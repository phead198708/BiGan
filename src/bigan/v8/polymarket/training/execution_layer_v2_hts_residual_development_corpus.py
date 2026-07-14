"""Build a causal, development-only HTS residual corpus from Phase 2 rounds."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.execution_layer_v2_estimand_reformulation import (
    _normalize_development_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_edge import (
    _bootstrap_market_mean,
    _candidate_ranking_key,
    _evaluate_candidate_oof,
    _quantile,
    fit_residual_offset_contract,
)
from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    PINNED_ISSUE_160_MANIFEST_SHA256,
    _fresh_public_ranking_row_from_canonical,
    _fresh_public_row_from_provider_feature_context,
    score_frozen_o_decision_rows,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_apply_simulated_order_to_state,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
    _v8_initial_runtime_state,
)

SCHEMA_PREFIX = "bigan-v8-hts-residual-development-corpus"
PHASE2_MARKET_PROBABILITY_MAPPING_RULE_ID = (
    "phase2_complementary_book_midpoint_ratio_v1"
)
REQUIRED_ACTIONS = {
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "NO_TRADE",
}
HTS_ACTIONS = {
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
}
FROZEN_OOF_PNL_ENTRY_EDGE_THRESHOLD = 0.02
FORBIDDEN_DECISION_FIELDS = {
    "resolved_outcome",
    "settlement_pnl",
    "settlement_return",
    "settlement_payout",
    "oracle_action",
    "future_return",
    "total_net_return",
    "total_net_pnl_per_notional",
}


@dataclass(frozen=True, slots=True)
class HTSResidualDevelopmentCorpusConfig:
    """Inputs for a frozen-protocol, post-cutoff development corpus build."""

    run_id: str
    output_dir: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    source_corpus_dirs: tuple[Path | str, ...]
    prior_development_rows_path: Path | str
    paper_candidate_unlock_dir: Path | str
    expected_unlock_manifest_sha256: str = PINNED_ISSUE_160_MANIFEST_SHA256
    canonical_o_source_manifest_path: Path | str | None = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if len(self.expected_protocol_sha256) != 64:
            raise ValueError("expected_protocol_sha256 must be a SHA-256 digest")
        if not self.source_corpus_dirs:
            raise ValueError("source_corpus_dirs must not be empty")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "protocol_path", Path(self.protocol_path))
        object.__setattr__(
            self,
            "source_corpus_dirs",
            tuple(Path(path) for path in self.source_corpus_dirs),
        )
        object.__setattr__(
            self, "prior_development_rows_path", Path(self.prior_development_rows_path)
        )
        object.__setattr__(
            self, "paper_candidate_unlock_dir", Path(self.paper_candidate_unlock_dir)
        )
        if self.canonical_o_source_manifest_path is not None:
            object.__setattr__(
                self,
                "canonical_o_source_manifest_path",
                Path(self.canonical_o_source_manifest_path),
            )


@dataclass(frozen=True, slots=True)
class HTSResidualForwardOOFConfig:
    """Frozen protocol inputs for development-only forward OOF evaluation."""

    run_id: str
    output_dir: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    development_corpus_manifest_paths: tuple[Path | str, ...]

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if len(self.expected_protocol_sha256) != 64:
            raise ValueError("expected_protocol_sha256 must be a SHA-256 digest")
        if not self.development_corpus_manifest_paths:
            raise ValueError("development_corpus_manifest_paths must not be empty")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "protocol_path", Path(self.protocol_path))
        object.__setattr__(
            self,
            "development_corpus_manifest_paths",
            tuple(Path(path) for path in self.development_corpus_manifest_paths),
        )


def build_hts_residual_development_corpus(
    config: HTSResidualDevelopmentCorpusConfig,
) -> dict[str, Any]:
    """Build residual rows without fitting or evaluating any candidate."""

    output_dir = Path(config.output_dir) / config.run_id
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    protocol_hash = _sha256_file(Path(config.protocol_path))
    if protocol_hash != config.expected_protocol_sha256:
        raise ValueError("development protocol SHA-256 mismatch")
    protocol = _load_json(Path(config.protocol_path))
    _validate_protocol(protocol)
    prior_rows = _load_jsonl(Path(config.prior_development_rows_path))
    prior_market_ids = {str(row.get("market_id") or "") for row in prior_rows}
    prior_max_decision_ts = max(
        (int(float(row.get("decision_ts") or 0)) for row in prior_rows),
        default=0,
    )

    public_rows: list[dict[str, Any]] = []
    targets: dict[tuple[str, int], dict[str, Any]] = {}
    corpus_audits: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for corpus_dir in sorted(Path(path) for path in config.source_corpus_dirs):
        preflight = _residual_source_chainlink_preflight(corpus_dir)
        if preflight["source_corpus_residual_eligible"] is not True:
            corpus_audits.append(preflight)
            feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
            feature_rows = _load_jsonl(feature_path) if feature_path.exists() else []
            if feature_rows:
                rejected.extend(
                    _rejection(
                        str(row.get("market_id") or ""),
                        int(row.get("decision_ts") or 0),
                        "source_corpus_chainlink_evidence_unavailable",
                        source_corpus_dir=str(corpus_dir.resolve()),
                        source_reason_codes=preflight["reason_codes"],
                    )
                    for row in feature_rows
                )
            else:
                rejected.append(
                    _rejection(
                        "",
                        0,
                        "source_corpus_chainlink_evidence_unavailable",
                        source_corpus_dir=str(corpus_dir.resolve()),
                        source_reason_codes=preflight["reason_codes"],
                    )
                )
            continue
        audit, corpus_public_rows, corpus_targets, corpus_rejections = (
            _load_verified_phase2_corpus(
                corpus_dir=corpus_dir,
                protocol=protocol,
                prior_market_ids=prior_market_ids,
                prior_max_decision_ts=prior_max_decision_ts,
            )
        )
        corpus_audits.append(audit)
        public_rows.extend(corpus_public_rows)
        targets.update(corpus_targets)
        rejected.extend(corpus_rejections)

    scoring = score_frozen_o_decision_rows(
        run_id=f"{config.run_id}-frozen-o-scoring",
        decision_rows=public_rows,
        paper_candidate_unlock_dir=config.paper_candidate_unlock_dir,
        expected_paper_candidate_unlock_manifest_sha256=(
            config.expected_unlock_manifest_sha256
        ),
        canonical_o_source_manifest_path=config.canonical_o_source_manifest_path,
    )
    if not scoring["scoring_passed"]:
        raise ValueError(
            "frozen O scorer failed closed: "
            + ",".join(
                scoring["canonical_scorer_report"][
                    "canonical_scorer_blocking_reason_codes"
                ]
            )
        )

    residual_rows: list[dict[str, Any]] = []
    selected_action_counts: Counter[str] = Counter()
    scored_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scored in scoring["canonical_scorer_report"][
        "canonical_scored_action_rows"
    ]:
        scored_by_group[str(scored["decision_group_id"])].append(scored)
    for selected in scoring["canonical_scorer_report"][
        "canonical_selected_decision_rows"
    ]:
        market_id = str(selected["market_id"])
        decision_ts = int(selected["decision_ts"])
        action = str(selected["action"])
        selected_action_counts[action] += 1
        if action not in HTS_ACTIONS:
            rejected.append(
                _rejection(
                    market_id,
                    decision_ts,
                    "frozen_o_selected_non_hts_action",
                    selected_action=action,
                )
            )
            continue
        target = targets.get((market_id, decision_ts))
        public_row = next(
            row
            for row in public_rows
            if str(row["market_id"]) == market_id
            and int(row["decision_ts"]) == decision_ts
        )
        if target is None:
            rejected.append(_rejection(market_id, decision_ts, "target_row_missing"))
            continue
        residual_rows.append(
            _residual_row(
                selected=selected,
                group_scored_rows=scored_by_group[str(selected["decision_group_id"])],
                public_row=public_row,
                target=target,
                protocol_hash=protocol_hash,
            )
        )

    residual_rows.sort(
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            str(row["row_identity"]),
        )
    )
    row_path = output_dir / "hts_residual_new_development_rows.jsonl"
    rejected_path = output_dir / "hts_residual_new_development_rejected_rows.jsonl"
    _write_jsonl(row_path, residual_rows)
    _write_jsonl(rejected_path, rejected)

    market_count = len({str(row["market_id"]) for row in residual_rows})
    minimum_markets = int(protocol["development_evaluation_support"][
        "minimum_market_count"
    ])
    development_support = _development_support(residual_rows, protocol)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "status": "DEVELOPMENT_CORPUS_READY"
        if development_support["passed"]
        else "DEVELOPMENT_CORPUS_SUPPORT_INSUFFICIENT",
        "protocol_path": str(Path(config.protocol_path).resolve()),
        "protocol_sha256": protocol_hash,
        "protocol_frozen_before_included_rows": True,
        "source_corpus_count": len(config.source_corpus_dirs),
        "source_corpus_residual_eligible_count": sum(
            audit.get("source_corpus_residual_eligible") is True
            for audit in corpus_audits
        ),
        "source_corpus_residual_ineligible_count": sum(
            audit.get("source_corpus_residual_eligible") is not True
            for audit in corpus_audits
        ),
        "source_corpus_ineligible_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for audit in corpus_audits
                    if audit.get("source_corpus_residual_eligible") is not True
                    for reason in audit.get("reason_codes", [])
                ).items()
            )
        ),
        "source_corpus_audits": corpus_audits,
        "source_decision_row_count": len(public_rows),
        "selected_action_distribution": dict(sorted(selected_action_counts.items())),
        "residual_row_count": len(residual_rows),
        "residual_market_count": market_count,
        "residual_source_run_count": len(
            {str(row["source_run_id"]) for row in residual_rows}
        ),
        "selected_side_distribution": dict(
            sorted(Counter(str(row["selected_side"]) for row in residual_rows).items())
        ),
        "resolved_outcome_distribution": dict(
            sorted(
                Counter(
                    str(row["target_provenance"]["resolved_outcome"])
                    for row in residual_rows
                ).items()
            )
        ),
        "rejected_row_count": len(rejected),
        "rejected_reason_distribution": dict(
            sorted(Counter(str(row["reason_code"]) for row in rejected).items())
        ),
        "feature_causality_violation_count": sum(
            int(row["max_input_ts"]) > int(row["decision_ts"])
            for row in residual_rows
        ),
        "market_probability_mapping_rule_id": (
            PHASE2_MARKET_PROBABILITY_MAPPING_RULE_ID
        ),
        "market_probability_mapping_provenance_valid_count": sum(
            (row.get("market_probability_mapping_provenance") or {}).get(
                "provenance_valid"
            )
            is True
            for row in public_rows
        ),
        "market_probability_mapping_violation_count": sum(
            row["reason_code"] == "market_probability_mapping_contract_violation"
            for row in rejected
        ),
        "source_chainlink_feature_coverage": _source_chainlink_feature_coverage(
            public_rows
        ),
        "residual_chainlink_feature_coverage": _chainlink_feature_coverage(
            residual_rows
        ),
        "chainlink_feature_coverage_scope": "residual_hts_rows",
        "chainlink_feature_coverage": _chainlink_feature_coverage(residual_rows),
        "minimum_development_market_count": minimum_markets,
        "development_support": development_support,
        "forward_oof_evaluation_ready": development_support["passed"],
        "forward_oof_blocking_reason_codes": development_support[
            "blocking_reason_codes"
        ],
        "candidate_fit_attempted": False,
        "candidate_selected": False,
        "confirmatory_validation_started": False,
        "future_confirmatory_validation_start_allowed": False,
        "uses_settlement_outcome_as_target_only": True,
        "uses_settlement_outcome_as_decision_input": False,
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    report_path = output_dir / "hts_residual_new_development_corpus_report.json"
    _write_json(report_path, report)
    _write_text(
        output_dir / "hts_residual_new_development_corpus_report.md",
        _report_markdown(report),
    )
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "report": _descriptor(report_path),
        "development_rows": _descriptor(row_path),
        "rejected_rows": _descriptor(rejected_path),
        "protocol": _descriptor(Path(config.protocol_path)),
        "frozen_o_source_manifest_sha256": scoring["canonical_context"].get(
            "source_manifest_sha256"
        ),
        "frozen_o_ranking_report_sha256": scoring["canonical_context"].get(
            "ranking_objective_report_sha256"
        ),
        "forward_oof_evaluation_ready": report["forward_oof_evaluation_ready"],
        "candidate_fit_attempted": False,
        "confirmatory_validation_started": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    manifest_path = output_dir / "hts_residual_new_development_corpus_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
    }


def run_hts_residual_development_forward_oof(
    config: HTSResidualForwardOOFConfig,
) -> dict[str, Any]:
    """Evaluate only the candidates frozen before the supplied development rows."""

    output_dir = Path(config.output_dir) / config.run_id
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    protocol_hash = _sha256_file(Path(config.protocol_path))
    if protocol_hash != config.expected_protocol_sha256:
        raise ValueError("development protocol SHA-256 mismatch")
    protocol = _load_json(Path(config.protocol_path))
    _validate_protocol(protocol)

    rows_by_identity: dict[str, dict[str, Any]] = {}
    input_manifests: list[dict[str, Any]] = []
    for manifest_path in sorted(config.development_corpus_manifest_paths):
        manifest = _load_json(manifest_path)
        if manifest.get("protocol", {}).get("sha256") != protocol_hash:
            raise ValueError("development corpus protocol lineage mismatch")
        row_descriptor = dict(manifest.get("development_rows") or {})
        row_path = Path(str(row_descriptor.get("path") or ""))
        if not row_path.exists() or _sha256_file(row_path) != row_descriptor.get(
            "sha256"
        ):
            raise ValueError("development rows descriptor hash mismatch")
        for source_row in _load_jsonl(row_path):
            identity = str(source_row.get("row_identity") or "")
            if not identity:
                raise ValueError("development row identity missing")
            previous = rows_by_identity.get(identity)
            if previous is not None and previous.get("row_content_sha256") != source_row.get(
                "row_content_sha256"
            ):
                raise ValueError("duplicate development row identity content mismatch")
            rows_by_identity[identity] = source_row
        input_manifests.append(_descriptor(manifest_path))
    protocol_lineage_hashes = {
        str(row.get("source_lineage", {}).get("frozen_development_protocol_sha256") or "")
        for row in rows_by_identity.values()
    }
    if protocol_lineage_hashes != {protocol_hash}:
        raise ValueError("development row protocol lineage mismatch")
    rows, invalid_rows = _normalize_development_rows(list(rows_by_identity.values()))
    if invalid_rows:
        reasons = Counter(
            reason
            for row in invalid_rows
            for reason in row.get("reason_codes", [])
        )
        raise ValueError(f"invalid development rows: {dict(reasons)}")
    if any(row.get("lineage") != "post_protocol_development_only" for row in rows):
        raise ValueError("non-post-protocol row in development OOF input")
    if any(int(row["max_input_ts"]) > int(row["decision_ts"]) for row in rows):
        raise ValueError("development OOF input contains causality violations")

    support = _development_support(rows, protocol)
    candidate_reports: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_contract: dict[str, Any] | None = None
    gate = {
        "passed": False,
        "checks": {"development_support_passed": support["passed"]},
        "blocking_reason_codes": support["blocking_reason_codes"],
    }
    if support["passed"]:
        minimum_training_runs = int(
            protocol["forward_oof_contract"]["minimum_training_source_runs"]
        )
        candidate_reports = [
            _evaluate_candidate_oof(
                rows,
                spec,
                minimum_training_runs=minimum_training_runs,
                include_prediction_rows=True,
            )
            for spec in _protocol_candidate_specs(protocol)
        ]
        candidate_reports.sort(key=_candidate_ranking_key)
        selected = candidate_reports[0]
        selected_spec = next(
            spec
            for spec in _protocol_candidate_specs(protocol)
            if spec["candidate_name"] == selected["candidate_name"]
        )
        selected_contract = fit_residual_offset_contract(rows, selected_spec)
        gate = _protocol_development_gate(
            selected=selected,
            selected_contract=selected_contract,
            support=support,
            protocol=protocol,
        )

    pnl_prediction_rows = [
        row
        for candidate in candidate_reports
        for row in candidate.pop("_forward_oof_prediction_rows", [])
    ]
    pnl_report, pnl_rows = _forward_oof_pnl_comparison(
        prediction_rows=pnl_prediction_rows,
        selected_candidate_name=(
            str(selected["candidate_name"]) if selected is not None else None
        ),
        protocol_hash=protocol_hash,
        run_id=config.run_id,
    )
    pnl_rows_path = output_dir / "hts_residual_development_forward_oof_pnl_rows.jsonl"
    _write_jsonl(pnl_rows_path, pnl_rows)
    pnl_report_path = (
        output_dir
        / "hts_residual_development_forward_oof_pnl_comparison_report.json"
    )
    _write_json(pnl_report_path, pnl_report)
    pnl_markdown_path = (
        output_dir / "hts_residual_development_forward_oof_pnl_comparison_report.md"
    )
    _write_text(pnl_markdown_path, _forward_oof_pnl_markdown(pnl_report))

    combined_rows_path = output_dir / "hts_residual_development_oof_rows.jsonl"
    _write_jsonl(combined_rows_path, rows)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-forward-oof-report-v1",
        "run_id": config.run_id,
        "status": "DEVELOPMENT_OOF_COMPLETE"
        if selected is not None
        else "DEVELOPMENT_OOF_BLOCKED_INSUFFICIENT_SUPPORT",
        "protocol": _descriptor(Path(config.protocol_path)),
        "input_development_corpus_manifests": input_manifests,
        "row_count": len(rows),
        "market_count": len({str(row["market_id"]) for row in rows}),
        "source_run_count": len({str(row["source_run_id"]) for row in rows}),
        "support": support,
        "candidate_reports": candidate_reports,
        "selected_candidate_name": selected.get("candidate_name")
        if selected is not None
        else None,
        "selected_candidate_contract": selected_contract,
        "development_candidate_gate": gate,
        "development_candidate_gate_passed": gate["passed"],
        "pnl_comparison_diagnostic": {
            "status": pnl_report["status"],
            "selected_candidate_name": pnl_report["selected_candidate_name"],
            "selected_candidate_pnl_improved_vs_raw_baseline": pnl_report[
                "selected_candidate_pnl_improved_vs_raw_baseline"
            ],
            "pnl_used_for_candidate_ranking": False,
            "pnl_used_for_threshold_tuning": False,
            "development_candidate_gate_unchanged": True,
        },
        "pnl_comparison_artifacts": {
            "report": _descriptor(pnl_report_path),
            "markdown": _descriptor(pnl_markdown_path),
            "rows": _descriptor(pnl_rows_path),
        },
        "candidate_freeze_review_allowed": gate["passed"],
        "candidate_frozen": False,
        "confirmatory_validation_started": False,
        "future_confirmatory_validation_start_allowed": False,
        "future_confirmatory_exactly_once_required": True,
        "uses_validation_labels_for_tuning": False,
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    report_path = output_dir / "hts_residual_development_forward_oof_report.json"
    _write_json(report_path, report)
    _write_text(
        output_dir / "hts_residual_development_forward_oof_report.md",
        _forward_oof_markdown(report),
    )
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-forward-oof-manifest-v1",
        "run_id": config.run_id,
        "report": _descriptor(report_path),
        "combined_rows": _descriptor(combined_rows_path),
        "pnl_comparison_report": _descriptor(pnl_report_path),
        "pnl_comparison_markdown": _descriptor(pnl_markdown_path),
        "pnl_comparison_rows": _descriptor(pnl_rows_path),
        "protocol": _descriptor(Path(config.protocol_path)),
        "development_candidate_gate_passed": gate["passed"],
        "candidate_frozen": False,
        "confirmatory_validation_started": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    manifest_path = output_dir / "hts_residual_development_forward_oof_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "pnl_comparison_report_path": pnl_report_path,
        "pnl_comparison_rows_path": pnl_rows_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
    }


def _forward_oof_pnl_comparison(
    *,
    prediction_rows: list[dict[str, Any]],
    selected_candidate_name: str | None,
    protocol_hash: str,
    run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    guard_config = _v8_execution_guard_config()
    contract = {
        "contract_id": "hts_residual_forward_oof_execution_bet_pnl_v1",
        "entry_edge_formula": (
            "predicted_probability - execution_price - "
            "decision_time_expected_execution_cost_per_unit"
        ),
        "entry_edge_threshold": FROZEN_OOF_PNL_ENTRY_EDGE_THRESHOLD,
        "entry_edge_threshold_source": (
            "preexisting_execution_layer_v2_entry_ev_threshold_default"
        ),
        "execution_guard_config": guard_config,
        "execution_guard_config_sha256": canonical_json_sha256(guard_config),
        "bet_size_source": "frozen_v8_execution_guard_proposed_order_size",
        "realized_bet_pnl_formula": (
            "guard_proposed_order_size_contracts * "
            "guarded_action_target_net_pnl_per_contract"
        ),
        "cost_basis_formula": (
            "guard_proposed_order_size_contracts * "
            "(execution_price + execution_cost_per_contract)"
        ),
        "pnl_unit_contract": "absolute_usd_like_pnl_per_prediction_contract",
        "guard_proposed_order_size_semantics": "prediction_contract_count",
        "threshold_frozen_before_oof_pnl_evaluation": True,
        "pnl_used_for_candidate_ranking": False,
        "pnl_used_for_threshold_tuning": False,
        "outcome_fields_used_for_selection": False,
        "outcome_fields_used_for_evaluation_only": True,
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[str(row["candidate_name"])].append(row)

    comparison_rows: list[dict[str, Any]] = []
    candidate_comparisons: list[dict[str, Any]] = []
    baseline_metrics: dict[str, Any] | None = None
    baseline_replay_rows: list[dict[str, Any]] | None = None
    expected_identities: set[str] | None = None
    for candidate_name in sorted(grouped):
        source_rows = sorted(
            grouped[candidate_name],
            key=lambda row: (
                int(row["decision_ts"]),
                str(row["market_id"]),
                str(row["row_identity"]),
            ),
        )
        identities = {str(row["row_identity"]) for row in source_rows}
        if expected_identities is None:
            expected_identities = identities
        elif identities != expected_identities:
            raise ValueError("candidate OOF PnL row identities do not match")
        if baseline_replay_rows is None:
            baseline_replay_rows = _frozen_execution_bet_replay(
                source_rows,
                policy_name="raw_market_probability_baseline",
                probability_field="raw_baseline_probability",
                guard_config=guard_config,
            )
            comparison_rows.extend(baseline_replay_rows)
        observed_baseline = _oof_execution_bet_metrics(baseline_replay_rows)
        if baseline_metrics is None:
            baseline_metrics = observed_baseline
        elif observed_baseline != baseline_metrics:
            raise ValueError("raw baseline OOF PnL metrics differ across candidates")
        candidate_replay_rows = _frozen_execution_bet_replay(
            source_rows,
            policy_name=candidate_name,
            probability_field="candidate_probability",
            guard_config=guard_config,
        )
        comparison_rows.extend(candidate_replay_rows)
        candidate_metrics = _oof_execution_bet_metrics(candidate_replay_rows)
        candidate_comparisons.append(
            {
                "candidate_name": candidate_name,
                "candidate_policy_metrics": candidate_metrics,
                "pnl_delta_vs_raw_baseline": (
                    candidate_metrics["settled_pnl_sum"]
                    - observed_baseline["settled_pnl_sum"]
                ),
                "execution_bet_count_delta_vs_raw_baseline": (
                    candidate_metrics["execution_bet_count"]
                    - observed_baseline["execution_bet_count"]
                ),
                "candidate_pnl_improved_vs_raw_baseline": (
                    candidate_metrics["settled_pnl_sum"]
                    > observed_baseline["settled_pnl_sum"]
                ),
                "candidate_probability_used_for_expected_net_return_gate": True,
                "frozen_execution_guard_applied_after_model_signal": True,
                "target_net_return_used_for_selection": False,
            }
        )

    baseline_metrics = baseline_metrics or _empty_oof_execution_bet_metrics()
    selected_comparison = next(
        (
            row
            for row in candidate_comparisons
            if row["candidate_name"] == selected_candidate_name
        ),
        None,
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-forward-oof-pnl-comparison-v1",
        "run_id": run_id,
        "status": (
            "DEVELOPMENT_OOF_PNL_DIAGNOSTIC_COMPLETE"
            if prediction_rows
            else "DEVELOPMENT_OOF_PNL_DIAGNOSTIC_BLOCKED_NO_PREDICTIONS"
        ),
        "protocol_sha256": protocol_hash,
        "pnl_diagnostic_contract": contract,
        "pnl_diagnostic_contract_sha256": canonical_json_sha256(contract),
        "selected_candidate_name": selected_candidate_name,
        "oof_prediction_row_count": len(prediction_rows),
        "unique_oof_row_count": len(expected_identities or set()),
        "raw_market_probability_baseline_policy_metrics": baseline_metrics,
        "candidate_comparisons": candidate_comparisons,
        "selected_candidate_comparison": selected_comparison,
        "selected_candidate_pnl_improved_vs_raw_baseline": bool(
            selected_comparison
            and selected_comparison["candidate_pnl_improved_vs_raw_baseline"]
        ),
        "raw_market_probability_used_as_direct_fair_value_ev_in_diagnostic_baseline": True,
        "raw_market_probability_direct_fair_value_execution_eligible": False,
        "explicit_decision_time_cost_field_available_for_selection": True,
        "execution_guard_applied_to_model_signal_candidates": True,
        "execution_bet_pnl_is_primary_comparison_metric": True,
        "execution_pnl_diagnostic_conclusion": (
            "no_policy_reached_frozen_execution_edge_threshold"
            if baseline_metrics["execution_bet_count"] == 0
            and all(
                row["candidate_policy_metrics"]["execution_bet_count"] == 0
                for row in candidate_comparisons
            )
            else "accepted_bet_pnl_available_for_diagnostic_comparison"
        ),
        "realized_target_includes_phase2_cost_model": True,
        "models_frozen_order_sizing_and_runtime_exposure": True,
        "models_actual_exchange_fills": False,
        "development_candidate_gate_unchanged": True,
        "pnl_used_for_candidate_ranking": False,
        "pnl_used_for_threshold_tuning": False,
        "confirmatory_validation_started": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    return report, comparison_rows


def _frozen_execution_bet_replay(
    source_rows: list[dict[str, Any]],
    *,
    policy_name: str,
    probability_field: str,
    guard_config: dict[str, Any],
) -> list[dict[str, Any]]:
    state = _v8_initial_runtime_state(guard_config)
    market_close_by_open_position: dict[str, int] = {}
    replay_rows: list[dict[str, Any]] = []
    ordered_rows = sorted(
        source_rows,
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            str(row["row_identity"]),
        ),
    )
    for index, row in enumerate(ordered_rows, start=1):
        _release_closed_simulated_positions(
            state=state,
            market_close_by_open_position=market_close_by_open_position,
            decision_ts=int(row["decision_ts"]),
        )
        probability = float(row[probability_field])
        expected_net_pnl_per_contract = (
            probability
            - float(row["execution_price"])
            - float(row["decision_time_expected_execution_cost_per_unit"])
        )
        signal_passed = (
            expected_net_pnl_per_contract >= FROZEN_OOF_PNL_ENTRY_EDGE_THRESHOLD
        )
        handoff = dict(row.get("execution_handoff_context") or {})
        blocking_reason_codes: list[str] = []
        guard_row: dict[str, Any] | None = None
        if not signal_passed:
            blocking_reason_codes.append(
                "model_expected_net_return_below_frozen_threshold"
            )
        elif not handoff:
            blocking_reason_codes.append("execution_handoff_context_missing")
        else:
            guard_row = _v8_execution_guard_decision(
                handoff,
                guard_config=guard_config,
                runtime_state=state,
                runtime_mode="simulated_runtime_state",
            )
            blocking_reason_codes.extend(
                guard_row["execution_blocking_reason_codes"]
            )
        guard_order_allowed = bool(guard_row and guard_row["order_allowed"])
        order_allowed = guard_order_allowed
        guarded_action = (
            str(guard_row["execution_guarded_action"])
            if guard_row is not None
            else None
        )
        proposed_size = (
            float(guard_row["proposed_order_size"]) if order_allowed else 0.0
        )
        target_by_action = dict(
            row.get("evaluation_target_net_pnl_per_contract_by_action") or {}
        )
        components_by_action = dict(
            row.get("evaluation_target_pnl_components_by_action") or {}
        )
        guarded_target = target_by_action.get(guarded_action)
        source_selected_target = target_by_action.get(str(row["selected_action"]))
        target_available = isinstance(guarded_target, int | float) and math.isfinite(
            float(guarded_target)
        )
        guarded_components = dict(components_by_action.get(guarded_action) or {})
        component_values = {
            name: guarded_components.get(name)
            for name in (
                "gross_pnl_per_contract",
                "fees_per_contract",
                "slippage_per_contract",
                "liquidity_impact_per_contract",
                "execution_cost_per_contract",
                "net_pnl_per_contract",
            )
        }
        component_values_available = all(
            isinstance(value, int | float) and math.isfinite(float(value))
            for value in component_values.values()
        )
        simulated_order_id = None
        gross_pnl = None
        execution_cost = None
        settled_pnl = None
        if order_allowed and target_available and component_values_available:
            simulated_order_id = f"{policy_name}-bet-{index:06d}"
            gross_pnl = proposed_size * float(
                component_values["gross_pnl_per_contract"]
            )
            execution_cost = proposed_size * float(
                component_values["execution_cost_per_contract"]
            )
            settled_pnl = proposed_size * float(guarded_target)
            _v8_apply_simulated_order_to_state(
                state=state,
                decision=guard_row,
                simulated_order_id=simulated_order_id,
            )
            market_close_by_open_position[str(row["market_id"])] = int(
                row["market_close_ts"]
            )
        elif order_allowed:
            if not target_available:
                blocking_reason_codes.append(
                    "guarded_action_net_pnl_per_contract_target_missing"
                )
            if not component_values_available:
                blocking_reason_codes.append(
                    "guarded_action_pnl_components_missing"
                )
            order_allowed = False
            proposed_size = 0.0
        replay_rows.append(
            {
                "policy_name": policy_name,
                "candidate_name": row.get("candidate_name"),
                "row_identity": str(row["row_identity"]),
                "market_id": str(row["market_id"]),
                "source_run_id": str(row["source_run_id"]),
                "decision_ts": int(row["decision_ts"]),
                "market_close_ts": int(row["market_close_ts"]),
                "source_selected_action": str(row["selected_action"]),
                "source_selected_side": str(row["selected_side"]),
                "model_probability": probability,
                "model_probability_source_field": probability_field,
                "execution_price": float(row["execution_price"]),
                "decision_time_expected_execution_cost_per_unit": float(
                    row["decision_time_expected_execution_cost_per_unit"]
                ),
                "model_expected_net_pnl_per_contract": (
                    expected_net_pnl_per_contract
                ),
                "model_entry_edge": expected_net_pnl_per_contract,
                "model_signal_passed": signal_passed,
                "execution_guard_evaluated": guard_row is not None,
                "execution_guard_order_allowed": guard_order_allowed,
                "execution_guarded_action": guarded_action,
                "execution_guarded_side": (
                    guard_row.get("execution_guarded_side") if guard_row else None
                ),
                "execution_order_allowed": order_allowed,
                "simulated_order_id": simulated_order_id,
                "paper_bet_contract_size": proposed_size,
                "paper_bet_entry_cost_basis": (
                    proposed_size
                    * (
                        float(row["execution_price"])
                        + float(component_values["execution_cost_per_contract"])
                    )
                    if order_allowed
                    else 0.0
                ),
                "guarded_action_target_net_pnl_per_contract": (
                    float(guarded_target) if target_available else None
                ),
                "source_selected_action_target_net_pnl_per_contract": (
                    float(source_selected_target)
                    if isinstance(source_selected_target, int | float)
                    and math.isfinite(float(source_selected_target))
                    else None
                ),
                "guarded_action_gross_pnl_per_contract": (
                    float(component_values["gross_pnl_per_contract"])
                    if component_values_available
                    else None
                ),
                "guarded_action_execution_cost_per_contract": (
                    float(component_values["execution_cost_per_contract"])
                    if component_values_available
                    else None
                ),
                "gross_pnl": gross_pnl,
                "execution_cost": execution_cost,
                "settled_pnl": settled_pnl,
                "settlement_target_available": (
                    target_available and component_values_available
                ),
                "execution_blocking_reason_codes": sorted(
                    set(blocking_reason_codes)
                ),
                "execution_guard_reason_codes": (
                    list(guard_row["execution_guard_reason_codes"])
                    if guard_row
                    else []
                ),
                "source_o_score_mutated": False,
                "source_ranking_mutated": False,
                "selection_uses_outcome_fields": False,
                "outcome_aware_evaluation_only": True,
                "promotion_evidence_eligible": False,
                "paper_only": True,
                "capital_at_risk": False,
            }
        )
    return replay_rows


def _release_closed_simulated_positions(
    *,
    state: dict[str, Any],
    market_close_by_open_position: dict[str, int],
    decision_ts: int,
) -> None:
    closed_markets = sorted(
        market_id
        for market_id, close_ts in market_close_by_open_position.items()
        if close_ts <= decision_ts
    )
    for market_id in closed_markets:
        position = state["open_position_by_market_id"].pop(market_id, None)
        market_close_by_open_position.pop(market_id, None)
        if not isinstance(position, dict):
            continue
        side = str(position.get("side") or "NONE")
        notional = float(position.get("notional") or 0.0)
        state["open_position_by_market_side"].pop(f"{market_id}|{side}", None)
        state["current_market_exposure_by_market_id"].pop(market_id, None)
        state["current_side_exposure_by_side"][side] = max(
            0.0,
            float(state["current_side_exposure_by_side"].get(side) or 0.0)
            - notional,
        )
        state["current_total_exposure"] = max(
            0.0, float(state.get("current_total_exposure") or 0.0) - notional
        )


def _oof_execution_bet_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bets = [row for row in rows if row["execution_order_allowed"] is True]
    settled = [row for row in bets if row["settled_pnl"] is not None]
    pnl_values = [float(row["settled_pnl"]) for row in settled]
    pnl_sum = sum(pnl_values)
    cost_basis = sum(float(row["paper_bet_entry_cost_basis"]) for row in bets)
    contract_size = sum(float(row["paper_bet_contract_size"]) for row in bets)
    gross_pnl_sum = sum(float(row["gross_pnl"]) for row in settled)
    execution_cost_sum = sum(float(row["execution_cost"]) for row in settled)
    market_pnl: dict[str, float] = defaultdict(float)
    for row in settled:
        market_pnl[str(row["market_id"])] += float(row["settled_pnl"])
    equity = 0.0
    peak = 0.0
    maximum_drawdown = 0.0
    for row in sorted(
        settled,
        key=lambda row: (
            int(row["market_close_ts"]),
            str(row["market_id"]),
            str(row["simulated_order_id"]),
        ),
    ):
        pnl = float(row["settled_pnl"])
        equity += pnl
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
    blockers = Counter(
        reason
        for row in rows
        for reason in row["execution_blocking_reason_codes"]
    )
    model_edges = [float(row["model_entry_edge"]) for row in rows]
    selected_action_targets = [
        float(row["source_selected_action_target_net_pnl_per_contract"])
        for row in rows
        if row["source_selected_action_target_net_pnl_per_contract"] is not None
    ]
    return {
        "source_decision_count": len(rows),
        "model_signal_candidate_count": sum(row["model_signal_passed"] for row in rows),
        "execution_guard_evaluated_count": sum(
            row["execution_guard_evaluated"] for row in rows
        ),
        "execution_guard_passed_count": sum(
            row["execution_guard_order_allowed"] for row in rows
        ),
        "execution_bet_count": len(bets),
        "settled_bet_count": len(settled),
        "unresolved_bet_count": len(bets) - len(settled),
        "bet_market_count": len({str(row["market_id"]) for row in bets}),
        "contract_size_sum": contract_size,
        "cost_basis": cost_basis,
        "gross_pnl_sum": gross_pnl_sum,
        "execution_cost_sum": execution_cost_sum,
        "settled_pnl_sum": pnl_sum,
        "pnl_accounting_reconciled": math.isclose(
            pnl_sum,
            gross_pnl_sum - execution_cost_sum,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ),
        "roi_on_cost_basis": pnl_sum / cost_basis if cost_basis > 0.0 else 0.0,
        "mean_pnl_per_bet": pnl_sum / len(settled) if settled else 0.0,
        "win_count": sum(value > 0.0 for value in pnl_values),
        "loss_count": sum(value < 0.0 for value in pnl_values),
        "flat_count": sum(value == 0.0 for value in pnl_values),
        "win_rate": (
            sum(value > 0.0 for value in pnl_values) / len(settled)
            if settled
            else 0.0
        ),
        "maximum_drawdown": maximum_drawdown,
        "max_drawdown_ordering": "market_close_ts_market_id_simulated_order_id",
        "profitable_market_count": sum(value > 0.0 for value in market_pnl.values()),
        "losing_market_count": sum(value < 0.0 for value in market_pnl.values()),
        "model_entry_edge_summary": _numeric_distribution(model_edges),
        "model_entry_edge_threshold": FROZEN_OOF_PNL_ENTRY_EDGE_THRESHOLD,
        "nonnegative_model_entry_edge_count": sum(
            value >= 0.0 for value in model_edges
        ),
        "within_one_cent_of_entry_threshold_count": sum(
            value >= FROZEN_OOF_PNL_ENTRY_EDGE_THRESHOLD - 0.01
            for value in model_edges
        ),
        "source_selected_action_target_diagnostic": {
            "evaluation_only": True,
            "target_available_count": len(selected_action_targets),
            "positive_count": sum(value > 0.0 for value in selected_action_targets),
            "negative_count": sum(value < 0.0 for value in selected_action_targets),
            "flat_count": sum(value == 0.0 for value in selected_action_targets),
            "net_pnl_per_contract_sum": sum(selected_action_targets),
            "positive_net_pnl_per_contract_sum": sum(
                value for value in selected_action_targets if value > 0.0
            ),
            "negative_net_pnl_per_contract_sum": sum(
                value for value in selected_action_targets if value < 0.0
            ),
        },
        "pnl_by_side": _oof_execution_pnl_group_summary(
            settled, "execution_guarded_side"
        ),
        "pnl_by_action": _oof_execution_pnl_group_summary(
            settled, "execution_guarded_action"
        ),
        "execution_blocking_reason_distribution": dict(sorted(blockers.items())),
        "bet_row_identity_set_sha256": canonical_json_sha256(
            sorted(str(row["row_identity"]) for row in bets)
        ),
    }


def _empty_oof_execution_bet_metrics() -> dict[str, Any]:
    return {
        "source_decision_count": 0,
        "model_signal_candidate_count": 0,
        "execution_guard_evaluated_count": 0,
        "execution_guard_passed_count": 0,
        "execution_bet_count": 0,
        "settled_bet_count": 0,
        "unresolved_bet_count": 0,
        "bet_market_count": 0,
        "contract_size_sum": 0.0,
        "cost_basis": 0.0,
        "gross_pnl_sum": 0.0,
        "execution_cost_sum": 0.0,
        "settled_pnl_sum": 0.0,
        "pnl_accounting_reconciled": True,
        "roi_on_cost_basis": 0.0,
        "mean_pnl_per_bet": 0.0,
        "win_count": 0,
        "loss_count": 0,
        "flat_count": 0,
        "win_rate": 0.0,
        "maximum_drawdown": 0.0,
        "max_drawdown_ordering": "market_close_ts_market_id_simulated_order_id",
        "profitable_market_count": 0,
        "losing_market_count": 0,
        "model_entry_edge_summary": _numeric_distribution([]),
        "model_entry_edge_threshold": FROZEN_OOF_PNL_ENTRY_EDGE_THRESHOLD,
        "nonnegative_model_entry_edge_count": 0,
        "within_one_cent_of_entry_threshold_count": 0,
        "source_selected_action_target_diagnostic": {
            "evaluation_only": True,
            "target_available_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "flat_count": 0,
            "net_pnl_per_contract_sum": 0.0,
            "positive_net_pnl_per_contract_sum": 0.0,
            "negative_net_pnl_per_contract_sum": 0.0,
        },
        "pnl_by_side": {},
        "pnl_by_action": {},
        "execution_blocking_reason_distribution": {},
        "bet_row_identity_set_sha256": canonical_json_sha256([]),
    }


def _oof_execution_pnl_group_summary(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(
            float(row["settled_pnl"])
        )
    return {
        name: {
            "bet_count": len(values),
            "settled_pnl_sum": sum(values),
            "mean_pnl_per_bet": sum(values) / len(values),
            "win_rate": sum(value > 0.0 for value in values) / len(values),
        }
        for name, values in sorted(grouped.items())
    }


def _numeric_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "p10": _quantile(values, 0.10),
        "median": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }


def _evaluation_pnl_components(label: dict[str, Any]) -> dict[str, float]:
    fees = float(label.get("fees") or 0.0)
    slippage = float(label.get("slippage") or 0.0)
    liquidity_impact = float(label.get("liquidity_impact") or 0.0)
    execution_cost = fees + slippage + liquidity_impact
    net_pnl = float(label["total_net_pnl_per_notional"])
    return {
        "gross_pnl_per_contract": net_pnl + execution_cost,
        "fees_per_contract": fees,
        "slippage_per_contract": slippage,
        "liquidity_impact_per_contract": liquidity_impact,
        "execution_cost_per_contract": execution_cost,
        "net_pnl_per_contract": net_pnl,
    }


def _protocol_candidate_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "model_family": "market_probability_logit_offset_regularized_residual",
        "market_probability_offset_field": "selected_side_probability",
        "market_probability_offset_transform": "logit",
        "market_probability_offset_coefficient": 1.0,
        "market_probability_offset_trainable": False,
        "residual_coefficients_shrinkage_target": 0.0,
        "probability_bounds": [0.01, 0.99],
        "decision_time_features_only": True,
        "settlement_outcome_used_as_target_only": True,
        "settlement_outcome_used_as_input": False,
    }
    return [{**common, **dict(spec)} for spec in protocol["candidate_specifications"]]


def _development_support(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    contract = protocol["development_evaluation_support"]
    market_count = len({str(row["market_id"]) for row in rows})
    source_run_count = len({str(row["source_run_id"]) for row in rows})
    sides = {str(row["selected_side"]) for row in rows}
    outcomes = {
        str(row["target_provenance"]["resolved_outcome"]) for row in rows
    }
    checks = {
        "minimum_market_count_met": market_count
        >= int(contract["minimum_market_count"]),
        "minimum_source_run_count_met": source_run_count
        >= int(contract["minimum_source_run_count"]),
        "both_selected_sides_present": sides == {"UP", "DOWN"},
        "both_resolved_outcomes_present": outcomes == {"UP", "DOWN"},
    }
    reason_by_check = {
        "minimum_market_count_met": "insufficient_development_market_support",
        "minimum_source_run_count_met": "insufficient_development_source_run_support",
        "both_selected_sides_present": "missing_selected_side_support",
        "both_resolved_outcomes_present": "missing_resolved_outcome_support",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "blocking_reason_codes": [
            reason_by_check[name] for name, passed in checks.items() if not passed
        ],
        "market_count": market_count,
        "source_run_count": source_run_count,
        "selected_side_counts": dict(
            sorted(Counter(str(row["selected_side"]) for row in rows).items())
        ),
        "resolved_outcome_counts": dict(
            sorted(
                Counter(
                    str(row["target_provenance"]["resolved_outcome"])
                    for row in rows
                ).items()
            )
        ),
    }


def _protocol_development_gate(
    *,
    selected: dict[str, Any],
    selected_contract: dict[str, Any],
    support: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gate_config = protocol["development_gate"]
    market_deltas = [
        float(row["brier_improvement"])
        for row in selected["market_level_error_deltas"]["by_market"]
    ]
    bootstrap = _bootstrap_market_mean(
        market_deltas,
        samples=int(gate_config["bootstrap_samples"]),
        seed=int(gate_config["bootstrap_seed"]),
    )
    confidence = float(gate_config["bootstrap_confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    interval = {
        "confidence_level": confidence,
        "lower": _quantile(bootstrap, alpha),
        "upper": _quantile(bootstrap, 1.0 - alpha),
    }
    checks = {
        "development_support_passed": support["passed"],
        "relative_brier_improvement_vs_raw_passed": selected[
            "relative_brier_improvement_vs_raw"
        ]
        >= float(gate_config["minimum_relative_brier_improvement_vs_raw"]),
        "relative_log_loss_improvement_vs_raw_passed": selected[
            "relative_log_loss_improvement_vs_raw"
        ]
        >= float(gate_config["minimum_relative_log_loss_improvement_vs_raw"]),
        "positive_market_mean_brier_improvement_passed": (
            selected["market_level_error_deltas"]["mean_brier_improvement"] > 0.0
        ),
        "market_bootstrap_lower_bound_positive_passed": interval["lower"] > 0.0,
        "finite_bounded_coefficients_passed": selected_contract[
            "finite_and_bounded"
        ]
        is True,
    }
    reason_by_check = {
        "development_support_passed": "development_support_gate_failed",
        "relative_brier_improvement_vs_raw_passed": (
            "relative_brier_improvement_vs_raw_failed"
        ),
        "relative_log_loss_improvement_vs_raw_passed": (
            "relative_log_loss_improvement_vs_raw_failed"
        ),
        "positive_market_mean_brier_improvement_passed": (
            "positive_market_mean_brier_improvement_failed"
        ),
        "market_bootstrap_lower_bound_positive_passed": (
            "development_market_bootstrap_interval_crosses_zero"
        ),
        "finite_bounded_coefficients_passed": "residual_coefficients_invalid",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "bootstrap_market_mean_improvement_interval": interval,
        "blocking_reason_codes": [
            reason_by_check[name] for name, passed in checks.items() if not passed
        ],
        "development_only": True,
        "does_not_automatically_start_confirmatory_validation": True,
    }


def _load_verified_phase2_corpus(
    *,
    corpus_dir: Path,
    protocol: dict[str, Any],
    prior_market_ids: set[str],
    prior_max_decision_ts: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
    list[dict[str, Any]],
]:
    required = {
        "polymarket_corpus_manifest.json",
        "polymarket_feature_rows.jsonl",
        "polymarket_label_rows.jsonl",
        "polymarket_market_metadata.jsonl",
        "polymarket_resolution_events.jsonl",
        "polymarket_chainlink_prices.jsonl",
        "polymarket_chainlink_decision_time_evidence_manifest.json",
        "training_corpus_provenance.json",
    }
    missing = sorted(name for name in required if not (corpus_dir / name).exists())
    if missing:
        raise ValueError(f"missing required corpus artifacts in {corpus_dir}: {missing}")
    corpus_manifest = _load_json(corpus_dir / "polymarket_corpus_manifest.json")
    normalized_hashes = dict(corpus_manifest.get("normalized_artifact_hashes") or {})
    expected_hashes = {
        "feature_rows": "polymarket_feature_rows.jsonl",
        "label_rows": "polymarket_label_rows.jsonl",
        "market_metadata": "polymarket_market_metadata.jsonl",
        "resolution_events": "polymarket_resolution_events.jsonl",
    }
    for key, filename in expected_hashes.items():
        if normalized_hashes.get(key) != _sha256_file(corpus_dir / filename):
            raise ValueError(f"Phase 2 normalized artifact hash mismatch: {filename}")
    chainlink_manifest_path = (
        corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json"
    )
    chainlink_manifest = _load_json(chainlink_manifest_path)
    chainlink_path = corpus_dir / "polymarket_chainlink_prices.jsonl"
    if chainlink_manifest.get("evidence_sha256") != _sha256_file(chainlink_path):
        raise ValueError("Chainlink evidence SHA-256 mismatch")
    if chainlink_manifest.get("timestamp_causality_violation_count") != 0:
        raise ValueError("Chainlink evidence reports timestamp causality violations")

    features = _load_jsonl(corpus_dir / "polymarket_feature_rows.jsonl")
    labels = _load_jsonl(corpus_dir / "polymarket_label_rows.jsonl")
    metadata_rows = _load_jsonl(corpus_dir / "polymarket_market_metadata.jsonl")
    resolutions = _load_jsonl(corpus_dir / "polymarket_resolution_events.jsonl")
    chainlink_rows = _load_jsonl(chainlink_path)
    provenance = _load_json(corpus_dir / "training_corpus_provenance.json")
    metadata_by_market = {str(row["market_id"]): row for row in metadata_rows}
    resolution_by_market = {str(row["market_id"]): row for row in resolutions}
    labels_by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        labels_by_decision[(str(row["market_id"]), int(row["decision_ts"]))].append(
            row
        )
    public_rows: list[dict[str, Any]] = []
    targets: dict[tuple[str, int], dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    not_before_ts = int(protocol["collection_not_before_ts"])
    excluded_corpus_ids = set(protocol.get("excluded_smoke_corpus_ids") or [])
    corpus_id = str(provenance.get("corpus_id") or corpus_dir.name)
    if corpus_id in excluded_corpus_ids:
        raise ValueError(f"pre-protocol smoke corpus is excluded: {corpus_id}")
    for feature_row in features:
        market_id = str(feature_row["market_id"])
        decision_ts = int(feature_row["decision_ts"])
        reasons: list[str] = []
        if market_id in prior_market_ids:
            reasons.append("market_overlaps_prior_development")
        if decision_ts <= prior_max_decision_ts:
            reasons.append("decision_not_strictly_later_than_prior_development")
        if decision_ts <= not_before_ts:
            reasons.append("decision_not_after_frozen_protocol")
        if int(feature_row.get("max_input_ts") or 0) > decision_ts:
            reasons.append("phase2_feature_causality_violation")
        action_rows = labels_by_decision.get((market_id, decision_ts), [])
        if {str(row.get("action") or "") for row in action_rows} != REQUIRED_ACTIONS:
            reasons.append("incomplete_5_action_label_grid")
        metadata = metadata_by_market.get(market_id)
        resolution = resolution_by_market.get(market_id)
        if metadata is None:
            reasons.append("market_metadata_missing")
        if resolution is None or resolution.get("resolved_outcome") not in {
            "UP",
            "DOWN",
        }:
            reasons.append("official_resolution_missing")
        if reasons:
            rejected.extend(
                _rejection(market_id, decision_ts, reason) for reason in sorted(reasons)
            )
            continue
        public_row = _phase2_feature_to_public_row(
            run_id=str(provenance.get("run_id") or corpus_id),
            row_index=len(public_rows),
            feature_row=feature_row,
            market=metadata,
            chainlink_rows=chainlink_rows,
        )
        mapping_provenance = dict(
            public_row.get("market_probability_mapping_provenance") or {}
        )
        expected_mapping_rule = str(
            (protocol.get("market_probability_mapping_contract") or {}).get(
                "rule_id"
            )
            or PHASE2_MARKET_PROBABILITY_MAPPING_RULE_ID
        )
        if (
            public_row.get("market_probability_mapping_rule_id")
            != expected_mapping_rule
            or mapping_provenance.get("provenance_valid") is not True
        ):
            rejected.append(
                _rejection(
                    market_id,
                    decision_ts,
                    "market_probability_mapping_contract_violation",
                    expected_rule_id=expected_mapping_rule,
                    observed_rule_id=public_row.get(
                        "market_probability_mapping_rule_id"
                    ),
                    mapping_provenance=mapping_provenance,
                )
            )
            continue
        required_chainlink_fields = (
            "chainlink_price_at_decision",
            "chainlink_reference_price_at_market_start",
            "chainlink_reference_distance_at_decision",
            "chainlink_momentum_30s",
            "chainlink_momentum_60s",
            "chainlink_momentum_120s",
            "chainlink_realized_volatility_120s",
        )
        missing_chainlink = [
            field
            for field in required_chainlink_fields
            if not isinstance(public_row.get(field), int | float)
            or not math.isfinite(float(public_row[field]))
        ]
        chainlink_provenance = dict(
            public_row.get("chainlink_regime_feature_provenance") or {}
        )
        if missing_chainlink or chainlink_provenance.get("provenance_valid") is not True:
            rejected.append(
                _rejection(
                    market_id,
                    decision_ts,
                    "complete_causal_chainlink_feature_block_missing",
                    missing_fields=missing_chainlink,
                    unavailable_reason_codes=chainlink_provenance.get(
                        "unavailable_reason_codes", []
                    ),
                )
            )
            continue
        if _forbidden_decision_fields(public_row):
            rejected.append(
                _rejection(market_id, decision_ts, "forbidden_decision_field_present")
            )
            continue
        if int(public_row["decision_time_feature_max_input_ts"]) > decision_ts:
            rejected.append(
                _rejection(market_id, decision_ts, "joined_feature_causality_violation")
            )
            continue
        public_rows.append(public_row)
        targets[(market_id, decision_ts)] = {
            "resolved_outcome": str(resolution["resolved_outcome"]),
            "raw_resolution_sha256": resolution.get("raw_resolution_sha256"),
            "source_run_id": str(provenance.get("run_id") or corpus_id),
            "source_corpus_dir": str(corpus_dir.resolve()),
            "source_corpus_manifest_sha256": _sha256_file(
                corpus_dir / "polymarket_corpus_manifest.json"
            ),
            "source_chainlink_evidence_sha256": _sha256_file(chainlink_path),
            "source_feature_rows_sha256": _sha256_file(
                corpus_dir / "polymarket_feature_rows.jsonl"
            ),
            "source_label_rows_sha256": _sha256_file(
                corpus_dir / "polymarket_label_rows.jsonl"
            ),
            "selected_action_labels": {
                str(row["action"]): row for row in action_rows
            },
        }
    audit = {
        "corpus_id": corpus_id,
        "corpus_dir": str(corpus_dir.resolve()),
        "source_corpus_residual_eligible": True,
        "reason_codes": [],
        "phase2_corpus_manifest_sha256": _sha256_file(
            corpus_dir / "polymarket_corpus_manifest.json"
        ),
        "chainlink_evidence_sha256": _sha256_file(chainlink_path),
        "chainlink_evidence_manifest_sha256": _sha256_file(chainlink_manifest_path),
        "feature_row_count": len(features),
        "label_row_count": len(labels),
        "public_decision_row_count": len(public_rows),
        "rejected_reason_count": len(rejected),
        "phase2_normalized_hashes_verified": True,
        "chainlink_hash_verified": True,
        "paper_only": True,
        "capital_at_risk": False,
    }
    return audit, public_rows, targets, rejected


def _residual_source_chainlink_preflight(corpus_dir: Path) -> dict[str, Any]:
    chainlink_path = corpus_dir / "polymarket_chainlink_prices.jsonl"
    manifest_path = (
        corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json"
    )
    reason_codes: list[str] = []
    if not chainlink_path.exists():
        reason_codes.append("chainlink_evidence_file_missing")
    elif chainlink_path.stat().st_size == 0:
        reason_codes.append("chainlink_evidence_file_empty")
    if not manifest_path.exists():
        reason_codes.append("chainlink_evidence_manifest_missing")
    return {
        "corpus_id": corpus_dir.name,
        "corpus_dir": str(corpus_dir.resolve()),
        "source_corpus_residual_eligible": not reason_codes,
        "reason_codes": reason_codes,
        "chainlink_evidence_path": str(chainlink_path.resolve()),
        "chainlink_evidence_file_exists": chainlink_path.exists(),
        "chainlink_evidence_file_nonempty": (
            chainlink_path.exists() and chainlink_path.stat().st_size > 0
        ),
        "chainlink_evidence_manifest_path": str(manifest_path.resolve()),
        "chainlink_evidence_manifest_exists": manifest_path.exists(),
        "phase2_normalized_hashes_verified": False,
        "chainlink_hash_verified": False,
        "paper_only": True,
        "capital_at_risk": False,
    }


def _phase2_feature_to_public_row(
    *,
    run_id: str,
    row_index: int,
    feature_row: dict[str, Any],
    market: dict[str, Any],
    chainlink_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    features = dict(feature_row["features"])
    decision_ts = int(feature_row["decision_ts"])
    up = _phase2_book(features, "up", decision_ts)
    down = _phase2_book(features, "down", decision_ts)
    close_price = float(features["btc_mid_price"])
    return_1m = float(features.get("btc_return_1m") or 0.0)
    open_price = close_price / (1.0 + return_1m) if return_1m > -1.0 else close_price
    btc_provenance = dict(
        (feature_row.get("feature_provenance") or {}).get("btc_mid_price") or {}
    )
    candle_available_at = int(
        btc_provenance.get("input_end_ts")
        or feature_row.get("max_input_ts")
        or decision_ts
    )
    candle = {
        "open_price": open_price,
        "close_price": close_price,
        "available_at_ts": candle_available_at,
        "ts": candle_available_at,
    }
    public_row = _fresh_public_row_from_provider_feature_context(
        run_id=run_id,
        row_index=row_index,
        market=dict(market),
        up=up,
        down=down,
        candle=candle,
        reference_candle=None,
        chainlink_rtds_prices=chainlink_rows,
        decision_ts=decision_ts,
    )
    up_mid = float(up["mid_price"])
    down_mid = float(down["mid_price"])
    midpoint_sum = up_mid + down_mid
    expected_p_up = up_mid / midpoint_sum if midpoint_sum > 0.0 else 0.5
    mapping_max_input_ts = max(
        int(up["available_at_ts"]), int(down["available_at_ts"])
    )
    observed_p_up = float(public_row["p_up"])
    public_row["market_probability_mapping_rule_id"] = (
        PHASE2_MARKET_PROBABILITY_MAPPING_RULE_ID
    )
    public_row["market_probability_mapping_provenance"] = {
        "rule_id": PHASE2_MARKET_PROBABILITY_MAPPING_RULE_ID,
        "source_fields_used": [
            "polymarket_feature_rows.features.up_bid",
            "polymarket_feature_rows.features.up_ask",
            "polymarket_feature_rows.features.down_bid",
            "polymarket_feature_rows.features.down_ask",
        ],
        "formula": "up_midpoint / (up_midpoint + down_midpoint)",
        "up_midpoint": up_mid,
        "down_midpoint": down_mid,
        "decision_ts": decision_ts,
        "max_input_ts": mapping_max_input_ts,
        "provenance_valid": bool(
            midpoint_sum > 0.0
            and mapping_max_input_ts <= decision_ts
            and abs(observed_p_up - expected_p_up) <= 1e-12
        ),
        "uses_settlement_or_future_fields": False,
    }
    public_row["phase2_features"] = features
    public_row["phase2_feature_max_input_ts"] = int(
        feature_row.get("max_input_ts") or 0
    )
    public_row["decision_time_feature_max_input_ts"] = max(
        int(public_row["decision_time_feature_max_input_ts"]),
        int(feature_row.get("max_input_ts") or 0),
    )
    return public_row


def _phase2_book(
    features: dict[str, Any], side: str, decision_ts: int
) -> dict[str, Any]:
    staleness = max(int(float(features.get(f"{side}_book_staleness_ms") or 0)), 0)
    bid = float(features[f"{side}_bid"])
    ask = float(features[f"{side}_ask"])
    return {
        "outcome": side.upper(),
        "bid_price": bid,
        "ask_price": ask,
        "mid_price": (bid + ask) / 2.0,
        "bid_size": float(features.get(f"{side}_bid_size") or 0.0),
        "ask_size": float(features.get(f"{side}_ask_size") or 0.0),
        "liquidity_depth": float(features.get(f"{side}_liquidity_depth") or 0.0),
        "available_at_ts": decision_ts - staleness,
        "ts": decision_ts - staleness,
    }


def _residual_row(
    *,
    selected: dict[str, Any],
    group_scored_rows: list[dict[str, Any]],
    public_row: dict[str, Any],
    target: dict[str, Any],
    protocol_hash: str,
) -> dict[str, Any]:
    action = str(selected["action"])
    side = "UP" if "BUY_UP" in action else "DOWN"
    micro = _selected_action_microstructure(public_row, action)
    selected_probability = (
        float(public_row["p_up"])
        if side == "UP"
        else float(public_row["p_down"])
    )
    execution_price = float(micro["entry_ask"])
    chainlink_provenance = dict(
        public_row.get("chainlink_regime_feature_provenance") or {}
    )
    max_input_ts = max(
        int(public_row["decision_time_feature_max_input_ts"]),
        int(chainlink_provenance.get("max_input_ts") or 0),
        int(selected.get("canonical_feature_mapping_max_input_ts") or 0),
    )
    decision_ts = int(public_row["decision_ts"])
    selected_score = float(selected["canonical_corrected_model_score"])
    second_best_score = max(
        float(row["canonical_corrected_model_score"])
        for row in group_scored_rows
        if str(row["action"]) != action
    )
    label = target["selected_action_labels"][action]
    canonical_ranking = [
        _fresh_public_ranking_row_from_canonical(row)
        for row in sorted(
            group_scored_rows,
            key=lambda row: (
                int(row.get("canonical_rank") or 999),
                str(row.get("action") or ""),
            ),
        )
    ]
    selected_ranking = next(
        row for row in canonical_ranking if row["selected_action"] == action
    )
    features = {
        "canonical_o_action_score": selected_score,
        "action_score_margin": selected_score - second_best_score,
        "btc_momentum": float(public_row["chainlink_momentum_60s"]),
        "reference_price_to_beat_distance_at_decision": float(
            public_row["chainlink_reference_distance_at_decision"]
        ),
        "chainlink_momentum_30s": float(public_row["chainlink_momentum_30s"]),
        "chainlink_momentum_60s": float(public_row["chainlink_momentum_60s"]),
        "chainlink_momentum_120s": float(public_row["chainlink_momentum_120s"]),
        "chainlink_realized_volatility_120s": float(
            public_row["chainlink_realized_volatility_120s"]
        ),
        "selected_side_probability": selected_probability,
        "execution_price": execution_price,
        "selected_side_probability_minus_execution_price": (
            selected_probability - execution_price
        ),
        "spread_bps": float(micro["spread_bps"]),
        "queue_fill_proxy": float(micro["queue_fill_proxy"]),
        "book_staleness_ms": float(micro["book_staleness_ms"]),
        "time_to_close_seconds": float(micro["time_to_close_seconds"]),
        "side_book_depth_imbalance": _side_depth_imbalance(public_row, side),
        "side_book_update_count_1m": _side_feature(
            public_row, side, "recent_book_update_count_1m"
        ),
        "side_recent_spread_stability_1m": _side_feature(
            public_row, side, "recent_spread_stability_1m"
        ),
        "cumulative_market_exposure_before_entry": 0.0,
        "same_side_reentry": 0.0,
        "side_flip": 0.0,
    }
    row = {
        "market_id": str(public_row["market_id"]),
        "condition_id": str(public_row["condition_id"]),
        "market_slug": str(public_row["slug"]),
        "decision_ts": decision_ts,
        "market_close_ts": int(public_row["market_end_ts"]),
        "max_input_ts": max_input_ts,
        "selected_action": action,
        "selected_side": side,
        "action_family": "HOLD_TO_SETTLEMENT",
        "decision_time_features": features,
        "selected_side_win_target": 1
        if target["resolved_outcome"] == side
        else 0,
        "target_net_return_after_cost": float(label["total_net_return"]),
        "evaluation_target_net_return_after_cost_by_action": {
            action_name: float(action_label["total_net_return"])
            for action_name, action_label in sorted(
                target["selected_action_labels"].items()
            )
        },
        "evaluation_target_net_pnl_per_contract_by_action": {
            action_name: float(action_label["total_net_pnl_per_notional"])
            for action_name, action_label in sorted(
                target["selected_action_labels"].items()
            )
        },
        "evaluation_target_pnl_components_by_action": {
            action_name: _evaluation_pnl_components(action_label)
            for action_name, action_label in sorted(
                target["selected_action_labels"].items()
            )
        },
        "execution_handoff_context": {
            "decision_group_id": public_row.get("decision_group_id"),
            "market_id": str(public_row["market_id"]),
            "decision_ts": decision_ts,
            "selected_action": action,
            "selected_side": side,
            "selected_action_family": "HOLD_TO_SETTLEMENT",
            "full_5_action_ranking": canonical_ranking,
            "corrected_model_score": selected_score,
            "raw_model_score": selected.get("canonical_raw_model_score"),
            "high_score_flag": bool(selected.get("high_score_flag")),
            "p_up": float(public_row["p_up"]),
            "p_down": float(public_row["p_down"]),
            "p_up_action_disagreement": bool(
                selected.get("p_up_action_disagreement")
            ),
            "microstructure_snapshot": selected_ranking[
                "microstructure_snapshot"
            ],
            "reference_price_feature_provenance": chainlink_provenance,
            "decision_time_feature_max_input_ts": max_input_ts,
        },
        "target_outcome_available_only_post_resolution": True,
        "target_provenance": {
            "source_type": "phase2_official_read_only_resolution",
            "resolved_outcome": target["resolved_outcome"],
            "raw_resolution_sha256": target["raw_resolution_sha256"],
            "outcome_used_as_training_target_only": True,
            "outcome_used_as_decision_input": False,
        },
        "source_run_id": target["source_run_id"],
        "source_lineage": {
            "source_corpus_dir": target["source_corpus_dir"],
            "source_corpus_manifest_sha256": target[
                "source_corpus_manifest_sha256"
            ],
            "source_chainlink_evidence_sha256": target[
                "source_chainlink_evidence_sha256"
            ],
            "source_feature_rows_sha256": target["source_feature_rows_sha256"],
            "source_label_rows_sha256": target["source_label_rows_sha256"],
            "market_probability_mapping_rule_id": public_row[
                "market_probability_mapping_rule_id"
            ],
            "frozen_development_protocol_sha256": protocol_hash,
            "frozen_o_scored_action_row_sha256": selected[
                "canonical_scored_action_row_hash"
            ],
        },
        "chainlink_feature_provenance": chainlink_provenance,
        "lineage": "post_protocol_development_only",
        "development_evidence_only": True,
        "unseen_validation_eligible": False,
        "future_confirmatory_validation_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    row["row_identity"] = canonical_json_sha256(
        {
            "market_id": row["market_id"],
            "decision_ts": decision_ts,
            "selected_action": action,
            "source_run_id": row["source_run_id"],
        }
    )
    row["row_content_sha256"] = canonical_json_sha256(row)
    return row


def _selected_action_microstructure(
    public_row: dict[str, Any], action: str
) -> dict[str, Any]:
    for row in public_row.get("full_5_action_ranking") or []:
        if row.get("selected_action") == action:
            return dict(row.get("microstructure_snapshot") or {})
    raise ValueError(f"selected action missing from ranking: {action}")


def _side_depth_imbalance(public_row: dict[str, Any], side: str) -> float:
    raw = dict(public_row.get("phase2_features") or {})
    own = float(raw.get(f"{side.lower()}_bid_size") or 0.0)
    opposite_side = "down" if side == "UP" else "up"
    opposite = float(raw.get(f"{opposite_side}_bid_size") or 0.0)
    denominator = own + opposite
    return (own - opposite) / denominator if denominator > 0.0 else 0.0


def _side_feature(public_row: dict[str, Any], side: str, suffix: str) -> float:
    raw = dict(public_row.get("phase2_features") or {})
    return float(raw.get(f"{side.lower()}_{suffix}") or 0.0)


def _forbidden_decision_fields(payload: Any) -> list[str]:
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key) in FORBIDDEN_DECISION_FIELDS:
                    found.add(str(key))
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return sorted(found)


def _chainlink_feature_coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "chainlink_momentum_30s",
        "chainlink_momentum_60s",
        "chainlink_momentum_120s",
        "chainlink_realized_volatility_120s",
        "reference_price_to_beat_distance_at_decision",
    )
    return {
        field: sum(
            isinstance(row["decision_time_features"].get(field), int | float)
            and math.isfinite(float(row["decision_time_features"][field]))
            for row in rows
        )
        for field in fields
    }


def _source_chainlink_feature_coverage(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    source_fields = {
        "chainlink_momentum_30s": "chainlink_momentum_30s",
        "chainlink_momentum_60s": "chainlink_momentum_60s",
        "chainlink_momentum_120s": "chainlink_momentum_120s",
        "chainlink_realized_volatility_120s": (
            "chainlink_realized_volatility_120s"
        ),
        "reference_price_to_beat_distance_at_decision": (
            "chainlink_reference_distance_at_decision"
        ),
    }
    return {
        report_field: sum(
            isinstance(row.get(source_field), int | float)
            and math.isfinite(float(row[source_field]))
            for row in rows
        )
        for report_field, source_field in source_fields.items()
    }


def _validate_protocol(protocol: dict[str, Any]) -> None:
    schema_version = protocol.get("schema_version")
    if schema_version not in {
        "bigan-v8-hts-residual-development-protocol-v2",
        "bigan-v8-hts-residual-development-protocol-v3",
    }:
        raise ValueError("unsupported residual development protocol")
    if protocol.get("protocol_frozen_before_new_development_collection") is not True:
        raise ValueError("development protocol is not frozen")
    if protocol.get("uses_validation_labels_for_tuning") is not False:
        raise ValueError("development protocol permits validation-label tuning")
    if protocol.get("future_confirmatory_validation_start_allowed") is not False:
        raise ValueError("development protocol unexpectedly unlocks confirmatory data")
    if schema_version.endswith("-v3"):
        mapping = dict(protocol.get("market_probability_mapping_contract") or {})
        if mapping.get("rule_id") != PHASE2_MARKET_PROBABILITY_MAPPING_RULE_ID:
            raise ValueError("development protocol market-probability rule mismatch")
        if mapping.get("uses_future_or_settlement_fields") is not False:
            raise ValueError("market-probability mapping permits future fields")


def _rejection(
    market_id: str, decision_ts: int, reason_code: str, **extra: Any
) -> dict[str, Any]:
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "reason_code": reason_code,
        **extra,
    }


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# HTS Residual New Development Corpus",
            "",
            f"- status: `{report['status']}`",
            f"- source corpora: `{report['source_corpus_count']}`",
            f"- decision rows: `{report['source_decision_row_count']}`",
            f"- residual rows / markets: `{report['residual_row_count']}` / `{report['residual_market_count']}`",
            f"- forward OOF ready: `{str(report['forward_oof_evaluation_ready']).lower()}`",
            f"- forward OOF blockers: `{report['forward_oof_blocking_reason_codes']}`",
            f"- feature causality violations: `{report['feature_causality_violation_count']}`",
            f"- source Chainlink coverage: `{report['source_chainlink_feature_coverage']}`",
            f"- residual Chainlink coverage: `{report['residual_chainlink_feature_coverage']}`",
            f"- rejected reasons: `{report['rejected_reason_distribution']}`",
            "- candidate fit attempted: `false`",
            "- confirmatory validation started: `false`",
            "- paper/live/promotion unlock: `false`",
            "",
        ]
    )


def _forward_oof_markdown(report: dict[str, Any]) -> str:
    gate = report["development_candidate_gate"]
    lines = [
        "# HTS Residual Development Forward OOF",
        "",
        f"- status: `{report['status']}`",
        f"- rows / markets: `{report['row_count']}` / `{report['market_count']}`",
        f"- selected candidate: `{report['selected_candidate_name']}`",
        f"- development gate passed: `{str(gate['passed']).lower()}`",
        f"- blocking reasons: `{gate['blocking_reason_codes']}`",
        "- candidate frozen: `false`",
        "- confirmatory validation started: `false`",
        "- source/freeze/promotion/paper/live unlock: `false`",
        "",
    ]
    if report["candidate_reports"]:
        lines.extend(
            [
                "| candidate | OOF markets | Brier | log loss | rel Brier | rel log loss |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for candidate in report["candidate_reports"]:
            metrics = candidate["candidate_metrics"]
            lines.append(
                "| {name} | {markets} | {brier:.6f} | {log_loss:.6f} | "
                "{rel_brier:.6f} | {rel_log:.6f} |".format(
                    name=candidate["candidate_name"],
                    markets=candidate["forward_oof_market_count"],
                    brier=metrics["market_weighted_brier_score"],
                    log_loss=metrics["market_weighted_log_loss"],
                    rel_brier=candidate["relative_brier_improvement_vs_raw"],
                    rel_log=candidate["relative_log_loss_improvement_vs_raw"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def _forward_oof_pnl_markdown(report: dict[str, Any]) -> str:
    baseline = report["raw_market_probability_baseline_policy_metrics"]
    lines = [
        "# HTS Residual Forward OOF Execution-Bet PnL Diagnostic",
        "",
        f"- status: `{report['status']}`",
        f"- selected candidate: `{report['selected_candidate_name']}`",
        "- diagnostic conclusion: "
        f"`{report['execution_pnl_diagnostic_conclusion']}`",
        "- path: `OOF probability -> frozen EV threshold -> frozen execution guard "
        "-> simulated paper bet -> settlement net PnL`",
        "- outcomes used for selection: `false`",
        "- PnL used for tuning/ranking: `false`",
        "- development calibration gate changed: `false`",
        "- promotion/paper/live unlock: `false`",
        "",
        "| policy | signal candidates | guard-passed bets | cost basis | "
        "settled PnL | ROI | max drawdown |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| raw market probability baseline | {signals} | {bets} | {cost:.6f} | "
        "{pnl:.6f} | {roi:.6f} | {drawdown:.6f} |".format(
            signals=baseline["model_signal_candidate_count"],
            bets=baseline["execution_bet_count"],
            cost=baseline["cost_basis"],
            pnl=baseline["settled_pnl_sum"],
            roi=baseline["roi_on_cost_basis"],
            drawdown=baseline["maximum_drawdown"],
        ),
    ]
    for comparison in report["candidate_comparisons"]:
        metrics = comparison["candidate_policy_metrics"]
        lines.append(
            "| {name} | {signals} | {bets} | {cost:.6f} | {pnl:.6f} | "
            "{roi:.6f} | {drawdown:.6f} |".format(
                name=comparison["candidate_name"],
                signals=metrics["model_signal_candidate_count"],
                bets=metrics["execution_bet_count"],
                cost=metrics["cost_basis"],
                pnl=metrics["settled_pnl_sum"],
                roi=metrics["roi_on_cost_basis"],
                drawdown=metrics["maximum_drawdown"],
            )
        )
    lines.extend(
        [
            "",
            "This is an outcome-aware development diagnostic on deterministic "
            "simulated paper bets. It is not actual exchange-fill PnL and is not "
            "promotion evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
