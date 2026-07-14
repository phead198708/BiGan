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
    _fresh_public_row_from_provider_feature_context,
    score_frozen_o_decision_rows,
)

SCHEMA_PREFIX = "bigan-v8-hts-residual-development-corpus"
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
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "status": "DEVELOPMENT_CORPUS_READY"
        if market_count >= minimum_markets
        else "DEVELOPMENT_CORPUS_SUPPORT_INSUFFICIENT",
        "protocol_path": str(Path(config.protocol_path).resolve()),
        "protocol_sha256": protocol_hash,
        "protocol_frozen_before_included_rows": True,
        "source_corpus_count": len(config.source_corpus_dirs),
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
        "source_chainlink_feature_coverage": _source_chainlink_feature_coverage(
            public_rows
        ),
        "residual_chainlink_feature_coverage": _chainlink_feature_coverage(
            residual_rows
        ),
        "chainlink_feature_coverage_scope": "residual_hts_rows",
        "chainlink_feature_coverage": _chainlink_feature_coverage(residual_rows),
        "minimum_development_market_count": minimum_markets,
        "forward_oof_evaluation_ready": market_count >= minimum_markets,
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
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
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
    return {
        "outcome": side.upper(),
        "bid_price": float(features[f"{side}_bid"]),
        "ask_price": float(features[f"{side}_ask"]),
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
    if protocol.get("schema_version") != (
        "bigan-v8-hts-residual-development-protocol-v2"
    ):
        raise ValueError("unsupported residual development protocol")
    if protocol.get("protocol_frozen_before_new_development_collection") is not True:
        raise ValueError("development protocol is not frozen")
    if protocol.get("uses_validation_labels_for_tuning") is not False:
        raise ValueError("development protocol permits validation-label tuning")
    if protocol.get("future_confirmatory_validation_start_allowed") is not False:
        raise ValueError("development protocol unexpectedly unlocks confirmatory data")


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
