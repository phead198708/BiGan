"""Outcome-blind future-batch canaries for the frozen v6.2 candidate."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    _raw_target_stripped_predictions,
)
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2 import (
    CANDIDATE_NAME,
    apply_market_clustered_mean_ev_scores,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    _outcome_blind_acceptance_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    _prepare_run_dir,
    _result,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    SIDES,
    _blocked_safety_fields,
    _descriptor,
    _find_nonempty_fields,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    TARGET_FIELDS,
    attach_frozen_execution_compatibility,
    select_sequential_policy_rows,
)

SCHEMA_PREFIX = "bigan-v8-market-clustered-mean-ev-v6-2-future-batch-canary"
FUTURE_QUALITY_VALID_MARKET_TARGET = 200
FUTURE_ATTEMPT_SCAN_CAP = 240
MINIMUM_ACCEPTED_UNIQUE_MARKETS = 120
MINIMUM_ACCEPTED_UNIQUE_MARKETS_PER_SIDE = 17
CONSECUTIVE_ZERO_ACTION_BATCH_LIMIT = 2
CONSECUTIVE_ZERO_ACTION_QUALITY_MARKET_MINIMUM = 24


@dataclass(frozen=True, slots=True)
class MarketClusteredMeanEVV62FutureBatchCanaryConfig:
    """Pinned inputs for one strictly-later v6.2 batch canary."""

    run_id: str
    output_dir: Path | str
    development_batch_canary_manifest_path: Path | str
    expected_development_batch_canary_manifest_sha256: str
    candidate_manifest_path: Path | str
    expected_candidate_manifest_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_development_batch_canary_manifest_sha256,
            name="development batch canary manifest sha256",
        )
        _require_sha256(
            self.expected_candidate_manifest_sha256,
            name="candidate manifest sha256",
        )
        for name in (
            "output_dir",
            "development_batch_canary_manifest_path",
            "candidate_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def run_market_clustered_mean_ev_v6_2_future_batch_canary(
    config: MarketClusteredMeanEVV62FutureBatchCanaryConfig,
) -> dict[str, Any]:
    """Score one complete strictly-later batch without opening any target."""

    development_manifest_path = config.development_batch_canary_manifest_path.resolve()
    candidate_manifest_path = config.candidate_manifest_path.resolve()
    _verify_pin(
        development_manifest_path,
        config.expected_development_batch_canary_manifest_sha256,
        "development batch canary manifest",
    )
    _verify_pin(
        candidate_manifest_path,
        config.expected_candidate_manifest_sha256,
        "v6.2 candidate manifest",
    )
    development_manifest = _load_json(development_manifest_path)
    candidate_manifest = _load_json(candidate_manifest_path)
    _validate_candidate_manifest(candidate_manifest)
    if development_manifest.get("development_data_canary_passed") is not True:
        raise ValueError("development_batch_canary_not_passed")
    if development_manifest.get("labels_outcomes_or_pnl_opened") is not False:
        raise ValueError("development_batch_target_sealing_invalid")
    development_report_descriptor = _verified_descriptor(
        development_manifest.get("report"), "development batch report"
    )
    development_report = _load_json(Path(development_report_descriptor["path"]))
    if development_report.get("development_data_canary_passed") is not True:
        raise ValueError("development_batch_report_not_passed")
    action_descriptor = _verified_descriptor(
        development_manifest.get("five_action_grid"), "five action grid"
    )
    action_rows = _load_jsonl(Path(action_descriptor["path"]))
    if _find_nonempty_fields(action_rows, TARGET_FIELDS):
        raise ValueError("future_batch_contains_forbidden_target_fields")

    freeze_ts = int(candidate_manifest["future_collection_minimum_created_ts_exclusive"])
    future_time_violations = [
        row
        for row in action_rows
        if int(row.get("decision_ts") or 0) <= freeze_ts
        or int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0)
    ]
    if future_time_violations:
        raise ValueError("future_batch_not_strictly_later_or_causal")
    prior_market_ids, prior_reference_hash = _prior_market_reference(candidate_manifest)
    future_market_ids = {str(row.get("market_id") or "") for row in action_rows}
    if "" in future_market_ids:
        raise ValueError("future_batch_market_identity_missing")
    overlap = sorted(future_market_ids & prior_market_ids)
    if overlap:
        raise ValueError("future_batch_market_overlap_with_candidate_lineage")

    pre_audit_descriptor = _verified_descriptor(
        candidate_manifest.get("pre_target_access_audit"), "pre-target access audit"
    )
    pre_audit = _load_json(Path(pre_audit_descriptor["path"]))
    feature_contract_descriptor = _verified_descriptor(
        pre_audit.get("feature_contract"), "feature contract"
    )
    model_descriptor = _verified_descriptor(
        candidate_manifest.get("source_model"), "source model"
    )
    calibration_descriptor = _verified_descriptor(
        candidate_manifest.get("market_clustered_mean_risk_calibration"),
        "market-clustered mean-risk calibration",
    )
    feature_contract = _load_json(Path(feature_contract_descriptor["path"]))
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    calibration = _load_json(Path(calibration_descriptor["path"]))
    if calibration.get("frozen") is not True or calibration.get("calibration_gate_passed") is not True:
        raise ValueError("v6_2_calibration_not_frozen_or_passed")

    booster = xgb.Booster()
    booster.load_model(model_descriptor["path"])
    raw_predictions = _raw_target_stripped_predictions(
        booster,
        action_rows,
        feature_columns=feature_columns,
    )
    compatible = attach_frozen_execution_compatibility(raw_predictions)
    scored = apply_market_clustered_mean_ev_scores(
        compatible,
        calibration_artifact=calibration,
    )
    selected = select_sequential_policy_rows(
        scored,
        score_field="mean_ev_lower_confidence_bound",
        require_positive=True,
    )
    replay = _outcome_blind_acceptance_replay(
        scored,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    accepted = [row for row in replay if row["execution_guard_order_allowed"]]
    selected_markets = {str(row["market_id"]) for row in selected}
    accepted_markets = {str(row["market_id"]) for row in accepted}
    accepted_by_side = {
        side: sorted(
            {
                str(row["market_id"])
                for row in accepted
                if str(row.get("selected_side") or "") == side
            }
        )
        for side in SIDES
    }
    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    scored_path = run_dir / "v6_2_future_batch_mean_ev_scored_rows.jsonl"
    selected_path = run_dir / "v6_2_future_batch_selected_rows.jsonl"
    replay_path = run_dir / "v6_2_future_batch_full_guard_replay.jsonl"
    accepted_path = run_dir / "v6_2_future_batch_guard_accepted_rows.jsonl"
    _write_jsonl(scored_path, scored)
    _write_jsonl(selected_path, selected)
    _write_jsonl(replay_path, replay)
    _write_jsonl(accepted_path, accepted)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "batch_id": development_manifest["batch_id"],
        "candidate_name": CANDIDATE_NAME,
        "candidate_manifest_sha256": config.expected_candidate_manifest_sha256.lower(),
        "candidate_freeze_created_ts": freeze_ts,
        "future_strictly_later_and_disjoint_passed": True,
        "prior_reference_market_count": len(prior_market_ids),
        "prior_reference_hash": prior_reference_hash,
        "bounded_batch_complete": development_report["bounded_batch_complete"],
        "source_sequence_start": int(development_report["source_sequence_start"]),
        "source_sequence_end": int(development_report["source_sequence_end"]),
        "indexed_market_count": int(development_report["indexed_market_count"]),
        "quality_valid_market_count": len(future_market_ids),
        "decision_group_count": len(replay),
        "action_row_count": len(action_rows),
        "positive_guard_compatible_trade_lcb_row_count": len(selected),
        "positive_mean_ev_lcb_unique_market_count": len(selected_markets),
        "positive_mean_ev_lcb_side_market_count": _selected_side_counts(selected),
        "positive_mean_ev_lcb_action_distribution": dict(
            sorted(Counter(str(row["action"]) for row in selected).items())
        ),
        "selected_no_positive_action_market_count": len(future_market_ids - selected_markets),
        "guard_accepted_decision_count": len(accepted),
        "guard_accepted_unique_market_count": len(accepted_markets),
        "guard_accepted_market_ids": sorted(accepted_markets),
        "guard_accepted_market_ids_by_side": accepted_by_side,
        "guard_accepted_by_side": {
            side: len(accepted_by_side[side]) for side in SIDES
        },
        "guard_accepted_action_distribution": dict(
            sorted(Counter(str(row["executed_action"]) for row in accepted).items())
        ),
        "guard_blocking_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in replay
                    for reason in row["execution_blocking_reason_codes"]
                ).items()
            )
        ),
        "candidate_model_scoring_attempted": True,
        "target_free_scoring_passed": True,
        "labels_outcomes_or_pnl_opened": False,
        "settlement_provider_called": False,
        "threshold_or_guard_tuning_performed": False,
        "model_or_source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_2_future_batch_action_canary_report.json"
    report_md_path = run_dir / "v6_2_future_batch_action_canary_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _batch_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "batch_id": report["batch_id"],
        "development_batch_canary_manifest": _descriptor(development_manifest_path),
        "candidate_manifest": _descriptor(candidate_manifest_path),
        "feature_contract": feature_contract_descriptor,
        "source_model": model_descriptor,
        "market_clustered_mean_risk_calibration": calibration_descriptor,
        "mean_ev_scored_rows": _descriptor(scored_path),
        "selected_rows": _descriptor(selected_path),
        "full_guard_replay": _descriptor(replay_path),
        "guard_accepted_rows": _descriptor(accepted_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_2_future_batch_action_canary_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def build_v6_2_future_cumulative_canary(
    batch_reports: list[dict[str, Any]],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Aggregate frozen future batches and apply only preregistered target-free stops."""

    if not batch_reports:
        raise ValueError("at least one v6.2 future batch report is required")
    seen_batches: set[str] = set()
    seen_markets: set[str] = set()
    accepted_markets: set[str] = set()
    accepted_by_side: dict[str, set[str]] = {side: set() for side in SIDES}
    previous_end: int | None = None
    for report in batch_reports:
        if report.get("candidate_name") != CANDIDATE_NAME:
            raise ValueError("future_batch_candidate_identity_mismatch")
        if report.get("future_strictly_later_and_disjoint_passed") is not True:
            raise ValueError("future_batch_provenance_not_passed")
        if report.get("labels_outcomes_or_pnl_opened") is not False:
            raise ValueError("future_batch_target_sealing_invalid")
        batch_id = str(report.get("batch_id") or "")
        if not batch_id or batch_id in seen_batches:
            raise ValueError("future_batch_identity_missing_or_duplicate")
        seen_batches.add(batch_id)
        start = int(report.get("source_sequence_start") or 0)
        end = int(report.get("source_sequence_end") or 0)
        if start <= 0 or end < start:
            raise ValueError("future_batch_sequence_invalid")
        if previous_end is not None and start != previous_end + 1:
            raise ValueError("future_batch_sequence_not_contiguous")
        previous_end = end
        batch_market_ids = set(report.get("guard_accepted_market_ids") or [])
        if seen_markets & batch_market_ids:
            raise ValueError("future_batch_market_identity_repeated")
        seen_markets.update(batch_market_ids)
        accepted_markets.update(batch_market_ids)
        for side in SIDES:
            accepted_by_side[side].update(
                str(value)
                for value in dict(report["guard_accepted_market_ids_by_side"]).get(side, [])
            )
        for key, expected in _blocked_safety_fields().items():
            if report.get(key) != expected:
                raise ValueError(f"future_batch_safety_invalid:{key}")

    attempted_count = sum(int(report["indexed_market_count"]) for report in batch_reports)
    quality_count = sum(int(report["quality_valid_market_count"]) for report in batch_reports)
    remaining_attempt_capacity = max(0, FUTURE_ATTEMPT_SCAN_CAP - attempted_count)
    trailing_zero_reports: list[dict[str, Any]] = []
    for report in reversed(batch_reports):
        if (
            report.get("bounded_batch_complete") is True
            and int(report["guard_accepted_unique_market_count"]) == 0
            and int(report["positive_mean_ev_lcb_unique_market_count"]) == 0
        ):
            trailing_zero_reports.append(report)
        else:
            break
    trailing_zero_quality = sum(
        int(report["quality_valid_market_count"]) for report in trailing_zero_reports
    )
    blockers: list[str] = []
    if (
        len(trailing_zero_reports) >= CONSECUTIVE_ZERO_ACTION_BATCH_LIMIT
        and trailing_zero_quality >= CONSECUTIVE_ZERO_ACTION_QUALITY_MARKET_MINIMUM
    ):
        blockers.append("two_consecutive_complete_batches_zero_v6_2_actions")
    if quality_count + remaining_attempt_capacity < FUTURE_QUALITY_VALID_MARKET_TARGET:
        blockers.append("remaining_scan_capacity_cannot_reach_future_quality_target")
    if len(accepted_markets) + remaining_attempt_capacity < MINIMUM_ACCEPTED_UNIQUE_MARKETS:
        blockers.append("remaining_scan_capacity_cannot_reach_accepted_market_support")
    for side in SIDES:
        if (
            len(accepted_by_side[side]) + remaining_attempt_capacity
            < MINIMUM_ACCEPTED_UNIQUE_MARKETS_PER_SIDE
        ):
            blockers.append(
                f"remaining_scan_capacity_cannot_reach_{side.lower()}_accepted_support"
            )
    collection_complete = quality_count >= FUTURE_QUALITY_VALID_MARKET_TARGET
    if attempted_count >= FUTURE_ATTEMPT_SCAN_CAP and not collection_complete:
        blockers.append("future_quality_target_not_met_within_scan_cap")
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-cumulative-report-v1",
        "run_id": run_id,
        "candidate_name": CANDIDATE_NAME,
        "batch_count": len(batch_reports),
        "batch_ids": [str(report["batch_id"]) for report in batch_reports],
        "attempted_market_count": attempted_count,
        "quality_valid_market_count": quality_count,
        "future_quality_valid_market_target": FUTURE_QUALITY_VALID_MARKET_TARGET,
        "future_attempt_scan_cap": FUTURE_ATTEMPT_SCAN_CAP,
        "guard_accepted_unique_market_count": len(accepted_markets),
        "guard_accepted_unique_market_count_by_side": {
            side: len(accepted_by_side[side]) for side in SIDES
        },
        "minimum_accepted_unique_market_count": MINIMUM_ACCEPTED_UNIQUE_MARKETS,
        "minimum_accepted_unique_market_count_per_side": (
            MINIMUM_ACCEPTED_UNIQUE_MARKETS_PER_SIDE
        ),
        "remaining_attempt_capacity": remaining_attempt_capacity,
        "consecutive_zero_action_batch_count": len(trailing_zero_reports),
        "consecutive_zero_action_quality_market_count": trailing_zero_quality,
        "future_holdout_collection_complete": collection_complete,
        "future_holdout_exact_earliest_200_freeze_required": True,
        "future_pnl_evaluation_allowed": False,
        "collector_should_stop": collection_complete or bool(blockers),
        "target_free_terminal_blocked": bool(blockers),
        "target_free_terminal_blocking_reason_codes": sorted(set(blockers)),
        "labels_outcomes_or_pnl_opened": False,
        "settlement_provider_called": False,
        "threshold_or_guard_tuning_performed": False,
        "model_or_source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def write_v6_2_future_cumulative_canary(
    *,
    report: dict[str, Any],
    batch_report_paths: list[Path],
    output_dir: Path,
    run_id: str,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """Persist cumulative support without opening future outcomes."""

    run_dir = _prepare_run_dir(output_dir, run_id, overwrite=overwrite_existing)
    report_path = run_dir / "v6_2_future_cumulative_action_canary_report.json"
    report_md_path = run_dir / "v6_2_future_cumulative_action_canary_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _cumulative_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-cumulative-manifest-v1",
        "run_id": run_id,
        "batch_reports": [_descriptor(path.resolve()) for path in batch_report_paths],
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "future_holdout_collection_complete": report["future_holdout_collection_complete"],
        "collector_should_stop": report["collector_should_stop"],
        "target_free_terminal_blocked": report["target_free_terminal_blocked"],
        "target_free_terminal_blocking_reason_codes": report[
            "target_free_terminal_blocking_reason_codes"
        ],
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_2_future_cumulative_action_canary_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _validate_candidate_manifest(manifest: dict[str, Any]) -> None:
    checks = {
        "candidate": manifest.get("candidate_name") == CANDIDATE_NAME,
        "actionability": manifest.get("target_free_actionability_gate_passed") is True,
        "frozen": manifest.get("research_actionability_candidate_frozen") is True,
        "collector_resume": manifest.get("collector_resume_allowed") is True,
        "future_required": manifest.get("new_strictly_later_future_holdout_required") is True,
        "sealed": (
            manifest.get("target_free_labels_outcomes_settlement_targets_or_pnl_opened")
            is False
        ),
    }
    for key, expected in _blocked_safety_fields().items():
        checks[f"safety_{key}"] = manifest.get(key) == expected
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError("v6_2_candidate_manifest_invalid:" + ",".join(failed))


def _prior_market_reference(manifest: dict[str, Any]) -> tuple[set[str], str]:
    pre_audit_descriptor = _verified_descriptor(
        manifest.get("pre_target_access_audit"), "pre-target access audit"
    )
    pre_audit = _load_json(Path(pre_audit_descriptor["path"]))
    v5_descriptor = _verified_descriptor(pre_audit.get("v5_freeze_manifest"), "v5 manifest")
    v5 = _load_json(Path(v5_descriptor["path"]))
    issue209_descriptor = _verified_descriptor(
        manifest.get("source_issue209_manifest"), "issue209 manifest"
    )
    issue209 = _load_json(Path(issue209_descriptor["path"]))
    descriptors = [
        _verified_descriptor(v5.get("development_train_action_rows"), "v5 train rows"),
        _verified_descriptor(
            v5.get("development_calibration_action_rows"), "v5 calibration rows"
        ),
        _verified_descriptor(
            issue209.get("target_free_five_action_rows"), "issue209 target-free rows"
        ),
    ]
    market_ids = {
        str(row["market_id"])
        for descriptor in descriptors
        for row in _load_jsonl(Path(descriptor["path"]))
    }
    return market_ids, canonical_json_sha256(sorted(market_ids))


def _selected_side_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        side: len({str(row["market_id"]) for row in rows if str(row["side"]) == side})
        for side in SIDES
    }


def _batch_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.2 strictly-later outcome-blind batch canary",
            "",
            f"- Batch: `{report['batch_id']}`",
            f"- Quality-valid markets: `{report['quality_valid_market_count']}`",
            f"- Positive mean-EV-LCB markets: `{report['positive_mean_ev_lcb_unique_market_count']}`",
            f"- Positive sides: `{report['positive_mean_ev_lcb_side_market_count']}`",
            f"- Full-guard accepted markets: `{report['guard_accepted_unique_market_count']}`",
            f"- Accepted sides: `{report['guard_accepted_by_side']}`",
            "- Strictly later/disjoint: `true`",
            "- Outcomes/labels/PnL opened: `false`",
            "- Paper/live/promotion unlock: `false`",
            "",
        ]
    )


def _cumulative_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.2 future cumulative action canary",
            "",
            f"- Batches: `{report['batch_count']}`",
            f"- Quality-valid markets: `{report['quality_valid_market_count']}` / `{report['future_quality_valid_market_target']}`",
            f"- Accepted unique markets: `{report['guard_accepted_unique_market_count']}`",
            f"- Accepted by side: `{report['guard_accepted_unique_market_count_by_side']}`",
            f"- Collection complete: `{str(report['future_holdout_collection_complete']).lower()}`",
            f"- Terminal blocked: `{str(report['target_free_terminal_blocked']).lower()}`",
            f"- Blockers: `{report['target_free_terminal_blocking_reason_codes']}`",
            "- Outcomes/labels/PnL opened: `false`",
            "- Exact earliest-200 freeze required before evaluation: `true`",
            "",
        ]
    )
