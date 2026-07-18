"""Outcome-blind accepted-bet viability audit for a frozen pairwise candidate."""

from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    validate_pairwise_action_advantage_lcb_feature_contract,
    validate_pairwise_action_advantage_lcb_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (
    FORBIDDEN_DECISION_FIELDS,
    _apply_action_advantage_lcb_scores,
    _blocked_safety_fields,
    _decision_features,
    _descriptor,
    _find_fields,
    _load_json,
    _load_jsonl,
    _p_up,
    _predict_role_rows,
    _release_closed_positions,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_apply_simulated_order_to_state,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
    _v8_initial_runtime_state,
)

SCHEMA_PREFIX = "bigan-v8-outcome-blind-accepted-bet-viability"
AUDITED_ROLE = "development_calibration"
ALLOWED_FEATURE_FILENAME = "polymarket_feature_rows.jsonl"
FORBIDDEN_OUTCOME_FIELDS = frozenset(
    set(FORBIDDEN_DECISION_FIELDS)
    | {
        "accepted_bet_net_pnl",
        "evaluation_target_net_pnl_per_contract_by_action",
        "final_outcome",
        "gross_pnl",
        "label",
        "net_pnl",
        "outcome",
        "outcomePrices",
        "payout_down",
        "payout_up",
        "resolved_outcome",
        "settlement_outcome",
        "target_cost_components",
        "target_net_pnl_per_contract",
        "target_resolved_outcome",
        "winning_outcome",
    }
)
TRADE_FAMILIES = ("HOLD_TO_SETTLEMENT", "SELL_BEFORE_CLOSE")


@dataclass(frozen=True, slots=True)
class OutcomeBlindAcceptanceViabilityConfig:
    """Pinned inputs for decision-time-only replay of a frozen candidate."""

    run_id: str
    output_dir: Path | str
    candidate_freeze_manifest_path: Path | str
    expected_candidate_freeze_manifest_sha256: str
    role_assignment_manifest_path: Path | str
    expected_role_assignment_manifest_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_candidate_freeze_manifest_sha256,
            name="candidate freeze manifest SHA-256",
        )
        _require_sha256(
            self.expected_role_assignment_manifest_sha256,
            name="role assignment manifest SHA-256",
        )
        for name in (
            "output_dir",
            "candidate_freeze_manifest_path",
            "role_assignment_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def run_outcome_blind_acceptance_viability_audit(
    config: OutcomeBlindAcceptanceViabilityConfig,
) -> dict[str, Any]:
    """Replay frozen inference and guard without opening target artifacts."""

    candidate_path = config.candidate_freeze_manifest_path.resolve()
    role_path = config.role_assignment_manifest_path.resolve()
    _verify_pin(
        candidate_path,
        config.expected_candidate_freeze_manifest_sha256,
        name="candidate freeze manifest",
    )
    _verify_pin(
        role_path,
        config.expected_role_assignment_manifest_sha256,
        name="role assignment manifest",
    )
    candidate = _load_json(candidate_path)
    role_manifest = _load_json(role_path)
    if candidate.get("role_assignment_manifest") != _descriptor(role_path):
        raise ValueError("candidate and supplied role assignment lineage mismatch")
    if role_manifest.get("role_assignment_ready") is not True:
        raise ValueError("role assignment is not ready")
    if role_manifest.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        raise ValueError("role assignment was not outcome blind")
    if not _safety_blocked(candidate, role_manifest):
        raise ValueError("input safety contract is not blocked")

    model_descriptor = _verified_descriptor(candidate.get("model"), name="model")
    calibration_descriptor = _verified_descriptor(
        candidate.get("action_advantage_lcb_calibration_artifact"),
        name="calibration artifact",
    )
    protocol_descriptor = _verified_descriptor(
        candidate.get("protocol"),
        name="pairwise protocol",
    )
    feature_contract_descriptor = _verified_descriptor(
        candidate.get("feature_contract"),
        name="feature contract",
    )
    protocol = _load_json(Path(protocol_descriptor["path"]))
    feature_contract = _load_json(Path(feature_contract_descriptor["path"]))
    calibration = _load_json(Path(calibration_descriptor["path"]))
    validate_pairwise_action_advantage_lcb_protocol(protocol)
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=protocol_descriptor["sha256"],
    )
    if not (
        candidate.get("candidate_name")
        == protocol.get("candidate_name")
        == calibration.get("candidate_name")
    ):
        raise ValueError("candidate identity mismatch across frozen inputs")
    _validate_frozen_calibration(calibration)
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    if calibration.get("feature_contract_sha256") != feature_contract_descriptor["sha256"]:
        raise ValueError("calibration feature contract lineage mismatch")

    selected_rows_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"),
        name="role assignment selected rows",
    )
    role_rows = _load_jsonl(Path(selected_rows_descriptor["path"]))
    forbidden_role_fields = _find_fields(
        {"rows": role_rows},
        set(FORBIDDEN_OUTCOME_FIELDS),
    )
    if forbidden_role_fields:
        raise ValueError("role rows contain forbidden outcome fields")
    audited_role_rows = [row for row in role_rows if str(row.get("role") or "") == AUDITED_ROLE]
    expected_markets = int(protocol["role_assignment"]["development_calibration_market_count"])
    if len(audited_role_rows) != expected_markets:
        raise ValueError("development calibration role market count mismatch")

    opened_feature_paths: list[dict[str, str]] = []
    action_rows: list[dict[str, Any]] = []
    source_feature_row_count = 0
    for role_row in audited_role_rows:
        feature_rows, descriptor = _load_outcome_blind_feature_rows(role_row)
        opened_feature_paths.append(descriptor)
        source_feature_row_count += len(feature_rows)
        action_rows.extend(
            _materialize_outcome_blind_action_rows(
                feature_rows,
                role_row=role_row,
                feature_columns=feature_columns,
            )
        )
    _validate_complete_action_grid(action_rows)
    booster = xgb.Booster()
    booster.load_model(model_descriptor["path"])
    predictions = _predict_role_rows(
        action_rows,
        booster=booster,
        feature_columns=feature_columns,
    )
    scored = _apply_action_advantage_lcb_scores(
        predictions,
        lcb_artifact=calibration,
    )
    execution_contract = dict(protocol["frozen_execution_contract"])
    viability_rows = _outcome_blind_acceptance_replay(
        scored,
        entry_threshold=float(execution_contract["entry_edge_threshold"]),
        runner_up_advantage_threshold=float(execution_contract["runner_up_advantage_threshold"]),
    )
    run_dir = _prepare_run_dir(
        config.output_dir,
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    rows_path = run_dir / "outcome_blind_accepted_bet_viability_rows.jsonl"
    _write_jsonl(rows_path, viability_rows)
    report = _viability_report(
        run_id=config.run_id,
        role_rows=audited_role_rows,
        source_feature_row_count=source_feature_row_count,
        action_rows=action_rows,
        viability_rows=viability_rows,
        protocol=protocol,
        candidate=candidate,
        opened_feature_paths=opened_feature_paths,
        input_descriptors={
            "candidate_freeze_manifest": _descriptor(candidate_path),
            "role_assignment_manifest": _descriptor(role_path),
            "role_assignment_selected_rows": selected_rows_descriptor,
            "model": model_descriptor,
            "calibration_artifact": calibration_descriptor,
            "protocol": protocol_descriptor,
            "feature_contract": feature_contract_descriptor,
        },
    )
    report_path = run_dir / "outcome_blind_accepted_bet_viability_report.json"
    markdown_path = run_dir / "outcome_blind_accepted_bet_viability_report.md"
    _write_json(report_path, report)
    _write_text(markdown_path, _report_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "candidate_freeze_manifest": _descriptor(candidate_path),
        "role_assignment_manifest": _descriptor(role_path),
        "role_assignment_selected_rows": selected_rows_descriptor,
        "model": model_descriptor,
        "calibration_artifact": calibration_descriptor,
        "protocol": protocol_descriptor,
        "feature_contract": feature_contract_descriptor,
        "opened_feature_rows": opened_feature_paths,
        "viability_rows": _descriptor(rows_path),
        "viability_report": _descriptor(report_path),
        "viability_report_markdown": _descriptor(markdown_path),
        "target_or_outcome_files_opened": False,
        "current_oof_or_validation_pnl_used": False,
        "threshold_sweep_performed": False,
        "model_or_score_mutated": False,
        "frozen_input_hashes_verified": True,
        "execution_guard_config_sha256": report["execution_guard_config_sha256"],
        "execution_guard_config_mutated": False,
        **_diagnostic_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "outcome_blind_accepted_bet_viability_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "rows_path": rows_path,
        "rows_sha256": _sha256_file(rows_path),
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "markdown_path": markdown_path,
        "markdown_sha256": _sha256_file(markdown_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _load_outcome_blind_feature_rows(
    role_row: dict[str, Any],
    *,
    allowed_corpus_root: Path = Path("/Volumes/PHILIPS/v8"),
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    descriptor = _verified_descriptor(
        role_row.get("feature_rows"),
        name="source feature rows",
    )
    path = Path(descriptor["path"])
    if path.name != ALLOWED_FEATURE_FILENAME:
        raise ValueError("forbidden non-feature artifact access attempted")
    if allowed_corpus_root.resolve() not in path.resolve().parents:
        raise ValueError("feature artifact is outside the direct training corpus root")
    rows = _load_jsonl(path)
    forbidden = _find_fields({"rows": rows}, set(FORBIDDEN_OUTCOME_FIELDS))
    if forbidden:
        raise ValueError("feature artifact contains forbidden outcome fields")
    expected_market_id = str(role_row.get("market_id") or "")
    if not rows or {str(row.get("market_id") or "") for row in rows} != {expected_market_id}:
        raise ValueError("feature artifact market identity mismatch")
    return rows, descriptor


def _validate_frozen_calibration(calibration: dict[str, Any]) -> None:
    checks = {
        "source_split": calibration.get("source_split") == "development_calibration_only",
        "confirmatory_labels_not_used": calibration.get(
            "uses_confirmatory_validation_labels_for_tuning"
        )
        is False,
        "quarantined_evidence_not_used": calibration.get(
            "uses_issue174_confirmatory_labels_for_tuning"
        )
        is False,
        "prior_or_future_evidence_not_used": calibration.get(
            "uses_prior_or_future_evidence_for_tuning"
        )
        is False,
        "raw_cross_model_scores_disabled": calibration.get(
            "raw_rank_score_cross_model_comparison_allowed"
        )
        is False,
        "safety_blocked": _safety_blocked(calibration),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid frozen calibration contract: " + ", ".join(failed))


def _materialize_outcome_blind_action_rows(
    feature_rows: list[dict[str, Any]],
    *,
    role_row: dict[str, Any],
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for feature_row in feature_rows:
        decision_ts = int(feature_row.get("decision_ts") or 0)
        max_input_ts = int(feature_row.get("max_input_ts") or 0)
        available_at_ts = int(feature_row.get("available_at_ts") or 0)
        feature_cutoff_ts = int(feature_row.get("feature_cutoff_ts") or 0)
        if (
            decision_ts <= 0
            or max_input_ts > decision_ts
            or available_at_ts > decision_ts
            or feature_cutoff_ts > decision_ts
        ):
            raise ValueError("feature timestamp causality violation")
        provenance_violations = _provenance_timestamp_violations(
            feature_row.get("feature_provenance") or {},
            decision_ts=decision_ts,
        )
        if provenance_violations:
            raise ValueError("feature provenance timestamp causality violation")
        raw = dict(feature_row.get("features") or {})
        p_up = _p_up(raw)
        p_down = 1.0 - p_up
        time_to_close_seconds = float(raw.get("time_to_close_seconds") or 0.0)
        market_close_ts = decision_ts + max(
            0,
            int(round(time_to_close_seconds * 1_000.0)),
        )
        reference_provenance = dict(
            (feature_row.get("feature_provenance") or {}).get(
                "reference_price_to_beat_distance_at_decision"
            )
            or {}
        )
        reference_valid = bool(
            reference_provenance.get("provenance_valid") is True
            and int(reference_provenance.get("max_input_ts") or 0) <= decision_ts
        )
        for action in REQUIRED_ACTIONS:
            side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
            family = (
                "HOLD_TO_SETTLEMENT"
                if "HOLD_TO_SETTLEMENT" in action
                else "SELL_BEFORE_CLOSE"
                if "SELL_BEFORE_CLOSE" in action
                else "NO_TRADE"
            )
            decision_features = _decision_features(
                raw,
                action=action,
                side=side,
                family=family,
            )
            missing = [name for name in feature_columns if name not in decision_features]
            if missing:
                raise ValueError(f"decision-time features are missing: {missing}")
            values = {name: float(decision_features[name]) for name in feature_columns}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("decision-time features must be finite")
            if action != "NO_TRADE" and not reference_valid:
                raise ValueError("reference price feature provenance is invalid")
            side_prefix = side.lower()
            microstructure = {
                "entry_bid": float(raw.get(f"{side_prefix}_bid") or 0.0),
                "entry_ask": float(raw.get(f"{side_prefix}_ask") or 0.0),
                "spread_bps": float(raw.get(f"{side_prefix}_spread_bps") or 0.0),
                "book_staleness_ms": float(raw.get(f"{side_prefix}_book_staleness_ms") or 0.0),
                "queue_fill_proxy": float(
                    raw.get(f"{side_prefix}_queue_fill_probability_proxy") or 0.0
                ),
                "time_to_close_seconds": time_to_close_seconds,
            }
            selected_probability = p_up if side == "UP" else p_down if side == "DOWN" else 0.0
            row = {
                "market_id": str(feature_row["market_id"]),
                "condition_id": str(feature_row.get("condition_id") or feature_row["market_id"]),
                "market_slug": str(feature_row.get("slug") or ""),
                "decision_ts": decision_ts,
                "market_close_ts": market_close_ts,
                "max_input_ts": max_input_ts,
                "role": AUDITED_ROLE,
                "market_selection_rank": int(role_row["selection_rank"]),
                "action": action,
                "side": side,
                "action_family": family,
                "decision_time_features": values,
                "p_up": p_up,
                "p_down": p_down,
                "selected_side_probability": selected_probability,
                "microstructure_snapshot": microstructure,
                "reference_price_feature_provenance": {
                    **reference_provenance,
                    "provenance_valid": reference_valid,
                },
                "p_up_action_disagreement": bool(
                    (side == "UP" and p_up < 0.5) or (side == "DOWN" and p_up > 0.5)
                ),
                "source_feature_row_sha256": canonical_json_sha256(feature_row),
                "target_used_as_decision_input": False,
                "outcome_fields_used_as_decision_input": False,
                "paper_only": True,
                "capital_at_risk": False,
            }
            row["action_row_sha256"] = canonical_json_sha256(row)
            output.append(row)
    return output


def _outcome_blind_acceptance_replay(
    predictions: list[dict[str, Any]],
    *,
    entry_threshold: float,
    runner_up_advantage_threshold: float,
    guard_decision_fn: Callable[..., dict[str, Any]] = _v8_execution_guard_decision,
) -> list[dict[str, Any]]:
    by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_decision[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    guard_config = _v8_execution_guard_config()
    state = _v8_initial_runtime_state(guard_config)
    closes: dict[str, int] = {}
    output: list[dict[str, Any]] = []
    for index, ((market_id, decision_ts), action_rows) in enumerate(
        sorted(by_decision.items(), key=lambda item: (item[0][1], item[0][0])),
        start=1,
    ):
        _release_closed_positions(
            state=state,
            market_close_by_open_position=closes,
            decision_ts=decision_ts,
        )
        if {str(row["action"]) for row in action_rows} != set(REQUIRED_ACTIONS):
            raise ValueError("acceptance replay action grid is incomplete")
        ranked = sorted(
            action_rows,
            key=lambda row: (
                -float(row["action_advantage_lcb_net_return"]),
                str(row["action"]),
            ),
        )
        selected = ranked[0]
        runner_up = ranked[1]
        trade_ranked = [row for row in ranked if str(row["action"]) != "NO_TRADE"]
        best_trade = trade_ranked[0]
        raw_ranked = sorted(
            action_rows,
            key=lambda row: (-float(row["raw_pairwise_rank_score"]), str(row["action"])),
        )
        selected_action = str(selected["action"])
        decision_score = float(selected["action_advantage_lcb_net_return"])
        runner_up_score = float(runner_up["action_advantage_lcb_net_return"])
        advantage = decision_score - runner_up_score
        blockers: list[str] = []
        guard_result: dict[str, Any] | None = None
        full_ranking = [
            {
                "rank": rank,
                "action": row["action"],
                "side": row["side"],
                "action_family": row["action_family"],
                "raw_pairwise_rank_score": float(row["raw_pairwise_rank_score"]),
                "pairwise_group_normalized_rank_score": float(
                    row["pairwise_group_normalized_rank_score"]
                ),
                "calibrated_action_expected_net_return": float(
                    row["calibrated_action_expected_net_return"]
                ),
                "action_advantage_lcb_net_return": float(row["action_advantage_lcb_net_return"]),
                "action_advantage_lcb_score_bucket": row["action_advantage_lcb_score_bucket"],
                "action_advantage_lcb_estimate_source": row["action_advantage_lcb_estimate_source"],
                "p_up_action_disagreement": row["p_up_action_disagreement"],
            }
            for rank, row in enumerate(ranked, start=1)
        ]
        if selected_action == "NO_TRADE":
            blockers.append("policy_selected_no_trade")
        elif decision_score < entry_threshold:
            blockers.append("expected_net_return_below_frozen_entry_threshold")
        elif advantage <= runner_up_advantage_threshold:
            blockers.append("selected_vs_runner_up_advantage_not_positive")
        else:
            guard_ranking = [
                {
                    "rank": rank,
                    "selected_action": row["action"],
                    "selected_side": row["side"],
                    "selected_action_family": row["action_family"],
                    "corrected_model_score": float(row["action_advantage_lcb_net_return"]),
                    "raw_model_score": float(row["raw_pairwise_rank_score"]),
                    "high_score_flag": float(row["action_advantage_lcb_net_return"])
                    >= entry_threshold,
                    "p_up_action_disagreement": row["p_up_action_disagreement"],
                    "microstructure_snapshot": row["microstructure_snapshot"],
                }
                for rank, row in enumerate(ranked, start=1)
            ]
            guard_result = guard_decision_fn(
                {
                    "decision_group_id": canonical_json_sha256(
                        {"market_id": market_id, "decision_ts": decision_ts}
                    ),
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "selected_action": selected_action,
                    "selected_side": selected["side"],
                    "selected_action_family": selected["action_family"],
                    "corrected_model_score": decision_score,
                    "raw_model_score": selected["raw_pairwise_rank_score"],
                    "high_score_flag": decision_score >= entry_threshold,
                    "p_up": selected["p_up"],
                    "p_down": selected["p_down"],
                    "p_up_action_disagreement": selected["p_up_action_disagreement"],
                    "microstructure_snapshot": selected["microstructure_snapshot"],
                    "reference_price_feature_provenance": selected[
                        "reference_price_feature_provenance"
                    ],
                    "decision_time_feature_max_input_ts": selected["max_input_ts"],
                    "full_5_action_ranking": guard_ranking,
                },
                guard_config=guard_config,
                runtime_state=state,
                runtime_mode="simulated_runtime_state",
            )
            blockers.extend(guard_result["execution_blocking_reason_codes"])
        allowed = bool(guard_result and guard_result["order_allowed"])
        executed_action = (
            str(guard_result["execution_guarded_action"]) if allowed else selected_action
        )
        executed = next(row for row in action_rows if str(row["action"]) == executed_action)
        if allowed:
            _v8_apply_simulated_order_to_state(
                state=state,
                decision=guard_result,
                simulated_order_id=f"outcome-blind-viability-{index:06d}",
            )
            closes[market_id] = int(executed["market_close_ts"])
        row = {
            "decision_index": index,
            "market_id": market_id,
            "decision_ts": decision_ts,
            "source_selected_action": selected_action,
            "source_selected_side": selected["side"],
            "source_selected_action_family": selected["action_family"],
            "executed_action": executed_action,
            "selected_side": executed["side"],
            "selected_action_family": executed["action_family"],
            "decision_score": decision_score,
            "runner_up_action": runner_up["action"],
            "runner_up_score": runner_up_score,
            "selected_vs_runner_up_advantage": advantage,
            "best_trade_action": best_trade["action"],
            "best_trade_side": best_trade["side"],
            "best_trade_action_family": best_trade["action_family"],
            "best_trade_action_advantage_lcb_net_return": float(
                best_trade["action_advantage_lcb_net_return"]
            ),
            "best_trade_to_no_trade_lcb_gap": float(
                best_trade["action_advantage_lcb_net_return"]
                - next(
                    value["action_advantage_lcb_net_return"]
                    for value in action_rows
                    if str(value["action"]) == "NO_TRADE"
                )
            ),
            "best_trade_to_frozen_entry_threshold_gap": float(
                best_trade["action_advantage_lcb_net_return"] - entry_threshold
            ),
            "all_trade_action_lcbs_nonpositive": all(
                float(value["action_advantage_lcb_net_return"]) <= 0.0 for value in trade_ranked
            ),
            "raw_ranker_top_action": raw_ranked[0]["action"],
            "raw_ranker_top_side": raw_ranked[0]["side"],
            "raw_ranker_top_action_family": raw_ranked[0]["action_family"],
            "raw_ranker_top_score": float(raw_ranked[0]["raw_pairwise_rank_score"]),
            "lcb_selection_differs_from_raw_ranker_top": (
                selected_action != str(raw_ranked[0]["action"])
            ),
            "frozen_entry_threshold": entry_threshold,
            "frozen_runner_up_advantage_threshold": (runner_up_advantage_threshold),
            "execution_guard_order_allowed": allowed,
            "execution_guard_evaluated": guard_result is not None,
            "guard_action_remapped": allowed and executed_action != selected_action,
            "proposed_order_size": (float(guard_result["proposed_order_size"]) if allowed else 0.0),
            "p_up": selected["p_up"],
            "p_down": selected["p_down"],
            "p_up_action_disagreement": selected["p_up_action_disagreement"],
            "microstructure_snapshot": selected["microstructure_snapshot"],
            "time_to_close_bucket": _time_to_close_bucket(
                float(selected["microstructure_snapshot"]["time_to_close_seconds"])
            ),
            "full_five_action_ranking": full_ranking,
            "execution_blocking_reason_codes": sorted(set(blockers)),
            "first_terminal_stage": _first_terminal_stage(
                allowed=allowed,
                blockers=blockers,
            ),
            "target_or_outcome_fields_used": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        row["viability_row_sha256"] = canonical_json_sha256(row)
        output.append(row)
    return output


def _time_to_close_bucket(seconds: float) -> str:
    if seconds < 0.0:
        return "negative"
    if seconds < 30.0:
        return "000_030"
    if seconds < 60.0:
        return "030_060"
    if seconds < 120.0:
        return "060_120"
    if seconds < 180.0:
        return "120_180"
    return "180_plus"


def _first_terminal_stage(*, allowed: bool, blockers: list[str]) -> str:
    if allowed:
        return "guard_allowed"
    categories = [_blocker_category(code) for code in blockers]
    priority = (
        "selected_no_trade",
        "entry_threshold",
        "runner_up_margin",
        "required_runtime_or_provenance",
        "p_up_disagreement",
        "time_to_close",
        "spread_staleness_queue_or_liquidity",
        "exposure_or_duplicate_position",
        "hts_guard",
        "guard_blocked_other",
    )
    return next((value for value in priority if value in categories), "guard_blocked_other")


def _blocker_category(reason_code: str) -> str:
    value = reason_code.lower()
    if reason_code == "policy_selected_no_trade":
        return "selected_no_trade"
    if reason_code == "expected_net_return_below_frozen_entry_threshold":
        return "entry_threshold"
    if reason_code == "selected_vs_runner_up_advantage_not_positive":
        return "runner_up_margin"
    if any(token in value for token in ("runtime", "provenance", "missing")):
        return "required_runtime_or_provenance"
    if "p_up" in value or "side_disagreement" in value:
        return "p_up_disagreement"
    if "time_to_close" in value or "time_window" in value:
        return "time_to_close"
    if any(token in value for token in ("spread", "stale", "queue", "liquidity", "book")):
        return "spread_staleness_queue_or_liquidity"
    if any(token in value for token in ("exposure", "duplicate", "position")):
        return "exposure_or_duplicate_position"
    if "hts" in value or "hold_to_settlement" in value:
        return "hts_guard"
    return "guard_blocked_other"


def _viability_report(
    *,
    run_id: str,
    role_rows: list[dict[str, Any]],
    source_feature_row_count: int,
    action_rows: list[dict[str, Any]],
    viability_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    candidate: dict[str, Any],
    opened_feature_paths: list[dict[str, str]],
    input_descriptors: dict[str, Any],
) -> dict[str, Any]:
    accepted = [row for row in viability_rows if row["execution_guard_order_allowed"]]
    selected_trade = [row for row in viability_rows if row["source_selected_action"] != "NO_TRADE"]
    stage_distribution = dict(
        sorted(Counter(row["first_terminal_stage"] for row in viability_rows).items())
    )
    blocker_distribution = dict(
        sorted(
            Counter(
                code for row in viability_rows for code in row["execution_blocking_reason_codes"]
            ).items()
        )
    )
    selected_scores = [float(row["decision_score"]) for row in viability_rows]
    best_trade_scores = [
        float(row["best_trade_action_advantage_lcb_net_return"]) for row in viability_rows
    ]
    best_trade_threshold_gaps = [
        float(row["best_trade_to_frozen_entry_threshold_gap"]) for row in viability_rows
    ]
    guard_config = _v8_execution_guard_config()
    gate = dict(protocol["development_freeze_gates"])
    required_count = int(gate["minimum_accepted_bet_count"])
    accepted_by_side = dict(sorted(Counter(row["selected_side"] for row in accepted).items()))
    accepted_by_family = dict(
        sorted(Counter(row["selected_action_family"] for row in accepted).items())
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": run_id,
        "status": "diagnostic_complete_fail_closed",
        "candidate_name": candidate["candidate_name"],
        "audit_role": AUDITED_ROLE,
        "audited_market_count": len(role_rows),
        "source_feature_row_count": source_feature_row_count,
        "decision_group_count": len(viability_rows),
        "materialized_action_row_count": len(action_rows),
        "expected_action_row_count": source_feature_row_count * len(REQUIRED_ACTIONS),
        "complete_five_action_grid_passed": len(action_rows)
        == source_feature_row_count * len(REQUIRED_ACTIONS),
        "feature_causality_violation_count": 0,
        "feature_causality_checked_row_count": source_feature_row_count,
        "target_or_outcome_files_opened": False,
        "target_or_outcome_field_access_count": 0,
        "forbidden_file_open_attempt_count": 0,
        "opened_file_policy": "polymarket_feature_rows_jsonl_only",
        "opened_feature_artifact_count": len(opened_feature_paths),
        "opened_feature_artifacts_sha256": canonical_json_sha256(opened_feature_paths),
        "current_oof_or_validation_pnl_used": False,
        "threshold_sweep_performed": False,
        "threshold_or_guard_mutated": False,
        "model_or_source_score_mutated": False,
        "frozen_input_hashes_verified": True,
        "frozen_model_sha256": input_descriptors["model"]["sha256"],
        "frozen_calibration_artifact_sha256": input_descriptors["calibration_artifact"]["sha256"],
        "frozen_protocol_sha256": input_descriptors["protocol"]["sha256"],
        "frozen_feature_contract_sha256": input_descriptors["feature_contract"]["sha256"],
        "frozen_execution_contract_sha256": canonical_json_sha256(
            protocol["frozen_execution_contract"]
        ),
        "execution_guard_config_sha256": canonical_json_sha256(guard_config),
        "execution_guard_config_mutated": False,
        "frozen_entry_threshold": float(
            protocol["frozen_execution_contract"]["entry_edge_threshold"]
        ),
        "frozen_runner_up_advantage_threshold": float(
            protocol["frozen_execution_contract"]["runner_up_advantage_threshold"]
        ),
        "selected_action_distribution": dict(
            sorted(Counter(row["source_selected_action"] for row in viability_rows).items())
        ),
        "selected_side_distribution": dict(
            sorted(Counter(row["selected_side"] for row in viability_rows).items())
        ),
        "selected_family_distribution": dict(
            sorted(Counter(row["selected_action_family"] for row in viability_rows).items())
        ),
        "time_to_close_bucket_distribution": dict(
            sorted(Counter(row["time_to_close_bucket"] for row in viability_rows).items())
        ),
        "raw_ranker_top_action_distribution": dict(
            sorted(Counter(row["raw_ranker_top_action"] for row in viability_rows).items())
        ),
        "best_trade_action_distribution": dict(
            sorted(Counter(row["best_trade_action"] for row in viability_rows).items())
        ),
        "best_trade_side_distribution": dict(
            sorted(Counter(row["best_trade_side"] for row in viability_rows).items())
        ),
        "best_trade_family_distribution": dict(
            sorted(Counter(row["best_trade_action_family"] for row in viability_rows).items())
        ),
        "raw_ranker_top_changed_by_lcb_count": sum(
            bool(row["lcb_selection_differs_from_raw_ranker_top"]) for row in viability_rows
        ),
        "raw_ranker_trade_top_changed_to_no_trade_by_lcb_count": sum(
            row["source_selected_action"] == "NO_TRADE"
            and row["raw_ranker_top_action"] != "NO_TRADE"
            for row in viability_rows
        ),
        "all_trade_action_lcbs_nonpositive_count": sum(
            bool(row["all_trade_action_lcbs_nonpositive"]) for row in viability_rows
        ),
        "selected_trade_decision_count": len(selected_trade),
        "execution_guard_evaluated_count": sum(
            bool(row["execution_guard_evaluated"]) for row in viability_rows
        ),
        "execution_guard_allowed_count": len(accepted),
        "execution_guard_allowed_unique_market_count": len(
            {str(row["market_id"]) for row in accepted}
        ),
        "execution_guard_allowed_count_by_side": accepted_by_side,
        "execution_guard_allowed_count_by_family": accepted_by_family,
        "required_accepted_bet_count": required_count,
        "accepted_bet_support_shortfall": max(0, required_count - len(accepted)),
        "accepted_bet_support_gate_would_pass": len(accepted) >= required_count,
        "first_terminal_stage_distribution": stage_distribution,
        "execution_blocking_reason_distribution": blocker_distribution,
        "first_terminal_stage_reconciled": sum(stage_distribution.values()) == len(viability_rows),
        "selected_score_summary": {
            "minimum": min(selected_scores, default=None),
            "median": median(selected_scores) if selected_scores else None,
            "maximum": max(selected_scores, default=None),
        },
        "best_trade_lcb_score_summary": _numeric_summary(best_trade_scores),
        "best_trade_to_frozen_entry_threshold_gap_summary": _numeric_summary(
            best_trade_threshold_gaps
        ),
        "zero_accepted_bet_explanation": (
            _zero_bet_explanation(stage_distribution)
            if not accepted
            else "not_applicable_nonzero_guard_allowed_support"
        ),
        "input_descriptors": input_descriptors,
        "diagnostic_only": True,
        "replacement_candidate_created": False,
        "future_redesign_requires_separate_pre_registration": True,
        **_diagnostic_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "minimum": min(values, default=None),
        "median": median(values) if values else None,
        "maximum": max(values, default=None),
    }


def _zero_bet_explanation(stage_distribution: dict[str, int]) -> str:
    if not stage_distribution:
        return "no_decisions_available"
    dominant = sorted(stage_distribution.items(), key=lambda item: (-item[1], item[0]))[0]
    return f"all_decisions_blocked_dominant_stage:{dominant[0]}:{dominant[1]}"


def _validate_complete_action_grid(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
    if not grouped or any(actions != set(REQUIRED_ACTIONS) for actions in grouped.values()):
        raise ValueError("outcome-blind action grid is incomplete")


def _provenance_timestamp_violations(
    payload: Any,
    *,
    decision_ts: int,
    prefix: str = "feature_provenance",
) -> list[str]:
    violations: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}"
            if (
                key in {"max_input_ts", "source_ts", "available_at_ts"}
                and isinstance(value, int | float)
                and int(value) > decision_ts
            ):
                violations.append(path)
            violations.extend(
                _provenance_timestamp_violations(
                    value,
                    decision_ts=decision_ts,
                    prefix=path,
                )
            )
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            violations.extend(
                _provenance_timestamp_violations(
                    value,
                    decision_ts=decision_ts,
                    prefix=f"{prefix}[{index}]",
                )
            )
    return violations


def _safety_blocked(*payloads: dict[str, Any]) -> bool:
    expected = _blocked_safety_fields()
    return all(
        all(payload.get(key) is value for key, value in expected.items()) for payload in payloads
    )


def _diagnostic_safety_fields() -> dict[str, Any]:
    return {
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
        "live_trading_enabled": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
    }


def _prepare_run_dir(
    output_dir: Path | str,
    run_id: str,
    *,
    overwrite: bool,
) -> Path:
    run_dir = Path(output_dir).expanduser().resolve() / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Outcome-Blind Accepted-Bet Viability Audit",
            "",
            f"- candidate: `{report['candidate_name']}`",
            f"- audited markets: `{report['audited_market_count']}`",
            f"- decision groups: `{report['decision_group_count']}`",
            f"- complete 5-action grid: "
            f"`{str(report['complete_five_action_grid_passed']).lower()}`",
            f"- selected trade decisions: `{report['selected_trade_decision_count']}`",
            f"- guard-allowed decisions: `{report['execution_guard_allowed_count']}`",
            f"- required support / shortfall: "
            f"`{report['required_accepted_bet_count']} / "
            f"{report['accepted_bet_support_shortfall']}`",
            f"- first terminal stages: `{report['first_terminal_stage_distribution']}`",
            f"- time-to-close buckets: `{report['time_to_close_bucket_distribution']}`",
            f"- raw ranker top actions: `{report['raw_ranker_top_action_distribution']}`",
            f"- best trade actions: `{report['best_trade_action_distribution']}`",
            f"- all trade LCBs non-positive: `{report['all_trade_action_lcbs_nonpositive_count']}`",
            f"- raw trade top changed to NO_TRADE by LCB: "
            f"`{report['raw_ranker_trade_top_changed_to_no_trade_by_lcb_count']}`",
            f"- best trade LCB summary: `{report['best_trade_lcb_score_summary']}`",
            f"- guard blockers: `{report['execution_blocking_reason_distribution']}`",
            f"- zero-bet explanation: `{report['zero_accepted_bet_explanation']}`",
            "- target/outcome files opened: `false`",
            "- OOF/validation PnL used: `false`",
            "- threshold sweep or mutation: `false`",
            "- replacement candidate created: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )
