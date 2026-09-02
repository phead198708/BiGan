"""Bounded local-paper runtime for the promoted market-clustered v6.2 policy."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import xgboost as xgb

from bigan.execution.position_manager import PositionManager
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    ALLOWED_RAW_FEATURE_FILES,
    _materialize_future_action_rows,
    _materialize_selected_window_features,
)
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
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    attach_frozen_execution_compatibility,
)
from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    O_V8_PAPER_FRESH_EXIT_DECISION_POLICY_SOURCE,
    O_V8_PAPER_FRESH_EXIT_FORCE_SECONDS_TO_CLOSE,
    O_V8_PAPER_FRESH_EXIT_THRESHOLD_PROFILE_NAME,
    _fresh_paper_adapter_sell_decision,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_execution_guard_config,
)

SCHEMA_PREFIX = "bigan-v8-market-clustered-mean-ev-v6-2-paper-canary"
EXPECTED_SCOPE = "v6_2_bounded_local_paper_canary_only"
EXPECTED_ROUND_COUNT = 12
MAXIMUM_PAPER_ORDER_SIZE = 0.2
HARD_CAPTURE_FAILURE_LIMIT = 3
SELL_BEFORE_CLOSE_EXIT_WINDOW_SECONDS = O_V8_PAPER_FRESH_EXIT_FORCE_SECONDS_TO_CLOSE
SELL_BEFORE_CLOSE_EXIT_RULE_ID = (
    "legacy_adapter_first_causal_guard_quality_sell_signal_v1"
)
FORBIDDEN_DECISION_FIELDS = frozenset(
    {
        "accepted_bet_net_pnl",
        "final_outcome",
        "future_return",
        "gross_pnl",
        "label",
        "net_pnl",
        "oracle_action",
        "outcome",
        "payout_down",
        "payout_up",
        "realized_pnl",
        "resolved_outcome",
        "settlement_outcome",
        "settlement_pnl",
        "target_net_pnl_per_contract",
        "winning_outcome",
    }
)


@dataclass(frozen=True, slots=True)
class MarketClusteredMeanEVV62PaperCanaryConfig:
    """Pinned inputs for one bounded v6.2 local-paper canary."""

    run_id: str
    output_dir: Path | str
    unlock_manifest_path: Path | str
    expected_unlock_manifest_sha256: str
    captured_round_dirs: tuple[Path | str, ...]
    runtime_created_ts: int
    builder_git_commit: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_unlock_manifest_sha256, "unlock manifest")
        if self.runtime_created_ts <= 0:
            raise ValueError("runtime_created_ts must be positive")
        if len(self.builder_git_commit) != 40:
            raise ValueError("builder_git_commit must be a 40-character git commit")


def validate_v6_2_paper_candidate_unlock(
    path: Path | str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate the human-approved #219 unlock before any provider or scoring use."""

    resolved = Path(path).expanduser().resolve()
    _verify_pin(resolved, expected_sha256, "approved #219 unlock manifest")
    manifest = _load_json(resolved)
    checks = {
        "candidate": manifest.get("candidate_name") == CANDIDATE_NAME,
        "paper_candidate_allowed": manifest.get("paper_candidate_allowed") is True,
        "paper_canary_handoff_allowed": manifest.get("paper_canary_handoff_allowed") is True,
        "scope": manifest.get("paper_candidate_allowed_scope") == EXPECTED_SCOPE,
        "paper_only": manifest.get("paper_only") is True,
        "capital": manifest.get("capital_at_risk") is False,
        "live": manifest.get("live_trading_enabled") is False,
        "write": manifest.get("polymarket_write_enabled") is False,
        "wallet": manifest.get("wallet_signing_enabled") is False,
        "handoff": manifest.get("v8_execution_handoff_allowed") is False,
        "issue_134": manifest.get("#134_resume_allowed") is False,
        "issue_146": manifest.get("#146_start_allowed") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("approved #219 unlock manifest invalid:" + ",".join(failed))
    for name in (
        "paper_candidate_gate_report",
        "paper_canary_input_contract",
        "promoted_research_candidate_manifest",
        "source_model",
        "market_clustered_mean_risk_calibration",
    ):
        _verified_descriptor(manifest.get(name), name)
    contract = _load_json(Path(manifest["paper_canary_input_contract"]["path"]))
    contract_checks = {
        "frozen": contract.get("frozen") is True,
        "rounds": contract.get("bounded_complete_round_count") == EXPECTED_ROUND_COUNT,
        "max_size": float(contract.get("maximum_paper_order_notional") or 0.0)
        == MAXIMUM_PAPER_ORDER_SIZE,
        "raw": contract.get("per_round_raw_evidence_persistence_required") is True,
        "five_actions": contract.get("full_five_action_grid_required") is True,
        "causality": contract.get("feature_max_input_ts_must_be_lte_decision_ts") is True,
        "forced": contract.get("forced_coverage_bets_allowed") is False,
        "settlement": contract.get("settlement_may_block_next_round_collection") is False,
        "legacy_o": contract.get("legacy_o_source_score_used") is False,
    }
    failed_contract = [name for name, passed in contract_checks.items() if not passed]
    if failed_contract:
        raise ValueError("approved paper canary contract invalid:" + ",".join(failed_contract))
    return manifest


def classify_capture_hard_failure(run_dir: Path | str) -> list[str]:
    """Classify only decision-critical capture failures for bounded fail-fast use."""

    resolved = Path(run_dir).expanduser().resolve()
    report_path = resolved / "pending_round_capture_report.json"
    manifest_path = resolved / "pending_round_capture_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        return ["capture_manifest_or_report_missing"]
    report = _load_json(report_path)
    reasons = []
    if int(report.get("raw_polymarket_market_count") or 0) != 1:
        reasons.append("decision_critical_market_identity_missing")
    if int(report.get("raw_orderbook_row_count") or 0) <= 0:
        reasons.append("decision_critical_orderbook_missing")
    if int(report.get("raw_btc_candle_row_count") or 0) <= 0:
        reasons.append("decision_critical_btc_feature_rows_missing")
    if report.get("pending_feature_enrichment") is True:
        reasons.append("decision_critical_feature_enrichment_pending")
    return sorted(set(reasons))


def run_market_clustered_mean_ev_v6_2_paper_canary(
    config: MarketClusteredMeanEVV62PaperCanaryConfig,
) -> dict[str, Any]:
    """Score captured outcome-blind rounds and emit guard-allowed local paper artifacts."""

    # This must remain the first external artifact read in the runtime.
    unlock_path = Path(config.unlock_manifest_path).expanduser().resolve()
    unlock = validate_v6_2_paper_candidate_unlock(
        unlock_path,
        config.expected_unlock_manifest_sha256,
    )
    lineage = _resolve_frozen_lineage(unlock)
    captured_dirs = tuple(Path(value).expanduser().resolve() for value in config.captured_round_dirs)
    capture_audits = [_capture_audit(path) for path in captured_dirs]
    valid_audits = [row for row in capture_audits if not row["hard_failure_reason_codes"]]
    selected_rows = [_selected_row_from_capture(row, index) for index, row in enumerate(valid_audits, 1)]

    feature_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    if selected_rows:
        feature_rows, _ = _materialize_selected_window_features(selected_rows)
        feature_columns = tuple(str(value) for value in lineage["feature_contract"]["feature_columns"])
        action_rows = _materialize_future_action_rows(
            feature_rows,
            selected_rows=selected_rows,
            feature_columns=feature_columns,
        )
        _validate_decision_artifacts(feature_rows, action_rows)
        booster = xgb.Booster()
        booster.load_model(lineage["model_descriptor"]["path"])
        raw_predictions = _raw_target_stripped_predictions(
            booster,
            action_rows,
            feature_columns=feature_columns,
        )
        scored_rows = apply_market_clustered_mean_ev_scores(
            attach_frozen_execution_compatibility(raw_predictions),
            calibration_artifact=lineage["calibration"],
        )
        guard_rows = _outcome_blind_acceptance_replay(
            scored_rows,
            entry_threshold=0.0,
            runner_up_advantage_threshold=0.0,
        )

    allowed = [row for row in guard_rows if row.get("execution_guard_order_allowed") is True]
    intents = _paper_intents(config.run_id, allowed, feature_rows=feature_rows)
    fills = _paper_fills(intents)
    lifecycle = _paper_position_lifecycle(
        run_id=config.run_id,
        feature_rows=feature_rows,
        entry_fills=fills,
    )
    exit_signals = lifecycle["exit_signals"]
    exit_intents = lifecycle["exit_intents"]
    exit_fills = lifecycle["exit_fills"]
    ledger = lifecycle["ledger"]
    positions = lifecycle["positions"]

    run_dir = Path(config.output_dir).expanduser().resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"paper canary run already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    paths = {
        "capture_audit": run_dir / "v6_2_paper_canary_capture_audit.jsonl",
        "feature_rows": run_dir / "v6_2_paper_canary_feature_rows.jsonl",
        "five_action_rows": run_dir / "v6_2_paper_canary_five_action_rows.jsonl",
        "score_rows": run_dir / "v6_2_paper_canary_score_rows.jsonl",
        "guard_rows": run_dir / "v6_2_paper_canary_execution_guard_rows.jsonl",
        "paper_intents": run_dir / "v6_2_paper_intents.jsonl",
        "paper_fills": run_dir / "v6_2_paper_fills.jsonl",
        "paper_exit_signals": run_dir / "v6_2_paper_exit_signals.jsonl",
        "paper_exit_intents": run_dir / "v6_2_paper_exit_intents.jsonl",
        "paper_exit_fills": run_dir / "v6_2_paper_exit_fills.jsonl",
        "paper_ledger": run_dir / "v6_2_paper_ledger.jsonl",
        "paper_positions": run_dir / "v6_2_paper_positions.json",
        "position_lifecycle_report": (
            run_dir / "v6_2_paper_position_lifecycle_report.json"
        ),
        "safety_report": run_dir / "v6_2_paper_canary_runtime_safety_report.json",
        "runtime_report": run_dir / "v6_2_paper_canary_runtime_report.json",
        "runtime_markdown": run_dir / "v6_2_paper_canary_runtime_report.md",
        "manifest": run_dir / "v6_2_paper_canary_manifest.json",
    }
    for name, rows in (
        ("capture_audit", capture_audits),
        ("feature_rows", feature_rows),
        ("five_action_rows", action_rows),
        ("score_rows", scored_rows),
        ("guard_rows", guard_rows),
        ("paper_intents", intents),
        ("paper_fills", fills),
        ("paper_exit_signals", exit_signals),
        ("paper_exit_intents", exit_intents),
        ("paper_exit_fills", exit_fills),
        ("paper_ledger", ledger),
    ):
        _write_jsonl(paths[name], rows)
    _write_json(paths["paper_positions"], positions)
    _write_json(paths["position_lifecycle_report"], lifecycle["report"])
    round_artifacts = _write_per_round_artifacts(
        run_dir / "rounds",
        capture_audits=capture_audits,
        feature_rows=feature_rows,
        action_rows=action_rows,
        scored_rows=scored_rows,
        guard_rows=guard_rows,
        intents=intents,
        fills=fills,
        exit_signals=exit_signals,
        exit_intents=exit_intents,
        exit_fills=exit_fills,
        ledger=ledger,
    )
    safety = _safety_report(
        run_id=config.run_id,
        capture_audits=capture_audits,
        feature_rows=feature_rows,
        action_rows=action_rows,
        guard_rows=guard_rows,
        intents=intents,
        fills=fills,
        exit_signals=exit_signals,
        exit_intents=exit_intents,
        exit_fills=exit_fills,
        ledger=ledger,
        positions=positions,
    )
    _write_json(paths["safety_report"], safety)
    report = _runtime_report(
        config=config,
        unlock_path=unlock_path,
        unlock=unlock,
        lineage=lineage,
        capture_audits=capture_audits,
        feature_rows=feature_rows,
        action_rows=action_rows,
        scored_rows=scored_rows,
        guard_rows=guard_rows,
        intents=intents,
        fills=fills,
        exit_signals=exit_signals,
        exit_intents=exit_intents,
        exit_fills=exit_fills,
        ledger=ledger,
        positions=positions,
        lifecycle_report=lifecycle["report"],
        safety=safety,
        round_artifacts=round_artifacts,
    )
    _write_json(paths["runtime_report"], report)
    _write_text(paths["runtime_markdown"], _runtime_markdown(report))
    artifact_descriptors = {
        name: _descriptor(path)
        for name, path in paths.items()
        if name != "manifest" and path.is_file()
    }
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v2",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "runtime_created_ts": config.runtime_created_ts,
        "approved_unlock_manifest": _descriptor(unlock_path),
        "artifacts": artifact_descriptors,
        "per_round_artifacts": round_artifacts,
        "complete_round_count": len(valid_audits),
        "paper_intent_count": len(intents),
        "paper_fill_count": len(fills),
        "paper_exit_intent_count": len(exit_intents),
        "paper_exit_fill_count": len(exit_fills),
        "sell_before_close_residual_position_count": positions[
            "sell_before_close_residual_position_count"
        ],
        "closed_before_settlement_realized_trade_pnl_sum": lifecycle["report"][
            "closed_before_settlement_realized_trade_pnl_sum"
        ],
        "runtime_safety_passed": safety["runtime_safety_passed"],
        "paper_candidate_allowed": True,
        "paper_only": True,
        "capital_at_risk": False,
        "live_trading_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    _write_json(paths["manifest"], manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": paths["runtime_report"],
        "report_sha256": _sha256_file(paths["runtime_report"]),
        "manifest": manifest,
        "manifest_path": paths["manifest"],
        "manifest_sha256": _sha256_file(paths["manifest"]),
    }


def _resolve_frozen_lineage(unlock: dict[str, Any]) -> dict[str, Any]:
    promoted_descriptor = _verified_descriptor(
        unlock["promoted_research_candidate_manifest"], "promoted candidate"
    )
    promoted = _load_json(Path(promoted_descriptor["path"]))
    original_descriptor = _verified_descriptor(
        promoted.get("original_frozen_candidate_manifest"), "original candidate"
    )
    original = _load_json(Path(original_descriptor["path"]))
    audit_descriptor = _verified_descriptor(original.get("pre_target_access_audit"), "pre-target audit")
    audit = _load_json(Path(audit_descriptor["path"]))
    feature_descriptor = _verified_descriptor(audit.get("feature_contract"), "feature contract")
    profile_descriptor = _verified_descriptor(original.get("profile"), "v6.2 profile")
    model_descriptor = _verified_descriptor(unlock.get("source_model"), "source model")
    calibration_descriptor = _verified_descriptor(
        unlock.get("market_clustered_mean_risk_calibration"), "v6.2 calibration"
    )
    if model_descriptor != _verified_descriptor(original.get("source_model"), "original model"):
        raise ValueError("approved unlock model lineage mismatch")
    if calibration_descriptor != _verified_descriptor(
        original.get("market_clustered_mean_risk_calibration"), "original calibration"
    ):
        raise ValueError("approved unlock calibration lineage mismatch")
    profile = _load_json(Path(profile_descriptor["path"]))
    if profile.get("candidate_name") != CANDIDATE_NAME or profile.get("frozen") is not True:
        raise ValueError("v6.2 frozen profile invalid")
    return {
        "promoted_descriptor": promoted_descriptor,
        "original_descriptor": original_descriptor,
        "audit_descriptor": audit_descriptor,
        "feature_descriptor": feature_descriptor,
        "profile_descriptor": profile_descriptor,
        "model_descriptor": model_descriptor,
        "calibration_descriptor": calibration_descriptor,
        "feature_contract": _load_json(Path(feature_descriptor["path"])),
        "profile": profile,
        "calibration": _load_json(Path(calibration_descriptor["path"])),
    }


def _capture_audit(run_dir: Path) -> dict[str, Any]:
    reasons = classify_capture_hard_failure(run_dir)
    report_path = run_dir / "pending_round_capture_report.json"
    manifest_path = run_dir / "pending_round_capture_manifest.json"
    report = _load_json(report_path) if report_path.is_file() else {}
    raw_descriptors = {}
    provider_raw_descriptors = {}
    for dirname, target in (("raw", raw_descriptors), ("provider_raw", provider_raw_descriptors)):
        for filename in ALLOWED_RAW_FEATURE_FILES:
            path = run_dir / dirname / filename
            if path.is_file():
                target[filename] = _jsonl_descriptor(path)
    row = {
        "run_id": str(report.get("run_id") or run_dir.name),
        "run_dir": str(run_dir),
        "capture_report": _descriptor(report_path) if report_path.is_file() else None,
        "capture_manifest": _descriptor(manifest_path) if manifest_path.is_file() else None,
        "raw_artifacts": raw_descriptors,
        "provider_raw_artifacts": provider_raw_descriptors,
        "hard_capture_failure": bool(reasons),
        "hard_failure_reason_codes": reasons,
        "optional_provider_failures_do_not_trigger_fail_fast": True,
        "resolution_artifact_opened_for_decision": False,
        "labels_outcomes_or_pnl_opened_for_decision": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    row["capture_audit_row_sha256"] = canonical_json_sha256(row)
    return row


def _selected_row_from_capture(audit: dict[str, Any], sequence: int) -> dict[str, Any]:
    raw = dict(audit["raw_artifacts"])
    missing = [name for name in ALLOWED_RAW_FEATURE_FILES if name not in raw]
    if missing:
        raise ValueError("valid capture missing raw feature artifacts:" + ",".join(missing))
    market_rows = _load_jsonl(Path(raw["raw_polymarket_markets.jsonl"]["path"]))
    if len(market_rows) != 1:
        raise ValueError("paper canary capture must contain exactly one market")
    market = market_rows[0]
    selected = {
        "sequence": sequence,
        "scheduled_round_start_ts": int(market["market_start_ts"]),
        "market_start_ts": int(market["market_start_ts"]),
        "market_end_ts": int(market["market_end_ts"]),
        "market_id": str(market["market_id"]),
        "capture_run_id": audit["run_id"],
        "capture_quality_valid": True,
        "labels_outcomes_or_pnl_opened": False,
        "raw_resolution_row_count": 0,
        "raw_artifacts": {name: raw[name] for name in ALLOWED_RAW_FEATURE_FILES},
    }
    selected["entry_sha256"] = canonical_json_sha256(selected)
    return selected


def _validate_decision_artifacts(
    feature_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]]
) -> None:
    if len(action_rows) != len(feature_rows) * 5:
        raise ValueError("paper canary full five-action grid is incomplete")
    for row in (*feature_rows, *action_rows):
        decision_ts = int(row["decision_ts"])
        if int(row["max_input_ts"]) > decision_ts:
            raise ValueError("paper canary feature causality violation")
        forbidden = _find_nonempty_fields(row, FORBIDDEN_DECISION_FIELDS)
        if forbidden:
            raise ValueError("paper canary decision row contains targets:" + ",".join(forbidden))


def _paper_intents(
    run_id: str,
    rows: list[dict[str, Any]],
    *,
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    features_by_decision = {
        (str(row["market_id"]), int(row["decision_ts"])): row
        for row in feature_rows
    }
    intents = []
    for index, row in enumerate(rows, 1):
        size = float(row.get("proposed_order_size") or 0.0)
        micro = dict(row.get("microstructure_snapshot") or {})
        price = float(micro.get("entry_ask") or 0.0)
        decision_ts = int(row["decision_ts"])
        source_feature = features_by_decision.get(
            (str(row["market_id"]), decision_ts)
        )
        if source_feature is None:
            raise ValueError("guard-allowed decision is missing its source feature row")
        time_to_close_seconds = float(
            (source_feature.get("features") or {}).get("time_to_close_seconds")
            or 0.0
        )
        if time_to_close_seconds <= 0.0:
            raise ValueError("guard-allowed decision has invalid source market close")
        market_close_ts = decision_ts + int(round(time_to_close_seconds * 1_000.0))
        action_family = str(row["selected_action_family"])
        intended_exit_policy = (
            "sell_before_close"
            if action_family == "SELL_BEFORE_CLOSE"
            else "hold_to_settlement"
        )
        planned_exit_before_ts = (
            market_close_ts - int(SELL_BEFORE_CLOSE_EXIT_WINDOW_SECONDS * 1_000.0)
            if intended_exit_policy == "sell_before_close"
            else None
        )
        if not 0.0 < size <= MAXIMUM_PAPER_ORDER_SIZE:
            raise ValueError("guard-allowed paper order size violates frozen maximum")
        if not 0.0 < price < 1.0:
            raise ValueError("guard-allowed paper order has invalid executable ask")
        intent = {
            "paper_intent_id": f"{run_id}-intent-{index:06d}",
            "market_id": row["market_id"],
            "decision_ts": decision_ts,
            "market_close_ts": market_close_ts,
            "source_selected_action": row["source_selected_action"],
            "executed_action": row["executed_action"],
            "selected_side": row["selected_side"],
            "selected_action_family": action_family,
            "intended_exit_policy": intended_exit_policy,
            "planned_exit_before_ts": planned_exit_before_ts,
            "sell_before_close_exit_rule_id": (
                SELL_BEFORE_CLOSE_EXIT_RULE_ID
                if intended_exit_policy == "sell_before_close"
                else None
            ),
            "decision_score": float(row["decision_score"]),
            "selected_vs_runner_up_advantage": float(row["selected_vs_runner_up_advantage"]),
            "paper_order_size": size,
            "paper_limit_price": price,
            "microstructure_snapshot": micro,
            "order_size_source": "frozen_execution_guard_proposed_order_size",
            "execution_guard_order_allowed": True,
            "execution_blocking_reason_codes": [],
            "forced_coverage_bet": False,
            "target_or_outcome_fields_used": False,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "v8_execution_handoff_allowed": False,
        }
        intent["paper_intent_sha256"] = canonical_json_sha256(intent)
        intents.append(intent)
    return intents


def _paper_fills(intents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fills = []
    for index, intent in enumerate(intents, 1):
        size = float(intent["paper_order_size"])
        price = float(intent["paper_limit_price"])
        fill = {
            "paper_fill_id": f"paper-fill-{index:06d}",
            "paper_intent_id": intent["paper_intent_id"],
            "market_id": intent["market_id"],
            "decision_ts": intent["decision_ts"],
            "market_close_ts": intent["market_close_ts"],
            "executed_action": intent["executed_action"],
            "selected_side": intent["selected_side"],
            "selected_action_family": intent["selected_action_family"],
            "intended_exit_policy": intent["intended_exit_policy"],
            "planned_exit_before_ts": intent["planned_exit_before_ts"],
            "requested_size": size,
            "filled_size": size,
            "paper_fill_price": price,
            "synthetic_paper_cash_delta": -(size * price),
            "fill_rule_id": "guard_allowed_executable_ask_full_fill_v1",
            "settlement_resolved": False,
            "settlement_pnl": None,
            "outcome_used_for_fill_decision": False,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "v8_execution_handoff_allowed": False,
        }
        fill["paper_fill_sha256"] = canonical_json_sha256(fill)
        fills.append(fill)
    return fills


def _paper_position_lifecycle(
    *,
    run_id: str,
    feature_rows: list[dict[str, Any]],
    entry_fills: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen action-family lifecycle using causal paper-only books."""

    guard_config = _v8_execution_guard_config()
    conn = duckdb.connect(":memory:")
    manager = PositionManager(conn=conn)
    exit_signals: list[dict[str, Any]] = []
    exit_intents: list[dict[str, Any]] = []
    exit_fills: list[dict[str, Any]] = []
    fill_by_event_id: dict[str, dict[str, Any]] = {}
    residual_reasons: dict[str, list[str]] = {}
    try:
        for fill in sorted(
            entry_fills,
            key=lambda row: (int(row["decision_ts"]), str(row["paper_fill_id"])),
        ):
            event_id = str(fill["paper_fill_id"])
            fill_by_event_id[event_id] = fill
            family = str(fill["selected_action_family"])
            manager.open_position(
                event_id=event_id,
                symbol=f"POLYMARKET:{fill['market_id']}:{fill['selected_side']}",
                side=str(fill["selected_side"]),
                entry_price=float(fill["paper_fill_price"]),
                size=float(fill["filled_size"]),
                order_id=str(fill["paper_intent_id"]),
                sleeve="volatility" if family == "SELL_BEFORE_CLOSE" else "settlement",
                entry_time=int(fill["decision_ts"]),
                fill_price=float(fill["paper_fill_price"]),
            )
            if family == "HOLD_TO_SETTLEMENT":
                exit_signals.append(
                    _paper_exit_signal(
                        run_id=run_id,
                        signal_index=len(exit_signals) + 1,
                        fill=fill,
                        decision="HOLD_TO_SETTLEMENT",
                        candidate=None,
                        reason_codes=["hold_to_settlement_policy_no_preclose_exit"],
                    )
                )
                continue

            candidates = _sell_before_close_exit_candidates(
                fill=fill,
                feature_rows=feature_rows,
            )
            if not candidates:
                reasons = ["sell_before_close_post_entry_snapshot_missing"]
                residual_reasons[event_id] = reasons
                exit_signals.append(
                    _paper_exit_signal(
                        run_id=run_id,
                        signal_index=len(exit_signals) + 1,
                        fill=fill,
                        decision="HOLD_POSITION",
                        candidate=None,
                        reason_codes=reasons,
                    )
                )
                continue

            closed = False
            last_reasons: list[str] = []
            for candidate in candidates:
                quality_reasons = _sell_before_close_exit_blockers(
                    candidate=candidate,
                    fill=fill,
                    guard_config=guard_config,
                )
                sell = False
                adapter_reasons: list[str] = []
                if not quality_reasons:
                    sell, adapter_reasons = _fresh_paper_adapter_sell_decision(
                        exit_price=float(candidate["exit_bid"]),
                        entry_price=float(fill["paper_fill_price"]),
                        p_side=float(candidate["selected_side_probability"]),
                        time_to_close=float(candidate["time_to_close_seconds"]),
                    )
                reasons = quality_reasons or adapter_reasons
                decision = "SELL_POSITION" if sell else "HOLD_POSITION"
                signal = _paper_exit_signal(
                    run_id=run_id,
                    signal_index=len(exit_signals) + 1,
                    fill=fill,
                    decision=decision,
                    candidate=candidate,
                    reason_codes=reasons,
                )
                exit_signals.append(signal)
                if not sell:
                    last_reasons = reasons
                    continue
                manager.close_position(
                    event_id,
                    float(candidate["exit_bid"]),
                    exit_time=int(candidate["decision_ts"]),
                )
                intent = _paper_exit_intent(
                    run_id=run_id,
                    intent_index=len(exit_intents) + 1,
                    signal=signal,
                )
                exit_intents.append(intent)
                exit_fills.append(
                    _paper_exit_fill(
                        fill_index=len(exit_fills) + 1,
                        intent=intent,
                    )
                )
                closed = True
                break
            if not closed:
                residual_reasons[event_id] = last_reasons or [
                    "sell_before_close_no_guard_quality_exit_snapshot"
                ]

        position_rows = _paper_position_rows(
            manager=manager,
            fill_by_event_id=fill_by_event_id,
            exit_fills=exit_fills,
            residual_reasons=residual_reasons,
        )
    finally:
        conn.close()

    positions = {
        "schema_version": f"{SCHEMA_PREFIX}-positions-v2",
        "position_count": len(position_rows),
        "open_position_count": sum(row["status"] == "open" for row in position_rows),
        "closed_position_count": sum(row["status"] == "closed" for row in position_rows),
        "sell_before_close_residual_position_count": sum(
            row["status"] == "open"
            and row["intended_exit_policy"] == "sell_before_close"
            for row in position_rows
        ),
        "hold_to_settlement_open_position_count": sum(
            row["status"] == "open"
            and row["intended_exit_policy"] == "hold_to_settlement"
            for row in position_rows
        ),
        "positions": position_rows,
        "paper_only": True,
        "capital_at_risk": False,
    }
    ledger = _paper_ledger(entry_fills, exit_fills)
    causal_violations = sum(
        signal.get("exit_max_input_ts") is not None
        and (
            int(signal["exit_max_input_ts"]) > int(signal["decision_ts"])
            or int(signal["decision_ts"]) >= int(signal["market_close_ts"])
        )
        for signal in exit_signals
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-position-lifecycle-report-v1",
        "run_id": run_id,
        "position_manager_reused": True,
        "position_manager_module": "bigan.execution.position_manager.PositionManager",
        "legacy_position_exit_adapter_reused": True,
        "exit_decision_policy_source": O_V8_PAPER_FRESH_EXIT_DECISION_POLICY_SOURCE,
        "exit_threshold_profile_name": O_V8_PAPER_FRESH_EXIT_THRESHOLD_PROFILE_NAME,
        "sell_before_close_exit_rule_id": SELL_BEFORE_CLOSE_EXIT_RULE_ID,
        "sell_before_close_exit_window_seconds": SELL_BEFORE_CLOSE_EXIT_WINDOW_SECONDS,
        "exit_quality_guard_config": {
            key: guard_config[key]
            for key in (
                "max_spread_bps",
                "max_book_staleness_ms",
                "min_queue_fill",
            )
        },
        "exit_quality_guard_config_sha256": canonical_json_sha256(
            {
                key: guard_config[key]
                for key in (
                    "max_spread_bps",
                    "max_book_staleness_ms",
                    "min_queue_fill",
                )
            }
        ),
        "entry_fill_count": len(entry_fills),
        "exit_signal_count": len(exit_signals),
        "exit_intent_count": len(exit_intents),
        "exit_fill_count": len(exit_fills),
        "closed_before_settlement_realized_trade_pnl_sum": sum(
            float(row["realized_trade_pnl"])
            for row in position_rows
            if row["realized_trade_pnl"] is not None
        ),
        "sell_before_close_residual_position_count": positions[
            "sell_before_close_residual_position_count"
        ],
        "hold_to_settlement_preclose_exit_count": sum(
            row["intended_exit_policy"] == "hold_to_settlement"
            for row in exit_fills
        ),
        "exit_causality_violation_count": causal_violations,
        "position_lifecycle_reconciled": (
            len(position_rows) == len(entry_fills)
            and len(exit_intents) == len(exit_fills)
            and causal_violations == 0
        ),
        "sell_before_close_lifecycle_complete": positions[
            "sell_before_close_residual_position_count"
        ]
        == 0,
        "outcome_label_or_pnl_used_for_exit_decision": False,
        "realized_exit_pnl_used_for_policy_tuning": False,
        "source_or_o_score_mutated": False,
        "threshold_cost_sizing_or_entry_guard_mutated": False,
        "paper_only": True,
        "capital_at_risk": False,
        "live_trading_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    report["report_id"] = canonical_json_sha256(report)
    return {
        "exit_signals": exit_signals,
        "exit_intents": exit_intents,
        "exit_fills": exit_fills,
        "ledger": ledger,
        "positions": positions,
        "report": report,
    }


def _sell_before_close_exit_candidates(
    *,
    fill: dict[str, Any],
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    market_id = str(fill["market_id"])
    entry_ts = int(fill["decision_ts"])
    market_close_ts = int(fill["market_close_ts"])
    side = str(fill["selected_side"]).lower()
    candidates = []
    for row in feature_rows:
        decision_ts = int(row.get("decision_ts") or 0)
        if (
            str(row.get("market_id") or "") != market_id
            or decision_ts <= entry_ts
            or decision_ts >= market_close_ts
        ):
            continue
        raw = dict(row.get("features") or {})
        candidate = {
            "market_id": market_id,
            "decision_ts": decision_ts,
            "market_close_ts": market_close_ts,
            "max_input_ts": int(row.get("max_input_ts") or 0),
            "available_at_ts": int(row.get("available_at_ts") or 0),
            "exit_bid": float(raw.get(f"{side}_bid") or 0.0),
            "exit_ask": float(raw.get(f"{side}_ask") or 0.0),
            "exit_bid_size": float(raw.get(f"{side}_bid_size") or 0.0),
            "exit_executable_bid_notional": float(
                raw.get(f"{side}_executable_bid_notional") or 0.0
            ),
            "spread_bps": float(raw.get(f"{side}_spread_bps") or 0.0),
            "book_staleness_ms": float(
                raw.get(f"{side}_book_staleness_ms") or 0.0
            ),
            "queue_fill_proxy": float(
                raw.get(f"{side}_queue_fill_probability_proxy") or 0.0
            ),
            "time_to_close_seconds": float(raw.get("time_to_close_seconds") or 0.0),
            "source_feature_row_sha256": str(
                row.get("future_feature_row_sha256")
                or canonical_json_sha256(row)
            ),
        }
        up_mid = float(raw.get("up_mid") or 0.0)
        down_mid = float(raw.get("down_mid") or 0.0)
        probability_denominator = up_mid + down_mid
        p_up = up_mid / probability_denominator if probability_denominator > 0.0 else 0.5
        candidate["p_up"] = p_up
        candidate["p_down"] = 1.0 - p_up
        candidate["selected_side_probability"] = (
            p_up if side == "up" else 1.0 - p_up
        )
        candidate["exit_candidate_sha256"] = canonical_json_sha256(candidate)
        candidates.append(candidate)
    return sorted(candidates, key=lambda row: (row["decision_ts"], row["max_input_ts"]))


def _sell_before_close_exit_blockers(
    *,
    candidate: dict[str, Any],
    fill: dict[str, Any],
    guard_config: dict[str, Any],
) -> list[str]:
    reasons = []
    decision_ts = int(candidate["decision_ts"])
    if (
        int(candidate["max_input_ts"]) <= 0
        or int(candidate["available_at_ts"]) <= 0
        or int(candidate["max_input_ts"]) > decision_ts
        or int(candidate["available_at_ts"]) > decision_ts
    ):
        reasons.append("exit_feature_timestamp_causality_violation")
    if decision_ts >= int(candidate["market_close_ts"]):
        reasons.append("exit_not_strictly_before_market_close")
    if not 0.0 < float(candidate["exit_bid"]) < 1.0:
        reasons.append("exit_executable_bid_missing_or_invalid")
    if not 0.0 < float(candidate["exit_ask"]) < 1.0:
        reasons.append("exit_executable_ask_missing_or_invalid")
    if (
        float(candidate["spread_bps"]) < 0.0
        or float(candidate["spread_bps"]) > float(guard_config["max_spread_bps"])
    ):
        reasons.append("exit_spread_too_wide")
    if (
        float(candidate["book_staleness_ms"]) < 0.0
        or float(candidate["book_staleness_ms"])
        > float(guard_config["max_book_staleness_ms"])
    ):
        reasons.append("exit_book_stale")
    if not (
        float(guard_config["min_queue_fill"])
        <= float(candidate["queue_fill_proxy"])
        <= 1.0
    ):
        reasons.append("exit_queue_fill_too_weak")
    size = float(fill["filled_size"])
    if float(candidate["exit_bid_size"]) < size:
        reasons.append("exit_bid_size_below_position_size")
    if float(candidate["exit_executable_bid_notional"]) < size * float(
        candidate["exit_bid"]
    ):
        reasons.append("exit_executable_notional_below_position_notional")
    return sorted(set(reasons))


def _paper_exit_signal(
    *,
    run_id: str,
    signal_index: int,
    fill: dict[str, Any],
    decision: str,
    candidate: dict[str, Any] | None,
    reason_codes: list[str],
) -> dict[str, Any]:
    row = {
        "paper_exit_signal_id": f"{run_id}-exit-signal-{signal_index:06d}",
        "paper_fill_id": fill["paper_fill_id"],
        "market_id": fill["market_id"],
        "selected_side": fill["selected_side"],
        "entry_action": fill["executed_action"],
        "intended_exit_policy": fill["intended_exit_policy"],
        "entry_decision_ts": fill["decision_ts"],
        "decision_ts": (
            int(candidate["decision_ts"])
            if candidate is not None
            else int(fill["decision_ts"])
        ),
        "market_close_ts": fill["market_close_ts"],
        "planned_exit_before_ts": fill["planned_exit_before_ts"],
        "paper_exit_decision": decision,
        "accepted_for_paper_exit_intent": decision == "SELL_POSITION",
        "exit_price": None if candidate is None else candidate["exit_bid"],
        "exit_size": (
            float(fill["filled_size"]) if decision == "SELL_POSITION" else 0.0
        ),
        "exit_max_input_ts": None if candidate is None else candidate["max_input_ts"],
        "exit_available_at_ts": (
            None if candidate is None else candidate["available_at_ts"]
        ),
        "exit_microstructure_snapshot": None if candidate is None else candidate,
        "exit_reason_codes": sorted(set(reason_codes)),
        "sell_before_close_exit_rule_id": (
            SELL_BEFORE_CLOSE_EXIT_RULE_ID
            if fill["intended_exit_policy"] == "sell_before_close"
            else None
        ),
        "outcome_label_or_pnl_used": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    row["paper_exit_signal_sha256"] = canonical_json_sha256(row)
    return row


def _paper_exit_intent(
    *, run_id: str, intent_index: int, signal: dict[str, Any]
) -> dict[str, Any]:
    intent = {
        "paper_exit_intent_id": f"{run_id}-exit-intent-{intent_index:06d}",
        "paper_exit_signal_id": signal["paper_exit_signal_id"],
        "paper_fill_id": signal["paper_fill_id"],
        "market_id": signal["market_id"],
        "selected_side": signal["selected_side"],
        "entry_action": signal["entry_action"],
        "intended_exit_policy": signal["intended_exit_policy"],
        "decision_ts": signal["decision_ts"],
        "market_close_ts": signal["market_close_ts"],
        "paper_exit_action": "SELL_POSITION",
        "paper_exit_size": signal["exit_size"],
        "paper_limit_price": signal["exit_price"],
        "exit_reason_codes": signal["exit_reason_codes"],
        "local_paper_exit_only": True,
        "outcome_label_or_pnl_used": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    intent["paper_exit_intent_sha256"] = canonical_json_sha256(intent)
    return intent


def _paper_exit_fill(*, fill_index: int, intent: dict[str, Any]) -> dict[str, Any]:
    size = float(intent["paper_exit_size"])
    price = float(intent["paper_limit_price"])
    fill = {
        "paper_exit_fill_id": f"paper-exit-fill-{fill_index:06d}",
        "paper_exit_intent_id": intent["paper_exit_intent_id"],
        "entry_paper_fill_id": intent["paper_fill_id"],
        "market_id": intent["market_id"],
        "selected_side": intent["selected_side"],
        "entry_action": intent["entry_action"],
        "intended_exit_policy": intent["intended_exit_policy"],
        "decision_ts": intent["decision_ts"],
        "market_close_ts": intent["market_close_ts"],
        "requested_size": size,
        "filled_size": size,
        "paper_fill_price": price,
        "synthetic_paper_cash_delta": size * price,
        "fill_rule_id": "causal_guard_quality_executable_bid_full_fill_v1",
        "outcome_used_for_fill_decision": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    fill["paper_exit_fill_sha256"] = canonical_json_sha256(fill)
    return fill


def _paper_position_rows(
    *,
    manager: PositionManager,
    fill_by_event_id: dict[str, dict[str, Any]],
    exit_fills: list[dict[str, Any]],
    residual_reasons: dict[str, list[str]],
) -> list[dict[str, Any]]:
    exit_by_entry = {str(row["entry_paper_fill_id"]): row for row in exit_fills}
    rows = []
    for position in sorted(manager.list_positions(), key=lambda value: value.event_id):
        entry = fill_by_event_id[position.event_id]
        exit_fill = exit_by_entry.get(position.event_id)
        row = {
            "position_id": position.event_id,
            "market_id": entry["market_id"],
            "selected_side": entry["selected_side"],
            "entry_action": entry["executed_action"],
            "action_family": entry["selected_action_family"],
            "intended_exit_policy": entry["intended_exit_policy"],
            "status": position.status,
            "entry_decision_ts": entry["decision_ts"],
            "market_close_ts": entry["market_close_ts"],
            "planned_exit_before_ts": entry["planned_exit_before_ts"],
            "entry_price": float(entry["paper_fill_price"]),
            "entry_contract_size": float(entry["filled_size"]),
            "open_contract_size": float(position.size) if position.status == "open" else 0.0,
            "exit_decision_ts": None if exit_fill is None else exit_fill["decision_ts"],
            "exit_price": None if exit_fill is None else exit_fill["paper_fill_price"],
            "realized_trade_pnl": (
                None
                if exit_fill is None
                else (
                    float(exit_fill["paper_fill_price"])
                    - float(entry["paper_fill_price"])
                )
                * float(entry["filled_size"])
            ),
            "settlement_residual": position.status == "open",
            "residual_reason_codes": residual_reasons.get(position.event_id, []),
            "settlement_resolved": False,
            "outcome_used_for_position_lifecycle": False,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
        }
        row["position_row_sha256"] = canonical_json_sha256(row)
        rows.append(row)
    return rows


def _paper_ledger(
    fills: list[dict[str, Any]], exit_fills: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    cash = 10_000.0
    exposure: defaultdict[str, float] = defaultdict(float)
    rows = []
    events = [
        (int(fill["decision_ts"]), 0, "ENTRY", fill) for fill in fills
    ] + [
        (int(fill["decision_ts"]), 1, "EXIT", fill) for fill in exit_fills
    ]
    for index, (_, _, event_type, fill) in enumerate(sorted(events), 1):
        market_id = str(fill["market_id"])
        before = cash
        cash += float(fill["synthetic_paper_cash_delta"])
        exposure_delta = float(fill["filled_size"]) * (
            1.0 if event_type == "ENTRY" else -1.0
        )
        exposure[market_id] += exposure_delta
        if abs(exposure[market_id]) < 1e-12:
            exposure[market_id] = 0.0
        row = {
            "paper_ledger_entry_id": f"paper-ledger-{index:06d}",
            "ledger_event_type": event_type,
            "paper_fill_id": fill.get("paper_fill_id"),
            "paper_exit_fill_id": fill.get("paper_exit_fill_id"),
            "market_id": market_id,
            "decision_ts": fill["decision_ts"],
            "cash_before": before,
            "cash_after": cash,
            "synthetic_cash_delta": float(fill["synthetic_paper_cash_delta"]),
            "synthetic_position_delta": exposure_delta,
            "market_contract_exposure_after": exposure[market_id],
            "total_contract_exposure_after": sum(exposure.values()),
            "settlement_resolved": False,
            "realized_pnl": None,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
        }
        row["paper_ledger_entry_sha256"] = canonical_json_sha256(row)
        rows.append(row)
    return rows


def _safety_report(
    *,
    run_id: str,
    capture_audits: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    exit_signals: list[dict[str, Any]],
    exit_intents: list[dict[str, Any]],
    exit_fills: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    positions: dict[str, Any],
) -> dict[str, Any]:
    causality_violations = sum(
        int(row["max_input_ts"]) > int(row["decision_ts"])
        for row in (*feature_rows, *action_rows)
    )
    forbidden = sum(
        bool(_find_nonempty_fields(row, FORBIDDEN_DECISION_FIELDS))
        for row in (
            *feature_rows,
            *action_rows,
            *guard_rows,
            *intents,
            *exit_signals,
            *exit_intents,
        )
    )
    exit_causality_violations = sum(
        row.get("exit_max_input_ts") is not None
        and (
            int(row["exit_max_input_ts"]) > int(row["decision_ts"])
            or int(row["decision_ts"]) >= int(row["market_close_ts"])
        )
        for row in exit_signals
    )
    checks = {
        "feature_causality": causality_violations == 0,
        "forbidden_decision_fields": forbidden == 0,
        "five_action_grid": len(action_rows) == len(feature_rows) * 5,
        "guard_only_intents": len(intents)
        == sum(row.get("execution_guard_order_allowed") is True for row in guard_rows),
        "deterministic_entry_fill_reconciliation": len(intents) == len(fills),
        "deterministic_exit_fill_reconciliation": len(exit_intents) == len(exit_fills),
        "deterministic_ledger_reconciliation": len(ledger)
        == len(fills) + len(exit_fills),
        "position_reconciliation": positions["position_count"] == len(fills),
        "sell_before_close_lifecycle_complete": positions[
            "sell_before_close_residual_position_count"
        ]
        == 0,
        "hold_to_settlement_never_exits_preclose": all(
            row["intended_exit_policy"] != "hold_to_settlement"
            for row in exit_fills
        ),
        "exit_feature_causality": exit_causality_violations == 0,
        "no_negative_position_exposure": all(
            float(row["open_contract_size"]) >= 0.0
            for row in positions["positions"]
        ),
        "order_size": all(
            0.0 < float(row["paper_order_size"]) <= MAXIMUM_PAPER_ORDER_SIZE
            for row in intents
        ),
        "raw_resolution_not_opened": all(
            row["resolution_artifact_opened_for_decision"] is False for row in capture_audits
        ),
    }
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-safety-v2",
        "run_id": run_id,
        "runtime_safety_checks": checks,
        "runtime_safety_passed": all(checks.values()),
        "feature_causality_violation_count": causality_violations,
        "forbidden_decision_field_count": forbidden,
        "exit_feature_causality_violation_count": exit_causality_violations,
        "paper_only": True,
        "capital_at_risk": False,
        "live_trading_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _runtime_report(
    *,
    config: MarketClusteredMeanEVV62PaperCanaryConfig,
    unlock_path: Path,
    unlock: dict[str, Any],
    lineage: dict[str, Any],
    capture_audits: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    exit_signals: list[dict[str, Any]],
    exit_intents: list[dict[str, Any]],
    exit_fills: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    positions: dict[str, Any],
    lifecycle_report: dict[str, Any],
    safety: dict[str, Any],
    round_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [row for row in capture_audits if not row["hard_capture_failure"]]
    hard_reasons = Counter(
        reason for row in capture_audits for reason in row["hard_failure_reason_codes"]
    )
    blockers = Counter(
        reason for row in guard_rows for reason in row["execution_blocking_reason_codes"]
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-runtime-report-v2",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "runtime_created_ts": config.runtime_created_ts,
        "candidate_name": CANDIDATE_NAME,
        "approved_unlock_manifest": _descriptor(unlock_path),
        "approved_unlock_manifest_id": unlock["manifest_id"],
        "paper_candidate_allowed_scope": EXPECTED_SCOPE,
        "bounded_complete_round_target": EXPECTED_ROUND_COUNT,
        "captured_round_attempt_count": len(capture_audits),
        "complete_round_count": len(valid),
        "hard_capture_failure_round_count": len(capture_audits) - len(valid),
        "hard_capture_failure_reason_distribution": dict(sorted(hard_reasons.items())),
        "provider_fail_fast_consecutive_hard_failure_limit": HARD_CAPTURE_FAILURE_LIMIT,
        "optional_provider_http_failure_may_continue": True,
        "raw_evidence_round_count": len(round_artifacts),
        "feature_row_count": len(feature_rows),
        "five_action_row_count": len(action_rows),
        "complete_five_action_grid": len(action_rows) == len(feature_rows) * 5,
        "score_row_count": len(scored_rows),
        "guard_decision_count": len(guard_rows),
        "guard_allowed_count": len(intents),
        "guard_blocked_count": len(guard_rows) - len(intents),
        "guard_blocking_reason_distribution": dict(sorted(blockers.items())),
        "paper_intent_count": len(intents),
        "paper_fill_count": len(fills),
        "paper_exit_signal_count": len(exit_signals),
        "paper_exit_intent_count": len(exit_intents),
        "paper_exit_fill_count": len(exit_fills),
        "paper_ledger_entry_count": len(ledger),
        "open_paper_position_count": positions["open_position_count"],
        "closed_paper_position_count": positions["closed_position_count"],
        "sell_before_close_residual_position_count": positions[
            "sell_before_close_residual_position_count"
        ],
        "hold_to_settlement_open_position_count": positions[
            "hold_to_settlement_open_position_count"
        ],
        "position_lifecycle_report_id": lifecycle_report["report_id"],
        "position_lifecycle_reconciled": lifecycle_report[
            "position_lifecycle_reconciled"
        ],
        "sell_before_close_lifecycle_complete": lifecycle_report[
            "sell_before_close_lifecycle_complete"
        ],
        "sell_before_close_exit_rule_id": SELL_BEFORE_CLOSE_EXIT_RULE_ID,
        "closed_before_settlement_realized_trade_pnl_sum": lifecycle_report[
            "closed_before_settlement_realized_trade_pnl_sum"
        ],
        "paper_order_size_distribution": dict(
            sorted(Counter(str(row["paper_order_size"]) for row in intents).items())
        ),
        "maximum_paper_order_size": MAXIMUM_PAPER_ORDER_SIZE,
        "decision_target_outcome_or_pnl_accessed": False,
        "settlement_mode": "asynchronous_after_decision_freeze",
        "settlement_may_block_next_round_collection": False,
        "unresolved_positions_remain_unresolved": True,
        "source_model": lineage["model_descriptor"],
        "market_clustered_mean_risk_calibration": lineage["calibration_descriptor"],
        "feature_contract": lineage["feature_descriptor"],
        "v6_2_profile": lineage["profile_descriptor"],
        "source_or_o_score_mutated": False,
        "threshold_cost_sizing_or_guard_mutated": False,
        "paper_results_used_for_tuning": False,
        "paper_results_are_promotion_evidence": False,
        "runtime_safety_passed": safety["runtime_safety_passed"],
        "paper_candidate_allowed": True,
        "paper_only": True,
        "capital_at_risk": False,
        "live_trading_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _write_per_round_artifacts(
    root: Path,
    *,
    capture_audits: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    exit_signals: list[dict[str, Any]],
    exit_intents: list[dict[str, Any]],
    exit_fills: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_market = defaultdict(lambda: defaultdict(list))
    for name, rows in (
        ("feature_rows", feature_rows),
        ("five_action_rows", action_rows),
        ("score_rows", scored_rows),
        ("guard_rows", guard_rows),
        ("paper_intents", intents),
        ("paper_fills", fills),
        ("paper_exit_signals", exit_signals),
        ("paper_exit_intents", exit_intents),
        ("paper_exit_fills", exit_fills),
        ("paper_ledger", ledger),
    ):
        for row in rows:
            by_market[str(row["market_id"])][name].append(row)
    output = []
    for audit in capture_audits:
        raw_market = audit.get("raw_artifacts", {}).get("raw_polymarket_markets.jsonl")
        market_rows = _load_jsonl(Path(raw_market["path"])) if raw_market else []
        market_id = str(market_rows[0]["market_id"]) if market_rows else audit["run_id"]
        slug = str(market_rows[0].get("slug") or market_id) if market_rows else market_id
        round_dir = root / slug
        round_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = round_dir / "raw_evidence_descriptors.json"
        _write_json(
            evidence_path,
            {
                "capture_run_id": audit["run_id"],
                "market_id": market_id,
                "raw_artifacts": audit["raw_artifacts"],
                "provider_raw_artifacts": audit["provider_raw_artifacts"],
                "resolution_artifact_opened_for_decision": False,
            },
        )
        descriptors = {"raw_evidence_descriptors": _descriptor(evidence_path)}
        for name in (
            "feature_rows",
            "five_action_rows",
            "score_rows",
            "guard_rows",
            "paper_intents",
            "paper_fills",
            "paper_exit_signals",
            "paper_exit_intents",
            "paper_exit_fills",
            "paper_ledger",
        ):
            path = round_dir / f"{name}.jsonl"
            _write_jsonl(path, by_market[market_id][name])
            descriptors[name] = _jsonl_descriptor(path)
        output.append(
            {
                "market_id": market_id,
                "slug": slug,
                "capture_run_id": audit["run_id"],
                "hard_capture_failure": audit["hard_capture_failure"],
                "artifacts": descriptors,
            }
        )
    return output


def _runtime_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.2 Bounded Paper Canary",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- complete_round_count: `{report['complete_round_count']}`",
            f"- guard_allowed_count: `{report['guard_allowed_count']}`",
            f"- paper_intent_count: `{report['paper_intent_count']}`",
            f"- paper_fill_count: `{report['paper_fill_count']}`",
            f"- paper_exit_fill_count: `{report['paper_exit_fill_count']}`",
            f"- open_paper_position_count: `{report['open_paper_position_count']}`",
            f"- sell_before_close_residual_position_count: "
            f"`{report['sell_before_close_residual_position_count']}`",
            f"- sell_before_close_lifecycle_complete: "
            f"`{str(report['sell_before_close_lifecycle_complete']).lower()}`",
            f"- runtime_safety_passed: `{str(report['runtime_safety_passed']).lower()}`",
            f"- paper_candidate_allowed: `{str(report['paper_candidate_allowed']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "This is a bounded local-paper canary. It is not live or promotion evidence.",
            "",
        ]
    )


def _find_nonempty_fields(payload: Any, forbidden: frozenset[str]) -> list[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in forbidden and value not in (None, "", [], {}):
                found.add(key)
            found.update(_find_nonempty_fields(value, forbidden))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_find_nonempty_fields(value, forbidden))
    return sorted(found)


def _verified_descriptor(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} descriptor missing")
    path = Path(str(payload.get("path") or "")).expanduser().resolve()
    expected = str(payload.get("sha256") or "").lower()
    _verify_pin(path, expected, name)
    return {**payload, "path": str(path), "sha256": expected}


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _jsonl_descriptor(path: Path) -> dict[str, Any]:
    return {**_descriptor(path), "row_count": len(_load_jsonl(path))}


def _verify_pin(path: Path, expected: str, name: str) -> None:
    _require_sha256(expected, name)
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if _sha256_file(path) != expected.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError(f"{name} SHA-256 must be a hex digest")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


__all__ = [
    "HARD_CAPTURE_FAILURE_LIMIT",
    "MarketClusteredMeanEVV62PaperCanaryConfig",
    "classify_capture_hard_failure",
    "run_market_clustered_mean_ev_v6_2_paper_canary",
    "validate_v6_2_paper_candidate_unlock",
]
