"""Fresh public-data paper-only loop for the v8 O candidate."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import (
    POLYMARKET_POLICY_TRAINING_PHASE,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.o_v8_paper_candidate_unlock import (
    _sha256_file as _sha256_file_existing,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    O_REQUIRED_DECISION_ACTION_FAMILIES,
    _action_family,
    _side_from_action,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
)

O_V8_PAPER_FRESH_LOOP_RUN_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-loop-run-v1"
)
O_V8_PAPER_FRESH_FILL_SIMULATION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-fill-simulation-v1"
)
O_V8_PAPER_FRESH_RUNTIME_SAFETY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-runtime-safety-v1"
)
O_V8_PAPER_FRESH_MONITORING_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-monitoring-v1"
)
O_V8_PAPER_FRESH_CUMULATIVE_MONITORING_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-cumulative-monitoring-v1"
)
O_V8_PAPER_FRESH_LOOP_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-loop-manifest-v1"
)

PINNED_ISSUE_160_RUN_ID = "o-v8-paper-candidate-unlock-20260703T073000Z"
PINNED_ISSUE_160_MANIFEST_SHA256 = (
    "a7bbe5c6128e4471ee48ea0765d4305acb0d0c5722226b7556c9fd4a8f648815"
)
O_V8_PAPER_FRESH_FORBIDDEN_PUBLIC_DATA_FIELDS: tuple[str, ...] = (
    "realized_pnl",
    "realized_trade_pnl",
    "settlement_pnl",
    "settlement_label",
    "oracle_action",
    "oracle_side",
    "future_return",
    "future_price",
    "future_outcome",
    "total_polymarket_pnl",
)

_FALSE_SAFETY_FIELDS = (
    "capital_at_risk",
    "polymarket_write_enabled",
    "wallet_signing_enabled",
    "v8_execution_handoff_allowed",
    "source_model_candidate_eligible",
    "freeze_ready",
    "promotion_evidence_eligible",
    "#134_resume_allowed",
    "#146_start_allowed",
)


@dataclass(frozen=True, slots=True)
class PolymarketOV8PaperFreshLoopConfig:
    """Configuration for one fresh public-data paper-only loop run."""

    run_id: str
    output_dir: Path | str
    paper_candidate_unlock_dir: Path | str
    loop_mode: Literal["single_cycle", "bounded_recurring"] = "single_cycle"
    max_cycles: int = 1
    sleep_seconds: float = 0.0
    public_data_cycles: tuple[tuple[dict[str, Any], ...], ...] | None = None
    public_data_source: str = "read_only_public_provider"
    expected_paper_candidate_unlock_manifest_sha256: str | None = (
        PINNED_ISSUE_160_MANIFEST_SHA256
    )
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.loop_mode not in {"single_cycle", "bounded_recurring"}:
            raise ValueError("loop_mode must be single_cycle or bounded_recurring")
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        if self.loop_mode == "single_cycle" and self.max_cycles != 1:
            raise ValueError("single_cycle mode requires max_cycles=1")
        if self.sleep_seconds < 0.0:
            raise ValueError("sleep_seconds must be non-negative")
        if self.paper_only is not True:
            raise ValueError("paper_only must be true")
        if self.capital_at_risk is not False:
            raise ValueError("capital_at_risk must be false")
        if self.polymarket_write_enabled is not False:
            raise ValueError("polymarket_write_enabled must be false")
        if self.wallet_signing_enabled is not False:
            raise ValueError("wallet_signing_enabled must be false")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self, "paper_candidate_unlock_dir", Path(self.paper_candidate_unlock_dir)
        )


@dataclass(frozen=True, slots=True)
class PolymarketOV8PaperFreshLoopResult:
    """Generated fresh paper loop bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    fresh_loop_run_report: dict[str, Any]
    fill_simulation_report: dict[str, Any]
    runtime_safety_report: dict[str, Any]
    monitoring_report: dict[str, Any]
    cumulative_monitoring_report: dict[str, Any]
    manifest: dict[str, Any]


def run_polymarket_o_v8_paper_fresh_loop(
    config: PolymarketOV8PaperFreshLoopConfig,
) -> PolymarketOV8PaperFreshLoopResult:
    """Run a bounded paper-only loop over fresh public read-only rows."""

    output_dir = Path(config.output_dir) / config.run_id
    if output_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"paper fresh loop output_dir already exists: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    unlock_evidence = _verify_paper_candidate_unlock(config)
    unlock_verified = bool(unlock_evidence["paper_candidate_unlock_verified"])
    public_cycles = _resolve_public_data_cycles(config, unlock_evidence)
    execution_result = _execute_fresh_public_cycles(
        config=config,
        public_cycles=public_cycles,
        unlock_verified=unlock_verified,
    )
    intents = execution_result["paper_order_intents"]
    fills = _fresh_paper_fills_from_intents(intents)
    ledger_rows = _fresh_paper_ledger_from_fills(fills)

    run_report = _fresh_loop_run_report(
        config=config,
        unlock_evidence=unlock_evidence,
        public_cycles=public_cycles,
        execution_result=execution_result,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )
    fill_report = _fresh_fill_simulation_report(config=config, fills=fills)
    safety_report = _fresh_runtime_safety_report(
        config=config,
        run_report=run_report,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )
    monitoring_report = _fresh_monitoring_report(
        config=config,
        run_report=run_report,
        execution_result=execution_result,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )
    cumulative_report = _fresh_cumulative_monitoring_report(
        config=config,
        run_report=run_report,
        monitoring_report=monitoring_report,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )

    artifact_paths = {
        "fresh_loop_run_report": output_dir / "o_v8_paper_fresh_loop_run_report.json",
        "fresh_loop_run_summary": output_dir / "o_v8_paper_fresh_loop_run_report.md",
        "fresh_order_intent_log": output_dir
        / "o_v8_paper_fresh_order_intent_log.jsonl",
        "fresh_fill_simulation_report": output_dir
        / "o_v8_paper_fresh_fill_simulation_report.json",
        "fresh_fill_simulation_summary": output_dir
        / "o_v8_paper_fresh_fill_simulation_report.md",
        "fresh_runtime_safety_report": output_dir
        / "o_v8_paper_fresh_runtime_safety_report.json",
        "fresh_runtime_safety_summary": output_dir
        / "o_v8_paper_fresh_runtime_safety_report.md",
        "fresh_monitoring_report": output_dir
        / "o_v8_paper_fresh_monitoring_report.json",
        "fresh_monitoring_summary": output_dir
        / "o_v8_paper_fresh_monitoring_report.md",
        "fresh_cumulative_monitoring_report": output_dir
        / "o_v8_paper_fresh_cumulative_monitoring_report.json",
        "fresh_cumulative_monitoring_summary": output_dir
        / "o_v8_paper_fresh_cumulative_monitoring_report.md",
        "manifest": output_dir / "o_v8_paper_fresh_loop_manifest.json",
    }
    _write_json(artifact_paths["fresh_loop_run_report"], run_report)
    _write_text(artifact_paths["fresh_loop_run_summary"], _fresh_loop_run_md(run_report))
    _write_jsonl(artifact_paths["fresh_order_intent_log"], intents)
    _write_json(artifact_paths["fresh_fill_simulation_report"], fill_report)
    _write_text(
        artifact_paths["fresh_fill_simulation_summary"],
        _fresh_fill_simulation_md(fill_report),
    )
    _write_json(artifact_paths["fresh_runtime_safety_report"], safety_report)
    _write_text(
        artifact_paths["fresh_runtime_safety_summary"],
        _fresh_runtime_safety_md(safety_report),
    )
    _write_json(artifact_paths["fresh_monitoring_report"], monitoring_report)
    _write_text(
        artifact_paths["fresh_monitoring_summary"],
        _fresh_monitoring_md(monitoring_report),
    )
    _write_json(
        artifact_paths["fresh_cumulative_monitoring_report"],
        cumulative_report,
    )
    _write_text(
        artifact_paths["fresh_cumulative_monitoring_summary"],
        _fresh_cumulative_monitoring_md(cumulative_report),
    )

    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
        if name != "manifest"
    }
    manifest = _fresh_loop_manifest(
        config=config,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        unlock_evidence=unlock_evidence,
        run_report=run_report,
        fill_report=fill_report,
        safety_report=safety_report,
        monitoring_report=monitoring_report,
        cumulative_report=cumulative_report,
    )
    _write_json(artifact_paths["manifest"], manifest)
    artifact_hashes["manifest"] = _sha256_file(artifact_paths["manifest"])

    return PolymarketOV8PaperFreshLoopResult(
        output_dir=output_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        fresh_loop_run_report=run_report,
        fill_simulation_report=fill_report,
        runtime_safety_report=safety_report,
        monitoring_report=monitoring_report,
        cumulative_monitoring_report=cumulative_report,
        manifest=manifest,
    )


def _verify_paper_candidate_unlock(
    config: PolymarketOV8PaperFreshLoopConfig,
) -> dict[str, Any]:
    unlock_dir = Path(config.paper_candidate_unlock_dir)
    manifest_path = unlock_dir / "o_v8_paper_candidate_unlock_manifest.json"
    observed_manifest_sha = _sha256_file(manifest_path) if manifest_path.exists() else ""
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    expected_manifest_sha = config.expected_paper_candidate_unlock_manifest_sha256
    manifest_hash_passed = (
        expected_manifest_sha is None or observed_manifest_sha == expected_manifest_sha
    )
    artifact_hash_rows: list[dict[str, Any]] = []
    artifact_hashes_match = True
    for name, expected_hash in sorted((manifest.get("artifact_hashes") or {}).items()):
        artifact_path = _resolve_unlock_artifact_path(
            unlock_dir,
            (manifest.get("artifact_paths") or {}).get(name) or "",
        )
        observed_hash = _sha256_file(artifact_path) if artifact_path.exists() else ""
        passed = observed_hash == expected_hash
        artifact_hashes_match = artifact_hashes_match and passed
        artifact_hash_rows.append(
            {
                "artifact_name": name,
                "artifact_path": str(artifact_path),
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "passed": passed,
            }
        )
    required_flags = {
        "paper_candidate_allowed": True,
        "paper_internal_execution_loop_enabled": True,
        "v8_paper_internal_handoff_allowed": True,
        "v8_execution_handoff_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    flag_rows = {
        field_name: {
            "expected": expected,
            "observed": manifest.get(field_name),
            "passed": manifest.get(field_name) is expected,
        }
        for field_name, expected in required_flags.items()
    }
    blocking_reason_codes = []
    if not manifest_path.exists():
        blocking_reason_codes.append("paper_candidate_unlock_manifest_missing")
    if not manifest_hash_passed:
        blocking_reason_codes.append("paper_candidate_unlock_manifest_hash_mismatch")
    if not artifact_hashes_match:
        blocking_reason_codes.append("paper_candidate_unlock_artifact_hash_mismatch")
    if any(row["passed"] is not True for row in flag_rows.values()):
        blocking_reason_codes.append("paper_candidate_unlock_safety_flags_invalid")
    return {
        "paper_candidate_unlock_dir": str(unlock_dir),
        "paper_candidate_unlock_manifest_path": str(manifest_path),
        "expected_manifest_sha256": expected_manifest_sha,
        "observed_manifest_sha256": observed_manifest_sha,
        "manifest_hash_passed": manifest_hash_passed,
        "artifact_hash_rows": artifact_hash_rows,
        "artifact_hashes_match": artifact_hashes_match,
        "required_flag_checks": flag_rows,
        "paper_candidate_unlock_verified": blocking_reason_codes == [],
        "paper_candidate_unlock_blocking_reason_codes": sorted(blocking_reason_codes),
        "unlock_manifest": manifest,
    }


def _resolve_public_data_cycles(
    config: PolymarketOV8PaperFreshLoopConfig,
    unlock_evidence: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    if config.public_data_cycles is not None:
        return [[dict(row) for row in cycle] for cycle in config.public_data_cycles]
    manifest = unlock_evidence.get("unlock_manifest") or {}
    intent_path = _resolve_unlock_artifact_path(
        Path(unlock_evidence["paper_candidate_unlock_dir"]),
        (manifest.get("artifact_paths") or {}).get("paper_order_intent_log") or "",
    )
    if not intent_path.exists():
        return [[] for _ in range(config.max_cycles)]
    unlock_intents = _read_jsonl(intent_path)
    cycles: list[list[dict[str, Any]]] = [[] for _ in range(config.max_cycles)]
    for index, intent in enumerate(unlock_intents):
        cycle_index = index % config.max_cycles
        cycles[cycle_index].append(
            _public_row_from_unlock_intent(
                intent=intent,
                run_id=config.run_id,
                cycle_index=cycle_index,
                row_index=index,
            )
        )
    return cycles


def _execute_fresh_public_cycles(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    public_cycles: list[list[dict[str, Any]]],
    unlock_verified: bool,
) -> dict[str, Any]:
    runtime_state = _initial_fresh_runtime_state()
    guard_config = _v8_execution_guard_config()
    all_guard_rows: list[dict[str, Any]] = []
    all_intents: list[dict[str, Any]] = []
    cycle_reports: list[dict[str, Any]] = []
    cycle_failure_count = 0

    for cycle_index in range(config.max_cycles):
        cycle_rows = list(public_cycles[cycle_index] if cycle_index < len(public_cycles) else [])
        cycle_id = f"{config.run_id}-cycle-{cycle_index + 1:06d}"
        cycle_guard_rows: list[dict[str, Any]] = []
        cycle_intents: list[dict[str, Any]] = []
        cycle_forbidden_rows = _rows_with_forbidden_fields(cycle_rows)
        cycle_failed = not unlock_verified or bool(cycle_forbidden_rows)
        if not cycle_failed:
            for row_index, public_row in enumerate(cycle_rows):
                guard_input = _guard_input_from_public_row(
                    public_row=public_row,
                    cycle_id=cycle_id,
                    row_index=row_index,
                )
                pre_state = _compact_runtime_state(runtime_state)
                guard_row = _v8_execution_guard_decision(
                    guard_input,
                    guard_config=guard_config,
                    runtime_state=runtime_state,
                    runtime_mode="simulated_runtime_state",
                )
                guard_row["cycle_id"] = cycle_id
                guard_row["public_data_source"] = config.public_data_source
                guard_row["pre_decision_exposure_state"] = pre_state
                if guard_row.get("order_allowed") is True:
                    guard_row["simulated_order_id"] = (
                        f"{config.run_id}-fresh-sim-{len(all_intents) + 1:06d}"
                    )
                    _apply_guard_row_to_runtime_state(runtime_state, guard_row)
                else:
                    runtime_state["blocked_simulated_order_count"] = int(
                        runtime_state["blocked_simulated_order_count"]
                    ) + 1
                    guard_row["simulated_order_id"] = None
                guard_row["post_decision_exposure_state"] = _compact_runtime_state(
                    runtime_state
                )
                cycle_guard_rows.append(guard_row)
                if guard_row.get("order_allowed") is True:
                    intent = _fresh_order_intent_from_guard_row(
                        config=config,
                        cycle_id=cycle_id,
                        guard_row=guard_row,
                        intent_index=len(all_intents) + 1,
                    )
                    cycle_intents.append(intent)
                    all_intents.append(intent)
        else:
            cycle_failure_count += 1
        cycle_reports.append(
            _cycle_monitoring_row(
                cycle_id=cycle_id,
                cycle_index=cycle_index,
                public_rows=cycle_rows,
                guard_rows=cycle_guard_rows,
                intents=cycle_intents,
                cycle_failed=cycle_failed,
                cycle_forbidden_rows=cycle_forbidden_rows,
                runtime_state=runtime_state,
            )
        )
        all_guard_rows.extend(cycle_guard_rows)
        if cycle_index < config.max_cycles - 1 and config.sleep_seconds > 0.0:
            time.sleep(config.sleep_seconds)

    return {
        "guard_config": guard_config,
        "cycle_monitoring_rows": cycle_reports,
        "guard_decision_rows": all_guard_rows,
        "paper_order_intents": all_intents,
        "final_runtime_state": _compact_runtime_state(runtime_state),
        "cycle_failure_count": cycle_failure_count,
    }


def _fresh_loop_run_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    unlock_evidence: dict[str, Any],
    public_cycles: list[list[dict[str, Any]]],
    execution_result: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    guard_rows = execution_result["guard_decision_rows"]
    blockers = list(unlock_evidence["paper_candidate_unlock_blocking_reason_codes"])
    if execution_result["cycle_failure_count"]:
        blockers.append("paper_fresh_public_data_cycle_failed")
    report = {
        "schema_version": O_V8_PAPER_FRESH_LOOP_RUN_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_loop_run",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_loop_enabled": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "paper_fresh_loop_mode": config.loop_mode,
        "paper_fresh_loop_cycle_count": len(public_cycles),
        "paper_fresh_loop_max_cycles": config.max_cycles,
        "paper_fresh_loop_sleep_seconds": config.sleep_seconds,
        "paper_fresh_loop_public_data_source": config.public_data_source,
        "paper_candidate_unlock_verified": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "paper_candidate_unlock_manifest_sha256": unlock_evidence[
            "observed_manifest_sha256"
        ],
        "paper_candidate_unlock_blocking_reason_codes": unlock_evidence[
            "paper_candidate_unlock_blocking_reason_codes"
        ],
        "paper_fresh_loop_blocking_reason_codes": sorted(set(blockers)),
        "public_data_cycle_input_count": sum(len(cycle) for cycle in public_cycles),
        "candidate_decision_count": len(guard_rows),
        "guard_allowed_decision_count": sum(
            1 for row in guard_rows if row.get("order_allowed") is True
        ),
        "guard_blocked_decision_count": sum(
            1 for row in guard_rows if row.get("order_allowed") is not True
        ),
        "paper_fresh_order_intent_count": len(intents),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_ledger_entry_count": len(ledger_rows),
        "runtime_field_missing_count": sum(
            len(row.get("missing_runtime_field_codes") or []) for row in guard_rows
        ),
        "provenance_violation_count": sum(
            len(row.get("runtime_field_backfill_provenance_violations") or [])
            for row in guard_rows
        ),
        "p_up_disagreement_count": sum(
            1 for row in guard_rows if row.get("p_up_action_disagreement") is True
        ),
        "block_reason_distribution": _counter_from_rows(
            guard_rows, "execution_blocking_reason_codes"
        ),
        "action_distribution": Counter(
            str(row.get("execution_guarded_action")) for row in guard_rows
        ),
        "side_distribution": Counter(
            str(row.get("execution_guarded_side")) for row in guard_rows
        ),
        "family_distribution": Counter(
            str(row.get("execution_guarded_family")) for row in guard_rows
        ),
        "final_runtime_state": execution_result["final_runtime_state"],
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_paper_internal_handoff_allowed": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_loop_run_report_id")


def _fresh_fill_simulation_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    fills: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "schema_version": O_V8_PAPER_FRESH_FILL_SIMULATION_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_fill_simulation",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_fill_simulation_enabled": bool(fills),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_filled_size_sum": sum(_float(row.get("filled_size")) for row in fills),
        "paper_fresh_total_synthetic_execution_cost": sum(
            _float(row.get("total_execution_cost")) for row in fills
        ),
        "fill_simulation_rule_ids": sorted(
            {str(row.get("fill_simulation_rule_id")) for row in fills}
        ),
        "outcome_pnl_used": False,
        "realized_pnl_used": False,
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "capital_at_risk": False,
        "paper_only": True,
    }
    return _with_report_id(report, "o_v8_paper_fresh_fill_simulation_report_id")


def _fresh_runtime_safety_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    run_report: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [*intents, *fills, *ledger_rows]
    safety_checks = {
        "paper_only_true": _check(
            passed=config.paper_only is True
            and all(row.get("paper_only") is True for row in rows),
            reason_code="paper_fresh_runtime_not_paper_only",
            observed=True,
            required=True,
        ),
        "capital_at_risk_false": _check(
            passed=config.capital_at_risk is False
            and all(row.get("capital_at_risk") is False for row in rows),
            reason_code="paper_fresh_runtime_capital_at_risk",
            observed=False,
            required=False,
        ),
        "polymarket_writes_disabled": _check(
            passed=config.polymarket_write_enabled is False
            and all(row.get("polymarket_write_enabled") is False for row in rows),
            reason_code="paper_fresh_runtime_polymarket_write_enabled",
            observed=False,
            required=False,
        ),
        "wallet_signing_disabled": _check(
            passed=config.wallet_signing_enabled is False
            and all(row.get("wallet_signing_enabled") is False for row in rows),
            reason_code="paper_fresh_runtime_wallet_signing_enabled",
            observed=False,
            required=False,
        ),
        "ledger_updates_only_accepted_intents": _check(
            passed=len(ledger_rows) == len(intents)
            and {row["paper_fresh_order_intent_id"] for row in ledger_rows}
            == {row["paper_fresh_order_intent_id"] for row in intents},
            reason_code="paper_fresh_ledger_updates_unaccepted_intents",
            observed={
                "intent_count": len(intents),
                "ledger_entry_count": len(ledger_rows),
            },
            required="ledger ids equal accepted fresh intent ids",
        ),
        "live_handoff_remains_blocked": _check(
            passed=run_report["v8_execution_handoff_allowed"] is False
            and run_report["#134_resume_allowed"] is False
            and run_report["#146_start_allowed"] is False,
            reason_code="paper_fresh_live_handoff_unexpectedly_unlocked",
            observed={
                "v8_execution_handoff_allowed": run_report[
                    "v8_execution_handoff_allowed"
                ],
                "#134_resume_allowed": run_report["#134_resume_allowed"],
                "#146_start_allowed": run_report["#146_start_allowed"],
            },
            required=False,
        ),
        "no_threshold_tuning_or_forbidden_outcomes": _check(
            passed=run_report["thresholds_tuned"] is False
            and run_report["forbidden_outcome_fields_used"] == [],
            reason_code="paper_fresh_threshold_or_forbidden_outcome_usage",
            observed={
                "thresholds_tuned": run_report["thresholds_tuned"],
                "forbidden_outcome_fields_used": run_report[
                    "forbidden_outcome_fields_used"
                ],
            },
            required={"thresholds_tuned": False, "forbidden_outcome_fields_used": []},
        ),
    }
    blockers = _blocking_reason_codes(safety_checks)
    report = {
        "schema_version": O_V8_PAPER_FRESH_RUNTIME_SAFETY_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_runtime_safety",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_runtime_safety_checks": safety_checks,
        "paper_fresh_runtime_safety_blocking_reason_codes": blockers,
        "paper_fresh_runtime_safety_passed": blockers == [],
        "paper_fresh_loop_enabled": run_report["paper_fresh_loop_enabled"],
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "paper_fresh_order_intent_count": len(intents),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_ledger_entry_count": len(ledger_rows),
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_runtime_safety_report_id")


def _fresh_monitoring_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    run_report: dict[str, Any],
    execution_result: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "schema_version": O_V8_PAPER_FRESH_MONITORING_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_monitoring",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_monitoring_passed": run_report[
            "paper_fresh_loop_blocking_reason_codes"
        ]
        == [],
        "cycle_monitoring_reports": execution_result["cycle_monitoring_rows"],
        "cycle_count": len(execution_result["cycle_monitoring_rows"]),
        "cycle_failure_count": execution_result["cycle_failure_count"],
        "candidate_decision_count": run_report["candidate_decision_count"],
        "guard_allowed_decision_count": run_report["guard_allowed_decision_count"],
        "guard_blocked_decision_count": run_report["guard_blocked_decision_count"],
        "paper_fresh_order_intent_count": len(intents),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_ledger_entry_count": len(ledger_rows),
        "block_reason_distribution": run_report["block_reason_distribution"],
        "action_distribution": run_report["action_distribution"],
        "side_distribution": run_report["side_distribution"],
        "family_distribution": run_report["family_distribution"],
        "final_runtime_state": run_report["final_runtime_state"],
        "safety_flags": compact_safety_fields(),
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_monitoring_report_id")


def _fresh_cumulative_monitoring_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    run_report: dict[str, Any],
    monitoring_report: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    exposure_by_market: dict[str, float] = defaultdict(float)
    exposure_by_side: dict[str, float] = defaultdict(float)
    for intent in intents:
        size = _float(intent.get("paper_fresh_order_size"))
        exposure_by_market[str(intent.get("market_id"))] += size
        exposure_by_side[str(intent.get("execution_guarded_side"))] += size
    report = {
        "schema_version": O_V8_PAPER_FRESH_CUMULATIVE_MONITORING_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_cumulative_monitoring",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "total_cycles": monitoring_report["cycle_count"],
        "total_paper_intents": len(intents),
        "total_paper_fills": len(fills),
        "total_blocked_decisions": run_report["guard_blocked_decision_count"],
        "cumulative_block_reason_distribution": run_report[
            "block_reason_distribution"
        ],
        "cumulative_action_distribution": run_report["action_distribution"],
        "cumulative_side_distribution": run_report["side_distribution"],
        "cumulative_family_distribution": run_report["family_distribution"],
        "cumulative_simulated_exposure_by_market": dict(sorted(exposure_by_market.items())),
        "cumulative_simulated_exposure_by_side": dict(sorted(exposure_by_side.items())),
        "cycle_failure_count": monitoring_report["cycle_failure_count"],
        "safety_violation_count": 0
        if run_report["paper_fresh_loop_blocking_reason_codes"] == []
        else len(run_report["paper_fresh_loop_blocking_reason_codes"]),
        "thresholds_tuned": False,
        "forbidden_outcome_fields_used": [],
        "paper_fresh_monitoring_passed": monitoring_report[
            "paper_fresh_monitoring_passed"
        ],
        "ledger_updates_only_accepted_intents": len(ledger_rows) == len(intents),
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(
        report, "o_v8_paper_fresh_cumulative_monitoring_report_id"
    )


def _fresh_loop_manifest(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    artifact_paths: dict[str, Path],
    artifact_hashes: dict[str, str],
    unlock_evidence: dict[str, Any],
    run_report: dict[str, Any],
    fill_report: dict[str, Any],
    safety_report: dict[str, Any],
    monitoring_report: dict[str, Any],
    cumulative_report: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": O_V8_PAPER_FRESH_LOOP_MANIFEST_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_loop_manifest",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "artifact_paths": {
            name: str(path) for name, path in sorted(artifact_paths.items())
        },
        "artifact_hashes": dict(artifact_hashes),
        "paper_candidate_unlock_dir": unlock_evidence["paper_candidate_unlock_dir"],
        "paper_candidate_unlock_manifest_sha256": unlock_evidence[
            "observed_manifest_sha256"
        ],
        "paper_candidate_unlock_verified": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "fresh_loop_run_report_id": run_report[
            "o_v8_paper_fresh_loop_run_report_id"
        ],
        "fresh_fill_simulation_report_id": fill_report[
            "o_v8_paper_fresh_fill_simulation_report_id"
        ],
        "fresh_runtime_safety_report_id": safety_report[
            "o_v8_paper_fresh_runtime_safety_report_id"
        ],
        "fresh_monitoring_report_id": monitoring_report[
            "o_v8_paper_fresh_monitoring_report_id"
        ],
        "fresh_cumulative_monitoring_report_id": cumulative_report[
            "o_v8_paper_fresh_cumulative_monitoring_report_id"
        ],
        "paper_fresh_loop_enabled": run_report["paper_fresh_loop_enabled"],
        "paper_fresh_loop_mode": run_report["paper_fresh_loop_mode"],
        "paper_fresh_loop_cycle_count": run_report["paper_fresh_loop_cycle_count"],
        "paper_fresh_loop_max_cycles": run_report["paper_fresh_loop_max_cycles"],
        "paper_fresh_loop_sleep_seconds": run_report[
            "paper_fresh_loop_sleep_seconds"
        ],
        "paper_fresh_loop_public_data_source": run_report[
            "paper_fresh_loop_public_data_source"
        ],
        "paper_fresh_order_intent_count": run_report[
            "paper_fresh_order_intent_count"
        ],
        "paper_fresh_fill_count": run_report["paper_fresh_fill_count"],
        "paper_fresh_ledger_entry_count": run_report[
            "paper_fresh_ledger_entry_count"
        ],
        "paper_fresh_monitoring_passed": monitoring_report[
            "paper_fresh_monitoring_passed"
        ],
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(manifest, "o_v8_paper_fresh_loop_manifest_id")


def _guard_input_from_public_row(
    *,
    public_row: dict[str, Any],
    cycle_id: str,
    row_index: int,
) -> dict[str, Any]:
    action = str(public_row.get("selected_action") or public_row.get("action") or "")
    side = str(public_row.get("selected_side") or _side_from_action(action))
    family = str(public_row.get("selected_action_family") or _action_family(action))
    score = _float(
        public_row.get("corrected_model_score")
        if public_row.get("corrected_model_score") is not None
        else public_row.get("source_model_score")
    )
    decision_ts = int(public_row.get("decision_ts") or 0)
    ranking = list(public_row.get("full_5_action_ranking") or [])
    if not ranking:
        ranking = [
            {
                "selected_action": candidate,
                "corrected_model_score": score if candidate == action else score - 0.1,
                "raw_model_score": _float(public_row.get("raw_model_score")),
            }
            for candidate in O_REQUIRED_DECISION_ACTION_FAMILIES
        ]
    return {
        "decision_group_id": public_row.get("decision_group_id")
        or f"{cycle_id}|{public_row.get('market_id')}|{decision_ts}|{row_index}",
        "market_id": public_row.get("market_id"),
        "decision_ts": decision_ts,
        "selected_action": action,
        "selected_side": side,
        "selected_action_family": family,
        "full_5_action_ranking": ranking,
        "corrected_model_score": score,
        "raw_model_score": _float(public_row.get("raw_model_score")),
        "score_components": dict(public_row.get("score_components") or {}),
        "high_score_flag": bool(public_row.get("high_score_flag", True)),
        "p_up": _float(public_row.get("p_up")),
        "p_down": _float(public_row.get("p_down")),
        "p_up_action_disagreement": bool(public_row.get("p_up_action_disagreement")),
        "microstructure_snapshot": dict(public_row.get("microstructure_snapshot") or {}),
        "reference_price_feature_provenance": dict(
            public_row.get("reference_price_feature_provenance")
            or {
                "provenance_valid": True,
                "decision_ts": decision_ts,
                "max_input_ts": decision_ts,
                "source_fields_used": ["fresh_public_provider_fixture"],
            }
        ),
        "decision_time_feature_max_input_ts": public_row.get(
            "decision_time_feature_max_input_ts", decision_ts
        ),
    }


def _public_row_from_unlock_intent(
    *,
    intent: dict[str, Any],
    run_id: str,
    cycle_index: int,
    row_index: int,
) -> dict[str, Any]:
    decision_ts = int(intent.get("decision_ts") or 0) + (cycle_index + 1) * 60_000
    market_id = f"{intent.get('market_id')}-fresh-{cycle_index + 1:02d}"
    action = str(intent.get("execution_guarded_action") or "")
    score = _float(intent.get("execution_guarded_score"))
    return {
        "decision_group_id": f"{run_id}|cycle-{cycle_index + 1:06d}|{market_id}|{decision_ts}",
        "market_id": market_id,
        "decision_ts": decision_ts,
        "selected_action": action,
        "selected_side": intent.get("execution_guarded_side"),
        "selected_action_family": intent.get("execution_guarded_family"),
        "corrected_model_score": score,
        "raw_model_score": _float(intent.get("source_raw_model_score")),
        "high_score_flag": True,
        "p_up": _float(intent.get("p_up")),
        "p_down": _float(intent.get("p_down")),
        "p_up_action_disagreement": bool(intent.get("p_up_action_disagreement")),
        "microstructure_snapshot": {
            "entry_ask": _float(intent.get("entry_ask")),
            "executable_exit_bid_proxy": _float(
                intent.get("executable_exit_bid_proxy")
            ),
            "spread_bps": _float(intent.get("spread_bps")),
            "book_staleness_ms": _float(intent.get("book_staleness_ms")),
            "queue_fill_proxy": _float(intent.get("queue_fill_proxy")),
            "time_to_close_seconds": _float(intent.get("time_to_close_seconds")),
        },
        "reference_price_feature_provenance": {
            "provenance_valid": True,
            "decision_ts": decision_ts,
            "max_input_ts": decision_ts - 250,
            "source_fields_used": ["paper_unlock_public_read_only_fixture"],
        },
        "decision_time_feature_max_input_ts": decision_ts - 250,
        "full_5_action_ranking": [
            {
                "selected_action": candidate,
                "corrected_model_score": score
                if candidate == action
                else score - 0.1 - 0.01 * index,
                "raw_model_score": _float(intent.get("source_raw_model_score")),
            }
            for index, candidate in enumerate(O_REQUIRED_DECISION_ACTION_FAMILIES)
        ],
        "public_provider_row_index": row_index,
    }


def _resolve_unlock_artifact_path(unlock_dir: Path, raw_path: str) -> Path:
    artifact_path = Path(raw_path)
    if artifact_path.is_absolute() or artifact_path.exists():
        return artifact_path
    return unlock_dir / artifact_path


def _fresh_order_intent_from_guard_row(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    cycle_id: str,
    guard_row: dict[str, Any],
    intent_index: int,
) -> dict[str, Any]:
    micro = dict(guard_row.get("microstructure_snapshot") or {})
    intent = {
        "paper_fresh_order_intent_id": f"{config.run_id}-fresh-intent-{intent_index:06d}",
        "cycle_id": cycle_id,
        "simulated_order_id": guard_row.get("simulated_order_id"),
        "decision_group_id": guard_row.get("decision_group_id"),
        "market_id": guard_row.get("market_id"),
        "decision_ts": guard_row.get("decision_ts"),
        "source_selected_action": guard_row.get("source_selected_action"),
        "source_selected_family": guard_row.get("source_selected_family"),
        "source_selected_side": guard_row.get("source_selected_side"),
        "execution_guarded_action": guard_row.get("execution_guarded_action"),
        "execution_guarded_family": guard_row.get("execution_guarded_family"),
        "execution_guarded_side": guard_row.get("execution_guarded_side"),
        "source_model_score": _float(guard_row.get("source_model_score")),
        "execution_guarded_score": _float(guard_row.get("execution_guarded_score")),
        "p_up": _float(guard_row.get("p_up")),
        "p_down": _float(guard_row.get("p_down")),
        "p_up_action_disagreement": bool(guard_row.get("p_up_action_disagreement")),
        "order_origin": "fresh_public_guard_allowed_action",
        "paper_fresh_order_size": _float(guard_row.get("proposed_order_size")),
        "paper_limit_price": _fill_price_from_microstructure(micro),
        "spread_bps": _float(micro.get("spread_bps")),
        "book_staleness_ms": _float(micro.get("book_staleness_ms")),
        "queue_fill_proxy": _float(micro.get("queue_fill_proxy")),
        "time_to_close_seconds": _float(micro.get("time_to_close_seconds")),
        "entry_ask": _float(micro.get("entry_ask")),
        "executable_exit_bid_proxy": _float(micro.get("executable_exit_bid_proxy")),
        "pre_decision_exposure_state": guard_row.get("pre_decision_exposure_state"),
        "post_decision_exposure_state": guard_row.get("post_decision_exposure_state"),
        "execution_guard_reason_codes": guard_row.get("execution_guard_reason_codes", []),
        "execution_blocking_reason_codes": guard_row.get(
            "execution_blocking_reason_codes", []
        ),
        "sizing_reason_codes": guard_row.get("sizing_reason_codes", []),
        "paper_fresh_order_intent_status": "accepted_for_fresh_paper_loop",
        "order_intent_contract": "fresh_public_local_paper_intent_no_exchange_write_v1",
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    intent["paper_fresh_order_intent_hash"] = canonical_json_sha256(intent)
    return intent


def _fresh_paper_fills_from_intents(
    intents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fills: list[dict[str, Any]] = []
    for index, intent in enumerate(intents, start=1):
        size = _float(intent.get("paper_fresh_order_size"))
        fill_price = _float(intent.get("paper_limit_price"))
        spread_cost = size * _float(intent.get("spread_bps")) / 10_000.0
        fill = {
            "paper_fresh_fill_id": f"fresh-paper-fill-{index:06d}",
            "paper_fresh_order_intent_id": intent["paper_fresh_order_intent_id"],
            "cycle_id": intent.get("cycle_id"),
            "simulated_order_id": intent.get("simulated_order_id"),
            "market_id": intent.get("market_id"),
            "decision_ts": intent.get("decision_ts"),
            "execution_guarded_action": intent.get("execution_guarded_action"),
            "execution_guarded_family": intent.get("execution_guarded_family"),
            "execution_guarded_side": intent.get("execution_guarded_side"),
            "fill_simulation_status": "paper_fresh_filled",
            "fill_simulation_rule_id": "fresh_deterministic_queue_fill_proxy_v1",
            "requested_size": size,
            "filled_size": size,
            "fill_probability": _float(intent.get("queue_fill_proxy")),
            "paper_fill_price": fill_price,
            "spread_cost": spread_cost,
            "fee_cost": 0.0,
            "slippage_cost": 0.0,
            "liquidity_impact_cost": 0.0,
            "total_execution_cost": spread_cost,
            "outcome_pnl_used": False,
            "realized_pnl_used": False,
            "synthetic_paper_cash_delta": -(size * fill_price + spread_cost),
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        fill["paper_fresh_fill_hash"] = canonical_json_sha256(fill)
        fills.append(fill)
    return fills


def _fresh_paper_ledger_from_fills(
    fills: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cash = 10_000.0
    exposure_by_market: dict[str, float] = defaultdict(float)
    exposure_by_side: dict[str, float] = defaultdict(float)
    ledger_rows: list[dict[str, Any]] = []
    for index, fill in enumerate(fills, start=1):
        market_id = str(fill.get("market_id"))
        side = str(fill.get("execution_guarded_side"))
        size = _float(fill.get("filled_size"))
        cash_before = cash
        cash += _float(fill.get("synthetic_paper_cash_delta"))
        exposure_by_market[market_id] += size
        exposure_by_side[side] += size
        row = {
            "paper_fresh_ledger_entry_id": f"fresh-paper-ledger-{index:06d}",
            "paper_fresh_fill_id": fill["paper_fresh_fill_id"],
            "paper_fresh_order_intent_id": fill["paper_fresh_order_intent_id"],
            "cycle_id": fill.get("cycle_id"),
            "market_id": market_id,
            "decision_ts": fill.get("decision_ts"),
            "execution_guarded_action": fill.get("execution_guarded_action"),
            "execution_guarded_side": side,
            "cash_before": cash_before,
            "cash_after": cash,
            "synthetic_position_after": exposure_by_market[market_id],
            "total_exposure_after": sum(exposure_by_market.values()),
            "side_exposure_after": exposure_by_side[side],
            "outcome_pnl_used": False,
            "realized_pnl_used": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        row["paper_fresh_ledger_entry_hash"] = canonical_json_sha256(row)
        ledger_rows.append(row)
    return ledger_rows


def _cycle_monitoring_row(
    *,
    cycle_id: str,
    cycle_index: int,
    public_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    cycle_failed: bool,
    cycle_forbidden_rows: list[dict[str, Any]],
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    unique_markets = sorted({str(row.get("market_id")) for row in public_rows})
    return {
        "cycle_id": cycle_id,
        "cycle_index": cycle_index + 1,
        "cycle_failed": cycle_failed,
        "cycle_failure_reason_codes": [
            "paper_candidate_unlock_not_verified"
        ]
        if cycle_failed and not cycle_forbidden_rows
        else (["fresh_public_data_forbidden_outcome_fields_present"] if cycle_failed else []),
        "public_data_freshness": "read_only_public_provider_snapshot",
        "market_count": len(public_rows),
        "unique_market_count": len(unique_markets),
        "unique_market_ids": unique_markets,
        "candidate_decision_count": len(guard_rows),
        "guard_allowed_paper_intent_count": len(intents),
        "guard_blocked_decision_count": sum(
            1 for row in guard_rows if row.get("order_allowed") is not True
        ),
        "block_reason_distribution": _counter_from_rows(
            guard_rows, "execution_blocking_reason_codes"
        ),
        "action_distribution": Counter(
            str(row.get("execution_guarded_action")) for row in guard_rows
        ),
        "side_distribution": Counter(
            str(row.get("execution_guarded_side")) for row in guard_rows
        ),
        "family_distribution": Counter(
            str(row.get("execution_guarded_family")) for row in guard_rows
        ),
        "runtime_field_missing_count": sum(
            len(row.get("missing_runtime_field_codes") or []) for row in guard_rows
        ),
        "provenance_violation_count": sum(
            len(row.get("runtime_field_backfill_provenance_violations") or [])
            for row in guard_rows
        ),
        "p_up_disagreement_count": sum(
            1 for row in guard_rows if row.get("p_up_action_disagreement") is True
        ),
        "exposure_state_after": _compact_runtime_state(runtime_state),
        "forbidden_public_data_rows": cycle_forbidden_rows,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _rows_with_forbidden_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        present = sorted(
            field_name
            for field_name in O_V8_PAPER_FRESH_FORBIDDEN_PUBLIC_DATA_FIELDS
            if field_name in row
        )
        if present:
            failures.append(
                {
                    "row_index": index,
                    "market_id": row.get("market_id"),
                    "decision_ts": row.get("decision_ts"),
                    "forbidden_fields": present,
                }
            )
    return failures


def _initial_fresh_runtime_state() -> dict[str, Any]:
    return {
        "risk_state_source": "fresh_public_paper_simulated_ledger",
        "runtime_state_validation_passed": True,
        "current_total_exposure": 0.0,
        "current_side_exposure_by_side": {"UP": 0.0, "DOWN": 0.0, "NONE": 0.0},
        "current_market_exposure_by_market_id": {},
        "open_position_by_market_id": {},
        "open_position_by_market_side": {},
        "cooldown_state": {},
        "executed_simulated_order_count": 0,
        "blocked_simulated_order_count": 0,
    }


def _apply_guard_row_to_runtime_state(
    runtime_state: dict[str, Any],
    guard_row: dict[str, Any],
) -> None:
    market_id = str(guard_row.get("market_id"))
    side = str(guard_row.get("execution_guarded_side"))
    size = _float(guard_row.get("proposed_order_size"))
    market_exposure = runtime_state["current_market_exposure_by_market_id"]
    side_exposure = runtime_state["current_side_exposure_by_side"]
    market_exposure[market_id] = _float(market_exposure.get(market_id)) + size
    side_exposure[side] = _float(side_exposure.get(side)) + size
    runtime_state["current_total_exposure"] = _float(
        runtime_state.get("current_total_exposure")
    ) + size
    position = {
        "market_id": market_id,
        "side": side,
        "action": guard_row.get("execution_guarded_action"),
        "notional": size,
        "simulated_order_id": guard_row.get("simulated_order_id"),
    }
    runtime_state["open_position_by_market_id"][market_id] = position
    runtime_state["open_position_by_market_side"][f"{market_id}|{side}"] = position
    runtime_state["executed_simulated_order_count"] = int(
        runtime_state.get("executed_simulated_order_count") or 0
    ) + 1


def _compact_runtime_state(runtime_state: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(runtime_state, sort_keys=True))


def _fill_price_from_microstructure(micro: dict[str, Any]) -> float:
    entry_ask = _float(micro.get("entry_ask"))
    if entry_ask > 0.0:
        return entry_ask
    exit_bid = _float(micro.get("executable_exit_bid_proxy"))
    return exit_bid if exit_bid > 0.0 else 1.0


def _counter_from_rows(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(field_name)
        if isinstance(value, list):
            for item in value:
                counter[str(item)] += 1
        elif value is not None:
            counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _check(
    *,
    passed: bool,
    reason_code: str,
    observed: Any,
    required: Any,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "reason_code": reason_code,
        "observed": observed,
        "required": required,
    }


def _blocking_reason_codes(checks: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        str(check["reason_code"])
        for check in checks.values()
        if check.get("passed") is not True
    )


def _with_report_id(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    report = dict(payload)
    report[field_name] = canonical_json_sha256(report)
    return report


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fresh_loop_run_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Loop Run",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_loop_enabled: `{str(report['paper_fresh_loop_enabled']).lower()}`",
            f"- paper_fresh_loop_mode: `{report['paper_fresh_loop_mode']}`",
            f"- paper_fresh_loop_cycle_count: `{report['paper_fresh_loop_cycle_count']}`",
            f"- paper_fresh_order_intent_count: `{report['paper_fresh_order_intent_count']}`",
            f"- paper_fresh_fill_count: `{report['paper_fresh_fill_count']}`",
            f"- paper_fresh_ledger_entry_count: `{report['paper_fresh_ledger_entry_count']}`",
            f"- v8_paper_internal_handoff_allowed: `{str(report['v8_paper_internal_handoff_allowed']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["paper_fresh_loop_blocking_reason_codes"]),
            "",
        ]
    )


def _fresh_fill_simulation_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Fill Simulation",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_fill_count: `{report['paper_fresh_fill_count']}`",
            f"- paper_fresh_filled_size_sum: `{report['paper_fresh_filled_size_sum']}`",
            f"- total_synthetic_execution_cost: `{report['paper_fresh_total_synthetic_execution_cost']}`",
            f"- outcome_pnl_used: `{str(report['outcome_pnl_used']).lower()}`",
            f"- realized_pnl_used: `{str(report['realized_pnl_used']).lower()}`",
            "",
        ]
    )


def _fresh_runtime_safety_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Runtime Safety",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_runtime_safety_passed: `{str(report['paper_fresh_runtime_safety_passed']).lower()}`",
            f"- v8_paper_internal_handoff_allowed: `{str(report['v8_paper_internal_handoff_allowed']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- paper_only: `{str(report['paper_only']).lower()}`",
            f"- capital_at_risk: `{str(report['capital_at_risk']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["paper_fresh_runtime_safety_blocking_reason_codes"]),
            "",
        ]
    )


def _fresh_monitoring_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Monitoring",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_monitoring_passed: `{str(report['paper_fresh_monitoring_passed']).lower()}`",
            f"- cycle_count: `{report['cycle_count']}`",
            f"- cycle_failure_count: `{report['cycle_failure_count']}`",
            f"- candidate_decision_count: `{report['candidate_decision_count']}`",
            f"- guard_allowed_decision_count: `{report['guard_allowed_decision_count']}`",
            f"- guard_blocked_decision_count: `{report['guard_blocked_decision_count']}`",
            "",
        ]
    )


def _fresh_cumulative_monitoring_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Cumulative Monitoring",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- total_cycles: `{report['total_cycles']}`",
            f"- total_paper_intents: `{report['total_paper_intents']}`",
            f"- total_paper_fills: `{report['total_paper_fills']}`",
            f"- total_blocked_decisions: `{report['total_blocked_decisions']}`",
            f"- cycle_failure_count: `{report['cycle_failure_count']}`",
            f"- safety_violation_count: `{report['safety_violation_count']}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
        ]
    )


def _markdown_list(rows: list[str]) -> list[str]:
    return ["- none"] if not rows else [f"- `{row}`" for row in rows]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return _sha256_file_existing(path) if path.exists() else ""
