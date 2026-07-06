"""One-hour paper-only goal diagnostics for Execution Layer v2 remaps."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
    O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE,
    PINNED_ISSUE_160_MANIFEST_SHA256,
    PolymarketOV8PaperFreshLoopConfig,
    run_polymarket_o_v8_paper_fresh_loop,
)

ONE_HOUR_REMAP_PAPER_GOAL_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-one-hour-remap-paper-goal-v1"
)
ONE_HOUR_REMAP_PAPER_GOAL_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-one-hour-remap-paper-goal-manifest-v1"
)
ONE_HOUR_REMAP_ROUND_COVERAGE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-one-hour-round-coverage-v1"
)
ONE_HOUR_REMAP_EXECUTION_REPORT_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-one-hour-remap-execution-v1"
)

DEFAULT_ONE_HOUR_UNLOCK_DIR = Path(
    "examples/v8/polymarket_runs/o-v8-paper-candidate-unlock-20260703T073000Z"
)
DEFAULT_FROZEN_EV_CALIBRATION_ARTIFACT = Path(
    "examples/v8/polymarket_configs/execution_layer_v2_frozen_ev_calibration_v1.json"
)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2OneHourRemapPaperGoalConfig:
    """Configuration for the one-hour paper-only remap goal runner."""

    run_id: str
    output_dir: Path | str
    duration_seconds: int = 3600
    poll_interval_seconds: float = 60.0
    paper_candidate_unlock_dir: Path | str = DEFAULT_ONE_HOUR_UNLOCK_DIR
    expected_paper_candidate_unlock_manifest_sha256: str | None = (
        PINNED_ISSUE_160_MANIFEST_SHA256
    )
    frozen_ev_calibration_artifact_path: Path | str = (
        DEFAULT_FROZEN_EV_CALIBRATION_ARTIFACT
    )
    canonical_o_source_manifest_path: Path | str | None = None
    public_data_cycles: tuple[tuple[dict[str, Any], ...], ...] | None = None
    public_provider: Any | None = None
    settlement_evaluation_rows: tuple[dict[str, Any], ...] = ()
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.duration_seconds < 3600:
            raise ValueError("duration_seconds must be at least 3600")
        if self.poll_interval_seconds < 0.0:
            raise ValueError("poll_interval_seconds must be non-negative")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self, "paper_candidate_unlock_dir", Path(self.paper_candidate_unlock_dir)
        )
        object.__setattr__(
            self,
            "frozen_ev_calibration_artifact_path",
            Path(self.frozen_ev_calibration_artifact_path),
        )
        if self.canonical_o_source_manifest_path is not None and not isinstance(
            self.canonical_o_source_manifest_path,
            Path,
        ):
            object.__setattr__(
                self,
                "canonical_o_source_manifest_path",
                Path(self.canonical_o_source_manifest_path),
            )
        if not isinstance(self.settlement_evaluation_rows, tuple):
            object.__setattr__(
                self,
                "settlement_evaluation_rows",
                tuple(dict(row) for row in self.settlement_evaluation_rows),
            )


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2OneHourRemapPaperGoalResult:
    """Generated one-hour remap goal bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    goal_report: dict[str, Any]
    round_coverage_report: dict[str, Any]
    remap_execution_report: dict[str, Any]
    manifest: dict[str, Any]


def run_execution_layer_v2_one_hour_remap_paper_goal(
    config: ExecutionLayerV2OneHourRemapPaperGoalConfig,
) -> ExecutionLayerV2OneHourRemapPaperGoalResult:
    """Run the paper-only remap path and write one-hour goal diagnostics."""

    goal_dir = Path(config.output_dir) / config.run_id
    if goal_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"one-hour remap goal output already exists: {goal_dir}")
        shutil.rmtree(goal_dir)
    goal_dir.mkdir(parents=True)

    public_data_source = (
        O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE
        if config.public_data_cycles is not None
        else O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
    )
    max_cycles = (
        len(config.public_data_cycles)
        if config.public_data_cycles is not None
        else max(1, int(config.duration_seconds / max(config.poll_interval_seconds, 1.0)))
    )
    fresh_result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id=f"{config.run_id}-fresh-loop",
            output_dir=goal_dir,
            paper_candidate_unlock_dir=config.paper_candidate_unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=(
                config.expected_paper_candidate_unlock_manifest_sha256
            ),
            loop_mode="bounded_recurring" if max_cycles > 1 else "single_cycle",
            max_cycles=max_cycles,
            sleep_seconds=0.0
            if config.public_data_cycles is not None
            else config.poll_interval_seconds,
            public_data_cycles=config.public_data_cycles,
            public_data_source=public_data_source,
            public_provider=config.public_provider,
            canonical_o_source_manifest_path=config.canonical_o_source_manifest_path,
            frozen_ev_calibration_artifact_path=(
                config.frozen_ev_calibration_artifact_path
            ),
            overwrite_existing=config.overwrite_existing,
        )
    )

    intents = _read_jsonl(fresh_result.artifact_paths["fresh_order_intent_log"])
    fills = _read_jsonl(fresh_result.artifact_paths["fresh_fill_log"])
    trace_rows = list(fresh_result.signal_trace_report.get("trace_rows") or [])
    settlement_rows = _settlement_pnl_rows(
        fills=fills,
        settlement_evaluation_rows=list(config.settlement_evaluation_rows),
    )
    round_coverage = _round_coverage_report(
        run_id=config.run_id,
        trace_rows=trace_rows,
        intents=intents,
    )
    remap_execution = _remap_execution_report(
        run_id=config.run_id,
        fresh_remap_report=fresh_result.execution_layer_v2_paper_remap_report,
        intents=intents,
        fills=fills,
    )
    goal_report = _one_hour_goal_report(
        config=config,
        fresh_result=fresh_result,
        round_coverage=round_coverage,
        remap_execution=remap_execution,
        intents=intents,
        fills=fills,
        settlement_rows=settlement_rows,
    )

    artifact_paths = {
        "one_hour_remap_paper_goal_report": goal_dir
        / "one_hour_remap_paper_goal_report.json",
        "one_hour_remap_paper_goal_summary": goal_dir
        / "one_hour_remap_paper_goal_report.md",
        "one_hour_remap_paper_goal_manifest": goal_dir
        / "one_hour_remap_paper_goal_manifest.json",
        "round_coverage_report": goal_dir / "round_coverage_report.json",
        "round_coverage_summary": goal_dir / "round_coverage_report.md",
        "remap_execution_report": goal_dir / "remap_execution_report.json",
        "remap_execution_summary": goal_dir / "remap_execution_report.md",
        "settlement_pnl_rows": goal_dir / "settlement_pnl_rows.jsonl",
        "paper_intent_log": fresh_result.artifact_paths["fresh_order_intent_log"],
        "paper_fill_log": fresh_result.artifact_paths["fresh_fill_log"],
        "paper_ledger_log": fresh_result.artifact_paths["fresh_ledger_log"],
        "paper_fresh_loop_manifest": fresh_result.artifact_paths["manifest"],
        "paper_remap_report": fresh_result.artifact_paths[
            "execution_layer_v2_paper_remap_report"
        ],
    }
    _write_json(artifact_paths["one_hour_remap_paper_goal_report"], goal_report)
    _write_text(
        artifact_paths["one_hour_remap_paper_goal_summary"],
        _one_hour_goal_md(goal_report),
    )
    _write_json(artifact_paths["round_coverage_report"], round_coverage)
    _write_text(
        artifact_paths["round_coverage_summary"],
        _round_coverage_md(round_coverage),
    )
    _write_json(artifact_paths["remap_execution_report"], remap_execution)
    _write_text(
        artifact_paths["remap_execution_summary"],
        _remap_execution_md(remap_execution),
    )
    _write_jsonl(artifact_paths["settlement_pnl_rows"], settlement_rows)
    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
        if name != "one_hour_remap_paper_goal_manifest"
    }
    manifest = _one_hour_goal_manifest(
        config=config,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        goal_report=goal_report,
        round_coverage=round_coverage,
        remap_execution=remap_execution,
        fresh_manifest=fresh_result.manifest,
    )
    _write_json(artifact_paths["one_hour_remap_paper_goal_manifest"], manifest)
    artifact_hashes["one_hour_remap_paper_goal_manifest"] = _sha256_file(
        artifact_paths["one_hour_remap_paper_goal_manifest"]
    )
    manifest["artifact_hashes"] = dict(artifact_hashes)
    _write_json(artifact_paths["one_hour_remap_paper_goal_manifest"], manifest)
    artifact_hashes["one_hour_remap_paper_goal_manifest"] = _sha256_file(
        artifact_paths["one_hour_remap_paper_goal_manifest"]
    )

    return ExecutionLayerV2OneHourRemapPaperGoalResult(
        output_dir=goal_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        goal_report=goal_report,
        round_coverage_report=round_coverage,
        remap_execution_report=remap_execution,
        manifest=manifest,
    )


def _round_coverage_report(
    *,
    run_id: str,
    trace_rows: list[dict[str, Any]],
    intents: list[dict[str, Any]],
) -> dict[str, Any]:
    complete_rounds = sorted({str(row.get("market_id")) for row in trace_rows if row.get("market_id")})
    bet_rounds = sorted({str(row.get("market_id")) for row in intents if row.get("market_id")})
    missing = sorted(set(complete_rounds) - set(bet_rounds))
    rows = [
        {
            "market_id": market_id,
            "complete_round": True,
            "paper_bet_created": market_id in set(bet_rounds),
            "paper_bet_count": sum(
                1 for intent in intents if str(intent.get("market_id")) == market_id
            ),
        }
        for market_id in complete_rounds
    ]
    report = {
        "schema_version": ONE_HOUR_REMAP_ROUND_COVERAGE_SCHEMA_VERSION,
        "report_type": "one_hour_remap_round_coverage",
        "run_id": run_id,
        "complete_round_count": len(complete_rounds),
        "complete_rounds_with_bet_count": len(bet_rounds),
        "missing_bet_round_count": len(missing),
        "missing_bet_round_ids": missing,
        "round_coverage_rows": rows,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    return _with_report_id(report, "round_coverage_report_id")


def _remap_execution_report(
    *,
    run_id: str,
    fresh_remap_report: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
) -> dict[str, Any]:
    remap_intents = [
        intent for intent in intents if intent.get("hts_time_window_remap_applied") is True
    ]
    report = {
        "schema_version": ONE_HOUR_REMAP_EXECUTION_REPORT_SCHEMA_VERSION,
        "report_type": "one_hour_remap_execution",
        "run_id": run_id,
        "hts_time_window_blocked_count": fresh_remap_report[
            "hts_time_window_blocked_count"
        ],
        "same_side_sbc_remap_guard_passed_count": fresh_remap_report[
            "remap_guard_passed_count"
        ],
        "remap_paper_bet_count": len(remap_intents),
        "normal_policy_bet_count": len(intents) - len(remap_intents),
        "paper_fill_count": len(fills),
        "forced_coverage_bet_count": 0,
        "remap_reason_distribution": fresh_remap_report[
            "remap_reason_distribution"
        ],
        "remap_execution_rows": fresh_remap_report.get("remap_rows", []),
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
    }
    return _with_report_id(report, "remap_execution_report_id")


def _settlement_pnl_rows(
    *,
    fills: list[dict[str, Any]],
    settlement_evaluation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_market = {
        str(row.get("market_id")): dict(row) for row in settlement_evaluation_rows
    }
    rows = []
    for fill in fills:
        market_id = str(fill.get("market_id"))
        evaluation = by_market.get(market_id, {})
        settled = "settlement_pnl" in evaluation
        pnl = _float(evaluation.get("settlement_pnl")) if settled else 0.0
        row = {
            "market_id": market_id,
            "decision_ts": fill.get("decision_ts"),
            "paper_fresh_fill_id": fill.get("paper_fresh_fill_id"),
            "paper_fresh_order_intent_id": fill.get("paper_fresh_order_intent_id"),
            "execution_guarded_action": fill.get("execution_guarded_action"),
            "execution_guarded_side": fill.get("execution_guarded_side"),
            "settlement_status": "settled" if settled else "unresolved",
            "settlement_pnl": pnl,
            "unresolved_pnl": 0.0 if settled else 0.0,
            "paper_only": True,
            "capital_at_risk": False,
            "uses_settlement_pnl_for_decision_time_logic": False,
        }
        row["settlement_pnl_row_hash"] = canonical_json_sha256(row)
        rows.append(row)
    return rows


def _one_hour_goal_report(
    *,
    config: ExecutionLayerV2OneHourRemapPaperGoalConfig,
    fresh_result: Any,
    round_coverage: dict[str, Any],
    remap_execution: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    settlement_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    settled_pnl = sum(_float(row.get("settlement_pnl")) for row in settlement_rows)
    unresolved_pnl = sum(_float(row.get("unresolved_pnl")) for row in settlement_rows)
    unresolved_count = sum(
        1 for row in settlement_rows if row.get("settlement_status") != "settled"
    )
    blockers = []
    if round_coverage["complete_round_count"] <= 0:
        blockers.append("no_complete_rounds_observed")
    if round_coverage["missing_bet_round_count"] > 0:
        blockers.append("complete_rounds_missing_paper_bets")
    if settled_pnl <= 0.0:
        blockers.append("settled_pnl_not_positive")
    if unresolved_count:
        blockers.append("unresolved_round_pnl_remaining")
    if fresh_result.runtime_safety_report["paper_fresh_runtime_safety_passed"] is not True:
        blockers.append("paper_fresh_runtime_safety_failed")
    final_success = blockers == []
    report = {
        "schema_version": ONE_HOUR_REMAP_PAPER_GOAL_SCHEMA_VERSION,
        "report_type": "one_hour_remap_paper_goal",
        "run_id": config.run_id,
        "duration_seconds": config.duration_seconds,
        "duration_requirement_passed": config.duration_seconds >= 3600,
        "public_data_source": fresh_result.manifest["paper_fresh_loop_public_data_source"],
        "read_only_public_provider_required_for_real_run": True,
        "uses_read_only_public_provider_only": (
            fresh_result.manifest["paper_fresh_loop_public_data_source"]
            == O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
        ),
        "frozen_ev_calibration_artifact_path": str(
            config.frozen_ev_calibration_artifact_path
        ),
        "complete_round_count": round_coverage["complete_round_count"],
        "complete_rounds_with_bet_count": round_coverage[
            "complete_rounds_with_bet_count"
        ],
        "missing_bet_round_count": round_coverage["missing_bet_round_count"],
        "normal_policy_bet_count": remap_execution["normal_policy_bet_count"],
        "remap_paper_bet_count": remap_execution["remap_paper_bet_count"],
        "forced_coverage_bet_count": 0,
        "paper_intent_count": len(intents),
        "paper_fill_count": len(fills),
        "settled_pnl": settled_pnl,
        "unresolved_pnl": unresolved_pnl,
        "unresolved_settlement_count": unresolved_count,
        "final_goal_success": final_success,
        "goal_failure_reason_codes": blockers,
        "losing_action_distribution": _action_distribution_by_pnl(
            settlement_rows, positive=False
        ),
        "winning_action_distribution": _action_distribution_by_pnl(
            settlement_rows, positive=True
        ),
        "guard_blocking_reason_distribution": fresh_result.fresh_loop_run_report[
            "block_reason_distribution"
        ],
        "uses_settlement_pnl_or_outcome_labels_in_decision_logic": False,
        "uses_oracle_actions_or_future_returns": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "one_hour_remap_paper_goal_report_id")


def _one_hour_goal_manifest(
    *,
    config: ExecutionLayerV2OneHourRemapPaperGoalConfig,
    artifact_paths: dict[str, Path],
    artifact_hashes: dict[str, str],
    goal_report: dict[str, Any],
    round_coverage: dict[str, Any],
    remap_execution: dict[str, Any],
    fresh_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": ONE_HOUR_REMAP_PAPER_GOAL_MANIFEST_SCHEMA_VERSION,
        "report_type": "one_hour_remap_paper_goal_manifest",
        "run_id": config.run_id,
        "artifact_paths": {
            name: str(path) for name, path in sorted(artifact_paths.items())
        },
        "artifact_hashes": dict(artifact_hashes),
        "one_hour_remap_paper_goal_report_id": goal_report[
            "one_hour_remap_paper_goal_report_id"
        ],
        "round_coverage_report_id": round_coverage["round_coverage_report_id"],
        "remap_execution_report_id": remap_execution["remap_execution_report_id"],
        "paper_fresh_loop_manifest_id": fresh_manifest[
            "o_v8_paper_fresh_loop_manifest_id"
        ],
        "duration_seconds": goal_report["duration_seconds"],
        "complete_round_count": goal_report["complete_round_count"],
        "complete_rounds_with_bet_count": goal_report[
            "complete_rounds_with_bet_count"
        ],
        "missing_bet_round_count": goal_report["missing_bet_round_count"],
        "normal_policy_bet_count": goal_report["normal_policy_bet_count"],
        "remap_paper_bet_count": goal_report["remap_paper_bet_count"],
        "forced_coverage_bet_count": goal_report["forced_coverage_bet_count"],
        "settled_pnl": goal_report["settled_pnl"],
        "unresolved_pnl": goal_report["unresolved_pnl"],
        "final_goal_success": goal_report["final_goal_success"],
        "goal_failure_reason_codes": goal_report["goal_failure_reason_codes"],
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    return _with_report_id(manifest, "one_hour_remap_paper_goal_manifest_id")


def _one_hour_goal_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# One-Hour Remap Paper Goal",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- duration_seconds: `{report['duration_seconds']}`",
            f"- complete_round_count: `{report['complete_round_count']}`",
            f"- complete_rounds_with_bet_count: `{report['complete_rounds_with_bet_count']}`",
            f"- missing_bet_round_count: `{report['missing_bet_round_count']}`",
            f"- normal_policy_bet_count: `{report['normal_policy_bet_count']}`",
            f"- remap_paper_bet_count: `{report['remap_paper_bet_count']}`",
            f"- forced_coverage_bet_count: `{report['forced_coverage_bet_count']}`",
            f"- settled_pnl: `{report['settled_pnl']}`",
            f"- unresolved_pnl: `{report['unresolved_pnl']}`",
            f"- final_goal_success: `{report['final_goal_success']}`",
            "",
            "## Failure Reasons",
            "",
            *[f"- `{reason}`" for reason in report["goal_failure_reason_codes"]],
            "",
        ]
    )


def _round_coverage_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Round Coverage",
            "",
            f"- complete_round_count: `{report['complete_round_count']}`",
            f"- complete_rounds_with_bet_count: `{report['complete_rounds_with_bet_count']}`",
            f"- missing_bet_round_count: `{report['missing_bet_round_count']}`",
            "",
        ]
    )


def _remap_execution_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remap Execution",
            "",
            f"- hts_time_window_blocked_count: `{report['hts_time_window_blocked_count']}`",
            f"- same_side_sbc_remap_guard_passed_count: `{report['same_side_sbc_remap_guard_passed_count']}`",
            f"- remap_paper_bet_count: `{report['remap_paper_bet_count']}`",
            f"- normal_policy_bet_count: `{report['normal_policy_bet_count']}`",
            "",
        ]
    )


def _action_distribution_by_pnl(
    rows: list[dict[str, Any]],
    *,
    positive: bool,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        pnl = _float(row.get("settlement_pnl"))
        if positive and pnl <= 0.0:
            continue
        if not positive and pnl >= 0.0:
            continue
        counter[str(row.get("execution_guarded_action"))] += 1
    return dict(sorted(counter.items()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_report_id(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    copied = dict(payload)
    copied[field_name] = canonical_json_sha256(copied)
    return copied


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
