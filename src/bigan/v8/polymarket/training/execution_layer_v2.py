"""Diagnostic-only v8 execution layer v2 for signal-to-position control.

This module intentionally stops at deterministic paper diagnostics.  It does
not place orders, mutate O source scores, or unlock paper/live execution.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields

EXECUTION_LAYER_V2_SCHEMA_VERSION = "bigan-v8-polymarket-execution-layer-v2-v1"
EXECUTION_LAYER_V2_REPORT_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-report-v1"
)
EXECUTION_LAYER_V2_BACKTEST_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-backtest-manifest-v1"
)
EXECUTION_LAYER_V2_POLICY_NAME = (
    "v8_execution_layer_v2_signal_position_dynamic_exit"
)
EXECUTION_LAYER_V2_BASELINE_NAME = "hold_to_settlement_v1"

ExecutionLayerV2Side = Literal["UP", "DOWN", "NONE"]
ExecutionLayerV2Action = Literal[
    "NO_ACTION",
    "ENTER_POSITION",
    "HOLD_POSITION",
    "EXIT_POSITION",
    "ROTATE_POSITION",
]
ExecutionLayerV2State = Literal["NO_POSITION", "ACTIVE", "DECAYING", "EXIT"]

EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS: tuple[str, ...] = (
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
    "winning_outcome",
    "resolved_outcome",
    "action_return_target",
    "label_return",
)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2Config:
    """Paper-only configuration for dynamic v8 position management."""

    entry_ev_threshold: float = 0.02
    min_confidence: float = 0.55
    hold_ev_floor_ratio: float = 0.60
    opposite_signal_ev_margin: float = 0.02
    time_exit_threshold_seconds: float = 60.0
    execution_cost_bps: float = 10.0
    nav_usdc: float = 10_000.0
    max_nav_fraction_per_position: float = 0.05
    min_nav_fraction_per_position: float = 0.0
    kelly_time_decay_lambda: float = 0.0005
    diagnostic_lambda_grid: tuple[float, ...] = (0.0, 0.0005, 0.001)
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False
    v8_execution_handoff_allowed: bool = False
    source_model_candidate_eligible: bool = False
    freeze_ready: bool = False
    promotion_evidence_eligible: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "entry_ev_threshold",
            "min_confidence",
            "hold_ev_floor_ratio",
            "opposite_signal_ev_margin",
            "time_exit_threshold_seconds",
            "execution_cost_bps",
            "nav_usdc",
            "max_nav_fraction_per_position",
            "min_nav_fraction_per_position",
            "kelly_time_decay_lambda",
        ):
            _require_finite(field_name, float(getattr(self, field_name)))
        if self.entry_ev_threshold < 0.0:
            raise ValueError("entry_ev_threshold must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 <= self.hold_ev_floor_ratio <= 1.0:
            raise ValueError("hold_ev_floor_ratio must be in [0, 1]")
        if self.opposite_signal_ev_margin < 0.0:
            raise ValueError("opposite_signal_ev_margin must be non-negative")
        if self.time_exit_threshold_seconds < 0.0:
            raise ValueError("time_exit_threshold_seconds must be non-negative")
        if self.execution_cost_bps < 0.0:
            raise ValueError("execution_cost_bps must be non-negative")
        if self.nav_usdc <= 0.0:
            raise ValueError("nav_usdc must be positive")
        if not 0.0 <= self.min_nav_fraction_per_position <= self.max_nav_fraction_per_position:
            raise ValueError("nav fractions must satisfy 0 <= min <= max")
        if self.max_nav_fraction_per_position > 1.0:
            raise ValueError("max_nav_fraction_per_position must be <= 1")
        if self.kelly_time_decay_lambda < 0.0:
            raise ValueError("kelly_time_decay_lambda must be non-negative")
        if not self.diagnostic_lambda_grid:
            raise ValueError("diagnostic_lambda_grid is required")
        for value in self.diagnostic_lambda_grid:
            _require_finite("diagnostic_lambda_grid", float(value))
            if value < 0.0:
                raise ValueError("diagnostic_lambda_grid values must be non-negative")
        _validate_safety_flags(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2BacktestConfig:
    """Configuration for writing a deterministic v2 backtest artifact bundle."""

    run_id: str
    output_dir: Path | str
    input_path: Path | str
    max_rows: int | None = None
    execution_config: ExecutionLayerV2Config = field(default_factory=ExecutionLayerV2Config)
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False
    v8_execution_handoff_allowed: bool = False
    source_model_candidate_eligible: bool = False
    freeze_ready: bool = False
    promotion_evidence_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive when provided")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "input_path", Path(self.input_path))
        _validate_safety_flags(self)

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["input_path"] = str(self.input_path)
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2BacktestResult:
    """Written execution layer v2 backtest bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    report: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2Signal:
    """One decision-time signal row available before settlement/outcome."""

    market_id: str
    decision_ts: int
    p_up: float
    ask_up: float
    ask_down: float
    p_down: float | None = None
    bid_up: float | None = None
    bid_down: float | None = None
    time_to_expiry_seconds: float | None = None
    source_signal_id: str | None = None
    model_score: float | None = None
    paper_only: bool = True
    capital_at_risk: bool = False

    def __post_init__(self) -> None:
        if not self.market_id.strip():
            raise ValueError("market_id is required")
        if self.decision_ts < 0:
            raise ValueError("decision_ts must be non-negative")
        _validate_probability("p_up", self.p_up)
        p_down = 1.0 - self.p_up if self.p_down is None else self.p_down
        _validate_probability("p_down", p_down)
        object.__setattr__(self, "p_down", p_down)
        for field_name in ("ask_up", "ask_down"):
            _validate_price(field_name, float(getattr(self, field_name)))
        for field_name in ("bid_up", "bid_down", "time_to_expiry_seconds", "model_score"):
            value = getattr(self, field_name)
            if value is not None:
                _require_finite(field_name, float(value))
        if self.bid_up is not None:
            _validate_price("bid_up", self.bid_up, allow_zero=True)
        if self.bid_down is not None:
            _validate_price("bid_down", self.bid_down, allow_zero=True)
        if self.time_to_expiry_seconds is not None and self.time_to_expiry_seconds < 0.0:
            raise ValueError("time_to_expiry_seconds must be non-negative")
        if self.paper_only is not True:
            raise ValueError("paper_only must be true")
        if self.capital_at_risk is not False:
            raise ValueError("capital_at_risk must be false")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2Position:
    """Paper-only open position state used by the v2 state machine."""

    market_id: str
    side: ExecutionLayerV2Side
    entry_ts: int
    entry_price: float
    entry_probability: float
    entry_ev: float
    size_usdc: float
    shares: float
    state: ExecutionLayerV2State = "ACTIVE"
    paper_only: bool = True
    capital_at_risk: bool = False

    def __post_init__(self) -> None:
        if not self.market_id.strip():
            raise ValueError("market_id is required")
        if self.side not in ("UP", "DOWN"):
            raise ValueError("position side must be UP or DOWN")
        if self.entry_ts < 0:
            raise ValueError("entry_ts must be non-negative")
        _validate_price("entry_price", self.entry_price)
        _validate_probability("entry_probability", self.entry_probability)
        for field_name in ("entry_ev", "size_usdc", "shares"):
            _require_finite(field_name, float(getattr(self, field_name)))
        if self.size_usdc <= 0.0 or self.shares <= 0.0:
            raise ValueError("position size_usdc and shares must be positive")
        if self.state not in ("ACTIVE", "DECAYING"):
            raise ValueError("open position state must be ACTIVE or DECAYING")
        if self.paper_only is not True:
            raise ValueError("paper_only must be true")
        if self.capital_at_risk is not False:
            raise ValueError("capital_at_risk must be false")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2Decision:
    """One paper-only state-machine decision."""

    market_id: str
    decision_ts: int
    action: ExecutionLayerV2Action
    target_side: ExecutionLayerV2Side
    state_before: ExecutionLayerV2State
    state_after: ExecutionLayerV2State
    selected_ev_t: float
    entry_ev_reference: float
    ev_ratio_to_entry: float | None
    confidence: float
    execution_price: float
    paper_notional: float
    shares: float
    kelly_fraction: float
    time_decay_multiplier: float
    reason_codes: tuple[str, ...]
    source_signal_id: str | None = None
    baseline_v1_action: str = EXECUTION_LAYER_V2_BASELINE_NAME
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False
    v8_execution_handoff_allowed: bool = False

    def __post_init__(self) -> None:
        if self.action not in {
            "NO_ACTION",
            "ENTER_POSITION",
            "HOLD_POSITION",
            "EXIT_POSITION",
            "ROTATE_POSITION",
        }:
            raise ValueError("unsupported execution layer v2 action")
        if self.target_side not in ("UP", "DOWN", "NONE"):
            raise ValueError("unsupported target_side")
        if self.state_before not in ("NO_POSITION", "ACTIVE", "DECAYING", "EXIT"):
            raise ValueError("unsupported state_before")
        if self.state_after not in ("NO_POSITION", "ACTIVE", "DECAYING", "EXIT"):
            raise ValueError("unsupported state_after")
        for field_name in (
            "selected_ev_t",
            "entry_ev_reference",
            "confidence",
            "execution_price",
            "paper_notional",
            "shares",
            "kelly_fraction",
            "time_decay_multiplier",
        ):
            _require_finite(field_name, float(getattr(self, field_name)))
        if self.ev_ratio_to_entry is not None:
            _require_finite("ev_ratio_to_entry", self.ev_ratio_to_entry)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.execution_price < 0.0 or self.paper_notional < 0.0 or self.shares < 0.0:
            raise ValueError("execution values must be non-negative")
        if not self.reason_codes:
            raise ValueError("reason_codes are required")
        _validate_safety_flags(self)

    @property
    def state_transition(self) -> str:
        return f"{self.state_before}->{self.state_after}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        payload["state_transition"] = self.state_transition
        payload["execution_layer_v2_decision_hash"] = canonical_json_sha256(payload)
        return payload


class ExecutionLayerV2Engine:
    """Sequential paper-only executor for v2 state-machine decisions."""

    def __init__(self, config: ExecutionLayerV2Config | None = None) -> None:
        self.config = config or ExecutionLayerV2Config()
        self._positions: dict[str, ExecutionLayerV2Position] = {}
        self._seen_keys: set[tuple[int, str]] = set()
        self._last_key: tuple[int, str] | None = None

    @property
    def positions(self) -> dict[str, ExecutionLayerV2Position]:
        return dict(self._positions)

    def decide_many(
        self,
        signals: tuple[ExecutionLayerV2Signal, ...] | list[ExecutionLayerV2Signal],
    ) -> tuple[ExecutionLayerV2Decision, ...]:
        decisions = []
        for signal in sorted(signals, key=lambda row: (row.decision_ts, row.market_id)):
            decisions.append(self.decide(signal))
        return tuple(decisions)

    def decide(self, signal: ExecutionLayerV2Signal) -> ExecutionLayerV2Decision:
        key = (int(signal.decision_ts), str(signal.market_id))
        if key in self._seen_keys:
            raise ValueError("duplicate_execution_layer_v2_decision_key")
        if self._last_key is not None and key < self._last_key:
            raise ValueError("execution_layer_v2_state_out_of_order")
        position = self._positions.get(signal.market_id)
        decision = decide_execution_layer_v2(
            signal=signal,
            position=position,
            config=self.config,
        )
        self._apply_decision(signal=signal, decision=decision)
        self._seen_keys.add(key)
        self._last_key = key
        return decision

    def _apply_decision(
        self,
        *,
        signal: ExecutionLayerV2Signal,
        decision: ExecutionLayerV2Decision,
    ) -> None:
        if decision.action == "ENTER_POSITION":
            self._positions[signal.market_id] = _position_from_decision(signal, decision)
            return
        if decision.action == "ROTATE_POSITION":
            self._positions[signal.market_id] = _position_from_decision(signal, decision)
            return
        if decision.action == "EXIT_POSITION":
            self._positions.pop(signal.market_id, None)


def decide_execution_layer_v2(
    *,
    signal: ExecutionLayerV2Signal,
    position: ExecutionLayerV2Position | None = None,
    config: ExecutionLayerV2Config | None = None,
) -> ExecutionLayerV2Decision:
    """Recalculate EV_t and emit one paper-only entry/hold/exit/rotate decision."""

    config = config or ExecutionLayerV2Config()
    up_entry_ev = _entry_ev(signal, "UP", config)
    down_entry_ev = _entry_ev(signal, "DOWN", config)
    best_side: ExecutionLayerV2Side = "UP" if up_entry_ev >= down_entry_ev else "DOWN"
    best_entry_ev = up_entry_ev if best_side == "UP" else down_entry_ev
    confidence = _probability(signal, best_side)

    if position is None:
        return _entry_or_no_action_decision(
            signal=signal,
            side=best_side,
            entry_ev=best_entry_ev,
            confidence=confidence,
            config=config,
        )
    return _active_position_decision(
        signal=signal,
        position=position,
        up_entry_ev=up_entry_ev,
        down_entry_ev=down_entry_ev,
        config=config,
    )


def build_execution_layer_v2_report(
    signals: tuple[ExecutionLayerV2Signal, ...] | list[ExecutionLayerV2Signal],
    *,
    config: ExecutionLayerV2Config | None = None,
) -> dict[str, Any]:
    """Build a deterministic v2 diagnostic report from decision-time signals."""

    config = config or ExecutionLayerV2Config()
    engine = ExecutionLayerV2Engine(config=config)
    decisions = engine.decide_many(tuple(signals))
    decision_rows = [decision.to_dict() for decision in decisions]
    action_counts = Counter(decision.action for decision in decisions)
    transition_counts = Counter(decision.state_transition for decision in decisions)
    reason_counts: Counter[str] = Counter()
    for decision in decisions:
        reason_counts.update(decision.reason_codes)
    v1_baseline = _v1_baseline_summary(
        tuple(signals),
        decisions=decisions,
        config=config,
    )
    lambda_diagnostics = _lambda_grid_diagnostics(tuple(signals), config=config)
    entry_ev_values = [_best_entry_ev(signal, config) for signal in signals]
    report = {
        "schema_version": EXECUTION_LAYER_V2_REPORT_SCHEMA_VERSION,
        "execution_layer_v2_policy_name": EXECUTION_LAYER_V2_POLICY_NAME,
        "execution_layer_v2_status": "diagnostic_only_fail_closed",
        "decision_count": len(decisions),
        "entry_decision_count": action_counts.get("ENTER_POSITION", 0),
        "hold_decision_count": action_counts.get("HOLD_POSITION", 0),
        "exit_decision_count": action_counts.get("EXIT_POSITION", 0),
        "rotation_decision_count": action_counts.get("ROTATE_POSITION", 0),
        "action_counts": dict(sorted(action_counts.items())),
        "state_transition_counts": dict(sorted(transition_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "entry_ev_threshold": config.entry_ev_threshold,
        "max_candidate_entry_ev": max(entry_ev_values, default=0.0),
        "mean_candidate_entry_ev": (
            sum(entry_ev_values) / len(entry_ev_values) if entry_ev_values else 0.0
        ),
        "positive_entry_ev_count": sum(
            value >= config.entry_ev_threshold for value in entry_ev_values
        ),
        "decision_rows": decision_rows,
        "open_position_count": len(engine.positions),
        "open_positions": [
            position.to_dict()
            for position in sorted(engine.positions.values(), key=lambda row: row.market_id)
        ],
        "ev_recalculation_loop_enabled": True,
        "dynamic_exit_engine_enabled": True,
        "state_machine_executor_enabled": True,
        "kelly_time_decay_sizing_enabled": True,
        "time_decay_function": "kelly_fraction * exp(-lambda * time_to_expiry_seconds)",
        "lambda_threshold_tuning_mode": "diagnostic_only_config_grid_no_outcomes",
        "uses_validation_labels_for_tuning": False,
        "uses_realized_pnl_or_settlement_outcomes": False,
        "forbidden_outcome_fields_used": [],
        "v1_baseline_comparison": v1_baseline,
        "lambda_threshold_diagnostics": lambda_diagnostics,
        "config": config.to_dict(),
        "source_scores_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(config),
    }
    report["execution_layer_v2_report_id"] = canonical_json_sha256(report)
    return report


def build_execution_layer_v2_report_from_rows(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    config: ExecutionLayerV2Config | None = None,
    input_path: str | None = None,
) -> dict[str, Any]:
    """Build a report from raw rows, fail-closed on forbidden outcome fields."""

    config = config or ExecutionLayerV2Config()
    rows = tuple(dict(row) for row in rows)
    forbidden = _forbidden_fields_by_row(rows)
    if forbidden:
        report = {
            "schema_version": EXECUTION_LAYER_V2_REPORT_SCHEMA_VERSION,
            "execution_layer_v2_policy_name": EXECUTION_LAYER_V2_POLICY_NAME,
            "execution_layer_v2_status": "blocked_fail_closed",
            "decision_count": 0,
            "decision_rows": [],
            "forbidden_outcome_fields_present": True,
            "forbidden_outcome_fields_by_row": forbidden,
            "forbidden_outcome_fields_used": sorted(
                {field for item in forbidden for field in item["forbidden_fields"]}
            ),
            "input_path": input_path,
            "source_scores_mutated": False,
            "paper_live_unlock_changed": False,
            **_safety_report_fields(config),
        }
        report["execution_layer_v2_report_id"] = canonical_json_sha256(report)
        return report
    loaded = _signals_from_rows(rows)
    if loaded["rejected_rows"]:
        report = {
            "schema_version": EXECUTION_LAYER_V2_REPORT_SCHEMA_VERSION,
            "execution_layer_v2_policy_name": EXECUTION_LAYER_V2_POLICY_NAME,
            "execution_layer_v2_status": "blocked_fail_closed",
            "decision_count": 0,
            "accepted_signal_row_count": loaded["accepted_signal_row_count"],
            "rejected_signal_row_count": len(loaded["rejected_rows"]),
            "rejected_signal_rows": loaded["rejected_rows"],
            "decision_rows": [],
            "forbidden_outcome_fields_present": False,
            "forbidden_outcome_fields_used": [],
            "input_path": input_path,
            "source_scores_mutated": False,
            "paper_live_unlock_changed": False,
            **_safety_report_fields(config),
        }
        report["execution_layer_v2_report_id"] = canonical_json_sha256(report)
        return report
    report = build_execution_layer_v2_report(tuple(loaded["signals"]), config=config)
    report["input_path"] = input_path
    report["accepted_signal_row_count"] = loaded["accepted_signal_row_count"]
    report["rejected_signal_row_count"] = 0
    report["rejected_signal_rows"] = []
    report["execution_layer_v2_report_id"] = canonical_json_sha256(report)
    return report


def run_execution_layer_v2_backtest(
    config: ExecutionLayerV2BacktestConfig,
) -> ExecutionLayerV2BacktestResult:
    """Write a deterministic paper-only v2 backtest artifact bundle."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"execution layer v2 backtest exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    rows = load_execution_layer_v2_input_rows(config.input_path, max_rows=config.max_rows)
    report = build_execution_layer_v2_report_from_rows(
        rows,
        config=config.execution_config,
        input_path=str(config.input_path),
    )
    report["run_id"] = config.run_id
    report["input_row_count"] = len(rows)
    report["backtest_artifact_mode"] = "deterministic_offline_diagnostic"
    report["outcome_evaluation_generated"] = False
    report["pnl_claim_generated"] = False
    report["execution_layer_v2_report_id"] = canonical_json_sha256(report)

    artifact_paths = {
        "execution_layer_v2_backtest_report": run_dir
        / "execution_layer_v2_backtest_report.json",
        "execution_layer_v2_backtest_summary": run_dir
        / "execution_layer_v2_backtest_report.md",
        "execution_layer_v2_backtest_manifest": run_dir
        / "execution_layer_v2_backtest_manifest.json",
    }
    _write_json(artifact_paths["execution_layer_v2_backtest_report"], report)
    _write_text(
        artifact_paths["execution_layer_v2_backtest_summary"],
        execution_layer_v2_report_to_markdown(report),
    )
    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in artifact_paths.items()
        if name != "execution_layer_v2_backtest_manifest"
    }
    manifest = {
        "schema_version": EXECUTION_LAYER_V2_BACKTEST_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "input_path": str(config.input_path),
        "input_row_count": len(rows),
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_hashes": dict(artifact_hashes),
        "execution_layer_v2_report_id": report["execution_layer_v2_report_id"],
        "execution_layer_v2_status": report["execution_layer_v2_status"],
        "entry_decision_count": report.get("entry_decision_count", 0),
        "hold_decision_count": report.get("hold_decision_count", 0),
        "exit_decision_count": report.get("exit_decision_count", 0),
        "rotation_decision_count": report.get("rotation_decision_count", 0),
        "outcome_evaluation_generated": False,
        "pnl_claim_generated": False,
        "source_scores_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(config.execution_config),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    _write_json(artifact_paths["execution_layer_v2_backtest_manifest"], manifest)
    artifact_hashes["execution_layer_v2_backtest_manifest"] = _sha256_file(
        artifact_paths["execution_layer_v2_backtest_manifest"]
    )
    return ExecutionLayerV2BacktestResult(
        output_dir=run_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        report=report,
        manifest=manifest,
    )


def load_execution_layer_v2_input_rows(
    input_path: Path | str,
    *,
    max_rows: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Load JSON/JSONL rows from a signal trace or future holdout raw manifest."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"execution layer v2 input not found: {path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = _rows_from_json_payload(payload)
    if max_rows is not None:
        rows = rows[:max_rows]
    return tuple(dict(row) for row in rows)


def execution_layer_v2_report_to_markdown(report: dict[str, Any]) -> str:
    """Render the v2 report as a compact Markdown evidence summary."""

    baseline = report.get("v1_baseline_comparison", {})
    lines = [
        "# v8 Execution Layer v2 Backtest",
        "",
        f"- status: `{report.get('execution_layer_v2_status')}`",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- input_path: `{report.get('input_path', '')}`",
        f"- input_row_count: `{report.get('input_row_count', report.get('decision_count', 0))}`",
        f"- decision_count: `{report.get('decision_count', 0)}`",
        f"- entry_decision_count: `{report.get('entry_decision_count', 0)}`",
        f"- hold_decision_count: `{report.get('hold_decision_count', 0)}`",
        f"- exit_decision_count: `{report.get('exit_decision_count', 0)}`",
        f"- rotation_decision_count: `{report.get('rotation_decision_count', 0)}`",
        f"- entry_ev_threshold: `{report.get('entry_ev_threshold', '')}`",
        f"- max_candidate_entry_ev: `{report.get('max_candidate_entry_ev', '')}`",
        f"- positive_entry_ev_count: `{report.get('positive_entry_ev_count', '')}`",
        f"- v1_baseline: `{baseline.get('baseline_name', EXECUTION_LAYER_V2_BASELINE_NAME)}`",
        f"- v2_differs_from_v1: `{baseline.get('v2_differs_from_v1', False)}`",
        f"- outcome_evaluation_generated: `{report.get('outcome_evaluation_generated', False)}`",
        f"- pnl_claim_generated: `{report.get('pnl_claim_generated', False)}`",
        f"- paper_only: `{report.get('paper_only')}`",
        f"- capital_at_risk: `{report.get('capital_at_risk')}`",
        f"- v8_execution_handoff_allowed: `{report.get('v8_execution_handoff_allowed')}`",
        f"- #134_resume_allowed: `{report.get('#134_resume_allowed')}`",
        f"- #146_start_allowed: `{report.get('#146_start_allowed')}`",
        "",
        "## Action Counts",
        "",
    ]
    for action, count in sorted(report.get("action_counts", {}).items()):
        lines.append(f"- {action}: `{count}`")
    lines.extend(["", "## State Transitions", ""])
    for transition, count in sorted(report.get("state_transition_counts", {}).items()):
        lines.append(f"- {transition}: `{count}`")
    lines.extend(["", "## Reason Counts", ""])
    for reason, count in sorted(report.get("reason_counts", {}).items()):
        lines.append(f"- {reason}: `{count}`")
    if report.get("rejected_signal_rows"):
        lines.extend(["", "## Rejected Signal Rows", ""])
        for row in report["rejected_signal_rows"][:10]:
            lines.append(
                "- "
                f"row_index=`{row.get('row_index')}` "
                f"market_id=`{row.get('market_id')}` "
                f"reason_codes=`{row.get('reason_codes')}`"
            )
    lines.append("")
    return "\n".join(lines)


def time_decay_multiplier(time_to_expiry_seconds: float | None, decay_lambda: float) -> float:
    """Issue #166 time-decay term: exp(-lambda * time_to_expiry)."""

    if time_to_expiry_seconds is None:
        return 1.0
    _require_finite("time_to_expiry_seconds", float(time_to_expiry_seconds))
    _require_finite("decay_lambda", float(decay_lambda))
    if time_to_expiry_seconds < 0.0 or decay_lambda < 0.0:
        raise ValueError("time_to_expiry_seconds and decay_lambda must be non-negative")
    return math.exp(-decay_lambda * time_to_expiry_seconds)


def binary_kelly_fraction(probability: float, price: float) -> float:
    """Kelly fraction for a binary $1 payout contract, clamped to [0, 1]."""

    _validate_probability("probability", probability)
    _validate_price("price", price)
    odds = (1.0 - price) / price
    if odds <= 0.0:
        return 0.0
    fraction = (odds * probability - (1.0 - probability)) / odds
    if not math.isfinite(fraction):
        return 0.0
    return min(1.0, max(0.0, fraction))


def time_decayed_kelly_notional(
    *,
    probability: float,
    price: float,
    time_to_expiry_seconds: float | None,
    config: ExecutionLayerV2Config | None = None,
) -> dict[str, float]:
    """Return deterministic paper notional from Kelly fraction and time decay."""

    config = config or ExecutionLayerV2Config()
    raw_kelly = binary_kelly_fraction(probability, price)
    decay = time_decay_multiplier(
        time_to_expiry_seconds,
        config.kelly_time_decay_lambda,
    )
    decayed_fraction = raw_kelly * decay
    clamped_fraction = min(
        config.max_nav_fraction_per_position,
        max(config.min_nav_fraction_per_position, decayed_fraction),
    )
    notional = config.nav_usdc * clamped_fraction
    return {
        "kelly_fraction": raw_kelly,
        "time_decay_multiplier": decay,
        "decayed_kelly_fraction": decayed_fraction,
        "clamped_nav_fraction": clamped_fraction,
        "paper_notional": notional,
    }


def _entry_or_no_action_decision(
    *,
    signal: ExecutionLayerV2Signal,
    side: ExecutionLayerV2Side,
    entry_ev: float,
    confidence: float,
    config: ExecutionLayerV2Config,
) -> ExecutionLayerV2Decision:
    price = _ask(signal, side)
    sizing = time_decayed_kelly_notional(
        probability=_probability(signal, side),
        price=price,
        time_to_expiry_seconds=signal.time_to_expiry_seconds,
        config=config,
    )
    if entry_ev < config.entry_ev_threshold:
        return _decision(
            signal=signal,
            action="NO_ACTION",
            target_side="NONE",
            state_before="NO_POSITION",
            state_after="NO_POSITION",
            selected_ev_t=entry_ev,
            entry_ev_reference=entry_ev,
            ev_ratio_to_entry=None,
            confidence=confidence,
            execution_price=0.0,
            paper_notional=0.0,
            shares=0.0,
            kelly_fraction=sizing["kelly_fraction"],
            time_decay_multiplier=sizing["time_decay_multiplier"],
            reason_codes=("entry_ev_threshold_not_met", "paper_only_guard"),
        )
    if confidence < config.min_confidence:
        return _decision(
            signal=signal,
            action="NO_ACTION",
            target_side="NONE",
            state_before="NO_POSITION",
            state_after="NO_POSITION",
            selected_ev_t=entry_ev,
            entry_ev_reference=entry_ev,
            ev_ratio_to_entry=None,
            confidence=confidence,
            execution_price=0.0,
            paper_notional=0.0,
            shares=0.0,
            kelly_fraction=sizing["kelly_fraction"],
            time_decay_multiplier=sizing["time_decay_multiplier"],
            reason_codes=("entry_confidence_threshold_not_met", "paper_only_guard"),
        )
    if sizing["paper_notional"] <= 0.0:
        return _decision(
            signal=signal,
            action="NO_ACTION",
            target_side="NONE",
            state_before="NO_POSITION",
            state_after="NO_POSITION",
            selected_ev_t=entry_ev,
            entry_ev_reference=entry_ev,
            ev_ratio_to_entry=None,
            confidence=confidence,
            execution_price=0.0,
            paper_notional=0.0,
            shares=0.0,
            kelly_fraction=sizing["kelly_fraction"],
            time_decay_multiplier=sizing["time_decay_multiplier"],
            reason_codes=("time_decayed_kelly_size_zero", "paper_only_guard"),
        )
    return _decision(
        signal=signal,
        action="ENTER_POSITION",
        target_side=side,
        state_before="NO_POSITION",
        state_after="ACTIVE",
        selected_ev_t=entry_ev,
        entry_ev_reference=entry_ev,
        ev_ratio_to_entry=1.0,
        confidence=confidence,
        execution_price=price,
        paper_notional=sizing["paper_notional"],
        shares=sizing["paper_notional"] / price,
        kelly_fraction=sizing["kelly_fraction"],
        time_decay_multiplier=sizing["time_decay_multiplier"],
        reason_codes=("positive_ev_entry", "time_decayed_kelly_sizing", "paper_only_guard"),
    )


def _active_position_decision(
    *,
    signal: ExecutionLayerV2Signal,
    position: ExecutionLayerV2Position,
    up_entry_ev: float,
    down_entry_ev: float,
    config: ExecutionLayerV2Config,
) -> ExecutionLayerV2Decision:
    held_side = position.side
    opposite_side: ExecutionLayerV2Side = "DOWN" if held_side == "UP" else "UP"
    held_ev_t = _probability(signal, held_side) - position.entry_price - _cost(config)
    ev_ratio = held_ev_t / position.entry_ev if position.entry_ev > 0.0 else 0.0
    opposite_ev = up_entry_ev if opposite_side == "UP" else down_entry_ev
    opposite_confidence = _probability(signal, opposite_side)
    time_to_close = signal.time_to_expiry_seconds
    if time_to_close is not None and time_to_close <= config.time_exit_threshold_seconds:
        return _exit_decision(
            signal=signal,
            position=position,
            held_ev_t=held_ev_t,
            ev_ratio=ev_ratio,
            reason_codes=("time_to_expiry_exit_threshold_crossed", "paper_only_guard"),
        )
    floor_ev = config.hold_ev_floor_ratio * position.entry_ev
    if (
        opposite_ev >= config.entry_ev_threshold
        and opposite_ev >= held_ev_t + config.opposite_signal_ev_margin
        and opposite_confidence >= config.min_confidence
    ):
        return _rotate_decision(
            signal=signal,
            position=position,
            target_side=opposite_side,
            target_entry_ev=opposite_ev,
            held_ev_t=held_ev_t,
            ev_ratio=ev_ratio,
            confidence=opposite_confidence,
            config=config,
        )
    if held_ev_t < floor_ev:
        return _exit_decision(
            signal=signal,
            position=position,
            held_ev_t=held_ev_t,
            ev_ratio=ev_ratio,
            reason_codes=("ev_t_decayed_below_hold_floor", "paper_only_guard"),
        )
    return _decision(
        signal=signal,
        action="HOLD_POSITION",
        target_side=held_side,
        state_before=position.state,
        state_after="ACTIVE",
        selected_ev_t=held_ev_t,
        entry_ev_reference=position.entry_ev,
        ev_ratio_to_entry=ev_ratio,
        confidence=_probability(signal, held_side),
        execution_price=0.0,
        paper_notional=0.0,
        shares=0.0,
        kelly_fraction=0.0,
        time_decay_multiplier=time_decay_multiplier(
            signal.time_to_expiry_seconds,
            config.kelly_time_decay_lambda,
        ),
        reason_codes=("ev_t_above_hold_floor", "hold_position", "paper_only_guard"),
    )


def _exit_decision(
    *,
    signal: ExecutionLayerV2Signal,
    position: ExecutionLayerV2Position,
    held_ev_t: float,
    ev_ratio: float,
    reason_codes: tuple[str, ...],
) -> ExecutionLayerV2Decision:
    price = _bid(signal, position.side)
    return _decision(
        signal=signal,
        action="EXIT_POSITION",
        target_side=position.side,
        state_before=position.state,
        state_after="EXIT",
        selected_ev_t=held_ev_t,
        entry_ev_reference=position.entry_ev,
        ev_ratio_to_entry=ev_ratio,
        confidence=_probability(signal, position.side),
        execution_price=price,
        paper_notional=position.shares * price,
        shares=position.shares,
        kelly_fraction=0.0,
        time_decay_multiplier=1.0,
        reason_codes=reason_codes,
    )


def _rotate_decision(
    *,
    signal: ExecutionLayerV2Signal,
    position: ExecutionLayerV2Position,
    target_side: ExecutionLayerV2Side,
    target_entry_ev: float,
    held_ev_t: float,
    ev_ratio: float,
    confidence: float,
    config: ExecutionLayerV2Config,
) -> ExecutionLayerV2Decision:
    price = _ask(signal, target_side)
    sizing = time_decayed_kelly_notional(
        probability=_probability(signal, target_side),
        price=price,
        time_to_expiry_seconds=signal.time_to_expiry_seconds,
        config=config,
    )
    return _decision(
        signal=signal,
        action="ROTATE_POSITION",
        target_side=target_side,
        state_before=position.state,
        state_after="ACTIVE",
        selected_ev_t=target_entry_ev,
        entry_ev_reference=position.entry_ev,
        ev_ratio_to_entry=ev_ratio,
        confidence=confidence,
        execution_price=price,
        paper_notional=sizing["paper_notional"],
        shares=sizing["paper_notional"] / price if price > 0.0 else 0.0,
        kelly_fraction=sizing["kelly_fraction"],
        time_decay_multiplier=sizing["time_decay_multiplier"],
        reason_codes=(
            "opposite_signal_ev_margin_crossed",
            "rotate_position",
            "paper_only_guard",
        ),
    )


def _decision(
    *,
    signal: ExecutionLayerV2Signal,
    action: ExecutionLayerV2Action,
    target_side: ExecutionLayerV2Side,
    state_before: ExecutionLayerV2State,
    state_after: ExecutionLayerV2State,
    selected_ev_t: float,
    entry_ev_reference: float,
    ev_ratio_to_entry: float | None,
    confidence: float,
    execution_price: float,
    paper_notional: float,
    shares: float,
    kelly_fraction: float,
    time_decay_multiplier: float,
    reason_codes: tuple[str, ...],
) -> ExecutionLayerV2Decision:
    return ExecutionLayerV2Decision(
        market_id=signal.market_id,
        decision_ts=signal.decision_ts,
        action=action,
        target_side=target_side,
        state_before=state_before,
        state_after=state_after,
        selected_ev_t=selected_ev_t,
        entry_ev_reference=entry_ev_reference,
        ev_ratio_to_entry=ev_ratio_to_entry,
        confidence=confidence,
        execution_price=execution_price,
        paper_notional=paper_notional,
        shares=shares,
        kelly_fraction=kelly_fraction,
        time_decay_multiplier=time_decay_multiplier,
        reason_codes=reason_codes,
        source_signal_id=signal.source_signal_id,
    )


def _position_from_decision(
    signal: ExecutionLayerV2Signal,
    decision: ExecutionLayerV2Decision,
) -> ExecutionLayerV2Position:
    return ExecutionLayerV2Position(
        market_id=decision.market_id,
        side=decision.target_side,
        entry_ts=decision.decision_ts,
        entry_price=decision.execution_price,
        entry_probability=_probability(signal, decision.target_side),
        entry_ev=decision.selected_ev_t,
        size_usdc=decision.paper_notional,
        shares=decision.shares,
        state="ACTIVE",
    )


def _signal_from_row(row: dict[str, Any]) -> ExecutionLayerV2Signal:
    normalized = _normalized_signal_fields(row)
    return ExecutionLayerV2Signal(
        market_id=str(normalized["market_id"]),
        decision_ts=int(normalized["decision_ts"]),
        p_up=float(normalized["p_up"]),
        p_down=float(normalized["p_down"]) if normalized.get("p_down") is not None else None,
        ask_up=float(normalized["ask_up"]),
        ask_down=float(normalized["ask_down"]),
        bid_up=float(normalized["bid_up"]) if normalized.get("bid_up") is not None else None,
        bid_down=float(normalized["bid_down"]) if normalized.get("bid_down") is not None else None,
        time_to_expiry_seconds=(
            float(normalized["time_to_expiry_seconds"])
            if normalized.get("time_to_expiry_seconds") is not None
            else None
        ),
        source_signal_id=normalized.get("source_signal_id"),
        model_score=(
            float(normalized["model_score"])
            if normalized.get("model_score") is not None
            else None
        ),
    )


def _signals_from_rows(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    signals: list[ExecutionLayerV2Signal] = []
    rejected = []
    for index, row in enumerate(rows):
        try:
            signals.append(_signal_from_row(row))
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "row_index": index,
                    "market_id": row.get("market_id"),
                    "decision_ts": row.get("decision_ts"),
                    "reason_codes": _signal_rejection_reason_codes(row, exc),
                    "error": str(exc),
                }
            )
    return {
        "signals": tuple(signals),
        "accepted_signal_row_count": len(signals),
        "rejected_rows": rejected,
    }


def _normalized_signal_fields(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("features")
    source = {**row, **features} if isinstance(features, dict) else dict(row)
    if "full_5_action_ranking" in row and (
        "ask_up" not in source or "ask_down" not in source
    ):
        source.update(_price_fields_from_full_action_ranking(row["full_5_action_ranking"]))
    snapshot = row.get("microstructure_snapshot")
    selected_side = row.get("selected_side") or row.get("canonical_selected_side")
    if isinstance(snapshot, dict) and selected_side in {"UP", "DOWN"}:
        suffix = "up" if selected_side == "UP" else "down"
        source.setdefault(f"ask_{suffix}", snapshot.get("entry_ask"))
        source.setdefault(f"bid_{suffix}", snapshot.get("executable_exit_bid_proxy"))
        source.setdefault("time_to_expiry_seconds", snapshot.get("time_to_close_seconds"))
    source.setdefault("time_to_expiry_seconds", source.get("time_to_close_seconds"))
    source.setdefault("model_score", source.get("corrected_model_score"))
    source.setdefault(
        "source_signal_id",
        source.get("decision_group_id")
        or source.get("o_v8_paper_fresh_signal_trace_row_hash")
        or canonical_json_sha256(
            {
                "market_id": source.get("market_id"),
                "decision_ts": source.get("decision_ts"),
            }
        ),
    )
    return source


def _price_fields_from_full_action_ranking(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {}
    fields: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        side = row.get("selected_side") or row.get("side")
        snapshot = row.get("microstructure_snapshot")
        if side not in {"UP", "DOWN"} or not isinstance(snapshot, dict):
            continue
        suffix = "up" if side == "UP" else "down"
        fields.setdefault(f"ask_{suffix}", snapshot.get("entry_ask"))
        fields.setdefault(f"bid_{suffix}", snapshot.get("executable_exit_bid_proxy"))
        fields.setdefault("time_to_expiry_seconds", snapshot.get("time_to_close_seconds"))
    return fields


def _signal_rejection_reason_codes(row: dict[str, Any], exc: Exception) -> list[str]:
    reasons = ["signal_row_not_convertible_fail_closed"]
    normalized = _normalized_signal_fields(row)
    for field_name in ("market_id", "decision_ts", "p_up", "ask_up", "ask_down"):
        if normalized.get(field_name) is None:
            reasons.append(f"missing_{field_name}")
    if "p_up" not in normalized and "p_down" not in normalized:
        reasons.append("missing_decision_time_probability")
    if isinstance(exc, ValueError):
        reasons.append("invalid_signal_field_value")
    return sorted(set(reasons))


def _rows_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if not isinstance(payload, dict):
        raise ValueError("execution layer v2 JSON input must be an object or list")
    for key in (
        "holdout_decision_rows",
        "trace_rows",
        "decision_rows",
        "signal_rows",
        "rows",
    ):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows]
    return [dict(payload)]


def _forbidden_fields_by_row(rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    failures = []
    forbidden_set = set(EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS)
    for index, row in enumerate(rows):
        present = sorted(forbidden_set.intersection(row))
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


def _v1_baseline_summary(
    signals: tuple[ExecutionLayerV2Signal, ...],
    *,
    decisions: tuple[ExecutionLayerV2Decision, ...],
    config: ExecutionLayerV2Config,
) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    state_machine_action_counts: Counter[str] = Counter()
    for signal in signals:
        up_ev = _entry_ev(signal, "UP", config)
        down_ev = _entry_ev(signal, "DOWN", config)
        best_ev = max(up_ev, down_ev)
        best_side = "UP" if up_ev >= down_ev else "DOWN"
        if best_ev >= config.entry_ev_threshold:
            action_counts[f"BUY_{best_side}_HOLD_TO_SETTLEMENT"] += 1
            state_machine_action_counts["ENTER_POSITION"] += 1
        else:
            action_counts["NO_TRADE"] += 1
            state_machine_action_counts["NO_ACTION"] += 1
    v2_action_counts = Counter(decision.action for decision in decisions)
    return {
        "baseline_name": EXECUTION_LAYER_V2_BASELINE_NAME,
        "baseline_assumption": "enter positive EV then hold to settlement",
        "baseline_action_counts": dict(sorted(action_counts.items())),
        "baseline_state_machine_action_counts": dict(
            sorted(state_machine_action_counts.items())
        ),
        "v2_action_counts": dict(sorted(v2_action_counts.items())),
        "v2_differs_from_v1": dict(state_machine_action_counts) != dict(v2_action_counts),
        "uses_realized_pnl_or_settlement_outcomes": False,
    }


def _lambda_grid_diagnostics(
    signals: tuple[ExecutionLayerV2Signal, ...],
    *,
    config: ExecutionLayerV2Config,
) -> list[dict[str, Any]]:
    rows = []
    for value in config.diagnostic_lambda_grid:
        notionals = []
        for signal in signals:
            up_ev = _entry_ev(signal, "UP", config)
            down_ev = _entry_ev(signal, "DOWN", config)
            side: ExecutionLayerV2Side = "UP" if up_ev >= down_ev else "DOWN"
            sizing_config = ExecutionLayerV2Config(
                **{
                    **config.to_dict(),
                    "kelly_time_decay_lambda": value,
                }
            )
            sizing = time_decayed_kelly_notional(
                probability=_probability(signal, side),
                price=_ask(signal, side),
                time_to_expiry_seconds=signal.time_to_expiry_seconds,
                config=sizing_config,
            )
            notionals.append(sizing["paper_notional"])
        rows.append(
            {
                "kelly_time_decay_lambda": value,
                "mean_candidate_notional": sum(notionals) / len(notionals)
                if notionals
                else 0.0,
                "max_candidate_notional": max(notionals, default=0.0),
                "selection_metric": "size_sensitivity_no_outcomes",
                "uses_validation_labels_for_tuning": False,
            }
        )
    return rows


def _entry_ev(
    signal: ExecutionLayerV2Signal,
    side: ExecutionLayerV2Side,
    config: ExecutionLayerV2Config,
) -> float:
    return _probability(signal, side) - _ask(signal, side) - _cost(config)


def _best_entry_ev(
    signal: ExecutionLayerV2Signal,
    config: ExecutionLayerV2Config,
) -> float:
    return max(_entry_ev(signal, "UP", config), _entry_ev(signal, "DOWN", config))


def _probability(signal: ExecutionLayerV2Signal, side: ExecutionLayerV2Side) -> float:
    if side == "UP":
        return signal.p_up
    if side == "DOWN":
        return float(signal.p_down)
    return 0.0


def _ask(signal: ExecutionLayerV2Signal, side: ExecutionLayerV2Side) -> float:
    if side == "UP":
        return signal.ask_up
    if side == "DOWN":
        return signal.ask_down
    return 0.0


def _bid(signal: ExecutionLayerV2Signal, side: ExecutionLayerV2Side) -> float:
    if side == "UP":
        return signal.bid_up if signal.bid_up is not None else signal.ask_up
    if side == "DOWN":
        return signal.bid_down if signal.bid_down is not None else signal.ask_down
    return 0.0


def _cost(config: ExecutionLayerV2Config) -> float:
    return config.execution_cost_bps / 10_000.0


def _safety_report_fields(config: ExecutionLayerV2Config) -> dict[str, Any]:
    fields = {
        **compact_safety_fields(),
        "polymarket_write_enabled": config.polymarket_write_enabled,
        "wallet_signing_enabled": config.wallet_signing_enabled,
        "v8_execution_handoff_allowed": config.v8_execution_handoff_allowed,
        "source_model_candidate_eligible": config.source_model_candidate_eligible,
        "freeze_ready": config.freeze_ready,
        "promotion_evidence_eligible": config.promotion_evidence_eligible,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    return fields


def _validate_safety_flags(obj: Any) -> None:
    expected = {
        **compact_safety_fields(),
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    for field_name, expected_value in expected.items():
        if getattr(obj, field_name) is not expected_value:
            raise ValueError(f"{field_name} must be {expected_value}")
    for field_name in (
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
    ):
        if hasattr(obj, field_name) and getattr(obj, field_name) is not False:
            raise ValueError(f"{field_name} must be false")


def _validate_probability(field_name: str, value: float) -> None:
    _require_finite(field_name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")


def _validate_price(field_name: str, value: float, *, allow_zero: bool = False) -> None:
    _require_finite(field_name, value)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value >= 1.0:
        lower = "[0, 1)" if allow_zero else "(0, 1)"
        raise ValueError(f"{field_name} must be in {lower}")


def _require_finite(field_name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
