"""Executor-ready prediction signal JSONL queue helpers."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from bigan.execution.v6_gate import (
    V6JointGateConfig,
    build_v6_signal_fields,
    is_v6_model_version,
)
from bigan.monitoring.events import prediction_event_from_prediction_row
from bigan.monitoring.market_quality import (
    round_end_ts_from_canonical_symbol,
    tradable_market_implied_probability,
)


@dataclass(frozen=True, slots=True)
class ExecutionSignal:
    """One executor-ready signal row for ``--signal-jsonl-path``."""

    event_id: str
    ts: int
    created_at: int
    model_version: str
    prob_up_15m: float
    canonical_symbol: str
    token_id: str
    outcome_side: str
    round_slug: str
    round_end_ts: int
    market_implied_prob: float
    token_probability: float
    edge: float
    bridged_at: int
    opposite_token_id: str = ""
    p_up: float | None = None
    p_down: float | None = None
    p_neutral: float | None = None
    p_vol_up: float | None = None
    p_vol_down: float | None = None
    v6_joint_side: str | None = None


def append_prediction_rows_as_signal_jsonl(
    path: Path | str,
    rows: list[dict[str, Any]],
    *,
    model_version: str,
    allowed_families: frozenset[str] = frozenset({"BTC-15M"}),
    allowed_outcome_sides: frozenset[str] | None = None,
    v6_joint_config: V6JointGateConfig | None = None,
    bridged_at: int | None = None,
    max_event_age_seconds: float | None = None,
    token_ids_by_market_side: Mapping[tuple[str, str, str], str] | None = None,
) -> int:
    """Append executor-ready signal rows built directly from prediction rows."""

    if max_event_age_seconds is not None and max_event_age_seconds <= 0:
        raise ValueError("max_event_age_seconds must be positive")
    signals = build_execution_signals_from_prediction_rows(
        rows,
        model_version=model_version,
        allowed_families=allowed_families,
        allowed_outcome_sides=allowed_outcome_sides,
        v6_joint_config=v6_joint_config,
        bridged_at=bridged_at,
        token_ids_by_market_side=token_ids_by_market_side,
    )
    if not signals:
        return 0
    current_round_slug = _current_round_slug(signals)
    signals = [signal for signal in signals if signal.round_slug == current_round_slug]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_round_slug = _existing_queue_round_slug(output_path)
    mode = "w" if existing_round_slug and existing_round_slug != current_round_slug else "a"
    if max_event_age_seconds is not None:
        signals = [
            signal
            for signal in signals
            if _signal_age_seconds(signal) <= max_event_age_seconds
        ]
    if not signals:
        if mode == "w":
            output_path.write_text("", encoding="utf-8")
        return 0
    existing_identities = (
        set() if mode == "w" else _existing_signal_identities(output_path, current_round_slug)
    )
    signals_to_write: list[ExecutionSignal] = []
    for signal in signals:
        identity = _signal_identity(signal)
        if identity in existing_identities:
            continue
        existing_identities.add(identity)
        signals_to_write.append(signal)
    if not signals_to_write:
        return 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for signal in signals_to_write:
            handle.write(json.dumps(asdict(signal), sort_keys=True) + "\n")
    return len(signals_to_write)


def build_execution_signals_from_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    model_version: str,
    allowed_families: frozenset[str] = frozenset({"BTC-15M"}),
    allowed_outcome_sides: frozenset[str] | None = None,
    v6_joint_config: V6JointGateConfig | None = None,
    bridged_at: int | None = None,
    token_ids_by_market_side: Mapping[tuple[str, str, str], str] | None = None,
) -> list[ExecutionSignal]:
    """Build executor-ready signals without rereading ``prediction_events``."""

    token_by_side = dict(token_ids_by_market_side or {})
    token_by_side.update(_token_ids_by_market_side(rows))
    emitted_at = _now_ms() if bridged_at is None else int(bridged_at)
    signals: list[ExecutionSignal] = []
    for row in rows:
        signal = _execution_signal_from_prediction_row(
            row,
            model_version=model_version,
            allowed_families=allowed_families,
            allowed_outcome_sides=allowed_outcome_sides,
            v6_joint_config=v6_joint_config,
            token_by_side=token_by_side,
            bridged_at=emitted_at,
        )
        if signal is not None:
            signals.append(signal)
    return signals


def _execution_signal_from_prediction_row(
    row: dict[str, Any],
    *,
    model_version: str,
    allowed_families: frozenset[str],
    allowed_outcome_sides: frozenset[str] | None,
    v6_joint_config: V6JointGateConfig | None,
    token_by_side: dict[tuple[str, str, str], str],
    bridged_at: int,
) -> ExecutionSignal | None:
    event = prediction_event_from_prediction_row(row)
    snapshot = _snapshot_from_prediction_row(row)
    canonical_symbol = str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
    parsed = _parse_canonical_symbol(canonical_symbol)
    if parsed is None:
        return None
    family, round_slug, token_side = parsed
    if family not in allowed_families or token_side not in {"UP", "DOWN"}:
        return None
    if (
        allowed_outcome_sides is not None
        and v6_joint_config is None
        and token_side not in allowed_outcome_sides
    ):
        return None
    round_end_ts = round_end_ts_from_canonical_symbol(canonical_symbol)
    if round_end_ts is None:
        return None
    opposite_side = "DOWN" if token_side == "UP" else "UP"
    opposite_token_id = token_by_side.get((family, round_slug, opposite_side), "")
    created_at = int(row.get("created_at") or row.get("ingest_ts") or _now_ms())
    if v6_joint_config is not None and is_v6_model_version(model_version):
        fields = build_v6_signal_fields(
            event_id=event.event_id,
            ts=int(event.ts),
            created_at=created_at,
            snapshot=snapshot,
            model_version=model_version,
            config=v6_joint_config,
            round_end_ts=int(round_end_ts),
            bridged_at=bridged_at,
            opposite_token_id=opposite_token_id,
        )
        if fields is None:
            return None
        if allowed_outcome_sides is not None and fields["outcome_side"] not in allowed_outcome_sides:
            return None
        return ExecutionSignal(
            event_id=str(fields["event_id"]),
            ts=int(fields["ts"]),
            created_at=int(fields["created_at"]),
            model_version=model_version,
            prob_up_15m=float(fields["prob_up_15m"]),
            canonical_symbol=str(fields["canonical_symbol"]),
            token_id=str(fields["token_id"]),
            outcome_side=str(fields["outcome_side"]),
            round_slug=str(fields["round_slug"]),
            round_end_ts=int(fields["round_end_ts"]),
            market_implied_prob=float(fields["market_implied_prob"]),
            token_probability=float(fields["token_probability"]),
            edge=float(fields["edge"]),
            bridged_at=int(fields["bridged_at"]),
            opposite_token_id=str(fields.get("opposite_token_id") or ""),
            p_up=float(fields["p_up"]),
            p_down=float(fields["p_down"]),
            p_neutral=float(fields["p_neutral"]),
            p_vol_up=float(fields["p_vol_up"]),
            p_vol_down=float(fields["p_vol_down"]),
            v6_joint_side=(
                str(fields["v6_joint_side"]) if fields.get("v6_joint_side") else None
            ),
        )

    token_id = str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
    market = tradable_market_implied_probability(snapshot, event_ts=int(event.ts))
    if not token_id or market is None:
        return None
    prob = float(row["prob_up_15m"])
    token_probability = 1.0 - prob if token_side == "DOWN" else prob
    return ExecutionSignal(
        event_id=event.event_id,
        ts=int(event.ts),
        created_at=created_at,
        model_version=model_version,
        prob_up_15m=prob,
        canonical_symbol=canonical_symbol,
        token_id=token_id,
        outcome_side=token_side,
        round_slug=round_slug,
        round_end_ts=int(round_end_ts),
        market_implied_prob=float(market),
        token_probability=token_probability,
        edge=token_probability - float(market),
        bridged_at=bridged_at,
        opposite_token_id=opposite_token_id,
    )


def _snapshot_from_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "source": row.get("source"),
        "source_symbol": row.get("source_symbol"),
        "source_market": row.get("source_market"),
        "canonical_symbol": row.get("canonical_symbol"),
        "symbol": row.get("symbol"),
        "market_implied_prob": row.get("market_implied_prob"),
        "p_up": row.get("p_up"),
        "p_down": row.get("p_down"),
        "p_neutral": row.get("p_neutral"),
        "p_vol_up": row.get("p_vol_up"),
        "p_vol_down": row.get("p_vol_down"),
    }
    feature_values_json = row.get("feature_values_json")
    if feature_values_json is not None:
        try:
            features = json.loads(str(feature_values_json))
        except json.JSONDecodeError:
            features = {}
        if isinstance(features, dict):
            for key, value in features.items():
                snapshot.setdefault(key, value)
            snapshot["features"] = features
    return snapshot


def _token_ids_by_market_side(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    token_by_side: dict[tuple[str, str, str], str] = {}
    for row in rows:
        snapshot = _snapshot_from_prediction_row(row)
        parsed = _parse_canonical_symbol(
            str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
        )
        if parsed is None:
            continue
        family, round_slug, token_side = parsed
        token_id = str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
        if token_side in {"UP", "DOWN"} and token_id:
            token_by_side[(family, round_slug, token_side)] = token_id
    return token_by_side


def _parse_canonical_symbol(canonical_symbol: str) -> tuple[str, str, str] | None:
    parts = canonical_symbol.split(":")
    if len(parts) < 3:
        return None
    return parts[0].upper(), parts[-2], parts[-1].upper()


def _current_round_slug(signals: list[ExecutionSignal]) -> str:
    """Return the round represented by the freshest executor signal batch."""

    if not signals:
        raise ValueError("signals is empty")
    newest = max(signals, key=lambda signal: (signal.ts, signal.round_end_ts))
    return newest.round_slug


def _existing_queue_round_slug(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            for raw in reversed(handle.readlines()):
                raw = raw.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                round_slug = payload.get("round_slug")
                return str(round_slug) if round_slug else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _existing_signal_identities(path: Path, round_slug: str) -> set[tuple[object, ...]]:
    if not path.exists():
        return set()
    identities: set[tuple[object, ...]] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                if payload.get("round_slug") != round_slug:
                    continue
                identities.add(_payload_signal_identity(payload))
    except (OSError, json.JSONDecodeError):
        return set()
    return identities


def _signal_identity(signal: ExecutionSignal) -> tuple[object, ...]:
    return (
        signal.ts,
        signal.created_at,
        signal.round_slug,
        signal.outcome_side,
        signal.token_id,
        signal.opposite_token_id,
        _rounded_probability(signal.p_up),
        _rounded_probability(signal.p_down),
        _rounded_probability(signal.p_neutral),
        _rounded_probability(signal.p_vol_up),
        _rounded_probability(signal.p_vol_down),
    )


def _payload_signal_identity(payload: dict[str, Any]) -> tuple[object, ...]:
    return (
        payload.get("ts"),
        payload.get("created_at"),
        payload.get("round_slug"),
        payload.get("outcome_side"),
        payload.get("token_id"),
        payload.get("opposite_token_id") or "",
        _rounded_probability(payload.get("p_up")),
        _rounded_probability(payload.get("p_down")),
        _rounded_probability(payload.get("p_neutral")),
        _rounded_probability(payload.get("p_vol_up")),
        _rounded_probability(payload.get("p_vol_down")),
    )


def _signal_age_seconds(signal: ExecutionSignal) -> float:
    return max(0.0, (signal.bridged_at - signal.ts) / 1000.0)


def _rounded_probability(value: object) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def _now_ms() -> int:
    return int(time.time() * 1000)
