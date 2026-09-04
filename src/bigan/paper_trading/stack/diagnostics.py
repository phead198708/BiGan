"""Safe projections of gate failures, independent of trading decisions."""

from __future__ import annotations

import re
from typing import Any

from bigan.paper_trading.operator.diagnostics import MAX_COUNTER, NUMERIC_FIELDS, DiagnosticCode
from bigan.paper_trading.operator.read_model import OperatorState

TRACE_LIMIT = 180
SOURCES = ("binance", "polymarket", "chainlink")


def _obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int | None:
    return value if type(value) is int and -MAX_COUNTER <= value <= MAX_COUNTER else None


def readiness_reasons(payload: dict[str, Any]) -> list[str]:
    """Describe the existing live_inputs_ready predicate; never relax its gates."""
    status = _obj(payload.get("status"))
    reasons = []
    if payload.get("stale") is not False:
        reasons.append("STATUS_STALE")
    if status.get("state") != "RUNNING":
        reasons.append("OPERATOR_NOT_RUNNING")
    if not status.get("run_id"):
        reasons.append("RUN_MISSING")
    market = status.get("active_market")
    if not isinstance(market, dict):
        reasons.append("MARKET_MISSING")
    else:
        now, start, end = (_int(payload.get("generated_at_ms")),
                           _int(market.get("window_start_ts_ms")), _int(market.get("window_end_ts_ms")))
        if now is None or start is None or end is None or not start <= now < end:
            reasons.append("MARKET_OUTSIDE_WINDOW")
    feeds = _obj(status.get("feeds"))
    for source in SOURCES:
        health = _obj(feeds.get(source))
        for field, suffix in (("connected", "DISCONNECTED"), ("synchronized", "UNSYNCHRONIZED"),
                              ("fresh", "NOT_FRESH")):
            if health.get(field) is not True:
                reasons.append(source.upper() + "_" + suffix)
    pricing = _obj(status.get("pricing_inputs"))
    if pricing.get("ready") is not True:
        reasons.append("PRICING_NOT_READY")
    if pricing.get("fresh") is not True:
        reasons.append("PRICING_NOT_FRESH")
    if _obj(status.get("alpha")).get("fresh") is not True:
        reasons.append("ALPHA_NOT_FRESH")
    session = _obj(status.get("session"))
    if session.get("healthy") is not True or session.get("failure_reason") is not None:
        reasons.append("SESSION_UNHEALTHY")
    return reasons


def _component_diagnostics(value: Any) -> dict[str, Any]:
    data = _obj(value)
    counts = _obj(data.get("counts"))
    safe_counts = {code.value: counts[code.value] for code in DiagnosticCode
                   if _int(counts.get(code.value)) is not None and counts[code.value] >= 0}
    recent = data.get("recent")
    events = []
    for item in (recent[-8:] if isinstance(recent, list) else []):
        item = _obj(item)
        if not isinstance(item.get("code"), str) or item["code"] not in DiagnosticCode._value2member_map_:
            continue
        events.append({"code": item["code"], **{key: item[key] for key in NUMERIC_FIELDS
                                               if _int(item.get(key)) is not None}})
    return {"counts": safe_counts, "recent": events}


def readiness_snapshot(payload: dict[str, Any], *, phase: str) -> dict[str, Any]:
    if phase not in {"startup", "runtime"}:
        raise ValueError("invalid readiness phase")
    status = _obj(payload.get("status"))
    feeds = _obj(status.get("feeds"))
    market = _obj(status.get("active_market"))
    pricing = _obj(status.get("pricing_inputs"))
    alpha = _obj(status.get("alpha"))
    run_id = status.get("run_id")
    run_id = run_id if isinstance(run_id, str) and re.fullmatch(r"paper-[0-9a-f]{24}", run_id) else None
    state = status.get("state")
    state = state if isinstance(state, str) and state in OperatorState._value2member_map_ else "UNKNOWN"
    components = {name: _obj(feeds.get(name)) for name in SOURCES}
    components.update(pricing=pricing, alpha=alpha)
    tokens = _obj(components["polymarket"].get("tokens"))
    for side in ("yes", "no"):
        components["polymarket_" + side] = _obj(tokens.get(side))
    return {
        "phase": phase, "observed_at_ms": _int(payload.get("generated_at_ms")),
        "status_updated_at_ms": _int(status.get("updated_at_ms")), "state": state, "run_id": run_id,
        "window_start_ts_ms": _int(market.get("window_start_ts_ms")),
        "window_end_ts_ms": _int(market.get("window_end_ts_ms")),
        "reasons": readiness_reasons(payload),
        "components": {name: {
            **{key: health[key] for key in ("connected", "synchronized", "fresh", "ready")
               if type(health.get(key)) is bool},
            **{key: _int(health.get(key)) for key in (
                "age_ms", "timestamp_ms", "last_event_ts_ms", "last_message_received_ms", "reconnect_count",
                "error_count", "spot_sample_count", "oracle_sample_count", "return_sample_count",
            )},
        } for name, health in components.items()},
        "diagnostics": {name: _component_diagnostics(components[name].get("diagnostics"))
                        for name in (*SOURCES, "pricing")},
    }
