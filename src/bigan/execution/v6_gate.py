"""Live/paper v6 gate helpers for Phase 4 execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.modeling.xgboost_v6 import joint_decision_from_payload


@dataclass(frozen=True, slots=True)
class V6JointGateConfig:
    """Issue #93/#94 v6 gate settings for live execution."""

    settlement_threshold: float = 0.50
    neutral_cap: float = 0.25
    volatility_threshold: float = 0.60
    round_trip_cost: float = 0.072
    ev_margin: float = 0.01
    gain_priors: tuple[tuple[str, float], ...] = (
        ("up", 0.29554594232059017),
        ("down", 0.29845284327323157),
    )

    @property
    def gain_priors_dict(self) -> dict[str, float]:
        return dict(self.gain_priors)

    def joint_rule(self) -> dict[str, float]:
        return {
            "settlement_threshold": self.settlement_threshold,
            "neutral_cap": self.neutral_cap,
            "volatility_threshold": self.volatility_threshold,
        }


def is_v6_model_version(model_version: str) -> bool:
    return model_version == "xgboost-v6" or model_version.startswith("xgboost-v6:")


def v6_payload_from_values(
    *,
    model_version: str,
    p_up: float,
    p_down: float,
    p_neutral: float,
    p_vol_up: float,
    p_vol_down: float,
) -> dict[str, float | str]:
    return {
        "model_version": model_version,
        "p_up": float(p_up),
        "p_down": float(p_down),
        "p_neutral": float(p_neutral),
        "p_vol_up": float(p_vol_up),
        "p_vol_down": float(p_vol_down),
    }


def v6_payload_from_snapshot(
    snapshot: dict[str, Any],
    *,
    model_version: str,
) -> dict[str, float | str] | None:
    """Read explicit v6 probabilities from a monitoring snapshot."""

    def _value(*keys: str) -> float | None:
        for key in keys:
            raw = snapshot.get(key)
            if raw is None and isinstance(snapshot.get("features"), dict):
                raw = snapshot["features"].get(key)
            if raw is not None:
                return float(raw)
        return None

    p_up = _value("p_up")
    p_down = _value("p_down")
    p_neutral = _value("p_neutral")
    p_vol_up = _value("p_vol_up")
    p_vol_down = _value("p_vol_down")
    if None in {p_up, p_down, p_neutral, p_vol_up, p_vol_down}:
        return None
    return v6_payload_from_values(
        model_version=model_version,
        p_up=p_up,
        p_down=p_down,
        p_neutral=p_neutral,
        p_vol_up=p_vol_up,
        p_vol_down=p_vol_down,
    )


def evaluate_v6_joint_side(
    payload: dict[str, float | str],
    config: V6JointGateConfig,
) -> str | None:
    """Return UP/DOWN when the offline-trained joint gate admits a trade."""

    return joint_decision_from_payload(
        payload,
        joint_rule=config.joint_rule(),
        round_trip_cost=config.round_trip_cost,
        ev_margin=config.ev_margin,
        gain_priors=config.gain_priors_dict,
    )


def evaluate_v6_settlement_side(
    payload: dict[str, float | str],
    config: V6JointGateConfig,
) -> str | None:
    """Return UP/DOWN when the settlement head alone admits a settlement bet."""

    p_up = float(payload["p_up"])
    p_down = float(payload["p_down"])
    if p_up >= p_down and p_up >= config.settlement_threshold:
        return "UP"
    if p_down > p_up and p_down >= config.settlement_threshold:
        return "DOWN"
    return None


def evaluate_v6_volatility_side(
    payload: dict[str, float | str],
    config: V6JointGateConfig,
) -> str | None:
    """Return UP/DOWN when the volatility heads alone admit a volatility bet."""

    p_neutral = float(payload["p_neutral"])
    if p_neutral > config.neutral_cap:
        return None
    p_vol_up = float(payload["p_vol_up"])
    p_vol_down = float(payload["p_vol_down"])
    if p_vol_up >= p_vol_down and p_vol_up >= config.volatility_threshold:
        return "UP"
    if p_vol_down > p_vol_up and p_vol_down >= config.volatility_threshold:
        return "DOWN"
    return None


def v6_selection_score(payload: dict[str, float | str], side: str) -> float:
    """Rank competing round signals by admitted settlement-side confidence."""

    if side == "UP":
        return float(payload["p_up"])
    return float(payload["p_down"])


def v6_joint_selection_score(payload: dict[str, float | str], side: str) -> float:
    """Rank competing opportunities when explicitly evaluating the old joint gate."""

    if side == "UP":
        return float(payload["p_up"]) * float(payload["p_vol_up"])
    return float(payload["p_down"]) * float(payload["p_vol_down"])


def load_v6_gain_priors(model_json_path: Path | str) -> dict[str, float]:
    artifact = json.loads(Path(model_json_path).read_text(encoding="utf-8"))
    priors = artifact.get("volatility_gain_priors", {})
    return {str(key): float(value) for key, value in priors.items()}


def build_v6_signal_fields(
    *,
    event_id: str,
    ts: int,
    created_at: int,
    snapshot: dict[str, Any],
    model_version: str,
    config: V6JointGateConfig,
    round_end_ts: int,
    bridged_at: int = 0,
    opposite_token_id: str = "",
) -> dict[str, Any] | None:
    """Map one prediction snapshot to executor/bridge signal fields."""

    payload = v6_payload_from_snapshot(snapshot, model_version=model_version)
    if payload is None:
        return None
    settlement_side = evaluate_v6_settlement_side(payload, config)
    volatility_side = evaluate_v6_volatility_side(payload, config)
    side = settlement_side or volatility_side
    if side is None:
        return None
    canonical_symbol = str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
    parts = canonical_symbol.split(":")
    if len(parts) < 3:
        return None
    family, round_slug, token_side = parts[0], parts[-2], parts[-1].upper()
    if token_side not in {"UP", "DOWN"}:
        return None
    market = snapshot.get("market_implied_prob")
    if market is None:
        return None
    market_implied_prob = float(market)
    source_token_id = str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
    if not source_token_id:
        return None
    if token_side == "UP":
        up_token_id = source_token_id
        down_token_id = opposite_token_id
    else:
        up_token_id = opposite_token_id
        down_token_id = source_token_id
    if side == "UP":
        token_id = up_token_id
        if not token_id:
            return None
        token_probability = (
            float(payload["p_up"]) if settlement_side is not None else float(payload["p_vol_up"])
        )
        outcome_canonical = f"{family}:{round_slug}:UP"
    else:
        token_id = down_token_id
        if not token_id:
            return None
        token_probability = (
            float(payload["p_down"])
            if settlement_side is not None
            else float(payload["p_vol_down"])
        )
        outcome_canonical = f"{family}:{round_slug}:DOWN"
    selected_market_implied_prob = (
        market_implied_prob if side == token_side else 1.0 - market_implied_prob
    )
    return {
        "event_id": event_id,
        "ts": int(ts),
        "created_at": int(created_at),
        "prob_up_15m": float(payload["p_up"]),
        "canonical_symbol": outcome_canonical,
        "token_id": token_id,
        "outcome_side": side,
        "round_slug": round_slug,
        "round_end_ts": int(round_end_ts),
        "market_implied_prob": selected_market_implied_prob,
        "token_probability": token_probability,
        "edge": token_probability - selected_market_implied_prob,
        "bridged_at": bridged_at,
        "opposite_token_id": down_token_id if side == "UP" else up_token_id,
        "p_up": float(payload["p_up"]),
        "p_down": float(payload["p_down"]),
        "p_neutral": float(payload["p_neutral"]),
        "p_vol_up": float(payload["p_vol_up"]),
        "p_vol_down": float(payload["p_vol_down"]),
        "v6_joint_side": settlement_side,
    }


def v6_joint_gate_config_from_model(
    model_json_path: Path | str,
    *,
    settlement_threshold: float = 0.50,
    neutral_cap: float = 0.25,
    volatility_threshold: float = 0.60,
    round_trip_cost: float = 0.072,
    ev_margin: float = 0.01,
) -> V6JointGateConfig:
    priors = load_v6_gain_priors(model_json_path)
    return V6JointGateConfig(
        settlement_threshold=settlement_threshold,
        neutral_cap=neutral_cap,
        volatility_threshold=volatility_threshold,
        round_trip_cost=round_trip_cost,
        ev_margin=ev_margin,
        gain_priors=tuple(sorted(priors.items())),
    )
