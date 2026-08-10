"""Offline-only, fail-closed runtime for a future frozen pooled residual candidate."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.challenge_model_15m_training import (
    BASE_FEATURE_NAMES,
    side_symmetric_features,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_candidate_evaluation import FEATURE_NAMES

BUNDLE_SCHEMA_VERSION = "bigan-btc-15m-pooled-residual-runtime-bundle-v1"
RUNTIME_RESULT_SCHEMA_VERSION = "bigan-btc-15m-pooled-residual-runtime-result-v1"
MARKET_FAMILY = "btc_updown_15m"
MARKET_HORIZON_MS = 900_000
ACTIONS = ("NO_TRADE", "BUY_UP_HOLD", "BUY_DOWN_HOLD")


class PooledResidualRuntimeError(ValueError):
    """Raised before scoring when frozen runtime artifacts are not trustworthy."""


@dataclass(frozen=True, slots=True)
class PooledResidualRuntime:
    """An immutable in-memory scorer with no network, wallet, or write surface."""

    candidate_id: str
    lineage_id: str
    manifest_sha256: str
    model_sha256: str
    feature_contract_sha256: str
    candidate_freeze_sha256: str
    maximum_decision_lag_ms: int
    maximum_source_age_ms: int
    booster: xgb.Booster

    def score_feature_row(
        self,
        feature_row: Mapping[str, Any],
        *,
        observed_at_ts: int,
    ) -> dict[str, Any]:
        """Score one canonical decision-time row or return a safe NO_TRADE."""

        try:
            matrix, decision_ts = _validated_runtime_matrix(
                feature_row,
                observed_at_ts=observed_at_ts,
                maximum_decision_lag_ms=self.maximum_decision_lag_ms,
                maximum_source_age_ms=self.maximum_source_age_ms,
            )
            raw = self.booster.predict(
                xgb.DMatrix(
                    matrix,
                    feature_names=list(FEATURE_NAMES),
                    missing=np.nan,
                )
            )
            if raw.shape != (2,) or any(not math.isfinite(float(value)) for value in raw):
                raise ValueError("model prediction is not a finite UP/DOWN pair")
            scores = {"UP": float(raw[0]), "DOWN": float(raw[1])}
            action = _select_action(scores)
            return _runtime_result(
                runtime=self,
                feature_row=feature_row,
                decision_ts=decision_ts,
                observed_at_ts=observed_at_ts,
                scores=scores,
                selected_action=action,
                fail_closed_reasons=[],
            )
        except (KeyError, TypeError, ValueError, xgb.core.XGBoostError) as exc:
            return _runtime_result(
                runtime=self,
                feature_row=feature_row,
                decision_ts=_safe_int(feature_row.get("decision_ts")),
                observed_at_ts=observed_at_ts,
                scores=None,
                selected_action="NO_TRADE",
                fail_closed_reasons=[str(exc) or exc.__class__.__name__],
            )


def load_pooled_residual_runtime(
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    repository_root: Path | str,
) -> PooledResidualRuntime:
    """Resolve and verify a repository-local UBJ bundle without depending on cwd."""

    root = Path(repository_root).resolve()
    manifest_input = Path(manifest_path)
    manifest_file = (
        manifest_input.resolve()
        if manifest_input.is_absolute()
        else (root / manifest_input).resolve()
    )
    expected_sha = _require_sha256(expected_manifest_sha256, "manifest SHA-256")
    if not manifest_file.is_relative_to(root) or not manifest_file.is_file():
        raise PooledResidualRuntimeError("runtime manifest must be repository-local")
    if sha256_file(manifest_file) != expected_sha:
        raise PooledResidualRuntimeError("runtime manifest SHA-256 mismatch")
    manifest = _load_json(manifest_file, "runtime manifest")
    blockers: list[str] = []
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        blockers.append("schema_version")
    candidate_id = str(manifest.get("candidate_id") or "")
    lineage_id = str(manifest.get("lineage_id") or "")
    if not candidate_id:
        blockers.append("candidate_id")
    if re.fullmatch(r"BTC-15M-cost-aware-market-residual-v[0-9]+", lineage_id) is None:
        blockers.append("lineage_id")
    if dict(manifest.get("market_contract") or {}) != {
        "family": MARKET_FAMILY,
        "horizon_seconds": 900,
        "sides": ["UP", "DOWN"],
    }:
        blockers.append("market_contract")
    if dict(manifest.get("decision_contract") or {}) != {
        "actions": list(ACTIONS),
        "NO_TRADE_value": 0.0,
        "accept_if": "highest_side_score>0",
        "fixed_acceptance_threshold": 0.0,
        "side_tie_break_order": ["UP", "DOWN"],
        "one_trade_maximum_per_market": True,
    }:
        blockers.append("decision_contract")
    runtime_authorization = dict(manifest.get("runtime_authorization") or {})
    if runtime_authorization != {
        "offline_readiness_only": True,
        "live_shadow_authorized": False,
        "paper_or_live_execution_authorized": False,
        "wallet_signing_authorized": False,
        "polymarket_write_authorized": False,
        "capital_at_risk": False,
    }:
        blockers.append("runtime_authorization")
    if dict(manifest.get("safety") or {}) != SAFETY:
        blockers.append("safety")
    freshness = dict(manifest.get("freshness_contract") or {})
    maximum_decision_lag_ms = _positive_int(
        freshness.get("maximum_decision_lag_ms"),
        "maximum_decision_lag_ms",
        blockers,
    )
    maximum_source_age_ms = _positive_int(
        freshness.get("maximum_source_age_ms"),
        "maximum_source_age_ms",
        blockers,
    )
    if freshness.get("stale_input_action") != "NO_TRADE":
        blockers.append("freshness_contract.stale_input_action")

    try:
        model_path = _verify_descriptor(manifest.get("model"), root, "model")
        feature_path = _verify_descriptor(
            manifest.get("feature_contract"), root, "feature_contract"
        )
        freeze_path = _verify_descriptor(
            manifest.get("candidate_freeze"), root, "candidate_freeze"
        )
    except PooledResidualRuntimeError as exc:
        blockers.append(str(exc))
        model_path = feature_path = freeze_path = None
    if blockers:
        raise PooledResidualRuntimeError(
            "invalid pooled residual runtime manifest: " + ", ".join(blockers)
        )
    assert model_path is not None and feature_path is not None and freeze_path is not None
    model_descriptor = dict(manifest["model"])
    if model_descriptor.get("format") != "xgboost_ubj":
        raise PooledResidualRuntimeError("runtime model format must be xgboost_ubj")
    if model_descriptor.get("objective") != "reg:squarederror":
        raise PooledResidualRuntimeError("runtime model objective mismatch")
    if model_descriptor.get("xgboost_version") != xgb.__version__:
        raise PooledResidualRuntimeError("runtime XGBoost version mismatch")

    feature_contract = _load_json(feature_path, "feature contract")
    _validate_feature_contract(feature_contract)
    freeze = _load_json(freeze_path, "candidate freeze")
    _validate_candidate_freeze(freeze, candidate_id=candidate_id, lineage_id=lineage_id)

    booster = xgb.Booster()
    try:
        booster.load_model(str(model_path))
    except xgb.core.XGBoostError as exc:
        raise PooledResidualRuntimeError("runtime UBJ model cannot be loaded") from exc
    if tuple(booster.feature_names or ()) != FEATURE_NAMES:
        raise PooledResidualRuntimeError("runtime UBJ feature names/order mismatch")
    if booster.num_features() != len(FEATURE_NAMES):
        raise PooledResidualRuntimeError("runtime UBJ feature count mismatch")
    objective = json.loads(booster.save_config())["learner"]["objective"]["name"]
    if objective != "reg:squarederror":
        raise PooledResidualRuntimeError("loaded runtime UBJ objective mismatch")
    if model_descriptor.get("ordered_feature_names_sha256") != canonical_json_sha256(
        list(FEATURE_NAMES)
    ):
        raise PooledResidualRuntimeError("runtime model feature contract hash mismatch")
    return PooledResidualRuntime(
        candidate_id=candidate_id,
        lineage_id=lineage_id,
        manifest_sha256=expected_sha,
        model_sha256=sha256_file(model_path),
        feature_contract_sha256=sha256_file(feature_path),
        candidate_freeze_sha256=sha256_file(freeze_path),
        maximum_decision_lag_ms=maximum_decision_lag_ms,
        maximum_source_age_ms=maximum_source_age_ms,
        booster=booster,
    )


def _validated_runtime_matrix(
    feature_row: Mapping[str, Any],
    *,
    observed_at_ts: int,
    maximum_decision_lag_ms: int,
    maximum_source_age_ms: int,
) -> tuple[np.ndarray, int]:
    if feature_row.get("market_family") != MARKET_FAMILY:
        raise ValueError("runtime market is not BTC 15m")
    if int(feature_row["horizon_ms"]) != MARKET_HORIZON_MS:
        raise ValueError("runtime market horizon is not 900 seconds")
    decision_ts = int(feature_row["decision_ts"])
    max_input_ts = int(feature_row["max_input_ts"])
    raw = dict(feature_row.get("features") or {})
    market_age_seconds = float(raw["market_age_seconds"])
    time_to_close_seconds = float(raw["time_to_close_seconds"])
    if (
        not math.isfinite(market_age_seconds)
        or not math.isfinite(time_to_close_seconds)
        or market_age_seconds < 0.0
        or time_to_close_seconds < 0.0
        or not math.isclose(
            market_age_seconds + time_to_close_seconds,
            MARKET_HORIZON_MS / 1000.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("runtime market horizon is not 900 seconds")
    market_start_ts = decision_ts - round(market_age_seconds * 1000.0)
    market_end_ts = decision_ts + round(time_to_close_seconds * 1000.0)
    if "market_start_ts" in feature_row and int(feature_row["market_start_ts"]) != (
        market_start_ts
    ):
        raise ValueError("runtime market start disagrees with decision-time features")
    if "market_end_ts" in feature_row and int(feature_row["market_end_ts"]) != market_end_ts:
        raise ValueError("runtime market end disagrees with decision-time features")
    if not market_start_ts <= decision_ts <= market_end_ts:
        raise ValueError("runtime decision timestamp is outside the market")
    if int(observed_at_ts) < decision_ts:
        raise ValueError("runtime observation precedes decision timestamp")
    if int(observed_at_ts) - decision_ts > maximum_decision_lag_ms:
        raise ValueError("runtime decision input is stale")
    if decision_ts - max_input_ts > maximum_source_age_ms:
        raise ValueError("runtime source input is stale")
    rows: list[list[float]] = []
    for side in ("UP", "DOWN"):
        transformed = side_symmetric_features(feature_row, side)
        if tuple(transformed) != FEATURE_NAMES:
            raise ValueError("runtime feature order mismatch")
        values = [float(transformed[name]) for name in FEATURE_NAMES]
        for index, name in enumerate(BASE_FEATURE_NAMES):
            value = values[index]
            missing = values[index + len(BASE_FEATURE_NAMES)]
            if missing not in (0.0, 1.0):
                raise ValueError(f"invalid missingness indicator: {name}")
            if math.isfinite(value) == bool(missing):
                raise ValueError(f"feature/missingness mismatch: {name}")
        rows.append(values)
    return np.asarray(rows, dtype=np.float64), decision_ts


def _select_action(scores: Mapping[str, float]) -> str:
    best_side = max(("UP", "DOWN"), key=lambda side: (scores[side], side == "UP"))
    if scores[best_side] <= 0.0:
        return "NO_TRADE"
    return "BUY_UP_HOLD" if best_side == "UP" else "BUY_DOWN_HOLD"


def _runtime_result(
    *,
    runtime: PooledResidualRuntime,
    feature_row: Mapping[str, Any],
    decision_ts: int | None,
    observed_at_ts: int,
    scores: Mapping[str, float] | None,
    selected_action: str,
    fail_closed_reasons: Sequence[str],
) -> dict[str, Any]:
    action_values = {
        "NO_TRADE": 0.0,
        "BUY_UP_HOLD": (float(scores["UP"]) if scores is not None else None),
        "BUY_DOWN_HOLD": (float(scores["DOWN"]) if scores is not None else None),
    }
    return {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "candidate_id": runtime.candidate_id,
        "lineage_id": runtime.lineage_id,
        "market_id": str(feature_row.get("market_id") or ""),
        "decision_ts": decision_ts,
        "observed_at_ts": int(observed_at_ts),
        "action_values": action_values,
        "selected_action": selected_action,
        "model_scored": scores is not None,
        "fail_closed": bool(fail_closed_reasons),
        "fail_closed_reasons": list(fail_closed_reasons),
        "manifest_sha256": runtime.manifest_sha256,
        "model_sha256": runtime.model_sha256,
        "offline_readiness_only": True,
        "live_shadow_authorized": False,
        "paper_or_live_execution_authorized": False,
        "wallet_signing_authorized": False,
        "polymarket_write_authorized": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }


def _validate_feature_contract(contract: Mapping[str, Any]) -> None:
    ordered = dict(contract.get("ordered_model_feature_contract") or {})
    causality = dict(contract.get("causality") or {})
    missingness = dict(contract.get("missingness") or {})
    valid = (
        tuple(contract.get("base_feature_names") or ()) == BASE_FEATURE_NAMES
        and ordered.get("base_feature_count") == len(BASE_FEATURE_NAMES)
        and ordered.get("missing_indicator_count") == len(BASE_FEATURE_NAMES)
        and ordered.get("ordered_feature_count") == len(FEATURE_NAMES)
        and ordered.get("ordered_feature_names_sha256")
        == canonical_json_sha256(list(FEATURE_NAMES))
        and causality.get("market_horizon_seconds") == 900
        and causality.get("available_at_ts_must_be_lte_decision_ts") is True
        and causality.get("feature_cutoff_ts_must_be_lte_decision_ts") is True
        and causality.get("max_input_ts_must_be_lte_decision_ts") is True
        and causality.get("settlement_outcome_allowed") is False
        and causality.get("target_or_pnl_allowed") is False
        and missingness.get("explicit_indicator_for_every_base_feature") is True
        and missingness.get("missing_encoded_as_numeric_zero_allowed") is False
        and missingness.get("native_model_missing_value") == "nan"
    )
    if not valid:
        raise PooledResidualRuntimeError("invalid pooled residual feature contract")


def _validate_candidate_freeze(
    freeze: Mapping[str, Any], *, candidate_id: str, lineage_id: str
) -> None:
    valid = (
        freeze.get("candidate_id") == candidate_id
        and freeze.get("lineage_id") == lineage_id
        and freeze.get("all_gates_passed") is True
        and freeze.get("candidate_freeze_allowed") is True
        and freeze.get("promotion_evidence_eligible") is False
        and freeze.get("live_shadow_start_allowed") is False
        and dict(freeze.get("safety") or {}) == SAFETY
    )
    if not valid:
        raise PooledResidualRuntimeError("candidate is not OOF-gated and frozen")


def _verify_descriptor(value: Any, root: Path, name: str) -> Path:
    descriptor = dict(value or {})
    relative = descriptor.get("path")
    expected = descriptor.get("sha256")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise PooledResidualRuntimeError(f"{name} path is not repository-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PooledResidualRuntimeError(f"{name} artifact is unavailable")
    if sha256_file(path) != _require_sha256(expected, f"{name} SHA-256"):
        raise PooledResidualRuntimeError(f"{name} SHA-256 mismatch")
    return path


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PooledResidualRuntimeError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PooledResidualRuntimeError(f"{name} must be a JSON object")
    return value


def _positive_int(value: Any, name: str, blockers: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        blockers.append(f"freshness_contract.{name}")
        return 0
    return value


def _require_sha256(value: Any, name: str) -> str:
    normalized = str(value or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise PooledResidualRuntimeError(f"{name} is invalid")
    return normalized


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ACTIONS",
    "BUNDLE_SCHEMA_VERSION",
    "PooledResidualRuntime",
    "PooledResidualRuntimeError",
    "RUNTIME_RESULT_SCHEMA_VERSION",
    "load_pooled_residual_runtime",
]
