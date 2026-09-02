"""Zero-capital rollback drill for the frozen residual promotion runtime."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    IMPLEMENTATION_PATH,
    LINEAGE_ID,
    _runtime_fixture_from_public_rows,
    load_residual_promotion_runtime,
)

ROLLBACK_DRILL_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-zero-capital-rollback-drill-v1"
)
CONFIG_REPOSITORY_PATH = (
    "examples/v8/polymarket_configs/"
    "BTC-15M-cost-aware-market-residual-promotion-v1"
)
BUNDLE_PATH = f"{CONFIG_REPOSITORY_PATH}/candidate_bundle/bundle_manifest.json"
PARITY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/candidate_bundle/offline_live_parity_report.json"
)
SOURCE_DATASET_PATH = (
    "examples/v8/polymarket_configs/"
    "BTC-15M-cost-aware-market-residual-v4/"
    "residual_v4_challenger_slot_002_oof/"
    "residual_v4_stacking_development_dataset_rows.jsonl"
)


def run_zero_capital_rollback_drill(
    *,
    repository_root: Path | str,
    output_path: Path | str,
    created_at: str,
) -> dict[str, Any]:
    """Exercise deterministic NO_TRADE rollback paths without outcomes or writes."""

    root = Path(repository_root).resolve()
    bundle_path = _repository_file(BUNDLE_PATH, root)
    parity_path = _repository_file(PARITY_PATH, root)
    implementation_path = _repository_file(root / IMPLEMENTATION_PATH, root)
    _verify_sidecar(bundle_path)
    _verify_sidecar(parity_path)
    runtime = load_residual_promotion_runtime(
        manifest_path=bundle_path,
        expected_manifest_sha256=sha256_file(bundle_path),
        repository_root=root,
    )
    public_rows = _load_jsonl(_repository_file(SOURCE_DATASET_PATH, root))
    fixture = _runtime_fixture_from_public_rows(public_rows[:2])
    feature_row = dict(fixture["live_feature_row"])
    observed_at_ts = int(fixture["observed_at_ts"])
    healthy = runtime.score_feature_row(feature_row, observed_at_ts=observed_at_ts)
    if healthy.get("fail_closed") is not False or healthy.get("model_scored") is not True:
        raise ValueError("healthy runtime fixture did not score")

    cases = {
        "market_identity_guard": _mutate(feature_row, market_family="not_btc_15m"),
        "decision_staleness_guard": copy.deepcopy(feature_row),
        "source_staleness_guard": _mutate(
            feature_row,
            max_input_ts=(
                int(feature_row["decision_ts"]) - runtime.maximum_source_age_ms - 1
            ),
        ),
        "causality_guard": _mutate(
            feature_row, max_input_ts=int(feature_row["decision_ts"]) + 1
        ),
        "missing_input_guard": _without_nested_feature(feature_row, "up_ask"),
    }
    observations: list[dict[str, Any]] = []
    for name, row in cases.items():
        case_observed_at = observed_at_ts
        if name == "decision_staleness_guard":
            case_observed_at += runtime.maximum_decision_lag_ms + 1
        result = runtime.score_feature_row(row, observed_at_ts=case_observed_at)
        passed = bool(
            result.get("selected_action") == "NO_TRADE"
            and result.get("fail_closed") is True
            and result.get("model_scored") is False
            and result.get("action_values")
            == {
                "NO_TRADE": 0.0,
                "BUY_UP_HOLD": None,
                "BUY_DOWN_HOLD": None,
            }
            and _runtime_safety_is_closed(result)
        )
        observations.append(
            {
                "case": name,
                "passed": passed,
                "selected_action": result.get("selected_action"),
                "model_scored": result.get("model_scored"),
                "fail_closed": result.get("fail_closed"),
                "fail_closed_reasons": list(result.get("fail_closed_reasons") or []),
            }
        )

    recovered = runtime.score_feature_row(feature_row, observed_at_ts=observed_at_ts)
    recovery_passed = _projection(recovered) == _projection(healthy)
    technical_passed = all(item["passed"] for item in observations) and recovery_passed
    if not technical_passed:
        raise ValueError("zero-capital rollback drill failed closed")
    report = {
        "schema_version": ROLLBACK_DRILL_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "candidate_bundle": _descriptor(bundle_path, root),
        "runtime_implementation": _descriptor(implementation_path, root),
        "frozen_runtime_parity": _descriptor(parity_path, root),
        "development_fixture_sha256": canonical_json_sha256(fixture),
        "development_fixture_only": True,
        "fresh_population_used": False,
        "rollback_target": "NO_TRADE",
        "rollback_cases": observations,
        "healthy_projection_sha256": canonical_json_sha256(_projection(healthy)),
        "recovered_projection_sha256": canonical_json_sha256(_projection(recovered)),
        "deterministic_recovery_passed": recovery_passed,
        "technical_rollback_drill_passed": technical_passed,
        "fresh_confirmation_passed": False,
        "phase6_passed": False,
        "ready_to_request_micro_live_approval": False,
        "micro_live_authorized": False,
        "automatic_live_unlock": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "wallet_signing_attempted": False,
        "polymarket_write_attempted": False,
        "capital_exposed": False,
        "safety": dict(SAFETY),
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    output.write_bytes(raw)
    output.with_suffix(output.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )
    return report


def _runtime_safety_is_closed(result: Mapping[str, Any]) -> bool:
    return bool(
        result.get("outcomes_accessed") is False
        and result.get("settlement_accessed") is False
        and result.get("pnl_accessed") is False
        and result.get("wallet_signing_allowed") is False
        and result.get("polymarket_write_allowed") is False
        and result.get("capital_at_risk") is False
        and dict(result.get("safety") or {}) == SAFETY
    )


def _mutate(row: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    output = copy.deepcopy(dict(row))
    output.update(changes)
    return output


def _without_nested_feature(row: Mapping[str, Any], name: str) -> dict[str, Any]:
    output = copy.deepcopy(dict(row))
    del output["features"][name]
    return output


def _projection(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_values": dict(result.get("action_values") or {}),
        "probabilities": result.get("probabilities"),
        "selected_action": result.get("selected_action"),
        "model_scored": result.get("model_scored"),
        "fail_closed": result.get("fail_closed"),
    }


def _repository_file(path: Path | str, repository_root: Path) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repository_root / candidate).resolve()
    )
    if not resolved.is_relative_to(repository_root) or not resolved.is_file():
        raise ValueError("repository artifact is missing or escaped repository root")
    return resolved


def _descriptor(path: Path, repository_root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(repository_root).as_posix(),
        "sha256": sha256_file(path),
    }


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path)
    ):
        raise ValueError("frozen artifact sidecar mismatch")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = ["run_zero_capital_rollback_drill"]
