"""Outcome-blind shadow-stability evidence for residual promotion v1."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_collection import (
    assert_outcome_blind,
    verify_attempt_chain,
)
from bigan.v8.polymarket.residual_promotion_finalization import (
    validate_frozen_population,
)
from bigan.v8.polymarket.residual_promotion_release_readiness import (
    FUNCTIONAL_ROLLBACK_REPOSITORY_PATH,
    MAX_ROLLBACK_LATENCY_MS,
    OPERATIONAL_ROLLBACK_SCHEMA_VERSION,
    SHADOW_SCHEMA_VERSION,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    MAXIMUM_ATTEMPTS,
    TARGET_MARKETS,
    _runtime_fixture_from_public_rows,
    load_residual_promotion_runtime,
)

IMPLEMENTATION_REPOSITORY_PATH = "src/bigan/v8/polymarket/residual_promotion_release_evidence.py"
CLI_REPOSITORY_PATH = "examples/v8/run_residual_promotion_shadow_stability.py"
OPERATIONAL_ROLLBACK_CLI_REPOSITORY_PATH = (
    "examples/v8/run_residual_promotion_operational_rollback.py"
)
SOURCE_DATASET_REPOSITORY_PATH = (
    "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v4/"
    "residual_v4_challenger_slot_002_oof/"
    "residual_v4_stacking_development_dataset_rows.jsonl"
)
CONFIG_REPOSITORY_PATH = (
    "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
)
BUNDLE_REPOSITORY_PATH = f"{CONFIG_REPOSITORY_PATH}/candidate_bundle/bundle_manifest.json"


def build_outcome_blind_shadow_stability_report(
    *,
    repository_root: Path | str,
    service_root: Path | str,
    freeze_dir: Path | str,
    expected_population_manifest_sha256: str,
    output_path: Path | str,
    created_at: str,
    target_market_count: int = TARGET_MARKETS,
    validation_fixture_only: bool = False,
) -> dict[str, Any]:
    """Freeze exact shadow evidence without reading outcomes, settlement, or PnL."""

    if validation_fixture_only is not (target_market_count != TARGET_MARKETS):
        raise ValueError("non-production shadow target requires validation_fixture_only")
    repo = Path(repository_root).resolve()
    root = Path(service_root).resolve()
    freeze = Path(freeze_dir).resolve()
    output = Path(output_path).resolve()
    if not freeze.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("shadow evidence paths must remain inside service root")
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError("shadow stability report already exists")
    validation = validate_frozen_population(
        freeze_dir=freeze,
        service_root=root,
        repository_root=repo,
        expected_manifest_sha256=expected_population_manifest_sha256,
        target_market_count=target_market_count,
        validation_fixture_only=validation_fixture_only,
    )
    if validation.get("validation_passed") is not True:
        raise ValueError("exact population validation did not pass")
    manifest_path = freeze / "exact_population_manifest.json"
    manifest = _verified_json(manifest_path)
    if sha256_file(manifest_path) != expected_population_manifest_sha256:
        raise ValueError("exact population manifest SHA-256 mismatch")
    artifacts = dict(manifest.get("artifacts") or {})
    population = _verified_freeze_rows(freeze, dict(artifacts.get("population_rows") or {}))
    candidate = _verified_freeze_rows(freeze, dict(artifacts.get("candidate_decision_rows") or {}))
    baseline = _verified_freeze_rows(freeze, dict(artifacts.get("baseline_decision_rows") or {}))
    market_ids = _reconcile_ordered_rows(
        population=population,
        candidate=candidate,
        baseline=baseline,
        target_market_count=target_market_count,
    )
    if manifest.get("ordered_market_ids_sha256") != canonical_json_sha256(market_ids):
        raise ValueError("shadow ordered market identity SHA-256 mismatch")
    attempts = _load_jsonl(root / "outcome_blind_attempts.jsonl")
    verify_attempt_chain(attempts)
    for attempt in attempts:
        assert_outcome_blind(attempt)
    progress = _load_json(root / "collection_progress.json")
    assert_outcome_blind(progress)
    _validate_complete_progress(
        progress,
        attempt_count=len(attempts),
        target_market_count=target_market_count,
        validation_fixture_only=validation_fixture_only,
    )
    parity_descriptor = _verified_repository_descriptor(repo, manifest.get("runtime_parity_report"))
    parity = _verified_json(repo / parity_descriptor["path"])
    rollback_descriptor = _repository_descriptor(repo, FUNCTIONAL_ROLLBACK_REPOSITORY_PATH)
    rollback = _verified_json(repo / rollback_descriptor["path"])
    implementation = _repository_descriptor(repo, IMPLEMENTATION_REPOSITORY_PATH)
    cli = _repository_descriptor(repo, CLI_REPOSITORY_PATH)
    production_complete = not validation_fixture_only and target_market_count == TARGET_MARKETS
    runtime_parity_passed = bool(
        parity.get("prediction_and_decision_parity") is True
        and parity.get("fresh_outcomes_accessed") is False
        and dict(parity.get("safety") or {}) == SAFETY
    )
    rollback_wired = bool(
        rollback.get("technical_rollback_drill_passed") is True
        and rollback.get("rollback_target") == "NO_TRADE"
        and dict(rollback.get("safety") or {}) == SAFETY
    )
    if not runtime_parity_passed or not rollback_wired:
        raise ValueError("static parity or fail-closed rollback evidence is invalid")
    report = {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "validation_fixture_only": validation_fixture_only,
        "candidate_bundle_sha256": dict(manifest["candidate_bundle"])["sha256"],
        "population_manifest_sha256": expected_population_manifest_sha256,
        "population_manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": expected_population_manifest_sha256,
        },
        "ordered_market_ids_sha256": str(manifest["ordered_market_ids_sha256"]),
        "candidate_row_count": len(candidate),
        "baseline_row_count": len(baseline),
        "paired_row_count": len(market_ids),
        "attempt_count": len(attempts),
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "attempt_ledger": {
            "path": "outcome_blind_attempts.jsonl",
            "sha256": sha256_file(root / "outcome_blind_attempts.jsonl"),
        },
        "collection_progress": {
            "path": "collection_progress.json",
            "sha256": sha256_file(root / "collection_progress.json"),
        },
        "runtime_parity_report": parity_descriptor,
        "runtime_decision_parity_passed": True,
        "functional_rollback_report": rollback_descriptor,
        "monitoring_enabled": True,
        "kill_switch_wired": True,
        "kill_switch_evidence": {
            "runtime_fail_closed_target": "NO_TRADE",
            "collector_stops_at_target_or_attempt_cap": True,
            "wallet_write_and_capital_paths_absent": True,
        },
        "hash_chain_status": "valid",
        "collection_complete": True,
        "zero_capital_read_only": True,
        "shadow_stability_passed": production_complete,
        "collection_population_changed": False,
        "outcomes_accessed_during_collection": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "promotion_evidence_eligible": False,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "implementation": implementation,
        "cli": cli,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, report)
    return report


def run_outcome_blind_operational_rollback_drill(
    *,
    repository_root: Path | str,
    output_path: Path | str,
    created_at: str,
    iterations: int = 5,
    clock_ns: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Measure fail-closed runtime rollback without fresh data or execution access."""

    if iterations <= 0:
        raise ValueError("operational rollback iterations must be positive")
    repo = Path(repository_root).resolve()
    output = Path(output_path).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError("operational rollback report already exists")
    bundle_path = _repository_file(repo, BUNDLE_REPOSITORY_PATH)
    _verify_sidecar(bundle_path)
    runtime = load_residual_promotion_runtime(
        manifest_path=bundle_path,
        expected_manifest_sha256=sha256_file(bundle_path),
        repository_root=repo,
    )
    development_rows = _load_jsonl(_repository_file(repo, SOURCE_DATASET_REPOSITORY_PATH))
    fixture = _runtime_fixture_from_public_rows(development_rows[:2])
    feature_row = dict(fixture["live_feature_row"])
    observed_at_ts = int(fixture["observed_at_ts"])
    healthy = runtime.score_feature_row(feature_row, observed_at_ts=observed_at_ts)
    if healthy.get("model_scored") is not True or healthy.get("fail_closed") is not False:
        raise ValueError("operational rollback healthy fixture did not score")
    rollback_row = copy.deepcopy(feature_row)
    rollback_row["market_family"] = "rollback_to_no_trade"
    clock = clock_ns or time.perf_counter_ns
    latencies: list[float] = []
    observations: list[dict[str, Any]] = []
    for index in range(1, iterations + 1):
        started = clock()
        result = runtime.score_feature_row(rollback_row, observed_at_ts=observed_at_ts)
        ended = clock()
        if ended < started:
            raise ValueError("operational rollback clock moved backwards")
        latency_ms = max(0.0, (ended - started) / 1_000_000.0)
        passed = bool(
            result.get("selected_action") == "NO_TRADE"
            and result.get("model_scored") is False
            and result.get("fail_closed") is True
            and result.get("wallet_signing_allowed") is False
            and result.get("polymarket_write_allowed") is False
            and result.get("capital_at_risk") is False
        )
        latencies.append(latency_ms)
        observations.append(
            {
                "iteration": index,
                "latency_ms": latency_ms,
                "rollback_to_no_trade_passed": passed,
                "fail_closed_reasons": list(result.get("fail_closed_reasons") or []),
            }
        )
    maximum = max(latencies)
    rollback_passed = bool(
        all(row["rollback_to_no_trade_passed"] for row in observations)
        and maximum <= MAX_ROLLBACK_LATENCY_MS
    )
    functional = _repository_descriptor(repo, FUNCTIONAL_ROLLBACK_REPOSITORY_PATH)
    safe_parameters = {"action": "NO_TRADE", "capital_fraction": 0.0}
    report = {
        "schema_version": OPERATIONAL_ROLLBACK_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "candidate_bundle_sha256": sha256_file(bundle_path),
        "functional_rollback_report_sha256": functional["sha256"],
        "functional_rollback_report": functional,
        "implementation": _repository_descriptor(repo, IMPLEMENTATION_REPOSITORY_PATH),
        "cli": _repository_descriptor(repo, OPERATIONAL_ROLLBACK_CLI_REPOSITORY_PATH),
        "development_fixture_only": True,
        "fresh_population_used": False,
        "development_fixture_sha256": canonical_json_sha256(fixture),
        "rollback_target": "NO_TRADE",
        "safe_parameters": safe_parameters,
        "safe_parameters_sha256": canonical_json_sha256(safe_parameters),
        "latency_measurements_ms": latencies,
        "maximum_observed_latency_ms": maximum,
        "maximum_allowed_latency_ms": MAX_ROLLBACK_LATENCY_MS,
        "observations": observations,
        "rollback_drill_passed": rollback_passed,
        "phase6_zero_capital_authorized": False,
        "micro_live_authorized": False,
        "development_outcomes_accessed": True,
        "development_outcome_scope": "development_only_forever",
        "fresh_outcomes_accessed": False,
        "outcomes_accessed": True,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, report)
    if not rollback_passed:
        raise ValueError("operational rollback drill failed closed")
    return report


def _validate_complete_progress(
    progress: Mapping[str, Any],
    *,
    attempt_count: int,
    target_market_count: int,
    validation_fixture_only: bool,
) -> None:
    attempted = int(progress.get("attempts_consumed") or 0)
    if not (
        progress.get("lineage_id") == LINEAGE_ID
        and progress.get("candidate_id") == CANDIDATE_ID
        and progress.get("quality_valid_market_count") == target_market_count
        and progress.get("target_quality_valid_market_count") == target_market_count
        and progress.get("remaining_quality_valid_markets") == 0
        and progress.get("collection_complete") is True
        and progress.get("attempt_cap_exhausted") is (attempted == MAXIMUM_ATTEMPTS)
        and attempted == attempt_count
        and 0 < attempted <= MAXIMUM_ATTEMPTS
        and progress.get("hash_chain_status") == "valid"
        and progress.get("fresh_outcomes_opened") is False
        and progress.get("interim_pnl_evaluated") is False
        and progress.get("collection_influenced_by_model_decisions") is False
        and progress.get("wallet_signing_allowed") is False
        and progress.get("polymarket_write_allowed") is False
        and progress.get("capital_at_risk") is False
        and dict(progress.get("safety") or {}) == SAFETY
    ):
        raise ValueError("collection completion record is invalid")
    if not validation_fixture_only and progress.get("maximum_attempts") != MAXIMUM_ATTEMPTS:
        raise ValueError("production collection attempt cap changed")


def _reconcile_ordered_rows(
    *,
    population: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    target_market_count: int,
) -> list[str]:
    if not len(population) == len(candidate) == len(baseline) == target_market_count:
        raise ValueError("shadow population row count mismatch")
    population_ids = [str(row.get("market_id") or "") for row in population]
    candidate_ids = [str(row.get("market_id") or "") for row in candidate]
    baseline_ids = [str(row.get("market_id") or "") for row in baseline]
    if not (
        population_ids == candidate_ids == baseline_ids
        and len(set(population_ids)) == target_market_count
        and all(
            int(row.get("population_position") or 0) == index
            for index, row in enumerate(population, start=1)
        )
        and all(
            int(row.get("population_position") or 0) == index
            for index, row in enumerate(candidate, start=1)
        )
        and all(
            int(row.get("population_position") or 0) == index
            for index, row in enumerate(baseline, start=1)
        )
    ):
        raise ValueError("shadow population identity/order mismatch")
    return population_ids


def _verified_repository_descriptor(repository_root: Path, value: Any) -> dict[str, str]:
    descriptor = dict(value or {})
    if set(descriptor) != {"path", "sha256"}:
        raise ValueError("repository descriptor is invalid")
    path = _repository_file(repository_root, str(descriptor["path"]))
    if sha256_file(path) != descriptor["sha256"]:
        raise ValueError("repository descriptor SHA-256 mismatch")
    return {"path": str(descriptor["path"]), "sha256": str(descriptor["sha256"])}


def _repository_descriptor(repository_root: Path, relative_path: str) -> dict[str, str]:
    path = _repository_file(repository_root, relative_path)
    return {"path": relative_path, "sha256": sha256_file(path)}


def _repository_file(root: Path, path: Path | str) -> Path:
    value = Path(path)
    resolved = value.resolve() if value.is_absolute() else (root / value).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("repository artifact is missing or escaped root")
    return resolved


def _verified_freeze_rows(freeze: Path, descriptor: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = (freeze / str(descriptor.get("path") or "")).resolve()
    if (
        not path.is_relative_to(freeze)
        or not path.is_file()
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise ValueError("frozen population child SHA-256 mismatch")
    _verify_sidecar(path)
    return _load_jsonl(path)


def _verified_json(path: Path) -> dict[str, Any]:
    _verify_sidecar(path)
    return _load_json(path)


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != sha256_file(path):
        raise ValueError("frozen artifact sidecar mismatch")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )


__all__ = [
    "build_outcome_blind_shadow_stability_report",
    "run_outcome_blind_operational_rollback_drill",
]
