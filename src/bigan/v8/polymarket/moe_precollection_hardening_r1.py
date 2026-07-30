"""Harden the BTC 15m MoE v2 package without granting collection authority."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import (
    _parse_pytest_junit,
    _parse_ruff_json,
)
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

LINEAGE_ID = "BTC-15M-MoE-confirmatory-v2"
REVISION_ID = "BTC-15M-MoE-confirmatory-v2-precollection-r1"
BASE_COMMIT = "a37320d4f8aa8ccb8176132def6429c1127758ba"
REVIEWED_HEAD_COMMIT = "ad8d17037552ccd4147f5f249476251ae22a6166"
CREATED_AT = "2026-07-30T12:00:00+00:00"
CANDIDATE_BUNDLE_HASH = (
    "fa6b1429e22b26a7aba32be264431ace0818a4cf613043e3f7e054a5c837b807"
)
TARGET_MARKET_COUNT = 800
REQUIRED_COMPLETION_PROBABILITY = 0.975
QUALITY_RATE_CONFIDENCE = 0.975
POWER_MONTE_CARLO_SEED = 26015
POWER_OUTER_SIMULATIONS = 1000
POWER_INNER_BOOTSTRAP_RESAMPLES = 1000

SAFETY = {
    "source_model_candidate_eligible": False,
    "freeze_ready": False,
    "promotion_evidence_eligible": False,
    "paper_candidate_allowed": False,
    "v8_execution_handoff_allowed": False,
    "#134_resume_allowed": False,
    "#146_start_allowed": False,
    "live_trading_allowed": False,
    "wallet_signing_allowed": False,
    "polymarket_write_allowed": False,
    "capital_at_risk": False,
}
STATE_BLOCKED = {
    "fresh_collection_authorized": False,
    "fresh_collection_started": False,
    "fresh_outcomes_opened": False,
}
FORBIDDEN_HEALTH_SNAPSHOT_FIELDS = {
    "settlement_outcome",
    "target",
    "realized_pnl",
    "model_prediction",
    "selected_side",
    "accepted",
    "router_route",
    "expert_id",
    "future_price",
    "future_return",
}
FORBIDDEN_COLLECTION_CONTROL_FIELDS = FORBIDDEN_HEALTH_SNAPSHOT_FIELDS | {
    "candidate_prediction",
    "baseline_prediction",
    "selection_score",
}
OLD_PACKAGE_HASHES = {
    "collection_quality_rate_analysis.json": (
        "6abe9f28c68f91944601a53d34b972e0de618a96c4292f66f0ee66d94da13097"
    ),
    "moe_confirmatory_collector_protocol.json": (
        "ddbdd51fe169f878e9cf0058163603e8899fbfce0818a328ed42095cca5d6312"
    ),
    "moe_confirmatory_protocol.json": (
        "e89c57201e3a92779ecdf2a9a0ee694832e6bef5b58333e87880f827f8ce2a80"
    ),
    "moe_fresh_collection_authorization_template.json": (
        "e00bd99e8719fc9775f56ee4cb661d3b3550fb307503399999d74384a83ef88d"
    ),
}


def build_precollection_hardening_r1(
    *,
    repository_root: Path | str | None = None,
    created_at: str = CREATED_AT,
) -> dict[str, Any]:
    """Build versioned r1 precollection artifacts without changing model bytes."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    v2_dir = (
        repo_root
        / "examples"
        / "v8"
        / "polymarket_configs"
        / LINEAGE_ID
    )
    graph = _load_verified_json(v2_dir / "moe_artifact_graph.json")
    if graph["bundle_hash"] != CANDIDATE_BUNDLE_HASH:
        raise ValueError("candidate bundle hash changed")
    bundle_dir = repo_root / graph["bundle_repo_path"]
    before_model_hashes = {
        path.name: sha256_file(path)
        for path in sorted(bundle_dir.glob("*.json"))
    }

    revision = _write_revision_record(v2_dir=v2_dir, created_at=created_at)
    snapshot = _write_health_snapshot(
        repo_root=repo_root,
        v2_dir=v2_dir,
        created_at=created_at,
    )
    quality = _write_quality_analysis_r1(
        v2_dir=v2_dir,
        snapshot=snapshot,
        created_at=created_at,
    )
    collector = _write_collector_protocol_r1(
        v2_dir=v2_dir,
        revision=revision,
        snapshot=snapshot,
        quality=quality,
        created_at=created_at,
    )
    power = _write_power_interpretation_r1(
        v2_dir=v2_dir,
        created_at=created_at,
    )
    protocol = _write_confirmatory_protocol_r1(
        v2_dir=v2_dir,
        revision=revision,
        snapshot=snapshot,
        quality=quality,
        collector=collector,
        power=power,
        created_at=created_at,
    )
    authorization = _write_authorization_template_r1(
        v2_dir=v2_dir,
        revision=revision,
        snapshot=snapshot,
        quality=quality,
        collector=collector,
        power=power,
        protocol=protocol,
        created_at=created_at,
    )

    after_model_hashes = {
        path.name: sha256_file(path)
        for path in sorted(bundle_dir.glob("*.json"))
    }
    if before_model_hashes != after_model_hashes:
        raise ValueError("frozen candidate model artifacts changed")
    return {
        "revision_record_sha256": revision["sha256"],
        "health_snapshot_sha256": snapshot["snapshot_sha256"],
        "health_manifest_sha256": snapshot["manifest_sha256"],
        "wilson_lower_bound": quality["payload"][
            "conservative_quality_rate_lower_bound"
        ],
        "attempt_cap": quality["payload"]["attempt_cap"],
        "collector_protocol_sha256": collector["sha256"],
        "power_interpretation_sha256": power["sha256"],
        "confirmatory_protocol_sha256": protocol["sha256"],
        "authorization_template_sha256": authorization["sha256"],
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "candidate_model_hashes_unchanged": True,
        **STATE_BLOCKED,
        "safety": dict(SAFETY),
    }


def wilson_lower_bound(
    *,
    success_count: int,
    attempt_count: int,
    confidence: float,
) -> float:
    """Return the deterministic one-sided Wilson lower confidence bound."""

    if not 0 <= success_count <= attempt_count or attempt_count <= 0:
        raise ValueError("invalid success/attempt counts")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    observed = success_count / attempt_count
    z_value = NormalDist().inv_cdf(confidence)
    denominator = 1.0 + (z_value * z_value) / attempt_count
    center = observed + (z_value * z_value) / (2.0 * attempt_count)
    margin = z_value * math.sqrt(
        observed * (1.0 - observed) / attempt_count
        + (z_value * z_value) / (4.0 * attempt_count * attempt_count)
    )
    return (center - margin) / denominator


def binomial_tail_probability(
    *,
    attempt_count: int,
    required_success_count: int,
    success_probability: float,
) -> float:
    """Compute P[Binomial(attempt_count, p) >= required] via log-sum-exp."""

    if attempt_count < 0 or required_success_count < 0:
        raise ValueError("counts must be non-negative")
    if not 0.0 <= success_probability <= 1.0:
        raise ValueError("success_probability must be in [0, 1]")
    if required_success_count == 0:
        return 1.0
    if required_success_count > attempt_count:
        return 0.0
    if success_probability == 0.0:
        return 0.0
    if success_probability == 1.0:
        return 1.0
    log_terms = [
        math.lgamma(attempt_count + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(attempt_count - successes + 1)
        + successes * math.log(success_probability)
        + (attempt_count - successes) * math.log1p(-success_probability)
        for successes in range(required_success_count, attempt_count + 1)
    ]
    maximum = max(log_terms)
    probability = math.exp(maximum) * math.fsum(
        math.exp(value - maximum) for value in log_terms
    )
    return min(1.0, max(0.0, probability))


def minimum_attempt_cap(
    *,
    target_quality_valid_market_count: int,
    conservative_quality_rate_lower_bound: float,
    required_completion_probability: float,
) -> tuple[int, float, float]:
    """Return the minimum cap and probabilities at cap and cap-minus-one."""

    if not 0.0 < required_completion_probability < 1.0:
        raise ValueError("required completion probability must be in (0, 1)")
    attempt_cap = target_quality_valid_market_count
    while True:
        probability = binomial_tail_probability(
            attempt_count=attempt_cap,
            required_success_count=target_quality_valid_market_count,
            success_probability=conservative_quality_rate_lower_bound,
        )
        if probability >= required_completion_probability:
            previous = binomial_tail_probability(
                attempt_count=attempt_cap - 1,
                required_success_count=target_quality_valid_market_count,
                success_probability=conservative_quality_rate_lower_bound,
            )
            return attempt_cap, probability, previous
        attempt_cap += 1


def validate_health_snapshot(
    *,
    snapshot_path: Path | str,
    manifest_path: Path | str,
) -> list[dict[str, Any]]:
    """Validate the vendored snapshot, sidecars, counts, schema, and hash chain."""

    snapshot = Path(snapshot_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    _verify_sidecar(snapshot)
    manifest = _load_verified_json(manifest_file)
    if sha256_file(snapshot) != manifest["snapshot_content_sha256"]:
        raise ValueError("health snapshot content SHA-256 mismatch")
    rows = [
        json.loads(line)
        for line in snapshot.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != manifest["snapshot_row_count"] != 120:
        raise ValueError("health snapshot row count mismatch")
    required = {
        "attempt_index",
        "attempt_id",
        "market_id_or_stable_hash",
        "attempt_ts",
        "quality_valid",
        "quality_failure_reason_codes",
        "provider_capture_complete",
        "paired_executable_ask_capture_attempted",
        "causality_checks_passed",
        "raw_evidence_manifest_sha256",
        "previous_snapshot_entry_sha256",
        "snapshot_entry_sha256",
    }
    previous = "0" * 64
    for expected_index, row in enumerate(rows, start=1):
        if set(row) != required:
            raise ValueError("health snapshot field set mismatch")
        if FORBIDDEN_HEALTH_SNAPSHOT_FIELDS & set(row):
            raise ValueError("forbidden health snapshot field")
        if row["attempt_index"] != expected_index:
            raise ValueError("health snapshot attempt index is not contiguous")
        if row["previous_snapshot_entry_sha256"] != previous:
            raise ValueError("health snapshot hash chain predecessor mismatch")
        unsigned = dict(row)
        entry_sha = unsigned.pop("snapshot_entry_sha256")
        if canonical_json_sha256(unsigned) != entry_sha:
            raise ValueError("health snapshot hash chain entry mismatch")
        previous = entry_sha
    if previous != manifest["snapshot_terminal_entry_sha256"]:
        raise ValueError("health snapshot terminal hash mismatch")
    valid_count = sum(bool(row["quality_valid"]) for row in rows)
    if len(rows) != 120 or valid_count != 113:
        raise ValueError("health snapshot must reconcile to 120 / 113")
    if manifest["attempted_market_count"] != 120:
        raise ValueError("manifest attempted count mismatch")
    if manifest["quality_valid_market_count"] != 113:
        raise ValueError("manifest quality-valid count mismatch")
    if manifest["outcomes_labels_or_pnl_read"] is not False:
        raise ValueError("health snapshot must remain outcome blind")
    if manifest["model_outputs_read"] is not False:
        raise ValueError("health snapshot must not use model outputs")
    return rows


def deterministic_exact_window(
    attempts: Sequence[Mapping[str, Any]],
    *,
    target_market_count: int = TARGET_MARKET_COUNT,
) -> list[dict[str, Any]]:
    """Select the chronological earliest exact quality-valid unique markets."""

    if target_market_count <= 0:
        raise ValueError("target market count must be positive")
    normalized = [dict(row) for row in attempts]
    for row in normalized:
        forbidden = _find_forbidden_keys(row, FORBIDDEN_COLLECTION_CONTROL_FIELDS)
        if forbidden:
            raise ValueError(
                "forbidden collection-control input: " + ",".join(forbidden)
            )
    eligible = [row for row in normalized if row.get("quality_valid") is True]
    market_ids = [str(row["market_id"]) for row in eligible]
    if len(set(market_ids)) != len(market_ids):
        raise ValueError("duplicate quality-valid market")
    ordered = sorted(
        eligible,
        key=lambda row: (
            int(row["market_start_ts"]),
            str(row["market_id"]),
            int(row["attempt_index"]),
        ),
    )
    if len(ordered) < target_market_count:
        raise ValueError("exact confirmatory window is incomplete")
    return ordered[:target_market_count]


def validate_exact_window(
    *,
    attempts: Sequence[Mapping[str, Any]],
    selected_markets: Sequence[Mapping[str, Any]],
    target_market_count: int = TARGET_MARKET_COUNT,
) -> None:
    """Fail closed unless selected markets are the deterministic exact window."""

    expected = deterministic_exact_window(
        attempts,
        target_market_count=target_market_count,
    )
    selected = [dict(row) for row in selected_markets]
    if len(selected) != target_market_count:
        raise ValueError("confirmatory window must contain exactly 800 markets")
    expected_identity = [
        (str(row["market_id"]), int(row["attempt_index"])) for row in expected
    ]
    selected_identity = [
        (str(row["market_id"]), int(row["attempt_index"])) for row in selected
    ]
    if selected_identity != expected_identity:
        raise ValueError("skipped, replaced, or out-of-order confirmatory market")


def validate_attempt_hash_chain(attempts: Sequence[Mapping[str, Any]]) -> None:
    """Validate an append-only attempt hash chain."""

    previous = "0" * 64
    for expected_index, raw in enumerate(attempts, start=1):
        row = dict(raw)
        if row.get("attempt_index") != expected_index:
            raise ValueError("attempt index is not contiguous")
        if row.get("previous_entry_sha256") != previous:
            raise ValueError("attempt predecessor hash mismatch")
        entry_sha = row.pop("entry_sha256", None)
        if not isinstance(entry_sha, str) or canonical_json_sha256(row) != entry_sha:
            raise ValueError("attempt entry hash mismatch")
        previous = entry_sha


def verify_raw_evidence_manifest_hash(
    *,
    manifest_path: Path | str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load one raw evidence manifest only after its pinned hash reconciles."""

    path = Path(manifest_path).resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError("raw evidence manifest SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("raw evidence manifest must be a JSON object")
    return payload


def validate_population_reconciliation(
    reconciliation: Mapping[str, Any],
) -> None:
    """Require the exact same 800-market population in every evaluation panel."""

    exact_count_fields = (
        "frozen_window_market_count",
        "reported_market_count",
        "candidate_market_row_count",
        "baseline_market_row_count",
        "paired_delta_market_row_count",
    )
    if any(int(reconciliation.get(field, -1)) != TARGET_MARKET_COUNT for field in exact_count_fields):
        raise ValueError("confirmatory population count must equal exactly 800")
    zero_fields = (
        "dropped_market_count",
        "duplicate_market_count",
        "out_of_window_market_count",
    )
    if any(int(reconciliation.get(field, -1)) != 0 for field in zero_fields):
        raise ValueError("confirmatory population reconciliation failed")


def assert_outcome_access_allowed(
    *,
    capture_manifest: Mapping[str, Any] | None,
    requested_market_ids: Sequence[str] | None = None,
) -> None:
    """Fail closed before partial or premature outcome access."""

    if capture_manifest is None:
        raise ValueError("exact-window capture manifest is required")
    manifest = dict(capture_manifest)
    required_true = (
        "capture_manifest_frozen",
        "decision_artifacts_frozen",
        "all_artifact_hashes_reconcile",
        "all_decisions_frozen",
    )
    if manifest.get("exact_market_count") != TARGET_MARKET_COUNT:
        raise ValueError("outcome access requires exactly 800 markets")
    if any(manifest.get(field) is not True for field in required_true):
        raise ValueError("outcome access boundary is not frozen")
    ordered = manifest.get("ordered_market_ids")
    if not isinstance(ordered, list) or len(ordered) != TARGET_MARKET_COUNT:
        raise ValueError("ordered exact-window market IDs are required")
    if canonical_json_sha256(ordered) != manifest.get("ordered_market_ids_sha256"):
        raise ValueError("ordered market ID hash mismatch")
    if requested_market_ids is not None and list(requested_market_ids) != ordered:
        raise ValueError("partial or reordered outcome opening is forbidden")


def empirical_bootstrap_lcb_crossing_power(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int = TARGET_MARKET_COUNT,
    seed: int = POWER_MONTE_CARLO_SEED,
    outer_simulations: int = POWER_OUTER_SIMULATIONS,
    inner_bootstrap_resamples: int = POWER_INNER_BOOTSTRAP_RESAMPLES,
    quantile: float = 0.025,
) -> dict[str, Any]:
    """Empirically validate LCB crossing using paired market-level resampling."""

    if not rows:
        raise ValueError("planning rows are required")
    candidate = np.asarray(
        [float(row["moe_unit_net_pnl"]) for row in rows],
        dtype=float,
    )
    baseline = np.asarray(
        [float(row["matched_global_baseline_proxy_unit_net_pnl"]) for row in rows],
        dtype=float,
    )
    delta = candidate - baseline
    recorded_delta = np.asarray(
        [float(row["paired_delta_unit_net_pnl"]) for row in rows],
        dtype=float,
    )
    if not np.array_equal(delta, recorded_delta):
        raise ValueError("paired planning delta does not reconcile exactly")
    rng = np.random.default_rng(seed)
    source_probability = np.full(len(rows), 1.0 / len(rows), dtype=float)
    candidate_crossings = 0
    delta_crossings = 0
    for _ in range(outer_simulations):
        outer_counts = rng.multinomial(sample_size, source_probability)
        empirical_probability = outer_counts / sample_size
        inner_counts = rng.multinomial(
            sample_size,
            empirical_probability,
            size=inner_bootstrap_resamples,
        )
        candidate_means = inner_counts @ candidate / sample_size
        delta_means = inner_counts @ delta / sample_size
        candidate_lcb = float(
            np.quantile(candidate_means, quantile, method="linear")
        )
        delta_lcb = float(np.quantile(delta_means, quantile, method="linear"))
        candidate_crossings += int(candidate_lcb > 0.0)
        delta_crossings += int(delta_lcb > 0.0)
    return {
        "seed": seed,
        "sample_size": sample_size,
        "source_market_count": len(rows),
        "outer_simulations": outer_simulations,
        "inner_bootstrap_resamples": inner_bootstrap_resamples,
        "bootstrap_quantile": quantile,
        "NO_TRADE_unit_net_pnl": 0.0,
        "market_level_paired_resampling": True,
        "candidate_and_baseline_use_identical_market_draws": True,
        "paired_delta_lcb_crossing_count": delta_crossings,
        "paired_delta_lcb_crossing_probability": (
            delta_crossings / outer_simulations
        ),
        "absolute_moe_lcb_crossing_count": candidate_crossings,
        "absolute_moe_lcb_crossing_probability": (
            candidate_crossings / outer_simulations
        ),
    }


def build_final_attestation_r1(
    *,
    repository_root: Path | str,
    base_pytest_junit_path: Path | str,
    head_pytest_junit_path: Path | str,
    base_ruff_json_path: Path | str,
    head_ruff_json_path: Path | str,
    executable_head_commit: str,
    created_at: str = CREATED_AT,
) -> dict[str, Any]:
    """Write Commit-B-only regression and final hardening attestations."""

    repo_root = Path(repository_root).resolve()
    v2_dir = (
        repo_root
        / "examples"
        / "v8"
        / "polymarket_configs"
        / LINEAGE_ID
    )
    base_junit_path = Path(base_pytest_junit_path).resolve()
    head_junit_path = Path(head_pytest_junit_path).resolve()
    base_ruff_path = Path(base_ruff_json_path).resolve()
    head_ruff_path = Path(head_ruff_json_path).resolve()
    base_pytest = _parse_pytest_junit(base_junit_path)
    head_pytest = _parse_pytest_junit(head_junit_path)
    base_ruff = _parse_ruff_json(base_ruff_path)
    head_ruff = _parse_ruff_json(head_ruff_path)
    base_by_node = {row["node_id"]: row for row in base_pytest["failures"]}
    head_by_node = {row["node_id"]: row for row in head_pytest["failures"]}
    base_nodes = set(base_by_node)
    head_nodes = set(head_by_node)
    added = sorted(head_nodes - base_nodes)
    removed = sorted(base_nodes - head_nodes)
    changed = sorted(
        node
        for node in base_nodes & head_nodes
        if base_by_node[node]["normalized_message_sha256"]
        != head_by_node[node]["normalized_message_sha256"]
    )
    unchanged = sorted((base_nodes & head_nodes) - set(changed))
    base_ruff_ids = {row["identity"] for row in base_ruff}
    head_ruff_ids = {row["identity"] for row in head_ruff}
    added_ruff = sorted(head_ruff_ids - base_ruff_ids)
    ledger_payload = {
        "schema_version": "bigan-btc-15m-moe-regression-failure-ledger-r1",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "base_commit": BASE_COMMIT,
        "executable_head_commit": executable_head_commit,
        "attestation_commit": None,
        "commands": {
            "pytest": "PYTHONPATH=src:. python -m pytest tests/v8 -q",
            "ruff": (
                "PYTHONPATH=src python -m ruff check "
                "src/bigan/v8/polymarket examples/v8 tests/v8"
            ),
        },
        "capture_artifact_hashes": {
            "base_pytest_junit_sha256": sha256_file(base_junit_path),
            "head_pytest_junit_sha256": sha256_file(head_junit_path),
            "base_ruff_json_sha256": sha256_file(base_ruff_path),
            "head_ruff_json_sha256": sha256_file(head_ruff_path),
        },
        "base_pytest": base_pytest,
        "head_pytest": head_pytest,
        "pytest_reconciliation": {
            "base_failure_node_ids": sorted(base_nodes),
            "head_failure_node_ids": sorted(head_nodes),
            "added_failure_node_ids": added,
            "removed_failure_node_ids": removed,
            "unchanged_failure_node_ids": unchanged,
            "changed_message_failure_node_ids": changed,
            "new_test_failure_count": len(added) + len(changed),
            "head_failures_subset_of_base_failures": not added and not changed,
        },
        "base_ruff_errors": base_ruff,
        "head_ruff_errors": head_ruff,
        "ruff_reconciliation": {
            "added_error_identities": added_ruff,
            "removed_error_identities": sorted(base_ruff_ids - head_ruff_ids),
            "unchanged_error_identities": sorted(base_ruff_ids & head_ruff_ids),
            "new_ruff_error_count": len(added_ruff),
        },
        "required_condition_passed": (
            not added and not changed and not added_ruff
        ),
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    ledger_path = v2_dir / "regression_failure_ledger_r1.json"
    ledger = _write_new_frozen_json(ledger_path, ledger_payload)

    quality = _load_verified_json(v2_dir / "collection_quality_rate_analysis_r1.json")
    power = _load_verified_json(
        v2_dir / "moe_confirmatory_power_interpretation_r1.json"
    )
    final_payload = {
        "schema_version": "bigan-btc-15m-moe-final-precollection-hardening-r1",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "base_commit": BASE_COMMIT,
        "executable_head_commit": executable_head_commit,
        "attestation_commit": None,
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "candidate_bundle_unchanged": True,
        "model_retraining_performed": False,
        "health_snapshot": _descriptor(
            v2_dir / "collection_attempt_health_snapshot.jsonl"
        ),
        "wilson_lower_bound": quality[
            "conservative_quality_rate_lower_bound"
        ],
        "required_completion_probability": quality[
            "required_window_completion_probability"
        ],
        "attempt_cap": quality["attempt_cap"],
        "completion_probability_at_attempt_cap": quality[
            "completion_probability_at_attempt_cap"
        ],
        "completion_probability_at_attempt_cap_minus_one": quality[
            "completion_probability_at_attempt_cap_minus_one"
        ],
        "collector_protocol": _descriptor(
            v2_dir / "moe_confirmatory_collector_protocol_r1.json"
        ),
        "confirmatory_protocol": _descriptor(
            v2_dir / "moe_confirmatory_protocol_r1.json"
        ),
        "authorization_template": _descriptor(
            v2_dir / "moe_fresh_collection_authorization_template_r1.json"
        ),
        "power_at_n800": {
            "normal_approximation": power[
                "normal_approximation_observed_effect_at_n800"
            ],
            "empirical_bootstrap": power[
                "empirical_paired_market_bootstrap_validation_at_n800"
            ],
        },
        "regression_failure_ledger": _descriptor(ledger["path"]),
        "regression_required_condition_passed": ledger_payload[
            "required_condition_passed"
        ],
        "collection_did_not_start": True,
        "fresh_outcome_opened": False,
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    final_path = v2_dir / "final_precollection_hardening_report.json"
    final_report = _write_new_frozen_json(final_path, final_payload)
    return {
        "ledger": ledger,
        "final_report": final_report,
        "required_condition_passed": ledger_payload[
            "required_condition_passed"
        ],
    }


def _write_revision_record(*, v2_dir: Path, created_at: str) -> dict[str, Any]:
    old_package = {}
    for name, expected_sha in OLD_PACKAGE_HASHES.items():
        path = v2_dir / name
        if sha256_file(path) != expected_sha:
            raise ValueError(f"historical artifact drift: {name}")
        old_package[name] = _descriptor(path)
    payload = {
        "schema_version": "bigan-btc-15m-moe-precollection-revision-r1",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "reviewed_through_commit": REVIEWED_HEAD_COMMIT,
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "candidate_bundle_unchanged": True,
        "model_retraining_performed": False,
        "supersession": {
            "old_attempt_cap": 905,
            "superseded_before_authorization": True,
            "old_package_may_not_authorize_collection": True,
            "reason": (
                "expectation_based_attempt_cap_replaced_by_a_minimum_"
                "completion_probability_cap_and_exact_window_contract"
            ),
        },
        "preserved_original_artifacts": old_package,
        "preserved_original_artifact_count": len(old_package),
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    return _write_new_frozen_json(
        v2_dir / "precollection_protocol_revision_record.json",
        payload,
    )


def _write_health_snapshot(
    *,
    repo_root: Path,
    v2_dir: Path,
    created_at: str,
) -> dict[str, Any]:
    lane_rel = Path(
        "examples/v8/polymarket_runs/"
        "challenge-model-development-btc-updown-15m-v1"
    )
    lane_root = repo_root / lane_rel
    health_path = lane_root / "development_lane_health_latest.json"
    capture_index_path = lane_root / "outcome_blind_capture_batch_index.jsonl"
    health = json.loads(health_path.read_text(encoding="utf-8"))
    source_rows = [
        json.loads(line)
        for line in capture_index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if health["cumulative"]["attempted_market_count"] != 120:
        raise ValueError("source health attempted count mismatch")
    if health["cumulative"]["quality_valid_market_count"] != 113:
        raise ValueError("source health quality-valid count mismatch")
    if health["outcomes_labels_or_pnl_read_for_health"] is not False:
        raise ValueError("source health is not outcome blind")
    if len(source_rows) != 30:
        raise ValueError("source capture index must contain 30 batches")
    previous_batch_sha = "0" * 64
    capture_rows: list[tuple[dict[str, Any], str, str]] = []
    batch_descriptors = []
    for expected_sequence, batch_entry in enumerate(source_rows, start=1):
        if batch_entry["sequence"] != expected_sequence:
            raise ValueError("source batch sequence mismatch")
        if batch_entry["previous_entry_sha256"] != previous_batch_sha:
            raise ValueError("source batch index chain mismatch")
        if batch_entry["capture_control_used_outcomes_labels_or_pnl"] is not False:
            raise ValueError("source capture batch used outcome information")
        previous_batch_sha = batch_entry["entry_sha256"]
        batch_path = (
            lane_root
            / "captures"
            / batch_entry["batch_id"]
            / "batch_summary.json"
        )
        if sha256_file(batch_path) != batch_entry["batch_summary_sha256"]:
            raise ValueError("source batch summary SHA-256 mismatch")
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        if batch["labels_or_outcomes_opened_during_collection"] is not False:
            raise ValueError("source batch opened labels or outcomes")
        if batch["settlement_pnl_opened_during_collection"] is not False:
            raise ValueError("source batch opened settlement PnL")
        if batch["resolution_provider_called"] is not False:
            raise ValueError("source batch called a resolution provider")
        if batch["training_corpus_export_attempted"] is not False:
            raise ValueError("source batch attempted a training export")
        if len(batch["captures"]) != batch_entry["capture_count"]:
            raise ValueError("source batch capture count mismatch")
        batch_descriptors.append(
            {
                "batch_id": batch_entry["batch_id"],
                "path": batch_path.relative_to(repo_root).as_posix(),
                "sha256": batch_entry["batch_summary_sha256"],
                "capture_count": batch_entry["capture_count"],
            }
        )
        for capture in batch["captures"]:
            manifest_path = (
                Path(capture["run_dir"]) / "pending_round_capture_manifest.json"
            )
            if not manifest_path.exists():
                manifest_path = (
                    lane_root
                    / "captures"
                    / Path(capture["run_dir"]).name
                    / "pending_round_capture_manifest.json"
                )
            capture_rows.append(
                (
                    capture,
                    sha256_file(manifest_path),
                    manifest_path.relative_to(repo_root).as_posix(),
                )
            )
    if len(capture_rows) != 120:
        raise ValueError("source captures must reconcile to 120 attempts")

    rows: list[dict[str, Any]] = []
    previous_entry = "0" * 64
    common_failure_reasons = {
        "book_causality_failed",
        "btc_candle_coverage_failed",
        "chainlink_capture_failed",
        "chainlink_causality_failed",
        "decision_row_count_not_2",
        "feature_rows_missing",
        "market_row_coverage_failed",
        "orderbook_full_window_coverage_failed",
        "paired_executable_ask_coverage_failed",
        "provider_orderbook_snapshot_coverage_failed",
    }
    raw_evidence_descriptors = []
    for attempt_index, (capture, manifest_sha, manifest_path) in enumerate(
        capture_rows,
        start=1,
    ):
        quality_valid = capture["capture_status"] == "pending_resolution"
        coverage = capture.get("orderbook_window_coverage_by_market") or {}
        if len(coverage) == 1:
            market_identity = str(next(iter(coverage)))
        else:
            stable = hashlib.sha256(
                str(capture["run_id"]).encode("utf-8")
            ).hexdigest()
            market_identity = f"sha256:{stable}"
        direct_reasons = set(capture.get("reject_reason_counts") or {})
        direct_reasons.update(capture.get("chainlink_capture_reason_codes") or [])
        direct_reasons.update(
            capture.get("feature_enrichment_reason_codes") or []
        )
        failure_reasons = (
            [] if quality_valid else sorted(direct_reasons | common_failure_reasons)
        )
        provider_complete = bool(
            capture.get("orderbook_full_window_coverage_passed")
            and capture.get("raw_chainlink_price_row_count", 0) > 0
            and capture.get("raw_polymarket_market_count") == 1
        )
        row = {
            "attempt_index": attempt_index,
            "attempt_id": str(capture["run_id"]),
            "market_id_or_stable_hash": market_identity,
            "attempt_ts": _timestamp_ms_to_iso(
                int(capture["scheduled_round_start_ts"])
            ),
            "quality_valid": quality_valid,
            "quality_failure_reason_codes": failure_reasons,
            "provider_capture_complete": provider_complete,
            "paired_executable_ask_capture_attempted": bool(
                capture.get("orderbook_full_window_coverage_required")
            ),
            "causality_checks_passed": bool(
                quality_valid
                and provider_complete
                and capture.get("capture_start_boundary_validation_passed")
            ),
            "raw_evidence_manifest_sha256": manifest_sha,
            "previous_snapshot_entry_sha256": previous_entry,
        }
        row["snapshot_entry_sha256"] = canonical_json_sha256(row)
        previous_entry = row["snapshot_entry_sha256"]
        rows.append(row)
        raw_evidence_descriptors.append(
            {
                "attempt_index": attempt_index,
                "path": manifest_path,
                "sha256": manifest_sha,
            }
        )
    if sum(bool(row["quality_valid"]) for row in rows) != 113:
        raise ValueError("derived snapshot does not reconcile to 113 valid attempts")

    snapshot_path = v2_dir / "collection_attempt_health_snapshot.jsonl"
    _write_new_jsonl(snapshot_path, rows)
    snapshot_sha = sha256_file(snapshot_path)
    module_path = Path(__file__).resolve()
    manifest_payload = {
        "schema_version": "bigan-btc-15m-attempt-health-manifest-r1",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "attempted_market_count": 120,
        "quality_valid_market_count": 113,
        "outcomes_labels_or_pnl_read": False,
        "model_outputs_read": False,
        "source_ledger_path": health_path.relative_to(repo_root).as_posix(),
        "source_ledger_sha256": sha256_file(health_path),
        "source_capture_index": _descriptor(capture_index_path),
        "source_capture_batch_count": len(batch_descriptors),
        "source_capture_batches": batch_descriptors,
        "source_fields_used": [
            "cumulative.attempted_market_count",
            "cumulative.quality_valid_market_count",
            "outcomes_labels_or_pnl_read_for_health",
            "capture_status",
            "capture_start_boundary_validation_passed",
            "orderbook_full_window_coverage_passed",
            "orderbook_full_window_coverage_required",
            "raw_chainlink_price_row_count",
            "raw_polymarket_market_count",
            "reject_reason_counts",
            "chainlink_capture_reason_codes",
            "feature_enrichment_reason_codes",
            "scheduled_round_start_ts",
            "run_id",
        ],
        "snapshot_derivation_code_path": module_path.relative_to(
            repo_root
        ).as_posix(),
        "snapshot_derivation_code_sha256": sha256_file(module_path),
        "snapshot_row_count": len(rows),
        "snapshot_path": snapshot_path.relative_to(repo_root).as_posix(),
        "snapshot_content_sha256": snapshot_sha,
        "snapshot_terminal_entry_sha256": previous_entry,
        "raw_evidence_manifest_count": len(raw_evidence_descriptors),
        "raw_evidence_manifests": raw_evidence_descriptors,
        "forbidden_snapshot_fields": sorted(
            FORBIDDEN_HEALTH_SNAPSHOT_FIELDS
        ),
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    manifest = _write_new_frozen_json(
        v2_dir / "collection_attempt_health_manifest.json",
        manifest_payload,
    )
    validate_health_snapshot(
        snapshot_path=snapshot_path,
        manifest_path=manifest["path"],
    )
    return {
        "snapshot_path": snapshot_path,
        "snapshot_sha256": snapshot_sha,
        "manifest_path": manifest["path"],
        "manifest_sha256": manifest["sha256"],
        "rows": rows,
    }


def _write_quality_analysis_r1(
    *,
    v2_dir: Path,
    snapshot: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    attempted = len(snapshot["rows"])
    valid = sum(bool(row["quality_valid"]) for row in snapshot["rows"])
    observed_rate = valid / attempted
    lower = wilson_lower_bound(
        success_count=valid,
        attempt_count=attempted,
        confidence=QUALITY_RATE_CONFIDENCE,
    )
    cap, at_cap, before_cap = minimum_attempt_cap(
        target_quality_valid_market_count=TARGET_MARKET_COUNT,
        conservative_quality_rate_lower_bound=lower,
        required_completion_probability=REQUIRED_COMPLETION_PROBABILITY,
    )
    payload = {
        "schema_version": "bigan-btc-15m-collection-quality-analysis-r1",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "role": "outcome_blind_completion_probability_attempt_cap",
        "target_quality_valid_market_count": TARGET_MARKET_COUNT,
        "attempted_market_count": attempted,
        "quality_valid_market_count": valid,
        "observed_quality_valid_rate": observed_rate,
        "quality_rate_confidence": QUALITY_RATE_CONFIDENCE,
        "quality_rate_method": "one_sided_Wilson_lower_bound",
        "quality_rate_z_value": NormalDist().inv_cdf(
            QUALITY_RATE_CONFIDENCE
        ),
        "conservative_quality_rate_lower_bound": lower,
        "required_window_completion_probability": (
            REQUIRED_COMPLETION_PROBABILITY
        ),
        "attempt_cap_method": (
            "wilson_lower_bound_plus_binomial_tail_quantile"
        ),
        "attempt_cap": cap,
        "completion_probability_at_attempt_cap": at_cap,
        "completion_probability_at_attempt_cap_minus_one": before_cap,
        "attempt_cap_is_minimal": (
            at_cap >= REQUIRED_COMPLETION_PROBABILITY
            and before_cap < REQUIRED_COMPLETION_PROBABILITY
        ),
        "completion_probability_sensitivity_at_observed_rate": (
            binomial_tail_probability(
                attempt_count=cap,
                required_success_count=TARGET_MARKET_COUNT,
                success_probability=observed_rate,
            )
        ),
        "completion_probability_sensitivity_with_rate_minus_0_02": (
            binomial_tail_probability(
                attempt_count=cap,
                required_success_count=TARGET_MARKET_COUNT,
                success_probability=observed_rate - 0.02,
            )
        ),
        "completion_probability_sensitivity_with_rate_minus_0_05": (
            binomial_tail_probability(
                attempt_count=cap,
                required_success_count=TARGET_MARKET_COUNT,
                success_probability=observed_rate - 0.05,
            )
        ),
        "invariants": {
            "completion_probability_at_attempt_cap_gte_0_975": (
                at_cap >= REQUIRED_COMPLETION_PROBABILITY
            ),
            "completion_probability_at_attempt_cap_minus_one_lt_0_975": (
                before_cap < REQUIRED_COMPLETION_PROBABILITY
            ),
            "attempt_cap_gte_target": cap >= TARGET_MARKET_COUNT,
        },
        "assumption_limitation": (
            "The binomial calculation assumes conditionally independent "
            "attempt validity and may be optimistic under clustered provider "
            "outages."
        ),
        "source_snapshot": _descriptor(snapshot["snapshot_path"]),
        "source_snapshot_manifest": _descriptor(snapshot["manifest_path"]),
        "outcomes_labels_or_pnl_read": False,
        "model_outputs_read": False,
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    if not payload["attempt_cap_is_minimal"] or not all(
        payload["invariants"].values()
    ):
        raise ValueError("attempt cap invariants failed")
    result = _write_new_frozen_json(
        v2_dir / "collection_quality_rate_analysis_r1.json",
        payload,
    )
    result["payload"] = payload
    return result


def _write_collector_protocol_r1(
    *,
    v2_dir: Path,
    revision: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    quality: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    cap = int(quality["payload"]["attempt_cap"])
    payload = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-collector-r1",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "population": {
            "definition": (
                "chronological_earliest_exact_800_quality_valid_unique_"
                "BTC_15m_markets_strictly_later_than_authorization_boundary"
            ),
            "target_quality_valid_market_count": TARGET_MARKET_COUNT,
            "required_final_quality_valid_market_count": TARGET_MARKET_COUNT,
            "more_than_800_markets_in_confirmatory_window_allowed": False,
            "less_than_800_markets_is_incomplete_failure": True,
            "maximum_attempts": cap,
        },
        "ordering": {
            "primary_order": "market_start_ts ascending",
            "tie_break_1": "market_id ascending",
            "tie_break_2": "attempt_index ascending",
        },
        "stopping": {
            "stop_window_selection_immediately_when_800th_market_is_accepted": True,
            "markets_after_exact_800_boundary_are_not_confirmatory_eligible": True,
            "post_boundary_markets_may_not_be_added": True,
            "market_replacement_after_outcome_access_allowed": False,
            "window_extension_allowed": False,
            "market_skip_allowed": False,
            "route_or_prediction_based_selection_allowed": False,
        },
        "required_raw_streams": [
            "raw_polymarket_markets.jsonl",
            "raw_polymarket_orderbooks.jsonl",
            "raw_polymarket_trades.jsonl",
            "raw_binance_btcusdt_klines.jsonl",
            "raw_polymarket_chainlink_prices.jsonl",
        ],
        "causality": {
            "available_at_ts_lte_decision_ts": True,
            "max_input_ts_lte_decision_ts": True,
            "feature_cutoff_ts_lte_decision_ts": True,
            "missing_not_numeric_zero": True,
            "complement_quote_proxy_forbidden": True,
        },
        "forbidden_collection_control_inputs": sorted(
            FORBIDDEN_COLLECTION_CONTROL_FIELDS
        ),
        "capture_stage_state": {
            "resolution_provider_enabled": False,
            "settlement_finalizer_enabled": False,
            "training_export_enabled": False,
            "outcome_access_enabled": False,
        },
        "append_only_audit": {
            "attempt_index_hash_chain_required": True,
            "failed_attempts_retained": True,
            "failure_reason_codes_required": True,
            "existing_attempt_rewrite_allowed": False,
        },
        "capture_manifest_required_before_outcome_access": {
            "path": "confirmatory_capture_manifest.json",
            "sha256_sidecar": "confirmatory_capture_manifest.sha256",
            "required_fields": [
                "exact_market_count",
                "ordered_market_ids",
                "ordered_market_ids_sha256",
                "first_market_start_ts",
                "last_market_start_ts",
                "attempts_consumed",
                "unused_attempt_capacity",
                "all_decisions_frozen",
                "candidate_decision_rows_sha256",
                "baseline_decision_rows_sha256",
                "raw_evidence_manifest_set_sha256",
            ],
            "partial_outcome_opening_forbidden": True,
        },
        "frozen_inputs": {
            "revision_record": _descriptor(revision["path"]),
            "attempt_health_snapshot": _descriptor(
                snapshot["snapshot_path"]
            ),
            "attempt_health_manifest": _descriptor(
                snapshot["manifest_path"]
            ),
            "attempt_cap_analysis": _descriptor(quality["path"]),
        },
        "state": {
            **STATE_BLOCKED,
            "collector_protocol_frozen": True,
            "outcome_access_enabled": False,
        },
        "safety": dict(SAFETY),
    }
    return _write_new_frozen_json(
        v2_dir / "moe_confirmatory_collector_protocol_r1.json",
        payload,
    )


def _write_power_interpretation_r1(
    *,
    v2_dir: Path,
    created_at: str,
) -> dict[str, Any]:
    original = _load_verified_json(
        v2_dir / "moe_confirmatory_power_analysis.json"
    )
    if original["selected_confirmatory_market_count"] != TARGET_MARKET_COUNT:
        raise ValueError("frozen 800-market power target changed")
    normal_rows = [
        row
        for row in original["core_effect_and_variance_sensitivity"]
        if row["sample_size"] == TARGET_MARKET_COUNT
    ]
    observed_normal = next(
        row
        for row in normal_rows
        if row["scenario_name"] == "observed_development_effect"
        and row["variance_multiplier"] == 1.0
    )
    design_normal = next(
        row
        for row in normal_rows
        if row["scenario_name"] == "observed_development_effect"
        and row["variance_multiplier"] == 1.25
    )
    report_only = [
        row
        for row in normal_rows
        if row["scenario_name"]
        in {"75pct_of_observed_effect", "50pct_of_observed_effect"}
    ]
    planning_rows_path = v2_dir / "development_paired_planning_rows.jsonl"
    planning_rows = [
        json.loads(line)
        for line in planning_rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    empirical = empirical_bootstrap_lcb_crossing_power(planning_rows)
    payload = {
        "schema_version": "bigan-btc-15m-moe-power-interpretation-r1",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "created_at": created_at,
        "selected_confirmatory_market_count": TARGET_MARKET_COUNT,
        "selected_target_unchanged": True,
        "design_criterion": {
            "effect_size_scenario": "observed_development_effect",
            "variance_multiplier": 1.25,
            "selected_sample_size": TARGET_MARKET_COUNT,
            "normal_approximation": design_normal,
        },
        "primary_and_absolute_lcb_design_ready": True,
        "overall_all_gate_success_probability_estimated": False,
        "overall_all_gate_success_probability_not_guaranteed": True,
        "winner_selection_bias_possible": True,
        "development_effect_may_be_optimistic": True,
        "normal_approximation_observed_effect_at_n800": observed_normal,
        "report_only_75pct_and_50pct_effect_power_at_n800": report_only,
        "report_only_sensitivities_are_not_hard_design_selection_scenarios": True,
        "empirical_paired_market_bootstrap_validation_at_n800": empirical,
        "empirical_validation_role": (
            "validation_of_normal_approximation_only"
        ),
        "empirical_validation_may_change_frozen_target": False,
        "source_power_analysis": _descriptor(
            v2_dir / "moe_confirmatory_power_analysis.json"
        ),
        "source_planning_rows": _descriptor(planning_rows_path),
        "development_evidence_role": "planning_only_not_promotion_evidence",
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    return _write_new_frozen_json(
        v2_dir / "moe_confirmatory_power_interpretation_r1.json",
        payload,
    )


def _write_confirmatory_protocol_r1(
    *,
    v2_dir: Path,
    revision: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    quality: Mapping[str, Any],
    collector: Mapping[str, Any],
    power: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    original = _load_verified_json(v2_dir / "moe_confirmatory_protocol.json")
    gates = json.loads(json.dumps(original["gates"]))
    gates["quality_valid_market_count"] = {
        "operator": "eq",
        "value": TARGET_MARKET_COUNT,
    }
    payload = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-protocol-r1",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "candidate_id": original["candidate_id"],
        "created_at": created_at,
        "bootstrap": original["bootstrap"],
        "design": {
            **original["design"],
            "target_quality_valid_market_count": TARGET_MARKET_COUNT,
            "required_final_quality_valid_market_count": TARGET_MARKET_COUNT,
            "exact_population_required": True,
            "pooled_extended_replacement_or_post_hoc_population_may_rescue_failure": False,
        },
        "population_reconciliation": {
            "frozen_window_market_count": TARGET_MARKET_COUNT,
            "reported_market_count": TARGET_MARKET_COUNT,
            "candidate_market_row_count": TARGET_MARKET_COUNT,
            "baseline_market_row_count": TARGET_MARKET_COUNT,
            "paired_delta_market_row_count": TARGET_MARKET_COUNT,
            "dropped_market_count": 0,
            "duplicate_market_count": 0,
            "out_of_window_market_count": 0,
        },
        "gates": gates,
        "hard_gate_definitions_unchanged_except_exact_population_operator": True,
        "outcome_access_boundary": {
            "confirmatory_capture_manifest_required": True,
            "exact_market_count_must_equal": TARGET_MARKET_COUNT,
            "capture_manifest_frozen_must_equal": True,
            "decision_artifacts_frozen_must_equal": True,
            "all_artifact_hashes_reconcile_must_equal": True,
            "partial_outcome_opening_forbidden": True,
        },
        "frozen_inputs": {
            **original["frozen_inputs"],
            "revision_record": _descriptor(revision["path"]),
            "collector_protocol": _descriptor(collector["path"]),
            "attempt_health_snapshot": _descriptor(
                snapshot["snapshot_path"]
            ),
            "attempt_health_manifest": _descriptor(
                snapshot["manifest_path"]
            ),
            "attempt_cap_analysis": _descriptor(quality["path"]),
            "power_interpretation": _descriptor(power["path"]),
        },
        "state": {
            **STATE_BLOCKED,
            "confirmatory_evaluation_started": False,
            "protocol_frozen": True,
        },
        "safety": dict(SAFETY),
    }
    validate_population_reconciliation(payload["population_reconciliation"])
    return _write_new_frozen_json(
        v2_dir / "moe_confirmatory_protocol_r1.json",
        payload,
    )


def _write_authorization_template_r1(
    *,
    v2_dir: Path,
    revision: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    quality: Mapping[str, Any],
    collector: Mapping[str, Any],
    power: Mapping[str, Any],
    protocol: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    graph = _load_verified_json(v2_dir / "moe_artifact_graph.json")
    baseline = _load_verified_json(
        v2_dir / "moe_matched_global_baseline_contract.json"
    )
    cap = quality["payload"]["attempt_cap"]
    payload = {
        "schema_version": "bigan-btc-15m-moe-authorization-template-r1",
        "lineage_id": LINEAGE_ID,
        "revision_id": REVISION_ID,
        "candidate_id": "mixture_of_experts",
        "created_at": created_at,
        "artifact_role": "inactive_template_only_not_collection_authority",
        "template_usable_as_collection_authorization": False,
        "activation_placeholders": {
            "authorization_artifact_id": None,
            "authorized_by": None,
            "authorized_at": None,
            "authorization_source_url": None,
            "authorization_source_id": None,
            "authorization_request_sha256": None,
            "authorization_decision_sha256": None,
            "strictly_later_than_timestamp": None,
            "maximum_attempts": cap,
            "maximum_markets": TARGET_MARKET_COUNT,
            "explicit_request_received": False,
        },
        "frozen_inputs": {
            "candidate_artifact_graph": _descriptor(
                v2_dir / "moe_artifact_graph.json"
            ),
            "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
            "matched_baseline_contract": _descriptor(
                v2_dir / "moe_matched_global_baseline_contract.json"
            ),
            "matched_baseline_artifact_sha256": baseline["artifact"]["sha256"],
            "router_sha256": graph["artifacts"]["moe_router_contract.json"][
                "sha256"
            ],
            "feature_contract_sha256": graph["artifacts"][
                "moe_feature_contract.json"
            ]["sha256"],
            "cost_action_sha256": graph["artifacts"][
                "moe_cost_and_action_contract.json"
            ]["sha256"],
            "collector_protocol": _descriptor(collector["path"]),
            "reporting_contract": _descriptor(
                v2_dir / "moe_future_evaluation_reporting_contract.json"
            ),
            "statistical_protocol": _descriptor(protocol["path"]),
            "attempt_health_snapshot": _descriptor(
                snapshot["snapshot_path"]
            ),
            "attempt_health_manifest": _descriptor(
                snapshot["manifest_path"]
            ),
            "attempt_cap_analysis": _descriptor(quality["path"]),
            "power_interpretation": _descriptor(power["path"]),
            "revision_record": _descriptor(revision["path"]),
            "runtime_validation_report": _descriptor(
                v2_dir / "moe_artifact_runtime_validation_report.json"
            ),
        },
        "later_authorization_must_freeze": [
            "candidate_artifact_graph_SHA",
            "matched_baseline_SHA",
            "router_SHA",
            "feature_contract_SHA",
            "cost_action_SHA",
            "collector_protocol_SHA",
            "reporting_contract_SHA",
            "statistical_protocol_SHA",
            "attempt_health_snapshot_SHA",
            "attempt_cap_analysis_SHA",
            "strictly_later_than_timestamp",
            "maximum_attempts",
            "exact_target_markets_800",
        ],
        "old_905_attempt_template_non_authoritative": True,
        "state": dict(STATE_BLOCKED),
        "safety": dict(SAFETY),
    }
    return _write_new_frozen_json(
        v2_dir / "moe_fresh_collection_authorization_template_r1.json",
        payload,
    )


def _load_verified_json(path: Path) -> dict[str, Any]:
    _verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        raise ValueError(f"missing SHA-256 sidecar: {path}")
    expected = sidecar.read_text(encoding="utf-8").strip()
    actual = sha256_file(path)
    if expected != actual:
        raise ValueError(f"SHA-256 mismatch: {path}")


def _descriptor(path: Path | str) -> dict[str, str]:
    resolved = Path(path).resolve()
    repo_root = REPO_ROOT.resolve()
    return {
        "path": resolved.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _write_new_frozen_json(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists() or path.with_suffix(".sha256").exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    _atomic_write_text(
        path,
        json.dumps(
            dict(payload),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    digest = sha256_file(path)
    _atomic_write_text(path.with_suffix(".sha256"), digest + "\n")
    return {"path": path, "sha256": digest}


def _write_new_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if path.exists() or path.with_suffix(".sha256").exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    content = "".join(
        json.dumps(
            dict(row),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    _atomic_write_text(path, content)
    _atomic_write_text(path.with_suffix(".sha256"), sha256_file(path) + "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _timestamp_ms_to_iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1000.0, tz=UTC).isoformat()


def _find_forbidden_keys(
    payload: Any,
    forbidden: set[str],
    *,
    prefix: str = "",
) -> list[str]:
    found = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in forbidden:
                found.append(path)
            found.extend(
                _find_forbidden_keys(value, forbidden, prefix=path)
            )
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(
                _find_forbidden_keys(
                    value,
                    forbidden,
                    prefix=f"{prefix}[{index}]",
                )
            )
    return sorted(found)


def main() -> int:
    """Build the blocked r1 precollection package."""

    result = build_precollection_hardening_r1(repository_root=REPO_ROOT)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
