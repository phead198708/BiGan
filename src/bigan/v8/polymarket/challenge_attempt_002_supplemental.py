"""Build #257/#258/#256 evidence for a frozen attempt-002 future result."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.canonical_payload import canonical_payload_sha256
from bigan.v8.polymarket.challenge_attempt_002 import (
    CANDIDATE_ID,
    NO_TRADE,
    validate_attempt_002_preregistration,
)
from bigan.v8.polymarket.challenge_attempt_002_promotion import (
    _load_json,
    _load_jsonl,
    _validated_future_bundle,
    _verified_descriptor,
)
from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.challenge_promotion_evidence import (
    AGGREGATE_PARITY_SCHEMA_VERSION,
    AGGREGATE_RECONCILIATION_SCHEMA_VERSION,
    AGGREGATE_SAFETY_SCHEMA_VERSION,
    _aggregate_policy_report,
    _execution_policy_inputs,
    _provider_health_diagnostic_decisions,
    _regime_assignments,
    _run_all_execution_policies,
    _write_policy_replays,
)
from bigan.v8.polymarket.feature_completeness import (
    build_provider_health_diagnostics,
)
from bigan.v8.polymarket.regime_diagnostics import (
    DIMENSION_BUCKETS,
    REGIME_REPORT_SCHEMA_VERSION,
    build_regime_stratified_diagnostics,
    regime_diagnostics_markdown,
    validate_regime_definition_contract,
)

SUPPLEMENTAL_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-supplemental-evidence-manifest-v1"
)
SUPPLEMENTAL_RUNTIME_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-supplemental-runtime-evidence-v1"
)


class ChallengeAttempt002SupplementalError(ValueError):
    """Raised when supplemental evidence cannot match the frozen future grid."""


@dataclass(frozen=True, slots=True)
class Attempt002SupplementalConfig:
    """Hash-pinned inputs for post-result diagnostics and policy validation."""

    run_id: str
    output_dir: Path | str
    repository_root: Path | str
    future_manifest_path: Path | str
    expected_future_manifest_sha256: str
    operator_authorization_path: Path | str
    expected_operator_authorization_sha256: str
    shared_source_rows_path: Path | str
    expected_shared_source_rows_sha256: str
    feature_rows_path: Path | str
    expected_feature_rows_sha256: str
    native_decisions_path: Path | str
    expected_native_decisions_sha256: str
    regime_contract_path: Path | str
    expected_regime_contract_sha256: str
    policy_manifest_path: Path | str
    expected_policy_manifest_sha256: str
    compatibility_manifest_path: Path | str
    expected_compatibility_manifest_sha256: str
    implementation_commit: str
    generated_at: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not _is_git_commit(self.implementation_commit):
            raise ValueError("implementation_commit must be a Git SHA-1")
        if not self.generated_at.endswith("Z"):
            raise ValueError("generated_at must be an explicit UTC timestamp")
        for name in (
            "output_dir",
            "repository_root",
            "future_manifest_path",
            "operator_authorization_path",
            "shared_source_rows_path",
            "feature_rows_path",
            "native_decisions_path",
            "regime_contract_path",
            "policy_manifest_path",
            "compatibility_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "expected_future_manifest_sha256",
            "expected_operator_authorization_sha256",
            "expected_shared_source_rows_sha256",
            "expected_feature_rows_sha256",
            "expected_native_decisions_sha256",
            "expected_regime_contract_sha256",
            "expected_policy_manifest_sha256",
            "expected_compatibility_manifest_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 digest")


def run_attempt_002_supplemental_evidence(
    config: Attempt002SupplementalConfig,
) -> dict[str, Any]:
    """Generate deterministic diagnostics without changing the future gate."""

    root = config.repository_root.resolve()
    protocol_path = (
        root
        / "examples/v8/polymarket_configs"
        / "challenge_attempt_002_preregistration.json"
    )
    protocol_sha256 = _sha256_file(protocol_path)
    protocol = _load_json(protocol_path)
    validate_attempt_002_preregistration(protocol)

    paths = {
        "future_manifest": config.future_manifest_path.resolve(),
        "operator_authorization": (
            config.operator_authorization_path.resolve()
        ),
        "shared_source_rows": config.shared_source_rows_path.resolve(),
        "feature_rows": config.feature_rows_path.resolve(),
        "native_decisions": config.native_decisions_path.resolve(),
        "regime_contract": config.regime_contract_path.resolve(),
        "policy_manifest": config.policy_manifest_path.resolve(),
        "compatibility_manifest": (
            config.compatibility_manifest_path.resolve()
        ),
    }
    expected = {
        "future_manifest": config.expected_future_manifest_sha256,
        "operator_authorization": (
            config.expected_operator_authorization_sha256
        ),
        "shared_source_rows": config.expected_shared_source_rows_sha256,
        "feature_rows": config.expected_feature_rows_sha256,
        "native_decisions": config.expected_native_decisions_sha256,
        "regime_contract": config.expected_regime_contract_sha256,
        "policy_manifest": config.expected_policy_manifest_sha256,
        "compatibility_manifest": (
            config.expected_compatibility_manifest_sha256
        ),
    }
    for name, path in paths.items():
        _verify(path, expected[name], label=name)

    future_manifest = _load_json(paths["future_manifest"])
    authorization_descriptor = _descriptor(paths["operator_authorization"])
    bundle = _validated_future_bundle(
        future_manifest=future_manifest,
        future_manifest_sha256=expected["future_manifest"],
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        runtime={"operator_authorization": authorization_descriptor},
    )
    if not all(
        bundle["checks"][name]
        for name in (
            "future_manifest_schema_exact",
            "future_manifest_real_single_use",
            "future_result_schema_exact",
            "future_result_real_and_not_historical",
            "target_access_claim_schema_exact",
            "target_access_claim_real_single_use_and_alpha_consumed",
            "operator_authorization_hash_reconciles",
            "exact_120_comparison_recomputed",
            "future_bundle_hash_lineage_reconciles",
            "historical_or_synthetic_evidence_not_substituted",
        )
    ):
        raise ChallengeAttempt002SupplementalError(
            "attempt-002 future bundle is not real, single-use, and reconciled"
        )

    pairs_path = _verified_descriptor(
        future_manifest["target_free_pairs"],
        label="supplemental target-free pairs",
    )
    comparison_path = _verified_descriptor(
        future_manifest["comparison"],
        label="supplemental future comparison",
    )
    pairs = _load_jsonl(pairs_path)
    comparison = _load_jsonl(comparison_path)
    source_rows = _load_jsonl(paths["shared_source_rows"])
    feature_rows = _load_jsonl(paths["feature_rows"])
    native_decisions = _load_jsonl(paths["native_decisions"])
    _validate_frozen_inputs(
        pairs=pairs,
        comparison=comparison,
        source_rows=source_rows,
        feature_rows=feature_rows,
        native_decisions=native_decisions,
    )

    regime_contract = _load_json(paths["regime_contract"])
    policy_manifest = _load_json(paths["policy_manifest"])
    compatibility = _load_json(paths["compatibility_manifest"])
    validate_regime_definition_contract(regime_contract)
    candidate_rows, baseline_rows = _diagnostic_decision_rows(
        comparison=comparison,
        source_rows=source_rows,
    )
    assignments = _regime_assignments(
        selected_candidate_rows=candidate_rows,
        source_rows=source_rows,
        feature_rows=feature_rows,
        regime_contract=regime_contract,
    )
    regime_artifacts = build_regime_stratified_diagnostics(
        assignments=assignments,
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        contract=regime_contract,
    )
    future_result_sha256 = bundle["hashes"]["future_result"]
    common = {
        "attempt_id": protocol["attempt_id"],
        "selected_candidate_id": CANDIDATE_ID,
        "source_attempt_002_future_manifest_sha256": expected[
            "future_manifest"
        ],
        "source_attempt_002_result_sha256": future_result_sha256,
        "source_target_free_pairs_sha256": _sha256_file(pairs_path),
        "source_future_comparison_sha256": _sha256_file(comparison_path),
        "safety": SAFE_FALSES,
    }
    regime_report = dict(
        regime_artifacts["regime_stratified_pnl_report"]
    )
    regime_report.pop("report_sha256", None)
    regime_report.update(common)
    regime_report["reported_dimensions"] = list(DIMENSION_BUCKETS)
    regime_report["report_sha256"] = canonical_payload_sha256(
        regime_report,
        payload_schema_version=REGIME_REPORT_SCHEMA_VERSION,
    )
    regime_artifacts["regime_stratified_pnl_report"] = regime_report

    provider_report = build_provider_health_diagnostics(
        feature_rows=feature_rows,
        decision_rows=_provider_health_diagnostic_decisions(
            selected_candidate_rows=candidate_rows,
            source_rows=source_rows,
        ),
    )
    provider_report.update(common)
    provider_report["report_id"] = canonical_payload_sha256(
        provider_report,
        payload_schema_version="bigan-v8-provider-health-diagnostics-v1",
    )
    if (
        provider_report["decision_row_count"] != 120
        or provider_report["matched_decision_count"] != 120
        or provider_report["unmatched_decision_count"] != 0
        or provider_report["feature_completeness_report"][
            "incomplete_feature_row_count"
        ]
        != 0
    ):
        raise ChallengeAttempt002SupplementalError(
            "provider-health evidence is incomplete or does not reconcile"
        )

    policy_inputs = _execution_policy_inputs(
        source_rows=source_rows,
        feature_rows=feature_rows,
        native_decisions=native_decisions,
        source_model_hash=str(compatibility["source_model_hash"]),
    )
    policy_results = _run_all_execution_policies(
        policy_inputs=policy_inputs,
        policy_manifest=policy_manifest,
        policy_manifest_path=paths["policy_manifest"],
        compatibility=compatibility,
    )
    parity_report = _attempt_002_policy_report(
        schema_version=AGGREGATE_PARITY_SCHEMA_VERSION,
        common=common,
        policy_results=policy_results,
        report_key="parity",
    )
    safety_report = _attempt_002_policy_report(
        schema_version=AGGREGATE_SAFETY_SCHEMA_VERSION,
        common=common,
        policy_results=policy_results,
        report_key="safety",
    )
    reconciliation_report = _attempt_002_policy_report(
        schema_version=AGGREGATE_RECONCILIATION_SCHEMA_VERSION,
        common=common,
        policy_results=policy_results,
        report_key="reconciliation",
    )

    run_dir = config.output_dir.resolve() / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "regime_assignments": run_dir / "attempt_002_regime_assignments.jsonl",
        "regime_stratified_pnl_report": (
            run_dir / "attempt_002_regime_stratified_pnl_report.json"
        ),
        "regime_bootstrap_report": (
            run_dir / "attempt_002_regime_bootstrap_report.json"
        ),
        "side_action_attribution_report": (
            run_dir / "attempt_002_side_action_attribution_report.json"
        ),
        "regime_markdown": run_dir / "attempt_002_regime_diagnostics.md",
        "provider_health_diagnostics_report": (
            run_dir / "attempt_002_provider_health_diagnostics.json"
        ),
        "execution_policy_inputs": (
            run_dir / "attempt_002_execution_policy_inputs.jsonl"
        ),
        "replay_parity_report": (
            run_dir / "attempt_002_replay_parity_report.json"
        ),
        "policy_safety_report": (
            run_dir / "attempt_002_policy_safety_report.json"
        ),
        "policy_reconciliation_report": (
            run_dir / "attempt_002_policy_reconciliation_report.json"
        ),
    }
    _write_jsonl(output_paths["regime_assignments"], assignments)
    _write_json(
        output_paths["regime_stratified_pnl_report"],
        regime_report,
    )
    _write_json(
        output_paths["regime_bootstrap_report"],
        regime_artifacts["regime_bootstrap_report"],
    )
    _write_json(
        output_paths["side_action_attribution_report"],
        regime_artifacts["side_action_attribution_report"],
    )
    output_paths["regime_markdown"].write_text(
        regime_diagnostics_markdown(regime_artifacts),
        encoding="utf-8",
    )
    _write_json(
        output_paths["provider_health_diagnostics_report"],
        provider_report,
    )
    _write_jsonl(output_paths["execution_policy_inputs"], policy_inputs)
    _write_json(output_paths["replay_parity_report"], parity_report)
    _write_json(output_paths["policy_safety_report"], safety_report)
    _write_json(
        output_paths["policy_reconciliation_report"],
        reconciliation_report,
    )
    policy_replays = _write_policy_replays(
        run_dir=run_dir,
        policy_results=policy_results,
    )

    runtime_evidence = {
        "schema_version": SUPPLEMENTAL_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "operator_authorization": authorization_descriptor,
        **{
            name: _descriptor(output_paths[name])
            for name in (
                "provider_health_diagnostics_report",
                "regime_stratified_pnl_report",
                "replay_parity_report",
                "policy_safety_report",
                "policy_reconciliation_report",
            )
        },
    }
    runtime_path = run_dir / "attempt_002_supplemental_runtime_evidence.json"
    _write_json(runtime_path, runtime_evidence)
    manifest = {
        "schema_version": SUPPLEMENTAL_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "attempt_id": protocol["attempt_id"],
        "candidate_id": CANDIDATE_ID,
        "implementation_commit": config.implementation_commit,
        "generated_at": config.generated_at,
        "future_manifest": _descriptor(paths["future_manifest"]),
        "shared_source_rows": _descriptor(paths["shared_source_rows"]),
        "feature_rows": _descriptor(paths["feature_rows"]),
        "native_decisions": _descriptor(paths["native_decisions"]),
        "regime_contract": _descriptor(paths["regime_contract"]),
        "policy_manifest": _descriptor(paths["policy_manifest"]),
        "compatibility_manifest": _descriptor(
            paths["compatibility_manifest"]
        ),
        "outputs": {
            name: _descriptor(path)
            for name, path in sorted(output_paths.items())
        },
        "policy_replays": policy_replays,
        "supplemental_runtime_evidence": _descriptor(runtime_path),
        "future_gate_changed": False,
        "result_selected_policy_used": False,
        "historical_or_synthetic_evidence_used": False,
        "promotion_decision_emitted": False,
        "safety": SAFE_FALSES,
    }
    manifest_path = run_dir / "attempt_002_supplemental_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "runtime_evidence_path": runtime_path,
        "runtime_evidence_sha256": _sha256_file(runtime_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "provider_health_report": provider_report,
        "regime_report": regime_report,
        "parity_report": parity_report,
        "safety_report": safety_report,
        "reconciliation_report": reconciliation_report,
    }


def _validate_frozen_inputs(
    *,
    pairs: Sequence[Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    native_decisions: Sequence[Mapping[str, Any]],
) -> None:
    if not all(
        len(rows) == 120
        for rows in (pairs, comparison, source_rows, native_decisions)
    ):
        raise ChallengeAttempt002SupplementalError(
            "pairs, comparison, source, and native decisions must be exact-120"
        )
    pair_by_market = _one_per_market(pairs, label="target-free pairs")
    comparison_by_market = _one_per_market(
        comparison,
        label="future comparison",
    )
    source_by_market = _one_per_market(
        source_rows,
        label="shared source rows",
    )
    native_by_market = _one_per_market(
        native_decisions,
        label="native decisions",
    )
    if not (
        set(pair_by_market)
        == set(comparison_by_market)
        == set(source_by_market)
        == set(native_by_market)
    ):
        raise ChallengeAttempt002SupplementalError(
            "supplemental evidence market grids differ"
        )
    for market_id, pair in pair_by_market.items():
        source = source_by_market[market_id]
        comparison_row = comparison_by_market[market_id]
        if (
            pair.get("shared_source_row_id")
            not in {
                source.get("shared_source_row_id"),
                source.get("source_row_id"),
            }
            or comparison_row.get("source_target_free_pair_id")
            != pair.get("pair_id")
            or source.get("capture_quality_valid") is not True
            or source.get("target_used_as_decision_input") is not False
            or int(source.get("policy_grid_decision_ts") or 0) <= 0
        ):
            raise ChallengeAttempt002SupplementalError(
                f"supplemental lineage mismatch for {market_id}"
            )
    feature_keys = {
        (str(row.get("market_id") or ""), int(row.get("decision_ts") or 0))
        for row in feature_rows
    }
    required_keys = {
        (
            market_id,
            int(source.get("policy_grid_decision_ts") or 0),
        )
        for market_id, source in source_by_market.items()
    }
    if not required_keys.issubset(feature_keys):
        raise ChallengeAttempt002SupplementalError(
            "decision-time feature rows do not cover the exact future grid"
        )


def _diagnostic_decision_rows(
    *,
    comparison: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_market = {
        str(row["market_id"]): row for row in source_rows
    }
    candidates = []
    baselines = []
    for row in comparison:
        market_id = str(row["market_id"])
        decision_ts = int(
            source_by_market[market_id]["policy_grid_decision_ts"]
        )
        candidate_action = str(row["candidate_action"])
        baseline_action = str(row["baseline_action"])
        candidates.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "selected_side": row["candidate_side"],
                "executed_action": candidate_action,
                "decision_origin": (
                    "abstention"
                    if candidate_action == NO_TRADE
                    else "primary"
                ),
                "selection_source": (
                    "abstention"
                    if candidate_action == NO_TRADE
                    else "primary"
                ),
                "execution_guard_order_allowed": (
                    candidate_action != NO_TRADE
                ),
                "after_cost_pnl": row["candidate_after_cost_pnl"],
            }
        )
        baselines.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "selected_side": row["baseline_side"],
                "executed_action": baseline_action,
                "decision_origin": "matched_frozen_v6_7",
                "execution_guard_order_allowed": (
                    baseline_action != NO_TRADE
                ),
                "after_cost_pnl": row["baseline_after_cost_pnl"],
            }
        )
    return candidates, baselines


def _attempt_002_policy_report(
    *,
    schema_version: str,
    common: Mapping[str, Any],
    policy_results: Mapping[str, dict[str, Any]],
    report_key: str,
) -> dict[str, Any]:
    report = _aggregate_policy_report(
        schema_version=schema_version,
        common={},
        policy_results=dict(policy_results),
        report_key=report_key,
    )
    report.pop("report_id", None)
    report.update(common)
    report["report_id"] = canonical_payload_sha256(
        report,
        payload_schema_version=schema_version,
    )
    return report


def _one_per_market(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    output = {}
    for row in rows:
        market_id = str(row.get("market_id") or "")
        if not market_id or market_id in output:
            raise ChallengeAttempt002SupplementalError(
                f"{label} market identity is missing or duplicated"
            )
        output[market_id] = row
    return output


def _descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify(path: Path, expected: str, *, label: str) -> None:
    actual = _sha256_file(path)
    if actual != expected.lower():
        raise ChallengeAttempt002SupplementalError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


__all__ = [
    "Attempt002SupplementalConfig",
    "ChallengeAttempt002SupplementalError",
    "SUPPLEMENTAL_MANIFEST_SCHEMA_VERSION",
    "SUPPLEMENTAL_RUNTIME_MANIFEST_SCHEMA_VERSION",
    "run_attempt_002_supplemental_evidence",
]
