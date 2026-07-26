from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_attempt_002_pipeline import (
    Attempt002EvaluationConfig,
    build_attempt_002_target_access_claim,
    build_attempt_002_target_free_pairs,
    run_attempt_002_future_evaluation,
)
from bigan.v8.polymarket.challenge_attempt_002_promotion import (
    ChallengeAttempt002PromotionError,
    attempt_002_promotion_readiness_markdown,
    audit_attempt_002_promotion,
)
from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.regime_diagnostics import DIMENSION_BUCKETS

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "examples/v8/polymarket_configs"
    / "challenge_attempt_002_preregistration.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _target_free_inputs() -> tuple[list[dict], list[dict], list[dict]]:
    protocol = _json(PROTOCOL_PATH)
    boundary = protocol["preregistration_freeze_created_ts"]
    selected = {0, 24, 48, 72, 96}
    shared = []
    candidate = []
    baseline = []
    for index in range(120):
        market_id = f"future-promotion-{index:03d}"
        start = boundary + (index + 1) * 300_000
        source = {
            "market_id": market_id,
            "market_start_ts": start,
            "market_end_ts": start + 300_000,
            "decision_ts": start + 60_000,
            "capture_quality_valid": True,
            "target_used_as_decision_input": False,
        }
        source["shared_source_row_id"] = canonical_json_sha256(source)
        shared.append(source)

        selected_candidate = index in selected
        candidate_row = {
            "candidate_id": (
                "v8_1_entry_price_floor_0_30_sized_1_0"
            ),
            "market_id": market_id,
            "selected_action": (
                "BUY_UP_SELL_BEFORE_CLOSE"
                if selected_candidate
                else "NO_TRADE"
            ),
            "selected_side": "UP" if selected_candidate else "NONE",
            "fixed_candidate_position_size": 1.0,
            "candidate_position_size": (
                1.0 if selected_candidate else 0.0
            ),
            "target_used_as_decision_time_input": False,
            "outcome_or_pnl_field_used_at_inference": False,
            "safety": SAFE_FALSES,
        }
        candidate_row["decision_id"] = canonical_json_sha256(
            candidate_row
        )
        candidate.append(candidate_row)

        baseline_row = {
            "market_id": market_id,
            "selected_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
            "selected_side": "DOWN",
            "baseline_fixed_position_size": 0.2,
            "target_used_as_decision_input": False,
            "outcomes_resolution_labels_or_pnl_opened": False,
            "safety": SAFE_FALSES,
        }
        baseline_row["decision_id"] = canonical_json_sha256(baseline_row)
        baseline.append(baseline_row)
    return shared, candidate, baseline


def _settlement_targets(pairs: list[dict]) -> list[dict]:
    keys = {
        (pair["market_id"], action)
        for pair in pairs
        for action in (
            pair["candidate_action"],
            pair["baseline_action"],
        )
        if action != "NO_TRADE"
    }
    rows = []
    for market_id, action in sorted(keys):
        target = {
            "market_id": market_id,
            "action": action,
            "side": "UP" if action.startswith("BUY_UP_") else "DOWN",
            "runtime_policy_after_cost_net_pnl_per_contract": (
                0.2 if action.startswith("BUY_UP_") else -0.05
            ),
            "target_used_as_decision_input": False,
            "target_available_only_post_exit_or_official_resolution": True,
            "settled_after_market_close": True,
            "cost_fields_subtracted_exactly_once": True,
            "official_read_only_resolution": True,
            "synthetic_only": False,
            "safety": SAFE_FALSES,
        }
        target["target_row_id"] = canonical_json_sha256(target)
        rows.append(target)
    return rows


def _real_future_bundle(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    protocol = _json(PROTOCOL_PATH)
    protocol_sha256 = _sha256(PROTOCOL_PATH)
    shared, candidate, baseline = _target_free_inputs()
    pairs = build_attempt_002_target_free_pairs(
        shared_source_rows=shared,
        candidate_decisions=candidate,
        baseline_decisions=baseline,
        protocol=protocol,
    )
    authorization = {
        "schema_version": (
            "bigan-v8-challenge-attempt-002-operator-authorization-v1"
        ),
        "attempt_id": protocol["attempt_id"],
        "protocol_sha256": protocol_sha256,
        "authorization_scope": (
            "outcome_blind_collection_of_exact_120_market_window_only"
        ),
        "exact_quality_valid_market_count": 120,
        "collection_authorized": True,
        "authorized_at": "2026-07-27T00:00:00Z",
        "authorization_source": "synthetic_test_fixture_only",
        "target_access_before_decision_freeze_authorized": False,
        "outcomes_during_collection_authorized": False,
        "paper_allowed": False,
        "live_allowed": False,
        "write_allowed": False,
        "wallet_allowed": False,
        "handoff_allowed": False,
        "promotion_allowed": False,
        "capital_at_risk": False,
    }
    authorization_path = tmp_path / "authorization.json"
    _write_json(authorization_path, authorization)
    claim = build_attempt_002_target_access_claim(
        target_free_pairs=pairs,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        target_access_started_ts=pairs[-1]["market_end_ts"] + 1,
        operator_authorization_sha256=_sha256(authorization_path),
        synthetic_only=False,
    )
    pairs_path = tmp_path / "pairs.jsonl"
    claim_path = tmp_path / "claim.json"
    targets_path = tmp_path / "targets.jsonl"
    _write_jsonl(pairs_path, pairs)
    _write_json(claim_path, claim)
    _write_jsonl(targets_path, _settlement_targets(pairs))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    evaluated = run_attempt_002_future_evaluation(
        Attempt002EvaluationConfig(
            run_id="real-future-test-fixture",
            output_dir=tmp_path / "runs",
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=protocol_sha256,
            target_free_pairs_path=pairs_path,
            expected_target_free_pairs_sha256=_sha256(pairs_path),
            target_access_claim_path=claim_path,
            expected_target_access_claim_sha256=_sha256(claim_path),
            settlement_targets_path=targets_path,
            expected_settlement_targets_sha256=_sha256(targets_path),
            implementation_commit=commit,
            evaluated_at="2026-07-27T01:00:00Z",
            operator_authorization_path=authorization_path,
            expected_operator_authorization_sha256=_sha256(
                authorization_path
            ),
        )
    )
    manifest_path = Path(evaluated["manifest_path"])
    result_sha256 = str(evaluated["result_sha256"])
    manifest_sha256 = str(evaluated["manifest_sha256"])
    common = {
        "attempt_id": protocol["attempt_id"],
        "selected_candidate_id": (
            "v8_1_entry_price_floor_0_30_sized_1_0"
        ),
        "source_attempt_002_future_manifest_sha256": manifest_sha256,
        "source_attempt_002_result_sha256": result_sha256,
        "safety": SAFE_FALSES,
    }
    reports = {
        "provider_health_diagnostics_report": {
            "schema_version": (
                "bigan-v8-provider-health-diagnostics-v1"
            ),
            **common,
            "feature_completeness_report": {
                "feature_row_count": 120,
                "complete_feature_row_count": 120,
                "incomplete_feature_row_count": 0,
            },
            "decision_row_count": 120,
            "matched_decision_count": 120,
            "unmatched_decision_count": 0,
            "diagnostic_only": True,
            "outcomes_settlement_pnl_or_future_information_used": False,
        },
        "regime_stratified_pnl_report": {
            "schema_version": (
                "bigan-v8-regime-stratified-pnl-report-v1"
            ),
            **common,
            "reported_dimensions": list(DIMENSION_BUCKETS),
            "all_dimension_partitions_reconcile": True,
            "diagnostic_only": True,
            "stratified_metrics_are_eligibility_blockers": False,
        },
        "replay_parity_report": {
            "schema_version": (
                "bigan-v8-challenge-execution-policy-replay-parity-v1"
            ),
            **common,
            "policy_candidate_count": 3,
            "all_preregistered_policy_candidates_evaluated": True,
            "outcome_selected_policy_used": False,
            "passed": True,
        },
        "policy_safety_report": {
            "schema_version": (
                "bigan-v8-challenge-execution-policy-safety-v1"
            ),
            **common,
            "policy_candidate_count": 3,
            "all_preregistered_policy_candidates_evaluated": True,
            "outcome_selected_policy_used": False,
            "passed": True,
        },
        "policy_reconciliation_report": {
            "schema_version": (
                "bigan-v8-challenge-execution-policy-reconciliation-v1"
            ),
            **common,
            "policy_candidate_count": 3,
            "all_preregistered_policy_candidates_evaluated": True,
            "outcome_selected_policy_used": False,
            "passed": True,
        },
    }
    runtime: dict[str, dict[str, object]] = {
        "operator_authorization": _descriptor(authorization_path)
    }
    for name, report in reports.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, report)
        runtime[name] = _descriptor(path)
    return _descriptor(manifest_path), runtime


def test_static_prerequisites_pass_but_future_evidence_is_required() -> None:
    report = audit_attempt_002_promotion(repository_root=ROOT)

    assert all(report["static_checks"].values())
    assert report["decision"] == "BLOCKED"
    assert report["challenge_model_promotion_eligible"] is False
    assert report["fresh_runtime_evidence_supplied"] is False
    assert "evidence:future_manifest_schema_exact" in report["blockers"]
    assert report["paper_candidate_unlocked"] is False
    assert report["live_unlocked"] is False
    assert report["write_enabled"] is False


def test_complete_real_hash_bound_attempt_002_evidence_can_promote(
    tmp_path: Path,
) -> None:
    future, runtime = _real_future_bundle(tmp_path)
    report = audit_attempt_002_promotion(
        repository_root=ROOT,
        future_evidence_manifest=future,
        supplemental_runtime_evidence=runtime,
    )

    assert all(report["static_checks"].values())
    assert all(report["evidence_checks"].values())
    assert report["decision"] == "PROMOTE_TO_CHAMPION"
    assert report["challenge_model_promotion_eligible"] is True
    assert report["selected_champion_candidate"] == (
        "v8_1_entry_price_floor_0_30_sized_1_0"
    )
    assert report["promotion_unlocked"] is True
    assert report["paper_candidate_unlocked"] is False
    assert report["live_unlocked"] is False
    assert report["capital_at_risk"] is False
    assert "Historical and synthetic evidence never substitute" in (
        attempt_002_promotion_readiness_markdown(report)
    )


def test_missing_issue_257_runtime_report_fails_closed(
    tmp_path: Path,
) -> None:
    future, runtime = _real_future_bundle(tmp_path)
    runtime.pop("provider_health_diagnostics_report")

    report = audit_attempt_002_promotion(
        repository_root=ROOT,
        future_evidence_manifest=future,
        supplemental_runtime_evidence=runtime,
    )

    assert report["decision"] == "BLOCKED"
    assert (
        "evidence:issue_257_future_provider_health_complete"
        in report["blockers"]
    )
    assert (
        "evidence:all_required_supplemental_runtime_evidence_present"
        in report["blockers"]
    )


def test_tampered_future_result_is_rejected_even_if_manifest_is_rehashed(
    tmp_path: Path,
) -> None:
    future, runtime = _real_future_bundle(tmp_path)
    manifest_path = Path(str(future["path"]))
    manifest = _json(manifest_path)
    result_path = Path(manifest["result"]["path"])
    result = _json(result_path)
    result["gate_result"]["metrics"][
        "candidate_total_after_cost_pnl"
    ] += 1.0
    _write_json(result_path, result)
    manifest["result"] = _descriptor(result_path)
    _write_json(manifest_path, manifest)
    future = _descriptor(manifest_path)
    new_manifest_sha256 = str(future["sha256"])
    for name, descriptor in runtime.items():
        if name == "operator_authorization":
            continue
        path = Path(str(descriptor["path"]))
        report = _json(path)
        report[
            "source_attempt_002_future_manifest_sha256"
        ] = new_manifest_sha256
        report["source_attempt_002_result_sha256"] = _sha256(result_path)
        _write_json(path, report)
        runtime[name] = _descriptor(path)

    with pytest.raises(
        ChallengeAttempt002PromotionError,
        match="does not match recomputation",
    ):
        audit_attempt_002_promotion(
            repository_root=ROOT,
            future_evidence_manifest=future,
            supplemental_runtime_evidence=runtime,
        )


def test_hash_tamper_is_rejected_before_promotion(tmp_path: Path) -> None:
    future, runtime = _real_future_bundle(tmp_path)
    future["sha256"] = "0" * 64

    with pytest.raises(
        ChallengeAttempt002PromotionError,
        match="hash mismatch",
    ):
        audit_attempt_002_promotion(
            repository_root=ROOT,
            future_evidence_manifest=future,
            supplemental_runtime_evidence=runtime,
        )
