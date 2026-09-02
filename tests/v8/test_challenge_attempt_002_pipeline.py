from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_attempt_002_pipeline import (
    ZERO_SHA256,
    Attempt002EvaluationConfig,
    ChallengeAttempt002PipelineError,
    build_attempt_002_settled_comparison,
    build_attempt_002_target_access_claim,
    build_attempt_002_target_free_pairs,
    run_attempt_002_future_evaluation,
    validate_attempt_002_operator_authorization,
)
from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.contracts import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "examples/v8/polymarket_configs"
    / "challenge_attempt_002_preregistration.json"
)


def _protocol() -> dict:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _inputs() -> tuple[list[dict], list[dict], list[dict]]:
    protocol = _protocol()
    boundary = protocol["preregistration_freeze_created_ts"]
    selected = {0, 24, 48, 72, 96}
    shared = []
    candidate = []
    baseline = []
    for index in range(120):
        market_id = f"future-{index:03d}"
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

        candidate_selected = index in selected
        candidate_decision = {
            "candidate_id": (
                "v8_1_entry_price_floor_0_30_sized_1_0"
            ),
            "market_id": market_id,
            "selected_action": (
                "BUY_UP_SELL_BEFORE_CLOSE"
                if candidate_selected
                else "NO_TRADE"
            ),
            "selected_side": "UP" if candidate_selected else "NONE",
            "fixed_candidate_position_size": 1.0,
            "candidate_position_size": 1.0 if candidate_selected else 0.0,
            "target_used_as_decision_time_input": False,
            "outcome_or_pnl_field_used_at_inference": False,
            "safety": SAFE_FALSES,
        }
        candidate_decision["decision_id"] = canonical_json_sha256(
            candidate_decision
        )
        candidate.append(candidate_decision)

        baseline_decision = {
            "market_id": market_id,
            "selected_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
            "selected_side": "DOWN",
            "baseline_fixed_position_size": 0.2,
            "target_used_as_decision_input": False,
            "outcomes_resolution_labels_or_pnl_opened": False,
            "safety": SAFE_FALSES,
        }
        baseline_decision["decision_id"] = canonical_json_sha256(
            baseline_decision
        )
        baseline.append(baseline_decision)
    return shared, candidate, baseline


def _targets(pairs: list[dict]) -> list[dict]:
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
            "official_read_only_resolution": False,
            "synthetic_only": True,
            "safety": SAFE_FALSES,
        }
        target["target_row_id"] = canonical_json_sha256(target)
        rows.append(target)
    return rows


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_free_pairs_bind_same_future_window_without_pnl() -> None:
    shared, candidate, baseline = _inputs()
    pairs = build_attempt_002_target_free_pairs(
        shared_source_rows=shared,
        candidate_decisions=candidate,
        baseline_decisions=baseline,
        protocol=_protocol(),
    )

    assert len(pairs) == 120
    assert sum(row["candidate_action"] != "NO_TRADE" for row in pairs) == 5
    assert all("candidate_after_cost_pnl" not in row for row in pairs)
    assert all("baseline_after_cost_pnl" not in row for row in pairs)
    assert all(
        row["outcomes_resolution_labels_or_pnl_opened"] is False
        for row in pairs
    )
    assert all(row["target_used_as_decision_input"] is False for row in pairs)
    assert all(row["safety"] == SAFE_FALSES for row in pairs)


def test_real_target_access_requires_separate_operator_authorization() -> None:
    shared, candidate, baseline = _inputs()
    protocol = _protocol()
    pairs = build_attempt_002_target_free_pairs(
        shared_source_rows=shared,
        candidate_decisions=candidate,
        baseline_decisions=baseline,
        protocol=protocol,
    )
    with pytest.raises(
        ChallengeAttempt002PipelineError,
        match="operator authorization",
    ):
        build_attempt_002_target_access_claim(
            target_free_pairs=pairs,
            protocol=protocol,
            protocol_sha256=_sha256(PROTOCOL_PATH),
            target_access_started_ts=pairs[-1]["market_end_ts"] + 1,
            operator_authorization_sha256=ZERO_SHA256,
            synthetic_only=False,
        )


def test_operator_authorization_is_collection_only() -> None:
    protocol = _protocol()
    authorization = {
        "schema_version": (
            "bigan-v8-challenge-attempt-002-operator-authorization-v1"
        ),
        "attempt_id": protocol["attempt_id"],
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "authorization_scope": (
            "outcome_blind_collection_of_exact_120_market_window_only"
        ),
        "exact_quality_valid_market_count": 120,
        "collection_authorized": True,
        "authorized_at": "2026-07-26T18:00:00Z",
        "authorization_source": "explicit_user_instruction",
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
    validate_attempt_002_operator_authorization(
        authorization,
        protocol=protocol,
        protocol_sha256=_sha256(PROTOCOL_PATH),
    )

    authorization["outcomes_during_collection_authorized"] = True
    with pytest.raises(
        ChallengeAttempt002PipelineError,
        match="target",
    ):
        validate_attempt_002_operator_authorization(
            authorization,
            protocol=protocol,
            protocol_sha256=_sha256(PROTOCOL_PATH),
        )


def test_synthetic_settlement_maps_sizes_and_never_unlocks_promotion() -> None:
    shared, candidate, baseline = _inputs()
    protocol = _protocol()
    protocol_sha256 = _sha256(PROTOCOL_PATH)
    pairs = build_attempt_002_target_free_pairs(
        shared_source_rows=shared,
        candidate_decisions=candidate,
        baseline_decisions=baseline,
        protocol=protocol,
    )
    claim = build_attempt_002_target_access_claim(
        target_free_pairs=pairs,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        target_access_started_ts=pairs[-1]["market_end_ts"] + 1,
        synthetic_only=True,
    )
    comparison = build_attempt_002_settled_comparison(
        target_free_pairs=pairs,
        settlement_targets=_targets(pairs),
        target_access_claim=claim,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )

    assert len(comparison) == 120
    assert sum(row["candidate_after_cost_pnl"] for row in comparison) == (
        pytest.approx(1.0)
    )
    assert sum(row["baseline_after_cost_pnl"] for row in comparison) == (
        pytest.approx(-1.2)
    )
    assert all(row["synthetic_only"] is True for row in comparison)


def test_single_use_runner_writes_hash_indexed_synthetic_evidence(
    tmp_path: Path,
) -> None:
    shared, candidate, baseline = _inputs()
    protocol = _protocol()
    protocol_sha256 = _sha256(PROTOCOL_PATH)
    pairs = build_attempt_002_target_free_pairs(
        shared_source_rows=shared,
        candidate_decisions=candidate,
        baseline_decisions=baseline,
        protocol=protocol,
    )
    claim = build_attempt_002_target_access_claim(
        target_free_pairs=pairs,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        target_access_started_ts=pairs[-1]["market_end_ts"] + 1,
        synthetic_only=True,
    )
    pairs_path = tmp_path / "pairs.jsonl"
    claim_path = tmp_path / "claim.json"
    targets_path = tmp_path / "targets.jsonl"
    _write_jsonl(pairs_path, pairs)
    _write_json(claim_path, claim)
    _write_jsonl(targets_path, _targets(pairs))

    output = run_attempt_002_future_evaluation(
        Attempt002EvaluationConfig(
            run_id="synthetic-pipeline",
            output_dir=tmp_path / "runs",
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=protocol_sha256,
            target_free_pairs_path=pairs_path,
            expected_target_free_pairs_sha256=_sha256(pairs_path),
            target_access_claim_path=claim_path,
            expected_target_access_claim_sha256=_sha256(claim_path),
            settlement_targets_path=targets_path,
            expected_settlement_targets_sha256=_sha256(targets_path),
            implementation_commit="a" * 40,
            evaluated_at="2026-07-26T18:00:00Z",
        )
    )

    assert output["result"]["all_future_success_criteria_passed"] is True
    assert output["result"]["synthetic_only"] is True
    assert output["result"]["real_future_evidence"] is False
    assert output["result"]["promotion_evidence_eligible"] is False
    assert output["manifest"]["promotion_evidence_eligible"] is False
    assert output["result"]["safety"] == SAFE_FALSES


def test_settlement_target_coverage_fails_closed() -> None:
    shared, candidate, baseline = _inputs()
    protocol = _protocol()
    protocol_sha256 = _sha256(PROTOCOL_PATH)
    pairs = build_attempt_002_target_free_pairs(
        shared_source_rows=shared,
        candidate_decisions=candidate,
        baseline_decisions=baseline,
        protocol=protocol,
    )
    claim = build_attempt_002_target_access_claim(
        target_free_pairs=pairs,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        target_access_started_ts=pairs[-1]["market_end_ts"] + 1,
        synthetic_only=True,
    )
    targets = _targets(pairs)
    targets.pop()
    with pytest.raises(
        ChallengeAttempt002PipelineError,
        match="exactly cover",
    ):
        build_attempt_002_settled_comparison(
            target_free_pairs=pairs,
            settlement_targets=targets,
            target_access_claim=claim,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
        )
