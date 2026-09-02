from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_hierarchical_side_family_lcb import (
    CANDIDATE_NAME,
    _apply_hierarchical_lcb_scores,
    _development_freeze_gate,
    _fresh_confirmatory_gate,
    _hierarchical_lcb_artifact,
    _validate_development_and_quarantine_rows,
    _validate_fresh_confirmatory_assignment_rows,
    validate_hierarchical_side_family_lcb_feature_contract,
    validate_hierarchical_side_family_lcb_protocol,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/execution_layer_v2_hierarchical_side_family_lcb_v1.json"
)
FEATURE_CONTRACT_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_hierarchical_side_family_lcb_feature_contract_v1.json"
)


def test_issue174_protocol_is_frozen_causal_and_confirmatory_blind() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    validate_hierarchical_side_family_lcb_protocol(protocol)

    assert protocol["candidate_name"] == CANDIDATE_NAME
    assert protocol["uses_issue173_confirmatory_labels_for_tuning"] is False
    assert protocol["fresh_confirmatory_collection"]["target_valid_unique_market_count"] == 60
    assert (
        protocol["hierarchical_expected_mean_lcb_protocol"][
            "forced_action_side_or_family_quota_enabled"
        ]
        is False
    )
    assert protocol["safety"]["v8_execution_handoff_allowed"] is False

    drifted = json.loads(json.dumps(protocol))
    drifted["development_source_contract"]["confirmatory_artifact_access_forbidden"] = False
    with pytest.raises(ValueError, match="development_roles"):
        validate_hierarchical_side_family_lcb_protocol(drifted)


def test_issue174_feature_contract_is_hash_pinned_and_quarantines_confirmatory() -> None:
    contract = _load_json(FEATURE_CONTRACT_PATH)
    validate_hierarchical_side_family_lcb_feature_contract(
        contract,
        expected_parent_protocol_sha256=_sha256(PROTOCOL_PATH),
    )

    assert contract["uses_issue173_confirmatory_labels_for_tuning"] is False
    assert contract["issue173_confirmatory_artifact_access_forbidden"] is True
    assert contract["settlement_or_outcome_fields_allowed_as_decision_inputs"] is False
    assert contract["forced_action_side_or_family_quota_enabled"] is False


def test_hierarchical_lcb_is_deterministic_and_uses_side_leaf_shrinkage() -> None:
    rows = _calibration_rows()
    protocol = _load_json(PROTOCOL_PATH)
    first = _hierarchical_lcb_artifact(
        rows,
        protocol=protocol,
        feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
    )
    second = _hierarchical_lcb_artifact(
        rows,
        protocol=protocol,
        feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
    )

    assert canonical_json_sha256(first) == canonical_json_sha256(second)
    assert first["method"] == ("market_grouped_bootstrap_hierarchical_mean_residual_lcb")
    assert all(group["support_passed"] for group in first["families"].values())
    assert all(group["support_passed"] for group in first["family_side_groups"].values())
    supported_leaves = [group for group in first["leaf_groups"].values() if group["support_passed"]]
    assert supported_leaves
    assert all(
        group["penalty_source"] == "leaf_shrunk_to_supported_family_side"
        for group in supported_leaves
    )
    assert first["forced_action_side_or_family_quota_enabled"] is False
    assert first["source_model_candidate_eligible"] is False


def test_unseen_leaf_falls_back_to_supported_side_without_forced_quota() -> None:
    artifact = _hierarchical_lcb_artifact(
        _calibration_rows(),
        protocol=_load_json(PROTOCOL_PATH),
        feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
    )
    prediction = _prediction(
        market_id="future-market",
        family="HOLD_TO_SETTLEMENT",
        side="UP",
        price=0.95,
        time_to_close=5.0,
        spread=900.0,
        queue=0.1,
        staleness=1_900.0,
    )
    scored = _apply_hierarchical_lcb_scores([prediction], artifact=artifact)[0]

    assert scored["hierarchical_score_available"] is True
    assert scored["hierarchical_penalty_source"] == (
        "unseen_leaf_fallback_to_family_side_or_family"
    )
    assert scored["forced_action_side_or_family_quota_enabled"] is False


def test_unsupported_family_fails_closed_to_no_trade_score() -> None:
    rows = [
        row
        for row in _calibration_rows()
        if row["action_family"] == "SELL_BEFORE_CLOSE" or int(row["market_id"].split("-")[-1]) < 10
    ]
    artifact = _hierarchical_lcb_artifact(
        rows,
        protocol=_load_json(PROTOCOL_PATH),
        feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
    )
    assert artifact["families"]["HOLD_TO_SETTLEMENT"]["support_passed"] is False

    scored = _apply_hierarchical_lcb_scores(
        [_prediction(family="HOLD_TO_SETTLEMENT", side="DOWN")],
        artifact=artifact,
    )[0]
    assert scored["hierarchical_score_available"] is False
    assert scored["hierarchical_expected_mean_lcb_net_return"] < -999_999.0
    assert scored["hierarchical_penalty_source"] == ("unsupported_family_fail_closed_no_trade")


def test_development_and_quarantine_roles_are_exact_and_disjoint() -> None:
    development = [
        {"role": "development_train", "market_id": f"train-{index}"} for index in range(60)
    ] + [{"role": "development_calibration", "market_id": f"cal-{index}"} for index in range(30)]
    quarantine = [
        {"role": "confirmatory_validation", "market_id": f"holdout-{index}"} for index in range(30)
    ]
    _validate_development_and_quarantine_rows(development, quarantine)

    with pytest.raises(ValueError, match="confirmatory quarantine"):
        _validate_development_and_quarantine_rows(development, quarantine[:-1])


def test_development_gate_fails_closed_on_quarantine_access_or_unsupported_family() -> None:
    artifact = _hierarchical_lcb_artifact(
        _calibration_rows(),
        protocol=_load_json(PROTOCOL_PATH),
        feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
    )
    metrics = {
        "accepted_bet_count": 20,
        "accepted_unique_market_count": 20,
        "net_pnl_sum": 1.0,
    }
    audits = [
        {"role": "development_train", "feature_causality_violation_count": 0} for _ in range(60)
    ] + [
        {
            "role": "development_calibration",
            "feature_causality_violation_count": 0,
        }
        for _ in range(30)
    ]
    gate = _development_freeze_gate(
        protocol=_load_json(PROTOCOL_PATH),
        candidate_metrics=metrics,
        baseline_metrics={**metrics, "net_pnl_sum": 0.5},
        hierarchy_artifact=artifact,
        corpus_audits=audits,
        quarantine_access_overlap={"forbidden-market"},
    )
    assert gate["passed"] is False
    assert "issue173_confirmatory_evidence_access_violation" in gate["reason_codes"]


def test_fresh_confirmatory_assignment_rows_are_exact_and_outcome_blind() -> None:
    rows = [
        {
            "market_id": f"fresh-{index}",
            "selection_rank": index + 1,
            "role": "fresh_confirmatory_validation",
            "execution_compatibility_validated_before_label_access": True,
            "labels_or_outcomes_opened_for_assignment": False,
        }
        for index in range(60)
    ]
    _validate_fresh_confirmatory_assignment_rows(rows)

    rows[-1]["labels_or_outcomes_opened_for_assignment"] = True
    with pytest.raises(ValueError, match="assignment contract failed"):
        _validate_fresh_confirmatory_assignment_rows(rows)


def test_fresh_confirmatory_gate_requires_positive_market_bootstrap_lower_bound() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    action_rows = [
        {
            "market_id": f"fresh-{index}",
            "action": "BUY_UP_HOLD_TO_SETTLEMENT",
            "decision_ts": 2_000_000_000_000 + index,
            "max_input_ts": 2_000_000_000_000 + index,
            "reference_price_feature_provenance": {"provenance_valid": True},
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
        }
        for index in range(60)
    ]
    candidate_replay = [
        {
            "execution_guard_order_allowed": True,
            "settlement_resolved_for_report_only": True,
            "required_runtime_fields_present": True,
        }
        for _ in range(30)
    ]
    candidate_metrics = {
        "accepted_bet_count": 30,
        "accepted_unique_market_count": 30,
        "accepted_bet_count_by_side": {"UP": 15, "DOWN": 15},
        "accepted_bet_count_by_family": {
            "HOLD_TO_SETTLEMENT": 15,
            "SELL_BEFORE_CLOSE": 15,
        },
        "net_pnl_sum": 1.0,
        "roi": 0.2,
    }
    baseline_metrics = {"net_pnl_sum": 0.5}
    robustness = {
        "market_bootstrap_interval_95": {
            "reported": True,
            "lower": 0.01,
            "upper": 0.8,
        },
        "leave_one_market_out": {"reported": True},
        "largest_winner_removal": {"reported": True},
    }
    audits = [{"feature_causality_violation_count": 0} for _ in range(60)]
    passed = _fresh_confirmatory_gate(
        protocol=protocol,
        action_rows=action_rows,
        candidate_replay=candidate_replay,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        robustness=robustness,
        corpus_audits=audits,
    )
    assert passed["passed"] is True

    robustness["market_bootstrap_interval_95"]["lower"] = -0.01
    blocked = _fresh_confirmatory_gate(
        protocol=protocol,
        action_rows=action_rows,
        candidate_replay=candidate_replay,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        robustness=robustness,
        corpus_audits=audits,
    )
    assert blocked["passed"] is False
    assert (
        "confirmatory_candidate_minus_baseline_bootstrap_lower_bound_not_positive"
        in blocked["reason_codes"]
    )


def _calibration_rows() -> list[dict]:
    rows = []
    for index in range(30):
        for family in ("HOLD_TO_SETTLEMENT", "SELL_BEFORE_CLOSE"):
            for side in ("UP", "DOWN"):
                row = _prediction(
                    market_id=f"market-{index}",
                    family=family,
                    side=side,
                    price=0.45 if index % 2 == 0 else 0.65,
                    time_to_close=250.0 if index % 2 == 0 else 150.0,
                    spread=300.0,
                    queue=0.8,
                    staleness=500.0,
                )
                row["target_net_pnl_per_contract"] = row["raw_family_expected_net_return"] - (
                    0.01 if side == "UP" else 0.02
                )
                rows.append(row)
    return rows


def _prediction(
    *,
    market_id: str = "market-0",
    family: str,
    side: str,
    price: float = 0.45,
    time_to_close: float = 250.0,
    spread: float = 300.0,
    queue: float = 0.8,
    staleness: float = 500.0,
) -> dict:
    action = f"BUY_{side}_{family}"
    return {
        "market_id": market_id,
        "decision_ts": 2_000_000_000_000,
        "action": action,
        "action_family": family,
        "side": side,
        "raw_family_expected_net_return": 0.08,
        "decision_time_features": {
            "execution_price": price,
            "time_to_close_seconds": time_to_close,
            "selected_side_spread_bps": spread,
            "selected_side_queue_fill_probability_proxy": queue,
            "selected_side_book_staleness_ms": staleness,
        },
        "target_net_pnl_per_contract": 0.06,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
