from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_paper_candidate import (
    FROZEN_HASHES,
    MANUAL_APPROVAL_SCOPE,
    MarketClusteredMeanEVV62PaperCandidateConfig,
    run_market_clustered_mean_ev_v6_2_paper_candidate_gate,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _blocked_execution() -> dict[str, object]:
    return {
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    model = tmp_path / "model.json"
    calibration = tmp_path / "calibration.json"
    model.write_text("model", encoding="utf-8")
    calibration.write_text("calibration", encoding="utf-8")
    review = {
        "manual_promotion_review_passed": True,
        "manual_promotion_review_blocking_reason_codes": [],
        "post_preregistration_source_audit": {
            "passed": True,
            "all_frozen_function_pins_unchanged": True,
        },
        "research_candidate_promoted": True,
        "source_model_candidate_eligible": True,
        "freeze_ready": True,
        "promotion_evidence_eligible": True,
        **_blocked_execution(),
    }
    side_gate = {
        "future_gate_passed": True,
        "future_gate_blocking_reason_codes": [],
        "future_result_driven_rerun_allowed": False,
        "pnl_hard_gate_aggregation": "selected_side_buy_up_buy_down_only",
        "action_and_action_family_pnl_diagnostic_only": True,
        "accepted_side_metrics": {
            "UP": {"accepted_bet_net_pnl_sum": 0.6},
            "DOWN": {"accepted_bet_net_pnl_sum": 2.0},
        },
        **_blocked_execution(),
    }
    handoff = {
        "paper_candidate_gate_required": True,
        "research_candidate_promoted": True,
        **_blocked_execution(),
    }
    review_path = tmp_path / "review.json"
    gate_path = tmp_path / "side_gate.json"
    handoff_path = tmp_path / "handoff.json"
    _write_json(review_path, review)
    _write_json(gate_path, side_gate)
    _write_json(handoff_path, handoff)
    promoted = {
        "candidate_name": "market_clustered_mean_ev_v6_2",
        "research_candidate_promoted": True,
        "source_model_candidate_eligible": True,
        "freeze_ready": True,
        "promotion_evidence_eligible": True,
        "manual_promotion_review": {"path": str(review_path), "sha256": _sha(review_path)},
        "side_only_gate_report": {"path": str(gate_path), "sha256": _sha(gate_path)},
        "source_model": {"path": str(model), "sha256": _sha(model)},
        "market_clustered_mean_risk_calibration": {
            "path": str(calibration),
            "sha256": _sha(calibration),
        },
        **_blocked_execution(),
    }
    promoted_path = tmp_path / "promoted.json"
    _write_json(promoted_path, promoted)
    monkeypatch.setitem(FROZEN_HASHES, "promoted_candidate_manifest", _sha(promoted_path))
    monkeypatch.setitem(FROZEN_HASHES, "manual_promotion_review", _sha(review_path))
    monkeypatch.setitem(FROZEN_HASHES, "paper_handoff_plan", _sha(handoff_path))
    monkeypatch.setitem(FROZEN_HASHES, "source_model", _sha(model))
    monkeypatch.setitem(FROZEN_HASHES, "calibration", _sha(calibration))
    monkeypatch.setitem(FROZEN_HASHES, "side_only_gate_report", _sha(gate_path))
    return promoted_path, handoff_path


def _config(
    tmp_path: Path,
    promoted: Path,
    handoff: Path,
    *,
    approved: bool,
    scope: str = MANUAL_APPROVAL_SCOPE,
    run_id: str = "paper-gate",
) -> MarketClusteredMeanEVV62PaperCandidateConfig:
    return MarketClusteredMeanEVV62PaperCandidateConfig(
        run_id=run_id,
        output_dir=tmp_path / "out",
        promoted_candidate_manifest_path=promoted,
        paper_handoff_plan_path=handoff,
        manual_approval_approved=approved,
        manual_approval_id="approval-1",
        manual_approval_operator="pytest",
        manual_approval_scope=scope,
        manual_approval_ts=1234,
        builder_git_commit="a" * 40,
    )


def test_v6_2_paper_candidate_gate_allows_only_bounded_paper_canary(
    tmp_path: Path, monkeypatch
) -> None:
    promoted, handoff = _fixture(tmp_path, monkeypatch)
    result = run_market_clustered_mean_ev_v6_2_paper_candidate_gate(
        _config(tmp_path, promoted, handoff, approved=True)
    )

    report = result["report"]
    payload = dict(report)
    report_id = payload.pop("report_id")
    assert canonical_json_sha256(payload) == report_id
    assert report["paper_candidate_allowed"] is True
    assert report["paper_canary_handoff_allowed"] is True
    assert report["source_model_candidate_eligible"] is True
    assert report["freeze_ready"] is True
    assert report["promotion_evidence_eligible"] is True
    assert report["v8_execution_handoff_allowed"] is False
    assert report["live_handoff_allowed"] is False
    assert report["paper_pnl_is_promotion_evidence"] is False
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False
    assert report["wallet_signing_enabled"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False

    contract = result["contract"]
    assert contract["scorer_lineage"] == "market_clustered_mean_ev_v6_2"
    assert contract["legacy_o_source_score_used"] is False
    assert contract["bounded_complete_round_count"] == 12
    assert contract["maximum_paper_order_notional"] == 0.2
    assert contract["decision_time_target_access_allowed"] is False
    assert contract["forced_coverage_bets_allowed"] is False

    manifest = result["manifest"]
    manifest_payload = dict(manifest)
    manifest_id = manifest_payload.pop("manifest_id")
    assert canonical_json_sha256(manifest_payload) == manifest_id
    assert manifest["paper_candidate_allowed"] is True
    assert manifest["paper_canary_handoff_allowed"] is True
    assert manifest["v8_execution_handoff_allowed"] is False


def test_v6_2_paper_candidate_gate_requires_explicit_approval(
    tmp_path: Path, monkeypatch
) -> None:
    promoted, handoff = _fixture(tmp_path, monkeypatch)
    result = run_market_clustered_mean_ev_v6_2_paper_candidate_gate(
        _config(tmp_path, promoted, handoff, approved=False)
    )

    report = result["report"]
    assert report["paper_candidate_allowed"] is False
    assert report["paper_canary_handoff_allowed"] is False
    assert report["paper_candidate_blocking_reason_codes"] == [
        "manual_approval_required_before_v6_2_paper_canary"
    ]
    assert result["manifest"]["v8_execution_handoff_allowed"] is False


def test_v6_2_paper_candidate_gate_rejects_wrong_approval_scope(
    tmp_path: Path, monkeypatch
) -> None:
    promoted, handoff = _fixture(tmp_path, monkeypatch)
    result = run_market_clustered_mean_ev_v6_2_paper_candidate_gate(
        _config(
            tmp_path,
            promoted,
            handoff,
            approved=True,
            scope="live_trading",
        )
    )

    assert result["report"]["paper_candidate_allowed"] is False
    assert "manual_approval_required_before_v6_2_paper_canary" in result["report"][
        "paper_candidate_blocking_reason_codes"
    ]


def test_v6_2_paper_candidate_gate_rejects_unpinned_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    promoted, handoff = _fixture(tmp_path, monkeypatch)
    promoted.write_text(promoted.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_market_clustered_mean_ev_v6_2_paper_candidate_gate(
            _config(tmp_path, promoted, handoff, approved=True)
        )
