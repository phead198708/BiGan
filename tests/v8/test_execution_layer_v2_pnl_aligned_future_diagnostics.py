from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_diagnostics import (
    PnLAlignedFutureDiagnosticsFreezeConfig,
    freeze_pnl_aligned_future_supplemental_diagnostics,
    run_pnl_aligned_future_supplemental_diagnostics,
    validate_pnl_aligned_future_supplemental_diagnostics_protocol,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pnl_aligned_future_supplemental_diagnostics_v1.json"
)
CANDIDATE = "pnl_aligned_action_conditioned_net_value_v1"
BASELINE = "raw_market_probability_selected_o_action_baseline"


def test_supplemental_diagnostics_protocol_is_frozen_and_report_only() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    validate_pnl_aligned_future_supplemental_diagnostics_protocol(protocol)
    assert protocol["primary_future_evidence_gate_mutation_allowed"] is False
    assert protocol["outcome_rows_used_for_selection_or_tuning"] is False

    drifted = json.loads(json.dumps(protocol))
    drifted["market_concentration"]["basis"] = "signed_pnl"
    with pytest.raises(ValueError, match="concentration"):
        validate_pnl_aligned_future_supplemental_diagnostics_protocol(drifted)


def test_supplemental_diagnostics_group_pnl_concentration_and_lomo(
    tmp_path: Path,
) -> None:
    freeze = _diagnostics_freeze(tmp_path)
    evaluation_path = _evaluation_artifacts(tmp_path)

    result = run_pnl_aligned_future_supplemental_diagnostics(
        run_id="supplemental-diagnostics",
        output_dir=tmp_path / "runs",
        diagnostics_freeze_manifest_path=freeze["manifest_path"],
        expected_diagnostics_freeze_manifest_sha256=freeze["manifest_sha256"],
        evaluation_manifest_path=evaluation_path,
        expected_evaluation_manifest_sha256=_sha256(evaluation_path),
    )

    report = result["report"]
    candidate = report["policy_diagnostics"][CANDIDATE]
    assert candidate["overall"]["settled_net_pnl_sum"] == pytest.approx(1.0)
    assert candidate["by_side"]["UP"]["settled_net_pnl_sum"] == pytest.approx(2.0)
    assert candidate["by_side"]["DOWN"]["settled_net_pnl_sum"] == pytest.approx(-1.0)
    concentration = candidate["market_concentration"]
    assert concentration["top_1_absolute_pnl_share"] == pytest.approx(2.0 / 3.0)
    assert concentration["top_3_absolute_pnl_share"] == pytest.approx(1.0)
    assert concentration["absolute_pnl_hhi"] == pytest.approx(5.0 / 9.0)
    largest_winner = candidate["largest_winner_dependency"]
    assert largest_winner["largest_winning_market_id"] == "market-1"
    assert largest_winner["net_pnl_after_largest_winner_removed"] == pytest.approx(-1.0)
    assert largest_winner["positive_after_largest_winner_removed"] is False
    leave_one_out = candidate["leave_one_market_out"]
    assert leave_one_out["minimum_net_pnl_after_one_market_removed"] == pytest.approx(-1.0)
    assert leave_one_out["maximum_net_pnl_after_one_market_removed"] == pytest.approx(2.0)
    assert leave_one_out["all_scenarios_positive"] is False
    assert report["primary_future_evidence_gate_passed"] is False
    assert report["supplemental_diagnostics_can_mutate_primary_gate"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False


def test_supplemental_diagnostics_rejects_policy_identity_mismatch(tmp_path: Path) -> None:
    freeze = _diagnostics_freeze(tmp_path)
    evaluation_path = _evaluation_artifacts(tmp_path, baseline_identity="different-row")

    with pytest.raises(ValueError, match="identities do not reconcile"):
        run_pnl_aligned_future_supplemental_diagnostics(
            run_id="identity-mismatch",
            output_dir=tmp_path / "runs",
            diagnostics_freeze_manifest_path=freeze["manifest_path"],
            expected_diagnostics_freeze_manifest_sha256=freeze["manifest_sha256"],
            evaluation_manifest_path=evaluation_path,
            expected_evaluation_manifest_sha256=_sha256(evaluation_path),
        )


def test_supplemental_diagnostics_rejects_hash_tamper(tmp_path: Path) -> None:
    freeze = _diagnostics_freeze(tmp_path)
    evaluation_path = _evaluation_artifacts(tmp_path)
    with evaluation_path.open("a") as handle:
        handle.write(" ")

    with pytest.raises(ValueError, match="evaluation manifest SHA-256 mismatch"):
        run_pnl_aligned_future_supplemental_diagnostics(
            run_id="hash-tamper",
            output_dir=tmp_path / "runs",
            diagnostics_freeze_manifest_path=freeze["manifest_path"],
            expected_diagnostics_freeze_manifest_sha256=freeze["manifest_sha256"],
            evaluation_manifest_path=evaluation_path,
            expected_evaluation_manifest_sha256="a" * 64,
        )


def _diagnostics_freeze(tmp_path: Path) -> dict:
    collection_path = tmp_path / "collection-freeze.json"
    _write_json(collection_path, {"collection_freeze_id": "collection"})
    evaluation_freeze_path = tmp_path / "evaluation-freeze.json"
    _write_json(
        evaluation_freeze_path,
        {
            "collection_freeze_manifest": _descriptor(collection_path),
            "future_outcome_targets_loaded": False,
            "outcome_reconciliation_started": False,
            **_safety(),
        },
    )
    return freeze_pnl_aligned_future_supplemental_diagnostics(
        PnLAlignedFutureDiagnosticsFreezeConfig(
            run_id="diagnostics-freeze",
            output_dir=tmp_path / "freezes",
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=_sha256(PROTOCOL_PATH),
            evaluation_freeze_manifest_path=evaluation_freeze_path,
            expected_evaluation_freeze_manifest_sha256=_sha256(evaluation_freeze_path),
            git_commit="a" * 40,
        )
    )


def _evaluation_artifacts(
    tmp_path: Path,
    *,
    baseline_identity: str | None = None,
) -> Path:
    rows_path = tmp_path / "accepted-bet-pnl-rows.jsonl"
    rows = [
        _pnl_row(CANDIDATE, "row-1", "market-1", "UP", 2.0),
        _pnl_row(CANDIDATE, "row-2", "market-2", "DOWN", -1.0),
        _pnl_row(BASELINE, baseline_identity or "row-1", "market-1", "UP", 0.5),
        _pnl_row(BASELINE, "row-2", "market-2", "DOWN", 0.5),
    ]
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report_path = tmp_path / "accepted-bet-pnl-report.json"
    _write_json(
        report_path,
        {
            "candidate_policy_name": CANDIDATE,
            "baseline_policy_name": BASELINE,
            "candidate_policy_metrics": {"settled_net_pnl_sum": 1.0},
            "baseline_policy_metrics": {"settled_net_pnl_sum": 1.0},
            "future_evidence_gate_passed": False,
            "future_evidence_gate_blocking_reason_codes": ["diagnostic_fixture_blocked"],
        },
    )
    evaluation_path = tmp_path / "evaluation-manifest.json"
    _write_json(
        evaluation_path,
        {
            "accepted_bet_pnl_rows": _descriptor(rows_path),
            "accepted_bet_pnl_report": _descriptor(report_path),
            "future_evidence_gate_passed": False,
            **_safety(),
        },
    )
    return evaluation_path


def _pnl_row(
    policy_name: str,
    identity: str,
    market_id: str,
    side: str,
    pnl: float,
) -> dict:
    action = f"BUY_{side}_HOLD_TO_SETTLEMENT"
    return {
        "policy_name": policy_name,
        "source_row_identity": identity,
        "market_id": market_id,
        "market_close_ts": 2_000 if market_id == "market-1" else 3_000,
        "simulated_order_id": f"{policy_name}-{market_id}",
        "execution_guard_order_allowed": True,
        "settlement_target_available": True,
        "execution_guarded_side": side,
        "execution_guarded_action": action,
        "paper_bet_contract_size": 1.0,
        "gross_pnl": pnl + 0.1,
        "execution_cost": 0.1,
        "cost_basis": 1.0,
        "settled_net_pnl": pnl,
    }


def _safety() -> dict:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
