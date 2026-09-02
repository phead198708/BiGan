from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import bigan.v8.polymarket.training.execution_layer_v2_pairwise_precollection_support as support_module
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_precollection_support import (
    PairwisePrecollectionSupportGateConfig,
    _batch_preflight,
    _continuation_decision,
    run_pairwise_precollection_support_gate,
)


def test_ready_at_initial_budget_stops_collection() -> None:
    decision = _continuation_decision(
        attempted_capture_count=210,
        initial_attempt_count=210,
        maximum_attempt_count=260,
        target_market_count=195,
        preflight_blocking_reason_codes=[],
        role_assignment_report={
            "role_assignment_ready": True,
            "selected_market_count": 195,
            "blocking_reason_codes": [],
        },
    )

    assert decision["status"] == "OUTCOME_BLIND_SUPPORT_TARGET_READY"
    assert decision["continuation_allowed"] is False
    assert decision["continuation_attempt_count"] == 0


def test_support_only_failure_allows_only_frozen_remaining_attempts() -> None:
    decision = _continuation_decision(
        attempted_capture_count=210,
        initial_attempt_count=210,
        maximum_attempt_count=260,
        target_market_count=195,
        preflight_blocking_reason_codes=[],
        role_assignment_report={
            "role_assignment_ready": False,
            "selected_market_count": 188,
            "blocking_reason_codes": [
                "insufficient_quality_valid_unique_market_support",
                "role_market_count_mismatch",
            ],
        },
    )

    assert decision["status"] == "BOUNDED_SUPPORT_CONTINUATION_ALLOWED"
    assert decision["support_only_failure"] is True
    assert decision["continuation_allowed"] is True
    assert decision["continuation_attempt_count"] == 50


def test_non_support_blocker_prevents_additional_collection() -> None:
    decision = _continuation_decision(
        attempted_capture_count=210,
        initial_attempt_count=210,
        maximum_attempt_count=260,
        target_market_count=195,
        preflight_blocking_reason_codes=[],
        role_assignment_report={
            "role_assignment_ready": False,
            "selected_market_count": 188,
            "blocking_reason_codes": [
                "earliest_quality_capture_not_finalized",
                "insufficient_quality_valid_unique_market_support",
            ],
        },
    )

    assert decision["status"] == "BLOCKED_FAIL_CLOSED"
    assert decision["continuation_allowed"] is False
    assert decision["continuation_attempt_count"] == 0
    assert "earliest_quality_capture_not_finalized" in decision[
        "blocking_reason_codes"
    ]


def test_insufficient_support_at_frozen_maximum_remains_blocked() -> None:
    decision = _continuation_decision(
        attempted_capture_count=260,
        initial_attempt_count=210,
        maximum_attempt_count=260,
        target_market_count=195,
        preflight_blocking_reason_codes=[],
        role_assignment_report={
            "role_assignment_ready": False,
            "selected_market_count": 194,
            "blocking_reason_codes": [
                "insufficient_quality_valid_unique_market_support",
                "role_market_count_mismatch",
            ],
        },
    )

    assert decision["status"] == (
        "BLOCKED_INSUFFICIENT_SUPPORT_AT_FROZEN_MAXIMUM"
    )
    assert decision["continuation_allowed"] is False
    assert decision["blocking_reason_codes"] == [
        "insufficient_support_at_frozen_maximum"
    ]


def test_exact_duplicate_batch_pin_counts_once(tmp_path: Path) -> None:
    batch = _batch_progress(tmp_path, batch_id="batch-1", count=2)
    digest = _sha256(batch)

    unique, duplicates, preflight = _batch_preflight(
        ((batch, digest), (batch, digest))
    )

    assert len(unique) == 1
    assert preflight["attempted_capture_count"] == 2
    assert preflight["capture_run_id_count"] == 2
    assert preflight["blocking_reason_codes"] == []
    assert len(duplicates) == 1
    assert duplicates[0]["reason_code"] == (
        "duplicate_exact_batch_progress_pin"
    )


def test_duplicate_capture_identity_is_fail_closed(tmp_path: Path) -> None:
    first = _batch_progress(
        tmp_path,
        batch_id="batch-1",
        count=1,
        run_id_prefix="shared",
    )
    second = _batch_progress(
        tmp_path,
        batch_id="batch-2",
        count=1,
        run_id_prefix="shared",
    )

    _, _, preflight = _batch_preflight(
        ((first, _sha256(first)), (second, _sha256(second)))
    )

    assert preflight["attempted_capture_count"] == 2
    assert "duplicate_capture_run_id" in preflight[
        "blocking_reason_codes"
    ]


def test_forbidden_outcome_field_and_collector_error_block_preflight(
    tmp_path: Path,
) -> None:
    batch = _batch_progress(
        tmp_path,
        batch_id="unsafe-batch",
        count=1,
    )
    payload = json.loads(batch.read_text(encoding="utf-8"))
    payload["error_count"] = 1
    payload["captures"][0]["settlement_pnl"] = 1.0
    batch.write_text(json.dumps(payload), encoding="utf-8")

    _, _, preflight = _batch_preflight(((batch, _sha256(batch)),))

    assert preflight["collector_batch_error_count"] == 1
    assert preflight["forbidden_batch_field_paths"]
    assert "collector_batch_error_count_nonzero" in preflight[
        "blocking_reason_codes"
    ]
    assert "batch_progress_forbidden_outcome_fields_present" in preflight[
        "blocking_reason_codes"
    ]


def test_support_gate_writes_hashable_fail_closed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "target_valid_market_count": 195,
                "initial_capture_attempt_count": 210,
                "maximum_total_capture_attempt_count": 260,
            }
        ),
        encoding="utf-8",
    )
    batch = _batch_progress(
        tmp_path,
        batch_id="batch-210",
        count=210,
    )

    def fake_role_assignment(config) -> dict:
        role_dir = Path(config.output_dir) / config.run_id
        role_dir.mkdir(parents=True)
        report_path = role_dir / "report.json"
        manifest_path = role_dir / "manifest.json"
        report = {
            "role_assignment_ready": False,
            "selected_market_count": 190,
            "excluded_capture_count": 20,
            "blocking_reason_codes": [
                "insufficient_quality_valid_unique_market_support",
                "role_market_count_mismatch",
            ],
        }
        report_path.write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps({"role_assignment_ready": False}),
            encoding="utf-8",
        )
        return {
            "report_path": report_path,
            "manifest_path": manifest_path,
            "report": report,
            "manifest": {"role_assignment_ready": False},
        }

    monkeypatch.setattr(
        support_module,
        "assign_pairwise_action_advantage_lcb_roles",
        fake_role_assignment,
    )

    result = run_pairwise_precollection_support_gate(
        PairwisePrecollectionSupportGateConfig(
            run_id="support-gate",
            output_dir=tmp_path / "runs",
            precollection_freeze_manifest_path=freeze,
            expected_precollection_freeze_manifest_sha256=(
                _sha256(freeze)
            ),
            batch_progress_pins=(
                (batch, _sha256(batch)),
                (batch, _sha256(batch)),
            ),
            training_corpus_root=tmp_path / "training",
        )
    )

    report = result["report"]
    manifest = result["manifest"]
    assert report["status"] == "BOUNDED_SUPPORT_CONTINUATION_ALLOWED"
    assert report["attempted_capture_count"] == 210
    assert report["continuation_attempt_count"] == 50
    assert report["duplicate_excluded_input_count"] == 1
    assert report["labels_or_outcomes_opened_for_continuation"] is False
    assert report["uses_oof_validation_or_confirmatory_pnl_for_continuation"] is False
    assert report["source_scores_mutated"] is False
    assert report["execution_thresholds_mutated"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert manifest["continuation_allowed"] is True
    assert manifest["labels_or_outcomes_opened_for_continuation"] is False
    assert result["report_sha256"] == _sha256(result["report_path"])
    assert result["manifest_sha256"] == _sha256(
        result["manifest_path"]
    )


def test_preflight_over_frozen_maximum_blocks_before_role_assignment() -> None:
    decision = _continuation_decision(
        attempted_capture_count=261,
        initial_attempt_count=210,
        maximum_attempt_count=260,
        target_market_count=195,
        preflight_blocking_reason_codes=[
            "frozen_maximum_capture_attempt_count_exceeded"
        ],
        role_assignment_report=None,
    )

    assert decision["status"] == "BLOCKED_FAIL_CLOSED"
    assert decision["continuation_allowed"] is False
    assert decision["blocking_reason_codes"] == [
        "frozen_maximum_capture_attempt_count_exceeded"
    ]


def _batch_progress(
    root: Path,
    *,
    batch_id: str,
    count: int,
    run_id_prefix: str | None = None,
) -> Path:
    path = root / f"{batch_id}.json"
    prefix = run_id_prefix or batch_id
    captures = [
        {"run_id": f"{prefix}-round-{index:03d}"}
        for index in range(count)
    ]
    path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "capture_count": count,
                "captures": captures,
                "error_count": 0,
                "paper_only": True,
                "capital_at_risk": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
