"""Tests for the outcome-blind hybrid prior-lineage quarantine."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.hybrid_pairwise_precollection_readiness import (
    _active_lineage_snapshot,
    _quarantine_checks,
)
from bigan.v8.polymarket.training.hybrid_pairwise_prior_lineage_quarantine import (
    HybridPairwisePriorLineageQuarantineConfig,
    build_hybrid_pairwise_prior_lineage_quarantine,
)

ROOT = Path(__file__).resolve().parents[2]
HYBRID_PROTOCOL_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_hybrid_pairwise_fresh_calibration_v1.json"
)


def test_builds_complete_outcome_blind_quarantine_for_readiness(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    result = _run(tmp_path, fixture)
    quarantine = result["quarantine"]

    assert quarantine["status"] == "prior_lineage_complete"
    assert quarantine["final"] is True
    assert quarantine["active_prior_lineage_complete"] is True
    assert quarantine["includes_issue175_through_issue179"] is True
    assert quarantine["historical_development_market_count"] == 90
    assert quarantine["terminal_batch_count"] == 2
    assert quarantine["terminal_capture_count"] == 3
    assert quarantine["terminal_capture_market_count"] == 2
    assert quarantine["terminal_empty_fail_closed_capture_count"] == 1
    assert quarantine["total_prior_unique_market_count"] == 94
    assert quarantine["maximum_prior_decision_ts"] == 3_500
    assert quarantine["minimum_future_decision_ts"] == 5_001
    assert quarantine["outcome_label_or_pnl_artifacts_opened"] is False
    assert quarantine["resolution_artifacts_opened"] is False
    assert quarantine["outcome_values_loaded"] is False
    assert quarantine["pnl_values_loaded"] is False
    _assert_blocked_safety(quarantine)

    protocol = deepcopy(_load_json(HYBRID_PROTOCOL_PATH))
    protocol["historical_development_registry"][
        "selected_market_ids_sha256"
    ] = quarantine["historical_development_market_ids_sha256"]
    checks = _quarantine_checks(
        quarantine=quarantine,
        protocol=protocol,
        freeze_created_at_ts=5_001,
    )
    assert all(checks.values())
    snapshot = _active_lineage_snapshot(result["quarantine_path"])
    assert snapshot["status"] == "prior_lineage_complete"
    assert snapshot["lineage_complete"] is True
    assert snapshot["forbidden_field_paths"] == []


def test_incomplete_supervisor_state_fails_before_quarantine(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state = _load_json(fixture["terminal_state_path"])
    state["status"] = "waiting_for_issue177_batch01"
    _write_json(fixture["terminal_state_path"], state)
    fixture["terminal_state_sha256"] = _sha256(
        fixture["terminal_state_path"]
    )

    with pytest.raises(
        ValueError,
        match="prior collection lineage is not terminal",
    ):
        _run(tmp_path, fixture)


def test_terminal_support_manifest_hash_drift_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    expected = fixture["support_manifest_sha256"]
    support = _load_json(fixture["support_manifest_path"])
    support["tampered"] = True
    _write_json(fixture["support_manifest_path"], support)

    with pytest.raises(
        ValueError,
        match="final support gate manifest SHA-256 mismatch",
    ):
        _run(
            tmp_path,
            fixture,
            support_manifest_sha256=expected,
        )


def test_forbidden_batch_outcome_field_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    batch_path = fixture["batch_paths"][0]
    batch = _load_json(batch_path)
    batch["settlement_pnl"] = 1.0
    _write_json(batch_path, batch)
    _refresh_support_chain(fixture)

    with pytest.raises(
        ValueError,
        match="terminal batch progress contains forbidden fields",
    ):
        _run(tmp_path, fixture)


def test_capture_market_identity_gap_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    batch_path = fixture["batch_paths"][0]
    batch = _load_json(batch_path)
    capture = batch["captures"][0]
    capture["run_dir"] = str(tmp_path / "missing-market-run")
    capture["raw_polymarket_market_count"] = 1
    _write_json(batch_path, batch)
    _refresh_support_chain(fixture)

    with pytest.raises(
        ValueError,
        match="capture market identity is incomplete",
    ):
        _run(tmp_path, fixture)


def test_historical_registry_market_hash_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    descriptor = _load_json(fixture["registry_descriptor_path"])
    descriptor["selected_market_ids_sha256"] = "f" * 64
    _write_json(fixture["registry_descriptor_path"], descriptor)
    fixture["registry_descriptor_sha256"] = _sha256(
        fixture["registry_descriptor_path"]
    )

    with pytest.raises(
        ValueError,
        match="historical registry market ID hash mismatch",
    ):
        _run(tmp_path, fixture)


def test_support_gate_unlock_flag_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    report = _load_json(fixture["support_report_path"])
    report["v8_execution_handoff_allowed"] = True
    _write_json(fixture["support_report_path"], report)
    _refresh_support_chain(fixture)

    with pytest.raises(
        ValueError,
        match="support gate report safety contract failed",
    ):
        _run(tmp_path, fixture)


def test_quarantine_creation_must_follow_all_prior_decisions(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(
        ValueError,
        match="creation timestamp must follow all prior decisions",
    ):
        _run(tmp_path, fixture, created_at_ts=3_500)


def test_terminal_blocked_support_chain_can_be_quarantined_without_unlock(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        terminal_status="blocked_fail_closed",
        support_status="BLOCKED_INSUFFICIENT_SUPPORT_AT_FROZEN_MAXIMUM",
    )

    result = _run(tmp_path, fixture)
    quarantine = result["quarantine"]

    assert quarantine["terminal_lineage_status"] == "blocked_fail_closed"
    assert (
        quarantine["terminal_support_gate_status"]
        == "BLOCKED_INSUFFICIENT_SUPPORT_AT_FROZEN_MAXIMUM"
    )
    assert quarantine["active_prior_lineage_complete"] is True
    _assert_blocked_safety(quarantine)


def test_output_is_deterministic_and_overwrite_removes_old_contents(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    first = _run(tmp_path, fixture, run_id="same")
    stale = Path(first["run_dir"]) / "stale.json"
    stale.write_text("{}\n", encoding="utf-8")

    second = _run(
        tmp_path,
        fixture,
        run_id="same",
        overwrite=True,
    )

    assert second["quarantine_sha256"] == first["quarantine_sha256"]
    assert second["manifest_sha256"] == first["manifest_sha256"]
    assert not stale.exists()


def _fixture(
    tmp_path: Path,
    *,
    terminal_status: str = "issue179_role_assignment_ready",
    support_status: str = "OUTCOME_BLIND_SUPPORT_TARGET_READY",
) -> dict[str, Any]:
    historical_ids = [f"historical-{index:03d}" for index in range(90)]
    registry_rows = [
        {
            "selection_rank": index + 1,
            "market_id": market_id,
            "maximum_decision_ts": 1_000 + index,
            "corpus_dir": str(tmp_path / "historical" / market_id),
            "artifact_pins": {
                "polymarket_label_rows.jsonl": {
                    "path": str(tmp_path / "never-open-labels.jsonl"),
                    "semantic_content_parsed": False,
                    "sha256": "a" * 64,
                },
                "polymarket_resolution_events.jsonl": {
                    "path": str(tmp_path / "never-open-resolution.jsonl"),
                    "semantic_content_parsed": False,
                    "sha256": "b" * 64,
                },
            },
            "labels_or_outcomes_used_for_selection": False,
            "outcome_values_loaded": False,
            "pnl_values_loaded": False,
        }
        for index, market_id in enumerate(historical_ids)
    ]
    registry_rows_path = tmp_path / "historical-registry-rows.jsonl"
    _write_jsonl(registry_rows_path, registry_rows)
    registry_descriptor = {
        "schema_version": "fixture-historical-registry-descriptor-v1",
        "selected_market_count": 90,
        "selected_market_ids_sha256": canonical_json_sha256(
            historical_ids
        ),
    }
    registry_descriptor_path = tmp_path / "historical-registry.json"
    _write_json(registry_descriptor_path, registry_descriptor)

    prior_ids = ["prior-a", "prior-b"]
    prior_registry = {
        "schema_version": "fixture-prior-registry-v1",
        "run_id": "fixture-prior-registry",
        "prior_market_ids": prior_ids,
        "prior_market_ids_sha256": canonical_json_sha256(prior_ids),
        "maximum_prior_decision_ts": 1_500,
        "market_entries": [
            {"market_id": "prior-a", "decision_ts": 1_400},
            {"market_id": "prior-b", "decision_ts": 1_500},
        ],
        "prior_outcome_or_pnl_values_loaded": False,
        "prior_validation_or_future_evidence_used_for_tuning": False,
        **_blocked_safety_fields(),
    }
    prior_registry_path = tmp_path / "prior-registry.json"
    _write_json(prior_registry_path, prior_registry)
    precollection_freeze = {
        "schema_version": "fixture-precollection-freeze-v1",
        "prior_evidence_exclusion_registry": _descriptor(
            prior_registry_path
        ),
        **_blocked_safety_fields(),
    }
    precollection_freeze_path = tmp_path / "precollection-freeze.json"
    _write_json(precollection_freeze_path, precollection_freeze)

    run_a = _raw_market_run(
        tmp_path,
        name="capture-a",
        market_id="fresh-a",
        market_end_ts=3_000,
    )
    run_b = _raw_market_run(
        tmp_path,
        name="capture-b",
        market_id="fresh-b",
        market_end_ts=3_500,
    )
    batch_one = {
        "batch_id": "batch-1",
        "paper_only": True,
        "capital_at_risk": False,
        "error_count": 0,
        "capture_count": 2,
        "captures": [
            {
                "run_id": "capture-a",
                "run_dir": str(run_a),
                "capture_status": "pending_resolution",
                "scheduled_round_start_ts": 2_900,
                "raw_polymarket_market_count": 1,
                "reject_reason_counts": {},
            },
            {
                "run_id": "capture-empty",
                "run_dir": str(tmp_path / "capture-empty"),
                "capture_status": "blocked_fail_closed",
                "scheduled_round_start_ts": 3_100,
                "raw_polymarket_market_count": 0,
                "reject_reason_counts": {"read_only_public_http_timeout": 1},
            },
        ],
    }
    batch_two = {
        "batch_id": "batch-2",
        "paper_only": True,
        "capital_at_risk": False,
        "error_count": 0,
        "capture_count": 1,
        "captures": [
            {
                "run_id": "capture-b",
                "run_dir": str(run_b),
                "capture_status": "exported",
                "scheduled_round_start_ts": 3_400,
                "raw_polymarket_market_count": 1,
                "reject_reason_counts": {},
            }
        ],
    }
    batch_one_path = tmp_path / "batch-1.json"
    batch_two_path = tmp_path / "batch-2.json"
    _write_json(batch_one_path, batch_one)
    _write_json(batch_two_path, batch_two)

    support_report = {
        "schema_version": "fixture-support-report-v1",
        "status": support_status,
        "unique_batch_progress_count": 2,
        "continuation_allowed": False,
        "continuation_required": False,
        "labels_or_outcomes_opened_for_continuation": False,
        "settlement_pnl_opened_for_continuation": False,
        "uses_oof_validation_or_confirmatory_pnl_for_continuation": False,
        **_blocked_safety_fields(),
    }
    support_report_path = tmp_path / "support-report.json"
    _write_json(support_report_path, support_report)
    support_manifest = {
        "schema_version": "fixture-support-manifest-v1",
        "precollection_freeze_manifest": _descriptor(
            precollection_freeze_path
        ),
        "support_gate_report": _descriptor(support_report_path),
        "batch_progress_inputs": [
            _descriptor(batch_one_path),
            _descriptor(batch_two_path),
        ],
        **_blocked_safety_fields(),
    }
    support_manifest_path = tmp_path / "support-manifest.json"
    _write_json(support_manifest_path, support_manifest)
    terminal_state = {
        "status": terminal_status,
        "completed_batch_count": 2,
        "capture_attempt_count": 3,
        "support_gate_status": support_status,
        "support_gate_report_path": str(support_report_path.resolve()),
        "support_gate_report_sha256": _sha256(support_report_path),
        "support_gate_manifest_path": str(support_manifest_path.resolve()),
        "support_gate_manifest_sha256": _sha256(support_manifest_path),
    }
    terminal_state_path = tmp_path / "terminal-state.json"
    _write_json(terminal_state_path, terminal_state)
    return {
        "registry_descriptor_path": registry_descriptor_path,
        "registry_descriptor_sha256": _sha256(registry_descriptor_path),
        "registry_rows_path": registry_rows_path,
        "registry_rows_sha256": _sha256(registry_rows_path),
        "terminal_state_path": terminal_state_path,
        "terminal_state_sha256": _sha256(terminal_state_path),
        "support_report_path": support_report_path,
        "support_manifest_path": support_manifest_path,
        "support_manifest_sha256": _sha256(support_manifest_path),
        "batch_paths": (batch_one_path, batch_two_path),
    }


def _refresh_support_chain(fixture: dict[str, Any]) -> None:
    support_manifest = _load_json(fixture["support_manifest_path"])
    support_manifest["support_gate_report"] = _descriptor(
        fixture["support_report_path"]
    )
    support_manifest["batch_progress_inputs"] = [
        _descriptor(path) for path in fixture["batch_paths"]
    ]
    _write_json(fixture["support_manifest_path"], support_manifest)
    fixture["support_manifest_sha256"] = _sha256(
        fixture["support_manifest_path"]
    )
    terminal_state = _load_json(fixture["terminal_state_path"])
    terminal_state["support_gate_report_sha256"] = _sha256(
        fixture["support_report_path"]
    )
    terminal_state["support_gate_manifest_sha256"] = fixture[
        "support_manifest_sha256"
    ]
    _write_json(fixture["terminal_state_path"], terminal_state)
    fixture["terminal_state_sha256"] = _sha256(
        fixture["terminal_state_path"]
    )


def _run(
    tmp_path: Path,
    fixture: dict[str, Any],
    *,
    run_id: str = "quarantine",
    created_at_ts: int = 5_000,
    overwrite: bool = False,
    support_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    return build_hybrid_pairwise_prior_lineage_quarantine(
        HybridPairwisePriorLineageQuarantineConfig(
            run_id=run_id,
            output_dir=tmp_path / "runs",
            created_at_ts=created_at_ts,
            historical_registry_descriptor_path=fixture[
                "registry_descriptor_path"
            ],
            expected_historical_registry_descriptor_sha256=fixture[
                "registry_descriptor_sha256"
            ],
            historical_registry_rows_path=fixture["registry_rows_path"],
            expected_historical_registry_rows_sha256=fixture[
                "registry_rows_sha256"
            ],
            terminal_lineage_state_path=fixture["terminal_state_path"],
            expected_terminal_lineage_state_sha256=fixture[
                "terminal_state_sha256"
            ],
            final_support_gate_manifest_path=fixture[
                "support_manifest_path"
            ],
            expected_final_support_gate_manifest_sha256=(
                support_manifest_sha256
                if support_manifest_sha256 is not None
                else fixture["support_manifest_sha256"]
            ),
            overwrite_existing=overwrite,
        )
    )


def _raw_market_run(
    tmp_path: Path,
    *,
    name: str,
    market_id: str,
    market_end_ts: int,
) -> Path:
    run_dir = tmp_path / name
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    _write_jsonl(
        raw_dir / "raw_polymarket_markets.jsonl",
        [
            {
                "market_id": market_id,
                "market_end_ts": market_end_ts,
                "outcome_tokens": ["UP", "DOWN"],
            }
        ],
    )
    return run_dir


def _assert_blocked_safety(payload: dict[str, Any]) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
    assert payload["source_model_candidate_eligible"] is False
    assert payload["freeze_ready"] is False
    assert payload["promotion_evidence_eligible"] is False
    assert payload["v8_execution_handoff_allowed"] is False
    assert payload["#134_resume_allowed"] is False
    assert payload["#146_start_allowed"] is False


def _blocked_safety_fields() -> dict[str, Any]:
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
