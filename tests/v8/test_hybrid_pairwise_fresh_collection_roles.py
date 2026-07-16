"""Tests for #185 authorized collection and outcome-blind 45/60 roles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.hybrid_pairwise_fresh_collection_roles import (
    AUTHORIZATION_SCHEMA_VERSION,
    HybridFreshCollectionStartGateConfig,
    HybridFreshCollectionSupportGateConfig,
    _assign_hybrid_fresh_roles,
    _role_chronology_passed,
    evaluate_hybrid_fresh_collection_start_gate,
    run_hybrid_fresh_collection_support_gate,
)
from bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration import (
    CALIBRATION_MARKET_COUNT,
    CALIBRATION_ROLE,
    CONFIRMATORY_MARKET_COUNT,
    CONFIRMATORY_ROLE,
    _validate_fresh_role_lineage,
    _validate_role_assignment,
)

ROOT = Path(__file__).resolve().parents[2]
HYBRID_PROTOCOL_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_hybrid_pairwise_fresh_calibration_v1.json"
)
SOURCE_PROTOCOL_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_v1.json"
)
FEATURE_CONTRACT_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)
COLLECTOR_GIT_COMMIT = "1" * 40
AUTHORIZATION_SCHEMA_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_hybrid_pairwise_fresh_collection_"
    "authorization_schema_v1.json"
)


def test_committed_authorization_schema_freezes_120_150_and_safety() -> None:
    schema = _load_json(AUTHORIZATION_SCHEMA_PATH)
    properties = schema["properties"]

    assert schema["$id"] == AUTHORIZATION_SCHEMA_VERSION
    assert properties["issue_number"]["const"] == 185
    assert (
        properties["collection_plan"]["properties"][
            "initial_capture_attempt_count"
        ]["const"]
        == 120
    )
    assert (
        properties["collection_plan"]["properties"][
            "maximum_total_capture_attempt_count"
        ]["const"]
        == 150
    )
    assert properties["paper_only"]["const"] is True
    assert properties["capital_at_risk"]["const"] is False
    assert properties["polymarket_write_enabled"]["const"] is False
    assert properties["wallet_signing_enabled"]["const"] is False


def test_missing_terminal_freeze_blocks_without_command(tmp_path: Path) -> None:
    fixture = _start_fixture(tmp_path)
    readiness = _load_json(fixture["readiness_path"])
    readiness["precollection_readiness_passed"] = False
    readiness["precollection_freeze_created"] = False
    readiness["precollection_freeze_manifest"] = None
    _write_json(fixture["readiness_path"], readiness)
    fixture["readiness_sha256"] = _sha256(fixture["readiness_path"])

    result = evaluate_hybrid_fresh_collection_start_gate(
        HybridFreshCollectionStartGateConfig(
            run_id="missing-terminal-freeze",
            output_dir=tmp_path / "runs",
            readiness_manifest_path=fixture["readiness_path"],
            expected_readiness_manifest_sha256=fixture[
                "readiness_sha256"
            ],
            collector_script_path=fixture["collector_script_path"],
            expected_collector_script_sha256=fixture[
                "collector_script_sha256"
            ],
            collector_git_commit=COLLECTOR_GIT_COMMIT,
        )
    )

    assert result["report"]["collection_start_allowed"] is False
    assert result["report"]["collection_start_command_generated"] is False
    assert "issue183_terminal_readiness_not_passed" in result["report"][
        "blocking_reason_codes"
    ]
    assert result["launch_plan_path"] is None


def test_missing_explicit_authorization_writes_blocked_start_artifacts(
    tmp_path: Path,
) -> None:
    fixture = _start_fixture(tmp_path)

    result = _run_start(tmp_path, fixture, authorization=False)
    report = result["report"]

    assert report["status"] == "blocked_fail_closed"
    assert report["collection_start_allowed"] is False
    assert report["collection_start_command_generated"] is False
    assert report["collector_execution_attempted"] is False
    assert result["launch_plan_path"] is None
    assert (
        "explicit_issue185_collection_authorization_missing"
        in report["blocking_reason_codes"]
    )
    _assert_blocked_safety(report)


def test_authorized_fixture_generates_deterministic_launch_plan_only(
    tmp_path: Path,
) -> None:
    fixture = _start_fixture(tmp_path)

    first = _run_start(tmp_path, fixture, run_id="authorized")
    plan = first["launch_plan"]
    assert plan is not None
    assert first["report"]["collection_start_allowed"] is True
    assert plan["initial_capture_attempt_count"] == 120
    assert plan["maximum_total_capture_attempt_count"] == 150
    assert plan["maximum_continuation_attempt_count"] == 30
    assert plan["market_family"] == "btc_updown_5m"
    assert plan["orderbook_websocket_primary"] is True
    assert plan["causal_rest_orderbook_fallback_only"] is True
    assert plan["collector_execution_attempted"] is False
    assert "--round-count" in plan["initial_collection_command_argv"]
    assert "120" in plan["initial_collection_command_argv"]
    _assert_blocked_safety(plan)

    second = _run_start(
        tmp_path,
        fixture,
        run_id="authorized",
        overwrite=True,
    )
    assert second["launch_plan_sha256"] == first["launch_plan_sha256"]
    assert second["report_sha256"] == first["report_sha256"]


def test_start_gate_hash_drift_fails_before_launch_plan(
    tmp_path: Path,
) -> None:
    fixture = _start_fixture(tmp_path)
    expected = fixture["collector_script_sha256"]
    fixture["collector_script_path"].write_text(
        "#!/usr/bin/env python3\nprint('changed')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="collector script SHA-256 mismatch"):
        _run_start(
            tmp_path,
            fixture,
            collector_script_sha256=expected,
        )


def test_frozen_ranker_model_hash_drift_fails_before_launch_plan(
    tmp_path: Path,
) -> None:
    fixture = _start_fixture(tmp_path)
    fixture["ranker_model_path"].write_text(
        '{"ranker":"tampered"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="historical ranker model SHA-256 mismatch",
    ):
        _run_start(tmp_path, fixture)


def test_initial_support_is_not_evaluated_before_all_120_attempts(
    tmp_path: Path,
) -> None:
    fixture = _authorized_fixture(tmp_path)
    batch = _batch_progress(tmp_path, fixture, count=119)

    result = _run_support(tmp_path, fixture, pins=((batch, _sha256(batch)),))
    report = result["report"]

    assert report["status"] == "initial_collection_incomplete"
    assert report["attempted_capture_count"] == 119
    assert report["role_assignment_attempted"] is False
    assert report["continuation_allowed"] is False
    assert report["continuation_attempt_count"] == 0
    assert report["blocking_reason_codes"] == [
        "frozen_initial_capture_attempts_incomplete"
    ]
    assert result["continuation_manifest_path"] is None


def test_support_shortfall_after_120_allows_only_frozen_30_remainder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _authorized_fixture(tmp_path)
    batch = _batch_progress(tmp_path, fixture, count=120)
    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_fresh_collection_roles."
        "_assign_hybrid_fresh_roles",
        _fake_role_assignment(
            tmp_path,
            selected_market_count=100,
            blockers=[
                "insufficient_quality_valid_unique_market_support",
                "role_market_count_mismatch",
            ],
        ),
    )

    result = _run_support(tmp_path, fixture, pins=((batch, _sha256(batch)),))
    report = result["report"]

    assert report["status"] == "bounded_support_continuation_allowed"
    assert report["continuation_allowed"] is True
    assert report["continuation_attempt_count"] == 30
    assert result["continuation_manifest_path"].is_file()
    assert result["continuation_manifest"][
        "labels_or_outcomes_opened_for_continuation"
    ] is False
    _assert_blocked_safety(report)


def test_non_support_failure_cannot_request_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _authorized_fixture(tmp_path)
    batch = _batch_progress(tmp_path, fixture, count=120)
    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_fresh_collection_roles."
        "_assign_hybrid_fresh_roles",
        _fake_role_assignment(
            tmp_path,
            selected_market_count=100,
            blockers=["calibration_confirmatory_chronology_failed"],
        ),
    )

    result = _run_support(tmp_path, fixture, pins=((batch, _sha256(batch)),))

    assert result["report"]["status"] == "blocked_fail_closed"
    assert result["report"]["continuation_allowed"] is False
    assert result["report"]["continuation_attempt_count"] == 0
    assert result["continuation_manifest_path"] is None


def test_duplicate_batch_pin_does_not_inflate_attempt_support(
    tmp_path: Path,
) -> None:
    fixture = _authorized_fixture(tmp_path)
    batch = _batch_progress(tmp_path, fixture, count=60)
    pin = (batch, _sha256(batch))

    result = _run_support(tmp_path, fixture, pins=(pin, pin))

    assert result["report"]["attempted_capture_count"] == 60
    assert result["report"]["duplicate_excluded_input_count"] == 1
    assert result["report"]["selected_market_count"] == 0
    assert result["report"]["continuation_allowed"] is False


def test_duplicate_capture_run_ids_block_without_continuation(
    tmp_path: Path,
) -> None:
    fixture = _authorized_fixture(tmp_path)
    batch = _batch_progress(tmp_path, fixture, count=120)
    payload = _load_json(batch)
    payload["captures"][1]["run_id"] = payload["captures"][0]["run_id"]
    _write_json(batch, payload)

    result = _run_support(tmp_path, fixture, pins=((batch, _sha256(batch)),))

    assert "duplicate_capture_run_id" in result["report"][
        "blocking_reason_codes"
    ]
    assert result["report"]["continuation_allowed"] is False


def test_forbidden_outcome_field_blocks_without_continuation(
    tmp_path: Path,
) -> None:
    fixture = _authorized_fixture(tmp_path)
    batch = _batch_progress(tmp_path, fixture, count=120)
    payload = _load_json(batch)
    payload["settlement_pnl"] = 1.0
    _write_json(batch, payload)

    result = _run_support(tmp_path, fixture, pins=((batch, _sha256(batch)),))

    assert result["report"]["status"] == "blocked_fail_closed"
    assert (
        "batch_progress_forbidden_outcome_fields_present"
        in result["report"]["blocking_reason_codes"]
    )
    assert result["report"]["continuation_allowed"] is False


def test_frozen_maximum_150_attempts_is_enforced(
    tmp_path: Path,
) -> None:
    fixture = _authorized_fixture(tmp_path)
    first = _batch_progress(
        tmp_path,
        fixture,
        count=120,
        batch_number=1,
    )
    second = _batch_progress(
        tmp_path,
        fixture,
        count=31,
        batch_number=2,
    )

    result = _run_support(
        tmp_path,
        fixture,
        pins=((first, _sha256(first)), (second, _sha256(second))),
    )

    assert result["report"]["attempted_capture_count"] == 151
    assert (
        "frozen_maximum_capture_attempt_count_exceeded"
        in result["report"]["blocking_reason_codes"]
    )
    assert result["report"]["continuation_allowed"] is False


def test_role_assignment_matches_exact_184_schema_and_excludes_prior_market(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _start_fixture(tmp_path)
    training_root = tmp_path / "training"
    training_root.mkdir()
    captures = []
    finalizations = []
    for index in range(120):
        corpus_dir = training_root / f"corpus-{index:03d}"
        corpus_dir.mkdir()
        captures.append(
            {
                "run_id": f"capture-{index:03d}",
                "round_index": index + 1,
                "scheduled_round_start_ts": 20_000 + index * 10,
            }
        )
        finalizations.append(
            {
                "run_id": f"capture-{index:03d}",
                "exported_training_corpus_dir": str(corpus_dir),
            }
        )
    batch = tmp_path / "role-batch.json"
    _write_json(
        batch,
        {
            "batch_id": "authorized-role-batch",
            "capture_count": 120,
            "captures": captures,
            "finalizations": finalizations,
        },
    )

    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_fresh_collection_roles._capture_quality_audit",
        lambda capture, **_: {
            "capture_run_id": capture["run_id"],
            "capture_round_index": capture["round_index"],
            "scheduled_round_start_ts": capture["scheduled_round_start_ts"],
            "source_batch_id": capture["source_batch_id"],
            "source_batch_ordinal": capture["source_batch_ordinal"],
            "source_batch_progress_sha256": capture[
                "source_batch_progress_sha256"
            ],
            "reason_codes": [],
        },
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_fresh_collection_roles._finalization_quality_reasons",
        lambda _: [],
    )

    def corpus_audit(
        *,
        corpus_dir: Path,
        prior_market_ids: set[str],
        minimum_decision_ts: int,
    ) -> dict[str, Any]:
        del prior_market_ids
        index = int(corpus_dir.name.rsplit("-", maxsplit=1)[1])
        market_id = "prior-000" if index == 0 else f"fresh-{index:03d}"
        timestamp = minimum_decision_ts + 100 + index * 10
        manifest = corpus_dir / "polymarket_corpus_manifest.json"
        features = corpus_dir / "polymarket_feature_rows.jsonl"
        _write_json(manifest, {"market_id": market_id})
        features.write_text("{}\n", encoding="utf-8")
        return {
            "market_id": market_id,
            "minimum_decision_ts": timestamp,
            "maximum_decision_ts": timestamp + 1,
            "decision_row_count": 1,
            "corpus_manifest": _descriptor(manifest),
            "feature_rows": _descriptor(features),
            "reason_codes": [],
        }

    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_fresh_collection_roles."
        "_outcome_blind_corpus_role_audit",
        corpus_audit,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_fresh_collection_roles."
        "_execution_compatibility_audit",
        lambda **_: {
            "decision_row_count": 1,
            "execution_compatible_row_count": 1,
            "blocking_reason_codes": [],
        },
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_fresh_collection_roles._load_jsonl",
        lambda path: [{"slug": path.parent.name}],
    )
    freeze = _load_json(fixture["freeze_path"])
    run_dir = tmp_path / "role-output"
    run_dir.mkdir()
    launch_plan_path = tmp_path / "launch-plan.json"
    _write_json(launch_plan_path, {"launch_plan_id": "a" * 64})
    result = _assign_hybrid_fresh_roles(
        run_id="role-test",
        run_dir=run_dir,
        freeze=freeze,
        freeze_path=fixture["freeze_path"],
        launch_plan={"launch_plan_id": "a" * 64},
        launch_plan_path=launch_plan_path,
        batch_progress_pins=((batch, _sha256(batch)),),
        collector_contract=_load_json(SOURCE_PROTOCOL_PATH)[
            "collector_contract"
        ],
        training_root=training_root,
    )
    manifest = result["manifest"]
    rows = _jsonl(result["selected_rows_path"])

    _validate_role_assignment(manifest, rows)
    _validate_fresh_role_lineage(
        role_manifest=manifest,
        role_rows=rows,
        hybrid_protocol=_load_json(HYBRID_PROTOCOL_PATH),
        precollection_freeze_descriptor=_descriptor(fixture["freeze_path"]),
        final_quarantine_descriptor=_descriptor(
            fixture["quarantine_path"]
        ),
    )
    assert manifest["role_assignment_ready"] is True
    assert manifest["selected_market_count"] == 105
    assert Counter(row["role"] for row in rows) == Counter(
        {
            CALIBRATION_ROLE: CALIBRATION_MARKET_COUNT,
            CONFIRMATORY_ROLE: CONFIRMATORY_MARKET_COUNT,
        }
    )
    assert all(row["market_id"] != "prior-000" for row in rows)
    assert manifest["prior_market_overlap_count"] == 0
    assert manifest["role_market_overlap_count"] == 0
    assert manifest["chronology_validation_passed"] is True
    assert manifest["confirmatory_labels_opened"] is False
    assert all(
        row["labels_or_outcomes_opened_for_role_assignment"] is False
        for row in rows
    )
    _assert_blocked_safety(manifest)


def test_role_chronology_rejects_overlapping_calibration_and_confirmatory() -> None:
    rows = [
        {
            "role": CALIBRATION_ROLE,
            "minimum_decision_ts": index + 1,
            "maximum_decision_ts": index + 1,
        }
        for index in range(CALIBRATION_MARKET_COUNT)
    ]
    rows.extend(
        {
            "role": CONFIRMATORY_ROLE,
            "minimum_decision_ts": 10 + index,
            "maximum_decision_ts": 10 + index,
        }
        for index in range(CONFIRMATORY_MARKET_COUNT)
    )

    assert _role_chronology_passed(rows) is False


def _start_fixture(tmp_path: Path) -> dict[str, Any]:
    hybrid = deepcopy(_load_json(HYBRID_PROTOCOL_PATH))
    registry_path = tmp_path / "historical-registry.json"
    ranker_descriptor_path = tmp_path / "ranker-descriptor.json"
    ranker_manifest_path = tmp_path / "ranker-manifest.json"
    ranker_model_path = tmp_path / "ranker-model.json"
    _write_json(registry_path, {"registry": "fixture"})
    _write_json(ranker_model_path, {"ranker": "fixture"})
    ranker_identity = {
        "freeze_id": "f" * 64,
        "model_sha256": _sha256(ranker_model_path),
        "dataset_hash": "a" * 64,
        "oof_dataset_hash": "b" * 64,
        "split_hash": "c" * 64,
        "model_config_hash": "d" * 64,
    }
    _write_json(
        ranker_manifest_path,
        {
            **ranker_identity,
            "rank_scores_execution_eligible": False,
        },
    )
    _write_json(
        ranker_descriptor_path,
        {
            **ranker_identity,
            "model": _descriptor(ranker_model_path),
            "freeze_manifest": _descriptor(ranker_manifest_path),
            "rank_scores_execution_eligible": False,
        },
    )
    hybrid["historical_ranker_freeze"] = {
        **hybrid["historical_ranker_freeze"],
        **ranker_identity,
        "descriptor_sha256": _sha256(ranker_descriptor_path),
    }
    hybrid_protocol_path = tmp_path / "hybrid-protocol.json"
    _write_json(hybrid_protocol_path, hybrid)
    quarantine = {
        "schema_version": (
            "bigan-v8-hybrid-pairwise-prior-lineage-quarantine-v1"
        ),
        "status": "prior_lineage_complete",
        "final": True,
        "active_prior_lineage_complete": True,
        "includes_issue175_through_issue179": True,
        "prior_market_ids": [f"prior-{index:03d}" for index in range(90)],
        "maximum_prior_decision_ts": 1_000,
        **_blocked_safety_fields(),
    }
    quarantine["prior_market_ids_sha256"] = canonical_json_sha256(
        quarantine["prior_market_ids"]
    )
    quarantine_path = tmp_path / "final-quarantine.json"
    _write_json(quarantine_path, quarantine)
    freeze = {
        "schema_version": (
            "bigan-v8-hybrid-pairwise-precollection-freeze-manifest-v1"
        ),
        "run_id": "fixture-freeze",
        "candidate_lineage": hybrid["candidate_lineage"],
        "freeze_created_at_ts": 2_000,
        "minimum_collection_decision_ts": 2_001,
        "hybrid_protocol": _descriptor(hybrid_protocol_path),
        "source_pairwise_protocol": _descriptor(SOURCE_PROTOCOL_PATH),
        "source_feature_contract": _descriptor(FEATURE_CONTRACT_PATH),
        "historical_registry_descriptor": _descriptor(registry_path),
        "historical_ranker_descriptor": _descriptor(
            ranker_descriptor_path
        ),
        "historical_ranker_manifest": _descriptor(ranker_manifest_path),
        "final_prior_lineage_quarantine": _descriptor(quarantine_path),
        "fresh_role_plan": hybrid["fresh_role_plan"],
        "collection_plan": hybrid["collection_plan"],
        "ranker_retraining_allowed": False,
        "ranker_score_mutation_allowed": False,
        "labels_or_outcomes_opened_for_role_assignment": False,
        "confirmatory_labels_opened": False,
        **_blocked_safety_fields(),
    }
    freeze_path = tmp_path / "precollection-freeze.json"
    _write_json(freeze_path, freeze)
    readiness = {
        "schema_version": (
            "bigan-v8-hybrid-pairwise-precollection-readiness-manifest-v1"
        ),
        "precollection_readiness_passed": True,
        "precollection_freeze_created": True,
        "precollection_freeze_manifest": _descriptor(freeze_path),
        "collection_start_allowed": False,
        "collection_start_command_generated": False,
        **_blocked_safety_fields(),
    }
    readiness_path = tmp_path / "readiness-manifest.json"
    _write_json(readiness_path, readiness)
    collector_script_path = tmp_path / "collector.py"
    collector_script_path.write_text(
        "#!/usr/bin/env python3\n",
        encoding="utf-8",
    )
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "issue_number": 185,
        "authorized": True,
        "authorization_created_at_ts": 2_001,
        "read_only_public_provider": True,
        "collector_execution_authorized": True,
        "hybrid_precollection_freeze": _descriptor(freeze_path),
        "final_prior_lineage_quarantine": _descriptor(quarantine_path),
        "collector_script_sha256": _sha256(collector_script_path),
        "collector_git_commit": COLLECTOR_GIT_COMMIT,
        "collection_plan": {
            "market_family": "btc_updown_5m",
            "initial_capture_attempt_count": 120,
            "maximum_total_capture_attempt_count": 150,
        },
        "collection_output_dir": str(tmp_path / "collection"),
        "market_identity_cache_path": str(tmp_path / "identity-cache.json"),
        "batch_id_prefix": "issue185-fixture",
        **_blocked_safety_fields(),
    }
    authorization_path = tmp_path / "authorization.json"
    _write_json(authorization_path, authorization)
    return {
        "readiness_path": readiness_path,
        "readiness_sha256": _sha256(readiness_path),
        "freeze_path": freeze_path,
        "freeze_sha256": _sha256(freeze_path),
        "quarantine_path": quarantine_path,
        "quarantine_sha256": _sha256(quarantine_path),
        "collector_script_path": collector_script_path,
        "collector_script_sha256": _sha256(collector_script_path),
        "authorization_path": authorization_path,
        "authorization_sha256": _sha256(authorization_path),
        "hybrid_protocol_path": hybrid_protocol_path,
        "ranker_model_path": ranker_model_path,
    }


def _authorized_fixture(tmp_path: Path) -> dict[str, Any]:
    fixture = _start_fixture(tmp_path)
    start = _run_start(tmp_path, fixture, run_id="authorized-start")
    fixture["launch_plan_path"] = start["launch_plan_path"]
    fixture["launch_plan_sha256"] = start["launch_plan_sha256"]
    fixture["launch_plan"] = start["launch_plan"]
    return fixture


def _run_start(
    tmp_path: Path,
    fixture: dict[str, Any],
    *,
    run_id: str = "start-gate",
    authorization: bool = True,
    overwrite: bool = False,
    collector_script_sha256: str | None = None,
) -> dict[str, Any]:
    return evaluate_hybrid_fresh_collection_start_gate(
        HybridFreshCollectionStartGateConfig(
            run_id=run_id,
            output_dir=tmp_path / "runs",
            readiness_manifest_path=fixture["readiness_path"],
            expected_readiness_manifest_sha256=fixture[
                "readiness_sha256"
            ],
            collector_script_path=fixture["collector_script_path"],
            expected_collector_script_sha256=(
                collector_script_sha256
                or fixture["collector_script_sha256"]
            ),
            collector_git_commit=COLLECTOR_GIT_COMMIT,
            precollection_freeze_manifest_path=fixture["freeze_path"],
            expected_precollection_freeze_manifest_sha256=fixture[
                "freeze_sha256"
            ],
            final_prior_quarantine_path=fixture["quarantine_path"],
            expected_final_prior_quarantine_sha256=fixture[
                "quarantine_sha256"
            ],
            authorization_path=(
                fixture["authorization_path"] if authorization else None
            ),
            expected_authorization_sha256=(
                fixture["authorization_sha256"] if authorization else None
            ),
            overwrite_existing=overwrite,
        )
    )


def _run_support(
    tmp_path: Path,
    fixture: dict[str, Any],
    *,
    pins: tuple[tuple[Path, str], ...],
) -> dict[str, Any]:
    return run_hybrid_fresh_collection_support_gate(
        HybridFreshCollectionSupportGateConfig(
            run_id="support",
            output_dir=tmp_path / "support-runs",
            collection_launch_plan_path=fixture["launch_plan_path"],
            expected_collection_launch_plan_sha256=fixture[
                "launch_plan_sha256"
            ],
            precollection_freeze_manifest_path=fixture["freeze_path"],
            expected_precollection_freeze_manifest_sha256=fixture[
                "freeze_sha256"
            ],
            batch_progress_pins=pins,
            training_corpus_root="/Volumes/PHILIPS/v8",
        )
    )


def _batch_progress(
    tmp_path: Path,
    fixture: dict[str, Any],
    *,
    count: int,
    batch_number: int = 1,
) -> Path:
    prefix = fixture["launch_plan"]["batch_id_prefix"]
    path = tmp_path / f"batch-{batch_number}.json"
    _write_json(
        path,
        {
            "batch_id": f"{prefix}-batch-{batch_number}",
            "paper_only": True,
            "capital_at_risk": False,
            "error_count": 0,
            "capture_count": count,
            "captures": [
                {
                    "run_id": f"batch-{batch_number}-capture-{index:03d}",
                    "market_family": "btc_updown_5m",
                    "orderbook_snapshot_interval_seconds": 1.0,
                }
                for index in range(count)
            ],
            "finalizations": [],
            "errors": [],
        },
    )
    return path


def _fake_role_assignment(
    tmp_path: Path,
    *,
    selected_market_count: int,
    blockers: list[str],
):
    def fake(**_: Any) -> dict[str, Any]:
        report_path = tmp_path / f"fake-role-{len(blockers)}.json"
        manifest_path = tmp_path / f"fake-role-manifest-{len(blockers)}.json"
        report = {
            "role_assignment_ready": not blockers,
            "selected_market_count": selected_market_count,
            "role_market_counts": {
                CALIBRATION_ROLE: min(
                    selected_market_count,
                    CALIBRATION_MARKET_COUNT,
                ),
                CONFIRMATORY_ROLE: max(
                    0,
                    selected_market_count - CALIBRATION_MARKET_COUNT,
                ),
            },
            "blocking_reason_codes": blockers,
        }
        _write_json(report_path, report)
        _write_json(manifest_path, {"fixture": True})
        return {
            "report": report,
            "report_path": report_path,
            "manifest_path": manifest_path,
        }

    return fake


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _assert_blocked_safety(payload: dict[str, Any]) -> None:
    for name, expected in _blocked_safety_fields().items():
        assert payload[name] is expected


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
