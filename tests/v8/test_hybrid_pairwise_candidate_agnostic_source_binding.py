from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.training.hybrid_pairwise_candidate_agnostic_source_binding import (
    HybridBoundedFinalizationViewConfig,
    HybridCandidateAgnosticSourceBindingConfig,
    HybridCandidateAgnosticSourceSnapshotConfig,
    _bounded_finalized_batch,
    create_hybrid_candidate_agnostic_source_binding,
    freeze_hybrid_candidate_agnostic_source_snapshot,
    prepare_hybrid_bounded_finalization_view,
)

PAIRWISE_PROTOCOL = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_v1.json"
)


def test_active_candidate_agnostic_source_binds_without_outcome_access(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, capture_count=3)
    result = create_hybrid_candidate_agnostic_source_binding(
        _binding_config(fixture)
    )

    report = result["report"]
    assert report["source_binding_ready"] is True
    assert report["source_collection_terminal"] is False
    assert report["source_snapshot_allowed"] is False
    assert report["source_observed_market_count"] == 3
    assert report["source_prior_market_overlap_count"] == 0
    assert report["duplicate_collector_started"] is False
    assert report["labels_or_outcomes_opened"] is False
    assert report["uses_issue189_oof_development_calibration_or_pnl"] is False
    assert result["manifest"]["maximum_source_capture_attempt_ordinal"] == 150
    assert result["manifest"]["roles"] == {
        "fresh_development_calibration": 45,
        "fresh_confirmatory_validation": 60,
    }
    _assert_safety_blocked(report)


def test_prior_quarantine_overlap_blocks_source_binding(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        capture_count=3,
        prior_market_ids=["market-0002"],
    )
    result = create_hybrid_candidate_agnostic_source_binding(
        _binding_config(fixture)
    )

    assert result["report"]["source_binding_ready"] is False
    assert result["report"]["source_prior_market_overlap_count"] == 1
    assert "source_market_overlaps_issue183_prior_quarantine" in result[
        "report"
    ]["blocking_reason_codes"]
    _assert_safety_blocked(result["report"])


def test_terminal_snapshot_includes_exactly_first_150_attempts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, capture_count=152)
    binding_result = create_hybrid_candidate_agnostic_source_binding(
        _binding_config(fixture)
    )
    batch_path = fixture["batch_path"]
    batch = _load_json(batch_path)
    batch["collection_stop_reason"] = "outcome_blind_quality_target_reached"
    batch["quality_target_reached"] = True
    _write_json(batch_path, batch)

    snapshot_result = freeze_hybrid_candidate_agnostic_source_snapshot(
        HybridCandidateAgnosticSourceSnapshotConfig(
            run_id="terminal-snapshot",
            output_dir=tmp_path / "runs",
            binding_manifest_path=binding_result["manifest_path"],
            expected_binding_manifest_sha256=binding_result["manifest_sha256"],
            terminal_batch_progress_path=batch_path,
            expected_terminal_batch_progress_sha256=_sha256(batch_path),
        )
    )

    report = snapshot_result["report"]
    assert report["source_snapshot_ready"] is True
    assert report["source_total_capture_count"] == 152
    assert report["bounded_capture_attempt_count"] == 150
    assert report["attempts_after_150_included"] is False
    rows_descriptor = snapshot_result["manifest"]["bounded_capture_rows"]
    rows = _load_jsonl(Path(rows_descriptor["path"]))
    assert len(rows) == 150
    assert [row["round_index"] for row in rows] == list(range(1, 151))
    assert all("round-0151" not in row["run_id"] for row in rows)
    allowlist = _load_json(
        Path(snapshot_result["manifest"]["finalization_allowlist"]["path"])
    )
    assert allowlist["allowed_capture_attempt_count"] == 150
    assert allowlist["attempts_after_150_allowed"] is False
    assert allowlist["labels_or_outcomes_opened_for_allowlist_creation"] is False
    _assert_safety_blocked(report)


def test_active_or_drifted_terminal_source_snapshot_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, capture_count=150)
    binding_result = create_hybrid_candidate_agnostic_source_binding(
        _binding_config(fixture)
    )
    batch_path = fixture["batch_path"]

    active = freeze_hybrid_candidate_agnostic_source_snapshot(
        HybridCandidateAgnosticSourceSnapshotConfig(
            run_id="active-source",
            output_dir=tmp_path / "runs",
            binding_manifest_path=binding_result["manifest_path"],
            expected_binding_manifest_sha256=binding_result["manifest_sha256"],
            terminal_batch_progress_path=batch_path,
            expected_terminal_batch_progress_sha256=_sha256(batch_path),
        )
    )
    assert active["report"]["source_snapshot_ready"] is False
    assert "source_collection_not_terminal" in active["report"][
        "blocking_reason_codes"
    ]

    batch = _load_json(batch_path)
    batch["collection_stop_reason"] = "outcome_blind_quality_target_reached"
    batch["captures"][149]["orderbook_snapshot_interval_seconds"] = 5.0
    _write_json(batch_path, batch)
    drifted = freeze_hybrid_candidate_agnostic_source_snapshot(
        HybridCandidateAgnosticSourceSnapshotConfig(
            run_id="drifted-source",
            output_dir=tmp_path / "runs",
            binding_manifest_path=binding_result["manifest_path"],
            expected_binding_manifest_sha256=binding_result["manifest_sha256"],
            terminal_batch_progress_path=batch_path,
            expected_terminal_batch_progress_sha256=_sha256(batch_path),
        )
    )
    assert drifted["report"]["source_snapshot_ready"] is False
    assert "issue190_capture_orderbook_snapshot_interval_seconds_drift" in drifted[
        "report"
    ]["blocking_reason_codes"]
    _assert_safety_blocked(drifted["report"])


def test_bounded_finalization_cannot_include_attempts_after_150(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, capture_count=152)
    batch = _load_json(fixture["batch_path"])
    snapshot_rows = batch["captures"][:150]
    batch["finalizations"] = [
        {
            "run_id": row["run_id"],
            "run_dir": row["run_dir"],
            "finalization_status": "exported",
            "training_eligible": True,
            "exported_training_corpus_dir": f"/Volumes/PHILIPS/v8/{row['run_id']}",
        }
        for row in batch["captures"]
    ]

    bounded, blockers = _bounded_finalized_batch(
        snapshot_rows=snapshot_rows,
        finalized_batch=batch,
    )

    assert blockers == []
    assert bounded["capture_count"] == 150
    assert len(bounded["captures"]) == 150
    assert len(bounded["finalizations"]) == 150
    assert bounded["source_attempts_after_150_included"] is False
    included_run_ids = {row["run_id"] for row in bounded["finalizations"]}
    assert batch["captures"][150]["run_id"] not in included_run_ids
    assert batch["captures"][151]["run_id"] not in included_run_ids


def test_bounded_finalization_view_exposes_only_snapshot_allowlist(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, capture_count=152)
    snapshot_result = _terminal_snapshot(fixture)
    finalizer_script = tmp_path / "frozen-finalizer.py"
    finalizer_script.write_text("print('frozen finalizer')\n", encoding="utf-8")
    source_batch_hash_before = _sha256(fixture["batch_path"])

    result = prepare_hybrid_bounded_finalization_view(
        HybridBoundedFinalizationViewConfig(
            run_id="bounded-finalization-view",
            output_dir=tmp_path / "runs",
            snapshot_manifest_path=snapshot_result["manifest_path"],
            expected_snapshot_manifest_sha256=snapshot_result["manifest_sha256"],
            finalizer_script_path=finalizer_script,
            expected_finalizer_script_sha256=_sha256(finalizer_script),
            finalizer_git_commit="b" * 40,
            python_executable=Path("/usr/bin/python3"),
            training_corpus_root=tmp_path / "training-corpus",
        )
    )

    view_batch = _load_json(result["view_batch_progress_path"])
    assert view_batch["capture_count"] == 150
    assert [row["round_index"] for row in view_batch["captures"]] == list(
        range(1, 151)
    )
    view_root = result["view_root"]
    assert len([path for path in view_root.iterdir() if path.is_symlink()]) == 150
    assert not (view_root / "bound-source-round-0151").exists()
    assert not (view_root / "bound-source-round-0152").exists()
    assert _sha256(fixture["batch_path"]) == source_batch_hash_before
    command = result["manifest"]["finalizer_command_argv"]
    assert command[command.index("--output-dir") + 1] == str(view_root)
    assert command[1] == str(finalizer_script.resolve())
    assert result["report"]["finalizer_executed"] is False
    assert result["report"]["source_batch_progress_mutated"] is False
    _assert_safety_blocked(result["report"])


def test_bounded_finalization_view_blocks_missing_pending_capture_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, capture_count=150)
    snapshot_result = _terminal_snapshot(fixture)
    finalizer_script = tmp_path / "frozen-finalizer.py"
    finalizer_script.write_text("print('frozen finalizer')\n", encoding="utf-8")
    missing_run_dir = Path(
        _load_json(fixture["batch_path"])["captures"][0]["run_dir"]
    )
    (missing_run_dir / "pending_round_capture_manifest.json").unlink()

    try:
        prepare_hybrid_bounded_finalization_view(
            HybridBoundedFinalizationViewConfig(
                run_id="missing-pending-evidence",
                output_dir=tmp_path / "runs",
                snapshot_manifest_path=snapshot_result["manifest_path"],
                expected_snapshot_manifest_sha256=snapshot_result["manifest_sha256"],
                finalizer_script_path=finalizer_script,
                expected_finalizer_script_sha256=_sha256(finalizer_script),
                finalizer_git_commit="b" * 40,
                python_executable=Path("/usr/bin/python3"),
            )
        )
    except ValueError as exc:
        assert "bounded_view_pending_capture_manifest_missing" in str(exc)
    else:
        raise AssertionError("missing pending evidence must fail closed")


def _fixture(
    tmp_path: Path,
    *,
    capture_count: int,
    prior_market_ids: list[str] | None = None,
) -> dict[str, Any]:
    raw_root = tmp_path / "raw-source"
    raw_root.mkdir()
    protocol_path = tmp_path / "pairwise-protocol.json"
    protocol_path.write_bytes(PAIRWISE_PROTOCOL.read_bytes())
    quarantine_path = tmp_path / "quarantine.json"
    frozen_prior_market_ids = prior_market_ids or ["historical-market"]
    _write_json(
        quarantine_path,
        {
            "prior_market_ids": frozen_prior_market_ids,
            "prior_market_ids_sha256": _canonical_sha256(
                sorted(frozen_prior_market_ids)
            ),
        },
    )
    freeze_path = tmp_path / "precollection-freeze.json"
    freeze = {
        "schema_version": (
            "bigan-v8-hybrid-pairwise-precollection-freeze-manifest-v1"
        ),
        "final_prior_lineage_quarantine": _descriptor(quarantine_path),
        "source_pairwise_protocol": _descriptor(protocol_path),
        "minimum_collection_decision_ts": 2_000,
        "collection_started": False,
        "labels_or_outcomes_opened_for_role_assignment": False,
        **_safety(),
    }
    _write_json(freeze_path, freeze)
    readiness_path = tmp_path / "readiness.json"
    _write_json(
        readiness_path,
        {
            "schema_version": (
                "bigan-v8-hybrid-pairwise-precollection-readiness-manifest-v1"
            ),
            "precollection_readiness_passed": True,
            "precollection_freeze_created": True,
            "precollection_freeze_manifest": _descriptor(freeze_path),
            **_safety(),
        },
    )
    pre_registration_path = tmp_path / "source-pre-registration.json"
    _write_json(
        pre_registration_path,
        {
            "schema_version": (
                "bigan-v8-pairwise-future-unseen-holdout-"
                "pre-registration-manifest-v1"
            ),
            "pre_registration_ready": True,
            "candidate_agnostic_raw_collection": True,
            "outcome_blind_collection_only_required": True,
            "collection_stop_rule_is_outcome_blind": True,
            "holdout_labels_or_outcomes_opened_before_pre_registration": False,
            "settlement_finalizer_started_during_collection": False,
            "training_corpus_export_during_collection_allowed": False,
            "collection_control_uses_model_scores_bets_or_pnl": False,
            **_safety(),
        },
    )
    collection_freeze_path = tmp_path / "source-collection-freeze.json"
    _write_json(
        collection_freeze_path,
        {
            "schema_version": (
                "bigan-v8-pairwise-future-unseen-collection-freeze-manifest-v1"
            ),
            "pre_registration_manifest": _descriptor(pre_registration_path),
            "collection_control_is_outcome_blind": True,
            "outcome_blind_collection_only_required": True,
            "labels_or_outcomes_opened_for_collection_freeze": False,
            "settlement_finalizer_started_during_collection": False,
            "training_corpus_export_during_collection_allowed": False,
            "minimum_collection_decision_ts": 3_000,
            **_safety(),
        },
    )
    protocol = _load_json(protocol_path)
    collector = protocol["collector_contract"]
    captures = []
    for index in range(1, capture_count + 1):
        run_id = f"bound-source-round-{index:04d}"
        run_dir = raw_root / run_id
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True)
        _write_jsonl(
            raw_dir / "raw_polymarket_markets.jsonl",
            [
                {
                    "market_id": f"market-{index:04d}",
                    "condition_id": f"market-{index:04d}",
                    "market_start_ts": 3_000 + index,
                    "paper_only": True,
                    "capital_at_risk": False,
                }
            ],
        )
        _write_json(
            run_dir / "pending_round_capture_manifest.json",
            {
                "run_id": run_id,
                "round_index": index,
                "paper_only": True,
                "capital_at_risk": False,
            },
        )
        captures.append(
            {
                "round_index": index,
                "scheduled_round_start_ts": 3_000 + index,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "market_family": collector["market_family"],
                "public_provider_timeout_seconds": collector[
                    "public_provider_timeout_seconds"
                ],
                "public_provider_http_timeout_seconds": collector[
                    "public_provider_http_timeout_seconds"
                ],
                "orderbook_snapshot_interval_seconds": collector[
                    "orderbook_snapshot_interval_seconds"
                ],
                "orderbook_ws_initial_complete_book_timeout_seconds": collector[
                    "orderbook_ws_initial_complete_book_timeout_seconds"
                ],
                "rest_orderbook_fallback_collection_seconds": collector[
                    "rest_orderbook_fallback_collection_seconds"
                ],
                "feature_enrichment_max_attempts": collector[
                    "feature_enrichment_max_attempts"
                ],
            }
        )
    batch_path = tmp_path / "batch-progress.json"
    _write_json(
        batch_path,
        {
            "batch_id": "candidate-agnostic-batch",
            "capture_count": capture_count,
            "captures": captures,
            "finalizations": [],
            "future_holdout_collection_freeze_manifest": _descriptor(
                collection_freeze_path
            ),
            "collection_stop_reason": "frozen_maximum_not_yet_reached",
            "outcome_blind_collection_only": True,
            "outcome_blind_quality_valid_capture_count": capture_count,
            "labels_or_outcomes_opened_during_collection": False,
            "labels_or_outcomes_opened_for_collection_control": False,
            "resolution_provider_called": False,
            "settlement_finalizer_started": False,
            "training_corpus_export_attempted": False,
            "uses_accepted_bet_count_for_collection_control": False,
            "uses_model_scores_for_collection_control": False,
            **_safety(),
        },
    )
    return {
        "raw_root": raw_root,
        "readiness_path": readiness_path,
        "freeze_path": freeze_path,
        "pre_registration_path": pre_registration_path,
        "collection_freeze_path": collection_freeze_path,
        "batch_path": batch_path,
    }


def _terminal_snapshot(fixture: dict[str, Any]) -> dict[str, Any]:
    binding_result = create_hybrid_candidate_agnostic_source_binding(
        _binding_config(fixture)
    )
    batch_path = fixture["batch_path"]
    batch = _load_json(batch_path)
    batch["collection_stop_reason"] = "outcome_blind_quality_target_reached"
    batch["quality_target_reached"] = True
    _write_json(batch_path, batch)
    return freeze_hybrid_candidate_agnostic_source_snapshot(
        HybridCandidateAgnosticSourceSnapshotConfig(
            run_id="terminal-snapshot-helper",
            output_dir=batch_path.parent / "runs",
            binding_manifest_path=binding_result["manifest_path"],
            expected_binding_manifest_sha256=binding_result["manifest_sha256"],
            terminal_batch_progress_path=batch_path,
            expected_terminal_batch_progress_sha256=_sha256(batch_path),
        )
    )


def _binding_config(fixture: dict[str, Any]) -> HybridCandidateAgnosticSourceBindingConfig:
    return HybridCandidateAgnosticSourceBindingConfig(
        run_id="source-binding",
        output_dir=fixture["batch_path"].parent / "runs",
        readiness_manifest_path=fixture["readiness_path"],
        expected_readiness_manifest_sha256=_sha256(fixture["readiness_path"]),
        precollection_freeze_manifest_path=fixture["freeze_path"],
        expected_precollection_freeze_manifest_sha256=_sha256(
            fixture["freeze_path"]
        ),
        source_pre_registration_manifest_path=fixture["pre_registration_path"],
        expected_source_pre_registration_manifest_sha256=_sha256(
            fixture["pre_registration_path"]
        ),
        source_collection_freeze_manifest_path=fixture["collection_freeze_path"],
        expected_source_collection_freeze_manifest_sha256=_sha256(
            fixture["collection_freeze_path"]
        ),
        source_batch_progress_path=fixture["batch_path"],
        expected_source_batch_progress_sha256=_sha256(fixture["batch_path"]),
        source_batch_id="candidate-agnostic-batch",
        source_raw_root=fixture["raw_root"],
        builder_git_commit="a" * 40,
    )


def _safety() -> dict[str, Any]:
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
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
    }


def _assert_safety_blocked(payload: dict[str, Any]) -> None:
    for key, value in _safety().items():
        assert payload[key] == value


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
