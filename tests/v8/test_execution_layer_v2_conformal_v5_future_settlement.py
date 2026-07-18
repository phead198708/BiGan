from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement as settlement_module
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    SETTLED_CORPUS_INDEX_SCHEMA_VERSION,
    ConformalV5FutureSettlementCorpusIndexConfig,
    _copy_and_finalize_selected_round,
    _feature_payload,
    _is_retryable_settlement_failure,
    _join_frozen_replay_targets,
    _validate_settled_corpus_index,
    build_conformal_v5_future_settled_corpus_index,
)


def test_settlement_index_config_rejects_unbounded_or_unpinned_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        ConformalV5FutureSettlementCorpusIndexConfig(
            run_id="future-settlement",
            output_dir=tmp_path,
            prediction_freeze_manifest_path=tmp_path / "freeze.json",
            expected_prediction_freeze_manifest_sha256="bad",
            builder_git_commit="a" * 40,
            target_access_started_ts=200,
        )
    with pytest.raises(ValueError, match="max_workers"):
        ConformalV5FutureSettlementCorpusIndexConfig(
            run_id="future-settlement",
            output_dir=tmp_path,
            prediction_freeze_manifest_path=tmp_path / "freeze.json",
            expected_prediction_freeze_manifest_sha256="b" * 64,
            builder_git_commit="a" * 40,
            target_access_started_ts=200,
            max_workers=0,
        )
    with pytest.raises(ValueError, match="settlement_max_wait_seconds"):
        ConformalV5FutureSettlementCorpusIndexConfig(
            run_id="future-settlement",
            output_dir=tmp_path,
            prediction_freeze_manifest_path=tmp_path / "freeze.json",
            expected_prediction_freeze_manifest_sha256="b" * 64,
            builder_git_commit="a" * 40,
            target_access_started_ts=200,
            settlement_max_wait_seconds=-1,
        )


def test_copy_then_settle_preserves_outcome_blind_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected, source_run_dir = _selected_capture_fixture(tmp_path)

    def fake_finalize(copied_run_dir: Path, **_: object) -> SimpleNamespace:
        corpus_dir = copied_run_dir / "phase2_corpus"
        corpus_dir.mkdir()
        for filename, payload in {
            "polymarket_corpus_manifest.json": {"market_count": 1},
            "polymarket_feature_rows.jsonl": {"market_id": "market-1"},
            "polymarket_label_rows.jsonl": {"market_id": "market-1"},
            "polymarket_resolution_events.jsonl": {
                "market_id": "market-1",
                "resolved_outcome": "UP",
            },
        }.items():
            path = corpus_dir / filename
            if filename.endswith(".jsonl"):
                path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            else:
                path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        (copied_run_dir / "pending_round_finalization_manifest.json").write_text(
            json.dumps({"finalization_status": "exported"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (copied_run_dir / "raw/raw_polymarket_resolutions.jsonl").write_text(
            json.dumps({"resolved_outcome": "UP"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(
            report={
                "finalization_status": "exported",
                "pending_resolution": False,
                "resolution_provider_called": True,
                "phase2_corpus_built": True,
            },
            corpus_dir=corpus_dir,
        )

    monkeypatch.setattr(settlement_module, "finalize_polymarket_pending_round", fake_finalize)
    run_dir = tmp_path / "settlement-run"
    (run_dir / "settled_round_copies").mkdir(parents=True)
    (run_dir / "settled_corpus_quarantine").mkdir()

    result = _copy_and_finalize_selected_round(
        selected,
        run_dir=run_dir,
        provider_factory=object,
    )

    assert result["settled_corpus_ready"] is True
    assert result["index_entry"]["official_read_only_resolution"] is True
    assert result["index_entry"]["source_outcome_blind_round_mutated"] is False
    assert result["index_entry"]["direct_training_corpus_exported"] is False
    assert (source_run_dir / "raw/raw_polymarket_resolutions.jsonl").read_text() == ""
    source_manifest = json.loads(
        (source_run_dir / "pending_round_capture_manifest.json").read_text(encoding="utf-8")
    )
    assert source_manifest["config"]["output_dir"] == str(source_run_dir.parent)


def test_unresolved_copy_fails_closed_without_index_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected, _ = _selected_capture_fixture(tmp_path)

    def fake_finalize(*_: object, **__: object) -> SimpleNamespace:
        return SimpleNamespace(
            report={
                "finalization_status": "pending_resolution",
                "pending_resolution": True,
                "resolution_provider_called": True,
                "phase2_corpus_built": False,
                "reject_reason_counts": {"missing_verified_resolution": 1},
                "phase2_error": None,
            },
            corpus_dir=None,
        )

    monkeypatch.setattr(settlement_module, "finalize_polymarket_pending_round", fake_finalize)
    run_dir = tmp_path / "settlement-run"
    (run_dir / "settled_round_copies").mkdir(parents=True)
    (run_dir / "settled_corpus_quarantine").mkdir()

    result = _copy_and_finalize_selected_round(
        selected,
        run_dir=run_dir,
        provider_factory=object,
    )

    assert result["settled_corpus_ready"] is False
    assert "official_resolution_still_pending" in result["failure"]["reason_codes"]
    assert "missing_verified_resolution" in result["failure"]["reason_codes"]
    assert "index_entry" not in result
    assert _is_retryable_settlement_failure(result["failure"]) is True


def test_pending_settlement_copy_can_be_retried_without_recopied_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected, source_run_dir = _selected_capture_fixture(tmp_path)
    call_count = 0

    def fake_finalize(copied_run_dir: Path, **_: object) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return SimpleNamespace(
                report={
                    "finalization_status": "pending_resolution",
                    "pending_resolution": True,
                    "resolution_provider_called": True,
                    "phase2_corpus_built": False,
                    "reject_reason_counts": {"missing_verified_resolution": 1},
                },
                corpus_dir=None,
            )
        corpus_dir = copied_run_dir / "phase2_corpus"
        corpus_dir.mkdir()
        for filename in (
            "polymarket_corpus_manifest.json",
            "polymarket_feature_rows.jsonl",
            "polymarket_label_rows.jsonl",
            "polymarket_resolution_events.jsonl",
        ):
            (corpus_dir / filename).write_text("{}\n", encoding="utf-8")
        (copied_run_dir / "pending_round_finalization_manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return SimpleNamespace(
            report={
                "finalization_status": "exported",
                "pending_resolution": False,
                "resolution_provider_called": True,
                "phase2_corpus_built": True,
            },
            corpus_dir=corpus_dir,
        )

    monkeypatch.setattr(settlement_module, "finalize_polymarket_pending_round", fake_finalize)
    run_dir = tmp_path / "settlement-run"
    (run_dir / "settled_round_copies").mkdir(parents=True)
    (run_dir / "settled_corpus_quarantine").mkdir()

    first = _copy_and_finalize_selected_round(
        selected,
        run_dir=run_dir,
        provider_factory=object,
        settlement_attempt=1,
    )
    second = _copy_and_finalize_selected_round(
        selected,
        run_dir=run_dir,
        provider_factory=object,
        settlement_attempt=2,
    )

    assert first["settled_corpus_ready"] is False
    assert second["settled_corpus_ready"] is True
    assert second["index_entry"]["settlement_attempt_count"] == 2
    assert call_count == 2
    assert (source_run_dir / "raw/raw_polymarket_resolutions.jsonl").read_text() == ""


def test_settled_corpus_builder_retries_pending_market_before_emitting_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_path, freeze_sha = _prediction_freeze_fixture(tmp_path)
    calls: list[tuple[int, list[str]]] = []

    def fake_finalize_rows(
        selected_rows: list[dict], *, settlement_attempt: int, **_: object
    ) -> list[dict]:
        market_ids = [str(row["market_id"]) for row in selected_rows]
        calls.append((settlement_attempt, market_ids))
        rows = []
        for market_id in market_ids:
            if market_id == "market-000" and settlement_attempt == 1:
                rows.append(
                    {
                        "market_id": market_id,
                        "settled_corpus_ready": False,
                        "failure": {
                            "market_id": market_id,
                            "run_id": "round-000",
                            "reason_codes": ["official_resolution_still_pending"],
                            "pending_resolution": True,
                            "settlement_attempt_count": 1,
                        },
                    }
                )
            else:
                rows.append(_successful_settlement_result(market_id, settlement_attempt))
        return rows

    monkeypatch.setattr(settlement_module, "_finalize_selected_rounds", fake_finalize_rows)
    result = build_conformal_v5_future_settled_corpus_index(
        ConformalV5FutureSettlementCorpusIndexConfig(
            run_id="future-settled",
            output_dir=tmp_path / "runs",
            prediction_freeze_manifest_path=freeze_path,
            expected_prediction_freeze_manifest_sha256=freeze_sha,
            builder_git_commit="a" * 40,
            target_access_started_ts=300,
            settlement_max_wait_seconds=10,
            settlement_poll_interval_seconds=1,
        ),
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda _: None,
        clock_ms_fn=lambda: 400,
    )

    assert result["report"]["settled_corpus_index_ready"] is True
    assert result["report"]["settlement_attempt_count"] == 2
    assert result["report"]["settlement_retry_market_count"] == 1
    assert result["index"]["entry_count"] == 220
    assert result["index"]["target_access_started_ts"] == 300
    assert result["index"]["index_finalized_ts"] == 400
    assert calls[1] == (2, ["market-000"])


def test_settled_corpus_builder_fails_closed_when_resolution_wait_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_path, freeze_sha = _prediction_freeze_fixture(tmp_path)

    def fake_finalize_rows(selected_rows: list[dict], **_: object) -> list[dict]:
        return [
            {
                "market_id": str(row["market_id"]),
                "settled_corpus_ready": False,
                "failure": {
                    "market_id": str(row["market_id"]),
                    "run_id": str(row["run_id"]),
                    "reason_codes": ["official_resolution_still_pending"],
                    "pending_resolution": True,
                    "settlement_attempt_count": 1,
                },
            }
            for row in selected_rows
        ]

    monotonic_values = iter([0.0, 11.0])
    monkeypatch.setattr(settlement_module, "_finalize_selected_rounds", fake_finalize_rows)
    result = build_conformal_v5_future_settled_corpus_index(
        ConformalV5FutureSettlementCorpusIndexConfig(
            run_id="future-unresolved",
            output_dir=tmp_path / "runs",
            prediction_freeze_manifest_path=freeze_path,
            expected_prediction_freeze_manifest_sha256=freeze_sha,
            builder_git_commit="a" * 40,
            target_access_started_ts=300,
            settlement_max_wait_seconds=10,
        ),
        monotonic_fn=lambda: next(monotonic_values),
        sleep_fn=lambda _: None,
        clock_ms_fn=lambda: 400,
    )

    assert result["report"]["settled_corpus_index_ready"] is False
    assert result["index"] is None
    assert result["index_path"] is None
    assert result["report"]["unresolved_or_failed_market_count"] == 220
    assert (
        result["report"]["unresolved_or_failed_reason_distribution"][
            "settlement_resolution_max_wait_elapsed"
        ]
        == 220
    )


def test_join_targets_uses_executed_action_order_size_and_side_only_safety() -> None:
    replay = [
        _replay_row(
            market_id="market-1",
            decision_ts=100,
            action="BUY_UP_SELL_BEFORE_CLOSE",
            side="UP",
            allowed=True,
            order_size=0.2,
        ),
        _replay_row(
            market_id="market-1",
            decision_ts=200,
            action="BUY_DOWN_HOLD_TO_SETTLEMENT",
            side="DOWN",
            allowed=False,
            order_size=0.0,
        ),
    ]
    targets = {
        ("market-1", 100): _target_row("market-1", 100, BUY_UP_SELL_BEFORE_CLOSE=0.5),
        ("market-1", 200): _target_row("market-1", 200, BUY_DOWN_HOLD_TO_SETTLEMENT=-0.4),
    }

    rows = _join_frozen_replay_targets(
        replay,
        targets_by_decision=targets,
        policy_name="candidate",
        decision_freeze_sha256="a" * 64,
    )

    assert rows[0]["accepted_bet_net_pnl"] == pytest.approx(0.1)
    assert rows[1]["accepted_bet_net_pnl"] == 0.0
    assert rows[0]["target_used_as_decision_input"] is False
    assert rows[0]["future_results_used_for_tuning"] is False
    assert rows[0]["source_model_candidate_eligible"] is False
    assert rows[0]["v8_execution_handoff_allowed"] is False
    assert rows[0]["capital_at_risk"] is False


def test_settled_index_requires_exact_frozen_market_set_and_post_freeze_time() -> None:
    selected = [
        {"market_id": "market-1"},
        {"market_id": "market-2"},
    ]
    index = _settled_index(["market-1", "market-2"])

    rows = _validate_settled_corpus_index(
        index,
        expected_decision_freeze_sha256="a" * 64,
        decision_freeze_created_ts=150,
        selected_rows=selected,
        reconciliation_started_ts=300,
    )

    assert [row["market_id"] for row in rows] == ["market-1", "market-2"]

    index["entries"] = index["entries"][:1]
    with pytest.raises(ValueError, match="complete_market_set"):
        _validate_settled_corpus_index(
            index,
            expected_decision_freeze_sha256="a" * 64,
            decision_freeze_created_ts=150,
            selected_rows=selected,
            reconciliation_started_ts=300,
        )


def test_settled_index_before_decision_freeze_fails_closed() -> None:
    index = _settled_index(["market-1"])
    index["target_access_started_ts"] = 100

    with pytest.raises(ValueError, match="target_access_after_decision_freeze"):
        _validate_settled_corpus_index(
            index,
            expected_decision_freeze_sha256="a" * 64,
            decision_freeze_created_ts=150,
            selected_rows=[{"market_id": "market-1"}],
            reconciliation_started_ts=300,
        )


def test_frozen_feature_comparison_ignores_only_freeze_metadata() -> None:
    feature = {
        "market_id": "market-1",
        "condition_id": "condition-1",
        "slug": "btc-updown-5m-1",
        "market_family": "btc_updown_5m",
        "horizon_ms": 300_000,
        "decision_ts": 100,
        "feature_cutoff_ts": 100,
        "max_input_ts": 100,
        "available_at_ts": 100,
        "features": {"p_up": 0.6},
        "feature_provenance": {"p_up": {"max_input_ts": 100}},
    }
    frozen = {
        **feature,
        "future_window_selection_rank": 1,
        "future_feature_row_sha256": "b" * 64,
        "target_used_as_decision_input": False,
    }

    assert _feature_payload(feature) == _feature_payload(frozen)
    frozen["features"] = {"p_up": 0.61}
    assert _feature_payload(feature) != _feature_payload(frozen)


def test_settlement_artifacts_are_hashable(tmp_path: Path) -> None:
    path = tmp_path / "settlement.json"
    path.write_text(json.dumps({"paper_only": True}, sort_keys=True) + "\n", encoding="utf-8")
    assert len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64


def _replay_row(
    *,
    market_id: str,
    decision_ts: int,
    action: str,
    side: str,
    allowed: bool,
    order_size: float,
) -> dict:
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "executed_action": action,
        "selected_side": side,
        "selected_action_family": (
            "SELL_BEFORE_CLOSE" if action.endswith("SELL_BEFORE_CLOSE") else "HOLD_TO_SETTLEMENT"
        ),
        "execution_guard_order_allowed": allowed,
        "proposed_order_size": order_size,
    }


def _target_row(market_id: str, decision_ts: int, **overrides: float) -> dict:
    values = {
        "BUY_UP_HOLD_TO_SETTLEMENT": 0.0,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.0,
        "BUY_UP_SELL_BEFORE_CLOSE": 0.0,
        "BUY_DOWN_SELL_BEFORE_CLOSE": 0.0,
        "NO_TRADE": 0.0,
        **overrides,
    }
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "resolved_outcome": "UP",
        "target_net_pnl_per_notional_by_action": values,
    }


def _settled_index(market_ids: list[str]) -> dict:
    safety = {
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "v8_execution_handoff_allowed": False,
        "paper_candidate_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    return {
        "schema_version": SETTLED_CORPUS_INDEX_SCHEMA_VERSION,
        "decision_freeze_sha256": "a" * 64,
        "target_access_started_ts": 200,
        "index_finalized_ts": 210,
        "outcomes_used_for_decision_or_selection": False,
        "outcomes_used_for_threshold_or_model_tuning": False,
        "entries": [
            {
                "market_id": market_id,
                "official_read_only_resolution": True,
                "corpus_built_after_decision_freeze": True,
                "settled_after_market_close": True,
            }
            for market_id in market_ids
        ],
        **safety,
    }


def _selected_capture_fixture(tmp_path: Path) -> tuple[dict, Path]:
    source_parent = tmp_path / "source"
    source_run_dir = source_parent / "round-001"
    raw_dir = source_run_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "raw_polymarket_resolutions.jsonl").write_text("", encoding="utf-8")
    manifest_path = source_run_dir / "pending_round_capture_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": {
                    "run_id": source_run_dir.name,
                    "output_dir": str(source_parent),
                },
                "pending_resolution": True,
                "resolution_provider_called": False,
                "paper_only": True,
                "capital_at_risk": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    selected = {
        "market_id": "market-1",
        "run_id": source_run_dir.name,
        "scheduled_round_start_ts": 100,
        "market_end_ts": 200,
        "pending_round_capture_manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    }
    return selected, source_run_dir


def _prediction_freeze_fixture(tmp_path: Path) -> tuple[Path, str]:
    freeze_dir = tmp_path / "freeze"
    freeze_dir.mkdir()
    selected_rows = [
        {
            "market_id": f"market-{index:03d}",
            "run_id": f"round-{index:03d}",
        }
        for index in range(220)
    ]
    selected_path = freeze_dir / "selected.jsonl"
    selected_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_rows),
        encoding="utf-8",
    )
    action_path = freeze_dir / "actions.jsonl"
    action_path.write_text(
        "".join(
            json.dumps({"market_id": row["market_id"], "market_close_ts": 200}, sort_keys=True)
            + "\n"
            for row in selected_rows
        ),
        encoding="utf-8",
    )
    decision_path = freeze_dir / "decision.json"
    decision_path.write_text(
        json.dumps({"decision_freeze_created_ts": 100}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    descriptor = lambda path: {  # noqa: E731
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    freeze = {
        "schema_version": (f"{settlement_module.PREDICTION_FREEZE_SCHEMA_PREFIX}-manifest-v1"),
        "decision_freeze_written_before_target_access": True,
        "future_labels_outcomes_or_pnl_opened": False,
        "resolution_artifact_opened": False,
        "accepted_bet_decision_freeze": descriptor(decision_path),
        "selected_window_rows": descriptor(selected_path),
        "target_free_five_action_rows": descriptor(action_path),
        **settlement_module._blocked_safety_fields(),
    }
    freeze_path = freeze_dir / "manifest.json"
    freeze_path.write_text(json.dumps(freeze, sort_keys=True) + "\n", encoding="utf-8")
    return freeze_path, hashlib.sha256(freeze_path.read_bytes()).hexdigest()


def _successful_settlement_result(market_id: str, attempt: int) -> dict:
    return {
        "market_id": market_id,
        "settled_corpus_ready": True,
        "index_entry": {
            "market_id": market_id,
            "official_read_only_resolution": True,
            "corpus_built_after_decision_freeze": True,
            "settled_after_market_close": True,
            "settlement_attempt_count": attempt,
        },
    }
