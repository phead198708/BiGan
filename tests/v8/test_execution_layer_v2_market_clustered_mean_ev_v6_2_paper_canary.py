from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_market_clustered_mean_ev_v6_2_paper_canary as subject,
)
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_paper_canary import (
    MarketClusteredMeanEVV62PaperCanaryConfig,
    classify_capture_hard_failure,
    run_market_clustered_mean_ev_v6_2_paper_canary,
    validate_v6_2_paper_candidate_unlock,
)
from examples.v8 import (
    run_execution_layer_v2_market_clustered_mean_ev_v6_2_paper_canary as runner_subject,
)
from examples.v8.run_execution_layer_v2_market_clustered_mean_ev_v6_2_paper_canary import (
    _wait_until_epoch_ms,
    run_v6_2_paper_canary_cli,
)


def test_unlock_failure_happens_before_capture_artifact_access(tmp_path: Path) -> None:
    unlock = tmp_path / "unlock.json"
    _write_json(unlock, {})
    missing_capture = tmp_path / "not-created"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_market_clustered_mean_ev_v6_2_paper_canary(
            MarketClusteredMeanEVV62PaperCanaryConfig(
                run_id="blocked-before-provider",
                output_dir=tmp_path / "out",
                unlock_manifest_path=unlock,
                expected_unlock_manifest_sha256="0" * 64,
                captured_round_dirs=(missing_capture,),
                runtime_created_ts=1,
                builder_git_commit="a" * 40,
            )
        )
    assert not missing_capture.exists()


def test_snapshot_canary_emits_only_guard_allowed_paper_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _lineage_bundle(tmp_path)
    capture = _synthetic_capture(tmp_path)
    _patch_scoring(monkeypatch, order_allowed=True)

    result = run_market_clustered_mean_ev_v6_2_paper_canary(
        MarketClusteredMeanEVV62PaperCanaryConfig(
            run_id="snapshot-canary",
            output_dir=tmp_path / "out",
            unlock_manifest_path=bundle["unlock"],
            expected_unlock_manifest_sha256=_sha(bundle["unlock"]),
            captured_round_dirs=(capture,),
            runtime_created_ts=1_784_472_600_000,
            builder_git_commit="a" * 40,
        )
    )

    report = result["report"]
    assert report["complete_round_count"] == 1
    assert report["feature_row_count"] > 0
    assert report["five_action_row_count"] == report["feature_row_count"] * 5
    assert report["paper_intent_count"] == 1
    assert report["paper_fill_count"] == 1
    assert report["paper_exit_intent_count"] == 1
    assert report["paper_exit_fill_count"] == 1
    assert report["paper_ledger_entry_count"] == 2
    assert report["sell_before_close_residual_position_count"] == 0
    assert report["sell_before_close_lifecycle_complete"] is True
    assert report["closed_before_settlement_realized_trade_pnl_sum"] == pytest.approx(
        -0.004
    )
    assert report["runtime_safety_passed"] is True
    assert report["decision_target_outcome_or_pnl_accessed"] is False
    assert report["paper_candidate_allowed"] is True
    assert report["v8_execution_handoff_allowed"] is False
    assert report["capital_at_risk"] is False
    assert result["manifest"]["#134_resume_allowed"] is False
    assert result["manifest"]["#146_start_allowed"] is False

    intents = _load_jsonl(Path(result["run_dir"]) / "v6_2_paper_intents.jsonl")
    assert intents[0]["paper_order_size"] == pytest.approx(0.2)
    assert intents[0]["execution_guard_order_allowed"] is True
    assert intents[0]["forced_coverage_bet"] is False
    audit = _load_jsonl(Path(result["run_dir"]) / "v6_2_paper_canary_capture_audit.jsonl")
    assert audit[0]["resolution_artifact_opened_for_decision"] is False


def test_hold_to_settlement_position_never_emits_preclose_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _lineage_bundle(tmp_path)
    capture = _synthetic_capture(tmp_path)
    _patch_scoring(
        monkeypatch,
        order_allowed=True,
        selected_action="BUY_DOWN_HOLD_TO_SETTLEMENT",
    )

    result = run_market_clustered_mean_ev_v6_2_paper_canary(
        MarketClusteredMeanEVV62PaperCanaryConfig(
            run_id="hts-lifecycle",
            output_dir=tmp_path / "out",
            unlock_manifest_path=bundle["unlock"],
            expected_unlock_manifest_sha256=_sha(bundle["unlock"]),
            captured_round_dirs=(capture,),
            runtime_created_ts=1_784_472_600_000,
            builder_git_commit="a" * 40,
        )
    )

    report = result["report"]
    assert report["paper_exit_signal_count"] == 1
    assert report["paper_exit_intent_count"] == 0
    assert report["paper_exit_fill_count"] == 0
    assert report["hold_to_settlement_open_position_count"] == 1
    signals = _load_jsonl(Path(result["run_dir"]) / "v6_2_paper_exit_signals.jsonl")
    assert signals[0]["paper_exit_decision"] == "HOLD_TO_SETTLEMENT"
    assert signals[0]["accepted_for_paper_exit_intent"] is False
    settlement = runner_subject._settlement_evaluation(
        run_id="hts-lifecycle",
        run_dir=Path(result["run_dir"]),
        settlement_source_dirs=[capture],
    )
    assert settlement["report"]["settled_position_count"] == 1
    assert settlement["report"]["unresolved_open_position_count"] == 0
    assert settlement["rows"][0]["resolved_outcome"] == "UP"
    assert settlement["rows"][0]["settlement_pnl"] == pytest.approx(-0.112)
    assert settlement["rows"][0]["settlement_used_for_decision"] is False


@pytest.mark.parametrize(
    ("reference_start", "reference_end", "expected_outcome"),
    ((100_000.0, 99_999.0, "DOWN"), (100_000.0, 100_000.0, "UP")),
)
def test_reference_price_only_resolution_is_normalized_by_frozen_market_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_start: float,
    reference_end: float,
    expected_outcome: str,
) -> None:
    bundle = _lineage_bundle(tmp_path)
    capture = _synthetic_capture(tmp_path)
    market = _load_jsonl(capture / "raw" / "raw_polymarket_markets.jsonl")[0]
    _write_jsonl(
        capture / "raw" / "raw_polymarket_resolutions.jsonl",
        [
            {
                "market_id": market["market_id"],
                "reference_price_start": reference_start,
                "reference_price_end": reference_end,
                "reference_price_source": market["reference_price_source"],
                "resolution_source_type": "reference_prices",
                "resolution_status": "normal",
                **_resolution_safety(),
            }
        ],
    )
    _patch_scoring(
        monkeypatch,
        order_allowed=True,
        selected_action="BUY_DOWN_HOLD_TO_SETTLEMENT",
    )
    result = run_market_clustered_mean_ev_v6_2_paper_canary(
        MarketClusteredMeanEVV62PaperCanaryConfig(
            run_id="reference-only-resolution",
            output_dir=tmp_path / "out",
            unlock_manifest_path=bundle["unlock"],
            expected_unlock_manifest_sha256=_sha(bundle["unlock"]),
            captured_round_dirs=(capture,),
            runtime_created_ts=1,
            builder_git_commit="a" * 40,
        )
    )

    settlement = runner_subject._settlement_evaluation(
        run_id="reference-only-resolution",
        run_dir=Path(result["run_dir"]),
        settlement_source_dirs=[capture],
    )

    assert settlement["report"]["unresolved_open_position_count"] == 0
    assert settlement["report"]["derived_reference_price_resolution_count"] == 1
    assert settlement["rows"][0]["resolved_outcome"] == expected_outcome
    assert settlement["rows"][0]["resolved_outcome_source"] == (
        "official_reference_price_rule_derivation"
    )
    assert settlement["rows"][0]["settlement_used_for_decision"] is False


def test_reference_price_resolution_with_invalid_provenance_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _lineage_bundle(tmp_path)
    capture = _synthetic_capture(tmp_path)
    market = _load_jsonl(capture / "raw" / "raw_polymarket_markets.jsonl")[0]
    _write_jsonl(
        capture / "raw" / "raw_polymarket_resolutions.jsonl",
        [
            {
                "market_id": market["market_id"],
                "reference_price_start": 100_000.0,
                "reference_price_end": 99_999.0,
                "reference_price_source": "unverified-source",
                "resolution_source_type": "reference_prices",
                "resolution_status": "normal",
                **_resolution_safety(),
            }
        ],
    )
    _patch_scoring(
        monkeypatch,
        order_allowed=True,
        selected_action="BUY_DOWN_HOLD_TO_SETTLEMENT",
    )
    result = run_market_clustered_mean_ev_v6_2_paper_canary(
        MarketClusteredMeanEVV62PaperCanaryConfig(
            run_id="invalid-reference-provenance",
            output_dir=tmp_path / "out",
            unlock_manifest_path=bundle["unlock"],
            expected_unlock_manifest_sha256=_sha(bundle["unlock"]),
            captured_round_dirs=(capture,),
            runtime_created_ts=1,
            builder_git_commit="a" * 40,
        )
    )

    settlement = runner_subject._settlement_evaluation(
        run_id="invalid-reference-provenance",
        run_dir=Path(result["run_dir"]),
        settlement_source_dirs=[capture],
    )

    assert settlement["report"]["unresolved_open_position_count"] == 1
    assert settlement["report"]["invalid_resolution_row_count"] == 1
    assert settlement["report"]["resolution_normalization_reason_distribution"] == {
        "resolution_source_mismatch": 1
    }
    assert settlement["rows"][0]["total_paper_pnl"] is None


def test_sell_before_close_without_exit_window_snapshot_remains_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _lineage_bundle(tmp_path)
    capture = _synthetic_capture(tmp_path)
    books_path = capture / "raw" / "raw_polymarket_orderbooks.jsonl"
    books = _load_jsonl(books_path)
    last_ts = max(int(row["ts"]) for row in books)
    _write_jsonl(books_path, [row for row in books if int(row["ts"]) < last_ts])
    _patch_scoring(monkeypatch, order_allowed=True)

    result = run_market_clustered_mean_ev_v6_2_paper_canary(
        MarketClusteredMeanEVV62PaperCanaryConfig(
            run_id="sbc-residual",
            output_dir=tmp_path / "out",
            unlock_manifest_path=bundle["unlock"],
            expected_unlock_manifest_sha256=_sha(bundle["unlock"]),
            captured_round_dirs=(capture,),
            runtime_created_ts=1_784_472_600_000,
            builder_git_commit="a" * 40,
        )
    )

    report = result["report"]
    assert report["paper_exit_intent_count"] == 0
    assert report["paper_exit_fill_count"] == 0
    assert report["sell_before_close_residual_position_count"] == 1
    assert report["sell_before_close_lifecycle_complete"] is False
    assert report["runtime_safety_passed"] is False
    signals = _load_jsonl(Path(result["run_dir"]) / "v6_2_paper_exit_signals.jsonl")
    assert signals[-1]["exit_reason_codes"] == ["exit_book_stale"]
    assert signals[-1]["accepted_for_paper_exit_intent"] is False


def test_snapshot_runner_reports_fixture_mode_and_keeps_handoff_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _lineage_bundle(tmp_path)
    capture = _synthetic_capture(tmp_path)
    _patch_scoring(monkeypatch, order_allowed=True)

    result = run_v6_2_paper_canary_cli(
        run_id="snapshot-runner",
        output_dir=tmp_path / "out",
        unlock_manifest_path=bundle["unlock"],
        expected_unlock_manifest_sha256=_sha(bundle["unlock"]),
        round_count=1,
        snapshot_capture_dirs=(capture,),
    )

    assert result["collection_status"]["public_data_source"] == "snapshot_fixture"
    assert result["manifest"]["v8_execution_handoff_allowed"] is False
    assert result["manifest"]["paper_only"] is True
    assert result["manifest"]["capital_at_risk"] is False
    assert result["settlement_status"]["finalization_attempted_round_count"] == 0
    assert "settlement_evaluation_report" in result["manifest"]["artifacts"]
    assert "settlement_evaluation_rows" in result["manifest"]["artifacts"]


def test_terminal_snapshot_rerun_returns_verified_artifacts_without_rescoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _lineage_bundle(tmp_path)
    capture = _synthetic_capture(tmp_path)
    _patch_scoring(monkeypatch, order_allowed=True)
    first = run_v6_2_paper_canary_cli(
        run_id="terminal-idempotent",
        output_dir=tmp_path / "out",
        unlock_manifest_path=bundle["unlock"],
        expected_unlock_manifest_sha256=_sha(bundle["unlock"]),
        round_count=1,
        snapshot_capture_dirs=(capture,),
    )
    monkeypatch.setattr(
        runner_subject,
        "_run_scoring",
        lambda **kwargs: pytest.fail("terminal rerun must not rescore captures"),
    )

    second = run_v6_2_paper_canary_cli(
        run_id="terminal-idempotent",
        output_dir=tmp_path / "out",
        unlock_manifest_path=bundle["unlock"],
        expected_unlock_manifest_sha256=_sha(bundle["unlock"]),
        round_count=1,
        snapshot_capture_dirs=(capture,),
        overwrite_existing=True,
    )

    assert second["terminal_idempotent_replay"] is True
    assert second["manifest_sha256"] == first["manifest_sha256"]
    assert second["report"]["decision_target_outcome_or_pnl_accessed"] is False
    assert second["manifest"]["v8_execution_handoff_allowed"] is False


def test_settlement_only_reconciliation_does_not_mutate_source_decisions(
    tmp_path: Path,
) -> None:
    bundle = _lineage_bundle(tmp_path)
    capture = _synthetic_capture(tmp_path)
    market = _load_jsonl(capture / "raw" / "raw_polymarket_markets.jsonl")[0]
    _write_jsonl(
        capture / "raw" / "raw_polymarket_resolutions.jsonl",
        [
            {
                "market_id": market["market_id"],
                "reference_price_start": 100_000.0,
                "reference_price_end": 99_999.0,
                "reference_price_source": market["reference_price_source"],
                "resolution_source_type": "reference_prices",
                "resolution_status": "normal",
                **_resolution_safety(),
            }
        ],
    )
    source = tmp_path / "source-run"
    source.mkdir()
    source_manifest = {"run_id": source.name, **runner_subject._safety()}
    source_runtime = {
        "run_id": source.name,
        "decision_target_outcome_or_pnl_accessed": False,
        "source_or_o_score_mutated": False,
        "threshold_cost_sizing_or_guard_mutated": False,
        **runner_subject._safety(),
    }
    positions = {
        "positions": [
            {
                "position_id": "paper-fill-1",
                "market_id": market["market_id"],
                "selected_side": "DOWN",
                "entry_action": "BUY_DOWN_HOLD_TO_SETTLEMENT",
                "intended_exit_policy": "hold_to_settlement",
                "status": "open",
                "entry_contract_size": 0.2,
                "entry_price": 0.6,
                "exit_price": None,
                "realized_trade_pnl": None,
            }
        ]
    }
    settlement_status = {"rows": [{"capture_run_dir": str(capture)}]}
    _write_json(source / "v6_2_paper_canary_manifest.json", source_manifest)
    _write_json(source / "v6_2_paper_canary_runtime_report.json", source_runtime)
    _write_json(source / "v6_2_paper_positions.json", positions)
    _write_json(
        source / "v6_2_paper_canary_async_settlement_status.json",
        settlement_status,
    )
    source_hashes_before = {
        path.name: _sha(path) for path in sorted(source.iterdir()) if path.is_file()
    }

    result = runner_subject.reconcile_v6_2_paper_canary_settlement(
        source_run_dir=source,
        run_id="settlement-reconciled",
        output_dir=tmp_path / "out",
        unlock_manifest_path=bundle["unlock"],
        expected_unlock_manifest_sha256=_sha(bundle["unlock"]),
    )

    report = result["settlement_evaluation"]["report"]
    assert report["unresolved_open_position_count"] == 0
    assert report["derived_reference_price_resolution_count"] == 1
    assert report["total_paper_pnl"] == pytest.approx(0.08)
    assert result["manifest"]["source_decision_artifacts_mutated"] is False
    assert result["manifest"]["promotion_evidence_eligible"] is False
    assert {
        path.name: _sha(path) for path in sorted(source.iterdir()) if path.is_file()
    } == source_hashes_before


def test_real_runner_uses_one_continuous_collector_decoupled_from_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _lineage_bundle(tmp_path)
    source = _synthetic_capture(tmp_path)
    for dirname in ("raw", "provider_raw"):
        _write_jsonl(source / dirname / "raw_polymarket_resolutions.jsonl", [])
    _patch_scoring(monkeypatch, order_allowed=True)
    collector_finished = threading.Event()
    collector_calls = []

    def fake_collector(**kwargs):
        collector_calls.append(kwargs)
        target = Path(kwargs["output_dir"]) / "continuous-capture-round-01"
        shutil.copytree(source, target)
        capture = {
            "round_index": 1,
            "run_id": target.name,
            "run_dir": str(target),
            "scheduled_round_start_ts": 1_784_472_000_000,
        }
        collector_finished.set()
        return {"captures": [capture], "errors": []}

    original_scoring = runner_subject._run_scoring

    def scoring_after_collector(**kwargs):
        assert collector_finished.is_set()
        return original_scoring(**kwargs)

    monkeypatch.setattr(
        runner_subject,
        "run_polymarket_async_round_collector_cli",
        fake_collector,
    )
    monkeypatch.setattr(runner_subject, "_run_scoring", scoring_after_collector)
    monkeypatch.setattr(
        runner_subject,
        "finalize_polymarket_pending_round",
        lambda *args, **kwargs: SimpleNamespace(
            corpus_dir=None,
            report={
                "finalization_status": "pending_resolution",
                "pending_resolution": True,
                "raw_resolution_count": 0,
            }
        ),
    )

    result = run_v6_2_paper_canary_cli(
        run_id="continuous-runner",
        output_dir=tmp_path / "out",
        unlock_manifest_path=bundle["unlock"],
        expected_unlock_manifest_sha256=_sha(bundle["unlock"]),
        round_count=12,
        settlement_poll_interval_seconds=0.01,
        settlement_grace_seconds=0.0,
    )

    assert len(collector_calls) == 1
    assert collector_calls[0]["round_count"] == 12
    assert isinstance(collector_calls[0]["external_stop_event"], threading.Event)
    assert result["collection_status"]["continuous_round_collector_enabled"] is True
    assert (
        result["collection_status"][
            "round_capture_scheduler_decoupled_from_scoring"
        ]
        is True
    )
    assert result["manifest"]["v8_execution_handoff_allowed"] is False


def test_guard_blocked_decision_creates_no_intent_fill_or_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _lineage_bundle(tmp_path)
    capture = _synthetic_capture(tmp_path)
    _patch_scoring(monkeypatch, order_allowed=False)
    result = run_market_clustered_mean_ev_v6_2_paper_canary(
        MarketClusteredMeanEVV62PaperCanaryConfig(
            run_id="blocked-canary",
            output_dir=tmp_path / "out",
            unlock_manifest_path=bundle["unlock"],
            expected_unlock_manifest_sha256=_sha(bundle["unlock"]),
            captured_round_dirs=(capture,),
            runtime_created_ts=1,
            builder_git_commit="a" * 40,
        )
    )
    assert result["report"]["guard_allowed_count"] == 0
    assert result["report"]["paper_intent_count"] == 0
    assert result["report"]["paper_fill_count"] == 0
    assert result["report"]["paper_ledger_entry_count"] == 0
    assert result["report"]["guard_blocking_reason_distribution"] == {
        "execution_time_to_close_unsafe": 1
    }


def test_capture_failure_classification_ignores_optional_trade_telemetry(
    tmp_path: Path,
) -> None:
    capture = _synthetic_capture(tmp_path)
    assert classify_capture_hard_failure(capture) == []
    report_path = capture / "pending_round_capture_report.json"
    report = _load_json(report_path)
    report["raw_orderbook_row_count"] = 0
    report["raw_trade_row_count"] = 0
    _write_json(report_path, report)
    assert classify_capture_hard_failure(capture) == [
        "decision_critical_orderbook_missing"
    ]


def test_legacy_or_ambiguous_unlock_scope_is_rejected(tmp_path: Path) -> None:
    bundle = _lineage_bundle(tmp_path)
    unlock = _load_json(bundle["unlock"])
    unlock["paper_candidate_allowed_scope"] = "live"
    _write_json(bundle["unlock"], unlock)
    with pytest.raises(ValueError, match="scope"):
        validate_v6_2_paper_candidate_unlock(bundle["unlock"], _sha(bundle["unlock"]))


def test_round_boundary_wait_survives_fast_failed_capture() -> None:
    now_values = iter((100.0, 100.25, 400.0))
    sleeps: list[float] = []

    _wait_until_epoch_ms(
        400_000,
        now_fn=lambda: next(now_values),
        sleep_fn=sleeps.append,
    )

    assert sleeps == [pytest.approx(300.0), pytest.approx(299.75)]


def _patch_scoring(
    monkeypatch: pytest.MonkeyPatch,
    *,
    order_allowed: bool,
    selected_action: str = "BUY_UP_SELL_BEFORE_CLOSE",
) -> None:
    monkeypatch.setattr(subject.xgb.Booster, "load_model", lambda self, path: None)

    def fake_raw(booster, rows, *, feature_columns):
        return [
            {
                **row,
                "raw_direct_predicted_net_return": (
                    0.1 if row["action"] == selected_action else 0.0
                ),
            }
            for row in rows
        ]

    monkeypatch.setattr(subject, "_raw_target_stripped_predictions", fake_raw)
    monkeypatch.setattr(
        subject,
        "attach_frozen_execution_compatibility",
        lambda rows: [
            {**row, "guard_compatible_before_ranking": row["action"] != "NO_TRADE"}
            for row in rows
        ],
    )
    monkeypatch.setattr(
        subject,
        "apply_market_clustered_mean_ev_scores",
        lambda rows, *, calibration_artifact: [
            {
                **row,
                "action_advantage_lcb_net_return": row["raw_direct_predicted_net_return"],
            }
            for row in rows
        ],
    )

    def fake_replay(rows, **kwargs):
        selected = next(row for row in rows if row["action"] == selected_action)
        side = "UP" if "BUY_UP" in selected_action else "DOWN"
        family = (
            "SELL_BEFORE_CLOSE"
            if "SELL_BEFORE_CLOSE" in selected_action
            else "HOLD_TO_SETTLEMENT"
        )
        return [
            {
                "market_id": selected["market_id"],
                "decision_ts": selected["decision_ts"],
                "source_selected_action": selected["action"],
                "executed_action": selected["action"],
                "selected_side": side,
                "selected_action_family": family,
                "decision_score": 0.1,
                "selected_vs_runner_up_advantage": 0.05,
                "execution_guard_order_allowed": order_allowed,
                "proposed_order_size": 0.2 if order_allowed else 0.0,
                "microstructure_snapshot": {
                    "entry_ask": 0.56,
                    "spread_bps": 357.0,
                    "book_staleness_ms": 0.0,
                    "queue_fill_proxy": 1.0,
                    "time_to_close_seconds": 240.0,
                },
                "execution_blocking_reason_codes": (
                    [] if order_allowed else ["execution_time_to_close_unsafe"]
                ),
                "paper_only": True,
                "capital_at_risk": False,
            }
        ]

    monkeypatch.setattr(subject, "_outcome_blind_acceptance_replay", fake_replay)


def _lineage_bundle(tmp_path: Path) -> dict[str, Path]:
    feature_contract = Path(
        "examples/v8/polymarket_configs/"
        "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
    ).resolve()
    model = tmp_path / "model.json"
    calibration = tmp_path / "calibration.json"
    model.write_text("test model", encoding="utf-8")
    _write_json(
        calibration,
        {
            "sides": {
                "UP": {"mean_residual": 0.0, "mean_residual_upper_confidence_bound": 0.0},
                "DOWN": {
                    "mean_residual": 0.0,
                    "mean_residual_upper_confidence_bound": 0.0,
                },
            }
        },
    )
    profile = tmp_path / "profile.json"
    _write_json(profile, {"candidate_name": subject.CANDIDATE_NAME, "frozen": True})
    audit = tmp_path / "audit.json"
    _write_json(audit, {"feature_contract": _descriptor(feature_contract)})
    original = tmp_path / "original.json"
    original_payload = {
        "pre_target_access_audit": _descriptor(audit),
        "profile": _descriptor(profile),
        "source_model": _descriptor(model),
        "market_clustered_mean_risk_calibration": _descriptor(calibration),
    }
    _write_json(original, original_payload)
    promoted = tmp_path / "promoted.json"
    _write_json(
        promoted,
        {"original_frozen_candidate_manifest": _descriptor(original)},
    )
    gate = tmp_path / "gate.json"
    _write_json(gate, {"paper_candidate_allowed": True})
    contract = tmp_path / "contract.json"
    _write_json(
        contract,
        {
            "frozen": True,
            "bounded_complete_round_count": 12,
            "maximum_paper_order_notional": 0.2,
            "per_round_raw_evidence_persistence_required": True,
            "full_five_action_grid_required": True,
            "feature_max_input_ts_must_be_lte_decision_ts": True,
            "forced_coverage_bets_allowed": False,
            "settlement_may_block_next_round_collection": False,
            "legacy_o_source_score_used": False,
        },
    )
    unlock = tmp_path / "unlock.json"
    unlock_payload = {
        "manifest_id": "approved-test-unlock",
        "candidate_name": subject.CANDIDATE_NAME,
        "paper_candidate_allowed": True,
        "paper_canary_handoff_allowed": True,
        "paper_candidate_allowed_scope": subject.EXPECTED_SCOPE,
        "paper_only": True,
        "capital_at_risk": False,
        "live_trading_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "paper_candidate_gate_report": _descriptor(gate),
        "paper_canary_input_contract": _descriptor(contract),
        "promoted_research_candidate_manifest": _descriptor(promoted),
        "source_model": _descriptor(model),
        "market_clustered_mean_risk_calibration": _descriptor(calibration),
    }
    _write_json(unlock, unlock_payload)
    return {"unlock": unlock}


def _synthetic_capture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "capture-round-01"
    raw_dir = run_dir / "raw"
    provider_raw_dir = run_dir / "provider_raw"
    raw_dir.mkdir(parents=True)
    provider_raw_dir.mkdir(parents=True)
    start = 1_784_472_000_000
    end = start + 300_000
    market_id = "future-market-001"
    payloads: dict[str, list[dict]] = {
        "raw_polymarket_markets.jsonl": [
            {
                "market_id": market_id,
                "condition_id": "condition-001",
                "slug": f"btc-updown-5m-{start // 1000}",
                "market_family": "btc_updown_5m",
                "horizon_ms": 300_000,
                "market_start_ts": start,
                "market_end_ts": end,
                "settlement_ts": end,
                "up_token_id": "up-token-001",
                "down_token_id": "down-token-001",
                "reference_price_source": "polymarket_rtds_chainlink",
                "settlement_rule": "UP if end reference is at least start reference",
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        ],
        "raw_polymarket_orderbooks.jsonl": [],
        "raw_polymarket_trades.jsonl": [],
        "raw_binance_btcusdt_klines.jsonl": [],
        "raw_polymarket_chainlink_prices.jsonl": [],
        "raw_polymarket_resolutions.jsonl": [
            {
                "market_id": market_id,
                "resolved_outcome": "UP",
                "winning_outcome": "UP",
                "payout_up": 1.0,
                "payout_down": 0.0,
                "reference_price_source": "polymarket_rtds_chainlink",
                "resolution_source_type": "polymarket_clob_market_tokens",
                "resolution_status": "normal",
                **_resolution_safety(),
            }
        ],
    }
    for offset in range(-15, 5):
        ts = start + offset * 60_000
        payloads["raw_binance_btcusdt_klines.jsonl"].append(
            {
                "ts": ts,
                "available_at_ts": ts + 60_000,
                "open_price": 100_000.0 + offset,
                "high_price": 100_010.0 + offset,
                "low_price": 99_990.0 + offset,
                "close_price": 100_001.0 + offset,
                "volume": 1.0,
                "timeframe_ms": 60_000,
                "source": "binance_btcusdt",
            }
        )
    for offset in range(5):
        ts = start + offset * 60_000
        for outcome, bid, ask in (("UP", 0.54, 0.56), ("DOWN", 0.44, 0.46)):
            payloads["raw_polymarket_orderbooks.jsonl"].append(
                {
                    "market_id": market_id,
                    "token_id": f"{outcome.lower()}-token-001",
                    "outcome": outcome,
                    "ts": ts,
                    "available_at_ts": ts,
                    "bid_price": bid,
                    "ask_price": ask,
                    "mid_price": (bid + ask) / 2.0,
                    "bid_size": 100.0,
                    "ask_size": 100.0,
                    "liquidity_depth": 200.0,
                    "paper_only": True,
                    "capital_at_risk": False,
                    "polymarket_write_enabled": False,
                    "wallet_signing_enabled": False,
                }
            )
        payloads["raw_polymarket_chainlink_prices.jsonl"].append(
            {
                "source_ts": ts,
                "available_at_ts": ts,
                "price": 100_000.0 + offset,
                "source_type": "polymarket_rtds_chainlink",
                "symbol": "BTC/USD",
                "read_only": True,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        )
    for filename, rows in payloads.items():
        _write_jsonl(raw_dir / filename, rows)
        _write_jsonl(provider_raw_dir / filename, rows)
    _write_json(
        run_dir / "pending_round_capture_report.json",
        {
            "run_id": run_dir.name,
            "raw_polymarket_market_count": 1,
            "raw_orderbook_row_count": 10,
            "raw_trade_row_count": 0,
            "raw_btc_candle_row_count": 20,
            "pending_feature_enrichment": False,
            "paper_only": True,
            "capital_at_risk": False,
        },
    )
    _write_json(
        run_dir / "pending_round_capture_manifest.json",
        {"run_id": run_dir.name, "pending_resolution": True},
    )
    return run_dir


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _resolution_safety() -> dict[str, bool]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
