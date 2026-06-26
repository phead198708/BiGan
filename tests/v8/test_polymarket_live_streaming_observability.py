"""Streaming observability tests for Polymarket live paper runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import bigan.v8.polymarket.live.operator as live_operator
from bigan.v8.polymarket import PolymarketLivePaperConfig, run_polymarket_live_paper
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.live.binance_reference_feed import MockBinanceBTCReferenceFeed
from bigan.v8.polymarket.live.contracts import (
    PolymarketLiveMarket,
    compact_safety_fields,
)
from bigan.v8.polymarket.live.polymarket_feed import MockPolymarketLiveFeed
from bigan.v8.polymarket.training.contracts import (
    ACTION_VALUE_LABEL_ACTIONS,
    PRIMARY_POLICY_TARGET_ACTION_VALUE,
    PolymarketPolicyModel,
)


def test_streaming_status_and_event_files_are_written(tmp_path: Path) -> None:
    result = _run_streaming(tmp_path, "streaming-healthy")

    live_status = _read_json(result.artifact_paths["live_status"])
    assert live_status["operator_status"] == "completed"
    assert live_status["stage"] == "final"
    assert live_status["prediction_count"] == 9
    assert live_status["decision_count"] == 9
    assert live_status["trade_count"] > 0
    assert live_status["paper_only"] is True
    assert live_status["capital_at_risk"] is False
    assert live_status["polymarket_write_enabled"] is False
    assert live_status["wallet_signing_enabled"] is False
    assert result.artifact_paths["live_status_md"].read_text(encoding="utf-8")

    heartbeats = _read_jsonl(result.artifact_paths["operator_heartbeat"])
    signals = _read_jsonl(result.artifact_paths["signal_events"])
    executions = _read_jsonl(result.artifact_paths["execution_events"])
    positions = _read_jsonl(result.artifact_paths["position_snapshots"])
    pnl = _read_jsonl(result.artifact_paths["pnl_snapshots"])

    assert heartbeats
    assert heartbeats[-1]["operator_status"] == "completed"
    assert len(signals) == result.operator_manifest["prediction_count"]
    assert len(executions) == result.operator_manifest["decision_count"]
    assert positions
    assert pnl

    first_signal = signals[0]
    for field in (
        "expected_return_by_action",
        "best_policy_action",
        "best_action_expected_return",
        "second_best_action_expected_return",
        "best_action_margin",
        "policy_confidence",
        "action_value_model_family",
        "feature_conditioned_action_value_model_enabled",
        "model_manifest_sha256",
    ):
        assert field in first_signal
    first_execution = executions[0]
    for field in (
        "entry_policy_action",
        "intended_exit_policy",
        "planned_exit_before_ts",
        "policy_exit_reason",
        "action_value_head_used",
        "probability_ev_fallback_used",
    ):
        assert field in first_execution
    for row in (*heartbeats, *signals, *executions, *positions, *pnl):
        _assert_safe(row)


def test_streaming_fail_closed_writes_blocked_status(tmp_path: Path) -> None:
    result = _run_streaming(
        tmp_path,
        "streaming-model-mismatch",
        inject_model_manifest_mismatch=True,
    )

    live_status = _read_json(result.artifact_paths["live_status"])
    heartbeats = _read_jsonl(result.artifact_paths["operator_heartbeat"])

    assert result.operator_manifest["operator_status"] == "blocked_fail_closed"
    assert "model_manifest_mismatch" in result.operator_manifest["critical_reason_codes"]
    assert live_status["operator_status"] == "blocked_fail_closed"
    assert "model_manifest_mismatch" in live_status["critical_reason_codes"]
    assert heartbeats[-1]["operator_status"] == "blocked_fail_closed"
    assert "model_manifest_mismatch" in heartbeats[-1]["critical_reason_codes"]
    _assert_safe(live_status)


def test_streaming_files_do_not_enter_training_raw(tmp_path: Path) -> None:
    result = _run_streaming(tmp_path, "streaming-training-boundary")
    streaming_names = {
        "live_status.json",
        "live_status.md",
        "operator_heartbeat.jsonl",
        "signal_events.jsonl",
        "execution_events.jsonl",
        "position_snapshots.jsonl",
        "pnl_snapshots.jsonl",
    }

    for row in _read_jsonl(result.artifact_paths["training_raw_index"]):
        training_raw_dir = result.run_dir / row["training_raw_dir"]
        assert not (streaming_names & {path.name for path in training_raw_dir.rglob("*")})
        _assert_training_raw_is_model_output_free(training_raw_dir)


def test_streaming_preserves_final_audit_artifacts(tmp_path: Path) -> None:
    baseline = run_polymarket_live_paper(
        PolymarketLivePaperConfig(
            run_id="baseline",
            output_dir=tmp_path / "baseline",
            overwrite_existing=True,
        )
    )
    streaming = _run_streaming(tmp_path / "streaming", "baseline")

    for artifact_name in (
        "polymarket_model_predictions",
        "polymarket_ev_decisions",
        "polymarket_pnl_breakdown",
        "paper_observability_report",
        "rounds_index",
        "training_raw_index",
        "paper_audit_index",
        "paper_run_summary_latest",
    ):
        assert _sha256(baseline.artifact_paths[artifact_name]) == _sha256(
            streaming.artifact_paths[artifact_name]
        )
    for streaming_name in (
        "live_status",
        "operator_heartbeat",
        "signal_events",
        "execution_events",
        "position_snapshots",
        "pnl_snapshots",
    ):
        assert streaming_name not in streaming.operator_manifest["artifact_hashes"]


def test_real_live_streaming_emits_incremental_events_before_loader_returns(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_id = "real-streaming-incremental"
    run_dir = (tmp_path / run_id).resolve()
    counts_seen_before_loader_return: list[dict[str, int | str]] = []
    status_stages: list[str] = []

    original_write_status = live_operator._StreamingObservabilityWriter.write_status

    def tracing_write_status(self: Any, *args: Any, **kwargs: Any) -> None:
        status_stages.append(str(kwargs["stage"]))
        original_write_status(self, *args, **kwargs)

    def fake_real_live_feed_rows(
        config: PolymarketLivePaperConfig,
        *,
        streaming_writer: Any | None = None,
        on_feed_snapshot: Any | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        assert streaming_writer is not None
        assert on_feed_snapshot is not None

        polymarket_feed = MockPolymarketLiveFeed(config)
        market_rows = polymarket_feed.market_rows()
        markets = tuple(PolymarketLiveMarket(**row) for row in market_rows)
        orderbook_rows = polymarket_feed.orderbook_rows(markets)
        trade_rows = polymarket_feed.trade_rows(markets)
        reference_feed = MockBinanceBTCReferenceFeed(config)
        tick_rows = reference_feed.tick_rows(markets)
        candle_rows = reference_feed.candle_rows(markets)

        accumulated_orderbooks: list[dict[str, Any]] = []
        snapshots = _first_complete_orderbook_snapshots(orderbook_rows, limit=3)
        assert len(snapshots) == 3

        for snapshot_rows in snapshots:
            accumulated_orderbooks.extend(snapshot_rows)
            streaming_writer.record_feed_checkpoint(
                stage="collecting_feed",
                market_count=len(market_rows),
                latest_market_id=market_rows[-1]["market_id"],
                orderbook_count=len(accumulated_orderbooks),
                trade_count=len(trade_rows),
                tick_count=len(tick_rows),
                candle_count=len(candle_rows),
                force=True,
            )
            assert _read_json(run_dir / "live_status.json")["stage"] == "collecting_feed"

            on_feed_snapshot(
                market_rows=market_rows,
                orderbook_rows=list(accumulated_orderbooks),
                trade_rows=trade_rows,
                tick_rows=tick_rows,
                candle_rows=candle_rows,
            )
            status = _read_json(run_dir / "live_status.json")
            counts_seen_before_loader_return.append(
                {
                    "stage": status["stage"],
                    "signals": len(_read_jsonl(run_dir / "signal_events.jsonl")),
                    "executions": len(
                        _read_jsonl(run_dir / "execution_events.jsonl")
                    ),
                    "positions": len(
                        _read_jsonl(run_dir / "position_snapshots.jsonl")
                    ),
                    "pnl": len(_read_jsonl(run_dir / "pnl_snapshots.jsonl")),
                }
            )

        return market_rows, orderbook_rows, trade_rows, tick_rows, candle_rows

    monkeypatch.setattr(
        live_operator._StreamingObservabilityWriter,
        "write_status",
        tracing_write_status,
    )
    monkeypatch.setattr(live_operator, "load_real_live_feed_rows", fake_real_live_feed_rows)

    result = live_operator.run_polymarket_live_paper(
        PolymarketLivePaperConfig(
            run_id=run_id,
            output_dir=tmp_path,
            mock_live=False,
            market_families=("btc_updown_5m",),
            duration_seconds=120,
            poll_interval_seconds=5,
            stream_observability=True,
            status_interval_seconds=1,
            heartbeat_interval_seconds=1,
            flush_event_files=True,
            overwrite_existing=True,
        )
    )

    assert result.operator_manifest["operator_status"] == "completed"
    assert len(counts_seen_before_loader_return) == 3
    first, second = counts_seen_before_loader_return[:2]
    for key in ("signals", "executions", "positions", "pnl"):
        assert int(first[key]) > 0
        assert int(second[key]) > int(first[key])
    assert first["stage"] == "pnl_updated"
    assert second["stage"] == "pnl_updated"
    assert {
        "collecting_feed",
        "signals_generated",
        "decisions_generated",
        "pnl_updated",
    } <= set(status_stages)
    assert len(_read_jsonl(result.artifact_paths["signal_events"])) == 3
    assert len(_read_jsonl(result.artifact_paths["execution_events"])) == 3
    assert len(_read_jsonl(result.artifact_paths["pnl_snapshots"])) == 3
    assert _read_json(result.artifact_paths["live_status"])["stage"] == "final"


def test_real_live_streaming_state_matches_final_batch_replay(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_id = "real-streaming-stateful"
    run_dir = (tmp_path / run_id).resolve()
    model_manifest_path, model_path = _write_sell_before_close_model_artifacts(tmp_path)
    streaming_sequences_seen: list[list[tuple[Any, ...]]] = []

    def fake_real_live_feed_rows(
        config: PolymarketLivePaperConfig,
        *,
        streaming_writer: Any | None = None,
        on_feed_snapshot: Any | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        assert streaming_writer is not None
        assert on_feed_snapshot is not None

        polymarket_feed = MockPolymarketLiveFeed(config)
        market_rows = polymarket_feed.market_rows()
        markets = tuple(PolymarketLiveMarket(**row) for row in market_rows)
        orderbook_rows = _sell_before_close_orderbook_rows(
            polymarket_feed.orderbook_rows(markets),
            market=markets[0],
        )
        trade_rows = polymarket_feed.trade_rows(markets)
        reference_feed = MockBinanceBTCReferenceFeed(config)
        tick_rows = reference_feed.tick_rows(markets)
        candle_rows = reference_feed.candle_rows(markets)

        accumulated_orderbooks: list[dict[str, Any]] = []
        snapshots = _first_complete_orderbook_snapshots(orderbook_rows, limit=3)
        assert len(snapshots) == 3
        for snapshot_rows in snapshots:
            accumulated_orderbooks.extend(snapshot_rows)
            streaming_writer.record_feed_checkpoint(
                stage="collecting_feed",
                market_count=len(market_rows),
                latest_market_id=market_rows[-1]["market_id"],
                orderbook_count=len(accumulated_orderbooks),
                trade_count=len(trade_rows),
                tick_count=len(tick_rows),
                candle_count=len(candle_rows),
                force=True,
            )
            on_feed_snapshot(
                market_rows=market_rows,
                orderbook_rows=list(accumulated_orderbooks),
                trade_rows=trade_rows,
                tick_rows=tick_rows,
                candle_rows=candle_rows,
            )
            streaming_sequences_seen.append(
                _decision_sequence(
                    _read_jsonl(run_dir / "execution_events.jsonl")
                )
            )

        return market_rows, orderbook_rows, trade_rows, tick_rows, candle_rows

    monkeypatch.setattr(live_operator, "load_real_live_feed_rows", fake_real_live_feed_rows)

    result = live_operator.run_polymarket_live_paper(
        PolymarketLivePaperConfig(
            run_id=run_id,
            output_dir=tmp_path,
            mock_live=False,
            market_families=("btc_updown_5m",),
            model_manifest=model_manifest_path,
            model_path=model_path,
            stream_observability=True,
            status_interval_seconds=1,
            heartbeat_interval_seconds=1,
            flush_event_files=True,
            overwrite_existing=True,
        )
    )

    streaming_decisions = _read_jsonl(result.artifact_paths["execution_events"])
    final_decisions = _read_jsonl(result.artifact_paths["polymarket_ev_decisions"])

    assert result.operator_manifest["operator_status"] == "completed"
    assert result.operator_manifest["critical_reason_codes"] == []
    assert [row["action"] for row in streaming_decisions] == [
        "BUY_UP",
        "HOLD",
        "SELL_UP",
    ]
    assert streaming_sequences_seen == [
        [_decision_sequence(streaming_decisions)[0]],
        _decision_sequence(streaming_decisions)[:2],
        _decision_sequence(streaming_decisions),
    ]
    assert _decision_sequence(streaming_decisions) == _decision_sequence(final_decisions)
    assert final_decisions[0]["entry_policy_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert final_decisions[0]["intended_exit_policy"] == "sell_before_close"
    assert final_decisions[1]["action"] == "HOLD"
    assert final_decisions[1]["policy_exit_reason"] == "hold_until_exit_condition"
    assert final_decisions[2]["action"] == "SELL_UP"
    assert "planned_sell_before_close_exit" in final_decisions[2]["reason_codes"]

    final_pnl = _read_json(result.artifact_paths["polymarket_pnl_breakdown"])
    streaming_pnl = _read_jsonl(result.artifact_paths["pnl_snapshots"])[-1]
    assert round(streaming_pnl["estimated_total_pnl"], 12) == round(
        final_pnl["total_polymarket_pnl"],
        12,
    )


def _run_streaming(tmp_path: Path, run_id: str, **overrides):
    return run_polymarket_live_paper(
        PolymarketLivePaperConfig(
            run_id=run_id,
            output_dir=tmp_path,
            stream_observability=True,
            status_interval_seconds=1,
            heartbeat_interval_seconds=1,
            flush_event_files=True,
            overwrite_existing=True,
            **overrides,
        )
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_complete_orderbook_snapshots(
    orderbook_rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[list[dict[str, Any]]]:
    rows_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in orderbook_rows:
        rows_by_key.setdefault((str(row["market_id"]), int(row["ts"])), []).append(row)

    snapshots: list[list[dict[str, Any]]] = []
    for _key, rows in sorted(rows_by_key.items()):
        if {str(row["outcome"]) for row in rows} == {"UP", "DOWN"}:
            snapshots.append(rows)
        if len(snapshots) == limit:
            break
    return snapshots


def _sell_before_close_orderbook_rows(
    orderbook_rows: list[dict[str, Any]],
    *,
    market: PolymarketLiveMarket,
) -> list[dict[str, Any]]:
    snapshots = _first_complete_orderbook_snapshots(orderbook_rows, limit=3)
    replacement_ts = (
        market.market_start_ts,
        market.market_start_ts + 100_000,
        market.market_end_ts - 20_000,
    )
    rows: list[dict[str, Any]] = []
    for ts, snapshot in zip(replacement_ts, snapshots, strict=True):
        for row in snapshot:
            updated = dict(row)
            updated["ts"] = ts
            updated["received_ts"] = ts + 1_000
            rows.append(updated)
    return rows


def _decision_sequence(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    sequence = []
    for row in rows:
        decision_ts = row["ts"] if "ts" in row else row["decision_ts"]
        sequence.append(
            (
                row["market_id"],
                decision_ts,
                row["action"],
                row["selected_outcome"],
                row["entry_policy_action"],
                row["intended_exit_policy"],
                row["planned_exit_before_ts"],
                row["policy_exit_reason"],
            )
        )
    return sequence


def _write_sell_before_close_model_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    action_returns = dict.fromkeys(ACTION_VALUE_LABEL_ACTIONS, -0.05)
    action_returns["NO_TRADE"] = 0.0
    action_returns["BUY_UP_SELL_BEFORE_CLOSE"] = 0.08
    feature_columns = (
        "down_ask",
        "down_bid",
        "down_liquidity_depth",
        "down_mid",
        "time_to_close_seconds",
        "up_ask",
        "up_bid",
        "up_liquidity_depth",
        "up_mid",
    )
    feature_schema_hash = canonical_json_sha256({"feature_columns": list(feature_columns)})
    label_schema_hash = canonical_json_sha256({"target": "action_expected_net_return"})
    training_corpus_hash = canonical_json_sha256({"source": "stateful_streaming_test"})
    dataset_hash = canonical_json_sha256({"dataset": "stateful_streaming_test"})
    model = PolymarketPolicyModel(
        model_version="stateful_streaming_action_value_model",
        feature_columns=feature_columns,
        global_probability=0.80,
        market_family_probabilities={"btc_updown_5m": 0.80},
        family_feature_offsets={"btc_updown_5m": 0.0},
        feature_schema_hash=feature_schema_hash,
        label_schema_hash=label_schema_hash,
        training_corpus_hash=training_corpus_hash,
        dataset_hash=dataset_hash,
        train_row_count=3,
        primary_policy_target=PRIMARY_POLICY_TARGET_ACTION_VALUE,
        outcome_probability_head_enabled=True,
        action_value_head_enabled=True,
        compatibility_probability_fallback_enabled=True,
        action_value_model_family="market_family_mean_baseline",
        fallback_action_value_model_family="market_family_mean_baseline",
        feature_conditioned_action_value_model_enabled=False,
        global_action_returns=action_returns,
        market_family_action_returns={"btc_updown_5m": action_returns},
    )
    model_path = tmp_path / "stateful_streaming_model.json"
    manifest_path = tmp_path / "stateful_streaming_model_manifest.json"
    model_path.write_text(
        json.dumps(model.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "bigan-v8-polymarket-policy-v1",
        "model_version": model.model_version,
        "model_sha256": _sha256(model_path),
        "feature_schema_hash": model.feature_schema_hash,
        "label_schema_hash": model.label_schema_hash,
        "training_corpus_hash": model.training_corpus_hash,
        "dataset_hash": model.dataset_hash,
        "trained_model_used": True,
        "policy_signal_source": "trained_model",
        "synthetic_fixture_signal_used": False,
        "primary_policy_target": PRIMARY_POLICY_TARGET_ACTION_VALUE,
        "outcome_probability_head_enabled": True,
        "action_value_head_enabled": True,
        "compatibility_probability_fallback_enabled": True,
        "action_value_model_family": "market_family_mean_baseline",
        "feature_conditioned_action_value_model_enabled": False,
        "direct_pnl_optimization": False,
        "real_historical_corpus_used": False,
        "fixture_corpus_used": False,
        "synthetic_corpus_used": False,
        "fixture_model_used": False,
        "manual_live_evidence_eligible": False,
        **compact_safety_fields(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path, model_path


def _assert_training_raw_is_model_output_free(training_raw_dir: Path) -> None:
    forbidden_fields = {
        "estimated_up_probability",
        "p_up_auxiliary",
        "expected_return_by_action",
        "best_policy_action",
        "best_action_expected_return",
        "second_best_action_expected_return",
        "best_action_margin",
        "policy_confidence",
        "action_value_model_family",
        "feature_conditioned_action_value_model_enabled",
        "paper_action",
        "paper_pnl",
        "edge",
        "selected_side",
        "entry_policy_action",
        "intended_exit_policy",
        "planned_exit_before_ts",
        "policy_exit_reason",
    }
    for path in training_raw_dir.glob("raw_*.jsonl"):
        for row in _read_jsonl(path):
            assert not (forbidden_fields & set(row)), path.name


def _assert_safe(payload: dict) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["broker_exchange_write_enabled"] is False
    assert payload["live_exchange_write_enabled"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
