"""Operator for Polymarket live-data paper-only BTC UP/DOWN runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256, looks_like_sha256
from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS
from bigan.v8.polymarket.execution_ev import build_polymarket_ev_decisions
from bigan.v8.polymarket.ledger import PolymarketPositionLedger
from bigan.v8.polymarket.live.binance_reference_feed import MockBinanceBTCReferenceFeed
from bigan.v8.polymarket.live.contracts import (
    POLYMARKET_LIVE_PHASE,
    POLYMARKET_LIVE_SCHEMA_VERSION,
    BinanceBTCCandle,
    BinanceBTCReferenceTick,
    PolymarketLiveMarket,
    PolymarketLiveOrderBook,
    PolymarketLivePaperConfig,
    PolymarketLivePaperResult,
    PolymarketLiveTrade,
    compact_safety_fields,
    safety_fields,
)
from bigan.v8.polymarket.live.polymarket_feed import MockPolymarketLiveFeed
from bigan.v8.polymarket.live.real_feed_loader import load_real_live_feed_rows
from bigan.v8.polymarket.rules import build_btc_updown_resolution_rule, resolve_polymarket_rule
from bigan.v8.polymarket.training import (
    PolymarketPolicyExample,
    PolymarketPolicyModel,
    PolymarketPolicyTrainingConfig,
    predict_polymarket_policy_examples,
)

TRAINING_ELIGIBILITY_POLICY = "min_one_complete_book_sample"
STREAMING_ARTIFACT_NAMES = {
    "live_status",
    "live_status_md",
    "operator_heartbeat",
    "signal_events",
    "execution_events",
    "position_snapshots",
    "pnl_snapshots",
}


def run_polymarket_live_paper(
    config: PolymarketLivePaperConfig,
) -> PolymarketLivePaperResult:
    """Run a live-data, paper-only Polymarket operator pass."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"polymarket live paper run_dir already exists: {run_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    artifact_paths = _artifact_paths(run_dir, stream_observability=config.stream_observability)
    streaming = (
        _StreamingObservabilityWriter(config=config, artifact_paths=artifact_paths)
        if config.stream_observability
        else None
    )
    if streaming is not None:
        streaming.record_operator_start()
    reason_codes: list[str] = []
    markets: tuple[PolymarketLiveMarket, ...] = ()
    orderbooks: tuple[PolymarketLiveOrderBook, ...] = ()
    trades: tuple[PolymarketLiveTrade, ...] = ()
    ticks: tuple[BinanceBTCReferenceTick, ...] = ()
    candles: tuple[BinanceBTCCandle, ...] = ()
    predictions: tuple[Any, ...] = ()
    decisions: tuple[Any, ...] = ()
    ledger_events: list[dict[str, Any]] = []
    settlement_events: list[dict[str, Any]] = []
    model_manifest: dict[str, Any] = {}
    model_manifest_path: Path | None = None
    model_manifest_sha256 = ""
    model: PolymarketPolicyModel | None = None
    real_time_streaming = config.stream_observability and not config.mock_live

    if real_time_streaming:
        try:
            model, model_manifest, model_manifest_path = _load_or_create_model(config, run_dir)
            model_manifest_sha256 = _sha256_file(model_manifest_path)
            if config.inject_model_manifest_mismatch:
                model_manifest["model_sha256"] = "0" * 64
                raise ValueError("model_manifest_mismatch")
            _verify_model_manifest(model_manifest=model_manifest, model_path=model_manifest_path)
        except Exception as exc:
            reason_codes.extend(
                _exception_reason_codes(exc, fallback="model_manifest_mismatch")
            )
    snapshot_callback = (
        _RealTimeStreamingSnapshotProcessor(
            config=config,
            run_dir=run_dir,
            streaming_writer=streaming,
            model=model,
            model_manifest_sha256=model_manifest_sha256,
        )
        if real_time_streaming and streaming is not None and model is not None
        else None
    )

    if not reason_codes:
        try:
            market_rows, orderbook_rows, trade_rows, tick_rows, candle_rows = _load_feed_rows(
                config,
                streaming_writer=streaming,
                on_feed_snapshot=snapshot_callback,
            )
            if streaming is not None:
                streaming.record_feed_checkpoint(
                    stage="feed_loaded",
                    market_count=len(market_rows),
                    latest_market_id=_latest_market_id_from_rows(market_rows),
                    orderbook_count=len(orderbook_rows),
                    trade_count=len(trade_rows),
                    tick_count=len(tick_rows),
                    candle_count=len(candle_rows),
                    force=True,
                )
            _write_jsonl(artifact_paths["live_market_metadata"], market_rows)
            _write_jsonl(artifact_paths["live_token_orderbooks"], orderbook_rows)
            _write_jsonl(artifact_paths["live_token_trades"], trade_rows)
            _write_jsonl(artifact_paths["live_btc_reference_ticks"], tick_rows)
            _write_jsonl(artifact_paths["live_btc_reference_candles"], candle_rows)
            markets = tuple(PolymarketLiveMarket(**row) for row in market_rows)
            orderbooks = tuple(PolymarketLiveOrderBook(**row) for row in orderbook_rows)
            trades = tuple(PolymarketLiveTrade(**row) for row in trade_rows)
            ticks = tuple(BinanceBTCReferenceTick(**row) for row in tick_rows)
            candles = tuple(BinanceBTCCandle(**row) for row in candle_rows)
        except Exception as exc:
            reason_codes.extend(_exception_reason_codes(exc, fallback="feed_contract_violation"))
            _write_missing_feed_artifacts(artifact_paths)
    else:
        _write_missing_feed_artifacts(artifact_paths)

    feed_health = _feed_health(
        config=config,
        markets=markets,
        orderbooks=orderbooks,
        trades=trades,
        ticks=ticks,
        candles=candles,
        existing_reason_codes=tuple(reason_codes),
    )
    reason_codes.extend(feed_health["critical_reason_codes"])

    if model is None and not reason_codes:
        try:
            model, model_manifest, model_manifest_path = _load_or_create_model(config, run_dir)
            model_manifest_sha256 = _sha256_file(model_manifest_path)
            if config.inject_model_manifest_mismatch:
                model_manifest["model_sha256"] = "0" * 64
                raise ValueError("model_manifest_mismatch")
            _verify_model_manifest(model_manifest=model_manifest, model_path=model_manifest_path)
        except Exception as exc:
            reason_codes.extend(
                _exception_reason_codes(exc, fallback="model_manifest_mismatch")
            )

    if model is not None and not reason_codes:
        examples = _policy_examples(markets=markets, orderbooks=orderbooks)
        predictions = predict_polymarket_policy_examples(model, examples)
        if streaming is not None and not real_time_streaming:
            streaming.append_signal_events(
                predictions=predictions,
                model_manifest_sha256=model_manifest_sha256,
            )
        decisions = build_polymarket_ev_decisions(
            predictions=predictions,
            config=_ev_config(config, run_dir),
        )
        if streaming is not None and not real_time_streaming:
            streaming.append_execution_events(decisions=decisions)
        ledgers = _apply_decisions(markets=markets, decisions=decisions, config=config)
        if real_time_streaming and snapshot_callback is not None:
            try:
                snapshot_callback.assert_replay_equivalent(
                    markets=markets,
                    predictions=predictions,
                    decisions=decisions,
                    ledgers=ledgers,
                )
            except Exception as exc:
                reason_codes.extend(
                    _exception_reason_codes(exc, fallback="streaming_replay_mismatch")
                )
    else:
        ledgers = _empty_ledgers(markets)
    settlement_events = _settle_markets(
        config=config,
        markets=markets,
        candles=candles,
        ledgers=ledgers,
    )
    ledger_events = [event.to_dict() for ledger in ledgers.values() for event in ledger.events]
    if streaming is not None and not real_time_streaming:
        streaming.append_position_snapshots(
            markets=markets,
            ledger_events=ledger_events,
            decisions=decisions,
        )

    pnl_breakdown = _pnl_breakdown(
        config=config,
        markets=markets,
        ledgers=ledgers,
        decisions=decisions,
        settlement_events=settlement_events,
    )
    if streaming is not None and not real_time_streaming:
        streaming.append_pnl_snapshots(
            markets=markets,
            ledger_events=ledger_events,
            decisions=decisions,
            pnl_breakdown=pnl_breakdown,
        )
    if pnl_breakdown["unresolved_market_count"] > 0 and config.settlement_mode == "delayed":
        reason_codes.append("settlement_pending")
    status, recommendation = _status_and_recommendation(
        config=config,
        critical_reason_codes=tuple(reason_codes),
    )
    observability_report = _observability_report(
        config=config,
        feed_health=feed_health,
        pnl_breakdown=pnl_breakdown,
        status=status,
        recommendation=recommendation,
        reason_codes=tuple(reason_codes),
    )
    if streaming is not None:
        streaming.write_status(
            operator_status=status,
            stage="final",
            markets=markets,
            predictions=predictions,
            decisions=decisions,
            ledger_events=ledger_events,
            pnl_breakdown=pnl_breakdown,
            critical_reason_codes=tuple(reason_codes),
            force=True,
        )
        streaming.emit_heartbeat(
            stage="final",
            operator_status=status,
            force=True,
            critical_reason_codes=tuple(reason_codes),
            prediction_count=len(predictions),
            decision_count=len(decisions),
            trade_count=pnl_breakdown["trade_count"],
        )

    _write_jsonl(
        artifact_paths["polymarket_model_predictions"],
        [_with_full_safety(prediction.to_dict()) for prediction in predictions],
    )
    _write_jsonl(
        artifact_paths["polymarket_ev_decisions"],
        [_with_full_safety(decision.to_dict()) for decision in decisions],
    )
    _write_jsonl(artifact_paths["polymarket_position_ledger"], ledger_events)
    _write_jsonl(artifact_paths["polymarket_settlement_events"], settlement_events)
    _write_json(artifact_paths["polymarket_pnl_breakdown"], pnl_breakdown)
    _write_json(artifact_paths["paper_observability_report"], observability_report)
    _write_text(
        artifact_paths["paper_operator_summary"],
        _operator_summary_markdown(
            config=config,
            observability_report=observability_report,
            pnl_breakdown=pnl_breakdown,
        ),
    )
    round_artifacts = _write_round_artifacts(
        config=config,
        run_dir=run_dir,
        artifact_paths=artifact_paths,
        markets=markets,
        orderbooks=orderbooks,
        trades=trades,
        candles=candles,
        predictions=predictions,
        decisions=decisions,
        ledger_events=ledger_events,
        settlement_events=settlement_events,
        observability_report=observability_report,
        model_manifest=model_manifest,
        model_manifest_sha256=model_manifest_sha256,
        reason_codes=tuple(reason_codes),
        status=status,
        recommendation=recommendation,
    )

    core_hashes = _artifact_hashes(
        artifact_paths,
        exclude={
            "polymarket_live_operator_manifest",
            "github_paper_comment_payload",
            "github_paper_comment_md",
            *STREAMING_ARTIFACT_NAMES,
        },
    )
    operator_manifest = _operator_manifest(
        config=config,
        status=status,
        recommendation=recommendation,
        feed_health=feed_health,
        pnl_breakdown=pnl_breakdown,
        observability_report=observability_report,
        artifact_hashes=core_hashes,
        model_manifest=model_manifest,
        model_manifest_path=model_manifest_path,
        model_manifest_sha256=model_manifest_sha256,
        reason_codes=tuple(reason_codes),
        round_artifacts=round_artifacts,
    )
    _write_json(artifact_paths["polymarket_live_operator_manifest"], operator_manifest)
    operator_manifest_sha256 = _sha256_file(
        artifact_paths["polymarket_live_operator_manifest"]
    )
    comment_payload = _github_comment_payload(
        config=config,
        operator_manifest=operator_manifest,
        operator_manifest_sha256=operator_manifest_sha256,
        pnl_breakdown_sha256=_sha256_file(artifact_paths["polymarket_pnl_breakdown"]),
        observability_report_sha256=_sha256_file(
            artifact_paths["paper_observability_report"]
        ),
        round_artifacts=round_artifacts,
    )
    payload_hash = canonical_json_sha256(comment_payload)
    comment_payload["github_comment_payload_sha256"] = payload_hash
    _write_json(artifact_paths["github_paper_comment_payload"], comment_payload)
    _write_text(
        artifact_paths["github_paper_comment_md"],
        _github_comment_markdown(comment_payload),
    )
    if config.mode == "gh-command":
        _post_github_comment(config=config, body_path=artifact_paths["github_paper_comment_md"])

    return PolymarketLivePaperResult(
        run_dir=run_dir,
        artifact_paths=artifact_paths,
        operator_manifest=operator_manifest,
        observability_report=observability_report,
        pnl_breakdown=pnl_breakdown,
    )


def _load_feed_rows(
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
    if not config.mock_live:
        return load_real_live_feed_rows(
            config,
            streaming_writer=streaming_writer,
            on_feed_snapshot=on_feed_snapshot,
        )
    polymarket_feed = MockPolymarketLiveFeed(config)
    market_rows = polymarket_feed.market_rows()
    markets = tuple(PolymarketLiveMarket(**row) for row in market_rows)
    binance_feed = MockBinanceBTCReferenceFeed(config)
    return (
        market_rows,
        polymarket_feed.orderbook_rows(markets),
        polymarket_feed.trade_rows(markets),
        binance_feed.tick_rows(markets),
        binance_feed.candle_rows(markets),
    )


def _load_or_create_model(
    config: PolymarketLivePaperConfig,
    run_dir: Path,
) -> tuple[PolymarketPolicyModel, dict[str, Any], Path]:
    if config.model_manifest is None:
        model = _fixture_model()
        model_path = run_dir / "polymarket_live_fixture_model.json"
        manifest_path = run_dir / "polymarket_live_fixture_model_manifest.json"
        _write_json(model_path, model.to_dict())
        model_sha256 = _sha256_file(model_path)
        manifest = {
            "schema_version": "bigan-v8-polymarket-policy-v1",
            "model_version": model.model_version,
            "model_sha256": model_sha256,
            "feature_schema_hash": model.feature_schema_hash,
            "label_schema_hash": model.label_schema_hash,
            "training_corpus_hash": model.training_corpus_hash,
            "dataset_hash": model.dataset_hash,
            "trained_model_used": True,
            "policy_signal_source": "trained_model",
            "synthetic_fixture_signal_used": False,
            "primary_policy_target": "resolved_up_probability_only",
            "outcome_probability_head_enabled": True,
            "action_value_head_enabled": False,
            "compatibility_probability_fallback_enabled": True,
            "action_value_model_family": "resolved_up_probability_only",
            "feature_conditioned_action_value_model_enabled": False,
            "direct_pnl_optimization": False,
            "out_of_sample_replay": True,
            **compact_safety_fields(),
        }
        _write_json(manifest_path, manifest)
        return model, manifest, manifest_path

    manifest_path = config.model_manifest.expanduser().resolve()
    manifest = _read_json(manifest_path)
    model_path = (
        config.model_path.expanduser().resolve()
        if config.model_path is not None
        else manifest_path.with_name("polymarket_policy_model.json")
    )
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_path}")
    payload = _read_json(model_path)
    model = PolymarketPolicyModel(
        model_version=str(payload["model_version"]),
        feature_columns=tuple(payload["feature_columns"]),
        global_probability=float(payload["global_probability"]),
        market_family_probabilities={
            str(k): float(v)
            for k, v in dict(payload["market_family_probabilities"]).items()
        },
        family_feature_offsets={
            str(k): float(v)
            for k, v in dict(payload["family_feature_offsets"]).items()
        },
        feature_schema_hash=str(payload["feature_schema_hash"]),
        label_schema_hash=str(payload["label_schema_hash"]),
        training_corpus_hash=str(payload["training_corpus_hash"]),
        dataset_hash=str(payload["dataset_hash"]),
        train_row_count=int(payload["train_row_count"]),
        primary_policy_target=str(
            payload.get("primary_policy_target", "resolved_up_probability_only")
        ),
        outcome_probability_head_enabled=bool(
            payload.get("outcome_probability_head_enabled", True)
        ),
        action_value_head_enabled=bool(payload.get("action_value_head_enabled", False)),
        compatibility_probability_fallback_enabled=bool(
            payload.get("compatibility_probability_fallback_enabled", True)
        ),
        action_value_model_family=str(
            payload.get("action_value_model_family", "market_family_mean_baseline")
        ),
        fallback_action_value_model_family=str(
            payload.get("fallback_action_value_model_family", "market_family_mean_baseline")
        ),
        feature_conditioned_action_value_model_enabled=bool(
            payload.get("feature_conditioned_action_value_model_enabled", False)
        ),
        action_value_feature_columns=tuple(payload.get("action_value_feature_columns", ())),
        action_return_feature_means={
            str(feature): float(value)
            for feature, value in dict(payload.get("action_return_feature_means", {})).items()
        },
        action_return_feature_coefficients={
            str(action): {
                str(feature): float(value)
                for feature, value in dict(coefficients).items()
            }
            for action, coefficients in dict(
                payload.get("action_return_feature_coefficients", {})
            ).items()
        },
        global_action_returns={
            str(action): float(value)
            for action, value in dict(payload.get("global_action_returns", {})).items()
        },
        market_family_action_returns={
            str(family): {
                str(action): float(value)
                for action, value in dict(action_returns).items()
            }
            for family, action_returns in dict(
                payload.get("market_family_action_returns", {})
            ).items()
        },
        family_action_feature_offsets={
            str(family): {
                str(action): float(value)
                for action, value in dict(action_returns).items()
            }
            for family, action_returns in dict(
                payload.get("family_action_feature_offsets", {})
            ).items()
        },
    )
    expected = manifest.get("model_sha256")
    actual = _sha256_file(model_path)
    if expected != actual:
        raise ValueError("model_manifest_mismatch")
    return model, manifest, manifest_path


def _verify_model_manifest(*, model_manifest: dict[str, Any], model_path: Path) -> None:
    for field_name in (
        "model_sha256",
        "feature_schema_hash",
        "training_corpus_hash",
        "dataset_hash",
    ):
        value = str(model_manifest.get(field_name, ""))
        if not looks_like_sha256(value):
            raise ValueError(f"{field_name} must be SHA-256")
    if model_manifest.get("trained_model_used") is not True:
        raise ValueError("trained_model_used must be true")
    if model_manifest.get("policy_signal_source") != "trained_model":
        raise ValueError("policy_signal_source must be trained_model")
    if model_manifest.get("direct_pnl_optimization") is not False:
        raise ValueError("direct_pnl_optimization must be false")
    if model_manifest.get("real_historical_corpus_used") is True:
        if model_manifest.get("primary_policy_target") != "action_expected_net_return":
            raise ValueError("primary_policy_target must be action_expected_net_return")
        if model_manifest.get("action_value_head_enabled") is not True:
            raise ValueError("action_value_head_enabled must be true")
        if model_manifest.get("outcome_probability_head_enabled") is not True:
            raise ValueError("outcome_probability_head_enabled must be true")
        if model_manifest.get("feature_conditioned_action_value_model_enabled") is not True:
            raise ValueError("feature_conditioned_action_value_model_enabled must be true")
        for field_name in ("policy_dataset_hash", "split_hash"):
            value = str(model_manifest.get(field_name, ""))
            if not looks_like_sha256(value):
                raise ValueError(f"{field_name} must be SHA-256")
        if model_manifest.get("manual_live_evidence_eligible") is not True:
            raise ValueError("manual_live_evidence_eligible must be true")
        for field_name in (
            "fixture_corpus_used",
            "synthetic_corpus_used",
            "fixture_model_used",
            "synthetic_fixture_signal_used",
        ):
            if model_manifest.get(field_name) is not False:
                raise ValueError(f"{field_name} must be false")
    for field_name, expected in compact_safety_fields().items():
        if model_manifest.get(field_name) is not expected:
            raise ValueError(f"model manifest violates {field_name}")
    if model_path.name.endswith("manifest.json") and not model_path.exists():
        raise FileNotFoundError(model_path)


def _policy_examples(
    *,
    markets: tuple[PolymarketLiveMarket, ...],
    orderbooks: tuple[PolymarketLiveOrderBook, ...],
) -> tuple[PolymarketPolicyExample, ...]:
    by_market = {market.market_id: market for market in markets}
    books_by_key: dict[tuple[str, int], dict[str, PolymarketLiveOrderBook]] = defaultdict(dict)
    for book in orderbooks:
        books_by_key[(book.market_id, book.ts)][book.outcome] = book
    examples = []
    for (market_id, decision_ts), books in sorted(books_by_key.items()):
        if set(books) != {"UP", "DOWN"}:
            continue
        market = by_market[market_id]
        up = books["UP"]
        down = books["DOWN"]
        decision_ts = max(up.received_ts, down.received_ts)
        features = _live_features(market=market, decision_ts=decision_ts, up=up, down=down)
        examples.append(
            PolymarketPolicyExample(
                market_id=market.market_id,
                condition_id=market.condition_id,
                slug=market.slug,
                market_family=market.market_family,
                horizon_ms=market.horizon_ms,
                decision_ts=decision_ts,
                feature_cutoff_ts=decision_ts,
                max_input_ts=decision_ts,
                features=features,
                target_up_probability=0.5,
                resolved_outcome="UNKNOWN_50_50",
                resolution_status="unknown_50_50",
            )
        )
    return tuple(examples)


def _live_features(
    *,
    market: PolymarketLiveMarket,
    decision_ts: int,
    up: PolymarketLiveOrderBook,
    down: PolymarketLiveOrderBook,
) -> dict[str, float]:
    features = {
        "up_bid": up.bid_price,
        "up_ask": up.ask_price,
        "up_mid": up.mid_price,
        "down_bid": down.bid_price,
        "down_ask": down.ask_price,
        "down_mid": down.mid_price,
        "up_liquidity_depth": up.liquidity_depth,
        "down_liquidity_depth": down.liquidity_depth,
        "time_to_close_seconds": max(0.0, (market.market_end_ts - decision_ts) / 1000.0),
    }
    for family in sorted(BTC_UPDOWN_MARKET_HORIZONS_MS):
        features[f"family_{family}"] = 1.0 if family == market.market_family else 0.0
    return features


def _apply_decisions(
    *,
    markets: tuple[PolymarketLiveMarket, ...],
    decisions: tuple[Any, ...],
    config: PolymarketLivePaperConfig,
) -> dict[str, PolymarketPositionLedger]:
    ledgers = _empty_ledgers(markets)
    _apply_decisions_to_ledgers(ledgers=ledgers, decisions=decisions, config=config)
    return ledgers


def _apply_decisions_to_ledgers(
    *,
    ledgers: dict[str, PolymarketPositionLedger],
    decisions: tuple[Any, ...],
    config: PolymarketLivePaperConfig,
) -> None:
    for decision in sorted(decisions, key=lambda item: (item.decision_ts, item.market_id)):
        ledger = ledgers[decision.market_id]
        fees = decision.paper_notional * config.fee_rate
        slippage = decision.paper_notional * config.slippage_rate
        if decision.action == "BUY_UP":
            ledger.buy(
                ts=decision.decision_ts,
                outcome="UP",
                qty=decision.paper_notional / decision.execution_price,
                ask_price=decision.execution_price,
                fees=fees,
                slippage=slippage,
                reason_codes=tuple(decision.reason_codes),
            )
        elif decision.action == "BUY_DOWN":
            ledger.buy(
                ts=decision.decision_ts,
                outcome="DOWN",
                qty=decision.paper_notional / decision.execution_price,
                ask_price=decision.execution_price,
                fees=fees,
                slippage=slippage,
                reason_codes=tuple(decision.reason_codes),
            )
        elif decision.action == "SELL_UP":
            qty = ledger.position_snapshot()["position_up"]
            if qty > 0.0:
                ledger.sell(
                    ts=decision.decision_ts,
                    outcome="UP",
                    qty=qty,
                    bid_price=decision.execution_price,
                    fees=fees,
                    slippage=slippage,
                    reason_codes=tuple(decision.reason_codes),
                )
        elif decision.action == "SELL_DOWN":
            qty = ledger.position_snapshot()["position_down"]
            if qty > 0.0:
                ledger.sell(
                    ts=decision.decision_ts,
                    outcome="DOWN",
                    qty=qty,
                    bid_price=decision.execution_price,
                    fees=fees,
                    slippage=slippage,
                    reason_codes=tuple(decision.reason_codes),
                )
        elif decision.action == "HOLD":
            ledger.hold(ts=decision.decision_ts, reason_codes=tuple(decision.reason_codes))
        else:
            ledger.no_trade(ts=decision.decision_ts, reason_codes=tuple(decision.reason_codes))


def _settle_markets(
    *,
    config: PolymarketLivePaperConfig,
    markets: tuple[PolymarketLiveMarket, ...],
    candles: tuple[BinanceBTCCandle, ...],
    ledgers: dict[str, PolymarketPositionLedger],
) -> list[dict[str, Any]]:
    if config.stop_requested or config.settlement_mode == "delayed":
        return []
    candles_by_market = {candle.market_id: candle for candle in candles}
    settlement_events = []
    for market in markets:
        if not market.resolution_available:
            continue
        if not market.settlement_rule:
            continue
        candle = candles_by_market.get(market.market_id)
        ledger = ledgers.get(market.market_id)
        if candle is None or ledger is None:
            continue
        pre = ledger.position_snapshot()
        rule = build_btc_updown_resolution_rule(
            market_id=market.market_id,
            condition_id=market.condition_id,
            slug=market.slug,
            market_family=market.market_family,
            resolution_source=market.reference_price_source,
            candle_open_ts=market.market_start_ts,
            candle_close_ts=market.market_end_ts,
            raw_rule_text=market.settlement_rule,
        )
        resolution = resolve_polymarket_rule(
            rule,
            reference_price_start=market.reference_price_at_start,
            reference_price_end=candle.close_price,
        )
        event = ledger.settle(
            ts=market.settlement_ts,
            payout_up=resolution.payout_up,
            payout_down=resolution.payout_down,
            reason_codes=("phase1_settlement_engine", "paper_settlement"),
        )
        settlement_events.append(
            {
                "market_id": market.market_id,
                "condition_id": market.condition_id,
                "slug": market.slug,
                "resolution_status": resolution.resolution_status,
                "resolved_outcome": resolution.resolved_outcome,
                "payout_up": resolution.payout_up,
                "payout_down": resolution.payout_down,
                "reference_price_start": market.reference_price_at_start,
                "reference_price_end": candle.close_price,
                "qty_up_settled": pre["position_up"],
                "qty_down_settled": pre["position_down"],
                "settlement_cashflow": event.cash_delta,
                "settlement_pnl": event.settlement_pnl,
                "raw_resolution_sha256": resolution.raw_resolution_sha256,
                **safety_fields(),
            }
        )
    return settlement_events


def _feed_health(
    *,
    config: PolymarketLivePaperConfig,
    markets: tuple[PolymarketLiveMarket, ...],
    orderbooks: tuple[PolymarketLiveOrderBook, ...],
    trades: tuple[PolymarketLiveTrade, ...],
    ticks: tuple[BinanceBTCReferenceTick, ...],
    candles: tuple[BinanceBTCCandle, ...],
    existing_reason_codes: tuple[str, ...],
) -> dict[str, Any]:
    reason_codes = list(existing_reason_codes)
    missing_market_rule_count = sum(1 for market in markets if not market.settlement_rule)
    if missing_market_rule_count:
        reason_codes.append("missing_market_rule")
    expected_book_keys = {
        (market.market_id, book_ts)
        for market in markets
        for book_ts in {
            book.ts for book in orderbooks if book.market_id == market.market_id
        }
    }
    missing_token_book_count = 0
    for market_id, ts in expected_book_keys:
        outcomes = {
            book.outcome
            for book in orderbooks
            if book.market_id == market_id and book.ts == ts
        }
        if outcomes != {"UP", "DOWN"}:
            missing_token_book_count += 1
    if missing_token_book_count:
        reason_codes.append("missing_token_book")
    stale_orderbook_count = sum(
        int(book.received_ts - book.ts > config.max_stale_orderbook_seconds * 1000)
        for book in orderbooks
    )
    stale_reference_price_count = sum(
        int(tick.received_ts - tick.ts > config.max_stale_reference_seconds * 1000)
        for tick in ticks
    )
    if stale_orderbook_count:
        reason_codes.append("stale_orderbook")
    if stale_reference_price_count:
        reason_codes.append("stale_reference_price")
    candle_market_ids = {candle.market_id for candle in candles}
    missing_reference_candle_count = sum(
        int(market.market_id not in candle_market_ids) for market in markets
    )
    if missing_reference_candle_count:
        reason_codes.append("missing_reference_candle")
    return {
        "polymarket_metadata_event_count": len(markets),
        "polymarket_orderbook_event_count": len(orderbooks),
        "polymarket_trade_event_count": len(trades),
        "binance_reference_event_count": len(ticks) + len(candles),
        "provider_disconnect_count": 0,
        "provider_reconnect_count": 0,
        "provider_error_count": 0,
        "stale_orderbook_count": stale_orderbook_count,
        "stale_reference_price_count": stale_reference_price_count,
        "missing_market_rule_count": missing_market_rule_count,
        "missing_token_book_count": missing_token_book_count,
        "missing_reference_candle_count": missing_reference_candle_count,
        "settlement_pending_count": int(config.settlement_mode == "delayed") * len(markets),
        "settlement_resolved_count": int(config.settlement_mode == "resolved") * len(markets),
        "critical_reason_codes": sorted(set(reason_codes)),
        **safety_fields(),
    }


def _pnl_breakdown(
    *,
    config: PolymarketLivePaperConfig,
    markets: tuple[PolymarketLiveMarket, ...],
    ledgers: dict[str, PolymarketPositionLedger],
    decisions: tuple[Any, ...],
    settlement_events: list[dict[str, Any]],
) -> dict[str, Any]:
    realized = sum(ledger.realized_trade_pnl for ledger in ledgers.values())
    settlement = sum(ledger.settlement_pnl for ledger in ledgers.values())
    complete_set = sum(ledger.complete_set_pnl for ledger in ledgers.values())
    fees = sum(ledger.fees for ledger in ledgers.values())
    slippage = sum(ledger.slippage for ledger in ledgers.values())
    unrealized = sum(ledger.unrealized_mark_pnl for ledger in ledgers.values())
    total = realized + settlement + complete_set + unrealized - fees - slippage
    unresolved = 0
    for ledger in ledgers.values():
        snapshot = ledger.position_snapshot()
        unresolved += int(snapshot["position_up"] > 0.0) + int(snapshot["position_down"] > 0.0)
    action_counts = Counter(decision.action for decision in decisions)
    return {
        "schema_version": POLYMARKET_LIVE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "market_count": len(markets),
        "resolved_market_count": len({row["market_id"] for row in settlement_events}),
        "unresolved_market_count": unresolved,
        "prediction_count": len(decisions),
        "decision_count": len(decisions),
        "trade_count": sum(
            count for action, count in action_counts.items() if action.startswith(("BUY", "SELL"))
        ),
        "no_trade_count": action_counts.get("NO_TRADE", 0),
        "settled_position_count": sum(
            int(row["qty_up_settled"] > 0.0 or row["qty_down_settled"] > 0.0)
            for row in settlement_events
        ),
        "realized_trade_pnl": realized,
        "unrealized_mark_pnl": unrealized,
        "settlement_pnl": settlement,
        "complete_set_pnl": complete_set,
        "fees": fees,
        "slippage": slippage,
        "total_polymarket_pnl": total,
        **safety_fields(),
    }


def _observability_report(
    *,
    config: PolymarketLivePaperConfig,
    feed_health: dict[str, Any],
    pnl_breakdown: dict[str, Any],
    status: str,
    recommendation: str,
    reason_codes: tuple[str, ...],
) -> dict[str, Any]:
    critical = sorted(set(reason_codes))
    return {
        "schema_version": POLYMARKET_LIVE_SCHEMA_VERSION,
        "phase": POLYMARKET_LIVE_PHASE,
        "run_id": config.run_id,
        "operator_status": status,
        "operator_recommendation": recommendation,
        "critical_alert_count": len(critical),
        "critical_reason_codes": critical,
        "capital_deployment_allowed": False,
        "live_deployment_allowed": False,
        "feed_health": feed_health,
        "pnl_breakdown": pnl_breakdown,
        **safety_fields(),
    }


def _operator_manifest(
    *,
    config: PolymarketLivePaperConfig,
    status: str,
    recommendation: str,
    feed_health: dict[str, Any],
    pnl_breakdown: dict[str, Any],
    observability_report: dict[str, Any],
    artifact_hashes: dict[str, str],
    model_manifest: dict[str, Any],
    model_manifest_path: Path | None,
    model_manifest_sha256: str,
    reason_codes: tuple[str, ...],
    round_artifacts: dict[str, Any],
) -> dict[str, Any]:
    ended_at = _ended_at(config)
    return {
        "schema_version": POLYMARKET_LIVE_SCHEMA_VERSION,
        "phase": POLYMARKET_LIVE_PHASE,
        "run_id": config.run_id,
        "commit_sha": _current_commit_sha(),
        "model_manifest_path": None if model_manifest_path is None else model_manifest_path.name,
        "model_manifest_sha256": model_manifest_sha256,
        "model_version": model_manifest.get("model_version"),
        "real_historical_corpus_used": model_manifest.get(
            "real_historical_corpus_used", False
        ),
        "fixture_corpus_used": model_manifest.get("fixture_corpus_used", False),
        "synthetic_corpus_used": model_manifest.get("synthetic_corpus_used", False),
        "fixture_model_used": model_manifest.get("fixture_model_used", False),
        "manual_live_evidence_eligible": model_manifest.get(
            "manual_live_evidence_eligible", False
        ),
        "policy_dataset_hash": model_manifest.get(
            "policy_dataset_hash", model_manifest.get("dataset_hash")
        ),
        "split_hash": model_manifest.get("split_hash"),
        "market_families": list(config.market_families),
        "started_at": config.started_at,
        "ended_at": ended_at,
        "wall_clock_duration_seconds": 0 if config.stop_requested else config.duration_seconds,
        "live_polymarket_data": not config.mock_live,
        "live_binance_reference_data": not config.mock_live,
        "deterministic_replay": config.mock_live,
        "operator_status": status,
        "operator_recommendation": recommendation,
        "capital_deployment_allowed": False,
        "live_deployment_allowed": False,
        **feed_health,
        "critical_alert_count": len(set(reason_codes)),
        "critical_reason_codes": sorted(set(reason_codes)),
        **{
            key: pnl_breakdown[key]
            for key in (
                "market_count",
                "resolved_market_count",
                "unresolved_market_count",
                "prediction_count",
                "decision_count",
                "trade_count",
                "no_trade_count",
                "settled_position_count",
                "realized_trade_pnl",
                "settlement_pnl",
                "total_polymarket_pnl",
            )
        },
        "observability_passed": observability_report["critical_alert_count"] == 0,
        "artifact_hashes": artifact_hashes,
        "round_artifact_export_mode": round_artifacts["round_artifact_export_mode"],
        "round_artifacts_written": round_artifacts["round_artifacts_written"],
        "training_raw_round_count": round_artifacts["training_raw_round_count"],
        "paper_audit_round_count": round_artifacts["paper_audit_round_count"],
        "latest_round_summary": round_artifacts["latest_round_summary"],
        "latest_run_summary": round_artifacts["latest_run_summary"],
        "latest_round_summary_sha256": round_artifacts.get("latest_round_summary_sha256"),
        "latest_run_summary_sha256": round_artifacts.get("latest_run_summary_sha256"),
        **safety_fields(),
    }


def _github_comment_payload(
    *,
    config: PolymarketLivePaperConfig,
    operator_manifest: dict[str, Any],
    operator_manifest_sha256: str,
    pnl_breakdown_sha256: str,
    observability_report_sha256: str,
    round_artifacts: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        key: operator_manifest[key]
        for key in (
            "run_id",
            "commit_sha",
            "model_manifest_sha256",
            "real_historical_corpus_used",
            "fixture_corpus_used",
            "synthetic_corpus_used",
            "fixture_model_used",
            "manual_live_evidence_eligible",
            "policy_dataset_hash",
            "split_hash",
            "market_families",
            "started_at",
            "ended_at",
            "wall_clock_duration_seconds",
            "live_polymarket_data",
            "live_binance_reference_data",
            "deterministic_replay",
            "market_count",
            "resolved_market_count",
            "unresolved_market_count",
            "prediction_count",
            "decision_count",
            "trade_count",
            "no_trade_count",
            "settled_position_count",
            "realized_trade_pnl",
            "settlement_pnl",
            "total_polymarket_pnl",
            "provider_disconnect_count",
            "provider_error_count",
            "stale_orderbook_count",
            "stale_reference_price_count",
            "critical_alert_count",
            "operator_recommendation",
            "paper_only",
            "capital_at_risk",
            "polymarket_write_enabled",
            "wallet_signing_enabled",
            "broker_exchange_write_enabled",
            "live_exchange_write_enabled",
        )
    }
    payload.update(
        {
            "repo_full_name": config.repo_full_name,
            "issue_number": config.issue_number,
            "operator_manifest_sha256": operator_manifest_sha256,
            "pnl_breakdown_sha256": pnl_breakdown_sha256,
            "observability_report_sha256": observability_report_sha256,
            "round_artifact_export_mode": round_artifacts[
                "round_artifact_export_mode"
            ],
            "last_completed_round_id": round_artifacts["latest_run_summary"].get(
                "last_completed_round_id"
            ),
            "rounds_seen": round_artifacts["latest_run_summary"].get("rounds_seen", 0),
            "rounds_resolved": round_artifacts["latest_run_summary"].get(
                "rounds_resolved", 0
            ),
            "rounds_failed_closed": round_artifacts["latest_run_summary"].get(
                "rounds_failed_closed", 0
            ),
            "rounds_pending_resolution": round_artifacts["latest_run_summary"].get(
                "rounds_pending_resolution", 0
            ),
            "training_raw_round_count": round_artifacts["training_raw_round_count"],
            "paper_audit_round_count": round_artifacts["paper_audit_round_count"],
            "latest_round_summary_sha256": round_artifacts.get(
                "latest_round_summary_sha256"
            ),
            "latest_run_summary_sha256": round_artifacts.get(
                "latest_run_summary_sha256"
            ),
        }
    )
    return payload


def _github_comment_markdown(payload: dict[str, Any]) -> str:
    fields = (
        "run_id",
        "commit_sha",
        "model_manifest_sha256",
        "market_families",
        "started_at",
        "ended_at",
        "wall_clock_duration_seconds",
        "live_polymarket_data",
        "live_binance_reference_data",
        "deterministic_replay",
        "market_count",
        "resolved_market_count",
        "unresolved_market_count",
        "prediction_count",
        "decision_count",
        "trade_count",
        "no_trade_count",
        "settled_position_count",
        "last_completed_round_id",
        "rounds_seen",
        "rounds_resolved",
        "rounds_failed_closed",
        "rounds_pending_resolution",
        "training_raw_round_count",
        "paper_audit_round_count",
        "realized_trade_pnl",
        "settlement_pnl",
        "total_polymarket_pnl",
        "provider_disconnect_count",
        "provider_error_count",
        "stale_orderbook_count",
        "stale_reference_price_count",
        "critical_alert_count",
        "operator_recommendation",
        "paper_only",
        "capital_at_risk",
        "broker_exchange_write_enabled",
        "live_exchange_write_enabled",
        "polymarket_write_enabled",
        "wallet_signing_enabled",
        "operator_manifest_sha256",
        "pnl_breakdown_sha256",
        "observability_report_sha256",
        "round_artifact_export_mode",
        "latest_round_summary_sha256",
        "latest_run_summary_sha256",
        "github_comment_payload_sha256",
    )
    lines = ["## v8 Polymarket live paper evidence", ""]
    for field in fields:
        value = payload.get(field)
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        lines.append(f"- {field}: {value}")
    lines.append("")
    return "\n".join(lines)


def _operator_summary_markdown(
    *,
    config: PolymarketLivePaperConfig,
    observability_report: dict[str, Any],
    pnl_breakdown: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Polymarket Live Paper Operator Summary",
            "",
            f"- run_id: {config.run_id}",
            f"- operator_status: {observability_report['operator_status']}",
            f"- operator_recommendation: {observability_report['operator_recommendation']}",
            f"- critical_alert_count: {observability_report['critical_alert_count']}",
            f"- prediction_count: {pnl_breakdown['prediction_count']}",
            f"- decision_count: {pnl_breakdown['decision_count']}",
            f"- trade_count: {pnl_breakdown['trade_count']}",
            f"- settlement_pnl: {pnl_breakdown['settlement_pnl']}",
            f"- total_polymarket_pnl: {pnl_breakdown['total_polymarket_pnl']}",
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )


def _write_round_artifacts(
    *,
    config: PolymarketLivePaperConfig,
    run_dir: Path,
    artifact_paths: dict[str, Path],
    markets: tuple[PolymarketLiveMarket, ...],
    orderbooks: tuple[PolymarketLiveOrderBook, ...],
    trades: tuple[PolymarketLiveTrade, ...],
    candles: tuple[BinanceBTCCandle, ...],
    predictions: tuple[Any, ...],
    decisions: tuple[Any, ...],
    ledger_events: list[dict[str, Any]],
    settlement_events: list[dict[str, Any]],
    observability_report: dict[str, Any],
    model_manifest: dict[str, Any],
    model_manifest_sha256: str,
    reason_codes: tuple[str, ...],
    status: str,
    recommendation: str,
) -> dict[str, Any]:
    rounds_root = run_dir / "rounds"
    rounds_root.mkdir(parents=True, exist_ok=True)
    candles_by_market = {candle.market_id: candle for candle in candles}
    settlement_by_market = {row["market_id"]: row for row in settlement_events}
    predictions_by_market = _group_dict_rows(
        [_with_full_safety(prediction.to_dict()) for prediction in predictions],
        key="market_id",
    )
    decisions_by_market = _group_dict_rows(
        [_with_full_safety(decision.to_dict()) for decision in decisions],
        key="market_id",
    )
    ledger_by_market = _group_dict_rows(ledger_events, key="market_id")
    orderbooks_by_market = _group_objects(orderbooks)
    trades_by_market = _group_objects(trades)

    round_index_rows: list[dict[str, Any]] = []
    training_index_rows: list[dict[str, Any]] = []
    paper_audit_index_rows: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    latest_run_summary: dict[str, Any] = _empty_run_summary(
        config=config,
        status=status,
        recommendation=recommendation,
        reason_codes=reason_codes,
    )
    latest_round_summary_sha256 = None
    latest_run_summary_sha256 = None

    for market in sorted(markets, key=lambda item: (item.market_start_ts, item.market_id)):
        finalized = _finalize_round_artifacts(
            config=config,
            run_dir=run_dir,
            rounds_root=rounds_root,
            market=market,
            orderbooks=orderbooks_by_market.get(market.market_id, []),
            trades=trades_by_market.get(market.market_id, []),
            candle=candles_by_market.get(market.market_id),
            predictions=predictions_by_market.get(market.market_id, []),
            decisions=decisions_by_market.get(market.market_id, []),
            ledger_events=ledger_by_market.get(market.market_id, []),
            settlement=settlement_by_market.get(market.market_id),
            observability_report=observability_report,
            completed_round_summaries=round_summaries,
            model_manifest=model_manifest,
            model_manifest_sha256=model_manifest_sha256,
            run_reason_codes=reason_codes,
            status=status,
            recommendation=recommendation,
        )
        round_summaries.append(finalized["round_summary"])
        index_row = finalized["index_row"]
        round_index_rows.append(index_row)
        paper_audit_index_rows.append(index_row)
        if finalized["training_eligible"]:
            training_index_rows.append(index_row)
        latest_run_summary = finalized["latest_run_summary"]
        latest_round_summary_sha256 = finalized["latest_round_summary_sha256"]
        latest_run_summary_sha256 = finalized["latest_run_summary_sha256"]
        _write_round_lifecycle_indexes(
            artifact_paths=artifact_paths,
            round_index_rows=round_index_rows,
            training_index_rows=training_index_rows,
            paper_audit_index_rows=paper_audit_index_rows,
            latest_run_summary=latest_run_summary,
        )

    _write_round_lifecycle_indexes(
        artifact_paths=artifact_paths,
        round_index_rows=round_index_rows,
        training_index_rows=training_index_rows,
        paper_audit_index_rows=paper_audit_index_rows,
        latest_run_summary=latest_run_summary,
    )
    return {
        "round_artifact_export_mode": "round_finalization_lifecycle",
        "round_artifacts_written": len(round_index_rows),
        "training_raw_round_count": len(training_index_rows),
        "paper_audit_round_count": len(paper_audit_index_rows),
        "latest_round_summary": round_summaries[-1] if round_summaries else {},
        "latest_run_summary": latest_run_summary,
        "latest_round_summary_sha256": latest_round_summary_sha256,
        "latest_run_summary_sha256": latest_run_summary_sha256,
    }


def _finalize_round_artifacts(
    *,
    config: PolymarketLivePaperConfig,
    run_dir: Path,
    rounds_root: Path,
    market: PolymarketLiveMarket,
    orderbooks: list[PolymarketLiveOrderBook],
    trades: list[PolymarketLiveTrade],
    candle: BinanceBTCCandle | None,
    predictions: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    ledger_events: list[dict[str, Any]],
    settlement: dict[str, Any] | None,
    observability_report: dict[str, Any],
    completed_round_summaries: list[dict[str, Any]],
    model_manifest: dict[str, Any],
    model_manifest_sha256: str,
    run_reason_codes: tuple[str, ...],
    status: str,
    recommendation: str,
) -> dict[str, Any]:
    round_id = _round_id(market)
    round_dir = rounds_root / round_id
    training_raw_dir = round_dir / "training_raw"
    paper_audit_dir = round_dir / "paper_audit"
    round_dir.mkdir(parents=True, exist_ok=True)
    paper_audit_dir.mkdir(parents=True, exist_ok=True)

    book_coverage = _book_coverage_metrics(orderbooks)
    training_eligible = (
        settlement is not None
        and candle is not None
        and book_coverage["complete_up_down_book_sample_count"] > 0
    )
    fail_closed_reason_codes = _round_fail_closed_reason_codes(
        training_eligible=training_eligible,
        has_settlement=settlement is not None,
        has_candle=candle is not None,
        book_coverage=book_coverage,
    )
    round_feed_health = _round_feed_health(
        book_coverage=book_coverage,
        has_candle=candle is not None,
    )
    round_model_health = _round_model_health(
        predictions=predictions,
        decisions=decisions,
        run_reason_codes=run_reason_codes,
    )
    round_resolution_health = _round_resolution_health(settlement=settlement, candle=candle)

    paper_audit_hashes = _write_paper_audit_bundle(
        paper_audit_dir=paper_audit_dir,
        predictions=predictions,
        decisions=decisions,
        ledger_events=ledger_events,
        settlement_events=[] if settlement is None else [settlement],
        observability_report=observability_report,
        market_id=market.market_id,
    )
    training_manifest_sha256 = None
    if training_eligible and candle is not None and settlement is not None:
        training_manifest_sha256 = _write_training_raw_bundle(
            config=config,
            training_raw_dir=training_raw_dir,
            market=market,
            orderbooks=orderbooks,
            trades=trades,
            candle=candle,
            settlement=settlement,
            book_coverage=book_coverage,
            model_manifest=model_manifest,
            model_manifest_sha256=model_manifest_sha256,
        )

    round_summary = _round_summary(
        market=market,
        candle=candle,
        settlement=settlement,
        predictions=predictions,
        decisions=decisions,
        ledger_events=ledger_events,
        reason_codes=fail_closed_reason_codes,
        training_eligible=training_eligible,
        book_coverage=book_coverage,
        round_feed_health=round_feed_health,
        round_model_health=round_model_health,
        round_resolution_health=round_resolution_health,
        model_manifest=model_manifest,
        model_manifest_sha256=model_manifest_sha256,
    )
    round_summary_path = round_dir / "round_summary.json"
    round_summary_md_path = round_dir / "round_summary.md"
    _write_json(round_summary_path, round_summary)
    _write_text(round_summary_md_path, _round_summary_markdown(round_summary))
    round_summary_sha256 = _sha256_file(round_summary_path)

    latest_run_summary = _run_summary_after_round(
        config=config,
        round_summaries=[*completed_round_summaries, round_summary],
        status=status,
        recommendation=recommendation,
        reason_codes=run_reason_codes,
    )
    run_summary_path = round_dir / "run_summary_after_round.json"
    run_summary_md_path = round_dir / "run_summary_after_round.md"
    _write_json(run_summary_path, latest_run_summary)
    _write_text(run_summary_md_path, _run_summary_markdown(latest_run_summary))

    index_row = {
        "round_id": round_id,
        "market_id": market.market_id,
        "slug": market.slug,
        "market_family": market.market_family,
        "round_dir": str(round_dir.relative_to(run_dir)),
        "training_raw_dir": str(training_raw_dir.relative_to(run_dir))
        if training_eligible
        else None,
        "paper_audit_dir": str(paper_audit_dir.relative_to(run_dir)),
        "round_summary_sha256": round_summary_sha256,
        "training_manifest_sha256": training_manifest_sha256,
        "paper_audit_manifest_sha256": paper_audit_hashes["paper_audit_manifest"],
        "resolution_status": round_summary["resolution_status"],
        "resolved_outcome": round_summary["resolved_outcome"],
        "training_eligible": training_eligible,
        "round_training_eligible": training_eligible,
        "round_reason_codes": fail_closed_reason_codes,
        "fail_closed_reason_codes": fail_closed_reason_codes,
        "training_eligibility_policy": TRAINING_ELIGIBILITY_POLICY,
        **book_coverage,
        **safety_fields(),
    }
    return {
        "round_summary": round_summary,
        "index_row": index_row,
        "training_eligible": training_eligible,
        "latest_run_summary": latest_run_summary,
        "latest_round_summary_sha256": round_summary_sha256,
        "latest_run_summary_sha256": _sha256_file(run_summary_path),
    }


def finalize_polymarket_round_artifacts(**kwargs: Any) -> dict[str, Any]:
    """Shared round finalization service for batch and async settlement paths."""

    return _finalize_round_artifacts(**kwargs)


def _write_round_lifecycle_indexes(
    *,
    artifact_paths: dict[str, Path],
    round_index_rows: list[dict[str, Any]],
    training_index_rows: list[dict[str, Any]],
    paper_audit_index_rows: list[dict[str, Any]],
    latest_run_summary: dict[str, Any],
) -> None:
    _write_jsonl(artifact_paths["rounds_index"], round_index_rows)
    _write_jsonl(artifact_paths["training_raw_index"], training_index_rows)
    _write_jsonl(artifact_paths["paper_audit_index"], paper_audit_index_rows)
    _write_json(artifact_paths["paper_run_summary_latest"], latest_run_summary)


def write_polymarket_round_lifecycle_indexes(
    *,
    artifact_paths: dict[str, Path],
    round_index_rows: list[dict[str, Any]],
    training_index_rows: list[dict[str, Any]],
    paper_audit_index_rows: list[dict[str, Any]],
    latest_run_summary: dict[str, Any],
) -> None:
    """Flush round lifecycle indexes after each durable round finalization."""

    _write_round_lifecycle_indexes(
        artifact_paths=artifact_paths,
        round_index_rows=round_index_rows,
        training_index_rows=training_index_rows,
        paper_audit_index_rows=paper_audit_index_rows,
        latest_run_summary=latest_run_summary,
    )


def _write_training_raw_bundle(
    *,
    config: PolymarketLivePaperConfig,
    training_raw_dir: Path,
    market: PolymarketLiveMarket,
    orderbooks: list[PolymarketLiveOrderBook],
    trades: list[PolymarketLiveTrade],
    candle: BinanceBTCCandle,
    settlement: dict[str, Any],
    book_coverage: dict[str, Any],
    model_manifest: dict[str, Any],
    model_manifest_sha256: str,
) -> str:
    training_raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_polymarket_markets": training_raw_dir / "raw_polymarket_markets.jsonl",
        "raw_polymarket_orderbooks": training_raw_dir / "raw_polymarket_orderbooks.jsonl",
        "raw_polymarket_trades": training_raw_dir / "raw_polymarket_trades.jsonl",
        "raw_binance_btcusdt_klines": training_raw_dir / "raw_binance_btcusdt_klines.jsonl",
        "raw_polymarket_resolutions": training_raw_dir / "raw_polymarket_resolutions.jsonl",
    }
    _write_jsonl(paths["raw_polymarket_markets"], [_training_market_row(market)])
    _write_jsonl(
        paths["raw_polymarket_orderbooks"],
        [_training_orderbook_row(row) for row in sorted(orderbooks, key=lambda item: (item.ts, item.outcome))],
    )
    _write_jsonl(
        paths["raw_polymarket_trades"],
        [_training_trade_row(row) for row in sorted(trades, key=lambda item: (item.ts, item.outcome))],
    )
    _write_jsonl(
        paths["raw_binance_btcusdt_klines"],
        [
            _training_feature_reference_candle_row(market),
            _training_candle_row(candle),
        ],
    )
    _write_jsonl(
        paths["raw_polymarket_resolutions"],
        [_training_resolution_row(market=market, candle=candle, settlement=settlement)],
    )
    artifact_hashes = {path.name: _sha256_file(path) for path in sorted(paths.values())}
    manifest = {
        "schema_version": "bigan-v8-polymarket-live-round-training-raw-v1",
        "phase": "polymarket_live_round_training_raw",
        "round_id": _round_id(market),
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "slug": market.slug,
        "training_eligible": True,
        "round_training_eligible": True,
        "training_eligibility_policy": TRAINING_ELIGIBILITY_POLICY,
        "phase2_raw_compatible": True,
        "source_operator_run_id": config.run_id,
        "source_round_id": _round_id(market),
        "source_market_id": market.market_id,
        "source_model_run_id": _model_run_id(model_manifest),
        "source_model_manifest_sha256": model_manifest_sha256,
        "source_collection_mode": "mock_live" if config.mock_live else "live_readonly",
        "live_polymarket_data": not config.mock_live,
        "live_btc_reference_data": not config.mock_live,
        "live_binance_reference_data": not config.mock_live,
        "deterministic_replay": config.mock_live,
        "round_finalization_only": bool(
            model_manifest.get("round_finalization_only", False)
        ),
        "model_signal_used": bool(model_manifest.get("model_signal_used", True)),
        "paper_decision_used": bool(model_manifest.get("paper_decision_used", True)),
        "paper_audit_only": bool(model_manifest.get("paper_audit_only", False)),
        **book_coverage,
        "artifact_hashes": artifact_hashes,
        "excluded_audit_fields": [
            "model_prediction",
            "model_probability",
            "paper_action",
            "paper_pnl",
            "selected_side",
            "edge",
            "ev_buy_up",
            "ev_buy_down",
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
            "entry_policy_action",
            "intended_exit_policy",
            "planned_exit_before_ts",
            "policy_exit_reason",
        ],
        **safety_fields(),
    }
    manifest_path = training_raw_dir / "round_training_manifest.json"
    _write_json(manifest_path, manifest)
    return _sha256_file(manifest_path)


def _write_paper_audit_bundle(
    *,
    paper_audit_dir: Path,
    predictions: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    ledger_events: list[dict[str, Any]],
    settlement_events: list[dict[str, Any]],
    observability_report: dict[str, Any],
    market_id: str,
) -> dict[str, str]:
    paths = {
        "polymarket_model_predictions": paper_audit_dir / "polymarket_model_predictions.jsonl",
        "polymarket_ev_decisions": paper_audit_dir / "polymarket_ev_decisions.jsonl",
        "polymarket_position_ledger": paper_audit_dir / "polymarket_position_ledger.jsonl",
        "polymarket_settlement_events": paper_audit_dir / "polymarket_settlement_events.jsonl",
        "polymarket_pnl_breakdown": paper_audit_dir / "polymarket_pnl_breakdown.json",
        "paper_observability_report": paper_audit_dir / "paper_observability_report.json",
    }
    _write_jsonl(paths["polymarket_model_predictions"], predictions)
    _write_jsonl(paths["polymarket_ev_decisions"], decisions)
    _write_jsonl(paths["polymarket_position_ledger"], ledger_events)
    _write_jsonl(paths["polymarket_settlement_events"], settlement_events)
    _write_json(paths["polymarket_pnl_breakdown"], _audit_pnl_breakdown(market_id, ledger_events))
    _write_json(paths["paper_observability_report"], observability_report)
    artifact_hashes = {name: _sha256_file(path) for name, path in sorted(paths.items())}
    manifest = {
        "schema_version": "bigan-v8-polymarket-live-round-paper-audit-v1",
        "phase": "polymarket_live_round_paper_audit",
        "market_id": market_id,
        "artifact_hashes": artifact_hashes,
        "round_finalization_only": bool(
            observability_report.get("round_finalization_only", False)
        ),
        "model_signal_used": bool(
            observability_report.get("model_signal_used", bool(predictions))
        ),
        "paper_decision_used": bool(
            observability_report.get("paper_decision_used", bool(decisions))
        ),
        "paper_audit_only": bool(observability_report.get("paper_audit_only", False)),
        **safety_fields(),
    }
    manifest_path = paper_audit_dir / "paper_audit_manifest.json"
    _write_json(manifest_path, manifest)
    artifact_hashes["paper_audit_manifest"] = _sha256_file(manifest_path)
    return artifact_hashes


def _round_summary(
    *,
    market: PolymarketLiveMarket,
    candle: BinanceBTCCandle | None,
    settlement: dict[str, Any] | None,
    predictions: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    ledger_events: list[dict[str, Any]],
    reason_codes: list[str],
    training_eligible: bool,
    book_coverage: dict[str, Any],
    round_feed_health: dict[str, Any],
    round_model_health: dict[str, Any],
    round_resolution_health: dict[str, Any],
    model_manifest: dict[str, Any],
    model_manifest_sha256: str,
) -> dict[str, Any]:
    last_event = ledger_events[-1] if ledger_events else {}
    action_counts = Counter(row["action"] for row in decisions)
    return {
        "schema_version": "bigan-v8-polymarket-live-round-summary-v1",
        "round_id": _round_id(market),
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "slug": market.slug,
        "market_family": market.market_family,
        "market_start_ts": market.market_start_ts,
        "market_end_ts": market.market_end_ts,
        "settlement_ts": market.settlement_ts,
        "resolution_status": None if settlement is None else settlement["resolution_status"],
        "resolved_outcome": None if settlement is None else settlement["resolved_outcome"],
        "payout_up": None if settlement is None else settlement["payout_up"],
        "payout_down": None if settlement is None else settlement["payout_down"],
        "reference_price_start": market.reference_price_at_start,
        "reference_price_end": None if candle is None else candle.close_price,
        "prediction_count": len(predictions),
        "decision_count": len(decisions),
        "action_counts": dict(sorted(action_counts.items())),
        "paper_position_up": float(last_event.get("position_up", 0.0)),
        "paper_position_down": float(last_event.get("position_down", 0.0)),
        "realized_trade_pnl": float(last_event.get("realized_trade_pnl", 0.0)),
        "settlement_pnl": float(last_event.get("settlement_pnl", 0.0)),
        "total_pnl": float(last_event.get("total_pnl", 0.0)),
        "reason_codes": reason_codes,
        "round_reason_codes": reason_codes,
        "round_feed_health": round_feed_health,
        "round_model_health": round_model_health,
        "round_resolution_health": round_resolution_health,
        "round_training_eligible": training_eligible,
        "training_eligibility_policy": TRAINING_ELIGIBILITY_POLICY,
        **book_coverage,
        "model_run_id": _model_run_id(model_manifest),
        "model_manifest_sha256": model_manifest_sha256,
        "model_sha256": model_manifest.get("model_sha256"),
        "real_historical_corpus_used": model_manifest.get(
            "real_historical_corpus_used", False
        ),
        "fixture_corpus_used": model_manifest.get("fixture_corpus_used", False),
        "synthetic_corpus_used": model_manifest.get("synthetic_corpus_used", False),
        "fixture_model_used": model_manifest.get("fixture_model_used", False),
        "round_finalization_only": bool(
            model_manifest.get("round_finalization_only", False)
        ),
        "model_signal_used": bool(
            model_manifest.get("model_signal_used", bool(predictions))
        ),
        "paper_decision_used": bool(
            model_manifest.get("paper_decision_used", bool(decisions))
        ),
        "paper_audit_only": bool(model_manifest.get("paper_audit_only", False)),
        **safety_fields(),
    }


def _run_summary_after_round(
    *,
    config: PolymarketLivePaperConfig,
    round_summaries: list[dict[str, Any]],
    status: str,
    recommendation: str,
    reason_codes: tuple[str, ...],
) -> dict[str, Any]:
    resolved = [row for row in round_summaries if row["resolved_outcome"] is not None]
    pending = [
        row
        for row in round_summaries
        if row["resolved_outcome"] is None and not row["reason_codes"]
    ]
    failed = [row for row in round_summaries if row["reason_codes"]]
    return {
        "schema_version": "bigan-v8-polymarket-live-run-summary-after-round-v1",
        "run_id": config.run_id,
        "rounds_seen": len(round_summaries),
        "rounds_resolved": len(resolved),
        "rounds_failed_closed": len(failed),
        "rounds_pending_resolution": len(pending),
        "total_prediction_count": sum(row["prediction_count"] for row in round_summaries),
        "total_decision_count": sum(row["decision_count"] for row in round_summaries),
        "total_trade_count": sum(
            count
            for row in round_summaries
            for action, count in row["action_counts"].items()
            if action.startswith(("BUY", "SELL"))
        ),
        "cumulative_trade_pnl": sum(row["realized_trade_pnl"] for row in round_summaries),
        "cumulative_settlement_pnl": sum(row["settlement_pnl"] for row in round_summaries),
        "cumulative_total_pnl": sum(row["total_pnl"] for row in round_summaries),
        "max_drawdown": _summary_max_drawdown(round_summaries),
        "critical_alert_count": len(set(reason_codes)),
        "critical_reason_codes": sorted(set(reason_codes)),
        "last_completed_round_id": None if not resolved else resolved[-1]["round_id"],
        "operator_status": status,
        "operator_recommendation": recommendation,
        **safety_fields(),
    }


def _empty_run_summary(
    *,
    config: PolymarketLivePaperConfig,
    status: str,
    recommendation: str,
    reason_codes: tuple[str, ...],
) -> dict[str, Any]:
    return _run_summary_after_round(
        config=config,
        round_summaries=[],
        status=status,
        recommendation=recommendation,
        reason_codes=reason_codes,
    )


def _training_market_row(market: PolymarketLiveMarket) -> dict[str, Any]:
    return {
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "slug": market.slug,
        "market_family": market.market_family,
        "market_start_ts": market.market_start_ts,
        "market_end_ts": market.market_end_ts,
        "settlement_ts": market.settlement_ts,
        "up_token_id": market.up_token_id,
        "down_token_id": market.down_token_id,
        "reference_price_source": market.reference_price_source,
        "settlement_rule": market.settlement_rule,
        **safety_fields(),
    }


def _training_orderbook_row(row: PolymarketLiveOrderBook) -> dict[str, Any]:
    return {
        "market_id": row.market_id,
        "token_id": row.token_id,
        "outcome": row.outcome,
        "ts": row.ts,
        "available_at_ts": row.received_ts,
        "bid_price": row.bid_price,
        "ask_price": row.ask_price,
        "mid_price": row.mid_price,
        "bid_size": row.bid_size,
        "ask_size": row.ask_size,
        "liquidity_depth": row.liquidity_depth,
        **safety_fields(),
    }


def _training_trade_row(row: PolymarketLiveTrade) -> dict[str, Any]:
    return {
        "market_id": row.market_id,
        "token_id": row.token_id,
        "outcome": row.outcome,
        "ts": row.ts,
        "available_at_ts": row.ts,
        "price": row.price,
        "size": row.size,
        "side": row.side,
        **safety_fields(),
    }


def _training_feature_reference_candle_row(market: PolymarketLiveMarket) -> dict[str, Any]:
    timeframe_ms = 60_000
    close_ts = market.market_start_ts
    open_ts = close_ts - timeframe_ms
    return {
        "ts": open_ts,
        "close_time": close_ts,
        "available_at_ts": close_ts,
        "open_price": market.reference_price_at_start,
        "high_price": market.reference_price_at_start,
        "low_price": market.reference_price_at_start,
        "close_price": market.reference_price_at_start,
        "volume": 0.0,
        "timeframe_ms": timeframe_ms,
        "source": market.reference_price_source,
        "reference_role": "causal_feature_reference",
        **safety_fields(),
    }


def _training_candle_row(candle: BinanceBTCCandle) -> dict[str, Any]:
    return {
        "ts": candle.open_ts,
        "close_time": candle.close_ts,
        "available_at_ts": candle.close_ts,
        "open_price": candle.open_price,
        "high_price": candle.high_price,
        "low_price": candle.low_price,
        "close_price": candle.close_price,
        "volume": 0.0,
        "timeframe_ms": candle.close_ts - candle.open_ts,
        "source": candle.source,
        **safety_fields(),
    }


def _training_resolution_row(
    *,
    market: PolymarketLiveMarket,
    candle: BinanceBTCCandle,
    settlement: dict[str, Any],
) -> dict[str, Any]:
    return {
        "market_id": market.market_id,
        "reference_price_start": market.reference_price_at_start,
        "reference_price_end": candle.close_price,
        "resolution_status": settlement["resolution_status"],
        "resolved_outcome": settlement["resolved_outcome"],
        "payout_up": settlement["payout_up"],
        "payout_down": settlement["payout_down"],
        "raw_resolution_text": (
            f"{market.slug} resolved {settlement['resolved_outcome']} "
            f"from {market.reference_price_at_start} to {candle.close_price}"
        ),
        **safety_fields(),
    }


def _audit_pnl_breakdown(market_id: str, ledger_events: list[dict[str, Any]]) -> dict[str, Any]:
    last_event = ledger_events[-1] if ledger_events else {}
    return {
        "schema_version": POLYMARKET_LIVE_SCHEMA_VERSION,
        "market_id": market_id,
        "realized_trade_pnl": float(last_event.get("realized_trade_pnl", 0.0)),
        "settlement_pnl": float(last_event.get("settlement_pnl", 0.0)),
        "complete_set_pnl": float(last_event.get("complete_set_pnl", 0.0)),
        "fees": sum(float(row.get("fees", 0.0)) for row in ledger_events),
        "slippage": sum(float(row.get("slippage", 0.0)) for row in ledger_events),
        "total_polymarket_pnl": float(last_event.get("total_pnl", 0.0)),
        **safety_fields(),
    }


def _round_summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Round Summary",
            "",
            f"- round_id: {summary['round_id']}",
            f"- resolved_outcome: {summary['resolved_outcome']}",
            f"- prediction_count: {summary['prediction_count']}",
            f"- decision_count: {summary['decision_count']}",
            f"- total_pnl: {summary['total_pnl']}",
            f"- reason_codes: {', '.join(summary['reason_codes'])}",
            f"- model_manifest_sha256: {summary['model_manifest_sha256']}",
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )


def _run_summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Run Summary After Round",
            "",
            f"- run_id: {summary['run_id']}",
            f"- rounds_seen: {summary['rounds_seen']}",
            f"- rounds_resolved: {summary['rounds_resolved']}",
            f"- rounds_failed_closed: {summary['rounds_failed_closed']}",
            f"- rounds_pending_resolution: {summary['rounds_pending_resolution']}",
            f"- cumulative_total_pnl: {summary['cumulative_total_pnl']}",
            f"- last_completed_round_id: {summary['last_completed_round_id']}",
            f"- operator_status: {summary['operator_status']}",
            f"- operator_recommendation: {summary['operator_recommendation']}",
            "- paper_only: true",
            "- capital_at_risk: false",
            "",
        ]
    )


def _round_fail_closed_reason_codes(
    *,
    training_eligible: bool,
    has_settlement: bool,
    has_candle: bool,
    book_coverage: dict[str, Any],
) -> list[str]:
    if training_eligible:
        return []
    reasons: set[str] = set()
    if not has_settlement:
        reasons.add("pending_resolution")
    if not has_candle:
        reasons.add("missing_reference_candle")
    if int(book_coverage["complete_up_down_book_sample_count"]) <= 0:
        reasons.add("missing_complete_up_down_orderbook")
    return sorted(reasons)


def _book_coverage_metrics(orderbooks: list[PolymarketLiveOrderBook]) -> dict[str, Any]:
    by_ts: dict[int, set[str]] = defaultdict(set)
    for row in orderbooks:
        by_ts[row.ts].add(row.outcome)
    complete_timestamps = sorted(ts for ts, outcomes in by_ts.items() if outcomes == {"UP", "DOWN"})
    expected_sample_count = len(by_ts)
    complete_count = len(complete_timestamps)
    incomplete_count = max(0, expected_sample_count - complete_count)
    return {
        "expected_sample_count": expected_sample_count,
        "complete_up_down_book_sample_count": complete_count,
        "incomplete_book_sample_count": incomplete_count,
        "book_coverage_ratio": (
            0.0 if expected_sample_count == 0 else complete_count / expected_sample_count
        ),
        "first_complete_book_ts": complete_timestamps[0] if complete_timestamps else None,
        "last_complete_book_ts": complete_timestamps[-1] if complete_timestamps else None,
    }


def _round_feed_health(
    *,
    book_coverage: dict[str, Any],
    has_candle: bool,
) -> dict[str, Any]:
    reason_codes = []
    if int(book_coverage["complete_up_down_book_sample_count"]) <= 0:
        reason_codes.append("missing_complete_up_down_orderbook")
    if not has_candle:
        reason_codes.append("missing_reference_candle")
    return {
        "book_coverage": book_coverage,
        "has_reference_candle": has_candle,
        "reason_codes": reason_codes,
    }


def _round_model_health(
    *,
    predictions: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    run_reason_codes: tuple[str, ...],
) -> dict[str, Any]:
    model_reason_codes = [
        code
        for code in run_reason_codes
        if "model" in code or "manifest" in code or "fixture" in code or "synthetic" in code
    ]
    return {
        "prediction_count": len(predictions),
        "decision_count": len(decisions),
        "reason_codes": sorted(set(model_reason_codes)),
    }


def _round_resolution_health(
    *,
    settlement: dict[str, Any] | None,
    candle: BinanceBTCCandle | None,
) -> dict[str, Any]:
    reason_codes = []
    if settlement is None:
        reason_codes.append("pending_resolution")
    if candle is None:
        reason_codes.append("missing_reference_candle")
    return {
        "resolution_available": settlement is not None,
        "reference_candle_available": candle is not None,
        "reason_codes": reason_codes,
    }


def _group_objects(rows: tuple[Any, ...]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[row.market_id].append(row)
    return dict(grouped)


def _group_dict_rows(rows: list[dict[str, Any]], *, key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


def _round_id(market: PolymarketLiveMarket) -> str:
    return market.slug or market.market_id


def _model_run_id(model_manifest: dict[str, Any]) -> str | None:
    for key in ("real_corpus_gate_run_id", "run_id", "recorder_run_id", "model_version"):
        value = model_manifest.get(key)
        if value:
            return str(value)
    return None


def _summary_max_drawdown(round_summaries: list[dict[str, Any]]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in round_summaries:
        cumulative += float(row["total_pnl"])
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
    return abs(max_drawdown)


def _status_and_recommendation(
    *,
    config: PolymarketLivePaperConfig,
    critical_reason_codes: tuple[str, ...],
) -> tuple[str, str]:
    if config.stop_requested:
        return "operator_stopped", "stop_paper_run"
    if critical_reason_codes:
        return "blocked_fail_closed", "blocked_fail_closed"
    return "completed", "continue_paper_run"


def _empty_ledgers(
    markets: tuple[PolymarketLiveMarket, ...],
) -> dict[str, PolymarketPositionLedger]:
    return {
        market.market_id: PolymarketPositionLedger(
            market_id=market.market_id,
            condition_id=market.condition_id,
            slug=market.slug,
            up_token_id=market.up_token_id,
            down_token_id=market.down_token_id,
        )
        for market in markets
    }


def _ev_config(config: PolymarketLivePaperConfig, run_dir: Path) -> PolymarketPolicyTrainingConfig:
    return PolymarketPolicyTrainingConfig(
        corpus_dir=run_dir,
        output_dir=run_dir,
        ev_threshold=config.ev_threshold,
        min_confidence=config.min_confidence,
        max_paper_notional=config.max_paper_notional,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        liquidity_impact_rate=config.liquidity_impact_rate,
        paper_only=True,
        capital_at_risk=False,
        polymarket_write_enabled=False,
        wallet_signing_enabled=False,
    )


def _fixture_model() -> PolymarketPolicyModel:
    feature_columns = [
        "down_ask",
        "down_bid",
        "down_liquidity_depth",
        "down_mid",
        "time_to_close_seconds",
        "up_ask",
        "up_bid",
        "up_liquidity_depth",
        "up_mid",
    ]
    feature_columns.extend(f"family_{family}" for family in sorted(BTC_UPDOWN_MARKET_HORIZONS_MS))
    feature_schema_hash = canonical_json_sha256({"feature_columns": feature_columns})
    label_schema_hash = canonical_json_sha256({"target": "resolved_up_probability"})
    training_corpus_hash = canonical_json_sha256({"source": "mock_live_fixture"})
    dataset_hash = canonical_json_sha256({"dataset": "mock_live_fixture"})
    return PolymarketPolicyModel(
        model_version="polymarket_policy_probability_v1",
        feature_columns=tuple(feature_columns),
        global_probability=0.68,
        market_family_probabilities={
            "btc_updown_5m": 0.68,
            "btc_updown_15m": 0.66,
            "btc_updown_1h": 0.64,
        },
        family_feature_offsets={
            "btc_updown_5m": 0.12,
            "btc_updown_15m": 0.10,
            "btc_updown_1h": 0.08,
        },
        feature_schema_hash=feature_schema_hash,
        label_schema_hash=label_schema_hash,
        training_corpus_hash=training_corpus_hash,
        dataset_hash=dataset_hash,
        train_row_count=9,
        primary_policy_target="resolved_up_probability_only",
        outcome_probability_head_enabled=True,
        action_value_head_enabled=False,
        compatibility_probability_fallback_enabled=True,
        action_value_model_family="resolved_up_probability_only",
        fallback_action_value_model_family="market_family_mean_baseline",
        feature_conditioned_action_value_model_enabled=False,
    )


class _RealTimeStreamingSnapshotProcessor:
    """Generate paper signals and PnL checkpoints from live feed snapshots."""

    def __init__(
        self,
        *,
        config: PolymarketLivePaperConfig,
        run_dir: Path,
        streaming_writer: _StreamingObservabilityWriter,
        model: PolymarketPolicyModel,
        model_manifest_sha256: str,
    ) -> None:
        self._config = config
        self._run_dir = run_dir
        self._streaming_writer = streaming_writer
        self._model = model
        self._model_manifest_sha256 = model_manifest_sha256
        self._seen_example_keys: set[tuple[str, int]] = set()
        self._ledgers: dict[str, PolymarketPositionLedger] = {}
        self._all_predictions: list[Any] = []
        self._decisions: list[Any] = []
        self._emitted_decisions_by_key: dict[tuple[str, int], Any] = {}

    def __call__(
        self,
        *,
        market_rows: list[dict[str, Any]],
        orderbook_rows: list[dict[str, Any]],
        trade_rows: list[dict[str, Any]],
        tick_rows: list[dict[str, Any]],
        candle_rows: list[dict[str, Any]],
    ) -> None:
        del trade_rows, tick_rows, candle_rows
        markets = tuple(PolymarketLiveMarket(**row) for row in market_rows)
        orderbooks = tuple(PolymarketLiveOrderBook(**row) for row in orderbook_rows)
        self._ensure_ledgers(markets)
        examples = _policy_examples(markets=markets, orderbooks=orderbooks)
        new_examples = tuple(
            example
            for example in examples
            if (example.market_id, example.decision_ts) not in self._seen_example_keys
        )
        if not new_examples:
            return
        for example in new_examples:
            self._seen_example_keys.add((example.market_id, example.decision_ts))
        predictions = predict_polymarket_policy_examples(self._model, new_examples)
        self._streaming_writer.append_signal_events(
            predictions=predictions,
            model_manifest_sha256=self._model_manifest_sha256,
        )
        self._all_predictions.extend(predictions)
        all_decisions = build_polymarket_ev_decisions(
            predictions=tuple(self._all_predictions),
            config=_ev_config(self._config, self._run_dir),
        )
        self._assert_emitted_decisions_stable(all_decisions)
        new_decisions = tuple(
            decision
            for decision in all_decisions
            if _decision_key(decision) not in self._emitted_decisions_by_key
        )
        if not new_decisions:
            return
        self._streaming_writer.append_execution_events(decisions=new_decisions)
        before_counts = {
            market_id: len(ledger.events) for market_id, ledger in self._ledgers.items()
        }
        _apply_decisions_to_ledgers(
            ledgers=self._ledgers,
            decisions=new_decisions,
            config=self._config,
        )
        new_ledger_events = []
        for market_id, ledger in sorted(self._ledgers.items()):
            previous_count = before_counts.get(market_id, 0)
            new_ledger_events.extend(
                event.to_dict() for event in ledger.events[previous_count:]
            )
        for decision in new_decisions:
            self._emitted_decisions_by_key[_decision_key(decision)] = decision
        self._decisions = list(all_decisions)
        pnl_breakdown = _pnl_breakdown(
            config=self._config,
            markets=markets,
            ledgers=self._ledgers,
            decisions=tuple(self._decisions),
            settlement_events=[],
        )
        self._streaming_writer.append_position_snapshots(
            markets=markets,
            ledger_events=new_ledger_events,
            decisions=new_decisions,
        )
        self._streaming_writer.append_pnl_snapshots(
            markets=markets,
            ledger_events=new_ledger_events,
            decisions=new_decisions,
            pnl_breakdown=pnl_breakdown,
        )

    def assert_replay_equivalent(
        self,
        *,
        markets: tuple[PolymarketLiveMarket, ...],
        predictions: tuple[Any, ...],
        decisions: tuple[Any, ...],
        ledgers: dict[str, PolymarketPositionLedger],
    ) -> None:
        if _prediction_sequence_signature(tuple(self._all_predictions)) != (
            _prediction_sequence_signature(predictions)
        ):
            raise RuntimeError("streaming_replay_mismatch: predictions diverged")
        if _decision_sequence_signature(tuple(self._decisions)) != (
            _decision_sequence_signature(decisions)
        ):
            raise RuntimeError("streaming_replay_mismatch: decisions diverged")
        if _ledger_sequence_signature(self._ledgers) != _ledger_sequence_signature(ledgers):
            raise RuntimeError("streaming_replay_mismatch: ledgers diverged")
        streaming_pnl = _pnl_breakdown(
            config=self._config,
            markets=markets,
            ledgers=self._ledgers,
            decisions=tuple(self._decisions),
            settlement_events=[],
        )
        final_replay_pnl = _pnl_breakdown(
            config=self._config,
            markets=markets,
            ledgers=ledgers,
            decisions=decisions,
            settlement_events=[],
        )
        if _pnl_equivalence_signature(streaming_pnl) != (
            _pnl_equivalence_signature(final_replay_pnl)
        ):
            raise RuntimeError("streaming_replay_mismatch: pnl diverged")

    def _ensure_ledgers(self, markets: tuple[PolymarketLiveMarket, ...]) -> None:
        for market in markets:
            if market.market_id in self._ledgers:
                continue
            self._ledgers[market.market_id] = PolymarketPositionLedger(
                market_id=market.market_id,
                condition_id=market.condition_id,
                slug=market.slug,
                up_token_id=market.up_token_id,
                down_token_id=market.down_token_id,
            )

    def _assert_emitted_decisions_stable(self, decisions: tuple[Any, ...]) -> None:
        for decision in decisions:
            key = _decision_key(decision)
            emitted = self._emitted_decisions_by_key.get(key)
            if emitted is None:
                continue
            if _decision_sequence_signature((emitted,)) != _decision_sequence_signature(
                (decision,)
            ):
                raise RuntimeError("streaming_replay_mismatch: emitted decision changed")


def _decision_key(decision: Any) -> tuple[str, int]:
    return str(decision.market_id), int(decision.decision_ts)


def _prediction_sequence_signature(predictions: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "market_id": str(prediction.market_id),
            "decision_ts": int(prediction.decision_ts),
            "estimated_up_probability": _signature_float(
                prediction.estimated_up_probability
            ),
            "best_policy_action": prediction.best_policy_action,
            "best_action_expected_return": _optional_signature_float(
                prediction.best_action_expected_return
            ),
        }
        for prediction in sorted(predictions, key=lambda row: (row.decision_ts, row.market_id))
    ]


def _decision_sequence_signature(decisions: tuple[Any, ...]) -> list[dict[str, Any]]:
    payloads = []
    for decision in sorted(decisions, key=lambda row: (row.decision_ts, row.market_id)):
        payload = decision.to_dict()
        payloads.append(
            {
                "market_id": payload["market_id"],
                "decision_ts": payload["decision_ts"],
                "action": payload["action"],
                "selected_outcome": payload["selected_outcome"],
                "execution_price": _signature_float(payload["execution_price"]),
                "paper_notional": _signature_float(payload["paper_notional"]),
                "reason_codes": list(payload["reason_codes"]),
                "entry_policy_action": payload.get("entry_policy_action"),
                "intended_exit_policy": payload.get("intended_exit_policy"),
                "planned_exit_before_ts": payload.get("planned_exit_before_ts"),
                "policy_exit_reason": payload.get("policy_exit_reason"),
                "best_policy_action": payload.get("best_policy_action"),
                "best_action_expected_return": _optional_signature_float(
                    payload.get("best_action_expected_return")
                ),
            }
        )
    return payloads


def _ledger_sequence_signature(
    ledgers: dict[str, PolymarketPositionLedger],
) -> list[dict[str, Any]]:
    events = []
    for market_id, ledger in sorted(ledgers.items()):
        for event in ledger.events:
            payload = event.to_dict()
            events.append(
                {
                    "market_id": market_id,
                    "ts": payload["ts"],
                    "action": payload["action"],
                    "outcome": payload["outcome"],
                    "qty": _signature_float(payload["qty"]),
                    "fill_price": _signature_float(payload["fill_price"]),
                    "position_up": _signature_float(payload["position_up"]),
                    "position_down": _signature_float(payload["position_down"]),
                    "realized_trade_pnl": _signature_float(
                        payload["realized_trade_pnl"]
                    ),
                    "unrealized_mark_pnl": _signature_float(
                        payload["unrealized_mark_pnl"]
                    ),
                    "total_pnl": _signature_float(payload["total_pnl"]),
                    "fees": _signature_float(payload["fees"]),
                    "slippage": _signature_float(payload["slippage"]),
                    "reason_codes": list(payload["reason_codes"]),
                }
            )
    return sorted(events, key=lambda row: (row["market_id"], row["ts"], row["action"]))


def _pnl_equivalence_signature(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "market_count",
        "unresolved_market_count",
        "prediction_count",
        "decision_count",
        "trade_count",
        "no_trade_count",
        "realized_trade_pnl",
        "unrealized_mark_pnl",
        "settlement_pnl",
        "complete_set_pnl",
        "fees",
        "slippage",
        "total_polymarket_pnl",
    )
    return {
        key: _signature_float(payload[key]) if isinstance(payload[key], float) else payload[key]
        for key in keys
    }


def _signature_float(value: float) -> float:
    return round(float(value), 12)


def _optional_signature_float(value: Any) -> float | None:
    if value is None:
        return None
    return _signature_float(float(value))


class _StreamingObservabilityWriter:
    """Durable operational checkpoints for long-running paper-only live runs."""

    def __init__(
        self,
        *,
        config: PolymarketLivePaperConfig,
        artifact_paths: dict[str, Path],
    ) -> None:
        self._config = config
        self._paths = artifact_paths
        self._started_monotonic = time.monotonic()
        self._last_status_monotonic = 0.0
        self._last_heartbeat_monotonic = 0.0
        for name in (
            "operator_heartbeat",
            "signal_events",
            "execution_events",
            "position_snapshots",
            "pnl_snapshots",
        ):
            self._paths[name].parent.mkdir(parents=True, exist_ok=True)
            self._paths[name].touch(exist_ok=True)

    def record_operator_start(self) -> None:
        self.write_status(operator_status="running", stage="operator_starting", force=True)
        self.emit_heartbeat(
            stage="operator_starting",
            operator_status="running",
            force=True,
        )

    def record_feed_checkpoint(
        self,
        *,
        stage: str,
        market_count: int,
        latest_market_id: str | None,
        orderbook_count: int,
        trade_count: int,
        tick_count: int,
        candle_count: int,
        force: bool = False,
    ) -> None:
        extra = {
            "rounds_seen": market_count,
            "latest_market_id": latest_market_id,
            "feed_orderbook_count": orderbook_count,
            "feed_trade_count": trade_count,
            "feed_tick_count": tick_count,
            "feed_candle_count": candle_count,
        }
        self.write_status(
            operator_status="running",
            stage=stage,
            count_overrides=extra,
            force=force,
        )
        self.emit_heartbeat(
            stage=stage,
            operator_status="running",
            force=force,
            **extra,
        )

    def append_signal_events(
        self,
        *,
        predictions: tuple[Any, ...],
        model_manifest_sha256: str,
    ) -> None:
        rows = []
        for prediction in predictions:
            payload = prediction.to_dict()
            rows.append(
                {
                    "ts": payload["decision_ts"],
                    "market_id": payload["market_id"],
                    "market_family": payload["market_family"],
                    "estimated_up_probability": payload["estimated_up_probability"],
                    "p_up_auxiliary": payload.get("p_up_auxiliary"),
                    "expected_return_by_action": payload.get("expected_return_by_action", {}),
                    "best_policy_action": payload.get("best_policy_action"),
                    "best_action_expected_return": payload.get(
                        "best_action_expected_return"
                    ),
                    "second_best_action_expected_return": payload.get(
                        "second_best_action_expected_return"
                    ),
                    "best_action_margin": payload.get("best_action_margin"),
                    "policy_confidence": payload.get("policy_confidence"),
                    "action_value_model_family": payload.get("action_value_model_family"),
                    "feature_conditioned_action_value_model_enabled": payload.get(
                        "feature_conditioned_action_value_model_enabled"
                    ),
                    "model_version": payload["model_version"],
                    "model_manifest_sha256": model_manifest_sha256,
                    **safety_fields(),
                }
            )
        self._append_jsonl(self._paths["signal_events"], rows)
        self.write_status(
            operator_status="running",
            stage="signals_generated",
            predictions=predictions,
            force=True,
        )

    def append_execution_events(self, *, decisions: tuple[Any, ...]) -> None:
        rows = []
        for decision in decisions:
            payload = decision.to_dict()
            rows.append(
                {
                    "ts": payload["decision_ts"],
                    "market_id": payload["market_id"],
                    "action": payload["action"],
                    "selected_outcome": payload["selected_outcome"],
                    "execution_price": payload["execution_price"],
                    "used_price_side": payload["used_price_side"],
                    "paper_notional": payload["paper_notional"],
                    "reason_codes": payload["reason_codes"],
                    "entry_policy_action": payload.get("entry_policy_action"),
                    "intended_exit_policy": payload.get("intended_exit_policy"),
                    "planned_exit_before_ts": payload.get("planned_exit_before_ts"),
                    "policy_exit_reason": payload.get("policy_exit_reason"),
                    "action_value_head_used": payload.get("action_value_head_used"),
                    "probability_ev_fallback_used": payload.get(
                        "probability_ev_fallback_used"
                    ),
                    **safety_fields(),
                }
            )
        self._append_jsonl(self._paths["execution_events"], rows)
        self.write_status(
            operator_status="running",
            stage="decisions_generated",
            decisions=decisions,
            force=True,
        )

    def append_position_snapshots(
        self,
        *,
        markets: tuple[PolymarketLiveMarket, ...],
        ledger_events: list[dict[str, Any]],
        decisions: tuple[Any, ...],
    ) -> None:
        decisions_by_key = {
            (decision.market_id, decision.decision_ts): decision.to_dict()
            for decision in decisions
        }
        rows = []
        for event in ledger_events:
            decision = decisions_by_key.get((event["market_id"], event["ts"]), {})
            rows.append(
                {
                    "ts": event["ts"],
                    "market_id": event["market_id"],
                    "position_up": event["position_up"],
                    "position_down": event["position_down"],
                    "entry_policy_action": decision.get("entry_policy_action"),
                    "intended_exit_policy": decision.get("intended_exit_policy"),
                    "planned_exit_before_ts": decision.get("planned_exit_before_ts"),
                    "last_action": event["action"],
                    **safety_fields(),
                }
            )
        if not rows:
            rows.extend(
                {
                    "ts": market.market_start_ts,
                    "market_id": market.market_id,
                    "position_up": 0.0,
                    "position_down": 0.0,
                    "entry_policy_action": None,
                    "intended_exit_policy": None,
                    "planned_exit_before_ts": None,
                    "last_action": "NO_TRADE",
                    **safety_fields(),
                }
                for market in markets
            )
        self._append_jsonl(self._paths["position_snapshots"], rows)

    def append_pnl_snapshots(
        self,
        *,
        markets: tuple[PolymarketLiveMarket, ...],
        ledger_events: list[dict[str, Any]],
        decisions: tuple[Any, ...],
        pnl_breakdown: dict[str, Any],
    ) -> None:
        trade_counts_by_market: Counter[str] = Counter()
        no_trade_counts_by_market: Counter[str] = Counter()
        for decision in decisions:
            if decision.action.startswith(("BUY", "SELL")):
                trade_counts_by_market[decision.market_id] += 1
            elif decision.action == "NO_TRADE":
                no_trade_counts_by_market[decision.market_id] += 1
        rows = []
        for event in ledger_events:
            rows.append(
                {
                    "ts": event["ts"],
                    "market_id": event["market_id"],
                    "open_position_up": event["position_up"],
                    "open_position_down": event["position_down"],
                    "realized_trade_pnl": event["realized_trade_pnl"],
                    "unrealized_mark_pnl": event["unrealized_mark_pnl"],
                    "settlement_pnl": event["settlement_pnl"],
                    "fees": event["fees"],
                    "slippage": event["slippage"],
                    "estimated_total_pnl": event["total_pnl"],
                    "trade_count": trade_counts_by_market[event["market_id"]],
                    "no_trade_count": no_trade_counts_by_market[event["market_id"]],
                    **safety_fields(),
                }
            )
        if not rows:
            rows.extend(
                {
                    "ts": market.market_start_ts,
                    "market_id": market.market_id,
                    "open_position_up": 0.0,
                    "open_position_down": 0.0,
                    "realized_trade_pnl": 0.0,
                    "unrealized_mark_pnl": 0.0,
                    "settlement_pnl": 0.0,
                    "fees": 0.0,
                    "slippage": 0.0,
                    "estimated_total_pnl": 0.0,
                    "trade_count": 0,
                    "no_trade_count": 0,
                    **safety_fields(),
                }
                for market in markets
            )
        self._append_jsonl(self._paths["pnl_snapshots"], rows)
        self.write_status(
            operator_status="running",
            stage="pnl_updated",
            markets=markets,
            decisions=decisions,
            ledger_events=ledger_events,
            pnl_breakdown=pnl_breakdown,
            force=True,
        )

    def write_status(
        self,
        *,
        operator_status: str,
        stage: str,
        markets: tuple[PolymarketLiveMarket, ...] = (),
        predictions: tuple[Any, ...] = (),
        decisions: tuple[Any, ...] = (),
        ledger_events: list[dict[str, Any]] | None = None,
        pnl_breakdown: dict[str, Any] | None = None,
        critical_reason_codes: tuple[str, ...] = (),
        count_overrides: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if (
            not force
            and now - self._last_status_monotonic
            < self._config.status_interval_seconds
        ):
            return
        self._last_status_monotonic = now
        ledger_events = ledger_events or []
        status = self._status_payload(
            operator_status=operator_status,
            stage=stage,
            markets=markets,
            predictions=predictions,
            decisions=decisions,
            ledger_events=ledger_events,
            pnl_breakdown=pnl_breakdown,
            critical_reason_codes=critical_reason_codes,
            count_overrides=count_overrides or {},
        )
        self._atomic_write_json(self._paths["live_status"], status)
        self._atomic_write_text(self._paths["live_status_md"], _live_status_markdown(status))

    def emit_heartbeat(
        self,
        *,
        stage: str,
        operator_status: str,
        force: bool = False,
        critical_reason_codes: tuple[str, ...] = (),
        **extra: Any,
    ) -> None:
        now = time.monotonic()
        if (
            not force
            and now - self._last_heartbeat_monotonic
            < self._config.heartbeat_interval_seconds
        ):
            return
        self._last_heartbeat_monotonic = now
        row = {
            "run_id": self._config.run_id,
            "operator_status": operator_status,
            "stage": stage,
            "heartbeat_at": _now_iso(),
            "elapsed_seconds": self._elapsed_seconds(),
            "critical_reason_codes": list(critical_reason_codes),
            **extra,
            **safety_fields(),
        }
        self._append_jsonl(self._paths["operator_heartbeat"], [row])

    def _status_payload(
        self,
        *,
        operator_status: str,
        stage: str,
        markets: tuple[PolymarketLiveMarket, ...],
        predictions: tuple[Any, ...],
        decisions: tuple[Any, ...],
        ledger_events: list[dict[str, Any]],
        pnl_breakdown: dict[str, Any] | None,
        critical_reason_codes: tuple[str, ...],
        count_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        latest_event_by_market: dict[str, dict[str, Any]] = {}
        for event in ledger_events:
            latest_event_by_market[event["market_id"]] = event
        open_up = sum(float(row.get("position_up", 0.0)) for row in latest_event_by_market.values())
        open_down = sum(
            float(row.get("position_down", 0.0)) for row in latest_event_by_market.values()
        )
        latest_decision_ts = None
        if decisions:
            latest_decision_ts = max(int(decision.decision_ts) for decision in decisions)
        elif predictions:
            latest_decision_ts = max(int(prediction.decision_ts) for prediction in predictions)
        latest_market_id = None
        if markets:
            latest_market_id = sorted(markets, key=lambda item: item.market_start_ts)[-1].market_id
        action_counts = Counter(decision.action for decision in decisions)
        realized = (
            float(pnl_breakdown.get("realized_trade_pnl", 0.0))
            if pnl_breakdown
            else sum(float(row.get("realized_trade_pnl", 0.0)) for row in latest_event_by_market.values())
        )
        unrealized = (
            float(pnl_breakdown.get("unrealized_mark_pnl", 0.0))
            if pnl_breakdown
            else sum(float(row.get("unrealized_mark_pnl", 0.0)) for row in latest_event_by_market.values())
        )
        settlement = (
            float(pnl_breakdown.get("settlement_pnl", 0.0))
            if pnl_breakdown
            else sum(float(row.get("settlement_pnl", 0.0)) for row in latest_event_by_market.values())
        )
        total = (
            float(pnl_breakdown.get("total_polymarket_pnl", 0.0))
            if pnl_breakdown
            else sum(float(row.get("total_pnl", 0.0)) for row in latest_event_by_market.values())
        )
        payload = {
            "run_id": self._config.run_id,
            "operator_status": operator_status,
            "stage": stage,
            "started_at": self._config.started_at,
            "last_update_at": _now_iso(),
            "elapsed_seconds": self._elapsed_seconds(),
            "configured_duration_seconds": self._config.duration_seconds,
            "remaining_seconds": max(
                0,
                self._config.duration_seconds - self._elapsed_seconds(),
            ),
            "market_family": self._config.market_families[0]
            if len(self._config.market_families) == 1
            else "mixed",
            "market_families": list(self._config.market_families),
            "source_collection_mode": "mock_live" if self._config.mock_live else "live_readonly",
            "rounds_seen": len(markets),
            "latest_market_id": latest_market_id,
            "latest_decision_ts": latest_decision_ts,
            "prediction_count": len(predictions),
            "decision_count": len(decisions),
            "trade_count": sum(
                count
                for action, count in action_counts.items()
                if action.startswith(("BUY", "SELL"))
            ),
            "no_trade_count": action_counts.get("NO_TRADE", 0),
            "open_position_up": open_up,
            "open_position_down": open_down,
            "realized_trade_pnl": realized,
            "unrealized_mark_pnl": unrealized,
            "settlement_pnl": settlement,
            "estimated_total_pnl": total,
            "critical_reason_codes": sorted(set(critical_reason_codes)),
            **safety_fields(),
        }
        payload.update(count_overrides)
        return payload

    def _append_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        _json_ready(row),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            if self._config.flush_event_files:
                os.fsync(handle.fileno())

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    @staticmethod
    def _atomic_write_text(path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)

    def _elapsed_seconds(self) -> int:
        return max(0, int(time.monotonic() - self._started_monotonic))


def _artifact_paths(run_dir: Path, *, stream_observability: bool = False) -> dict[str, Path]:
    paths = {
        "live_market_metadata": run_dir / "live_market_metadata.jsonl",
        "live_token_orderbooks": run_dir / "live_token_orderbooks.jsonl",
        "live_token_trades": run_dir / "live_token_trades.jsonl",
        "live_btc_reference_ticks": run_dir / "live_btc_reference_ticks.jsonl",
        "live_btc_reference_candles": run_dir / "live_btc_reference_candles.jsonl",
        "polymarket_model_predictions": run_dir / "polymarket_model_predictions.jsonl",
        "polymarket_ev_decisions": run_dir / "polymarket_ev_decisions.jsonl",
        "polymarket_position_ledger": run_dir / "polymarket_position_ledger.jsonl",
        "polymarket_settlement_events": run_dir / "polymarket_settlement_events.jsonl",
        "polymarket_pnl_breakdown": run_dir / "polymarket_pnl_breakdown.json",
        "polymarket_live_operator_manifest": run_dir / "polymarket_live_operator_manifest.json",
        "paper_observability_report": run_dir / "paper_observability_report.json",
        "paper_operator_summary": run_dir / "paper_operator_summary.md",
        "rounds_index": run_dir / "rounds_index.jsonl",
        "training_raw_index": run_dir / "training_raw_index.jsonl",
        "paper_audit_index": run_dir / "paper_audit_index.jsonl",
        "paper_run_summary_latest": run_dir / "paper_run_summary_latest.json",
        "github_paper_comment_payload": run_dir / "github_paper_comment_payload.json",
        "github_paper_comment_md": run_dir / "github_paper_comment.md",
    }
    if stream_observability:
        paths.update(
            {
                "live_status": run_dir / "live_status.json",
                "live_status_md": run_dir / "live_status.md",
                "operator_heartbeat": run_dir / "operator_heartbeat.jsonl",
                "signal_events": run_dir / "signal_events.jsonl",
                "execution_events": run_dir / "execution_events.jsonl",
                "position_snapshots": run_dir / "position_snapshots.jsonl",
                "pnl_snapshots": run_dir / "pnl_snapshots.jsonl",
            }
        )
    return paths


def _write_missing_feed_artifacts(paths: dict[str, Path]) -> None:
    for name in (
        "live_market_metadata",
        "live_token_orderbooks",
        "live_token_trades",
        "live_btc_reference_ticks",
        "live_btc_reference_candles",
    ):
        if not paths[name].exists():
            _write_jsonl(paths[name], [])


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _latest_market_id_from_rows(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    return str(
        sorted(
            rows,
            key=lambda row: (
                int(row.get("market_start_ts", 0)),
                str(row.get("market_id", "")),
            ),
        )[-1].get("market_id")
    )


def _live_status_markdown(status: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Polymarket Live Paper Status",
            "",
            f"- run_id: {status['run_id']}",
            f"- operator_status: {status['operator_status']}",
            f"- stage: {status['stage']}",
            (
                "- elapsed / remaining: "
                f"{status['elapsed_seconds']} / {status['remaining_seconds']} seconds"
            ),
            f"- latest round / latest market: {status.get('latest_market_id')}",
            f"- prediction_count: {status['prediction_count']}",
            f"- decision_count: {status['decision_count']}",
            f"- trade_count: {status['trade_count']}",
            (
                "- open position: "
                f"UP={status['open_position_up']} DOWN={status['open_position_down']}"
            ),
            f"- realized_trade_pnl: {status['realized_trade_pnl']}",
            f"- unrealized_mark_pnl: {status['unrealized_mark_pnl']}",
            f"- settlement_pnl: {status['settlement_pnl']}",
            f"- estimated_total_pnl: {status['estimated_total_pnl']}",
            (
                "- critical_reason_codes: "
                + ", ".join(status.get("critical_reason_codes", []))
            ),
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )


def _artifact_hashes(paths: dict[str, Path], *, exclude: set[str]) -> dict[str, str]:
    return {
        name: _sha256_file(path)
        for name, path in sorted(paths.items())
        if name not in exclude and path.exists()
    }


def _exception_reason_codes(exc: Exception, *, fallback: str) -> list[str]:
    codes = tuple(getattr(exc, "reason_codes", ()))
    if codes:
        return list(codes)
    text = str(exc)
    if "model_manifest_mismatch" in text:
        return ["model_manifest_mismatch"]
    if (
        "primary_policy_target" in text
        or "action_value_head_enabled" in text
        or "feature_conditioned_action_value_model_enabled" in text
    ):
        return ["probability_only_model_not_allowed"]
    if "settlement_rule is required" in text:
        return ["missing_market_rule"]
    if "streaming_replay_mismatch" in text:
        return ["streaming_replay_mismatch"]
    return [fallback]


def _with_full_safety(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, **safety_fields()}


def _ended_at(config: PolymarketLivePaperConfig) -> str:
    try:
        started = datetime.fromisoformat(config.started_at.replace("Z", "+00:00"))
    except ValueError:
        return config.started_at
    ended = started + timedelta(seconds=0 if config.stop_requested else config.duration_seconds)
    return ended.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _current_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _post_github_comment(*, config: PolymarketLivePaperConfig, body_path: Path) -> None:
    subprocess.run(
        [
            "gh",
            "issue",
            "comment",
            str(config.issue_number),
            "--repo",
            config.repo_full_name,
            "--body-file",
            str(body_path),
        ],
        check=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(_json_ready(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_text(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
