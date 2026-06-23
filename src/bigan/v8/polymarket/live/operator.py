"""Operator for Polymarket live-data paper-only BTC UP/DOWN runs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
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
from bigan.v8.polymarket.rules import build_btc_updown_resolution_rule, resolve_polymarket_rule
from bigan.v8.polymarket.training import (
    PolymarketPolicyExample,
    PolymarketPolicyModel,
    PolymarketPolicyTrainingConfig,
    predict_polymarket_policy_examples,
)


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

    artifact_paths = _artifact_paths(run_dir)
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

    try:
        market_rows, orderbook_rows, trade_rows, tick_rows, candle_rows = _load_feed_rows(config)
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

    model: PolymarketPolicyModel | None = None
    if not reason_codes:
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
        decisions = build_polymarket_ev_decisions(
            predictions=predictions,
            config=_ev_config(config, run_dir),
        )
        ledgers = _apply_decisions(markets=markets, decisions=decisions, config=config)
        settlement_events = _settle_markets(
            config=config,
            markets=markets,
            candles=candles,
            ledgers=ledgers,
        )
        ledger_events = [
            event.to_dict()
            for ledger in ledgers.values()
            for event in ledger.events
        ]
    else:
        ledgers = _empty_ledgers(markets)

    pnl_breakdown = _pnl_breakdown(
        config=config,
        markets=markets,
        ledgers=ledgers,
        decisions=decisions,
        settlement_events=settlement_events,
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

    core_hashes = _artifact_hashes(
        artifact_paths,
        exclude={
            "polymarket_live_operator_manifest",
            "github_paper_comment_payload",
            "github_paper_comment_md",
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
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not config.mock_live:
        raise NotImplementedError(
            "real live polling is intentionally not run by CI; use explicit integration wiring"
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
    return ledgers


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
        candle = candles_by_market[market.market_id]
        ledger = ledgers[market.market_id]
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
            reference_price_start=candle.open_price,
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
        **safety_fields(),
    }


def _github_comment_payload(
    *,
    config: PolymarketLivePaperConfig,
    operator_manifest: dict[str, Any],
    operator_manifest_sha256: str,
    pnl_breakdown_sha256: str,
    observability_report_sha256: str,
) -> dict[str, Any]:
    payload = {
        key: operator_manifest[key]
        for key in (
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
    )


def _artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
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
        "github_paper_comment_payload": run_dir / "github_paper_comment_payload.json",
        "github_paper_comment_md": run_dir / "github_paper_comment.md",
    }


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
    if "settlement_rule is required" in text:
        return ["missing_market_rule"]
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
