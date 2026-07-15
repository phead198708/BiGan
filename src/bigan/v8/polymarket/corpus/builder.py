"""Deterministic local corpus builder for Polymarket BTC UP/DOWN markets."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.corpus.contracts import (
    BTC_UPDOWN_MARKET_HORIZONS_MS,
    OPTIONAL_RAW_CORPUS_FILENAMES,
    POLYMARKET_CORPUS_PHASE,
    POLYMARKET_CORPUS_SCHEMA_VERSION,
    RAW_CORPUS_FILENAMES,
    BinanceBTCCandle,
    CorpusOutcome,
    PolymarketChainlinkPrice,
    PolymarketCorpusBookSnapshot,
    PolymarketCorpusBuildConfig,
    PolymarketCorpusBuildResult,
    PolymarketCorpusFeatureRow,
    PolymarketCorpusMarket,
    PolymarketCorpusResolutionEvent,
    PolymarketCorpusTrade,
    safety_fields,
    stable_hash,
)
from bigan.v8.polymarket.corpus.features import build_polymarket_corpus_feature_rows
from bigan.v8.polymarket.corpus.labels import (
    build_polymarket_corpus_label_rows,
    build_sell_before_close_label_redesign_report,
    sell_before_close_label_redesign_markdown,
)
from bigan.v8.polymarket.corpus.splits import build_polymarket_train_shadow_split
from bigan.v8.polymarket.rules import (
    PolymarketResolutionRule,
    PolymarketResolvedOutcome,
    build_btc_updown_resolution_rule,
    payout_for_resolved_outcome,
    resolve_polymarket_rule,
)


def build_polymarket_btc_corpus(
    config: PolymarketCorpusBuildConfig,
) -> PolymarketCorpusBuildResult:
    """Build deterministic corpus artifacts from local raw JSON/JSONL inputs."""

    input_dir = config.input_dir.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"corpus output_dir already exists: {output_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    raw_paths = {name: input_dir / name for name in RAW_CORPUS_FILENAMES}
    missing = [name for name, path in raw_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing raw corpus files: " + ", ".join(sorted(missing)))
    raw_payloads = {name: _read_jsonl(path) for name, path in raw_paths.items()}
    raw_hashes = {name: _sha256_file(path) for name, path in sorted(raw_paths.items())}
    optional_raw_paths = {
        name: input_dir / name
        for name in OPTIONAL_RAW_CORPUS_FILENAMES
        if (input_dir / name).exists()
    }
    for name, path in sorted(optional_raw_paths.items()):
        optional_rows = _read_jsonl(path)
        if optional_rows:
            raw_payloads[name] = optional_rows
            raw_hashes[name] = _sha256_file(path)

    markets = _normalize_markets(raw_payloads["raw_polymarket_markets.jsonl"], config)
    rules = _build_rules(markets)
    book_snapshots = _normalize_book_snapshots(
        raw_payloads["raw_polymarket_orderbooks.jsonl"],
        markets,
    )
    trades = _normalize_trades(raw_payloads["raw_polymarket_trades.jsonl"], markets)
    candles = _normalize_candles(raw_payloads["raw_binance_btcusdt_klines.jsonl"])
    chainlink_prices = _normalize_chainlink_prices(
        raw_payloads.get("raw_polymarket_chainlink_prices.jsonl", [])
    )
    resolution_events = _normalize_resolutions(
        raw_payloads["raw_polymarket_resolutions.jsonl"],
        markets=markets,
        rules=rules,
    )
    feature_rows = build_polymarket_corpus_feature_rows(
        markets=markets,
        book_snapshots=book_snapshots,
        trades=trades,
        btc_candles=candles,
        chainlink_prices=chainlink_prices,
        config=config,
    )
    label_rows = build_polymarket_corpus_label_rows(
        markets=markets,
        rules=rules,
        book_snapshots=book_snapshots,
        resolution_events={event.market_id: event for event in resolution_events},
        feature_rows=feature_rows,
        config=config,
    )
    sell_before_close_label_redesign_report = (
        build_sell_before_close_label_redesign_report(
            label_rows=label_rows,
            config=config,
        )
    )
    split = build_polymarket_train_shadow_split(label_rows=label_rows, config=config)

    paths = {
        "market_rules": output_dir / "polymarket_market_rules.jsonl",
        "market_metadata": output_dir / "polymarket_market_metadata.jsonl",
        "token_book_snapshots": output_dir / "polymarket_token_book_snapshots.jsonl",
        "token_trades": output_dir / "polymarket_token_trades.jsonl",
        "btc_reference_candles": output_dir / "polymarket_btc_reference_candles.jsonl",
        "resolution_events": output_dir / "polymarket_resolution_events.jsonl",
        "feature_rows": output_dir / "polymarket_feature_rows.jsonl",
        "label_rows": output_dir / "polymarket_label_rows.jsonl",
        "sell_before_close_label_redesign_report": (
            output_dir / "sell_before_close_label_redesign_report.json"
        ),
        "sell_before_close_label_redesign_summary": (
            output_dir / "sell_before_close_label_redesign_report.md"
        ),
        "train_shadow_split": output_dir / "polymarket_train_shadow_split.json",
        "corpus_summary": output_dir / "polymarket_corpus_summary.json",
        "corpus_manifest": output_dir / "polymarket_corpus_manifest.json",
    }
    if chainlink_prices:
        paths["chainlink_prices"] = output_dir / "polymarket_chainlink_prices.jsonl"
        paths["chainlink_decision_time_evidence_manifest"] = (
            output_dir / "polymarket_chainlink_decision_time_evidence_manifest.json"
        )
    _write_jsonl(paths["market_rules"], [rule.to_dict() for rule in rules.values()])
    _write_jsonl(paths["market_metadata"], [market.to_dict() for market in markets])
    _write_jsonl(paths["token_book_snapshots"], [row.to_dict() for row in book_snapshots])
    _write_jsonl(paths["token_trades"], [row.to_dict() for row in trades])
    _write_jsonl(paths["btc_reference_candles"], [row.to_dict() for row in candles])
    chainlink_integration = _empty_chainlink_feature_integration()
    if chainlink_prices:
        _write_jsonl(
            paths["chainlink_prices"],
            [row.to_dict() for row in chainlink_prices],
        )
        chainlink_integration = _chainlink_feature_integration_manifest(
            feature_rows=feature_rows,
            chainlink_prices=chainlink_prices,
            evidence_path=paths["chainlink_prices"],
        )
        _write_json(
            paths["chainlink_decision_time_evidence_manifest"],
            chainlink_integration,
        )
    _write_jsonl(paths["resolution_events"], [row.to_dict() for row in resolution_events])
    _write_jsonl(paths["feature_rows"], [row.to_dict() for row in feature_rows])
    _write_jsonl(paths["label_rows"], [row.to_dict() for row in label_rows])
    _write_json(
        paths["sell_before_close_label_redesign_report"],
        sell_before_close_label_redesign_report,
    )
    paths["sell_before_close_label_redesign_summary"].write_text(
        sell_before_close_label_redesign_markdown(
            sell_before_close_label_redesign_report
        ),
        encoding="utf-8",
    )
    _write_json(paths["train_shadow_split"], split.to_dict())

    normalized_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(paths.items())
        if name not in {"corpus_manifest", "corpus_summary"} and path.exists()
    }
    summary = _corpus_summary(
        config=config,
        markets=markets,
        feature_count=len(feature_rows),
        label_count=len(label_rows),
        split=split.to_dict(),
        raw_hashes=raw_hashes,
        normalized_hashes=normalized_hashes,
        rules=rules,
        resolutions=resolution_events,
        sell_before_close_label_redesign_report=(
            sell_before_close_label_redesign_report
        ),
    )
    _write_json(paths["corpus_summary"], summary)
    normalized_hashes["corpus_summary"] = _sha256_file(paths["corpus_summary"])
    manifest = {
        "schema_version": POLYMARKET_CORPUS_SCHEMA_VERSION,
        "phase": POLYMARKET_CORPUS_PHASE,
        "created_at": config.created_at,
        "market_family_counts": _family_counts(markets),
        "market_count": len(markets),
        "feature_row_count": len(feature_rows),
        "label_row_count": len(label_rows),
        "raw_artifact_hashes": raw_hashes,
        "normalized_artifact_hashes": normalized_hashes,
        "chainlink_decision_time_feature_integration": chainlink_integration,
        "rule_hashes": {market_id: rule.raw_rule_sha256 for market_id, rule in rules.items()},
        "resolution_hashes": {
            event.market_id: event.raw_resolution_sha256 for event in resolution_events
        },
        "sell_before_close_label_schema_version": (
            sell_before_close_label_redesign_report[
                "sell_before_close_label_schema_version"
            ]
        ),
        "sell_before_close_fixed_terminal_bid_only_labels_allowed": (
            sell_before_close_label_redesign_report[
                "fixed_terminal_bid_only_labels_allowed"
            ]
        ),
        "sell_before_close_label_redesign_report_path": (
            "sell_before_close_label_redesign_report.json"
        ),
        "sell_before_close_label_redesign_summary_path": (
            "sell_before_close_label_redesign_report.md"
        ),
        "sell_before_close_label_redesign_report_id": (
            sell_before_close_label_redesign_report[
                "sell_before_close_label_redesign_report_id"
            ]
        ),
        "sell_before_close_label_gate_passed": (
            sell_before_close_label_redesign_report["label_gate_passed"]
        ),
        "sell_before_close_execution_class_counts": (
            sell_before_close_label_redesign_report[
                "sell_before_close_execution_class_counts"
            ]
        ),
        "sample_config": config.to_manifest_dict(),
        **safety_fields(),
    }
    _write_json(paths["corpus_manifest"], manifest)
    artifact_hashes = {
        name: _sha256_file(path) for name, path in sorted(paths.items()) if path.exists()
    }
    return PolymarketCorpusBuildResult(
        output_dir=output_dir,
        artifact_paths=paths,
        artifact_hashes=artifact_hashes,
        manifest=manifest,
        summary=summary,
    )


def write_deterministic_polymarket_corpus_fixtures(
    input_dir: Path | str,
) -> dict[str, Path]:
    """Write small deterministic local raw fixtures for tests and examples."""

    root = Path(input_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_ts = 1_780_100_000_000
    market_specs = (
        ("btc5m-up", "btc_updown_5m", base_ts, "gt", "normal", 65_000.0, 65_030.0),
        (
            "btc15m-gte-tie",
            "btc_updown_15m",
            base_ts + 1_000_000,
            "gte",
            "normal",
            65_100.0,
            65_100.0,
        ),
        (
            "btc1h-unknown",
            "btc_updown_1h",
            base_ts + 3_000_000,
            "gt_unknown",
            "unknown_50_50",
            65_200.0,
            65_200.0,
        ),
    )
    markets: list[dict[str, Any]] = []
    orderbooks: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []
    for index, (market_id, family, start_ts, rule_kind, status, open_px, close_px) in enumerate(
        market_specs
    ):
        horizon_ms = BTC_UPDOWN_MARKET_HORIZONS_MS[family]
        end_ts = start_ts + horizon_ms
        slug = f"{market_id}-fixture"
        up_token = f"{market_id}-up-token"
        down_token = f"{market_id}-down-token"
        if rule_kind == "gte":
            rule_text = "UP wins if close is greater than or equal to open; otherwise DOWN wins."
        elif rule_kind == "gt_unknown":
            rule_text = "UP wins if close is greater than open; if unresolved, resolves 50-50."
        else:
            rule_text = "UP wins if close is greater than open; otherwise DOWN wins."
        markets.append(
            {
                "market_id": market_id,
                "condition_id": f"0xcondition{index:02d}",
                "slug": slug,
                "market_family": family,
                "market_start_ts": start_ts,
                "market_end_ts": end_ts,
                "settlement_ts": end_ts,
                "up_token_id": up_token,
                "down_token_id": down_token,
                "reference_price_source": "binance_btcusdt",
                "settlement_rule": rule_text,
                **safety_fields(),
            }
        )
        sample_step = {"btc_updown_5m": 60_000, "btc_updown_15m": 300_000, "btc_updown_1h": 900_000}[family]
        sample_count = max(2, horizon_ms // sample_step)
        for sample_index in range(sample_count):
            ts = start_ts + sample_index * sample_step
            up_mid = min(0.92, 0.48 + 0.03 * sample_index + 0.02 * index)
            if status == "unknown_50_50":
                up_mid = 0.50 + 0.005 * ((sample_index % 2) * 2 - 1)
            down_mid = max(0.08, 1.0 - up_mid)
            for outcome, token_id, mid in (("UP", up_token, up_mid), ("DOWN", down_token, down_mid)):
                bid = max(0.01, mid - 0.01)
                ask = min(0.99, mid + 0.01)
                orderbooks.append(
                    {
                        "market_id": market_id,
                        "token_id": token_id,
                        "outcome": outcome,
                        "ts": ts,
                        "available_at_ts": ts,
                        "bid_price": bid,
                        "ask_price": ask,
                        "mid_price": mid,
                        "bid_size": 1000.0 + sample_index * 10.0,
                        "ask_size": 950.0 + sample_index * 8.0,
                        "liquidity_depth": 1950.0 + sample_index * 18.0,
                        **safety_fields(),
                    }
                )
                trades.append(
                    {
                        "market_id": market_id,
                        "token_id": token_id,
                        "outcome": outcome,
                        "ts": ts,
                        "available_at_ts": ts,
                        "price": mid,
                        "size": 10.0 + sample_index,
                        "side": "buy" if outcome == "UP" else "sell",
                        **safety_fields(),
                    }
                )
        resolutions.append(
            {
                "market_id": market_id,
                "reference_price_start": open_px,
                "reference_price_end": close_px,
                "resolution_status": status,
                "raw_resolution_text": f"{market_id} resolved as {status}",
                **safety_fields(),
            }
        )
    candle_start = base_ts - 900_000
    candle_end = base_ts + 3_000_000 + BTC_UPDOWN_MARKET_HORIZONS_MS["btc_updown_1h"]
    candles = []
    sequence = 0
    ts = candle_start
    while ts <= candle_end:
        close = 65_000.0 + sequence * 3.0 + (sequence % 5)
        timeframe_ms = 60_000
        candles.append(
            {
                "ts": ts,
                "close_time": ts + timeframe_ms,
                "available_at_ts": ts + timeframe_ms,
                "open_price": close - 1.0,
                "high_price": close + 2.0,
                "low_price": close - 2.0,
                "close_price": close,
                "volume": 100.0 + sequence,
                "timeframe_ms": timeframe_ms,
                "source": "binance_btcusdt",
            }
        )
        sequence += 1
        ts += 60_000
    payloads = {
        "raw_polymarket_markets.jsonl": markets,
        "raw_polymarket_orderbooks.jsonl": orderbooks,
        "raw_polymarket_trades.jsonl": trades,
        "raw_binance_btcusdt_klines.jsonl": candles,
        "raw_polymarket_resolutions.jsonl": resolutions,
    }
    paths = {}
    for filename, rows in payloads.items():
        path = root / filename
        _write_jsonl(path, rows)
        paths[filename] = path
    return paths


def _normalize_markets(
    rows: list[dict[str, Any]],
    config: PolymarketCorpusBuildConfig,
) -> tuple[PolymarketCorpusMarket, ...]:
    markets = []
    for row in rows:
        family = str(row["market_family"])
        if family not in config.market_families:
            continue
        markets.append(
            PolymarketCorpusMarket(
                market_id=str(row["market_id"]),
                condition_id=str(row["condition_id"]),
                slug=str(row["slug"]),
                market_family=family,  # type: ignore[arg-type]
                horizon_ms=int(row.get("horizon_ms") or BTC_UPDOWN_MARKET_HORIZONS_MS[family]),
                market_start_ts=int(row["market_start_ts"]),
                market_end_ts=int(row["market_end_ts"]),
                settlement_ts=int(row.get("settlement_ts") or row["market_end_ts"]),
                up_token_id=str(row["up_token_id"]),
                down_token_id=str(row["down_token_id"]),
                reference_price_source=str(row.get("reference_price_source") or "binance_btcusdt"),
                settlement_rule=str(row["settlement_rule"]),
                raw_market_sha256=stable_hash(row),
                reference_price_start=_optional_positive_float(
                    row.get("reference_price_start")
                    if row.get("reference_price_start") is not None
                    else row.get("reference_price_at_start")
                ),
                paper_only=row.get("paper_only", True) is True,
                capital_at_risk=row.get("capital_at_risk", False) is True,
                broker_exchange_write_enabled=row.get("broker_exchange_write_enabled", False) is True,
                live_exchange_write_enabled=row.get("live_exchange_write_enabled", False) is True,
                polymarket_write_enabled=row.get("polymarket_write_enabled", False) is True,
                wallet_signing_enabled=row.get("wallet_signing_enabled", False) is True,
            )
        )
    if not markets:
        raise ValueError("no supported Polymarket corpus markets")
    return tuple(sorted(markets, key=lambda item: (item.market_start_ts, item.market_id)))


def _build_rules(
    markets: tuple[PolymarketCorpusMarket, ...],
) -> dict[str, PolymarketResolutionRule]:
    return {
        market.market_id: build_btc_updown_resolution_rule(
            market_id=market.market_id,
            condition_id=market.condition_id,
            slug=market.slug,
            market_family=market.market_family,
            resolution_source=market.reference_price_source,
            candle_open_ts=market.market_start_ts,
            candle_close_ts=market.market_end_ts,
            raw_rule_text=market.settlement_rule,
        )
        for market in markets
    }


def _normalize_book_snapshots(
    rows: list[dict[str, Any]],
    markets: tuple[PolymarketCorpusMarket, ...],
) -> tuple[PolymarketCorpusBookSnapshot, ...]:
    market_by_id = {market.market_id: market for market in markets}
    snapshots = []
    for row in rows:
        market = market_by_id.get(str(row["market_id"]))
        if market is None:
            continue
        outcome = _outcome_for_row(market=market, row=row)
        token_id = market.token_id_for_outcome(outcome)
        bid = float(row.get("bid_price") or row.get("bid"))
        ask = float(row.get("ask_price") or row.get("ask"))
        snapshots.append(
            PolymarketCorpusBookSnapshot(
                market_id=market.market_id,
                token_id=token_id,
                outcome=outcome,
                ts=int(row["ts"]),
                available_at_ts=int(row.get("available_at_ts") or row["ts"]),
                bid_price=bid,
                ask_price=ask,
                mid_price=float(row.get("mid_price") or (bid + ask) / 2.0),
                bid_size=float(row.get("bid_size") or 0.0),
                ask_size=float(row.get("ask_size") or 0.0),
                liquidity_depth=float(
                    row.get("liquidity_depth")
                    or float(row.get("bid_size") or 0.0)
                    + float(row.get("ask_size") or 0.0)
                ),
                paper_only=row.get("paper_only", True) is True,
                capital_at_risk=row.get("capital_at_risk", False) is True,
                broker_exchange_write_enabled=row.get("broker_exchange_write_enabled", False) is True,
                live_exchange_write_enabled=row.get("live_exchange_write_enabled", False) is True,
                polymarket_write_enabled=row.get("polymarket_write_enabled", False) is True,
                wallet_signing_enabled=row.get("wallet_signing_enabled", False) is True,
            )
        )
    return tuple(sorted(snapshots, key=lambda item: (item.market_id, item.ts, item.outcome)))


def _normalize_trades(
    rows: list[dict[str, Any]],
    markets: tuple[PolymarketCorpusMarket, ...],
) -> tuple[PolymarketCorpusTrade, ...]:
    market_by_id = {market.market_id: market for market in markets}
    trades = []
    for row in rows:
        market = market_by_id.get(str(row["market_id"]))
        if market is None:
            continue
        outcome = _outcome_for_row(market=market, row=row)
        token_id = market.token_id_for_outcome(outcome)
        trades.append(
            PolymarketCorpusTrade(
                market_id=market.market_id,
                token_id=token_id,
                outcome=outcome,
                ts=int(row["ts"]),
                available_at_ts=int(row.get("available_at_ts") or row["ts"]),
                price=float(row["price"]),
                size=float(row.get("size") or 0.0),
                side=str(row.get("side") or "unknown"),
                paper_only=row.get("paper_only", True) is True,
                capital_at_risk=row.get("capital_at_risk", False) is True,
                broker_exchange_write_enabled=row.get("broker_exchange_write_enabled", False) is True,
                live_exchange_write_enabled=row.get("live_exchange_write_enabled", False) is True,
                polymarket_write_enabled=row.get("polymarket_write_enabled", False) is True,
                wallet_signing_enabled=row.get("wallet_signing_enabled", False) is True,
            )
        )
    return tuple(sorted(trades, key=lambda item: (item.market_id, item.ts, item.outcome)))


def _normalize_candles(rows: list[dict[str, Any]]) -> tuple[BinanceBTCCandle, ...]:
    candles = []
    for row in rows:
        ts = int(row["ts"])
        timeframe_ms = int(row.get("timeframe_ms") or 60_000)
        candles.append(
            BinanceBTCCandle(
                ts=ts,
                available_at_ts=_candle_available_at_ts(
                    row=row,
                    ts=ts,
                    timeframe_ms=timeframe_ms,
                ),
                open_price=float(row.get("open_price") or row["open"]),
                high_price=float(row.get("high_price") or row["high"]),
                low_price=float(row.get("low_price") or row["low"]),
                close_price=float(row.get("close_price") or row["close"]),
                volume=float(row.get("volume") or 0.0),
                timeframe_ms=timeframe_ms,
                source=str(row.get("source") or "binance_btcusdt"),
            )
        )
    return tuple(sorted(candles, key=lambda item: item.ts))


def _normalize_chainlink_prices(
    rows: list[dict[str, Any]],
) -> tuple[PolymarketChainlinkPrice, ...]:
    prices = []
    for row in rows:
        if row.get("timestamp_causality_valid") is False:
            raise ValueError("Chainlink row timestamp causality is invalid")
        prices.append(
            PolymarketChainlinkPrice(
                source_ts=int(row["source_ts"]),
                available_at_ts=int(row["available_at_ts"]),
                price=float(row["price"]),
                source_type=str(row.get("source_type") or ""),
                symbol=str(row.get("symbol") or ""),
                read_only=row.get("read_only", True) is True,
                paper_only=row.get("paper_only", True) is True,
                capital_at_risk=row.get("capital_at_risk", False) is True,
                broker_exchange_write_enabled=(
                    row.get("broker_exchange_write_enabled", False) is True
                ),
                live_exchange_write_enabled=(
                    row.get("live_exchange_write_enabled", False) is True
                ),
                polymarket_write_enabled=(
                    row.get("polymarket_write_enabled", False) is True
                ),
                wallet_signing_enabled=(
                    row.get("wallet_signing_enabled", False) is True
                ),
            )
        )
    return tuple(
        sorted(prices, key=lambda item: (item.source_ts, item.available_at_ts, item.price))
    )


def _empty_chainlink_feature_integration() -> dict[str, Any]:
    return {
        "schema_version": "bigan-v8-polymarket-chainlink-decision-time-evidence-v2",
        "source_type": "polymarket_rtds_chainlink",
        "decision_time_only": True,
        "row_count": 0,
        "feature_row_count": 0,
        "integrated_feature_row_count": 0,
        "missing_or_invalid_feature_row_count": 0,
        "feature_reference_source_distribution": {},
        "feature_integration_reason_distribution": {},
        "timestamp_causality_violation_count": 0,
        "feature_builder_integration_passed": False,
        "feature_builder_integration_required": True,
        "read_only": True,
        **safety_fields(),
    }


def _chainlink_feature_integration_manifest(
    *,
    feature_rows: tuple[PolymarketCorpusFeatureRow, ...],
    chainlink_prices: tuple[PolymarketChainlinkPrice, ...],
    evidence_path: Path,
) -> dict[str, Any]:
    source_distribution: Counter[str] = Counter()
    reason_distribution: Counter[str] = Counter()
    integrated_count = 0
    for row in feature_rows:
        provenance = row.feature_provenance.get(
            "reference_price_to_beat_distance_at_decision", {}
        )
        source = str(provenance.get("reference_price_to_beat_source") or "missing")
        source_distribution[source] += 1
        reasons = []
        if source != "polymarket_rtds_chainlink_market_start":
            reasons.append("reference_distance_not_sourced_from_chainlink")
        source_fields = str(provenance.get("source_fields_used") or "")
        if (
            "raw_polymarket_chainlink_prices.price_at_or_before_market_start"
            not in source_fields
            or "raw_polymarket_chainlink_prices.price_at_or_before_decision"
            not in source_fields
        ):
            reasons.append("chainlink_reference_source_fields_incomplete")
        if provenance.get("provenance_valid") is not True:
            reasons.append("chainlink_reference_provenance_invalid")
        if int(provenance.get("max_input_ts") or 0) > row.decision_ts:
            reasons.append("chainlink_reference_max_input_after_decision")
        if int(provenance.get("available_at_ts") or 0) > row.decision_ts:
            reasons.append("chainlink_reference_available_after_decision")
        if row.features.get("reference_price_to_beat_distance_at_decision") is None:
            reasons.append("chainlink_reference_distance_missing")
        if reasons:
            reason_distribution.update(set(reasons))
        else:
            integrated_count += 1
    feature_row_count = len(feature_rows)
    integration_passed = feature_row_count > 0 and integrated_count == feature_row_count
    return {
        "schema_version": "bigan-v8-polymarket-chainlink-decision-time-evidence-v2",
        "source_type": "polymarket_rtds_chainlink",
        "decision_time_only": True,
        "row_count": len(chainlink_prices),
        "evidence_path": evidence_path.name,
        "evidence_sha256": _sha256_file(evidence_path),
        "feature_row_count": feature_row_count,
        "integrated_feature_row_count": integrated_count,
        "missing_or_invalid_feature_row_count": feature_row_count - integrated_count,
        "feature_reference_source_distribution": dict(sorted(source_distribution.items())),
        "feature_integration_reason_distribution": dict(
            sorted(reason_distribution.items())
        ),
        "timestamp_causality_violation_count": sum(
            1
            for row in chainlink_prices
            if row.source_ts > row.available_at_ts
        ),
        "feature_builder_integration_passed": integration_passed,
        "feature_builder_integration_required": not integration_passed,
        "read_only": True,
        **safety_fields(),
    }


def _normalize_resolutions(
    rows: list[dict[str, Any]],
    *,
    markets: tuple[PolymarketCorpusMarket, ...],
    rules: dict[str, PolymarketResolutionRule],
) -> tuple[PolymarketCorpusResolutionEvent, ...]:
    market_by_id = {market.market_id: market for market in markets}
    events = []
    for row in rows:
        market = market_by_id.get(str(row["market_id"]))
        if market is None:
            continue
        rule = rules[market.market_id]
        start = _optional_positive_float(row.get("reference_price_start"))
        end = _optional_positive_float(row.get("reference_price_end"))
        status = str(row.get("resolution_status") or "normal")
        if (start is None) != (end is None):
            raise ValueError("reference prices must both be present or both be null")
        if start is not None and end is not None:
            rule = _rule_for_resolution_status(rule, status)
            resolved = resolve_polymarket_rule(
                rule,
                reference_price_start=start,
                reference_price_end=end,
                resolution_status=status,  # type: ignore[arg-type]
            )
            reference_price_start = resolved.reference_price_start
            reference_price_end = resolved.reference_price_end
            resolved_outcome = resolved.resolved_outcome
            payout_up = resolved.payout_up
            payout_down = resolved.payout_down
            resolution_status = resolved.resolution_status
        else:
            resolved_outcome = _resolved_outcome_from_row(row)
            payout_up, payout_down = _resolution_payouts_from_row(
                row=row,
                resolved_outcome=resolved_outcome,
            )
            reference_price_start = None
            reference_price_end = None
            resolution_status = status
        events.append(
            PolymarketCorpusResolutionEvent(
                market_id=market.market_id,
                condition_id=market.condition_id,
                slug=market.slug,
                market_family=market.market_family,
                reference_price_start=reference_price_start,
                reference_price_end=reference_price_end,
                resolution_status=resolution_status,  # type: ignore[arg-type]
                resolved_outcome=resolved_outcome,
                payout_up=payout_up,
                payout_down=payout_down,
                resolution_rule_sha256=rule.raw_rule_sha256,
                raw_resolution_sha256=stable_hash(
                    {
                        **row,
                        "resolved_outcome": resolved_outcome,
                        "payout_up": payout_up,
                        "payout_down": payout_down,
                    }
                ),
                paper_only=row.get("paper_only", True) is True,
                capital_at_risk=row.get("capital_at_risk", False) is True,
                broker_exchange_write_enabled=row.get("broker_exchange_write_enabled", False) is True,
                live_exchange_write_enabled=row.get("live_exchange_write_enabled", False) is True,
                polymarket_write_enabled=row.get("polymarket_write_enabled", False) is True,
                wallet_signing_enabled=row.get("wallet_signing_enabled", False) is True,
            )
        )
    if len(events) != len(markets):
        raise ValueError("every market must have one resolution event")
    return tuple(sorted(events, key=lambda item: (item.market_id, item.resolved_outcome)))


def _rule_for_resolution_status(
    rule: PolymarketResolutionRule,
    status: str,
) -> PolymarketResolutionRule:
    if status != "unknown_50_50" or rule.unknown_50_50_enabled:
        return rule
    return build_btc_updown_resolution_rule(
        market_id=rule.market_id,
        condition_id=rule.condition_id,
        slug=rule.slug,
        market_family=rule.market_family,
        resolution_source=rule.resolution_source,
        candle_open_ts=rule.candle_open_ts,
        candle_close_ts=rule.candle_close_ts,
        raw_rule_text=rule.raw_rule_text,
        comparator=rule.comparator,
        tie_breaker=rule.tie_breaker,
        unknown_50_50_enabled=True,
    )


def _resolved_outcome_from_row(row: dict[str, Any]) -> PolymarketResolvedOutcome:
    resolved_outcome = str(row.get("resolved_outcome") or "").upper()
    if resolved_outcome == "UNKNOWN":
        resolved_outcome = "UNKNOWN_50_50"
    if resolved_outcome not in {"UP", "DOWN", "UNKNOWN_50_50"}:
        raise ValueError("payout-only resolution requires resolved_outcome")
    return resolved_outcome  # type: ignore[return-value]


def _resolution_payouts_from_row(
    *,
    row: dict[str, Any],
    resolved_outcome: PolymarketResolvedOutcome,
) -> tuple[float, float]:
    expected_up, expected_down = payout_for_resolved_outcome(resolved_outcome)
    payout_up = _optional_float(row.get("payout_up"))
    payout_down = _optional_float(row.get("payout_down"))
    if payout_up is None and payout_down is None:
        return expected_up, expected_down
    if payout_up != expected_up or payout_down != expected_down:
        raise ValueError("resolution payout vector does not match resolved_outcome")
    return payout_up, payout_down


def _optional_positive_float(value: Any) -> float | None:
    numeric = _optional_float(value)
    if numeric is None:
        return None
    if numeric <= 0.0 or not math.isfinite(numeric):
        raise ValueError("reference prices must be positive")
    return numeric


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _corpus_summary(
    *,
    config: PolymarketCorpusBuildConfig,
    markets: tuple[PolymarketCorpusMarket, ...],
    feature_count: int,
    label_count: int,
    split: dict[str, Any],
    raw_hashes: dict[str, str],
    normalized_hashes: dict[str, str],
    rules: dict[str, PolymarketResolutionRule],
    resolutions: tuple[PolymarketCorpusResolutionEvent, ...],
    sell_before_close_label_redesign_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": POLYMARKET_CORPUS_SCHEMA_VERSION,
        "phase": POLYMARKET_CORPUS_PHASE,
        "created_at": config.created_at,
        "market_family_counts": _family_counts(markets),
        "market_count": len(markets),
        "feature_row_count": feature_count,
        "label_row_count": label_count,
        "raw_artifact_hashes": raw_hashes,
        "normalized_artifact_hashes": normalized_hashes,
        "rule_hashes": {market_id: rule.raw_rule_sha256 for market_id, rule in rules.items()},
        "resolution_hashes": {
            event.market_id: event.raw_resolution_sha256 for event in resolutions
        },
        "sell_before_close_label_schema_version": (
            sell_before_close_label_redesign_report[
                "sell_before_close_label_schema_version"
            ]
        ),
        "sell_before_close_label_gate_passed": (
            sell_before_close_label_redesign_report["label_gate_passed"]
        ),
        "sell_before_close_execution_class_counts": (
            sell_before_close_label_redesign_report[
                "sell_before_close_execution_class_counts"
            ]
        ),
        "sell_before_close_label_gate_reason_codes": (
            sell_before_close_label_redesign_report["label_gate_reason_codes"]
        ),
        "sample_config": config.to_manifest_dict(),
        "split": split,
        **safety_fields(),
    }


def _family_counts(markets: tuple[PolymarketCorpusMarket, ...]) -> dict[str, int]:
    counts = dict.fromkeys(BTC_UPDOWN_MARKET_HORIZONS_MS, 0)
    for market in markets:
        counts[market.market_family] += 1
    return {family: count for family, count in counts.items() if count > 0}


def _outcome_for_row(
    *,
    market: PolymarketCorpusMarket,
    row: dict[str, Any],
) -> CorpusOutcome:
    token_id = str(row.get("token_id") or "").strip()
    outcome = str(row.get("outcome") or "").upper().strip()
    if outcome and outcome not in {"UP", "DOWN"}:
        raise ValueError(f"unsupported outcome for market {market.market_id}: {outcome}")
    if token_id:
        if token_id == market.up_token_id:
            token_outcome = "UP"
        elif token_id == market.down_token_id:
            token_outcome = "DOWN"
        else:
            raise ValueError(f"unknown token_id for market {market.market_id}: {token_id}")
        if outcome and outcome != token_outcome:
            raise ValueError(
                f"token_id/outcome mismatch for market {market.market_id}: "
                f"{token_id} implies {token_outcome}, row says {outcome}"
            )
        return token_outcome
    if outcome in {"UP", "DOWN"}:
        return outcome
    raise ValueError("cannot infer UP/DOWN outcome")


def _candle_available_at_ts(
    *,
    row: dict[str, Any],
    ts: int,
    timeframe_ms: int,
) -> int:
    for field_name in ("available_at_ts", "close_time", "close_ts", "candle_close_ts"):
        value = row.get(field_name)
        if value is not None:
            return int(value)
    return ts + timeframe_ms


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                _json_ready(row),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


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
