#!/usr/bin/env python3
"""Build a 5s/10s event-driven xgboost-v7 convergence corpus from raw queues.

The corpus keeps the existing ``features_15m_v1`` feature schema, but aggregates
feature snapshots on a configurable sub-minute bucket. Settlement labels are
kept for artifact compatibility; the v7 objective of record is the side-specific
future executable exit value from the raw UP/DOWN quote paths.
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse, request

import pyarrow as pa
import pyarrow.parquet as pq

from bigan.features.aggregation import aggregate_features_15m_v1
from bigan.labels.v6 import VolatilityLabelConfig, two_sided_volatility_fields
from bigan.modeling.dataset import _feature_columns
from bigan.monitoring.market_quality import (
    is_degenerate_quote,
    round_end_ts_from_canonical_symbol,
    round_start_ts_from_canonical_symbol,
)

DATASET_VERSION = "xgboost-v7-event-driven-convergence-v1.0.0"
LABEL_VERSION = "xgboost-v7-event-driven-convergence-v1"
DEFAULT_GAMMA_API_BASE = "https://gamma-api.polymarket.com"
BTC15M_PREFIX = "BTC-15M:"
SIDES = ("UP", "DOWN")
HIT_5C_DELTA = 0.05
HIT_10C_DELTA = 0.10
LOSS_10C_DELTA = -0.10


@dataclass(frozen=True, slots=True)
class BuildStats:
    raw_queue_paths: list[str]
    bucket_seconds: float
    raw_throttle_ms: int
    raw_rows_read: int
    top_of_book_rows: int
    orderbook_rows: int
    trade_rows: int
    feature_rows_generated: int
    rows_written: int
    rows_skipped_quality: int
    rows_skipped_unresolved_outcome: int
    rows_skipped_missing_quote: int
    round_count: int
    outcome_counts: dict[str, int]
    splits: dict[str, dict[str, Any]]


def build_event_driven_dataset(
    raw_queue_paths: Sequence[Path | str],
    output_dir: Path | str,
    *,
    bucket_seconds: float = 10.0,
    max_raw_records: int | None = None,
    max_rounds: int | None = None,
    raw_throttle_ms: int = 1000,
    min_completeness_score: float = 0.8,
    resolve_gamma_outcomes: bool = False,
    gamma_api_base: str = DEFAULT_GAMMA_API_BASE,
    request_timeout_seconds: float = 8.0,
    outcome_cache_path: Path | str | None = None,
    infer_outcomes_from_final_quotes: bool = True,
    require_outcome: bool = True,
    volatility_label_config: VolatilityLabelConfig | None = None,
) -> BuildStats:
    """Build train/val/test parquet splits from one or more low-latency queues."""

    if bucket_seconds <= 0.0:
        raise ValueError("bucket_seconds must be positive")
    if max_raw_records is not None and max_raw_records <= 0:
        raise ValueError("max_raw_records must be positive when set")
    if max_rounds is not None and max_rounds <= 0:
        raise ValueError("max_rounds must be positive when set")
    if raw_throttle_ms <= 0:
        raise ValueError("raw_throttle_ms must be positive")
    if not 0.0 <= min_completeness_score <= 1.0:
        raise ValueError("min_completeness_score must be in [0, 1]")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cache_path = (
        target / "gamma-outcomes-cache.json"
        if outcome_cache_path is None
        else Path(outcome_cache_path)
    )
    outcome_cache = _read_outcome_cache(cache_path)
    raw = _read_raw_queues(
        [Path(path) for path in raw_queue_paths],
        max_raw_records=max_raw_records,
        max_rounds=max_rounds,
        raw_throttle_ms=raw_throttle_ms,
    )
    inferred_outcomes = (
        _infer_outcomes_from_quotes(raw.quotes_by_round_side)
        if infer_outcomes_from_final_quotes
        else {}
    )
    outcomes = dict(inferred_outcomes)
    outcomes.update(dict(outcome_cache.items()))
    if resolve_gamma_outcomes:
        missing = sorted(set(raw.quotes_by_round_side) - set(outcomes))
        outcomes.update(
            _resolve_gamma_outcomes(
                missing,
                gamma_api_base=gamma_api_base,
                timeout_seconds=request_timeout_seconds,
            )
        )
        _write_outcome_cache(cache_path, outcomes)

    feature_rows = aggregate_features_15m_v1(
        top_of_book_rows=raw.top_of_book_rows,
        orderbook_rows=raw.orderbook_rows,
        trade_rows=raw.trade_rows,
        ingest_ts=raw.max_ts or 0,
        bucket_ms=int(bucket_seconds * 1000),
    )
    labeled_rows, label_stats = _attach_labels(
        feature_rows,
        quotes_by_round_side=raw.quotes_by_round_side,
        outcomes=outcomes,
        min_completeness_score=min_completeness_score,
        require_outcome=require_outcome,
        volatility_label_config=volatility_label_config
        or VolatilityLabelConfig(
            min_exit_gain=0.02,
            buy_slippage=0.02,
            sell_slippage=0.02,
            max_entry_wait_ms=max(1, int(bucket_seconds * 1000)),
            min_exit_seconds_before_expiry=300.0,
            min_entry_price=0.0,
        ),
    )
    split_rows = _split_rows_by_round(labeled_rows)
    output_schema = pa.Table.from_pylist(labeled_rows).schema if labeled_rows else pa.schema([])
    for split, rows in split_rows.items():
        _write_split(target / f"{split}.parquet", rows, schema=output_schema)

    stats = BuildStats(
        raw_queue_paths=[str(Path(path)) for path in raw_queue_paths],
        bucket_seconds=float(bucket_seconds),
        raw_throttle_ms=int(raw_throttle_ms),
        raw_rows_read=raw.raw_rows_read,
        top_of_book_rows=len(raw.top_of_book_rows),
        orderbook_rows=len(raw.orderbook_rows),
        trade_rows=len(raw.trade_rows),
        feature_rows_generated=len(feature_rows),
        rows_written=len(labeled_rows),
        rows_skipped_quality=label_stats["rows_skipped_quality"],
        rows_skipped_unresolved_outcome=label_stats["rows_skipped_unresolved_outcome"],
        rows_skipped_missing_quote=label_stats["rows_skipped_missing_quote"],
        round_count=len({str(row["round_slug"]) for row in labeled_rows}),
        outcome_counts=dict(Counter(outcomes.values())),
        splits={split: _split_stats(rows) for split, rows in split_rows.items()},
    )
    manifest = {
        "dataset_version": DATASET_VERSION,
        "label_version": LABEL_VERSION,
        "feature_columns": list(_feature_columns()),
        "feature_versions": sorted(
            {
                str(row.get("feature_version"))
                for row in labeled_rows
                if row.get("feature_version") is not None
            }
        ),
        "label_versions": [LABEL_VERSION],
        "bucket_seconds": float(bucket_seconds),
        "row_grain": "event_driven_bucket",
        "source": "low_latency_raw_queue",
        "settlement_outcome_source": (
            "gamma_outcome_prices+final_quote_inference"
            if resolve_gamma_outcomes
            else "final_quote_inference"
        ),
        "convergence_label_formula": (
            "best executable future bid by side minus current side entry worst price"
        ),
        "start_target_price_note": (
            "start_price/target_price are directional placeholders when built from "
            "raw market queues; v7 convergence labels use quote paths."
        ),
        "build": asdict(stats),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return stats


@dataclass(slots=True)
class _RawReadResult:
    raw_rows_read: int
    top_of_book_rows: list[dict[str, Any]]
    orderbook_rows: list[dict[str, Any]]
    trade_rows: list[dict[str, Any]]
    quotes_by_round_side: dict[str, dict[str, list[dict[str, float | int]]]]
    max_ts: int | None


def _read_raw_queues(
    paths: Sequence[Path],
    *,
    max_raw_records: int | None,
    max_rounds: int | None,
    raw_throttle_ms: int,
) -> _RawReadResult:
    selected_rounds: set[str] = set()
    top_by_bucket: dict[tuple[str, int], dict[str, Any]] = {}
    depth_by_bucket: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    trade_rows: list[dict[str, Any]] = []
    quote_path_by_bucket: dict[tuple[str, int], dict[str, Any]] = {}
    rows_read = 0
    max_ts: int | None = None
    latest_selected_round_end: int | None = None
    stop = False
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if max_raw_records is not None and rows_read >= max_raw_records:
                    stop = True
                    break
                text = line.strip()
                if not text:
                    continue
                rows_read += 1
                payload = json.loads(text)
                row = dict(payload.get("row") or {})
                parsed = _parse_symbol(row.get("canonical_symbol"))
                if parsed is None:
                    continue
                _family, round_slug, side = parsed
                ts = _optional_int(row.get("ts") or row.get("message_ts"))
                if ts is None:
                    continue
                max_ts = ts if max_ts is None else max(max_ts, ts)
                if max_rounds is not None and round_slug not in selected_rounds:
                    if len(selected_rounds) >= max_rounds:
                        if latest_selected_round_end is not None and ts > latest_selected_round_end + 600_000:
                            stop = True
                            break
                        continue
                    selected_rounds.add(round_slug)
                    canonical = str(row.get("canonical_symbol") or "")
                    round_end = round_end_ts_from_canonical_symbol(canonical)
                    if round_end is not None:
                        latest_selected_round_end = (
                            round_end
                            if latest_selected_round_end is None
                            else max(latest_selected_round_end, round_end)
                        )
                if max_rounds is not None and round_slug not in selected_rounds:
                    continue
                table = str(payload.get("table") or "")
                if table == "raw_top_of_book":
                    quote = _normalise_top_of_book_row(row)
                    if quote is None:
                        continue
                    canonical = str(quote["canonical_symbol"])
                    bucket = int(quote["ts"]) // raw_throttle_ms
                    top_by_bucket[(canonical, bucket)] = quote
                    quote_path_by_bucket[(canonical, bucket)] = quote
                elif table == "raw_orderbook_snapshot":
                    depth = _normalise_orderbook_row(row)
                    if depth is not None:
                        canonical = str(depth["canonical_symbol"])
                        bucket = int(depth["ts"]) // raw_throttle_ms
                        depth_by_bucket[
                            (
                                canonical,
                                bucket,
                                str(depth["side"]),
                                int(depth["level"]),
                            )
                        ] = depth
                elif table == "raw_trades":
                    trade = _normalise_trade_row(row)
                    if trade is not None:
                        trade_rows.append(trade)
            if stop:
                break
        if stop:
            break
    top_of_book_rows = sorted(top_by_bucket.values(), key=lambda row: int(row["ts"]))
    orderbook_rows = sorted(depth_by_bucket.values(), key=lambda row: int(row["ts"]))
    quotes_by_round_side: dict[str, dict[str, list[dict[str, float | int]]]] = defaultdict(
        lambda: {side: [] for side in SIDES}
    )
    for quote in sorted(quote_path_by_bucket.values(), key=lambda row: int(row["ts"])):
        parsed = _parse_symbol(quote.get("canonical_symbol"))
        if parsed is None:
            continue
        _family, round_slug, side = parsed
        quotes_by_round_side[round_slug][side].append(
            {
                "ts": int(quote["ts"]),
                "bid": float(quote["bid_price"]),
                "ask": float(quote["ask_price"]),
            }
        )
    for by_side in quotes_by_round_side.values():
        for quotes in by_side.values():
            quotes.sort(key=lambda quote: int(quote["ts"]))
    return _RawReadResult(
        rows_read,
        top_of_book_rows,
        orderbook_rows,
        trade_rows,
        quotes_by_round_side,
        max_ts,
    )


def _attach_labels(
    feature_rows: Sequence[dict[str, Any]],
    *,
    quotes_by_round_side: Mapping[str, Mapping[str, Sequence[dict[str, float | int]]]],
    outcomes: Mapping[str, str],
    min_completeness_score: float,
    require_outcome: bool,
    volatility_label_config: VolatilityLabelConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = Counter()
    path_index = {
        round_slug: {
            side: _QuotePath.from_quotes(quotes)
            for side, quotes in by_side.items()
        }
        for round_slug, by_side in quotes_by_round_side.items()
    }
    for feature in feature_rows:
        if not _quality_ok(feature, min_completeness_score=min_completeness_score):
            stats["rows_skipped_quality"] += 1
            continue
        parsed = _parse_symbol(feature.get("canonical_symbol") or feature.get("symbol"))
        if parsed is None:
            stats["rows_skipped_missing_quote"] += 1
            continue
        _family, round_slug, token_side = parsed
        canonical = str(feature.get("canonical_symbol") or "")
        feature_ts = _optional_int(feature.get("feature_ts"))
        round_start_ts = round_start_ts_from_canonical_symbol(canonical)
        round_end_ts = round_end_ts_from_canonical_symbol(canonical)
        if feature_ts is None or round_start_ts is None or round_end_ts is None:
            stats["rows_skipped_missing_quote"] += 1
            continue
        if feature_ts < round_start_ts or feature_ts >= round_end_ts:
            stats["rows_skipped_quality"] += 1
            continue
        quote_sides = quotes_by_round_side.get(round_slug) or {}
        entry_quotes = {
            side: _latest_quote_at(quote_sides.get(side) or (), feature_ts)
            for side in SIDES
        }
        if entry_quotes["UP"] is None or entry_quotes["DOWN"] is None:
            stats["rows_skipped_missing_quote"] += 1
            continue
        outcome = str(outcomes.get(round_slug) or "").upper()
        if outcome not in SIDES:
            if require_outcome:
                stats["rows_skipped_unresolved_outcome"] += 1
                continue
            outcome = "NEUTRAL"
        path_fields = _two_sided_path_fields_fast(
            quote_paths=path_index.get(round_slug) or {},
            decision_ts=feature_ts,
            round_end_ts=round_end_ts,
            config=volatility_label_config,
        )
        side_quote = entry_quotes[token_side]
        assert side_quote is not None
        up_quote = entry_quotes["UP"]
        down_quote = entry_quotes["DOWN"]
        assert up_quote is not None and down_quote is not None
        row = {
            **feature,
            "label_version": LABEL_VERSION,
            "label_kind": "event_driven_convergence",
            "target_ts": round_end_ts,
            "round_slug": round_slug,
            "round_start_ts": round_start_ts,
            "round_end_ts": round_end_ts,
            "start_price": 0.0,
            "target_price": _directional_target_placeholder(outcome),
            "direction_up_15m": outcome == "UP",
            "entry_ask_price": float(side_quote["ask"]),
            "entry_ask_price_up": float(up_quote["ask"]),
            "entry_ask_price_down": float(down_quote["ask"]),
            "entry_bid_price_up": float(up_quote["bid"]),
            "entry_bid_price_down": float(down_quote["bid"]),
            "settlement_price": _settlement_price(outcome, token_side),
            "entry_fee": 0.0,
            "entry_cost": float(side_quote["ask"]),
            "realized_return": _realized_return(outcome, token_side, float(side_quote["ask"])),
            "fee_bps": 0.0,
            "settlement_margin": _settlement_margin_placeholder(outcome),
            "settlement_abs_margin": abs(_settlement_margin_placeholder(outcome)),
            "settlement_neutral_margin": 0.0,
            "label_settlement_3way": outcome,
            "label_profit_up_15m": _profit_label(outcome, float(up_quote["ask"]), "UP"),
            "label_profit_down_15m": _profit_label(
                outcome,
                float(down_quote["ask"]),
                "DOWN",
            ),
            "label_up_15m": outcome == "UP",
            "label_down_15m": outcome == "DOWN",
            "label_source": LABEL_VERSION,
            **path_fields,
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["feature_ts"], row["source"], row["source_symbol"]))
    return rows, {
        "rows_skipped_quality": int(stats["rows_skipped_quality"]),
        "rows_skipped_unresolved_outcome": int(stats["rows_skipped_unresolved_outcome"]),
        "rows_skipped_missing_quote": int(stats["rows_skipped_missing_quote"]),
    }


def _split_rows_by_round(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_round: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_round[str(row["round_slug"])].append(dict(row))
    ordered_rounds = sorted(
        by_round,
        key=lambda round_slug: min(int(row["feature_ts"]) for row in by_round[round_slug]),
    )
    count = len(ordered_rounds)
    train_cut = max(1, int(count * 0.6)) if count else 0
    val_cut = max(train_cut, int(count * 0.8))
    if count >= 3 and val_cut == train_cut:
        val_cut = train_cut + 1
    split_rounds = {
        "train": ordered_rounds[:train_cut],
        "val": ordered_rounds[train_cut:val_cut],
        "test": ordered_rounds[val_cut:],
    }
    return {
        split: sorted(
            [row for round_slug in round_slugs for row in by_round[round_slug]],
            key=lambda row: (row["feature_ts"], row["source"], row["source_symbol"]),
        )
        for split, round_slugs in split_rounds.items()
    }


@dataclass(frozen=True, slots=True)
class _QuotePath:
    ts: tuple[int, ...]
    bids: tuple[float, ...]
    asks: tuple[float, ...]

    @classmethod
    def from_quotes(cls, quotes: Sequence[Mapping[str, float | int]]) -> _QuotePath:
        ordered = sorted(quotes, key=lambda quote: int(quote["ts"]))
        return cls(
            ts=tuple(int(quote["ts"]) for quote in ordered),
            bids=tuple(float(quote["bid"]) for quote in ordered),
            asks=tuple(float(quote["ask"]) for quote in ordered),
        )

    def latest_at(self, ts: int) -> dict[str, float | int] | None:
        idx = bisect_right(self.ts, int(ts)) - 1
        if idx < 0:
            return None
        return {"ts": self.ts[idx], "bid": self.bids[idx], "ask": self.asks[idx]}

    def path_label(
        self,
        *,
        decision_ts: int,
        round_end_ts: int,
        config: VolatilityLabelConfig,
    ) -> dict[str, Any]:
        exit_deadline_ts = int(round_end_ts - int(config.min_exit_seconds_before_expiry * 1000))
        if exit_deadline_ts <= int(decision_ts):
            return _empty_path_fields(exit_deadline_ts, "no_exit_window")
        entry_idx = bisect_left(self.ts, int(decision_ts))
        if entry_idx >= len(self.ts):
            return _empty_path_fields(exit_deadline_ts, "missing_entry_quote")
        entry_ts = self.ts[entry_idx]
        if entry_ts > int(decision_ts) + int(config.max_entry_wait_ms):
            return _empty_path_fields(exit_deadline_ts, "missing_entry_quote")
        entry_ask = self.asks[entry_idx]
        entry_worst = min(0.99, entry_ask + config.buy_slippage + _fee(entry_ask, config.fee_bps))
        start_idx = max(entry_idx, bisect_right(self.ts, int(decision_ts)) - 1)
        end_idx = bisect_right(self.ts, exit_deadline_ts)
        candidates = [
            idx
            for idx in range(start_idx, end_idx)
            if self.ts[idx] > int(decision_ts)
            and self.ts[idx] >= entry_ts
            and math.isfinite(self.bids[idx])
        ]
        if not candidates:
            return {
                **_empty_path_fields(exit_deadline_ts, "missing_exit_path"),
                "entry_quote_ts": entry_ts,
                "entry_ask": entry_ask,
                "entry_worst_price": entry_worst,
            }
        barrier = _barrier_fields(
            ts=self.ts,
            bids=self.bids,
            candidates=candidates,
            entry_worst=entry_worst,
            fee_bps=config.fee_bps,
            sell_slippage=config.sell_slippage,
        )
        best_idx = max(candidates, key=lambda idx: self.bids[idx])
        best_exit_bid = self.bids[best_idx]
        best_exit_price = max(
            0.01,
            best_exit_bid - config.sell_slippage - _fee(best_exit_bid, config.fee_bps),
        )
        max_exit_gain = best_exit_price - entry_worst
        flag = (
            "entry_price_below_min"
            if entry_worst < config.min_entry_price
            else "valid"
        )
        return {
            "label": bool(flag == "valid" and max_exit_gain + 1e-12 >= config.min_exit_gain),
            "max_exit_gain": max_exit_gain,
            "max_exit_return_per_usdc": (
                (best_exit_price / entry_worst) - 1.0
                if entry_worst > 0.0
                else None
            ),
            "time_to_best_exit_seconds": (self.ts[best_idx] - entry_ts) / 1000.0,
            "best_exit_price": best_exit_price,
            "best_exit_bid": best_exit_bid,
            "best_exit_ts": self.ts[best_idx],
            "entry_quote_ts": entry_ts,
            "entry_ask": entry_ask,
            "entry_worst_price": entry_worst,
            "exit_deadline_ts": exit_deadline_ts,
            "path_validity_flag": flag,
            **barrier,
        }


def _barrier_fields(
    *,
    ts: tuple[int, ...],
    bids: tuple[float, ...],
    candidates: list[int],
    entry_worst: float,
    fee_bps: float,
    sell_slippage: float,
) -> dict[str, Any]:
    first_hit_5c_ts: int | None = None
    first_hit_10c_ts: int | None = None
    first_loss_10c_ts: int | None = None
    for idx in candidates:
        exit_price = max(0.01, bids[idx] - sell_slippage - _fee(bids[idx], fee_bps))
        move = exit_price - entry_worst
        if first_hit_5c_ts is None and move >= HIT_5C_DELTA - 1e-12:
            first_hit_5c_ts = ts[idx]
        if first_hit_10c_ts is None and move >= HIT_10C_DELTA - 1e-12:
            first_hit_10c_ts = ts[idx]
        if first_loss_10c_ts is None and move <= LOSS_10C_DELTA + 1e-12:
            first_loss_10c_ts = ts[idx]
    hit_5c_before_loss_10c = first_hit_5c_ts is not None and (
        first_loss_10c_ts is None or first_hit_5c_ts < first_loss_10c_ts
    )
    hit_10c_before_loss_10c = first_hit_10c_ts is not None and (
        first_loss_10c_ts is None or first_hit_10c_ts < first_loss_10c_ts
    )
    loss_10c_before_hit_5c = first_loss_10c_ts is not None and (
        first_hit_5c_ts is None or first_loss_10c_ts < first_hit_5c_ts
    )
    return {
        "first_hit_5c_ts": first_hit_5c_ts,
        "first_hit_10c_ts": first_hit_10c_ts,
        "first_loss_10c_ts": first_loss_10c_ts,
        "hit_5c_before_loss_10c": hit_5c_before_loss_10c,
        "hit_10c_before_loss_10c": hit_10c_before_loss_10c,
        "loss_10c_before_hit_5c": loss_10c_before_hit_5c,
    }


def _two_sided_path_fields_fast(
    *,
    quote_paths: Mapping[str, _QuotePath],
    decision_ts: int,
    round_end_ts: int,
    config: VolatilityLabelConfig,
) -> dict[str, Any]:
    if not quote_paths:
        return two_sided_volatility_fields(
            quotes_by_side={},
            decision_ts=decision_ts,
            round_end_ts=round_end_ts,
            config=config,
        )
    fields: dict[str, Any] = {}
    for side_lower, side in (("up", "UP"), ("down", "DOWN")):
        path = quote_paths.get(side)
        result = (
            path.path_label(decision_ts=decision_ts, round_end_ts=round_end_ts, config=config)
            if path is not None
            else _empty_path_fields(
                int(round_end_ts - int(config.min_exit_seconds_before_expiry * 1000)),
                "missing_price_path",
            )
        )
        fields[f"max_exit_gain_{side_lower}"] = result["max_exit_gain"]
        fields[f"max_exit_return_per_usdc_{side_lower}"] = result["max_exit_return_per_usdc"]
        fields[f"time_to_best_exit_{side_lower}"] = result["time_to_best_exit_seconds"]
        fields[f"best_exit_price_{side_lower}"] = result["best_exit_price"]
        fields[f"label_volatility_{side_lower}"] = result["label"]
        fields[f"volatility_path_validity_{side_lower}"] = result["path_validity_flag"]
        fields[f"first_hit_5c_ts_{side_lower}"] = result["first_hit_5c_ts"]
        fields[f"first_hit_10c_ts_{side_lower}"] = result["first_hit_10c_ts"]
        fields[f"first_loss_10c_ts_{side_lower}"] = result["first_loss_10c_ts"]
        fields[f"hit_5c_before_loss_10c_{side_lower}"] = result[
            "hit_5c_before_loss_10c"
        ]
        fields[f"hit_10c_before_loss_10c_{side_lower}"] = result[
            "hit_10c_before_loss_10c"
        ]
        fields[f"loss_10c_before_hit_5c_{side_lower}"] = result[
            "loss_10c_before_hit_5c"
        ]
    return fields


def _empty_path_fields(exit_deadline_ts: int, flag: str) -> dict[str, Any]:
    return {
        "label": None,
        "max_exit_gain": None,
        "max_exit_return_per_usdc": None,
        "time_to_best_exit_seconds": None,
        "best_exit_price": None,
        "best_exit_bid": None,
        "best_exit_ts": None,
        "entry_quote_ts": None,
        "entry_ask": None,
        "entry_worst_price": None,
        "exit_deadline_ts": exit_deadline_ts,
        "path_validity_flag": flag,
        "first_hit_5c_ts": None,
        "first_hit_10c_ts": None,
        "first_loss_10c_ts": None,
        "hit_5c_before_loss_10c": None,
        "hit_10c_before_loss_10c": None,
        "loss_10c_before_hit_5c": None,
    }


def _write_split(path: Path, rows: Sequence[dict[str, Any]], *, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=schema) if rows else pa.Table.from_pylist([], schema=schema)
    pq.write_table(table, path)


def _split_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(str(row.get("label_settlement_3way")) for row in rows)
    rounds = {str(row.get("round_slug")) for row in rows}
    feature_ts = [int(row["feature_ts"]) for row in rows] if rows else []
    return {
        "row_count": len(rows),
        "round_count": len(rounds),
        "outcomes": dict(outcomes),
        "start_ts": min(feature_ts) if feature_ts else None,
        "end_ts": max(feature_ts) if feature_ts else None,
        "valid_path_up": sum(1 for row in rows if row.get("best_exit_price_up") is not None),
        "valid_path_down": sum(1 for row in rows if row.get("best_exit_price_down") is not None),
        "hit_5c_before_loss_10c_up": sum(
            1 for row in rows if row.get("hit_5c_before_loss_10c_up") is True
        ),
        "hit_5c_before_loss_10c_down": sum(
            1 for row in rows if row.get("hit_5c_before_loss_10c_down") is True
        ),
        "loss_10c_before_hit_5c_up": sum(
            1 for row in rows if row.get("loss_10c_before_hit_5c_up") is True
        ),
        "loss_10c_before_hit_5c_down": sum(
            1 for row in rows if row.get("loss_10c_before_hit_5c_down") is True
        ),
    }


def _normalise_top_of_book_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    ts = _optional_int(row.get("ts") or row.get("message_ts"))
    bid = _optional_float(row.get("bid_price"))
    ask = _optional_float(row.get("ask_price"))
    if ts is None or bid is None or ask is None:
        return None
    out = _identity_fields(row, ts)
    out.update(
        {
            "bid_price": bid,
            "ask_price": ask,
            "spread": _optional_float(row.get("spread")),
        }
    )
    if is_degenerate_quote(out, market_implied_prob=ask):
        return None
    return out


def _normalise_orderbook_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    ts = _optional_int(row.get("ts") or row.get("message_ts"))
    level = _optional_int(row.get("level"))
    price = _optional_float(row.get("price"))
    size = _optional_float(row.get("size"))
    side = str(row.get("side") or "").upper()
    if ts is None or level is None or price is None or size is None or side not in {"BID", "ASK"}:
        return None
    out = _identity_fields(row, ts)
    out.update({"side": side, "level": level, "price": price, "size": size})
    return out


def _normalise_trade_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    ts = _optional_int(row.get("ts") or row.get("message_ts"))
    price = _optional_float(row.get("price"))
    size = _optional_float(row.get("size"))
    side = str(row.get("side") or "").upper()
    if ts is None or price is None or size is None:
        return None
    out = _identity_fields(row, ts)
    out.update({"price": price, "size": size, "side": side})
    return out


def _identity_fields(row: Mapping[str, Any], ts: int) -> dict[str, Any]:
    return {
        "ts": ts,
        "message_ts": _optional_int(row.get("message_ts")) or ts,
        "ingest_ts": _optional_int(row.get("ingest_ts") or row.get("capture_timestamp_ms")) or ts,
        "source": str(row.get("source") or "polymarket"),
        "source_symbol": str(row.get("source_symbol") or ""),
        "source_market": str(row.get("source_market") or ""),
        "canonical_symbol": str(row.get("canonical_symbol") or ""),
    }


def _quality_ok(row: Mapping[str, Any], *, min_completeness_score: float) -> bool:
    if row.get("quality_filter_pass") is False:
        return False
    completeness = _optional_float(row.get("completeness_score"))
    return completeness is None or completeness >= min_completeness_score


def _latest_quote_at(
    quotes: Iterable[Mapping[str, float | int]],
    ts: int,
) -> dict[str, float | int] | None:
    latest: dict[str, float | int] | None = None
    for quote in quotes:
        if int(quote["ts"]) > ts:
            break
        latest = dict(quote)
    return latest


def _parse_symbol(value: Any) -> tuple[str, str, str] | None:
    canonical = str(value or "")
    parts = canonical.split(":")
    if len(parts) < 3:
        return None
    family = parts[0].upper()
    round_slug = parts[-2]
    side = parts[-1].upper()
    if not family.startswith("BTC-15M") or side not in SIDES:
        return None
    return family, round_slug, side


def _read_outcome_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(key): str(value).upper()
        for key, value in payload.items()
        if str(value).upper() in SIDES
    }


def _write_outcome_cache(path: Path, outcomes: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(sorted(outcomes.items())), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _resolve_gamma_outcomes(
    round_slugs: Sequence[str],
    *,
    gamma_api_base: str,
    timeout_seconds: float,
) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for round_slug in round_slugs:
        outcome = _fetch_gamma_outcome(
            round_slug,
            gamma_api_base=gamma_api_base,
            timeout_seconds=timeout_seconds,
        )
        if outcome in SIDES:
            outcomes[round_slug] = outcome
    return outcomes


def _fetch_gamma_outcome(
    round_slug: str,
    *,
    gamma_api_base: str,
    timeout_seconds: float,
) -> str | None:
    url = f"{gamma_api_base.rstrip('/')}/markets?{parse.urlencode({'slug': round_slug, 'limit': '1'})}"
    req = request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "BiGan-v7-event-corpus/1.0"},
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, urlerror.URLError, json.JSONDecodeError):
        return None
    market = None
    if isinstance(payload, list) and payload:
        market = payload[0]
    elif isinstance(payload, Mapping):
        markets = payload.get("markets") or payload.get("data") or []
        if isinstance(markets, list) and markets:
            market = markets[0]
    if not isinstance(market, Mapping):
        return None
    outcomes_raw = market.get("outcomes")
    prices_raw = market.get("outcomePrices")
    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
    except json.JSONDecodeError:
        return None
    if not isinstance(outcomes, list) or not isinstance(prices, list) or len(outcomes) != len(prices):
        return None
    parsed_prices = [_optional_float(price) for price in prices]
    if any(price is None for price in parsed_prices):
        return None
    winner_idx = max(range(len(parsed_prices)), key=lambda idx: float(parsed_prices[idx] or 0.0))
    if float(parsed_prices[winner_idx] or 0.0) < 0.95:
        return None
    outcome = str(outcomes[winner_idx]).upper()
    return outcome if outcome in SIDES else None


def _infer_outcomes_from_quotes(
    quotes_by_round_side: Mapping[str, Mapping[str, Sequence[dict[str, float | int]]]],
) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for round_slug, by_side in quotes_by_round_side.items():
        scores = {}
        for side in SIDES:
            quotes = list(by_side.get(side) or ())
            if not quotes:
                continue
            tail = quotes[-20:]
            scores[side] = max(max(float(q["bid"]), float(q["ask"])) for q in tail)
        if scores.get("UP", 0.0) >= 0.95 and scores.get("UP", 0.0) > scores.get("DOWN", 0.0):
            outcomes[round_slug] = "UP"
        elif scores.get("DOWN", 0.0) >= 0.95 and scores.get("DOWN", 0.0) > scores.get("UP", 0.0):
            outcomes[round_slug] = "DOWN"
    return outcomes


def _settlement_price(outcome: str, token_side: str) -> float | None:
    if outcome not in SIDES:
        return None
    return 1.0 if outcome == token_side else 0.0


def _realized_return(outcome: str, token_side: str, ask: float) -> float | None:
    settlement = _settlement_price(outcome, token_side)
    if settlement is None or ask <= 0.0:
        return None
    return (settlement - ask) / ask


def _profit_label(outcome: str, ask: float, side: str) -> bool | None:
    settlement = _settlement_price(outcome, side)
    if settlement is None:
        return None
    return bool(settlement - ask > 0.0)


def _directional_target_placeholder(outcome: str) -> float | None:
    if outcome == "UP":
        return 1.0
    if outcome == "DOWN":
        return -1.0
    return 0.0 if outcome == "NEUTRAL" else None


def _settlement_margin_placeholder(outcome: str) -> float:
    value = _directional_target_placeholder(outcome)
    return 0.0 if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _fee(price: float, fee_bps: float) -> float:
    return float(price) * float(fee_bps) / 10_000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-queue-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-seconds", type=float, default=10.0)
    parser.add_argument("--max-raw-records", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--raw-throttle-ms", type=int, default=1000)
    parser.add_argument("--min-completeness-score", type=float, default=0.8)
    parser.add_argument("--resolve-gamma-outcomes", action="store_true")
    parser.add_argument("--gamma-api-base", default=DEFAULT_GAMMA_API_BASE)
    parser.add_argument("--request-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--outcome-cache-path")
    parser.add_argument("--no-final-quote-outcome-inference", action="store_true")
    parser.add_argument("--allow-unresolved-outcomes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = build_event_driven_dataset(
        [Path(path) for path in args.raw_queue_path],
        Path(args.output_dir),
        bucket_seconds=args.bucket_seconds,
        max_raw_records=args.max_raw_records,
        max_rounds=args.max_rounds,
        raw_throttle_ms=args.raw_throttle_ms,
        min_completeness_score=args.min_completeness_score,
        resolve_gamma_outcomes=args.resolve_gamma_outcomes,
        gamma_api_base=args.gamma_api_base,
        request_timeout_seconds=args.request_timeout_seconds,
        outcome_cache_path=args.outcome_cache_path,
        infer_outcomes_from_final_quotes=not args.no_final_quote_outcome_inference,
        require_outcome=not args.allow_unresolved_outcomes,
    )
    print(json.dumps(asdict(stats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
