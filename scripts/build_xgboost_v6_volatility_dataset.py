#!/usr/bin/env python
"""Fill xgboost-v6 volatility labels from rollup best-bid/ask paths."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from bigan.labels.v6 import VolatilityLabelConfig, compute_volatility_path_label
from bigan.modeling.dataset import SPLITS, _v6_label_diagnostics
from bigan.modeling.families import market_family_from_symbol

VOLATILITY_COLUMNS: tuple[str, ...] = (
    "max_exit_gain_up",
    "max_exit_gain_down",
    "max_exit_return_per_usdc_up",
    "max_exit_return_per_usdc_down",
    "time_to_best_exit_up",
    "time_to_best_exit_down",
    "best_exit_price_up",
    "best_exit_price_down",
    "label_volatility_up",
    "label_volatility_down",
    "volatility_path_validity_up",
    "volatility_path_validity_down",
)


def main() -> None:
    args = _parse_args()
    base_dataset = args.base_dataset
    rollup_root = args.rollup_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = VolatilityLabelConfig(
        min_exit_gain=args.min_exit_gain,
        buy_slippage=args.buy_slippage,
        sell_slippage=args.sell_slippage,
        max_entry_wait_ms=args.max_entry_wait_ms,
        min_exit_seconds_before_expiry=args.min_exit_seconds_before_expiry,
        min_entry_price=args.min_entry_price,
        fee_bps=args.fee_bps,
    )
    split_rows = _read_split_rows(base_dataset)
    rows_by_date = _rows_by_decision_date(split_rows)
    summary = {
        "base_dataset": str(base_dataset),
        "rollup_root": str(rollup_root),
        "output_dir": str(output_dir),
        "volatility_config": {
            "min_exit_gain": config.min_exit_gain,
            "buy_slippage": config.buy_slippage,
            "sell_slippage": config.sell_slippage,
            "max_entry_wait_ms": config.max_entry_wait_ms,
            "min_exit_seconds_before_expiry": config.min_exit_seconds_before_expiry,
            "min_entry_price": config.min_entry_price,
            "fee_bps": config.fee_bps,
        },
        "date_batches": {},
    }

    for date, refs in sorted(rows_by_date.items()):
        needed_symbols = _needed_symbols(split_rows, refs)
        quote_paths = _rollup_quote_paths(rollup_root, date)
        quotes_by_symbol = _load_rollup_quotes(quote_paths, needed_symbols)
        date_summary = _fill_refs_for_date(
            split_rows,
            refs,
            quotes_by_symbol,
            config=config,
        )
        date_summary["quote_files"] = len(quote_paths)
        date_summary["needed_symbols"] = len(needed_symbols)
        date_summary["quote_symbols"] = len(quotes_by_symbol)
        date_summary["quote_rows"] = sum(len(quotes) for quotes in quotes_by_symbol.values())
        summary["date_batches"][date] = date_summary

    split_tables = _write_splits(output_dir, split_rows)
    manifest = _build_manifest(base_dataset, output_dir, split_tables, summary)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "volatility_rollup_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_compact_summary(manifest, summary), indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--rollup-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-exit-gain", type=float, default=0.15)
    parser.add_argument("--buy-slippage", type=float, default=0.02)
    parser.add_argument("--sell-slippage", type=float, default=0.02)
    parser.add_argument("--max-entry-wait-ms", type=int, default=60_000)
    parser.add_argument("--min-exit-seconds-before-expiry", type=float, default=300.0)
    parser.add_argument("--min-entry-price", type=float, default=0.35)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    return parser.parse_args()


def _read_split_rows(base_dataset: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        rows = pq.read_table(base_dataset / f"{split}.parquet").to_pylist()
        for row in rows:
            for column in VOLATILITY_COLUMNS:
                row[column] = None
            row["volatility_label_source"] = None
        rows_by_split[split] = rows
    return rows_by_split


def _rows_by_decision_date(
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, list[tuple[str, int]]]:
    by_date: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for split, rows in split_rows.items():
        for idx, row in enumerate(rows):
            feature_ts = row.get("feature_ts")
            if feature_ts is None:
                continue
            date = _date_from_ms(int(feature_ts))
            by_date[date].append((split, idx))
    return dict(by_date)


def _needed_symbols(
    split_rows: dict[str, list[dict[str, Any]]],
    refs: Iterable[tuple[str, int]],
) -> set[str]:
    symbols: set[str] = set()
    for split, idx in refs:
        row = split_rows[split][idx]
        symbol = str(row.get("canonical_symbol") or "")
        if not symbol:
            continue
        symbols.add(_side_symbol(symbol, "UP"))
        symbols.add(_side_symbol(symbol, "DOWN"))
    return symbols


def _rollup_quote_paths(rollup_root: Path, date: str) -> list[str]:
    dates = [date, _next_date(date)]
    paths: list[str] = []
    for partition_date in dates:
        paths.extend(
            str(path)
            for path in (
                rollup_root
                / "rollup"
                / "ws_market"
                / f"date={partition_date}"
                / "event_type=best_bid_ask"
            ).glob("*.parquet")
        )
    return sorted(paths)


def _load_rollup_quotes(
    quote_paths: list[str],
    needed_symbols: set[str],
) -> dict[str, list[dict[str, float | int | None]]]:
    if not quote_paths or not needed_symbols:
        return {}
    con = duckdb.connect()
    con.execute("create temp table needed(symbol varchar)")
    con.executemany("insert into needed values (?)", [(symbol,) for symbol in sorted(needed_symbols)])
    table = con.execute(
        """
        select
          coalesce(
            cast(json_extract_string(raw_payload, '$.timestamp') as bigint),
            exchange_time,
            receive_time
          ) as ts,
          json_extract_string(raw_payload, '$.canonical_symbol') as canonical_symbol,
          cast(json_extract_string(raw_payload, '$.best_bid') as double) as bid,
          cast(json_extract_string(raw_payload, '$.best_ask') as double) as ask
        from read_parquet(?) q
        inner join needed n
          on json_extract_string(raw_payload, '$.canonical_symbol') = n.symbol
        where json_extract_string(raw_payload, '$.canonical_symbol') is not null
        """,
        [quote_paths],
    ).to_arrow_table()
    quotes: dict[str, list[dict[str, float | int | None]]] = defaultdict(list)
    for row in table.to_pylist():
        symbol = row.get("canonical_symbol")
        ts = row.get("ts")
        if symbol is None or ts is None:
            continue
        quotes[str(symbol)].append(
            {
                "ts": int(ts),
                "bid": _float_or_none(row.get("bid")),
                "ask": _float_or_none(row.get("ask")),
            }
        )
    for symbol_quotes in quotes.values():
        symbol_quotes.sort(key=lambda quote: int(quote["ts"] or 0))
    return dict(quotes)


def _fill_refs_for_date(
    split_rows: dict[str, list[dict[str, Any]]],
    refs: list[tuple[str, int]],
    quotes_by_symbol: dict[str, list[dict[str, float | int | None]]],
    *,
    config: VolatilityLabelConfig,
) -> dict[str, Any]:
    quote_times = {
        symbol: [int(quote["ts"] or 0) for quote in quotes]
        for symbol, quotes in quotes_by_symbol.items()
    }
    counters: Counter[str] = Counter()
    family_counters: dict[str, Counter[str]] = defaultdict(Counter)
    for split, idx in refs:
        row = split_rows[split][idx]
        feature_ts = int(row["feature_ts"])
        round_end_ts = row.get("round_end_ts")
        if round_end_ts is None:
            continue
        symbol = str(row.get("canonical_symbol") or "")
        family = market_family_from_symbol(symbol)
        for side in ("up", "down"):
            side_symbol = _side_symbol(symbol, side.upper())
            result = _compute_label_from_symbol_quotes(
                quotes_by_symbol.get(side_symbol, []),
                quote_times.get(side_symbol, []),
                decision_ts=feature_ts,
                round_end_ts=int(round_end_ts),
                config=config,
            )
            row[f"max_exit_gain_{side}"] = result.max_exit_gain
            row[f"max_exit_return_per_usdc_{side}"] = result.max_exit_return_per_usdc
            row[f"time_to_best_exit_{side}"] = result.time_to_best_exit_seconds
            row[f"best_exit_price_{side}"] = result.best_exit_price
            row[f"label_volatility_{side}"] = result.label
            row[f"volatility_path_validity_{side}"] = result.path_validity_flag
            key = f"{side}:{result.path_validity_flag}"
            counters[key] += 1
            family_counters[f"{family}:{side}"][result.path_validity_flag] += 1
            if result.label is not None:
                counters[f"{side}:known"] += 1
                family_counters[f"{family}:{side}"]["known"] += 1
                row["volatility_label_source"] = "rollup_ws_market_best_bid_ask"
                if result.label:
                    counters[f"{side}:positive"] += 1
                    family_counters[f"{family}:{side}"]["positive"] += 1
    return {
        "rows": len(refs),
        "counts": dict(sorted(counters.items())),
        "family_counts": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(family_counters.items())
        },
    }


def _compute_label_from_symbol_quotes(
    quotes: list[dict[str, float | int | None]],
    times: list[int],
    *,
    decision_ts: int,
    round_end_ts: int,
    config: VolatilityLabelConfig,
):
    exit_deadline_ts = round_end_ts - int(config.min_exit_seconds_before_expiry * 1000)
    if not quotes or not times:
        return compute_volatility_path_label(
            (),
            decision_ts=decision_ts,
            round_end_ts=round_end_ts,
            config=config,
        )
    start = bisect.bisect_left(times, decision_ts)
    end = bisect.bisect_right(times, exit_deadline_ts)
    return compute_volatility_path_label(
        quotes[start:end],
        decision_ts=decision_ts,
        round_end_ts=round_end_ts,
        config=config,
    )


def _write_splits(output_dir: Path, split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, pa.Table]:
    tables: dict[str, pa.Table] = {}
    for split, rows in split_rows.items():
        table = pa.Table.from_pylist(rows)
        tables[split] = table
        pq.write_table(table, output_dir / f"{split}.parquet")
    return tables


def _build_manifest(
    base_dataset: Path,
    output_dir: Path,
    split_tables: dict[str, pa.Table],
    summary: dict[str, Any],
) -> dict[str, Any]:
    base_manifest_path = base_dataset / "manifest.json"
    manifest = (
        json.loads(base_manifest_path.read_text(encoding="utf-8"))
        if base_manifest_path.exists()
        else {}
    )
    manifest["output_dir"] = str(output_dir)
    manifest["source_dataset"] = str(base_dataset)
    manifest["volatility_rollup_summary_path"] = str(output_dir / "volatility_rollup_summary.json")
    manifest["volatility_rollup_source"] = {
        "rollup_root": summary["rollup_root"],
        "quote_event_type": "best_bid_ask",
    }
    manifest["v6_label_diagnostics"] = _v6_label_diagnostics(split_tables)
    manifest["schema_extra_columns"] = sorted(
        set(manifest.get("schema_extra_columns", [])) | {"volatility_label_source"}
    )
    return manifest


def _compact_summary(manifest: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    diagnostics = manifest["v6_label_diagnostics"]["family_volatility_label_rates"]
    compact: dict[str, Any] = {
        "output_dir": manifest["output_dir"],
        "date_batches": summary["date_batches"],
        "family_known_labels": {},
    }
    for family, split_stats in diagnostics.items():
        compact["family_known_labels"][family] = {
            split: {
                side: split_stats[split][side]["known_label_count"]
                for side in ("up", "down")
            }
            for split in SPLITS
        }
    return compact


def _side_symbol(symbol: str, side: str) -> str:
    if ":" in symbol:
        return f"{symbol.rsplit(':', 1)[0]}:{side.upper()}"
    return symbol


def _date_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).date().isoformat()


def _next_date(date: str) -> str:
    parsed = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
    return (parsed + timedelta(days=1)).date().isoformat()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


if __name__ == "__main__":
    main()
