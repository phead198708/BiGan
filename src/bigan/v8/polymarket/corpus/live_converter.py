"""Convert historical data/live Polymarket observations into v8 Phase 2 corpus inputs."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus.builder import build_polymarket_btc_corpus
from bigan.v8.polymarket.corpus.contracts import (
    BTC_UPDOWN_MARKET_HORIZONS_MS,
    DEFAULT_CORPUS_CREATED_AT,
    RAW_CORPUS_FILENAMES,
    PolymarketCorpusBuildConfig,
    PolymarketCorpusBuildResult,
    safety_fields,
)

LIVE_TO_CORPUS_SCHEMA_VERSION = "bigan-v8-polymarket-live-to-corpus-v1"
LIVE_TO_CORPUS_PHASE = "polymarket_live_to_phase2_corpus"
SUPPORTED_SIGNAL_FILENAMES = ("signals.jsonl", "signals-v7-event-driven.jsonl", "signals-event-driven.jsonl")
DEFAULT_SETTLEMENT_RULE = (
    "UP wins if the BTC reference price at market end is greater than the BTC reference "
    "price at market start; otherwise DOWN wins."
)


@dataclass(frozen=True, slots=True)
class LiveSignalCorpusConversionConfig:
    """Configuration for converting historical live signal observations."""

    input_path: Path | str
    output_dir: Path | str
    created_at: str = DEFAULT_CORPUS_CREATED_AT
    market_families: tuple[str, ...] = tuple(BTC_UPDOWN_MARKET_HORIZONS_MS)
    build_phase2_corpus: bool = True
    allow_midpoint_price_proxy: bool = False
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.input_path, Path):
            object.__setattr__(self, "input_path", Path(self.input_path))
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.created_at:
            raise ValueError("created_at is required")
        unsupported = set(self.market_families) - set(BTC_UPDOWN_MARKET_HORIZONS_MS)
        if unsupported:
            raise ValueError("unsupported market families: " + ", ".join(sorted(unsupported)))
        for field_name, expected in safety_fields().items():
            if getattr(self, field_name) is not expected:
                raise ValueError(f"{field_name} must be {str(expected).lower()}")

    @property
    def raw_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / "raw"

    @property
    def corpus_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / "phase2_corpus"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["output_dir"] = str(self.output_dir)
        payload["market_families"] = list(self.market_families)
        return payload


@dataclass(frozen=True, slots=True)
class LiveSignalCorpusConversionResult:
    """Output handles for one live-to-corpus conversion."""

    output_dir: Path
    raw_dir: Path
    corpus_dir: Path | None
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    report: dict[str, Any]
    phase2_result: PolymarketCorpusBuildResult | None


@dataclass(slots=True)
class _MarketAccumulator:
    market_id: str
    slug: str
    market_family: str
    horizon_ms: int
    market_start_ts: int
    market_end_ts: int
    up_token_id: str | None = None
    down_token_id: str | None = None
    condition_id: str | None = None
    reference_price_start: float | None = None
    reference_price_end: float | None = None
    resolution_status: str = "normal"
    settlement_rule: str = DEFAULT_SETTLEMENT_RULE
    rows: list[dict[str, Any]] | None = None
    reject_reasons: set[str] | None = None

    def __post_init__(self) -> None:
        self.rows = [] if self.rows is None else self.rows
        self.reject_reasons = set() if self.reject_reasons is None else self.reject_reasons


def convert_live_signals_to_phase2_corpus(
    config: LiveSignalCorpusConversionConfig,
) -> LiveSignalCorpusConversionResult:
    """Convert historical live signal rows into raw and normalized v8 corpus artifacts."""

    output_dir = config.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"live-to-corpus output_dir already exists: {output_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    config.raw_dir.mkdir(parents=True)

    signal_files = _discover_signal_files(config.input_path.expanduser().resolve())
    raw_rows = _read_signal_rows(signal_files)
    markets, rejected_rows = _accumulate_markets(config=config, rows=raw_rows)
    raw_payloads, rejected_markets = _build_raw_payloads(config=config, markets=markets)
    rejected_rows.extend(rejected_markets)
    _write_raw_files(config.raw_dir, raw_payloads)

    artifact_paths = {
        "conversion_manifest": output_dir / "live_signal_conversion_manifest.json",
        "conversion_report": output_dir / "live_signal_conversion_report.json",
        "rejected_rows": output_dir / "live_signal_rejected_rows.jsonl",
        **{filename: config.raw_dir / filename for filename in RAW_CORPUS_FILENAMES},
    }
    _write_jsonl(artifact_paths["rejected_rows"], rejected_rows)
    phase2_result = None
    phase2_error: str | None = None
    if config.build_phase2_corpus and raw_payloads["raw_polymarket_markets.jsonl"]:
        try:
            phase2_result = build_polymarket_btc_corpus(
                PolymarketCorpusBuildConfig(
                    input_dir=config.raw_dir,
                    output_dir=config.corpus_dir,
                    created_at=config.created_at,
                    market_families=tuple(config.market_families),  # type: ignore[arg-type]
                    overwrite_existing=True,
                )
            )
        except Exception as exc:
            phase2_error = str(exc)
    report = _conversion_report(
        config=config,
        signal_files=signal_files,
        raw_rows=raw_rows,
        raw_payloads=raw_payloads,
        rejected_rows=rejected_rows,
        phase2_result=phase2_result,
        phase2_error=phase2_error,
    )
    manifest = {
        "schema_version": LIVE_TO_CORPUS_SCHEMA_VERSION,
        "phase": LIVE_TO_CORPUS_PHASE,
        "created_at": config.created_at,
        "config": config.to_dict(),
        "input_files": [str(path) for path in signal_files],
        "raw_artifact_hashes": {
            filename: _sha256_file(config.raw_dir / filename)
            for filename in RAW_CORPUS_FILENAMES
        },
        "phase2_corpus_built": phase2_result is not None,
        "phase2_corpus_dir": None if phase2_result is None else str(phase2_result.output_dir),
        "training_eligible": report["training_eligible"],
        **safety_fields(),
    }
    _write_json(artifact_paths["conversion_report"], report)
    _write_json(artifact_paths["conversion_manifest"], manifest)
    if phase2_result is not None:
        artifact_paths.update(
            {
                f"phase2_{name}": path
                for name, path in phase2_result.artifact_paths.items()
            }
        )
    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
        if path.exists()
    }
    return LiveSignalCorpusConversionResult(
        output_dir=output_dir,
        raw_dir=config.raw_dir,
        corpus_dir=None if phase2_result is None else phase2_result.output_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        report=report,
        phase2_result=phase2_result,
    )


def _discover_signal_files(input_path: Path) -> tuple[Path, ...]:
    if input_path.is_file():
        return (input_path,)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    files = [
        path
        for path in sorted(input_path.rglob("*.jsonl"))
        if path.name in SUPPORTED_SIGNAL_FILENAMES or path.name.startswith("signals")
    ]
    if not files:
        raise FileNotFoundError(f"no signal jsonl files found under {input_path}")
    return tuple(files)


def _read_signal_rows(signal_files: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in signal_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["_source_file"] = str(path)
            payload["_source_line"] = line_number
            rows.append(payload)
    return rows


def _accumulate_markets(
    *,
    config: LiveSignalCorpusConversionConfig,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, _MarketAccumulator], list[dict[str, Any]]]:
    markets: dict[str, _MarketAccumulator] = {}
    rejected: list[dict[str, Any]] = []
    for row in rows:
        parsed = _parse_signal_row(row)
        if parsed["reject_reasons"]:
            rejected.append(_rejected_row(row, parsed["reject_reasons"]))
            continue
        if parsed["market_family"] not in config.market_families:
            rejected.append(_rejected_row(row, ("unsupported_market_family",)))
            continue
        market = markets.setdefault(
            parsed["market_id"],
            _MarketAccumulator(
                market_id=parsed["market_id"],
                slug=parsed["slug"],
                market_family=parsed["market_family"],
                horizon_ms=BTC_UPDOWN_MARKET_HORIZONS_MS[parsed["market_family"]],
                market_start_ts=parsed["market_start_ts"],
                market_end_ts=parsed["market_end_ts"],
            ),
        )
        if market.market_start_ts != parsed["market_start_ts"] or market.market_end_ts != parsed["market_end_ts"]:
            market.reject_reasons.add("inconsistent_market_window")
        market.rows.append({**row, **parsed})
        _update_market_tokens(market, parsed)
        _update_market_resolution(market, row)
    return markets, rejected


def _parse_signal_row(row: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    slug = str(row.get("round_slug") or row.get("slug") or "").strip()
    if not slug:
        reasons.append("missing_round_slug")
    family = _family_from_slug(slug)
    if family is None:
        reasons.append("unsupported_or_missing_market_family")
        family = "unsupported"
    try:
        market_start_ts = _market_start_ts_from_slug(slug)
    except ValueError:
        reasons.append("missing_market_start_ts")
        market_start_ts = 0
    horizon_ms = BTC_UPDOWN_MARKET_HORIZONS_MS.get(family, 0)
    market_end_ts = int(row.get("round_end_ts") or market_start_ts + horizon_ms)
    ts = _optional_int(row, "ts")
    if ts is None:
        reasons.append("missing_decision_ts")
        ts = market_start_ts
    outcome = str(row.get("outcome_side") or row.get("selected_side") or "").upper().strip()
    if outcome not in {"UP", "DOWN"}:
        reasons.append("missing_outcome_side")
        outcome = "UP"
    token_id = str(row.get("token_id") or "").strip()
    opposite_token_id = str(row.get("opposite_token_id") or "").strip()
    if not token_id or not opposite_token_id:
        reasons.append("missing_up_down_token_ids")
    market_id = str(row.get("market_id") or slug).strip()
    return {
        "reject_reasons": tuple(reasons),
        "market_id": market_id,
        "slug": slug,
        "market_family": family,
        "market_start_ts": market_start_ts,
        "market_end_ts": market_end_ts,
        "decision_ts": ts,
        "outcome": outcome,
        "token_id": token_id,
        "opposite_token_id": opposite_token_id,
    }


def _family_from_slug(slug: str) -> str | None:
    normalized = slug.lower()
    if "updown-5m" in normalized or "up-or-down-5m" in normalized:
        return "btc_updown_5m"
    if "updown-15m" in normalized or "up-or-down-15m" in normalized:
        return "btc_updown_15m"
    if "updown-1h" in normalized or "up-or-down-1h" in normalized:
        return "btc_updown_1h"
    if "btc-15m" in normalized:
        return "btc_updown_15m"
    return None


def _market_start_ts_from_slug(slug: str) -> int:
    last = slug.rsplit("-", 1)[-1]
    if not last.isdigit():
        raise ValueError("slug does not end with start epoch seconds")
    return int(last) * 1000


def _update_market_tokens(market: _MarketAccumulator, parsed: dict[str, Any]) -> None:
    if parsed["outcome"] == "UP":
        up_token_id = parsed["token_id"]
        down_token_id = parsed["opposite_token_id"]
    else:
        up_token_id = parsed["opposite_token_id"]
        down_token_id = parsed["token_id"]
    if market.up_token_id is None:
        market.up_token_id = up_token_id
    elif market.up_token_id != up_token_id:
        market.reject_reasons.add("inconsistent_up_token_id")
    if market.down_token_id is None:
        market.down_token_id = down_token_id
    elif market.down_token_id != down_token_id:
        market.reject_reasons.add("inconsistent_down_token_id")
    if market.condition_id is None:
        market.condition_id = "condition-" + canonical_json_sha256(market.market_id)[:16]


def _update_market_resolution(market: _MarketAccumulator, row: dict[str, Any]) -> None:
    start = _first_float(row, ("reference_price_start", "btc_reference_price_start", "open_price"))
    end = _first_float(row, ("reference_price_end", "btc_reference_price_end", "close_price"))
    if start is not None:
        market.reference_price_start = start
    if end is not None:
        market.reference_price_end = end
    resolution_status = str(row.get("resolution_status") or "").strip()
    if resolution_status:
        market.resolution_status = resolution_status
    raw_rule = str(row.get("settlement_rule") or "").strip()
    if raw_rule:
        market.settlement_rule = raw_rule


def _build_raw_payloads(
    *,
    config: LiveSignalCorpusConversionConfig,
    markets: dict[str, _MarketAccumulator],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    raw_payloads = {filename: [] for filename in RAW_CORPUS_FILENAMES}
    rejected: list[dict[str, Any]] = []
    for market in sorted(markets.values(), key=lambda item: (item.market_start_ts, item.market_id)):
        reasons = set(market.reject_reasons)
        if not market.up_token_id or not market.down_token_id:
            reasons.add("missing_up_down_token_ids")
        books = _book_rows_for_market(config=config, market=market)
        if not books:
            reasons.add("missing_executable_up_down_orderbook")
        candles = _candle_rows_for_market(market)
        if not candles:
            reasons.add("missing_btc_reference_candles")
        if market.reference_price_start is None or market.reference_price_end is None:
            reasons.add("missing_verified_resolution")
        if reasons:
            rejected.append(
                {
                    "market_id": market.market_id,
                    "slug": market.slug,
                    "reject_reasons": sorted(reasons),
                    "row_count": len(market.rows or []),
                }
            )
            continue
        raw_payloads["raw_polymarket_markets.jsonl"].append(_market_row(market))
        raw_payloads["raw_polymarket_orderbooks.jsonl"].extend(books)
        raw_payloads["raw_polymarket_trades.jsonl"].extend(_trade_rows_for_market(market))
        raw_payloads["raw_binance_btcusdt_klines.jsonl"].extend(candles)
        raw_payloads["raw_polymarket_resolutions.jsonl"].append(_resolution_row(market))
    for filename, rows in raw_payloads.items():
        raw_payloads[filename] = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))
    return raw_payloads, rejected


def _market_row(market: _MarketAccumulator) -> dict[str, Any]:
    return {
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "slug": market.slug,
        "market_family": market.market_family,
        "horizon_ms": market.horizon_ms,
        "market_start_ts": market.market_start_ts,
        "market_end_ts": market.market_end_ts,
        "settlement_ts": market.market_end_ts,
        "up_token_id": market.up_token_id,
        "down_token_id": market.down_token_id,
        "reference_price_source": "binance_btcusdt",
        "settlement_rule": market.settlement_rule,
        **safety_fields(),
    }


def _book_rows_for_market(
    *,
    config: LiveSignalCorpusConversionConfig,
    market: _MarketAccumulator,
) -> list[dict[str, Any]]:
    rows_by_ts: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in market.rows or []:
        decision_ts = int(row["decision_ts"])
        for outcome in ("UP", "DOWN"):
            book = _book_price(row, outcome=outcome, allow_proxy=config.allow_midpoint_price_proxy)
            if book is not None:
                rows_by_ts[decision_ts][outcome] = book
    rows: list[dict[str, Any]] = []
    for decision_ts, by_outcome in sorted(rows_by_ts.items()):
        if set(by_outcome) != {"UP", "DOWN"}:
            continue
        for outcome in ("UP", "DOWN"):
            book = by_outcome[outcome]
            rows.append(
                {
                    "market_id": market.market_id,
                    "token_id": market.up_token_id if outcome == "UP" else market.down_token_id,
                    "outcome": outcome,
                    "ts": decision_ts,
                    "available_at_ts": decision_ts,
                    "bid_price": book["bid"],
                    "ask_price": book["ask"],
                    "mid_price": (book["bid"] + book["ask"]) / 2.0,
                    "bid_size": book.get("bid_size", 0.0),
                    "ask_size": book.get("ask_size", 0.0),
                    "liquidity_depth": book.get("liquidity_depth", 0.0),
                    **safety_fields(),
                }
            )
    return rows


def _book_price(
    row: dict[str, Any],
    *,
    outcome: str,
    allow_proxy: bool,
) -> dict[str, float] | None:
    lower = outcome.lower()
    bid = _first_float(row, (f"{lower}_bid", f"{lower}_bid_price", "bid_price" if row.get("outcome") == outcome else ""))
    ask = _first_float(row, (f"{lower}_ask", f"{lower}_ask_price", "ask_price" if row.get("outcome") == outcome else ""))
    if bid is not None and ask is not None:
        if not 0.0 < bid <= ask <= 1.0:
            return None
        return {
            "bid": bid,
            "ask": ask,
            "bid_size": _first_float(row, (f"{lower}_bid_size", "bid_size")) or 0.0,
            "ask_size": _first_float(row, (f"{lower}_ask_size", "ask_size")) or 0.0,
            "liquidity_depth": _first_float(row, (f"{lower}_liquidity_depth", "liquidity_depth")) or 0.0,
        }
    if not allow_proxy:
        return None
    price = _first_float(row, (f"{lower}_mid", f"{lower}_price"))
    if price is None and row.get("outcome") == outcome:
        price = _first_float(row, ("polymarket_price", "market_implied_prob"))
    if price is None and row.get("outcome") != outcome:
        selected_price = _first_float(row, ("polymarket_price", "market_implied_prob"))
        price = None if selected_price is None else 1.0 - selected_price
    if price is None or not 0.02 <= price <= 0.98:
        return None
    return {
        "bid": max(0.001, price - 0.01),
        "ask": min(0.999, price + 0.01),
        "bid_size": 0.0,
        "ask_size": 0.0,
        "liquidity_depth": 0.0,
    }


def _trade_rows_for_market(market: _MarketAccumulator) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in market.rows or []:
        price = _first_float(row, ("trade_price",))
        if price is None:
            continue
        outcome = row["outcome"]
        rows.append(
            {
                "market_id": market.market_id,
                "token_id": market.up_token_id if outcome == "UP" else market.down_token_id,
                "outcome": outcome,
                "ts": row["decision_ts"],
                "available_at_ts": row["decision_ts"],
                "price": price,
                "size": _first_float(row, ("trade_size", "size")) or 0.0,
                "side": str(row.get("trade_side") or "unknown"),
                **safety_fields(),
            }
        )
    return rows


def _candle_rows_for_market(market: _MarketAccumulator) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for row in sorted(market.rows or [], key=lambda item: item["decision_ts"]):
        price = _first_float(row, ("btc_mid_price", "btc_reference_price", "reference_price"))
        if price is None:
            continue
        decision_ts = int(row["decision_ts"])
        candle_ts = decision_ts - 60_000
        if candle_ts in seen:
            continue
        seen.add(candle_ts)
        rows.append(
            {
                "ts": candle_ts,
                "available_at_ts": decision_ts,
                "open_price": price,
                "high_price": price,
                "low_price": price,
                "close_price": price,
                "volume": _first_float(row, ("btc_volume", "volume")) or 0.0,
                "timeframe_ms": 60_000,
                "source": "data_live_signal",
            }
        )
    if market.reference_price_start is not None and market.market_start_ts - 60_000 not in seen:
        rows.append(
            {
                "ts": market.market_start_ts - 60_000,
                "available_at_ts": market.market_start_ts,
                "open_price": market.reference_price_start,
                "high_price": market.reference_price_start,
                "low_price": market.reference_price_start,
                "close_price": market.reference_price_start,
                "volume": 0.0,
                "timeframe_ms": 60_000,
                "source": "data_live_signal_resolution_start",
            }
        )
    return rows


def _resolution_row(market: _MarketAccumulator) -> dict[str, Any]:
    return {
        "market_id": market.market_id,
        "reference_price_start": market.reference_price_start,
        "reference_price_end": market.reference_price_end,
        "resolution_status": market.resolution_status,
        **safety_fields(),
    }


def _write_raw_files(raw_dir: Path, raw_payloads: dict[str, list[dict[str, Any]]]) -> None:
    for filename in RAW_CORPUS_FILENAMES:
        _write_jsonl(raw_dir / filename, raw_payloads[filename])


def _conversion_report(
    *,
    config: LiveSignalCorpusConversionConfig,
    signal_files: tuple[Path, ...],
    raw_rows: list[dict[str, Any]],
    raw_payloads: dict[str, list[dict[str, Any]]],
    rejected_rows: list[dict[str, Any]],
    phase2_result: PolymarketCorpusBuildResult | None,
    phase2_error: str | None,
) -> dict[str, Any]:
    reject_counts = Counter()
    for row in rejected_rows:
        for reason in row.get("reject_reasons", []):
            reject_counts[reason] += 1
    market_count = len(raw_payloads["raw_polymarket_markets.jsonl"])
    training_eligible = phase2_result is not None and market_count > 0 and not phase2_error
    return {
        "schema_version": LIVE_TO_CORPUS_SCHEMA_VERSION,
        "phase": LIVE_TO_CORPUS_PHASE,
        "created_at": config.created_at,
        "input_file_count": len(signal_files),
        "input_row_count": len(raw_rows),
        "accepted_market_count": market_count,
        "accepted_orderbook_row_count": len(raw_payloads["raw_polymarket_orderbooks.jsonl"]),
        "accepted_trade_row_count": len(raw_payloads["raw_polymarket_trades.jsonl"]),
        "accepted_btc_candle_count": len(raw_payloads["raw_binance_btcusdt_klines.jsonl"]),
        "accepted_resolution_count": len(raw_payloads["raw_polymarket_resolutions.jsonl"]),
        "rejected_item_count": len(rejected_rows),
        "reject_reason_counts": dict(sorted(reject_counts.items())),
        "phase2_corpus_built": phase2_result is not None,
        "phase2_error": phase2_error,
        "training_eligible": training_eligible,
        "conversion_policy": {
            "requires_executable_bid_ask": not config.allow_midpoint_price_proxy,
            "allows_midpoint_price_proxy": config.allow_midpoint_price_proxy,
            "requires_verified_resolution": True,
            "uses_model_signal_as_label": False,
        },
        **safety_fields(),
    }


def _rejected_row(row: dict[str, Any], reasons: tuple[str, ...] | list[str]) -> dict[str, Any]:
    return {
        "source_file": row.get("_source_file"),
        "source_line": row.get("_source_line"),
        "round_slug": row.get("round_slug") or row.get("slug"),
        "event_id": row.get("event_id"),
        "reject_reasons": sorted(set(reasons)),
    }


def _optional_int(row: dict[str, Any], field_name: str) -> int | None:
    value = row.get(field_name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(row: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if not name:
            continue
        value = row.get(name)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return None


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
