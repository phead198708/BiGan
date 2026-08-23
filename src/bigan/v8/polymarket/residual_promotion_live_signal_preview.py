"""Outcome-blind, monitoring-only in-round signal preview sidecar."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus.builder import (
    _normalize_book_snapshots,
    _normalize_candles,
    _normalize_chainlink_prices,
    _normalize_markets,
    _normalize_trades,
)
from bigan.v8.polymarket.corpus.contracts import PolymarketCorpusBuildConfig
from bigan.v8.polymarket.corpus.features import build_polymarket_corpus_feature_rows
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.recorder.async_settlement import (
    _causal_chainlink_rows_for_markets,
)
from bigan.v8.polymarket.recorder.operator import _raw_market_row
from bigan.v8.polymarket.residual_promotion_v1 import CANDIDATE_ID, LINEAGE_ID

SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-outcome-blind-live-signal-preview-v1"
)
DECISION_OFFSETS_MS = (300_000, 600_000)
MARKET_HORIZON_MS = 900_000
ALLOWED_ACTIONS = frozenset({"NO_TRADE", "BUY_UP_HOLD", "BUY_DOWN_HOLD"})


class LiveSignalPreviewError(ValueError):
    """Fail-closed live preview error."""


def feature_rows_from_outcome_blind_snapshots(
    *,
    market_rows: Sequence[Mapping[str, Any]],
    orderbook_rows: Sequence[Mapping[str, Any]],
    trade_rows: Sequence[Mapping[str, Any]],
    candle_rows: Sequence[Mapping[str, Any]],
    chainlink_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the frozen decision-time feature rows without a target stream."""

    config = PolymarketCorpusBuildConfig(
        input_dir=Path("monitoring_input_never_read"),
        output_dir=Path("monitoring_output_never_written"),
        market_families=("btc_updown_15m",),
        sample_interval_seconds={"btc_updown_15m": 300},
        min_time_to_close_seconds=0,
        include_trade_labels=True,
        include_settlement_labels=False,
        overwrite_existing=False,
    )
    normalized_markets = _normalize_markets(
        [dict(row) for row in market_rows], config
    )
    books = _normalize_book_snapshots(
        [dict(row) for row in orderbook_rows], normalized_markets
    )
    trades = _normalize_trades(
        [dict(row) for row in trade_rows], normalized_markets
    )
    candles = _normalize_candles([dict(row) for row in candle_rows])
    chainlink = _normalize_chainlink_prices(
        [dict(row) for row in chainlink_rows]
    )
    rows = build_polymarket_corpus_feature_rows(
        markets=normalized_markets,
        book_snapshots=books,
        trades=trades,
        btc_candles=candles,
        chainlink_prices=chainlink,
        config=config,
    )
    return [row.to_dict() for row in rows]


def signal_from_feature_row(
    *,
    feature_row: Mapping[str, Any],
    runtime: Any,
    decision_number: int,
    already_accepted: bool,
) -> dict[str, Any]:
    """Score one causal row with the already-frozen candidate runtime."""

    if decision_number not in {1, 2}:
        raise LiveSignalPreviewError("decision number is invalid")
    decision_ts = _positive_int(feature_row.get("decision_ts"), "decision_ts")
    result = runtime.score_feature_row(
        feature_row,
        observed_at_ts=decision_ts,
    )
    if result.get("fail_closed") is True:
        reasons = "; ".join(map(str, result.get("fail_closed_reasons") or []))
        raise LiveSignalPreviewError(
            "frozen runtime failed closed" + (f": {reasons}" if reasons else "")
        )
    action = str(result.get("selected_action") or "")
    action_values = _action_values(result.get("action_values"))
    if action not in ALLOWED_ACTIONS:
        raise LiveSignalPreviewError("runtime selected an unsupported action")
    accepted = not already_accepted and action != "NO_TRADE"
    market_age_seconds = DECISION_OFFSETS_MS[decision_number - 1] // 1000
    return {
        "decision_number": decision_number,
        "decision_ts": decision_ts,
        "market_age_seconds": market_age_seconds,
        "time_to_close_seconds": 900 - market_age_seconds,
        "action_values": action_values,
        "selected_action": action,
        "accepted_at_this_decision": accepted,
        "provisional_until_ledger_close": True,
    }


def write_live_preview(
    *,
    output_path: Path | str,
    candidate_bundle_sha256: str,
    market: Mapping[str, Any],
    signals: Sequence[Mapping[str, Any]],
    generated_at: datetime | None = None,
    last_error: str | None = None,
) -> dict[str, Any]:
    """Atomically publish one sanitized in-progress round preview."""

    output = Path(output_path).expanduser().resolve()
    start_ts = _positive_int(market.get("market_start_ts"), "market_start_ts")
    end_ts = _positive_int(market.get("market_end_ts"), "market_end_ts")
    slug = str(market.get("slug") or "")
    if (
        end_ts - start_ts != MARKET_HORIZON_MS
        or slug != f"btc-updown-15m-{start_ts // 1000}"
        or market.get("market_family") != "btc_updown_15m"
    ):
        raise LiveSignalPreviewError("preview market identity is invalid")
    clean_signals = _validated_signals(
        signals,
        market_start_ts=start_ts,
        market_end_ts=end_ts,
    )
    accepted = next(
        (row for row in clean_signals if row["accepted_at_this_decision"]),
        None,
    )
    now = generated_at or datetime.now(UTC)
    state = (
        "awaiting_ledger_close"
        if len(clean_signals) == 2 or int(now.timestamp() * 1000) >= end_ts
        else "in_progress"
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "generated_at": now.isoformat(),
        "round_state": state,
        "slug": slug,
        "market_start_ts": start_ts,
        "market_end_ts": end_ts,
        "signals": clean_signals,
        "accepted_action": (
            accepted["selected_action"] if accepted is not None else "NO_TRADE"
        ),
        "accepted_decision_number": (
            accepted["decision_number"] if accepted is not None else None
        ),
        "last_error": last_error,
        "preview_is_provisional": True,
        "canonical_source_after_close": "outcome_blind_hash_chained_ledger",
        "monitoring_only": True,
        "monitoring_influences_collection": False,
        "monitoring_influences_model": False,
        "collection_state_mutated": False,
        "fresh_outcomes_accessed": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "paper_candidate_allowed": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    payload = {**identity, "content_sha256": canonical_json_sha256(identity)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def run_live_preview_monitor(
    *,
    output_path: Path | str,
    runtime: Any,
    candidate_bundle_sha256: str,
    provider_factory: Any,
    config_factory: Any,
    chainlink: Any,
    snapshot_lead_seconds: float = 3.0,
    market_retry_seconds: float = 2.0,
) -> None:
    """Continuously publish the 5m/10m sidecar without touching collection state."""

    maximum_source_age_seconds = float(runtime.maximum_source_age_ms) / 1000.0
    if not (
        0.0 < snapshot_lead_seconds < maximum_source_age_seconds
        and market_retry_seconds > 0.0
    ):
        raise LiveSignalPreviewError("live preview timing policy is invalid")
    while True:
        now_ms = int(time.time() * 1000)
        start_ts = (now_ms // MARKET_HORIZON_MS) * MARKET_HORIZON_MS
        if now_ms >= start_ts + DECISION_OFFSETS_MS[0] - int(
            snapshot_lead_seconds * 1000
        ):
            _sleep_until(start_ts + MARKET_HORIZON_MS)
            continue
        slug = f"btc-updown-15m-{start_ts // 1000}"
        provider = provider_factory(slug)
        config = config_factory(slug)
        markets = _discover_exact_market(
            provider=provider,
            config=config,
            slug=slug,
            deadline_ts=start_ts + DECISION_OFFSETS_MS[0] - 30_000,
            retry_seconds=market_retry_seconds,
        )
        market = markets[0]
        write_live_preview(
            output_path=output_path,
            candidate_bundle_sha256=candidate_bundle_sha256,
            market=market,
            signals=[],
        )
        book_rows: list[dict[str, Any]] = []
        signals: list[dict[str, Any]] = []
        for decision_number, offset_ms in enumerate(DECISION_OFFSETS_MS, start=1):
            decision_ts = start_ts + offset_ms
            _sleep_until(decision_ts - int(snapshot_lead_seconds * 1000))
            last_error: str | None = None
            try:
                current_books = provider.orderbook_rows(markets, config)
                book_rows.extend(dict(row) for row in current_books)
                trades = provider.trade_rows(markets, config)
                raw_market = _market_row_after_trade_collection(market)
                candles = provider.btc_feature_candle_rows(markets, config)
                causal_chainlink, _ = _causal_chainlink_rows_for_markets(
                    rows=chainlink.rows(),
                    markets=markets,
                )
                _sleep_until(decision_ts)
                feature_rows = feature_rows_from_outcome_blind_snapshots(
                    market_rows=[raw_market],
                    orderbook_rows=book_rows,
                    trade_rows=trades,
                    candle_rows=candles,
                    chainlink_rows=causal_chainlink,
                )
                matches = [
                    row
                    for row in feature_rows
                    if int(row.get("decision_ts") or 0) == decision_ts
                ]
                if len(matches) != 1:
                    raise LiveSignalPreviewError(
                        "exact causal decision row was unavailable"
                    )
                signals.append(
                    signal_from_feature_row(
                        feature_row=matches[0],
                        runtime=runtime,
                        decision_number=decision_number,
                        already_accepted=any(
                            row["accepted_at_this_decision"] for row in signals
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"{exc.__class__.__name__}: {exc}"
            report = write_live_preview(
                output_path=output_path,
                candidate_bundle_sha256=candidate_bundle_sha256,
                market=market,
                signals=signals,
                last_error=last_error,
            )
            print(
                json.dumps(
                    {
                        "slug": slug,
                        "round_state": report["round_state"],
                        "published_signal_count": len(report["signals"]),
                        "last_error": last_error,
                        "outcomes_accessed": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        _sleep_until(start_ts + MARKET_HORIZON_MS)


def _discover_exact_market(
    *,
    provider: Any,
    config: Any,
    slug: str,
    deadline_ts: int,
    retry_seconds: float,
) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    while int(time.time() * 1000) < deadline_ts:
        try:
            rows = [
                dict(row)
                for row in provider.market_rows(config)
                if row.get("slug") == slug
            ]
            if len(rows) == 1:
                return rows
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(retry_seconds)
    raise LiveSignalPreviewError("exact current market was unavailable") from last_error


def _market_row_after_trade_collection(
    market: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze market metadata only after the causal provider report is attached."""

    if not all(
        field in market
        for field in (
            "trade_collection_mode",
            "trade_api_collection_ts",
            "trade_api_request_failed",
            "trade_rest_rows_truncated",
        )
    ):
        raise LiveSignalPreviewError(
            "decision-time trade provider metadata is unavailable"
        )
    return _raw_market_row(dict(market))


def _sleep_until(timestamp_ms: int) -> None:
    while True:
        remaining = timestamp_ms / 1000.0 - time.time()
        if remaining <= 0.0:
            return
        time.sleep(min(1.0, remaining))


def _validated_signals(
    rows: Sequence[Mapping[str, Any]],
    *,
    market_start_ts: int,
    market_end_ts: int,
) -> list[dict[str, Any]]:
    if len(rows) > 2:
        raise LiveSignalPreviewError("preview has too many signals")
    clean: list[dict[str, Any]] = []
    already_accepted = False
    for index, row in enumerate(rows, start=1):
        decision_ts = _positive_int(row.get("decision_ts"), "decision_ts")
        action = str(row.get("selected_action") or "")
        expected_offset = DECISION_OFFSETS_MS[index - 1]
        expected_accepted = not already_accepted and action != "NO_TRADE"
        if not (
            int(row.get("decision_number") or 0) == index
            and decision_ts == market_start_ts + expected_offset
            and decision_ts < market_end_ts
            and action in ALLOWED_ACTIONS
            and row.get("accepted_at_this_decision") is expected_accepted
        ):
            raise LiveSignalPreviewError("preview signal contract is invalid")
        already_accepted = already_accepted or expected_accepted
        clean.append(
            {
                "decision_number": index,
                "decision_ts": decision_ts,
                "market_age_seconds": expected_offset // 1000,
                "time_to_close_seconds": (market_end_ts - decision_ts) // 1000,
                "action_values": _action_values(row.get("action_values")),
                "selected_action": action,
                "accepted_at_this_decision": expected_accepted,
                "provisional_until_ledger_close": True,
            }
        )
    return clean


def _action_values(value: Any) -> dict[str, float]:
    raw = dict(value or {})
    if set(raw) != ALLOWED_ACTIONS:
        raise LiveSignalPreviewError("action-value keys are invalid")
    result: dict[str, float] = {}
    for action in sorted(ALLOWED_ACTIONS):
        number = raw[action]
        if isinstance(number, bool) or not isinstance(number, Real):
            raise LiveSignalPreviewError("action value is not numeric")
        numeric = float(number)
        if not math.isfinite(numeric):
            raise LiveSignalPreviewError("action value is non-finite")
        result[action] = numeric
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise LiveSignalPreviewError(f"{name} is invalid")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise LiveSignalPreviewError(f"{name} is invalid") from exc
    if number <= 0:
        raise LiveSignalPreviewError(f"{name} is invalid")
    return number
