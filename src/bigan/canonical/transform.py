"""WS event payload -> canonical row dicts.

The transform layer is the only place that knows about Polymarket's wire
format. It produces dicts whose keys match the canonical schemas in
``schemas.py`` exactly; nothing downstream should peek at the original
``event_type``-shaped payload.

For multi-source support (#22 / #24), additional ``transform_*`` factories
will be added (e.g. ``transform_coinbase_ticker``) and dispatched on the
``source`` argument by the ETL runner.
"""

from __future__ import annotations

from typing import Any

from .schemas import PROVENANCE_WS
from .symbols import symbol_mapping_row

# Source identifier for Polymarket payloads. Multi-source extension (#24)
# would add ``coinbase``, ``binance``, etc.
SOURCE_POLYMARKET = "polymarket"


def _as_int_ms(v: Any) -> int | None:
    """Coerce stringified ms-epoch timestamps into ints; return None on failure."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first_int_ms(*values: Any) -> int | None:
    for value in values:
        ts = _as_int_ms(value)
        if ts is not None:
            return ts
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _identity_columns(
    *,
    raw: dict[str, Any],
    ingest_ts: int,
    source_timestamp_ms: int | None = None,
    capture_timestamp_ms: int | None = None,
    source_channel: str | None = None,
    fallback_ts: int | None = None,
    record_provenance: str | None = None,
) -> dict[str, Any] | None:
    """Build the shared identity columns. Returns ``None`` if essential
    fields (asset_id / timestamp) are absent — caller should skip the row.

    Timestamps and provenance prefer the surrounding NDJSON record metadata.
    ``raw["timestamp"]`` / ``raw["provenance"]`` remain fallbacks for legacy
    archives written before the outer contract existed.

    ``provenance`` is sourced (in priority order):
      1. ``record_provenance`` — caller-supplied default for the surrounding
         NDJSON record (the runner sets this to ``"ws"`` for WS messages).
      2. ``raw["provenance"]`` — carried inside older synthesized records.
      3. :data:`PROVENANCE_WS` — final fallback so provenance is never NULL
         for rows produced by the live pipeline.
    """
    asset_id = raw.get("asset_id")
    if asset_id is None:
        return None
    ts = _first_int_ms(
        source_timestamp_ms,
        raw.get("source_timestamp_ms"),
        raw.get("timestamp"),
        fallback_ts,
    )
    if ts is None:
        return None
    provenance = record_provenance or raw.get("provenance") or PROVENANCE_WS
    return {
        "ts": int(ts),
        "message_ts": int(ts),
        "ingest_ts": int(ingest_ts),
        "capture_timestamp_ms": int(
            capture_timestamp_ms if capture_timestamp_ms is not None else ingest_ts
        ),
        "source": SOURCE_POLYMARKET,
        "source_symbol": str(asset_id),
        "source_market": str(raw["market"]) if raw.get("market") is not None else None,
        # The live runner annotates Gamma-discovered outcome tokens before
        # archiving. Unknown or historical rows can still be enriched later
        # by the #22 symbol mapping path.
        "canonical_symbol": _optional_str(raw.get("canonical_symbol")),
        "source_channel": source_channel or _optional_str(raw.get("source_channel")),
        "provenance": str(provenance),
    }


# ---------------------------------------------------------------------------
# best_bid_ask -> raw_top_of_book
# ---------------------------------------------------------------------------


def transform_top_of_book_event(
    raw: dict[str, Any],
    *,
    ingest_ts: int,
    source_timestamp_ms: int | None = None,
    capture_timestamp_ms: int | None = None,
    source_channel: str | None = None,
    record_provenance: str | None = None,
) -> dict[str, Any] | None:
    """Map one ``best_bid_ask`` event to a single ``raw_top_of_book`` row."""
    if raw.get("event_type") != "best_bid_ask":
        return None
    identity = _identity_columns(
        raw=raw,
        ingest_ts=ingest_ts,
        source_timestamp_ms=source_timestamp_ms,
        capture_timestamp_ms=capture_timestamp_ms,
        source_channel=source_channel,
        record_provenance=record_provenance,
    )
    if identity is None:
        return None
    bid = _as_float(raw.get("best_bid"))
    ask = _as_float(raw.get("best_ask"))
    spread = _as_float(raw.get("spread"))
    if spread is None and bid is not None and ask is not None:
        # Some payloads omit ``spread``; recompute defensively.
        spread = ask - bid
    return {
        **identity,
        "bid_price": bid,
        "ask_price": ask,
        "spread": spread,
    }


def derive_top_of_book_from_book(
    raw: dict[str, Any],
    *,
    ingest_ts: int,
    source_timestamp_ms: int | None = None,
    capture_timestamp_ms: int | None = None,
    source_channel: str | None = None,
    record_provenance: str | None = None,
) -> dict[str, Any] | None:
    """Derive a ``raw_top_of_book`` row from a ``book`` snapshot.

    Polymarket's market channel does not always emit standalone
    ``best_bid_ask`` events; for many markets the only top-of-book
    information arrives implicitly via book / price_change. We project
    the BBO out of the snapshot here so downstream features have a
    dense top_of_book stream regardless of which event types the source
    chose to emit.
    """
    if raw.get("event_type") != "book":
        return None
    identity = _identity_columns(
        raw=raw,
        ingest_ts=ingest_ts,
        source_timestamp_ms=source_timestamp_ms,
        capture_timestamp_ms=capture_timestamp_ms,
        source_channel=source_channel,
        record_provenance=record_provenance,
    )
    if identity is None:
        return None
    bids = _coerce_levels(raw.get("bids"))
    asks = _coerce_levels(raw.get("ask") if "ask" in raw else raw.get("asks"))
    if not bids and not asks:
        return None
    best_bid = max(bids, key=lambda lvl: lvl[0])[0] if bids else None
    best_ask = min(asks, key=lambda lvl: lvl[0])[0] if asks else None
    spread = (
        best_ask - best_bid
        if best_bid is not None and best_ask is not None
        else None
    )
    return {
        **identity,
        "bid_price": best_bid,
        "ask_price": best_ask,
        "spread": spread,
    }


def derive_top_of_book_from_price_change(
    raw: dict[str, Any],
    *,
    ingest_ts: int,
    source_timestamp_ms: int | None = None,
    capture_timestamp_ms: int | None = None,
    source_channel: str | None = None,
    record_provenance: str | None = None,
) -> list[dict[str, Any]]:
    """Derive ``raw_top_of_book`` rows from a ``price_change`` event.

    Polymarket's ``price_change`` payload carries ``best_bid`` /
    ``best_ask`` on each entry inside ``price_changes[]`` so we can
    extract a top-of-book row per asset without replaying the full
    book delta.
    """
    if raw.get("event_type") != "price_change":
        return []
    market = raw.get("market")
    ts = _first_int_ms(
        source_timestamp_ms,
        raw.get("source_timestamp_ms"),
        raw.get("timestamp"),
    )
    if ts is None:
        return []
    provenance = record_provenance or raw.get("provenance") or PROVENANCE_WS
    resolved_source_channel = source_channel or _optional_str(raw.get("source_channel"))
    rows: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}  # last best/ask per asset within event
    for entry in raw.get("price_changes") or []:
        if not isinstance(entry, dict):
            continue
        asset_id = entry.get("asset_id")
        if asset_id is None:
            continue
        bid = _as_float(entry.get("best_bid"))
        ask = _as_float(entry.get("best_ask"))
        if bid is None and ask is None:
            continue
        # Multiple price_changes for the same asset in one event → keep last.
        seen[str(asset_id)] = {
            "ts": ts,
            "message_ts": ts,
            "ingest_ts": int(ingest_ts),
            "capture_timestamp_ms": int(
                capture_timestamp_ms if capture_timestamp_ms is not None else ingest_ts
            ),
            "source": SOURCE_POLYMARKET,
            "source_symbol": str(asset_id),
            "source_market": str(market) if market is not None else None,
            "canonical_symbol": _optional_str(entry.get("canonical_symbol")),
            "source_channel": resolved_source_channel,
            "provenance": str(provenance),
            "bid_price": bid,
            "ask_price": ask,
            "spread": (
                ask - bid if bid is not None and ask is not None else None
            ),
        }
    rows.extend(seen.values())
    return rows


# ---------------------------------------------------------------------------
# book -> raw_orderbook_snapshot (one row per (side, level))
# ---------------------------------------------------------------------------


def transform_book_event(
    raw: dict[str, Any],
    *,
    ingest_ts: int,
    source_timestamp_ms: int | None = None,
    capture_timestamp_ms: int | None = None,
    source_channel: str | None = None,
    record_provenance: str | None = None,
) -> list[dict[str, Any]]:
    """Map one ``book`` snapshot event to N + M canonical rows.

    Bids and asks are sorted by their canonical ordering before assigning
    ``level``: bids by descending price (best bid first, level=0), asks by
    ascending price. This makes ``WHERE level=0`` always return the inside
    market regardless of how the source ordered the payload.
    """
    if raw.get("event_type") != "book":
        return []
    identity = _identity_columns(
        raw=raw,
        ingest_ts=ingest_ts,
        source_timestamp_ms=source_timestamp_ms,
        capture_timestamp_ms=capture_timestamp_ms,
        source_channel=source_channel,
        record_provenance=record_provenance,
    )
    if identity is None:
        return []
    snapshot_hash = raw.get("hash")

    rows: list[dict[str, Any]] = []
    bids = _coerce_levels(raw.get("bids"))
    asks = _coerce_levels(raw.get("ask") if "ask" in raw else raw.get("asks"))

    bids.sort(key=lambda lvl: -lvl[0])  # descending by price
    asks.sort(key=lambda lvl: lvl[0])  # ascending by price

    for level, (price, size) in enumerate(bids):
        rows.append(
            {
                **identity,
                "side": "BID",
                "level": level,
                "price": price,
                "size": size,
                "snapshot_hash": str(snapshot_hash) if snapshot_hash is not None else None,
            }
        )
    for level, (price, size) in enumerate(asks):
        rows.append(
            {
                **identity,
                "side": "ASK",
                "level": level,
                "price": price,
                "size": size,
                "snapshot_hash": str(snapshot_hash) if snapshot_hash is not None else None,
            }
        )
    return rows


def _coerce_levels(raw_levels: Any) -> list[tuple[float, float]]:
    if not isinstance(raw_levels, list):
        return []
    out: list[tuple[float, float]] = []
    for lvl in raw_levels:
        if not isinstance(lvl, dict):
            continue
        price = _as_float(lvl.get("price"))
        size = _as_float(lvl.get("size"))
        if price is None or size is None:
            continue
        out.append((price, size))
    return out


# ---------------------------------------------------------------------------
# last_trade_price -> raw_trades
# ---------------------------------------------------------------------------


def transform_last_trade_price_event(
    raw: dict[str, Any],
    *,
    ingest_ts: int,
    source_timestamp_ms: int | None = None,
    capture_timestamp_ms: int | None = None,
    source_channel: str | None = None,
    record_provenance: str | None = None,
) -> dict[str, Any] | None:
    """Map one ``last_trade_price`` event to a single ``raw_trades`` row."""
    if raw.get("event_type") != "last_trade_price":
        return None
    identity = _identity_columns(
        raw=raw,
        ingest_ts=ingest_ts,
        source_timestamp_ms=source_timestamp_ms,
        capture_timestamp_ms=capture_timestamp_ms,
        source_channel=source_channel,
        record_provenance=record_provenance,
    )
    if identity is None:
        return None
    price = _as_float(raw.get("price"))
    size = _as_float(raw.get("size"))
    side = raw.get("side")
    if price is None or size is None or side not in ("BUY", "SELL"):
        return None
    fee_rate = _as_float(raw.get("fee_rate_bps"))
    trade_id = (
        f"{SOURCE_POLYMARKET}-{identity['source_symbol']}-{identity['ts']}-"
        f"{price}-{size}-{side}"
    )
    return {
        **identity,
        "price": price,
        "size": size,
        "side": side,
        "fee_rate_bps": fee_rate,
        "trade_id": trade_id,
    }


# ---------------------------------------------------------------------------
# symbol_mapping -> symbol_mapping
# ---------------------------------------------------------------------------


def transform_symbol_mapping_event(
    raw: dict[str, Any],
    *,
    ingest_ts: int,
    source_timestamp_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Map Gamma-discovered token metadata to symbol_mapping rows."""

    if raw.get("event_type") != "symbol_mapping":
        return []
    mappings = raw.get("mappings")
    if not isinstance(mappings, list):
        return []
    fallback_ts = source_timestamp_ms if source_timestamp_ms is not None else ingest_ts
    rows: list[dict[str, Any]] = []
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        try:
            rows.append(
                symbol_mapping_row(
                    source=mapping["source"],
                    source_symbol=mapping["source_symbol"],
                    source_market=mapping.get("source_market"),
                    canonical_symbol=mapping["canonical_symbol"],
                    effective_from_ts=int(
                        mapping.get("effective_from_ts")
                        if mapping.get("effective_from_ts") is not None
                        else fallback_ts
                    ),
                    effective_to_ts=mapping.get("effective_to_ts"),
                    ingest_ts=int(mapping.get("ingest_ts") or ingest_ts),
                    message_ts=int(mapping.get("message_ts") or fallback_ts),
                    symbol_kind=mapping.get("symbol_kind"),
                    metadata_json=mapping.get("metadata_json"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return rows


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------


def transform_event(
    record: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Dispatch a single NDJSON record to the
    appropriate canonical table(s).

    Returns a dict ``{table_name: [rows...]}``. Tables that don't apply for
    this event type are absent. Empty rows lists never appear.

    Records of types we don't currently materialise into canonical tables
    (``price_change``, ``tick_size_change``, ``new_market``, ...) yield an
    empty dict.
    """
    raw = record.get("raw")
    if not isinstance(raw, dict):
        return {}
    capture_timestamp_ms = _first_int_ms(
        record.get("capture_timestamp_ms"),
        record.get("receive_time"),
    )
    if capture_timestamp_ms is None:
        return {}
    ingest_ts = capture_timestamp_ms
    source_timestamp_ms = _first_int_ms(record.get("source_timestamp_ms"))
    source_channel = _optional_str(record.get("source_channel"))
    record_provenance = _optional_str(record.get("provenance"))
    event_type = raw.get("event_type")

    if event_type == "best_bid_ask":
        row = transform_top_of_book_event(
            raw,
            ingest_ts=ingest_ts,
            source_timestamp_ms=source_timestamp_ms,
            capture_timestamp_ms=capture_timestamp_ms,
            source_channel=source_channel,
            record_provenance=record_provenance,
        )
        return {"raw_top_of_book": [row]} if row is not None else {}

    if event_type == "book":
        out: dict[str, list[dict[str, Any]]] = {}
        snapshot_rows = transform_book_event(
            raw,
            ingest_ts=ingest_ts,
            source_timestamp_ms=source_timestamp_ms,
            capture_timestamp_ms=capture_timestamp_ms,
            source_channel=source_channel,
            record_provenance=record_provenance,
        )
        if snapshot_rows:
            out["raw_orderbook_snapshot"] = snapshot_rows
        bbo_row = derive_top_of_book_from_book(
            raw,
            ingest_ts=ingest_ts,
            source_timestamp_ms=source_timestamp_ms,
            capture_timestamp_ms=capture_timestamp_ms,
            source_channel=source_channel,
            record_provenance=record_provenance,
        )
        if bbo_row is not None:
            out["raw_top_of_book"] = [bbo_row]
        return out

    if event_type == "price_change":
        bbo_rows = derive_top_of_book_from_price_change(
            raw,
            ingest_ts=ingest_ts,
            source_timestamp_ms=source_timestamp_ms,
            capture_timestamp_ms=capture_timestamp_ms,
            source_channel=source_channel,
            record_provenance=record_provenance,
        )
        return {"raw_top_of_book": bbo_rows} if bbo_rows else {}

    if event_type == "last_trade_price":
        row = transform_last_trade_price_event(
            raw,
            ingest_ts=ingest_ts,
            source_timestamp_ms=source_timestamp_ms,
            capture_timestamp_ms=capture_timestamp_ms,
            source_channel=source_channel,
            record_provenance=record_provenance,
        )
        return {"raw_trades": [row]} if row is not None else {}

    if event_type == "symbol_mapping":
        rows = transform_symbol_mapping_event(
            raw,
            ingest_ts=ingest_ts,
            source_timestamp_ms=source_timestamp_ms,
        )
        return {"symbol_mapping": rows} if rows else {}

    return {}
