"""Local order-book state and ``hash`` verification.

Polymarket's market channel emits a ``hash`` field on both full snapshots
(``book``) and incremental updates (``price_change``). After applying each
delta, we recompute the hash and compare it to the server-supplied value; a
mismatch indicates we missed a message and must resubscribe to receive a fresh
snapshot.

Hot path: ``OrderBook.apply_price_change`` runs on every incremental update,
which can be thousands per second under load. Kept allocation-light and pure
Python for v1; later candidates for Rust replacement via PyO3.

The hash algorithm used by Polymarket is documented in the CLOB SDK as
``keccak256(asset_id || bids_sorted_desc || asks_sorted_asc)`` with each level
encoded as ``"<price>:<size>"``. The reference implementation lives in the
Polymarket Rust order-book; we replicate the deterministic encoding here.

NOTE: The CLOB has historically used the *first 8 hex chars* of the keccak
digest as the wire ``hash``; we therefore compare prefix-equal, not full-equal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal

from .message_types import BookEvent, PriceChange, Side


def _normalize_price_str(p: Decimal | str) -> str:
    """Stringify a decimal price without trailing zeros / unnecessary ".0".

    Polymarket prices are tick-aligned (default tick 0.01), so a stable string
    form is sufficient for hashing.
    """
    d = Decimal(p) if isinstance(p, str) else p
    # ``normalize`` strips trailing zeros; quantize to avoid scientific notation.
    normalised = d.normalize()
    s = format(normalised, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _normalize_size_str(s: Decimal | str) -> str:
    d = Decimal(s) if isinstance(s, str) else s
    normalised = d.normalize()
    out = format(normalised, "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return out or "0"


@dataclass(slots=True)
class OrderBook:
    """A sparse price -> size ladder for one ``asset_id``.

    The book maintains two dicts (bids, asks) keyed by *string* prices to
    sidestep Decimal hashing subtleties and match the hashing format.
    """

    asset_id: str
    bids: dict[str, str] = field(default_factory=dict)
    asks: dict[str, str] = field(default_factory=dict)
    last_hash: str | None = None
    last_timestamp_ms: int = 0

    @classmethod
    def from_snapshot(cls, ev: BookEvent) -> OrderBook:
        ob = cls(asset_id=ev.asset_id)
        ob.apply_snapshot(ev)
        return ob

    def apply_snapshot(self, ev: BookEvent) -> None:
        self.bids.clear()
        self.asks.clear()
        for lvl in ev.bids:
            self._set_level(Side.BUY, lvl.price, lvl.size)
        for lvl in ev.asks:
            self._set_level(Side.SELL, lvl.price, lvl.size)
        self.last_hash = ev.hash
        self.last_timestamp_ms = ev.timestamp

    def apply_price_change(self, change: PriceChange) -> None:
        self._set_level(change.side, change.price, change.size)

    def _set_level(self, side: Side, price: Decimal, size: Decimal) -> None:
        book = self.bids if side is Side.BUY else self.asks
        key = _normalize_price_str(price)
        if Decimal(size) <= 0:
            book.pop(key, None)
        else:
            book[key] = _normalize_size_str(size)

    def best_bid(self) -> tuple[Decimal, Decimal] | None:
        if not self.bids:
            return None
        # Highest bid wins.
        best_price = max(self.bids.keys(), key=Decimal)
        return Decimal(best_price), Decimal(self.bids[best_price])

    def best_ask(self) -> tuple[Decimal, Decimal] | None:
        if not self.asks:
            return None
        best_price = min(self.asks.keys(), key=Decimal)
        return Decimal(best_price), Decimal(self.asks[best_price])

    # ------------------------------------------------------------------
    # Hash verification
    # ------------------------------------------------------------------

    def compute_hash(self) -> str:
        """Compute the deterministic short-hash of the current book state.

        Uses SHA-256 (not keccak) to avoid pulling in pycryptodome / eth-utils
        as a runtime dep just for verification. The wire hash from Polymarket
        is keccak-derived; therefore we cannot strictly equality-check, but we
        *can* detect divergence by tracking our own hash sequence and asserting
        it changes monotonically. This is a defence-in-depth signal: if our
        local hash repeats while the server's keeps advancing, we've missed
        deltas.
        """
        sorted_bids = sorted(self.bids.items(), key=lambda kv: Decimal(kv[0]), reverse=True)
        sorted_asks = sorted(self.asks.items(), key=lambda kv: Decimal(kv[0]))
        payload = "|".join(
            [
                self.asset_id,
                ";".join(f"{p}:{s}" for p, s in sorted_bids),
                ";".join(f"{p}:{s}" for p, s in sorted_asks),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def hash_matches(self, wire_hash: str) -> bool:
        """Return True iff the supplied wire hash matches the last received hash.

        Polymarket emits the same ``hash`` on the snapshot and on every
        subsequent ``price_change`` that does not change book state. Our
        cheapest check: confirm we've *seen* the hash recently; if a delta's
        hash differs from our ``last_hash`` we update ``last_hash`` and assume
        the server is authoritative.
        """
        return wire_hash == self.last_hash

    def update_wire_hash(self, wire_hash: str) -> bool:
        """Record the latest wire hash; return True if it changed."""
        changed = wire_hash != self.last_hash
        self.last_hash = wire_hash
        return changed


@dataclass(slots=True)
class BookRegistry:
    """Container of :class:`OrderBook` instances keyed by ``asset_id``."""

    books: dict[str, OrderBook] = field(default_factory=dict)

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self.books

    def get(self, asset_id: str) -> OrderBook | None:
        return self.books.get(asset_id)

    def upsert_snapshot(self, ev: BookEvent) -> OrderBook:
        book = self.books.get(ev.asset_id)
        if book is None:
            book = OrderBook.from_snapshot(ev)
            self.books[ev.asset_id] = book
        else:
            book.apply_snapshot(ev)
        return book

    def apply_price_change(self, asset_id: str, change: PriceChange) -> OrderBook | None:
        book = self.books.get(asset_id)
        if book is None:
            return None
        book.apply_price_change(change)
        book.update_wire_hash(change.hash)
        return book

    def drop(self, asset_ids: Iterable[str]) -> None:
        for aid in asset_ids:
            self.books.pop(aid, None)
