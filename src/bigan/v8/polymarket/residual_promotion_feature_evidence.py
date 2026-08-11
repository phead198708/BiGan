"""Provider-byte proof for BTC-15M residual execution feature rows."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

PROVIDER_FEATURE_FILENAMES = (
    "raw_polymarket_markets.jsonl",
    "raw_polymarket_orderbooks.jsonl",
    "raw_polymarket_trades.jsonl",
    "raw_binance_btcusdt_klines.jsonl",
    "raw_polymarket_chainlink_prices.jsonl",
)
_REQUIRED_NONEMPTY_FILES = frozenset(
    {
        "raw_polymarket_markets.jsonl",
        "raw_polymarket_orderbooks.jsonl",
        "raw_binance_btcusdt_klines.jsonl",
    }
)
_FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "official_settlement",
        "payout",
        "payout_down",
        "payout_per_token",
        "payout_up",
        "raw_resolution_text",
        "resolution_status",
        "resolved_outcome",
        "settlement_price",
        "winner",
        "winning_outcome",
        "winning_token_id",
    }
)
_FORBIDDEN_RESULT_KEY_TOKENS = ("label", "pnl", "profit")
_LOCKED_FALSE_KEYS = frozenset(
    {
        "broker_exchange_write_enabled",
        "capital_at_risk",
        "live_exchange_write_enabled",
        "live_trading_allowed",
        "polymarket_write_allowed",
        "polymarket_write_enabled",
        "wallet_signing_allowed",
        "wallet_signing_enabled",
    }
)


class ProviderFeatureEvidenceError(ValueError):
    """Fail-closed provider evidence validation error."""


@dataclass(frozen=True, slots=True)
class VerifiedProviderFeatureEvidence:
    """Exact raw streams and the feature row deterministically rebuilt from them."""

    file_sha256_items: tuple[tuple[str, str], ...]
    raw_jsonl_items: tuple[tuple[str, str], ...]
    reconstructed_feature_row_sha256: str
    evidence_graph_sha256: str

    @property
    def file_sha256(self) -> dict[str, str]:
        return dict(self.file_sha256_items)

    @property
    def raw_jsonl(self) -> dict[str, str]:
        return dict(self.raw_jsonl_items)


def build_provider_bound_feature_rows(
    raw_evidence: Mapping[str, bytes],
) -> tuple[dict[str, Any], ...]:
    """Rebuild causal feature rows from exactly five strict raw provider streams."""

    rows, _, _ = _decode_provider_evidence(raw_evidence)
    return _reconstruct_feature_rows(rows)


def verify_provider_feature_evidence(
    *,
    raw_evidence: Mapping[str, bytes],
    signal: Mapping[str, Any],
    feature_row: Mapping[str, Any],
) -> VerifiedProviderFeatureEvidence:
    """Bind one submitted feature row to exact provider bytes and frozen semantics."""

    rows, raw_jsonl, file_sha256 = _decode_provider_evidence(raw_evidence)
    feature_rows = _reconstruct_feature_rows(rows)
    market_id = signal.get("market_id")
    decision_ts = signal.get("decision_ts_ms")
    matches = [
        row
        for row in feature_rows
        if row.get("market_id") == market_id and row.get("decision_ts") == decision_ts
    ]
    if len(matches) != 1:
        raise ProviderFeatureEvidenceError(
            "provider evidence does not reconstruct exactly one decision feature row"
        )
    reconstructed = matches[0]
    submitted = dict(feature_row)
    if canonical_json_sha256(reconstructed) != canonical_json_sha256(submitted):
        raise ProviderFeatureEvidenceError(
            "provider-reconstructed feature row does not match submitted feature bytes"
        )
    market_rows = rows[PROVIDER_FEATURE_FILENAMES[0]]
    if len(market_rows) != 1:
        raise ProviderFeatureEvidenceError(
            "provider evidence must contain exactly one BTC-15M market"
        )
    market = market_rows[0]
    slug = str(signal.get("slug") or "")
    try:
        start_ts = int(slug.rsplit("-", maxsplit=1)[1]) * 1_000
    except (IndexError, ValueError) as exc:
        raise ProviderFeatureEvidenceError("provider signal slug is invalid") from exc
    if not (
        market.get("market_id") == market_id
        and market.get("condition_id") == market_id
        and market.get("slug") == slug
        and market.get("market_family") == "btc_updown_15m"
        and market.get("market_start_ts") == start_ts
        and market.get("market_end_ts") == start_ts + 900_000
        and market.get("up_token_id") == signal.get("up_token_id")
        and market.get("down_token_id") == signal.get("down_token_id")
    ):
        raise ProviderFeatureEvidenceError(
            "provider market identity does not match executable signal"
        )
    reconstructed_sha256 = canonical_json_sha256(reconstructed)
    graph = {
        "market_id": market_id,
        "decision_ts_ms": decision_ts,
        "file_sha256": file_sha256,
        "reconstructed_feature_row_sha256": reconstructed_sha256,
    }
    return VerifiedProviderFeatureEvidence(
        file_sha256_items=tuple(sorted(file_sha256.items())),
        raw_jsonl_items=tuple((name, raw_jsonl[name]) for name in PROVIDER_FEATURE_FILENAMES),
        reconstructed_feature_row_sha256=reconstructed_sha256,
        evidence_graph_sha256=canonical_json_sha256(graph),
    )


def _decode_provider_evidence(
    raw_evidence: Mapping[str, bytes],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], dict[str, str]]:
    if not isinstance(raw_evidence, Mapping) or set(raw_evidence) != set(
        PROVIDER_FEATURE_FILENAMES
    ):
        raise ProviderFeatureEvidenceError(
            "provider feature evidence schema is not exact"
        )
    rows: dict[str, list[dict[str, Any]]] = {}
    raw_jsonl: dict[str, str] = {}
    file_sha256: dict[str, str] = {}
    for name in PROVIDER_FEATURE_FILENAMES:
        raw = raw_evidence.get(name)
        if not isinstance(raw, bytes):
            raise ProviderFeatureEvidenceError(
                f"provider feature evidence is not raw bytes: {name}"
            )
        decoded_rows, text = _strict_jsonl(raw, name)
        if name in _REQUIRED_NONEMPTY_FILES and not decoded_rows:
            raise ProviderFeatureEvidenceError(
                f"required provider feature evidence is empty: {name}"
            )
        for row in decoded_rows:
            _assert_outcome_blind_provider_row(row, path=name)
        rows[name] = decoded_rows
        raw_jsonl[name] = text
        file_sha256[name] = hashlib.sha256(raw).hexdigest()
    return rows, raw_jsonl, file_sha256


def _reconstruct_feature_rows(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    config = PolymarketCorpusBuildConfig(
        input_dir=Path(__file__).resolve().parent,
        output_dir=Path(__file__).resolve().parent,
        market_families=("btc_updown_15m",),
        sample_interval_seconds={"btc_updown_15m": 300},
        min_time_to_close_seconds=0,
        include_trade_labels=True,
        include_settlement_labels=False,
        overwrite_existing=False,
    )
    try:
        markets = _normalize_markets(
            [dict(row) for row in rows[PROVIDER_FEATURE_FILENAMES[0]]],
            config,
        )
        if len(markets) != 1:
            raise ProviderFeatureEvidenceError(
                "provider feature evidence market population is not exactly one"
            )
        books = _normalize_book_snapshots(
            [dict(row) for row in rows[PROVIDER_FEATURE_FILENAMES[1]]],
            markets,
        )
        trades = _normalize_trades(
            [dict(row) for row in rows[PROVIDER_FEATURE_FILENAMES[2]]],
            markets,
        )
        candles = _normalize_candles(
            [dict(row) for row in rows[PROVIDER_FEATURE_FILENAMES[3]]]
        )
        chainlink = _normalize_chainlink_prices(
            [dict(row) for row in rows[PROVIDER_FEATURE_FILENAMES[4]]]
        )
        feature_rows = build_polymarket_corpus_feature_rows(
            markets=markets,
            book_snapshots=books,
            trades=trades,
            btc_candles=candles,
            chainlink_prices=chainlink,
            config=config,
        )
    except ProviderFeatureEvidenceError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderFeatureEvidenceError(
            "provider feature evidence cannot reconstruct causal features"
        ) from exc
    return tuple(row.to_dict() for row in feature_rows)


def _strict_jsonl(raw: bytes, label: str) -> tuple[list[dict[str, Any]], str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProviderFeatureEvidenceError(
            f"provider feature evidence is not UTF-8: {label}"
        ) from exc
    if not raw:
        return [], text
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ProviderFeatureEvidenceError(
            f"provider feature evidence contains an empty JSONL row: {label}"
        )
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            )
            _validate_finite_json_tree(value)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderFeatureEvidenceError(
                f"provider feature evidence is not strict JSONL: {label}"
            ) from exc
        if not isinstance(value, dict):
            raise ProviderFeatureEvidenceError(
                f"provider feature evidence row is not an object: {label}"
            )
        rows.append(value)
    return rows, text


def _assert_outcome_blind_provider_row(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProviderFeatureEvidenceError(
                    "provider feature evidence key is not a string"
                )
            lowered = key.lower()
            if lowered in _FORBIDDEN_RESULT_KEYS or any(
                token in lowered for token in _FORBIDDEN_RESULT_KEY_TOKENS
            ):
                raise ProviderFeatureEvidenceError(
                    f"provider feature evidence contains result-bearing field: {path}.{key}"
                )
            if lowered in _LOCKED_FALSE_KEYS and child is not False:
                raise ProviderFeatureEvidenceError(
                    f"provider feature evidence safety field is not false: {path}.{key}"
                )
            if (
                lowered in {"outcomes_accessed", "settlement_accessed", "pnl_accessed"}
                and child is not False
            ):
                raise ProviderFeatureEvidenceError(
                    f"provider feature evidence accessed a forbidden stream: {path}.{key}"
                )
            _assert_outcome_blind_provider_row(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_outcome_blind_provider_row(child, path=f"{path}[{index}]")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _validate_finite_json_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("decoded JSON contains a non-finite number")
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_finite_json_tree(child)
    elif isinstance(value, list):
        for child in value:
            _validate_finite_json_tree(child)


__all__ = [
    "PROVIDER_FEATURE_FILENAMES",
    "ProviderFeatureEvidenceError",
    "VerifiedProviderFeatureEvidence",
    "build_provider_bound_feature_rows",
    "verify_provider_feature_evidence",
]
