"""Causal read-only cache for pre-fetched Polymarket market identity."""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus.contracts import safety_fields

GAMMA_MARKET_IDENTITY_CACHE_SCHEMA_VERSION = (
    "bigan-v8-gamma-market-identity-cache-v1"
)
GAMMA_MARKET_IDENTITY_CACHE_SOURCE_TYPE = "gamma_prefetch_cache"
GAMMA_MARKET_IDENTITY_CACHE_FALLBACK_SOURCE_TYPE = (
    "gamma_prefetch_cache_fallback"
)
GAMMA_MARKET_IDENTITY_ALLOWED_PAYLOAD_FIELDS = (
    "conditionId",
    "condition_id",
    "slug",
    "market_slug",
    "question",
    "description",
    "resolutionSource",
    "priceToBeat",
    "referencePriceStart",
    "outcomes",
    "clobTokenIds",
    "endDate",
    "end_date_iso",
)


class GammaMarketIdentityCacheError(RuntimeError):
    """Raised when a cached market identity cannot be trusted."""

    def __init__(self, message: str, *, reason_codes: tuple[str, ...]) -> None:
        super().__init__(message)
        self.reason_codes = reason_codes


class GammaMarketIdentityCache:
    """Atomic cache whose entries are immutable after their first write."""

    def __init__(
        self,
        path: Path | str,
        *,
        max_age_seconds: float,
        current_time_ms: int | None = None,
    ) -> None:
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.max_age_ms = int(max_age_seconds * 1000)
        self.current_time_ms = current_time_ms

    def store_prefetched_payload(
        self,
        *,
        payload: dict[str, Any],
        slug: str,
        market_family: str,
        market_start_ts: int,
        market_end_ts: int,
        condition_id: str,
        up_token_id: str,
        down_token_id: str,
        reference_price_source: str,
        settlement_rule: str,
        fetched_at_ts: int,
        source_endpoint: str,
    ) -> dict[str, Any]:
        """Persist one future identity only when it was observed before start."""

        if fetched_at_ts <= 0:
            raise GammaMarketIdentityCacheError(
                "Gamma market identity fetched_at_ts must be positive.",
                reason_codes=("gamma_market_identity_cache_invalid_fetched_at_ts",),
            )
        if fetched_at_ts > market_start_ts:
            raise GammaMarketIdentityCacheError(
                "Gamma market identity was not prefetched before market start.",
                reason_codes=("gamma_market_identity_cache_post_start_prefetch",),
            )
        entry = {
            "schema_version": GAMMA_MARKET_IDENTITY_CACHE_SCHEMA_VERSION,
            "source_type": GAMMA_MARKET_IDENTITY_CACHE_SOURCE_TYPE,
            "source_endpoint": source_endpoint,
            "slug": slug,
            "market_family": market_family,
            "market_start_ts": market_start_ts,
            "market_end_ts": market_end_ts,
            "condition_id": condition_id,
            "up_token_id": up_token_id,
            "down_token_id": down_token_id,
            "reference_price_source": reference_price_source,
            "settlement_rule": settlement_rule,
            "raw_market_sha256": canonical_json_sha256(payload),
            "raw_public_payload": payload,
            "identity_payload": _identity_payload_projection(payload),
            "fetched_at_ts": fetched_at_ts,
            **safety_fields(),
        }
        entry["identity_payload_sha256"] = canonical_json_sha256(
            entry["identity_payload"]
        )
        entry["identity_payload_field_names"] = sorted(
            entry["identity_payload"]
        )
        entry["forbidden_fields_used_for_identity"] = []
        _validate_identity_projection_matches_entry(entry)
        _validate_entry_shape(entry)
        entry["cache_entry_sha256"] = _entry_sha256(entry)

        with self._locked():
            cache = self._read_cache_unlocked()
            existing = dict(cache["entries"].get(slug) or {})
            if existing:
                _validate_entry_integrity(existing)
                if _identity_tuple(existing) != _identity_tuple(entry):
                    raise GammaMarketIdentityCacheError(
                        "Existing cached market identity conflicts with Gamma payload.",
                        reason_codes=(
                            "gamma_market_identity_cache_immutable_identity_mismatch",
                        ),
                    )
                return existing
            cache["entries"][slug] = entry
            cache["updated_at_ts"] = self._now_ms()
            self._write_cache_unlocked(cache)
        return entry

    def lookup(
        self,
        *,
        slug: str,
        decision_ts: int,
        expected_market_family: str,
        expected_market_start_ts: int,
        expected_market_end_ts: int,
    ) -> dict[str, Any]:
        """Return an exact, causal cache entry or fail closed."""

        with self._locked():
            cache = self._read_cache_unlocked()
            raw_entry = cache["entries"].get(slug)
        if not isinstance(raw_entry, dict):
            raise GammaMarketIdentityCacheError(
                "No prefetched Gamma identity exists for the current market.",
                reason_codes=("gamma_market_identity_cache_missing",),
            )
        entry = dict(raw_entry)
        _validate_entry_integrity(entry)
        if entry["slug"] != slug:
            raise GammaMarketIdentityCacheError(
                "Cached Gamma identity slug does not match requested market.",
                reason_codes=("gamma_market_identity_cache_slug_mismatch",),
            )
        if (
            entry["market_family"] != expected_market_family
            or int(entry["market_start_ts"]) != expected_market_start_ts
            or int(entry["market_end_ts"]) != expected_market_end_ts
        ):
            raise GammaMarketIdentityCacheError(
                "Cached Gamma identity market window does not match requested market.",
                reason_codes=("gamma_market_identity_cache_window_mismatch",),
            )
        fetched_at_ts = int(entry["fetched_at_ts"])
        if fetched_at_ts > expected_market_start_ts or fetched_at_ts > decision_ts:
            raise GammaMarketIdentityCacheError(
                "Cached Gamma identity was not available before this decision.",
                reason_codes=("gamma_market_identity_cache_post_decision",),
            )
        if not (expected_market_start_ts <= decision_ts < expected_market_end_ts):
            raise GammaMarketIdentityCacheError(
                "Requested decision is outside the cached market window.",
                reason_codes=("gamma_market_identity_cache_decision_outside_window",),
            )
        age_ms = decision_ts - fetched_at_ts
        if age_ms < 0 or age_ms > self.max_age_ms:
            raise GammaMarketIdentityCacheError(
                "Cached Gamma identity exceeds the configured causal age limit.",
                reason_codes=("gamma_market_identity_cache_stale",),
            )
        entry["cache_age_ms"] = age_ms
        entry["cache_provenance_valid"] = True
        return entry

    def report(self) -> dict[str, Any]:
        """Return a deterministic operational snapshot of the cache."""

        with self._locked():
            cache = self._read_cache_unlocked()
        return {
            "schema_version": GAMMA_MARKET_IDENTITY_CACHE_SCHEMA_VERSION,
            "cache_path": str(self.path),
            "cache_exists": self.path.is_file(),
            "cache_entry_count": len(cache["entries"]),
            "cache_payload_sha256": canonical_json_sha256(cache),
            "max_age_ms": self.max_age_ms,
            **safety_fields(),
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_cache_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return _empty_cache()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GammaMarketIdentityCacheError(
                "Gamma market identity cache is unreadable.",
                reason_codes=("gamma_market_identity_cache_invalid_json",),
            ) from exc
        if not isinstance(payload, dict):
            raise GammaMarketIdentityCacheError(
                "Gamma market identity cache root must be an object.",
                reason_codes=("gamma_market_identity_cache_invalid_shape",),
            )
        if payload.get("schema_version") != GAMMA_MARKET_IDENTITY_CACHE_SCHEMA_VERSION:
            raise GammaMarketIdentityCacheError(
                "Gamma market identity cache schema is unsupported.",
                reason_codes=("gamma_market_identity_cache_schema_mismatch",),
            )
        if not isinstance(payload.get("entries"), dict):
            raise GammaMarketIdentityCacheError(
                "Gamma market identity cache entries must be an object.",
                reason_codes=("gamma_market_identity_cache_invalid_shape",),
            )
        for entry in payload["entries"].values():
            if not isinstance(entry, dict):
                raise GammaMarketIdentityCacheError(
                    "Gamma market identity cache contains a non-object entry.",
                    reason_codes=("gamma_market_identity_cache_invalid_shape",),
                )
            _validate_entry_integrity(dict(entry))
        return payload

    def _write_cache_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _now_ms(self) -> int:
        if self.current_time_ms is not None:
            return self.current_time_ms
        return int(time.time() * 1000)


def _empty_cache() -> dict[str, Any]:
    return {
        "schema_version": GAMMA_MARKET_IDENTITY_CACHE_SCHEMA_VERSION,
        "updated_at_ts": 0,
        "entries": {},
        **safety_fields(),
    }


def _entry_sha256(entry: dict[str, Any]) -> str:
    payload = dict(entry)
    payload.pop("cache_entry_sha256", None)
    payload.pop("cache_age_ms", None)
    payload.pop("cache_provenance_valid", None)
    return canonical_json_sha256(payload)


def _validate_entry_integrity(entry: dict[str, Any]) -> None:
    _validate_entry_shape(entry)
    expected = str(entry.get("cache_entry_sha256") or "")
    if not expected or expected != _entry_sha256(entry):
        raise GammaMarketIdentityCacheError(
            "Gamma market identity cache entry hash is invalid.",
            reason_codes=("gamma_market_identity_cache_hash_mismatch",),
        )
    _validate_identity_projection_matches_entry(entry)


def _validate_entry_shape(entry: dict[str, Any]) -> None:
    required_text = (
        "slug",
        "market_family",
        "condition_id",
        "up_token_id",
        "down_token_id",
        "reference_price_source",
        "settlement_rule",
        "source_endpoint",
        "raw_market_sha256",
    )
    if any(not str(entry.get(name) or "").strip() for name in required_text):
        raise GammaMarketIdentityCacheError(
            "Gamma market identity cache entry is incomplete.",
            reason_codes=("gamma_market_identity_cache_incomplete",),
        )
    if entry["up_token_id"] == entry["down_token_id"]:
        raise GammaMarketIdentityCacheError(
            "Gamma market identity cache entry duplicates UP/DOWN token ids.",
            reason_codes=("gamma_market_identity_cache_duplicate_tokens",),
        )
    start_ts = int(entry.get("market_start_ts") or 0)
    end_ts = int(entry.get("market_end_ts") or 0)
    fetched_at_ts = int(entry.get("fetched_at_ts") or 0)
    if start_ts <= 0 or end_ts <= start_ts or fetched_at_ts <= 0:
        raise GammaMarketIdentityCacheError(
            "Gamma market identity cache entry has an invalid time contract.",
            reason_codes=("gamma_market_identity_cache_invalid_time_contract",),
        )
    payload = entry.get("raw_public_payload")
    if not isinstance(payload, dict):
        raise GammaMarketIdentityCacheError(
            "Gamma market identity cache entry lacks its raw public payload.",
            reason_codes=("gamma_market_identity_cache_missing_raw_payload",),
        )
    if canonical_json_sha256(payload) != entry["raw_market_sha256"]:
        raise GammaMarketIdentityCacheError(
            "Gamma market identity raw payload hash does not match.",
            reason_codes=("gamma_market_identity_cache_raw_payload_hash_mismatch",),
        )
    identity_payload = entry.get("identity_payload")
    if not isinstance(identity_payload, dict) or not identity_payload:
        raise GammaMarketIdentityCacheError(
            "Gamma market identity cache entry lacks its identity projection.",
            reason_codes=("gamma_market_identity_cache_missing_identity_payload",),
        )
    if (
        canonical_json_sha256(identity_payload)
        != entry.get("identity_payload_sha256")
        or sorted(identity_payload)
        != entry.get("identity_payload_field_names")
    ):
        raise GammaMarketIdentityCacheError(
            "Gamma market identity projection hash does not match.",
            reason_codes=(
                "gamma_market_identity_cache_identity_payload_hash_mismatch",
            ),
        )
    if set(identity_payload) - set(
        GAMMA_MARKET_IDENTITY_ALLOWED_PAYLOAD_FIELDS
    ):
        raise GammaMarketIdentityCacheError(
            "Gamma market identity projection contains a forbidden field.",
            reason_codes=(
                "gamma_market_identity_cache_forbidden_identity_field",
            ),
        )
    if entry.get("forbidden_fields_used_for_identity") != []:
        raise GammaMarketIdentityCacheError(
            "Gamma market identity entry reports forbidden field usage.",
            reason_codes=(
                "gamma_market_identity_cache_forbidden_identity_field",
            ),
        )
    for name, expected in safety_fields().items():
        if entry.get(name) is not expected:
            raise GammaMarketIdentityCacheError(
                "Gamma market identity cache entry violates the safety contract.",
                reason_codes=(f"gamma_market_identity_cache_unsafe_{name}",),
            )


def _identity_tuple(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry["slug"],
        entry["market_family"],
        int(entry["market_start_ts"]),
        int(entry["market_end_ts"]),
        entry["condition_id"],
        entry["up_token_id"],
        entry["down_token_id"],
    )


def _identity_payload_projection(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        field_name: payload[field_name]
        for field_name in GAMMA_MARKET_IDENTITY_ALLOWED_PAYLOAD_FIELDS
        if field_name in payload
    }


def _validate_identity_projection_matches_entry(entry: dict[str, Any]) -> None:
    payload = dict(entry["identity_payload"])
    payload_slug = str(payload.get("slug") or payload.get("market_slug") or "")
    payload_condition_id = str(
        payload.get("conditionId") or payload.get("condition_id") or ""
    )
    outcomes = _json_list(payload.get("outcomes"))
    token_ids = _json_list(payload.get("clobTokenIds"))
    token_by_outcome: dict[str, str] = {}
    if len(outcomes) == len(token_ids):
        for outcome, token_id in zip(outcomes, token_ids, strict=True):
            normalized = str(outcome).strip().upper()
            if normalized in {"UP", "DOWN"}:
                token_by_outcome[normalized] = str(token_id)
    if (
        payload_slug != entry["slug"]
        or payload_condition_id != entry["condition_id"]
        or token_by_outcome
        != {
            "UP": entry["up_token_id"],
            "DOWN": entry["down_token_id"],
        }
    ):
        raise GammaMarketIdentityCacheError(
            "Gamma identity projection disagrees with normalized cache fields.",
            reason_codes=(
                "gamma_market_identity_cache_payload_identity_mismatch",
            ),
        )


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []
