"""Outcome-blind slug and signal feed for the running promotion collection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_capture_archive import (
    EMPTY_RESOLUTION_STREAM_NAME,
    FORBIDDEN_CAPTURE_FILE_TOKENS,
)
from bigan.v8.polymarket.residual_promotion_collection import (
    assert_outcome_blind,
    verify_attempt_chain,
)
from bigan.v8.polymarket.residual_promotion_v1 import CANDIDATE_ID, LINEAGE_ID

SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-outcome-blind-signal-feed-v1"
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_signal_feed.py"
)
CLI_REPOSITORY_PATH = "examples/v8/run_residual_promotion_signal_feed.py"
RAW_MARKET_FILENAME = "raw_polymarket_markets.jsonl"
EXPECTED_DECISION_OFFSETS_MS = (300_000, 600_000)
ALLOWED_ACTIONS = frozenset({"NO_TRADE", "BUY_UP_HOLD", "BUY_DOWN_HOLD"})
SLUG_PATTERN = re.compile(r"^btc-updown-15m-(\d+)$")


class SignalFeedError(ValueError):
    """Fail-closed signal feed validation error."""


def export_outcome_blind_signal_feed(
    *,
    service_root: Path | str,
    output_path: Path | str,
    stable_read_attempts: int = 20,
    stable_read_delay_seconds: float = 0.05,
) -> dict[str, Any]:
    """Atomically refresh one monitoring-only feed from ledger-closed attempts."""

    root = Path(service_root).resolve()
    output = Path(output_path).resolve()
    if not root.is_dir():
        raise SignalFeedError("service root is missing")
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise SignalFeedError("signal feed output must be disjoint from service root")
    attempts, progress, ledger_sha256, progress_sha256 = _stable_snapshot(
        root,
        attempts=stable_read_attempts,
        delay_seconds=stable_read_delay_seconds,
    )
    unique_valid: list[Mapping[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    duplicate_valid: list[dict[str, Any]] = []
    seen_market_ids: set[str] = set()
    for attempt in attempts:
        quality = dict(attempt.get("quality") or {})
        market_id = attempt.get("market_id")
        if quality.get("quality_valid") is not True or not isinstance(market_id, str):
            excluded.append(_excluded_attempt(attempt))
            continue
        if market_id in seen_market_ids:
            duplicate_valid.append(_excluded_attempt(attempt, reason="duplicate_quality_valid_market"))
            continue
        seen_market_ids.add(market_id)
        unique_valid.append(attempt)
    if len(unique_valid) != int(progress.get("quality_valid_market_count", -1)):
        raise SignalFeedError("signal feed population does not reconcile with progress")

    matches = [
        _match_row(root=root, attempt=attempt, population_position=position)
        for position, attempt in enumerate(unique_valid, start=1)
    ]
    raw_actions = Counter(
        signal["selected_action"] for match in matches for signal in match["signals"]
    )
    accepted_actions = Counter(match["accepted_action"] for match in matches)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "source_service_root": str(root),
        "source_ledger_sha256": ledger_sha256,
        "source_progress_sha256": progress_sha256,
        "source_progress_updated_at": progress.get("updated_at"),
        "candidate_bundle_sha256": progress.get("candidate_bundle_sha256"),
        "authorization_sha256": progress.get("authorization_sha256"),
        "collector_protocol_sha256": progress.get("collector_protocol_sha256"),
        "attempts_consumed": len(attempts),
        "quality_valid_match_count": len(matches),
        "target_quality_valid_market_count": progress.get(
            "target_quality_valid_market_count"
        ),
        "remaining_quality_valid_markets": progress.get(
            "remaining_quality_valid_markets"
        ),
        "population_order": "chronological_first_quality_valid_unique_without_reordering",
        "matches": matches,
        "excluded_attempts": excluded,
        "duplicate_quality_valid_attempts": duplicate_valid,
        "raw_signal_distribution": dict(sorted(raw_actions.items())),
        "accepted_match_signal_distribution": dict(sorted(accepted_actions.items())),
        "signal_contract": {
            "sampling_mode": "two_frozen_decision_points_per_quality_valid_match",
            "expected_market_age_seconds": [300, 600],
            "acceptance_policy": "first_non_no_trade_once_per_market",
            "subsequent_signal_can_never_create_a_second_acceptance": True,
            "refresh_boundary": "after_ledger_close_only",
            "in_progress_attempts_read": False,
            "execution_enabled": False,
        },
        "monitoring_only": True,
        "monitoring_influences_collection": False,
        "monitoring_influences_model": False,
        "collection_state_mutated": False,
        "fresh_outcomes_accessed": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "paper_candidate_allowed": False,
        "paper_run_started": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    payload = {**identity, "content_sha256": canonical_json_sha256(identity)}
    _atomic_write_if_changed(output, payload)
    return payload


def run_signal_feed_monitor(
    *,
    service_root: Path | str,
    output_path: Path | str,
    poll_seconds: float,
) -> None:
    """Continuously refresh after ledger closures without touching the collector."""

    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise SignalFeedError("poll seconds must be positive and finite")
    previous_sha256: str | None = None
    while True:
        report = export_outcome_blind_signal_feed(
            service_root=service_root,
            output_path=output_path,
        )
        current_sha256 = str(report["content_sha256"])
        if current_sha256 != previous_sha256:
            print(
                json.dumps(
                    {
                        "attempts_consumed": report["attempts_consumed"],
                        "content_sha256": current_sha256,
                        "quality_valid_match_count": report[
                            "quality_valid_match_count"
                        ],
                        "output_path": str(Path(output_path).resolve()),
                        "fresh_outcomes_accessed": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            previous_sha256 = current_sha256
        time.sleep(poll_seconds)


def _stable_snapshot(
    root: Path, *, attempts: int, delay_seconds: float
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    if attempts <= 0 or not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise SignalFeedError("stable-read policy is invalid")
    ledger_path = root / "outcome_blind_attempts.jsonl"
    progress_path = root / "collection_progress.json"
    last_error: Exception | None = None
    for index in range(attempts):
        ledger_bytes = ledger_path.read_bytes()
        progress_bytes = progress_path.read_bytes()
        if ledger_bytes != ledger_path.read_bytes() or progress_bytes != progress_path.read_bytes():
            last_error = SignalFeedError("collection state changed during snapshot")
        else:
            try:
                ledger_rows = _jsonl_bytes(ledger_bytes)
                progress = _json_bytes(progress_bytes)
                verify_attempt_chain(ledger_rows)
                for row in ledger_rows:
                    assert_outcome_blind(row)
                assert_outcome_blind(progress)
                _validate_progress(progress, attempts=ledger_rows)
                return (
                    ledger_rows,
                    progress,
                    hashlib.sha256(ledger_bytes).hexdigest(),
                    hashlib.sha256(progress_bytes).hexdigest(),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        if index + 1 < attempts:
            time.sleep(delay_seconds)
    raise SignalFeedError("could not obtain a stable reconciled collection snapshot") from last_error


def _validate_progress(
    progress: Mapping[str, Any], *, attempts: Sequence[Mapping[str, Any]]
) -> None:
    if not (
        progress.get("lineage_id") == LINEAGE_ID
        and progress.get("candidate_id") == CANDIDATE_ID
        and int(progress.get("attempts_consumed", -1)) == len(attempts)
        and progress.get("hash_chain_status") == "valid"
        and progress.get("fresh_outcomes_opened") is False
        and progress.get("interim_pnl_evaluated") is False
        and progress.get("collection_influenced_by_model_decisions") is False
        and progress.get("wallet_signing_allowed") is False
        and progress.get("polymarket_write_allowed") is False
        and progress.get("capital_at_risk") is False
        and dict(progress.get("safety") or {}) == SAFETY
    ):
        raise SignalFeedError("collection progress is not outcome-blind or reconciled")


def _match_row(
    *, root: Path, attempt: Mapping[str, Any], population_position: int
) -> dict[str, Any]:
    attempt_id = str(attempt.get("attempt_id") or "")
    run_dir = (root / "captures" / attempt_id).resolve()
    if not run_dir.is_relative_to(root) or not run_dir.is_dir():
        raise SignalFeedError("ledger-bound capture directory is missing or escaped root")
    manifest_path = run_dir / "pending_round_capture_manifest.json"
    report_path = run_dir / "pending_round_capture_report.json"
    if (
        sha256_file(manifest_path) != attempt.get("capture_manifest_sha256")
        or sha256_file(report_path) != attempt.get("capture_report_sha256")
    ):
        raise SignalFeedError("ledger-bound capture hash mismatch")
    manifest = _load_json(manifest_path)
    capture_report = _load_json(report_path)
    if not (
        manifest.get("resolution_provider_called") is False
        and capture_report.get("resolution_provider_called") is False
        and manifest.get("wallet_signing_enabled") is False
        and manifest.get("polymarket_write_enabled") is False
        and manifest.get("capital_at_risk") is False
    ):
        raise SignalFeedError("capture safety boundary is open")
    _validate_capture_names_and_resolution_streams(run_dir)
    raw_market_path = run_dir / "raw" / RAW_MARKET_FILENAME
    raw_hashes = dict(manifest.get("raw_artifact_hashes") or {})
    if raw_hashes.get(RAW_MARKET_FILENAME) != sha256_file(raw_market_path):
        raise SignalFeedError("raw market artifact SHA-256 mismatch")
    market_rows = _load_jsonl(raw_market_path)
    market_id = str(attempt.get("market_id") or "")
    matches = [row for row in market_rows if str(row.get("market_id") or "") == market_id]
    if len(matches) != 1:
        raise SignalFeedError("ledger market identity does not resolve exactly once")
    market = matches[0]
    slug = str(market.get("slug") or "")
    slug_match = SLUG_PATTERN.fullmatch(slug)
    start_ts = _positive_int(market.get("market_start_ts"), "market start")
    end_ts = _positive_int(market.get("market_end_ts"), "market end")
    if (
        slug_match is None
        or int(slug_match.group(1)) * 1000 != start_ts
        or end_ts - start_ts != 900_000
        or market.get("market_family") != "btc_updown_15m"
    ):
        raise SignalFeedError("BTC 15m slug identity is invalid")
    decisions = sorted(
        [dict(row) for row in attempt.get("decision_rows") or []],
        key=lambda row: int(row.get("decision_ts", 0)),
    )
    if len(decisions) != 2:
        raise SignalFeedError("quality-valid match must have exactly two decision signals")
    signals: list[dict[str, Any]] = []
    already_accepted = False
    for number, (decision, expected_offset) in enumerate(
        zip(decisions, EXPECTED_DECISION_OFFSETS_MS, strict=True), start=1
    ):
        decision_ts = _positive_int(decision.get("decision_ts"), "decision")
        action = str(decision.get("candidate_selected_action") or "")
        if action not in ALLOWED_ACTIONS or decision.get("market_id") != market_id:
            raise SignalFeedError("candidate decision identity or action is invalid")
        action_values = _action_values(decision.get("candidate_action_values"))
        expected_accept = not already_accepted and action != "NO_TRADE"
        if decision.get("candidate_accepted_at_this_decision") is not expected_accept:
            raise SignalFeedError("candidate first-signal acceptance semantics drifted")
        already_accepted = already_accepted or expected_accept
        if decision_ts - start_ts != expected_offset or not start_ts < decision_ts < end_ts:
            raise SignalFeedError("candidate decision timestamp drifted from frozen schedule")
        signals.append(
            {
                "decision_number": number,
                "decision_ts": decision_ts,
                "market_age_seconds": expected_offset // 1000,
                "time_to_close_seconds": (end_ts - decision_ts) // 1000,
                "action_values": action_values,
                "selected_action": action,
                "accepted_at_this_decision": expected_accept,
            }
        )
    accepted = next(
        (signal for signal in signals if signal["accepted_at_this_decision"]), None
    )
    return {
        "population_position": population_position,
        "attempt_index": int(attempt["attempt_index"]),
        "attempt_id": attempt_id,
        "attempt_hash": str(attempt["attempt_hash"]),
        "market_id": market_id,
        "condition_id": str(market.get("condition_id") or ""),
        "slug": slug,
        "market_start_ts": start_ts,
        "market_end_ts": end_ts,
        "quality_valid": True,
        "signals": signals,
        "accepted_action": accepted["selected_action"] if accepted else "NO_TRADE",
        "accepted_decision_number": accepted["decision_number"] if accepted else None,
        "execution_status": "MONITORING_ONLY_NO_INTENT",
        "decision_influenced_collection": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
    }


def _action_values(value: Any) -> dict[str, float]:
    raw = dict(value or {})
    if set(raw) != ALLOWED_ACTIONS:
        raise SignalFeedError("candidate action-value keys are invalid")
    output: dict[str, float] = {}
    for action in sorted(ALLOWED_ACTIONS):
        number = raw[action]
        if isinstance(number, bool) or not isinstance(number, int | float):
            raise SignalFeedError("candidate action value is not numeric")
        number = float(number)
        if not math.isfinite(number):
            raise SignalFeedError("candidate action value is non-finite")
        output[action] = number
    return output


def _excluded_attempt(
    attempt: Mapping[str, Any], *, reason: str | None = None
) -> dict[str, Any]:
    quality = dict(attempt.get("quality") or {})
    reasons = list(quality.get("invalid_reason_codes") or [])
    if reason is not None:
        reasons.append(reason)
    return {
        "attempt_index": int(attempt.get("attempt_index", -1)),
        "attempt_id": str(attempt.get("attempt_id") or ""),
        "market_id": attempt.get("market_id"),
        "reason_codes": sorted(set(map(str, reasons))),
    }


def _validate_capture_names_and_resolution_streams(run_dir: Path) -> None:
    resolution_count = 0
    for path in run_dir.rglob("*"):
        if path.name.startswith("._"):
            raise SignalFeedError("capture contains AppleDouble metadata")
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if lowered == EMPTY_RESOLUTION_STREAM_NAME:
            resolution_count += 1
            if path.stat().st_size != 0:
                raise SignalFeedError("capture has a non-empty resolution stream")
        elif any(token in lowered for token in FORBIDDEN_CAPTURE_FILE_TOKENS):
            raise SignalFeedError("capture contains an outcome-bearing filename")
    if resolution_count == 0:
        raise SignalFeedError("capture is missing its empty resolution stream")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise SignalFeedError(f"{label} timestamp is invalid")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SignalFeedError(f"{label} timestamp is invalid") from exc
    if result <= 0:
        raise SignalFeedError(f"{label} timestamp is invalid")
    return result


def _jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in raw.decode().splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise SignalFeedError("JSONL contains a non-object row")
    return rows


def _json_bytes(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise SignalFeedError("JSON root must be an object")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return _json_bytes(path.read_bytes())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return _jsonl_bytes(path.read_bytes())


def _atomic_write_if_changed(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    if path.is_file() and path.read_bytes() == raw:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "SignalFeedError",
    "export_outcome_blind_signal_feed",
    "run_signal_feed_monitor",
]
