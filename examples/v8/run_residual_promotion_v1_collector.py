"""Run the authorized BTC 15m promotion-v1 outcome-blind shadow collector."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket import (  # noqa: E402
    DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    PolymarketChainlinkRTDSCollector,
    PolymarketPublicHTTPRealCorpusProvider,
    PolymarketRealCorpusRecorderConfig,
    capture_polymarket_pending_round,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY  # noqa: E402
from bigan.v8.polymarket.residual_promotion_collection import (  # noqa: E402
    append_attempt,
    build_progress,
    observe_outcome_blind_capture,
    validate_collection_authorization,
)
from bigan.v8.polymarket.residual_promotion_v1 import (  # noqa: E402
    LINEAGE_ID,
    MAXIMUM_ATTEMPTS,
    TARGET_MARKETS,
    load_matched_baseline,
)

CONFIG = ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", type=Path, required=True)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=CONFIG / "manual_collection_authorization_v2.json",
    )
    parser.add_argument(
        "--collector-protocol",
        type=Path,
        default=CONFIG / "prospective_collector_protocol_v2.json",
    )
    parser.add_argument("--http-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--capture-timeout-seconds", type=float, default=930.0)
    parser.add_argument("--snapshot-interval-seconds", type=float, default=1.0)
    parser.add_argument("--max-round-start-lag-seconds", type=float, default=30.0)
    parser.add_argument(
        "--clob-ws-url", default=DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.service_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    validation = validate_collection_authorization(
        authorization_path=args.authorization,
        collector_protocol_path=args.collector_protocol,
        repository_root=ROOT,
    )
    runtime = validation["runtime"]
    baseline = load_matched_baseline(repository_root=ROOT)
    chainlink = PolymarketChainlinkRTDSCollector()
    chainlink.start()
    try:
        return _collect(
            args=args,
            root=root,
            validation=validation,
            runtime=runtime,
            baseline=baseline,
            chainlink=chainlink,
        )
    finally:
        chainlink.stop()


def _collect(
    *,
    args: argparse.Namespace,
    root: Path,
    validation: dict[str, Any],
    runtime: Any,
    baseline: Any,
    chainlink: PolymarketChainlinkRTDSCollector,
) -> int:
    bundle_sha = str(validation["bundle"]["sha256"])
    attempts_path = root / "outcome_blind_attempts.jsonl"
    existing = _load_jsonl(attempts_path) if attempts_path.exists() else []
    progress = build_progress(
        existing,
        authorization_sha256=validation["authorization_sha256"],
        collector_protocol_sha256=validation["collector_protocol_sha256"],
        candidate_bundle_sha256=bundle_sha,
    )
    _write_start_record(
        root,
        validation=validation,
        resumed_attempt_count=len(existing),
    )
    while not progress["collection_complete"] and not progress["attempt_cap_exhausted"]:
        scheduled_start = _wait_for_round_start(
            max_lag_seconds=args.max_round_start_lag_seconds
        )
        attempt_index = int(progress["attempts_consumed"]) + 1
        run_id = (
            f"{LINEAGE_ID}-attempt{attempt_index:04d}-"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        try:
            provider = PolymarketPublicHTTPRealCorpusProvider(
                max_markets=1,
                clob_ws_url=args.clob_ws_url,
                timeout_seconds=args.capture_timeout_seconds,
                http_timeout_seconds=args.http_timeout_seconds,
                orderbook_snapshot_interval_seconds=args.snapshot_interval_seconds,
            )
            config = PolymarketRealCorpusRecorderConfig(
                run_id=run_id,
                output_dir=root / "captures",
                market_families=("btc_updown_15m",),
                mock_public_data=False,
                overwrite_existing=False,
            )
            capture = capture_polymarket_pending_round(
                config,
                public_provider=provider,
                chainlink_rtds_collector=chainlink,
            )
            capture_summary = {
                **dict(capture.report),
                "round_index": attempt_index,
                "run_id": run_id,
                "run_dir": str(capture.run_dir),
                "scheduled_round_start_ts": scheduled_start,
            }
            attempt = observe_outcome_blind_capture(
                capture_summary,
                runtime=runtime,
                baseline=baseline,
            )
        except Exception as exc:  # noqa: BLE001
            attempt = _failed_attempt(
                attempt_index=attempt_index,
                run_id=run_id,
                scheduled_start=scheduled_start,
                error=exc,
            )
        progress = append_attempt(
            service_root=root,
            attempt=attempt,
            authorization_sha256=validation["authorization_sha256"],
            collector_protocol_sha256=validation["collector_protocol_sha256"],
            candidate_bundle_sha256=bundle_sha,
        )
        print(json.dumps(_progress_line(progress), sort_keys=True), flush=True)
    return 0


def _failed_attempt(
    *, attempt_index: int, run_id: str, scheduled_start: int, error: Exception
) -> dict[str, Any]:
    return {
        "schema_version": "bigan-btc-15m-residual-promotion-attempt-v1",
        "lineage_id": LINEAGE_ID,
        "attempt_index": attempt_index,
        "attempt_id": run_id,
        "scheduled_round_start_ts": scheduled_start,
        "market_id": None,
        "capture_manifest_sha256": None,
        "capture_report_sha256": None,
        "quality": {
            "quality_valid": False,
            "quality_observations": {},
            "invalid_reason_codes": ["capture_exception"],
            "paired_executable_ask_decision_count": 0,
            "observed_decision_count": 0,
            "btc_feature_complete_decision_count": 0,
            "causality_violation_count": 0,
            "missing_feature_count": 0,
            "missing_feature_counts": {},
            "missing_values_encoded_as_zero": False,
        },
        "provider_health": {
            "provider_failed": True,
            "retry_used": False,
            "paired_executable_ask_decision_count": 0,
            "causality_violation_count": 0,
            "missing_feature_count": 0,
            "error_type": error.__class__.__name__,
            "error_message": str(error),
        },
        "decision_rows": [],
        "collection_decision_inputs": "quality_only_model_decisions_excluded",
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }


def _wait_for_round_start(*, max_lag_seconds: float) -> int:
    period = 15 * 60
    while True:
        now = time.time()
        current = int(now // period) * period
        lag = now - current
        if lag <= max_lag_seconds:
            return current * 1000
        next_start = current + period
        time.sleep(min(60.0, max(0.05, next_start - now)))


def _write_start_record(
    root: Path,
    *,
    validation: dict[str, Any],
    resumed_attempt_count: int,
) -> None:
    path = root / "collection_start_record.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if not (
            previous["authorization_sha256"] == validation["authorization_sha256"]
            and previous["collector_protocol_sha256"]
            == validation["collector_protocol_sha256"]
            and previous["candidate_bundle_sha256"] == validation["bundle"]["sha256"]
        ):
            raise ValueError("existing collection start record binding mismatch")
        return
    payload = {
        "schema_version": "bigan-btc-15m-residual-promotion-start-v1",
        "lineage_id": LINEAGE_ID,
        "started_at": datetime.now(UTC).isoformat(),
        "collector_pid": os.getpid(),
        "authorization_sha256": validation["authorization_sha256"],
        "collector_protocol_sha256": validation["collector_protocol_sha256"],
        "candidate_bundle_sha256": validation["bundle"]["sha256"],
        "target_quality_valid_market_count": TARGET_MARKETS,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "resumed_attempt_count": resumed_attempt_count,
        "fresh_collection_started": True,
        "fresh_outcomes_opened": False,
        "zero_capital_read_only": True,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _progress_line(progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempts_consumed": progress["attempts_consumed"],
        "quality_valid_market_count": progress["quality_valid_market_count"],
        "remaining_quality_valid_markets": progress["remaining_quality_valid_markets"],
        "observed_quality_valid_rate": progress["observed_quality_valid_rate"],
        "estimated_remaining_days_at_96_markets_per_day": progress[
            "estimated_remaining_days_at_96_markets_per_day"
        ],
        "fresh_outcomes_opened": False,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    raise SystemExit(main())
