#!/usr/bin/env python3
"""Prefetch causal future Polymarket market identities into a read-only cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket import (  # noqa: E402
    PolymarketPublicHTTPRealCorpusProvider,
    PolymarketRealCorpusRecorderConfig,
)
from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS  # noqa: E402
from bigan.v8.polymarket.corpus.contracts import safety_fields  # noqa: E402


def run_gamma_market_identity_prefetch_cli(
    *,
    run_id: str,
    output_dir: Path | str,
    cache_path: Path | str,
    market_family: str,
    prefetch_round_count: int,
    http_timeout_seconds: float,
    cache_max_age_seconds: float,
    current_time_ms: int | None = None,
) -> dict:
    if not run_id.strip():
        raise ValueError("run_id is required")
    if market_family not in BTC_UPDOWN_MARKET_HORIZONS_MS:
        raise ValueError("unsupported market_family")
    run_dir = Path(output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_cache_path = Path(cache_path).expanduser().resolve()
    provider = PolymarketPublicHTTPRealCorpusProvider(
        max_markets=1,
        http_timeout_seconds=http_timeout_seconds,
        current_time_ms=current_time_ms,
        market_identity_cache_path=resolved_cache_path,
        market_identity_cache_max_age_seconds=cache_max_age_seconds,
        gamma_market_identity_prefetch_round_count=prefetch_round_count,
    )
    config = PolymarketRealCorpusRecorderConfig(
        run_id=run_id,
        output_dir=Path(output_dir).expanduser().resolve(),
        market_families=(market_family,),
        mock_public_data=False,
    )
    report = {
        "schema_version": "bigan-v8-gamma-market-identity-prefetch-report-v1",
        "run_id": run_id,
        "market_family": market_family,
        "cache_path": str(resolved_cache_path),
        "http_timeout_seconds": http_timeout_seconds,
        "cache_max_age_seconds": cache_max_age_seconds,
        "prefetch_round_count": prefetch_round_count,
        "gamma_primary_remains_required": True,
        "cache_fallback_requires_exact_slug_window": True,
        "cache_fallback_requires_pre_start_fetch": True,
        "cache_fallback_requires_clob_revalidation": True,
        "cache_fallback_requires_live_orderbook_validation": True,
        "outcome_or_pnl_fields_used": False,
        **provider.prefetch_gamma_market_identities(config=config),
        **safety_fields(),
    }
    report_path = run_dir / "gamma_market_identity_prefetch_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema_version": "bigan-v8-gamma-market-identity-prefetch-manifest-v1",
        "run_id": run_id,
        "report_path": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "cache_path": str(resolved_cache_path),
        "cache_payload_sha256": report["cache_report"][
            "cache_payload_sha256"
        ],
        "stored_slug_count": report["stored_slug_count"],
        "reason_codes": report["reason_codes"],
        **safety_fields(),
    }
    manifest_path = run_dir / "gamma_market_identity_prefetch_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
        "manifest": manifest,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
    )
    parser.add_argument("--cache-path", required=True)
    parser.add_argument(
        "--market-family",
        choices=tuple(BTC_UPDOWN_MARKET_HORIZONS_MS),
        default="btc_updown_5m",
    )
    parser.add_argument("--prefetch-round-count", type=int, default=12)
    parser.add_argument("--http-timeout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--cache-max-age-seconds",
        type=float,
        default=7_200.0,
    )
    args = parser.parse_args()
    result = run_gamma_market_identity_prefetch_cli(
        run_id=args.run_id,
        output_dir=args.output_dir,
        cache_path=args.cache_path,
        market_family=args.market_family,
        prefetch_round_count=args.prefetch_round_count,
        http_timeout_seconds=args.http_timeout_seconds,
        cache_max_age_seconds=args.cache_max_age_seconds,
    )
    print(
        json.dumps(
            {
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "stored_slug_count": result["report"]["stored_slug_count"],
                "reason_codes": result["report"]["reason_codes"],
                **safety_fields(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
