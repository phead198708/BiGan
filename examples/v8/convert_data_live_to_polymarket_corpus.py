"""Convert historical data/live Polymarket signal observations into a v8 Phase 2 corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.corpus import (  # noqa: E402
    BTC_UPDOWN_MARKET_HORIZONS_MS,
    DEFAULT_CORPUS_CREATED_AT,
    LiveSignalCorpusConversionConfig,
    convert_live_signals_to_phase2_corpus,
)


def run_live_to_corpus_cli(
    *,
    input_path: Path | str,
    output_dir: Path | str,
    created_at: str = DEFAULT_CORPUS_CREATED_AT,
    market_families: tuple[str, ...] | None = None,
    build_phase2_corpus: bool = True,
    allow_midpoint_price_proxy: bool = False,
    overwrite_existing: bool = False,
) -> dict:
    """Run the fail-closed data/live to Phase 2 corpus conversion."""

    config = LiveSignalCorpusConversionConfig(
        input_path=input_path,
        output_dir=output_dir,
        created_at=created_at,
        market_families=tuple(market_families or BTC_UPDOWN_MARKET_HORIZONS_MS),
        build_phase2_corpus=build_phase2_corpus,
        allow_midpoint_price_proxy=allow_midpoint_price_proxy,
        overwrite_existing=overwrite_existing,
    )
    result = convert_live_signals_to_phase2_corpus(config)
    report = result.report
    return {
        "output_dir": str(result.output_dir),
        "raw_dir": str(result.raw_dir),
        "corpus_dir": None if result.corpus_dir is None else str(result.corpus_dir),
        "conversion_manifest_path": str(result.artifact_paths["conversion_manifest"]),
        "conversion_report_path": str(result.artifact_paths["conversion_report"]),
        "rejected_rows_path": str(result.artifact_paths["rejected_rows"]),
        "input_row_count": report["input_row_count"],
        "accepted_market_count": report["accepted_market_count"],
        "accepted_orderbook_row_count": report["accepted_orderbook_row_count"],
        "accepted_btc_candle_count": report["accepted_btc_candle_count"],
        "accepted_resolution_count": report["accepted_resolution_count"],
        "rejected_item_count": report["rejected_item_count"],
        "reject_reason_counts": report["reject_reason_counts"],
        "training_eligible": report["training_eligible"],
        "phase2_corpus_built": report["phase2_corpus_built"],
        "phase2_error": report["phase2_error"],
        "artifact_hashes": result.artifact_hashes,
        "paper_only": report["paper_only"],
        "capital_at_risk": report["capital_at_risk"],
        "polymarket_write_enabled": report["polymarket_write_enabled"],
        "wallet_signing_enabled": report["wallet_signing_enabled"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="data/live directory or one signals JSONL file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--created-at", default=DEFAULT_CORPUS_CREATED_AT)
    parser.add_argument(
        "--market-family",
        action="append",
        choices=sorted(BTC_UPDOWN_MARKET_HORIZONS_MS),
        help="Restrict conversion to one market family. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--allow-midpoint-price-proxy",
        action="store_true",
        help="Permit synthetic bid/ask from midpoint prices. Default is disabled.",
    )
    parser.add_argument(
        "--no-build-phase2-corpus",
        action="store_true",
        help="Only write raw conversion artifacts and validation report.",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)

    summary = run_live_to_corpus_cli(
        input_path=args.input_path,
        output_dir=args.output_dir,
        created_at=args.created_at,
        market_families=tuple(args.market_family) if args.market_family else None,
        build_phase2_corpus=not args.no_build_phase2_corpus,
        allow_midpoint_price_proxy=args.allow_midpoint_price_proxy,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
