"""Build a real Polymarket aggregate corpus with strict training exclusions."""

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
    DEFAULT_CORPUS_CREATED_AT,
    PolymarketRealCorpusAggregateConfig,
    build_polymarket_real_corpus_aggregate,
)


def run_polymarket_real_corpus_aggregate_cli(
    *,
    source_root: Path | str,
    output_dir: Path | str,
    run_id: str,
    market_families: tuple[str, ...] = ("btc_updown_5m",),
    sell_before_close_entry_notional: float = 1.0,
    sell_before_close_min_exit_notional: float = 1.0,
    created_at: str = DEFAULT_CORPUS_CREATED_AT,
    overwrite_existing: bool = False,
    exclude_sparse_theoretical_sell_before_close: bool = True,
) -> dict:
    result = build_polymarket_real_corpus_aggregate(
        PolymarketRealCorpusAggregateConfig(
            source_root=source_root,
            output_dir=output_dir,
            run_id=run_id,
            market_families=market_families,
            sell_before_close_entry_notional=sell_before_close_entry_notional,
            sell_before_close_min_exit_notional=sell_before_close_min_exit_notional,
            created_at=created_at,
            overwrite_existing=overwrite_existing,
            exclude_sparse_theoretical_sell_before_close=(
                exclude_sparse_theoretical_sell_before_close
            ),
        )
    )
    report = result.report
    return {
        "run_id": report["run_id"],
        "run_dir": str(result.run_dir),
        "corpus_dir": str(result.corpus_dir),
        "included_source_corpus_count": report["included_source_corpus_count"],
        "excluded_source_corpus_count": report["excluded_source_corpus_count"],
        "source_corpora_excluded_count": report["source_corpora_excluded_count"],
        "excluded_market_count": report["excluded_market_count"],
        "excluded_slugs": report["excluded_slugs"],
        "excluded_reason_counts": report["excluded_reason_counts"],
        "sell_before_close_label_gate_passed": report[
            "sell_before_close_label_gate_passed"
        ],
        "theoretical_sell_before_close_count": report[
            "theoretical_sell_before_close_count"
        ],
        "sparse_theoretical_sell_before_close_count": report[
            "sparse_theoretical_sell_before_close_count"
        ],
        "real_historical_training_eligible": report[
            "real_historical_training_eligible"
        ],
        "phase2_corpus_manifest_sha256": report["phase2_corpus_manifest_sha256"],
        "phase2_train_shadow_split_sha256": report[
            "phase2_train_shadow_split_sha256"
        ],
        "real_corpus_recorder_report_path": str(
            result.artifact_paths["real_corpus_recorder_report"]
        ),
        "real_corpus_recorder_manifest_path": str(
            result.artifact_paths["real_corpus_recorder_manifest"]
        ),
        "aggregate_source_corpora_path": str(
            result.artifact_paths["aggregate_source_corpora"]
        ),
        "aggregate_summary_path": str(result.artifact_paths["aggregate_summary"]),
        "paper_only": report["paper_only"],
        "capital_at_risk": report["capital_at_risk"],
        "polymarket_write_enabled": report["polymarket_write_enabled"],
        "wallet_signing_enabled": report["wallet_signing_enabled"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--market-family", action="append", dest="market_families")
    parser.add_argument("--created-at", default=DEFAULT_CORPUS_CREATED_AT)
    parser.add_argument("--sell-before-close-entry-notional", type=float, default=1.0)
    parser.add_argument("--sell-before-close-min-exit-notional", type=float, default=1.0)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--include-sparse-theoretical",
        action="store_true",
        help="Disable the default sparse theoretical sell-before-close exclusion.",
    )
    args = parser.parse_args(argv)
    summary = run_polymarket_real_corpus_aggregate_cli(
        source_root=args.source_root,
        output_dir=args.output_dir,
        run_id=args.run_id,
        market_families=tuple(args.market_families or ("btc_updown_5m",)),
        sell_before_close_entry_notional=args.sell_before_close_entry_notional,
        sell_before_close_min_exit_notional=args.sell_before_close_min_exit_notional,
        created_at=args.created_at,
        overwrite_existing=args.overwrite_existing,
        exclude_sparse_theoretical_sell_before_close=not args.include_sparse_theoretical,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
