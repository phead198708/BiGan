"""Run development or frozen-model outcome-blind batch canaries for #208."""

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

from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (  # noqa: E402
    FrozenModelBatchCanaryConfig,
    OutcomeBlindDevelopmentBatchCanaryConfig,
    build_frozen_model_cumulative_canary,
    build_v5_retrospective_no_trade_canary_report,
    run_frozen_model_batch_canary,
    run_outcome_blind_development_batch_canary,
    write_frozen_model_cumulative_canary,
    write_v5_retrospective_no_trade_canary_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    development = subparsers.add_parser("development")
    _common_run_args(development)
    development.add_argument("--collector-index", required=True)
    development.add_argument("--collector-index-sha256", required=True)
    development.add_argument("--batch-id", required=True)
    development.add_argument("--feature-contract", required=True)
    development.add_argument("--feature-contract-sha256", required=True)

    frozen = subparsers.add_parser("frozen-model-batch")
    _common_run_args(frozen)
    frozen.add_argument("--development-canary-manifest", required=True)
    frozen.add_argument("--development-canary-manifest-sha256", required=True)
    frozen.add_argument("--research-candidate-manifest", required=True)
    frozen.add_argument("--research-candidate-manifest-sha256", required=True)

    cumulative = subparsers.add_parser("cumulative")
    _common_run_args(cumulative)
    cumulative.add_argument("--batch-report", action="append", required=True)
    cumulative.add_argument("--batch-report-sha256", action="append", required=True)
    cumulative.add_argument("--minimum-accepted-market-count", type=int, default=120)
    cumulative.add_argument("--minimum-side-market-count", type=int, default=17)
    cumulative.add_argument("--maximum-index-scan-count", type=int, default=462)

    retrospective = subparsers.add_parser("v5-retrospective")
    _common_run_args(retrospective)
    retrospective.add_argument("--target-free-predictions", required=True)
    retrospective.add_argument("--target-free-predictions-sha256", required=True)
    retrospective.add_argument("--batch-market-count", type=int, default=12)

    args = parser.parse_args(argv)
    if args.mode == "development":
        result = run_outcome_blind_development_batch_canary(
            OutcomeBlindDevelopmentBatchCanaryConfig(
                run_id=args.run_id,
                output_dir=args.output_dir,
                collector_index_path=args.collector_index,
                expected_collector_index_sha256=args.collector_index_sha256,
                batch_id=args.batch_id,
                feature_contract_path=args.feature_contract,
                expected_feature_contract_sha256=args.feature_contract_sha256,
                overwrite_existing=args.overwrite_existing,
            )
        )
    elif args.mode == "frozen-model-batch":
        result = run_frozen_model_batch_canary(
            FrozenModelBatchCanaryConfig(
                run_id=args.run_id,
                output_dir=args.output_dir,
                development_batch_canary_manifest_path=args.development_canary_manifest,
                expected_development_batch_canary_manifest_sha256=(
                    args.development_canary_manifest_sha256
                ),
                research_candidate_manifest_path=args.research_candidate_manifest,
                expected_research_candidate_manifest_sha256=(
                    args.research_candidate_manifest_sha256
                ),
                overwrite_existing=args.overwrite_existing,
            )
        )
    elif args.mode == "cumulative":
        paths = [Path(value).resolve() for value in args.batch_report]
        hashes = list(args.batch_report_sha256)
        if len(paths) != len(hashes):
            parser.error("--batch-report and --batch-report-sha256 counts must match")
        reports = []
        for path, expected in zip(paths, hashes, strict=True):
            if _sha256(path) != expected:
                parser.error(f"batch report hash mismatch: {path}")
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        report = build_frozen_model_cumulative_canary(
            reports,
            run_id=args.run_id,
            minimum_accepted_market_count=args.minimum_accepted_market_count,
            minimum_side_market_count=args.minimum_side_market_count,
            maximum_index_scan_count=args.maximum_index_scan_count,
        )
        result = write_frozen_model_cumulative_canary(
            report=report,
            batch_report_paths=paths,
            output_dir=Path(args.output_dir),
            run_id=args.run_id,
            overwrite_existing=args.overwrite_existing,
        )
    else:
        prediction_path = Path(args.target_free_predictions).resolve()
        if _sha256(prediction_path) != args.target_free_predictions_sha256:
            parser.error(f"target-free prediction hash mismatch: {prediction_path}")
        prediction_rows = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        report = build_v5_retrospective_no_trade_canary_report(
            prediction_rows,
            run_id=args.run_id,
            batch_market_count=args.batch_market_count,
        )
        result = write_v5_retrospective_no_trade_canary_report(
            report=report,
            source_prediction_path=prediction_path,
            output_dir=Path(args.output_dir),
            run_id=args.run_id,
            overwrite_existing=args.overwrite_existing,
        )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "labels_outcomes_or_pnl_opened": False,
                "paper_candidate_allowed": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--overwrite-existing", action="store_true")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
