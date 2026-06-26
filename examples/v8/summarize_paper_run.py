"""Summarize a v8 paper run into operator observability artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bigan.v8.paper import (  # noqa: E402
    DEFAULT_OBSERVABILITY_CREATED_AT,
    PaperObservabilityThresholds,
    summarize_paper_run,
)


def summarize_paper_run_cli(
    *,
    run_dir: Path | str,
    output_dir: Path | str,
    compare_run_dir: Path | str | None = None,
    overwrite_existing: bool = False,
) -> dict[str, object]:
    """Write paper observability artifacts and return a console summary."""

    result = summarize_paper_run(
        run_dir=run_dir,
        output_dir=output_dir,
        thresholds=PaperObservabilityThresholds(),
        created_at=DEFAULT_OBSERVABILITY_CREATED_AT,
        compare_run_dir=compare_run_dir,
        overwrite_existing=overwrite_existing,
    )
    report = result.report
    return {
        "run_id": report.run_id,
        "phase6_deployment_status": report.phase6_status,
        "feed_health_status": report.feed_health_status,
        "alert_count": report.alert_count,
        "critical_alert_count": report.alert_severity_counts["critical"],
        "operator_recommendation": report.operator_recommendation,
        "operator_summary_path": str(result.artifact_paths["operator_summary"]),
        "observability_report_path": str(
            result.artifact_paths["observability_report"]
        ),
        "observability_report_sha256": _sha256_file(
            result.artifact_paths["observability_report"]
        ),
        "dashboard_summary_path": str(result.artifact_paths["dashboard_summary"]),
        "alerts_path": str(result.artifact_paths["alerts"]),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compare-run-dir",
        type=Path,
        default=None,
        help="Optional second paper run directory for comparison outputs.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace an existing observability output bundle.",
    )
    args = parser.parse_args(argv)
    summary = summarize_paper_run_cli(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        compare_run_dir=args.compare_run_dir,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
