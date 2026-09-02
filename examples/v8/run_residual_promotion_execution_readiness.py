#!/usr/bin/env python3
"""Generate non-authorizing execution engineering readiness evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.residual_promotion_execution_readiness import (
    CONFIG_REPOSITORY_PATH,
    build_execution_readiness_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(CONFIG_REPOSITORY_PATH) / "execution_engineering_readiness_report.json",
    )
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = build_execution_readiness_report(
        repository_root=root,
        output_path=output,
        created_at=args.created_at,
    )
    print(f"report={output}")
    print(f"engineering_readiness_passed={report['engineering_readiness_passed']}")
    print(f"security_review_passed={report['security_review_passed']}")
    print(f"paper_run_started={report['paper_run_started']}")
    print(f"micro_live_authorized={report['micro_live_authorized']}")


if __name__ == "__main__":
    main()
