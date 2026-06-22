"""Replay Strategy Discovery candidates through the v8 paper pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bigan.v8.strategy_discovery import (  # noqa: E402
    StrategyCandidateReplayConfig,
    StrategyDiscoveryError,
    load_strategy_candidates_jsonl,
    run_strategy_candidate_replay_batch,
)

_MODE_MAP = {
    "dry-run": "dry_run",
    "gh-command": "gh_command",
    "direct-comment": "direct_comment",
}


def run_strategy_candidate_replay_cli(
    *,
    candidate_file: Path | str,
    output_dir: Path | str,
    repo: str,
    issue_number: int,
    mode: str = "dry-run",
    batch_id: str = "strategy_candidate_batch_001",
    duration_seconds: int = 300,
    overwrite_existing: bool = False,
) -> dict[str, object]:
    """Run a deterministic candidate replay batch and return summary fields."""

    result = run_strategy_candidate_replay_batch(
        candidates=load_strategy_candidates_jsonl(candidate_file),
        config=StrategyCandidateReplayConfig(
            batch_id=batch_id,
            output_dir=output_dir,
            repo_full_name=repo,
            issue_number=issue_number,
            post_mode=_MODE_MAP[mode],  # type: ignore[arg-type]
            duration_seconds=duration_seconds,
            overwrite_existing=overwrite_existing,
        ),
    )
    return result.console_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--issue-number", type=int, required=True)
    parser.add_argument("--batch-id", default="strategy_candidate_batch_001")
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument(
        "--mode",
        choices=tuple(_MODE_MAP),
        default="dry-run",
        help="Delivery mode. direct-comment posts via gh and must be explicit.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace an existing strategy candidate batch directory.",
    )
    args = parser.parse_args(argv)
    try:
        summary = run_strategy_candidate_replay_cli(
            candidate_file=args.candidate_file,
            output_dir=args.output_dir,
            repo=args.repo,
            issue_number=args.issue_number,
            mode=args.mode,
            batch_id=args.batch_id,
            duration_seconds=args.duration_seconds,
            overwrite_existing=args.overwrite_existing,
        )
    except (FileExistsError, StrategyDiscoveryError, ValueError) as exc:
        print(json.dumps({"status": "failed_fail_closed", "error": str(exc)}))
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
