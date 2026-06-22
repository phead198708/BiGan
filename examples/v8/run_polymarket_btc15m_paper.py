"""Run deterministic v8 Polymarket BTC 15m UP/DOWN paper adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bigan.v8.polymarket import (  # noqa: E402
    PolymarketAdapterRunConfig,
    run_polymarket_btc15m_paper_pipeline,
)

_MODE_MAP = {
    "dry-run": "dry_run",
    "gh-command": "gh_command",
    "direct-comment": "direct_comment",
}


def run_polymarket_btc15m_paper_cli(
    *,
    run_id: str,
    output_dir: Path | str,
    repo: str,
    issue_number: int,
    mode: str = "dry-run",
    overwrite_existing: bool = False,
) -> dict[str, object]:
    """Run deterministic mocked Polymarket BTC 15m adapter pipeline."""

    result = run_polymarket_btc15m_paper_pipeline(
        config=PolymarketAdapterRunConfig(
            run_id=run_id,
            output_dir=output_dir,
            repo_full_name=repo,
            issue_number=issue_number,
            comment_post_mode=_MODE_MAP[mode],
            overwrite_existing=overwrite_existing,
        )
    )
    return result.console_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", default="phead198708/BiGan")
    parser.add_argument("--issue-number", type=int, default=130)
    parser.add_argument(
        "--mode",
        choices=tuple(_MODE_MAP),
        default="dry-run",
        help="Comment mode. direct-comment posts via gh and must be explicit.",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)

    summary = run_polymarket_btc15m_paper_cli(
        run_id=args.run_id,
        output_dir=args.output_dir,
        repo=args.repo,
        issue_number=args.issue_number,
        mode=args.mode,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
