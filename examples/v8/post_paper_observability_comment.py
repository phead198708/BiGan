"""Build or post a GitHub comment for v8 paper observability results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bigan.v8.paper import (  # noqa: E402
    DEFAULT_ALERT_DELIVERY_CREATED_AT,
    GitHubCommentDeliveryConfig,
    deliver_github_paper_comment,
)

_MODE_MAP = {
    "dry-run": "dry_run",
    "gh-command": "gh_command",
    "direct-comment": "direct_comment",
}


def post_paper_observability_comment_cli(
    *,
    observability_dir: Path | str,
    repo: str,
    issue_number: int,
    output_dir: Path | str,
    mode: str = "dry-run",
    overwrite_existing: bool = False,
) -> dict[str, object]:
    """Write deterministic GitHub paper comment outputs."""

    post_mode = _MODE_MAP[mode]
    result = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=GitHubCommentDeliveryConfig(
            repo_full_name=repo,
            issue_number=issue_number,
            output_dir=output_dir,
            post_mode=post_mode,
            created_at=DEFAULT_ALERT_DELIVERY_CREATED_AT,
            overwrite_existing=overwrite_existing,
        ),
    )
    payload = result.payload
    return {
        "run_id": payload.run_id,
        "issue_number": payload.issue_number,
        "operator_recommendation": payload.operator_recommendation,
        "critical_alert_count": payload.critical_alert_count,
        "phase6_deployment_status": payload.phase6_deployment_status,
        "comment_body_path": str(result.artifact_paths["comment_body"]),
        "payload_path": str(result.artifact_paths["payload"]),
        "gh_command_path": (
            None
            if "gh_command" not in result.artifact_paths
            else str(result.artifact_paths["gh_command"])
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observability-dir", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--issue-number", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=tuple(_MODE_MAP),
        default="dry-run",
        help="Delivery mode. direct-comment posts via gh and must be explicit.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace an existing comment payload output directory.",
    )
    args = parser.parse_args(argv)
    summary = post_paper_observability_comment_cli(
        observability_dir=args.observability_dir,
        repo=args.repo,
        issue_number=args.issue_number,
        output_dir=args.output_dir,
        mode=args.mode,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
