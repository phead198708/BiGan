"""Run the preregistered BTC 15m rolling-origin OOF diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.challenge_model_15m_training import (  # noqa: E402
    run_challenge_model_15m_rolling_origin_oof,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--created-at")
    args = parser.parse_args()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = run_challenge_model_15m_rolling_origin_oof(
        preregistration_path=args.preregistration,
        expected_preregistration_sha256=args.preregistration_sha256,
        output_dir=args.output_dir,
        source_commit=source_commit,
        created_at=args.created_at,
        repository_root=ROOT,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
