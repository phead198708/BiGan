"""Run the single-use BTC 15m MoE v2 confirmatory evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.moe_confirmatory_evaluation import (  # noqa: E402
    run_exact_confirmatory_evaluation,
)

DEFAULT_CONFIG = ROOT / (
    "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-dir",
        default=str(DEFAULT_CONFIG / "confirmatory_collection_freeze_001"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_CONFIG / "confirmatory_evaluation_001"),
    )
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--authorization-text", required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--provider-attempts", type=int, default=3)
    args = parser.parse_args()

    def progress(attempt: int, settled: int, remaining_in_pass: int) -> None:
        print(
            f"provider_attempt={attempt} settled={settled} "
            f"remaining_in_pass={remaining_in_pass}",
            flush=True,
        )

    result = run_exact_confirmatory_evaluation(
        freeze_dir=args.freeze_dir,
        output_dir=args.output_dir,
        implementation_commit=args.implementation_commit,
        authorization_text=args.authorization_text,
        repository_root=ROOT,
        max_workers=args.max_workers,
        provider_attempts=args.provider_attempts,
        progress_callback=progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
