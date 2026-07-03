"""Run the #161 v8 O fresh public-data paper-only loop."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    PINNED_ISSUE_160_MANIFEST_SHA256,
    PolymarketOV8PaperFreshLoopConfig,
    run_polymarket_o_v8_paper_fresh_loop,
)

DEFAULT_ISSUE_160_UNLOCK_DIR = Path(
    "examples/v8/polymarket_runs/o-v8-paper-candidate-unlock-20260703T073000Z"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bounded paper-only v8 O fresh public-data loop."
    )
    parser.add_argument(
        "--run-id",
        default=f"o-v8-paper-fresh-loop-{_utc_stamp()}",
        help="Run id for the output bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
        help="Parent output directory.",
    )
    parser.add_argument(
        "--paper-candidate-unlock-dir",
        type=Path,
        default=DEFAULT_ISSUE_160_UNLOCK_DIR,
        help="Pinned #160 paper-candidate unlock directory.",
    )
    parser.add_argument(
        "--paper-candidate-unlock-manifest-sha256",
        default=PINNED_ISSUE_160_MANIFEST_SHA256,
        help="Expected #160 unlock manifest SHA-256.",
    )
    parser.add_argument(
        "--loop-mode",
        choices=("single_cycle", "bounded_recurring"),
        default="single_cycle",
    )
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--public-data-snapshot",
        type=Path,
        default=None,
        help=(
            "Optional read-only public data snapshot. Accepts either "
            "{'cycles': [[rows...]]} or a single list of rows."
        ),
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace an existing run directory with the same run id.",
    )
    args = parser.parse_args()

    public_cycles = (
        _load_public_data_cycles(args.public_data_snapshot)
        if args.public_data_snapshot is not None
        else None
    )
    config = PolymarketOV8PaperFreshLoopConfig(
        run_id=args.run_id,
        output_dir=args.output_dir,
        paper_candidate_unlock_dir=args.paper_candidate_unlock_dir,
        expected_paper_candidate_unlock_manifest_sha256=(
            args.paper_candidate_unlock_manifest_sha256
        ),
        loop_mode=args.loop_mode,
        max_cycles=args.max_cycles,
        sleep_seconds=args.sleep_seconds,
        public_data_cycles=public_cycles,
        overwrite_existing=args.overwrite_existing,
    )
    result = run_polymarket_o_v8_paper_fresh_loop(config)
    manifest = result.manifest
    print(f"run_id={args.run_id}")
    print(f"output_dir={result.output_dir}")
    print(f"paper_fresh_loop_enabled={manifest['paper_fresh_loop_enabled']}")
    print(f"paper_fresh_loop_mode={manifest['paper_fresh_loop_mode']}")
    print(f"paper_fresh_loop_cycle_count={manifest['paper_fresh_loop_cycle_count']}")
    print(f"paper_fresh_order_intent_count={manifest['paper_fresh_order_intent_count']}")
    print(f"paper_fresh_fill_count={manifest['paper_fresh_fill_count']}")
    print(f"paper_fresh_monitoring_passed={manifest['paper_fresh_monitoring_passed']}")
    print(
        "v8_paper_internal_handoff_allowed="
        f"{manifest['v8_paper_internal_handoff_allowed']}"
    )
    print(f"v8_execution_handoff_allowed={manifest['v8_execution_handoff_allowed']}")
    print(f"manifest={result.artifact_paths['manifest']}")
    print(f"manifest_sha256={result.artifact_hashes['manifest']}")


def _load_public_data_cycles(path: Path) -> tuple[tuple[dict, ...], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cycles = payload.get("cycles") if isinstance(payload, dict) else payload
    if not isinstance(cycles, list):
        raise ValueError("public data snapshot must contain a list of cycles or rows")
    if cycles and all(isinstance(row, dict) for row in cycles):
        return (tuple(dict(row) for row in cycles),)
    return tuple(tuple(dict(row) for row in cycle) for cycle in cycles)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    main()
