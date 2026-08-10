"""Operate the frozen BTC 15m residual promotion-v1 preparation stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.challenge_development_lane import sha256_file  # noqa: E402
from bigan.v8.polymarket.residual_promotion_v1 import (  # noqa: E402
    CONFIG_DIR,
    complete_post_fit_parity,
    freeze_prospective_program,
    load_residual_promotion_runtime,
    prepare_post_fit_parity_correction,
    prepare_pre_fit_engineering_correction,
    prepare_pretraining_freeze,
    run_final_fit,
    validate_final_fit_protocol,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--created-at")
    correction = commands.add_parser("pre-fit-correction")
    correction.add_argument("--corrected-source-commit", required=True)
    correction.add_argument("--created-at")
    post_fit_correction = commands.add_parser("post-fit-parity-correction")
    post_fit_correction.add_argument("--corrected-source-commit", required=True)
    post_fit_correction.add_argument("--created-at")
    commands.add_parser("complete-parity")
    final_fit = commands.add_parser("final-fit")
    final_fit.add_argument("--source-commit", required=True)
    final_fit.add_argument(
        "--protocol",
        type=Path,
        default=CONFIG_DIR / "final_fit_protocol.json",
    )
    final_fit.add_argument("--protocol-sha256")
    freeze = commands.add_parser("freeze-prospective")
    freeze.add_argument(
        "--bundle-manifest",
        type=Path,
        default=CONFIG_DIR / "candidate_bundle/bundle_manifest.json",
    )
    freeze.add_argument("--bundle-manifest-sha256")
    freeze.add_argument(
        "--collector-implementation",
        type=Path,
        default=ROOT / "examples/v8/run_residual_promotion_v1_collector.py",
    )
    freeze.add_argument("--created-at")
    verify = commands.add_parser("verify")
    verify.add_argument(
        "--bundle-manifest",
        type=Path,
        default=CONFIG_DIR / "candidate_bundle/bundle_manifest.json",
    )
    verify.add_argument("--bundle-manifest-sha256")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        report = prepare_pretraining_freeze(
            repository_root=ROOT,
            source_commit=args.source_commit,
            created_at=args.created_at,
        )
    elif args.command == "pre-fit-correction":
        report = prepare_pre_fit_engineering_correction(
            repository_root=ROOT,
            corrected_source_commit=args.corrected_source_commit,
            created_at=args.created_at,
        )
    elif args.command == "post-fit-parity-correction":
        report = prepare_post_fit_parity_correction(
            repository_root=ROOT,
            corrected_source_commit=args.corrected_source_commit,
            created_at=args.created_at,
        )
    elif args.command == "complete-parity":
        report = complete_post_fit_parity(repository_root=ROOT)
    elif args.command == "final-fit":
        protocol_sha = args.protocol_sha256 or sha256_file(args.protocol)
        report = run_final_fit(
            protocol_path=args.protocol,
            expected_protocol_sha256=protocol_sha,
            repository_root=ROOT,
            source_commit=args.source_commit,
        )
    elif args.command == "freeze-prospective":
        manifest_sha = args.bundle_manifest_sha256 or sha256_file(
            args.bundle_manifest
        )
        report = freeze_prospective_program(
            repository_root=ROOT,
            bundle_manifest_path=args.bundle_manifest,
            expected_bundle_manifest_sha256=manifest_sha,
            collector_implementation_path=args.collector_implementation,
            created_at=args.created_at,
        )
    else:
        manifest_sha = args.bundle_manifest_sha256 or sha256_file(
            args.bundle_manifest
        )
        runtime = load_residual_promotion_runtime(
            manifest_path=args.bundle_manifest,
            expected_manifest_sha256=manifest_sha,
            repository_root=ROOT,
        )
        protocol = json.loads(
            (CONFIG_DIR / "final_fit_protocol.json").read_text(encoding="utf-8")
        )
        validate_final_fit_protocol(protocol, repository_root=ROOT)
        report = {
            "verification_passed": True,
            "candidate_id": runtime.candidate_id,
            "lineage_id": runtime.lineage_id,
            "manifest_sha256": runtime.manifest_sha256,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
