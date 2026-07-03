"""Run the v8 Polymarket real-corpus retraining gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket import (  # noqa: E402
    DEFAULT_REAL_CORPUS_MODEL_VERSION,
    PolymarketRealCorpusRetrainingGateConfig,
    run_polymarket_real_corpus_retraining_gate,
)
from bigan.v8.polymarket.training import DEFAULT_POLICY_CREATED_AT  # noqa: E402


def run_polymarket_real_corpus_retraining_gate_cli(
    *,
    recorder_run_dir: Path | str,
    output_dir: Path | str,
    run_id: str = "polymarket_real_corpus_retraining_gate",
    model_version: str = DEFAULT_REAL_CORPUS_MODEL_VERSION,
    created_at: str = DEFAULT_POLICY_CREATED_AT,
    overwrite_existing: bool = False,
) -> dict:
    result = run_polymarket_real_corpus_retraining_gate(
        PolymarketRealCorpusRetrainingGateConfig(
            recorder_run_dir=recorder_run_dir,
            output_dir=output_dir,
            run_id=run_id,
            model_version=model_version,
            created_at=created_at,
            overwrite_existing=overwrite_existing,
        )
    )
    report = result.report
    return {
        "run_id": report["run_id"],
        "run_dir": str(result.run_dir),
        "gate_status": report["gate_status"],
        "accepted_for_retraining": report["accepted_for_retraining"],
        "training_completed": report["training_completed"],
        "reason_codes": report["reason_codes"],
        "gate_report_path": str(result.artifact_paths["gate_report"]),
        "training_manifest_path": str(result.artifact_paths["gate_manifest"]),
        "model_manifest_path": report["model_manifest_path"],
        "model_manifest_sha256": report["model_manifest_sha256"],
        "model_sha256": report["model_sha256"],
        "policy_dataset_hash": report["policy_dataset_hash"],
        "split_hash": report["split_hash"],
        "train_dataset_hash": report["train_dataset_hash"],
        "shadow_dataset_hash": report["shadow_dataset_hash"],
        "real_historical_corpus_used": report["real_historical_corpus_used"],
        "fixture_corpus_used": report["fixture_corpus_used"],
        "synthetic_corpus_used": report["synthetic_corpus_used"],
        "synthetic_fixture_signal_used": report["synthetic_fixture_signal_used"],
        "fixture_model_used": report["fixture_model_used"],
        "manual_live_evidence_eligible": report["manual_live_evidence_eligible"],
        "paper_only": report["paper_only"],
        "capital_at_risk": report["capital_at_risk"],
        "polymarket_write_enabled": report["polymarket_write_enabled"],
        "wallet_signing_enabled": report["wallet_signing_enabled"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="polymarket_real_corpus_retraining_gate")
    parser.add_argument("--model-version", default=DEFAULT_REAL_CORPUS_MODEL_VERSION)
    parser.add_argument("--created-at", default=DEFAULT_POLICY_CREATED_AT)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)

    summary = run_polymarket_real_corpus_retraining_gate_cli(
        recorder_run_dir=args.recorder_run_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_version=args.model_version,
        created_at=args.created_at,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
