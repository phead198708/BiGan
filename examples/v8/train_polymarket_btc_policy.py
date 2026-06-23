"""Train a deterministic paper-only Polymarket BTC Up/Down policy fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.corpus import (  # noqa: E402
    PolymarketCorpusBuildConfig,
    build_polymarket_btc_corpus,
    write_deterministic_polymarket_corpus_fixtures,
)
from bigan.v8.polymarket.training import (  # noqa: E402
    DEFAULT_POLICY_CREATED_AT,
    PolymarketPolicyTrainingConfig,
    run_polymarket_policy_training,
)


def run_polymarket_policy_training_cli(
    *,
    output_dir: Path | str,
    corpus_dir: Path | str | None = None,
    created_at: str = DEFAULT_POLICY_CREATED_AT,
    overwrite_existing: bool = False,
    generate_fixture_corpus: bool = True,
) -> dict:
    resolved_output = Path(output_dir).expanduser().resolve()
    if corpus_dir is None:
        corpus_root = resolved_output / "fixture_corpus"
        raw_dir = corpus_root / "raw"
        generated_corpus_dir = corpus_root / "corpus"
        if generate_fixture_corpus:
            write_deterministic_polymarket_corpus_fixtures(raw_dir)
            build_polymarket_btc_corpus(
                PolymarketCorpusBuildConfig(
                    input_dir=raw_dir,
                    output_dir=generated_corpus_dir,
                    overwrite_existing=overwrite_existing,
                )
            )
        resolved_corpus = generated_corpus_dir
    else:
        resolved_corpus = Path(corpus_dir).expanduser().resolve()
    result = run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=resolved_corpus,
            output_dir=resolved_output / "policy_runs",
            created_at=created_at,
            overwrite_existing=overwrite_existing,
        )
    )
    return {
        "run_dir": str(result.run_dir),
        "model_manifest_path": str(result.artifact_paths["model_manifest"]),
        "calibration_report_path": str(result.artifact_paths["calibration_report"]),
        "validation_report_path": str(result.artifact_paths["validation_report"]),
        "ev_threshold_report_path": str(result.artifact_paths["ev_threshold_report"]),
        "replay_report_path": str(result.artifact_paths["replay_report"]),
        "market_families": result.model_manifest["market_families"],
        "train_row_count": result.model_manifest["train_row_count"],
        "validation_row_count": result.model_manifest["validation_row_count"],
        "shadow_row_count": result.model_manifest["shadow_row_count"],
        "trained_model_used": result.model_manifest["trained_model_used"],
        "synthetic_fixture_signal_used": result.model_manifest["synthetic_fixture_signal_used"],
        "paper_only": result.model_manifest["paper_only"],
        "capital_at_risk": result.model_manifest["capital_at_risk"],
        "polymarket_write_enabled": result.model_manifest["polymarket_write_enabled"],
        "wallet_signing_enabled": result.model_manifest["wallet_signing_enabled"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corpus-dir")
    parser.add_argument("--created-at", default=DEFAULT_POLICY_CREATED_AT)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--no-generate-fixture-corpus",
        action="store_true",
        help="Use an existing corpus directory instead of generating deterministic fixtures.",
    )
    args = parser.parse_args(argv)
    summary = run_polymarket_policy_training_cli(
        output_dir=args.output_dir,
        corpus_dir=args.corpus_dir,
        created_at=args.created_at,
        overwrite_existing=args.overwrite_existing,
        generate_fixture_corpus=not args.no_generate_fixture_corpus,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
