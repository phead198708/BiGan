"""Build a deterministic local Polymarket BTC UP/DOWN corpus."""

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
    DEFAULT_CORPUS_CREATED_AT,
    PolymarketCorpusBuildConfig,
    build_polymarket_btc_corpus,
    write_deterministic_polymarket_corpus_fixtures,
)


def run_polymarket_btc_corpus_cli(
    *,
    output_dir: Path | str,
    input_dir: Path | str | None = None,
    created_at: str = DEFAULT_CORPUS_CREATED_AT,
    overwrite_existing: bool = False,
    generate_fixture_inputs: bool = True,
) -> dict:
    resolved_output = Path(output_dir)
    resolved_input = (
        Path(input_dir) if input_dir is not None else resolved_output / "raw_fixture_inputs"
    )
    if generate_fixture_inputs:
        write_deterministic_polymarket_corpus_fixtures(resolved_input)
    result = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=resolved_input,
            output_dir=resolved_output / "corpus",
            created_at=created_at,
            overwrite_existing=overwrite_existing,
        )
    )
    return {
        "output_dir": str(result.output_dir),
        "corpus_manifest_path": str(result.artifact_paths["corpus_manifest"]),
        "corpus_summary_path": str(result.artifact_paths["corpus_summary"]),
        "market_count": result.manifest["market_count"],
        "feature_row_count": result.manifest["feature_row_count"],
        "label_row_count": result.manifest["label_row_count"],
        "market_family_counts": result.manifest["market_family_counts"],
        "paper_only": result.manifest["paper_only"],
        "capital_at_risk": result.manifest["capital_at_risk"],
        "polymarket_write_enabled": result.manifest["polymarket_write_enabled"],
        "wallet_signing_enabled": result.manifest["wallet_signing_enabled"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-dir")
    parser.add_argument("--created-at", default=DEFAULT_CORPUS_CREATED_AT)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--no-generate-fixtures",
        action="store_true",
        help="Use an existing local input directory instead of generating fixtures.",
    )
    args = parser.parse_args(argv)
    summary = run_polymarket_btc_corpus_cli(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        created_at=args.created_at,
        overwrite_existing=args.overwrite_existing,
        generate_fixture_inputs=not args.no_generate_fixtures,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
