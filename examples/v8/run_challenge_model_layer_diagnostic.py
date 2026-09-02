"""Build the challenge model-layer market/feature diagnostic without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.challenge_model_layer_diagnostic import (  # noqa: E402
    build_challenge_model_layer_diagnostic,
    load_and_verify_inputs,
    render_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        default=str(
            ROOT
            / "examples/v8/polymarket_configs/challenge_model_layer_diagnostic_inputs.json"
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--source-base-commit", required=True)
    args = parser.parse_args()

    inputs_manifest, paths = load_and_verify_inputs(args.inputs, repo_root=ROOT)
    report = build_challenge_model_layer_diagnostic(
        inputs_manifest=inputs_manifest,
        paths=paths,
        source_base_commit=args.source_base_commit.strip(),
    )
    output_json = Path(args.output_json).resolve()
    output_markdown = Path(args.output_markdown).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "market_recommendation": report["market_selection"]["recommendation"],
                "report_payload_sha256": report["report_payload_sha256"],
                "training_started": report["training_started"],
                "safety": report["safety"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
