#!/usr/bin/env python
"""Per-family shadow comparison of xgboost-v5 (challenger) vs xgboost-v4 (champion).

Scores both calibrated models over a warehouse window, keeps the market family for
every row, and reports per-family discrimination (Brier / ROC AUC) plus per-family
edge-trigger and simulated-PnL sweeps. Use this to pick per-family edge thresholds.

Example:
  .venv/bin/python scripts/shadow_per_family_v5_vs_v4.py \
    --warehouse-dir data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z/warehouse \
    --since-ms 1779848280000 --until-ms 1779941760000 \
    --output-dir docs/reports/shadow_v5_vs_v4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.mlops.shadow import run_per_family_shadow_analysis

DEFAULT_CHAMPION_MODEL = Path(
    "data/xgboost-v4-run-20260523T103814Z/artifacts/models/xgboost-v4/model.json"
)
DEFAULT_CHAMPION_CALIB = Path(
    "data/xgboost-v4-run-20260523T103814Z/artifacts/models/"
    "xgboost-v4-selected-calibration/calibration.json"
)
DEFAULT_CHALLENGER_MODEL = Path(
    "data/model-runs/xgboost-v5-run-20260529T053000Z/model/model.json"
)
DEFAULT_CHALLENGER_CALIB = Path(
    "data/model-runs/xgboost-v5-run-20260529T053000Z/calibration-family/calibration.json"
)


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_markdown(result: dict) -> str:
    champ = result["champion_model_version"]
    chal = result["challenger_model_version"]
    lines = [
        "# Per-Family Shadow: v5 (challenger) vs v4 (champion)",
        "",
        f"- Champion: `{champ}`",
        f"- Challenger: `{chal}`",
        f"- Window: {result['window_start_ts']} .. {result['window_end_ts']}",
        f"- Scored rows: {result['row_count']}",
        "",
        "## Discrimination by family (labelled rows)",
        "| Family | Rows | Labelled | Pos rate | v4 Brier | v5 Brier | v4 AUC | v5 AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    families = result["families"]
    order = sorted(families) + ["all"]
    summaries = {**families, "all": result["all"]}
    for name in order:
        s = summaries[name]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                s["row_count"],
                s["labelled_count"],
                _fmt(s["positive_rate"]),
                _fmt(s["champion_brier"]),
                _fmt(s["challenger_brier"]),
                _fmt(s["champion_roc_auc"]),
                _fmt(s["challenger_roc_auc"]),
            )
        )

    lines += [
        "",
        "## Challenger (v5) edge-trigger and simulated UP-long PnL by family",
        "| Family | Edge thr | Trigger rate | Trades | Net PnL |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in order:
        s = summaries[name]
        for threshold, detail in s["edge_thresholds"].items():
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    name,
                    threshold,
                    _fmt(detail["challenger_trigger_rate"]),
                    _fmt(detail["challenger_trade_count"]),
                    _fmt(detail["challenger_net_pnl"]),
                )
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse-dir", type=Path, required=True)
    parser.add_argument("--champion-model-path", type=Path, default=DEFAULT_CHAMPION_MODEL)
    parser.add_argument("--champion-calibration-path", type=Path, default=DEFAULT_CHAMPION_CALIB)
    parser.add_argument("--challenger-model-path", type=Path, default=DEFAULT_CHALLENGER_MODEL)
    parser.add_argument(
        "--challenger-calibration-path", type=Path, default=DEFAULT_CHALLENGER_CALIB
    )
    parser.add_argument("--since-ms", type=int, default=None)
    parser.add_argument("--until-ms", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--edge-thresholds",
        type=str,
        default="0.02,0.03,0.05,0.08",
        help="Comma-separated edge thresholds for the per-family sweep.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    thresholds = tuple(float(value) for value in args.edge_thresholds.split(",") if value)
    result = run_per_family_shadow_analysis(
        warehouse_dir=args.warehouse_dir,
        champion_model_path=args.champion_model_path,
        challenger_model_path=args.challenger_model_path,
        champion_calibration_path=args.champion_calibration_path,
        challenger_calibration_path=args.challenger_calibration_path,
        since_ms=args.since_ms,
        until_ms=args.until_ms,
        limit=args.limit,
        edge_thresholds=thresholds,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "per_family_shadow.json"
    md_path = output_dir / "per_family_shadow.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(result) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
