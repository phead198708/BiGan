"""Dataset distribution and stability reports for promotion evidence."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .dataset import SPLITS

DATASET_STABILITY_SCHEMA_VERSION = "dataset_stability_report_v1"
DEFAULT_CORE_FEATURES: tuple[str, ...] = ("spread", "microprice", "trade_volume_1m")


def generate_dataset_stability_report(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    core_features: tuple[str, ...] = DEFAULT_CORE_FEATURES,
) -> dict[str, Any]:
    """Write JSON and Markdown checks for label and core-feature stability."""

    dataset_path = Path(dataset_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest(dataset_path)
    rows_by_split = {
        split: pq.read_table(dataset_path / f"{split}.parquet").to_pylist()
        for split in SPLITS
    }
    label_summary = {
        split: _label_summary(rows)
        for split, rows in rows_by_split.items()
    }
    feature_distributions = {
        feature: {
            split: _distribution([row.get(feature) for row in rows])
            for split, rows in rows_by_split.items()
        }
        for feature in core_features
    }
    drift_vs_train = {
        "label_positive_rate_abs_diff": {
            split: _abs_diff(
                label_summary["train"]["positive_rate"],
                label_summary[split]["positive_rate"],
            )
            for split in ("val", "test")
        },
        "feature_mean_relative_diff": {
            feature: {
                split: _relative_diff(
                    feature_distributions[feature]["train"]["mean"],
                    feature_distributions[feature][split]["mean"],
                )
                for split in ("val", "test")
            }
            for feature in core_features
        },
    }
    report = {
        "schema_version": DATASET_STABILITY_SCHEMA_VERSION,
        "generated_at_ms": int(time.time() * 1_000),
        "dataset_dir": str(dataset_path),
        "dataset_version": manifest.get("dataset_version"),
        "core_features": list(core_features),
        "feature_columns": manifest.get("feature_columns", []),
        "label_versions": manifest.get("label_versions", []),
        "feature_versions": manifest.get("feature_versions", []),
        "split_label_summary": label_summary,
        "core_feature_distributions": feature_distributions,
        "drift_vs_train": drift_vs_train,
        "family_splits": manifest.get("family_splits", {}),
    }
    json_path = target / "dataset_stability_report.json"
    markdown_path = target / "dataset_stability_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return {
        **report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def _read_manifest(dataset_path: Path) -> dict[str, Any]:
    manifest_path = dataset_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset manifest must be a JSON object: {manifest_path}")
    return payload


def _label_summary(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    labels = [_label_value(row) for row in rows]
    positives = sum(1 for value in labels if value is True)
    negatives = sum(1 for value in labels if value is False)
    row_count = len(labels)
    return {
        "row_count": row_count,
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": None if row_count == 0 else positives / row_count,
    }


def _label_value(row: dict[str, Any]) -> bool:
    value = row.get("label_profit_up_15m")
    if value is None:
        value = row.get("label_up_15m")
    return bool(value)


def _distribution(values: list[Any]) -> dict[str, int | float | None]:
    cleaned = sorted(value for value in (_as_float(value) for value in values) if value is not None)
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
        }
    mean = sum(cleaned) / len(cleaned)
    variance = sum((value - mean) ** 2 for value in cleaned) / len(cleaned)
    return {
        "count": len(cleaned),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": cleaned[0],
        "p25": _quantile(cleaned, 0.25),
        "p50": _quantile(cleaned, 0.50),
        "p75": _quantile(cleaned, 0.75),
        "max": cleaned[-1],
    }


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _quantile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _abs_diff(reference: Any, current: Any) -> float | None:
    ref = _as_float(reference)
    cur = _as_float(current)
    if ref is None or cur is None:
        return None
    return abs(cur - ref)


def _relative_diff(reference: Any, current: Any) -> float | None:
    ref = _as_float(reference)
    cur = _as_float(current)
    if ref is None or cur is None:
        return None
    denominator = max(abs(ref), 1e-12)
    return abs(cur - ref) / denominator


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Stability Report",
        "",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Dataset version: `{report.get('dataset_version')}`",
        "",
        "## Label Positive Rate",
        "",
        "| Split | Rows | Positive rate |",
        "|---|---:|---:|",
    ]
    label_summary = report["split_label_summary"]
    for split in SPLITS:
        summary = label_summary[split]
        lines.append(
            f"| {split} | {summary['row_count']} | {_format_optional(summary['positive_rate'])} |"
        )
    lines.extend(["", "## Core Feature Distributions", ""])
    feature_distributions = report["core_feature_distributions"]
    for feature in report["core_features"]:
        lines.extend(
            [
                f"### `{feature}`",
                "",
                "| Split | Count | Mean | Std | Min | P50 | Max |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in SPLITS:
            dist = feature_distributions[feature][split]
            lines.append(
                "| "
                f"{split} | {dist['count']} | {_format_optional(dist['mean'])} | "
                f"{_format_optional(dist['std'])} | {_format_optional(dist['min'])} | "
                f"{_format_optional(dist['p50'])} | {_format_optional(dist['max'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Conclusion",
            "",
            (
                "Use this report as the issue #55 distribution/stability evidence. "
                "Large split-to-train changes should be reviewed before promotion."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    parsed = _as_float(value)
    return "n/a" if parsed is None else f"{parsed:.6f}"
