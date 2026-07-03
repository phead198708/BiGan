"""Dataset stability report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bigan.modeling import generate_dataset_stability_report


def test_generate_dataset_stability_report_writes_json_and_markdown(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "training"
    dataset_dir.mkdir()
    manifest = {
        "dataset_version": "dataset-v1",
        "feature_columns": ["spread", "microprice", "trade_volume_1m"],
        "feature_versions": ["features-v1"],
        "label_versions": ["labels-v1"],
        "family_splits": {},
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for split, offset in (("train", 0.0), ("val", 0.1), ("test", 0.2)):
        table = pa.Table.from_pylist(
            [
                {
                    "label_profit_up_15m": idx % 2 == 0,
                    "spread": 0.01 + offset + idx / 1000,
                    "microprice": 0.50 + offset + idx / 100,
                    "trade_volume_1m": 10.0 + offset + idx,
                }
                for idx in range(4)
            ]
        )
        pq.write_table(table, dataset_dir / f"{split}.parquet")

    report = generate_dataset_stability_report(
        dataset_dir,
        tmp_path / "stability",
    )

    json_path = Path(str(report["json_path"]))
    markdown_path = Path(str(report["markdown_path"]))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "dataset_stability_report_v1"
    assert payload["split_label_summary"]["train"]["positive_rate"] == 0.5
    assert payload["core_feature_distributions"]["spread"]["test"]["count"] == 4
    assert markdown_path.exists()
    assert "Dataset Stability Report" in markdown_path.read_text(encoding="utf-8")
