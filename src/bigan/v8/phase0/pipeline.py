"""End-to-end Phase 0 dataset generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from bigan.v8.phase0.alignment import TimeAlignmentEngine
from bigan.v8.phase0.contracts import (
    FEATURE_COLUMNS,
    FEATURE_VECTOR_SCHEMA,
    LABEL_SCHEMA,
    PHASE0_DATASET_VERSION,
    FeatureVector,
    Label,
    MarketData,
)
from bigan.v8.phase0.costs import CostModelConfig, TradingCostModel
from bigan.v8.phase0.features import CausalFeatureBuilder, CausalFeatureBuilderConfig
from bigan.v8.phase0.labels import CostAwareLabelBuilder, CostAwareLabelBuilderConfig
from bigan.v8.phase0.loader import MarketDataLoader
from bigan.v8.phase0.validation import IntegrityValidator, ValidationConfig, ValidationReport


@dataclass(frozen=True, slots=True)
class Phase0PipelineConfig:
    """Configuration for reproducible Phase 0 dataset builds."""

    feature_config: CausalFeatureBuilderConfig = CausalFeatureBuilderConfig()
    label_config: CostAwareLabelBuilderConfig = CostAwareLabelBuilderConfig()
    cost_config: CostModelConfig = CostModelConfig()
    validation_config: ValidationConfig = ValidationConfig()
    fail_on_validation_error: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_config": asdict(self.feature_config),
            "label_config": asdict(self.label_config),
            "cost_config": self.cost_config.to_dict(),
            "validation_config": asdict(self.validation_config),
            "fail_on_validation_error": self.fail_on_validation_error,
        }


@dataclass(slots=True)
class Phase0Dataset:
    """Generated Phase 0 dataset and validation manifest."""

    market_data: list[MarketData]
    features: list[FeatureVector]
    labels: list[Label]
    validation_report: ValidationReport
    manifest: dict[str, Any]

    def feature_table(self) -> pa.Table:
        return pa.Table.from_pylist(
            [feature.flat_row() for feature in self.features],
            schema=FEATURE_VECTOR_SCHEMA,
        )

    def label_table(self) -> pa.Table:
        return pa.Table.from_pylist(
            [label.to_row() for label in self.labels],
            schema=LABEL_SCHEMA,
        )

    def write(self, output_dir: Path | str) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        pq.write_table(self.feature_table(), target / "features.parquet")
        pq.write_table(self.label_table(), target / "labels.parquet")
        (target / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class Phase0Pipeline:
    """Production-grade Phase 0 data-correctness firewall."""

    def __init__(
        self,
        config: Phase0PipelineConfig | None = None,
        *,
        loader: MarketDataLoader | None = None,
    ) -> None:
        self.config = config or Phase0PipelineConfig()
        self.loader = loader or MarketDataLoader()
        self.alignment_engine = TimeAlignmentEngine()
        self.feature_builder = CausalFeatureBuilder(
            self.config.feature_config,
            alignment_engine=self.alignment_engine,
        )
        self.cost_model = TradingCostModel(self.config.cost_config)
        self.label_builder = CostAwareLabelBuilder(
            self.cost_model,
            self.config.label_config,
            alignment_engine=self.alignment_engine,
        )
        self.validator = IntegrityValidator(self.config.validation_config)

    def build(
        self,
        rows: Iterable[Mapping[str, Any]] | list[MarketData],
        *,
        output_dir: Path | str | None = None,
    ) -> Phase0Dataset:
        market_data = (
            rows
            if rows and isinstance(rows, list) and all(isinstance(row, MarketData) for row in rows)
            else self.loader.load_rows(rows)  # type: ignore[arg-type]
        )
        aligned = self.alignment_engine.align_market_data(market_data)
        features = self.feature_builder.build(aligned)
        labels = self.label_builder.build(aligned, features)
        dataset_hash = _dataset_hash(features, labels)
        validation_report = self.validator.validate_all(
            features=features,
            labels=labels,
            market_data=market_data,
            dataset_hash=dataset_hash,
        )
        manifest = {
            "dataset_version": PHASE0_DATASET_VERSION,
            "dataset_hash": dataset_hash,
            "market_rows": len(market_data),
            "feature_rows": len(features),
            "label_rows": len(labels),
            "feature_columns": list(FEATURE_COLUMNS),
            "config": self.config.to_dict(),
            "validation": validation_report.to_dict(),
        }
        dataset = Phase0Dataset(
            market_data=list(market_data),
            features=features,
            labels=labels,
            validation_report=validation_report,
            manifest=manifest,
        )
        if self.config.fail_on_validation_error and not validation_report.passed:
            failures = "; ".join(f"{failure.code}: {failure.message}" for failure in validation_report.failures)
            raise ValueError(f"Phase 0 validation failed: {failures}")
        if output_dir is not None:
            dataset.write(output_dir)
        return dataset


def _dataset_hash(features: list[FeatureVector], labels: list[Label]) -> str:
    payload = {
        "features": [
            feature.flat_row()
            for feature in sorted(
                features,
                key=lambda row: (row.source, row.instrument_id, row.decision_ts),
            )
        ],
        "labels": [
            label.to_row()
            for label in sorted(
                labels,
                key=lambda row: (row.source, row.instrument_id, row.decision_ts, row.horizon_ms),
            )
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()

