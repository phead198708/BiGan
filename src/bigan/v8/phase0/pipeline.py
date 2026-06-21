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
    MARKET_DATA_SCHEMA,
    PHASE0_DATASET_VERSION,
    DatasetContract,
    FeatureVector,
    Label,
    MarketData,
)
from bigan.v8.phase0.costs import (
    CostCalibrationBucketConfig,
    CostCalibrationBucketReport,
    CostCalibrationConfig,
    CostModelConfig,
    ExecutionCostSample,
    TradingCostModel,
)
from bigan.v8.phase0.features import CausalFeatureBuilder, CausalFeatureBuilderConfig
from bigan.v8.phase0.labels import CostAwareLabelBuilder, CostAwareLabelBuilderConfig
from bigan.v8.phase0.loader import MarketDataLoader
from bigan.v8.phase0.validation import (
    IntegrityValidator,
    ValidationConfig,
    ValidationFailure,
    ValidationReport,
)


@dataclass(frozen=True, slots=True)
class Phase0PipelineConfig:
    """Configuration for reproducible Phase 0 dataset builds."""

    feature_config: CausalFeatureBuilderConfig = CausalFeatureBuilderConfig()
    label_config: CostAwareLabelBuilderConfig = CostAwareLabelBuilderConfig()
    cost_config: CostModelConfig = CostModelConfig()
    validation_config: ValidationConfig = ValidationConfig()
    fail_on_validation_error: bool = True
    require_cost_calibration: bool = False
    cost_calibration_config: CostCalibrationConfig = CostCalibrationConfig()
    cost_calibration_bucket_config: CostCalibrationBucketConfig = CostCalibrationBucketConfig()

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_config": asdict(self.feature_config),
            "label_config": asdict(self.label_config),
            "cost_config": self.cost_config.to_dict(),
            "validation_config": asdict(self.validation_config),
            "fail_on_validation_error": self.fail_on_validation_error,
            "require_cost_calibration": self.require_cost_calibration,
            "cost_calibration_config": asdict(self.cost_calibration_config),
            "cost_calibration_bucket_config": asdict(self.cost_calibration_bucket_config),
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
        cost_calibration_samples: list[ExecutionCostSample] | None = None,
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
        contract = DatasetContract(
            dataset_version=PHASE0_DATASET_VERSION,
            dataset_hash=dataset_hash,
            market_schema=tuple(MARKET_DATA_SCHEMA.names),
            feature_schema=tuple(FEATURE_VECTOR_SCHEMA.names),
            label_schema=tuple(LABEL_SCHEMA.names),
            metadata={
                "market_rows": len(market_data),
                "feature_rows": len(features),
                "label_rows": len(labels),
            },
        )
        validation_report = self.validator.validate_all(
            features=features,
            labels=labels,
            market_data=market_data,
            dataset_hash=dataset_hash,
        )
        cost_calibration = (
            None
            if cost_calibration_samples is None
            else self.cost_model.validate_calibration_by_bucket(
                cost_calibration_samples,
                bucket_config=self.config.cost_calibration_bucket_config,
                config=self.config.cost_calibration_config,
            )
        )
        _merge_cost_calibration_validation(
            validation_report,
            require_cost_calibration=self.config.require_cost_calibration,
            cost_calibration=cost_calibration,
        )
        validation_payload = validation_report.to_dict()
        manifest = {
            "dataset_version": PHASE0_DATASET_VERSION,
            "dataset_hash": dataset_hash,
            "market_rows": len(market_data),
            "feature_rows": len(features),
            "label_rows": len(labels),
            "feature_columns": list(FEATURE_COLUMNS),
            "dataset_contract": contract.to_dict(),
            "config": self.config.to_dict(),
            "validation": validation_payload,
        }
        if cost_calibration is not None:
            manifest["cost_calibration"] = cost_calibration.to_dict()
        dataset = Phase0Dataset(
            market_data=list(market_data),
            features=features,
            labels=labels,
            validation_report=validation_report,
            manifest=manifest,
        )
        if self.config.fail_on_validation_error and not validation_payload["passed"]:
            failures = "; ".join(
                f"{failure.code}: {failure.message}"
                for failure in validation_report.failures
            )
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


def _merge_cost_calibration_validation(
    validation_report: ValidationReport,
    *,
    require_cost_calibration: bool,
    cost_calibration: CostCalibrationBucketReport | None,
) -> None:
    validation_report.metrics["cost_calibration_required"] = require_cost_calibration
    if cost_calibration is not None:
        validation_report.metrics["cost_calibration_passed"] = cost_calibration.passed
        validation_report.metrics["cost_calibration_checked_sample_ratio"] = (
            cost_calibration.checked_sample_ratio
        )
        validation_report.metrics["cost_calibration_checked_bucket_count"] = (
            cost_calibration.checked_bucket_count
        )

    if cost_calibration is None:
        if require_cost_calibration:
            validation_report.failures.append(
                ValidationFailure(
                    code="cost_calibration_missing",
                    message="required cost calibration samples were not provided",
                )
            )
        return

    message_prefix = "required " if require_cost_calibration else ""
    if not cost_calibration.aggregate.passed:
        validation_report.failures.append(
            ValidationFailure(
                code="cost_calibration_failed",
                message=f"{message_prefix}aggregate cost calibration failed",
                row_count=cost_calibration.aggregate.sample_count,
            )
        )
    if cost_calibration.failed_buckets:
        validation_report.failures.append(
            ValidationFailure(
                code="cost_calibration_bucket_failed",
                message=f"{message_prefix}bucketed cost calibration failed",
                row_count=len(cost_calibration.failed_buckets),
                column="cost_calibration.failed_buckets",
            )
        )
    if not cost_calibration.coverage_passed:
        reason_text = ", ".join(cost_calibration.coverage_failure_reasons)
        validation_report.failures.append(
            ValidationFailure(
                code="cost_calibration_coverage_failed",
                message=f"{message_prefix}bucket coverage failed: {reason_text}",
                row_count=cost_calibration.skipped_sample_count,
                column="cost_calibration.coverage_failure_reasons",
            )
        )
