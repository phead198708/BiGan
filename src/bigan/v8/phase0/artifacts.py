"""Runtime artifact gate for Phase 0 datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from bigan.v8.phase0.contracts import (
    COST_COLUMNS,
    FEATURE_VECTOR_SCHEMA,
    LABEL_SCHEMA,
    MARKET_DATA_SCHEMA,
    PHASE0_DATASET_VERSION,
    DatasetContract,
    schema_names_hash,
)

MANDATORY_ACCEPTANCE_CRITERIA: tuple[str, ...] = (
    "zero_detectable_leakage",
    "feature_causality_strictly_enforced",
    "label_correctness_verified",
    "statistical_validity_verified",
    "cost_model_realistic",
    "dataset_reproducible",
)

COST_CALIBRATION_REQUIRED_FIELDS: tuple[str, ...] = (
    "aggregate",
    "buckets",
    "skipped_buckets",
    "failed_buckets",
    "checked_sample_count",
    "skipped_sample_count",
    "checked_sample_ratio",
    "checked_bucket_count",
    "skipped_bucket_count",
    "coverage_passed",
    "coverage_failure_reasons",
)

COST_CALIBRATION_COUNT_FIELDS: tuple[str, ...] = (
    "checked_sample_count",
    "skipped_sample_count",
    "checked_bucket_count",
    "skipped_bucket_count",
)

COST_CALIBRATION_COVERAGE_REASONS: frozenset[str] = frozenset(
    {
        "all_buckets_skipped",
        "checked_sample_ratio_below_min",
        "checked_bucket_count_below_min",
    }
)

COST_CALIBRATION_REPORT_REQUIRED_FIELDS: tuple[str, ...] = (
    "sample_count",
    "passed",
    "estimated_mean_cost",
    "observed_mean_cost",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "bias",
    "max_absolute_error",
    "weighted_mean_absolute_percentage_error",
    "median_absolute_error",
    "median_absolute_percentage_error",
    "symmetric_mean_absolute_percentage_error",
)

COST_CALIBRATION_REPORT_METRIC_FIELDS: tuple[str, ...] = (
    "estimated_mean_cost",
    "observed_mean_cost",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "bias",
    "max_absolute_error",
    "weighted_mean_absolute_percentage_error",
    "median_absolute_error",
    "median_absolute_percentage_error",
    "symmetric_mean_absolute_percentage_error",
)


class Phase0ArtifactError(RuntimeError):
    """Raised when a Phase 0 artifact is not safe to consume."""


@dataclass(frozen=True, slots=True)
class Phase0ArtifactValidationFailure:
    """One manifest validation failure."""

    code: str
    message: str
    field: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
        }


@dataclass(frozen=True, slots=True)
class Phase0ArtifactValidationReport:
    """Runtime manifest validation result."""

    failures: tuple[Phase0ArtifactValidationFailure, ...]
    contract: DatasetContract | None = None

    @property
    def passed(self) -> bool:
        return not self.failures

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        message = "; ".join(
            f"{failure.code}: {failure.message}"
            for failure in self.failures
        )
        raise Phase0ArtifactError(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": [failure.to_dict() for failure in self.failures],
            "contract": None if self.contract is None else self.contract.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class Phase0ArtifactGate:
    """Hard gate for downstream Phase 0 artifact consumption."""

    expected_dataset_version: str = PHASE0_DATASET_VERSION
    require_canonical_order: bool = True
    require_cost_calibration: bool = False
    mandatory_acceptance_criteria: tuple[str, ...] = MANDATORY_ACCEPTANCE_CRITERIA

    def validate_manifest(
        self,
        manifest: Mapping[str, Any],
    ) -> Phase0ArtifactValidationReport:
        failures: list[Phase0ArtifactValidationFailure] = []
        contract_data = manifest.get("dataset_contract")
        if not isinstance(contract_data, Mapping):
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="missing_dataset_contract",
                    message="manifest must include dataset_contract",
                    field="dataset_contract",
                )
            )
            return Phase0ArtifactValidationReport(failures=tuple(failures))

        failures.extend(_schema_failures(contract_data))
        contract: DatasetContract | None = None
        try:
            contract = DatasetContract(**dict(contract_data))
        except (TypeError, ValueError, ValidationError) as exc:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="invalid_dataset_contract",
                    message=str(exc),
                    field="dataset_contract",
                )
            )

        if contract is not None:
            failures.extend(self._validate_contract(manifest, contract))

        failures.extend(self._validate_validation_block(manifest))
        failures.extend(self._validate_cost_calibration(manifest))
        return Phase0ArtifactValidationReport(
            failures=tuple(failures),
            contract=contract,
        )

    def _validate_contract(
        self,
        manifest: Mapping[str, Any],
        contract: DatasetContract,
    ) -> list[Phase0ArtifactValidationFailure]:
        failures: list[Phase0ArtifactValidationFailure] = []
        manifest_version = manifest.get("dataset_version")
        if manifest_version is None:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="missing_manifest_dataset_version",
                    message="manifest must include dataset_version",
                    field="dataset_version",
                )
            )
        elif manifest_version != contract.dataset_version:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="manifest_dataset_version_mismatch",
                    message="manifest dataset_version does not match dataset_contract",
                    field="dataset_version",
                )
            )
        elif manifest_version != self.expected_dataset_version:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="manifest_dataset_version_mismatch",
                    message=(
                        f"expected manifest dataset_version {self.expected_dataset_version}, "
                        f"got {manifest_version}"
                    ),
                    field="dataset_version",
                )
            )
        if contract.dataset_version != self.expected_dataset_version:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="stale_dataset_version",
                    message=(
                        f"expected {self.expected_dataset_version}, "
                        f"got {contract.dataset_version}"
                    ),
                    field="dataset_contract.dataset_version",
                )
            )
        manifest_hash = manifest.get("dataset_hash")
        if not manifest_hash:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="missing_dataset_hash",
                    message="manifest must include dataset_hash",
                    field="dataset_hash",
                )
            )
        elif manifest_hash != contract.dataset_hash:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="dataset_hash_mismatch",
                    message="manifest dataset_hash does not match dataset_contract",
                    field="dataset_hash",
                )
            )
        if self.require_canonical_order:
            failures.extend(_order_failures(contract))
        failures.extend(_schema_hash_failures(contract))
        return failures

    def _validate_validation_block(
        self,
        manifest: Mapping[str, Any],
    ) -> list[Phase0ArtifactValidationFailure]:
        failures: list[Phase0ArtifactValidationFailure] = []
        validation = manifest.get("validation")
        if not isinstance(validation, Mapping):
            return [
                Phase0ArtifactValidationFailure(
                    code="missing_validation",
                    message="manifest must include validation block",
                    field="validation",
                )
            ]
        if validation.get("passed") is not True:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="validation_failed",
                    message="validation.passed must be true",
                    field="validation.passed",
                )
            )
        criteria = validation.get("acceptance_criteria")
        if not isinstance(criteria, Mapping):
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="missing_acceptance_criteria",
                    message="validation.acceptance_criteria is required",
                    field="validation.acceptance_criteria",
                )
            )
            return failures
        for criterion in self.mandatory_acceptance_criteria:
            if criterion not in criteria:
                failures.append(
                    Phase0ArtifactValidationFailure(
                        code="missing_acceptance_criterion",
                        message=f"missing mandatory criterion: {criterion}",
                        field=f"validation.acceptance_criteria.{criterion}",
                    )
                )
            elif criteria.get(criterion) is not True:
                failures.append(
                    Phase0ArtifactValidationFailure(
                        code="acceptance_criterion_failed",
                        message=f"mandatory criterion failed: {criterion}",
                        field=f"validation.acceptance_criteria.{criterion}",
                    )
                )
        return failures

    def _validate_cost_calibration(
        self,
        manifest: Mapping[str, Any],
    ) -> list[Phase0ArtifactValidationFailure]:
        calibration = manifest.get("cost_calibration")
        if not isinstance(calibration, Mapping):
            if not self.require_cost_calibration:
                return []
            return [
                Phase0ArtifactValidationFailure(
                    code="missing_cost_calibration",
                    message="manifest must include cost_calibration",
                    field="cost_calibration",
                )
            ]

        failures: list[Phase0ArtifactValidationFailure] = []
        failures.extend(_cost_calibration_structure_failures(calibration))
        if calibration.get("passed") is not True:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_failed",
                    message="cost_calibration.passed must be true",
                    field="cost_calibration.passed",
                )
            )
        failed_buckets = calibration.get("failed_buckets")
        if (
            isinstance(failed_buckets, list)
            and all(isinstance(bucket, str) for bucket in failed_buckets)
            and failed_buckets
        ):
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_bucket_failed",
                    message="cost calibration contains failed buckets",
                    field="cost_calibration.failed_buckets",
                )
            )
        coverage_passed = calibration.get("coverage_passed")
        coverage_reasons = calibration.get("coverage_failure_reasons")
        if coverage_passed is False and calibration.get("passed") is True:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_coverage_invalid",
                    message="coverage_passed cannot be false when cost_calibration.passed is true",
                    field="cost_calibration.coverage_passed",
                )
            )
        if (
            isinstance(coverage_reasons, list)
            and coverage_reasons
            and calibration.get("passed") is True
        ):
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_coverage_invalid",
                    message=(
                        "coverage_failure_reasons cannot be non-empty when "
                        "cost_calibration.passed is true"
                    ),
                    field="cost_calibration.coverage_failure_reasons",
                )
            )
        return failures


def assert_phase0_artifact_ready(
    manifest: Mapping[str, Any],
    *,
    require_cost_calibration: bool | None = None,
) -> DatasetContract:
    """Raise unless a Phase 0 artifact manifest is safe to consume."""

    if require_cost_calibration is None:
        config = manifest.get("config", {})
        require_cost_calibration = bool(
            isinstance(config, Mapping)
            and config.get("require_cost_calibration", False)
        )
    report = Phase0ArtifactGate(
        require_cost_calibration=require_cost_calibration,
    ).validate_manifest(manifest)
    report.raise_if_failed()
    if report.contract is None:
        raise Phase0ArtifactError("Phase 0 artifact contract was not parsed")
    return report.contract


def _cost_calibration_structure_failures(
    calibration: Mapping[str, Any],
) -> list[Phase0ArtifactValidationFailure]:
    failures: list[Phase0ArtifactValidationFailure] = []
    for field_name in COST_CALIBRATION_REQUIRED_FIELDS:
        if field_name not in calibration:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_missing_field",
                    message=f"cost_calibration.{field_name} is required",
                    field=f"cost_calibration.{field_name}",
                )
            )

    aggregate = calibration.get("aggregate")
    if "aggregate" in calibration and not isinstance(aggregate, Mapping):
        failures.append(
            Phase0ArtifactValidationFailure(
                code="cost_calibration_invalid_field",
                message="cost_calibration.aggregate must be an object",
                field="cost_calibration.aggregate",
            )
        )
    if isinstance(aggregate, Mapping) and aggregate.get("passed") is not True:
        failures.append(
            Phase0ArtifactValidationFailure(
                code="cost_calibration_failed",
                message="cost_calibration.aggregate.passed must be true",
                field="cost_calibration.aggregate.passed",
            )
        )
    if isinstance(aggregate, Mapping):
        failures.extend(
            _cost_calibration_report_failures(
                aggregate,
                field_prefix="cost_calibration.aggregate",
            )
        )

    buckets = calibration.get("buckets")
    if "buckets" in calibration and not isinstance(buckets, Mapping):
        failures.append(
            Phase0ArtifactValidationFailure(
                code="cost_calibration_invalid_field",
                message="cost_calibration.buckets must be an object",
                field="cost_calibration.buckets",
            )
        )
    elif isinstance(buckets, Mapping):
        for bucket_name, bucket_report in buckets.items():
            bucket_field_prefix = f"cost_calibration.buckets.{bucket_name}"
            if not isinstance(bucket_name, str):
                failures.append(
                    Phase0ArtifactValidationFailure(
                        code="cost_calibration_invalid_field",
                        message="cost_calibration.buckets keys must be strings",
                        field="cost_calibration.buckets",
                    )
                )
                continue
            if not isinstance(bucket_report, Mapping):
                failures.append(
                    Phase0ArtifactValidationFailure(
                        code="cost_calibration_invalid_field",
                        message=f"{bucket_field_prefix} must be an object",
                        field=bucket_field_prefix,
                    )
                )
                continue
            failures.extend(
                _cost_calibration_report_failures(
                    bucket_report,
                    field_prefix=bucket_field_prefix,
                )
            )
            if bucket_report.get("passed") is False:
                failures.append(
                    Phase0ArtifactValidationFailure(
                        code="cost_calibration_bucket_failed",
                        message=f"{bucket_field_prefix}.passed must be true",
                        field=f"{bucket_field_prefix}.passed",
                    )
                )

    for field_name in ("skipped_buckets", "failed_buckets"):
        value = calibration.get(field_name)
        if field_name not in calibration:
            continue
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_invalid_field",
                    message=f"cost_calibration.{field_name} must be a list of strings",
                    field=f"cost_calibration.{field_name}",
                )
            )

    for field_name in COST_CALIBRATION_COUNT_FIELDS:
        value = calibration.get(field_name)
        if field_name not in calibration:
            continue
        if type(value) is not int:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_invalid_field",
                    message=f"cost_calibration.{field_name} must be an integer",
                    field=f"cost_calibration.{field_name}",
                )
            )
        elif value < 0:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_coverage_invalid",
                    message=f"cost_calibration.{field_name} must be non-negative",
                    field=f"cost_calibration.{field_name}",
                )
            )

    checked_sample_ratio = calibration.get("checked_sample_ratio")
    if "checked_sample_ratio" in calibration:
        if (
            isinstance(checked_sample_ratio, bool)
            or not isinstance(checked_sample_ratio, (int, float))
            or not math.isfinite(float(checked_sample_ratio))
        ):
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_invalid_field",
                    message="cost_calibration.checked_sample_ratio must be numeric",
                    field="cost_calibration.checked_sample_ratio",
                )
            )
        elif not 0.0 <= float(checked_sample_ratio) <= 1.0:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_coverage_invalid",
                    message="cost_calibration.checked_sample_ratio must be in [0, 1]",
                    field="cost_calibration.checked_sample_ratio",
                )
            )

    coverage_passed = calibration.get("coverage_passed")
    if "coverage_passed" in calibration and not isinstance(coverage_passed, bool):
        failures.append(
            Phase0ArtifactValidationFailure(
                code="cost_calibration_invalid_field",
                message="cost_calibration.coverage_passed must be boolean",
                field="cost_calibration.coverage_passed",
            )
        )

    coverage_reasons = calibration.get("coverage_failure_reasons")
    if "coverage_failure_reasons" in calibration:
        if not isinstance(coverage_reasons, list) or not all(
            isinstance(reason, str) for reason in coverage_reasons
        ):
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_invalid_field",
                    message="cost_calibration.coverage_failure_reasons must be a list of strings",
                    field="cost_calibration.coverage_failure_reasons",
                )
            )
        else:
            unknown_reasons = [
                reason
                for reason in coverage_reasons
                if reason not in COST_CALIBRATION_COVERAGE_REASONS
            ]
            if unknown_reasons:
                failures.append(
                    Phase0ArtifactValidationFailure(
                        code="cost_calibration_invalid_field",
                        message=(
                            "unknown cost calibration coverage reasons: "
                            + ", ".join(sorted(unknown_reasons))
                        ),
                        field="cost_calibration.coverage_failure_reasons",
                    )
                )
    return failures


def _cost_calibration_report_failures(
    report: Mapping[str, Any],
    *,
    field_prefix: str,
) -> list[Phase0ArtifactValidationFailure]:
    failures: list[Phase0ArtifactValidationFailure] = []
    for field_name in COST_CALIBRATION_REPORT_REQUIRED_FIELDS:
        if field_name not in report:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_missing_field",
                    message=f"{field_prefix}.{field_name} is required",
                    field=f"{field_prefix}.{field_name}",
                )
            )

    sample_count = report.get("sample_count")
    if "sample_count" in report:
        if type(sample_count) is not int:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_invalid_field",
                    message=f"{field_prefix}.sample_count must be an integer",
                    field=f"{field_prefix}.sample_count",
                )
            )
        elif sample_count < 0:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_invalid_field",
                    message=f"{field_prefix}.sample_count must be non-negative",
                    field=f"{field_prefix}.sample_count",
                )
            )

    passed = report.get("passed")
    if "passed" in report and not isinstance(passed, bool):
        failures.append(
            Phase0ArtifactValidationFailure(
                code="cost_calibration_invalid_field",
                message=f"{field_prefix}.passed must be boolean",
                field=f"{field_prefix}.passed",
            )
        )

    for field_name in COST_CALIBRATION_REPORT_METRIC_FIELDS:
        value = report.get(field_name)
        if field_name not in report or value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            failures.append(
                Phase0ArtifactValidationFailure(
                    code="cost_calibration_invalid_field",
                    message=f"{field_prefix}.{field_name} must be numeric or null",
                    field=f"{field_prefix}.{field_name}",
                )
            )
    return failures


def _schema_failures(contract_data: Mapping[str, Any]) -> list[Phase0ArtifactValidationFailure]:
    failures: list[Phase0ArtifactValidationFailure] = []
    for field_name, required_names in (
        ("market_schema", tuple(MARKET_DATA_SCHEMA.names)),
        ("feature_schema", tuple(FEATURE_VECTOR_SCHEMA.names)),
        ("label_schema", tuple(LABEL_SCHEMA.names)),
    ):
        observed = contract_data.get(field_name)
        if not isinstance(observed, (list, tuple)):
            failures.append(
                Phase0ArtifactValidationFailure(
                    code=f"{field_name}_missing",
                    message=f"{field_name} must be present",
                    field=f"dataset_contract.{field_name}",
                )
            )
            continue
        missing = set(required_names) - set(observed)
        if missing:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code=f"{field_name}_missing_columns",
                    message="missing required columns: " + ", ".join(sorted(missing)),
                    field=f"dataset_contract.{field_name}",
                )
            )
    cost_columns = contract_data.get("cost_columns")
    label_schema = contract_data.get("label_schema")
    cost_source = cost_columns if isinstance(cost_columns, (list, tuple)) else ()
    label_source = label_schema if isinstance(label_schema, (list, tuple)) else ()
    missing_cost_columns = (
        (set(COST_COLUMNS) - set(cost_source))
        | (set(COST_COLUMNS) - set(label_source))
    )
    if missing_cost_columns:
        failures.append(
            Phase0ArtifactValidationFailure(
                code="missing_cost_columns",
                message="missing cost columns: " + ", ".join(sorted(missing_cost_columns)),
                field="dataset_contract.cost_columns",
            )
        )
    return failures


def _order_failures(contract: DatasetContract) -> list[Phase0ArtifactValidationFailure]:
    failures: list[Phase0ArtifactValidationFailure] = []
    for field_name, matches in (
        ("market_schema", contract.market_schema_order_matches),
        ("feature_schema", contract.feature_schema_order_matches),
        ("label_schema", contract.label_schema_order_matches),
    ):
        if not matches:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code=f"{field_name}_order_mismatch",
                    message=f"{field_name} does not match canonical ordering",
                    field=f"dataset_contract.{field_name}",
                )
            )
    return failures


def _schema_hash_failures(contract: DatasetContract) -> list[Phase0ArtifactValidationFailure]:
    failures: list[Phase0ArtifactValidationFailure] = []
    for field_name, schema_names, observed_hash in (
        ("market_schema", contract.market_schema, contract.market_schema_hash),
        ("feature_schema", contract.feature_schema, contract.feature_schema_hash),
        ("label_schema", contract.label_schema, contract.label_schema_hash),
    ):
        expected_hash = schema_names_hash(schema_names)
        if observed_hash != expected_hash:
            failures.append(
                Phase0ArtifactValidationFailure(
                    code=f"{field_name}_hash_mismatch",
                    message=f"{field_name}_hash does not match schema names",
                    field=f"dataset_contract.{field_name}_hash",
                )
            )
    return failures
