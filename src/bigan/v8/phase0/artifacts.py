"""Runtime artifact gate for Phase 0 datasets."""

from __future__ import annotations

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
)

MANDATORY_ACCEPTANCE_CRITERIA: tuple[str, ...] = (
    "zero_detectable_leakage",
    "feature_causality_strictly_enforced",
    "label_correctness_verified",
    "statistical_validity_verified",
    "cost_model_realistic",
    "dataset_reproducible",
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


def assert_phase0_artifact_ready(manifest: Mapping[str, Any]) -> DatasetContract:
    """Raise unless a Phase 0 artifact manifest is safe to consume."""

    report = Phase0ArtifactGate().validate_manifest(manifest)
    report.raise_if_failed()
    if report.contract is None:
        raise Phase0ArtifactError("Phase 0 artifact contract was not parsed")
    return report.contract


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
