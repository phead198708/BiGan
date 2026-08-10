"""Governed prequential pooled ensemble for BTC 15m residual lineage v4."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.cost_aware_residual import (
    FOLD_SCHEMA_VERSION as V1_FOLD_SCHEMA_VERSION,
)
from bigan.v8.polymarket.cost_aware_residual import (
    LINEAGE_ID as V1_LINEAGE_ID,
)
from bigan.v8.polymarket.cost_aware_residual import (
    MARKET_RESULT_SCHEMA_VERSION as V1_MARKET_RESULT_SCHEMA_VERSION,
)
from bigan.v8.polymarket.cost_aware_residual import (
    PREDICTION_SCHEMA_VERSION as V1_PREDICTION_SCHEMA_VERSION,
)
from bigan.v8.polymarket.cost_aware_residual import (
    _descriptor,
    _load_frozen_development_rows,
    _load_json,
    _load_jsonl,
    _looks_like_git_sha,
    _public_dataset_row,
    _validate_frozen_population,
    _verified_json,
    _verify_descriptor,
    build_residual_oof_report,
    market_results_from_predictions,
    render_residual_oof_markdown,
)
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    GATE_NAMES,
    PAIR_CLIP_EPSILON,
    _action_policy,
    _bootstrap_contract,
    _cost_stress_contract,
    _power_contract,
    _probability_residual_label,
    _rolling_contract,
    pair_anchored_action_values,
)
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    _dmatrix as _residual_dmatrix,
)
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    _model_parameters as _residual_model_parameters,
)
from bigan.v8.polymarket.cost_aware_residual_v3_logit import (
    _binary_payout_label,
    _logit_model_parameters,
    logit_offset_action_values,
)
from bigan.v8.polymarket.cost_aware_residual_v3_logit import (
    _dmatrix as _logit_dmatrix,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    _write_new_frozen_json,
    _write_new_jsonl,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES
from bigan.v8.polymarket.moe_confirmatory_evaluation import _write_new_frozen_text
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.moe_terminal_diagnostic import _assert_semantically_equal
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

LINEAGE_ID = "BTC-15M-cost-aware-market-residual-v4"
PARENT_LINEAGE_ID = "BTC-15M-cost-aware-market-residual-v3"
PROTOCOL_SCHEMA_VERSION = "bigan-btc-15m-prequential-pooled-ensemble-oof-protocol-v4"
PREDICTION_SCHEMA_VERSION = "bigan-btc-15m-prequential-pooled-ensemble-oof-prediction-v4"
FOLD_SCHEMA_VERSION = "bigan-btc-15m-prequential-pooled-ensemble-oof-fold-v4"
MARKET_RESULT_SCHEMA_VERSION = "bigan-btc-15m-prequential-pooled-ensemble-market-result-v4"
REPORT_SCHEMA_VERSION = "bigan-btc-15m-prequential-pooled-ensemble-oof-report-v4"
MANIFEST_SCHEMA_VERSION = "bigan-btc-15m-prequential-pooled-ensemble-oof-manifest-v4"
DATASET_SCHEMA_VERSION = "bigan-btc-15m-prequential-pooled-ensemble-dataset-row-v4"

DEFAULT_CONFIG_DIR = (
    REPO_ROOT / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v4"
)
DEFAULT_PROTOCOL = DEFAULT_CONFIG_DIR / "residual_v4_primary_slot_001_protocol.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "residual_v4_primary_slot_001_oof"
DEFAULT_AUTHORIZATION = DEFAULT_CONFIG_DIR / "lineage_authorization.json"
DEFAULT_REGISTRY = DEFAULT_CONFIG_DIR / "development_data_registry.json"

AUTHORIZATION_INSTRUCTION = (
    "授权建立 BTC-15M-cost-aware-market-residual-v4，最多 2 个预注册候选 slot；允许使用 "
    "development_only_forever 数据进行 outcome-aware 开发。每个候选在唯一一次 OOF 前"
    "冻结协议、实际执行模块路径和 SHA；禁止网格搜索；保持既有 gates、零阈值、"
    "N_max=2000、成本、基线、人口、失败 artifacts 和全部安全状态不变；不授权 "
    "collection、shadow、paper/live、wallet、write、promotion 或 capital risk。"
)
AUTHORIZATION_INSTRUCTION_SHA256 = (
    "cbb3097160af4b02477a0576463286373be5a28856caeac781af2113564ba06e"
)
PARENT_V3_TERMINAL_SHA256 = "1a38ad26eebd795cc0a5f2746fd19b0479d85bdb57027ef6b0e3cade0bda4bae"
PARENT_V3_BINDING_AUDIT_SHA256 = "7f1e336642cebdd4fba76a68671ba33cbcc3ed896e2281fb835be7e84721bb85"
IMMUTABLE_GATE_IMPLEMENTATION = {
    "path": "src/bigan/v8/polymarket/cost_aware_residual.py",
    "sha256": "491a329f708a16d5aecdd952552cbff3fa13d8f7446bfe3e0c78fade3b36f78c",
}
IMMUTABLE_RESIDUAL_BASE_IMPLEMENTATION = {
    "path": "src/bigan/v8/polymarket/cost_aware_residual_v2.py",
    "sha256": "e9f570667c4d7fcc4e477bbfbd26ba1161aadb56bbeb7be096dad4e3338b4947",
}
IMMUTABLE_LOGIT_BASE_IMPLEMENTATION = {
    "path": "src/bigan/v8/polymarket/cost_aware_residual_v3_logit.py",
    "sha256": "d9d0128a24acf16d483301e579e70fba3788e8056901343f68706fa28da76e1d",
}
EXPERT_NAMES = ("probability_residual", "logit_offset_binomial")
LOG_LOSS_CLIP_EPSILON = 1e-6


def validate_v4_lineage_authorization(
    *,
    authorization_path: Path | str = DEFAULT_AUTHORIZATION,
    registry_path: Path | str = DEFAULT_REGISTRY,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify the exact user authorization and immutable v3 parent boundary."""

    root = Path(repository_root).resolve()
    authorization_file = Path(authorization_path).resolve()
    registry_file = Path(registry_path).resolve()
    authorization = _verified_json(authorization_file)
    registry = _verified_json(registry_file)
    blockers: list[str] = []
    source = dict(authorization.get("authorization_source") or {})
    if authorization.get("schema_version") != (
        "bigan-btc-15m-cost-aware-residual-lineage-authorization-v4"
    ):
        blockers.append("authorization.schema_version")
    if source.get("type") != "explicit_user_instruction":
        blockers.append("authorization_source.type")
    if source.get("instruction") != AUTHORIZATION_INSTRUCTION:
        blockers.append("authorization_source.instruction")
    if _raw_text_sha256(str(source.get("instruction") or "")) != (AUTHORIZATION_INSTRUCTION_SHA256):
        blockers.append("authorization_source.instruction_sha256")
    if source.get("instruction_sha256") != AUTHORIZATION_INSTRUCTION_SHA256:
        blockers.append("authorization_source.recorded_sha256")
    if authorization.get("lineage_id") != LINEAGE_ID:
        blockers.append("authorization.lineage_id")
    if dict(authorization.get("authorization_scope") or {}) != {
        "candidate_slot_budget": {
            "maximum_total_slots": 2,
            "slot_budget_may_be_increased": False,
        },
        "actual_executing_module_path_and_sha_binding_required": True,
        "fresh_collection_authorized": False,
        "fresh_outcome_opening_authorized": False,
        "live_shadow_authorized": False,
        "outcome_aware_development_authorized": True,
        "paper_or_live_execution_authorized": False,
        "promotion_authorized": False,
        "training_may_start_only_after_slot_protocol_is_sha_frozen": True,
        "wallet_or_write_authorized": False,
    }:
        blockers.append("authorization_scope")
    if dict(authorization.get("state") or {}) != _authorization_state_contract():
        blockers.append("authorization_state")
    if dict(authorization.get("safety") or {}) != SAFETY:
        blockers.append("authorization.safety")
    parent = dict(authorization.get("parent_lineage") or {})
    if not (
        parent.get("lineage_id") == PARENT_LINEAGE_ID
        and parent.get("status") == "phase_1_terminal_failed"
        and parent.get("candidate_budget_consumed") == 2
        and parent.get("candidate_budget_maximum") == 2
        and parent.get("failed_artifacts_mutable") is False
        and parent.get("gate_or_threshold_change_allowed") is False
    ):
        blockers.append("parent_lineage")
    try:
        terminal_path = _verify_descriptor(dict(parent["terminal_review"]), repository_root=root)
        terminal = _load_json(terminal_path)
        if not (
            sha256_file(terminal_path) == PARENT_V3_TERMINAL_SHA256
            and terminal.get("lineage_id") == PARENT_LINEAGE_ID
            and terminal.get("phase_1_terminal_failed") is True
            and terminal.get("candidate_budget_exhausted") is True
            and terminal.get("candidate_selected") is None
            and terminal.get("candidate_freeze_allowed") is False
            and dict(terminal.get("safety") or {}) == SAFETY
        ):
            blockers.append("parent_terminal_semantics")
    except (KeyError, OSError, TypeError, ValueError):
        blockers.append("parent_terminal_review")
    try:
        audit_path = _verify_descriptor(
            dict(authorization["required_parent_binding_audit"]),
            repository_root=root,
        )
        audit = _load_json(audit_path)
        if not (
            sha256_file(audit_path) == PARENT_V3_BINDING_AUDIT_SHA256
            and audit.get("audit_passed") is True
            and audit.get("lineage_id") == PARENT_LINEAGE_ID
            and dict(audit.get("safety") or {}) == SAFETY
        ):
            blockers.append("parent_binding_audit_semantics")
    except (KeyError, OSError, TypeError, ValueError):
        blockers.append("parent_binding_audit")
    try:
        registered = _verify_descriptor(
            dict(authorization["registered_development_data"]),
            repository_root=root,
        )
        if registered != registry_file:
            blockers.append("registered_development_data.path")
    except (KeyError, OSError, TypeError, ValueError):
        blockers.append("registered_development_data")
    if not (
        registry.get("lineage_id") == LINEAGE_ID
        and registry.get("development_only_forever") is True
        and registry.get("promotion_evidence_eligible") is False
        and dict(registry.get("safety") or {}) == SAFETY
        and dict(registry.get("rules") or {}).get(
            "candidate_evaluation_requires_actual_executing_module_path_and_sha_binding"
        )
        is True
    ):
        blockers.append("development_data_registry")
    for name, descriptor in dict(registry.get("registered_sources") or {}).items():
        try:
            _verify_descriptor(dict(descriptor), repository_root=root)
        except (KeyError, OSError, TypeError, ValueError):
            blockers.append(f"development_data_registry.{name}")
    if blockers:
        raise ValueError("residual v4 authorization invalid: " + ", ".join(blockers))
    return {
        "authorization_valid": True,
        "lineage_id": LINEAGE_ID,
        "maximum_total_slots": 2,
        "parent_v3_immutable": True,
        "actual_executing_module_binding_required": True,
        "authorization_sha256": sha256_file(authorization_file),
        "registry_sha256": sha256_file(registry_file),
        "safety": dict(SAFETY),
    }


def require_v4_candidate_implementation_binding(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, str]:
    """Bind the declaration to this exact module, including its current bytes."""

    root = Path(repository_root).resolve()
    expected = _descriptor(Path(__file__), root)
    try:
        declared = dict(payload["inputs"]["candidate_implementation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v4 candidate implementation descriptor unavailable") from exc
    if declared != expected:
        raise ValueError("v4 candidate implementation does not identify the executing module")
    if _verify_descriptor(declared, repository_root=root) != Path(__file__).resolve():
        raise ValueError("v4 candidate implementation resolved to another module")
    return expected


def validate_residual_v4_protocol(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Fail closed unless slot 1 and its actual executing module are frozen."""

    blockers: list[str] = []
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        blockers.append("schema_version")
    if payload.get("lineage_id") != LINEAGE_ID:
        blockers.append("lineage_id")
    if payload.get("slot_id") != "residual-v4-primary-slot-001":
        blockers.append("slot_id")
    if payload.get("candidate_role") != "primary":
        blockers.append("candidate_role")
    if payload.get("development_only_forever") is not True:
        blockers.append("development_only_forever")
    if payload.get("promotion_evidence_eligible") is not False:
        blockers.append("promotion_evidence_eligible")
    if dict(payload.get("candidate_budget") or {}) != _candidate_budget_contract():
        blockers.append("candidate_budget")
    if dict(payload.get("target") or {}) != _target_contract():
        blockers.append("target")
    if dict(payload.get("pair_coherence") or {}) != _pair_contract():
        blockers.append("pair_coherence")
    if dict(payload.get("feature_contract") or {}) != _feature_contract():
        blockers.append("feature_contract")
    if dict(payload.get("model") or {}) != _model_contract():
        blockers.append("model")
    if dict(payload.get("prequential_weighting") or {}) != _weighting_contract():
        blockers.append("prequential_weighting")
    if dict(payload.get("action_policy") or {}) != _action_policy():
        blockers.append("action_policy")
    if dict(payload.get("bootstrap") or {}) != _bootstrap_contract():
        blockers.append("bootstrap")
    if dict(payload.get("cost_stress") or {}) != _cost_stress_contract():
        blockers.append("cost_stress")
    gates = dict(payload.get("gates") or {})
    if set(gates) != GATE_NAMES or any(value is not True for value in gates.values()):
        blockers.append("gates")
    if dict(payload.get("prospective_power") or {}) != _power_contract():
        blockers.append("prospective_power")
    if dict(payload.get("rolling_origin") or {}) != _rolling_contract():
        blockers.append("rolling_origin")
    if dict(payload.get("dataset") or {}) != _dataset_contract():
        blockers.append("dataset")
    if dict(payload.get("baseline") or {}) != _baseline_contract():
        blockers.append("baseline")
    if dict(payload.get("development_discipline") or {}) != _discipline_contract():
        blockers.append("development_discipline")
    if dict(payload.get("state") or {}) != _state_contract():
        blockers.append("state")
    if dict(payload.get("safety") or {}) != SAFETY:
        blockers.append("safety")
    root = Path(repository_root).resolve()
    try:
        require_v4_candidate_implementation_binding(payload, repository_root=root)
    except ValueError:
        blockers.append("candidate_implementation_exact_binding")
    inputs = dict(payload.get("inputs") or {})
    required_inputs = {
        "lineage_authorization",
        "development_data_registry",
        "parent_v3_terminal_review",
        "parent_v3_binding_audit",
        "terminal_diagnostic_scored_rows",
        "confirmatory_capture_manifest",
        "confirmatory_market_evaluation_rows",
        "baseline_decision_rows",
        "matched_global_baseline_contract",
        "parent_feature_contract",
        "parent_cost_and_action_contract",
        "raw_capture_recovery_bundle_manifest",
        "candidate_implementation",
        "residual_base_implementation",
        "logit_base_implementation",
        "gate_implementation",
    }
    if set(inputs) != required_inputs:
        blockers.append("inputs")
    if dict(inputs.get("gate_implementation") or {}) != IMMUTABLE_GATE_IMPLEMENTATION:
        blockers.append("gate_implementation")
    if dict(inputs.get("residual_base_implementation") or {}) != (
        IMMUTABLE_RESIDUAL_BASE_IMPLEMENTATION
    ):
        blockers.append("residual_base_implementation")
    if dict(inputs.get("logit_base_implementation") or {}) != (IMMUTABLE_LOGIT_BASE_IMPLEMENTATION):
        blockers.append("logit_base_implementation")
    if verify_artifacts and not blockers:
        resolved: dict[str, Path] = {}
        for name, descriptor in inputs.items():
            try:
                resolved[name] = _verify_descriptor(dict(descriptor), repository_root=root)
            except (KeyError, OSError, TypeError, ValueError):
                blockers.append(f"inputs.{name}")
        if not blockers:
            try:
                authorization = validate_v4_lineage_authorization(
                    authorization_path=resolved["lineage_authorization"],
                    registry_path=resolved["development_data_registry"],
                    repository_root=root,
                )
                if authorization["maximum_total_slots"] != 2:
                    blockers.append("lineage_authorization.slot_budget")
            except ValueError:
                blockers.append("lineage_authorization")
            terminal = _load_json(resolved["parent_v3_terminal_review"])
            if not (
                terminal.get("phase_1_terminal_failed") is True
                and terminal.get("candidate_budget_exhausted") is True
                and dict(terminal.get("safety") or {}) == SAFETY
            ):
                blockers.append("parent_v3_terminal_review")
            audit = _load_json(resolved["parent_v3_binding_audit"])
            if not (
                audit.get("audit_passed") is True and dict(audit.get("safety") or {}) == SAFETY
            ):
                blockers.append("parent_v3_binding_audit")
    if blockers:
        raise ValueError("residual v4 protocol invalid: " + ", ".join(blockers))


def run_residual_v4_rolling_origin_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the first v4 candidate exactly once after complete SHA freeze."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("residual v4 OOF paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("residual v4 protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != (
        expected_protocol_sha256
    ):
        raise ValueError("residual v4 protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_residual_v4_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"residual v4 OOF output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset_rows, baseline_by_market, population_order = _load_frozen_development_rows(
        protocol=protocol,
        repository_root=root,
    )
    predictions, folds = rolling_origin_prequential_ensemble_predict(
        rows=dataset_rows,
        population_order=population_order,
        protocol=protocol,
    )
    markets = _market_results_from_predictions(
        predictions=predictions,
        baseline_by_market=baseline_by_market,
        population_order=population_order,
        protocol=protocol,
    )
    report = _build_report(
        protocol=protocol,
        protocol_sha256=expected_protocol_sha256,
        source_commit=source_commit,
        market_results=markets,
        fold_audits=folds,
    )

    dataset_path = output / "residual_v4_development_dataset_rows.jsonl"
    prediction_path = output / "residual_v4_oof_predictions.jsonl"
    fold_path = output / "residual_v4_oof_fold_audits.jsonl"
    market_path = output / "residual_v4_oof_market_results.jsonl"
    report_path = output / "residual_v4_oof_report.json"
    markdown_path = output / "residual_v4_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_dataset_v4_row(row) for row in dataset_rows])
    _write_new_jsonl(prediction_path, predictions)
    _write_new_jsonl(fold_path, folds)
    _write_new_jsonl(market_path, markets)
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_artifact = _write_new_frozen_text(markdown_path, render_v4_markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": protocol["slot_id"],
        "candidate_role": "primary",
        "created_at": protocol["created_at"],
        "source_commit": source_commit,
        "protocol": _descriptor(protocol_file, root),
        "candidate_implementation": dict(protocol["inputs"]["candidate_implementation"]),
        "residual_base_implementation": dict(protocol["inputs"]["residual_base_implementation"]),
        "logit_base_implementation": dict(protocol["inputs"]["logit_base_implementation"]),
        "immutable_gate_implementation": dict(protocol["inputs"]["gate_implementation"]),
        "artifacts": {
            "dataset_rows": _descriptor(dataset_path, root),
            "predictions": _descriptor(prediction_path, root),
            "fold_audits": _descriptor(fold_path, root),
            "market_results": _descriptor(market_path, root),
            "report": _descriptor(Path(report_artifact["path"]), root),
            "report_markdown": _descriptor(Path(markdown_artifact["path"]), root),
        },
        "evaluation_executed_exactly_once": True,
        "remaining_candidate_slots": 1,
        "candidate_freeze_allowed": report["all_gates_passed"],
        "next_stage_authorization_required_even_if_all_gates_pass": True,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_artifact = _write_new_frozen_json(output / "residual_v4_oof_manifest.json", manifest)
    return {
        "manifest": _descriptor(Path(manifest_artifact["path"]), root),
        "report": _descriptor(Path(report_artifact["path"]), root),
        "all_gates_passed": report["all_gates_passed"],
        "failed_gates": report["failed_gates"],
        "remaining_candidate_slots": 1,
        "oof_market_count": len(markets),
        "next_stage_authorization_required": True,
        "safety": dict(SAFETY),
    }


def verify_frozen_residual_v4_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify hashes, implementation binding, population, and report rebuild."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    protocol = _verified_json(protocol_file)
    validate_residual_v4_protocol(protocol, repository_root=root)
    manifest = _verified_json(output / "residual_v4_oof_manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("residual v4 manifest schema mismatch")
    if manifest.get("protocol") != _descriptor(protocol_file, root):
        raise ValueError("residual v4 manifest protocol binding mismatch")
    if manifest.get("candidate_implementation") != require_v4_candidate_implementation_binding(
        protocol, repository_root=root
    ):
        raise ValueError("residual v4 manifest implementation binding mismatch")
    artifacts = {
        name: _verify_descriptor(dict(descriptor), repository_root=root)
        for name, descriptor in dict(manifest.get("artifacts") or {}).items()
    }
    if set(artifacts) != {
        "dataset_rows",
        "predictions",
        "fold_audits",
        "market_results",
        "report",
        "report_markdown",
    }:
        raise ValueError("residual v4 manifest artifact set mismatch")
    dataset = _load_jsonl(artifacts["dataset_rows"])
    predictions = _load_jsonl(artifacts["predictions"])
    folds = _load_jsonl(artifacts["fold_audits"])
    markets = _load_jsonl(artifacts["market_results"])
    if len(dataset) != 3200 or any(
        row.get("schema_version") != DATASET_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or row.get("development_only_forever") is not True
        or row.get("promotion_evidence_eligible") is not False
        or dict(row.get("safety") or {}) != SAFETY
        for row in dataset
    ):
        raise ValueError("residual v4 dataset governance mismatch")
    _validate_population(
        predictions=predictions,
        fold_audits=folds,
        market_results=markets,
        protocol=protocol,
    )
    rebuilt = _build_report(
        protocol=protocol,
        protocol_sha256=sha256_file(protocol_file),
        source_commit=str(manifest["source_commit"]),
        market_results=markets,
        fold_audits=folds,
    )
    frozen = _load_json(artifacts["report"])
    _assert_semantically_equal(rebuilt, frozen, path="residual_v4_oof_report")
    if render_v4_markdown(rebuilt) != artifacts["report_markdown"].read_text(encoding="utf-8"):
        raise ValueError("residual v4 Markdown does not reproduce")
    return {
        "verification_passed": True,
        "all_gates_passed": bool(frozen["all_gates_passed"]),
        "failed_gates": list(frozen["failed_gates"]),
        "remaining_candidate_slots": 1,
        "oof_market_count": len(markets),
        "manifest_sha256": sha256_file(output / "residual_v4_oof_manifest.json"),
        "actual_executing_module_binding_verified": True,
        "parent_v3_immutable": True,
        "safety": dict(SAFETY),
    }


def prequential_expert_weights(
    expert_normalized_log_loss_history: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """Compute horizon-adaptive Hedge weights from strictly prior rows only."""

    if set(expert_normalized_log_loss_history) != set(EXPERT_NAMES):
        raise ValueError("prequential expert history set mismatch")
    lengths = {len(expert_normalized_log_loss_history[name]) for name in EXPERT_NAMES}
    if len(lengths) != 1:
        raise ValueError("prequential expert history length mismatch")
    count = lengths.pop()
    for name in EXPERT_NAMES:
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in expert_normalized_log_loss_history[name]
        ):
            raise ValueError("prequential normalized log loss is invalid")
    if count == 0:
        return dict.fromkeys(EXPERT_NAMES, 0.5)
    eta = math.sqrt(8.0 * math.log(len(EXPERT_NAMES)) / count)
    scores = {
        name: -eta * math.fsum(expert_normalized_log_loss_history[name]) for name in EXPERT_NAMES
    }
    maximum = max(scores.values())
    exponentials = {name: math.exp(scores[name] - maximum) for name in EXPERT_NAMES}
    denominator = math.fsum(exponentials.values())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("prequential weight denominator is invalid")
    weights = {name: exponentials[name] / denominator for name in EXPERT_NAMES}
    if not math.isclose(math.fsum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("prequential weights do not sum to one")
    return weights


def blend_probability_action_values(
    rows: Sequence[Mapping[str, Any]],
    residual_actions: Sequence[Mapping[str, float]],
    logit_actions: Sequence[Mapping[str, float]],
    *,
    weights: Mapping[str, float],
) -> list[dict[str, float]]:
    """Convex-blend coherent base probabilities, then subtract frozen costs."""

    if not (len(rows) == len(residual_actions) == len(logit_actions)):
        raise ValueError("v4 ensemble row count mismatch")
    if set(weights) != set(EXPERT_NAMES) or any(
        not math.isfinite(float(value)) or float(value) < 0.0 for value in weights.values()
    ):
        raise ValueError("v4 ensemble weights are invalid")
    if not math.isclose(
        math.fsum(float(value) for value in weights.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("v4 ensemble weights do not sum to one")
    output = []
    for row, residual, logit in zip(rows, residual_actions, logit_actions, strict=True):
        residual_probability = float(residual["predicted_probability"])
        logit_probability = float(logit["predicted_probability"])
        probability = (
            float(weights["probability_residual"]) * residual_probability
            + float(weights["logit_offset_binomial"]) * logit_probability
        )
        cost = dict(row["cost_decomposition"])
        entry_ask = float(cost["entry_ask"])
        non_entry_cost = float(cost["total_cost_excluding_entry_ask"])
        action_value = probability - entry_ask - non_entry_cost
        if not all(
            math.isfinite(value)
            for value in (
                residual_probability,
                logit_probability,
                probability,
                entry_ask,
                non_entry_cost,
                action_value,
            )
        ):
            raise ValueError("v4 ensemble action value is invalid")
        output.append(
            {
                "probability_residual_probability": residual_probability,
                "logit_offset_probability": logit_probability,
                "probability_residual_weight": float(weights["probability_residual"]),
                "logit_offset_binomial_weight": float(weights["logit_offset_binomial"]),
                "predicted_probability": probability,
                "entry_ask": entry_ask,
                "non_entry_cost": non_entry_cost,
                "action_value": action_value,
            }
        )
    _require_pair_coherence(rows, output)
    return output


def rolling_origin_prequential_ensemble_predict(
    *,
    rows: Sequence[Mapping[str, Any]],
    population_order: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the fixed two-model ensemble with strictly prior weight updates."""

    rows_by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_market[str(row["market_id"])].append(row)
    rolling = dict(protocol["rolling_origin"])
    initial = int(rolling["initial_training_market_count"])
    block_size = int(rolling["target_block_size"])
    block_count = int(rolling["target_block_count"])
    model = dict(protocol["model"])
    residual_spec = dict(model["base_learners"]["probability_residual"])
    logit_spec = dict(model["base_learners"]["logit_offset_binomial"])
    history: dict[str, list[float]] = {name: [] for name in EXPERT_NAMES}
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for block_index in range(block_count):
        target_start = initial + block_index * block_size
        target_end = target_start + block_size
        training_ids = list(population_order[:target_start])
        target_ids = list(population_order[target_start:target_end])
        if len(target_ids) != block_size:
            raise ValueError("residual v4 target block population mismatch")
        train_rows = [row for market_id in training_ids for row in rows_by_market[market_id]]
        target_rows = [row for market_id in target_ids for row in rows_by_market[market_id]]
        weights = prequential_expert_weights(history)
        history_before = {name: list(history[name]) for name in EXPERT_NAMES}

        residual_labels = [_probability_residual_label(row) for row in train_rows]
        residual_booster = xgb.train(
            params=dict(residual_spec["parameters"]),
            dtrain=_residual_dmatrix(train_rows, labels=residual_labels),
            num_boost_round=int(residual_spec["fixed_num_boost_round"]),
            verbose_eval=False,
        )
        residual_predictions = [
            float(value)
            for value in residual_booster.predict(_residual_dmatrix(target_rows, labels=None))
        ]
        residual_actions = pair_anchored_action_values(target_rows, residual_predictions)

        binary_labels = [_binary_payout_label(row) for row in train_rows]
        logit_booster = xgb.train(
            params=dict(logit_spec["parameters"]),
            dtrain=_logit_dmatrix(train_rows, labels=binary_labels),
            num_boost_round=int(logit_spec["fixed_num_boost_round"]),
            verbose_eval=False,
        )
        logit_predictions = [
            float(value)
            for value in logit_booster.predict(_logit_dmatrix(target_rows, labels=None))
        ]
        logit_actions = logit_offset_action_values(target_rows, logit_predictions)
        ensemble_actions = blend_probability_action_values(
            target_rows,
            residual_actions,
            logit_actions,
            weights=weights,
        )
        target_binary_labels = [_binary_payout_label(row) for row in target_rows]
        block_losses: dict[str, list[float]] = {name: [] for name in EXPERT_NAMES}
        for row, action, label in zip(
            target_rows, ensemble_actions, target_binary_labels, strict=True
        ):
            block_losses["probability_residual"].append(
                _normalized_binary_log_loss(label, action["probability_residual_probability"])
            )
            block_losses["logit_offset_binomial"].append(
                _normalized_binary_log_loss(label, action["logit_offset_probability"])
            )
            predictions.append(
                {
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "lineage_id": LINEAGE_ID,
                    "slot_id": protocol["slot_id"],
                    "market_id": row["market_id"],
                    "market_start_ts": row["market_start_ts"],
                    "decision_ts": row["decision_ts"],
                    "side": row["side"],
                    "prediction": action["action_value"],
                    "probability_residual_probability": action["probability_residual_probability"],
                    "logit_offset_probability": action["logit_offset_probability"],
                    "probability_residual_weight": action["probability_residual_weight"],
                    "logit_offset_binomial_weight": action["logit_offset_binomial_weight"],
                    "predicted_probability": action["predicted_probability"],
                    "realized_unit_net_pnl_if_action": row["target"],
                    "resolved_outcome": row["resolved_outcome"],
                    "cost_decomposition": row["cost_decomposition"],
                    "feature_row_sha256": row["feature_row_sha256"],
                    "chronological_block": block_index + 1,
                    "strictly_prior_training_market_count": len(training_ids),
                    "weight_prior_oof_side_decision_row_count": len(
                        history_before[EXPERT_NAMES[0]]
                    ),
                    "target_or_future_label_used_for_fit": False,
                    "target_or_future_label_used_for_weight": False,
                    "development_only_forever": True,
                    "promotion_evidence_eligible": False,
                    "safety": dict(SAFETY),
                }
            )
        for name in EXPERT_NAMES:
            history[name].extend(block_losses[name])
        audits.append(
            {
                "schema_version": FOLD_SCHEMA_VERSION,
                "lineage_id": LINEAGE_ID,
                "slot_id": protocol["slot_id"],
                "chronological_block": block_index + 1,
                "strictly_prior_training_market_count": len(training_ids),
                "target_market_count": len(target_ids),
                "training_market_ids_sha256": canonical_json_sha256(training_ids),
                "target_market_ids_sha256": canonical_json_sha256(target_ids),
                "last_training_market_position": target_start,
                "first_target_market_position": target_start + 1,
                "target_or_future_label_leakage_count": 0,
                "weight_prior_oof_side_decision_row_count": len(history_before[EXPERT_NAMES[0]]),
                "weight_history_sha256": canonical_json_sha256(history_before),
                "weights": dict(weights),
                "weight_update_used_current_block_label_count": 0,
                "residual_training_labels_sha256": canonical_json_sha256(residual_labels),
                "logit_training_labels_sha256": canonical_json_sha256(binary_labels),
                "residual_model_parameters_sha256": canonical_json_sha256(
                    residual_spec["parameters"]
                ),
                "logit_model_parameters_sha256": canonical_json_sha256(logit_spec["parameters"]),
                "block_normalized_log_loss_mean": {
                    name: math.fsum(block_losses[name]) / len(block_losses[name])
                    for name in EXPERT_NAMES
                },
                "pair_coherence_applied_before_probability_blend": True,
                "cost_subtraction_applied_after_probability_blend": True,
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
        )
    return predictions, audits


def render_v4_markdown(report: Mapping[str, Any]) -> str:
    """Render unchanged gates and explicit prequential ensemble semantics."""

    base = (
        render_residual_oof_markdown(report)
        .replace(
            "# BTC 15m cost-aware residual primary slot 001",
            "# BTC 15m prequential pooled residual v4 primary slot 001",
            1,
        )
        .rstrip()
    )
    return (
        base
        + "\n\n## Architecture and governance\n\n"
        + "- Architecture: two pooled side-symmetric probability learners with convex blending.\n"
        + "- Weight update: horizon-adaptive Hedge using strictly prior OOF normalized log loss.\n"
        + "- Initial weights: `0.5 / 0.5`; weight, parameter and threshold search: `False`\n"
        + "- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`\n"
        + "- Parent v1/v2/v3 failed artifacts changed: `False`\n"
        + "- Candidate slots remaining after this evaluation: `1`\n"
        + "- Next-stage authorization required even if every gate passes: `True`\n"
        + "- Collection, shadow, paper/live, wallet, write, promotion or capital authorized: `False`\n"
    )


def _market_results_from_predictions(
    *,
    predictions: Sequence[Mapping[str, Any]],
    baseline_by_market: Mapping[str, Mapping[str, Any]],
    population_order: Sequence[str],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    v1_rows = market_results_from_predictions(
        predictions=_as_v1_predictions(predictions),
        baseline_by_market=baseline_by_market,
        population_order=population_order,
        initial_training_market_count=int(
            protocol["rolling_origin"]["initial_training_market_count"]
        ),
        target_block_size=int(protocol["rolling_origin"]["target_block_size"]),
    )
    return [_replace_governance(row, MARKET_RESULT_SCHEMA_VERSION) for row in v1_rows]


def _build_report(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    source_commit: str,
    market_results: Sequence[Mapping[str, Any]],
    fold_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = build_residual_oof_report(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        source_commit=source_commit,
        market_results=_as_v1_market_results(market_results),
        fold_audits=_as_v1_folds(fold_audits),
    )
    report["schema_version"] = REPORT_SCHEMA_VERSION
    report["lineage_id"] = LINEAGE_ID
    report["architecture_type"] = (
        "prequential_horizon_adaptive_convex_ensemble_of_two_pooled_"
        "side_symmetric_market_anchored_probability_learners"
    )
    report["immutable_gate_implementation_sha256"] = IMMUTABLE_GATE_IMPLEMENTATION["sha256"]
    report["actual_executing_module_exactly_bound"] = True
    report["existing_gate_threshold_cost_baseline_population_changed"] = False
    report["parent_v1_v2_or_v3_failed_artifacts_changed"] = False
    report["remaining_candidate_slots"] = 1
    report["next_stage_authorization_required_even_if_all_gates_pass"] = True
    report["prequential_weighting"] = {
        "method": _weighting_contract()["method"],
        "future_or_current_block_labels_used_for_weight": False,
        "fold_weights": [dict(row["weights"]) for row in fold_audits],
    }
    return report


def _validate_population(
    *,
    predictions: Sequence[Mapping[str, Any]],
    fold_audits: Sequence[Mapping[str, Any]],
    market_results: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> None:
    if any(
        row.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or row.get("target_or_future_label_used_for_fit") is not False
        or row.get("target_or_future_label_used_for_weight") is not False
        or dict(row.get("safety") or {}) != SAFETY
        for row in predictions
    ):
        raise ValueError("residual v4 prediction governance mismatch")
    if any(
        row.get("schema_version") != FOLD_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or row.get("weight_update_used_current_block_label_count") != 0
        or dict(row.get("safety") or {}) != SAFETY
        for row in fold_audits
    ):
        raise ValueError("residual v4 fold governance mismatch")
    if any(
        row.get("schema_version") != MARKET_RESULT_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or dict(row.get("safety") or {}) != SAFETY
        for row in market_results
    ):
        raise ValueError("residual v4 market governance mismatch")
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
        cost = dict(row["cost_decomposition"])
        expected = (
            float(row["predicted_probability"])
            - float(cost["entry_ask"])
            - float(cost["total_cost_excluding_entry_ask"])
        )
        if not math.isclose(expected, float(row["prediction"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("residual v4 action-value reconciliation failed")
    for rows in grouped.values():
        if sorted(str(row["side"]) for row in rows) != ["DOWN", "UP"]:
            raise ValueError("residual v4 pair side mismatch")
        if not math.isclose(
            math.fsum(float(row["predicted_probability"]) for row in rows),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("residual v4 pair probability does not sum to one")
    _validate_frozen_population(
        predictions=_as_v1_predictions(predictions),
        fold_audits=_as_v1_folds(fold_audits),
        market_results=_as_v1_market_results(market_results),
        protocol=protocol,
    )


def _require_pair_coherence(
    rows: Sequence[Mapping[str, Any]], actions: Sequence[Mapping[str, float]]
) -> None:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    sides: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row, action in zip(rows, actions, strict=True):
        key = (str(row["market_id"]), int(row["decision_ts"]))
        grouped[key].append(float(action["predicted_probability"]))
        sides[key].append(str(row["side"]))
    for key in grouped:
        if sorted(sides[key]) != ["DOWN", "UP"]:
            raise ValueError(f"v4 ensemble UP/DOWN pair is incomplete: {key}")
        if not math.isclose(math.fsum(grouped[key]), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("v4 ensemble pair probability is incoherent")


def _normalized_binary_log_loss(label: float, probability: float) -> float:
    value = min(1.0 - LOG_LOSS_CLIP_EPSILON, max(LOG_LOSS_CLIP_EPSILON, probability))
    loss = -(label * math.log(value) + (1.0 - label) * math.log(1.0 - value))
    normalized = loss / -math.log(LOG_LOSS_CLIP_EPSILON)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("normalized binary log loss is invalid")
    return normalized


def _public_dataset_v4_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = _public_dataset_row(row)
    output["schema_version"] = DATASET_SCHEMA_VERSION
    output["lineage_id"] = LINEAGE_ID
    output["probability_residual_target"] = _probability_residual_label(row)
    output["binary_payout_target"] = _binary_payout_label(row)
    return output


def _as_v1_predictions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_replace_governance(row, V1_PREDICTION_SCHEMA_VERSION, V1_LINEAGE_ID) for row in rows]


def _as_v1_folds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_replace_governance(row, V1_FOLD_SCHEMA_VERSION, V1_LINEAGE_ID) for row in rows]


def _as_v1_market_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _replace_governance(row, V1_MARKET_RESULT_SCHEMA_VERSION, V1_LINEAGE_ID) for row in rows
    ]


def _replace_governance(
    row: Mapping[str, Any], schema_version: str, lineage_id: str = LINEAGE_ID
) -> dict[str, Any]:
    output = deepcopy(dict(row))
    output["schema_version"] = schema_version
    output["lineage_id"] = lineage_id
    return output


def _raw_text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_budget_contract() -> dict[str, Any]:
    return {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 1,
        "slots_consumed_before_run": 0,
        "slots_remaining_after_run": 1,
        "slot_budget_may_be_increased": False,
    }


def _target_contract() -> dict[str, Any]:
    return {
        "action_value_formula": (
            "prequential_convex_blend_of_pair_coherent_probability_residual_and_"
            "logit_offset_probabilities-entry_ask-frozen_fees-slippage-liquidity_impact"
        ),
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "NO_TRADE_value": 0.0,
        "post_close_training_label_only": True,
        "base_labels": {
            "probability_residual": "settlement_payout-selected_mid",
            "logit_offset_binomial": "settlement_payout",
        },
    }


def _pair_contract() -> dict[str, Any]:
    return {
        "base_probabilities_pair_normalized_before_blend": True,
        "blend_is_convex_and_preserves_UP_DOWN_sum_one": True,
        "cost_subtraction_happens_after_blend": True,
        "clip_epsilon": PAIR_CLIP_EPSILON,
        "missing_anchor_behavior": "fail_closed_NO_TRADE_in_runtime",
        "missing_values_encoded_as_zero": False,
    }


def _feature_contract() -> dict[str, Any]:
    return {
        "ordered_feature_count": 108,
        "base_feature_count": 54,
        "ordered_feature_names_sha256": canonical_json_sha256(list(FEATURE_NAMES)),
        "shared_side_symmetric_model": True,
        "side_identity_feature_allowed": False,
        "decision_time_causal_inputs_only": True,
        "native_missing_value": "nan",
        "missing_values_encoded_as_zero": False,
        "feature_search_allowed": False,
        "market_horizon_seconds": 900,
        "source_contract_reused_without_feature_addition_or_removal": True,
    }


def _model_contract() -> dict[str, Any]:
    return {
        "family": (
            "prequential_convex_ensemble_of_pooled_side_symmetric_market_"
            "anchored_probability_learners"
        ),
        "route_or_expert_filtering_allowed": False,
        "model_selection_or_early_stopping_performed": False,
        "base_learners": {
            "probability_residual": {
                "family": "market_anchored_probability_residual_xgboost",
                "fixed_num_boost_round": 128,
                "parameters": _residual_model_parameters(),
            },
            "logit_offset_binomial": {
                "family": "binomial_xgboost_with_market_logit_base_margin",
                "fixed_num_boost_round": 128,
                "parameters": _logit_model_parameters(),
            },
        },
    }


def _weighting_contract() -> dict[str, Any]:
    return {
        "method": "horizon_adaptive_hedge_on_strictly_prior_oof_binary_log_loss",
        "expert_names": list(EXPERT_NAMES),
        "initial_weights": dict.fromkeys(EXPERT_NAMES, 0.5),
        "binary_log_loss_clip_epsilon": LOG_LOSS_CLIP_EPSILON,
        "loss_normalization": "divide_by_negative_log_clip_epsilon",
        "learning_rate_formula": "sqrt(8*ln(2)/prior_oof_side_decision_row_count)",
        "current_or_future_block_label_used_for_weight": False,
        "weight_parameter_or_window_search_allowed": False,
    }


def _baseline_contract() -> dict[str, Any]:
    return {
        "candidate_and_baseline_population_must_match": True,
        "candidate_and_baseline_share_bootstrap_indices": True,
        "matched_global_baseline_behavior_is_frozen": True,
        "policy": "HOLD_TO_SETTLEMENT",
    }


def _dataset_contract() -> dict[str, Any]:
    return {
        "market_count": 800,
        "side_decision_row_count": 3200,
        "decision_rows_per_market": 2,
        "sides_per_decision": 2,
        "development_only_forever": True,
        "population_order": "frozen_confirmatory_capture_manifest_order",
    }


def _discipline_contract() -> dict[str, Any]:
    return {
        "one_candidate_this_slot": True,
        "hyperparameter_search_allowed": False,
        "weight_search_allowed": False,
        "threshold_search_allowed": False,
        "feature_search_allowed": False,
        "route_side_missingness_or_outlier_filtering_allowed": False,
        "post_result_mutation_allowed": False,
        "challenger_requires_separate_preregistration": True,
    }


def _state_contract() -> dict[str, Any]:
    return {
        "training_started": False,
        "candidate_frozen": False,
        "live_shadow_started": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "promotion_started": False,
    }


def _authorization_state_contract() -> dict[str, Any]:
    return {
        "candidate_frozen": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "lineage_authorized_for_governed_development": True,
        "live_shadow_started": False,
        "promotion_started": False,
        "training_started": False,
    }


__all__ = [
    "blend_probability_action_values",
    "prequential_expert_weights",
    "require_v4_candidate_implementation_binding",
    "rolling_origin_prequential_ensemble_predict",
    "run_residual_v4_rolling_origin_oof",
    "validate_residual_v4_protocol",
    "validate_v4_lineage_authorization",
    "verify_frozen_residual_v4_oof",
]
