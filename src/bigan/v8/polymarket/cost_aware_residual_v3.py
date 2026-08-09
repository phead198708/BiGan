"""Governed causal-feature, time-adaptive BTC 15m residual development v3."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
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
    _looks_like_git_sha,
    _public_dataset_row,
    _verified_json,
    _verify_descriptor,
    build_residual_oof_report,
    market_results_from_predictions,
    render_residual_oof_markdown,
)
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    GATE_NAMES,
    PAIR_CLIP_EPSILON,
    SELECTED_MID_INDEX,
    _action_policy,
    _bootstrap_contract,
    _cost_stress_contract,
    _model_parameters,
    _power_contract,
    _rolling_contract,
    pair_anchored_action_values,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    _write_new_frozen_json,
    _write_new_jsonl,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES
from bigan.v8.polymarket.moe_confirmatory_evaluation import _write_new_frozen_text
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

LINEAGE_ID = "BTC-15M-cost-aware-market-residual-v3"
PARENT_LINEAGE_ID = "BTC-15M-cost-aware-market-residual-v2"
PROTOCOL_SCHEMA_VERSION = "bigan-btc-15m-causal-time-adaptive-residual-oof-protocol-v3"
PREDICTION_SCHEMA_VERSION = "bigan-btc-15m-causal-time-adaptive-residual-oof-prediction-v3"
FOLD_SCHEMA_VERSION = "bigan-btc-15m-causal-time-adaptive-residual-oof-fold-v3"
MARKET_RESULT_SCHEMA_VERSION = "bigan-btc-15m-causal-time-adaptive-residual-market-result-v3"
REPORT_SCHEMA_VERSION = "bigan-btc-15m-causal-time-adaptive-residual-oof-report-v3"
MANIFEST_SCHEMA_VERSION = "bigan-btc-15m-causal-time-adaptive-residual-oof-manifest-v3"
DATASET_SCHEMA_VERSION = "bigan-btc-15m-causal-time-adaptive-residual-dataset-row-v3"

DEFAULT_CONFIG_DIR = (
    REPO_ROOT
    / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v3"
)
DEFAULT_PROTOCOL = DEFAULT_CONFIG_DIR / "residual_v3_primary_slot_001_protocol.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "residual_v3_primary_slot_001_oof"
DEFAULT_AUTHORIZATION = DEFAULT_CONFIG_DIR / "lineage_authorization.json"
DEFAULT_REGISTRY = DEFAULT_CONFIG_DIR / "development_data_registry.json"

AUTHORIZATION_INSTRUCTION = (
    "授权建立 BTC-15M-cost-aware-market-residual-v3，候选开发预算最多 2 个预注册 slot。"
    "允许使用已登记的 development-only 历史数据进行 outcome-aware 开发，允许新增因果特征"
    "和时间自适应训练。每个候选必须在评估前冻结协议和 SHA，禁止参数或阈值网格搜索。"
    "既有 gate、零阈值、N_max=2000、成本模型、基线、人口定义、v1/v2 失败 artifact 及"
    "全部安全状态不得修改。该授权不包含 fresh collection、outcome opening、live shadow、"
    "paper/live trading、wallet、write、promotion 或 capital risk；只有候选通过全部冻结 "
    "gate 后，才能另行申请下一阶段授权。"
)
AUTHORIZATION_INSTRUCTION_SHA256 = (
    "28fe8a1c7bba099c3897edb7ffd1ca71ff754d10a9ffd5100f974d73c4a1d8d3"
)
PARENT_V2_TERMINAL_SHA256 = (
    "3850abf82477c8aedfdc5bd831390989504184ad539bda9666a89bd3d452ab11"
)
IMMUTABLE_GATE_IMPLEMENTATION = {
    "path": "src/bigan/v8/polymarket/cost_aware_residual.py",
    "sha256": "491a329f708a16d5aecdd952552cbff3fa13d8f7446bfe3e0c78fade3b36f78c",
}
RECENCY_HALF_LIFE_MARKETS = 200.0

ENGINEERED_VALUE_FEATURE_NAMES = (
    "selected_spread_fraction",
    "opposite_spread_fraction",
    "paired_ask_overround",
    "paired_mid_deviation",
    "log_liquidity_depth_ratio",
    "log_executable_ask_notional_ratio",
    "book_staleness_delta_ms",
    "signed_return_consensus",
    "signed_return_acceleration_10s_vs_1m",
    "reference_dislocation_x_market_progress",
    "signed_return_5m_x_market_progress",
    "provider_health_x_paired_ask_overround",
)
ENGINEERED_MISSING_FEATURE_NAMES = tuple(
    f"{name}__missing" for name in ENGINEERED_VALUE_FEATURE_NAMES
)
V3_FEATURE_NAMES = (
    tuple(FEATURE_NAMES)
    + ENGINEERED_VALUE_FEATURE_NAMES
    + ENGINEERED_MISSING_FEATURE_NAMES
)


def validate_v3_lineage_authorization(
    *,
    authorization_path: Path | str = DEFAULT_AUTHORIZATION,
    registry_path: Path | str = DEFAULT_REGISTRY,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify the exact user grant and immutable v2 terminal boundary."""

    root = Path(repository_root).resolve()
    authorization_file = Path(authorization_path).resolve()
    registry_file = Path(registry_path).resolve()
    authorization = _verified_json(authorization_file)
    registry = _verified_json(registry_file)
    blockers: list[str] = []
    source = dict(authorization.get("authorization_source") or {})
    if authorization.get("schema_version") != (
        "bigan-btc-15m-cost-aware-residual-lineage-authorization-v3"
    ):
        blockers.append("authorization.schema_version")
    if source.get("type") != "explicit_user_instruction":
        blockers.append("authorization_source.type")
    if source.get("instruction") != AUTHORIZATION_INSTRUCTION:
        blockers.append("authorization_source.instruction")
    if _raw_text_sha256(str(source.get("instruction") or "")) != (
        AUTHORIZATION_INSTRUCTION_SHA256
    ):
        blockers.append("authorization_source.instruction_sha256")
    if source.get("instruction_sha256") != AUTHORIZATION_INSTRUCTION_SHA256:
        blockers.append("authorization_source.recorded_sha256")
    if authorization.get("lineage_id") != LINEAGE_ID:
        blockers.append("authorization.lineage_id")
    expected_scope = {
        "candidate_slot_budget": {
            "maximum_total_slots": 2,
            "slot_budget_may_be_increased": False,
        },
        "causal_feature_addition_authorized": True,
        "fresh_collection_authorized": False,
        "fresh_outcome_opening_authorized": False,
        "live_shadow_authorized": False,
        "outcome_aware_development_authorized": True,
        "paper_or_live_execution_authorized": False,
        "promotion_authorized": False,
        "time_adaptive_training_authorized": True,
        "training_may_start_only_after_slot_protocol_is_sha_frozen": True,
        "wallet_or_write_authorized": False,
    }
    if dict(authorization.get("authorization_scope") or {}) != expected_scope:
        blockers.append("authorization_scope")
    if dict(authorization.get("safety") or {}) != SAFETY:
        blockers.append("authorization.safety")
    if dict(authorization.get("state") or {}) != {
        "candidate_frozen": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "lineage_authorized_for_governed_development": True,
        "live_shadow_started": False,
        "promotion_started": False,
        "training_started": False,
    }:
        blockers.append("authorization.state")
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
        terminal_path = _verify_descriptor(
            dict(parent["terminal_review"]), repository_root=root
        )
        terminal = _load_json(terminal_path)
        if not (
            sha256_file(terminal_path) == PARENT_V2_TERMINAL_SHA256
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
        registered = _verify_descriptor(
            dict(authorization["registered_development_data"]), repository_root=root
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
    ):
        blockers.append("development_data_registry")
    for name, descriptor in dict(registry.get("registered_sources") or {}).items():
        try:
            _verify_descriptor(dict(descriptor), repository_root=root)
        except (OSError, TypeError, ValueError):
            blockers.append(f"development_data_registry.{name}")
    if blockers:
        raise ValueError("residual v3 authorization invalid: " + ", ".join(blockers))
    return {
        "authorization_valid": True,
        "lineage_id": LINEAGE_ID,
        "maximum_total_slots": 2,
        "parent_v2_immutable": True,
        "authorization_sha256": sha256_file(authorization_file),
        "registry_sha256": sha256_file(registry_file),
        "safety": dict(SAFETY),
    }


def engineer_causal_features(features: Sequence[float]) -> np.ndarray:
    """Append fixed causal interactions while preserving native NaNs."""

    source = np.asarray(features, dtype=float)
    if source.shape != (len(FEATURE_NAMES),):
        raise ValueError("residual v3 source feature vector shape mismatch")
    values = {name: float(source[index]) for index, name in enumerate(FEATURE_NAMES)}

    def finite(*names: str) -> bool:
        return all(math.isfinite(values[name]) for name in names)

    def difference(left: str, right: str) -> float:
        return values[left] - values[right] if finite(left, right) else math.nan

    def log_ratio(left: str, right: str) -> float:
        if not finite(left, right) or values[left] < 0.0 or values[right] < 0.0:
            return math.nan
        return math.log1p(values[left]) - math.log1p(values[right])

    engineered = [
        difference("selected_ask", "selected_bid"),
        difference("opposite_ask", "opposite_bid"),
        values["paired_ask_sum"] - 1.0
        if finite("paired_ask_sum")
        else math.nan,
        values["paired_mid_sum"] - 1.0
        if finite("paired_mid_sum")
        else math.nan,
        log_ratio("selected_liquidity_depth", "opposite_liquidity_depth"),
        log_ratio(
            "selected_executable_ask_notional",
            "opposite_executable_ask_notional",
        ),
        difference("selected_book_staleness_ms", "opposite_book_staleness_ms"),
        sum(
            values[name]
            for name in (
                "signed_btc_return_10s",
                "signed_btc_return_1m",
                "signed_btc_return_5m",
                "signed_btc_return_15m",
            )
        )
        if finite(
            "signed_btc_return_10s",
            "signed_btc_return_1m",
            "signed_btc_return_5m",
            "signed_btc_return_15m",
        )
        else math.nan,
        difference("signed_btc_return_10s", "signed_btc_return_1m"),
        values["signed_btc_mid_to_chainlink_relative_distance"]
        * values["market_progress_fraction"]
        if finite(
            "signed_btc_mid_to_chainlink_relative_distance",
            "market_progress_fraction",
        )
        else math.nan,
        values["signed_btc_return_5m"] * values["market_progress_fraction"]
        if finite("signed_btc_return_5m", "market_progress_fraction")
        else math.nan,
        values["provider_health_score"] * (values["paired_ask_sum"] - 1.0)
        if finite("provider_health_score", "paired_ask_sum")
        else math.nan,
    ]
    missing = [1.0 if not math.isfinite(value) else 0.0 for value in engineered]
    return np.concatenate((source, np.asarray(engineered), np.asarray(missing)))


def recency_weights(
    market_ids: Sequence[str],
    *,
    population_order: Sequence[str],
    half_life_markets: float = RECENCY_HALF_LIFE_MARKETS,
) -> np.ndarray:
    """Return fixed exponentially decayed, mean-one, market-grouped weights."""

    if not math.isfinite(half_life_markets) or half_life_markets <= 0.0:
        raise ValueError("residual v3 recency half-life must be positive")
    positions = {market_id: index for index, market_id in enumerate(population_order)}
    if len(positions) != len(population_order) or not market_ids:
        raise ValueError("residual v3 recency population is invalid")
    try:
        newest_position = max(positions[market_id] for market_id in market_ids)
        raw = np.asarray(
            [
                0.5
                ** ((newest_position - positions[market_id]) / half_life_markets)
                for market_id in market_ids
            ],
            dtype=float,
        )
    except KeyError as exc:
        raise ValueError("residual v3 row is outside the frozen population") from exc
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("residual v3 recency weight is invalid")
    return raw / float(np.mean(raw))


def validate_residual_v3_protocol(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Fail closed unless slot 1 freezes the authorized change and old gates."""

    blockers: list[str] = []
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        blockers.append("schema_version")
    if payload.get("lineage_id") != LINEAGE_ID:
        blockers.append("lineage_id")
    if payload.get("slot_id") != "residual-v3-primary-slot-001":
        blockers.append("slot_id")
    if payload.get("candidate_role") != "primary":
        blockers.append("candidate_role")
    if payload.get("development_only_forever") is not True:
        blockers.append("development_only_forever")
    if payload.get("promotion_evidence_eligible") is not False:
        blockers.append("promotion_evidence_eligible")
    if dict(payload.get("candidate_budget") or {}) != {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 1,
        "slots_consumed_before_run": 0,
        "slots_remaining_after_run": 1,
        "slot_budget_may_be_increased": False,
    }:
        blockers.append("candidate_budget")
    if dict(payload.get("target") or {}) != _target_contract():
        blockers.append("target")
    if dict(payload.get("pair_coherence") or {}) != _pair_contract():
        blockers.append("pair_coherence")
    if dict(payload.get("feature_contract") or {}) != _feature_contract():
        blockers.append("feature_contract")
    if dict(payload.get("temporal_adaptation") or {}) != _temporal_contract():
        blockers.append("temporal_adaptation")
    if dict(payload.get("model") or {}) != _model_contract():
        blockers.append("model")
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
    if dict(payload.get("baseline") or {}) != _baseline_contract():
        blockers.append("baseline")
    if dict(payload.get("dataset") or {}) != _dataset_contract():
        blockers.append("dataset")
    if dict(payload.get("development_discipline") or {}) != _discipline_contract():
        blockers.append("development_discipline")
    if dict(payload.get("state") or {}) != _state_contract():
        blockers.append("state")
    if dict(payload.get("safety") or {}) != SAFETY:
        blockers.append("safety")
    inputs = dict(payload.get("inputs") or {})
    required_inputs = {
        "lineage_authorization",
        "development_data_registry",
        "parent_v2_terminal_review",
        "terminal_diagnostic_scored_rows",
        "confirmatory_capture_manifest",
        "confirmatory_market_evaluation_rows",
        "baseline_decision_rows",
        "matched_global_baseline_contract",
        "parent_feature_contract",
        "parent_cost_and_action_contract",
        "raw_capture_recovery_bundle_manifest",
        "candidate_implementation",
        "gate_implementation",
    }
    if set(inputs) != required_inputs:
        blockers.append("inputs")
    root = Path(repository_root).resolve()
    if verify_artifacts and not blockers:
        resolved: dict[str, Path] = {}
        for name, descriptor in inputs.items():
            try:
                resolved[name] = _verify_descriptor(
                    dict(descriptor), repository_root=root
                )
            except (KeyError, OSError, TypeError, ValueError):
                blockers.append(f"inputs.{name}")
        if not blockers:
            try:
                authorization = validate_v3_lineage_authorization(
                    authorization_path=resolved["lineage_authorization"],
                    registry_path=resolved["development_data_registry"],
                    repository_root=root,
                )
                if authorization["maximum_total_slots"] != 2:
                    blockers.append("lineage_authorization.slot_budget")
            except ValueError:
                blockers.append("lineage_authorization")
            terminal = _load_json(resolved["parent_v2_terminal_review"])
            if not (
                sha256_file(resolved["parent_v2_terminal_review"])
                == PARENT_V2_TERMINAL_SHA256
                and terminal.get("phase_1_terminal_failed") is True
                and terminal.get("candidate_budget_exhausted") is True
                and terminal.get("candidate_selected") is None
                and dict(terminal.get("safety") or {}) == SAFETY
            ):
                blockers.append("parent_v2_terminal_review")
            if dict(inputs["gate_implementation"]) != IMMUTABLE_GATE_IMPLEMENTATION:
                blockers.append("gate_implementation")
    if blockers:
        raise ValueError("residual v3 protocol invalid: " + ", ".join(blockers))


def run_residual_v3_rolling_origin_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Execute v3 slot 1 once after protocol and implementation SHA freeze."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("residual v3 OOF paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("residual v3 protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != (
        expected_protocol_sha256
    ):
        raise ValueError("residual v3 protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_residual_v3_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"residual v3 OOF output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset_rows, baseline_by_market, population_order = _load_frozen_development_rows(
        protocol=protocol,
        repository_root=root,
    )
    predictions, folds = rolling_origin_causal_time_adaptive_predict(
        rows=dataset_rows,
        population_order=population_order,
        protocol=protocol,
    )
    market_results = _market_results_from_predictions(
        predictions=predictions,
        baseline_by_market=baseline_by_market,
        population_order=population_order,
        protocol=protocol,
    )
    report = _build_report(
        protocol=protocol,
        protocol_sha256=expected_protocol_sha256,
        source_commit=source_commit,
        market_results=market_results,
        fold_audits=folds,
    )

    dataset_path = output / "residual_v3_development_dataset_rows.jsonl"
    prediction_path = output / "residual_v3_oof_predictions.jsonl"
    fold_path = output / "residual_v3_oof_fold_audits.jsonl"
    market_path = output / "residual_v3_oof_market_results.jsonl"
    report_path = output / "residual_v3_oof_report.json"
    markdown_path = output / "residual_v3_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_v3_dataset_row(row) for row in dataset_rows])
    _write_new_jsonl(prediction_path, predictions)
    _write_new_jsonl(fold_path, folds)
    _write_new_jsonl(market_path, market_results)
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_artifact = _write_new_frozen_text(markdown_path, render_v3_markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": protocol["slot_id"],
        "candidate_role": "primary",
        "created_at": protocol["created_at"],
        "source_commit": source_commit,
        "protocol": _descriptor(protocol_file, root),
        "candidate_implementation": dict(protocol["inputs"]["candidate_implementation"]),
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
    manifest_artifact = _write_new_frozen_json(
        output / "residual_v3_oof_manifest.json", manifest
    )
    return {
        "manifest": _descriptor(Path(manifest_artifact["path"]), root),
        "report": _descriptor(Path(report_artifact["path"]), root),
        "all_gates_passed": report["all_gates_passed"],
        "failed_gates": report["failed_gates"],
        "remaining_candidate_slots": 1,
        "oof_market_count": len(market_results),
        "next_stage_authorization_required": True,
        "safety": dict(SAFETY),
    }


def rolling_origin_causal_time_adaptive_predict(
    *,
    rows: Sequence[Mapping[str, Any]],
    population_order: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run fixed rolling-origin OOF with strictly prior exponentially weighted fit."""

    rows_by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_market[str(row["market_id"])].append(row)
    rolling = dict(protocol["rolling_origin"])
    initial = int(rolling["initial_training_market_count"])
    block_size = int(rolling["target_block_size"])
    block_count = int(rolling["target_block_count"])
    parameters = dict(protocol["model"]["parameters"])
    boost_rounds = int(protocol["model"]["fixed_num_boost_round"])
    half_life = float(protocol["temporal_adaptation"]["half_life_markets"])
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for block_index in range(block_count):
        target_start = initial + block_index * block_size
        target_end = target_start + block_size
        training_ids = list(population_order[:target_start])
        target_ids = list(population_order[target_start:target_end])
        if len(target_ids) != block_size:
            raise ValueError("residual v3 target block population mismatch")
        train_rows = [row for market_id in training_ids for row in rows_by_market[market_id]]
        target_rows = [row for market_id in target_ids for row in rows_by_market[market_id]]
        residual_labels = [_probability_residual_label(row) for row in train_rows]
        weights = recency_weights(
            [str(row["market_id"]) for row in train_rows],
            population_order=training_ids,
            half_life_markets=half_life,
        )
        train_matrix = _dmatrix(train_rows, labels=residual_labels, weights=weights)
        target_matrix = _dmatrix(target_rows, labels=None, weights=None)
        booster = xgb.train(
            params=parameters,
            dtrain=train_matrix,
            num_boost_round=boost_rounds,
            verbose_eval=False,
        )
        residual_values = [float(value) for value in booster.predict(target_matrix)]
        action_rows = pair_anchored_action_values(target_rows, residual_values)
        for row, action in zip(target_rows, action_rows, strict=True):
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
                    "market_anchor_probability": action["market_anchor_probability"],
                    "predicted_probability_residual": action["predicted_probability_residual"],
                    "predicted_probability_before_pair_normalization": action[
                        "predicted_probability_before_pair_normalization"
                    ],
                    "predicted_probability": action["predicted_probability"],
                    "realized_unit_net_pnl_if_action": row["target"],
                    "resolved_outcome": row["resolved_outcome"],
                    "cost_decomposition": row["cost_decomposition"],
                    "feature_row_sha256": row["feature_row_sha256"],
                    "engineered_feature_row_sha256": canonical_json_sha256(
                        _canonical_feature_values(engineer_causal_features(row["features"]))
                    ),
                    "chronological_block": block_index + 1,
                    "strictly_prior_training_market_count": len(training_ids),
                    "target_or_future_label_used_for_fit": False,
                    "development_only_forever": True,
                    "promotion_evidence_eligible": False,
                    "safety": dict(SAFETY),
                }
            )
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
                "training_residual_labels_sha256": canonical_json_sha256(residual_labels),
                "training_weights_sha256": canonical_json_sha256(weights.tolist()),
                "training_weight_min": float(np.min(weights)),
                "training_weight_max": float(np.max(weights)),
                "training_weight_mean": float(np.mean(weights)),
                "recency_half_life_markets": half_life,
                "last_training_market_position": target_start,
                "first_target_market_position": target_start + 1,
                "target_or_future_label_leakage_count": 0,
                "fixed_num_boost_round": boost_rounds,
                "model_parameters_sha256": canonical_json_sha256(parameters),
                "engineered_feature_names_sha256": canonical_json_sha256(
                    list(V3_FEATURE_NAMES)
                ),
                "pair_coherence_applied_before_cost_subtraction": True,
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
        )
    return predictions, audits


def _probability_residual_label(row: Mapping[str, Any]) -> float:
    side = str(row["side"])
    outcome = str(row["resolved_outcome"])
    if side not in {"UP", "DOWN"} or outcome not in {"UP", "DOWN"}:
        raise ValueError("residual v3 side or outcome is invalid")
    anchor = float(np.asarray(row["features"], dtype=float)[SELECTED_MID_INDEX])
    cost = dict(row["cost_decomposition"])
    payout = 1.0 if side == outcome else 0.0
    expected_target = (
        payout
        - float(cost["entry_ask"])
        - float(cost["total_cost_excluding_entry_ask"])
    )
    if not math.isclose(
        expected_target, float(row["target"]), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("residual v3 target/cost reconciliation failed")
    if not math.isfinite(anchor):
        raise ValueError("residual v3 selected_mid anchor is invalid")
    return payout - anchor


def _dmatrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[float] | None,
    weights: Sequence[float] | None,
) -> xgb.DMatrix:
    values = np.vstack([engineer_causal_features(row["features"]) for row in rows])
    label_values = np.asarray(labels, dtype=float) if labels is not None else None
    weight_values = np.asarray(weights, dtype=float) if weights is not None else None
    return xgb.DMatrix(
        values,
        label=label_values,
        weight=weight_values,
        feature_names=list(V3_FEATURE_NAMES),
        missing=np.nan,
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
        "pooled_side_symmetric_market_anchored_probability_residual_with_"
        "fixed_causal_interactions_and_exponential_recency_weighting"
    )
    report["immutable_gate_implementation_sha256"] = (
        IMMUTABLE_GATE_IMPLEMENTATION["sha256"]
    )
    report["existing_gate_threshold_cost_baseline_population_changed"] = False
    report["parent_v1_or_v2_failed_artifacts_changed"] = False
    report["remaining_candidate_slots"] = 1
    report["next_stage_authorization_required_even_if_all_gates_pass"] = True
    return report


def render_v3_markdown(report: Mapping[str, Any]) -> str:
    """Render unchanged gates and v3-only architectural changes."""

    base = render_residual_oof_markdown(report).replace(
        "# BTC 15m cost-aware residual primary slot 001",
        "# BTC 15m causal time-adaptive residual v3 primary slot 001",
        1,
    ).rstrip()
    return (
        base
        + "\n\n## Architecture and governance\n\n"
        + "- Added causal engineered feature values and explicit missingness indicators: `12 + 12`\n"
        + f"- Fixed exponential recency half-life in markets: `{RECENCY_HALF_LIFE_MARKETS:g}`\n"
        + "- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`\n"
        + "- Parent v1/v2 failed artifacts changed: `False`\n"
        + "- Candidate slots remaining after this evaluation: `1`\n"
        + "- Next-stage authorization required even if every gate passes: `True`\n"
        + "- Collection, shadow, paper/live, wallet, write, promotion or capital authorized: `False`\n"
    )


def _public_v3_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = _public_dataset_row(row)
    output["schema_version"] = DATASET_SCHEMA_VERSION
    output["lineage_id"] = LINEAGE_ID
    engineered = engineer_causal_features(row["features"])
    output["market_anchor_probability"] = float(
        np.asarray(row["features"], dtype=float)[SELECTED_MID_INDEX]
    )
    output["probability_residual_target"] = _probability_residual_label(row)
    output["engineered_features"] = dict(
        zip(V3_FEATURE_NAMES, _canonical_feature_values(engineered), strict=True)
    )
    output["engineered_feature_row_sha256"] = canonical_json_sha256(
        _canonical_feature_values(engineered)
    )
    return output


def _canonical_feature_values(values: Sequence[float]) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def _as_v1_predictions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _replace_governance(row, V1_PREDICTION_SCHEMA_VERSION, V1_LINEAGE_ID)
        for row in rows
    ]


def _as_v1_folds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _replace_governance(row, V1_FOLD_SCHEMA_VERSION, V1_LINEAGE_ID)
        for row in rows
    ]


def _as_v1_market_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _replace_governance(row, V1_MARKET_RESULT_SCHEMA_VERSION, V1_LINEAGE_ID)
        for row in rows
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


def _target_contract() -> dict[str, Any]:
    return {
        "action_value_formula": (
            "pair_normalized_clipped_selected_mid_plus_predicted_probability_"
            "residual-entry_ask-frozen_fees-slippage-liquidity_impact"
        ),
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "NO_TRADE_value": 0.0,
        "post_close_training_label_only": True,
        "regression_label": "settlement_payout-selected_mid",
    }


def _pair_contract() -> dict[str, Any]:
    return {
        "anchor": "decision_time_selected_mid",
        "clip_epsilon": PAIR_CLIP_EPSILON,
        "normalization": "UP_DOWN_probabilities_sum_to_one_per_decision",
        "normalization_happens_before_cost_subtraction": True,
        "missing_anchor_behavior": "fail_closed_NO_TRADE_in_runtime",
        "missing_values_encoded_as_zero": False,
    }


def _feature_contract() -> dict[str, Any]:
    return {
        "source_ordered_feature_count": 108,
        "source_base_feature_count": 54,
        "engineered_value_feature_count": len(ENGINEERED_VALUE_FEATURE_NAMES),
        "engineered_missing_indicator_count": len(ENGINEERED_MISSING_FEATURE_NAMES),
        "ordered_feature_count": len(V3_FEATURE_NAMES),
        "ordered_feature_names_sha256": canonical_json_sha256(list(V3_FEATURE_NAMES)),
        "engineered_value_feature_names": list(ENGINEERED_VALUE_FEATURE_NAMES),
        "shared_side_symmetric_model": True,
        "side_identity_feature_allowed": False,
        "decision_time_causal_inputs_only": True,
        "native_missing_value": "nan",
        "derived_value_missing_if_any_required_input_missing": True,
        "explicit_derived_missing_indicators": True,
        "missing_values_encoded_as_zero": False,
        "feature_search_allowed": False,
        "market_horizon_seconds": 900,
    }


def _temporal_contract() -> dict[str, Any]:
    return {
        "method": "exponential_market_recency_sample_weight",
        "half_life_markets": RECENCY_HALF_LIFE_MARKETS,
        "age_unit": "frozen_population_market_position",
        "same_weight_for_all_rows_in_market": True,
        "newest_training_market_raw_weight": 1.0,
        "normalize_fold_training_weights_to_mean_one": True,
        "future_market_used_to_compute_weight": False,
        "half_life_search_allowed": False,
    }


def _model_contract() -> dict[str, Any]:
    return {
        "family": (
            "pooled_side_symmetric_market_anchored_probability_residual_xgboost_"
            "with_fixed_causal_features_and_recency_weights"
        ),
        "route_or_expert_allowed": False,
        "fixed_num_boost_round": 128,
        "model_selection_or_early_stopping_performed": False,
        "parameters": _model_parameters(),
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
        "threshold_search_allowed": False,
        "feature_grid_search_allowed": False,
        "recency_half_life_search_allowed": False,
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


__all__ = [
    "ENGINEERED_VALUE_FEATURE_NAMES",
    "V3_FEATURE_NAMES",
    "engineer_causal_features",
    "recency_weights",
    "rolling_origin_causal_time_adaptive_predict",
    "run_residual_v3_rolling_origin_oof",
    "validate_residual_v3_protocol",
    "validate_v3_lineage_authorization",
]
