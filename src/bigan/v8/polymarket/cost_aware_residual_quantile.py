"""Single preregistered lower-quantile challenger for the BTC 15m residual."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.cost_aware_residual import (
    DEFAULT_PROTOCOL as PRIMARY_PROTOCOL,
)
from bigan.v8.polymarket.cost_aware_residual import (
    LINEAGE_ID,
    MANIFEST_SCHEMA_VERSION,
    _descriptor,
    _load_frozen_development_rows,
    _load_json,
    _load_jsonl,
    _looks_like_git_sha,
    _public_dataset_row,
    _rolling_origin_predict,
    _validate_frozen_population,
    _verified_json,
    _verify_descriptor,
    build_residual_oof_report,
    market_results_from_predictions,
    render_residual_oof_markdown,
    validate_residual_oof_protocol,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    _write_new_frozen_json,
    _write_new_jsonl,
)
from bigan.v8.polymarket.moe_confirmatory_evaluation import _write_new_frozen_text
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

CHALLENGER_PROTOCOL_SCHEMA_VERSION = "bigan-btc-15m-cost-aware-residual-quantile-oof-protocol-v1"
DEFAULT_CONFIG_DIR = (
    REPO_ROOT / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v1"
)
DEFAULT_CHALLENGER_PROTOCOL = DEFAULT_CONFIG_DIR / "residual_challenger_slot_002_protocol.json"
DEFAULT_CHALLENGER_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "residual_challenger_slot_002_oof"


def validate_quantile_challenger_protocol(
    payload: dict[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Validate the sole structurally distinct challenger without relaxing gates."""

    blockers = []
    if payload.get("schema_version") != CHALLENGER_PROTOCOL_SCHEMA_VERSION:
        blockers.append("schema_version")
    if payload.get("candidate_role") != "challenger":
        blockers.append("candidate_role")
    budget = dict(payload.get("candidate_budget") or {})
    if budget != {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 2,
        "slots_consumed_before_run": 1,
        "slots_remaining_after_run": 0,
        "slot_budget_may_be_increased": False,
    }:
        blockers.append("candidate_budget")
    model = dict(payload.get("model") or {})
    parameters = dict(model.get("parameters") or {})
    if not (
        model.get("family") == "pooled_global_xgboost_direct_lower_quantile_regressor"
        and model.get("route_or_expert_allowed") is False
        and model.get("fixed_num_boost_round") == 128
        and parameters.get("objective") == "reg:quantileerror"
        and parameters.get("eval_metric") == "quantile"
        and parameters.get("quantile_alpha") == 0.35
        and parameters.get("nthread") == 1
    ):
        blockers.append("model")
    if payload.get("structural_change") != {
        "changed_component": "fixed_training_loss_only",
        "from": "conditional_mean_squared_error",
        "to": "lower_conditional_quantile_loss_alpha_0_35",
        "reason": (
            "primary accepted 570_of_600 markets and exceeded the N_max_2000 "
            "variance budget despite positive absolute and paired LCBs"
        ),
        "expected_mechanism": (
            "a positive lower conditional action-value quantile abstains unless the "
            "after-cost edge is robust across the lower tail"
        ),
        "threshold_changed": False,
        "feature_set_changed": False,
        "rolling_population_changed": False,
    }:
        blockers.append("structural_change")
    prior = dict(payload.get("prior_slot_result") or {})
    if set(prior) != {"manifest", "report", "failed_gates"} or prior.get("failed_gates") != [
        "every_chronological_block_paired_delta_total_gte_zero",
        "prospective_power_required_market_count_lte_2000",
    ]:
        blockers.append("prior_slot_result")
    root = Path(repository_root).resolve()
    if verify_artifacts and not blockers:
        for field in ("manifest", "report"):
            try:
                _verify_descriptor(prior[field], repository_root=root)
            except (KeyError, OSError, TypeError, ValueError):
                blockers.append(f"prior_slot_result.{field}")
        try:
            _verify_descriptor(
                payload["inputs"]["implementation"], repository_root=root
            )
        except (KeyError, OSError, TypeError, ValueError):
            blockers.append("inputs.implementation")

    normalized = deepcopy(payload)
    normalized["schema_version"] = "bigan-btc-15m-cost-aware-residual-oof-protocol-v1"
    normalized["candidate_role"] = "primary"
    normalized["candidate_budget"] = {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 1,
        "slots_consumed_before_run": 0,
        "slot_budget_may_be_increased": False,
    }
    normalized["model"]["family"] = "pooled_global_xgboost_direct_regressor"
    normalized["model"]["parameters"]["objective"] = "reg:squarederror"
    normalized["model"]["parameters"]["eval_metric"] = "rmse"
    normalized["model"]["parameters"].pop("quantile_alpha", None)
    primary = _load_json(Path(PRIMARY_PROTOCOL))
    normalized["inputs"]["implementation"] = primary["inputs"]["implementation"]
    normalized.pop("prior_slot_result", None)
    normalized.pop("structural_change", None)
    try:
        validate_residual_oof_protocol(
            normalized,
            repository_root=root,
            verify_artifacts=verify_artifacts,
        )
    except ValueError as error:
        blockers.append(f"shared_protocol:{error}")
    if blockers:
        raise ValueError("quantile challenger protocol invalid: " + ", ".join(blockers))


def run_quantile_challenger_oof(
    *,
    protocol_path: Path | str = DEFAULT_CHALLENGER_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_CHALLENGER_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Consume the second and final preregistered development slot exactly once."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("quantile challenger paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("quantile challenger protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != expected_protocol_sha256
    ):
        raise ValueError("quantile challenger protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_quantile_challenger_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"quantile challenger output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset_rows, baseline_by_market, population_order = _load_frozen_development_rows(
        protocol=protocol,
        repository_root=root,
    )
    predictions, folds = _rolling_origin_predict(
        rows=dataset_rows,
        population_order=population_order,
        protocol=protocol,
    )
    market_results = market_results_from_predictions(
        predictions=predictions,
        baseline_by_market=baseline_by_market,
        population_order=population_order,
        initial_training_market_count=int(
            protocol["rolling_origin"]["initial_training_market_count"]
        ),
        target_block_size=int(protocol["rolling_origin"]["target_block_size"]),
    )
    report = build_residual_oof_report(
        protocol=protocol,
        protocol_sha256=expected_protocol_sha256,
        source_commit=source_commit,
        market_results=market_results,
        fold_audits=folds,
    )
    report["candidate_budget_exhausted"] = True
    report["additional_candidate_allowed"] = False
    report["structural_change"] = dict(protocol["structural_change"])

    dataset_path = output / "residual_development_dataset_rows.jsonl"
    prediction_path = output / "residual_oof_predictions.jsonl"
    fold_path = output / "residual_oof_fold_audits.jsonl"
    market_path = output / "residual_oof_market_results.jsonl"
    report_path = output / "residual_oof_report.json"
    markdown_path = output / "residual_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_dataset_row(row) for row in dataset_rows])
    _write_new_jsonl(prediction_path, predictions)
    _write_new_jsonl(fold_path, folds)
    _write_new_jsonl(market_path, market_results)
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_artifact = _write_new_frozen_text(
        markdown_path, render_quantile_challenger_markdown(report)
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": protocol["slot_id"],
        "candidate_role": "challenger",
        "created_at": protocol["created_at"],
        "source_commit": source_commit,
        "protocol": _descriptor(protocol_file, root),
        "prior_slot_result": dict(protocol["prior_slot_result"]),
        "artifacts": {
            "dataset_rows": _descriptor(dataset_path, root),
            "predictions": _descriptor(prediction_path, root),
            "fold_audits": _descriptor(fold_path, root),
            "market_results": _descriptor(market_path, root),
            "report": _descriptor(Path(report_artifact["path"]), root),
            "report_markdown": _descriptor(Path(markdown_artifact["path"]), root),
        },
        "evaluation_executed_exactly_once": True,
        "candidate_budget_exhausted": True,
        "additional_candidate_allowed": False,
        "candidate_freeze_allowed": report["all_gates_passed"],
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_artifact = _write_new_frozen_json(output / "residual_oof_manifest.json", manifest)
    return {
        "manifest": _descriptor(Path(manifest_artifact["path"]), root),
        "report": _descriptor(Path(report_artifact["path"]), root),
        "all_gates_passed": report["all_gates_passed"],
        "failed_gates": report["failed_gates"],
        "candidate_budget_exhausted": True,
        "oof_market_count": len(market_results),
        "safety": dict(SAFETY),
    }


def verify_frozen_quantile_challenger_oof(
    *,
    protocol_path: Path | str = DEFAULT_CHALLENGER_PROTOCOL,
    output_dir: Path | str = DEFAULT_CHALLENGER_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify all challenger artifacts and rebuild the report deterministically."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    protocol = _verified_json(protocol_file)
    validate_quantile_challenger_protocol(protocol, repository_root=root)
    manifest = _verified_json(output / "residual_oof_manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("quantile challenger manifest schema mismatch")
    if manifest.get("protocol") != _descriptor(protocol_file, root):
        raise ValueError("quantile challenger protocol binding mismatch")
    artifact_paths = {
        name: _verify_descriptor(descriptor, repository_root=root)
        for name, descriptor in dict(manifest["artifacts"]).items()
    }
    predictions = _load_jsonl(artifact_paths["predictions"])
    folds = _load_jsonl(artifact_paths["fold_audits"])
    markets = _load_jsonl(artifact_paths["market_results"])
    _validate_frozen_population(
        predictions=predictions,
        fold_audits=folds,
        market_results=markets,
        protocol=protocol,
    )
    rebuilt = build_residual_oof_report(
        protocol=protocol,
        protocol_sha256=sha256_file(protocol_file),
        source_commit=str(manifest["source_commit"]),
        market_results=markets,
        fold_audits=folds,
    )
    rebuilt["candidate_budget_exhausted"] = True
    rebuilt["additional_candidate_allowed"] = False
    rebuilt["structural_change"] = dict(protocol["structural_change"])
    frozen_report = _load_json(artifact_paths["report"])
    if rebuilt != frozen_report:
        raise ValueError("quantile challenger report does not reproduce")
    if render_quantile_challenger_markdown(rebuilt) != artifact_paths["report_markdown"].read_text(
        encoding="utf-8"
    ):
        raise ValueError("quantile challenger markdown does not reproduce")
    return {
        "verification_passed": True,
        "all_gates_passed": rebuilt["all_gates_passed"],
        "failed_gates": rebuilt["failed_gates"],
        "candidate_budget_exhausted": True,
        "oof_market_count": len(markets),
        "manifest_sha256": sha256_file(output / "residual_oof_manifest.json"),
        "safety": dict(SAFETY),
    }


def render_quantile_challenger_markdown(report: dict[str, Any]) -> str:
    """Render base metrics plus the final-slot governance boundary."""

    base = render_residual_oof_markdown(report).rstrip()
    return (
        base
        + "\n\n## Candidate budget\n\n"
        + "- Second and final preregistered slot consumed: `True`\n"
        + "- Additional candidate allowed: `False`\n"
        + "- Structural change: fixed lower-quantile training loss only; threshold, "
        + "features, population, costs, bootstrap, and gates are unchanged.\n"
    )


__all__ = [
    "run_quantile_challenger_oof",
    "validate_quantile_challenger_protocol",
    "verify_frozen_quantile_challenger_oof",
]
