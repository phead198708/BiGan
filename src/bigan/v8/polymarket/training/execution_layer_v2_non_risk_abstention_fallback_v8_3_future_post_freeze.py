"""Single-use read-only settlement and PnL gate for issue #249 v8.3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_post_freeze import (
    AdaptiveSupportControllerV81FutureSettlementConfig,
    _evaluation_only_frozen_features_by_market,
    _validate_settled_index,
    _write_single_use_claim,
    build_adaptive_support_controller_v8_1_future_settled_index,
)
from bigan.v8.polymarket.training.execution_layer_v2_non_risk_abstention_fallback_v8_3 import (
    FUTURE_EXACT_MARKET_COUNT,
    FUTURE_MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
    FUTURE_SCHEMA_PREFIX,
    build_non_risk_abstention_fallback_v8_3_future_pnl_gate,
    validate_non_risk_abstention_fallback_v8_3_future_plan,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    _prepare_run_dir,
    _result,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze import (
    _runtime_targets_for_decisions,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)


@dataclass(frozen=True, slots=True)
class NonRiskAbstentionFallbackV83FutureSettlementConfig:
    run_id: str
    output_dir: Path | str
    target_free_freeze_manifest_path: Path | str
    expected_target_free_freeze_manifest_sha256: str
    implementation_commit: str
    target_access_started_ts: int
    provider_timeout_seconds: float = 15.0
    provider_http_timeout_seconds: float = 5.0
    settlement_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    max_workers: int = 8
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        _validate_common(
            run_id=self.run_id,
            implementation_commit=self.implementation_commit,
            stage_started_ts=self.target_access_started_ts,
        )
        _require_sha256(
            self.expected_target_free_freeze_manifest_sha256,
            name="expected_target_free_freeze_manifest_sha256",
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "target_free_freeze_manifest_path",
            Path(self.target_free_freeze_manifest_path),
        )


@dataclass(frozen=True, slots=True)
class NonRiskAbstentionFallbackV83FutureEvaluationConfig:
    run_id: str
    output_dir: Path | str
    target_free_freeze_manifest_path: Path | str
    expected_target_free_freeze_manifest_sha256: str
    settled_index_path: Path | str
    expected_settled_index_sha256: str
    runtime_policy_profile_path: Path | str
    expected_runtime_policy_profile_sha256: str
    implementation_commit: str
    evaluation_started_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        _validate_common(
            run_id=self.run_id,
            implementation_commit=self.implementation_commit,
            stage_started_ts=self.evaluation_started_ts,
        )
        for name in (
            "expected_target_free_freeze_manifest_sha256",
            "expected_settled_index_sha256",
            "expected_runtime_policy_profile_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "target_free_freeze_manifest_path",
            "settled_index_path",
            "runtime_policy_profile_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def build_non_risk_abstention_fallback_v8_3_future_settled_index(
    config: NonRiskAbstentionFallbackV83FutureSettlementConfig,
    **settlement_dependencies: Any,
) -> dict[str, Any]:
    """Use the proven read-only settlement engine through a sealed adapter."""

    freeze_path = config.target_free_freeze_manifest_path.resolve()
    freeze, selected, _, _ = _validated_v8_3_freeze(
        freeze_path,
        expected_sha256=config.expected_target_free_freeze_manifest_sha256,
    )
    if config.target_access_started_ts <= max(
        int(freeze["decision_freeze_created_ts"]),
        max(int(row["market_end_ts"]) for row in selected),
    ):
        raise ValueError("#249 target access attempted before freeze or market close")
    adapter_path = freeze_path.parent / "v8_3_future_settlement_engine_adapter.json"
    adapter = _settlement_engine_adapter(
        freeze=freeze,
        freeze_path=freeze_path,
        freeze_sha256=config.expected_target_free_freeze_manifest_sha256,
    )
    if adapter_path.exists():
        if _load_json(adapter_path) != adapter:
            raise ValueError("#249 settlement adapter already exists with different content")
    else:
        _write_json(adapter_path, adapter)
    adapter_sha256 = _sha256_file(adapter_path)
    underlying = build_adaptive_support_controller_v8_1_future_settled_index(
        AdaptiveSupportControllerV81FutureSettlementConfig(
            run_id=config.run_id,
            output_dir=config.output_dir,
            target_free_freeze_manifest_path=adapter_path,
            expected_target_free_freeze_manifest_sha256=adapter_sha256,
            implementation_commit=config.implementation_commit,
            target_access_started_ts=config.target_access_started_ts,
            provider_timeout_seconds=config.provider_timeout_seconds,
            provider_http_timeout_seconds=config.provider_http_timeout_seconds,
            settlement_max_wait_seconds=config.settlement_max_wait_seconds,
            settlement_poll_interval_seconds=config.settlement_poll_interval_seconds,
            max_workers=config.max_workers,
            overwrite_existing=config.overwrite_existing,
        ),
        **settlement_dependencies,
    )
    run_dir = Path(underlying["run_dir"])
    source_report = underlying["report"]
    report = {
        "schema_version": f"{FUTURE_SCHEMA_PREFIX}-settlement-report-v1",
        "run_id": config.run_id,
        "selected_market_count": source_report["selected_market_count"],
        "settled_market_count": source_report["settled_market_count"],
        "unresolved_or_failed_market_count": source_report[
            "unresolved_or_failed_market_count"
        ],
        "settlement_attempt_count": source_report["settlement_attempt_count"],
        "unresolved_or_failed_reason_distribution": source_report[
            "unresolved_or_failed_reason_distribution"
        ],
        "settled_index_ready": source_report["settled_index_ready"],
        "target_access_started_ts": config.target_access_started_ts,
        "v8_1_settlement_engine_used_without_policy_changes": True,
        "settlement_engine_adapter": _descriptor(adapter_path),
        "source_outcome_blind_rounds_mutated": False,
        "official_read_only_resolution_only": True,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        "blocking_reason_codes": source_report["blocking_reason_codes"],
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v8_3_future_settlement_report.json"
    report_md_path = run_dir / "v8_3_future_settlement_report.md"
    _write_json(report_path, report)
    _write_text(
        report_md_path,
        "\n".join(
            [
                "# v8.3 Future Read-Only Settlement",
                "",
                f"- selected markets: `{report['selected_market_count']}`",
                f"- settled markets: `{report['settled_market_count']}`",
                "- unresolved/failed markets: "
                f"`{report['unresolved_or_failed_market_count']}`",
                f"- settled index ready: `{str(report['settled_index_ready']).lower()}`",
                f"- blockers: `{report['blocking_reason_codes']}`",
                "- source outcome-blind rounds mutated: `false`",
                "- paper/live/write/wallet/capital/handoff remain blocked.",
                "",
            ]
        ),
    )
    manifest = {
        "schema_version": f"{FUTURE_SCHEMA_PREFIX}-settlement-manifest-v1",
        "run_id": config.run_id,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "settlement_engine_adapter": _descriptor(adapter_path),
        "underlying_settlement_manifest": _descriptor(
            Path(underlying["manifest_path"])
        ),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "settled_index": (
            _descriptor(Path(underlying["index_path"]))
            if underlying.get("index_path")
            else None
        ),
        "settled_index_ready": report["settled_index_ready"],
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_3_future_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    result = _result(run_dir, report, report_path, manifest, manifest_path)
    result.update(
        {
            "index": underlying.get("index"),
            "index_path": underlying.get("index_path"),
            "index_sha256": underlying.get("index_sha256"),
        }
    )
    return result


def evaluate_non_risk_abstention_fallback_v8_3_future_pnl_gate(
    config: NonRiskAbstentionFallbackV83FutureEvaluationConfig,
) -> dict[str, Any]:
    """Consume complete official targets once for the #249 comparison."""

    freeze_path = config.target_free_freeze_manifest_path.resolve()
    freeze, selected, candidate_decisions, baseline_decisions = (
        _validated_v8_3_freeze(
            freeze_path,
            expected_sha256=config.expected_target_free_freeze_manifest_sha256,
        )
    )
    settled_path = config.settled_index_path.resolve()
    runtime_path = config.runtime_policy_profile_path.resolve()
    _verify_pin(settled_path, config.expected_settled_index_sha256, "#249 settled index")
    _verify_pin(
        runtime_path,
        config.expected_runtime_policy_profile_sha256,
        "#249 runtime profile",
    )
    plan_descriptor = _verified_descriptor(freeze["plan"], "#249 frozen plan")
    plan = _load_json(Path(plan_descriptor["path"]))
    validate_non_risk_abstention_fallback_v8_3_future_plan(plan)
    if (
        config.expected_runtime_policy_profile_sha256.lower()
        != plan["lineage"]["runtime_policy_profile_sha256"]
    ):
        raise ValueError("#249 runtime profile pin drifted")
    settled = _load_json(settled_path)
    adapter_descriptor = _verified_descriptor(
        settled["target_free_freeze_manifest"], "#249 settlement adapter"
    )
    adapter = _load_json(Path(adapter_descriptor["path"]))
    if (
        _verified_descriptor(
            adapter["v8_3_target_free_freeze_manifest"],
            "#249 authoritative freeze",
        )
        != _descriptor(freeze_path)
    ):
        raise ValueError("#249 settlement adapter lineage mismatch")
    market_ids = [str(row["market_id"]) for row in selected]
    entries = _validate_settled_index(
        settled,
        freeze_path=Path(adapter_descriptor["path"]),
        freeze_sha256=adapter_descriptor["sha256"],
        selected_market_ids=market_ids,
        evaluation_started_ts=config.evaluation_started_ts,
    )
    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    claim = {
        "schema_version": f"{FUTURE_SCHEMA_PREFIX}-pnl-gate-claim-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "evaluation_started_ts": config.evaluation_started_ts,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "settled_index": _descriptor(settled_path),
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    claim_path = freeze_path.parent / "v8_3_future_single_use_pnl_gate_claim.json"
    _write_single_use_claim(claim_path, claim)
    runtime_profile = _load_json(runtime_path)
    candidate_targets = _runtime_targets_for_decisions(
        candidate_decisions,
        settled_entries=entries,
        runtime_profile=runtime_profile,
        run_id=config.run_id,
        role="future_unseen_holdout_v8_3_candidate",
    )
    baseline_targets = _runtime_targets_for_decisions(
        baseline_decisions,
        settled_entries=entries,
        runtime_profile=runtime_profile,
        run_id=config.run_id,
        role="future_unseen_holdout_v6_7_baseline",
    )
    gate = build_non_risk_abstention_fallback_v8_3_future_pnl_gate(
        candidate_targets,
        baseline_rows=baseline_targets,
        evaluation_market_ids=market_ids,
        settled_market_ids=[str(row["market_id"]) for row in entries],
        plan=plan,
        target_free_freeze_sha256=config.expected_target_free_freeze_manifest_sha256,
    )
    candidate_path = run_dir / "v8_3_future_candidate_runtime_targets.jsonl"
    baseline_path = run_dir / "v8_3_future_v6_7_runtime_targets.jsonl"
    _write_jsonl(candidate_path, candidate_targets)
    _write_jsonl(baseline_path, baseline_targets)
    report = {
        **gate,
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "evaluation_started_ts": config.evaluation_started_ts,
        "pnl_gate_claim": _descriptor(claim_path),
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        "automatic_paper_or_live_unlock_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    report_path = run_dir / "v8_3_future_pnl_gate_report.json"
    report_md_path = run_dir / "v8_3_future_pnl_gate_report.md"
    _write_json(report_path, report)
    _write_text(
        report_md_path,
        "\n".join(
            [
                "# v8.3 Future PnL Gate",
                "",
                f"- candidate after-cost PnL: `{report['candidate_after_cost_pnl']}`",
                f"- v6.7 after-cost PnL: `{report['v6_7_after_cost_pnl']}`",
                f"- total delta: `{report['candidate_minus_v6_7_after_cost_pnl']}`",
                "- largest-winner-removed delta: "
                f"`{report['candidate_minus_v6_7_largest_winner_removed_after_cost_pnl']}`",
                f"- gate passed: `{str(report['future_pnl_gate_passed']).lower()}`",
                f"- blockers: `{report['future_pnl_gate_blocking_reason_codes']}`",
                "- automatic paper/live unlock: `false`",
                "- future results used for tuning/rerun: `false`",
                "",
            ]
        ),
    )
    manifest = {
        "schema_version": f"{FUTURE_SCHEMA_PREFIX}-pnl-gate-manifest-v1",
        "run_id": config.run_id,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "settled_index": _descriptor(settled_path),
        "runtime_policy_profile": _descriptor(runtime_path),
        "pnl_gate_claim": _descriptor(claim_path),
        "candidate_runtime_targets": _descriptor(candidate_path),
        "v6_7_runtime_targets": _descriptor(baseline_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "future_pnl_gate_passed": report["future_pnl_gate_passed"],
        "future_pnl_gate_blocking_reason_codes": report[
            "future_pnl_gate_blocking_reason_codes"
        ],
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        "automatic_paper_or_live_unlock_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_3_future_pnl_gate_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _validated_v8_3_freeze(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    _verify_pin(path, expected_sha256, "#249 target-free freeze")
    freeze = _load_json(path)
    if (
        freeze.get("schema_version")
        != f"{FUTURE_SCHEMA_PREFIX}-target-free-freeze-manifest-v1"
        or freeze.get("exact_market_count") != FUTURE_EXACT_MARKET_COUNT
        or freeze.get("target_free_freeze_passed") is not True
        or freeze.get("future_target_access_allowed") is not True
        or freeze.get("decision_freeze_written_before_target_access") is not True
        or freeze.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or freeze.get("settlement_provider_called") is not False
        or freeze.get("source_scores_mutated") is not False
        or freeze.get(
            "threshold_model_cost_sizing_guard_or_gate_tuning_performed"
        )
        is not False
    ):
        raise ValueError("#249 target-free freeze is not target-access eligible")
    for field, expected in _v7_0_blocked_safety_fields().items():
        if freeze.get(field) != expected:
            raise ValueError(f"#249 target-free freeze safety mismatch: {field}")
    selected = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_rows"], "#249 selected rows")["path"])
    )
    candidate = _load_jsonl(
        Path(
            _verified_descriptor(
                freeze["candidate_runtime"], "#249 v8.3 runtime decisions"
            )["path"]
        )
    )
    baseline = _load_jsonl(
        Path(
            _verified_descriptor(
                freeze["v6_7_runtime"], "#249 v6.7 runtime decisions"
            )["path"]
        )
    )
    market_ids = [str(row.get("market_id") or "") for row in selected]
    _evaluation_only_frozen_features_by_market(
        freeze, selected_market_ids=market_ids
    )
    if (
        len(selected) != FUTURE_EXACT_MARKET_COUNT
        or "" in market_ids
        or len(set(market_ids)) != FUTURE_EXACT_MARKET_COUNT
        or len(candidate) < FUTURE_MINIMUM_GUARD_ACCEPTED_MARKET_COUNT
        or any(
            str(row.get("market_id") or "") not in set(market_ids)
            for row in candidate + baseline
        )
    ):
        raise ValueError("#249 frozen market or accepted-decision support invalid")
    return freeze, selected, candidate, baseline


def _settlement_engine_adapter(
    *,
    freeze: dict[str, Any],
    freeze_path: Path,
    freeze_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": (
            "bigan-v8-adaptive-support-controller-v8-1-future-holdout-"
            "target-free-freeze-manifest-v1"
        ),
        "candidate_name": freeze["candidate_name"],
        "decision_freeze_created_ts": freeze["decision_freeze_created_ts"],
        "exact_market_count": freeze["exact_market_count"],
        "selected_rows": freeze["selected_rows"],
        "target_free_feature_rows": freeze["target_free_feature_rows"],
        "candidate_runtime": freeze["candidate_runtime"],
        "v6_7_runtime": freeze["v6_7_runtime"],
        "v8_3_target_free_freeze_manifest": {
            "path": str(freeze_path),
            "sha256": freeze_sha256.lower(),
        },
        "target_free_freeze_passed": True,
        "future_target_access_allowed": True,
        "decision_freeze_written_before_target_access": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_scores_mutated": False,
        "threshold_model_or_controller_tuning_performed": False,
        **_v7_0_blocked_safety_fields(),
    }


def _validate_common(
    *,
    run_id: str,
    implementation_commit: str,
    stage_started_ts: int,
) -> None:
    if not run_id.strip():
        raise ValueError("run_id is required")
    if len(implementation_commit) != 40:
        raise ValueError("implementation_commit must be a Git SHA-1")
    if stage_started_ts <= 0:
        raise ValueError("stage timestamp must be positive")


__all__ = [
    "NonRiskAbstentionFallbackV83FutureEvaluationConfig",
    "NonRiskAbstentionFallbackV83FutureSettlementConfig",
    "build_non_risk_abstention_fallback_v8_3_future_settled_index",
    "evaluate_non_risk_abstention_fallback_v8_3_future_pnl_gate",
]
