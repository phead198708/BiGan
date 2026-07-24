"""Hash-pinned target-free future freeze for issue #249 v8.3."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training import (
    execution_layer_v2_adaptive_support_controller_v8_1 as v81,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_canary import (
    _score_window,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout import (
    materialize_adaptive_support_controller_v8_1_runtime_decisions,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_pipeline import (
    _baseline_guard_window,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4_canary import (
    _canonicalize_target_free_sbc_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_non_risk_abstention_fallback_v8_3 import (
    CANDIDATE_NAME,
    FUTURE_SCHEMA_PREFIX,
    build_non_risk_abstention_fallback_v8_3_canary,
    build_non_risk_abstention_fallback_v8_3_target_free_freeze_report,
    select_non_risk_abstention_fallback_v8_3_future_window,
    validate_non_risk_abstention_fallback_v8_3_future_plan,
    validate_non_risk_abstention_fallback_v8_3_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    _prepare_run_dir,
    _result,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    build_v6_7_target_free_candidate_rows,
    select_v6_7_target_free_rows,
    validate_p_up_semantic_compatibility_v6_7_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    _verify_index_raw_descriptors,
    load_and_validate_persistent_outcome_blind_index,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _descriptor,
    _find_nonempty_fields,
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
from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    FORBIDDEN_INFERENCE_FIELDS,
)


@dataclass(frozen=True, slots=True)
class NonRiskAbstentionFallbackV83FutureFreezeConfig:
    """Pinned inputs for the authoritative #249 decision freeze."""

    run_id: str
    output_dir: Path | str
    plan_path: Path | str
    expected_plan_sha256: str
    profile_path: Path | str
    expected_profile_sha256: str
    collector_protocol_path: Path | str
    expected_collector_protocol_sha256: str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    historical_gate_manifest_path: Path | str
    expected_historical_gate_manifest_sha256: str
    issue246_target_free_manifest_path: Path | str
    expected_issue246_target_free_manifest_sha256: str
    target_free_canary_manifest_path: Path | str
    expected_target_free_canary_manifest_sha256: str
    development_batch_manifest_paths: tuple[Path | str, ...]
    expected_development_batch_manifest_sha256s: tuple[str, ...]
    v6_2_batch_manifest_paths: tuple[Path | str, ...]
    expected_v6_2_batch_manifest_sha256s: tuple[str, ...]
    implementation_commit: str
    stage_started_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if len(self.implementation_commit) != 40:
            raise ValueError("implementation_commit must be a Git SHA-1")
        if self.stage_started_ts <= 0:
            raise ValueError("stage_started_ts must be positive")
        for name in (
            "output_dir",
            "plan_path",
            "profile_path",
            "collector_protocol_path",
            "collector_index_path",
            "historical_gate_manifest_path",
            "issue246_target_free_manifest_path",
            "target_free_canary_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(
            self,
            "development_batch_manifest_paths",
            tuple(Path(path) for path in self.development_batch_manifest_paths),
        )
        object.__setattr__(
            self,
            "v6_2_batch_manifest_paths",
            tuple(Path(path) for path in self.v6_2_batch_manifest_paths),
        )
        for name in (
            "expected_plan_sha256",
            "expected_profile_sha256",
            "expected_collector_protocol_sha256",
            "expected_collector_index_sha256",
            "expected_historical_gate_manifest_sha256",
            "expected_issue246_target_free_manifest_sha256",
            "expected_target_free_canary_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for values in (
            self.expected_development_batch_manifest_sha256s,
            self.expected_v6_2_batch_manifest_sha256s,
        ):
            for value in values:
                _require_sha256(value, name="batch_manifest_sha256")
        if (
            not self.development_batch_manifest_paths
            or len(self.development_batch_manifest_paths)
            != len(self.expected_development_batch_manifest_sha256s)
            or len(self.v6_2_batch_manifest_paths)
            != len(self.expected_v6_2_batch_manifest_sha256s)
            or len(self.development_batch_manifest_paths)
            != len(self.v6_2_batch_manifest_paths)
        ):
            raise ValueError("development/v6.2 batch pins must be nonempty and aligned")


def run_non_risk_abstention_fallback_v8_3_future_target_free_freeze(
    config: NonRiskAbstentionFallbackV83FutureFreezeConfig,
) -> dict[str, Any]:
    """Freeze v8.3 and v6.7 decisions before any target access."""

    paths = {
        "plan": config.plan_path.resolve(),
        "profile": config.profile_path.resolve(),
        "protocol": config.collector_protocol_path.resolve(),
        "index": config.collector_index_path.resolve(),
        "historical_gate": config.historical_gate_manifest_path.resolve(),
        "issue246": config.issue246_target_free_manifest_path.resolve(),
        "canary": config.target_free_canary_manifest_path.resolve(),
    }
    pins = {
        "plan": config.expected_plan_sha256,
        "profile": config.expected_profile_sha256,
        "protocol": config.expected_collector_protocol_sha256,
        "index": config.expected_collector_index_sha256,
        "historical_gate": config.expected_historical_gate_manifest_sha256,
        "issue246": config.expected_issue246_target_free_manifest_sha256,
        "canary": config.expected_target_free_canary_manifest_sha256,
    }
    for name, path in paths.items():
        _verify_pin(path, pins[name], f"#249 future {name}")
    plan = _load_json(paths["plan"])
    profile = _load_json(paths["profile"])
    historical_gate = _load_json(paths["historical_gate"])
    issue246 = _load_json(paths["issue246"])
    canary = _load_json(paths["canary"])
    canary_report = _load_json(
        Path(
            _verified_descriptor(
                canary["report"], "#249 target-free canary report"
            )["path"]
        )
    )
    validate_non_risk_abstention_fallback_v8_3_future_plan(plan)
    validate_non_risk_abstention_fallback_v8_3_profile(profile)
    _validate_frozen_lineage(
        plan=plan,
        profile_sha256=pins["profile"],
        protocol_sha256=pins["protocol"],
        historical_gate=historical_gate,
        historical_gate_sha256=pins["historical_gate"],
        issue246=issue246,
        issue246_sha256=pins["issue246"],
        canary=canary,
        canary_report=canary_report,
        canary_sha256=pins["canary"],
    )

    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    snapshot = run_dir / "v8_3_future_collector_index_snapshot.jsonl"
    shutil.copyfile(paths["index"], snapshot)
    if _sha256_file(snapshot) != pins["index"].lower():
        raise ValueError("#249 collector index changed while snapshotting")
    index_rows = load_and_validate_persistent_outcome_blind_index(snapshot)
    v8_1_historical_path = Path(
        _verified_descriptor(
            historical_gate["v8_1_historical_manifest"],
            "#249 v8.1 historical manifest",
        )["path"]
    )
    v8_1_historical = _load_json(v8_1_historical_path)
    prior = _prior_reference_sets(
        v8_1_historical=v8_1_historical,
        issue246=issue246,
        canary=canary,
    )
    selected, attempted, selection = (
        select_non_risk_abstention_fallback_v8_3_future_window(
            index_rows,
            plan=plan,
            prior_market_ids=prior["market_ids"],
            prior_slugs=prior["slugs"],
            prior_decision_ids=prior["decision_ids"],
            prior_source_row_hashes=prior["source_row_hashes"],
        )
    )
    if selection["exact_window_ready"] is not True:
        raise ValueError("#249 exact-120 target-free window is not ready")
    for row in selected:
        _verify_index_raw_descriptors(row)

    development, v6_2 = _load_pinned_batch_manifests(config, plan=plan)
    action_rows, feature_rows, scored_rows = _load_target_free_rows(
        development, v6_2
    )
    selected_set = {str(row["market_id"]) for row in selected}
    action_rows = [
        row for row in action_rows if str(row.get("market_id")) in selected_set
    ]
    feature_rows = [
        row for row in feature_rows if str(row.get("market_id")) in selected_set
    ]
    scored_rows = [
        row for row in scored_rows if str(row.get("market_id")) in selected_set
    ]
    forbidden = sorted(
        set(_find_nonempty_fields(action_rows, FORBIDDEN_INFERENCE_FIELDS))
        | set(_find_nonempty_fields(feature_rows, FORBIDDEN_INFERENCE_FIELDS))
        | set(_find_nonempty_fields(scored_rows, FORBIDDEN_INFERENCE_FIELDS))
    )
    if forbidden:
        raise ValueError("#249 target-free inputs contain targets: " + ",".join(forbidden))

    model_descriptor = _verified_descriptor(
        v8_1_historical["model"], "#249 v8.1 model"
    )
    v8_1_profile_descriptor = _verified_descriptor(
        v8_1_historical["profile"], "#249 v8.1 profile"
    )
    v6_7_descriptor = _verified_descriptor(
        v8_1_historical["v6_7_candidate_profile"], "#249 v6.7 profile"
    )
    v7_0_descriptor = _verified_descriptor(
        v8_1_historical["v7_0_training_profile"], "#249 v7.0 profile"
    )
    model = _load_json(Path(model_descriptor["path"]))
    v8_1_profile = _load_json(Path(v8_1_profile_descriptor["path"]))
    v6_7_profile = _load_json(Path(v6_7_descriptor["path"]))
    v7_0_profile = _load_json(Path(v7_0_descriptor["path"]))
    v81.validate_adaptive_support_controller_v8_1_profile(v8_1_profile)
    validate_p_up_semantic_compatibility_v6_7_profile(v6_7_profile)

    selected_ids = [str(row["market_id"]) for row in selected]
    v6_7_candidates, candidate_summary = build_v6_7_target_free_candidate_rows(
        scored_rows, action_rows=action_rows, profile=v6_7_profile
    )
    baseline_rows = select_v6_7_target_free_rows(
        v6_7_candidates, profile=v6_7_profile
    )
    canonical_rows, canonical_summary = _canonicalize_target_free_sbc_rows(
        scored_rows,
        action_rows=action_rows,
        v6_7_profile=v6_7_profile,
        v7_0_profile=v7_0_profile,
    )
    v8_1_decisions, candidate_guard, final_state = _score_window(
        selected_ids,
        canonical_rows=canonical_rows,
        baseline_rows=baseline_rows,
        action_rows=action_rows,
        model=model,
        v6_7_profile=v6_7_profile,
    )
    baseline_guard = _baseline_guard_window(
        selected_ids,
        baseline_rows=baseline_rows,
        action_rows=action_rows,
        v6_7_profile=v6_7_profile,
    )
    overlay = build_non_risk_abstention_fallback_v8_3_canary(
        candidate_rows=candidate_guard,
        baseline_rows=baseline_guard,
        profile=profile,
    )
    overlay_decisions = overlay["decisions"]
    candidate_runtime = materialize_adaptive_support_controller_v8_1_runtime_decisions(
        overlay_decisions, action_rows=action_rows
    )
    baseline_runtime = materialize_adaptive_support_controller_v8_1_runtime_decisions(
        baseline_guard, action_rows=action_rows
    )
    report = build_non_risk_abstention_fallback_v8_3_target_free_freeze_report(
        selected,
        attempted_rows=attempted,
        action_rows=action_rows,
        overlay_decisions=overlay_decisions,
        baseline_guard_rows=baseline_guard,
        selection_summary=selection,
        plan=plan,
        stage_started_ts=config.stage_started_ts,
        collector_index_sha256=pins["index"].lower(),
    )
    report.update(
        {
            "run_id": config.run_id,
            "implementation_commit": config.implementation_commit,
            "decision_freeze_created_ts": config.stage_started_ts,
            "prior_reference_hash": prior["prior_reference_hash"],
            "prior_reference_source_row_counts": prior["source_row_counts"],
            "v6_7_candidate_summary": candidate_summary,
            "canonical_mapping_summary": canonical_summary,
        }
    )
    report["report_id"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    return _write_outputs(
        run_dir=run_dir,
        report=report,
        plan_path=paths["plan"],
        profile_path=paths["profile"],
        protocol_path=paths["protocol"],
        index_snapshot=snapshot,
        historical_gate_path=paths["historical_gate"],
        issue246_path=paths["issue246"],
        canary_path=paths["canary"],
        selected=selected,
        attempted=attempted,
        action_rows=action_rows,
        canonical_rows=canonical_rows,
        baseline_rows=baseline_rows,
        v8_1_decisions=v8_1_decisions,
        candidate_guard=candidate_guard,
        baseline_guard=baseline_guard,
        overlay_decisions=overlay_decisions,
        candidate_runtime=candidate_runtime,
        baseline_runtime=baseline_runtime,
        final_state=final_state,
        development=development,
        v6_2=v6_2,
        model_descriptor=model_descriptor,
        v8_1_profile_descriptor=v8_1_profile_descriptor,
        v6_7_descriptor=v6_7_descriptor,
        v7_0_descriptor=v7_0_descriptor,
    )


def _validate_frozen_lineage(
    *,
    plan: dict[str, Any],
    profile_sha256: str,
    protocol_sha256: str,
    historical_gate: dict[str, Any],
    historical_gate_sha256: str,
    issue246: dict[str, Any],
    issue246_sha256: str,
    canary: dict[str, Any],
    canary_report: dict[str, Any],
    canary_sha256: str,
) -> None:
    lineage = dict(plan["lineage"])
    if (
        profile_sha256.lower() != lineage["candidate_profile_sha256"]
        or protocol_sha256.lower() != lineage["collector_protocol_sha256"]
        or historical_gate_sha256.lower()
        != lineage["historical_gate_manifest_sha256"]
        or issue246_sha256.lower()
        != lineage["issue246_target_free_manifest_sha256"]
        or canary_sha256.lower()
        != lineage["issue246_target_free_canary_manifest_sha256"]
        or historical_gate.get("historical_noninferiority_gate_passed") is not True
        or issue246.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or issue246.get("settlement_provider_called") is not False
        or canary.get("target_free_canary_passed") is not True
        or canary.get("issue246_outcomes_opened") is not False
        or canary_report.get("labels_outcomes_resolution_or_pnl_opened") is not False
    ):
        raise ValueError("#249 frozen lineage is invalid")


def _prior_reference_sets(
    *,
    v8_1_historical: dict[str, Any],
    issue246: dict[str, Any],
    canary: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        *_load_jsonl(
            Path(
                _verified_descriptor(
                    v8_1_historical["rank_lineage_rows"],
                    "#249 historical rank lineage",
                )["path"]
            )
        ),
        *_load_jsonl(
            Path(
                _verified_descriptor(
                    issue246["selected_rows"], "#249 issue246 selected rows"
                )["path"]
            )
        ),
        *_load_jsonl(
            Path(
                _verified_descriptor(
                    issue246["action_rows"], "#249 issue246 action rows"
                )["path"]
            )
        ),
        *_load_jsonl(
            Path(
                _verified_descriptor(
                    canary["decision_rows"], "#249 prior canary decisions"
                )["path"]
            )
        ),
    ]
    forbidden = _find_nonempty_fields(rows, FORBIDDEN_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError("#249 prior identity registry opened targets")
    output = {
        "market_ids": {str(row.get("market_id") or "") for row in rows} - {""},
        "slugs": {
            str(row.get("slug") or row.get("market_slug") or "") for row in rows
        }
        - {""},
        "decision_ids": {str(row.get("decision_id") or "") for row in rows} - {""},
        "source_row_hashes": {
            str(
                row.get("source_row_hash")
                or row.get("source_feature_row_sha256")
                or ""
            )
            for row in rows
        }
        - {""},
        "source_row_counts": {"combined_prior_rows": len(rows)},
    }
    output["prior_reference_hash"] = canonical_json_sha256(
        {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in output.items()
        }
    )
    return output


def _load_pinned_batch_manifests(
    config: NonRiskAbstentionFallbackV83FutureFreezeConfig,
    *,
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    development: list[dict[str, Any]] = []
    v6_2: list[dict[str, Any]] = []
    expected_feature_contract = str(plan["lineage"]["feature_contract_sha256"])
    source_candidate_sha256: str | None = None
    for dev_path, dev_pin, score_path, score_pin in zip(
        config.development_batch_manifest_paths,
        config.expected_development_batch_manifest_sha256s,
        config.v6_2_batch_manifest_paths,
        config.expected_v6_2_batch_manifest_sha256s,
        strict=True,
    ):
        dev_path = dev_path.resolve()
        score_path = score_path.resolve()
        _verify_pin(dev_path, dev_pin, "#249 development batch manifest")
        _verify_pin(score_path, score_pin, "#249 v6.2 batch manifest")
        dev = _load_json(dev_path)
        score = _load_json(score_path)
        candidate = _verified_descriptor(
            score["candidate_manifest"], "#249 v6.2 source candidate"
        )
        if source_candidate_sha256 is None:
            source_candidate_sha256 = candidate["sha256"]
        if (
            dev.get("development_data_canary_passed") is not True
            or dev.get("candidate_model_scoring_attempted") is not False
            or dev.get("labels_outcomes_or_pnl_opened") is not False
            or _verified_descriptor(
                dev["feature_contract"], "#249 feature contract"
            )["sha256"]
            != expected_feature_contract
            or score.get("labels_outcomes_or_pnl_opened") is not False
            or score.get("batch_id") != dev.get("batch_id")
            or _verified_descriptor(
                score["development_batch_canary_manifest"],
                "#249 matched development",
            )["sha256"]
            != dev_pin.lower()
            or candidate["sha256"] != source_candidate_sha256
        ):
            raise ValueError("#249 batch scoring lineage is invalid")
        dev["_manifest_path"] = str(dev_path)
        score["_manifest_path"] = str(score_path)
        development.append(dev)
        v6_2.append(score)
    batch_ids = [str(row.get("batch_id") or "") for row in development]
    if "" in batch_ids or len(set(batch_ids)) != len(batch_ids):
        raise ValueError("#249 batch identity missing or duplicated")
    return development, v6_2


def _load_target_free_rows(
    development: list[dict[str, Any]],
    v6_2: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    for dev, score in zip(development, v6_2, strict=True):
        actions.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        dev["five_action_grid"], "#249 five-action grid"
                    )["path"]
                )
            )
        )
        features.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        dev["feature_rows"], "#249 feature rows"
                    )["path"]
                )
            )
        )
        scored.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        score["mean_ev_scored_rows"], "#249 scored rows"
                    )["path"]
                )
            )
        )
    return actions, features, scored


def _write_outputs(
    *,
    run_dir: Path,
    report: dict[str, Any],
    plan_path: Path,
    profile_path: Path,
    protocol_path: Path,
    index_snapshot: Path,
    historical_gate_path: Path,
    issue246_path: Path,
    canary_path: Path,
    selected: list[dict[str, Any]],
    attempted: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    v8_1_decisions: list[dict[str, Any]],
    candidate_guard: list[dict[str, Any]],
    baseline_guard: list[dict[str, Any]],
    overlay_decisions: list[dict[str, Any]],
    candidate_runtime: list[dict[str, Any]],
    baseline_runtime: list[dict[str, Any]],
    final_state: dict[str, Any],
    development: list[dict[str, Any]],
    v6_2: list[dict[str, Any]],
    model_descriptor: dict[str, Any],
    v8_1_profile_descriptor: dict[str, Any],
    v6_7_descriptor: dict[str, Any],
    v7_0_descriptor: dict[str, Any],
) -> dict[str, Any]:
    outputs = {
        "selected_rows": run_dir / "v8_3_future_selected_index_rows.jsonl",
        "attempted_rows": run_dir / "v8_3_future_attempted_index_rows.jsonl",
        "action_rows": run_dir / "v8_3_future_five_action_rows.jsonl",
        "canonical_rows": run_dir / "v8_3_future_canonical_sbc_rows.jsonl",
        "baseline_rows": run_dir / "v8_3_future_v6_7_baseline_rows.jsonl",
        "v8_1_decisions": run_dir / "v8_3_future_v8_1_decisions.jsonl",
        "candidate_guard": run_dir / "v8_3_future_v8_1_guard_replay.jsonl",
        "baseline_guard": run_dir / "v8_3_future_v6_7_guard_replay.jsonl",
        "overlay_decisions": run_dir / "v8_3_future_overlay_decisions.jsonl",
        "candidate_runtime": run_dir / "v8_3_future_candidate_runtime.jsonl",
        "v6_7_runtime": run_dir / "v8_3_future_v6_7_runtime.jsonl",
    }
    rows_by_name = {
        "selected_rows": selected,
        "attempted_rows": attempted,
        "action_rows": action_rows,
        "canonical_rows": canonical_rows,
        "baseline_rows": baseline_rows,
        "v8_1_decisions": v8_1_decisions,
        "candidate_guard": candidate_guard,
        "baseline_guard": baseline_guard,
        "overlay_decisions": overlay_decisions,
        "candidate_runtime": candidate_runtime,
        "v6_7_runtime": baseline_runtime,
    }
    for name, rows in rows_by_name.items():
        _write_jsonl(outputs[name], rows)
    state_path = run_dir / "v8_3_future_final_controller_state.json"
    _write_json(state_path, final_state)
    report_path = run_dir / "v8_3_future_target_free_freeze_report.json"
    report_md_path = run_dir / "v8_3_future_target_free_freeze_report.md"
    _write_json(report_path, report)
    _write_text(
        report_md_path,
        "\n".join(
            [
                "# v8.3 Future Target-Free Freeze",
                "",
                f"- selected markets: `{report['selected_market_count']}`",
                "- candidate guard accepted: "
                f"`{report['candidate_guard_accepted_market_count']}`",
                f"- selection sources: `{report['selection_source_distribution']}`",
                "- target-free freeze passed: "
                f"`{str(report['target_free_freeze_passed']).lower()}`",
                f"- blockers: `{report['target_free_blocking_reason_codes']}`",
                "- outcomes/resolution/labels/PnL opened: `false`",
                "- paper/live/write/wallet/capital/handoff remain blocked.",
                "",
            ]
        ),
    )
    manifest = {
        "schema_version": f"{FUTURE_SCHEMA_PREFIX}-target-free-freeze-manifest-v1",
        "run_id": report["run_id"],
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": report["implementation_commit"],
        "decision_freeze_created_ts": report["decision_freeze_created_ts"],
        "exact_market_count": report["selected_market_count"],
        "plan": _descriptor(plan_path),
        "profile": _descriptor(profile_path),
        "collector_protocol": _descriptor(protocol_path),
        "collector_index_snapshot": _descriptor(index_snapshot),
        "historical_gate_manifest": _descriptor(historical_gate_path),
        "issue246_target_free_manifest": _descriptor(issue246_path),
        "target_free_canary_manifest": _descriptor(canary_path),
        "model": model_descriptor,
        "v8_1_profile": v8_1_profile_descriptor,
        "v6_7_profile": v6_7_descriptor,
        "v7_0_profile": v7_0_descriptor,
        "development_batch_manifests": [
            _descriptor(Path(row["_manifest_path"])) for row in development
        ],
        "v6_2_batch_manifests": [
            _descriptor(Path(row["_manifest_path"])) for row in v6_2
        ],
        **{name: _descriptor(path) for name, path in outputs.items()},
        "final_controller_state": _descriptor(state_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "target_free_freeze_passed": report["target_free_freeze_passed"],
        "target_free_blocking_reason_codes": report[
            "target_free_blocking_reason_codes"
        ],
        "future_target_access_allowed": report["future_target_access_allowed"],
        "decision_freeze_written_before_target_access": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_scores_mutated": False,
        "threshold_model_cost_sizing_guard_or_gate_tuning_performed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_3_future_target_free_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


__all__ = [
    "NonRiskAbstentionFallbackV83FutureFreezeConfig",
    "run_non_risk_abstention_fallback_v8_3_future_target_free_freeze",
]
