"""Hash-pinned target-free future freeze pipeline for issue #246 v8.1."""

from __future__ import annotations

import shutil
from collections import Counter
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
    FORBIDDEN_TARGET_FIELDS,
    FROZEN_PLAN_SHA256,
    _v7_0_blocked_safety_fields,
    build_adaptive_support_controller_v8_1_target_free_freeze_report,
    materialize_adaptive_support_controller_v8_1_runtime_decisions,
    select_adaptive_support_controller_v8_1_future_holdout_window,
    validate_adaptive_support_controller_v8_1_future_holdout_plan,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4_canary import (
    _canonicalize_target_free_sbc_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    _prepare_run_dir,
    _result,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    _microstructure_blocking_reasons,
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

SCHEMA_PREFIX = "bigan-v8-adaptive-support-controller-v8-1-future-holdout"


@dataclass(frozen=True, slots=True)
class AdaptiveSupportControllerV81FutureFreezeConfig:
    """Pinned inputs for the authoritative target-free decision freeze."""

    run_id: str
    output_dir: Path | str
    plan_path: Path | str
    expected_plan_sha256: str
    collector_protocol_path: Path | str
    expected_collector_protocol_sha256: str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    historical_manifest_path: Path | str
    expected_historical_manifest_sha256: str
    prior_canary_index_path: Path | str
    expected_prior_canary_index_sha256: str
    prior_canary_manifest_path: Path | str
    expected_prior_canary_manifest_sha256: str
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
            "collector_protocol_path",
            "collector_index_path",
            "historical_manifest_path",
            "prior_canary_index_path",
            "prior_canary_manifest_path",
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
            "expected_collector_protocol_sha256",
            "expected_collector_index_sha256",
            "expected_historical_manifest_sha256",
            "expected_prior_canary_index_sha256",
            "expected_prior_canary_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for name, values in (
            (
                "expected_development_batch_manifest_sha256s",
                self.expected_development_batch_manifest_sha256s,
            ),
            (
                "expected_v6_2_batch_manifest_sha256s",
                self.expected_v6_2_batch_manifest_sha256s,
            ),
        ):
            for value in values:
                _require_sha256(value, name=name)
        if (
            not self.development_batch_manifest_paths
            or len(self.development_batch_manifest_paths)
            != len(self.expected_development_batch_manifest_sha256s)
            or len(self.v6_2_batch_manifest_paths)
            != len(self.expected_v6_2_batch_manifest_sha256s)
            or len(self.development_batch_manifest_paths)
            != len(self.v6_2_batch_manifest_paths)
        ):
            raise ValueError(
                "development/v6.2 batch manifest pins must be nonempty and aligned"
            )


def run_adaptive_support_controller_v8_1_future_target_free_freeze(
    config: AdaptiveSupportControllerV81FutureFreezeConfig,
) -> dict[str, Any]:
    """Select exact-120 and seal both policies before any target access."""

    paths = {
        "plan": config.plan_path.resolve(),
        "protocol": config.collector_protocol_path.resolve(),
        "index": config.collector_index_path.resolve(),
        "historical": config.historical_manifest_path.resolve(),
        "prior_canary_index": config.prior_canary_index_path.resolve(),
        "prior_canary_manifest": config.prior_canary_manifest_path.resolve(),
    }
    pins = {
        "plan": config.expected_plan_sha256,
        "protocol": config.expected_collector_protocol_sha256,
        "index": config.expected_collector_index_sha256,
        "historical": config.expected_historical_manifest_sha256,
        "prior_canary_index": config.expected_prior_canary_index_sha256,
        "prior_canary_manifest": config.expected_prior_canary_manifest_sha256,
    }
    for name, path in paths.items():
        _verify_pin(path, pins[name], f"#246 {name}")
    if pins["plan"].lower() != FROZEN_PLAN_SHA256:
        raise ValueError("#246 future target-free freeze plan pin drifted")
    plan = _load_json(paths["plan"])
    validate_adaptive_support_controller_v8_1_future_holdout_plan(plan)
    lineage = dict(plan["lineage"])
    if pins["protocol"].lower() != lineage["collector_protocol_sha256"]:
        raise ValueError("#246 collector protocol pin drifted")
    _validate_prior_canary(paths, pins=pins, plan=plan)
    historical = _load_json(paths["historical"])
    _validate_historical_lineage(
        historical,
        plan=plan,
        manifest_sha256=pins["historical"],
    )

    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    index_snapshot = run_dir / "v8_1_future_collector_index_snapshot.jsonl"
    shutil.copyfile(paths["index"], index_snapshot)
    if _sha256_file(index_snapshot) != pins["index"].lower():
        raise ValueError("#246 collector index changed while snapshotting")
    index_rows = load_and_validate_persistent_outcome_blind_index(index_snapshot)
    prior = _prior_reference_sets(
        historical=historical,
        prior_canary_index_path=paths["prior_canary_index"],
    )
    selected, attempted, selection = (
        select_adaptive_support_controller_v8_1_future_holdout_window(
            index_rows,
            plan=plan,
            prior_market_ids=prior["market_ids"],
            prior_slugs=prior["slugs"],
            prior_decision_ids=prior["decision_ids"],
            prior_source_row_hashes=prior["source_row_hashes"],
        )
    )
    if selection["exact_window_ready"] is not True:
        raise ValueError("#246 exact-120 target-free window is not ready")
    for row in selected:
        _verify_index_raw_descriptors(row)

    development, v6_2 = _load_pinned_batch_manifests(config, plan=plan)
    action_rows, feature_rows, scored_rows = _load_batch_target_free_rows(
        development,
        v6_2,
    )
    selected_ids = [str(row["market_id"]) for row in selected]
    selected_set = set(selected_ids)
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
        set(_find_nonempty_fields(action_rows, FORBIDDEN_TARGET_FIELDS))
        | set(_find_nonempty_fields(feature_rows, FORBIDDEN_TARGET_FIELDS))
        | set(_find_nonempty_fields(scored_rows, FORBIDDEN_TARGET_FIELDS))
    )
    if forbidden:
        raise ValueError(
            "#246 future target-free inputs contain targets: " + ",".join(forbidden)
        )

    model_descriptor = _verified_descriptor(historical["model"], "#246 model")
    profile_descriptor = _verified_descriptor(
        historical["profile"], "#246 v8.1 profile"
    )
    v6_7_descriptor = _verified_descriptor(
        historical["v6_7_candidate_profile"], "#246 v6.7 profile"
    )
    v7_0_descriptor = _verified_descriptor(
        historical["v7_0_training_profile"], "#246 v7.0 profile"
    )
    model = _load_json(Path(model_descriptor["path"]))
    profile = _load_json(Path(profile_descriptor["path"]))
    v6_7_profile = _load_json(Path(v6_7_descriptor["path"]))
    v7_0_profile = _load_json(Path(v7_0_descriptor["path"]))
    v81.validate_adaptive_support_controller_v8_1_profile(profile)
    validate_p_up_semantic_compatibility_v6_7_profile(v6_7_profile)

    v6_7_candidates, candidate_summary = build_v6_7_target_free_candidate_rows(
        scored_rows,
        action_rows=action_rows,
        profile=v6_7_profile,
    )
    baseline_rows = select_v6_7_target_free_rows(
        v6_7_candidates,
        profile=v6_7_profile,
    )
    canonical_rows, canonical_summary = _canonicalize_target_free_sbc_rows(
        scored_rows,
        action_rows=action_rows,
        v6_7_profile=v6_7_profile,
        v7_0_profile=v7_0_profile,
    )
    decisions, candidate_guard, final_state = _score_window(
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
    report = build_adaptive_support_controller_v8_1_target_free_freeze_report(
        selected,
        attempted_rows=attempted,
        action_rows=action_rows,
        candidate_guard_rows=candidate_guard,
        baseline_guard_rows=baseline_guard,
        selection_summary=selection,
        plan=plan,
        stage_started_ts=config.stage_started_ts,
        collector_index_sha256=pins["index"].lower(),
    )
    report["run_id"] = config.run_id
    candidate_runtime = (
        materialize_adaptive_support_controller_v8_1_runtime_decisions(
            candidate_guard,
            action_rows=action_rows,
        )
    )
    baseline_runtime = (
        materialize_adaptive_support_controller_v8_1_runtime_decisions(
            baseline_guard,
            action_rows=action_rows,
        )
    )
    return _write_freeze_artifacts(
        run_dir=run_dir,
        report=report,
        plan_path=paths["plan"],
        protocol_path=paths["protocol"],
        index_snapshot=index_snapshot,
        historical_path=paths["historical"],
        prior_canary_index_path=paths["prior_canary_index"],
        prior_canary_manifest_path=paths["prior_canary_manifest"],
        selected=selected,
        attempted=attempted,
        action_rows=action_rows,
        canonical_rows=canonical_rows,
        baseline_rows=baseline_rows,
        decisions=decisions,
        candidate_guard=candidate_guard,
        baseline_guard=baseline_guard,
        candidate_runtime=candidate_runtime,
        baseline_runtime=baseline_runtime,
        final_state=final_state,
        development=development,
        v6_2=v6_2,
        prior=prior,
        candidate_summary=candidate_summary,
        canonical_summary=canonical_summary,
        model_descriptor=model_descriptor,
        profile_descriptor=profile_descriptor,
        v6_7_descriptor=v6_7_descriptor,
        v7_0_descriptor=v7_0_descriptor,
        implementation_commit=config.implementation_commit,
    )


def _validate_prior_canary(
    paths: dict[str, Path],
    *,
    pins: dict[str, str],
    plan: dict[str, Any],
) -> None:
    lineage = dict(plan["lineage"])
    canary_rows = load_and_validate_persistent_outcome_blind_index(
        paths["prior_canary_index"]
    )
    canary_manifest = _load_json(paths["prior_canary_manifest"])
    if (
        pins["prior_canary_index"].lower()
        != lineage["target_free_canary_batch_index_sha256"]
        or not canary_rows
        or canary_rows[-1]["entry_sha256"]
        != lineage["target_free_canary_batch_last_entry_sha256"]
        or pins["prior_canary_manifest"].lower()
        != lineage["target_free_canary_manifest_sha256"]
        or canary_manifest.get("target_free_canary_passed") is not True
        or canary_manifest.get("labels_outcomes_resolution_or_pnl_opened") is not False
    ):
        raise ValueError("#246 prior target-free canary lineage is invalid")


def _validate_historical_lineage(
    historical: dict[str, Any],
    *,
    plan: dict[str, Any],
    manifest_sha256: str,
) -> None:
    lineage = dict(plan["lineage"])
    model = _verified_descriptor(historical["model"], "#246 model")
    profile = _verified_descriptor(historical["profile"], "#246 profile")
    v6_7 = _verified_descriptor(
        historical["v6_7_candidate_profile"], "#246 v6.7 profile"
    )
    v7_0 = _verified_descriptor(
        historical["v7_0_training_profile"], "#246 v7.0 profile"
    )
    runtime = _verified_descriptor(
        historical["runtime_policy_profile"], "#246 runtime profile"
    )
    model_payload = _load_json(Path(model["path"]))
    if (
        manifest_sha256.lower() != lineage["historical_manifest_sha256"]
        or model["sha256"] != lineage["historical_model_sha256"]
        or profile["sha256"] != lineage["candidate_profile_sha256"]
        or v6_7["sha256"] != lineage["v6_7_profile_sha256"]
        or v7_0["sha256"] != lineage["v7_0_training_profile_sha256"]
        or runtime["sha256"] != lineage["runtime_policy_profile_sha256"]
        or historical.get("historical_hard_gate_passed") is not True
        or historical.get("target_free_canary_collection_allowed") is not True
        or historical.get("fit_leakage_audit_passed") is not True
        or model_payload.get("schema_version") != v81.MODEL_SCHEMA_VERSION
        or _sha256_file(Path(v81.__file__))
        != lineage["candidate_decision_policy_source_sha256"]
    ):
        raise ValueError("#246 frozen historical lineage is invalid")


def _prior_reference_sets(
    *,
    historical: dict[str, Any],
    prior_canary_index_path: Path,
) -> dict[str, Any]:
    canary_rows = load_and_validate_persistent_outcome_blind_index(
        prior_canary_index_path
    )
    historical_rows: list[dict[str, Any]] = []
    for key in ("seed_runtime_target_rows", "consumed_stream_five_action_rows"):
        historical_rows.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        historical[key],
                        f"#246 prior {key}",
                    )["path"]
                )
            )
        )
    all_rows = [*historical_rows, *canary_rows]
    forbidden = _find_nonempty_fields(all_rows, FORBIDDEN_TARGET_FIELDS)
    if forbidden:
        raise ValueError("#246 prior identity registry opened targets")
    output = {
        "market_ids": {str(row.get("market_id") or "") for row in all_rows} - {""},
        "slugs": {
            str(row.get("slug") or row.get("market_slug") or "")
            for row in all_rows
        }
        - {""},
        "decision_ids": {str(row.get("decision_id") or "") for row in all_rows}
        - {""},
        "source_row_hashes": {
            str(
                row.get("source_row_hash")
                or row.get("source_feature_row_sha256")
                or ""
            )
            for row in all_rows
        }
        - {""},
        "source_row_counts": {
            "historical_rows": len(historical_rows),
            "target_free_canary_index": len(canary_rows),
        },
    }
    output["prior_reference_hash"] = canonical_json_sha256(
        {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in output.items()
        }
    )
    return output


def _load_pinned_batch_manifests(
    config: AdaptiveSupportControllerV81FutureFreezeConfig,
    *,
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    development: list[dict[str, Any]] = []
    v6_2: list[dict[str, Any]] = []
    expected_source = str(plan["lineage"]["v6_2_source_candidate_manifest_sha256"])
    expected_feature_contract = str(plan["lineage"]["feature_contract_sha256"])
    for dev_path, dev_pin, score_path, score_pin in zip(
        config.development_batch_manifest_paths,
        config.expected_development_batch_manifest_sha256s,
        config.v6_2_batch_manifest_paths,
        config.expected_v6_2_batch_manifest_sha256s,
        strict=True,
    ):
        dev_path = dev_path.resolve()
        score_path = score_path.resolve()
        _verify_pin(dev_path, dev_pin, "#246 development batch manifest")
        _verify_pin(score_path, score_pin, "#246 v6.2 batch manifest")
        dev = _load_json(dev_path)
        score = _load_json(score_path)
        if (
            dev.get("development_data_canary_passed") is not True
            or dev.get("candidate_model_scoring_attempted") is not False
            or dev.get("labels_outcomes_or_pnl_opened") is not False
            or _verified_descriptor(dev["feature_contract"], "#246 feature contract")[
                "sha256"
            ]
            != expected_feature_contract
        ):
            raise ValueError("#246 development batch is not target-free eligible")
        if (
            score.get("labels_outcomes_or_pnl_opened") is not False
            or score.get("batch_id") != dev.get("batch_id")
            or _verified_descriptor(
                score["development_batch_canary_manifest"],
                "#246 matched development",
            )["sha256"]
            != dev_pin.lower()
            or _verified_descriptor(score["candidate_manifest"], "#246 v6.2 source")[
                "sha256"
            ]
            != expected_source
        ):
            raise ValueError("#246 v6.2 batch scoring lineage is invalid")
        dev["_manifest_path"] = str(dev_path)
        score["_manifest_path"] = str(score_path)
        development.append(dev)
        v6_2.append(score)
    batch_ids = [str(row.get("batch_id") or "") for row in development]
    if "" in batch_ids or len(set(batch_ids)) != len(batch_ids):
        raise ValueError("#246 batch identity is missing or duplicated")
    return development, v6_2


def _load_batch_target_free_rows(
    development: list[dict[str, Any]],
    v6_2: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    action_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    for dev, score in zip(development, v6_2, strict=True):
        action_rows.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        dev["five_action_grid"], "#246 action grid"
                    )["path"]
                )
            )
        )
        feature_rows.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(dev["feature_rows"], "#246 feature rows")[
                        "path"
                    ]
                )
            )
        )
        scored_rows.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        score["mean_ev_scored_rows"], "#246 scored rows"
                    )["path"]
                )
            )
        )
    return action_rows, feature_rows, scored_rows


def _baseline_guard_window(
    selected_market_ids: list[str],
    *,
    baseline_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    v6_7_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_by_market = {str(row["market_id"]): row for row in baseline_rows}
    source_by_key = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["action"])): row
        for row in action_rows
    }
    output = []
    for market_id in selected_market_ids:
        baseline = baseline_by_market.get(market_id)
        if baseline is None:
            action = "NO_TRADE"
            side = "NONE"
            reasons = ["v6_7_no_positive_guard_compatible_action"]
            source = None
        else:
            action = str(baseline["action"])
            side = "UP" if action.startswith("BUY_UP_") else "DOWN"
            source = source_by_key.get(
                (market_id, int(baseline["decision_ts"]), action)
            )
            reasons = (
                ["selected_action_source_row_missing"]
                if source is None
                else _microstructure_blocking_reasons(
                    source,
                    guard=v6_7_profile["hard_execution_safety"],
                )
            )
        row = {
            "market_id": market_id,
            "decision_ts": int(source.get("decision_ts") or 0) if source else 0,
            "selected_action": action,
            "selected_side": side,
            "execution_guard_order_allowed": source is not None and not reasons,
            "execution_blocking_reason_codes": reasons,
            "p_up_action_disagreement_diagnostic_only": True,
            "p_up_side_alignment_filter_enabled": False,
            "pre_entry_market_exposure": 0.0,
            "same_market_position_exists": False,
            "same_side_position_exists": False,
            "full_execution_guard_unchanged": True,
            "source_score_mutated": False,
            "labels_outcomes_or_pnl_opened": False,
        }
        row["guard_replay_row_id"] = canonical_json_sha256(row)
        output.append(row)
    return output


def _write_freeze_artifacts(
    *,
    run_dir: Path,
    report: dict[str, Any],
    plan_path: Path,
    protocol_path: Path,
    index_snapshot: Path,
    historical_path: Path,
    prior_canary_index_path: Path,
    prior_canary_manifest_path: Path,
    selected: list[dict[str, Any]],
    attempted: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    candidate_guard: list[dict[str, Any]],
    baseline_guard: list[dict[str, Any]],
    candidate_runtime: list[dict[str, Any]],
    baseline_runtime: list[dict[str, Any]],
    final_state: dict[str, Any],
    development: list[dict[str, Any]],
    v6_2: list[dict[str, Any]],
    prior: dict[str, Any],
    candidate_summary: dict[str, Any],
    canonical_summary: dict[str, Any],
    model_descriptor: dict[str, Any],
    profile_descriptor: dict[str, Any],
    v6_7_descriptor: dict[str, Any],
    v7_0_descriptor: dict[str, Any],
    implementation_commit: str,
) -> dict[str, Any]:
    outputs = {
        "selected_index_rows": run_dir / "v8_1_future_selected_index_rows.jsonl",
        "attempted_index_rows": run_dir / "v8_1_future_attempted_index_rows.jsonl",
        "action_rows": run_dir / "v8_1_future_five_action_rows.jsonl",
        "canonical_rows": run_dir / "v8_1_future_canonical_sbc_rows.jsonl",
        "baseline_rows": run_dir / "v8_1_future_v6_7_baseline_rows.jsonl",
        "candidate_decisions": run_dir / "v8_1_future_decisions.jsonl",
        "candidate_guard": run_dir / "v8_1_future_candidate_guard_replay.jsonl",
        "baseline_guard": run_dir / "v8_1_future_v6_7_guard_replay.jsonl",
        "candidate_runtime": run_dir
        / "v8_1_future_candidate_runtime_decisions.jsonl",
        "baseline_runtime": run_dir / "v8_1_future_v6_7_runtime_decisions.jsonl",
        "final_controller_state": run_dir
        / "v8_1_future_final_controller_state.json",
    }
    for name, rows in (
        ("selected_index_rows", selected),
        ("attempted_index_rows", attempted),
        ("action_rows", action_rows),
        ("canonical_rows", canonical_rows),
        ("baseline_rows", baseline_rows),
        ("candidate_decisions", decisions),
        ("candidate_guard", candidate_guard),
        ("baseline_guard", baseline_guard),
        ("candidate_runtime", candidate_runtime),
        ("baseline_runtime", baseline_runtime),
    ):
        _write_jsonl(outputs[name], rows)
    _write_json(outputs["final_controller_state"], final_state)
    report = {
        **report,
        "controller_band_distribution": dict(
            sorted(
                Counter(
                    row["rank_controller_decision"]["controller_band"]
                    for row in decisions
                    if row.get("rank_controller_decision")
                ).items()
            )
        ),
        "selected_action_distribution": dict(
            sorted(Counter(row["selected_action"] for row in decisions).items())
        ),
        "prior_reference_hash": prior["prior_reference_hash"],
        "prior_reference_source_row_counts": prior["source_row_counts"],
        "v6_7_candidate_summary": candidate_summary,
        "canonical_mapping_summary": canonical_summary,
    }
    report["report_id"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    report_path = run_dir / "v8_1_future_target_free_freeze_report.json"
    report_md_path = run_dir / "v8_1_future_target_free_freeze_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-target-free-freeze-manifest-v1",
        "run_id": report.get("run_id"),
        "candidate_name": v81.CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "plan": _descriptor(plan_path),
        "collector_protocol": _descriptor(protocol_path),
        "collector_index_snapshot": _descriptor(index_snapshot),
        "historical_manifest": _descriptor(historical_path),
        "prior_canary_index": _descriptor(prior_canary_index_path),
        "prior_canary_manifest": _descriptor(prior_canary_manifest_path),
        "model": model_descriptor,
        "profile": profile_descriptor,
        "v6_7_profile": v6_7_descriptor,
        "v7_0_profile": v7_0_descriptor,
        "development_batch_manifests": [
            _descriptor(Path(row["_manifest_path"])) for row in development
        ],
        "v6_2_batch_manifests": [
            _descriptor(Path(row["_manifest_path"])) for row in v6_2
        ],
        **{name: _descriptor(path) for name, path in outputs.items()},
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "target_free_freeze_passed": report["target_free_freeze_passed"],
        "target_free_blocking_reason_codes": report[
            "target_free_blocking_reason_codes"
        ],
        "future_target_access_allowed": report["future_target_access_allowed"],
        "labels_outcomes_resolution_or_pnl_opened": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_1_future_target_free_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v8.1 future target-free freeze",
            "",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- attempted markets: `{report['attempted_market_count']}`",
            "- candidate guard-accepted markets: "
            f"`{report['candidate_guard_accepted_market_count']}`",
            "- v6.7 guard-accepted markets: "
            f"`{report['v6_7_guard_accepted_market_count']}`",
            f"- controller bands: `{report['controller_band_distribution']}`",
            f"- selected actions: `{report['selected_action_distribution']}`",
            "- target-free freeze passed: "
            f"`{str(report['target_free_freeze_passed']).lower()}`",
            f"- blockers: `{report['target_free_blocking_reason_codes']}`",
            "- labels/outcomes/resolution/PnL opened: `false`",
            "- side quota: `false`",
            "- paper/live/write/wallet/capital/handoff remain blocked.",
            "",
        ]
    )
