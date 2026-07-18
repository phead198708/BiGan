"""Reconcile post-freeze official targets and run the #204 side-only PnL gate."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    _blocked_safety_fields,
    _descriptor,
    _is_git_sha,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _write_json,
    _write_text,
    build_conformal_v5_side_only_future_pnl_gate,
    validate_conformal_v5_future_evaluation_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    ALLOWED_RAW_FEATURE_FILES,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    SCHEMA_PREFIX as PREDICTION_FREEZE_SCHEMA_PREFIX,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

SCHEMA_PREFIX = "bigan-v8-conformal-v5-strict-future-settlement"
SETTLED_CORPUS_INDEX_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-corpus-index-v1"


@dataclass(frozen=True, slots=True)
class ConformalV5FutureSettlementConfig:
    """Pinned post-freeze inputs for one-shot official outcome reconciliation."""

    run_id: str
    output_dir: Path | str
    prediction_freeze_manifest_path: Path | str
    expected_prediction_freeze_manifest_sha256: str
    settled_corpus_index_path: Path | str
    expected_settled_corpus_index_sha256: str
    builder_git_commit: str
    reconciliation_started_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field in (
            "expected_prediction_freeze_manifest_sha256",
            "expected_settled_corpus_index_sha256",
        ):
            _require_sha256(str(getattr(self, field)), name=field)
        if not _is_git_sha(self.builder_git_commit):
            raise ValueError("builder_git_commit must be a Git SHA-1")
        if self.reconciliation_started_ts <= 0:
            raise ValueError("reconciliation_started_ts must be positive")
        for field in (
            "output_dir",
            "prediction_freeze_manifest_path",
            "settled_corpus_index_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def reconcile_conformal_v5_future_settlement(
    config: ConformalV5FutureSettlementConfig,
) -> dict[str, Any]:
    """Open targets only after freeze, join both policies, and evaluate by side."""

    freeze_manifest_path = config.prediction_freeze_manifest_path.resolve()
    _verify_pin(
        freeze_manifest_path,
        config.expected_prediction_freeze_manifest_sha256,
        "prediction freeze manifest",
    )
    freeze_manifest = _load_json(freeze_manifest_path)
    if (
        freeze_manifest.get("schema_version") != f"{PREDICTION_FREEZE_SCHEMA_PREFIX}-manifest-v1"
        or freeze_manifest.get("decision_freeze_written_before_target_access") is not True
        or freeze_manifest.get("future_labels_outcomes_or_pnl_opened") is not False
        or freeze_manifest.get("resolution_artifact_opened") is not False
    ):
        raise ValueError("prediction freeze is not eligible for settlement reconciliation")
    safety_mismatches = [
        field
        for field, expected in _blocked_safety_fields().items()
        if freeze_manifest.get(field) != expected
    ]
    if safety_mismatches:
        raise ValueError("prediction freeze safety mismatch: " + ", ".join(safety_mismatches))

    decision_freeze_descriptor = _verified_descriptor(
        freeze_manifest["accepted_bet_decision_freeze"], "accepted-bet decision freeze"
    )
    decision_freeze = _load_json(Path(decision_freeze_descriptor["path"]))
    if (
        decision_freeze.get("decision_freeze_written_before_target_access") is not True
        or decision_freeze.get("future_labels_outcomes_or_pnl_opened") is not False
        or decision_freeze.get("target_or_outcome_used_for_decision") is not False
    ):
        raise ValueError("accepted-bet decision freeze contract is invalid")
    action_rows_descriptor = _verified_descriptor(
        freeze_manifest["target_free_five_action_rows"], "target-free five-action rows"
    )
    action_rows = _load_jsonl(Path(action_rows_descriptor["path"]))
    max_market_close_ts = max(int(row["market_close_ts"]) for row in action_rows)
    if config.reconciliation_started_ts <= max_market_close_ts:
        raise ValueError("settlement reconciliation attempted before all markets closed")

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-target-access-marker-v1",
        "run_id": config.run_id,
        "reconciliation_started_ts": config.reconciliation_started_ts,
        "prediction_freeze_manifest": _descriptor(freeze_manifest_path),
        "accepted_bet_decision_freeze": decision_freeze_descriptor,
        "max_market_close_ts": max_market_close_ts,
        "all_markets_closed_before_target_access": True,
        "future_outcomes_opened_before_decision_freeze": False,
        "target_access_started_after_decision_freeze": True,
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        **_blocked_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / "future_settlement_target_access_started.json"
    _write_json(marker_path, marker)

    settled_index_path = config.settled_corpus_index_path.resolve()
    _verify_pin(
        settled_index_path,
        config.expected_settled_corpus_index_sha256,
        "settled corpus index",
    )
    settled_index = _load_json(settled_index_path)
    selected_descriptor = _verified_descriptor(
        freeze_manifest["selected_window_rows"], "selected window rows"
    )
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    feature_descriptor = _verified_descriptor(
        freeze_manifest["target_free_feature_rows"], "target-free feature rows"
    )
    frozen_features = _load_jsonl(Path(feature_descriptor["path"]))
    index_entries = _validate_settled_corpus_index(
        settled_index,
        expected_decision_freeze_sha256=decision_freeze_descriptor["sha256"],
        decision_freeze_created_ts=int(decision_freeze["decision_freeze_created_ts"]),
        selected_rows=selected_rows,
        reconciliation_started_ts=config.reconciliation_started_ts,
    )
    targets, source_descriptors = _load_and_validate_targets(
        index_entries,
        selected_rows=selected_rows,
        frozen_features=frozen_features,
    )
    target_path = run_dir / "conformal_v5_future_settled_five_action_targets.jsonl"
    _write_jsonl(target_path, targets)

    candidate_replay_descriptor = _verified_descriptor(
        freeze_manifest["candidate_outcome_blind_guard_replay"], "candidate guard replay"
    )
    baseline_replay_descriptor = _verified_descriptor(
        freeze_manifest["matched_baseline_outcome_blind_guard_replay"],
        "matched baseline guard replay",
    )
    candidate_replay = _load_jsonl(Path(candidate_replay_descriptor["path"]))
    baseline_replay = _load_jsonl(Path(baseline_replay_descriptor["path"]))
    targets_by_decision = {(str(row["market_id"]), int(row["decision_ts"])): row for row in targets}
    candidate_evaluation = _join_frozen_replay_targets(
        candidate_replay,
        targets_by_decision=targets_by_decision,
        policy_name="guard_compatible_conformal_net_return_v5",
        decision_freeze_sha256=decision_freeze_descriptor["sha256"],
    )
    baseline_evaluation = _join_frozen_replay_targets(
        baseline_replay,
        targets_by_decision=targets_by_decision,
        policy_name="guard_compatible_direct_net_return_v4",
        decision_freeze_sha256=decision_freeze_descriptor["sha256"],
    )
    candidate_evaluation_path = run_dir / "conformal_v5_future_settled_evaluation_rows.jsonl"
    baseline_evaluation_path = run_dir / "matched_v4_future_settled_evaluation_rows.jsonl"
    _write_jsonl(candidate_evaluation_path, candidate_evaluation)
    _write_jsonl(baseline_evaluation_path, baseline_evaluation)

    profile_descriptor = _verified_descriptor(
        freeze_manifest["evaluation_profile"], "future evaluation profile"
    )
    profile = _load_json(Path(profile_descriptor["path"]))
    validate_conformal_v5_future_evaluation_profile(profile)
    evaluation_market_ids = [str(row["market_id"]) for row in selected_rows]
    gate = build_conformal_v5_side_only_future_pnl_gate(
        candidate_evaluation,
        matched_baseline_evaluation_rows=baseline_evaluation,
        evaluation_market_ids=evaluation_market_ids,
        profile=profile,
        decision_freeze_sha256=decision_freeze_descriptor["sha256"],
    )
    gate.update(
        {
            "schema_version": f"{SCHEMA_PREFIX}-side-only-gate-report-v1",
            "run_id": config.run_id,
            "builder_git_commit": config.builder_git_commit,
            "settled_corpus_index": _descriptor(settled_index_path),
            "settled_target_rows": _descriptor(target_path),
            "candidate_evaluation_rows": _descriptor(candidate_evaluation_path),
            "matched_baseline_evaluation_rows": _descriptor(baseline_evaluation_path),
            "target_access_marker": _descriptor(marker_path),
            "target_opened_after_decision_freeze": True,
            "future_results_used_for_tuning": False,
            "future_results_used_for_rerun": False,
            "future_results_used_for_automatic_unlock": False,
            **_blocked_safety_fields(),
        }
    )
    gate["report_id"] = canonical_json_sha256(gate)
    gate_path = run_dir / "conformal_v5_future_side_only_pnl_gate_report.json"
    _write_json(gate_path, gate)
    gate_md_path = run_dir / "conformal_v5_future_side_only_pnl_gate_report.md"
    _write_text(gate_md_path, _gate_markdown(gate))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "prediction_freeze_manifest": _descriptor(freeze_manifest_path),
        "accepted_bet_decision_freeze": decision_freeze_descriptor,
        "settled_corpus_index": _descriptor(settled_index_path),
        "source_settled_corpora": source_descriptors,
        "target_access_marker": _descriptor(marker_path),
        "settled_five_action_targets": _descriptor(target_path),
        "candidate_settled_evaluation_rows": _descriptor(candidate_evaluation_path),
        "matched_baseline_settled_evaluation_rows": _descriptor(baseline_evaluation_path),
        "side_only_pnl_gate_report": _descriptor(gate_path),
        "side_only_pnl_gate_report_markdown": _descriptor(gate_md_path),
        "future_gate_passed": gate["future_gate_passed"],
        "future_gate_blocking_reason_codes": gate["future_gate_blocking_reason_codes"],
        "target_opened_after_decision_freeze": True,
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        "future_results_used_for_automatic_unlock": False,
        **_blocked_safety_fields(),
    }
    manifest["settlement_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v5_future_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "gate": gate,
        "gate_path": gate_path,
        "gate_sha256": _sha256_file(gate_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _validate_settled_corpus_index(
    index: dict[str, Any],
    *,
    expected_decision_freeze_sha256: str,
    decision_freeze_created_ts: int,
    selected_rows: list[dict[str, Any]],
    reconciliation_started_ts: int,
) -> list[dict[str, Any]]:
    entries = list(index.get("entries") or [])
    expected_markets = {str(row["market_id"]) for row in selected_rows}
    entry_markets = {str(row.get("market_id") or "") for row in entries}
    checks = {
        "schema": index.get("schema_version") == SETTLED_CORPUS_INDEX_SCHEMA_VERSION,
        "freeze_hash": index.get("decision_freeze_sha256") == expected_decision_freeze_sha256,
        "complete_market_set": len(entries) == len(expected_markets)
        and entry_markets == expected_markets,
        "official_read_only": all(
            row.get("official_read_only_resolution") is True for row in entries
        ),
        "post_freeze": all(
            row.get("corpus_built_after_decision_freeze") is True for row in entries
        ),
        "post_close": all(row.get("settled_after_market_close") is True for row in entries),
        "index_after_decision_freeze": int(index.get("index_created_ts") or 0)
        > decision_freeze_created_ts,
        "before_reconciliation": int(index.get("index_created_ts") or 0)
        <= reconciliation_started_ts,
        "no_selection_tuning": index.get("outcomes_used_for_decision_or_selection") is False
        and index.get("outcomes_used_for_threshold_or_model_tuning") is False,
        "safety": all(
            index.get(field) == expected for field, expected in _blocked_safety_fields().items()
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("settled corpus index validation failed: " + ", ".join(blockers))
    return sorted(entries, key=lambda row: str(row["market_id"]))


def _load_and_validate_targets(
    entries: list[dict[str, Any]],
    *,
    selected_rows: list[dict[str, Any]],
    frozen_features: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_by_market = {str(row["market_id"]): row for row in selected_rows}
    frozen_feature_by_key = {
        (str(row["market_id"]), int(row["decision_ts"])): row for row in frozen_features
    }
    targets: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for entry in entries:
        market_id = str(entry["market_id"])
        corpus_descriptor = _verified_descriptor(entry["corpus_manifest"], "corpus manifest")
        feature_descriptor = _verified_descriptor(entry["feature_rows"], "settled feature rows")
        label_descriptor = _verified_descriptor(entry["label_rows"], "settled label rows")
        resolution_descriptor = _verified_descriptor(
            entry["resolution_events"], "official resolution events"
        )
        corpus = _load_json(Path(corpus_descriptor["path"]))
        normalized = dict(corpus.get("normalized_artifact_hashes") or {})
        if (
            normalized.get("feature_rows") != feature_descriptor["sha256"]
            or normalized.get("label_rows") != label_descriptor["sha256"]
            or normalized.get("resolution_events") != resolution_descriptor["sha256"]
        ):
            raise ValueError("settled corpus normalized artifact lineage mismatch")
        source_raw = dict(selected_by_market[market_id].get("raw_artifacts") or {})
        corpus_raw = dict(corpus.get("raw_artifact_hashes") or {})
        for filename in ALLOWED_RAW_FEATURE_FILES:
            if corpus_raw.get(filename) != (source_raw.get(filename) or {}).get("sha256"):
                raise ValueError(f"settled corpus changed frozen raw feature input: {filename}")
        feature_rows = _load_jsonl(Path(feature_descriptor["path"]))
        expected_feature_keys = {key for key in frozen_feature_by_key if key[0] == market_id}
        settled_feature_keys = {
            (str(feature["market_id"]), int(feature["decision_ts"])) for feature in feature_rows
        }
        if (
            len(settled_feature_keys) != len(feature_rows)
            or settled_feature_keys != expected_feature_keys
        ):
            raise ValueError("settled corpus feature decision grid differs from freeze")
        for feature in feature_rows:
            key = (str(feature["market_id"]), int(feature["decision_ts"]))
            frozen = frozen_feature_by_key.get(key)
            if frozen is None or _feature_payload(feature) != _feature_payload(frozen):
                raise ValueError("settled corpus feature rows differ from decision freeze")
        resolutions = _load_jsonl(Path(resolution_descriptor["path"]))
        if len(resolutions) != 1 or str(resolutions[0].get("market_id") or "") != market_id:
            raise ValueError("official resolution row is missing or duplicated")
        resolution = resolutions[0]
        if (
            resolution.get("resolution_status") != "normal"
            or resolution.get("resolved_outcome") not in {"UP", "DOWN"}
            or not str(resolution.get("raw_resolution_sha256") or "")
        ):
            raise ValueError("official resolution is not final")
        labels = _load_jsonl(Path(label_descriptor["path"]))
        if len(labels) != len(feature_rows) * len(REQUIRED_ACTIONS):
            raise ValueError("settled five-action target row count is incomplete")
        by_decision: dict[int, dict[str, dict[str, Any]]] = {}
        for label in labels:
            if (
                str(label.get("market_id") or "") != market_id
                or label.get("resolved_outcome") != resolution["resolved_outcome"]
                or label.get("raw_resolution_sha256") != resolution["raw_resolution_sha256"]
            ):
                raise ValueError("label and official resolution provenance mismatch")
            decision_labels = by_decision.setdefault(int(label["decision_ts"]), {})
            action = str(label["action"])
            if action in decision_labels:
                raise ValueError("settled five-action target contains duplicate action")
            decision_labels[action] = label
        for feature in feature_rows:
            decision_ts = int(feature["decision_ts"])
            action_labels = by_decision.get(decision_ts, {})
            if set(action_labels) != set(REQUIRED_ACTIONS):
                raise ValueError("settled five-action target grid is incomplete")
            values = {
                action: float(action_labels[action]["total_net_pnl_per_notional"])
                for action in REQUIRED_ACTIONS
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("settled action target is non-finite")
            target = {
                "schema_version": f"{SCHEMA_PREFIX}-five-action-target-v1",
                "market_id": market_id,
                "decision_ts": decision_ts,
                "resolved_outcome": resolution["resolved_outcome"],
                "target_net_pnl_per_notional_by_action": values,
                "raw_resolution_sha256": resolution["raw_resolution_sha256"],
                "resolution_rule_sha256": resolution["resolution_rule_sha256"],
                "target_available_only_after_market_close": True,
                "target_joined_after_decision_freeze": True,
                "target_used_as_decision_input": False,
                "future_results_used_for_tuning": False,
                **_blocked_safety_fields(),
            }
            target["target_row_sha256"] = canonical_json_sha256(target)
            targets.append(target)
        sources.append(
            {
                "market_id": market_id,
                "corpus_manifest": corpus_descriptor,
                "feature_rows": feature_descriptor,
                "label_rows": label_descriptor,
                "resolution_events": resolution_descriptor,
            }
        )
    targets.sort(key=lambda row: (int(row["decision_ts"]), str(row["market_id"])))
    target_keys = {(str(row["market_id"]), int(row["decision_ts"])) for row in targets}
    if len(targets) != len(frozen_features) or len(target_keys) != len(targets):
        raise ValueError("settled target count does not match frozen feature rows")
    return targets, sources


def _join_frozen_replay_targets(
    replay_rows: list[dict[str, Any]],
    *,
    targets_by_decision: dict[tuple[str, int], dict[str, Any]],
    policy_name: str,
    decision_freeze_sha256: str,
) -> list[dict[str, Any]]:
    output = []
    for replay in replay_rows:
        key = (str(replay["market_id"]), int(replay["decision_ts"]))
        target = targets_by_decision.get(key)
        if target is None:
            raise ValueError("frozen replay decision has no settled target")
        allowed = replay.get("execution_guard_order_allowed") is True
        action = str(replay["executed_action"])
        target_value = float(target["target_net_pnl_per_notional_by_action"][action])
        order_size = float(replay.get("proposed_order_size") or 0.0)
        net_pnl = order_size * target_value if allowed else 0.0
        row = {
            **replay,
            "policy_name": policy_name,
            "decision_freeze_sha256": decision_freeze_sha256,
            "settlement_resolved": True,
            "resolved_outcome": target["resolved_outcome"],
            "target_net_pnl_per_notional": target_value,
            "accepted_bet_net_pnl": net_pnl,
            "target_joined_after_decision_freeze": True,
            "target_used_as_decision_input": False,
            "forbidden_outcome_field_used_for_decision": False,
            "feature_causality_violation": False,
            "provenance_violation": False,
            "runtime_state_violation": False,
            "future_results_used_for_tuning": False,
            **_blocked_safety_fields(),
        }
        row["settled_evaluation_row_sha256"] = canonical_json_sha256(row)
        output.append(row)
    return output


def _feature_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": str(row["market_id"]),
        "condition_id": str(row["condition_id"]),
        "slug": str(row["slug"]),
        "market_family": str(row["market_family"]),
        "horizon_ms": int(row["horizon_ms"]),
        "decision_ts": int(row["decision_ts"]),
        "feature_cutoff_ts": int(row["feature_cutoff_ts"]),
        "max_input_ts": int(row["max_input_ts"]),
        "available_at_ts": int(row["available_at_ts"]),
        "features": row["features"],
        "feature_provenance": row["feature_provenance"],
    }


def _verify_pin(path: Path, expected: str, name: str) -> None:
    _require_sha256(expected, name=f"expected_{name.replace(' ', '_')}_sha256")
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if _sha256_file(path) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _gate_markdown(gate: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Conformal v5 strictly-later side-only PnL gate",
            "",
            f"- passed: `{str(gate['future_gate_passed']).lower()}`",
            f"- candidate PnL: `{gate['candidate_post_cost_net_pnl']:.8f}`",
            f"- matched baseline PnL: `{gate['matched_baseline_post_cost_net_pnl']:.8f}`",
            f"- candidate - baseline: `{gate['candidate_minus_matched_baseline_post_cost_net_pnl']:.8f}`",
            f"- accepted bets / markets: `{gate['guard_accepted_bet_count']} / {gate['guard_accepted_unique_market_count']}`",
            "- hard PnL aggregation: `BUY_UP / BUY_DOWN side-only`",
            "- action/family PnL: `diagnostic_only`",
            "- future results used for tuning/rerun/unlock: `false/false/false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


__all__ = [
    "ConformalV5FutureSettlementConfig",
    "SETTLED_CORPUS_INDEX_SCHEMA_VERSION",
    "reconcile_conformal_v5_future_settlement",
]
