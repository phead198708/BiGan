"""Adopt the sealed #227 exact-60 decisions under the #229 support contract."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8 import (
    build_regime_emergent_target_free_support,
    validate_regime_emergent_pnl_v6_8_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_text,
)

SCHEMA_PREFIX = "bigan-v8-regime-emergent-pnl-v6-8-sealed-decision-adoption"
PARENT_SCHEMA = (
    "bigan-v8-p-up-semantic-execution-compatibility-v6-7-window-freeze-"
    "manifest-v1"
)


@dataclass(frozen=True, slots=True)
class V68SealedDecisionAdoptionConfig:
    """Pinned inputs for the target-free #227 to #229 contract transition."""

    run_id: str
    output_dir: Path | str
    evaluation_profile_path: Path | str
    expected_evaluation_profile_sha256: str
    parent_prediction_freeze_manifest_path: Path | str
    expected_parent_prediction_freeze_manifest_sha256: str
    implementation_commit: str
    decision_adoption_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip() or self.decision_adoption_created_ts <= 0:
            raise ValueError("#229 run_id and adoption timestamp are required")
        _require_git_sha(self.implementation_commit)
        for name in (
            "expected_evaluation_profile_sha256",
            "expected_parent_prediction_freeze_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "evaluation_profile_path",
            "parent_prediction_freeze_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def adopt_sealed_v6_7_decisions_for_v6_8(
    config: V68SealedDecisionAdoptionConfig,
) -> dict[str, Any]:
    """Freeze a new support decision without rescoring or opening targets."""

    profile_path = Path(config.evaluation_profile_path).resolve()
    parent_path = Path(config.parent_prediction_freeze_manifest_path).resolve()
    _verify_pin(
        profile_path,
        config.expected_evaluation_profile_sha256,
        "#229 evaluation profile",
    )
    _verify_pin(
        parent_path,
        config.expected_parent_prediction_freeze_manifest_sha256,
        "#229 parent prediction freeze",
    )
    profile = _load_json(profile_path)
    validate_regime_emergent_pnl_v6_8_profile(profile)
    if (
        config.expected_parent_prediction_freeze_manifest_sha256
        != profile["lineage"]["parent_v6_7_prediction_freeze_manifest_sha256"]
    ):
        raise ValueError("#229 parent prediction freeze lineage mismatch")
    parent = _load_json(parent_path)
    _validate_parent_manifest(parent, profile=profile)

    parent_selected_descriptor = _verified_descriptor(
        parent["selected_window_rows"], "#229 parent selected window"
    )
    parent_attempted_descriptor = _verified_descriptor(
        parent["attempted_window_rows"], "#229 parent attempted window"
    )
    parent_decisions_descriptor = _verified_descriptor(
        parent["v6_7_selected_decisions"], "#229 parent selected decisions"
    )
    parent_decision_freeze_descriptor = _verified_descriptor(
        parent["accepted_bet_decision_freeze"], "#229 parent decision freeze"
    )
    if (
        parent_selected_descriptor["sha256"]
        != profile["lineage"]["parent_v6_7_selected_index_rows_sha256"]
        or parent_decisions_descriptor["sha256"]
        != profile["lineage"]["parent_v6_7_selected_decisions_sha256"]
        or parent_decision_freeze_descriptor["sha256"]
        != profile["lineage"]["parent_v6_7_decision_freeze_sha256"]
        or _verified_descriptor(parent["collector_index"], "#229 parent index")[
            "sha256"
        ]
        != profile["lineage"]["collector_index_sha256"]
    ):
        raise ValueError("#229 sealed decision artifact lineage mismatch")

    selected_rows = _load_jsonl(Path(parent_selected_descriptor["path"]))
    attempted_rows = _load_jsonl(Path(parent_attempted_descriptor["path"]))
    decisions = _load_jsonl(Path(parent_decisions_descriptor["path"]))
    parent_decision = _load_json(Path(parent_decision_freeze_descriptor["path"]))
    parent_support = dict(parent_decision.get("target_free_support") or {})
    parent_checks = dict(parent_support.get("checks") or {})
    parent_non_side_checks = {
        name: passed
        for name, passed in parent_checks.items()
        if name not in {"buy_up_support", "buy_down_support"}
    }
    if (
        parent_support.get("target_free_support_gate_passed") is not False
        or not set(parent_support.get("blocking_reason_codes") or []).issubset(
            {"buy_up_support_gate_failed", "buy_down_support_gate_failed"}
        )
        or not parent_non_side_checks
        or not all(parent_non_side_checks.values())
    ):
        raise ValueError("#229 parent freeze has a non-side support blocker")
    selected_market_ids = [str(row.get("market_id") or "") for row in selected_rows]
    if (
        len(selected_rows) != 60
        or len(attempted_rows) != 66
        or len(decisions) != 60
        or "" in selected_market_ids
        or len(set(selected_market_ids)) != 60
        or parent_decision.get("selected_window_market_ids") != selected_market_ids
        or config.decision_adoption_created_ts
        <= max(int(row["market_end_ts"]) for row in selected_rows)
    ):
        raise ValueError("#229 sealed exact-60 decision identity invalid")
    support = build_regime_emergent_target_free_support(
        decisions,
        exact_window_market_count=len(selected_rows),
        required_total_market_count=60,
        score_field="v6_7_base_score",
    )
    if support["target_free_support_gate_passed"] is not True:
        raise ValueError(
            "#229 regime-emergent target-free support failed: "
            + ",".join(support["blocking_reason_codes"])
        )

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    copied = {}
    source_descriptors = {
        "selected_window_rows": parent_selected_descriptor,
        "attempted_window_rows": parent_attempted_descriptor,
        "target_free_feature_rows": _verified_descriptor(
            parent["target_free_feature_rows"], "#229 parent feature rows"
        ),
        "target_free_five_action_rows": _verified_descriptor(
            parent["target_free_five_action_rows"], "#229 parent action rows"
        ),
        "v6_2_target_free_predictions": _verified_descriptor(
            parent["v6_2_target_free_predictions"], "#229 parent predictions"
        ),
        "v6_7_candidate_rows": _verified_descriptor(
            parent["v6_7_candidate_rows"], "#229 parent candidates"
        ),
        "v6_7_base_selected_rows": _verified_descriptor(
            parent["v6_7_base_selected_rows"], "#229 parent base selection"
        ),
        "v6_8_selected_decisions": parent_decisions_descriptor,
        "matched_legacy_guard_replay": _verified_descriptor(
            parent["matched_legacy_guard_replay"], "#229 parent legacy replay"
        ),
    }
    for name, descriptor in source_descriptors.items():
        suffix = Path(str(descriptor["path"])).suffix
        destination = run_dir / f"{name}{suffix}"
        shutil.copyfile(descriptor["path"], destination)
        if _sha256_file(destination) != descriptor["sha256"]:
            raise ValueError(f"#229 copied artifact hash mismatch: {name}")
        copied[name] = _descriptor(destination)

    decision = {
        "schema_version": f"{SCHEMA_PREFIX}-decision-v1",
        "run_id": config.run_id,
        "role": "fresh_calibration",
        "decision_adoption_created_ts": config.decision_adoption_created_ts,
        "selected_window_market_count": 60,
        "selected_window_market_ids": selected_market_ids,
        "attempted_index_row_count": 66,
        "attempted_sequence_start": int(attempted_rows[0]["sequence"]),
        "attempted_sequence_end": int(attempted_rows[-1]["sequence"]),
        "regime_emergent_target_free_support": support,
        "future_target_access_allowed": True,
        "all_selected_markets_closed_before_adoption": True,
        "parent_decisions_rescored": False,
        "parent_decisions_reselected": False,
        "side_quota_applied": False,
        "manual_approval_scope": "offline_v6_8_calibration_and_confirmatory_only",
        "manual_approval_does_not_bypass_execution_pnl_gate": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        "threshold_or_guard_tuning_performed": False,
        **_blocked_safety_fields(),
    }
    decision["decision_freeze_id"] = canonical_json_sha256(decision)
    decision_path = run_dir / "v6_8_accepted_bet_decision_freeze.json"
    _write_json(decision_path, decision)

    side_count = dict(sorted(Counter(row["side"] for row in decisions).items()))
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "parent_prediction_freeze_manifest": _descriptor(parent_path),
        "selected_window_market_count": 60,
        "attempted_index_row_count": 66,
        "selected_side_count_diagnostic": side_count,
        "side_count_hard_gate_enabled": False,
        "side_composition_is_regime_emergent": True,
        "regime_emergent_target_free_support_gate_passed": True,
        "regime_emergent_target_free_support_blocking_reason_codes": [],
        "future_target_access_allowed": True,
        "parent_decisions_rescored": False,
        "parent_decisions_reselected": False,
        "feature_causality_violation_count": 0,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_8_sealed_decision_adoption_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "role": "fresh_calibration",
        "implementation_commit": config.implementation_commit,
        "evaluation_profile": _descriptor(profile_path),
        "parent_prediction_freeze_manifest": _descriptor(parent_path),
        **copied,
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "future_target_access_allowed": True,
        "side_count_hard_gate_enabled": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_8_sealed_decision_adoption_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "decision_freeze_path": decision_path,
        "decision_freeze_sha256": _sha256_file(decision_path),
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _validate_parent_manifest(
    parent: dict[str, Any], *, profile: dict[str, Any]
) -> None:
    if (
        parent.get("schema_version") != PARENT_SCHEMA
        or parent.get("role") != "fresh_calibration"
        or parent.get("future_target_access_allowed") is not False
        or parent.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or parent.get("settlement_provider_called") is not False
        or parent.get("source_score_mutated") is not False
        or _verified_descriptor(
            parent["evaluation_profile"], "#229 parent evaluation profile"
        )["sha256"]
        != profile["lineage"]["v6_7_evaluation_profile_sha256"]
        or _verified_descriptor(
            parent["candidate_freeze_manifest"], "#229 parent candidate freeze"
        )["sha256"]
        != profile["lineage"]["candidate_freeze_manifest_sha256"]
    ):
        raise ValueError("#229 parent target-free freeze is invalid")
    for field, expected in _blocked_safety_fields().items():
        if parent.get(field) != expected:
            raise ValueError(f"#229 parent freeze safety mismatch: {field}")


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.8 Sealed Decision Adoption",
            "",
            f"- selected markets: `{report['selected_window_market_count']}`",
            f"- side count diagnostic: `{report['selected_side_count_diagnostic']}`",
            "- side-count hard gate: `false`",
            "- parent decisions rescored/reselected: `false / false`",
            "- labels/outcomes/resolution/PnL opened: `false`",
            "- future calibration target access allowed: `true`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


__all__ = [
    "V68SealedDecisionAdoptionConfig",
    "adopt_sealed_v6_7_decisions_for_v6_8",
]
