"""Terminal governance review for the exhausted BTC 15m residual v4 lineage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.cost_aware_residual import (
    _descriptor,
    _load_json,
    _verified_json,
    _verify_descriptor,
)
from bigan.v8.polymarket.cost_aware_residual_v4 import DEFAULT_CONFIG_DIR, LINEAGE_ID
from bigan.v8.polymarket.moe_collection_boundary_r2 import _write_new_frozen_json
from bigan.v8.polymarket.moe_confirmatory_evaluation import _write_new_frozen_text
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.moe_terminal_diagnostic import _assert_semantically_equal
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

DEFAULT_PRIMARY_DIR = DEFAULT_CONFIG_DIR / "residual_v4_primary_slot_001_oof"
DEFAULT_CHALLENGER_DIR = DEFAULT_CONFIG_DIR / "residual_v4_challenger_slot_002_oof"
DEFAULT_REVIEW_PATH = DEFAULT_CONFIG_DIR / "residual_v4_development_terminal_review.json"
DEFAULT_REVIEW_MARKDOWN_PATH = (
    DEFAULT_CONFIG_DIR / "residual_v4_development_terminal_review.md"
)
TERMINAL_REVIEW_CREATED_AT = "2026-08-10T03:15:13Z"
SCHEMA_VERSION = "bigan-btc-15m-cost-aware-residual-terminal-review-v4"
PRIMARY_FAILED_GATES = [
    "every_chronological_block_paired_delta_total_gte_zero",
    "prospective_power_required_market_count_lte_2000",
]
CHALLENGER_FAILED_GATES = ["prospective_power_required_market_count_lte_2000"]


def generate_residual_v4_terminal_review(
    *,
    primary_dir: Path | str = DEFAULT_PRIMARY_DIR,
    challenger_dir: Path | str = DEFAULT_CHALLENGER_DIR,
    review_path: Path | str = DEFAULT_REVIEW_PATH,
    markdown_path: Path | str = DEFAULT_REVIEW_MARKDOWN_PATH,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Write one immutable fail-closed review after both v4 slots finish."""

    root = Path(repository_root).resolve()
    output = Path(review_path).resolve()
    markdown = Path(markdown_path).resolve()
    if output.exists() or markdown.exists():
        raise FileExistsError("residual v4 terminal review already exists")
    sources = _source_artifacts(
        Path(primary_dir).resolve(), Path(challenger_dir).resolve(), root
    )
    review = build_residual_v4_terminal_review(
        primary=_load_json(sources["paths"]["primary_report"]),
        challenger=_load_json(sources["paths"]["challenger_report"]),
        source_descriptors=sources["descriptors"],
    )
    artifact = _write_new_frozen_json(output, review)
    markdown_artifact = _write_new_frozen_text(
        markdown, render_residual_v4_terminal_review(review)
    )
    return {
        "review": _descriptor(Path(artifact["path"]), root),
        "review_markdown": _descriptor(Path(markdown_artifact["path"]), root),
        "phase_1_terminal_failed": True,
        "candidate_budget_exhausted": True,
        "candidate_freeze_allowed": False,
        "safety": dict(SAFETY),
    }


def verify_residual_v4_terminal_review(
    *,
    review_path: Path | str = DEFAULT_REVIEW_PATH,
    markdown_path: Path | str = DEFAULT_REVIEW_MARKDOWN_PATH,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify hashes and independently rebuild the frozen v4 terminal review."""

    root = Path(repository_root).resolve()
    output = Path(review_path).resolve()
    markdown = Path(markdown_path).resolve()
    frozen = _verified_json(output)
    descriptors = dict(frozen.get("source_artifacts") or {})
    paths = {
        name: _verify_descriptor(dict(descriptor), repository_root=root)
        for name, descriptor in descriptors.items()
    }
    if set(paths) != _expected_source_names():
        raise ValueError("residual v4 terminal source artifact set mismatch")
    rebuilt = build_residual_v4_terminal_review(
        primary=_load_json(paths["primary_report"]),
        challenger=_load_json(paths["challenger_report"]),
        source_descriptors=descriptors,
    )
    _assert_semantically_equal(rebuilt, frozen, path="residual_v4_terminal_review")
    markdown_sidecar = markdown.with_suffix(".md.sha256")
    if not markdown.is_file() or not markdown_sidecar.is_file():
        raise ValueError("residual v4 terminal Markdown unavailable")
    if markdown_sidecar.read_text(encoding="utf-8").strip() != sha256_file(markdown):
        raise ValueError("residual v4 terminal Markdown SHA mismatch")
    if render_residual_v4_terminal_review(rebuilt) != markdown.read_text(
        encoding="utf-8"
    ):
        raise ValueError("residual v4 terminal Markdown does not reproduce")
    return {
        "verification_passed": True,
        "phase_1_terminal_failed": True,
        "candidate_budget_exhausted": True,
        "candidate_freeze_allowed": False,
        "live_shadow_start_allowed": False,
        "fresh_collection_authorized": False,
        "review_sha256": sha256_file(output),
        "safety": dict(SAFETY),
    }


def build_residual_v4_terminal_review(
    *,
    primary: Mapping[str, Any],
    challenger: Mapping[str, Any],
    source_descriptors: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Build terminal state only when both preregistered v4 candidates failed."""

    blockers: list[str] = []
    if not _valid_failed_result(
        primary,
        role="primary",
        failed_gates=PRIMARY_FAILED_GATES,
        budget_exhausted=False,
    ):
        blockers.append("primary_result")
    if not _valid_failed_result(
        challenger,
        role="challenger",
        failed_gates=CHALLENGER_FAILED_GATES,
        budget_exhausted=True,
    ):
        blockers.append("challenger_result")
    if set(source_descriptors) != _expected_source_names():
        blockers.append("source_artifacts")
    if blockers:
        raise ValueError("residual v4 terminal inputs invalid: " + ", ".join(blockers))
    return {
        "schema_version": SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "created_at": TERMINAL_REVIEW_CREATED_AT,
        "role": "outcome_aware_development_terminal_review_only",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "phase_1_terminal_failed": True,
        "terminal_reason": (
            "neither_of_two_user_authorized_preregistered_candidates_passed_"
            "every_unchanged_gate"
        ),
        "candidate_budget": {
            "maximum_total_slots": 2,
            "consumed_slots": 2,
            "remaining_slots": 0,
            "candidate_budget_exhausted": True,
            "budget_increase_allowed_in_current_lineage": False,
        },
        "candidate_budget_exhausted": True,
        "best_development_candidate_by_gate_count": (
            "residual-v4-challenger-slot-002"
        ),
        "best_candidate_failed_gates": list(CHALLENGER_FAILED_GATES),
        "best_candidate_required_prospective_market_count": int(
            challenger["prospective_power"]["required_market_count"]
        ),
        "best_candidate_fast_track_maximum_market_count": 2000,
        "candidate_selected": None,
        "candidate_freeze_allowed": False,
        "live_shadow_start_allowed": False,
        "fresh_collection_authorized": False,
        "fresh_outcomes_opened": False,
        "paper_or_live_execution_authorized": False,
        "wallet_or_write_authorized": False,
        "promotion_authorized": False,
        "capital_at_risk": False,
        "phase_status": {
            "v4_lineage_authorization_complete": True,
            "v4_two_slot_development_complete": True,
            "phase_1_candidate_freeze_complete": False,
            "next_stage_authorization_request_allowed": False,
            "live_shadow_start_allowed": False,
            "fresh_collection_authorized": False,
        },
        "slot_results": {
            "primary": _slot_summary(primary),
            "challenger": _slot_summary(challenger),
        },
        "source_artifacts": {
            name: dict(descriptor) for name, descriptor in source_descriptors.items()
        },
        "immutability": {
            "existing_gate_or_threshold_changed": False,
            "zero_threshold_or_N_max_2000_changed": False,
            "cost_model_baseline_or_population_changed": False,
            "v1_v2_v3_or_v4_failed_artifact_changed": False,
            "safety_permission_change_allowed": False,
        },
        "required_next_governance": (
            "new_explicit_user_authorization_required_for_any_new_lineage;_"
            "current_gates_thresholds_cost_baseline_population_failed_artifacts_"
            "and_safety_must_not_be_rewritten"
        ),
        "safety": dict(SAFETY),
    }


def render_residual_v4_terminal_review(review: Mapping[str, Any]) -> str:
    primary = review["slot_results"]["primary"]
    challenger = review["slot_results"]["challenger"]
    return "\n".join(
        [
            "# BTC 15m cost-aware residual v4 development terminal review",
            "",
            "- Phase 1 terminal failed: `True`",
            "- Candidate budget consumed: `2 / 2`",
            "- Candidate selected/frozen: `None / False`",
            "- Live shadow allowed: `False`",
            "- Fresh collection authorized: `False`",
            "",
            "## Primary prequential convex ensemble",
            "",
            *_slot_markdown(primary),
            "",
            "## Challenger nested rolling-origin soft stacker",
            "",
            *_slot_markdown(challenger),
            "",
            "The challenger passed every unchanged gate except the frozen "
            "prospective-power N_max=2000 gate. No gate, zero threshold, N_max, "
            "cost, baseline, population, failed artifact, safety permission, or "
            "slot budget was changed. No candidate may enter collection, shadow, "
            "paper/live execution, wallet signing, writes, promotion, or capital risk.",
            "",
        ]
    )


def _valid_failed_result(
    report: Mapping[str, Any],
    *,
    role: str,
    failed_gates: list[str],
    budget_exhausted: bool,
) -> bool:
    common = (
        report.get("lineage_id") == LINEAGE_ID
        and report.get("candidate_role") == role
        and report.get("all_gates_passed") is False
        and report.get("failed_gates") == failed_gates
        and report.get("candidate_freeze_allowed") is False
        and report.get("promotion_evidence_eligible") is False
        and report.get("fresh_confirmatory_collection_authorized") is False
        and report.get("live_shadow_start_allowed") is False
        and report.get("existing_gate_threshold_cost_baseline_population_changed")
        is False
        and dict(report.get("safety") or {}) == SAFETY
    )
    if role == "primary":
        return common and budget_exhausted is False
    return (
        common
        and report.get("candidate_budget_exhausted") is budget_exhausted
        and report.get("additional_candidate_allowed") is False
        and report.get("parent_v1_v2_v3_or_primary_failed_artifacts_changed") is False
    )


def _slot_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    return {
        "slot_id": report["slot_id"],
        "candidate_role": report["candidate_role"],
        "all_gates_passed": False,
        "failed_gates": list(report["failed_gates"]),
        "candidate_accepted_market_count": int(
            overall["candidate_accepted_market_count"]
        ),
        "candidate_total_unit_net_pnl": float(
            overall["candidate_total_unit_net_pnl"]
        ),
        "paired_delta_total_unit_net_pnl": float(
            overall["paired_delta_total_unit_net_pnl"]
        ),
        "candidate_bootstrap_lcb": float(
            overall["candidate_bootstrap_interval"]["lower"]
        ),
        "paired_delta_bootstrap_lcb": float(
            overall["paired_delta_bootstrap_interval"]["lower"]
        ),
        "required_prospective_market_count": int(
            report["prospective_power"]["required_market_count"]
        ),
    }


def _slot_markdown(slot: Mapping[str, Any]) -> list[str]:
    return [
        f"- Accepted markets: `{slot['candidate_accepted_market_count']}`",
        f"- Candidate total unit PnL: `{slot['candidate_total_unit_net_pnl']:.8f}`",
        f"- Paired delta total: `{slot['paired_delta_total_unit_net_pnl']:.8f}`",
        f"- Candidate 97.5% LCB: `{slot['candidate_bootstrap_lcb']:.8f}`",
        f"- Paired-delta 97.5% LCB: `{slot['paired_delta_bootstrap_lcb']:.8f}`",
        f"- Required prospective N: `{slot['required_prospective_market_count']}`",
        f"- Failed gates: `{', '.join(slot['failed_gates'])}`",
    ]


def _expected_source_names() -> set[str]:
    return {
        "primary_protocol",
        "primary_report",
        "primary_manifest",
        "challenger_protocol",
        "challenger_report",
        "challenger_manifest",
        "lineage_authorization",
        "development_data_registry",
        "parent_v3_terminal_review",
        "parent_v3_binding_audit",
        "primary_implementation",
        "challenger_implementation",
        "terminal_review_implementation",
    }


def _source_artifacts(
    primary_dir: Path, challenger_dir: Path, root: Path
) -> dict[str, dict[str, Any]]:
    paths = {
        "primary_protocol": DEFAULT_CONFIG_DIR / "residual_v4_primary_slot_001_protocol.json",
        "primary_report": primary_dir / "residual_v4_oof_report.json",
        "primary_manifest": primary_dir / "residual_v4_oof_manifest.json",
        "challenger_protocol": (
            DEFAULT_CONFIG_DIR / "residual_v4_challenger_slot_002_protocol.json"
        ),
        "challenger_report": challenger_dir / "residual_v4_stacking_oof_report.json",
        "challenger_manifest": challenger_dir / "residual_v4_stacking_oof_manifest.json",
        "lineage_authorization": DEFAULT_CONFIG_DIR / "lineage_authorization.json",
        "development_data_registry": DEFAULT_CONFIG_DIR / "development_data_registry.json",
        "parent_v3_terminal_review": (
            root
            / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v3/"
            "residual_v3_development_terminal_review.json"
        ),
        "parent_v3_binding_audit": (
            root
            / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v3/"
            "residual_v3_frozen_artifact_binding_audit.json"
        ),
        "primary_implementation": (
            root / "src/bigan/v8/polymarket/cost_aware_residual_v4.py"
        ),
        "challenger_implementation": (
            root / "src/bigan/v8/polymarket/cost_aware_residual_v4_stacking.py"
        ),
        "terminal_review_implementation": Path(__file__).resolve(),
    }
    for name, path in paths.items():
        if name.endswith("implementation"):
            if not path.is_file():
                raise ValueError(f"residual v4 {name} unavailable")
        else:
            _verified_json(path)
    return {
        "paths": paths,
        "descriptors": {name: _descriptor(path, root) for name, path in paths.items()},
    }


__all__ = [
    "build_residual_v4_terminal_review",
    "generate_residual_v4_terminal_review",
    "render_residual_v4_terminal_review",
    "verify_residual_v4_terminal_review",
]
