"""Deterministic terminal review for the two-slot BTC 15m residual development."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.cost_aware_residual import LINEAGE_ID, _descriptor
from bigan.v8.polymarket.moe_collection_boundary_r2 import _write_new_frozen_json
from bigan.v8.polymarket.moe_confirmatory_evaluation import _write_new_frozen_text
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

SCHEMA_VERSION = "bigan-btc-15m-cost-aware-residual-terminal-review-v1"
CONFIG_DIR = (
    REPO_ROOT
    / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v1"
)
PRIMARY_DIR = CONFIG_DIR / "residual_primary_slot_001_oof"
CHALLENGER_DIR = CONFIG_DIR / "residual_challenger_slot_002_oof"
DEFAULT_REVIEW_PATH = CONFIG_DIR / "residual_development_terminal_review.json"


def generate_residual_terminal_review(
    *,
    created_at: str,
    repository_root: Path | str = REPO_ROOT,
    output_path: Path | str = DEFAULT_REVIEW_PATH,
) -> dict[str, Any]:
    """Freeze the two failed slots without selecting or mutating either candidate."""

    root = Path(repository_root).resolve()
    output = Path(output_path).resolve()
    if not output.is_relative_to(root):
        raise ValueError("residual terminal review escaped repository")
    sources = _load_sources(root)
    review = build_residual_terminal_review(created_at=created_at, sources=sources)
    json_artifact = _write_new_frozen_json(output, review)
    markdown_artifact = _write_new_frozen_text(
        output.with_suffix(".md"), render_residual_terminal_review(review)
    )
    return {
        "review": _descriptor(Path(json_artifact["path"]), root),
        "review_markdown": _descriptor(Path(markdown_artifact["path"]), root),
        "phase_1_terminal_failed": True,
        "candidate_budget_exhausted": True,
        "safety": dict(SAFETY),
    }


def verify_residual_terminal_review(
    *,
    repository_root: Path | str = REPO_ROOT,
    review_path: Path | str = DEFAULT_REVIEW_PATH,
) -> dict[str, Any]:
    """Rebuild the review from the two immutable slot reports."""

    root = Path(repository_root).resolve()
    path = Path(review_path).resolve()
    review = _verified_json(path)
    sources = _load_sources(root)
    rebuilt = build_residual_terminal_review(
        created_at=str(review["created_at"]), sources=sources
    )
    if rebuilt != review:
        raise ValueError("residual terminal review does not reproduce")
    markdown_path = path.with_suffix(".md")
    _verify_sidecar(markdown_path)
    if render_residual_terminal_review(rebuilt) != markdown_path.read_text(
        encoding="utf-8"
    ):
        raise ValueError("residual terminal review markdown does not reproduce")
    return {
        "verification_passed": True,
        "phase_1_terminal_failed": review["phase_1_terminal_failed"],
        "candidate_budget_exhausted": review["candidate_budget_exhausted"],
        "candidate_freeze_allowed": review["candidate_freeze_allowed"],
        "live_shadow_start_allowed": review["live_shadow_start_allowed"],
        "fresh_confirmatory_collection_authorized": review[
            "fresh_confirmatory_collection_authorized"
        ],
        "review_sha256": sha256_file(path),
        "safety": dict(SAFETY),
    }


def build_residual_terminal_review(
    *, created_at: str, sources: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Build the terminal decision from exact frozen slot evidence."""

    primary = dict(sources["primary_report"])
    challenger = dict(sources["challenger_report"])
    _validate_failed_report(
        primary,
        expected_slot="residual-primary-slot-001",
        expected_failed_gates=[
            "every_chronological_block_paired_delta_total_gte_zero",
            "prospective_power_required_market_count_lte_2000",
        ],
    )
    _validate_failed_report(
        challenger,
        expected_slot="residual-challenger-slot-002",
        expected_failed_gates=[
            "every_chronological_block_candidate_total_gte_zero",
            "prospective_power_required_market_count_lte_2000",
        ],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "created_at": created_at,
        "role": "outcome_aware_development_terminal_review_only",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "phase_1_terminal_failed": True,
        "terminal_reason": (
            "neither_of_two_preregistered_candidates_passed_every_frozen_gate"
        ),
        "candidate_budget_exhausted": True,
        "candidate_budget": {
            "maximum_total_slots": 2,
            "consumed_slots": 2,
            "remaining_slots": 0,
            "candidate_budget_exhausted": True,
            "budget_increase_allowed_in_current_lineage": False,
        },
        "slot_results": {
            "primary": _slot_summary(primary),
            "challenger": _slot_summary(challenger),
        },
        "source_artifacts": {
            name: dict(descriptor)
            for name, descriptor in sources["descriptors"].items()
        },
        "phase_status": {
            "phase_0_evidence_repair_complete": True,
            "phase_1_candidate_freeze_complete": False,
            "phase_2_live_shadow_start_allowed": False,
            "phase_3_strictly_later_collection_authorized": False,
            "phase_4_micro_live_build_or_launch_allowed": False,
        },
        "candidate_selected": None,
        "candidate_freeze_allowed": False,
        "live_shadow_start_allowed": False,
        "fresh_confirmatory_collection_authorized": False,
        "fresh_outcomes_opened": False,
        "paper_or_live_execution_authorized": False,
        "micro_live_authorized": False,
        "required_next_governance": (
            "new_lineage_and_new_explicit_candidate_budget_authorization_required;"
            "current_gates_thresholds_and_failed_artifacts_must_not_be_rewritten"
        ),
        "safety": dict(SAFETY),
    }


def render_residual_terminal_review(review: Mapping[str, Any]) -> str:
    """Render the deterministic terminal review."""

    primary = review["slot_results"]["primary"]
    challenger = review["slot_results"]["challenger"]
    return "\n".join(
        [
            "# BTC 15m cost-aware residual development terminal review",
            "",
            "- Phase 1 terminal failed: `True`",
            "- Candidate budget consumed: `2 / 2`",
            "- Candidate selected/frozen: `None / False`",
            "- Live shadow allowed: `False`",
            "- Fresh confirmatory collection authorized: `False`",
            "",
            "## Primary slot",
            "",
            f"- Candidate total unit PnL: `{primary['candidate_total_unit_net_pnl']:.8f}`",
            f"- Paired delta total: `{primary['paired_delta_total_unit_net_pnl']:.8f}`",
            f"- Required prospective N: `{primary['required_prospective_market_count']}`",
            f"- Failed gates: `{', '.join(primary['failed_gates'])}`",
            "",
            "## Challenger slot",
            "",
            f"- Candidate total unit PnL: `{challenger['candidate_total_unit_net_pnl']:.8f}`",
            f"- Paired delta total: `{challenger['paired_delta_total_unit_net_pnl']:.8f}`",
            f"- Required prospective N: `{challenger['required_prospective_market_count']}`",
            f"- Failed gates: `{', '.join(challenger['failed_gates'])}`",
            "",
            "No gate, threshold, population, artifact, or slot budget was changed. "
            "A new lineage and explicit new authorization are required before any "
            "additional candidate development.",
            "",
        ]
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


def _validate_failed_report(
    report: Mapping[str, Any], *, expected_slot: str, expected_failed_gates: list[str]
) -> None:
    if not (
        report.get("lineage_id") == LINEAGE_ID
        and report.get("slot_id") == expected_slot
        and report.get("all_gates_passed") is False
        and report.get("candidate_freeze_allowed") is False
        and report.get("failed_gates") == expected_failed_gates
        and report.get("development_only_forever") is True
        and report.get("promotion_evidence_eligible") is False
        and dict(report.get("safety") or {}) == SAFETY
    ):
        raise ValueError(f"slot report is not the expected frozen failure: {expected_slot}")


def _load_sources(root: Path) -> dict[str, Any]:
    paths = {
        "primary_manifest": PRIMARY_DIR / "residual_oof_manifest.json",
        "primary_report": PRIMARY_DIR / "residual_oof_report.json",
        "challenger_manifest": CHALLENGER_DIR / "residual_oof_manifest.json",
        "challenger_report": CHALLENGER_DIR / "residual_oof_report.json",
    }
    payloads = {name: _verified_json(path) for name, path in paths.items()}
    return {
        **payloads,
        "descriptors": {name: _descriptor(path, root) for name, path in paths.items()},
    }


def _verified_json(path: Path) -> dict[str, Any]:
    _verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _verify_sidecar(path: Path) -> None:
    sidecar = (
        path.with_name(f"{path.name}.sha256")
        if path.suffix == ".md"
        else path.with_suffix(".sha256")
    )
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"frozen residual review source unavailable: {path}")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise ValueError(f"frozen residual review source SHA mismatch: {path}")


__all__ = [
    "build_residual_terminal_review",
    "generate_residual_terminal_review",
    "render_residual_terminal_review",
    "verify_residual_terminal_review",
]
