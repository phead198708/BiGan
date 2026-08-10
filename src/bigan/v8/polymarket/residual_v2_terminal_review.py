"""Terminal governance review for the exhausted BTC 15m residual v2 lineage."""

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
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    DEFAULT_CONFIG_DIR,
    LINEAGE_ID,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import _write_new_frozen_json
from bigan.v8.polymarket.moe_confirmatory_evaluation import _write_new_frozen_text
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.moe_terminal_diagnostic import _assert_semantically_equal
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

DEFAULT_PRIMARY_DIR = DEFAULT_CONFIG_DIR / "residual_v2_primary_slot_001_oof"
DEFAULT_CHALLENGER_DIR = DEFAULT_CONFIG_DIR / "residual_v2_challenger_slot_002_oof"
DEFAULT_REVIEW_PATH = DEFAULT_CONFIG_DIR / "residual_v2_development_terminal_review.json"
DEFAULT_REVIEW_MARKDOWN_PATH = (
    DEFAULT_CONFIG_DIR / "residual_v2_development_terminal_review.md"
)
TERMINAL_REVIEW_CREATED_AT = "2026-08-09T08:36:00Z"
SCHEMA_VERSION = "bigan-btc-15m-cost-aware-residual-terminal-review-v2"


def generate_residual_v2_terminal_review(
    *,
    primary_dir: Path | str = DEFAULT_PRIMARY_DIR,
    challenger_dir: Path | str = DEFAULT_CHALLENGER_DIR,
    review_path: Path | str = DEFAULT_REVIEW_PATH,
    markdown_path: Path | str = DEFAULT_REVIEW_MARKDOWN_PATH,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Write one immutable review after both authorized candidate slots finish."""

    root = Path(repository_root).resolve()
    primary_path = Path(primary_dir).resolve()
    challenger_path = Path(challenger_dir).resolve()
    output = Path(review_path).resolve()
    markdown = Path(markdown_path).resolve()
    if output.exists() or markdown.exists():
        raise FileExistsError("residual v2 terminal review already exists")
    sources = _source_artifacts(primary_path, challenger_path, root)
    primary = _load_json(sources["paths"]["primary_report"])
    challenger = _load_json(sources["paths"]["challenger_report"])
    review = build_residual_v2_terminal_review(
        primary=primary,
        challenger=challenger,
        source_descriptors=sources["descriptors"],
    )
    artifact = _write_new_frozen_json(output, review)
    markdown_artifact = _write_new_frozen_text(
        markdown, render_residual_v2_terminal_review(review)
    )
    return {
        "review": _descriptor(Path(artifact["path"]), root),
        "review_markdown": _descriptor(Path(markdown_artifact["path"]), root),
        "phase_1_terminal_failed": True,
        "candidate_budget_exhausted": True,
        "candidate_freeze_allowed": False,
        "safety": dict(SAFETY),
    }


def verify_residual_v2_terminal_review(
    *,
    review_path: Path | str = DEFAULT_REVIEW_PATH,
    markdown_path: Path | str = DEFAULT_REVIEW_MARKDOWN_PATH,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify source hashes and independently rebuild the terminal review."""

    root = Path(repository_root).resolve()
    output = Path(review_path).resolve()
    markdown = Path(markdown_path).resolve()
    frozen = _verified_json(output)
    descriptors = dict(frozen.get("source_artifacts") or {})
    paths = {
        name: _verify_descriptor(dict(descriptor), repository_root=root)
        for name, descriptor in descriptors.items()
    }
    expected_names = {
        "primary_report",
        "primary_manifest",
        "challenger_report",
        "challenger_manifest",
        "parent_v1_terminal_review",
        "implementation",
    }
    if set(paths) != expected_names:
        raise ValueError("residual v2 terminal source artifact set mismatch")
    rebuilt = build_residual_v2_terminal_review(
        primary=_load_json(paths["primary_report"]),
        challenger=_load_json(paths["challenger_report"]),
        source_descriptors=descriptors,
    )
    _assert_semantically_equal(rebuilt, frozen, path="residual_v2_terminal_review")
    markdown_sidecar = markdown.with_suffix(".md.sha256")
    if not markdown.is_file() or not markdown_sidecar.is_file():
        raise ValueError("residual v2 terminal Markdown unavailable")
    if markdown_sidecar.read_text(encoding="utf-8").strip() != sha256_file(markdown):
        raise ValueError("residual v2 terminal Markdown SHA mismatch")
    if render_residual_v2_terminal_review(rebuilt) != markdown.read_text(
        encoding="utf-8"
    ):
        raise ValueError("residual v2 terminal Markdown does not reproduce")
    return {
        "verification_passed": True,
        "phase_1_terminal_failed": True,
        "candidate_budget_exhausted": True,
        "candidate_freeze_allowed": False,
        "live_shadow_start_allowed": False,
        "fresh_confirmatory_collection_authorized": False,
        "review_sha256": sha256_file(output),
        "safety": dict(SAFETY),
    }


def build_residual_v2_terminal_review(
    *,
    primary: Mapping[str, Any],
    challenger: Mapping[str, Any],
    source_descriptors: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Build the terminal state only when both candidates are immutable failures."""

    blockers = []
    if not (
        primary.get("lineage_id") == LINEAGE_ID
        and primary.get("candidate_role") == "primary"
        and primary.get("all_gates_passed") is False
        and primary.get("failed_gates")
        == ["prospective_power_required_market_count_lte_2000"]
        and primary.get("candidate_freeze_allowed") is False
        and dict(primary.get("safety") or {}) == SAFETY
    ):
        blockers.append("primary_result")
    if not (
        challenger.get("lineage_id") == LINEAGE_ID
        and challenger.get("candidate_role") == "challenger"
        and challenger.get("all_gates_passed") is False
        and challenger.get("candidate_budget_exhausted") is True
        and challenger.get("additional_candidate_allowed") is False
        and challenger.get("candidate_freeze_allowed") is False
        and dict(challenger.get("safety") or {}) == SAFETY
    ):
        blockers.append("challenger_result")
    if blockers:
        raise ValueError("residual v2 terminal inputs invalid: " + ", ".join(blockers))
    return {
        "schema_version": SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "created_at": TERMINAL_REVIEW_CREATED_AT,
        "role": "outcome_aware_development_terminal_review_only",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "phase_1_terminal_failed": True,
        "terminal_reason": (
            "neither_of_two_user_authorized_preregistered_candidates_passed_every_"
            "unchanged_gate"
        ),
        "candidate_budget": {
            "maximum_total_slots": 2,
            "consumed_slots": 2,
            "remaining_slots": 0,
            "candidate_budget_exhausted": True,
            "budget_increase_allowed_in_current_lineage": False,
        },
        "candidate_budget_exhausted": True,
        "best_development_candidate_by_gate_count": "residual-v2-primary-slot-001",
        "best_candidate_failed_gates": [
            "prospective_power_required_market_count_lte_2000"
        ],
        "best_candidate_required_prospective_market_count": int(
            primary["prospective_power"]["required_market_count"]
        ),
        "best_candidate_fast_track_maximum_market_count": 2000,
        "candidate_selected": None,
        "candidate_freeze_allowed": False,
        "live_shadow_start_allowed": False,
        "fresh_confirmatory_collection_authorized": False,
        "fresh_outcomes_opened": False,
        "paper_or_live_execution_authorized": False,
        "micro_live_authorized": False,
        "phase_status": {
            "v2_lineage_authorization_complete": True,
            "v2_two_slot_development_complete": True,
            "phase_1_candidate_freeze_complete": False,
            "phase_2_live_shadow_start_allowed": False,
            "phase_3_strictly_later_collection_authorized": False,
            "phase_4_micro_live_build_or_launch_allowed": False,
        },
        "slot_results": {
            "primary": _slot_summary(primary),
            "challenger": _slot_summary(challenger),
        },
        "source_artifacts": {
            name: dict(descriptor)
            for name, descriptor in source_descriptors.items()
        },
        "immutability": {
            "parent_v1_gate_or_threshold_changed": False,
            "parent_v1_failed_report_or_manifest_changed": False,
            "v2_failed_report_or_manifest_change_allowed": False,
            "safety_permission_change_allowed": False,
        },
        "required_next_governance": (
            "new_explicit_user_authorization_required;current_v1_and_v2_gates_"
            "thresholds_failed_artifacts_and_safety_must_not_be_rewritten"
        ),
        "safety": dict(SAFETY),
    }


def render_residual_v2_terminal_review(review: Mapping[str, Any]) -> str:
    """Render the exact two-slot result and immutable stop boundary."""

    primary = review["slot_results"]["primary"]
    challenger = review["slot_results"]["challenger"]
    return "\n".join(
        [
            "# BTC 15m cost-aware residual v2 development terminal review",
            "",
            "- Phase 1 terminal failed: `True`",
            "- Candidate budget consumed: `2 / 2`",
            "- Candidate selected/frozen: `None / False`",
            "- Live shadow allowed: `False`",
            "- Fresh confirmatory collection authorized: `False`",
            "",
            "## Primary market-anchored residual",
            "",
            f"- Accepted markets: `{primary['candidate_accepted_market_count']}`",
            f"- Candidate total unit PnL: `{primary['candidate_total_unit_net_pnl']:.8f}`",
            f"- Paired delta total: `{primary['paired_delta_total_unit_net_pnl']:.8f}`",
            f"- Required prospective N: `{primary['required_prospective_market_count']}`",
            f"- Failed gates: `{', '.join(primary['failed_gates'])}`",
            "",
            "## Uncertainty challenger",
            "",
            f"- Accepted markets: `{challenger['candidate_accepted_market_count']}`",
            f"- Candidate total unit PnL: `{challenger['candidate_total_unit_net_pnl']:.8f}`",
            f"- Paired delta total: `{challenger['paired_delta_total_unit_net_pnl']:.8f}`",
            f"- Required prospective N: `{challenger['required_prospective_market_count']}`",
            f"- Failed gates: `{', '.join(challenger['failed_gates'])}`",
            "",
            "No gate, threshold, population, failed report, safety permission, or slot "
            "budget was changed. No candidate may enter live shadow or fresh collection.",
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


def _source_artifacts(
    primary_dir: Path, challenger_dir: Path, root: Path
) -> dict[str, dict[str, Any]]:
    paths = {
        "primary_report": primary_dir / "residual_v2_oof_report.json",
        "primary_manifest": primary_dir / "residual_v2_oof_manifest.json",
        "challenger_report": (
            challenger_dir / "residual_v2_uncertainty_oof_report.json"
        ),
        "challenger_manifest": (
            challenger_dir / "residual_v2_uncertainty_oof_manifest.json"
        ),
        "parent_v1_terminal_review": (
            root
            / "examples/v8/polymarket_configs/"
            "BTC-15M-cost-aware-market-residual-v1/"
            "residual_development_terminal_review.json"
        ),
        "implementation": Path(__file__).resolve(),
    }
    for name, path in paths.items():
        if name == "implementation":
            if not path.is_file():
                raise ValueError("residual v2 terminal implementation unavailable")
        else:
            _verified_json(path)
    return {
        "paths": paths,
        "descriptors": {name: _descriptor(path, root) for name, path in paths.items()},
    }


__all__ = [
    "build_residual_v2_terminal_review",
    "generate_residual_v2_terminal_review",
    "render_residual_v2_terminal_review",
    "verify_residual_v2_terminal_review",
]
