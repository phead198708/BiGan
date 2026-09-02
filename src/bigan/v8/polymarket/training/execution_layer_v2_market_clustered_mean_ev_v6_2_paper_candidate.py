"""Dedicated paper-candidate gate for the promoted v6.2 research candidate."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _descriptor,
    _load_json,
    _require_git_sha,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_text,
)

SCHEMA_PREFIX = "bigan-v8-market-clustered-mean-ev-v6-2-paper-candidate"
CANDIDATE_NAME = "market_clustered_mean_ev_v6_2"
ISSUE_NUMBER = 219
MANUAL_APPROVAL_SCOPE = "v6_2_bounded_local_paper_canary_only"

FROZEN_HASHES = {
    "promoted_candidate_manifest": (
        "623d4ed91b4021cca14410b6486b4c9edff8b6854fa249e821c5f1f4409846cc"
    ),
    "manual_promotion_review": (
        "4911c9b10d8a9a198d3ea2beedcaf4485082236edd33bcc6f03b0d5a88905efc"
    ),
    "paper_handoff_plan": (
        "276b769b38bb89dff3eaf8cc11c0b8ec2155f150cb8828e2dd3612e277683209"
    ),
    "source_model": "7e292852673fe2072017effc2d40fce000be81734f0c8c3d6950c02e957bcf0c",
    "calibration": "dc82ddebc51e95e46477894f2a0ba7bd8fa2f6845b22ced43402822b66b68e43",
    "side_only_gate_report": (
        "af4c33e7e272fe9a12aa37836d2cee82ea9b71e0c6b8be7d7292f6d9ab887a5b"
    ),
    "prediction_freeze_manifest": (
        "0833fed7c17b67937911cf4e5bc8b5acda6a8380f0571422592e8395b5c7bd91"
    ),
    "single_use_claim": (
        "92b5cb4a9b39ec1b5a32d6a0388e8cf3953a8d502f388bb5c42e276676d0cea4"
    ),
}


@dataclass(frozen=True, slots=True)
class MarketClusteredMeanEVV62PaperCandidateConfig:
    """Inputs for the v6.2-only bounded local paper unlock."""

    run_id: str
    output_dir: Path | str
    promoted_candidate_manifest_path: Path | str
    paper_handoff_plan_path: Path | str
    manual_approval_approved: bool
    manual_approval_id: str
    manual_approval_operator: str
    manual_approval_ts: int
    builder_git_commit: str
    manual_approval_scope: str = MANUAL_APPROVAL_SCOPE
    bounded_complete_round_count: int = 12
    maximum_paper_order_notional: float = 0.2
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.manual_approval_id.strip():
            raise ValueError("manual_approval_id is required")
        if not self.manual_approval_operator.strip():
            raise ValueError("manual_approval_operator is required")
        if self.manual_approval_ts <= 0:
            raise ValueError("manual_approval_ts must be positive")
        if self.bounded_complete_round_count <= 0:
            raise ValueError("bounded_complete_round_count must be positive")
        if self.maximum_paper_order_notional <= 0.0:
            raise ValueError("maximum_paper_order_notional must be positive")
        _require_git_sha(self.builder_git_commit)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "promoted_candidate_manifest_path",
            Path(self.promoted_candidate_manifest_path),
        )
        object.__setattr__(
            self,
            "paper_handoff_plan_path",
            Path(self.paper_handoff_plan_path),
        )


def run_market_clustered_mean_ev_v6_2_paper_candidate_gate(
    config: MarketClusteredMeanEVV62PaperCandidateConfig,
) -> dict[str, Any]:
    """Validate the promoted lineage and emit a scoped paper-only unlock."""

    _verify_pin(
        config.promoted_candidate_manifest_path,
        FROZEN_HASHES["promoted_candidate_manifest"],
        "promoted candidate manifest",
    )
    _verify_pin(
        config.paper_handoff_plan_path,
        FROZEN_HASHES["paper_handoff_plan"],
        "paper handoff plan",
    )
    promoted = _load_json(config.promoted_candidate_manifest_path)
    handoff = _load_json(config.paper_handoff_plan_path)
    review_path = Path(
        _verified_descriptor(
            promoted.get("manual_promotion_review"), "manual promotion review"
        )["path"]
    )
    review = _load_json(review_path)
    side_gate_path = Path(
        _verified_descriptor(promoted.get("side_only_gate_report"), "side-only gate")[
            "path"
        ]
    )
    side_gate = _load_json(side_gate_path)
    source_model = _verified_descriptor(promoted.get("source_model"), "source model")
    calibration = _verified_descriptor(
        promoted.get("market_clustered_mean_risk_calibration"), "calibration"
    )
    approval = _manual_approval(config)
    canary_contract = _paper_canary_contract(
        config=config,
        promoted=promoted,
        source_model=source_model,
        calibration=calibration,
    )
    report = _paper_candidate_report(
        config=config,
        promoted=promoted,
        handoff=handoff,
        review=review,
        side_gate=side_gate,
        source_model=source_model,
        calibration=calibration,
        approval=approval,
        canary_contract=canary_contract,
    )

    run_dir = _prepare_run_dir(
        config.output_dir, config.run_id, overwrite=config.overwrite_existing
    )
    report_path = run_dir / "v6_2_paper_candidate_gate_report.json"
    report_md_path = run_dir / "v6_2_paper_candidate_gate_report.md"
    contract_path = run_dir / "v6_2_paper_canary_input_contract.json"
    contract_md_path = run_dir / "v6_2_paper_canary_input_contract.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _report_markdown(report))
    _write_json(contract_path, canary_contract)
    _write_text(contract_md_path, _contract_markdown(canary_contract))

    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-unlock-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "builder_git_commit": config.builder_git_commit,
        "promoted_research_candidate_manifest": _descriptor(
            config.promoted_candidate_manifest_path
        ),
        "manual_promotion_review": _descriptor(review_path),
        "paper_handoff_plan": _descriptor(config.paper_handoff_plan_path),
        "side_only_gate_report": _descriptor(side_gate_path),
        "source_model": source_model,
        "market_clustered_mean_risk_calibration": calibration,
        "paper_candidate_gate_report": _descriptor(report_path),
        "paper_canary_input_contract": _descriptor(contract_path),
        "manual_approval_hash": report["manual_approval_hash"],
        "paper_candidate_allowed": report["paper_candidate_allowed"],
        "paper_candidate_allowed_scope": MANUAL_APPROVAL_SCOPE,
        "paper_canary_handoff_allowed": report["paper_candidate_allowed"],
        "source_model_candidate_eligible": True,
        "freeze_ready": True,
        "promotion_evidence_eligible": True,
        "v8_execution_handoff_allowed": False,
        **_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_2_paper_candidate_unlock_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "contract": canary_contract,
        "contract_path": contract_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
    }


def _paper_candidate_report(
    *,
    config: MarketClusteredMeanEVV62PaperCandidateConfig,
    promoted: dict[str, Any],
    handoff: dict[str, Any],
    review: dict[str, Any],
    side_gate: dict[str, Any],
    source_model: dict[str, str],
    calibration: dict[str, str],
    approval: dict[str, Any],
    canary_contract: dict[str, Any],
) -> dict[str, Any]:
    """Build the fail-closed gate without looking at any new outcome."""

    review_source_audit = dict(review.get("post_preregistration_source_audit") or {})
    checks = {
        "promoted_manifest_hash_pinned": _sha256(
            config.promoted_candidate_manifest_path
        )
        == FROZEN_HASHES["promoted_candidate_manifest"],
        "manual_promotion_review_hash_pinned": dict(
            promoted.get("manual_promotion_review") or {}
        ).get("sha256")
        == FROZEN_HASHES["manual_promotion_review"],
        "paper_handoff_plan_hash_pinned": _sha256(config.paper_handoff_plan_path)
        == FROZEN_HASHES["paper_handoff_plan"],
        "research_candidate_promoted": promoted.get("research_candidate_promoted")
        is True
        and promoted.get("source_model_candidate_eligible") is True
        and promoted.get("freeze_ready") is True
        and promoted.get("promotion_evidence_eligible") is True,
        "manual_promotion_review_passed": review.get("manual_promotion_review_passed")
        is True
        and review.get("manual_promotion_review_blocking_reason_codes") == [],
        "frozen_model_and_calibration_hashes_match": source_model["sha256"]
        == FROZEN_HASHES["source_model"]
        and calibration["sha256"] == FROZEN_HASHES["calibration"],
        "frozen_scoring_guard_cost_and_gate_source_unchanged": review_source_audit.get(
            "passed"
        )
        is True
        and review_source_audit.get("all_frozen_function_pins_unchanged") is True,
        "single_use_future_side_gate_passed": dict(
            promoted.get("side_only_gate_report") or {}
        ).get("sha256")
        == FROZEN_HASHES["side_only_gate_report"]
        and side_gate.get("future_gate_passed") is True
        and side_gate.get("future_gate_blocking_reason_codes") == []
        and side_gate.get("future_result_driven_rerun_allowed") is False,
        "side_only_buy_up_buy_down_evidence_positive": side_gate.get(
            "pnl_hard_gate_aggregation"
        )
        == "selected_side_buy_up_buy_down_only"
        and side_gate.get("action_and_action_family_pnl_diagnostic_only") is True
        and all(
            float(
                ((side_gate.get("accepted_side_metrics") or {}).get(side) or {}).get(
                    "accepted_bet_net_pnl_sum"
                )
                or 0.0
            )
            > 0.0
            for side in ("UP", "DOWN")
        ),
        "paper_handoff_requires_separate_gate": handoff.get(
            "paper_candidate_gate_required"
        )
        is True
        and handoff.get("paper_candidate_allowed") is False
        and handoff.get("v8_execution_handoff_allowed") is False,
        "manual_approval_explicit_and_hashable": approval["manual_approval_approved"]
        is True
        and approval["manual_approval_scope"] == MANUAL_APPROVAL_SCOPE
        and bool(approval["manual_approval_id"])
        and bool(approval["manual_approval_operator"]),
        "paper_canary_contract_frozen_and_target_free": canary_contract.get("frozen")
        is True
        and canary_contract.get("decision_time_target_access_allowed") is False
        and canary_contract.get("outcomes_used_for_model_threshold_cost_sizing_or_guard_tuning")
        is False,
        "upstream_live_write_wallet_capital_flags_blocked": all(
            payload.get("paper_candidate_allowed") is False
            and payload.get("v8_execution_handoff_allowed") is False
            and payload.get("paper_only") is True
            and payload.get("capital_at_risk") is False
            and payload.get("polymarket_write_enabled") is False
            and payload.get("wallet_signing_enabled") is False
            and payload.get("#134_resume_allowed") is False
            and payload.get("#146_start_allowed") is False
            for payload in (promoted, review, handoff)
        ),
        "legacy_o_source_score_path_not_used": canary_contract.get("scorer_lineage")
        == CANDIDATE_NAME
        and canary_contract.get("legacy_o_source_score_used") is False,
    }
    reason_map = {
        "promoted_manifest_hash_pinned": "promoted_manifest_hash_mismatch",
        "manual_promotion_review_hash_pinned": "manual_promotion_review_hash_mismatch",
        "paper_handoff_plan_hash_pinned": "paper_handoff_plan_hash_mismatch",
        "research_candidate_promoted": "research_candidate_not_promoted",
        "manual_promotion_review_passed": "manual_promotion_review_not_passed",
        "frozen_model_and_calibration_hashes_match": "frozen_model_or_calibration_hash_mismatch",
        "frozen_scoring_guard_cost_and_gate_source_unchanged": "frozen_execution_source_changed",
        "single_use_future_side_gate_passed": "single_use_future_side_gate_not_passed",
        "side_only_buy_up_buy_down_evidence_positive": "side_only_pnl_evidence_not_positive",
        "paper_handoff_requires_separate_gate": "paper_handoff_contract_invalid",
        "manual_approval_explicit_and_hashable": "manual_approval_required_before_v6_2_paper_canary",
        "paper_canary_contract_frozen_and_target_free": "paper_canary_contract_invalid",
        "upstream_live_write_wallet_capital_flags_blocked": "upstream_execution_safety_flags_invalid",
        "legacy_o_source_score_path_not_used": "legacy_o_source_score_path_selected",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    allowed = not blockers
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-gate-report-v1",
        "run_id": config.run_id,
        "issue": ISSUE_NUMBER,
        "candidate_name": CANDIDATE_NAME,
        "builder_git_commit": config.builder_git_commit,
        "paper_candidate_required_checks": checks,
        "paper_candidate_blocking_reason_codes": blockers,
        "manual_approval_payload": approval,
        "manual_approval_hash": canonical_json_sha256(approval),
        "paper_candidate_allowed": allowed,
        "paper_candidate_allowed_scope": MANUAL_APPROVAL_SCOPE,
        "paper_canary_handoff_allowed": allowed,
        "source_model_candidate_eligible": True,
        "freeze_ready": True,
        "promotion_evidence_eligible": True,
        "v8_execution_handoff_allowed": False,
        "live_handoff_allowed": False,
        "paper_pnl_is_promotion_evidence": False,
        "paper_results_used_for_tuning_allowed": False,
        **_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _paper_canary_contract(
    *,
    config: MarketClusteredMeanEVV62PaperCandidateConfig,
    promoted: dict[str, Any],
    source_model: dict[str, str],
    calibration: dict[str, str],
) -> dict[str, Any]:
    contract = {
        "schema_version": f"{SCHEMA_PREFIX}-canary-input-contract-v1",
        "candidate_name": CANDIDATE_NAME,
        "frozen": True,
        "scorer_lineage": CANDIDATE_NAME,
        "legacy_o_source_score_used": False,
        "source_model": source_model,
        "market_clustered_mean_risk_calibration": calibration,
        "promoted_candidate_manifest_sha256": _sha256(
            config.promoted_candidate_manifest_path
        ),
        "market_family": "btc_updown_5m",
        "public_data_source": "read_only_public_provider",
        "bounded_complete_round_count": config.bounded_complete_round_count,
        "maximum_paper_order_notional": config.maximum_paper_order_notional,
        "paper_order_size_source": "frozen_execution_guard_proposed_order_size",
        "paper_order_size_must_not_exceed_maximum": True,
        "full_five_action_grid_required": True,
        "execution_guard_source": "frozen_v6_2_outcome_blind_acceptance_replay",
        "execution_guard_mutation_allowed": False,
        "score_threshold_cost_or_sizing_tuning_allowed": False,
        "decision_time_target_access_allowed": False,
        "feature_max_input_ts_must_be_lte_decision_ts": True,
        "settlement_polling_mode": "asynchronous_after_decision_freeze",
        "settlement_may_block_next_round_collection": False,
        "outcomes_used_for_model_threshold_cost_sizing_or_guard_tuning": False,
        "forced_coverage_bets_allowed": False,
        "provider_fail_fast_consecutive_hard_failure_limit": 3,
        "provider_optional_http_failure_may_continue": True,
        "per_round_raw_evidence_persistence_required": True,
        "paper_intent_fill_ledger_persistence_required": True,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "live_trading_enabled": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    contract["contract_id"] = canonical_json_sha256(contract)
    return contract


def _manual_approval(
    config: MarketClusteredMeanEVV62PaperCandidateConfig,
) -> dict[str, Any]:
    return {
        "manual_approval_approved": config.manual_approval_approved,
        "manual_approval_id": config.manual_approval_id,
        "manual_approval_operator": config.manual_approval_operator,
        "manual_approval_scope": config.manual_approval_scope,
        "manual_approval_ts": config.manual_approval_ts,
        "approval_does_not_enable_live_trading": True,
    }


def _safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "live_trading_enabled": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_run_dir(output_dir: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = output_dir.resolve() / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.2 paper-candidate gate",
            "",
            f"- paper candidate allowed: `{str(report['paper_candidate_allowed']).lower()}`",
            f"- scope: `{report['paper_candidate_allowed_scope']}`",
            f"- blockers: `{', '.join(report['paper_candidate_blocking_reason_codes']) or 'none'}`",
            "- live/write/wallet/capital allowed: `false`",
            "- paper PnL promotion evidence: `false`",
            "",
        ]
    )


def _contract_markdown(contract: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.2 paper canary input contract",
            "",
            f"- candidate: `{contract['candidate_name']}`",
            f"- complete rounds: `{contract['bounded_complete_round_count']}`",
            f"- maximum paper notional: `{contract['maximum_paper_order_notional']}`",
            "- provider: `read_only_public_provider`",
            "- decision-time target access: `false`",
            "- live/write/wallet/capital allowed: `false`",
            "",
        ]
    )
