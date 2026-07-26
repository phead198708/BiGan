"""Target-free freeze, settlement mapping, and single-use attempt-002 runner."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_attempt_002 import (
    CANDIDATE_ID,
    NO_TRADE,
    TRADE_ACTIONS,
    evaluate_attempt_002_future_rows,
    validate_attempt_002_preregistration,
)
from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.contracts import canonical_json_sha256

TARGET_FREE_PAIR_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-target-free-pair-v1"
)
TARGET_ACCESS_CLAIM_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-target-access-claim-v1"
)
OPERATOR_AUTHORIZATION_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-operator-authorization-v1"
)
PIPELINE_RESULT_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-pipeline-result-v1"
)
PIPELINE_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-pipeline-manifest-v1"
)
BASELINE_ID = "matched_frozen_v6_7"
ZERO_SHA256 = "0" * 64


class ChallengeAttempt002PipelineError(ValueError):
    """Raised when attempt-002 pipeline inputs cannot be reconciled."""


@dataclass(frozen=True, slots=True)
class Attempt002EvaluationConfig:
    """Pinned inputs for one synthetic dry-run or real future evaluation."""

    run_id: str
    output_dir: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    target_free_pairs_path: Path | str
    expected_target_free_pairs_sha256: str
    target_access_claim_path: Path | str
    expected_target_access_claim_sha256: str
    settlement_targets_path: Path | str
    expected_settlement_targets_sha256: str
    implementation_commit: str
    evaluated_at: str
    operator_authorization_path: Path | str | None = None
    expected_operator_authorization_sha256: str = ZERO_SHA256

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not _is_git_commit(self.implementation_commit):
            raise ValueError("implementation_commit must be a Git SHA-1")
        if not self.evaluated_at.endswith("Z"):
            raise ValueError("evaluated_at must be an explicit UTC timestamp")
        for field in (
            "expected_protocol_sha256",
            "expected_target_free_pairs_sha256",
            "expected_target_access_claim_sha256",
            "expected_settlement_targets_sha256",
            "expected_operator_authorization_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a SHA-256 digest")
        for field in (
            "output_dir",
            "protocol_path",
            "target_free_pairs_path",
            "target_access_claim_path",
            "settlement_targets_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))
        if self.operator_authorization_path is not None:
            object.__setattr__(
                self,
                "operator_authorization_path",
                Path(self.operator_authorization_path),
            )


def build_attempt_002_target_free_pairs(
    *,
    shared_source_rows: Sequence[Mapping[str, Any]],
    candidate_decisions: Sequence[Mapping[str, Any]],
    baseline_decisions: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind candidate and champion decisions before target access."""

    validate_attempt_002_preregistration(protocol)
    expected_count = int(
        protocol["future_window"]["exact_quality_valid_market_count"]
    )
    if not all(
        len(rows) == expected_count
        for rows in (
            shared_source_rows,
            candidate_decisions,
            baseline_decisions,
        )
    ):
        raise ChallengeAttempt002PipelineError(
            "target-free inputs must each contain exactly 120 rows"
        )
    candidate_by_market = _one_per_market(
        candidate_decisions,
        label="candidate decisions",
    )
    baseline_by_market = _one_per_market(
        baseline_decisions,
        label="baseline decisions",
    )
    boundary = int(
        protocol["future_window"][
            "strictly_later_minimum_market_start_ts_exclusive"
        ]
    )
    pairs = []
    prior_start = boundary
    seen_source_ids = set()
    for index, source in enumerate(shared_source_rows):
        market_id = str(source.get("market_id") or "")
        market_start_ts = _integer(
            source.get("market_start_ts"),
            field=f"source row {index} market_start_ts",
        )
        market_end_ts = _integer(
            source.get("market_end_ts"),
            field=f"source row {index} market_end_ts",
        )
        decision_ts = _integer(
            source.get("decision_ts"),
            field=f"source row {index} decision_ts",
        )
        source_id = str(
            source.get("shared_source_row_id")
            or source.get("source_row_id")
            or ""
        )
        if (
            not market_id
            or not _is_sha256(source_id)
            or source_id in seen_source_ids
            or market_start_ts <= prior_start
            or market_end_ts <= market_start_ts
            or decision_ts < market_start_ts
            or decision_ts > market_end_ts
            or source.get("capture_quality_valid") is not True
            or source.get("target_used_as_decision_input") is not False
        ):
            raise ChallengeAttempt002PipelineError(
                f"shared source row {index} is not future-freeze eligible"
            )
        candidate = candidate_by_market.get(market_id)
        baseline = baseline_by_market.get(market_id)
        if candidate is None or baseline is None:
            raise ChallengeAttempt002PipelineError(
                f"target-free decision missing for {market_id}"
            )
        candidate_fields = _normalize_target_free_decision(
            candidate,
            market_id=market_id,
            candidate=True,
        )
        baseline_fields = _normalize_target_free_decision(
            baseline,
            market_id=market_id,
            candidate=False,
        )
        pair = {
            "schema_version": TARGET_FREE_PAIR_SCHEMA_VERSION,
            "attempt_id": protocol["attempt_id"],
            "candidate_id": CANDIDATE_ID,
            "baseline_id": BASELINE_ID,
            "market_id": market_id,
            "market_start_ts": market_start_ts,
            "market_end_ts": market_end_ts,
            "shared_decision_ts": decision_ts,
            "shared_source_row_id": source_id,
            "candidate_action": candidate_fields["action"],
            "candidate_side": candidate_fields["side"],
            "candidate_fixed_position_size": 1.0,
            "candidate_position_size": candidate_fields[
                "allocated_position_size"
            ],
            "candidate_decision_id": candidate_fields["decision_id"],
            "baseline_action": baseline_fields["action"],
            "baseline_side": baseline_fields["side"],
            "baseline_fixed_position_size": 0.2,
            "baseline_position_size": baseline_fields[
                "allocated_position_size"
            ],
            "baseline_decision_id": baseline_fields["decision_id"],
            "candidate_decision_frozen_before_target_access": True,
            "baseline_decision_frozen_before_target_access": True,
            "target_used_as_decision_input": False,
            "outcomes_resolution_labels_or_pnl_opened": False,
            "historical_development_data_used": False,
            "safety": SAFE_FALSES,
        }
        pair["pair_id"] = canonical_json_sha256(pair)
        pairs.append(pair)
        prior_start = market_start_ts
        seen_source_ids.add(source_id)
    if set(candidate_by_market) != {
        row["market_id"] for row in pairs
    } or set(baseline_by_market) != {
        row["market_id"] for row in pairs
    }:
        raise ChallengeAttempt002PipelineError(
            "target-free decisions contain rows outside the shared window"
        )
    return pairs


def build_attempt_002_target_access_claim(
    *,
    target_free_pairs: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    target_access_started_ts: int,
    operator_authorization_sha256: str = ZERO_SHA256,
    synthetic_only: bool,
) -> dict[str, Any]:
    """Build a claim; a real claim requires separately hash-pinned permission."""

    validate_attempt_002_preregistration(protocol)
    _validate_target_free_pairs(target_free_pairs, protocol=protocol)
    if not _is_sha256(protocol_sha256):
        raise ChallengeAttempt002PipelineError(
            "protocol_sha256 must be a SHA-256 digest"
        )
    if not _is_sha256(operator_authorization_sha256):
        raise ChallengeAttempt002PipelineError(
            "operator authorization SHA-256 is invalid"
        )
    if not synthetic_only and operator_authorization_sha256 == ZERO_SHA256:
        raise ChallengeAttempt002PipelineError(
            "real target access requires explicit operator authorization"
        )
    max_market_end_ts = max(
        int(row["market_end_ts"]) for row in target_free_pairs
    )
    if target_access_started_ts <= max_market_end_ts:
        raise ChallengeAttempt002PipelineError(
            "target access must follow every market close"
        )
    claim = {
        "schema_version": TARGET_ACCESS_CLAIM_SCHEMA_VERSION,
        "attempt_id": protocol["attempt_id"],
        "protocol_sha256": protocol_sha256,
        "target_free_pair_set_sha256": canonical_json_sha256(
            list(target_free_pairs)
        ),
        "target_free_market_count": len(target_free_pairs),
        "target_access_started_ts": target_access_started_ts,
        "operator_authorization_sha256": operator_authorization_sha256,
        "synthetic_only": synthetic_only,
        "real_future_outcomes_opened": not synthetic_only,
        "attempt_and_promotion_alpha_consumed": not synthetic_only,
        "single_use": True,
        "result_selected_rerun_allowed": False,
        "historical_development_data_used": False,
        "safety": SAFE_FALSES,
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    return claim


def validate_attempt_002_operator_authorization(
    authorization: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> None:
    """Limit a future user approval to outcome-blind collection only."""

    checks = {
        "schema": authorization.get("schema_version")
        == OPERATOR_AUTHORIZATION_SCHEMA_VERSION,
        "identity": authorization.get("attempt_id")
        == protocol["attempt_id"]
        and authorization.get("protocol_sha256") == protocol_sha256,
        "scope": authorization.get("authorization_scope")
        == "outcome_blind_collection_of_exact_120_market_window_only"
        and authorization.get("exact_quality_valid_market_count") == 120,
        "approval": authorization.get("collection_authorized") is True
        and isinstance(authorization.get("authorized_at"), str)
        and authorization["authorized_at"].endswith("Z")
        and isinstance(authorization.get("authorization_source"), str)
        and bool(authorization["authorization_source"].strip()),
        "target": authorization.get(
            "target_access_before_decision_freeze_authorized"
        )
        is False
        and authorization.get("outcomes_during_collection_authorized")
        is False,
        "no_expansion": authorization.get("paper_allowed") is False
        and authorization.get("live_allowed") is False
        and authorization.get("write_allowed") is False
        and authorization.get("wallet_allowed") is False
        and authorization.get("handoff_allowed") is False
        and authorization.get("promotion_allowed") is False
        and authorization.get("capital_at_risk") is False,
    }
    _raise_failed_checks("operator authorization", checks)


def build_attempt_002_settled_comparison(
    *,
    target_free_pairs: Sequence[Mapping[str, Any]],
    settlement_targets: Sequence[Mapping[str, Any]],
    target_access_claim: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> list[dict[str, Any]]:
    """Map post-close targets onto already-frozen candidate/baseline pairs."""

    validate_attempt_002_preregistration(protocol)
    _validate_target_free_pairs(target_free_pairs, protocol=protocol)
    synthetic_only = _validate_target_access_claim(
        target_access_claim,
        target_free_pairs=target_free_pairs,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    targets: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, target in enumerate(settlement_targets):
        market_id = str(target.get("market_id") or "")
        action = str(target.get("action") or "")
        key = (market_id, action)
        if (
            not market_id
            or action not in TRADE_ACTIONS
            or key in targets
            or target.get("side") != TRADE_ACTIONS[action]
            or target.get("target_used_as_decision_input") is not False
            or target.get(
                "target_available_only_post_exit_or_official_resolution"
            )
            is not True
            or target.get("settled_after_market_close") is not True
            or target.get("cost_fields_subtracted_exactly_once") is not True
            or target.get("safety") != SAFE_FALSES
            or target.get("synthetic_only") is not synthetic_only
            or (
                not synthetic_only
                and target.get("official_read_only_resolution") is not True
            )
            or (
                synthetic_only
                and target.get("official_read_only_resolution") is not False
            )
        ):
            raise ChallengeAttempt002PipelineError(
                f"settlement target {index} is invalid"
            )
        _finite_float(
            target.get("runtime_policy_after_cost_net_pnl_per_contract"),
            field=f"settlement target {index} per-contract PnL",
        )
        target_payload = {
            key: value
            for key, value in target.items()
            if key != "target_row_id"
        }
        if target.get("target_row_id") != canonical_json_sha256(
            target_payload
        ):
            raise ChallengeAttempt002PipelineError(
                f"settlement target {index} target_row_id is invalid"
            )
        targets[key] = target

    required_keys = {
        (str(pair["market_id"]), action)
        for pair in target_free_pairs
        for action in (
            str(pair["candidate_action"]),
            str(pair["baseline_action"]),
        )
        if action != NO_TRADE
    }
    if set(targets) != required_keys:
        raise ChallengeAttempt002PipelineError(
            "settlement targets do not exactly cover selected trade actions"
        )
    rows = []
    for pair in target_free_pairs:
        candidate_action = str(pair["candidate_action"])
        baseline_action = str(pair["baseline_action"])
        candidate_target = (
            targets[(str(pair["market_id"]), candidate_action)]
            if candidate_action != NO_TRADE
            else None
        )
        baseline_target = (
            targets[(str(pair["market_id"]), baseline_action)]
            if baseline_action != NO_TRADE
            else None
        )
        candidate_pnl = (
            _target_pnl(candidate_target) * 1.0
            if candidate_target is not None
            else 0.0
        )
        baseline_pnl = (
            _target_pnl(baseline_target) * 0.2
            if baseline_target is not None
            else 0.0
        )
        rows.append(
            {
                "market_id": pair["market_id"],
                "market_start_ts": pair["market_start_ts"],
                "candidate_action": candidate_action,
                "candidate_side": pair["candidate_side"],
                "candidate_after_cost_pnl": candidate_pnl,
                "candidate_fixed_position_size": 1.0,
                "candidate_position_size": pair[
                    "candidate_position_size"
                ],
                "baseline_action": baseline_action,
                "baseline_side": pair["baseline_side"],
                "baseline_after_cost_pnl": baseline_pnl,
                "baseline_fixed_position_size": 0.2,
                "baseline_position_size": pair[
                    "baseline_position_size"
                ],
                "candidate_decision_frozen_before_target_access": True,
                "baseline_decision_frozen_before_target_access": True,
                "target_used_as_decision_input": False,
                "settled_after_market_close": True,
                "same_settled_market_for_candidate_and_baseline": True,
                "source_target_free_pair_id": pair["pair_id"],
                "candidate_target_row_id": (
                    candidate_target["target_row_id"]
                    if candidate_target is not None
                    else None
                ),
                "baseline_target_row_id": (
                    baseline_target["target_row_id"]
                    if baseline_target is not None
                    else None
                ),
                "synthetic_only": synthetic_only,
                "historical_development_data_used": False,
                "safety": SAFE_FALSES,
            }
        )
    return rows


def run_attempt_002_future_evaluation(
    config: Attempt002EvaluationConfig,
) -> dict[str, Any]:
    """Write one hash-indexed evaluation without mutating source evidence."""

    protocol_path = config.protocol_path.resolve()
    pairs_path = config.target_free_pairs_path.resolve()
    claim_path = config.target_access_claim_path.resolve()
    targets_path = config.settlement_targets_path.resolve()
    for path, expected, label in (
        (
            protocol_path,
            config.expected_protocol_sha256,
            "attempt-002 protocol",
        ),
        (
            pairs_path,
            config.expected_target_free_pairs_sha256,
            "target-free pairs",
        ),
        (
            claim_path,
            config.expected_target_access_claim_sha256,
            "target-access claim",
        ),
        (
            targets_path,
            config.expected_settlement_targets_sha256,
            "settlement targets",
        ),
    ):
        _verify_sha256(path, expected, label=label)
    protocol = _load_json(protocol_path)
    pairs = _load_jsonl(pairs_path)
    claim = _load_json(claim_path)
    targets = _load_jsonl(targets_path)
    synthetic_only = claim.get("synthetic_only") is True
    authorization_path = config.operator_authorization_path
    if synthetic_only:
        if (
            authorization_path is not None
            or config.expected_operator_authorization_sha256 != ZERO_SHA256
        ):
            raise ChallengeAttempt002PipelineError(
                "synthetic evaluation must not consume operator authorization"
            )
    else:
        if (
            authorization_path is None
            or config.expected_operator_authorization_sha256 == ZERO_SHA256
        ):
            raise ChallengeAttempt002PipelineError(
                "real evaluation requires the pinned operator authorization"
            )
        resolved_authorization = authorization_path.resolve()
        _verify_sha256(
            resolved_authorization,
            config.expected_operator_authorization_sha256,
            label="operator authorization",
        )
        authorization = _load_json(resolved_authorization)
        validate_attempt_002_operator_authorization(
            authorization,
            protocol=protocol,
            protocol_sha256=config.expected_protocol_sha256,
        )
        if claim.get("operator_authorization_sha256") != (
            config.expected_operator_authorization_sha256
        ):
            raise ChallengeAttempt002PipelineError(
                "claim does not bind the supplied operator authorization"
            )
    comparison = build_attempt_002_settled_comparison(
        target_free_pairs=pairs,
        settlement_targets=targets,
        target_access_claim=claim,
        protocol=protocol,
        protocol_sha256=config.expected_protocol_sha256,
    )
    gate_result = evaluate_attempt_002_future_rows(
        comparison,
        protocol=protocol,
    )
    synthetic_only = claim["synthetic_only"] is True
    result = {
        "schema_version": PIPELINE_RESULT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "attempt_id": protocol["attempt_id"],
        "implementation_commit": config.implementation_commit,
        "evaluated_at": config.evaluated_at,
        "synthetic_only": synthetic_only,
        "real_future_evidence": not synthetic_only,
        "gate_result": gate_result,
        "all_future_success_criteria_passed": gate_result[
            "all_future_success_criteria_passed"
        ],
        "promotion_evidence_eligible": (
            not synthetic_only
            and gate_result["promotion_evidence_eligible"] is True
        ),
        "promotion_audit_required": True,
        "automatic_promotion_allowed": False,
        "historical_development_data_used": False,
        "safety": SAFE_FALSES,
    }
    output_dir = config.output_dir.resolve()
    run_dir = output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    comparison_path = run_dir / "attempt_002_future_comparison.jsonl"
    result_path = run_dir / "attempt_002_future_result.json"
    _write_jsonl(comparison_path, comparison)
    _write_json(result_path, result)
    manifest = {
        "schema_version": PIPELINE_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "attempt_id": protocol["attempt_id"],
        "implementation_commit": config.implementation_commit,
        "protocol": _descriptor(protocol_path),
        "target_free_pairs": _descriptor(pairs_path),
        "target_access_claim": _descriptor(claim_path),
        "settlement_targets": _descriptor(targets_path),
        "comparison": _descriptor(comparison_path),
        "result": _descriptor(result_path),
        "synthetic_only": synthetic_only,
        "real_future_evidence": not synthetic_only,
        "promotion_evidence_eligible": result[
            "promotion_evidence_eligible"
        ],
        "single_use": True,
        "historical_development_data_used": False,
        "safety": SAFE_FALSES,
    }
    manifest_path = run_dir / "attempt_002_future_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "comparison_path": comparison_path,
        "result_path": result_path,
        "manifest_path": manifest_path,
        "comparison_sha256": _sha256_file(comparison_path),
        "result_sha256": _sha256_file(result_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "result": result,
        "manifest": manifest,
    }


def _validate_target_free_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> None:
    expected_count = int(
        protocol["future_window"]["exact_quality_valid_market_count"]
    )
    boundary = int(
        protocol["future_window"][
            "strictly_later_minimum_market_start_ts_exclusive"
        ]
    )
    prior_start = boundary
    market_ids = set()
    pair_ids = set()
    for index, pair in enumerate(pairs):
        market_id = str(pair.get("market_id") or "")
        market_start_ts = _integer(
            pair.get("market_start_ts"),
            field=f"pair {index} market_start_ts",
        )
        candidate_action = str(pair.get("candidate_action") or "")
        baseline_action = str(pair.get("baseline_action") or "")
        candidate_selected = candidate_action != NO_TRADE
        baseline_selected = baseline_action != NO_TRADE
        payload = {
            key: value for key, value in pair.items() if key != "pair_id"
        }
        checks = {
            "schema": pair.get("schema_version")
            == TARGET_FREE_PAIR_SCHEMA_VERSION,
            "attempt": pair.get("attempt_id") == protocol["attempt_id"]
            and pair.get("candidate_id") == CANDIDATE_ID
            and pair.get("baseline_id") == BASELINE_ID,
            "identity": bool(market_id)
            and market_id not in market_ids
            and pair.get("pair_id") == canonical_json_sha256(payload)
            and pair.get("pair_id") not in pair_ids,
            "time": market_start_ts > prior_start
            and int(pair.get("market_end_ts") or 0) > market_start_ts,
            "candidate": _side(candidate_action)
            == pair.get("candidate_side")
            and _float_equal(
                pair.get("candidate_fixed_position_size"),
                1.0,
            )
            and _float_equal(
                pair.get("candidate_position_size"),
                1.0 if candidate_selected else 0.0,
            )
            and _is_sha256(pair.get("candidate_decision_id")),
            "baseline": _side(baseline_action)
            == pair.get("baseline_side")
            and _float_equal(
                pair.get("baseline_fixed_position_size"),
                0.2,
            )
            and _float_equal(
                pair.get("baseline_position_size"),
                0.2 if baseline_selected else 0.0,
            )
            and _is_sha256(pair.get("baseline_decision_id")),
            "freeze": pair.get(
                "candidate_decision_frozen_before_target_access"
            )
            is True
            and pair.get(
                "baseline_decision_frozen_before_target_access"
            )
            is True
            and pair.get("target_used_as_decision_input") is False
            and pair.get(
                "outcomes_resolution_labels_or_pnl_opened"
            )
            is False,
            "lineage": pair.get("historical_development_data_used")
            is False,
            "safety": pair.get("safety") == SAFE_FALSES,
        }
        _raise_failed_checks(f"target-free pair {index}", checks)
        market_ids.add(market_id)
        pair_ids.add(str(pair["pair_id"]))
        prior_start = market_start_ts
    if len(pairs) != expected_count:
        raise ChallengeAttempt002PipelineError(
            f"target-free pair count must be exactly {expected_count}"
        )


def _validate_target_access_claim(
    claim: Mapping[str, Any],
    *,
    target_free_pairs: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> bool:
    payload = {
        key: value for key, value in claim.items() if key != "claim_id"
    }
    synthetic_only = claim.get("synthetic_only") is True
    target_access_ts = _integer(
        claim.get("target_access_started_ts"),
        field="target access timestamp",
    )
    max_end = max(int(row["market_end_ts"]) for row in target_free_pairs)
    authorization_sha256 = str(
        claim.get("operator_authorization_sha256") or ""
    )
    checks = {
        "schema": claim.get("schema_version")
        == TARGET_ACCESS_CLAIM_SCHEMA_VERSION,
        "identity": claim.get("attempt_id") == protocol["attempt_id"]
        and claim.get("protocol_sha256") == protocol_sha256
        and claim.get("claim_id") == canonical_json_sha256(payload),
        "freeze": claim.get("target_free_pair_set_sha256")
        == canonical_json_sha256(list(target_free_pairs))
        and claim.get("target_free_market_count") == len(target_free_pairs)
        == 120,
        "time": target_access_ts > max_end,
        "authorization": _is_sha256(authorization_sha256)
        and (
            (synthetic_only and authorization_sha256 == ZERO_SHA256)
            or (
                not synthetic_only
                and authorization_sha256 != ZERO_SHA256
            )
        ),
        "consumption": claim.get(
            "attempt_and_promotion_alpha_consumed"
        )
        is (not synthetic_only)
        and claim.get("real_future_outcomes_opened") is (not synthetic_only),
        "single_use": claim.get("single_use") is True
        and claim.get("result_selected_rerun_allowed") is False,
        "lineage": claim.get("historical_development_data_used") is False,
        "safety": claim.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("target-access claim", checks)
    return synthetic_only


def _normalize_target_free_decision(
    row: Mapping[str, Any],
    *,
    market_id: str,
    candidate: bool,
) -> dict[str, Any]:
    action = str(
        row.get("selected_action")
        or row.get("executed_action")
        or ""
    )
    side = str(row.get("selected_side") or "")
    expected_fixed = 1.0 if candidate else 0.2
    fixed = row.get(
        "fixed_candidate_position_size"
        if candidate
        else "baseline_fixed_position_size"
    )
    if fixed is None:
        fixed = row.get(
            "proposed_order_size"
            if not candidate
            else "candidate_fixed_position_size"
        )
    selected = action != NO_TRADE
    allocated = expected_fixed if selected else 0.0
    decision_id = row.get("decision_id")
    decision_payload = {
        key: value for key, value in row.items() if key != "decision_id"
    }
    checks = {
        "market": row.get("market_id") == market_id,
        "candidate": (
            not candidate or row.get("candidate_id") == CANDIDATE_ID
        ),
        "action": _side(action) == side,
        "size": _float_equal(fixed, expected_fixed)
        and (
            not candidate
            or _float_equal(
                row.get("candidate_position_size"),
                allocated,
            )
        ),
        "decision": decision_id
        == canonical_json_sha256(decision_payload),
        "target": row.get("target_used_as_decision_time_input") is False
        or row.get("target_used_as_decision_input") is False,
        "outcome": (
            row.get("outcome_or_pnl_field_used_at_inference") is False
            or row.get("outcomes_resolution_labels_or_pnl_opened") is False
        ),
        "safety": row.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks(
        "candidate decision" if candidate else "baseline decision",
        checks,
    )
    return {
        "action": action,
        "side": side,
        "allocated_position_size": allocated,
        "decision_id": str(
            decision_id
        ),
    }


def _target_pnl(target: Mapping[str, Any]) -> float:
    return _finite_float(
        target.get("runtime_policy_after_cost_net_pnl_per_contract"),
        field="runtime target per-contract PnL",
    )


def _one_per_market(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    output = {}
    for row in rows:
        market_id = str(row.get("market_id") or "")
        if not market_id or market_id in output:
            raise ChallengeAttempt002PipelineError(
                f"{label} market identity is missing or duplicated"
            )
        output[market_id] = row
    return output


def _side(action: str) -> str:
    if action == NO_TRADE:
        return "NONE"
    return TRADE_ACTIONS.get(action, "")


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ChallengeAttempt002PipelineError(
            f"{field} is not an integer"
        )
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ChallengeAttempt002PipelineError(
            f"{field} is not an integer"
        ) from error
    if number <= 0 or value != number:
        raise ChallengeAttempt002PipelineError(
            f"{field} must be a positive integer"
        )
    return number


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ChallengeAttempt002PipelineError(
            f"{field} is not numeric"
        ) from error
    if not math.isfinite(number):
        raise ChallengeAttempt002PipelineError(
            f"{field} must be finite"
        )
    return number


def _float_equal(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        return False


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _raise_failed_checks(label: str, checks: Mapping[str, bool]) -> None:
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ChallengeAttempt002PipelineError(
            f"{label} invalid: {','.join(blockers)}"
        )


def _verify_sha256(path: Path, expected: str, *, label: str) -> None:
    actual = _sha256_file(path)
    if actual != expected.lower():
        raise ChallengeAttempt002PipelineError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ChallengeAttempt002PipelineError(
            f"JSON object required: {path}"
        )
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ChallengeAttempt002PipelineError(
            f"JSONL objects required: {path}"
        )
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


__all__ = [
    "Attempt002EvaluationConfig",
    "ChallengeAttempt002PipelineError",
    "PIPELINE_MANIFEST_SCHEMA_VERSION",
    "PIPELINE_RESULT_SCHEMA_VERSION",
    "OPERATOR_AUTHORIZATION_SCHEMA_VERSION",
    "TARGET_ACCESS_CLAIM_SCHEMA_VERSION",
    "TARGET_FREE_PAIR_SCHEMA_VERSION",
    "ZERO_SHA256",
    "build_attempt_002_settled_comparison",
    "build_attempt_002_target_access_claim",
    "build_attempt_002_target_free_pairs",
    "run_attempt_002_future_evaluation",
    "validate_attempt_002_operator_authorization",
]
