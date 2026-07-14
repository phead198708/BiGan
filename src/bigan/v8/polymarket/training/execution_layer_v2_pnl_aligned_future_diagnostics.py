"""Frozen report-only diagnostics for #169 future accepted-bet PnL."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.training.contracts import compact_safety_fields

SCHEMA_VERSION = "bigan-v8-execution-layer-v2-pnl-aligned-future-supplemental-diagnostics-v1"


@dataclass(frozen=True, slots=True)
class PnLAlignedFutureDiagnosticsFreezeConfig:
    run_id: str
    output_dir: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    evaluation_freeze_manifest_path: Path | str
    expected_evaluation_freeze_manifest_sha256: str
    git_commit: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name, value in (
            ("expected_protocol_sha256", self.expected_protocol_sha256),
            (
                "expected_evaluation_freeze_manifest_sha256",
                self.expected_evaluation_freeze_manifest_sha256,
            ),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if len(self.git_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.git_commit.lower()
        ):
            raise ValueError("git_commit must be a 40-character hex digest")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "protocol_path", Path(self.protocol_path))
        object.__setattr__(
            self,
            "evaluation_freeze_manifest_path",
            Path(self.evaluation_freeze_manifest_path),
        )


def validate_pnl_aligned_future_supplemental_diagnostics_protocol(
    protocol: dict[str, Any],
) -> None:
    """Reject post-freeze metric, grouping, or safety drift."""

    concentration = dict(protocol.get("market_concentration") or {})
    leave_one_out = dict(protocol.get("leave_one_market_out") or {})
    checks = {
        "schema": protocol.get("schema_version") == SCHEMA_VERSION,
        "frozen": protocol.get("frozen") is True,
        "diagnostic_only": protocol.get("diagnostic_only") is True,
        "report_only": protocol.get("report_only") is True,
        "no_primary_gate_mutation": protocol.get("primary_future_evidence_gate_mutation_allowed")
        is False,
        "dimensions": protocol.get("group_dimensions")
        == [
            "execution_guarded_side",
            "execution_guarded_action",
            "execution_guarded_action_family",
        ],
        "group_metrics": protocol.get("group_metrics")
        == [
            "accepted_bet_count",
            "settled_bet_count",
            "contract_size_sum",
            "gross_pnl_sum",
            "execution_cost_sum",
            "cost_basis_sum",
            "settled_net_pnl_sum",
            "roi",
            "win_rate",
        ],
        "concentration": (
            concentration.get("basis") == "absolute_settled_net_pnl_by_market"
            and concentration.get("metrics")
            == [
                "top_1_absolute_pnl_share",
                "top_3_absolute_pnl_share",
                "absolute_pnl_hhi",
            ]
        ),
        "leave_one_out": (
            leave_one_out.get("sampling_unit") == "market_id"
            and leave_one_out.get("metrics")
            == [
                "net_pnl_after_market_removed",
                "minimum_net_pnl_after_one_market_removed",
                "maximum_net_pnl_after_one_market_removed",
                "all_scenarios_positive",
            ]
        ),
        "ordering": protocol.get("chronological_ordering")
        == ["market_close_ts", "market_id", "simulated_order_id"],
        "outcomes_report_only": protocol.get("outcome_rows_used_for_report_only") is True,
        "outcomes_not_selection": protocol.get("outcome_rows_used_for_selection_or_tuning")
        is False,
        "safety": _safety_fields_pass(protocol),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid supplemental diagnostics protocol: " + ", ".join(failed))


def freeze_pnl_aligned_future_supplemental_diagnostics(
    config: PnLAlignedFutureDiagnosticsFreezeConfig,
) -> dict[str, Any]:
    """Pin report-only formulas and implementation before reconciliation."""

    protocol_path = config.protocol_path.resolve()
    evaluation_freeze_path = config.evaluation_freeze_manifest_path.resolve()
    if _sha256(protocol_path) != config.expected_protocol_sha256:
        raise ValueError("supplemental diagnostics protocol SHA-256 mismatch")
    if _sha256(evaluation_freeze_path) != config.expected_evaluation_freeze_manifest_sha256:
        raise ValueError("evaluation freeze manifest SHA-256 mismatch")
    protocol = _load_json(protocol_path)
    validate_pnl_aligned_future_supplemental_diagnostics_protocol(protocol)
    evaluation_freeze = _load_json(evaluation_freeze_path)
    if not (
        evaluation_freeze.get("future_outcome_targets_loaded") is False
        and evaluation_freeze.get("outcome_reconciliation_started") is False
        and _safety_fields_pass(evaluation_freeze)
    ):
        raise ValueError("evaluation freeze is not outcome-blind and fail-closed")
    collection_freeze_descriptor = _verified_descriptor(
        evaluation_freeze.get("collection_freeze_manifest"),
        name="collection_freeze_manifest",
    )
    output_dir = config.output_dir / config.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}-freeze-manifest",
        "run_id": config.run_id,
        "git_commit": config.git_commit.lower(),
        "protocol": _descriptor(protocol_path),
        "evaluation_freeze_manifest": _descriptor(evaluation_freeze_path),
        "collection_freeze_manifest": collection_freeze_descriptor,
        "report_only": True,
        "primary_future_evidence_gate_mutation_allowed": False,
        "future_outcome_targets_loaded": False,
        "outcome_reconciliation_started": False,
        "outcome_rows_used_for_selection_or_tuning": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    manifest_path = output_dir / "pnl_aligned_future_supplemental_diagnostics_freeze.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
    }


def run_pnl_aligned_future_supplemental_diagnostics(
    *,
    run_id: str,
    output_dir: Path | str,
    diagnostics_freeze_manifest_path: Path | str,
    expected_diagnostics_freeze_manifest_sha256: str,
    evaluation_manifest_path: Path | str,
    expected_evaluation_manifest_sha256: str,
) -> dict[str, Any]:
    """Generate append-only diagnostics without changing the primary gate."""

    freeze_path = Path(diagnostics_freeze_manifest_path).resolve()
    evaluation_path = Path(evaluation_manifest_path).resolve()
    if _sha256(freeze_path) != expected_diagnostics_freeze_manifest_sha256:
        raise ValueError("supplemental diagnostics freeze SHA-256 mismatch")
    if _sha256(evaluation_path) != expected_evaluation_manifest_sha256:
        raise ValueError("primary evaluation manifest SHA-256 mismatch")
    freeze = _load_json(freeze_path)
    protocol_descriptor = _verified_descriptor(freeze.get("protocol"), name="protocol")
    protocol = _load_json(Path(protocol_descriptor["path"]))
    validate_pnl_aligned_future_supplemental_diagnostics_protocol(protocol)
    if not (
        freeze.get("report_only") is True
        and freeze.get("primary_future_evidence_gate_mutation_allowed") is False
        and _safety_fields_pass(freeze)
    ):
        raise ValueError("supplemental diagnostics freeze failed closed")
    evaluation = _load_json(evaluation_path)
    if not _safety_fields_pass(evaluation):
        raise ValueError("primary evaluation safety fields failed closed")
    rows_descriptor = _verified_descriptor(
        evaluation.get("accepted_bet_pnl_rows"), name="accepted_bet_pnl_rows"
    )
    report_descriptor = _verified_descriptor(
        evaluation.get("accepted_bet_pnl_report"), name="accepted_bet_pnl_report"
    )
    rows = _load_jsonl(Path(rows_descriptor["path"]))
    primary_report = _load_json(Path(report_descriptor["path"]))
    candidate_name = str(primary_report["candidate_policy_name"])
    baseline_name = str(primary_report["baseline_policy_name"])
    _validate_policy_identity(rows, candidate_name=candidate_name, baseline_name=baseline_name)
    policy_rows = {
        candidate_name: [row for row in rows if row["policy_name"] == candidate_name],
        baseline_name: [row for row in rows if row["policy_name"] == baseline_name],
    }
    policy_diagnostics = {
        policy_name: _policy_diagnostics(values) for policy_name, values in policy_rows.items()
    }
    for policy_name, key in (
        (candidate_name, "candidate_policy_metrics"),
        (baseline_name, "baseline_policy_metrics"),
    ):
        expected_pnl = float(primary_report[key]["settled_net_pnl_sum"])
        actual_pnl = float(policy_diagnostics[policy_name]["overall"]["settled_net_pnl_sum"])
        if not math.isclose(expected_pnl, actual_pnl, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("supplemental diagnostics do not reconcile to primary PnL")
    market_delta = _market_delta(policy_rows[candidate_name], policy_rows[baseline_name])
    candidate_minus_baseline_lomo = _candidate_minus_baseline_leave_one_out(market_delta)
    report = {
        "schema_version": f"{SCHEMA_VERSION}-report",
        "run_id": run_id,
        "status": "SUPPLEMENTAL_ACCEPTED_BET_DIAGNOSTICS_COMPLETE",
        "candidate_policy_name": candidate_name,
        "baseline_policy_name": baseline_name,
        "policy_diagnostics": policy_diagnostics,
        "candidate_minus_baseline_market_pnl": market_delta,
        "candidate_minus_baseline_leave_one_market_out": candidate_minus_baseline_lomo,
        "primary_future_evidence_gate_passed": bool(primary_report["future_evidence_gate_passed"]),
        "primary_future_evidence_gate_blocking_reason_codes": list(
            primary_report["future_evidence_gate_blocking_reason_codes"]
        ),
        "supplemental_diagnostics_report_only": True,
        "supplemental_diagnostics_can_mutate_primary_gate": False,
        "outcome_rows_used_for_report_only": True,
        "outcome_rows_used_for_selection_or_tuning": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    destination = Path(output_dir) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    report_path = destination / "pnl_aligned_future_supplemental_diagnostics_report.json"
    markdown_path = destination / "pnl_aligned_future_supplemental_diagnostics_report.md"
    _write_json(report_path, report)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}-manifest",
        "run_id": run_id,
        "diagnostics_freeze_manifest": _descriptor(freeze_path),
        "primary_evaluation_manifest": _descriptor(evaluation_path),
        "primary_accepted_bet_pnl_rows": rows_descriptor,
        "primary_accepted_bet_pnl_report": report_descriptor,
        "supplemental_diagnostics_report": _descriptor(report_path),
        "supplemental_diagnostics_markdown": _descriptor(markdown_path),
        "primary_future_evidence_gate_passed": report["primary_future_evidence_gate_passed"],
        "supplemental_diagnostics_report_only": True,
        "supplemental_diagnostics_can_mutate_primary_gate": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    manifest_path = destination / "pnl_aligned_future_supplemental_diagnostics_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": destination,
        "report_path": report_path,
        "markdown_path": markdown_path,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "report": report,
    }


def _policy_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("execution_guard_order_allowed") is True]
    settled = [row for row in accepted if row.get("settlement_target_available") is True]
    return {
        "overall": _group_metrics(accepted),
        "by_side": _grouped_metrics(accepted, "execution_guarded_side"),
        "by_action": _grouped_metrics(accepted, "execution_guarded_action"),
        "by_family": _grouped_metrics(
            accepted,
            "execution_guarded_action",
            transform=_action_family,
        ),
        "market_concentration": _market_concentration(settled),
        "largest_winner_dependency": _largest_winner_dependency(settled),
        "leave_one_market_out": _leave_one_market_out(settled),
    }


def _grouped_metrics(
    rows: list[dict[str, Any]],
    field: str,
    *,
    transform: Any | None = None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(field) or "UNKNOWN")
        grouped[transform(value) if transform else value].append(row)
    return {key: _group_metrics(values) for key, values in sorted(grouped.items())}


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    settled = [row for row in rows if row.get("settlement_target_available") is True]
    cost_basis = sum(float(row["cost_basis"]) for row in settled)
    net_pnl = sum(float(row["settled_net_pnl"]) for row in settled)
    return {
        "accepted_bet_count": len(rows),
        "settled_bet_count": len(settled),
        "contract_size_sum": sum(float(row["paper_bet_contract_size"]) for row in rows),
        "gross_pnl_sum": sum(float(row["gross_pnl"]) for row in settled),
        "execution_cost_sum": sum(float(row["execution_cost"]) for row in settled),
        "cost_basis_sum": cost_basis,
        "settled_net_pnl_sum": net_pnl,
        "roi": net_pnl / cost_basis if cost_basis > 0.0 else 0.0,
        "win_rate": sum(float(row["settled_net_pnl"]) > 0.0 for row in settled) / len(settled)
        if settled
        else 0.0,
    }


def _market_pnl(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    for row in rows:
        values[str(row["market_id"])] += float(row["settled_net_pnl"])
    return dict(sorted(values.items()))


def _market_concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    market_pnl = _market_pnl(rows)
    absolute_values = sorted((abs(value) for value in market_pnl.values()), reverse=True)
    total = sum(absolute_values)
    shares = [value / total for value in absolute_values] if total > 0.0 else []
    return {
        "basis": "absolute_settled_net_pnl_by_market",
        "market_count": len(market_pnl),
        "absolute_market_pnl_sum": total,
        "top_1_absolute_pnl_share": shares[0] if shares else 0.0,
        "top_3_absolute_pnl_share": sum(shares[:3]),
        "absolute_pnl_hhi": sum(value * value for value in shares),
        "reported": True,
    }


def _largest_winner_dependency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    market_pnl = _market_pnl(rows)
    full_pnl = sum(market_pnl.values())
    winning = [(market_id, value) for market_id, value in market_pnl.items() if value > 0.0]
    largest_market, largest_value = (
        max(winning, key=lambda item: item[1]) if winning else (None, 0.0)
    )
    without_largest = full_pnl - largest_value
    return {
        "largest_winning_market_id": largest_market,
        "largest_winning_market_net_pnl": largest_value,
        "full_net_pnl": full_pnl,
        "net_pnl_after_largest_winner_removed": without_largest,
        "positive_after_largest_winner_removed": without_largest > 0.0,
        "reported": True,
    }


def _leave_one_market_out(rows: list[dict[str, Any]]) -> dict[str, Any]:
    market_pnl = _market_pnl(rows)
    total = sum(market_pnl.values())
    scenarios = [
        {
            "excluded_market_id": market_id,
            "excluded_market_net_pnl": value,
            "net_pnl_after_market_removed": total - value,
        }
        for market_id, value in market_pnl.items()
    ]
    remaining = [float(row["net_pnl_after_market_removed"]) for row in scenarios]
    return {
        "sampling_unit": "market_id",
        "market_count": len(market_pnl),
        "full_net_pnl": total,
        "scenarios": scenarios,
        "minimum_net_pnl_after_one_market_removed": min(remaining) if remaining else total,
        "maximum_net_pnl_after_one_market_removed": max(remaining) if remaining else total,
        "all_scenarios_positive": bool(remaining) and all(value > 0.0 for value in remaining),
        "reported": True,
    }


def _market_delta(
    candidate_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidate = _market_pnl(
        [row for row in candidate_rows if row.get("settlement_target_available") is True]
    )
    baseline = _market_pnl(
        [row for row in baseline_rows if row.get("settlement_target_available") is True]
    )
    return [
        {
            "market_id": market_id,
            "candidate_net_pnl": candidate.get(market_id, 0.0),
            "baseline_net_pnl": baseline.get(market_id, 0.0),
            "candidate_minus_baseline_net_pnl": candidate.get(market_id, 0.0)
            - baseline.get(market_id, 0.0),
        }
        for market_id in sorted(set(candidate) | set(baseline))
    ]


def _candidate_minus_baseline_leave_one_out(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = sum(float(row["candidate_minus_baseline_net_pnl"]) for row in rows)
    scenarios = [
        {
            "excluded_market_id": str(row["market_id"]),
            "excluded_market_delta": float(row["candidate_minus_baseline_net_pnl"]),
            "candidate_minus_baseline_after_market_removed": total
            - float(row["candidate_minus_baseline_net_pnl"]),
        }
        for row in rows
    ]
    remaining = [float(row["candidate_minus_baseline_after_market_removed"]) for row in scenarios]
    return {
        "sampling_unit": "market_id",
        "market_count": len(rows),
        "full_candidate_minus_baseline_net_pnl": total,
        "scenarios": scenarios,
        "minimum_delta_after_one_market_removed": min(remaining) if remaining else total,
        "maximum_delta_after_one_market_removed": max(remaining) if remaining else total,
        "all_scenarios_candidate_better": bool(remaining)
        and all(value > 0.0 for value in remaining),
        "reported": True,
    }


def _validate_policy_identity(
    rows: list[dict[str, Any]], *, candidate_name: str, baseline_name: str
) -> None:
    policies = {str(row.get("policy_name") or "") for row in rows}
    if policies != {candidate_name, baseline_name}:
        raise ValueError("supplemental diagnostics policy set mismatch")
    identities = {
        policy_name: [
            str(row.get("source_row_identity") or "")
            for row in rows
            if row.get("policy_name") == policy_name
        ]
        for policy_name in policies
    }
    if any(
        not values or len(values) != len(set(values)) or any(not value for value in values)
        for values in identities.values()
    ) or set(identities[candidate_name]) != set(identities[baseline_name]):
        raise ValueError("supplemental diagnostics policy identities do not reconcile")


def _action_family(action: str) -> str:
    if action.endswith("HOLD_TO_SETTLEMENT"):
        return "HOLD_TO_SETTLEMENT"
    if action.endswith("SELL_BEFORE_CLOSE"):
        return "SELL_BEFORE_CLOSE"
    return "NO_TRADE"


def _safety_fields_pass(payload: dict[str, Any]) -> bool:
    return all(
        payload.get(key) is value
        for key, value in {
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "source_model_candidate_eligible": False,
            "freeze_ready": False,
            "promotion_evidence_eligible": False,
            "v8_execution_handoff_allowed": False,
            "#134_resume_allowed": False,
            "#146_start_allowed": False,
        }.items()
    )


def _markdown(report: dict[str, Any]) -> str:
    candidate = report["policy_diagnostics"][report["candidate_policy_name"]]
    baseline = report["policy_diagnostics"][report["baseline_policy_name"]]
    return "\n".join(
        [
            "# PnL-Aligned Future Supplemental Diagnostics",
            "",
            f"- status: `{report['status']}`",
            f"- primary gate passed: `{str(report['primary_future_evidence_gate_passed']).lower()}`",
            f"- candidate net PnL: `{candidate['overall']['settled_net_pnl_sum']}`",
            f"- baseline net PnL: `{baseline['overall']['settled_net_pnl_sum']}`",
            f"- candidate concentration: `{candidate['market_concentration']}`",
            f"- candidate leave-one-market-out: `{candidate['leave_one_market_out']}`",
            "",
            "Report-only diagnostics. The primary frozen evidence gate is unchanged.",
            "",
        ]
    )


def _verified_descriptor(value: Any, *, name: str) -> dict[str, str]:
    descriptor = dict(value or {})
    path = Path(str(descriptor.get("path") or ""))
    if not path.is_file() or descriptor.get("sha256") != _sha256(path):
        raise ValueError(f"{name} descriptor hash mismatch")
    return {"path": str(path.resolve()), "sha256": str(descriptor["sha256"])}


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())
