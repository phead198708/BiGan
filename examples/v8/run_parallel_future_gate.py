#!/usr/bin/env python3
"""Run the preregistered v8.1/v8.3/v6.7 parallel future gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.canonical_payload import canonical_payload_sha256  # noqa: E402
from bigan.v8.polymarket.parallel_future_gate import (  # noqa: E402
    REQUIRED_CANDIDATES,
    build_parallel_target_free_freeze,
    evaluate_parallel_future_gate,
    validate_parallel_future_collection_plan,
)

CONFIG = ROOT / "examples/v8/polymarket_configs"
DEFAULT_PLAN = CONFIG / "parallel_future_collection_plan.json"
DEFAULT_PROTOCOL = CONFIG / "parallel_candidate_protocol.json"
DEFAULT_CONTRACTS = {
    "v8_1_primary_no_fallback": (
        CONFIG / "parallel_candidate_v8_1_primary_no_fallback_contract.json"
    ),
    "v8_3_primary_with_fallback": (
        CONFIG / "parallel_candidate_v8_3_primary_with_fallback_contract.json"
    ),
    "matched_frozen_v6_7": (
        CONFIG / "parallel_candidate_matched_frozen_v6_7_contract.json"
    ),
}
DEFAULT_COLLECTOR_PROTOCOL = (
    CONFIG / "execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)
DEFAULT_FEATURE_CONTRACT = (
    CONFIG
    / "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)
DEFAULT_FEATURE_MISSINGNESS_CONTRACT = (
    CONFIG / "feature_missingness_contract.json"
)
DEFAULT_FEATURE_MISSINGNESS_RUNTIME_SCHEMA = (
    CONFIG / "feature_missingness_runtime.schema.json"
)
DEFAULT_PROMOTION_EVIDENCE_PROTOCOL = (
    CONFIG / "challenge_promotion_evidence_protocol.json"
)
DEFAULT_HISTORICAL_GATE_CONTRACT = (
    CONFIG / "historical_replay_superiority_contract.json"
)
DEFAULT_HISTORICAL_REPLAY_REPORT = (
    CONFIG / "historical_replay_superiority_report.json"
)
DEFAULT_FROZEN_MODEL_BINDING = (
    CONFIG / "parallel_frozen_v8_1_model_binding.json"
)
DEFAULT_PREFREEZE_CHECKLIST = (
    CONFIG / "challenge_prefreeze_checklist.json"
)
DEFAULT_EXCLUDED_CAPTURE_LEDGER = (
    CONFIG / "challenge_prefreeze_excluded_capture_ledger.json"
)


def validate_plan(
    *,
    plan_path: Path,
    protocol_path: Path,
    contract_paths: dict[str, Path],
    collector_protocol_path: Path,
    feature_contract_path: Path,
    feature_missingness_contract_path: Path = (
        DEFAULT_FEATURE_MISSINGNESS_CONTRACT
    ),
    feature_missingness_runtime_schema_path: Path = (
        DEFAULT_FEATURE_MISSINGNESS_RUNTIME_SCHEMA
    ),
    promotion_evidence_protocol_path: Path = (
        DEFAULT_PROMOTION_EVIDENCE_PROTOCOL
    ),
    frozen_model_binding_path: Path,
    prefreeze_checklist_path: Path = DEFAULT_PREFREEZE_CHECKLIST,
    excluded_capture_ledger_path: Path = DEFAULT_EXCLUDED_CAPTURE_LEDGER,
    historical_gate_contract_path: Path,
    historical_replay_report_path: Path,
    collection_started_ts: int | None = None,
) -> dict[str, Any]:
    """Load and validate every raw-file hash bound by the collection plan."""

    plan = _read_json(plan_path)
    validate_parallel_future_collection_plan(
        plan,
        protocol_sha256=_sha256(protocol_path),
        candidate_contract_sha256s={
            candidate_id: _sha256(path)
            for candidate_id, path in contract_paths.items()
        },
        collector_protocol_sha256=_sha256(collector_protocol_path),
        feature_contract_sha256=_sha256(feature_contract_path),
        feature_missingness_contract_sha256=_sha256(
            feature_missingness_contract_path
        ),
        feature_missingness_runtime_schema_sha256=_sha256(
            feature_missingness_runtime_schema_path
        ),
        promotion_evidence_protocol_sha256=_sha256(
            promotion_evidence_protocol_path
        ),
        frozen_model_binding_sha256=_sha256(frozen_model_binding_path),
        frozen_model_binding=_read_json(frozen_model_binding_path),
        candidate_contracts={
            candidate_id: _read_json(path)
            for candidate_id, path in contract_paths.items()
        },
        prefreeze_checklist_sha256=_sha256(prefreeze_checklist_path),
        prefreeze_checklist=_read_json(prefreeze_checklist_path),
        excluded_capture_ledger_sha256=_sha256(
            excluded_capture_ledger_path
        ),
        excluded_capture_ledger=_read_json(
            excluded_capture_ledger_path
        ),
        historical_gate_contract_sha256=_sha256(
            historical_gate_contract_path
        ),
        historical_replay_report_sha256=_sha256(
            historical_replay_report_path
        ),
        historical_replay_report=_read_json(
            historical_replay_report_path
        ),
        collection_started_ts=collection_started_ts,
    )
    return {
        "plan": plan,
        "plan_sha256": _sha256(plan_path),
        "protocol": _read_json(protocol_path),
        "candidate_contracts": {
            candidate_id: _read_json(path)
            for candidate_id, path in contract_paths.items()
        },
    }


def freeze_from_files(
    *,
    protocol_path: Path,
    contract_paths: dict[str, Path],
    source_rows_path: Path,
    decision_paths: dict[str, Path],
    output_dir: Path,
    decision_freeze_created_ts: int,
) -> dict[str, Any]:
    """Freeze one shared target-free row grid and all three decision streams."""

    protocol = _read_json(protocol_path)
    contracts = {
        candidate_id: _read_json(path)
        for candidate_id, path in contract_paths.items()
    }
    source_rows = _read_jsonl(source_rows_path)
    decisions = {
        candidate_id: _read_jsonl(path)
        for candidate_id, path in decision_paths.items()
    }
    freeze = build_parallel_target_free_freeze(
        protocol=protocol,
        candidate_contracts=contracts,
        source_rows=source_rows,
        decisions_by_candidate=decisions,
        decision_freeze_created_ts=decision_freeze_created_ts,
        target_access_started=False,
    )
    _ensure_empty_output_dir(output_dir)
    _write_jsonl(output_dir / "shared_target_free_source_rows.jsonl", source_rows)
    for candidate_id, rows in decisions.items():
        _write_jsonl(output_dir / f"{candidate_id}_decisions.jsonl", rows)
    _write_json(output_dir / "parallel_target_free_freeze.json", freeze)
    manifest = {
        "schema_version": "bigan-v8-parallel-target-free-freeze-manifest-v1",
        "freeze_sha256": freeze["freeze_sha256"],
        "source_rows": _descriptor(
            output_dir / "shared_target_free_source_rows.jsonl"
        ),
        "decision_streams": {
            candidate_id: _descriptor(
                output_dir / f"{candidate_id}_decisions.jsonl"
            )
            for candidate_id in REQUIRED_CANDIDATES
        },
        "outcomes_labels_settlement_returns_or_pnl_opened": False,
        "promotion_unlocked": False,
    }
    _write_json(output_dir / "parallel_target_free_freeze_manifest.json", manifest)
    return {
        "freeze": freeze,
        "manifest": manifest,
        "output_dir": str(output_dir.resolve()),
    }


def evaluate_from_files(
    *,
    protocol_path: Path,
    freeze_path: Path,
    settled_targets_path: Path,
    output_dir: Path,
    evaluation_started_ts: int,
    consumed_freeze_sha256s: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Consume a target-free freeze exactly once and persist all gate artifacts."""

    result = evaluate_parallel_future_gate(
        protocol=_read_json(protocol_path),
        freeze=_read_json(freeze_path),
        settled_targets=_read_jsonl(settled_targets_path),
        evaluation_started_ts=evaluation_started_ts,
        consumed_freeze_sha256s=consumed_freeze_sha256s,
    )
    _ensure_empty_output_dir(output_dir)
    _write_json(output_dir / "single_use_settlement_claim.json", result["claim"])
    for candidate_id, rows in result["candidate_rows"].items():
        _write_jsonl(output_dir / f"{candidate_id}_settled_rows.jsonl", rows)
    _write_json(
        output_dir / "multiplicity_aware_comparison_report.json",
        result["report"],
    )
    _write_json(
        output_dir / "parallel_future_final_manifest.json",
        result["final_manifest"],
    )
    return {
        **result,
        "output_dir": str(output_dir.resolve()),
    }


def build_legacy_v8_3_smoke_inputs(
    *,
    overlay_rows: list[dict[str, Any]],
    candidate_target_rows: list[dict[str, Any]],
    baseline_target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Adapt an already-consumed v8.3 window for non-promotional CLI smoke."""

    candidate_targets = {
        _key(row): row for row in candidate_target_rows
    }
    baseline_targets = {
        _key(row): row for row in baseline_target_rows
    }
    source_rows: list[dict[str, Any]] = []
    decisions = {candidate_id: [] for candidate_id in REQUIRED_CANDIDATES}
    settled_targets: list[dict[str, Any]] = []
    for overlay in overlay_rows:
        key = _key(overlay)
        market_id, decision_ts = key
        source_rows.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "legacy_overlay_decision_id": str(
                    overlay.get("overlay_decision_id") or ""
                ),
            }
        )
        v8_1_action = str(overlay["original_v8_1_action"])
        v8_3_action = str(overlay["selected_action"])
        baseline_action = str(overlay["original_v6_7_action"])
        v8_1_allowed = (
            overlay.get("original_v8_1_guard_allowed") is True
            and v8_1_action != "NO_TRADE"
        )
        v8_3_allowed = (
            overlay.get("execution_guard_order_allowed") is True
            and v8_3_action != "NO_TRADE"
        )
        baseline_allowed = (
            overlay.get("original_v6_7_guard_allowed") is True
            and baseline_action != "NO_TRADE"
        )
        candidate_target = candidate_targets.get(key)
        baseline_target = baseline_targets.get(key)
        decisions["v8_1_primary_no_fallback"].append(
            _decision(
                market_id=market_id,
                decision_ts=decision_ts,
                action=v8_1_action,
                side=str(overlay.get("original_v8_1_side") or "NONE"),
                origin="primary" if v8_1_allowed else "primary_abstention",
                allowed=v8_1_allowed,
                size=_size(candidate_target) if v8_1_allowed else 0.0,
                primary_abstained=not v8_1_allowed,
                fallback_used=False,
            )
        )
        decisions["v8_3_primary_with_fallback"].append(
            {
                **_decision(
                    market_id=market_id,
                    decision_ts=decision_ts,
                    action=v8_3_action,
                    side=str(overlay.get("selected_side") or "NONE"),
                    origin=str(overlay.get("selection_source") or ""),
                    allowed=v8_3_allowed,
                    size=_size(candidate_target) if v8_3_allowed else 0.0,
                    primary_abstained=(
                        overlay.get("original_v8_1_guard_allowed") is not True
                    ),
                    fallback_used=overlay.get("fallback_applied") is True,
                ),
                "v8_3_frozen_contract_reproduced": True,
            }
        )
        decisions["matched_frozen_v6_7"].append(
            {
                **_decision(
                    market_id=market_id,
                    decision_ts=decision_ts,
                    action=baseline_action,
                    side=str(overlay.get("original_v6_7_side") or "NONE"),
                    origin="matched_frozen_v6_7",
                    allowed=baseline_allowed,
                    size=_size(baseline_target) if baseline_allowed else 0.0,
                    primary_abstained=not baseline_allowed,
                    fallback_used=False,
                ),
                "matched_baseline_frozen_contract_reproduced": True,
            }
        )
        action_targets = {"NO_TRADE": 0.0}
        _add_action_target(action_targets, candidate_target)
        _add_action_target(action_targets, baseline_target)
        settled_targets.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "after_cost_pnl_per_notional_by_action": action_targets,
                "target_available_after_decision_freeze": True,
                "target_used_as_decision_input": False,
            }
        )
    return {
        "source_rows": source_rows,
        "decisions_by_candidate": decisions,
        "settled_targets": settled_targets,
    }


def run_legacy_smoke(
    *,
    protocol_path: Path,
    contract_paths: dict[str, Path],
    overlay_path: Path,
    candidate_targets_path: Path,
    baseline_targets_path: Path,
    output_dir: Path,
    legacy_decision_freeze_created_ts: int,
    evaluation_started_ts: int,
) -> dict[str, Any]:
    """Run a prior consumed window without creating promotion evidence."""

    inputs = build_legacy_v8_3_smoke_inputs(
        overlay_rows=_read_jsonl(overlay_path),
        candidate_target_rows=_read_jsonl(candidate_targets_path),
        baseline_target_rows=_read_jsonl(baseline_targets_path),
    )
    protocol = _read_json(protocol_path)
    freeze = build_parallel_target_free_freeze(
        protocol=protocol,
        candidate_contracts={
            candidate_id: _read_json(path)
            for candidate_id, path in contract_paths.items()
        },
        source_rows=inputs["source_rows"],
        decisions_by_candidate=inputs["decisions_by_candidate"],
        decision_freeze_created_ts=legacy_decision_freeze_created_ts,
        target_access_started=False,
    )
    result = evaluate_parallel_future_gate(
        protocol=protocol,
        freeze=freeze,
        settled_targets=inputs["settled_targets"],
        evaluation_started_ts=evaluation_started_ts,
        consumed_freeze_sha256s=set(),
    )
    _ensure_empty_output_dir(output_dir)
    smoke_report = {
        "schema_version": "bigan-v8-parallel-legacy-smoke-report-v1",
        "source_window_already_consumed": True,
        "fresh_attempt_alpha_consumed": False,
        "promotion_evidence_eligible": False,
        "promotion_unlocked": False,
        "input_sha256s": {
            "overlay_decisions": _sha256(overlay_path),
            "candidate_targets": _sha256(candidate_targets_path),
            "baseline_targets": _sha256(baseline_targets_path),
        },
        "parallel_report": result["report"],
    }
    smoke_report["report_sha256"] = canonical_payload_sha256(
        smoke_report,
        payload_schema_version=str(smoke_report["schema_version"]),
    )
    _write_json(output_dir / "parallel_legacy_smoke_report.json", smoke_report)
    return {
        "smoke_report": smoke_report,
        "output_dir": str(output_dir.resolve()),
    }


def _decision(
    *,
    market_id: str,
    decision_ts: int,
    action: str,
    side: str,
    origin: str,
    allowed: bool,
    size: float,
    primary_abstained: bool,
    fallback_used: bool,
) -> dict[str, Any]:
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "executed_action": action,
        "selected_side": side,
        "decision_origin": origin,
        "primary_abstained": primary_abstained,
        "fallback_used": fallback_used,
        "execution_guard_order_allowed": allowed,
        "proposed_order_size": size,
        "target_used_as_decision_input": False,
    }


def _size(target: dict[str, Any] | None) -> float:
    if target is None:
        raise ValueError("accepted decision is missing its frozen target row")
    return float(target["paper_position_size"])


def _add_action_target(
    output: dict[str, float],
    target: dict[str, Any] | None,
) -> None:
    if target is None:
        return
    action = str(target["action"])
    value = float(target["runtime_policy_after_cost_net_pnl_per_contract"])
    existing = output.get(action)
    if existing is not None and abs(existing - value) > 1e-12:
        raise ValueError("candidate and baseline target value mismatch for same action")
    output[action] = value


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["market_id"]), int(row["decision_ts"])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _ensure_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _contract_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "v8_1_primary_no_fallback": args.v8_1_contract,
        "v8_3_primary_with_fallback": args.v8_3_contract,
        "matched_frozen_v6_7": args.v6_7_contract,
    }


def _add_contract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--v8-1-contract",
        type=Path,
        default=DEFAULT_CONTRACTS["v8_1_primary_no_fallback"],
    )
    parser.add_argument(
        "--v8-3-contract",
        type=Path,
        default=DEFAULT_CONTRACTS["v8_3_primary_with_fallback"],
    )
    parser.add_argument(
        "--v6-7-contract",
        type=Path,
        default=DEFAULT_CONTRACTS["matched_frozen_v6_7"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-plan")
    validate.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    validate.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    validate.add_argument(
        "--collector-protocol",
        type=Path,
        default=DEFAULT_COLLECTOR_PROTOCOL,
    )
    validate.add_argument(
        "--feature-contract",
        type=Path,
        default=DEFAULT_FEATURE_CONTRACT,
    )
    validate.add_argument(
        "--historical-gate-contract",
        type=Path,
        default=DEFAULT_HISTORICAL_GATE_CONTRACT,
    )
    validate.add_argument(
        "--frozen-model-binding",
        type=Path,
        default=DEFAULT_FROZEN_MODEL_BINDING,
    )
    validate.add_argument(
        "--historical-replay-report",
        type=Path,
        default=DEFAULT_HISTORICAL_REPLAY_REPORT,
    )
    validate.add_argument(
        "--prefreeze-checklist",
        type=Path,
        default=DEFAULT_PREFREEZE_CHECKLIST,
    )
    validate.add_argument(
        "--excluded-capture-ledger",
        type=Path,
        default=DEFAULT_EXCLUDED_CAPTURE_LEDGER,
    )
    validate.add_argument("--collection-started-ts", type=int)
    _add_contract_args(validate)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze.add_argument("--source-rows", type=Path, required=True)
    freeze.add_argument("--v8-1-decisions", type=Path, required=True)
    freeze.add_argument("--v8-3-decisions", type=Path, required=True)
    freeze.add_argument("--v6-7-decisions", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--decision-freeze-created-ts", type=int, required=True)
    _add_contract_args(freeze)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    evaluate.add_argument("--freeze", type=Path, required=True)
    evaluate.add_argument("--settled-targets", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--evaluation-started-ts", type=int, required=True)
    evaluate.add_argument("--consumed-freeze-sha256", action="append", default=[])

    smoke = subparsers.add_parser("legacy-smoke")
    smoke.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    smoke.add_argument("--overlay-decisions", type=Path, required=True)
    smoke.add_argument("--candidate-targets", type=Path, required=True)
    smoke.add_argument("--baseline-targets", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument(
        "--legacy-decision-freeze-created-ts",
        type=int,
        required=True,
    )
    smoke.add_argument("--evaluation-started-ts", type=int, required=True)
    _add_contract_args(smoke)

    args = parser.parse_args(argv)
    if args.command == "validate-plan":
        result = validate_plan(
            plan_path=args.plan,
            protocol_path=args.protocol,
            contract_paths=_contract_paths(args),
            collector_protocol_path=args.collector_protocol,
            feature_contract_path=args.feature_contract,
            frozen_model_binding_path=args.frozen_model_binding,
            prefreeze_checklist_path=args.prefreeze_checklist,
            excluded_capture_ledger_path=args.excluded_capture_ledger,
            historical_gate_contract_path=args.historical_gate_contract,
            historical_replay_report_path=args.historical_replay_report,
            collection_started_ts=args.collection_started_ts,
        )
        output = {
            "plan_sha256": result["plan_sha256"],
            "fresh_attempt_id": result["plan"]["fresh_attempt_id"],
            "collection_plan_valid": True,
        }
    elif args.command == "freeze":
        result = freeze_from_files(
            protocol_path=args.protocol,
            contract_paths=_contract_paths(args),
            source_rows_path=args.source_rows,
            decision_paths={
                "v8_1_primary_no_fallback": args.v8_1_decisions,
                "v8_3_primary_with_fallback": args.v8_3_decisions,
                "matched_frozen_v6_7": args.v6_7_decisions,
            },
            output_dir=args.output_dir,
            decision_freeze_created_ts=args.decision_freeze_created_ts,
        )
        output = {
            "freeze_sha256": result["freeze"]["freeze_sha256"],
            "shared_source_row_count": result["freeze"][
                "shared_source_row_count"
            ],
            "output_dir": result["output_dir"],
        }
    elif args.command == "evaluate":
        result = evaluate_from_files(
            protocol_path=args.protocol,
            freeze_path=args.freeze,
            settled_targets_path=args.settled_targets,
            output_dir=args.output_dir,
            evaluation_started_ts=args.evaluation_started_ts,
            consumed_freeze_sha256s=set(args.consumed_freeze_sha256),
        )
        output = {
            "report_sha256": result["report"]["report_sha256"],
            "selected_candidate": result["report"][
                "multiplicity_aware_selected_candidate"
            ],
            "output_dir": result["output_dir"],
        }
    else:
        result = run_legacy_smoke(
            protocol_path=args.protocol,
            contract_paths=_contract_paths(args),
            overlay_path=args.overlay_decisions,
            candidate_targets_path=args.candidate_targets,
            baseline_targets_path=args.baseline_targets,
            output_dir=args.output_dir,
            legacy_decision_freeze_created_ts=(
                args.legacy_decision_freeze_created_ts
            ),
            evaluation_started_ts=args.evaluation_started_ts,
        )
        output = {
            "report_sha256": result["smoke_report"]["report_sha256"],
            "selected_candidate": result["smoke_report"]["parallel_report"][
                "multiplicity_aware_selected_candidate"
            ],
            "promotion_evidence_eligible": False,
            "output_dir": result["output_dir"],
        }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
