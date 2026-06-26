"""Run a deterministic v8 replay paper soak.

The soak is paper-only and replay-backed. It reuses the v8 paper harness over a
longer deterministic Phase 4 decision stream, writes a paper run summary, and
produces Phase 5 plus Phase 6 paper-mode evidence. It never connects to a broker,
exchange, live feed, or order-writing adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bigan.v8.paper import (  # noqa: E402
    PaperDegradationConfig,
    PaperHarnessConfig,
    PaperHarnessResult,
    run_paper_trading_harness,
    synthetic_phase4_decisions,
)
from bigan.v8.paper.contracts import json_ready  # noqa: E402
from bigan.v8.phase4 import AdaptiveDecision  # noqa: E402

DEFAULT_RUN_ID = "paper_soak_replay_v1"
DEFAULT_ROW_COUNT = 512
FIXED_STARTED_AT = "2026-06-22T02:00:00Z"
FIXED_CREATED_AT = "2026-06-22T02:00:00Z"


@dataclass(frozen=True, slots=True)
class PaperSoakRunResult:
    """In-memory handles and artifact paths from one paper replay soak."""

    run_id: str
    output_dir: Path
    decisions: tuple[AdaptiveDecision, ...]
    harness_result: PaperHarnessResult
    paper_run_summary: dict[str, Any]
    paper_run_summary_path: Path
    bundle_manifest: dict[str, Any]
    paper_bundle_manifest_sha256: str
    artifact_paths: dict[str, Path]


def run_paper_soak(
    output_dir: Path | str,
    *,
    run_id: str = DEFAULT_RUN_ID,
    row_count: int = DEFAULT_ROW_COUNT,
    inject_degradation: bool = False,
    overwrite_existing: bool = False,
) -> PaperSoakRunResult:
    """Run deterministic replay paper soak and write a run-scoped bundle."""

    if row_count < 24:
        raise ValueError("row_count must be at least 24 for paper soak coverage")
    if not run_id.strip():
        raise ValueError("run_id is required")

    run_dir = Path(output_dir).expanduser().resolve() / run_id
    decisions = synthetic_phase4_decisions(row_count=row_count)
    config = PaperHarnessConfig(
        run_id=run_id,
        candidate_run_id="paper-soak-candidate-001",
        model_sha256=_sha256_text("paper-soak-model"),
        policy_dataset_hash=_sha256_text("paper-soak-policy-dataset"),
        split_hash=_sha256_text("paper-soak-split"),
        upstream_training_report_sha256=_sha256_text("paper-soak-training-report"),
        upstream_validation_report_sha256=_sha256_text("paper-soak-validation-report"),
        output_dir=run_dir,
        created_at=FIXED_CREATED_AT,
        degradation=(
            PaperDegradationConfig(
                start_index=max(8, row_count // 3),
                net_return_shift=0.035,
                cost_multiplier=5.0,
                live_regime="high_volatility",
            )
            if inject_degradation
            else None
        ),
        overwrite_existing=overwrite_existing,
    )
    harness_result = run_paper_trading_harness(
        decisions=decisions,
        config=config,
    )
    artifact_paths = dict(harness_result.artifact_paths)
    summary = _paper_run_summary(
        run_id=run_id,
        decisions=decisions,
        harness_result=harness_result,
        artifact_paths=artifact_paths,
        inject_degradation=inject_degradation,
    )
    summary_path = harness_result.output_dir / "paper_run_summary.json"
    _write_json(summary_path, summary)
    artifact_paths["paper_run_summary"] = summary_path

    bundle_manifest = _bundle_manifest_with_soak_summary(
        harness_result.bundle_manifest,
        summary_path=summary_path,
    )
    _write_json(harness_result.artifact_paths["paper_bundle_manifest"], bundle_manifest)
    artifact_paths["paper_bundle_manifest"] = harness_result.artifact_paths[
        "paper_bundle_manifest"
    ]
    paper_bundle_manifest_sha256 = _sha256_file(artifact_paths["paper_bundle_manifest"])

    return PaperSoakRunResult(
        run_id=run_id,
        output_dir=harness_result.output_dir,
        decisions=decisions,
        harness_result=harness_result,
        paper_run_summary=summary,
        paper_run_summary_path=summary_path,
        bundle_manifest=bundle_manifest,
        paper_bundle_manifest_sha256=paper_bundle_manifest_sha256,
        artifact_paths=artifact_paths,
    )


def _paper_run_summary(
    *,
    run_id: str,
    decisions: tuple[AdaptiveDecision, ...],
    harness_result: PaperHarnessResult,
    artifact_paths: dict[str, Path],
    inject_degradation: bool,
) -> dict[str, Any]:
    fills = harness_result.fills
    phase5_report = harness_result.phase5_result.report
    drift_metrics = phase5_report.drift_metrics
    live_risk_metrics = phase5_report.live_risk_metrics
    safety_action = phase5_report.safety_action
    total_execution_cost = sum(fill.total_execution_cost for fill in fills)
    cumulative_net_return = sum(fill.net_return for fill in fills)
    duration_seconds = _duration_seconds(decisions)
    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
        if name != "paper_bundle_manifest"
    }
    return {
        "schema_version": "bigan-v8-paper-soak-summary-v1",
        "run_id": run_id,
        "mode": "deterministic_replay",
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_calls": False,
        "real_orders": False,
        "real_capital": False,
        "strategy_discovery_enabled": False,
        "injected_degradation": inject_degradation,
        "started_at": FIXED_STARTED_AT,
        "ended_at": _ended_at(FIXED_STARTED_AT, duration_seconds),
        "duration_seconds": duration_seconds,
        "row_count": len(decisions),
        "order_count": len(harness_result.orders),
        "fill_count": len(fills),
        "fill_rate": len(fills) / len(harness_result.orders)
        if harness_result.orders
        else 0.0,
        "ledger_entry_count": len(harness_result.ledger_entries),
        "final_position_count": len(harness_result.positions),
        "mean_net_return": harness_result.paper_report.mean_net_return,
        "cumulative_net_return": cumulative_net_return,
        "max_drawdown": harness_result.paper_report.max_drawdown,
        "total_execution_cost": total_execution_cost,
        "mean_execution_cost": total_execution_cost / len(fills) if fills else 0.0,
        "shadow_live_correlation": drift_metrics["shadow_live_correlation"],
        "pnl_drift": drift_metrics["mean_pnl_drift"],
        "cost_drift_ratio": drift_metrics["cost_drift_ratio"],
        "regime_mismatch_rate": drift_metrics["regime_mismatch_rate"],
        "max_live_drawdown": live_risk_metrics["max_live_drawdown"],
        "phase5_passed": harness_result.phase5_result.passed,
        "phase5_kill_switch_triggered": safety_action["kill_switch_triggered"],
        "phase5_reason_codes": list(safety_action["reason_codes"]),
        "rollback_model_id": safety_action["rollback_model_id"],
        "phase6_passed": harness_result.phase6_result.passed,
        "phase6_candidate_identity_verified": (
            harness_result.phase6_result.report.candidate_identity_verified
        ),
        "phase6_deployment_status": (
            harness_result.phase6_result.report.deployment_status
        ),
        "paper_order_stream_sha256": (
            harness_result.paper_report.paper_order_stream_sha256
        ),
        "paper_fill_stream_sha256": (
            harness_result.paper_report.paper_fill_stream_sha256
        ),
        "paper_ledger_sha256": harness_result.paper_report.paper_ledger_sha256,
        "paper_positions_sha256": harness_result.paper_report.paper_positions_sha256,
        "artifact_hash_scope": "sibling_artifacts_excluding_bundle_manifest",
        "artifact_hashes": artifact_hashes,
    }


def _bundle_manifest_with_soak_summary(
    bundle_manifest: dict[str, Any],
    *,
    summary_path: Path,
) -> dict[str, Any]:
    updated = dict(bundle_manifest)
    updated["schema_version"] = "bigan-v8-paper-soak-bundle-v1"
    updated["paper_run_summary_sha256"] = _sha256_file(summary_path)
    updated["paper_run_summary_path"] = summary_path.name
    updated["paper_only"] = True
    updated["capital_at_risk"] = False
    updated["broker_exchange_write_enabled"] = False
    artifacts = dict(updated["artifacts"])
    artifacts["paper_run_summary"] = {
        "path": summary_path.name,
        "sha256": _sha256_file(summary_path),
        "bytes": summary_path.stat().st_size,
    }
    updated["artifacts"] = dict(sorted(artifacts.items()))
    return updated


def _duration_seconds(decisions: tuple[AdaptiveDecision, ...]) -> int:
    if len(decisions) <= 1:
        return 0
    return int((decisions[-1].decision_ts - decisions[0].decision_ts) / 1000)


def _ended_at(started_at: str, duration_seconds: int) -> str:
    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    ended = started + timedelta(seconds=duration_seconds)
    return ended.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "examples" / "v8" / "paper_runs",
        help="Directory that will contain the run-scoped paper soak bundle.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--row-count",
        type=int,
        default=DEFAULT_ROW_COUNT,
        help="Deterministic replay decision count.",
    )
    parser.add_argument(
        "--inject-degradation",
        action="store_true",
        help="Inject deterministic paper degradation to exercise the Phase 5 kill-switch.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace an existing run-scoped paper soak artifact bundle.",
    )
    args = parser.parse_args(argv)
    result = run_paper_soak(
        args.output_dir,
        run_id=args.run_id,
        row_count=args.row_count,
        inject_degradation=args.inject_degradation,
        overwrite_existing=args.overwrite_existing,
    )
    print(
        json.dumps(
            {
                "paper_run_summary": str(result.paper_run_summary_path),
                "paper_bundle_manifest": str(
                    result.artifact_paths["paper_bundle_manifest"]
                ),
                "paper_bundle_manifest_sha256": result.paper_bundle_manifest_sha256,
                "paper_only": result.paper_run_summary["paper_only"],
                "capital_at_risk": result.paper_run_summary["capital_at_risk"],
                "phase5_kill_switch_triggered": result.paper_run_summary[
                    "phase5_kill_switch_triggered"
                ],
                "phase6_deployment_status": result.paper_run_summary[
                    "phase6_deployment_status"
                ],
                "row_count": result.paper_run_summary["row_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
