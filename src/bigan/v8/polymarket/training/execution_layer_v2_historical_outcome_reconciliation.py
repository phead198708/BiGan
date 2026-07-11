"""Read-only CLOB reconciliation for unresolved historical paper fills."""

from __future__ import annotations

import json
import shutil
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_one_hour_goal import (
    _settlement_evaluation_rows_from_resolutions,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_replay import (
    _safety_report_fields,
    _sha256_file,
    _write_json,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev_corpus import (
    _fill_id,
    _ingest_source_manifest,
    _load_rows,
)

HISTORICAL_OUTCOME_RECONCILIATION_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-historical-outcome-reconciliation-v1"
)
CLOB_MARKET_ENDPOINT = "https://clob.polymarket.com/markets/{condition_id}"


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2HistoricalOutcomeReconciliationConfig:
    run_id: str
    source_manifest_paths: tuple[Path | str, ...]
    output_dir: Path | str
    request_timeout_seconds: float = 10.0
    max_workers: int = 4
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.source_manifest_paths:
            raise ValueError("at least one source manifest is required")
        object.__setattr__(
            self,
            "source_manifest_paths",
            tuple(
                Path(path).expanduser().resolve()
                for path in self.source_manifest_paths
            ),
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.request_timeout_seconds <= 0.0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2HistoricalOutcomeReconciliationResult:
    output_dir: Path
    report: dict[str, Any]
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    source_bundle_manifests: tuple[Path, ...]


def run_execution_layer_v2_historical_outcome_reconciliation(
    config: ExecutionLayerV2HistoricalOutcomeReconciliationConfig,
    *,
    fetch_market: Callable[[str, float], dict[str, Any]] | None = None,
    outcome_observed_at_ts: float | None = None,
) -> ExecutionLayerV2HistoricalOutcomeReconciliationResult:
    """Resolve historical outcomes without mutating the source run artifacts."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"reconciliation output exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    fetcher = fetch_market or _fetch_clob_market
    observed_at_ts = (
        float(outcome_observed_at_ts)
        if outcome_observed_at_ts is not None
        else time.time() * 1000.0
    )

    source_reports: list[dict[str, Any]] = []
    bundle_manifests: list[Path] = []
    for source_manifest_path in config.source_manifest_paths:
        source_report, _ = _ingest_source_manifest(source_manifest_path)
        if not source_report["included"]:
            source_reports.append(
                {
                    "source_run_id": source_report["source_run_id"],
                    "source_manifest_path": str(source_manifest_path),
                    "reconciliation_status": "blocked_source_manifest",
                    "reason_codes": source_report["blocking_reason_codes"],
                    "unresolved_fill_count_before": 0,
                    "resolved_fill_count": 0,
                    "unresolved_fill_count_after": 0,
                }
            )
            continue
        source_result = _reconcile_source_run(
            run_dir=run_dir,
            source_manifest_path=source_manifest_path,
            source_report=source_report,
            request_timeout_seconds=config.request_timeout_seconds,
            max_workers=config.max_workers,
            fetch_market=fetcher,
            outcome_observed_at_ts=observed_at_ts,
        )
        source_reports.append(source_result["report"])
        bundle_manifests.append(source_result["manifest_path"])

    report = {
        "schema_version": HISTORICAL_OUTCOME_RECONCILIATION_SCHEMA_VERSION,
        "run_id": config.run_id,
        "endpoint_template": CLOB_MARKET_ENDPOINT,
        "read_only_public_provider": True,
        "bounded_request_timeout_seconds": config.request_timeout_seconds,
        "bounded_worker_count": config.max_workers,
        "source_run_count": len(config.source_manifest_paths),
        "source_bundle_created_count": len(bundle_manifests),
        "unresolved_fill_count_before": sum(
            row["unresolved_fill_count_before"] for row in source_reports
        ),
        "resolved_fill_count": sum(row["resolved_fill_count"] for row in source_reports),
        "unresolved_fill_count_after": sum(
            row["unresolved_fill_count_after"] for row in source_reports
        ),
        "source_reports": source_reports,
        "original_source_artifacts_mutated": False,
        "settlement_timestamp_required": False,
        "outcome_observed_at_ts_is_audit_only": True,
        "uses_outcome_in_decision_time_logic": False,
        "diagnostic_only": True,
        **_safety_report_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    artifact_paths = {
        "report": run_dir / "historical_outcome_reconciliation_report.json",
        "summary": run_dir / "historical_outcome_reconciliation_report.md",
    }
    _write_json(artifact_paths["report"], report)
    _write_text(artifact_paths["summary"], _report_to_markdown(report))
    artifact_hashes = {
        name: _sha256_file(path) for name, path in artifact_paths.items()
    }
    return ExecutionLayerV2HistoricalOutcomeReconciliationResult(
        output_dir=run_dir,
        report=report,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        source_bundle_manifests=tuple(bundle_manifests),
    )


def _reconcile_source_run(
    *,
    run_dir: Path,
    source_manifest_path: Path,
    source_report: dict[str, Any],
    request_timeout_seconds: float,
    max_workers: int,
    fetch_market: Callable[[str, float], dict[str, Any]],
    outcome_observed_at_ts: float,
) -> dict[str, Any]:
    source_run_id = source_report["source_run_id"]
    artifacts = source_report["resolved_artifacts"]
    fills = _load_rows(Path(artifacts["paper_fill_log"]["path"]))
    settlements = _load_rows(Path(artifacts["settlement_rows"]["path"]))
    unresolved = [row for row in settlements if not _settlement_row_resolved(row)]
    unresolved_fill_ids = {_fill_id(row) for row in unresolved}
    unresolved_fills = [row for row in fills if _fill_id(row) in unresolved_fill_ids]
    market_ids = sorted(
        {str(row.get("market_id") or "") for row in unresolved_fills}
        - {""}
    )
    responses, failures = _fetch_market_resolutions(
        market_ids,
        request_timeout_seconds=request_timeout_seconds,
        max_workers=max_workers,
        fetch_market=fetch_market,
        outcome_observed_at_ts=outcome_observed_at_ts,
    )
    resolutions_by_market = {
        row["market_id"]: row
        for row in responses
        if row["resolution_status"] == "resolved"
    }
    evaluation_rows = _settlement_evaluation_rows_from_resolutions(
        fills=unresolved_fills,
        resolutions_by_market=resolutions_by_market,
    )
    evaluations_by_fill = {_fill_id(row): row for row in evaluation_rows}
    reconciled_rows = []
    for original in settlements:
        fill_id = _fill_id(original)
        evaluation = evaluations_by_fill.get(fill_id)
        if evaluation is None:
            reconciled_rows.append(dict(original))
            continue
        row = dict(original)
        row.update(evaluation)
        row["resolution_status"] = "resolved"
        row["settlement_status"] = "settled"
        row["outcome_observed_at_ts"] = outcome_observed_at_ts
        row["outcome_observation_time_source"] = "provider_response_clock"
        row.pop("settlement_pnl_row_hash", None)
        row["settlement_evaluation_row_hash"] = canonical_json_sha256(row)
        reconciled_rows.append(row)
    unresolved_after = [row for row in reconciled_rows if not _settlement_row_resolved(row)]

    bundle_dir = run_dir / source_run_id
    bundle_dir.mkdir(parents=True)
    trace_path = _copy_artifact(
        Path(artifacts["signal_trace"]["path"]),
        bundle_dir / "incremental_fresh_loop/o_v8_paper_fresh_signal_trace.json",
    )
    intent_path = _copy_artifact(
        Path(artifacts["paper_intent_log"]["path"]),
        bundle_dir / "one_hour_paper_intent_log.jsonl",
    )
    fill_path = _copy_artifact(
        Path(artifacts["paper_fill_log"]["path"]),
        bundle_dir / "one_hour_paper_fill_log.jsonl",
    )
    settlement_path = bundle_dir / "settlement_evaluation_rows.jsonl"
    response_path = bundle_dir / "clob_resolution_rows.jsonl"
    _write_jsonl(settlement_path, reconciled_rows)
    _write_jsonl(response_path, responses)
    source_bundle_report = {
        "schema_version": HISTORICAL_OUTCOME_RECONCILIATION_SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "unique_market_count": len(market_ids),
        "unresolved_fill_count_before": len(unresolved),
        "resolved_fill_count": len(evaluation_rows),
        "unresolved_fill_count_after": len(unresolved_after),
        "resolution_failure_reason_distribution": _reason_distribution(failures),
        "resolution_failures": failures,
        "original_source_artifacts_mutated": False,
        "uses_outcome_in_decision_time_logic": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    source_bundle_report["report_id"] = canonical_json_sha256(source_bundle_report)
    source_report_path = bundle_dir / "clob_settlement_reconciliation_report.json"
    _write_json(source_report_path, source_bundle_report)
    artifact_paths = {
        "incremental_fresh_loop/o_v8_paper_fresh_signal_trace.json": trace_path,
        "one_hour_paper_intent_log.jsonl": intent_path,
        "one_hour_paper_fill_log.jsonl": fill_path,
        "settlement_evaluation_rows.jsonl": settlement_path,
        "clob_resolution_rows.jsonl": response_path,
        "clob_settlement_reconciliation_report.json": source_report_path,
    }
    manifest = {
        "schema_version": "bigan-v8-clob-settlement-reconciliation-manifest-v1",
        "run_id": f"{source_run_id}-clob-settlement-reconciled",
        "source_run_id": source_run_id,
        "completed": True,
        "unresolved_settlement_count": len(unresolved_after),
        "resolved_market_count": len(resolutions_by_market),
        "settled_fill_count": len(reconciled_rows) - len(unresolved_after),
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_hashes": {
            name: _sha256_file(path) for name, path in artifact_paths.items()
        },
        "diagnostic_only": True,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = bundle_dir / "clob_settlement_reconciliation_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest_path": manifest_path,
        "report": {
            "source_run_id": source_run_id,
            "source_manifest_path": str(source_manifest_path),
            "reconciliation_status": (
                "resolved" if not unresolved_after else "partially_unresolved"
            ),
            "reason_codes": sorted({row["reason_code"] for row in failures}),
            "unique_market_count": len(market_ids),
            "unresolved_fill_count_before": len(unresolved),
            "resolved_fill_count": len(evaluation_rows),
            "unresolved_fill_count_after": len(unresolved_after),
            "bundle_manifest_path": str(manifest_path),
            "bundle_manifest_sha256": _sha256_file(manifest_path),
        },
    }


def _fetch_market_resolutions(
    market_ids: list[str],
    *,
    request_timeout_seconds: float,
    max_workers: int,
    fetch_market: Callable[[str, float], dict[str, Any]],
    outcome_observed_at_ts: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    responses: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(market_ids) or 1)) as pool:
        futures = {
            pool.submit(fetch_market, market_id, request_timeout_seconds): market_id
            for market_id in market_ids
        }
        for future in as_completed(futures):
            market_id = futures[future]
            try:
                payload = future.result()
                responses.append(
                    _resolution_row(
                        market_id,
                        payload,
                        outcome_observed_at_ts=outcome_observed_at_ts,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "market_id": market_id,
                        "reason_code": f"clob_resolution_fetch_failed:{exc.__class__.__name__}",
                    }
                )
    return sorted(responses, key=lambda row: row["market_id"]), sorted(
        failures, key=lambda row: (row["market_id"], row["reason_code"])
    )


def _resolution_row(
    market_id: str,
    payload: dict[str, Any],
    *,
    outcome_observed_at_ts: float,
) -> dict[str, Any]:
    if payload.get("closed") is not True:
        raise ValueError("market_not_closed")
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError("market_tokens_missing")
    winners = [token for token in tokens if token.get("winner") is True]
    if len(winners) != 1:
        raise ValueError("winner_token_count_not_one")
    outcome = str(winners[0].get("outcome") or "").upper()
    if outcome not in {"UP", "DOWN"}:
        raise ValueError("winner_outcome_invalid")
    raw_response = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "market_id": market_id,
        "condition_id": str(payload.get("condition_id") or market_id),
        "resolution_status": "resolved",
        "resolved_outcome": outcome,
        "resolution_source_type": "polymarket_clob_read_only_settlement",
        "outcome_observed_at_ts": outcome_observed_at_ts,
        "outcome_observation_time_source": "provider_response_clock",
        "raw_response": raw_response,
        "raw_response_sha256": canonical_json_sha256(payload),
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _fetch_clob_market(condition_id: str, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(
        CLOB_MARKET_ENDPOINT.format(condition_id=condition_id),
        headers={"User-Agent": "bigan-v8-paper-read-only-reconciliation/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("clob_market_response_not_object")
    return payload


def _settlement_row_resolved(row: dict[str, Any]) -> bool:
    status = str(row.get("resolution_status") or row.get("settlement_status") or "").lower()
    outcome = str(row.get("resolved_outcome") or "").upper()
    return status in {"normal", "resolved", "settled"} and outcome in {"UP", "DOWN"}


def _copy_artifact(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _reason_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        reason: sum(row["reason_code"] == reason for row in rows)
        for reason in sorted({row["reason_code"] for row in rows})
    }


def _report_to_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Historical Paper Outcome Reconciliation",
            "",
            f"- source runs: `{report['source_run_count']}`",
            f"- unresolved fills before: `{report['unresolved_fill_count_before']}`",
            f"- resolved fills: `{report['resolved_fill_count']}`",
            f"- unresolved fills after: `{report['unresolved_fill_count_after']}`",
            "- source artifacts mutated: `false`",
            "- read-only CLOB provider: `true`",
            "- paper only: `true`",
            "",
        ]
    )


__all__ = [
    "ExecutionLayerV2HistoricalOutcomeReconciliationConfig",
    "ExecutionLayerV2HistoricalOutcomeReconciliationResult",
    "run_execution_layer_v2_historical_outcome_reconciliation",
]
