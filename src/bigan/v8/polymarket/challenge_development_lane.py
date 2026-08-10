"""Persistent, candidate-agnostic challenge-model development data lane."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA_VERSION = "bigan-challenge-model-development-lane-protocol-v1"
BATCH_INDEX_SCHEMA_VERSION = "bigan-challenge-model-development-lane-batch-index-v1"
FINALIZED_INDEX_SCHEMA_VERSION = "bigan-challenge-model-development-lane-finalized-index-v1"
SAFETY = {
    "paper_allowed": False,
    "live_allowed": False,
    "write_allowed": False,
    "wallet_allowed": False,
    "handoff_allowed": False,
    "promotion_allowed": False,
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_development_lane_protocol(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path | str,
) -> dict[str, Any]:
    """Validate the 15m-only development lane and its diagnostic authorization."""

    expected = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "lane_id": "challenge-model-development-btc-updown-15m-v1",
        "market_family": "btc_updown_15m",
        "lane_role": "persistent_candidate_agnostic_development_data",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "confirmatory_collection_lane": False,
        "safety": SAFETY,
    }
    blockers = [
        key for key, value in expected.items() if protocol.get(key) != value
    ]
    capture = dict(protocol.get("outcome_blind_capture") or {})
    if capture != {
        "enabled": True,
        "outcomes_labels_or_pnl_available_to_capture_control": False,
        "resolution_provider_enabled_in_capture_process": False,
        "settlement_finalizer_enabled_in_capture_process": False,
        "training_export_enabled_in_capture_process": False,
    }:
        blockers.append("outcome_blind_capture")
    finalization = dict(protocol.get("post_close_development_finalization") or {})
    if finalization != {
        "enabled_in_separate_process": True,
        "official_read_only_resolution_required": True,
        "market_close_must_precede_target_access": True,
        "finalized_corpus_role": "development_training_only",
        "feedback_to_capture_schedule_allowed": False,
        "promotion_evidence_eligible": False,
    }:
        blockers.append("post_close_development_finalization")
    schedule = dict(protocol.get("collection_schedule") or {})
    if (
        int(schedule.get("batch_round_count") or 0) <= 0
        or schedule.get("continuous") is not True
        or schedule.get("daily_summary_enabled") is not True
        or schedule.get("single_experiment_freeze") is not False
        or schedule.get("model_score_or_acceptance_control_allowed") is not False
    ):
        blockers.append("collection_schedule")
    authorization = dict(protocol.get("authorization_checkpoint") or {})
    if authorization != {
        "previous_120_round_permission_requirement_preserved": True,
        "maximum_capture_attempts_before_additional_permission": 120,
        "explicit_120_round_authorization_recorded": True,
        "authorization_extension": {
            "path": (
                "examples/v8/polymarket_configs/"
                "challenge_model_development_lane_attempt_120_authorization.json"
            ),
            "sha256": "a40e962c6f5da521726061d66298746900d8fea4a07ae0fa78909c2baff06b0a",
        },
        "stop_before_attempt_121": True,
        "attempt_121_authorized": False,
    }:
        blockers.append("authorization_checkpoint")
    authorization_descriptor = dict(authorization.get("authorization_extension") or {})
    authorization_path = Path(str(authorization_descriptor.get("path") or ""))
    if not authorization_path.is_absolute():
        authorization_path = Path(repo_root).resolve() / authorization_path
    authorization_sha256 = str(authorization_descriptor.get("sha256") or "").lower()
    previous_protocol_sha256 = ""
    if (
        not authorization_path.is_file()
        or sha256_file(authorization_path) != authorization_sha256
    ):
        blockers.append("authorization_extension")
    else:
        extension = _load_json(authorization_path)
        extension_authorization = dict(extension.get("authorization") or {})
        authorization_source = dict(extension.get("authorization_source") or {})
        superseded = dict(extension.get("supersedes_protocol") or {})
        previous_protocol_sha256 = str(superseded.get("sha256") or "").lower()
        if (
            extension.get("schema_version")
            != "bigan-challenge-model-development-lane-authorization-extension-v1"
            or extension.get("lane_id") != protocol.get("lane_id")
            or authorization_source.get("type")
            != "explicit_user_instruction_in_codex_task"
            or authorization_source.get("instruction") != "OK，开始Attempt 120"
            or superseded
            != {
                "commit": "6fbc74a7af242b437447ee0459d0c3557ddcebd6",
                "sha256": "19badf0185b16df00b58b029ff39896ffadbc167a7f760a0ee931142411a0fc2",
                "maximum_capture_attempts": 119,
            }
            or extension_authorization
            != {
                "exact_newly_authorized_attempts": [120],
                "maximum_capture_attempts": 120,
                "attempt_120_authorized": True,
                "attempt_121_authorized": False,
                "stop_before_attempt_121": True,
                "outcome_blind_capture_only": True,
                "post_close_development_finalization_allowed": True,
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
            }
            or extension.get("training_started") is not False
            or extension.get("candidate_lineage_changed") is not False
            or extension.get("thresholds_changed") is not False
            or extension.get("safety") != SAFETY
        ):
            blockers.append("authorization_extension_semantics")
    causality = dict(protocol.get("causality") or {})
    if causality != {
        "available_at_must_not_exceed_decision_ts": True,
        "max_input_ts_must_not_exceed_decision_ts": True,
        "future_exit_or_resolution_as_feature_allowed": False,
        "missing_trade_tape_as_zero_allowed": False,
    }:
        blockers.append("causality")
    descriptor = dict(protocol.get("market_selection_diagnostic_freeze") or {})
    freeze_path = Path(str(descriptor.get("path") or ""))
    if not freeze_path.is_absolute():
        freeze_path = Path(repo_root).resolve() / freeze_path
    expected_sha = str(descriptor.get("sha256") or "").lower()
    if not freeze_path.is_file() or sha256_file(freeze_path) != expected_sha:
        blockers.append("market_selection_diagnostic_freeze")
    else:
        freeze = _load_json(freeze_path)
        selection = dict(freeze.get("market_selection") or {})
        if (
            selection.get("recommendation") != "turn_to_15m"
            or selection.get("new_persistent_collection_market_families")
            != ["btc_updown_15m"]
            or selection.get("training_started") is not False
        ):
            blockers.append("market_selection_diagnostic_freeze_semantics")
    if blockers:
        raise ValueError(
            "development lane protocol validation failed: "
            + ", ".join(sorted(set(blockers)))
        )
    return {
        "lane_id": str(protocol["lane_id"]),
        "market_family": str(protocol["market_family"]),
        "batch_round_count": int(schedule["batch_round_count"]),
        "maximum_capture_attempts_before_additional_permission": int(
            authorization["maximum_capture_attempts_before_additional_permission"]
        ),
        "attempt_120_authorized": True,
        "attempt_121_authorized": False,
        "authorization_extension_path": str(authorization_path.resolve()),
        "authorization_extension_sha256": authorization_sha256,
        "previous_protocol_sha256": previous_protocol_sha256,
        "diagnostic_freeze_path": str(freeze_path.resolve()),
        "diagnostic_freeze_sha256": expected_sha,
    }


def validate_outcome_blind_batch_summary(summary: Mapping[str, Any]) -> None:
    """Fail closed if capture exposed a target or enabled a write/trading path."""

    expected = {
        "outcome_blind_collection_only": True,
        "labels_or_outcomes_opened_during_collection": False,
        "settlement_pnl_opened_during_collection": False,
        "settlement_finalizer_started": False,
        "resolution_provider_called": False,
        "training_corpus_export_attempted": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "live_exchange_write_enabled": False,
        "broker_exchange_write_enabled": False,
    }
    blockers = [key for key, value in expected.items() if summary.get(key) != value]
    if int(summary.get("finalization_attempt_count") or 0) != 0:
        blockers.append("finalization_attempt_count")
    if list(summary.get("finalizations") or []):
        blockers.append("finalizations")
    if blockers:
        raise ValueError(
            "development lane batch exposed forbidden state: "
            + ", ".join(sorted(set(blockers)))
        )


def append_outcome_blind_batch(
    *,
    index_path: Path | str,
    summary_path: Path | str,
    collector_commit: str,
    protocol_sha256: str,
    diagnostic_freeze_sha256: str,
    collected_at: str | None = None,
) -> dict[str, Any]:
    """Append one completed outcome-blind batch to the development index."""

    path = Path(index_path).resolve()
    summary_file = Path(summary_path).resolve()
    summary = _load_json(summary_file)
    validate_outcome_blind_batch_summary(summary)
    existing = load_jsonl(path)
    batch_id = str(summary["batch_id"])
    if batch_id in {str(row["batch_id"]) for row in existing}:
        raise ValueError(f"development batch already indexed: {batch_id}")
    previous = str(existing[-1]["entry_sha256"]) if existing else "0" * 64
    captures = list(summary.get("captures") or [])
    entry: dict[str, Any] = {
        "schema_version": BATCH_INDEX_SCHEMA_VERSION,
        "sequence": len(existing) + 1,
        "previous_entry_sha256": previous,
        "batch_id": batch_id,
        "collected_at": collected_at or datetime.now(UTC).isoformat(),
        "collector_commit": collector_commit,
        "protocol_sha256": protocol_sha256,
        "diagnostic_freeze_sha256": diagnostic_freeze_sha256,
        "batch_summary_path": str(summary_file),
        "batch_summary_sha256": sha256_file(summary_file),
        "market_family": _single_value(captures, "market_family"),
        "capture_count": int(summary.get("capture_count") or 0),
        "capture_status_distribution": dict(
            sorted(Counter(str(row.get("capture_status") or "unknown") for row in captures).items())
        ),
        "provider_health": {
            "error_count": int(summary.get("error_count") or 0),
            "chainlink_covered_capture_count": int(
                summary.get("chainlink_covered_capture_count") or 0
            ),
            "chainlink_fresh_capture_count": int(
                summary.get("chainlink_fresh_capture_count") or 0
            ),
            "orderbook_full_window_coverage_passed_capture_count": int(
                summary.get("orderbook_full_window_coverage_passed_capture_count") or 0
            ),
            "orderbook_full_window_coverage_failed_capture_count": int(
                summary.get("orderbook_full_window_coverage_failed_capture_count") or 0
            ),
            "raw_trade_nonempty_capture_count": sum(
                int(row.get("raw_trade_row_count") or 0) > 0 for row in captures
            ),
        },
        "outcome_blind_capture": True,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "capture_control_used_outcomes_labels_or_pnl": False,
        "safety": dict(SAFETY),
    }
    if entry["market_family"] != "btc_updown_15m":
        raise ValueError("development lane batch market family is not btc_updown_15m")
    entry["entry_sha256"] = _canonical_json_sha256(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return entry


def build_daily_capture_summary(
    *,
    index_rows: Sequence[Mapping[str, Any]],
    date_utc: str,
    collector_pid: int,
    service_status: str,
) -> dict[str, Any]:
    """Aggregate completed batches for one UTC day without reading any outcome."""

    rows = [
        row
        for row in index_rows
        if str(row.get("collected_at") or "").startswith(date_utc)
    ]
    status: Counter[str] = Counter()
    for row in rows:
        status.update(dict(row.get("capture_status_distribution") or {}))
    return {
        "schema_version": "bigan-challenge-model-development-lane-daily-capture-summary-v1",
        "date_utc": date_utc,
        "collector_pid": collector_pid,
        "service_status": service_status,
        "completed_batch_count": len(rows),
        "attempted_market_count": sum(int(row.get("capture_count") or 0) for row in rows),
        "capture_status_distribution": dict(sorted(status.items())),
        "provider_health": {
            key: sum(
                int((row.get("provider_health") or {}).get(key) or 0) for row in rows
            )
            for key in (
                "error_count",
                "chainlink_covered_capture_count",
                "chainlink_fresh_capture_count",
                "orderbook_full_window_coverage_passed_capture_count",
                "orderbook_full_window_coverage_failed_capture_count",
                "raw_trade_nonempty_capture_count",
            )
        },
        "outcomes_labels_or_pnl_read": False,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }


def append_finalized_development_rows(
    *,
    index_path: Path | str,
    finalizer_summary: Mapping[str, Any],
    finalizer_summary_path: Path | str,
    protocol_sha256: str,
    finalized_at: str | None = None,
) -> list[dict[str, Any]]:
    """Register post-close exported corpora without feeding capture control."""

    path = Path(index_path).resolve()
    existing = load_jsonl(path)
    seen = {str(row["run_id"]) for row in existing}
    appended: list[dict[str, Any]] = []
    for row in list(finalizer_summary.get("finalizations") or []):
        if row.get("finalization_status") != "exported":
            continue
        run_id = str(row["run_id"])
        if run_id in seen:
            continue
        corpus_dir = Path(str(row["exported_training_corpus_dir"])).resolve()
        manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"exported development corpus manifest missing: {run_id}")
        corpus_manifest = _load_json(manifest_path)
        if not (
            corpus_manifest.get("paper_only") is True
            and corpus_manifest.get("capital_at_risk") is False
            and corpus_manifest.get("polymarket_write_enabled") is False
            and corpus_manifest.get("wallet_signing_enabled") is False
        ):
            raise ValueError(f"exported development corpus safety failed: {run_id}")
        previous = str(existing[-1]["entry_sha256"]) if existing else "0" * 64
        entry: dict[str, Any] = {
            "schema_version": FINALIZED_INDEX_SCHEMA_VERSION,
            "sequence": len(existing) + 1,
            "previous_entry_sha256": previous,
            "run_id": run_id,
            "finalized_at": finalized_at or datetime.now(UTC).isoformat(),
            "protocol_sha256": protocol_sha256,
            "finalizer_summary_path": str(Path(finalizer_summary_path).resolve()),
            "finalizer_summary_sha256": sha256_file(finalizer_summary_path),
            "exported_corpus_manifest_path": str(manifest_path),
            "exported_corpus_manifest_sha256": sha256_file(manifest_path),
            "official_post_close_resolution_opened": True,
            "target_used_by_capture_control": False,
            "corpus_role": "development_training_only",
            "development_only_forever": True,
            "promotion_evidence_eligible": False,
            "safety": dict(SAFETY),
        }
        entry["entry_sha256"] = _canonical_json_sha256(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        existing.append(entry)
        seen.add(run_id)
        appended.append(entry)
    return appended


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {target}:{line_number}")
            rows.append(row)
    return rows


def _load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _single_value(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    values = {str(row.get(key) or "") for row in rows}
    if len(values) != 1:
        raise ValueError(f"batch captures do not have one {key}")
    return values.pop()
