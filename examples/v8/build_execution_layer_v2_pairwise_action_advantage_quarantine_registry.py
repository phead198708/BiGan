#!/usr/bin/env python3
"""Build the outcome-blind #175 prior-market quarantine registry."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    FORBIDDEN_REGISTRY_FIELDS,
    _blocked_safety_fields,
    _descriptor,
    _extract_decision_timestamps,
    _extract_market_ids,
    _find_fields,
    _load_json,
    _load_json_or_jsonl,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
)

SCHEMA_VERSION = "bigan-v8-execution-layer-v2-pairwise-action-advantage-lcb-quarantine-registry-v1"


def build_quarantine_registry(
    *,
    run_id: str,
    output_dir: Path,
    created_at_ts: int,
    source_registry_pins: tuple[tuple[Path, str], ...],
    assignment_rows_pins: tuple[tuple[Path, str], ...],
    batch_progress_pins: tuple[tuple[Path, str], ...],
) -> dict[str, Any]:
    if not run_id.strip():
        raise ValueError("run_id is required")
    if created_at_ts <= 0:
        raise ValueError("created_at_ts must be positive")
    if not source_registry_pins or not batch_progress_pins:
        raise ValueError("source registry and batch progress pins are required")

    source_descriptors: list[dict[str, str]] = []
    market_sources: dict[str, set[str]] = defaultdict(set)
    market_decision_timestamps: dict[str, list[int]] = defaultdict(list)

    for source_kind, pins in (
        ("source_registry", source_registry_pins),
        ("assignment_rows", assignment_rows_pins),
    ):
        for path, expected_sha256 in pins:
            resolved = path.resolve()
            _verify_pin(
                resolved,
                expected_sha256,
                name=f"#175 {source_kind}",
            )
            payload = _load_json_or_jsonl(resolved)
            forbidden = sorted(_find_fields(payload, FORBIDDEN_REGISTRY_FIELDS))
            if forbidden:
                raise ValueError(
                    f"{source_kind} contains forbidden outcome fields: " + ", ".join(forbidden)
                )
            market_ids = _extract_market_ids(payload)
            timestamps = _extract_decision_timestamps(payload)
            if not market_ids or not timestamps:
                raise ValueError(f"{source_kind} has incomplete market/time coverage")
            source_descriptors.append(
                {
                    **_descriptor(resolved),
                    "source_kind": source_kind,
                }
            )
            for market_id in market_ids:
                market_sources[market_id].add(str(resolved))
                market_decision_timestamps[market_id].extend(timestamps)

    batch_descriptors: list[dict[str, str]] = []
    capture_count = 0
    capture_market_count = 0
    missing_capture_market_identity_count = 0
    for path, expected_sha256 in batch_progress_pins:
        resolved = path.resolve()
        _verify_pin(resolved, expected_sha256, name="#175 batch progress")
        batch = _load_json(resolved)
        forbidden = sorted(_find_fields(batch, FORBIDDEN_REGISTRY_FIELDS))
        if forbidden:
            raise ValueError(
                "batch progress contains forbidden outcome fields: " + ", ".join(forbidden)
            )
        captures = [dict(row) for row in batch.get("captures") or []]
        if int(batch.get("capture_count") or 0) != len(captures):
            raise ValueError("batch capture count mismatch")
        batch_descriptors.append(
            {
                **_descriptor(resolved),
                "source_kind": "batch_progress",
            }
        )
        for capture in captures:
            capture_count += 1
            run_dir = Path(str(capture.get("run_dir") or "")).resolve()
            market_path = run_dir / "raw" / "raw_polymarket_markets.jsonl"
            if not market_path.is_file():
                market_path = run_dir / "provider_raw" / "raw_polymarket_markets.jsonl"
            if not market_path.is_file():
                missing_capture_market_identity_count += 1
                continue
            market_rows = _load_json_or_jsonl(market_path)
            market_ids = _extract_market_ids(market_rows)
            if not market_ids:
                missing_capture_market_identity_count += 1
                continue
            capture_market_count += len(market_ids)
            scheduled_ts = int(capture.get("scheduled_round_start_ts") or 0)
            for market_row in market_rows:
                if not isinstance(market_row, dict):
                    continue
                market_id = str(market_row.get("market_id") or market_row.get("condition_id") or "")
                if not market_id:
                    continue
                conservative_latest_decision_ts = max(
                    scheduled_ts,
                    int(market_row.get("market_end_ts") or 0),
                )
                if conservative_latest_decision_ts <= 0:
                    raise ValueError("capture market decision boundary is missing")
                market_sources[market_id].add(str(market_path))
                market_decision_timestamps[market_id].append(conservative_latest_decision_ts)

    if missing_capture_market_identity_count:
        raise ValueError(
            f"capture market identity is incomplete: {missing_capture_market_identity_count}"
        )
    if not market_sources:
        raise ValueError("quarantine registry has no market identities")

    entries = []
    for market_id in sorted(market_sources):
        timestamps = market_decision_timestamps[market_id]
        if not timestamps or any(value <= 0 for value in timestamps):
            raise ValueError(f"market decision-time boundary is incomplete: {market_id}")
        entries.append(
            {
                "market_id": market_id,
                "decision_ts": max(timestamps),
                "source_paths": sorted(market_sources[market_id]),
            }
        )
    market_ids = [row["market_id"] for row in entries]
    registry = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_ts": created_at_ts,
        "quarantine_all_issues_through": 174,
        "source_registry_inputs": sorted(
            source_descriptors,
            key=lambda row: (row["source_kind"], row["path"]),
        ),
        "batch_progress_inputs": sorted(
            batch_descriptors,
            key=lambda row: row["path"],
        ),
        "capture_count": capture_count,
        "capture_market_count": capture_market_count,
        "missing_capture_market_identity_count": 0,
        "prior_unique_market_count": len(market_ids),
        "prior_market_ids": market_ids,
        "prior_market_ids_sha256": canonical_json_sha256(market_ids),
        "maximum_prior_decision_ts": max(row["decision_ts"] for row in entries),
        "market_entries": entries,
        "outcome_label_or_pnl_artifacts_opened": False,
        "resolution_artifacts_opened": False,
        "uses_issue174_confirmatory_labels_for_tuning": False,
        "registry_complete": True,
        **_blocked_safety_fields(),
    }
    registry["quarantine_registry_id"] = canonical_json_sha256(registry)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    registry_path = run_dir / "issue175_prior_market_quarantine_registry.json"
    _write_json(registry_path, registry)
    descriptor = {
        "schema_version": f"{SCHEMA_VERSION}-descriptor",
        "run_id": run_id,
        "registry": _descriptor(registry_path),
        "prior_unique_market_count": len(market_ids),
        "prior_market_ids_sha256": registry["prior_market_ids_sha256"],
        "maximum_prior_decision_ts": registry["maximum_prior_decision_ts"],
        "outcome_label_or_pnl_artifacts_opened": False,
        "resolution_artifacts_opened": False,
        **_blocked_safety_fields(),
    }
    descriptor_path = run_dir / "issue175_prior_market_quarantine_descriptor.json"
    _write_json(descriptor_path, descriptor)
    return {
        "run_dir": run_dir,
        "registry_path": registry_path,
        "registry_sha256": _sha256_file(registry_path),
        "descriptor_path": descriptor_path,
        "descriptor_sha256": _sha256_file(descriptor_path),
        "registry": registry,
    }


def _pins(values: list[str]) -> tuple[tuple[Path, str], ...]:
    pins = []
    for value in values:
        path, separator, digest = value.rpartition("=")
        if not separator or not path or not digest:
            raise ValueError("artifact pins must use PATH=SHA256")
        _require_sha256(digest, name="artifact SHA-256")
        pins.append((Path(path), digest.lower()))
    return tuple(pins)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    parser.add_argument("--created-at-ts", type=int, required=True)
    parser.add_argument("--source-registry-pin", action="append", default=[])
    parser.add_argument("--assignment-rows-pin", action="append", default=[])
    parser.add_argument("--batch-progress-pin", action="append", default=[])
    args = parser.parse_args()
    result = build_quarantine_registry(
        run_id=args.run_id,
        output_dir=args.output_dir,
        created_at_ts=args.created_at_ts,
        source_registry_pins=_pins(args.source_registry_pin),
        assignment_rows_pins=_pins(args.assignment_rows_pin),
        batch_progress_pins=_pins(args.batch_progress_pin),
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "prior_unique_market_count": result["registry"]["prior_unique_market_count"],
                "prior_market_ids_sha256": result["registry"]["prior_market_ids_sha256"],
                "maximum_prior_decision_ts": result["registry"]["maximum_prior_decision_ts"],
                "registry_path": str(result["registry_path"]),
                "registry_sha256": result["registry_sha256"],
                "descriptor_path": str(result["descriptor_path"]),
                "descriptor_sha256": result["descriptor_sha256"],
                "outcome_label_or_pnl_artifacts_opened": False,
                "resolution_artifacts_opened": False,
                "paper_only": True,
                "capital_at_risk": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
