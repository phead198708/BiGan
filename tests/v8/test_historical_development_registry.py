"""Tests for the frozen historical development-market registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.historical_corpus_compatibility import (
    HISTORICAL_DEVELOPMENT_COMPATIBLE,
    HISTORICAL_INCOMPATIBLE,
    REQUIRED_FILES,
)
from bigan.v8.polymarket.training.historical_development_registry import (
    HistoricalDevelopmentRegistryConfig,
    freeze_historical_development_registry,
)


def test_selects_earliest_compatible_unique_pre_boundary_markets(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        rows=[
            _row_spec("market-c", 300),
            _row_spec("market-a", 100),
            _row_spec("market-post", 1_100),
            _row_spec("market-b", 200),
            _row_spec("market-bad", 50, classification=HISTORICAL_INCOMPATIBLE),
        ],
        boundary=1_000,
    )

    result = _run(tmp_path, fixture, selected_count=2)
    rows = _jsonl(result["registry_rows_path"])

    assert [row["market_id"] for row in rows] == ["market-a", "market-b"]
    assert [row["selection_rank"] for row in rows] == [1, 2]
    assert result["report"]["eligible_pre_boundary_market_count"] == 3
    assert result["report"]["all_selected_strictly_before_boundary"] is True
    assert result["report"]["duplicate_selected_market_count"] == 0
    assert result["report"]["maximum_selected_decision_ts"] == 200


def test_hash_drift_fails_before_selection(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        rows=[_row_spec("market-a", 100)],
        boundary=1_000,
    )
    fixture["rows_path"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="compatibility rows SHA-256 mismatch"):
        _run(tmp_path, fixture, selected_count=1)


def test_insufficient_pre_boundary_support_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        rows=[_row_spec("market-a", 100), _row_spec("market-post", 1_100)],
        boundary=1_000,
    )

    with pytest.raises(ValueError, match="insufficient eligible pre-boundary"):
        _run(tmp_path, fixture, selected_count=2)


def test_duplicate_market_identity_cannot_inflate_selected_support(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        rows=[
            _row_spec("market-a", 100, corpus_suffix="first"),
            _row_spec("market-a", 200, corpus_suffix="second"),
        ],
        boundary=1_000,
    )

    with pytest.raises(ValueError, match="duplicate market identities"):
        _run(tmp_path, fixture, selected_count=2)


def test_feature_causality_and_forbidden_fields_are_revalidated(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        rows=[
            _row_spec("market-good", 100),
            _row_spec("market-leak", 200, max_input_ts=201),
            _row_spec("market-forbidden", 300, forbidden_field=True),
        ],
        boundary=1_000,
    )

    result = _run(tmp_path, fixture, selected_count=1)

    assert result["report"]["eligible_pre_boundary_market_count"] == 1
    reasons = result["report"]["exclusion_reason_distribution"]
    assert reasons["feature_timestamp_causality_violation"] == 1
    assert reasons["forbidden_decision_fields_present"] == 1


def test_label_and_resolution_semantics_are_not_parsed(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        rows=[_row_spec("market-a", 100, malformed_outcome_files=True)],
        boundary=1_000,
    )

    result = _run(tmp_path, fixture, selected_count=1)
    row = _jsonl(result["registry_rows_path"])[0]

    assert (
        row["artifact_pins"]["polymarket_label_rows.jsonl"][
            "semantic_content_parsed"
        ]
        is False
    )
    assert (
        row["artifact_pins"]["polymarket_resolution_events.jsonl"][
            "semantic_content_parsed"
        ]
        is False
    )
    access = result["report"]["forbidden_evidence_access_audit"]
    assert access["label_rows_semantic_content_parsed"] is False
    assert access["resolution_rows_semantic_content_parsed"] is False
    assert access["outcome_values_loaded"] is False
    assert access["pnl_values_loaded"] is False


def test_registry_is_deterministic_and_safety_remains_blocked(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        rows=[_row_spec("market-b", 200), _row_spec("market-a", 100)],
        boundary=1_000,
    )
    first = _run(tmp_path, fixture, selected_count=2)
    first_descriptor_hash = _sha256(first["descriptor_path"])
    second = _run(tmp_path, fixture, selected_count=2, overwrite=True)

    assert _sha256(second["descriptor_path"]) == first_descriptor_hash
    report = second["report"]
    assert report["selected_market_count"] == 2
    assert report["future_hybrid_role_plan"]["fresh_calibration_market_count"] == 45
    assert report["future_hybrid_role_plan"]["fresh_confirmatory_market_count"] == 60
    assert (
        report["future_hybrid_role_plan"]["estimated_initial_capture_attempt_count"]
        == 120
    )
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False
    assert report["wallet_signing_enabled"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False


def _run(
    tmp_path: Path,
    fixture: dict[str, Path | str],
    *,
    selected_count: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    return freeze_historical_development_registry(
        HistoricalDevelopmentRegistryConfig(
            run_id="registry-run",
            output_dir=tmp_path / "runs",
            compatibility_report_path=fixture["report_path"],
            expected_compatibility_report_sha256=str(fixture["report_sha256"]),
            compatibility_rows_path=fixture["rows_path"],
            expected_compatibility_rows_sha256=str(fixture["rows_sha256"]),
            compatibility_manifest_path=fixture["manifest_path"],
            expected_compatibility_manifest_sha256=str(fixture["manifest_sha256"]),
            boundary_freeze_manifest_path=fixture["boundary_path"],
            expected_boundary_freeze_manifest_sha256=str(fixture["boundary_sha256"]),
            selected_market_count=selected_count,
            overwrite_existing=overwrite,
        )
    )


def _fixture(
    tmp_path: Path,
    *,
    rows: list[dict[str, Any]],
    boundary: int,
) -> dict[str, Path | str]:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    compatibility_rows = [
        _compatibility_row(input_dir, index=index, **spec)
        for index, spec in enumerate(rows)
    ]
    rows_path = input_dir / "compatibility_rows.jsonl"
    _write_jsonl(rows_path, compatibility_rows)
    input_inventory = [
        {
            "corpus_dir": row["corpus_dir"],
            "corpus_manifest_sha256": row["file_inventory"][
                "polymarket_corpus_manifest.json"
            ]["sha256"],
            "training_corpus_provenance_sha256": row["file_inventory"][
                "training_corpus_provenance.json"
            ]["sha256"],
        }
        for row in compatibility_rows
    ]
    report = {
        "schema_version": "bigan-v8-historical-corpus-compatibility-report-v1",
        "audit_mode": "outcome_blind_read_only_historical_compatibility",
        "discovered_corpus_count": len(compatibility_rows),
        "input_inventory_hash": canonical_json_sha256(input_inventory),
        "compatibility_rows": _descriptor(rows_path),
        "outcome_blind_access_audit": {
            "label_rows_content_parsed": False,
            "resolution_rows_content_parsed": False,
            "outcome_values_loaded": False,
            "pnl_values_loaded": False,
            "oracle_values_loaded": False,
            "validation_metrics_loaded": False,
        },
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = input_dir / "compatibility_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema_version": "bigan-v8-historical-corpus-compatibility-manifest-v1",
        "report": _descriptor(report_path),
        "compatibility_rows": _descriptor(rows_path),
        "input_inventory_hash": report["input_inventory_hash"],
        "outcome_values_loaded": False,
        "pnl_values_loaded": False,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = input_dir / "compatibility_manifest.json"
    _write_json(manifest_path, manifest)
    boundary_payload = {
        "schema_version": (
            "bigan-v8-execution-layer-v2-pairwise-action-advantage-lcb-"
            "precollection-role-freeze-v1"
        ),
        "minimum_collection_decision_ts": boundary,
        "git_commit": "a" * 40,
    }
    boundary_payload["precollection_freeze_id"] = canonical_json_sha256(
        boundary_payload
    )
    boundary_path = input_dir / "boundary_freeze.json"
    _write_json(boundary_path, boundary_payload)
    return {
        "rows_path": rows_path,
        "rows_sha256": _sha256(rows_path),
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "boundary_path": boundary_path,
        "boundary_sha256": _sha256(boundary_path),
    }


def _row_spec(
    market_id: str,
    decision_ts: int,
    *,
    max_input_ts: int | None = None,
    classification: str = HISTORICAL_DEVELOPMENT_COMPATIBLE,
    corpus_suffix: str = "",
    forbidden_field: bool = False,
    malformed_outcome_files: bool = False,
) -> dict[str, Any]:
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts if max_input_ts is None else max_input_ts,
        "classification": classification,
        "corpus_suffix": corpus_suffix,
        "forbidden_field": forbidden_field,
        "malformed_outcome_files": malformed_outcome_files,
    }


def _compatibility_row(
    input_dir: Path,
    *,
    index: int,
    market_id: str,
    decision_ts: int,
    max_input_ts: int,
    classification: str,
    corpus_suffix: str,
    forbidden_field: bool,
    malformed_outcome_files: bool,
) -> dict[str, Any]:
    corpus_dir = input_dir / f"corpus-{index:03d}-{corpus_suffix or market_id}"
    corpus_dir.mkdir()
    feature = {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": max_input_ts,
        "features": {"time_to_close_seconds": 100.0},
    }
    if forbidden_field:
        feature["features"]["settlement_pnl"] = 1.0
    inventory = {}
    for filename in REQUIRED_FILES:
        path = corpus_dir / filename
        if filename == "polymarket_feature_rows.jsonl":
            _write_jsonl(path, [feature])
        elif filename in {
            "polymarket_label_rows.jsonl",
            "polymarket_resolution_events.jsonl",
        } and malformed_outcome_files:
            path.write_text("not-json-outcome-content\n", encoding="utf-8")
        elif filename.endswith(".jsonl"):
            _write_jsonl(path, [{"market_id": market_id}])
        else:
            _write_json(path, {"market_id": market_id})
        inventory[filename] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": 1 if filename.endswith(".jsonl") else None,
        }
    row = {
        "schema_version": "bigan-v8-historical-corpus-compatibility-row-v1",
        "corpus_dir": str(corpus_dir.resolve()),
        "corpus_id": f"corpus-{index:03d}",
        "market_id": market_id,
        "round_slug": f"btc-updown-5m-{decision_ts // 1000}",
        "classification": classification,
        "historical_development_fit_eligible": (
            classification == HISTORICAL_DEVELOPMENT_COMPATIBLE
        ),
        "fresh_confirmatory_eligible": False,
        "deduplication_status": "selected_unique_market",
        "file_inventory": inventory,
        "outcome_values_loaded": False,
        "pnl_values_loaded": False,
    }
    row["row_id"] = canonical_json_sha256(row)
    return row


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
