from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_pipeline import (
    select_v6_7_window_index_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_p_up_semantic_compatibility_v6_7_evaluation_v1.json"
)


def _profile() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text())


def _index_row(
    sequence: int,
    *,
    valid: bool = True,
    market_id: str | None = None,
    start_ts: int | None = None,
) -> dict[str, object]:
    start = start_ts or 1_784_592_000_000 + sequence * 300_000
    return {
        "sequence": sequence,
        "scheduled_round_start_ts": start,
        "market_start_ts": start,
        "market_end_ts": start + 300_000,
        "market_id": market_id or f"market-{sequence:03d}",
        "capture_quality_valid": valid,
        "capture_quality_reason_codes": [] if valid else ["quality_failed"],
        "labels_outcomes_or_pnl_opened": False,
        "raw_resolution_row_count": 0,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _descriptor(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _calibration_freeze(tmp_path: Path) -> dict[str, object]:
    selected_path = tmp_path / "calibration-selected.jsonl"
    selected = [
        {
            "market_id": f"calibration-{index:03d}",
            "market_end_ts": 1_800_000_000_000 + index,
        }
        for index in range(60)
    ]
    _write_jsonl(selected_path, selected)
    decision_path = tmp_path / "calibration-decision.json"
    _write_json(decision_path, {"attempted_sequence_end": 65})
    return {
        "schema_version": (
            "bigan-v8-p-up-semantic-execution-compatibility-v6-7-window-"
            "freeze-manifest-v1"
        ),
        "role": "fresh_calibration",
        "future_target_access_allowed": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "selected_window_rows": _descriptor(selected_path),
        "accepted_bet_decision_freeze": _descriptor(decision_path),
    }


def test_v6_7_calibration_window_uses_earliest_quality_rows() -> None:
    rows = [
        _index_row(sequence, valid=sequence not in {2, 7, 11, 19, 23})
        for sequence in range(1, 66)
    ]
    selected, attempted = select_v6_7_window_index_rows(
        rows,
        profile=_profile(),
        role="fresh_calibration",
        calibration_prediction_freeze=None,
    )

    assert len(selected) == 60
    assert len(attempted) == 65
    assert [row["sequence"] for row in selected[:3]] == [1, 3, 4]
    assert selected[-1]["sequence"] == 65


def test_v6_7_confirmatory_starts_after_calibration_attempt_boundary(
    tmp_path: Path,
) -> None:
    freeze = _calibration_freeze(tmp_path)
    start = 1_800_000_001_000
    rows = [
        _index_row(
            sequence,
            start_ts=start + (sequence - 66) * 300_000,
        )
        for sequence in range(1, 186)
    ]
    selected, attempted = select_v6_7_window_index_rows(
        rows,
        profile=_profile(),
        role="future_confirmatory",
        calibration_prediction_freeze=freeze,
    )

    assert len(selected) == 120
    assert len(attempted) == 120
    assert selected[0]["sequence"] == 66
    assert selected[-1]["sequence"] == 185


def test_v6_7_confirmatory_fails_closed_on_calibration_market_overlap(
    tmp_path: Path,
) -> None:
    freeze = _calibration_freeze(tmp_path)
    rows = [
        _index_row(
            sequence,
            start_ts=1_800_000_001_000 + (sequence - 66) * 300_000,
            market_id=("calibration-000" if sequence == 66 else None),
        )
        for sequence in range(1, 186)
    ]

    with pytest.raises(ValueError, match="overlapping"):
        select_v6_7_window_index_rows(
            rows,
            profile=_profile(),
            role="future_confirmatory",
            calibration_prediction_freeze=freeze,
        )
