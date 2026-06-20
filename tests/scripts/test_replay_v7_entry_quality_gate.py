from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "replay_v7_entry_quality_gate.py"

spec = importlib.util.spec_from_file_location("replay_v7_entry_quality_gate", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "entry_max_signal_age_seconds": None,
        "entry_max_price_drift_from_signal": None,
        "entry_raw_side_min_probability": None,
        "entry_raw_side_min_margin": None,
        "entry_raw_side_max_opposite_lead": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_entry_quality_gate_skips_stale_signal_first() -> None:
    reason = module._entry_quality_skip_reason(
        {
            "entry_signal_age_seconds": 24.0,
            "entry_price_drift_from_signal": 0.01,
        },
        _args(entry_max_signal_age_seconds=20.0, entry_max_price_drift_from_signal=0.0),
    )

    assert reason == "entry_signal_age_above_replay_threshold"


def test_entry_quality_gate_skips_adverse_price_drift() -> None:
    reason = module._entry_quality_skip_reason(
        {
            "entry_signal_age_seconds": 12.0,
            "entry_price_drift_from_signal": 0.04,
        },
        _args(entry_max_signal_age_seconds=20.0, entry_max_price_drift_from_signal=0.02),
    )

    assert reason == "entry_price_drift_above_replay_threshold"


def test_entry_quality_gate_skips_raw_side_disagreement() -> None:
    reason = module._entry_quality_skip_reason(
        {
            "side": "DOWN",
            "entry_p_up": 0.58,
            "entry_p_down": 0.42,
        },
        _args(entry_raw_side_min_probability=0.50, entry_raw_side_max_opposite_lead=0.03),
    )

    assert reason == "entry_raw_side_probability_below_replay_threshold"


def test_entry_quality_gate_allows_small_raw_opposite_lead() -> None:
    reason = module._entry_quality_skip_reason(
        {
            "side": "DOWN",
            "entry_p_up": 0.52,
            "entry_p_down": 0.50,
        },
        _args(entry_raw_side_min_probability=0.50, entry_raw_side_max_opposite_lead=0.03),
    )

    assert reason is None


def test_entry_quality_gate_skips_raw_margin_below_min() -> None:
    reason = module._entry_quality_skip_reason(
        {
            "side": "UP",
            "entry_p_up": 0.511,
            "entry_p_down": 0.489,
        },
        _args(entry_raw_side_min_margin=0.03),
    )

    assert reason == "entry_raw_side_margin_below_replay_threshold"


def test_build_report_recomputes_kept_pnl() -> None:
    entries = {
        "entry-a": {
            "entry_signal_age_seconds": 10.0,
            "entry_price_drift_from_signal": 0.01,
        },
        "entry-b": {
            "entry_signal_age_seconds": 25.0,
            "entry_price_drift_from_signal": 0.01,
        },
    }
    bets = [
        {"run_id": "run", "event_id": "entry-a", "round_slug": "r1", "pnl": 0.5},
        {"run_id": "run", "event_id": "entry-b", "round_slug": "r2", "pnl": -1.0},
    ]

    report = module._build_report(
        entries=entries,
        bets=bets,
        args=_args(entry_max_signal_age_seconds=20.0),
    )

    assert report["summary"]["matched_bets"] == 2
    assert report["summary"]["kept_bets"] == 1
    assert report["summary"]["skipped_bets"] == 1
    assert report["summary"]["baseline_pnl"] == pytest.approx(-0.5)
    assert report["summary"]["replay_pnl"] == pytest.approx(0.5)
