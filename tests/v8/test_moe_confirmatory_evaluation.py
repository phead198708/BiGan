from __future__ import annotations

import numpy as np

from bigan.v8.polymarket.moe_collection_boundary_r2 import _write_new_frozen_json
from bigan.v8.polymarket.moe_confirmatory_evaluation import (
    _artifact_sidecar_path,
    _gate_passes,
    _group_panel,
    _policy_result,
    _verified_path,
    _write_new_frozen_text,
)


def _features() -> list[dict]:
    return [
        {
            "decision_ts": 100,
            "features": {
                "up_ask": 0.55,
                "up_bid": 0.53,
                "up_liquidity_depth": 10.0,
                "down_ask": 0.47,
                "down_bid": 0.45,
                "down_liquidity_depth": 10.0,
            },
        }
    ]


def test_policy_pnl_uses_frozen_hold_to_settlement_cost_formula() -> None:
    result = _policy_result(
        {"accepted": True, "selected_side": "UP", "decision_ts": 100},
        _features(),
        {"payout_up": 1.0, "payout_down": 0.0},
    )

    assert result["selected_side"] == "UP"
    assert np.isclose(result["unit_net_pnl"], 1.0 - 0.55 - 0.0002 - 0.01 - 0.00005)
    costs = result["cost_decomposition"]
    assert np.isclose(costs["gross_price_edge"], 1.0 - 0.54)
    assert np.isclose(costs["entry_spread_cost"], 0.01)
    assert np.isclose(costs["unit_net_pnl"], result["unit_net_pnl"])


def test_no_trade_is_zero_and_participates_in_group_panel() -> None:
    result = _policy_result(
        {"accepted": False, "selected_side": None, "decision_ts": 100},
        _features(),
        {"payout_up": 0.0, "payout_down": 1.0},
    )
    assert result["unit_net_pnl"] == 0.0
    rows = [
        {
            "candidate_accepted": False,
            "baseline_accepted": False,
            "candidate_unit_net_pnl": 0.0,
            "baseline_unit_net_pnl": 0.0,
            "paired_delta_unit_net_pnl": 0.0,
            "requested_route": "low_vol",
        }
    ]
    panel = _group_panel(rows, lambda row: row["requested_route"])
    assert panel["low_vol"]["market_count"] == 1
    assert panel["low_vol"]["candidate_total_unit_net_pnl"] == 0.0


def test_frozen_gate_operators_evaluate_observed_values() -> None:
    assert _gate_passes(800, {"operator": "eq", "value": 800})
    assert _gate_passes(0, {"operator": "eq", "value": 0})
    assert not _gate_passes(1, {"operator": "eq", "value": 0})
    assert _gate_passes(0.951, {"operator": "gte", "value": 0.95})
    assert not _gate_passes(0.0, {"operator": "gt", "value": 0.0})


def test_json_and_markdown_sidecars_generate_and_verify_independently(tmp_path) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    json_artifact = _write_new_frozen_json(json_path, {"passed": False})
    markdown_artifact = _write_new_frozen_text(markdown_path, "# Report\n")

    assert _artifact_sidecar_path(json_path) == tmp_path / "report.sha256"
    assert _artifact_sidecar_path(markdown_path) == tmp_path / "report.md.sha256"
    assert json_artifact["sha256"] != markdown_artifact["sha256"]
    assert _verified_path(json_path)["sha256"] == json_artifact["sha256"]
    assert _verified_path(markdown_path)["sha256"] == markdown_artifact["sha256"]
