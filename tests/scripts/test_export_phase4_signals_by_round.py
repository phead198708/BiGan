from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "export_phase4_signals_by_round.py"

spec = importlib.util.spec_from_file_location("export_phase4_signals_by_round", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_load_phase4_signal_records_groups_by_round(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4-test.jsonl"
    summary_path = tmp_path / "phase4-test-summary.json"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "signal_batch_received",
                        "ts": "2026-06-03T10:00:00+00:00",
                        "signals": [
                            {
                                "event_id": "pred-a",
                                "round_slug": "btc-updown-15m-100",
                                "side": "UP",
                                "edge": 0.1,
                                "timestamps": {
                                    "event_ts": "2026-06-03T10:01:00Z",
                                    "signal_created_at": "2026-06-03T10:00:30Z",
                                },
                            },
                            {
                                "event_id": "pred-b",
                                "round_slug": "btc-updown-15m-200",
                                "side": "DOWN",
                                "edge": 0.2,
                                "timestamps": {
                                    "event_ts": "2026-06-03T10:02:00Z",
                                    "signal_created_at": "2026-06-03T10:01:30Z",
                                },
                            },
                        ],
                    }
                ),
                json.dumps(
                    {
                        "event": "entry_gate_evaluated",
                        "signal": {
                            "event_id": "pred-a",
                            "round_slug": "btc-updown-15m-100",
                            "outcome_side": "UP",
                            "p_up": 0.6,
                            "p_down": 0.3,
                            "edge": 0.1,
                        },
                        "gate_evaluation": {
                            "settlement_gate_passed": True,
                            "volatility_gate_passed": False,
                            "settlement_edge": 0.12,
                            "volatility_score": -0.03,
                        },
                        "worst_price": 0.55,
                        "seconds_to_expiry": 500.0,
                        "fresh_edge_at_worst": 0.12,
                    }
                ),
                json.dumps(
                    {
                        "event": "entry_skipped",
                        "sleeve": "volatility",
                        "reason": "volatility_gate_below_cost",
                        "signal": {"event_id": "pred-a", "outcome_side": "UP", "round_slug": "btc-updown-15m-100"},
                    }
                ),
                json.dumps(
                    {
                        "event": "paper_entry_filled",
                        "position": {
                            "entry_signal_event_id": "pred-b",
                            "round_slug": "btc-updown-15m-200",
                            "side": "DOWN",
                            "fill_price": 0.42,
                        },
                        "signal": {
                            "event_id": "pred-b",
                            "round_slug": "btc-updown-15m-200",
                            "outcome_side": "DOWN",
                            "p_up": 0.4,
                            "p_down": 0.55,
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "paper_settlement_resolved",
                        "settlement_result": "UP",
                        "realized_pnl": 0.5,
                        "position": {"entry_signal_event_id": "pred-b"},
                    }
                ),
                json.dumps(
                    {
                        "event": "phase4_summary",
                        "status": "LIFECYCLE_PASS",
                        "observed_round_count": 2,
                        "processed_event_count": 2,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "volatility_budget_balances": {
                    "btc-updown-15m-100": 1.0,
                    "btc-updown-15m-200": 1.0,
                },
                "observed_round_count": 2,
            }
        ),
        encoding="utf-8",
    )

    records, run_summary, _ = module.load_phase4_signal_records(log_path, summary_path=summary_path)
    assert len(records) == 2
    by_round: dict[str, list[module.SignalRecord]] = {}
    for row in records:
        by_round.setdefault(row.round_slug, []).append(row)

    round_a = next(row for row in records if row.event_id == "pred-a")
    assert round_a.settlement_gate_passed is True
    assert round_a.action == "skipped"
    assert "volatility:volatility_gate_below_cost" in round_a.skip_reasons

    round_b = next(row for row in records if row.event_id == "pred-b")
    assert round_b.action == "filled"
    assert round_b.fill_price == 0.42
    assert round_b.settlement_result == "UP"
    assert round_b.realized_pnl == 0.5

    markdown = module.render_markdown_report(
        run_summary=run_summary,
        filter_totals={},
        by_round=by_round,
        round_order=["btc-updown-15m-100", "btc-updown-15m-200"],
        observed=set(run_summary.observed_round_slugs),
    )
    assert "## Round `btc-updown-15m-100`" in markdown
    assert "`pred-a`" in markdown
    assert "## Round `btc-updown-15m-200`" in markdown
    assert "filled: `1`" in markdown or "filled: `1`" in markdown.replace("'", "`")
