from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_issue_github_round_comments.py"

spec = importlib.util.spec_from_file_location("generate_issue_github_round_comments", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_pm_reduce_comment_uses_realized_delta_not_cumulative(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    log_path.write_text(
        json.dumps(
            {
                "ts": "2026-06-10T05:18:19.806000+00:00",
                "event": "paper_v7_settlement_position_reduced",
                "position": {
                    "round_slug": "btc-updown-15m-1781068500",
                    "side": "DOWN",
                    "entry_signal_event_id": "pred-abcdef123456",
                },
                "evaluation": {
                    "action": "REDUCE",
                    "reason": "residual_divergence_reduce",
                },
                "hold_bid": 0.49,
                "realized_pnl_delta": -0.0839,
                "cumulative_position_realized_pnl": -0.2551,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run = module.load_run(
        run_id="test-run",
        description="test",
        log_path=log_path,
        summary_path=summary_path,
        gamma_path=None,
    )

    row = run.pm_events_by_round["btc-updown-15m-1781068500"][0]
    assert row.exit_type == "pm_reduce"
    assert row.pnl == -0.0839
