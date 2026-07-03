from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "replay_v7_meta_confidence.py"


def test_meta_confidence_replay_writes_report(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    report_path = tmp_path / "report.md"
    output_path = tmp_path / "report.json"

    _write_fixture(train_path, run_id="train", start_round=1)
    _write_fixture(replay_path, run_id="replay", start_round=10)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train-jsonl",
            str(train_path),
            "--replay-jsonl",
            str(replay_path),
            "--min-bucket-size",
            "1",
            "--min-execution-edge-grid",
            "0.10",
            "--min-p-hit-5c-grid",
            "0",
            "--min-p-hit-10c-grid",
            "0",
            "--max-p-loss-10c-grid",
            "1.0",
            "--min-confidence-score-grid=-1.0",
            "--output-json-path",
            str(output_path),
            "--report-path",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["recommended"]["trade_count"] >= 1
    assert output_path.exists()
    assert "v7 meta-confidence replay" in report_path.read_text(encoding="utf-8")


def _write_fixture(path: Path, *, run_id: str, start_round: int) -> None:
    rows: list[dict[str, object]] = []
    for idx in range(4):
        round_end = 1_800_000_000 + (start_round + idx) * 900
        side = "UP" if idx % 2 == 0 else "DOWN"
        prices = [0.46, 0.54, 0.60] if idx != 1 else [0.50, 0.43, 0.39]
        for step, price in enumerate(prices):
            ts = round_end - 600 + step * 120
            signal = {
                "event_id": f"{run_id}-{idx}-{side}-{step}",
                "round_slug": f"btc-updown-15m-{round_end - 900}",
                "selected_side": side,
                "created_at": ts,
                "ts": ts,
                "round_end_ts": round_end,
                "polymarket_price": price,
                "model_probability": 0.72,
                "p_up": 0.64 if side == "UP" else 0.36,
                "p_down": 0.64 if side == "DOWN" else 0.36,
            }
            rows.append(
                {
                    "event": "entry_gate_evaluated",
                    "signal": signal,
                    "worst_price": price,
                    "seconds_to_expiry": round_end - ts,
                    "gate_evaluation": {"settlement_gate_passed": True},
                }
            )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
