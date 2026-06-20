import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT = SCRIPTS_DIR / "build_xgboost_v7_event_driven_dataset_from_predictions.py"
RAW_BUILDER_SCRIPT = SCRIPTS_DIR / "build_xgboost_v7_event_driven_dataset.py"

raw_spec = importlib.util.spec_from_file_location(
    "build_xgboost_v7_event_driven_dataset",
    RAW_BUILDER_SCRIPT,
)
assert raw_spec is not None
raw_module = importlib.util.module_from_spec(raw_spec)
sys.modules[raw_spec.name] = raw_module
assert raw_spec.loader is not None
raw_spec.loader.exec_module(raw_module)

spec = importlib.util.spec_from_file_location(
    "build_xgboost_v7_event_driven_dataset_from_predictions",
    SCRIPT,
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_build_event_driven_dataset_from_prediction_warehouse(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    output_dir = tmp_path / "dataset"
    rows = _prediction_rows()
    pq.write_table(pa.Table.from_pylist(rows), prediction_path)

    stats = module.build_event_driven_dataset_from_predictions(
        prediction_path,
        output_dir,
        bucket_seconds=5,
        min_completeness_score=0.0,
    )

    assert stats.prediction_rows_read == len(rows)
    assert stats.feature_rows_generated > 0
    assert stats.rows_written > 0
    assert stats.round_count == 5
    assert stats.outcome_counts == {"UP": 3, "DOWN": 2}

    train = pq.read_table(output_dir / "train.parquet")
    assert "entry_ask_price_up" in train.schema.names
    assert "entry_ask_price_down" in train.schema.names
    assert "best_exit_price_up" in train.schema.names
    train_rows = train.to_pylist()
    assert all(int(row["feature_ts"]) % 5000 == 0 for row in train_rows)
    assert {row["label_settlement_3way"] for row in train_rows} <= {"UP", "DOWN"}
    assert any(row["best_exit_price_up"] is not None for row in train_rows)
    assert any(row["best_exit_price_down"] is not None for row in train_rows)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "prediction_warehouse"
    assert manifest["bucket_seconds"] == 5.0
    assert manifest["build"]["prediction_rows_read"] == len(rows)


def _prediction_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    outcomes = ["UP", "DOWN", "UP", "DOWN", "UP"]
    base_round_start_s = 1_780_000_000
    for round_idx, outcome in enumerate(outcomes):
        round_start_s = base_round_start_s + round_idx * 900
        round_slug = f"btc-updown-15m-{round_start_s}"
        start_ms = round_start_s * 1000
        for offset_s in range(5, 605, 5):
            progress = offset_s / 600.0
            winner_prob = min(0.98, 0.42 + 0.56 * progress)
            loser_prob = max(0.02, 1.0 - winner_prob)
            side_probs = (
                {"UP": winner_prob, "DOWN": loser_prob}
                if outcome == "UP"
                else {"UP": loser_prob, "DOWN": winner_prob}
            )
            for side in ("UP", "DOWN"):
                ts = start_ms + offset_s * 1000
                market_implied_prob = side_probs[side]
                features = _feature_values(market_implied_prob)
                rows.append(
                    {
                        "canonical_symbol": f"BTC-15M:{round_slug}:{side}",
                        "ts": ts,
                        "message_ts": ts,
                        "prediction_ts": ts,
                        "ingest_ts": ts,
                        "source": "polymarket",
                        "source_symbol": f"{round_slug}-{side.lower()}",
                        "source_market": f"market-{round_idx}",
                        "symbol": f"BTC-15M:{round_slug}:{side}",
                        "feature_version": "bigan-mvp-v1.0.0",
                        "market_implied_prob": market_implied_prob,
                        "feature_values_json": json.dumps(features, sort_keys=True),
                    }
                )
    return rows


def _feature_values(market_implied_prob: float) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for column in module._feature_columns():
        values[column] = 0 if column in {"trade_count_1m", "day_of_week"} else 0.0
    values.update(
        {
            "spread": 0.02,
            "tick_spread": 0.02,
            "market_implied_prob": market_implied_prob,
            "mid_price": market_implied_prob,
            "microprice": market_implied_prob,
            "tick_mid_price": market_implied_prob,
            "minute_of_day": 0.5,
            "horizon_minutes": 15.0,
            "liquidity_bucket": 1.0,
        }
    )
    return values
