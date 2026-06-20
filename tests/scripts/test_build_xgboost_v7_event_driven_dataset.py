import importlib.util
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from bigan.features.low_latency import JsonlRawQueue

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build_xgboost_v7_event_driven_dataset.py"

spec = importlib.util.spec_from_file_location("build_xgboost_v7_event_driven_dataset", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_build_event_driven_v7_dataset_writes_subminute_labels(tmp_path: Path) -> None:
    queue = JsonlRawQueue(tmp_path / "raw.jsonl")
    round_start_s = 1_780_000_000
    round_slug = f"btc-updown-15m-{round_start_s}"
    start_ms = round_start_s * 1000
    token_ids = {"UP": "up-token", "DOWN": "down-token"}

    for offset_s in range(5, 125, 5):
        ts = start_ms + offset_s * 1000
        up_ask = 0.40 + (offset_s / 1000.0)
        down_ask = 0.62 - (offset_s / 1200.0)
        for side, ask in (("UP", up_ask), ("DOWN", down_ask)):
            bid = max(0.01, ask - 0.02)
            queue.append(
                "raw_top_of_book",
                {
                    "ts": ts,
                    "message_ts": ts,
                    "ingest_ts": ts,
                    "source": "polymarket",
                    "source_symbol": token_ids[side],
                    "source_market": "0xmarket",
                    "canonical_symbol": f"BTC-15M:{round_slug}:{side}",
                    "bid_price": bid,
                    "ask_price": ask,
                    "spread": ask - bid,
                },
                published_at_ms=ts,
            )
            for book_side, price in (("BID", bid), ("ASK", ask)):
                queue.append(
                    "raw_orderbook_snapshot",
                    {
                        "ts": ts,
                        "message_ts": ts,
                        "ingest_ts": ts,
                        "source": "polymarket",
                        "source_symbol": token_ids[side],
                        "source_market": "0xmarket",
                        "canonical_symbol": f"BTC-15M:{round_slug}:{side}",
                        "side": book_side,
                        "level": 0,
                        "price": price,
                        "size": 10.0,
                    },
                    published_at_ms=ts,
                )

    outcome_cache = tmp_path / "outcomes.json"
    outcome_cache.write_text(json.dumps({round_slug: "UP"}), encoding="utf-8")
    output_dir = tmp_path / "dataset"
    stats = module.build_event_driven_dataset(
        [queue.path],
        output_dir,
        bucket_seconds=5,
        raw_throttle_ms=1000,
        min_completeness_score=0.0,
        outcome_cache_path=outcome_cache,
    )

    assert stats.rows_written > 0
    train = pq.read_table(output_dir / "train.parquet")
    assert "entry_ask_price_up" in train.schema.names
    assert "entry_ask_price_down" in train.schema.names
    assert "hit_5c_before_loss_10c_up" in train.schema.names
    assert "loss_10c_before_hit_5c_down" in train.schema.names
    rows = train.to_pylist()
    assert all(int(row["feature_ts"]) % 5000 == 0 for row in rows)
    assert any(row["best_exit_price_up"] is not None for row in rows)
    assert any(row["hit_5c_before_loss_10c_up"] is not None for row in rows)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bucket_seconds"] == 5.0
