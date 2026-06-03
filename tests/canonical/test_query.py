from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from bigan.canonical.query import open_warehouse


def test_open_warehouse_skips_incomplete_parquet_files(tmp_path):
    partition = tmp_path / "raw_top_of_book" / "source=polymarket" / "dt=2026-06-03"
    partition.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "ts": 1,
                    "source_symbol": "token-up",
                    "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                    "bid_price": 0.49,
                    "ask_price": 0.51,
                }
            ]
        ),
        partition / "part-good.parquet",
    )
    (partition / "part-half-written.parquet").write_bytes(b"PAR1broken")

    with open_warehouse(tmp_path) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM raw_top_of_book").fetchone()[0]

    assert row_count == 1
