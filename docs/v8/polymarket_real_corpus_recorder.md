# v8 Polymarket Real Corpus Recorder

The real corpus recorder writes Polymarket BTC UP/DOWN market facts directly into
the Phase 2 `raw_*` corpus contract. It is a read-only data collection path, not a
trading or model-promotion path.

## Safety Boundary

Allowed:

- Read public market metadata.
- Read public token order books and trades.
- Read configured BTC feature candles.
- Write local JSONL artifacts, manifests, reports, and hashes.
- Optionally run the Phase 2 corpus builder after capture.

Forbidden:

- Real orders.
- Wallet signing.
- Private keys.
- CLOB write APIs.
- Real capital.
- Automatic model training, promotion, or live deployment.

Every recorder artifact preserves:

- `paper_only=true`
- `capital_at_risk=false`
- `broker_exchange_write_enabled=false`
- `live_exchange_write_enabled=false`
- `polymarket_write_enabled=false`
- `wallet_signing_enabled=false`

## Command

Deterministic CI-safe smoke:

```bash
python examples/v8/record_polymarket_real_corpus.py \
  --run-id recorder-smoke \
  --output-dir /tmp/bigan-v8-real-corpus-recorder \
  --overwrite-existing
```

The command writes:

- `real_corpus_recorder_manifest.json`
- `real_corpus_recorder_report.json`
- `real_corpus_rejected_rows.jsonl`
- `raw/raw_polymarket_markets.jsonl`
- `raw/raw_polymarket_orderbooks.jsonl`
- `raw/raw_polymarket_trades.jsonl`
- `raw/raw_binance_btcusdt_klines.jsonl`
- `raw/raw_polymarket_resolutions.jsonl`
- `phase2_corpus/*` when the Phase 2 build succeeds

## Settlement Source Rule

Do not use Binance BTC prices as Polymarket settlement references unless the
specific market rule explicitly declares Binance as the official resolution
source. The current Phase 2 raw filename `raw_binance_btcusdt_klines.jsonl` is a
causal feature-candle input. It is not a settlement oracle.

For a market to become training eligible, the recorder must capture:

- UP/DOWN token ids.
- Complete executable UP/DOWN bid/ask samples.
- Causal BTC feature candles with close availability enforced.
- Verified resolution start/end prices.
- Official settlement reference source provenance from the market rule or
  resolution metadata.

Missing evidence is written to `real_corpus_rejected_rows.jsonl`; rejected markets
do not enter the Phase 2 builder.

## Reports

The manifest and report include raw artifact hashes, row counts, reject reason
counts, training eligibility, and whether a Phase 2 corpus was built. Mock smoke
runs set `deterministic_replay=true` and `real_historical_corpus_used=false`.
Real public-data collection should set `real_historical_corpus_used=true` once the
read-only API client is wired into the recorder seam.
