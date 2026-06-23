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
Real public-data collection sets `real_historical_corpus_used=true` only when a
configured read-only provider returns accepted Polymarket rows, accepted BTC
feature candles, and the Phase 2 corpus build succeeds.

Eligibility is intentionally split:

- `phase2_corpus_build_eligible`: the raw bundle can be normalized by the Phase 2
  corpus builder.
- `real_historical_training_eligible`: the bundle came from real public data and
  may be used by the real-history training gate.
- `manual_live_evidence_eligible`: the bundle is eligible to unblock downstream
  manual live evidence.

Mocked recorder output may set `phase2_corpus_build_eligible=true` for smoke
coverage, but it must keep:

- `mock_public_data_used=true`
- `synthetic_public_data_used=true`
- `synthetic_corpus_used=true`
- `real_historical_training_eligible=false`
- `manual_live_evidence_eligible=false`

When `--no-mock-public-data` is used before the real read-only providers are
wired, the recorder fails closed. It still writes the run bundle and empty raw
files, and `real_corpus_rejected_rows.jsonl` includes provider-level reasons such
as `real_public_collection_not_configured`.

## Real Provider Seam

The operator supports an explicit `public_provider=` argument for non-mock runs.
That provider must implement the `PolymarketRealCorpusPublicProvider` contract and
return normalized raw rows for:

- Polymarket Gamma market discovery.
- Polymarket CLOB orderbook collection.
- Polymarket CLOB trade collection.
- BTC feature candle collection.
- Official Polymarket resolution/reference collection.

The default CLI does not auto-connect to public APIs. Without a configured
provider, `mock_public_data=false` remains fail-closed by design. With a
configured provider, any fetch or normalization exception is written to
`real_corpus_rejected_rows.jsonl` with provider/stage/reason details and the
real-history training gate remains closed.

Granular read flags are separate from training eligibility:

- `live_polymarket_data_read`: accepted real Polymarket market/orderbook rows
  were written.
- `live_btc_reference_data_read`: accepted real BTC feature candle rows were
  written.
- `real_historical_training_eligible`: the accepted real raw bundle produced a
  Phase 2 corpus and had no provider-stage failures.
