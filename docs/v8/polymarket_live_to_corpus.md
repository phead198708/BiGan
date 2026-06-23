# v8 Polymarket data/live to Phase 2 Corpus

This converter audits historical `data/live` Polymarket signal observations and writes
raw inputs compatible with the v8 Phase 2 Polymarket BTC UP/DOWN corpus builder.

It is intentionally fail-closed. A historical signal row is an observation, not a
training label. Rows only become Phase 2 training corpus evidence when the converter
can verify all required market evidence:

- BTC UP/DOWN market slug and deterministic market window.
- UP and DOWN token IDs.
- Point-in-time executable UP and DOWN bid/ask quotes.
- Point-in-time BTC observations for causal features.
- Verified settlement reference prices for market start and market end from the
  official Polymarket market rule source.
- Explicit settlement reference source provenance from the Polymarket market
  rules or resolution metadata.

The converter never uses `selected_side`, `outcome_side`, `model_probability`,
`prob_up_15m`, `edge`, or realized paper PnL as labels. Settlement labels are built
only by the public Phase 2 corpus builder from reference prices and market rules.

Do not substitute public BTC feature prices for Polymarket settlement references
unless the specific Polymarket market rule names that provider as the official
resolution source. The existing `raw_binance_btcusdt_klines.jsonl` corpus
filename is a legacy causal BTC feature input contract; row-level `source`
records whether the data came from Coinbase, Kraken, or Binance.

## Command

```bash
python examples/v8/convert_data_live_to_polymarket_corpus.py \
  --input-path /Users/tcscoder/Workspaces/BiGan/data/live \
  --output-dir /tmp/bigan-v8-live-to-corpus-audit \
  --no-build-phase2-corpus \
  --overwrite-existing
```

To build the Phase 2 corpus after conversion, omit `--no-build-phase2-corpus`.
The default remains strict: midpoint price proxying is disabled. Use
`--allow-midpoint-price-proxy` only for explicit offline experiments; proxy output
should not be treated as production-grade executable-cost training evidence.

## Outputs

The output bundle contains:

- `live_signal_conversion_manifest.json`
- `live_signal_conversion_report.json`
- `live_signal_rejected_rows.jsonl`
- `raw/raw_polymarket_markets.jsonl`
- `raw/raw_polymarket_orderbooks.jsonl`
- `raw/raw_polymarket_trades.jsonl`
- `raw/raw_binance_btcusdt_klines.jsonl`
- `raw/raw_polymarket_resolutions.jsonl`
- `phase2_corpus/*` when Phase 2 build succeeds

Every artifact is deterministic for identical input files and config. Reports include
SHA-256 hashes, safety flags, accepted/rejected counts, and reject reason counts.

## Current Historical Signal Caveat

Existing v7-style `data/live` signal files generally contain model-side observations
such as market price, selected side, token IDs, and timestamps. They often do not
contain executable bid/ask snapshots, official settlement reference prices, or
settlement reference source provenance. Those rows are expected to be rejected by
this converter until the live recorder captures the missing evidence.

This is the desired behavior: rejected observations can guide recorder improvements,
but they must not silently become training labels.
