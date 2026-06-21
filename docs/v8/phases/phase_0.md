# v8 Phase 0 - Data Correctness Firewall

## Objective

Phase 0 produces leakage-free, cost-aware, deterministic datasets for the v8
trading architecture. It is infrastructure, not model research: no model
training happens here, and every downstream policy or PnL result is invalid if
this layer fails.

## Architecture

Raw market rows flow through:

1. `bigan.v8.phase0.loader.MarketDataLoader`
2. `bigan.v8.phase0.alignment.TimeAlignmentEngine`
3. `bigan.v8.phase0.features.CausalFeatureBuilder`
4. `bigan.v8.phase0.labels.CostAwareLabelBuilder`
5. `bigan.v8.phase0.validation.IntegrityValidator`
6. `bigan.v8.phase0.pipeline.Phase0Pipeline`

Only the label builder may read future rows. Features must be built from
`ts <= decision_ts` and `available_at_ts <= decision_ts`, with per-feature
provenance persisted for validation.

## Data Contracts

The strict contracts live in `src/bigan/v8/phase0/contracts.py`:

- `MarketData`: event time, availability time, source, instrument, prices,
  spread/depth inputs, and liquidity.
- `FeatureVector`: decision timestamp, feature cutoff, max input timestamp,
  feature values, and per-column provenance.
- `Label`: future target timestamp, horizon, entry/exit price, gross return,
  spread cost, fee cost, volatility slippage, liquidity impact, total cost, and
  net return.

Arrow schemas are exported as `MARKET_DATA_SCHEMA`, `FEATURE_VECTOR_SCHEMA`, and
`LABEL_SCHEMA`.

## Cost Model

`TradingCostModel` includes:

- spread cost from entry and exit top-of-book spread
- explicit fee cost in bps
- volatility-scaled slippage
- square-root liquidity impact approximation
- slippage stress multipliers, including `1.2`, `1.5`, and `2.0`

Labels persist `net_return = gross_return - total_cost`.

## Hard Gates

`IntegrityValidator` rejects:

- feature inputs after `decision_ts`
- feature availability after `decision_ts`
- rolling-window provenance that crosses the feature cutoff
- cross-timeframe/future feature provenance
- feature names that indicate target/future/label leakage
- suspicious feature-to-future-return correlations
- label horizon violations
- label cost/math mismatches
- label/market-data entry and exit inconsistencies
- duplicate labels or non-monotonic target timestamps across horizons

## Tests

The mandatory Phase 0 tests are in `tests/v8/test_phase0.py`:

- leakage detection via feature-to-future-return correlation
- timestamp reversal/shuffle diagnostic
- slippage stress at `1.2`, `1.5`, and `2.0`
- label consistency across horizons
- strict timestamp and cross-timeframe leakage rejection
- deterministic dataset generation and schema checks

Run:

```bash
pytest tests/v8/test_phase0.py -q
```

## Acceptance Checklist

- zero detectable leakage
- strict feature causality enforced
- label correctness verified against market rows
- net returns include spread, fee, slippage, and liquidity impact
- deterministic dataset hash is stable across input row order
- no model training inside Phase 0

## Failure Modes

- lookahead feature reads
- data availability timestamp ignored
- higher-timeframe bar used before close
- target or label field included as a feature
- future-return correlation too high to trust
- horizon target timestamp mismatch
- gross/net return math drift
- understated trading costs
- duplicate or contradictory market rows
- non-reproducible dataset manifests
