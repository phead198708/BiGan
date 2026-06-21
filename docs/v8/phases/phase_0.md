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
7. `bigan.v8.phase0.artifacts.Phase0ArtifactGate`

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
- `DatasetContract`: deterministic dataset hash, market schema, feature schema,
  label/cost schema, and artifact metadata.

Arrow schemas are exported as `MARKET_DATA_SCHEMA`, `FEATURE_VECTOR_SCHEMA`, and
`LABEL_SCHEMA`. Downstream phases must reject artifacts whose
`dataset_contract` does not match these schemas.

`DatasetContract` validates required column sets separately from canonical
ordering. Runtime consumers can require canonical ordering for production
artifacts while still receiving precise failure reasons for missing columns,
ordering drift, or hash/version mismatches.

## Cost Model

`TradingCostModel` includes:

- spread cost from entry and exit top-of-book spread
- explicit fee cost in bps
- volatility-scaled slippage
- square-root liquidity impact approximation
- slippage stress multipliers, including `1.2`, `1.5`, and `2.0`
- real-execution calibration via `ExecutionCostSample` and
  `TradingCostModel.validate_calibration`
- regime-aware calibration via `validate_calibration_by_bucket`, with buckets
  for source, instrument, volatility, spread, liquidity, and order size
- robust low-cost diagnostics: weighted MAPE, median AE, median APE, and
  symmetric MAPE

Labels persist `net_return = gross_return - total_cost`.

## Hard Gates

`IntegrityValidator` rejects:

- feature inputs after `decision_ts`
- feature availability after `decision_ts`
- paired feature/label rows where `max(feature_timestamp) > label_start_time`
- unsafe runtime artifacts via `assert_phase0_artifact_ready(manifest)`
- rolling-window provenance that crosses the feature cutoff
- cross-timeframe/future feature provenance
- feature names that indicate target/future/label leakage
- suspicious feature-to-future-return correlations
- temporal feature-distribution drift via KS, PSI, and KL divergence
- label horizon violations
- label cost/math mismatches
- label/market-data entry and exit inconsistencies
- duplicate labels or non-monotonic target timestamps across horizons

Drift checks have explicit policy:

- `statistical_integrity_mode="fail"` for production hard failure
- `statistical_integrity_mode="warn"` for diagnostic-only review
- `fail_on_insufficient_drift_rows=True` when too-little data should block
  artifact consumption

The artifact gate rejects manifests with missing contracts, stale dataset
versions, hash mismatches, schema/cost-column mismatches, failed validation,
missing acceptance criteria, or failed mandatory criteria.

## Tests

The mandatory Phase 0 tests are in `tests/v8/test_phase0.py`:

- leakage detection via feature-to-future-return correlation
- timestamp reversal/shuffle diagnostic
- slippage stress at `1.2`, `1.5`, and `2.0`
- label consistency across horizons
- strict timestamp and cross-timeframe leakage rejection
- deterministic dataset generation and schema checks
- runtime feature/label causality checks
- runtime artifact gate acceptance/rejection
- cost calibration against observed execution-cost samples
- bucketed cost calibration pass/fail
- robust low-cost calibration metrics
- KS/PSI/KL distribution drift detection
- drift warning/fail/insufficient-row policy

Run:

```bash
pytest tests/v8/test_phase0.py -q
```

CI hard gates live in `.github/workflows/v8-phase0.yml` and run ruff plus the
Phase 0 pytest suite for v8 changes.

## Acceptance Checklist

- zero detectable leakage
- strict feature causality enforced
- label correctness verified against market rows
- feature/label causality verified with `max(feature_timestamp) <= label_start_time`
- runtime artifact gate passes before downstream consumption
- net returns include spread, fee, slippage, and liquidity impact
- cost model calibrated against observed execution samples before production use
- sufficiently sampled cost regimes pass bucketed calibration
- KS/PSI/KL statistical integrity checks pass
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
