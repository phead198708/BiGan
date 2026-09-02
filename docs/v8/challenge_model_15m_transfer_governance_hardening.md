# BTC 15m transfer governance hardening

Reviewed commit: `9bfca2f77555eef4980e9fb0f9fb300c3a6bc73e`

This review changed governance and artifact loading only. It did not start
training, collect or finalize a new real outcome, change model decisions,
change thresholds, consume transfer evidence, or unlock any safety capability.

## Governance report

The collector, outcome finalizer, and governance monitor were stopped at 32
attempts and 31 quality-valid finalized markets. The 40-market transfer trigger
has not been reached, so no real transfer report or freeze was produced or
consumed.

The v8.1 panel is explicitly marked `bridge_handicapped_not_native=true`. It
reports native feature coverage, native action-score coverage, bridge
limitations, accepted count, mean unit net PnL, a bootstrap interval, and weak
evidence status. It cannot emit a model-retraining mandate, a policy ranking,
or another strong transfer conclusion.

Legacy `HOLD_TO_SETTLEMENT` and bridge `SELL_BEFORE_CLOSE` results remain in
independent panels. Cross-policy information is diagnostic only.

## Portability report

The legacy v7 artifact graph now uses a single content-addressed bundle:

```text
sha256/8b7fc96af7f23c1e658cb816fc7d58028c71a9d530d2193bad9b2088547ffbcc/
├── bundle_manifest.json
├── model.json
├── settlement_model.json
└── settlement_residual_model.json
```

Dependency graph:

```text
repository registry
└── legacy v7 bundle manifest
    └── model.json
        ├── settlement.path ──────────> settlement_model.json
        └── settlement_residual.path ─> settlement_residual_model.json
```

Paths resolve from an explicit repository root and dependencies resolve from
the verified bundle directory, never from the process working directory. A
fresh-clone simulation verified the registry, bundle identity, all member
hashes, metadata, both XGBoost models, and finite synthetic predictions.
Tampering with a member SHA fails closed.

## Readiness-consistency report

The readiness gate requires 100 quality-valid outcome-finalized markets. The
collector authorization permits at most 119 single-market attempts. Both sides
now declare their units explicitly, with at most one quality-valid market per
attempt:

```text
100 required markets <= 119 authorized single-market attempts
```

The gate is mathematically reachable. No authorization extension exists or is
required, and attempt 120 remains unauthorized.

## Feature-scaling audit

`late_window_pressure` is the only time-normalized feature in the v8.1 bridge.
It reads `feature_row.horizon_ms`; BTC-15M requires exactly 900 seconds. A
300-second horizon fails closed.

## Remaining known limitations

- Native action-score coverage is zero.
- BTC anchor direction is reconstructed from available causal 15m inputs.
- Side mid-price is a value-conditioning bridge, not a native model score.
- Pre-entry exposure state remains an explicitly documented neutral assumption.
- Bootstrap results are weak development evidence only.
- The real transfer diagnostic remains unrun while the lane is paused at 31/40.

BTC 15m transfer governance protocol is internally consistent.
