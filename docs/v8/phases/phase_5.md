# v8 Phase 5 - Safety Layer, Shadow Mode, and Rollback

Phase 5 is the fail-closed safety layer for the v8 trading stack. It consumes
Phase 4 shadow decisions and paired live execution observations, then decides
whether trading can continue, whether model updates must freeze, and whether
the system must roll back to the last stable model snapshot.

It does not train a model and does not optimize PnL. Its job is degradation
detection and operational containment.

## Flow

```text
Phase 4 adaptive shadow decisions + live execution observations
  -> verify streams are non-empty, aligned, and ordered
  -> build shadow/live audit records
  -> assert shadow mode has no capital risk
  -> compute rolling PnL drift, cost drift, regime mismatch, and correlation
  -> detect silent degradation before drawdown breach
  -> trigger kill-switch when degradation or drawdown breach is detected
  -> optionally flatten positions
  -> freeze model updates
  -> restore last stable model parameters
  -> write Phase 5 safety-layer report
```

The public entrypoint is:

```python
run_phase5_safety_layer(
    shadow_decisions=phase4_decisions,
    live_observations=live_observations,
    stable_model=StableModelSnapshot(...),
    config=SafetyLayerConfig(...),
)
```

## Shadow Mode

Phase 5 treats the Phase 4 adaptive decisions as the shadow simulation stream.
Each shadow record is paired with a live observation at the same
`decision_ts/source/instrument_id`. Shadow records are explicitly marked as
`shadow_capital_at_risk=false`; if a shadow record carries capital risk, the
contract fails closed.

The report records:

```text
parallel_streams_aligned
shadow_mode_enabled
shadow_capital_risk_free
live_capital_at_risk
full_simulation_pipeline
```

## Silent Failure Detection

Phase 5 monitors rolling windows for:

- PnL drift
- execution-cost drift
- regime mismatch
- shadow/live correlation break

The first rolling window that breaches any threshold emits a degradation
timestamp and reason codes. That timestamp is compared with the first live
drawdown breach to prove degradation was detected before drawdown.

## Kill-switch

If degradation or live drawdown breach is detected, Phase 5 emits a
`SafetyAction` with:

```text
kill_switch_triggered=true
stop_trading=true
flatten_positions=<config>
freeze_model_updates=<config>
rollback_model_id=<last stable model>
restored_safe_parameters=<last stable parameters>
reason_codes=[...]
triggered_at_ts=<first degradation or drawdown timestamp>
```

When no degradation is present, the report must keep the kill-switch inactive.

## Rollback

Rollback targets are represented by `StableModelSnapshot`, which includes:

```text
model_id
model_sha256
policy_dataset_hash
split_hash
safe_parameter_sha256
safe_parameters
```

All snapshot hashes must be canonical SHA-256 hex digests. On kill-switch,
the safety action must restore the exact safe parameter payload from the stable
snapshot.

## Acceptance Criteria

The report acceptance criteria include:

```text
shadow_mode_parallel
shadow_mode_no_capital_risk
full_simulation_pipeline
pnl_drift_monitored
cost_drift_monitored
regime_mismatch_monitored
shadow_live_correlation_stable_or_kill_triggered
degradation_detected_before_drawdown
kill_switch_reliable
rollback_executes_reliably
safe_when_no_degradation
metrics_finite
```

Phase 5 is valid when normal shadow/live streams remain unblocked, degraded
streams trigger the kill-switch before drawdown, and rollback restores the last
stable model configuration exactly.
