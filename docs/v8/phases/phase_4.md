# v8 Phase 4 - Online Adaptive System

Phase 4 adds a causal online adaptation layer on top of frozen policy
predictions. It does not retrain the Phase 1.5 model and does not tune the
Phase 3 optimizer. It replays the policy stream through adaptive controls that
must be safe under stress before downstream live execution may consume them.

## Flow

```text
policy examples + frozen policy predictions + input provenance
  -> verify non-decreasing decision timestamps
  -> verify prediction keys match examples before replay
  -> hash exact example and prediction streams
  -> verify upstream candidate/model/split provenance
  -> classify current market regime from causal features only
  -> update lambda from regime, prior drawdown, and current volatility
  -> update execution aggressiveness from cost and liquidity state
  -> evaluate adaptive net return after the decision is fixed
  -> run volatility stress replays
  -> write Phase 4 adaptive-system report
```

The public entrypoint is:

```python
run_phase4_adaptive_system(
    examples=shadow_examples,
    predictions=frozen_policy_predictions,
    provenance=Phase4InputProvenance(...),
    config=Phase4AdaptiveSystemConfig(...),
)
```

## Causal Boundary

Phase 4 makes online decisions from:

- current causal features on the policy example
- current frozen policy prediction
- prior adaptive state for the same source/instrument

`shadow_net_return` is used only after lambda and execution aggressiveness have
already been chosen. It is therefore an offline acceptance measurement, not an
input to adaptation.

The replay fails closed before adaptation when:

- the stream is empty
- examples and predictions have different lengths
- timestamps are not globally non-decreasing
- prediction keys do not match example keys

Missing or stale provenance is not promotion quality. The runner may still
produce a diagnostic report, but `input_provenance_verified=false` makes
`passed=false`. Policy dataset, split, model, report, example-stream, and
prediction-stream hashes must be canonical SHA-256 hex digests.

The helper below computes deterministic stream hashes from replay-order
payloads:

```python
build_phase4_input_provenance(
    examples=shadow_examples,
    predictions=frozen_policy_predictions,
    candidate_run_id=...,
    policy_dataset_hash=...,
    split_hash=...,
    model_sha256=...,
    phase2_report_sha256=...,
    phase3_report_sha256=...,
)
```

## Components

### Regime Detector

The detector emits `trend`, `range`, `high_volatility`, or `liquidity_stress`
from trend score, volatility, liquidity depth, and spread. Regime changes must
persist for `transition_confirmation_count` rows before the confirmed regime
changes, reducing noisy regime flicker.

The report records both raw detector transitions and confirmed regime
transitions:

```text
raw_regime_transition_count
raw_regime_transition_rate
confirmed_regime_transition_count
confirmed_regime_transition_rate
pending_regime_count
pending_regime_rate
raw_to_confirmed_transition_suppression_ratio
```

This prevents confirmation smoothing from hiding a noisy raw detector.

### Lambda Controller

`lambda` is a bounded and smoothed risk-appetite scalar:

```text
lambda = base_lambda
  * regime_multiplier
  * volatility_dampener
  * drawdown_dampener
```

Trend regimes can increase risk appetite. High-volatility, liquidity-stress,
and drawdown states reduce it. Step limits and stress replay guard against
lambda oscillation.

### Execution Adaptation

Execution aggressiveness is bounded and smoothed separately from lambda. It
uses the same Phase 0 `TradingCostModel` cost basis for spread, fees,
volatility slippage, and liquidity impact. High cost, weak liquidity, and high
volatility reduce filled size.

## Report

`Phase4AdaptiveSystemReport` records:

- candidate run id
- policy dataset, split, model, Phase 2, and Phase 3 hashes
- exact example and prediction stream hashes
- provenance verification result
- baseline identity and baseline execution-config hash
- decision trace hash, count, and timestamp span
- adaptive metrics
- baseline non-adaptive metrics
- tail-risk comparison
- regime counts
- volatility stress metrics
- acceptance criteria
- full config snapshot

Acceptance criteria include:

```text
causal_stream_ordered
prediction_keys_match_examples
input_provenance_verified
regime_detector_active
raw_regime_flicker_bounded
confirmed_regime_transitions_stable
regime_transitions_stable
lambda_values_bounded
lambda_stability
lambda_stability_under_stress
execution_adaptation_stable
tail_risk_performance_improved
adaptive_metrics_finite
comparison_metrics_finite
stress_metrics_finite
```

Phase 4 fails closed when regime transitions flicker, lambda oscillates,
execution aggressiveness jumps beyond configured bounds, or adaptive sizing
does not improve tail-loss performance versus the non-adaptive baseline.
