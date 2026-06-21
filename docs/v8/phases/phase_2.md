# v8 Phase 2 - Hybrid PnL-Aware Evaluation

Phase 2 evaluates economic behavior around a frozen accepted Phase 1.5 policy
candidate. It does not retrain the policy model and it does not introduce
direct differentiable PnL optimization.

## Flow

```text
accepted Phase 1.5 candidate artifact
  -> verify run_manifest.json and registry hashes
  -> verify training/shadow/split provenance
  -> load model.xgb as immutable policy input
  -> run shadow inference on the supplied temporal split
  -> apply execution fill/cost/risk/turnover overlay
  -> write Phase 2 execution PnL report
```

The public entrypoint is:

```python
run_phase2_evaluation(candidate_artifact_dir, split, Phase2EvaluationConfig(...))
```

## Required Phase 1.5 Input

The candidate artifact directory must contain:

```text
run_manifest.json
training_manifest.json
split_manifest.json
shadow_acceptance_report.json
model.xgb
```

Before any prediction runs, Phase 2 verifies:

```text
accepted = true
candidate_status = accepted
acceptance_report_passed = true
model.xgb exists
run_manifest.json exists
training_manifest.json exists
shadow_acceptance_report.json exists
split_manifest.json exists
split_provenance_verified = true
direct_pnl_optimization = false
shadow_return_used_for_training = false
model_sha256 recorded and matching
artifact paths recorded for model, policy dataset manifest, split manifest,
training manifest, and shadow acceptance report
training_manifest_sha256 recorded and matching
shadow_acceptance_report_sha256 recorded and matching
split_manifest_sha256 recorded and matching
policy_dataset_manifest_sha256 recorded and matching
run_manifest_canonical_sha256 recorded and matching
policy_dataset_hash recorded
split_hash recorded
train_dataset_hash recorded
shadow_dataset_hash recorded
```

Missing, rejected, hash-mismatched, or provenance-mismatched artifacts raise
`Phase2ArtifactError`. The runner also verifies that the supplied
`PolicyTrainShadowSplit` hashes match the candidate before calling
`predict_examples(...)`.

Phase 2 also requires Phase 1.5 shadow baseline metrics for comparison:

```text
metrics.shadow_sharpe
metrics.mean_shadow_return
metrics.row_count > 0
metrics.action_distribution.mean_abs_turnover
metrics.action_distribution.active_rate
```

Missing or non-finite baseline values are hard failures. Phase 2 comparison
metrics must never be computed from implicit default-zero Phase 1.5 metrics.

## Execution Overlay

Phase 2 applies a deterministic execution simulation:

- volatility/spread/liquidity cost estimate
- fill probability approximation from liquidity
- risk penalty from action size and volatility
- turnover penalty
- static lambda-weighted hybrid score for diagnostics
- optional cost-aware low-EV trade filter

When reconstructing execution rows from policy examples, `spread` is treated as
a price spread and used directly. If `spread` is absent and `spread_bps` is
present, Phase 2 converts basis points as:

```text
spread = spread_bps / 10_000 * mid_price
```

The filter uses only policy confidence and pre-trade cost/risk estimates. It
does not use realized future PnL to decide the action.

## Report

`Phase2EvaluationReport` keeps Phase 1.5 shadow acceptance metrics separate
from execution-adjusted PnL metrics.

It records:

- exact Phase 1.5 `run_id`
- Phase 1.5 dataset/split/model hashes
- Phase 1.5 shadow Sharpe and turnover
- Phase 2 execution Sharpe and turnover
- execution costs, risk penalties, net returns, and filter rate
- Sharpe improvement ratio
- turnover reduction ratio

Acceptance criteria are reported as booleans:

```text
phase1_5_candidate_verified
execution_adjusted_pnl_reported
execution_metrics_finite
sharpe_improvement_ge_min
reduced_turnover
cost_aware_behavior_emerged
```

Default acceptance requires at least 10% Sharpe improvement over the Phase 1.5
shadow acceptance baseline and non-negative turnover reduction. Callers may
adjust thresholds in `Phase2EvaluationConfig` for smoke tests or diagnostics.
