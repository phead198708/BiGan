# v8 Phase 3 - Differentiable PnL Optimization

Phase 3 directly optimizes a differentiable net-PnL objective after Phase 2 has
verified the accepted Phase 1.5 candidate artifact. It does not mutate Phase 0,
Phase 1, Phase 1.5, or Phase 2 artifacts.

## Flow

```text
accepted Phase 1.5 candidate artifact
  -> verify run_manifest.json and registry hashes through the Phase 2 loader
  -> verify supplied train/shadow split hashes before prediction
  -> run frozen candidate inference on train rows and shadow rows
  -> compute Phase 2 execution baseline on shadow rows
  -> optimize differentiable action overlay on train rows only
  -> evaluate optimized overlay on shadow rows as OOS
  -> run cost perturbation stress tests
  -> write Phase 3 optimization report
```

The public entrypoint is:

```python
run_phase3_optimization(candidate_artifact_dir, split, DifferentiablePnlOptimizationConfig(...))
```

## Optimization Boundary

Phase 3 is the first v8 phase where direct PnL optimization is allowed. The
direct PnL optimizer is a differentiable action overlay on top of the frozen
Phase 1.5 policy predictions. It does not retrain or rewrite the Phase 1.5
model.

Training and evaluation are separated:

- Phase 1.5 train rows are used for differentiable PnL optimization.
- Phase 1.5 shadow rows are reserved for Phase 3 OOS evaluation.
- Phase 2 execution metrics on the same shadow rows are the baseline.

This keeps Phase 3 from tuning on the same rows used to prove OOS acceptance.

## Loss

The optimized objective is:

```text
loss = -mean(net_return) + return_variance_penalty * variance(net_return)
```

The differentiable net return includes:

- filled action times shadow net return
- spread cost
- fee cost
- volatility slippage cost
- liquidity impact cost
- quadratic risk penalty
- smooth turnover penalty

Turnover uses a smooth absolute-value approximation so the objective remains
gradient-friendly.

## Report

`Phase3OptimizationReport` records:

- exact Phase 1.5 `run_id`
- Phase 1.5 dataset/split/model hashes
- Phase 2 shadow execution baseline metrics
- Phase 3 train metrics
- Phase 3 OOS metrics
- Sharpe and mean-net-return deltas over Phase 2
- cost stress metrics for configured slippage multipliers
- optimization trace with loss and gradient norm
- optimized action-head parameters
- acceptance criteria

Acceptance criteria include:

```text
phase1_5_candidate_verified
phase2_baseline_reported
direct_pnl_optimization
gradient_flow_verified
gradient_norms_finite
gradient_norms_below_limit
optimization_loss_decreased
train_metrics_finite
oos_metrics_finite
sharpe_improvement_over_phase2
stable_oos_performance
cost_perturbation_robust
```

Phase 3 fails closed when candidate provenance, split provenance, gradient
health, OOS performance, or cost-stress robustness does not meet configured
thresholds.
