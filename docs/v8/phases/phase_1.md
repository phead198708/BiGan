# v8 Phase 1 - Pure Policy Learning

Phase 1 learns a stable trading policy from Phase 0-correct datasets. It does
not train on direct PnL, Sharpe, profit factor, drawdown, or return objectives.
PnL is used only after inference in the shadow acceptance validator.

## Architecture

```
Phase 0 artifact gate
  -> causal feature rows
  -> cost-aware labels
  -> Phase 1 dataset adapter
  -> supervised or ranking XGBoost policy
  -> action, confidence, regime embedding
  -> execution simulator / shadow acceptance
```

Phase 1 depends on Phase 0 for data integrity, causal feature construction, and
cost-aware label correctness. The adapter calls `assert_phase0_artifact_ready`
before building examples, so failed Phase 0 manifests cannot enter policy
learning.

## Modules

- `bigan.v8.phase1.contracts`: immutable schemas for policy dataset config,
  training examples, policy datasets, XGBoost config, and policy predictions.
- `bigan.v8.phase1.dataset`: deterministic adapter from accepted Phase 0
  features and labels into Phase 1 training examples.
- `bigan.v8.phase1.model`: XGBoost v8 policy trainer and inference wrapper.
- `bigan.v8.phase1.validation`: shadow acceptance gates for Sharpe, action
  distribution, and monotonic PnL bucket behavior.

## Model Boundary

Allowed training objectives:

- `binary:logistic` supervised policy classification
- `rank:pairwise` ranking policy learning

Forbidden objective, eval, and selection metric tokens:

- `pnl`
- `profit`
- `sharpe`
- `sortino`
- `drawdown`
- `roi`
- `return`
- `realized`

The training manifest always records:

- `direct_pnl_optimization: false`
- source Phase 0 dataset hash and version
- Phase 1 policy dataset hash
- feature columns
- policy objective and model config

## Outputs

Each prediction emits:

- `action`: position size in `[0, max_position_size]`
- `confidence`: normalized certainty in `[0, 1]`
- `regime_embedding`: configured regime feature values
- `score`: raw model score or probability

## Validation Checklist

Phase 1 is acceptable only when:

- The consumed Phase 0 manifest passes the artifact gate.
- The policy dataset hash is deterministic.
- Label and cost fields are not present in policy features.
- The XGBoost objective is supervised or ranking only.
- Direct PnL optimization knobs are rejected.
- Shadow Sharpe is positive.
- The action distribution is stable and not collapsed.
- Mean shadow PnL is monotonic by action bucket.

## Failure Modes

- Action collapse: predictions are always flat or always max size.
- Regime overfitting: one regime dominates active exposure.
- Unstable turnover: adjacent actions swing excessively.
- Non-monotonic buckets: higher action buckets do not produce higher shadow PnL.
- PnL leakage into training: objective or model-selection config uses profit,
  Sharpe, return, or related direct PnL terms.
- Unsafe upstream data: Phase 0 artifact gate rejects the source manifest.
