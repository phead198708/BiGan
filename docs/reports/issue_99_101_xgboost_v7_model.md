# xgboost-v7 Model Implementation Notes

## Scope

This change implements the first xgboost-v7 model artifact for issues #99 and #101.

v7 is settlement-only in this iteration. Volatility signals are intentionally excluded from the v7 promotion decision because the current strategy work is focused on settlement execution quality, signal latency, and price-aware entry.

## Issue #99 Mapping

The v7 model keeps the calibrated settlement probability head and adds a settlement residual head.

- Outcome head: `p_up`, `p_down`, `p_neutral`
- Residual target: `settlement_tradable_edge = win - market_implied_prob`
- Executable price formula: `entry_worst = min(0.99, ask + buy_slippage + ask * fee_bps / 10000)`
- Analytical EV fields: `expected_edge_up/down = p_side - entry_worst_price_side`
- Residual EV fields: `residual_expected_edge_up/down`
- Metric of record: `executable_one_way_settlement_pnl`

The report keeps outcome metrics separate from tradable EV metrics, so raw hit rate cannot promote a model by itself.

## Issue #101 Mapping

Issue #101 exposed that live settlement evidence can be misleading while settlement or cashflow reconciliation is pending. The v7 artifact writes an executor integration contract that keeps those checks in the executor layer:

- signal age
- price freshness
- account and funder balance
- max concurrent positions
- one settlement bet per round
- daily loss limit
- expiry and no-new-entry windows
- cashflow reconciliation

The artifact also records the guardrail:

> Do not use live run PnL as promotion evidence while pending settlement or account reconciliation is unresolved.

## Generated Artifacts

`train_xgboost_v7` writes:

- `model.json`
- `settlement_model.json`
- `settlement_residual_model.json`
- `xgboost_v7_config.json`
- `metrics.json`
- `outcome_metrics.json`
- `family_outcome_metrics.json`
- `residual_metrics.json`
- `tradable_ev_metrics.json`
- `feature_schema.json`
- `executor_integration.md`

The CLI entry point is:

```bash
python -m bigan.ingestion xgboost-v7 \
  --dataset-dir data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/dataset \
  --output-dir data/model-runs/xgboost-v7/<run-id>
```

The wrapper script is:

```bash
bash scripts/run_xgboost_v7_training.sh
```

Serving support is wired through `run_prediction_batch`: v7 artifacts are detected
from `model.json`, scored without external calibration, and written with
settlement residual, executable edge, selected side, and entry-worst fields.
The direct signal JSONL queue path preserves those v7 fields so the executor can
apply v7-specific gates later without rereading DuckDB.

## Smoke Check

Before adding the trainer, the evaluation-function smoke test reused existing v6 probabilities to test whether the proposed v7 metric can select a rule that survives holdout.

- Report: `docs/reports/xgboost_v7_settlement_eval_smoke_20260606T120255Z.md`
- Result: `V7_EVAL_OVERFITS_VALIDATION`
- Validation-selected rule: `min_confidence=0.85`, `min_expected_edge=0.0`
- Validation PnL: `+1.29`
- Test PnL: `-3.55`

Interpretation: the metric is useful diagnostically, but current data did not prove generalization. The trainer is therefore built to report promotion evidence conservatively, not to assume v7 is better than v6.

## Acceptance

- v7 artifacts include both settlement outcome and residual heads.
- Serving payload includes `expected_edge_*` fields.
- Training report separates outcome accuracy, residual fit, and executable one-way PnL.
- Tradable EV metrics include round-age and entry-price buckets.
- Executor contract states v7 settlement-only behavior and issue #101 reconciliation guardrail.
- Tests compile/train a tiny v7 artifact and validate core formulas.
