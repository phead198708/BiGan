# Bootstrap Champion Runbook

Owner: ML platform  
Scope: first production champion for BTC 15-minute direction probabilities

Use this runbook only while there is no incumbent production ML champion. After
the first champion is live, use the normal champion/challenger process.

## Required Evidence

Before running a bootstrap decision, collect:

- Explicit baseline model directory with `manifest.json`, `metrics.json`, and `model.json`.
- Candidate model directory with `manifest.json`, `metrics.json`, `model.json`, and `feature_schema.json`.
- Candidate `calibration_report.json` with calibrated Brier and ECE.
- Candidate and baseline edge-threshold backtest summaries with fees, slippage, drawdown, Sharpe or Sortino, turnover, trade count, and net PnL.
- Serving readiness JSON with p95 latency, error rate, and pass/fail limits.
- Rollback/fallback runbook and a loadable fallback baseline artifact.

## Initial Candidate Iteration

Train the conservative v2 XGBoost candidate with:

```bash
bigan-ingest xgboost-v2 \
  --dataset-dir data/training-datasets/bigan-training-15m-v1 \
  --output-dir data/model-runs/xgboost-v2
```

Then fit calibration against the saved v2 model:

```bash
bigan-ingest calibration-v1 \
  --model-path data/model-runs/xgboost-v2/model.json \
  --dataset-dir data/training-datasets/bigan-training-15m-v1 \
  --output-dir data/model-runs/xgboost-v2-calibration
```

## Backtest Sanity Gate

Before trusting any candidate backtest, run the perfect-label oracle check:

```bash
bigan-ingest backtest-oracle-sanity-v1 \
  --dataset-dir data/training-datasets/bigan-training-15m-v1 \
  --warehouse-dir data/warehouse \
  --output-dir data/backtests/oracle-label-sanity-v1
```

This check turns `label_profit_up_15m` into a perfect trade-profitability probability. If it still has
zero win rate, negative net PnL, or no trades before costs, do not promote any
candidate. Fix the label economics, UP/DOWN token mapping, or backtest price
alignment first.

The default oracle check requires `canonical_symbol` to end in `:UP`. Missing
UP mappings are a hard failure, because `prob_up_15m` must not be backtested
against a mixed set of UP and DOWN outcome tokens.

Candidate and baseline prediction backtests should use the same UP-only gate:

```bash
bigan-ingest backtest-predictions-v1 \
  --warehouse-dir data/warehouse \
  --model-version xgboost-v1 \
  --thresholds 0.00,0.03,0.05 \
  --output-dir data/backtests/xgboost-v1-up-only
```

The strategy buys only when `prob_up_15m - market_implied_prob >= edge_threshold`.
When a prediction row does not provide `market_implied_prob`, the backtest uses
the executable UP-token ask price at entry time as the conservative market
implied probability.

## Command

```bash
bigan-ingest bootstrap-champion-v1 \
  --baseline-dir data/model-runs/logreg-baseline-v1 \
  --baseline-backtest-summary-path data/backtests/logreg-baseline-v1/summary.json \
  --candidate-dir data/model-runs/xgboost-v1 \
  --calibration-dir data/model-runs/xgboost-v1-calibration \
  --candidate-backtest-summary-path data/backtests/xgboost-v1/summary.json \
  --serving-readiness-path data/model-runs/xgboost-v1/serving_readiness.json \
  --output-dir data/model-runs/bootstrap-champion-v1
```

The command writes:

- `bootstrap_decision.json`
- `bootstrap_decision.md`

## Decision Rules

- `PROMOTE_FIRST_CHAMPION:<model_version>` means the candidate is good enough,
  safe enough, and simple enough for v1 production. It does not mean it is the
  globally best possible model.
- `KEEP_BASELINE_TEMPORARILY` means the best candidate has explicit hard-gate
  failures, such as unacceptable backtest utility or invalid evaluation.
- `CONTINUE_BOOTSTRAP_EXPERIMENTATION` means the candidate may be promising, but
  critical evidence is incomplete.

Do not promote a candidate when any bootstrap checklist item is unchecked.
