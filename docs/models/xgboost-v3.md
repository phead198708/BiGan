# XGBoost-v3 Model Card

Status: initial challenger evidence for bootstrap champion review  
Model version: `xgboost-v3`  
Training data version: `bigan-training-15m-profitability-v1.0.0`  
Feature version: `bigan-mvp-v1.0.0`  
Serving readiness report: `docs/models/xgboost-v3-serving-readiness.json`

## Intended Use

`xgboost-v3` estimates `prob_up_15m`, the probability that buying the UP token is profitable after comparing the model probability with the current market-implied probability. The label definition is UP token profitability: `label_profit_up_15m` is positive when the resolved UP-token payoff exceeds the entry ask cost after the configured trading economics.

This model is intended as a simple, monitored v1 champion candidate, not as evidence that the globally best model has been found.

## Features

The model uses 17 numeric features in the exact order saved in `feature_schema.json`:

| # | Feature |
|---:|---|
| 1 | `spread` |
| 2 | `market_implied_prob` |
| 3 | `mid_price` |
| 4 | `microprice` |
| 5 | `obi_l1` |
| 6 | `obi_l5` |
| 7 | `obi_l10` |
| 8 | `signed_volume_1m` |
| 9 | `trade_imbalance_1m` |
| 10 | `trade_count_1m` |
| 11 | `trade_volume_1m` |
| 12 | `ret_1m` |
| 13 | `ret_5m` |
| 14 | `ret_15m` |
| 15 | `rv_1m` |
| 16 | `rv_5m` |
| 17 | `rv_15m` |

## Training Configuration

Best parameters:

| Parameter | Value |
|---|---:|
| `rounds` | 200 |
| `eta` | 0.05 |
| `lambda` | 5.0 |
| `max_depth` | 4 |
| `min_child_weight` | 2.0 |
| `subsample` | 0.8 |
| `colsample_bytree` | 1.0 |
| `tree_method` | `hist` |
| `nthread` | 1 |

Dependencies: Python 3.12, `xgboost`, `pyarrow`, `duckdb`, and the local `bigan` training/serving modules.

Training cost: the v3 grid contains 648 XGBoost candidates and uses single-threaded XGBoost training (`nthread=1`) for deterministic local runs. The original rerun did not record wall-clock training time, so the next training pipeline should persist elapsed seconds and host metadata as a first-class artifact.

Retraining: rebuild the profitability-labeled training dataset, rerun `bigan-ingest xgboost-v3`, rerun calibration, regenerate predictions, rerun edge-based settlement backtests, and rerun this serving readiness check before any promotion decision.

## Offline Metrics

| Split | AUC | Brier | PR AUC | ECE | Accuracy |
|---|---:|---:|---:|---:|---:|
| train | 0.9508 | 0.0906 | 0.9640 | 0.0545 | 0.8624 |
| val | 0.9585 | 0.0882 | 0.9626 | 0.1258 | 0.9135 |
| test | 0.8503 | 0.1331 | 0.9223 | 0.1304 | 0.7979 |

Platt-calibrated test metrics from the rerun: Brier `0.1235`, ECE `0.0927`, AUC `0.8503`, PR AUC `0.9223`.

## Data Regime

The profitability-labeled split is time ordered and the held-out test slice is unusually friendly to UP-token profitability:

| Split | Time Range (Asia/Shanghai) | Rows | Positive Rate |
|---|---|---:|---:|
| train | 2026-05-19 08:15 to 2026-05-19 23:19 | 7806 | 55.74% |
| val | 2026-05-19 23:19 to 2026-05-20 01:47 | 2602 | 57.84% |
| test | 2026-05-20 01:47 to 2026-05-20 08:36 | 2603 | 72.22% |

## Known Behavior

- v3 behaves well in the current data regime. The held-out test window has a high UP-token profitability positive rate, and the model's strong test metrics/backtest utility should be read as evidence that v3 is well matched to this observed regime, not proof that it is robust across all future regimes.
- v3 may be more fragile in a reverse regime where UP-token profitability deteriorates or the market-implied probability becomes less exploitable. After deployment, the two highest-priority live monitors are prediction distribution drift and label hit rate once outcomes settle.

## Backtest Summary

Best row under the issue 49 worst-case cost scenario, 20 bps fee plus 20 bps slippage:

| Edge | Net PnL | Trades | Win Rate | Max Drawdown | Sharpe | Sortino | Trades / 1000 Signals | Top-1 Market Abs PnL Share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 8.6651 | 71 | 0.6338 | 2.2168 | 0.2519 | 0.3948 | 5.46 | 1.45% |

Baseline under the same cost scenario produced net PnL `4.2572`, so v3's worst-case cost delta versus baseline was `+4.4079`.

## Serving Readiness

Local serving readiness was generated on macOS arm64 with 8 CPUs using `docs/models/xgboost-v3-serving-readiness.json`.

| Check | Result |
|---|---:|
| p50 single-row latency | 0.2416 ms |
| p95 single-row latency | 0.3112 ms |
| p99 single-row latency | 0.3870 ms |
| valid-input error rate | 0.0000 |
| 10k batch throughput | 104843.87 rows/sec |
| 100k batch throughput | 102108.69 rows/sec |
| schema validation | valid input accepted, invalid input rejected |
| fallback/rollback | fallback model and rollback runbook present |

Readiness caveat: the finite-feature benchmark subset had 4 unique test rows, repeated for latency and throughput measurement. This is enough to verify local serving mechanics but should be strengthened by upstream finite-feature enforcement or imputation before production launch.

## Interpretability

Top feature importance by gain:

| Rank | Feature | Gain | Split Count |
|---:|---|---:|---:|
| 1 | `obi_l1` | 41.0032 | 116 |
| 2 | `microprice` | 34.5244 | 441 |
| 3 | `obi_l10` | 30.8061 | 614 |
| 4 | `spread` | 24.9162 | 25 |
| 5 | `obi_l5` | 21.4456 | 542 |

Example contribution summaries:

| Example | Probability | Largest Contributions |
|---:|---:|---|
| 1 | 0.6761 | `obi_l5` +0.8353, `microprice` -0.1979, `obi_l10` -0.0635 |
| 2 | 0.8550 | `microprice` +1.8111, `obi_l10` -1.2701, `obi_l5` +0.7023 |

## Feature Stability

Feature stability depends on the `feature_schema.json` artifact. Serving uses fail-closed schema validation: missing, extra, wrongly ordered, non-numeric, or non-finite inputs are rejected instead of silently producing predictions.

The observed finite-row caveat means online feature generation should monitor null and NaN rates for every feature, especially order-book imbalance and volatility fields.

## Monitoring Plan

Monitoring after deployment should include:

- Prediction distribution drift: mean, quantiles, and edge distribution by market.
- Label hit rate after settlement, especially when the live positive rate moves away from the 72.22% test-window regime.
- Calibration drift: Brier, ECE, reliability bins, and label shift once outcomes settle.
- Trading utility: net PnL after costs, drawdown, Sharpe/Sortino, turnover, and market concentration.
- Serving health: p50/p95/p99 latency, error rate, schema rejection rate, and stale prediction rate.
- Feature quality: per-feature null/NaN rate, schema hash mismatches, and market data staleness.

## Shadow Mode Plan

Run v3 as a shadow challenger against the current baseline/champion path before routing production decisions to it. Shadow mode scores both models on the same eligible `features_15m_v1` rows and writes a comparison report; it does not change the champion output.

Historical smoke run:

```bash
bigan-ingest shadow-v1 \
  --warehouse-dir data/model-train-backtest-rerun-20260520T134420Z/warehouse \
  --champion-model-path data/model-train-backtest-rerun-20260520T134420Z/artifacts/models/logreg-baseline-v1/model.json \
  --champion-calibration-path data/model-train-backtest-rerun-20260520T134420Z/artifacts/models/logreg-baseline-v1-platt-calibration/calibration.json \
  --challenger-model-path data/model-train-backtest-rerun-20260520T134420Z/artifacts/models/xgboost-v3/model.json \
  --challenger-calibration-path data/model-train-backtest-rerun-20260520T134420Z/artifacts/models/xgboost-v3-calibration/calibration.json \
  --output-path data/shadow/xgboost-v3-shadow-smoke.json
```

For a 1-2 day live shadow window, run the same command on a schedule after each feature batch, using `--lookback-hours 48` or explicit `--since-ms` / `--until-ms` windows. Review `challenger_error_count`, probability distribution drift, average latency, and label hit rate once outcomes settle.

## Promotion Notes

`xgboost-v3` is good enough to continue bootstrap promotion review because it has stronger held-out probability quality than the baseline and still beats baseline backtest utility under 20 bps costs. It is not evidence of a globally best model. Promotion should still require a fresh bootstrap report using this model card and serving readiness JSON as inputs.
