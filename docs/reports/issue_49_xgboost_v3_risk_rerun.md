# Issue 49 XGBoost-v3 Risk Rerun

Run date: 2026-05-21  
Run root: `data/model-train-backtest-rerun-20260520T134420Z`

## Scope

Issue 49 adds risk and sensitivity evidence to the edge-based settlement
backtests used by bootstrap champion decisions. This rerun uses the existing
baseline, xgboost-v2, and xgboost-v3 prediction artifacts and regenerates
cost-adjusted summaries with risk fields.

Backtest quote source: Polymarket CLOB top-of-book rows from
`warehouse/raw_top_of_book/source=polymarket`. The strategy buys UP tokens only
when `prob_up_15m - market_implied_prob >= edge_threshold`.

## Code Changes Covered

- Added grouped backtest risk fields: `max_drawdown`, `max_drawdown_pct`,
  `sharpe_ratio`, `sortino_ratio`, turnover, and per-market concentration.
- Added cost/latency knobs to `backtest-predictions-v1`:
  `--fee-bps`, `--slippage-bps`, and `--latency-ms`.
- Updated bootstrap backtest gates to accept `sharpe_ratio`/`sortino_ratio`,
  require turnover and concentration evidence, and show candidate net PnL delta
  versus the baseline.
- Documented the summary JSON fields in `docs/runbooks/bootstrap_champion.md`.

## Offline Test Metrics

| Model | Raw AUC | Raw Brier | Raw ECE | Calibrated Brier | Calibrated ECE |
|---|---:|---:|---:|---:|---:|
| logreg-baseline-v1 | 0.6287 | 0.2012 | 0.1436 | 0.1978 | 0.1318 |
| xgboost-v2 | 0.7582 | 0.2033 | 0.1985 | 0.2278 | 0.2274 |
| xgboost-v3 | 0.8503 | 0.1331 | 0.1304 | 0.1235 | 0.0927 |

v3 is the strongest probability-quality candidate. v2 has better AUC than the
baseline, but its held-out Brier score is worse than the baseline, so it remains
unsafe as a probability champion despite profitable backtests.

## Cost Sensitivity Best Rows

Each row is the best edge threshold by net PnL for that model and cost scenario.

| Model | Scenario | Edge | Net PnL | Trades | Max DD | Sharpe | Sortino |
|---|---:|---:|---:|---:|---:|---:|---:|
| logreg-baseline-v1 | 0 bps, 0 ms | 0.10 | 4.3000 | 21 | 1.0600 | 0.4428 | 0.7529 |
| logreg-baseline-v1 | 5 bps, 0 ms | 0.10 | 4.2893 | 21 | 1.0631 | 0.4417 | 0.7503 |
| logreg-baseline-v1 | 10 bps, 0 ms | 0.10 | 4.2786 | 21 | 1.0661 | 0.4406 | 0.7476 |
| logreg-baseline-v1 | 20 bps, 0 ms | 0.10 | 4.2572 | 21 | 1.0723 | 0.4384 | 0.7424 |
| logreg-baseline-v1 | 5 bps, 500 ms | 0.10 | 4.2893 | 21 | 1.0631 | 0.4417 | 0.7503 |
| xgboost-v2 | 0 bps, 0 ms | 0.30 | 11.8700 | 65 | 2.1400 | 0.3929 | 0.6466 |
| xgboost-v2 | 5 bps, 0 ms | 0.30 | 11.8369 | 65 | 2.1471 | 0.3918 | 0.6442 |
| xgboost-v2 | 10 bps, 0 ms | 0.30 | 11.8037 | 65 | 2.1543 | 0.3907 | 0.6417 |
| xgboost-v2 | 20 bps, 0 ms | 0.30 | 11.7373 | 65 | 2.1686 | 0.3885 | 0.6368 |
| xgboost-v2 | 5 bps, 500 ms | 0.30 | 11.8369 | 65 | 2.1471 | 0.3918 | 0.6442 |
| xgboost-v3 | 0 bps, 0 ms | 0.30 | 8.8100 | 71 | 2.1800 | 0.2561 | 0.4030 |
| xgboost-v3 | 5 bps, 0 ms | 0.30 | 8.7738 | 71 | 2.1892 | 0.2550 | 0.4009 |
| xgboost-v3 | 10 bps, 0 ms | 0.30 | 8.7376 | 71 | 2.1984 | 0.2540 | 0.3989 |
| xgboost-v3 | 20 bps, 0 ms | 0.30 | 8.6651 | 71 | 2.2168 | 0.2519 | 0.3948 |
| xgboost-v3 | 5 bps, 500 ms | 0.30 | 8.7738 | 71 | 2.1892 | 0.2550 | 0.4009 |

The 500 ms rows match the 5 bps, 0 ms rows in this snapshot because the
available top-of-book quote timestamps do not change the executable entry quote
for the selected fills.

## Worst-Case Risk Comparison

Worst-case cost here means 20 bps fee plus 20 bps slippage with zero added
latency.

| Model | Edge | Net PnL | Delta vs Baseline | Trades | Trades / 1000 Signals | Max DD | Sharpe | Sortino | Top-1 Market Abs PnL Share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg-baseline-v1 | 0.10 | 4.2572 | 0.0000 | 21 | 1.61 | 1.0723 | 0.4384 | 0.7424 | 4.92% |
| xgboost-v2 | 0.30 | 11.7373 | 7.4802 | 65 | 5.00 | 2.1686 | 0.3885 | 0.6368 | 1.59% |
| xgboost-v3 | 0.30 | 8.6651 | 4.4079 | 71 | 5.46 | 2.2168 | 0.2519 | 0.3948 | 1.45% |

Both xgboost candidates still beat the baseline under the worst-case cost
scenario on net PnL. v2 has the strongest cost-adjusted utility, but it fails the
offline probability-quality gate. v3 has lower net PnL than v2 but much stronger
offline probability quality and remains the better first-champion candidate path.

## Bootstrap Gate Rerun

| Candidate | Action | Backtest Acceptable | Primary Remaining Blockers |
|---|---|---|---|
| xgboost-v2 | KEEP_BASELINE_TEMPORARILY | yes | Worse Brier than baseline; serving readiness missing; complexity notes missing |
| xgboost-v3 | CONTINUE_BOOTSTRAP_EXPERIMENTATION | yes | Serving readiness missing; complexity notes missing |

The important issue 49 fix is confirmed: bootstrap no longer auto-fails the
backtest gate because risk metrics are missing. For xgboost-v3, the backtest
summary now reads `net_pnl 8.6651, trades 71, delta_vs_baseline 4.4079,
max_dd 2.2168, sharpe 0.2519`.

## Artifacts

- Best-row sensitivity summary:
  `data/model-train-backtest-rerun-20260520T134420Z/artifacts/backtests/issue-49-risk-cost-sensitivity/best_rows.json`
- Worst-cost baseline summary:
  `data/model-train-backtest-rerun-20260520T134420Z/artifacts/backtests/issue-49-risk-cost-sensitivity/logreg-baseline-v1/cost-20bps-fee-20bps-slippage-latency-0ms/summary.json`
- Worst-cost xgboost-v2 summary:
  `data/model-train-backtest-rerun-20260520T134420Z/artifacts/backtests/issue-49-risk-cost-sensitivity/xgboost-v2/cost-20bps-fee-20bps-slippage-latency-0ms/summary.json`
- Worst-cost xgboost-v3 summary:
  `data/model-train-backtest-rerun-20260520T134420Z/artifacts/backtests/issue-49-risk-cost-sensitivity/xgboost-v3/cost-20bps-fee-20bps-slippage-latency-0ms/summary.json`
- Bootstrap rerun, xgboost-v2:
  `data/model-train-backtest-rerun-20260520T134420Z/artifacts/bootstrap-champion-issue-49-xgboost-v2/bootstrap_decision.md`
- Bootstrap rerun, xgboost-v3:
  `data/model-train-backtest-rerun-20260520T134420Z/artifacts/bootstrap-champion-issue-49-xgboost-v3/bootstrap_decision.md`

## Recommendation

Do not promote yet. Continue with xgboost-v3 as the primary candidate path
because it is the only challenger here with clearly stronger held-out
probability quality than the baseline while still beating baseline utility under
20 bps costs. The next promotion blockers are serving latency/error evidence and
short model-complexity notes; once those exist, rerun the bootstrap gate before
making a first-champion decision.
