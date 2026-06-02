# Issue 93/94 xgboost-v6 Multi-Head Design

Implemented surface:

- `xgboost-v6` trains a three-class settlement head on `label_settlement_3way`.
- Settlement inference returns explicit `p_up`, `p_down`, and `p_neutral`; DOWN is never derived from `1 - p_up`.
- Settlement calibration uses temperature scaling with per-family temperatures when enough validation samples exist and a global fallback otherwise.
- Reports include accuracy, log loss, multiclass Brier, per-class ROC-AUC/PR-AUC/Brier/ECE, and reliability-curve buckets.
- Independent volatility heads train from `label_volatility_up` and `label_volatility_down`, producing `p_vol_up` and `p_vol_down`.
- Volatility reports include per-side base rate, calibration metrics, bucket hit rate, high-probability max-exit-gain distribution, and a trivial realized-vol/spread/OBI baseline comparison.
- Cost-adjusted/account-cashflow proxy PnL is the metric of record for the joint gate.

Executor contract:

- Payload fields are `p_up`, `p_down`, `p_neutral`, `p_vol_up`, `p_vol_down`, and `model_version`.
- `prob_up_15m` remains only a clipped legacy alias for `p_up`.
- UP entry requires `p_up` to be the settlement-side max, `p_up >= settlement_threshold`, `p_neutral <= neutral_cap`, `p_vol_up >= volatility_threshold`, and expected max exit gain to clear round-trip cost plus margin.
- DOWN is symmetric and must read explicit `p_down` and `p_vol_down`.
- `p_vol_*` is an entry gate for the issue #90 volatility sleeve only. Running-bankroll sizing and the min-size floor remain separate controls.

Promotion evidence still required before live use:

- Train on the issue #91/#92 v6 corpus with expected sample counts by family in the dataset manifest.
- Add same-dataset `v5_prob_up_15m` reference predictions and inspect `v5_comparison.json`.
- Pass per-family v5-vs-v6 cost-adjusted/account-cashflow PnL, not only AUC/Brier.
- Run paper/orderbook-only shadow before any real FOK execution.

Primary artifacts produced by `python -m bigan.ingestion.__main__ xgboost-v6`:

- `model.json`
- `settlement_model.json`
- `volatility_up_model.json`
- `volatility_down_model.json`
- `metrics.json`
- `family_metrics.json`
- `volatility_metrics.json`
- `cost_adjusted_backtest.json`
- `v5_comparison.json`
- `executor_integration.md`
