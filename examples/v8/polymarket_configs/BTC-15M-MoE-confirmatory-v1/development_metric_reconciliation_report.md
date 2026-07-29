# BTC 15m MoE parent metric reconciliation

- Reconciliation passed: `true`
- Prediction rows: 1460
- Fold audits: 365
- OOF markets: 73
- Floating tolerance: 1e-12

| Candidate | Accepted | PnL | 95% LCB | Metric match | Gate match |
|---|---:|---:|---:|:---:|:---:|
| global_baseline | 72 | 0.532000 | -0.091142 | yes | yes |
| regime_conditioned_calibration | 72 | 0.672000 | -0.089711 | yes | yes |
| mixture_of_experts | 72 | 4.372000 | -0.035937 | yes | yes |
| drift_aware_rolling_calibration | 73 | 0.381250 | -0.099438 | yes | yes |
| uncertainty_aware_abstention | 53 | -2.532750 | -0.102943 | yes | yes |

Metric reconciliation does not repair the unresolvable recorded source commit. Candidate freeze remains blocked by provenance.
