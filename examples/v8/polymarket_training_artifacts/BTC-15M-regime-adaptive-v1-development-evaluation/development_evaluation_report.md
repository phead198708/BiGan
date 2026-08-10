# BTC 15m regime-adaptive development evaluation

- Lineage: `BTC-15M-regime-adaptive-v1`
- Evidence role: development selection only; never promotion evidence
- Rolling-origin markets: 73
- Candidate executions: 5/5
- Fresh collection authorized: no

## Candidate results

| Candidate | Accepted | PnL | 95% LCB | First | Second | Eligible |
|---|---:|---:|---:|---:|---:|:---:|
| global_baseline | 72 | 0.532000 | -0.091142 | -1.324000 | 1.856000 | no |
| regime_conditioned_calibration | 72 | 0.672000 | -0.089711 | -2.914000 | 3.586000 | no |
| mixture_of_experts | 72 | 4.372000 | -0.035937 | 0.146000 | 4.226000 | no |
| drift_aware_rolling_calibration | 73 | 0.381250 | -0.099438 | -0.779500 | 1.160750 | no |
| uncertainty_aware_abstention | 53 | -2.532750 | -0.102943 | -0.736750 | -1.796000 | no |

## Selection

- Status: `no_candidate_met_all_frozen_development_gates`
- Selected candidate: `None`
- Fresh collection allowed by this result: `false`

The result is outcome-aware development evidence. It cannot be reused as validation evidence and makes no promotion, paper, live, wallet, write, or capital-at-risk claim.
