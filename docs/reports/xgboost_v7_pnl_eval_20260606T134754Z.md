# xgboost-v7 Execution-Restricted PnL Evaluation

Status: **V7_EVAL_PROMISING_SMALL_SCOPE**

Model artifact: `data/model-runs/xgboost-v7/20260606T132859Z-stable-gate/model.json`
Model artifact kind: `xgboost-v7`
Dataset: `data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/dataset`

This report ranks policies by executable one-way settlement PnL under entry-window, slippage, and one-trade-per-round constraints. Hit rate and settlement accuracy are diagnostics, not the metric of record.

## Metric

- Probability edge: `p_side - entry_worst_price`.
- Residual edge: `residual_expected_edge_side`, when the artifact emits it.
- Hybrid edge: probability gate plus residual edge floor.
- Market baseline: buy the first market-favorite side without model edge.
- Entry worst price: `ask + buy_slippage + fee`, capped at `0.99`.
- PnL: win pays `1 - entry_worst_price`; loss pays `-entry_worst_price`.
- Settlement buy-and-hold does not subtract the old volatility round-trip cost.
- Each policy admits at most one settlement trade per round.

## Selected Policy

- selection splits: `train,val`
- validation split: `val`
- test split: `test`
- signal source: `probability`
- min confidence: `0.75`
- min expected edge: `0.04`

## Split Results

| Split | Policy | Trades | Coverage | PnL | Avg PnL | Hit rate | Mean edge | Mean price | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | v6_current_gate | 20 | 0.0676 | 3.6000 | 0.1800 | 1.0000 | 0.0923 | 0.8200 | 0.0000 |
| train | v7_selected_by_pnl_stability | 70 | 0.2365 | 9.5200 | 0.1360 | 0.9429 | 0.0653 | 0.8069 | -1.5600 |
| train | v7_residual_edge_gate | 93 | 0.3142 | -15.0400 | -0.1617 | 0.3441 | -0.2264 | 0.5058 | -21.3300 |
| train | v7_hybrid_edge_gate | 5 | 0.0169 | 1.6300 | 0.3260 | 1.0000 | 0.1058 | 0.6740 | 0.0000 |
| train | market_favorite_baseline | 296 | 1.0000 | -27.0500 | -0.0914 | 0.4595 | -0.1121 | 0.5508 | -28.8900 |
| train | first_model_side_no_edge_floor | 239 | 0.8074 | 16.8900 | 0.0707 | 0.6485 | -0.0137 | 0.5779 | -5.0200 |
| val | v6_current_gate | 4 | 0.0769 | -1.2700 | -0.3175 | 0.5000 | 0.0977 | 0.8175 | -1.2700 |
| val | v7_selected_by_pnl_stability | 17 | 0.3269 | 1.7800 | 0.1047 | 0.8824 | 0.0776 | 0.7776 | -0.6800 |
| val | v7_residual_edge_gate | 4 | 0.0769 | 0.3900 | 0.0975 | 0.5000 | 0.0362 | 0.4025 | -0.5300 |
| val | v7_hybrid_edge_gate | 2 | 0.0385 | -1.3100 | -0.6550 | 0.0000 | 0.1099 | 0.6550 | -1.3100 |
| val | market_favorite_baseline | 52 | 1.0000 | -0.6900 | -0.0133 | 0.5385 | -0.0238 | 0.5517 | -4.8600 |
| val | first_model_side_no_edge_floor | 52 | 1.0000 | 3.3400 | 0.0642 | 0.6154 | 0.0255 | 0.5512 | -3.3400 |
| test | v6_current_gate | 1 | 0.0114 | -0.7900 | -0.7900 | 0.0000 | 0.0930 | 0.7900 | -0.7900 |
| test | v7_selected_by_pnl_stability | 32 | 0.3636 | 1.3200 | 0.0412 | 0.8438 | 0.0618 | 0.8025 | -1.9200 |
| test | v7_residual_edge_gate | 10 | 0.1136 | 0.8000 | 0.0800 | 0.5000 | 0.0079 | 0.4200 | -0.8800 |
| test | v7_hybrid_edge_gate | 0 | 0.0000 | 0.0000 |  |  |  |  | 0.0000 |
| test | market_favorite_baseline | 88 | 1.0000 | -0.9800 | -0.0111 | 0.5341 | -0.0161 | 0.5452 | -4.9000 |
| test | first_model_side_no_edge_floor | 88 | 1.0000 | -4.1300 | -0.0469 | 0.5227 | 0.0071 | 0.5697 | -4.7200 |

## Candidate Counts

| Split | Candidate rows | Candidate rounds | Skips |
|---|---:|---:|---|
| train | 6294 | 296 | `{"before_round_start": 9538, "no_new_entry_window": 1150}` |
| val | 1088 | 52 | `{"before_round_start": 1873, "no_new_entry_window": 197}` |
| test | 1858 | 88 | `{"before_round_start": 2275, "no_new_entry_window": 343}` |

## Top Validation Grid Results

| Rank | Min confidence | Min edge | Trades | PnL | Avg PnL | Hit rate | Mean price |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.700 | 0.040 | 24 | 1.8600 | 0.0775 | 0.7917 | 0.7142 |
| 2 | 0.750 | 0.040 | 17 | 1.7800 | 0.1047 | 0.8824 | 0.7776 |
| 3 | 0.700 | 0.020 | 30 | 1.7200 | 0.0573 | 0.8000 | 0.7427 |
| 4 | 0.650 | 0.100 | 13 | 1.6400 | 0.1262 | 0.6923 | 0.5662 |
| 5 | 0.650 | 0.082 | 18 | 1.5000 | 0.0833 | 0.6667 | 0.5833 |
| 6 | 0.650 | 0.040 | 30 | 1.4200 | 0.0473 | 0.7000 | 0.6527 |
| 7 | 0.600 | 0.100 | 16 | 1.2100 | 0.0756 | 0.6250 | 0.5494 |
| 8 | 0.700 | 0.082 | 12 | 1.1900 | 0.0992 | 0.7500 | 0.6508 |
| 9 | 0.600 | 0.082 | 21 | 1.1500 | 0.0548 | 0.6190 | 0.5643 |
| 10 | 0.750 | 0.020 | 25 | 1.0100 | 0.0404 | 0.8400 | 0.7996 |
| 11 | 0.650 | 0.060 | 23 | 0.9800 | 0.0426 | 0.6522 | 0.6096 |
| 12 | 0.650 | 0.020 | 35 | 0.9600 | 0.0274 | 0.7143 | 0.6869 |

## Interpretation

The PnL-stability-selected v7 policy beats the fixed v6 gate on the held-out split and produces positive one-way settlement PnL. This is a small-scope offline check; the next step is executor integration or paper-only shadow with the same policy thresholds.
