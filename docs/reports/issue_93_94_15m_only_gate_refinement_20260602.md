# Issue 93/94 15M-Only V6 Gate Refinement

Status: **PROMISING BUT NOT PROMOTION EVIDENCE**

Run root: `data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z`

Source dataset: `data/model-runs/xgboost-v6-issue93-94-multifamily-volatility-20260602T133537Z/dataset`

Subset filter: `market_family in (BTC-15M, ETH-15M)`

## Dataset

| Split | Rows | Volatility UP known | Volatility DOWN known |
|---|---:|---:|---:|
| train | 25721 | 7275 | 7275 |
| val | 4742 | 1571 | 1571 |
| test | 6055 | 2591 | 2591 |

## Single-Grid V6 Metrics

Settlement head:

| Split | Accuracy | Log Loss | ECE UP | ECE DOWN | ECE NEUTRAL |
|---|---:|---:|---:|---:|---:|
| train | 0.6081 | 0.8151 | 0.0389 | 0.0479 | 0.0550 |
| val | 0.5555 | 0.8012 | 0.0213 | 0.0219 | 0.0316 |
| test | 0.5690 | 0.7944 | 0.0520 | 0.0641 | 0.0307 |

Volatility heads:

| Split | Side | Samples | Positives | ROC AUC | PR AUC | Learned Brier | Trivial Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| train | UP | 7275 | 2982 | 0.7689 | 0.6584 | 0.1879 | 0.3370 |
| train | DOWN | 7275 | 2884 | 0.7772 | 0.6711 | 0.1862 | 0.3273 |
| val | UP | 1571 | 615 | 0.7094 | 0.4961 | 0.1909 | 0.3016 |
| val | DOWN | 1571 | 532 | 0.7129 | 0.5356 | 0.1908 | 0.2718 |
| test | UP | 2591 | 1071 | 0.7296 | 0.5486 | 0.1926 | 0.3271 |
| test | DOWN | 2591 | 1032 | 0.7154 | 0.5259 | 0.1958 | 0.3146 |

## Selected Joint Rule

The standard single-grid trainer selected:

- `settlement_threshold=0.50`
- `neutral_cap=0.25`
- `volatility_threshold=0.60`

| Split | Trades | PnL | Hit Rate | Max Drawdown | Family |
|---|---:|---:|---:|---:|---|
| train | 239 | 19.0220 | 0.6569 | -10.9870 | BTC-15M only |
| val | 16 | 4.6480 | 0.8750 | -0.5670 | BTC-15M only |
| test | 78 | 0.6390 | 0.5897 | -8.7150 | BTC-15M only |

## Threshold Sweep

Wider sweep preserving the same cost rule:

| Split | Best nonzero candidate | Result |
|---|---|---|
| val | `settlement_threshold=0.45`, `neutral_cap=0.15`, `volatility_threshold=0.60` | 66 trades, PnL `13.8280`, hit rate 0.7879 |
| test | `settlement_threshold=0.45`, `neutral_cap=0.15`, `volatility_threshold=0.60` | 290 trades, PnL `6.3600`, hit rate 0.5931 |

The test candidate is mixed by family:

| Family | Trades | PnL | Hit Rate |
|---|---:|---:|---:|
| BTC-15M | 283 | 10.4940 | 0.6078 |
| ETH-15M | 7 | -4.1340 | 0.0000 |

## Per-Family Training Check

Separate family-only datasets were also trained with the same single-grid configuration:

- BTC-only run: `data/model-runs/xgboost-v6-issue93-94-btc15-only-volatility-20260602T135044Z`
- ETH-only run: `data/model-runs/xgboost-v6-issue93-94-eth15-only-volatility-20260602T135044Z`

The standard trainer-selected rules returned zero trades for both family-only models. A wider threshold sweep showed the instability:

| Run | Split | Best nonzero candidate | Result |
|---|---|---|---|
| BTC-15M-only | val | `settlement_threshold=0.60`, `neutral_cap=0.15`, `volatility_threshold=0.20` | 144 trades, PnL `-8.1080`, hit rate 0.6181 |
| BTC-15M-only | test | `settlement_threshold=0.50`, `neutral_cap=0.15`, `volatility_threshold=0.50` | 1657 trades, PnL `28.1010`, hit rate 0.5957 |
| ETH-15M-only | val | `settlement_threshold=0.60`, `neutral_cap=0.25`, `volatility_threshold=0.20` | 1 trade, PnL `0.1830`, hit rate 1.0000 |
| ETH-15M-only | test | `settlement_threshold=0.55`, `neutral_cap=0.25`, `volatility_threshold=0.40` | 2 trades, PnL `0.4910`, hit rate 1.0000 |

This does not support promoting a separately trained BTC-15M-only model: its best validation candidates are negative while test looks positive. It also does not support an ETH-15M-only strategy because candidate trade counts are too small.

## Conclusion

Removing 5M rows materially improves offline joint-gate behavior. The signal remains too thin for promotion:

- The selected rule produces only `0.6390` test PnL with `-8.7150` max drawdown.
- The wider sweep is positive on test, but most edge is BTC-15M; ETH-15M contributes losses.
- The family-only retrains are not stable enough to promote.
- This should feed the next refinement step: keep the 15M mixed model, replay an execution-restricted BTC-15M gate, and require a paper/orderbook-only shadow before any promotion decision.
