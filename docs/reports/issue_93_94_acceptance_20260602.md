# Issue 93/94 Acceptance Run

Status: **PARTIAL / BLOCKED FOR PROMOTION**

Run root: `data/model-runs/xgboost-v6-issue93-94-acceptance-20260602T130107Z`

## What Ran

- Source dataset: `data/model-runs/xgboost-v5-run-20260531T100500Z-atomic/dataset`
- Source v5 model: `data/model-runs/xgboost-v5-run-20260531T100500Z-atomic/model/model.json`
- Source v5 calibration: `data/model-runs/xgboost-v5-run-20260531T100500Z-atomic/calibration-family/calibration.json`
- Derived v6 dataset: `data/model-runs/xgboost-v6-issue93-94-acceptance-20260602T130107Z/dataset`
- v6 model output: `data/model-runs/xgboost-v6-issue93-94-acceptance-20260602T130107Z/model-single-grid`
- Neutral margin used for the derived settlement label: `1.0`

## Dataset Coverage

Rows: `115344`

Family counts:

| Family | Rows |
|---|---:|
| BTC-15M | 19996 |
| BTC-5M | 46260 |
| ETH-15M | 16522 |
| ETH-5M | 32566 |

Settlement labels are populated on all rows:

| Split | UP | DOWN | NEUTRAL |
|---|---:|---:|---:|
| train | 32186 | 30923 | 19279 |
| val | 6775 | 6789 | 2913 |
| test | 7053 | 7537 | 1889 |

Volatility labels are **not** populated:

| Split | `label_volatility_up` known | `label_volatility_down` known |
|---|---:|---:|
| train | 0 | 0 |
| val | 0 | 0 |
| test | 0 | 0 |

This blocks issue #94 promotion acceptance because the existing v5 dataset does not contain the intra-round bid/ask path needed by issue #91.

## Feature Parity

Feature parity vs v5 is clean:

- added: `[]`
- removed: `[]`
- v5 required missing: `[]`

## Settlement Metrics

| Split | Samples | Accuracy | Log Loss | Multiclass Brier |
|---|---:|---:|---:|---:|
| train | 82388 | 0.5238 | 0.8998 | 0.5565 |
| val | 16477 | 0.4840 | 0.8729 | 0.5495 |
| test | 16479 | 0.5153 | 0.8313 | 0.5257 |

Per-class ECE on test:

| Class | ECE |
|---|---:|
| UP | 0.0163 |
| DOWN | 0.0341 |
| NEUTRAL | 0.0312 |

## Cost-Adjusted PnL

Metric of record: `cost_adjusted_account_cashflow_proxy_pnl`.

Because volatility labels are missing, v6 `p_vol_*` heads are constant and the #90-compatible joint gate takes no trades:

| Split | v6 Trades | v6 PnL |
|---|---:|---:|
| train | 0 | 0.0 |
| val | 0 | 0.0 |
| test | 0 | 0.0 |

The same-dataset calibrated v5 reference was populated and compared:

| Split | v5 Trades | v5 PnL | v5 Hit Rate |
|---|---:|---:|---:|
| train | 82388 | -13395.1940 | 0.4237 |
| val | 16477 | -2440.8730 | 0.4334 |
| test | 16479 | -1989.1555 | 0.4643 |

The positive v6-vs-v5 PnL delta is **not promotion evidence** because v6 achieved it by taking zero trades under the missing-volatility-label gate.

## Issue Checklist

| Requirement | Status | Evidence |
|---|---|---|
| #93 three-class settlement head | PASS | `model-single-grid/model.json`, `settlement_model.json` |
| #93 explicit `p_up/p_down/p_neutral` payload | PASS | serving contract tests and v6 artifact contract |
| #93 no `1-p_up` DOWN inference | PASS | `tests/serving/test_contracts.py` regression |
| #93 per-class ECE/reliability | PASS | `model-single-grid/metrics.json` |
| #93 v5-vs-v6 cost-adjusted comparison | PARTIAL | comparison exists, but v6 has 0 trades |
| #94 independent `p_vol_up/p_vol_down` heads | CODE PASS / DATA BLOCKED | heads exist, but no known labels |
| #94 trivial volatility baseline | DATA BLOCKED | baseline sample count is 0 |
| #94 per-side volatility calibration/hit-rate | DATA BLOCKED | up/down known label count is 0 |
| #94 #90 gate vs sizing documentation | PASS | `model-single-grid/executor_integration.md` |
| #94 paper/orderbook-only run | NOT RUN | requires nonzero volatility coverage first |

## Next Required Step

Build a v6 corpus from Phase 4 capture windows that includes intra-round bid/ask paths, so `label_volatility_up` and `label_volatility_down` have nonzero coverage per family. Then rerun:

```bash
python -m bigan.ingestion.__main__ xgboost-v6 \
  --dataset-dir data/model-runs/<v6-corpus-with-price-path>/dataset \
  --output-dir data/model-runs/<v6-acceptance-run>/model
```

Promotion remains blocked until the rerun has nonzero v6 trade count, per-side volatility metrics versus the trivial baseline, and a paper/orderbook-only run judged by account-cashflow PnL.

## Follow-Up: Low-Latency Volatility Overlay

Status: **PARTIAL / PROMOTION STILL BLOCKED**

Follow-up run root: `data/model-runs/xgboost-v6-issue93-94-lowlat-volatility-20260602T153000Z`

This follow-up used the Phase 4 low-latency BTC-15M capture to add real forward book-path volatility labels:

- Base dataset retained: `data/model-runs/xgboost-v6-issue93-94-acceptance-20260602T130107Z/dataset`
- Low-latency feature source: `data/live/xgboost-v5-btc15-lowlat-20260531T105815Z/warehouse/features_15m_v1`
- Quote path source: `data/live/xgboost-v5-btc15-lowlat-20260531T105815Z/rollup/ws_market/date=*/event_type=best_bid_ask/*.parquet`
- Settlement source for overlay rows: `market_resolved.winning_outcome`
- Derived dataset: `data/model-runs/xgboost-v6-issue93-94-lowlat-volatility-20260602T153000Z/dataset`
- v6 model output: `data/model-runs/xgboost-v6-issue93-94-lowlat-volatility-20260602T153000Z/model-single-grid`

The overlay added `3117` resolved BTC-15M UP-side feature snapshots. It intentionally kept the original margin-derived dataset so train still contains `UP/DOWN/NEUTRAL`; the low-latency overlay itself only has `UP/DOWN` settlement labels because the stream does not carry `start_price/target_price` margin fields.

### Overlay Label Coverage

| Split | Added rows | `label_volatility_up` known | Up positives | `label_volatility_down` known | Down positives |
|---|---:|---:|---:|---:|---:|
| train | 2181 | 1543 | 449 | 1543 | 444 |
| val | 468 | 316 | 71 | 316 | 92 |
| test | 468 | 338 | 46 | 338 | 73 |

Low-latency path diagnostics:

| Field | Count |
|---|---:|
| resolved UP-side feature rows | 3117 |
| missing v5 prediction | 0 |
| `label_volatility_up=True` | 566 |
| `label_volatility_down=True` | 609 |
| `volatility_path_validity_up=valid` | 1574 |
| `volatility_path_validity_down=valid` | 1662 |
| `volatility_path_validity_up=no_exit_window` | 916 |
| `volatility_path_validity_down=no_exit_window` | 916 |

### Follow-Up Settlement Metrics

| Split | Samples | Accuracy | Log Loss | Multiclass Brier |
|---|---:|---:|---:|---:|
| test | 16947 | 0.5193 | 0.8209 | 0.5192 |

Per-class ECE on test:

| Class | ECE |
|---|---:|
| UP | 0.0205 |
| DOWN | 0.0381 |
| NEUTRAL | 0.0326 |

### Follow-Up Volatility Metrics

Both volatility heads now train on real labels and beat the trivial volatility baseline on the labeled BTC-15M overlay splits.

| Split | Side | Samples | Positives | Base Rate | Beats trivial baseline |
|---|---|---:|---:|---:|---|
| train | UP | 1543 | 449 | 0.2910 | yes |
| train | DOWN | 1543 | 444 | 0.2878 | yes |
| val | UP | 316 | 71 | 0.2247 | yes |
| val | DOWN | 316 | 92 | 0.2911 | yes |
| test | UP | 338 | 46 | 0.1361 | yes |
| test | DOWN | 338 | 73 | 0.2160 | yes |

Corrected test bucket hit-rate:

| Side | Bucket | Samples | Hit Rate |
|---|---|---:|---:|
| UP | 0.00-0.50 | 329 | 0.1277 |
| UP | 0.50-0.60 | 9 | 0.4444 |
| UP | 0.60-0.70 | 0 | n/a |
| UP | 0.70-1.00 | 0 | n/a |
| DOWN | 0.00-0.50 | 277 | 0.1011 |
| DOWN | 0.50-0.60 | 34 | 0.8235 |
| DOWN | 0.60-0.70 | 27 | 0.6296 |
| DOWN | 0.70-1.00 | 0 | n/a |

Note: the first pass exposed a bucket accounting bug where the `0.70-1.00` bucket included all probabilities. This was fixed in `src/bigan/modeling/xgboost_v6.py` and locked with `tests/modeling/test_xgboost_v6.py::test_volatility_bucket_hit_rate_keeps_high_bucket_exclusive`.

### Follow-Up Cost-Adjusted PnL

Selected joint rule:

- `settlement_threshold=0.50`
- `neutral_cap=0.25`
- `volatility_threshold=0.50`

| Split | v6 Trades | v6 PnL | v6 Hit Rate | v5 Reference Trades | v5 Reference PnL | v5 Hit Rate |
|---|---:|---:|---:|---:|---:|---:|
| train | 6164 | -114.4530 | 0.5638 | 84569 | -13554.0335 | 0.4276 |
| val | 108 | 13.4190 | 0.7870 | 16945 | -2474.3235 | 0.4362 |
| test | 153 | 3.4290 | 0.6797 | 16947 | -2054.1350 | 0.4665 |

This is **not promotion evidence** yet. The volatility labels are real, but they only cover BTC-15M overlay rows; the joint-rule backtest still scores BTC-5M and ETH rows using heads trained from BTC-15M-only volatility labels. Promotion still requires broader price-path coverage and a paper/orderbook-only run judged by account-cashflow PnL.

### Updated Issue Checklist

| Requirement | Status | Evidence |
|---|---|---|
| #93 three-class settlement head | PASS | `model-single-grid/model.json`, `settlement_model.json` |
| #93 explicit `p_up/p_down/p_neutral` payload | PASS | serving contract tests and v6 artifact contract |
| #93 no `1-p_up` DOWN inference | PASS | `tests/serving/test_contracts.py` regression |
| #93 per-class ECE/reliability | PASS | `model-single-grid/metrics.json` |
| #93 v5-vs-v6 cost-adjusted comparison | PARTIAL | comparison exists with nonzero v6 trades, but not promotion evidence |
| #94 independent `p_vol_up/p_vol_down` heads | PARTIAL PASS | heads train on real BTC-15M price-path labels |
| #94 trivial volatility baseline | PARTIAL PASS | BTC-15M overlay heads beat baseline on train/val/test |
| #94 per-side volatility calibration/hit-rate | PARTIAL PASS | populated for BTC-15M overlay only |
| #94 #90 gate vs sizing documentation | PASS | `model-single-grid/executor_integration.md` |
| #94 paper/orderbook-only run | NOT RUN | still required before promotion |

## Remaining Promotion Blockers

- Add or recover low-latency `start_price/target_price` fields if the overlay itself must carry margin-derived `NEUTRAL` labels instead of relying on the base corpus for NEUTRAL coverage.
- Run the v6 rule in paper/orderbook-only mode and reconcile account-cashflow PnL before any live promotion decision.

## Follow-Up: Multifamily Rollup Volatility Dataset

Status: **DATA COVERAGE IMPROVED / PROMOTION STILL BLOCKED**

Follow-up run root: `data/model-runs/xgboost-v6-issue93-94-multifamily-volatility-20260602T133537Z`

This follow-up used the historical rollup best-bid/ask paths from the v4 multimarket atomic capture to fill volatility labels on the original v6 acceptance corpus:

- Base dataset: `data/model-runs/xgboost-v6-issue93-94-acceptance-20260602T130107Z/dataset`
- Rollup quote source: `data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z/rollup/ws_market/date=*/event_type=best_bid_ask/*.parquet`
- Builder: `scripts/build_xgboost_v6_volatility_dataset.py`
- Derived dataset: `data/model-runs/xgboost-v6-issue93-94-multifamily-volatility-20260602T133537Z/dataset`
- v6 model output: `data/model-runs/xgboost-v6-issue93-94-multifamily-volatility-20260602T133537Z/model-single-grid`

### Multifamily Label Coverage

| Split | Rows | `label_volatility_up` known | Up positives | `label_volatility_down` known | Down positives |
|---|---:|---:|---:|---:|---:|
| train | 82388 | 13612 | 3014 | 13612 | 2935 |
| val | 16477 | 2705 | 620 | 2705 | 532 |
| test | 16479 | 4458 | 1073 | 4458 | 1032 |

Known volatility labels by family:

| Family | train up/down | val up/down | test up/down |
|---|---:|---:|---:|
| BTC-15M | 4076 / 4076 | 834 / 834 | 1427 / 1427 |
| BTC-5M | 4270 / 4270 | 777 / 777 | 1310 / 1310 |
| ETH-15M | 3199 / 3199 | 737 / 737 | 1164 / 1164 |
| ETH-5M | 2067 / 2067 | 357 / 357 | 557 / 557 |

The 5M labels are mostly negative under the current execution-aware volatility rule. On test, BTC-5M has 2 UP positives and 0 DOWN positives; ETH-5M has 0 positives on both sides. That appears to be a real consequence of the 5M exit window and `min_exit_gain=0.15`, not a label plumbing failure.

### Multifamily Model Metrics

Settlement head:

| Split | Accuracy | Log Loss | ECE UP | ECE DOWN | ECE NEUTRAL |
|---|---:|---:|---:|---:|---:|
| train | 0.5238 | 0.8998 | 0.0234 | 0.0256 | 0.0456 |
| val | 0.4840 | 0.8729 | 0.0290 | 0.0338 | 0.0612 |
| test | 0.5153 | 0.8313 | 0.0163 | 0.0341 | 0.0312 |

Volatility heads:

| Split | Side | Samples | Positives | Base Rate | ROC AUC | PR AUC | Learned Brier | Trivial Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | UP | 13612 | 3014 | 0.2214 | 0.9017 | 0.6346 | 0.1043 | 0.1871 |
| train | DOWN | 13612 | 2935 | 0.2156 | 0.8954 | 0.6298 | 0.1050 | 0.1830 |
| val | UP | 2705 | 620 | 0.2292 | 0.8646 | 0.5018 | 0.1137 | 0.1944 |
| val | DOWN | 2705 | 532 | 0.1967 | 0.8762 | 0.5631 | 0.1087 | 0.1764 |
| test | UP | 4458 | 1073 | 0.2407 | 0.8803 | 0.5947 | 0.1131 | 0.1963 |
| test | DOWN | 4458 | 1032 | 0.2315 | 0.8718 | 0.5446 | 0.1147 | 0.1887 |

Both volatility heads beat the trivial baseline on train/val/test with real labels across BTC and ETH, 15M and 5M.

### Multifamily Cost-Adjusted PnL

Selected joint rule:

- `settlement_threshold=0.50`
- `neutral_cap=0.25`
- `volatility_threshold=0.60`

| Split | v6 Trades | v6 PnL | v6 Hit Rate | v5 Reference Trades | v5 Reference PnL |
|---|---:|---:|---:|---:|---:|
| train | 0 | 0.0000 | n/a | 82388 | -13395.1940 |
| val | 0 | 0.0000 | n/a | 16477 | -2440.8730 |
| test | 0 | 0.0000 | n/a | 16479 | -1989.1555 |

Threshold sweep diagnostic, preserving the cost rule:

| Split | Best nonzero candidate | Result |
|---|---|---|
| val | `settlement_threshold=0.55`, `neutral_cap=0.25`, `volatility_threshold=0.50` | 18 trades, PnL `-0.2160`, hit rate 0.6667 |
| test | `settlement_threshold=0.55`, `neutral_cap=0.25`, `volatility_threshold=0.50` | 12 trades, PnL `-3.2740`, hit rate 0.4167 |

Conclusion: the data blocker is materially improved, but the current settlement+volatility joint gate still does not produce a positive out-of-sample cost-adjusted account-cashflow proxy. The v6 multifamily run is therefore **not promotion evidence**. Next useful work is gate/model refinement: likely 15M-only volatility evaluation, family-specific gate priors, or a cost-aware training objective before any paper/orderbook-only live shadow run.

### Multifamily Issue Checklist

| Requirement | Status | Evidence |
|---|---|---|
| #93 three-class settlement head | PASS | `model-single-grid/model.json`, `settlement_model.json` |
| #93 explicit `p_up/p_down/p_neutral` payload | PASS | serving contract tests and v6 artifact contract |
| #93 no `1-p_up` DOWN inference | PASS | `tests/serving/test_contracts.py` regression |
| #93 per-class ECE/reliability | PASS | `model-single-grid/metrics.json` |
| #93 v5-vs-v6 cost-adjusted comparison | PARTIAL | comparison exists, but v6 selected rule has 0 trades |
| #94 independent `p_vol_up/p_vol_down` heads | PASS | heads train on real multifamily price-path labels |
| #94 trivial volatility baseline | PASS | both heads beat baseline on train/val/test |
| #94 per-side volatility calibration/hit-rate | PASS | populated in `model-single-grid/volatility_metrics.json` |
| #94 #90 gate vs sizing documentation | PASS | `model-single-grid/executor_integration.md` |
| #94 paper/orderbook-only run | NOT RUN | blocked by no positive offline joint-gate evidence |
