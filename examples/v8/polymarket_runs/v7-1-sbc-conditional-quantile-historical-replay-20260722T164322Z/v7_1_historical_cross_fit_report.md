# v7.1 SBC Conditional-Quantile Historical Report

- historical gate passed: `false`
- blockers: `['historical_same_dataset_candidate_pnl_not_strictly_better_than_v6_7', 'historical_same_dataset_largest_winner_removed_pnl_worse_than_v6_7']`
- leakage audit passed: `true`
- OOF markets: `90`
- OOF pinball improvement (report-only): `0.013979104679842933`
- conformal coverage: `0.8222222222222222`
- positive-LCB markets: `31`
- positive-LCB PnL (report-only): `-0.6707499999999994`
- historical same-dataset replay gate passed: `false`
- v7.1 frozen-size PnL: `-0.13414999999999988`
- v6.7 frozen-size PnL: `0.7030000000000001`
- candidate-minus-v6.7 frozen-size PnL: `-0.83715`
- HTS: `explicitly unavailable fail-closed`
- historical PnL used for model/feature/threshold tuning: `false`
- historical PnL used for pre-collection screening: `true`
- historical replay is promotion evidence: `false`
- target-free canary started: `false`
- paper/live/write/wallet/capital unlock: `false`
