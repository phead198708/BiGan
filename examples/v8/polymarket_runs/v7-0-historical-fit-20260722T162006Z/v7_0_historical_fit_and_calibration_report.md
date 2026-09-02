# v7.0 Historical Fit and Calibration Report

- historical markets: `199`
- development gate passed: `false`
- blockers: `['hold_to_settlement_forward_oof_mae_improved_failed', 'hold_to_settlement_forward_oof_mse_improved_failed']`
- leakage audit passed: `true`
- validation/OOF PnL used for tuning or gate: `false`
- #229/#231 outcomes opened: `false`
- future confirmatory started: `false`

## Family Diagnostics

### SELL_BEFORE_CLOSE

- fit/calibration markets: `89 / 45`
- OOF relative MAE improvement: `0.0006721909457764329`
- OOF relative MSE improvement: `0.006801552922599793`
- calibration coverage: `0.8055555555555556`
- report-only selected PnL: `0`
- gate passed: `true`
- blockers: `[]`

### HOLD_TO_SETTLEMENT

- fit/calibration markets: `44 / 21`
- OOF relative MAE improvement: `-0.022633176069674445`
- OOF relative MSE improvement: `-0.009629154441057235`
- calibration coverage: `0.8134920634920635`
- report-only selected PnL: `0`
- gate passed: `false`
- blockers: `['hold_to_settlement_forward_oof_mae_improved_failed', 'hold_to_settlement_forward_oof_mse_improved_failed']`

No paper/live/write/wallet/capital/handoff/source/freeze/promotion unlock.
