# XGBoost v5 Phase 4 Runbook 12h Trade Records - 20260601T165038Z

## Summary

- Generated at UTC: `2026-06-02T01:48:50+00:00`
- Remote host: `ubuntu@54.250.242.139`
- Account-history source: `/Users/tcscoder/Downloads/Polymarket-History-2026-06-02.csv`
- Closed trades matched to account history: `6/6`
- Event-derived open positions: `0`
- Account cash-flow PnL: `-1.079153` USDC
- Executor theoretical PnL: `-0.670952` USDC
- Cash minus executor: `-0.408201` USDC

Account cash-flow PnL is the promotion/account-impact value. Executor PnL is retained only as a theoretical fill-price diagnostic until account-history reconciliation is wired into summaries.

The prior preflight `20260601T160428Z-stage4` row is excluded from this fresh 12h campaign. Its account cash-flow PnL was `-1.042689` USDC.

## Trade Records

| Stage | Market | Side | Entry UTC | Exit UTC | Entry Hash | Exit Hash | Buy USDC | Sell USDC | Account PnL | Executor PnL | Delta | Exit Reason |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Bitcoin Up or Down - June 1, 1:00PM-1:15PM ET | DOWN | 2026-06-01T17:03:20.516000+00:00 | 2026-06-01T17:10:25.782000+00:00 | `0xd4ecd4f9...f1050c1c` | `0xc8ea6be8...1846bd1c` | 1.036389 | 0.573230 | -0.463159 | -0.395833 | -0.067326 | exit_signal |
| 2 | Bitcoin Up or Down - June 1, 3:15PM-3:30PM ET | DOWN | 2026-06-01T19:20:21.347000+00:00 | 2026-06-01T19:26:06.298000+00:00 | `0x1db641cc...fbfc5cbc` | `0x350498e2...7356f304` | 1.040589 | 0.381100 | -0.659489 | -0.595238 | -0.064251 | soft_force_exit |
| 3 | Bitcoin Up or Down - June 1, 5:00PM-5:15PM ET | DOWN | 2026-06-01T21:03:15.063000+00:00 | 2026-06-01T21:04:18.588000+00:00 | `0x2e16d809...38c2f7ca` | `0x0e0489c1...862eb7f1` | 1.033589 | 1.080870 | 0.047281 | 0.115384 | -0.068103 | exit_signal |
| 3 | Bitcoin Up or Down - June 1, 5:15PM-5:30PM ET | DOWN | 2026-06-01T21:22:23.458000+00:00 | 2026-06-01T21:26:09.882000+00:00 | `0xa6992d28...3c672da8` | `0x85f26ec1...dac2f4d8` | 1.034289 | 1.044050 | 0.009761 | 0.078431 | -0.068670 | soft_force_exit |
| 4 | Bitcoin Up or Down - June 1, 7:15PM-7:30PM ET | DOWN | 2026-06-01T23:21:25.524000+00:00 | 2026-06-01T23:25:32.322000+00:00 | `0xb1db68e7...65137e67` | `0x598c6b9f...15c533eb` | 1.039889 | 1.447390 | 0.407501 | 0.488372 | -0.080871 | exit_signal |
| 5 | Bitcoin Up or Down - June 1, 9:00PM-9:15PM ET | DOWN | 2026-06-02T01:04:24.831000+00:00 | 2026-06-02T01:06:28.353000+00:00 | `0xa21c223e...6f0d2fdf` | `0xa6dc2fc5...b57f291b` | 1.029388 | 0.608340 | -0.421048 | -0.362069 | -0.058979 | exit_signal |

Full transaction hashes, order ids, token amounts, latency fields, dust values, and source account-history rows are in `docs/reports/xgboost_v5_phase4_runbook_12h_20260601_trade_records.json`.

## Readout

- All matched filled trades in this slice are `BTC-15M DOWN`.
- CSV/account cashflow is worse than executor theoretical PnL by roughly `0.058979` to `0.080871` USDC per trade.
- The systematic delta means Phase 4 summaries should not be used for promotion PnL until they consume account cashflow or an equivalent hash reconciliation.
