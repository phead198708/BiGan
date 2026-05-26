# Polymarket Cash-Flow Reconciliation

## Summary

- Positions: 15
- Matched cash-flow rows: 15
- Missing cash-flow rows: 0
- Ambiguous redeem rows: 0
- Account cash-flow PnL: -0.500671
- Theoretical fill-price PnL: -1.061990
- Cash minus theoretical: -0.779941
- Dust token amount: 0.023387

## Positions

| Event | Status | Side | Match | Account PnL | Theoretical PnL | Delta | Dust |
|---|---|---|---|---:|---:|---:|---:|
| phase4-btc-updown-15m-1779771600-UP-dabd1b92 | closed | UP | matched | -0.071769 | 0.000000 | -0.071769 | 0.000815 |
| phase4-btc-updown-15m-1779772500-DOWN-042ec3c3 | closed | DOWN | matched | -0.089980 | -0.020000 | -0.069980 | 0.000000 |
| phase4-btc-updown-15m-1779772500-UP-28f9fe94 | closed | UP | matched | -0.092129 | -0.020408 | -0.071721 | 0.000815 |
| phase4-btc-updown-15m-1779774300-UP-a7fc2f63 | open | UP | matched | 1.090569 | - | - | 0.000000 |
| phase4-btc-updown-15m-1779776100-DOWN-ea5eb6c5 | closed | DOWN | matched | -0.089980 | -0.020000 | -0.069980 | 0.000000 |
| phase4-btc-updown-15m-1779777000-UP-7b1e1f1d | closed | UP | matched | 0.092111 | 0.163265 | -0.071154 | 0.000815 |
| phase4-btc-updown-15m-1779777900-UP-6324d07b | closed | UP | matched | -0.334189 | -0.265306 | -0.068883 | 0.000815 |
| phase4-btc-updown-15m-1779778800-UP-a5d012f5 | open | UP | matched | -1.035000 | - | - | 0.000000 |
| phase4-btc-updown-15m-1779779700-UP-98231ca9 | open | UP | matched | 1.285691 | - | - | 0.000000 |
| phase4-btc-updown-15m-1779780600-UP-ecb2d49c | expired | UP | matched | -1.046889 | -1.000000 | -0.046889 | 0.000000 |
| phase4-btc-updown-15m-1779782400-UP-c88ea957 | closed | UP | matched | 0.669471 | 0.729166 | -0.059695 | 0.003332 |
| phase4-btc-updown-15m-1779783300-UP-c2e54d62 | closed | UP | matched | 0.768251 | 0.829787 | -0.061536 | 0.007658 |
| phase4-btc-updown-15m-1779784200-DOWN-9f1adb81 | closed | DOWN | matched | -0.049980 | 0.020000 | -0.069980 | 0.000000 |
| phase4-btc-updown-15m-1779785100-UP-53969803 | closed | UP | matched | -0.880699 | -0.833333 | -0.047366 | 0.003332 |
| phase4-btc-updown-15m-1779786000-UP-10f86d62 | closed | UP | matched | -0.716149 | -0.645161 | -0.070988 | 0.005805 |

Account PnL uses Polymarket account-history cash flow: `BUY=-usdcAmount`, `SELL=+usdcAmount`, `REDEEM=+usdcAmount`.
Theoretical PnL is kept separate because it is derived from executor fill price and size.
