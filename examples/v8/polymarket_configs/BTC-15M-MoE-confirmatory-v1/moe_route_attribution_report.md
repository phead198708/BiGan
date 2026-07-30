# BTC 15m MoE route and fallback attribution

- Attribution reconciled: `true`
- Markets: 73
- Accepted: 72
- Total PnL: 4.372000
- Fallback share: 0.205479
- Native expert PnL: 2.425750
- Global fallback PnL: 1.946250

| Route | Markets | Fallback | PnL |
|---|---:|---:|---:|
| bearish | 24 | 0 | -2.050750 |
| bullish | 18 | 4 | 1.110500 |
| high_vol | 26 | 6 | 3.288500 |
| low_vol | 5 | 5 | 2.023750 |

## Fallback over time

| Q1 | Q2 | Q3 | Q4 |
|---:|---:|---:|---:|
| 0.526316 | 0.111111 | 0.111111 | 0.055556 |

## Expert support growth

| Route | First observed support | Last observed support |
|---|---:|---:|
| bearish | 23 | 56 |
| bullish | 14 | 39 |
| high_vol | 12 | 45 |
| low_vol | 7 | 17 |

## Concentration and missingness

- Native-expert largest winner: 0.834750
- Global-fallback largest winner: 0.734750
- Trade-volume missing markets: 60/73
- Depth missing markets: 0/73
- Spread missing markets: 0/73

This is diagnostic development attribution only. No router, expert, filter, support threshold, or fallback behavior was changed.
