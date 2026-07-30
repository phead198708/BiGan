# BTC 15m MoE collection distribution shift

- Role: outcome-blind diagnostic monitoring only
- Reporting timestamp: `2026-07-30T10:28:45.920961+00:00`
- Candidate bundle: `fa6b1429e22b26a7aba32be264431ace0818a4cf613043e3f7e054a5c837b807`
- Development markets: `113`
- Collection markets: `7`
- Development population hash: `afa913aa3655bda94a426ee44ff0499daf4d56c2e9ff02a9b33f47f2771d5163`
- Collection population hash: `1cac873d680fe8d5ff199005c475f1b8808c8493ad22e80759f78f0fcd0232c1`

## Direction regime

| Regime | Development | Collection | Delta (pp) |
|---|---:|---:|---:|
| bullish | 48 (42.48%) | 4 (57.14%) | +14.66 |
| bearish | 54 (47.79%) | 0 (0.00%) | -47.79 |
| sideways_or_unknown | 11 (9.73%) | 3 (42.86%) | +33.12 |

## Requested route

| Route | Development | Collection | Delta (pp) |
|---|---:|---:|---:|
| high_vol | 36 (31.86%) | 0 (0.00%) | -31.86 |
| bullish | 28 (24.78%) | 4 (57.14%) | +32.36 |
| bearish | 40 (35.40%) | 0 (0.00%) | -35.40 |
| low_vol | 9 (7.96%) | 3 (42.86%) | +34.89 |

- Development fallback ratio: `0.079646`
- Collection fallback ratio: `0.428571`

## Important feature missingness

| Feature group | Development missing | Collection missing | Delta |
|---|---:|---:|---:|
| recent_trade_volume | 0.7965 | 1.0000 | +0.2035 |
| opposite_trade_volume | 0.7965 | 1.0000 | +0.2035 |
| orderbook_depth | 0.0000 | 0.0000 | +0.0000 |
| trade_tape_coverage | 0.7965 | 1.0000 | +0.2035 |
| btc_feature_coverage | 0.0000 | 0.0000 | +0.0000 |
| chainlink_reference_coverage | 0.0000 | 0.0000 | +0.0000 |

## Collection provider quality

- Raw market coverage: `1.000000`
- Orderbook coverage: `1.000000`
- Trade availability: `1.000000`
- Paired executable ask coverage: `1.000000`
- Retry rate: `0.142857`
- Invalid attempt rate: `0.000000`
- Causality violations: `0`

- No drift threshold or materiality gate is assigned.
- No market is filtered or reordered.
- Monitoring influences collection: false
- Monitoring influences model: false
- Outcomes accessed: false
- Settlement accessed: false
- PnL accessed: false

## Safety

- source_model_candidate_eligible=false
- freeze_ready=false
- promotion_evidence_eligible=false
- paper_candidate_allowed=false
- v8_execution_handoff_allowed=false
- live_trading_allowed=false
- wallet_signing_allowed=false
- polymarket_write_allowed=false
- capital_at_risk=false
