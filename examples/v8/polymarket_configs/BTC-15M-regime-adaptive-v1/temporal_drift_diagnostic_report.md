# BTC 15m temporal drift diagnostic

- Lineage: `BTC-15M-regime-adaptive-v1`
- Role: development-only parent OOF diagnosis; never promotion evidence
- Training performed: no
- New outcomes collected: no
- OOF markets: 73
- Accepted markets: 72
- Unit net PnL: 1.207000
- Bootstrap 95% interval: [-0.082169, 0.113385]

## Temporal instability

- first half: 35 accepted, PnL -2.428750, mean -0.067465
- second half: 37 accepted, PnL 3.635750, mean 0.098264
- Trading residual slope per chronological rank: -0.00441713
- Probability residual slope per chronological rank: -0.00441713

## Diagnosis

- Primary finding: `parent_global_edge_is_temporally_unstable`
- Regime dependence observed: `true`
- Liquidity dependence observed: `true`
- Time drift observed: `true`
- Recommended candidate family: `bounded_regime_adaptive_family_of_five`

## Governance

- Parent OOF remains immutable negative development evidence.
- This report does not validate a candidate or authorize training.
- Fresh strictly-later evidence remains mandatory.
