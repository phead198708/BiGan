# BTC 15m cost-aware residual v3 development terminal review

- Phase 1 terminal failed: `True`
- Candidate budget consumed: `2 / 2`
- Candidate selected/frozen: `None / False`
- Live shadow allowed: `False`
- Fresh collection authorized: `False`

## Primary causal time-adaptive residual

- Accepted markets: `573`
- Candidate total unit PnL: `23.32625000`
- Paired delta total: `30.72050000`
- Required prospective N: `3576`
- Failed gates: `every_chronological_block_candidate_total_gte_zero, every_chronological_block_paired_delta_total_gte_zero, prospective_power_required_market_count_lte_2000`

## Logit-offset challenger

- Accepted markets: `559`
- Candidate total unit PnL: `25.78975000`
- Paired delta total: `33.18400000`
- Required prospective N: `3043`
- Failed gates: `prospective_power_required_market_count_lte_2000`

No gate, zero threshold, N_max, cost, baseline, population, failed artifact, safety permission, or slot budget was changed. No candidate may enter live shadow or fresh collection.
