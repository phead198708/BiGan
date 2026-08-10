# BTC 15m cost-aware residual v2 development terminal review

- Phase 1 terminal failed: `True`
- Candidate budget consumed: `2 / 2`
- Candidate selected/frozen: `None / False`
- Live shadow allowed: `False`
- Fresh confirmatory collection authorized: `False`

## Primary market-anchored residual

- Accepted markets: `578`
- Candidate total unit PnL: `26.73600000`
- Paired delta total: `34.13025000`
- Required prospective N: `2764`
- Failed gates: `prospective_power_required_market_count_lte_2000`

## Uncertainty challenger

- Accepted markets: `20`
- Candidate total unit PnL: `1.34150000`
- Paired delta total: `8.73575000`
- Required prospective N: `32878`
- Failed gates: `absolute_market_bootstrap_97_5pct_lcb_gt_zero, paired_delta_market_bootstrap_97_5pct_lcb_gt_zero, every_chronological_block_candidate_total_gte_zero, every_chronological_block_paired_delta_total_gte_zero, stable_score_to_realized_pnl_ordering, prospective_power_required_market_count_lte_2000`

No gate, threshold, population, failed report, safety permission, or slot budget was changed. No candidate may enter live shadow or fresh collection.
