# BTC 15m lower-quantile residual v6 challenger slot 002

- All OOF gates passed: `False`
- OOF markets: `600`
- Candidate accepted markets: `371`
- Candidate total unit PnL: `18.35225000`
- Matched baseline total unit PnL: `-7.39425000`
- Paired delta total: `25.74650000`
- Candidate 97.5% LCB: `0.00310857`
- Paired-delta 97.5% LCB: `0.00731268`
- Conservative prospective required N: `3778`

## Frozen gates

- absolute_market_bootstrap_97_5pct_lcb_gt_zero: `True`
- paired_delta_market_bootstrap_97_5pct_lcb_gt_zero: `True`
- every_chronological_block_candidate_total_gte_zero: `False`
- every_chronological_block_paired_delta_total_gte_zero: `True`
- largest_winner_removed_candidate_total_gte_zero: `True`
- largest_positive_delta_removed_total_gte_zero: `True`
- stable_score_to_realized_pnl_ordering: `True`
- all_cost_stress_candidate_totals_gte_zero: `True`
- all_cost_stress_paired_delta_totals_gte_zero: `True`
- prospective_power_required_market_count_lte_2000: `False`
- population_and_leakage_reconciliation: `True`

This is outcome-aware development evidence only. It is permanently ineligible for promotion evidence and does not authorize live shadow, fresh collection, paper/live execution, wallet signing, writes, or capital risk.

## Proposal-quality architecture

- Proposal source: frozen v5 side and decision action.
- Risk target: strictly prior conditional 40th percentile of unit PnL.
- First block: frozen v5 identity because no prior v5 OOF proposal labels exist.
- Grid, feature, weight or threshold search: `False`
- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`
- Candidate budget exhausted after this evaluation: `True`
- Collection, shadow, paper/live, wallet, write, promotion or capital authorized: `False`
