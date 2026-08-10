# BTC 15m dynamic optimal-stopping residual v6 primary slot 001

- All OOF gates passed: `False`
- OOF markets: `600`
- Candidate accepted markets: `582`
- Candidate total unit PnL: `8.81500000`
- Matched baseline total unit PnL: `-7.39425000`
- Paired delta total: `16.20925000`
- Candidate 97.5% LCB: `-0.01859739`
- Paired-delta 97.5% LCB: `-0.01731112`
- Conservative prospective required N: `24673`

## Frozen gates

- absolute_market_bootstrap_97_5pct_lcb_gt_zero: `False`
- paired_delta_market_bootstrap_97_5pct_lcb_gt_zero: `False`
- every_chronological_block_candidate_total_gte_zero: `False`
- every_chronological_block_paired_delta_total_gte_zero: `False`
- largest_winner_removed_candidate_total_gte_zero: `True`
- largest_positive_delta_removed_total_gte_zero: `True`
- stable_score_to_realized_pnl_ordering: `False`
- all_cost_stress_candidate_totals_gte_zero: `True`
- all_cost_stress_paired_delta_totals_gte_zero: `True`
- prospective_power_required_market_count_lte_2000: `False`
- population_and_leakage_reconciliation: `True`

This is outcome-aware development evidence only. It is permanently ineligible for promotion evidence and does not authorize live shadow, fresh collection, paper/live execution, wallet signing, writes, or capital risk.

## Sequential architecture

- Late decision: direct after-cost action-value model.
- Early decision: incremental value over a strictly prior, inner-OOF late continuation policy.
- Outer target late features used for early scoring: `False`
- Grid, feature, weight or threshold search: `False`
- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`
- Collection, shadow, paper/live, wallet, write, promotion or capital authorized: `False`
