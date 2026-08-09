# BTC 15m cost-aware residual primary slot 001

- All OOF gates passed: `False`
- OOF markets: `600`
- Candidate accepted markets: `529`
- Candidate total unit PnL: `25.38575000`
- Matched baseline total unit PnL: `-7.39425000`
- Paired delta total: `32.78000000`
- Candidate 97.5% LCB: `0.01100899`
- Paired-delta 97.5% LCB: `0.01961065`
- Conservative prospective required N: `2696`

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

## Candidate budget

- Second and final preregistered slot consumed: `True`
- Additional candidate allowed: `False`
- Structural change: fixed lower-quantile training loss only; threshold, features, population, costs, bootstrap, and gates are unchanged.
