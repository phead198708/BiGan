# BTC 15m prequential pooled residual v4 primary slot 001

- All OOF gates passed: `False`
- OOF markets: `600`
- Candidate accepted markets: `577`
- Candidate total unit PnL: `25.92625000`
- Matched baseline total unit PnL: `-7.39425000`
- Paired delta total: `33.32050000`
- Candidate 97.5% LCB: `0.00962138`
- Paired-delta 97.5% LCB: `0.01616786`
- Conservative prospective required N: `2999`

## Frozen gates

- absolute_market_bootstrap_97_5pct_lcb_gt_zero: `True`
- paired_delta_market_bootstrap_97_5pct_lcb_gt_zero: `True`
- every_chronological_block_candidate_total_gte_zero: `True`
- every_chronological_block_paired_delta_total_gte_zero: `False`
- largest_winner_removed_candidate_total_gte_zero: `True`
- largest_positive_delta_removed_total_gte_zero: `True`
- stable_score_to_realized_pnl_ordering: `True`
- all_cost_stress_candidate_totals_gte_zero: `True`
- all_cost_stress_paired_delta_totals_gte_zero: `True`
- prospective_power_required_market_count_lte_2000: `False`
- population_and_leakage_reconciliation: `True`

This is outcome-aware development evidence only. It is permanently ineligible for promotion evidence and does not authorize live shadow, fresh collection, paper/live execution, wallet signing, writes, or capital risk.

## Architecture and governance

- Architecture: two pooled side-symmetric probability learners with convex blending.
- Weight update: horizon-adaptive Hedge using strictly prior OOF normalized log loss.
- Initial weights: `0.5 / 0.5`; weight, parameter and threshold search: `False`
- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`
- Parent v1/v2/v3 failed artifacts changed: `False`
- Candidate slots remaining after this evaluation: `1`
- Next-stage authorization required even if every gate passes: `True`
- Collection, shadow, paper/live, wallet, write, promotion or capital authorized: `False`
