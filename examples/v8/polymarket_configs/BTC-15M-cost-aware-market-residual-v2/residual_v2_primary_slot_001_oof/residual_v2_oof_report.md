# BTC 15m market-anchored residual v2 primary slot 001

- All OOF gates passed: `False`
- OOF markets: `600`
- Candidate accepted markets: `578`
- Candidate total unit PnL: `26.73600000`
- Matched baseline total unit PnL: `-7.39425000`
- Paired delta total: `34.13025000`
- Candidate 97.5% LCB: `0.01064081`
- Paired-delta 97.5% LCB: `0.01712973`
- Conservative prospective required N: `2764`

## Frozen gates

- absolute_market_bootstrap_97_5pct_lcb_gt_zero: `True`
- paired_delta_market_bootstrap_97_5pct_lcb_gt_zero: `True`
- every_chronological_block_candidate_total_gte_zero: `True`
- every_chronological_block_paired_delta_total_gte_zero: `True`
- largest_winner_removed_candidate_total_gte_zero: `True`
- largest_positive_delta_removed_total_gte_zero: `True`
- stable_score_to_realized_pnl_ordering: `True`
- all_cost_stress_candidate_totals_gte_zero: `True`
- all_cost_stress_paired_delta_totals_gte_zero: `True`
- prospective_power_required_market_count_lte_2000: `False`
- population_and_leakage_reconciliation: `True`

This is outcome-aware development evidence only. It is permanently ineligible for promotion evidence and does not authorize live shadow, fresh collection, paper/live execution, wallet signing, writes, or capital risk.

## Architecture and governance

- Architecture: pooled side-symmetric market-anchored probability residual.
- Pair coherence: UP/DOWN probabilities normalized to sum to one before costs.
- Parent v1 gates and thresholds changed: `False`
- Candidate slots remaining after this evaluation: `1`
- Live shadow or fresh collection authorized by this report: `False`
