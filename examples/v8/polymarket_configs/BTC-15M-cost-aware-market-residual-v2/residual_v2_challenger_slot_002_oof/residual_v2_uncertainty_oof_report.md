# BTC 15m market-anchored residual v2 uncertainty challenger slot 002

- All OOF gates passed: `False`
- OOF markets: `600`
- Candidate accepted markets: `20`
- Candidate total unit PnL: `1.34150000`
- Matched baseline total unit PnL: `-7.39425000`
- Paired delta total: `8.73575000`
- Candidate 97.5% LCB: `-0.00340053`
- Paired-delta 97.5% LCB: `-0.02115625`
- Conservative prospective required N: `32878`

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

## Architecture and governance

- Architecture: pooled side-symmetric market-anchored probability residual.
- Pair coherence: UP/DOWN probabilities normalized to sum to one before costs.
- Parent v1 gates and thresholds changed: `False`
- Candidate slots remaining after this evaluation: `1`
- Live shadow or fresh collection authorized by this report: `False`

## Uncertainty challenger and candidate budget

- Fixed heads: conditional mean, q25, q75.
- Action-value uncertainty deduction: `max(0, q75-q25)/2`.
- Acceptance threshold changed from zero: `False`
- Second and final candidate slot consumed: `True`
- Additional candidate allowed in this lineage: `False`
