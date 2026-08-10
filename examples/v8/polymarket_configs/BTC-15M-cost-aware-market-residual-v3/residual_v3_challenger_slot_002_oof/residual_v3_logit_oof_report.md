# BTC 15m logit-offset residual v3 challenger slot 002

- All OOF gates passed: `False`
- OOF markets: `600`
- Candidate accepted markets: `559`
- Candidate total unit PnL: `25.78975000`
- Matched baseline total unit PnL: `-7.39425000`
- Paired delta total: `33.18400000`
- Candidate 97.5% LCB: `0.00937398`
- Paired-delta 97.5% LCB: `0.01506905`
- Conservative prospective required N: `3043`

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

- Added causal engineered feature values and explicit missingness indicators: `12 + 12`
- Fixed exponential recency half-life in markets: `200`
- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`
- Parent v1/v2 failed artifacts changed: `False`
- Candidate slots remaining after this evaluation: `1`
- Next-stage authorization required even if every gate passes: `True`
- Collection, shadow, paper/live, wallet, write, promotion or capital authorized: `False`

## Logit-offset challenger and candidate budget

- Training likelihood: fixed binary log loss.
- Per-row base margin: `logit(decision_time_selected_mid)`.
- Source feature bytes and native NaN semantics reused: `True`
- Acceptance threshold changed from zero: `False`
- Second and final candidate slot consumed: `True`
- Additional candidate allowed in this lineage: `False`
