# BTC 15m nested soft-stacking residual v4 challenger slot 002

- All OOF gates passed: `False`
- OOF markets: `600`
- Candidate accepted markets: `563`
- Candidate total unit PnL: `28.17225000`
- Matched baseline total unit PnL: `-7.39425000`
- Paired delta total: `35.56650000`
- Candidate 97.5% LCB: `0.01277441`
- Paired-delta 97.5% LCB: `0.01821378`
- Conservative prospective required N: `2488`

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

## Architecture and terminal candidate budget

- Architecture: nested rolling-origin L2 logistic soft stacker over two pooled base learners.
- Meta inputs: base probability logits from strictly prior inner-OOF rows only.
- Fixed L2 regularization: `20`; parameter or threshold search: `False`
- Hard routing, side filters, missingness filters and outlier deletion: `False`
- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`
- Second and final v4 candidate slot consumed: `True`
- Collection, shadow, paper/live, wallet, write, promotion or capital authorized: `False`
