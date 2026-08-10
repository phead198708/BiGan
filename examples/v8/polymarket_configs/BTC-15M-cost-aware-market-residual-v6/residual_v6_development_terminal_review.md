# BTC-15M-cost-aware-market-residual-v6 terminal review

The v6 lineage is terminal failed. Both authorized candidate slots are consumed, no candidate is selected, and candidate freeze is not allowed.

## Slot outcomes

- Slot 1 replaced independent early/late action scoring with nested fitted-Q dynamic stopping. Across 600 OOF markets it accepted 582, produced total unit net PnL `+8.81500000`, and paired delta `+16.20925000`.
- Slot 1 absolute and paired 97.5% bootstrap LCBs were `-0.01859739` and `-0.01731112`; six frozen gates failed and required N was `24673`.
- Slot 2 retained the frozen v5 side and decision proposal and applied a strictly prior conditional 40th-percentile unit-PnL quality model. It accepted 371, produced total unit net PnL `+18.35225000`, and paired delta `+25.74650000`.
- Slot 2 absolute and paired LCBs were positive at `+0.00310857` and `+0.00731268`. Score ordering, cost stress, largest-winner removal, and paired block stability passed.
- Slot 2 still failed `every_chronological_block_candidate_total_gte_zero` because block 2 PnL was `-0.85125000`, and failed `prospective_power_required_market_count_lte_2000` because required N was `3778`.

## Governance decision

No third v6 candidate is permitted. The existing gates, zero threshold, N_max=2000, costs, baseline, population, parent failures and safety state remain unchanged. No collection, outcome opening, shadow, paper/live execution, wallet signing, Polymarket write, promotion, handoff or capital risk is authorized. A new development lineage requires separate explicit user authorization.
