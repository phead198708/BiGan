# BTC-15M-cost-aware-market-residual-v5 terminal review

The v5 lineage is terminal failed. Both authorized candidate slots are consumed, no candidate is selected, and candidate freeze is not allowed.

## Slot outcomes

- Slot 1 failed closed before its first model fit because the frozen canonical JSON feature object did not preserve semantic `FEATURE_NAMES` insertion order. It produced no metrics and no gate decision. The failure record is immutable.
- Slot 2 used a frozen deterministic adapter that resolves the same values in explicit `FEATURE_NAMES` order. The candidate algorithm, model parameters, threshold, costs, baseline, population and gates were unchanged.
- Slot 2 evaluated 600 OOF markets, accepted 585, produced total unit net PnL `+28.21975000`, and paired delta `+35.61400000`.
- Absolute 97.5% bootstrap LCB was `+0.01263104`; paired-delta LCB was `+0.01646790`.
- Largest-winner-removed candidate PnL was `+27.30500000`; largest-positive-delta-removed paired PnL was `+33.85400000`.
- Ten of eleven frozen gates passed. The only failure was `prospective_power_required_market_count_lte_2000`: required N was `2598` against immutable `N_max=2000`.

## Comparison with v4

v4 slot 2 required `2488` markets. v5 required `2598`, a deterioration of `110` markets despite a small increase in observed total PnL. The residual corrector therefore did not improve the conservative effect-to-variance ratio.

## Reconciliation and limitation

An independent reconstruction verified all file hashes and sidecars, the 2,400 prediction rows, six fold audits, 600 paired market rows, shared bootstrap indices, every gate result, the JSON report and Markdown rendering. The frozen built-in verifier has a field-name adapter limitation: it expects the legacy `target_or_future_label_used_for_fit` flag, while v5 correctly records `current_or_future_label_used_for_corrector=false`. The audit mapped this equivalent flag in memory only; no frozen artifact was modified.

## Governance decision

No third v5 candidate is permitted. No collection, new outcome opening, shadow, paper/live execution, wallet signing, Polymarket write, promotion, handoff or capital risk is authorized. A further candidate lineage requires separate explicit user authorization.
