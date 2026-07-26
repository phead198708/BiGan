# Issue #254 Parallel Future Gate

The first statistically eligible future attempt freezes one shared,
strictly-later, target-free source grid and three immutable decision streams:

- `v8_1_primary_no_fallback`: primary abstention remains `NO_TRADE`;
- `v8_3_primary_with_fallback`: the frozen v8.3 fallback contract is reproduced;
- `matched_frozen_v6_7`: the legacy baseline uses the same rows, costs, sizing,
  guards, and runtime semantics.

All decisions and their canonical hashes are written before any outcome,
settlement, return, label, or PnL access. Evaluation consumes the freeze once.
Each tested candidate has a Bonferroni-adjusted alpha of 0.0125. Insufficient
support is terminal for the window and cannot trigger an extension.

Reports separate primary, fallback, abstention, and no-bet contributions. A
multiplicity-aware winner is diagnostic evidence only: issue #254 never
unlocks paper, promotion, live, write, wallet, capital, handoff, source/freeze,
#134, or #146 permissions.
