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

`parallel_future_collection_plan.json` is the concrete first fresh-attempt
preregistration. It pins the collector and feature contracts, all three
candidate contracts, the implementation commit, a strictly-later millisecond
boundary, 120 quality-valid markets, a maximum of 180 attempts, and 12-market
batches. During collection, candidate scoring, settlement finalization,
resolution access, labels, outcomes, returns, and PnL remain disabled.

Reports separate primary, fallback, abstention, and no-bet contributions. A
multiplicity-aware winner is diagnostic evidence only: issue #254 never
unlocks paper, promotion, live, write, wallet, capital, handoff, source/freeze,
#134, or #146 permissions.

The executable entry point is `examples/v8/run_parallel_future_gate.py`:

- `validate-plan` verifies the preregistration and every bound raw-file hash;
- `freeze` writes one shared source grid and three immutable decision streams;
- `evaluate` consumes a freeze once and writes the multiplicity-aware report;
- `legacy-smoke` exercises the full evaluator on an already-consumed v8.3
  window while explicitly keeping promotion eligibility and alpha consumption
  false.

The production collector bridge is
`examples/v8/run_challenge_future_freeze.py`:

- `status` validates the append-only collector index and reports exact-window
  readiness without writing artifacts or opening targets;
- `freeze` snapshots the ready index, verifies every selected raw descriptor
  and matched development/v6.2 batch manifest, binds the exact historical
  v8.1 model bytes and initial controller state, reconstructs v8.1, v8.3, and
  matched v6.7 decisions on one source grid, and writes the canonical parallel
  freeze before settlement access.

`freeze` requires the exact historical fit manifest named by
`parallel_frozen_v8_1_model_binding.json`. It also requires a clean committed
worktree so the implementation commit in the freeze manifest identifies the
code that produced the decision streams.
