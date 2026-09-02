# Issue #258 Regime-Stratified Diagnostics

Regime assignments are frozen from decision-time inputs using the hash-pinned
`regime_definition_contract.json`. Boundaries cannot change after outcome or
PnL access. Missing inputs receive explicit `unknown` strata; future inputs or
target-like fields fail closed.

Each final report includes side, reference direction, bullish/bearish/sideways,
realized volatility, spread/liquidity, UTC time-of-day, provider health,
primary/fallback/abstention, and action-family partitions. Empty strata are
present, low-support strata are marked `insufficient_support`, and every
mutually exclusive dimension must reconcile support and after-cost PnL exactly
to the aggregate.

All bucket thresholds, including provider-health healthy/degraded boundaries,
are read from the frozen contract. Persisted assignments are revalidated by
their assignment hash and contract hash before reporting, and their selected
side, decision origin, and action family must still match the candidate row.
Non-finite candidate or baseline PnL fails closed. The primary, bootstrap, and
side/action reports each carry the full closed safety state.

These metrics are descriptive. They do not add a side quota or alter the
aggregate hard gate. No paper, promotion, live, write, wallet, capital,
handoff, source/freeze, #134, or #146 permission follows from a stratum result.
