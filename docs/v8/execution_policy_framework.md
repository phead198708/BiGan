# Issue #256 Execution-Policy Framework

The source model and execution policy are independently hash-bound. The policy
may rank immutable source scores and apply causal abstention, opportunity,
exposure, cooldown, fill-quality, provider-health, replacement, sizing, and
kill-switch rules. It cannot mutate the source model or scores, consume target
information, relax execution safety, or search future outcomes.

Three bounded fixtures demonstrate materially different behavior:

- strict high-signal adaptive abstention;
- per-window opportunity and portfolio risk budgeting;
- fill-quality-aware fixed-risk position replacement.

Offline replay and paper runtime use the same state machine. Exact decision and
risk-state hashes must match. Parity derives those hashes again from the raw
rows; caller-supplied stream digests are never accepted as proof. Decision
attribution binds the source-input, source-score, source-model, policy, and
decision hashes. Risk rows form a previous-hash chain, each `before` state must
equal the preceding `after` state, and the final state must equal the reported
positions.

Every accepted or replacement transition creates paper-only intents, fills,
positions, and ledger deltas. Reconciliation checks the full identifier chain,
decision linkage, transition shape, and both per-market and per-side exposure,
not only the global notional sum. Missing, unknown, mistyped, non-finite,
provider-incomplete, forbidden future/target, and active kill-switch inputs all
fail closed to attributed `NO_TRADE`.

The contract, compatibility manifest, three-candidate manifest, each fixture,
and future-validation template receive semantic validation in addition to raw
SHA-256 pinning. Omitted safety fields, relaxed flags, malformed budget
constraints, changed candidate identities, family drift, and a reordered
outcome-blind freeze protocol are rejected.

This framework creates no candidate permission by itself. Paper, promotion,
live, write, wallet, capital, handoff, source/freeze, #134, and #146 remain
closed until a separately preregistered fresh evidence program passes. It does
not start or resume the 120-round collection.
