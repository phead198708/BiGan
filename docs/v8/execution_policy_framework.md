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
risk-state hashes must match. Every accepted or replacement transition creates
paper-only intents, fills, positions, and ledger deltas which reconcile before
the replay passes.

This framework creates no candidate permission by itself. Paper, promotion,
live, write, wallet, capital, handoff, source/freeze, #134, and #146 remain
closed until a separately preregistered fresh evidence program passes.
