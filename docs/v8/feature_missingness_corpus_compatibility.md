# Issue #257 Corpus Compatibility

Legacy frozen corpora and active frozen candidates retain their existing bytes,
schemas, and semantics. Rows without trade-tape coverage metadata remain on the
legacy path. New captures carrying the #230 websocket/paginated-REST coverage
proof automatically derive a causal `TradeTapeCoverageStatus` for each decision
timestamp and add the #257 fields without changing the frozen model's selected
feature columns.

Every new candidate lineage that consumes provider-dependent trade-volume
features must opt into `require_feature_completeness=True`, bind the
`bigan-v8-feature-missingness-v1` contract hash, and declare whether the model
consumes the missingness indicators. Missing, timeout, truncated, censored, or
historical-backfill evidence produces null volume values and an explicit reason;
it never produces a valid numeric zero.

The persistent challenge collector counts a market as quality-valid only when
the websocket trade stream proves full-round continuity, has zero timestamp
causality violations, and is not censored. A complete websocket tape remains
valid if the read-only REST reconciliation times out or reaches its pagination
limit; REST evidence is never made available retroactively to an earlier
decision.

Every development canary writes hash-bound provider-health rows plus feature
completeness and missing-versus-zero diagnostics. Frozen-model canaries and the
final promotion-evidence runner additionally bind selection, fallback,
`NO_TRADE`, execution-guard, side, and action-family attribution to those rows.
All selected promotion decisions must reconcile to the policy-grid feature
identity before promotion can pass.

The compatibility boundary is additive. No historical raw row, normalized
feature row, manifest, model, threshold, guard, cost, sizing rule, or policy is
rewritten. Diagnostics are outcome-blind and cannot unlock paper, promotion,
live, write, wallet, capital, handoff, source/freeze changes, #134, or #146.
