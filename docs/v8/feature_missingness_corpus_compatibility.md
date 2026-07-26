# Issue #257 Corpus Compatibility

Legacy frozen corpora and active frozen candidates retain their existing bytes,
schemas, and semantics. The feature builder's default path remains unchanged:
it does not add missingness fields unless causal `TradeTapeCoverageStatus`
evidence is explicitly supplied.

Every new candidate lineage that consumes provider-dependent trade-volume
features must opt into `require_feature_completeness=True`, bind the
`bigan-v8-feature-missingness-v1` contract hash, and declare whether the model
consumes the missingness indicators. Missing, timeout, truncated, censored, or
historical-backfill evidence produces null volume values and an explicit reason;
it never produces a valid numeric zero.

The compatibility boundary is additive. No historical raw row, normalized
feature row, manifest, model, threshold, guard, cost, sizing rule, or policy is
rewritten. Diagnostics are outcome-blind and cannot unlock paper, promotion,
live, write, wallet, capital, handoff, source/freeze changes, #134, or #146.
