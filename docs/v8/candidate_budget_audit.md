# Issue #255 Candidate-Family Budget Audit

The new `v8-high-signal-vs-fallback-2026q3` family starts with a fresh evidence
program. Its first strictly-later window may compare exactly two immutable
candidates in parallel. Attempt-one family-wise alpha is 0.025 and Bonferroni
allocates 0.0125 to each candidate. Later windows spend 0.015 and 0.01,
respectively. A fourth confirmatory attempt is rejected.

Repository-known prior future outcome windows for issues #238, #241, #246,
#249, and #250 are recorded as permanently consumed in hash-chained,
append-only ledgers. They cannot be reset by changing a branch, version, or
candidate name. The identity key binds source-model, execution-policy, and
candidate-definition hashes.

An identical engineering reproduction consumes no alpha. A changed decision
candidate consumes budget once its target is opened. A pre-target invalid
window or pre-target bug fix consumes no attempt; a bug discovered after target
access consumes the attempt and makes the old candidate hash terminal.

The committed machine decision says only that issue #254 is statistically
eligible to freeze and collect its first window. It does not grant paper,
promotion, live, write, wallet, capital, handoff, source/freeze, #134, or #146
permission.
