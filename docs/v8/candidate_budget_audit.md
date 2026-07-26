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

The repository validator now checks the semantics behind every hash:

- candidate identities are recomputed from source-model, execution-policy, and
  candidate-definition hashes, with duplicate identities rejected;
- the case table, attempt cap, preregistered alpha-spending sequence, Bonferroni
  allocation, and family-wise alpha sum are validated as one contract;
- an opened attempt must consume both its attempt and alpha and become
  terminal, while opened evidence must be permanently consumed and never
  reusable for promotion, even if an attacker recomputes a valid ledger hash
  chain after changing those fields;
- the proposed gate pins the ordered candidate IDs and their stable identities,
  so a branch/version rename or alias cannot reset history;
- every budget contract artifact and the machine decision carries the complete
  closed paper/live/write/wallet/capital/handoff/source/freeze/promotion/
  #134/#146 safety state. The append-only ledger bytes and entries are
  unchanged.

The committed machine decision says only that issue #254 is statistically
eligible to freeze and collect its first window. It does not grant paper,
promotion, live, write, wallet, capital, handoff, source/freeze, #134, or #146
permission.

This hardening does not change the accepted collection plan, pre-freeze
checklist, excluded-capture ledger, candidate set, attempt number, or alpha
allocation. Collection remains an independent operator-authorized action.
