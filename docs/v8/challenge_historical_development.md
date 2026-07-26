# v8.1 Challenge historical-development runbook

This workflow is development-only. Historical outcomes can never promote a
candidate. A complete historical pass may only unlock preregistration of a new
future `attempt-002`; it never starts collection.

## Frozen state

- `attempt-001` is closed in issue `#262`.
- The outcome-blind collector wrote 12 raw captures. All 12 are permanently
  excluded; quality-valid and promotion-eligible capture counts are both zero.
- The exact-195 corpus and every other outcome-opened corpus in the registry are
  development data forever.
- Paper, live, write, wallet, handoff, collection, and promotion remain false.
- The immutable ledger genesis allows at most five preregistered development
  evaluations. After five, stop for comprehensive review.

The governing files are:

- `challenge_attempt_001_closure.json`
- `challenge_historical_development_data_registry.json`
- `challenge_historical_development_exact_195_market_ids.txt`
- `challenge_historical_development_success_standard.json`
- `challenge_historical_development_iteration_ledger.json`

Each file has a sibling `.sha256` sidecar. Do not rewrite these files after the
freeze commit. Each completed iteration creates a separate hash-chained entry.

## Success gates

The evaluator uses all 195 markets in the frozen chronological order and counts
`NO_TRADE` as zero after-cost PnL.

1. The 97.5% paired-bootstrap LCB of candidate minus matched v6.7 total PnL is
   strictly positive.
2. The 97.5% bootstrap LCB of candidate total PnL is strictly positive.
3. Candidate total PnL remains positive after removing its largest winner.
4. For each chronological half, the one-sided 97.5% bootstrap UCB is
   nonnegative, so the half is not significantly negative.
5. Support is consistent with the future protocol. The frozen choice is a
   full-window paired protocol with no minimum accepted-support count.
6. UP/DOWN/NONE and single-market concentration metrics are reported but are
   not hard gates.

Attempt-002 must later use structurally identical full-window paired and
absolute-LCB gates on a not-yet-collected future window.

## One development iteration

One iteration uses two ordered commits:

1. Add and commit one preregistration JSON plus its SHA-256 sidecar before
   changing the candidate. The preregistration explains what will change, why,
   and the expected mechanism. It must declare one candidate, no grid search,
   no result-selected parameter search, and all safety flags false. Its
   `implementation_commit` field is the prechange parent commit and its
   `implementation_commit_role` is `prechange_base_commit`.
2. Implement exactly that one candidate and commit it without rewriting the
   preregistration or sidecar.

The evaluator requires a clean worktree. It verifies that only the committed
preregistration and sidecar differ between the prechange base and the
preregistration commit, and that candidate implementation changes follow that
commit. This enforces preregistration before candidate change without creating
a self-referential Git hash.

Run one evaluation:

```bash
PYTHONPATH=src python examples/v8/run_challenge_historical_development.py \
  --run-id challenge-v8-1-historical-development-iteration-001 \
  --iteration-number 1 \
  --candidate-id <candidate-id> \
  --comparison-rows <exact-195-comparison.jsonl> \
  --comparison-rows-sha256 <sha256> \
  --preregistration <committed-preregistration.json> \
  --preregistration-sha256 <sha256> \
  --evaluated-at <UTC timestamp>
```

For iterations 2-5, also supply the immediately preceding entry:

```bash
  --previous-entry <previous-iteration-entry.json> \
  --previous-entry-sha256 <previous-entry-semantic-sha256>
```

The command exits `0` only when every historical gate passes and `2` for a
valid evaluation that does not pass. Either result consumes one development
iteration slot and must be retained in the alpha-spending ledger. A failure
does not authorize an unregistered threshold scan.

## Terminal decisions

- If an iteration passes all gates, stop historical iteration and prepare an
  additive attempt-002 preregistration. Do not collect until that protocol is
  reviewed and explicitly authorized.
- If five iterations are consumed without a pass, stop and perform a
  comprehensive review.
- Never use a historical report or iteration entry as promotion evidence.
