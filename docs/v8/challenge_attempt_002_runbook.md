# Challenge attempt-002 runbook

## Current status

Attempt-002 is preregistered but collection is not authorized and has not
started. Its collector PID is null, attempted and quality-valid counts are
zero, outcomes have not been opened, and all paper/live/write/wallet/handoff/
promotion/collection safety flags remain false.

The model is v8.1 with candidate ID
`v8_1_entry_price_floor_0_30_sized_1_0`: the v8.1 controller and terminal
execution-price floor `0.30` select the same trade set, and selected trades use
fixed size `1.0`. The matched frozen v6.7 champion remains at size `0.2`.

Governing additive artifacts:

- `challenge_attempt_002_preregistration.json`
- `challenge_attempt_002_synthetic_dry_run.json`
- `challenge_attempt_002_collection_execution_manifest.json`
- `challenge_attempt_002_promotion_execution_manifest.json`
- `challenge_historical_development_iteration_003_result.json`

Each has a SHA-256 sidecar. The earlier frozen attempt-001 plan, governance,
checklist, historical registry, and historical success standard are not
modified.

## Authorization boundary

Do not launch a collector from this preregistration alone. Collection requires
the operator's explicit permission for the 120-market run. After permission,
record it in a new, hash-pinned authorization artifact; do not rewrite the
preregistration.

The guarded supervisor is
`run_challenge_attempt_002_collection.py`. Both its `preflight` and `run`
subcommands require the separately hash-pinned authorization. Missing or
invalid authorization fails before the service root is created and before any
network operation. The supervisor:

- invokes the generic raw collector for exactly one 12-market batch at a time;
- stops as soon as the chronological quality-valid count reaches 120;
- fails closed at 180 attempted markets or 15 batches;
- never enables scoring, settlement, resolution, labels, or PnL;
- locks against concurrent supervisors and supports hash-bound resume;
- writes one issue-#260-ready summary per completed batch with PID, commit,
  frozen plan hash, attempted/valid counts, exclusion reasons, and provider
  health.

The future service root is:

```text
examples/v8/polymarket_live_runs/challenge-model-v8-1-attempt-002
```

Only markets with start timestamps strictly greater than
`1785088622878` are eligible. The target is exactly 120 chronological,
quality-valid BTC UP/DOWN 5-minute markets, collected in bounded batches of 12.
The maximum attempted count is 180. Batch summaries go to issue #260; there is
no per-round comment.

After explicit permission and creation of the authorization artifact, run the
read-only preflight first:

```bash
PYTHONPATH=src:. python examples/v8/run_challenge_attempt_002_collection.py \
  preflight \
  --run-id <collection-run-id> \
  --protocol-sha256 <attempt-002 protocol sha256> \
  --operator-authorization <authorization.json> \
  --operator-authorization-sha256 <sha256> \
  --collector-protocol-sha256 <collector protocol sha256> \
  --feature-contract-sha256 <feature contract sha256> \
  --service-root \
    examples/v8/polymarket_live_runs/challenge-model-v8-1-attempt-002
```

The collection `run` invocation is intentionally withheld until that
authorization artifact exists.

## Outcome-blind collection

During raw capture:

- do not score candidates;
- do not run settlement or a resolution provider;
- do not open labels, outcomes, or PnL;
- retain the same source market rows for candidate and baseline;
- verify raw artifact hashes and timestamp causality;
- do not extend the window based on observed results.

If infrastructure fails, fail closed or record a separate issue. Do not mutate
the preregistration, candidate, thresholds, features, sizing, or evaluation
code.

## Decision freeze and settlement

After exactly 120 quality-valid markets:

1. Run the post-collection target-free adapter. It verifies the terminal
   supervisor state and collector-index hash, reconstructs the earliest exact
   120 quality-valid markets, verifies every raw descriptor, performs frozen
   v6.2/v6.7/v8.1 scoring only after collection has stopped, applies the
   preregistered `0.30` entry-price floor and `1.0` sizing overlay, and freezes
   all 120 candidate and matched v6.7 decisions before target access. Every
   market stays in the comparison; `NO_TRADE` is zero.
2. Write the single-use target-access claim. This consumes attempt-002's
   one-sided alpha `0.025`.
3. Settle all 120 markets using official read-only resolution on quarantine
   copies, after market close. Costs are subtracted exactly once and source
   outcome-blind rows are never mutated.
4. Build one chronological 120-row paired comparison. Candidate selected-trade
   PnL is per-contract after-cost PnL times `1.0`; matched v6.7 PnL uses its
   frozen `0.2` size.

No manual code or artifact changes are allowed after target access.

## Single-use future gate

Run `evaluate_attempt_002_future_rows` from
`src/bigan/v8/polymarket/challenge_attempt_002.py` once on the frozen comparison.
Promotion evidence is eligible only if every hard gate passes:

1. paired market-bootstrap 97.5% LCB of candidate minus v6.7 is greater than
   zero;
2. candidate absolute market-bootstrap 97.5% LCB is greater than zero;
3. candidate total PnL after removing its largest winner remains greater than
   zero;
4. the 97.5% bootstrap UCB in each chronological 60-market half is
   nonnegative.

There is no accepted-support hard gate: all 120 markets participate in both
bootstrap gates. UP/DOWN/NONE distribution and single-market concentration are
reported only.

A passing result is future promotion evidence, not automatic promotion.
The promotion audit still runs separately and every safety flag remains false
until that audit succeeds.

The final audit is attempt-002-specific. It does not reuse attempt-001's
parallel-candidate alpha allocation or candidate identities.

## Executable evidence pipeline

`run_challenge_attempt_002_pipeline.py` controls evidence freezing and
evaluation only; it has no collector-start operation.

After an authorized collection reaches 120 quality-valid markets and the
collector PID is null, run the deterministic target-free adapter. All input
hashes are required explicitly, including the terminal supervisor state and
collector index. The v6.2 and historical v8.1 manifests remain immutable and
are consumed through their existing SHA-pinned descriptors:

```bash
PYTHONPATH=src:. python \
  examples/v8/run_challenge_attempt_002_target_freeze.py \
  --run-id <target-free-freeze-run-id> \
  --output-dir examples/v8/polymarket_runs \
  --service-root \
    examples/v8/polymarket_live_runs/challenge-model-v8-1-attempt-002 \
  --protocol-sha256 \
    0fa091610966a3a3470872a7e1b5832c8a32985fc312235366ad41aa891f249f \
  --supervisor-state <attempt_002_collection_supervisor_state.json> \
  --supervisor-state-sha256 <sha256> \
  --collector-index <persistent_outcome_blind_round_index.jsonl> \
  --collector-index-sha256 <sha256> \
  --feature-contract-sha256 \
    a4819ad6beec8d72612aa25ef2af751c357e807d514dcf1d2c94b37eba07c959 \
  --v6-2-candidate-manifest <frozen-v6.2-manifest.json> \
  --v6-2-candidate-manifest-sha256 \
    b9441b04fb595a927cbf9af9311612b037c36fc8c623ac8a92b6f4cb8ece84b9 \
  --historical-fit-manifest <frozen-v8.1-fit-manifest.json> \
  --historical-fit-manifest-sha256 \
    3fff5785a53cb32fb26d839786e3f48c2ff2bd7cc9dcf84e801c916a6ebb0fb7 \
  --frozen-model-binding-sha256 \
    64fa0f227ce97ec8ea238c3d8285d55efd96be65c95bf7c091aa95c1c185ccfd \
  --v8-1-candidate-contract-sha256 \
    b06919aadbea6821f44c1decf4f488fd3e34fc2612a115466caad3fe6173ad90 \
  --entry-price-floor-profile-sha256 \
    ea54d339c3ead15188a5fe1ede947e20e8f82cb422418f34a11277633180305e \
  --sizing-profile-sha256 \
    b04b25fb7dfad6a8949bd630f407abb156128ba59e33682816846344f1c130ff \
  --decision-freeze-created-ts <UTC epoch milliseconds after final close>
```

The adapter writes the shared source rows, exact decision-time feature rows,
v8.1 native decisions, candidate and baseline decisions, canonical
`attempt_002_target_free_pairs.jsonl`, an index snapshot, and a hash-indexed
manifest. It does not create a target-access claim, call settlement or a
resolution provider, or control collection. Any index change during the run
fails closed.

The real single-use target claim requires a separate
`challenge-attempt-002-operator-authorization-v1` artifact and its valid
sidecar. A missing or synthetic authorization cannot create a real claim:

```bash
PYTHONPATH=src:. python examples/v8/run_challenge_attempt_002_pipeline.py \
  claim \
  --target-free-pairs <target-free-pairs.jsonl> \
  --target-free-pairs-sha256 <sha256> \
  --operator-authorization <authorization.json> \
  --output <single-use-target-claim.json>
```

After official settlement writes exact action-level targets, run the gate once:

```bash
PYTHONPATH=src:. python examples/v8/run_challenge_attempt_002_pipeline.py \
  evaluate \
  --run-id <future-evaluation-run-id> \
  --target-free-pairs <target-free-pairs.jsonl> \
  --target-free-pairs-sha256 <sha256> \
  --target-access-claim <single-use-target-claim.json> \
  --target-access-claim-sha256 <sha256> \
  --settlement-targets <settlement-targets.jsonl> \
  --settlement-targets-sha256 <sha256> \
  --operator-authorization <authorization.json> \
  --evaluated-at <UTC timestamp>
```

The runner requires a clean committed worktree, validates every input hash,
rejects non-canonical decision and target row IDs, and writes an immutable
comparison, result, and manifest. Synthetic claims never consume promotion
alpha and can never emit promotion-eligible evidence, even when their
statistical gates pass.

## Supplemental issue evidence

After the real future gate has run, generate the preregistered diagnostic and
execution-policy evidence from the same frozen source grid. This step does not
change or rerun the future gate and does not make a promotion decision:

```bash
PYTHONPATH=src:. python examples/v8/run_challenge_attempt_002_supplemental.py \
  --run-id <supplemental-run-id> \
  --future-manifest <attempt_002_future_manifest.json> \
  --future-manifest-sha256 <sha256> \
  --operator-authorization <authorization.json> \
  --operator-authorization-sha256 <sha256> \
  --shared-source-rows <shared-source.jsonl> \
  --shared-source-rows-sha256 <sha256> \
  --feature-rows <decision-time-feature-rows.jsonl> \
  --feature-rows-sha256 <sha256> \
  --native-decisions <frozen-v8.1-native-decisions.jsonl> \
  --native-decisions-sha256 <sha256> \
  --regime-contract-sha256 <sha256> \
  --policy-manifest-sha256 <sha256> \
  --compatibility-manifest-sha256 <sha256> \
  --generated-at <UTC timestamp>
```

The generator requires exact market-grid reconciliation and complete
decision-time provider features. It creates:

- provider-health and feature-completeness diagnostics for issue #257;
- causal regime assignments, stratified PnL, bootstrap, and side/action
  reports for issue #258;
- all three preregistered execution-policy offline/paper replays, exact parity,
  safety, and reconciliation reports for issue #256;
- a hash-indexed supplemental runtime-evidence manifest.

All reports bind the attempt ID, v8.1 candidate ID, future-manifest SHA-256,
future-result SHA-256, and locked safety fields. Regime metrics remain
diagnostic only, and execution-policy performance cannot replace or alter the
attempt-002 future gate.

## Final promotion audit

Run the final audit exactly once against the immutable future manifest and
supplemental runtime-evidence manifest:

```bash
PYTHONPATH=src:. python examples/v8/run_challenge_attempt_002_promotion_audit.py \
  --future-manifest <attempt_002_future_manifest.json> \
  --future-manifest-sha256 <sha256> \
  --supplemental-runtime-evidence \
    <attempt_002_supplemental_runtime_evidence.json> \
  --supplemental-runtime-evidence-sha256 <sha256> \
  --output <attempt_002_promotion_readiness.json>
```

The audit independently verifies every descriptor hash, reconstructs the
settled 120-market comparison from the frozen decisions and targets, reruns the
preregistered bootstrap gate, verifies the real single-use target claim and
operator authorization, checks all issue prerequisites, and requires every
supplemental report. Missing, synthetic, historical, rehashed-but-altered, or
incomplete evidence remains `BLOCKED`.

Only an exact `PROMOTE_TO_CHAMPION` decision names
`v8_1_entry_price_floor_0_30_sized_1_0` as the champion. Paper, live, write,
wallet, capital, handoff, #134, and #146 permissions remain false.

## Pre-collection verification

The pipeline has already passed a synthetic 120-market dry-run without opening
real labels. Recheck before any authorized launch:

```bash
python -m ruff check \
  src/bigan/v8/polymarket/challenge_attempt_002.py \
  src/bigan/v8/polymarket/challenge_attempt_002_collection.py \
  src/bigan/v8/polymarket/challenge_attempt_002_pipeline.py \
  src/bigan/v8/polymarket/challenge_attempt_002_promotion.py \
  src/bigan/v8/polymarket/challenge_attempt_002_supplemental.py \
  src/bigan/v8/polymarket/challenge_attempt_002_target_freeze.py \
  examples/v8/run_challenge_attempt_002_collection.py \
  examples/v8/run_challenge_attempt_002_pipeline.py \
  examples/v8/run_challenge_attempt_002_promotion_audit.py \
  examples/v8/run_challenge_attempt_002_supplemental.py \
  examples/v8/run_challenge_attempt_002_target_freeze.py \
  tests/v8/test_challenge_attempt_002.py \
  tests/v8/test_challenge_attempt_002_collection.py \
  tests/v8/test_challenge_attempt_002_collection_manifest.py \
  tests/v8/test_challenge_attempt_002_preregistration.py \
  tests/v8/test_challenge_attempt_002_pipeline.py \
  tests/v8/test_challenge_attempt_002_execution_manifest.py \
  tests/v8/test_challenge_attempt_002_promotion.py \
  tests/v8/test_challenge_attempt_002_promotion_manifest.py \
  tests/v8/test_challenge_attempt_002_target_freeze.py

PYTHONPATH=src pytest -q \
  tests/v8/test_challenge_attempt_002.py \
  tests/v8/test_challenge_attempt_002_collection.py \
  tests/v8/test_challenge_attempt_002_collection_manifest.py \
  tests/v8/test_challenge_attempt_002_preregistration.py \
  tests/v8/test_challenge_attempt_002_pipeline.py \
  tests/v8/test_challenge_attempt_002_execution_manifest.py \
  tests/v8/test_challenge_attempt_002_promotion.py \
  tests/v8/test_challenge_attempt_002_promotion_manifest.py \
  tests/v8/test_challenge_attempt_002_target_freeze.py
```

Do not add a collection command to an operational checklist until the separate
authorization artifact exists.
