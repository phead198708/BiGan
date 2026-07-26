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
- `challenge_historical_development_iteration_003_result.json`

Each has a SHA-256 sidecar. The earlier frozen attempt-001 plan, governance,
checklist, historical registry, and historical success standard are not
modified.

## Authorization boundary

Do not launch a collector from this preregistration alone. Collection requires
the operator's explicit permission for the 120-market run. After permission,
record it in a new, hash-pinned authorization artifact; do not rewrite the
preregistration.

The future service root is:

```text
examples/v8/polymarket_live_runs/challenge-model-v8-1-attempt-002
```

Only markets with start timestamps strictly greater than
`1785088622878` are eligible. The target is exactly 120 chronological,
quality-valid BTC UP/DOWN 5-minute markets, collected in bounded batches of 12.
The maximum attempted count is 180. Batch summaries go to issue #260; there is
no per-round comment.

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

1. Freeze all 120 candidate and matched v6.7 decisions before target access.
   Every market stays in the comparison; `NO_TRADE` is zero.
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

## Executable evidence pipeline

`run_challenge_attempt_002_pipeline.py` controls evidence freezing and
evaluation only; it has no collector-start operation.

After an authorized collection reaches 120 quality-valid markets, freeze the
shared source grid and both target-free decision streams:

```bash
PYTHONPATH=src:. python examples/v8/run_challenge_attempt_002_pipeline.py \
  freeze-pairs \
  --run-id <target-free-freeze-run-id> \
  --shared-source-rows <shared-source.jsonl> \
  --shared-source-rows-sha256 <sha256> \
  --candidate-decisions <candidate-decisions.jsonl> \
  --candidate-decisions-sha256 <sha256> \
  --baseline-decisions <baseline-decisions.jsonl> \
  --baseline-decisions-sha256 <sha256>
```

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

## Pre-collection verification

The pipeline has already passed a synthetic 120-market dry-run without opening
real labels. Recheck before any authorized launch:

```bash
python -m ruff check \
  src/bigan/v8/polymarket/challenge_attempt_002.py \
  src/bigan/v8/polymarket/challenge_attempt_002_pipeline.py \
  examples/v8/run_challenge_attempt_002_pipeline.py \
  tests/v8/test_challenge_attempt_002.py \
  tests/v8/test_challenge_attempt_002_preregistration.py \
  tests/v8/test_challenge_attempt_002_pipeline.py \
  tests/v8/test_challenge_attempt_002_execution_manifest.py

PYTHONPATH=src pytest -q \
  tests/v8/test_challenge_attempt_002.py \
  tests/v8/test_challenge_attempt_002_preregistration.py \
  tests/v8/test_challenge_attempt_002_pipeline.py \
  tests/v8/test_challenge_attempt_002_execution_manifest.py
```

Do not add a collection command to an operational checklist until the separate
authorization artifact exists.
