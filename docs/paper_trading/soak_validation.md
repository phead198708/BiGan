# Paper stack soak validation

The observer only performs serial HTTP GET requests to `/healthz`, `/readyz`
and `/api/v1/dashboard`. It never opens a manifest, checkpoint, JSONL, SQLite
database or ownership lock. DashboardReader remains responsible for canonical
read-model consistency; this is **sampled observation, not deterministic audit**.

## Report contract

An explicit disjoint empty report directory receives `soak_report.json` (schema
1, `publication_schema_version=1`), `soak_summary.md`, and `soak_complete.json`.
**The JSON alone is never acceptance evidence.** Consumers must verify the
completion manifest and both artifact hashes, and reject `.soak-report.lock`.
The supported read-only consumer is:

```python
from pathlib import Path
from bigan.paper_trading.stack.report import load_completed_report
report = load_completed_report(Path("artifacts/paper_soak/run-001"))
```

The writer exclusively claims `.soak-report.lock`, stages and fsyncs every file,
publishes both artifacts, then atomically renames the incomplete marker into the
completion manifest. This final rename is the publication commit point. The
manifest contains both SHA-256 hashes and the result; consumers reject missing,
changed or mixed artifacts. Closing the writer does **not** remove an incomplete
marker. On exceptions the run becomes FAIL and this writer's candidate artifacts
(including a previously published PASS JSON) are withdrawn, while the incomplete
marker remains. No existing user report is overwritten or repaired. Persistent
I/O failures/power loss can leave candidates: they are invalid without a matching
completion manifest and absent incomplete marker, even if a candidate says PASS.
Paper output is never used for report storage or changed by publication failures.

The JSON includes deployed/config identities, verified build provenance for live
runs (null for unsealed mock runs), exact paper safety invariants,
timestamps, poll attempts/success/failures/longest continuous outage, state sample
counts and observed durations, feed fresh sample ratios, run indices/rollovers,
first/final canonical account observations, activity deltas and bounded diagnostic
codes. No response bodies, market titles, source URLs, config files, secrets,
private paths or raw child logs are retained. Safety values describe the required
boundary; any observed violation is a hard failure, not rewritten as safe evidence.

Account equity/cash/PnL/fees/drawdown come directly from the dashboard account.
`initial_equity` is the **first observed equity**, not original bankroll. Maximum
observed drawdown is the maximum sampled **current-run canonical drawdown**;
it is not a new cross-window drawdown/PnL calculation. Activity is a lower bound
from counter changes between samples; the first observation of each run is a
baseline, so events before it or between missed run frontiers can be omitted.
Settlement counters are process-wide. Feed ratios are sample-weighted, not an
assertion of continuous availability between polls. State durations charge the
interval between successful observations to the last observed state.

Memory is bounded: 2 MB per HTTP body, depth 32, at most 50 rows per history
section (or the config's smaller bound), 64 distinct diagnostic codes per severity,
and 1,024 run identities. Reaching the run cap is an explicit hard failure rather
than silently losing identity-check coverage. Polls cannot overlap; no full
payload history is retained; aiohttp sessions and child pipes are closed.

Cross-run validation uses the returned run chain, not the mistaken assumption
that every history row belongs to the active run. It checks immutable run/index/
market/window/opening identities, current status/account/positions, history event
contracts, and predecessor settled cash against successor opening cash. If an
independently bounded event page reaches beyond the observed run page, its
identity is **unverified/WARN**, never silently certified. A missing historical
identity in an otherwise complete run page is a hard failure. This does not
replace the operator's persistent replay/idempotency checks.

## PASS / WARN / FAIL

- **PASS:** valid observations and clean STOPPED shutdown, no warning-severity or
  hard failures. Informational no-fill/no-decision/no-settlement, discovery,
  warm-up and normal brief rollover transitions may still be PASS.
- **WARN:** recoverable HTTP errors, DEGRADED periods, stale samples below the
  continuous-stale deadline, unavailable optional sections or unverifiable
  historical identities. Inspect before accepting a deployment.
- **FAIL:** safety/schema/finite-value failures, negative cash/equity, mixed run
  identities, regressing run index, changing source/config/process identity,
  inconsistent settlement handoff, FAILED/unexpected child exit, continuous
  unreadability/staleness or delayed rollover beyond the configured deadline,
  forced termination, missing final STOPPED, or no valid observations.

PASS and WARN return 0 after clean shutdown; FAIL returns 1. Preflight/config
errors return 2 without children. Reports never turn a losing strategy or zero
fills into a stability failure. A PASS also does not prove an entire live market
window was covered: check mode, duration, run coverage and public-feed evidence.

## Automated, offline acceptance

```bash
PYTHONPATH=src python -m pytest tests/paper_trading/stack -q -o addopts=''
PYTHONPATH=src python -m pytest tests/paper_trading -q -o addopts=''
```

Tests use real loopback dashboard/operator children and temporary account/report
trees. Normal and expensive-quote fixtures exercise decisions with fills and
HOLD-only outcomes, settlement, rollover and STOPPED. The expensive-quote fixture
changes offline input quotes, not strategy parameters or generated signals.
OS-signal tests run the entire supervisor as a subprocess. Failure cases include
crashed children, existing account locks, port races, startup deadlines, frozen
operator escalation, config changes, malformed/stale/nonfinite HTTP views and
bounded serial polling. CI makes no Binance/Polymarket/Chainlink requests.

## Current publication/provenance acceptance

On 2026-09-04, the ordinary wheel built from committed runtime
`84e71d0cdb7e907415d96873805a0eac8841e593` was installed into an isolated
target and verified independently of repository cwd. The build seal SHA-256 is
`8405d36a97058a2ce37114fc485c6cc6342c72ce6427c3227468d40eaf4b144c`.
The installed package passed live preflight with its exact source SHA, rejected
a different SHA, and rejected the unchanged bundled live template. A wheel
built before committing the runtime changes correctly returned
`UNVERIFIED_BUILD_SOURCE`. Preflight created no paper output in these checks.

That installed wheel then completed a fresh **two-minute offline mock soak**:
**PASS**, 124.132 seconds including startup/shutdown, 61/61 successful polls,
one settlement/rollover, final STOPPED, no warnings or hard failures. Activity
lower bounds were 453 decisions, 23 fills, and 300 HOLDs. Strategy parameters
were defaults; only the independent account/output/source identity was supplied.
The raw report intentionally has null build_provenance because mock mode does
not require or assert the live provenance gate, even when using a sealed wheel.

- [JSON artifact](evidence/mock-2m-publication/soak_report.json)
- [Markdown artifact](evidence/mock-2m-publication/soak_summary.md)
- [Completion manifest](evidence/mock-2m-publication/soak_complete.json)

All three artifacts are copied byte-for-byte from the new run and verified with
`load_completed_report()`. The incomplete marker is absent. This is a new
publication, not a completion marker added to historical reports. The external
browser script also passed running/mobile/rollover/STOPPED/retained-view checks
without page errors, and the stack reaped both owned children (port released).

Automated results: **673 passed** in the agent-stack suite, including 513
paper-trading tests and 112 stack tests. Ruff, MyPy (48 source files), compileall,
and diff checks passed. Failure injection covers each of the four publication
renames, including the first and second artifact failures in real Supervisor
runs; candidates are withdrawn, incomplete markers retained, and consumers
reject the result. Live 30-minute acceptance remains outstanding below.

## Historical manual acceptance evidence

The artifacts below are historical **pre-completion-protocol** evidence from
`6227e20`, not newly published reports under the current contract. They are kept
unchanged for provenance and will be rejected by `load_completed_report` because
they have no completion manifest. Never retrofit a marker to old evidence.

Implementation commit: `6227e20a753f3adb3d2901bc0e7f202127585044`.

On 2026-09-04 the **two-minute mock stack passed** (124.211 s including process
startup/shutdown): 61/61 polls successful, all sampled feeds fresh, 1 observed
settlement/rollover, final STOPPED, no warnings/hard failures. Activity lower
bounds: 453 decisions, 23 fills, 299 HOLDs. The deployment used default strategy
parameters, a temporary independent output/report tree and the exact commit
above as source identity. Its config SHA is recorded in the unmodified report;
the deployment-specific config/output paths are not published.

- [Machine report](evidence/mock-2m/soak_report.json)
- [Attachable Markdown summary](evidence/mock-2m/soak_summary.md)
- [Running desktop](images/stack/stack-running.png), [mobile](images/stack/stack-mobile.png),
  [after rollover](images/stack/stack-rollover.png), [STOPPED](images/stack/stack-stopped.png),
  [last view retained after disconnect](images/stack/stack-disconnected.png)

External Playwright/Chrome acceptance verified the permanent paper banner,
account cards, mobile overflow, two-run history, STOPPED before dashboard exit,
last-view retention after disconnect, and no page errors. The reusable optional
script is `tests/paper_trading/stack/browser_acceptance.cjs`; no Node/browser
runtime dependency is added to the package. All owned children were reaped.

Automated results: 90 stack tests; 491 paper-trading tests (included in the
651-test full agent-stack suite). Ruff, MyPy (47 source files), compileall,
diff whitespace checks and wheel build/isolated installation checks passed.
The installed wheel includes both console entry points and serves all assets
independently of source cwd. Real subprocess tests cover SIGINT/SIGTERM,
crashes, deadlines, locked accounts and no automatic writer restart. A rare
concurrent rollback-journal read or briefly observed settlement/rollover transition
can truthfully produce WARN in offline E2E; tests permit these expected
availability/state warnings, never hide them to force PASS.

A live report must come from an actual run, not a mock renamed as live. On this host,
the public Binance depth probe returned **HTTP 451** on 2026-09-04. That blocks
the required REST bootstrap. Consequently the required **30-minute live soak
has not been executed** and remains a deployment acceptance prerequisite from
an environment with legitimate access to all required public endpoints. No
endpoint allowlist bypass, alternate private source or fabricated fill is used.

These mock artifacts demonstrate process/ledger/UI integration, not public
feed availability, a complete 15-minute live market or production profitability.
