# Paper stack soak validation

The observer only performs serial HTTP GET requests to `/healthz`, `/readyz`
and `/api/v1/dashboard`. It never opens a manifest, checkpoint, JSONL, SQLite
database or ownership lock. DashboardReader remains responsible for canonical
read-model consistency; this is **sampled observation, not deterministic audit**.

## Report contract

An explicit disjoint empty report directory receives `soak_report.json` (schema
1) and `soak_summary.md`. An exclusive report marker prevents concurrent reuse;
each file is fsync'd and atomically renamed. Files are never overwritten. The
two artifacts are not a multi-file transaction: on disk errors the command
returns nonzero, and incomplete artifacts must not be accepted as a completed
soak. A crash may leave a marker requiring manual inspection. Paper output is
never used for report storage or altered by report failure handling.

The JSON includes deployed/config identities, exact paper safety invariants,
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

## Manual acceptance evidence

Evidence is recorded after the final implementation checks in this PR. A live
report must come from an actual run, not a mock renamed as live. On this host,
the public Binance depth probe returned **HTTP 451** on 2026-09-04. That blocks
the required REST bootstrap. Consequently the required **30-minute live soak
has not been executed** and remains a deployment acceptance prerequisite from
an environment with legitimate access to all required public endpoints. No
endpoint allowlist bypass, alternate private source or fabricated fill is used.

The mock report and dashboard screenshots are attached separately after the
two-minute local acceptance run. They demonstrate process/ledger/UI integration,
not public feed availability or production profitability.
