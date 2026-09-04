# One-command local paper stack (PR-D)

This stack consumes live public market data but performs simulated paper execution only. It has no wallet, signing capability, private exchange credentials, or real order path.

## Quick start

Install the package with `python -m pip install -e '.[dev]'`, or install its wheel.
Run from the repository root. For a two-minute, no-network demo:

```bash
python -m bigan.paper_trading.stack \
  --config config/paper_operator.example.toml --mock-demo \
  --dashboard-host 127.0.0.1 --dashboard-port 8080 \
  --duration 2m --report-dir artifacts/paper_soak/mock-smoke
```

Open <http://127.0.0.1:8080>. `bigan-paper-stack` is the equivalent installed
console command. Choose another port if occupied. Use a new empty report directory
for each invocation. Existing paper accounts may be resumed by their operator;
the stack does not reset, repair, migrate, or take ownership of them.

The long-running mock child uses the real Operator, Session, pricing/alpha/OMS
and ledger with deterministic **offline** quotes. Its first market is already in
progress: it ends after twice the configured tail cutoff (minimum one second,
60 seconds with the example defaults). Later markets retain their configured
5m/15m duration. It uses wall-clock timestamps, no strategy overrides, no forced
orders and no public network. A mock settlement is fixture truth, not exchange
resolution evidence. The original operator CLI's one-shot `--mock-demo` remains
unchanged. Mock and live must use separate configuration identities/output trees.

## Live public-feed paper execution

Live execution requires a **regular wheel**, not `PYTHONPATH=src` or an editable
install. Build from a committed checkout, then install into a clean environment:

```bash
python -m pip wheel . --no-deps --wheel-dir dist
python -m venv /path/to/paper-venv
/path/to/paper-venv/bin/python -m pip install dist/bigan-0.1.0-py3-none-any.whl
/path/to/paper-venv/bin/python -m bigan.build_provenance
```

Use that environment's Python for the commands below. Do not overlay it with
an editable install or source-tree `PYTHONPATH`. Copy the **verified installed
source_commit** printed by the command into the deployment config.

```bash
cp config/paper_operator.live.example.toml config/paper_operator.live.toml
# Set source_commit to the verified installed wheel SHA; select an independent output_dir.
# Do not commit this deployment-specific file or add credentials.
python -m bigan.paper_trading.stack --config config/paper_operator.live.toml --preflight
python -m bigan.paper_trading.stack \
  --config config/paper_operator.live.toml \
  --dashboard-host 127.0.0.1 --dashboard-port 8080 \
  --duration 30m --report-dir artifacts/paper_soak/live-30m
```

`dry_run=false` means **execute the public-feed paper loop**, rather than just
check its configuration. It does not enable live trading. All six safety fields
remain invariant:

```toml
paper_only = true
capital_at_risk = false
broker_exchange_write_enabled = false
live_exchange_write_enabled = false
polymarket_write_enabled = false
wallet_signing_enabled = false
```

The existing strict configuration loader rejects dangerous/unknown fields and
non-allowlisted public endpoints. Live startup additionally requires `mock=false`,
`dry_run=false`, `config_check_only=false`, a full 40-character lowercase source
SHA matching the verified executing wheel, safe display IDs and an explicit
output directory. The bundled template intentionally contains a placeholder and
**must fail** until replaced. No Git command is run during preflight: it verifies
the `_build_provenance.json` generated inside the wheel, including SHA-256 hashes
of its packaged code/assets. Cwd, environment SHA declarations, and configuration
values cannot supply build provenance. Both children reverify their own package
before startup. Standalone Operator live execution and Dashboard live display
enforce the same check based on their actual mode, without requiring the hidden
Supervisor arguments. Missing metadata, dirty/unverifiable builds, mismatched source
claims and altered/extra/missing packaged files fail closed.

The setuptools build hook attests the Git HEAD only when packaged bytes and
build inputs match that revision. Source archives without Git and dirty builds
remain installable for development/mock but carry no verified source SHA and
cannot run live. Editable builds never stamp the mutable source tree. Unrelated
notes/tests/docs do not invalidate identical package/build inputs. Build metadata
is not a cryptographic signature: the builder and distribution channel must be
trusted. This does not defend against an actor replacing both verifier and seal.
See the [setuptools customization contract](https://setuptools.pypa.io/en/stable/userguide/extension.html).
There is no environment-based trading configuration override.

Preflight binds and releases a loopback socket to test availability, but makes
no outbound connections, creates no directories/locks and launches no processes.
Its output contains only safe identity/URL/mode fields. A report directory must
be disjoint from the paper output tree, including symlink resolution; existing
nonempty directories are rejected. Relative `output_dir` keeps its existing
**startup working directory** semantics. Both children receive the same absolute
config filename, cwd and expected hash from `OperatorConfig.config_sha256`.
Each child validates that hash before starting; the hash algorithm is not duplicated.

## Lifecycle and boundaries

```text
Supervisor (no paper file writes, no writer lock)
  ├─ Dashboard child → existing read-only reader → paper files
  ├─ Operator child  → existing ownership locks → sole paper writer
  └─ Soak observer  → GET healthz / readyz / dashboard → separate report directory
```

Dashboard starts first. Its health response must match the expected configuration
and a unique launch ID, preventing accidental attachment to another listener.
Only then is Operator launched. Readiness must refer to a process started after
that launch, not a previous `STOPPED` status. The single startup deadline covers
both phases. No shell commands or automatic writer restarts are used.

Remove `--duration` to run until SIGINT/SIGTERM. On normal stop the supervisor
cancels further polls, sends SIGTERM to its operator child, waits for its exit and
final `STOPPED` status, reads one final dashboard view, stops the dashboard, then
writes the report. Before stopping a healthy dashboard it allows one browser
refresh interval (2.5 seconds) to display STOPPED. The page's last view is retained with a disconnected banner
after its server stops; the persisted final status remains `STOPPED`. A child
crash stops its sibling and returns nonzero. Grace-period exhaustion escalates
to kill and **FAIL**; only process handles created by this supervisor are signaled.
SIGKILL/power loss cannot be handled; normal operator recovery remains authoritative.

Raw child stdout/stderr is drained in bounded chunks and suppressed, with fixed
`[operator]`/`[dashboard]` notices. Parent diagnostics use fixed `[soak]` codes.
This intentionally avoids leaking tracebacks, configuration or private paths.
The dashboard remains loopback-only, read-only and unauthenticated. Never expose
it through a proxy, tunnel or port forward. No new UI controls or write routes.

## Options and troubleshooting

Times use positive integer `s`, `m`, `h` units, maximum seven days. Defaults:
`--startup-timeout 60s`, `--poll-interval 2s`, `--request-timeout 3s`,
`--shutdown-grace 15s`, `--unreadable-timeout 30s`, `--stale-timeout 30s`,
`--rollover-timeout 15m`. Duration starts after readiness; report duration also
includes startup/shutdown. `--no-soak-report` skips artifacts, **not safety gates**.

| Symptom | Response |
| --- | --- |
| No market / DISCOVERING | Wait for eligible public markets; no forced signal. |
| Feed stale / DEGRADED | Check public endpoint reachability, clocks, authoritative start reference and warm-up; never disable freshness gates. |
| Dashboard 503 | During startup/frontier changes retry; continuous unreadability fails at the configured deadline. |
| Writer lock occupied | Stop the existing owner intentionally or use a different account/output; never remove a held lock. |
| Port occupied | Select another loopback port; stack never kills its owner. |
| Rollover pending | Wait for final public resolution/next market; no bid-derived payout. Deadline exhaustion fails the soak. |
| No fill / HOLD / loss | Valid strategy outcome, not a stability failure. Do not tune parameters to force acceptance. |
| Report directory nonempty | Choose a new directory; no overwrite or automatic deletion. |

See [soak validation](soak_validation.md) for report semantics and evidence.
