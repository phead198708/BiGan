# v8 Paper Runbook

This runbook covers the deterministic replay paper run introduced for v8 paper
soak validation. It is a paper-only workflow. It must not place real orders,
mutate real positions, connect to broker or exchange write APIs, or put capital
at risk.

## Safety Boundary

The replay paper run is allowed to:

- generate deterministic Phase 4 replay decisions
- emit paper orders, fills, ledger entries, and positions
- feed paper observations into Phase 5
- produce Phase 6 paper-mode CI/CD evidence
- write a run-scoped artifact bundle

The replay paper run is not allowed to:

- send real orders
- use real capital
- mutate real positions
- auto-switch into live mode
- bypass Phase 5 or Phase 6
- call broker or exchange write APIs

Every run summary and bundle manifest must record:

```json
{
  "paper_only": true,
  "capital_at_risk": false,
  "broker_exchange_write_enabled": false
}
```

## Run A Healthy Replay

```bash
PYTHONPATH=src python examples/v8/run_paper_soak.py \
  --output-dir /tmp/bigan-v8-paper-soak \
  --run-id paper_soak_replay_v1 \
  --row-count 512 \
  --overwrite-existing
```

Expected outcome:

- `paper_run_summary.json` exists
- `paper_bundle_manifest.json` exists
- Phase 5 passes with `phase5_kill_switch_triggered=false`
- Phase 6 reports `deployment_status=approved_for_staged_live`

## Run An Injected Degradation Replay

```bash
PYTHONPATH=src python examples/v8/run_paper_soak.py \
  --output-dir /tmp/bigan-v8-paper-soak \
  --run-id paper_soak_degraded_v1 \
  --row-count 512 \
  --inject-degradation \
  --overwrite-existing
```

Expected outcome:

- Phase 5 still produces a valid report
- `phase5_kill_switch_triggered=true`
- `phase5_reason_codes` is non-empty
- Phase 6 reports `deployment_status=blocked_fail_closed`

## Required Artifacts

Each run writes a run-scoped directory:

```text
<output-dir>/<run-id>/
```

Required files:

```text
paper_orders.jsonl
paper_fills.jsonl
paper_ledger.jsonl
paper_positions.json
paper_pnl_report.json
phase5_safety_layer_report.json
phase6_cicd_pipeline_report_<release_id>.json
paper_bundle_manifest.json
paper_run_summary.json
```

`paper_bundle_manifest.json` records SHA-256 hashes for the run artifacts it
indexes, including `paper_run_summary.json`. The manifest's own SHA-256 is
printed by the runner and can be recomputed with:

```bash
python - <<'PY'
import hashlib
from pathlib import Path

path = Path("/tmp/bigan-v8-paper-soak/paper_soak_replay_v1/paper_bundle_manifest.json")
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
```

## Inspect The Run Summary

```bash
python - <<'PY'
import json
from pathlib import Path

summary = json.loads(
    Path("/tmp/bigan-v8-paper-soak/paper_soak_replay_v1/paper_run_summary.json")
    .read_text()
)
print(json.dumps({
    "row_count": summary["row_count"],
    "phase5_kill_switch_triggered": summary["phase5_kill_switch_triggered"],
    "phase5_reason_codes": summary["phase5_reason_codes"],
    "phase6_deployment_status": summary["phase6_deployment_status"],
    "paper_only": summary["paper_only"],
    "capital_at_risk": summary["capital_at_risk"],
}, indent=2, sort_keys=True))
PY
```

## Stop Procedure

This first implementation is a bounded replay command, not a daemon. Stopping it
means terminating the local process. There is no live order writer to disable and
no real position to flatten. If a future read-only feed shadow run becomes a
long-running process, it must keep the same paper-only safety boundary and add
operator stop/alert steps before it can run outside replay.

## CI Gate

The lightweight CI smoke path is:

```bash
PYTHONPATH=src python -m pytest tests/v8/test_paper_soak.py -q
```

The full v8 hard gate also runs the existing Phase 0-6 tests, the golden-path
dry run, the paper harness test, and the paper soak test.
