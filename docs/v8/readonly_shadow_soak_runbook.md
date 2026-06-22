# v8 Read-Only Shadow Soak Runbook

This runbook covers the v8 paper-only read-only shadow soak runner. The first
implementation uses a deterministic replay feed fallback, so it can simulate a
bounded short run or a 24h window without waiting in CI. It does not place real
orders, mutate real positions, or connect to broker/exchange write APIs.

## Safety Boundary

Allowed:

- read-only market feed events
- deterministic replay feed fallback
- paper orders, fills, ledger entries, and positions
- heartbeat snapshots
- periodic summaries
- feed health reports
- Phase 5 safety reports
- Phase 6 paper-mode CI/CD reports

Forbidden:

- real orders
- real capital
- broker/exchange write APIs
- automatic live-mode switch
- unbounded daemon behavior
- bypassing Phase 5 or Phase 6

Every summary and manifest must record:

```json
{
  "paper_only": true,
  "capital_at_risk": false,
  "broker_exchange_write_enabled": false,
  "live_exchange_write_enabled": false
}
```

## Short Deterministic Validation

```bash
PYTHONPATH=src python examples/v8/run_readonly_shadow_soak.py \
  --output-dir /tmp/bigan-v8-readonly-shadow \
  --run-id readonly_shadow_short_001 \
  --duration-seconds 300 \
  --feed-event-interval-seconds 60 \
  --heartbeat-interval-seconds 10 \
  --summary-interval-seconds 60 \
  --overwrite-existing
```

Expected result:

- `feed_event_count > 0`
- `feed_health_passed=true`
- `heartbeat_count > 0`
- `periodic_summary_count > 0`
- Phase 5 kill-switch is not triggered
- Phase 6 reports `approved_for_staged_live`

## Degraded Validation

```bash
PYTHONPATH=src python examples/v8/run_readonly_shadow_soak.py \
  --output-dir /tmp/bigan-v8-readonly-shadow \
  --run-id readonly_shadow_degraded_001 \
  --duration-seconds 900 \
  --inject-degradation \
  --overwrite-existing
```

Expected result:

- Phase 5 report is produced
- Phase 5 kill-switch is triggered
- Phase 5 reason codes are non-empty
- Phase 6 reports `blocked_fail_closed`

## Feed-Health Hard Gate

Feed-health anomalies are hard gates for the read-only shadow soak. The runner
records the feed-health acceptance report in `feed_health_report.json`, mirrors
it into `paper_run_summary.json`, and injects it into the Phase 6 `monitoring`
stage metadata.

Any non-empty feed-health reason code blocks Phase 6 fail closed:

```text
feed_gap_breach
feed_late_event_breach
feed_out_of_order_breach
heartbeat_missing
```

Expected blocked result for those cases:

- `feed_health_passed=false`
- `feed_health_reason_codes` is non-empty
- Phase 6 reports `blocked_fail_closed`

## Operator Stop Validation

The runner supports a `<run_dir>/STOP` file. In deterministic tests, the
`--stop-after-events` option creates that file after a configured number of feed
events to exercise the same stop branch.

```bash
PYTHONPATH=src python examples/v8/run_readonly_shadow_soak.py \
  --output-dir /tmp/bigan-v8-readonly-shadow \
  --run-id readonly_shadow_stop_001 \
  --duration-seconds 1200 \
  --stop-after-events 5 \
  --overwrite-existing
```

Expected result:

- `stop_reason=operator_stop`
- artifacts are flushed
- Phase 5 and Phase 6 evidence are written
- no broker/exchange action is needed

## 24h Deterministic Replay Fallback

This command simulates a 24h bounded read-only feed window using deterministic
events. It does not wait 24 real hours.

```bash
PYTHONPATH=src python examples/v8/run_readonly_shadow_soak.py \
  --output-dir examples/v8/readonly_shadow_runs \
  --run-id readonly_shadow_24h_001 \
  --duration-hours 24 \
  --feed-event-interval-seconds 60 \
  --heartbeat-interval-seconds 60 \
  --summary-interval-seconds 300 \
  --overwrite-existing
```

Record the resulting `paper_run_summary.json` and `paper_bundle_manifest.json`
paths and SHA-256 hashes back into the GitHub issue. Until a real read-only live
feed adapter is configured, this deterministic fallback is the supported local
and CI-safe mode.

## Required Artifacts

Each run writes:

```text
readonly_feed_events.jsonl
paper_orders.jsonl
paper_fills.jsonl
paper_ledger.jsonl
paper_positions.json
paper_pnl_report.json
paper_run_summary.json
paper_soak_heartbeat.jsonl
paper_soak_periodic_summaries.jsonl
feed_health_report.json
phase5_safety_layer_report.json
phase6_cicd_pipeline_report_<release_id>.json
paper_bundle_manifest.json
```

`paper_bundle_manifest.json` records SHA-256 hashes for all run artifacts except
the manifest itself. Recompute the manifest hash separately when pasting issue
evidence.

## Inspect A Run

```bash
python - <<'PY'
import json
from pathlib import Path

run_dir = Path("/tmp/bigan-v8-readonly-shadow/readonly_shadow_short_001")
summary = json.loads((run_dir / "paper_run_summary.json").read_text())
print(json.dumps({
    "run_id": summary["run_id"],
    "stop_reason": summary["stop_reason"],
    "feed_event_count": summary["feed_event_count"],
    "feed_health_passed": summary["feed_health_passed"],
    "feed_health_reason_codes": summary["feed_health_reason_codes"],
    "heartbeat_count": summary["heartbeat_count"],
    "periodic_summary_count": summary["periodic_summary_count"],
    "phase5_kill_switch_triggered": summary["phase5_kill_switch_triggered"],
    "phase5_reason_codes": summary["phase5_reason_codes"],
    "phase6_deployment_status": summary["phase6_deployment_status"],
    "paper_only": summary["paper_only"],
    "capital_at_risk": summary["capital_at_risk"],
}, indent=2, sort_keys=True))
PY
```

## CI Gate

The lightweight gate is:

```bash
PYTHONPATH=src python -m pytest tests/v8/test_readonly_shadow_soak.py -q
```

The full v8 hard gate also runs Phase 0-6, golden path, paper harness, replay
paper soak, and read-only shadow soak tests.
