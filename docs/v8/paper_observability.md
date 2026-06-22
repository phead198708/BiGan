# v8 Paper Observability

This document covers the paper-only observability layer for v8 paper runs. It
reads existing paper artifacts, evaluates deterministic alert rules, and writes
operator-facing reports. It does not place orders, touch capital, send external
alerts, or promote a strategy.

## Safety Boundary

Allowed:

- read paper run artifacts
- read Phase 5 and Phase 6 reports
- compute derived metrics
- write operator reports
- write dashboard JSON and alert JSONL
- write markdown summaries for GitHub comments

Forbidden:

- real orders
- real capital
- broker/exchange write APIs
- automatic live-mode switch
- automatic deployment promotion
- modifying paper ledger artifacts after the run

Every generated report preserves:

```json
{
  "paper_only": true,
  "capital_at_risk": false,
  "broker_exchange_write_enabled": false,
  "live_exchange_write_enabled": false
}
```

If source artifacts violate the paper boundary, the observability report emits
critical alerts and recommends stopping the paper run.

## Summarize A Run

```bash
PYTHONPATH=src python examples/v8/summarize_paper_run.py \
  --run-dir examples/v8/readonly_shadow_runs/readonly_shadow_24h_001 \
  --output-dir examples/v8/operator_reports/readonly_shadow_24h_001 \
  --overwrite-existing
```

Console output includes:

```text
run_id
phase6_deployment_status
feed_health_status
alert_count
critical_alert_count
operator_recommendation
operator_summary_path
observability_report_path
```

## Output Artifacts

Each observability run writes:

```text
paper_observability_report.json
paper_operator_summary.md
paper_alerts.jsonl
paper_dashboard_summary.json
paper_periodic_metrics.csv
```

If `--compare-run-dir` is provided, it also writes:

```text
paper_run_comparison.json
paper_run_comparison.md
```

Outputs are deterministic for the same source run directory.

## Alert Categories

Feed alerts:

- `feed_gap_breach`
- `feed_late_event_breach`
- `feed_out_of_order_breach`
- `heartbeat_missing`
- `periodic_summary_missing`

Safety alerts:

- `kill_switch_triggered`
- `safety_reason_codes_present`
- `rollback_not_reliable`
- `phase5_not_passed`

Phase 6 alerts:

- `candidate_identity_not_verified`
- `phase6_blocked`
- `phase6_unknown_status`

Paper-boundary alerts:

- `paper_only_violation`
- `capital_risk_violation`
- `broker_write_enabled`
- `live_write_enabled`

Performance and execution alerts:

- `drawdown_threshold_breach`
- `cost_drift_breach`
- `pnl_drift_breach`
- `regime_mismatch_breach`

## Operator Recommendation

The recommendation is advisory only:

```text
continue_paper_run
investigate_warning
stop_paper_run
blocked_fail_closed
```

Mapping:

- no alerts: `continue_paper_run`
- warning alerts only: `investigate_warning`
- critical alerts: `stop_paper_run`
- Phase 6 `blocked_fail_closed`: `blocked_fail_closed`

No recommendation can trigger a live deployment.

## Required Source Artifacts

The source run directory must contain:

```text
paper_run_summary.json
paper_bundle_manifest.json
feed_health_report.json
phase5_safety_layer_report.json
phase6_cicd_pipeline_report_<release_id>.json
paper_orders.jsonl
paper_fills.jsonl
paper_ledger.jsonl
paper_positions.json
paper_pnl_report.json
paper_soak_heartbeat.jsonl
paper_soak_periodic_summaries.jsonl
```

Missing required artifacts fail closed before output reports are written.

## CI Gate

```bash
PYTHONPATH=src python -m pytest tests/v8/test_paper_observability.py -q
```

The full v8 hard gate includes paper observability alongside Phase 0-6, golden
path, paper harness, paper soak, and read-only shadow soak tests.
