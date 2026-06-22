# v8 24h Paper Operator Runbook

This runbook covers the one-command, paper-only operator workflow for a bounded
read-only shadow soak. The command runs the paper soak, produces observability,
generates a GitHub issue comment payload, and writes an operator manifest.

It does not place orders, touch capital, call broker/exchange write APIs,
promote a model, close issues, or enable live deployment.

## Safety Boundary

Every run must preserve:

```text
paper_only=true
capital_at_risk=false
broker_exchange_write_enabled=false
live_exchange_write_enabled=false
```

The final manifest also records:

```text
capital_deployment_allowed=false
live_deployment_allowed=false
broker_exchange_write_allowed=false
```

Degraded, blocked, feed-anomaly, and operator-stop runs remain paper-only and
fail closed.

The production operator API does not expose a post-run artifact mutation hook.
Paper artifacts written by the soak are consumed directly by observability and
comment generation.

## Manual 24h Run

```bash
PYTHONPATH=src python examples/v8/run_24h_paper_operator.py \
  --run-id readonly_shadow_24h_001 \
  --output-dir examples/v8/operator_runs \
  --repo phead198708/BiGan \
  --issue-number 124 \
  --mode gh-command \
  --duration-hours 24 \
  --heartbeat-interval-seconds 60 \
  --summary-interval-seconds 300
```

The command prints a JSON console summary with:

```text
run_id
run_dir
paper_summary_path
phase5_status
phase6_deployment_status
feed_health_status
alert_count
critical_alert_count
operator_recommendation
observability_report_path
operator_summary_path
comment_body_path
gh_command_path
paper_only
capital_at_risk
status
```

## Short Smoke Run

```bash
PYTHONPATH=src python examples/v8/run_24h_paper_operator.py \
  --run-id readonly_shadow_smoke_001 \
  --output-dir examples/v8/operator_runs \
  --repo phead198708/BiGan \
  --issue-number 124 \
  --mode dry-run \
  --duration-seconds 300 \
  --overwrite-existing
```

Use short deterministic runs for CI and local validation.

## Output Structure

```text
operator_runs/<run_id>/
  paper_run/
    readonly_feed_events.jsonl
    paper_orders.jsonl
    paper_fills.jsonl
    paper_ledger.jsonl
    paper_positions.json
    paper_pnl_report.json
    paper_run_summary.json
    paper_bundle_manifest.json
    phase5_safety_layer_report.json
    phase6_cicd_pipeline_report_<release_id>.json
    feed_health_report.json
    paper_soak_heartbeat.jsonl
    paper_soak_periodic_summaries.jsonl
  observability/
    paper_observability_report.json
    paper_operator_summary.md
    paper_alerts.jsonl
    paper_dashboard_summary.json
    paper_periodic_metrics.csv
  github_comment/
    github_paper_comment_payload.json
    github_paper_comment.md
    github_paper_comment_gh_command.sh
  operator_run_manifest.json
```

The GitHub command uses an absolute `--body-file` path, so it can be copied from
the console or executed from outside the output directory.

## STOP Behavior

The underlying read-only soak honors the existing `STOP` file mechanism. For
deterministic rehearsals, the CLI exposes:

```bash
--stop-after-events 5
```

Operator stop runs still write paper artifacts, observability artifacts, GitHub
comment outputs, and `operator_run_manifest.json` when enough paper evidence
exists.

The manifest status is:

```text
operator_stopped
```

and `reason_codes` includes:

```text
operator_stop
```

## Failure Behavior

If an orchestration step fails after the operator run directory exists, the
command writes a fail-closed manifest:

```text
status=failed_fail_closed
capital_deployment_allowed=false
live_deployment_allowed=false
broker_exchange_write_allowed=false
```

The CLI exits non-zero. Existing paper artifacts are not promoted, and no live
path is enabled.

## Validation

```bash
PYTHONPATH=src python -m pytest tests/v8/test_24h_paper_operator_cli.py -q
```

The full v8 hard gate runs the operator CLI smoke tests alongside the existing
Phase 0-6, paper harness, paper soak, read-only shadow soak, observability, and
alert delivery gates.
