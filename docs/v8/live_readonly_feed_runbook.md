# v8 Live Read-only Feed Runbook

This runbook covers the live-data, paper-only operator path added for #129.
It connects a public/read-only market data feed to the existing v8 paper
operator pipeline.

It does not place orders, touch capital, use private trading keys, call
broker/exchange write APIs, promote a candidate, or enable live deployment.

## Safety Boundary

Every live feed run must preserve:

```text
feed_mode=live-readonly
read_only=true
write_capable=false
paper_only=true
capital_at_risk=false
broker_exchange_write_enabled=false
live_exchange_write_enabled=false
```

Live mode never silently falls back to deterministic replay. If the live adapter
cannot be created or the provided feed is not tagged `feed_mode=live-readonly`,
the operator writes a fail-closed manifest when possible and exits non-zero.

## Manual 24h Live-data Run

```bash
PYTHONPATH=src python examples/v8/run_live_24h_paper_operator.py \
  --run-id live_readonly_24h_001 \
  --output-dir examples/v8/live_operator_runs \
  --repo phead198708/BiGan \
  --issue-number 129 \
  --mode gh-command \
  --feed-mode live-readonly \
  --provider coinbase_public_ticker \
  --provider-endpoint https://api.exchange.coinbase.com/products/BTC-USD/ticker \
  --instrument BTC-USD \
  --duration-hours 24 \
  --poll-interval-seconds 60 \
  --heartbeat-interval-seconds 60 \
  --summary-interval-seconds 300
```

The default provider adapter uses a public Coinbase REST ticker endpoint and
does not require trading credentials. The adapter also supports Binance-style
`bidPrice` / `askPrice` ticker payloads when that public endpoint is available
from the runtime environment.

## Short Local Smoke

The live adapter is intended for real wall-clock runs. CI uses mocked live
feeds, not a real provider. For local deterministic pipeline checks, use the
regular operator CLI with replay mode:

```bash
PYTHONPATH=src python examples/v8/run_24h_paper_operator.py \
  --run-id readonly_replay_smoke_001 \
  --output-dir examples/v8/operator_runs \
  --repo phead198708/BiGan \
  --issue-number 129 \
  --mode dry-run \
  --feed-mode deterministic-replay \
  --duration-seconds 300 \
  --overwrite-existing
```

## Required Live Artifacts

A live-data run writes all existing paper operator artifacts plus:

```text
paper_run/live_feed_metadata.json
paper_run/live_feed_health_report.json
```

`operator_run_manifest.json` records:

```text
feed_mode
real_live_data
deterministic_replay
provider_name
provider_endpoint_or_endpoint_type
instrument_id
started_at_wall_clock
ended_at_wall_clock
wall_clock_duration_seconds
configured_duration_seconds
live_feed_metadata_sha256
live_feed_health_sha256
provider_disconnect_count
provider_reconnect_count
provider_error_count
stale_event_count
empty_response_count
rate_limit_count
```

The observability report and GitHub comment include the same feed mode,
provider, instrument, and feed health fields.

## Fail-closed Feed Health

The live feed path blocks Phase 6 / operator approval for:

```text
feed_gap_breach
feed_late_event_breach
feed_out_of_order_breach
stale_event_breach
provider_error_breach
empty_response_breach
heartbeat_missing
```

Blocked runs keep all generated candidates and artifacts visible for audit.
They do not allow capital deployment or live deployment.

## STOP Behavior

The STOP-file path remains active for live-data runs. If a STOP file is detected
after enough feed evidence exists, the operator still writes live feed metadata,
paper summary, observability, GitHub comment payload, and manifest outputs.

Expected STOP manifest fields:

```text
status=operator_stopped
stop_reason=operator_stop
paper_only=true
capital_at_risk=false
```

## Manual Evidence Checklist

Post the completed 24h evidence to #129 with:

```text
run_id
commit_sha
feed_mode=live-readonly
real_live_data=true
deterministic_replay=false
provider_name
instrument_id
started_at
ended_at
started_at_wall_clock
ended_at_wall_clock
wall_clock_duration_seconds >= 86340
configured_duration_seconds
feed_event_count > 0
heartbeat_count > 0
periodic_summary_count > 0
provider_disconnect_count
provider_reconnect_count
provider_error_count
stale_event_count
feed_gap_count
feed_late_event_count
feed_out_of_order_count
phase5_passed
phase5_kill_switch_triggered
phase6_deployment_status
operator_recommendation
paper_only=true
capital_at_risk=false
broker_exchange_write_enabled=false
live_exchange_write_enabled=false
operator_run_manifest_sha256
paper_run_summary_sha256
observability_report_sha256
github_comment_payload_sha256
```

## CI Validation

```bash
PYTHONPATH=src python -m pytest \
  tests/v8/test_live_readonly_feed.py \
  tests/v8/test_live_24h_paper_operator.py -q
```

These tests use deterministic mocked live feed behavior and do not connect to a
real provider.
