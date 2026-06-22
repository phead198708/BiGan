# v8 Paper Alert Delivery

This document covers GitHub issue-comment delivery for v8 paper observability
results. The delivery layer reads existing observability artifacts, builds a
deterministic GitHub markdown comment, and optionally generates or runs a
`gh issue comment` command.

It does not place orders, touch capital, promote a model, close issues, or send
Slack/PagerDuty/email alerts.

## Safety Boundary

Allowed:

- read paper observability artifacts
- build a GitHub markdown comment body
- write dry-run markdown and payload files
- generate a manual `gh issue comment` command
- optionally post to a specified issue only with explicit `direct-comment` mode

Forbidden:

- real orders
- real capital
- broker/exchange write APIs
- automatic live-mode switch
- automatic deployment promotion
- automatic issue closing
- posting without explicit repo and issue number
- dropping critical alerts from the comment

Every comment must show:

```text
paper_only=true
capital_at_risk=false
broker_exchange_write_enabled=false
live_exchange_write_enabled=false
```

Blocked, degraded, or boundary-violating runs must include a clear do-not-promote
message.

Critical alerts are never capped or omitted from the GitHub comment. The
`max_alerts_to_inline` limit only applies to warning/info alerts, and omitted
counts are reported as non-critical omissions.

## Dry Run

```bash
PYTHONPATH=src python examples/v8/post_paper_observability_comment.py \
  --observability-dir examples/v8/operator_reports/readonly_shadow_24h_001 \
  --repo phead198708/BiGan \
  --issue-number 124 \
  --output-dir examples/v8/comment_payloads/readonly_shadow_24h_001 \
  --mode dry-run \
  --overwrite-existing
```

Dry-run mode writes:

```text
github_paper_comment_payload.json
github_paper_comment.md
```

## GH Command Mode

```bash
PYTHONPATH=src python examples/v8/post_paper_observability_comment.py \
  --observability-dir examples/v8/operator_reports/readonly_shadow_24h_001 \
  --repo phead198708/BiGan \
  --issue-number 124 \
  --output-dir examples/v8/comment_payloads/readonly_shadow_24h_001 \
  --mode gh-command \
  --overwrite-existing
```

GH-command mode also writes:

```text
github_paper_comment_gh_command.sh
```

The command is copy-pasteable:

```bash
gh issue comment 124 --repo phead198708/BiGan --body-file /absolute/path/to/github_paper_comment.md
```

The generated `--body-file` path is absolute so the command can be copied or run
from outside the output directory.

## Direct Comment Mode

Direct posting is disabled unless explicitly requested:

```bash
PYTHONPATH=src python examples/v8/post_paper_observability_comment.py \
  --observability-dir examples/v8/operator_reports/readonly_shadow_24h_001 \
  --repo phead198708/BiGan \
  --issue-number 124 \
  --output-dir examples/v8/comment_payloads/readonly_shadow_24h_001 \
  --mode direct-comment \
  --overwrite-existing
```

Direct mode requires an authenticated `gh` CLI and writes:

```text
github_paper_comment_delivery_receipt.json
```

## Required Inputs

The observability directory must contain:

```text
paper_observability_report.json
paper_operator_summary.md
paper_alerts.jsonl
paper_dashboard_summary.json
```

Optional inputs are included when present:

```text
paper_periodic_metrics.csv
paper_run_comparison.json
paper_run_comparison.md
```

Missing required inputs fail closed before delivery outputs are written.

## Generated Comment Contents

The comment includes:

- run id
- operator recommendation
- Phase 6 deployment status
- feed health status
- safety status
- alert counts
- paper-only and no-capital-risk flags
- top critical/warning alerts
- next recommended operator action
- observability artifact hashes
- source paper artifact hashes

The paper safety section reports observed-vs-expected values:

```text
paper_only: <actual> expected true
capital_at_risk: <actual> expected false
broker_exchange_write_enabled: <actual> expected false
live_exchange_write_enabled: <actual> expected false
automatic_deployment_promotion: false expected false
```

Healthy runs should show:

```text
Recommendation: continue_paper_run
Critical alerts: 0
Phase 6: approved_for_staged_live
```

Blocked or degraded runs should show:

```text
Do not promote to live trading
Phase 6: blocked_fail_closed
Critical alerts: N
```

## CI Gate

```bash
PYTHONPATH=src python -m pytest tests/v8/test_paper_alert_delivery.py -q
```

The full v8 hard gate includes alert delivery alongside Phase 0-6, golden path,
paper harness, paper soak, read-only shadow soak, and paper observability tests.
