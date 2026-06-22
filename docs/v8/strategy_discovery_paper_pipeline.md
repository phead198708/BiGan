# v8 Strategy Discovery Paper Pipeline

This document covers the paper-only bridge between Strategy Discovery candidates
and the v8 validation stack. Every candidate is normalized into a deterministic
manifest, replayed through paper execution, checked by Phase 5 and Phase 6,
summarized by Paper Observability, and prepared for GitHub issue review.

It does not place orders, touch capital, call broker/exchange write APIs,
promote a candidate, close issues, or enable live deployment.

## Safety Boundary

Every candidate and batch output must preserve:

```text
paper_only=true
capital_at_risk=false
broker_exchange_write_enabled=false
live_exchange_write_enabled=false
```

Unsafe candidates with broker/live write flags are rejected and retained in the
batch audit output with `status=candidate_invalid`.

## Candidate Contract

Candidate JSONL rows use the `StrategyCandidate` contract:

```text
candidate_id
candidate_family
strategy_name
created_at
source
source_commit_sha
source_artifact_sha256
feature_contract_sha256
dataset_contract_sha256
policy_config
execution_config
risk_config
expected_instruments
expected_regime_keys
paper_only=true
capital_at_risk=false
broker_exchange_write_enabled=false
live_exchange_write_enabled=false
```

The candidate SHA-256 and `candidate_manifest.json` are deterministic for the
same payload and replay config.

## CLI

```bash
PYTHONPATH=src python examples/v8/run_strategy_candidate_replay.py \
  --candidate-file examples/v8/strategy_candidates/candidates.jsonl \
  --output-dir examples/v8/strategy_candidate_runs/batch_001 \
  --repo phead198708/BiGan \
  --issue-number 127 \
  --mode gh-command \
  --duration-seconds 300
```

Expected console summary:

```text
batch_id
candidate_count
ready_for_manual_review_count
blocked_count
critical_alert_candidate_count
ranking_path
batch_summary_path
```

## Per-Candidate Outputs

```text
<batch>/<candidate_id>/
  candidate_manifest.json
  paper_run/
    paper_run_summary.json
    paper_bundle_manifest.json
    phase5_safety_layer_report.json
    phase6_cicd_pipeline_report_<release_id>.json
    feed_health_report.json
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
  candidate_replay_summary.json
```

Invalid candidates still receive `candidate_replay_summary.json` and remain in
the batch registry/ranking.

## Batch Outputs

```text
strategy_candidate_batch_manifest.json
strategy_candidate_registry.json
strategy_candidate_ranking.json
strategy_candidate_ranking.md
strategy_candidate_batch_summary.md
```

Ranking is deterministic and conservative:

```text
1. ready candidates first
2. demote invalid / critical / Phase 5 or Phase 6 blocked candidates
3. prefer lower drawdown
4. prefer lower cost drift
5. then compare paper return metrics
```

No ranking status implies live approval. `ready_for_manual_review` means only
that the candidate passed the paper-only replay evidence for human review.

## Validation

```bash
PYTHONPATH=src python -m pytest tests/v8/test_strategy_discovery_paper_integration.py -q
```

The v8 hard gate includes this test alongside the existing Phase 0-6, paper
harness, paper soak, read-only shadow soak, observability, and alert delivery
tests.
