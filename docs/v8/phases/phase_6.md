# Phase 6: CI/CD Strategy Deployment Pipeline

Phase 6 is the deployment correctness gate for the v8 trading system.

It does not train, trade, or optimize directly. It consumes deterministic
evidence from the earlier phases and decides whether a candidate is allowed to
enter staged live rollout.

## Lifecycle

```text
training
validation
shadow_deployment
live_deployment
monitoring
rollback
```

Each stage must produce immutable evidence:

```text
artifact_sha256
report_sha256
run_id
passed
metadata
```

The stage evidence stream is ordered and hashed. Missing, duplicated, or
out-of-order stages fail closed before deployment.

## Hard Gates

Phase 6 enforces these deployment rules:

```text
no unvalidated strategy goes live
rollback must be available before live deployment
rollback latency must be below threshold
training must be reproducible
validation must include OOS backtest and cost stress evidence
shadow deployment must have zero capital at risk
live deployment must be staged by capital fraction
monitoring must include performance and risk tracking
```

Live deployment is never allowed unless training, validation, and shadow gates
all pass first.

## Validation Evidence

Validation must include:

```text
oos_backtest_passed=true
cost_stress_passed=true
cost_stress_multipliers=[1.2, 1.5, 2.0]
```

The required cost-stress multipliers are configurable, but the default matches
the v8 cost-stress hard gate used by earlier phases.

## Staged Live Rollout

The rollout policy is deterministic:

```text
0.00
0.01
0.05
0.10
```

The first live step must not exceed the configured initial capital limit, and
the final live fraction must not exceed the configured max live capital limit.

Manual approval is required by default before the live deployment gate can pass.

## Rollback

Rollback evidence includes:

```text
stable_model_id
stable_model_sha256
safe_parameter_sha256
safe_parameters
rollback_artifact_sha256
latency_measurements_ms
```

`safe_parameter_sha256` is bound to the canonical hash of `safe_parameters`.
Any mismatch fails before a pipeline report is produced. Live deployment is
blocked when the maximum observed rollback latency exceeds the configured
threshold.

## Outputs

Phase 6 emits:

```text
phase6_cicd_pipeline_report.json
release_manifest
pipeline_input_sha256
release_manifest_sha256
stage_gates
rollback_gate
acceptance_criteria
deployment_status
```

`deployment_status` is either:

```text
approved_for_staged_live
blocked_fail_closed
```

## Acceptance Criteria

Phase 6 passes only if:

```text
full_pipeline_deterministic=true
reproducible_training_pipeline=true
validation_passed=true
shadow_deployment_passed=true
staged_live_deployment_passed=true
monitoring_enabled=true
rollback_available=true
rollback_latency_within_threshold=true
no_unvalidated_strategy_goes_live=true
```
