# Model Rollback Runbook

Owner: ML serving on-call  
Scope: BiGan online 15-minute direction models

## Trigger Conditions

Start rollback triage when any production model violates one of these gates:

- Health check fails for 3 consecutive probes.
- p95 serving latency is above the configured SLA for 5 consecutive minutes.
- Prediction error rate is above 2% for 5 consecutive minutes.
- Prediction distribution drift is critical: PSI >= 0.25 or Wasserstein distance >= 0.15 on a key feature/probability stream.
- Data missing rate is above 5% for 5 consecutive minutes.
- Schema mismatch incident is critical for any production request path.

## Manual Rollback Steps

1. Confirm the current online model from `current_online_models`.
2. Confirm the previous healthy champion from `model_registry`.
3. Freeze new deployments for the affected environment.
4. Route traffic to the rollback target model version.
5. Record the rollback with `rollback_deployment`, including operator and reason.
6. Keep the failed version in `rolled_back` deployment status and retire or demote it in `model_registry` after root cause review.

## Automatic Rollback Preconditions

Automatic rollback is allowed only when all checks pass:

- A previous healthy champion exists for the same `model_family`.
- The rollback target has compatible `feature_schema.json`.
- The rollback target artifact and calibration artifact can be loaded.
- The deployment table has no active rollback for the same environment.
- Current data quality incidents do not also affect the rollback target.

## Post-Rollback Verification

Run these checks after traffic is restored:

- `GET /health` returns `status=ok`.
- `GET /model-info` reports the rollback target model version.
- p95 serving latency is back under SLA for 10 minutes.
- Prediction events are still written to `prediction_events`.
- Outcomes and daily monitoring continue to populate `prediction_outcomes` and `model_monitoring_daily`.
- No new critical `data_quality_incidents` are opened.

## Notification And Audit

- Notify engineering and product owners with the model version, environment, trigger, rollback target, and operator.
- Link the alert ID, deployment ID, and incident ID in the incident channel.
- Attach the promotion report, shadow report if available, and monitoring snapshot.
- File a postmortem when user-facing predictions were stale, missing, or materially degraded.

## Database Schema Compatibility

Before rollback, verify:

- The rollback target feature schema matches the live feature producer.
- `prediction_events`, `prediction_outcomes`, and `model_monitoring_daily` schemas still accept the rollback target output.
- `model_registry` contains the rollback target artifact URI and status history.
- `model_deployments` can record the rollback event and target version.

## Drill Procedure

Perform a dry-run drill before enabling automatic rollback:

1. Insert a succeeded deployment for a known healthy model in a non-production DuckDB catalog.
2. Insert a second succeeded deployment for a test challenger.
3. Call `rollback_deployment` on the challenger deployment with the healthy model as `rollback_to_version`.
4. Query `current_online_models` and verify it resolves to the healthy model.
5. Record the drill evidence in the issue or release checklist.
