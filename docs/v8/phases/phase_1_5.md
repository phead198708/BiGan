# v8 Phase 1.5 - Policy Training Runner

Phase 1.5 is the first model-training node after the Phase 0 data firewall and
Phase 1 pure policy-learning primitives. It does not perform live trading or
execution simulation.

## Flow

```
accepted Phase 0 dataset
  -> Phase 1 policy dataset
  -> temporal train/shadow split
  -> train XGBoost policy on train rows only
  -> validate shadow acceptance on shadow rows only
  -> write candidate registry artifacts
```

The public entrypoint is:

```python
run_policy_training(phase0_dataset, PolicyTrainingRunConfig(...))
```

## Candidate Registry

When `output_dir` is provided, the runner writes a local registry record:

```text
<output_dir>/<run_id>/
  policy_dataset_manifest.json
  split_manifest.json
  training_manifest.json
  shadow_acceptance_report.json
  run_manifest.json
  model.xgb
```

The model artifact is written after model training succeeds. The candidate is
marked accepted only when `acceptance_report.passed` is true.

Rejected runs may still be written for debugging, but `run_manifest.json`
records:

```json
{
  "accepted": false,
  "candidate_status": "rejected"
}
```

## Hard Gates

The runner fails closed for:

- Phase 0 artifact gate failure
- policy dataset construction failure
- temporal split failure
- model training failure
- split provenance validation failure
- shadow acceptance failure

The run manifest records the hashes needed to distinguish train rows, shadow
rows, direct training labels, and shadow acceptance returns.

## Reproducibility

`run_id`, `policy_dataset_hash`, and `split_hash` are deterministic for the same
Phase 0 input and config. `created_at` is part of the config so callers can make
full artifact output deterministic in tests and audits.
