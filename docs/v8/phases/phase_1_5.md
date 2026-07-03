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
  dataset_profile.json
  split_manifest.json
  training_manifest.json
  shadow_acceptance_report.json
  run_manifest.json
  model.xgb
```

The model artifact is written after model training succeeds. The candidate is
marked accepted only when `acceptance_report.passed` is true.

Registry records are overwrite-protected by default. `run_id` is deterministic
and identifies a reproducible candidate spec:

```text
same Phase 0 input + same policy config + same split config + same model config
  -> same run_id
```

`created_at` is intentionally excluded from `run_id`. This keeps candidate
identity reproducible while the default `overwrite_existing=False` setting
prevents a later invocation with a different timestamp from silently mutating an
existing registry record. Callers must set `overwrite_existing=True` explicitly
to replace `<output_dir>/<run_id>/`.

Rejected runs may still be written for debugging, but `run_manifest.json`
records:

```json
{
  "accepted": false,
  "candidate_status": "rejected"
}
```

`dataset_profile.json` is immutable candidate metadata for training-corpus
observability. It records row counts, train/shadow UP-DOWN target balance,
source/instrument/regime distributions, model objective, boost rounds, target
encoding, positive-return threshold, and feature columns. The same payload is
embedded in `run_manifest.json` under `dataset_profile`, while the `artifacts`
block records `dataset_profile_path` and `dataset_profile_sha256`.

## Hard Gates

The runner fails closed for:

- Phase 0 artifact gate failure
- policy dataset construction failure
- temporal split failure
- model training failure
- split provenance validation failure
- shadow acceptance failure

The run manifest records the hashes needed to distinguish train rows, shadow
rows, direct training labels, and shadow acceptance returns. It also records
artifact paths and SHA-256 hashes for the local registry files, including the
dataset profile. The
`run_manifest_canonical_sha256` field is computed over the canonical JSON
payload with that self-hash field blanked, avoiding a circular file-hash
dependency.

## Reproducibility

`run_id`, `policy_dataset_hash`, and `split_hash` are deterministic for the same
Phase 0 input and config. `created_at` is part of the manifest/config for audit
metadata, but it is not part of the deterministic `run_id`.
