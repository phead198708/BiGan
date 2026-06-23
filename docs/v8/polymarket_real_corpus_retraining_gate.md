# v8 Polymarket Real Corpus Retraining Gate

This gate connects a successful real Polymarket recorder bundle to the existing
Polymarket policy training runner. It does not collect new data, trade, promote,
or deploy a model.

## Required Input

The recorder run must already contain:

- `real_corpus_recorder_manifest.json`
- `real_corpus_recorder_report.json`
- `phase2_corpus/polymarket_corpus_manifest.json`
- `phase2_corpus/polymarket_train_shadow_split.json`

The gate only trains when the recorder report proves:

- `phase2_corpus_build_eligible=true`
- `real_historical_training_eligible=true`
- `manual_live_evidence_eligible=true`
- `mock_public_data_used=false`
- `synthetic_public_data_used=false`
- `synthetic_corpus_used=false`
- `real_historical_corpus_used=true`
- `live_polymarket_data_read=true`
- `live_btc_reference_data_read=true`
- `public_collection_status=completed`

Mocked or synthetic recorder bundles are rejected fail-closed.

## Command

```bash
PYTHONPATH=src python examples/v8/run_polymarket_real_corpus_retraining_gate.py \
  --recorder-run-dir /path/to/real-recorder-run \
  --output-dir /tmp/bigan-v8-real-corpus-retraining \
  --overwrite-existing
```

## Outputs

The gate writes:

- `real_corpus_retraining_gate_report.json`
- `real_corpus_training_manifest.json`
- `policy_training/*/polymarket_policy_model_manifest.json`
- all standard #133 policy training artifacts when accepted

The model manifest is augmented with #134-readable provenance:

- `real_historical_corpus_used=true`
- `fixture_corpus_used=false`
- `synthetic_corpus_used=false`
- `synthetic_fixture_signal_used=false`
- `fixture_model_used=false`
- `manual_live_evidence_eligible=true`
- `policy_dataset_hash`
- `split_hash`
- `train_dataset_hash`
- `shadow_dataset_hash`
- `model_sha256`
- recorder and Phase 2 corpus manifest hashes

If the input bundle is not real-history eligible, training is not started and the
gate report records `gate_status=blocked_fail_closed` with explicit reason codes.
