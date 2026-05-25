# Issue 67 Probability Clipping Hotfix

## Summary

The serving contract now clips served `prob_up_15m` values to `[0.05, 0.95]`. This is a serving-layer hotfix only: it does not mutate model artifacts, retrain xgboost-v4, or rewrite historical prediction rows.

## Behavior

- Input probability must still validate inside `[0.0, 1.0]`.
- After validation, `PredictResponse.prob_up_15m` is clipped to `[0.05, 0.95]`.
- The API contract exposes the postprocessing bounds under `probability_postprocessing`.
- Batch scoring artifacts still preserve raw model output in `raw_prob_up_15m`.

## Required Offline Comparison

Before treating the hotfix as promotion evidence, compare current live `prediction_events` against settled outcomes under:

| Variant | Bounds |
|---|---|
| no clip | none |
| light clip | `[0.05, 0.95]` |
| stronger clip | `[0.10, 0.90]` |

Report bucket-level Brier/ECE and high-up/high-down realized rates. The hotfix is a risk reducer, not a substitute for per-family calibration.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/serving/test_contracts.py
```
