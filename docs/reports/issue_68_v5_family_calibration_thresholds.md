# Issue 68 V5 Family Calibration And Threshold Plan

## Implemented Plumbing

- Added family-aware calibration artifacts keyed by market family/horizon such as `BTC-15M`, `BTC-5M`, `ETH-15M`, and `ETH-5M`.
- Family calibration can compare Platt, isotonic, temperature, and beta calibration, selecting the lowest validation ECE per family.
- Family-aware calibration artifacts include a global fallback for small or single-class families.
- Added calibration clip bounds support, intended for post-calibration clipping such as `[0.03, 0.97]`.
- Added per-family edge-threshold search with the requested grid `[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]`, minimum expected-value gating, and optional turnover caps.
- Added bootstrap calibration gates for high-up/high-down bucket realized rates, global ECE regression, and positive family average realized return.
- Updated `docs/runbooks/champion_promotion.md` with the new calibration pass/fail requirements.

## Remaining Evidence

This code does not claim v5 is ready. The final v5 evidence still requires the settled seven-day multi-market corpus:

- Fit per-family calibrators on the final train/validation split.
- Compare per-family ECE against the current global v4 calibrator.
- Run per-family threshold search on the same validation/backtest window.
- Rerun bootstrap with bucket-level calibration metrics populated from settled outcomes.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/modeling/test_calibration.py \
  tests/backtest/test_strategy.py \
  tests/modeling/test_bootstrap.py
```
