# Issue 66 Feature Null Investigation

## Summary

The high null rates were caused by sparse live Polymarket microstructure, not by a missing model artifact. The v5 feature layer now emits conservative neutral defaults for features that cannot be observed in a short round, so online inference does not silently receive `NaN` for core v4/v5 inputs.

## Findings

| Feature | Root Cause | Fix |
|---|---|---|
| `aggressor_buy_ratio_1m` | Many BTC/ETH 5m/15m token minutes have no `raw_trades` rows. The old aggregation returned `None` whenever the 1-minute trade window was empty. | Empty trade window now maps to neutral `0.5`. If trades exist, the observed BUY share is still used. |
| `avg_trade_size_1m` | Same sparse-trade cause. Empty trade window made `trade_volume / trade_count` undefined. | Empty trade window now maps to `0.0`. |
| `ret_30m` | Polymarket round tokens are short-lived. A 5m or 15m token usually cannot have 30 minutes of same-token quote history, so direct same-symbol lookup is structurally unavailable. | Direct 30m return is used when available; otherwise fallback uses `ret_15m * 2`, then `ret_5m * 6`, then `ret_1m * 30`, and finally neutral `0.0`. |
| `tick_obi_l1` / `tick_obi_l3` | Depth snapshots are occasionally missing or have zero usable bid/ask size. | Missing/zero denominator imbalance now maps to neutral `0.0`. |

## Production Interpretation

These defaults are intentionally conservative:

- `0.5` aggressor BUY ratio means no observed trade-side pressure.
- `0.0` average trade size means no observed trades.
- `0.0` imbalance means no observed depth skew.
- `0.0` fallback return means no usable same-token lookback evidence.

This reduces online schema/NaN risk while preserving observed values whenever the feed provides them. Historical clean-corpus rows are not rewritten by this code change; regenerate `features_15m_v1` for the target 7-day corpus before using this fix as training evidence.

## Verification

Focused tests cover sparse trade defaults, short-round `ret_30m` fallback, and neutral tick imbalance behavior:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/features/test_aggregation.py
```
