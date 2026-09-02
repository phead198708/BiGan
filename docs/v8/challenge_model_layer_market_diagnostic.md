# Challenge model-layer market diagnostic

Status: **diagnostic complete; training not started**

This is outcome-aware development analysis. The exact-195 and legacy 15m corpora are permanently ineligible for promotion evidence.

## Market decision

**Recommendation: turn the new development lane to BTC 15m.** Keep exact-195 5m as a secondary research corpus, but do not start a new 5m lane yet.

The broad-support 5m variants do not earn their structural costs: iteration 4 has unit net PnL `-0.389000` over `76` bets and iteration 5 has `-0.183250` over `53` bets. The sparse 23-bet variant is positive (`1.014250`) but does not provide broad support.

The legacy 15m v7 selection has 119 bets and reported after-cost PnL `12.620000`. After charging an extra full observed source-token spread to every opposite-side quote proxy, the conservative PnL remains `12.390000`. This is development evidence only and is not directly comparable to a future confirmatory window.

## 5m cost/edge decomposition

| Variant | Bets | DOWN/UP | Mid-mark edge | Spread | Fee | Slippage | Liquidity | Net PnL | Cost / positive signal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6_7 | 193 | 137/56 | -2.370000 | 1.930000 | 0.038600 | 1.130000 | 0.009650 | -5.478250 | n/a |
| v8_1_23_bet | 23 | 22/1 | 1.395000 | 0.225000 | 0.004600 | 0.150000 | 0.001150 | 1.014250 | 0.273 |
| iteration_1 | 5 | 4/1 | 1.285000 | 0.065000 | 0.001000 | 0.035000 | 0.000250 | 1.183750 | 0.079 |
| iteration_3 | 5 | 4/1 | 1.285000 | 0.065000 | 0.001000 | 0.035000 | 0.000250 | 1.183750 | 0.079 |
| iteration_4 | 76 | 75/1 | 0.910000 | 0.810000 | 0.015200 | 0.470000 | 0.003800 | -0.389000 | 1.427 |
| iteration_5 | 53 | 52/1 | 0.780000 | 0.620000 | 0.010600 | 0.330000 | 0.002650 | -0.183250 | 1.235 |

For sell-before-close rows, spread cost includes the entry half-spread and the half-spread at the actual runtime guard exit snapshot. Every identity `mid-mark edge - spread - fee - slippage - liquidity = unit net PnL` is checked before report generation.

## DOWN concentration

- Exact-195 outcomes: UP `96`, DOWN `99`.
- BTC market returns: positive/tie `96`, negative `99`.
- v8.1 23-bet side mix: DOWN `22`, UP `1`.
- All-decision UP mid-price minus realized UP frequency: `0.011747`; market bootstrap 95% interval `[-0.042604, 0.065454]`.

The corpus is not a DOWN regime. The point estimate is compatible with mild UP overpricing, but its interval includes zero. The defensible attribution is model/controller side asymmetry, not proven structural longshot bias. New labels, features, calibration, and evaluation must therefore be side-symmetric and report UP/DOWN strata separately.

## Feature inventory

| Feature family | Archived stream | Exact-195 causal coverage | Constraint / caveat |
|---|---|---:|---|
| paired top-of-book and structural spread | order books | 195/195 | latest paired book available no later than decision_ts |
| depth dynamics and queue state | order books | 195/195 | rolling windows end at decision_ts; no exit snapshots |
| order-flow imbalance and trade intensity | trade tape + order books | 193/195 | trade available_at <= decision_ts; 2 legacy markets have no tape, and numeric zero must not substitute for missing |
| Chainlink reference displacement | Chainlink RTDS + decision-time market price | 195/195 | reference available_at and max_input_ts must both be <= decision_ts |
| causal BTC momentum and volatility | BTC reference klines / ticks | 195/195 | closed candle/tick available_at <= decision_ts; current candle close forbidden |
| side-symmetric relative-value transforms | paired order books + Chainlink + model-free BTC features | 195/195 | derive both sides from the same causal snapshot and share transformations |

Trade-tape raw rows are non-empty for 193/195 markets. Legacy action rows contain numeric zero even when missingness was not separately encoded, so the safe coverage for order-flow features is 193/195, not 195/195.

## 15m limitation and retraining target

The legacy evaluator had true source-token asks for 96/119 selected trades. For 23 DOWN trades it used the complement of an UP ask, which is an optimistic proxy rather than a true paired DOWN ask. The report preserves both the original numbers and a conservative one-full-spread correction. New 15m collection must store both token books and train only on executable side-specific asks.

No model training was started. All paper/live/write/wallet/handoff/promotion paths remain false.

JSON report payload SHA-256: `b72c870c57d44a1c06677ad0766421e3d4732e48075ed25e85f13cb95c37414e`
