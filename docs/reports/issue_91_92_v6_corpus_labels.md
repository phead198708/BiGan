# Issue 91/92 V6 Corpus Labels

## Field Audit

The canonical feature/label join now exposes v6-ready training rows through
`src/bigan/modeling/dataset.py::assemble_training_dataset`. New label batches
write `bigan-labels-15m-profitability-v1.2.0`.

Available fields:

| Requirement | Current field |
| --- | --- |
| event id | `event_id` synthetic sample key: `round_slug/source_market/source_symbol:feature_ts` |
| market id | `market_id` from `source_market` |
| family / horizon | `family`, `horizon`, plus `canonical_symbol` |
| decision time | `decision_ts` and `feature_ts`; feature snapshot is point-in-time at `feature_ts` |
| settlement outcome | `direction_up_15m`, old profitability labels, and `label_settlement_3way` |
| settlement margin | `settlement_margin`, `settlement_abs_margin`, `settlement_neutral_margin` |
| volatility path labels | `max_exit_gain_*`, `label_volatility_*`, `time_to_best_exit_*`, `best_exit_price_*`, `volatility_path_validity_*` |

Coverage caveat: historical `labels_15m_v1` generated from Gamma round metadata
has only `priceToBeat` / `finalPrice`. It does not contain intra-round top-of-book
paths. Rows generated from that source therefore set
`volatility_path_validity_up/down = missing_price_path` and leave volatility
labels null. Usable volatility labels require raw WS / low-latency
`raw_top_of_book` coverage for both token sides from the decision time through
the safe exit window.

## Settlement 3-Way Rule

`label_settlement_3way` is direction-based, not profitability-based:

- `UP`: `target_price - start_price > settlement_neutral_margin`
- `DOWN`: `target_price - start_price < -settlement_neutral_margin`
- `NEUTRAL`: `abs(target_price - start_price) <= settlement_neutral_margin`

The old labels remain unchanged:

- `direction_up_15m`: pure binary settlement direction used by existing code.
- `label_profit_up_15m` / `label_profit_down_15m`: whether buying that token at
  the feature row's entry price would be profitable after fees.

This keeps the v6 settlement head from treating `1 - p_up` as down confidence,
while preserving profitability as a separate execution target.

## Volatility Window

The shared implementation is `src/bigan/labels/v6.py`.

For each sample and side (`UP`, `DOWN`):

1. Decision time is `feature_ts == decision_ts`.
2. Entry uses the first top-of-book ask at or after `decision_ts`, within
   `max_entry_wait_ms`.
3. Exit candidates use bid quotes strictly after `decision_ts`, no earlier than
   the entry quote, and no later than `round_end_ts - safety_window`.
4. Entry worst price is `min(0.99, ask + buy_slippage + entry_fee)`.
5. Exit worst price is `max(0.01, bid - sell_slippage - exit_fee)`.
6. `max_exit_gain = best_exit_worst_price - entry_worst_price`.

Quality flags:

- `valid`: path was usable and entry passed the cheap-token floor.
- `entry_price_below_min`: path exists, but the sample violates the configured
  cheap-token floor.
- `missing_entry_quote`, `missing_exit_path`, `no_exit_window`,
  `missing_price_path`: label is not usable for training.

The Phase 4 analyzer now calls the same shared max-exit math, so live replay and
v6 corpus labels use one bid/ask + slippage definition.

## Thresholds

Default volatility label threshold is `min_exit_gain = 0.15`.

Candidate sweep set shipped in code:

`0.08, 0.10, 0.12, 0.15, 0.20`

The threshold is intentionally above the observed round-trip drag:

- buy + sell slippage baseline is about `0.04`
- issue #87 account-vs-theoretical drag was about `0.072` per trade
- the remaining margin is a safety buffer for fill quality and dust

Training manifests now include `v6_label_diagnostics`, with settlement class
balance and volatility positive/coverage rates per split and per family.

## Re-Run

Regenerate settlement labels for a window:

```bash
python -m bigan.ingestion labels-15m-v1 \
  --since-ms 1780000000000 \
  --until-ms 1780003600000 \
  --settlement-neutral-margin 0.0
```

Assemble a v6-ready training dataset:

```bash
python - <<'PY'
from bigan.modeling.dataset import assemble_training_dataset

assemble_training_dataset("data/warehouse", "data/datasets/v6-settlement-volatility")
PY
```

Manual replay acceptance still needs an operator-reviewed sample of at least
200 rows for the target time window. The checks to record are:

- feature snapshot timestamp equals `decision_ts`
- no exit quote at or before `decision_ts` contributes to `max_exit_gain`
- both UP and DOWN side paths are checked independently
- class balance and price-path coverage are copied from `manifest.json`
