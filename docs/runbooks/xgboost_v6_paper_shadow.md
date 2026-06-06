# xgboost-v6 Paper / Orderbook-Only Shadow

Status: **evidence path only - not promotion**

## Purpose

Exercise the 15M mixed `xgboost-v6` joint gate (`p_up`, `p_down`, `p_neutral`, `p_vol_up`, `p_vol_down`) against live CLOB quotes without posting orders. Reconcile **account cashflow** from [Polymarket history](https://polymarket.com/portfolio?tab=history) before any champion decision.

## Default gate (15M mixed model)

- `settlement_threshold=0.50`
- `neutral_cap=0.25`
- `volatility_threshold=0.60`
- `round_trip_cost=0.072`, `ev_margin=0.01`

## Local scorer

```bash
export V6_SIGNAL_QUEUE=data/live/xgboost-v6-paper-signals.jsonl
: > "${V6_SIGNAL_QUEUE}"

MODEL_VERSION=xgboost-v6 \
MODEL_PATH=data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/model-single-grid/model.json \
LOW_LATENCY_FEATURE_QUEUE_ENABLED=true \
LOW_LATENCY_RAW_QUEUE_CANONICAL_SYMBOL_PREFIX='BTC-15M:' \
SCORING_CANONICAL_SYMBOL_LIKE='BTC-15M:%' \
SIGNAL_JSONL_OUTPUT_PATH="${V6_SIGNAL_QUEUE}" \
SIGNAL_JSONL_MARKET_FAMILIES=BTC-15M \
SIGNAL_JSONL_OUTCOME_SIDE=ANY \
SIGNAL_JSONL_MAX_EVENT_AGE_SECONDS=180 \
MARKET_SPECS_JSON='[{"slug_prefix":"btc-updown-15m-","underlying":"BTC","horizon_minutes":15}]' \
./scripts/run_champion_live.sh
```

`predictions-v1` auto-detects the v6 artifact schema and writes `p_up` / `p_vol_*` into `predictions`, `prediction_events`, and the executor-ready `SIGNAL_JSONL_OUTPUT_PATH` queue. Use the queue for paper shadow; DuckDB scanning is diagnostic-only because it can replay stale signals.

The scorer queue is **current-round only**: appends are limited to the freshest round in each scorer batch, stale signals older than `SIGNAL_JSONL_MAX_EVENT_AGE_SECONDS` are not appended, and the queue is rewritten when that round changes. Executors started with `SIGNAL_JSONL_START=tail` must still tolerate this rewrite; the Phase 4 executor resets its line cursor automatically if the file is truncated by a round rotation.

## Paper executor

```bash
SIGNAL_JSONL_PATH="${V6_SIGNAL_QUEUE}" \
SIGNAL_JSONL_START=tail \
V6_SETTLEMENT_MIN_CONFIDENCE=0.80 \
MAX_SIGNAL_AGE_SECONDS=180 \
bash scripts/run_xgboost_v6_paper_shadow.sh
```

Split topology: either rsync/SSH the scorer-written queue to the executor host, or use `scripts/champion_signal_bridge.py` as a fallback bridge. In both cases, the executor must start with `SIGNAL_JSONL_PATH`; set `REQUIRE_SIGNAL_JSONL=false` only for diagnostic DuckDB scans.

## Experimental settlement exits

The settlement sleeve still defaults to hold-to-settlement. Issue #97/#98 exit policies are opt-in and paper-shadow evidence only:

```bash
SETTLEMENT_ALLOW_MID_ROUND_EXIT=true \
SETTLEMENT_REVERSAL_MIN_CONFIDENCE=0.75 \
SETTLEMENT_REVERSAL_HYSTERESIS_BARS=2 \
SETTLEMENT_CONFIDENCE_DECAY_ENABLED=true \
SETTLEMENT_DECAY_FLOOR=0.55 \
SETTLEMENT_DECAY_DELTA=0.25 \
SETTLEMENT_DECAY_OPPOSITE_MIN_CONFIDENCE=0.75 \
SETTLEMENT_DECAY_HYSTERESIS_BARS=2 \
SETTLEMENT_PRICE_STOP_ENABLED=true \
SETTLEMENT_STOP_PRICE_DELTA=0.15 \
SETTLEMENT_STOP_LOSS_USDC=0.50 \
SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_VETO_ENABLED=true \
SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_MIN_CONFIDENCE=0.80 \
SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_MAX_AGE_SECONDS=180 \
bash scripts/run_xgboost_v6_paper_shadow.sh
```

- Reversal exit uses only settlement probabilities: the opposite side must be admitted above the reversal confidence for consecutive fresh signals before selling the existing settlement position. The current data-tuned default is `p_opposite >= 0.75` for `2` consecutive signals; in the 2026-06-04 queue-first 30-round shadow log, this caught the two filled rounds that later resolved opposite and did not trigger on winning filled rounds, while `0.80/2` missed one of those two reversals.
- Confidence decay exit uses the fill-time settlement confidence baseline and exits when same-side confidence weakens below floor or delta while the opposite side becomes larger and passes the same consecutive strong-opposite confirmation.
- Price stop exit uses the current bid versus fill price / unrealized PnL, but the stop can execute only after the same confirmed opposite-reversal condition is present (`p_opposite >= 0.75` for `2` consecutive fresh signals by default). A price or loss breach without confirmed reversal is held and logged as `reversal_confirmation_required`; when same-side confirmation veto is enabled, a fresh post-entry same-side settlement confidence confirmation can still skip the confirmed-reversal price stop for that poll.

## Artifacts

- Executor log: `logs/xgboost-v6-paper-shadow/phase4-*.jsonl`
- Summary: `logs/xgboost-v6-paper-shadow/phase4-*-summary.json`
- Offline replay reference: `docs/reports/issue_93_94_v6_btc15m_execution_restricted_replay_20260602.md`

## Do not

- Set `CONFIRM=yes` on the v6 script (live settlement is blocked).
- Promote on paper PnL or fill-price PnL alone.
