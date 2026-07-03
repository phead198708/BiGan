# xgboost-v5 Capped Live Shadow

Runbook for the capped Phase 4 **live shadow** of `xgboost-v5` using the operating
policy validated by the drawdown-gated cost-adjusted backtest
(`data/model-runs/xgboost-v5-run-20260529T053000Z/backtest-drawdown-gated/`).
This is the next gate before registering v5 and running the champion cutover.

Runtime evidence belongs in GitHub issue [#84](https://github.com/phead198708/BiGan/issues/84);
engineering follow-ups in [#85](https://github.com/phead198708/BiGan/issues/85).

## Operating Policy (locked)

| Control | Value | Rationale |
|---|---|---|
| `model_version` | `xgboost-v5` | First model with a cost-adjusted positive edge |
| `market_families` | `BTC-15M,ETH-15M` | Only 15M families showed real out-of-sample edge (AUC ~0.68-0.69); 5M was noise |
| `edge_threshold` | legacy alias | Kept for runner compatibility; do not use it as the single v5 decision threshold |
| `settlement_edge_threshold` | `0.45` | Settlement-confidence live-entry gate. This remains separate from volatility diagnostics |
| `volatility_score_threshold` | `0.50` | Legacy diagnostic field; issue #90 volatility entries are gated by orderbook-only expected exit gain |
| `volatility_min_entry_price` | `0.20` | Diagnostic orderbook floor for volatility opportunity analysis |
| `volatility_round_trip_cost` | `0.04` | Estimated buy+sell cost drag for volatility sleeve paper evidence |
| `volatility_safety_margin` | `0.02` | Required margin above cost drag before a paper volatility entry |
| `volatility_round_bankroll_usdc` | `1.0` | Per-round volatility sleeve bankroll reset |
| `volatility_min_order_size_usdc` | `0.05` | Stop volatility re-entry when the round bankroll falls below the floor |
| `max_position_size_usdc` | `1.0` | Capped blast radius for a live shadow |
| `max_concurrent_positions` | `1` | One position at a time during validation |
| `max_combined_concurrent_positions` | `2` | Allows one settlement hold plus one volatility paper position when explicitly enabled |
| `settlement_max_filled_per_side_per_round` | `1` | Prevents settlement concentration into repeated same-side fills in one round |
| `volatility_max_filled_per_side_per_round` | `1` | Allows volatility re-entry after sell, but caps same-side concentration per round |
| `max_rounds` | `6` | Small total entry budget |
| `daily_loss_limit_usdc` | `3.0` | Hard realized-loss stop |

The old backtest/replay threshold of `0.14` is diagnostic evidence only. For v5
live-shadow execution, settlement and volatility are separate sleeves:

- `settlement` can place capped FOK BUY orders, at most one filled position per
  round, and is held to `REDEEM` rather than sold before expiry.
- `volatility` may re-enter the same round only after the prior volatility
  position is sold. It uses a per-round $1 bankroll that resets each round, is
  refilled by realized account cash-flow PnL up to the $1 cap, and stops below
  the configured min-order floor.
- Both sleeves enforce per-round, per-side filled-entry caps to keep the v5
  down-side concentration from silently becoming repeated same-side exposure.
- `volatility` is orderbook-only paper until explicitly promoted. The gate is
  `expected_volatility_exit_gain >= round_trip_cost + safety_margin`; model edge
  is logged for diagnostics but is not the volatility decision surface.

v5 uses **family-aware calibration**: `BTC-15M` and `ETH-15M` each get an
independent calibrator with a global fallback. The live scorer
(`run_prediction_batch`) and the executor both resolve the family from
`canonical_symbol`, so probabilities/edges are calibrated per family end to end.

## Decision Gate

Same as `phase4_execution_validation.md`: Phase 4 runs are **diagnostic only**
until account cash-flow PnL reconciliation passes. Do not use
`realized_pnl_usdc` for promotion, capital sizing, or calibration weighting. Only
register v5 and proceed to the cutover gate after a capped run completes with
`open_positions_at_shutdown = 0`, `exits_pending_confirmation = 0`,
`exits_pending_settlement = 0`, and a matched reconciliation with no
`missing_cash_flow` rows.

The executor summary keeps `promotion_or_capital_sizing_evidence = false` and
`decision_evidence_allowed = false` until the account-history reconciliation
also shows acceptable account PnL. Volatility paper records are opportunity
diagnostics only; they must not influence capital sizing or calibration weights.

## Topology

Split deployment:

- **Local** (this machine): heavy capture/feature/model pipeline. Runs the v5
  scorer + the `champion_signal_bridge.py`, which SSHes validated signals to the
  execution host.
- **Execution host** (`54.250.242.139`): holds Polymarket credentials and runs
  the capped Phase 4 executor against a bridged signal JSONL queue. CPU stays
  low because it only re-checks CLOB liquidity and places orders. Account history
  export + cash-flow reconciliation also happen here (creds + history live here).

```
local: capture -> features -> v5 score (DuckDB) -> bridge --remote ssh
                                                       |
                                                       v
host 54.250.242.139:  champion-signals.jsonl -> phase4 executor (caps) -> CLOB
```

## Artifacts

- Model: `data/model-runs/xgboost-v5-run-20260529T053000Z/model/model.json`
- Family-aware calibration: `data/model-runs/xgboost-v5-run-20260529T053000Z/calibration-family/calibration.json`

## 1. Local: start the v5 scorer (15M only)

In one local terminal, run the live scorer restricted to 15M families with the v5
model and family-aware calibration. It captures BTC/ETH 15M markets, scores them,
and writes `model_version=xgboost-v5` predictions into the local monitoring
catalog.

```bash
MODEL_VERSION=xgboost-v5 \
MODEL_PATH=data/model-runs/xgboost-v5-run-20260529T053000Z/model/model.json \
CALIBRATION_PATH=data/model-runs/xgboost-v5-run-20260529T053000Z/calibration-family/calibration.json \
SCORING_CANONICAL_SYMBOL_LIKE='%-15M:%' \
MARKET_SPECS_JSON='[
  {"slug_prefix":"btc-updown-15m-","underlying":"BTC","horizon_minutes":15},
  {"slug_prefix":"eth-updown-15m-","underlying":"ETH","horizon_minutes":15}
]' \
./scripts/run_champion_live.sh
```

The `%-15M:%` LIKE matches both `BTC-15M:` and `ETH-15M:` canonical symbols.

## 2. Local: bridge v5 signals to the execution host

In a second local terminal, tail the local v5 predictions and append them to the
remote signal queue over SSH. The bridge is now family-aware — pass both 15M
families so ETH-15M signals are not dropped (the bridge historically hardcoded
`BTC-15M`).

```bash
python scripts/champion_signal_bridge.py \
  --monitoring-db-path data/mlops/champion_catalog.duckdb \
  --model-version xgboost-v5 \
  --market-families BTC-15M,ETH-15M \
  --remote ubuntu@54.250.242.139 \
  --remote-path /home/ubuntu/BiGan/data/live/remote-signals/xgboost-v5-signals.jsonl \
  --start latest
```

## 3. Execution host: start the capped live-shadow executor

On `54.250.242.139`, with Polymarket credentials already present, read the bridged
queue and trade the v5 settlement policy. This default mode places REAL
settlement-sleeve FOK BUY orders and holds them to settlement, so keep the caps
small. The runner refuses to start unless `CONFIRM=yes`.

```bash
# on 54.250.242.139, inside the BiGan checkout
CONFIRM=yes \
SIGNAL_JSONL_PATH=/home/ubuntu/BiGan/data/live/remote-signals/xgboost-v5-signals.jsonl \
./scripts/run_xgboost_v5_capped_live_shadow.sh
```

## 3a. Execution host: collect volatility sleeve paper evidence

Before any real volatility trading, run the orderbook-only paper path. This still
uses the CLOB client for quotes, but it does not post BUY or SELL orders.

```bash
# on 54.250.242.139, inside the BiGan checkout
PAPER=true \
ENABLE_VOLATILITY_SLEEVE=true \
SIGNAL_JSONL_PATH=/home/ubuntu/BiGan/data/live/remote-signals/xgboost-v5-signals.jsonl \
./scripts/run_xgboost_v5_capped_live_shadow.sh
```

Do not set `--enable-volatility-live-entries` for Phase 4 v5. Promotion to real
volatility orders requires a separate paper evidence review that shows positive
account cash-flow reconciliation after costs and no min-size/concurrency
breaches.

If the runner is not yet deployed to the host, call the executor directly with the
same locked policy:

```bash
python scripts/polymarket_phase4_live_champion_executor.py \
  --model-version xgboost-v5 \
  --market-families BTC-15M,ETH-15M \
  --settlement-edge-threshold 0.45 \
  --volatility-score-threshold 0.50 \
  --volatility-min-entry-price 0.20 \
  --volatility-round-trip-cost 0.04 \
  --volatility-safety-margin 0.02 \
  --signal-jsonl-path /home/ubuntu/BiGan/data/live/remote-signals/xgboost-v5-signals.jsonl \
  --signal-jsonl-start tail \
  --max-position-size-usdc 1.0 \
  --max-concurrent-positions 1 \
  --max-combined-concurrent-positions 2 \
  --settlement-max-filled-per-side-per-round 1 \
  --volatility-max-filled-per-side-per-round 1 \
  --max-rounds 6 \
  --daily-loss-limit-usdc 3.0 \
  --max-runtime-minutes 120 \
  --log-path logs/xgboost-v5-live-shadow/phase4.jsonl \
  --summary-path logs/xgboost-v5-live-shadow/phase4-summary.json
```

The executor logs a `phase4_started` config (including `model_version`,
`market_families`, and `edge_threshold`) and writes a summary JSON at shutdown.
Family filtering is defense-in-depth: the bridge filters families on the way out,
and the executor skips any disallowed family on the way in (counted under
`skipped.market_family_not_allowed`).

## 4. Execution host: reconcile account cash flow

After the run, export the Polymarket account history CSV on the host and reconcile,
as in `phase4_execution_validation.md`:

```bash
python scripts/reconcile_polymarket_cashflows.py \
  --history-csv /path/to/Polymarket-History.csv \
  --db-path data/mlops/champion_catalog.duckdb \
  --write-db \
  --report-path docs/reports/xgboost_v5_live_shadow_cashflow.md \
  --summary-json-path docs/reports/xgboost_v5_live_shadow_cashflow_summary.json

python scripts/reconcile_stale_execution_positions.py \
  --history-csv /path/to/Polymarket-History.csv \
  --db-path data/mlops/champion_catalog.duckdb \
  --write-cashflow-db \
  --report-path docs/reports/xgboost_v5_live_shadow_stale_positions.md
```

## 5. Promotion decision

Only if the reconciled `account_cash_pnl_usdc` is non-negative and the lifecycle
is clean: register v5 and proceed to the champion cutover gate. Otherwise, treat
the run as diagnostic, capture findings on issue #84, and iterate before any
capital sizing.

## Related

- `data/model-runs/xgboost-v5-run-20260529T053000Z/backtest-drawdown-gated/` — drawdown-gated cost-adjusted edge evidence
- `docs/reports/backtest_v5_vs_v4/README.md` — initial cost-adjusted edge evidence
- `docs/reports/shadow_v5_vs_v4/README.md` — per-family shadow comparison
- `docs/runbooks/phase4_execution_validation.md` — base Phase 4 controls
- `docs/runbooks/champion_promotion.md` — cutover gate
