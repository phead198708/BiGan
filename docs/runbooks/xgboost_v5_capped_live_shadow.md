# xgboost-v5 Capped Live Shadow

Runbook for the capped Phase 4 **live shadow** of `xgboost-v5` using the operating
policy validated by the cost-adjusted backtest
(`docs/reports/backtest_v5_vs_v4/README.md`). This is the next gate before
registering v5 and running the champion cutover.

Runtime evidence belongs in GitHub issue [#84](https://github.com/phead198708/BiGan/issues/84);
engineering follow-ups in [#85](https://github.com/phead198708/BiGan/issues/85).

## Operating Policy (locked)

| Control | Value | Rationale |
|---|---|---|
| `model_version` | `xgboost-v5` | First model with a cost-adjusted positive edge |
| `market_families` | `BTC-15M,ETH-15M` | Only 15M families showed real out-of-sample edge (AUC ~0.68-0.69); 5M was noise |
| `edge_threshold` | `0.08` | Net-positive, monotonic operating point from the backtest (+34.15 PnL @ 533 trades) |
| `max_position_size_usdc` | `1.0` | Capped blast radius for a live shadow |
| `max_concurrent_positions` | `1` | One position at a time during validation |
| `max_rounds` | `6` | Small total entry budget |
| `daily_loss_limit_usdc` | `3.0` | Hard realized-loss stop |

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
queue and trade the v5 policy. It places REAL FOK orders, so keep the caps small.
The runner refuses to start unless `CONFIRM=yes`.

```bash
# on 54.250.242.139, inside the BiGan checkout
CONFIRM=yes \
SIGNAL_JSONL_PATH=/home/ubuntu/BiGan/data/live/remote-signals/xgboost-v5-signals.jsonl \
./scripts/run_xgboost_v5_capped_live_shadow.sh
```

If the runner is not yet deployed to the host, call the executor directly with the
same locked policy:

```bash
python scripts/polymarket_phase4_live_champion_executor.py \
  --model-version xgboost-v5 \
  --market-families BTC-15M,ETH-15M \
  --edge-threshold 0.08 \
  --signal-jsonl-path /home/ubuntu/BiGan/data/live/remote-signals/xgboost-v5-signals.jsonl \
  --signal-jsonl-start tail \
  --max-position-size-usdc 1.0 \
  --max-concurrent-positions 1 \
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

## 4. Promotion decision

Only if the reconciled `account_cash_pnl_usdc` is non-negative and the lifecycle
is clean: register v5 and proceed to the champion cutover gate. Otherwise, treat
the run as diagnostic, capture findings on issue #84, and iterate before any
capital sizing.

## Related

- `docs/reports/backtest_v5_vs_v4/README.md` — cost-adjusted edge evidence
- `docs/reports/shadow_v5_vs_v4/README.md` — per-family shadow comparison
- `docs/runbooks/phase4_execution_validation.md` — base Phase 4 controls
- `docs/runbooks/champion_promotion.md` — cutover gate
