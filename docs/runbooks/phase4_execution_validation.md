# Phase 4 Execution Validation

Local runbook for bounded BTC-15M champion-signal execution. Runtime evidence belongs in GitHub issue [#84](https://github.com/phead198708/BiGan/issues/84); engineering follow-ups are tracked in [#85](https://github.com/phead198708/BiGan/issues/85).

## Decision Gate

Phase 4 runs are **diagnostic only** until account-cash-flow PnL reconciliation passes:

- Do not use executor `realized_pnl_usdc` for promotion, capital sizing, or champion calibration weighting.
- Treat `theoretical_pnl_usdc` (fill-price ledger) and `account_cash_pnl_usdc` (Polymarket cash flow) as separate views.
- Only accept promotion evidence after a capped validation run completes with:
  - `open_positions_at_shutdown = 0`
  - `exits_pending_confirmation = 0`
  - `exits_pending_settlement = 0`
  - matched account-history reconciliation with no `missing_cash_flow` rows

## Default Execution Controls (issue #85)

| Control | Default | Rationale |
|---|---|---|
| `min_entry_price` | `0.35` | Raised from `0.30` after the `0.31 -> 0.11` soft-force-exit loss in run `20260527T143533Z` |
| `near_min_price_band` | `0.05` | Applies stricter gates to quotes just above the floor |
| `near_min_fresh_edge_threshold` | `0.50` | Cheap entries need stronger fresh-edge confirmation |
| `near_min_seconds_to_expiry` | `420` | Cheap entries need more time before expiry |
| `soft_force_exit_min_bid` | `0.15` | Defers soft force exits into fire-sale bids; hard force exit still applies later |

## Reconciliation Workflow

After each capped run:

1. Export Polymarket account history CSV.
2. Reconcile cash flow and persist dual PnL:

```bash
python scripts/reconcile_polymarket_cashflows.py \
  --history-csv /path/to/Polymarket-History.csv \
  --db-path data/mlops/champion_catalog.duckdb \
  --write-db \
  --report-path docs/reports/issue_85_cashflow_reconciliation.md \
  --summary-json-path docs/reports/issue_85_cashflow_reconciliation_summary.json
```

3. Reconcile any stale `open` rows still in the execution DB:

```bash
python scripts/reconcile_stale_execution_positions.py \
  --history-csv /path/to/Polymarket-History.csv \
  --db-path data/mlops/champion_catalog.duckdb \
  --write-cashflow-db \
  --report-path docs/reports/issue_85_stale_position_reconciliation.md
```

4. Compare executor summary:
   - `theoretical_pnl_usdc` from `execution_positions`
   - `account_cash_pnl` from `execution_cashflow_reconciliations`
   - per-leg `execution_cash_legs` when CLOB fills were persisted during the run

## Executor Summary Fields

The Phase 4 executor summary JSON now includes:

- `status`: lifecycle-only result — `LIFECYCLE_PASS`, `LIFECYCLE_INCOMPLETE`, `CHECK`, or `FAIL`
- `lifecycle_complete`: `true` only when the run has fills and no open/pending exit or settlement rows at shutdown
- `realized_pnl_usdc`: in-session fill-price ledger total
- `theoretical_pnl_usdc`: persisted position ledger total at shutdown
- `account_cash_pnl_usdc`: `null` until history reconciliation is run offline
- `pnl_reconciliation_status`: `theoretical_only` until account cash flow is attached
- `promotion_or_capital_sizing_evidence`: always `false` until reconciliation passes
- `account_cashflow_reconciliation_required`: always `true` for Phase 4 runs

## Related Reports

- `docs/reports/issue_76_polymarket_history_reconciliation.md`
- `docs/reports/issue_79_cashflow_reconciliation.md`
