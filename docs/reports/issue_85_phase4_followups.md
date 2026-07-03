# Issue 85 Phase 4 Follow-Ups

Generated: 2026-05-28

GitHub: [#85 Phase 4 follow-ups: cash-flow PnL reconciliation and execution controls](https://github.com/phead198708/BiGan/issues/85)

## Summary

Issue #85 tracks engineering work from the capped Phase 4 run `20260527T143533Z` (`entries_filled=7`, clean lifecycle, `realized_pnl_usdc=-0.09387443`). That run proved execution lifecycle coverage but not account-accurate PnL or promotion readiness.

This change set implements the code and runbook pieces needed before another capped validation run can be treated as go/no-go evidence.

## Work Item Status

| Item | Status | Notes |
|---|---|---|
| Account-cash-flow PnL reconciliation | Implemented | `execution_cashflow_reconciliations` (#79) plus per-leg `execution_cash_legs` persisted from CLOB fills |
| Dual PnL in reports | Implemented | Reconcile script + executor summary expose theoretical vs account views |
| Stale/open row reconciliation | Implemented | `reconcile_stale_open_positions()` + `scripts/reconcile_stale_execution_positions.py` |
| Cheap / near-threshold entry controls | Implemented | Default `min_entry_price=0.35` with near-min edge/time gates |
| Force-exit behavior | Implemented | Soft force exit defers when `bid < 0.15`; hard force exit unchanged |
| Entry filtering review | Documented | Latest skip mix remains conservative by design; see below |
| Runbook decision gate | Implemented | `docs/runbooks/phase4_execution_validation.md` |

## Entry Policy Decision

Runtime used `min_entry_price=0.30`. The largest loss came from the lowest fill (`0.31`) followed by a `soft_force_exit` into `0.11` on `btc-updown-15m-1779913800`.

Decision for the next capped run:

- Raise the floor to **`0.35`**.
- For quotes in `(min_entry_price, min_entry_price + 0.05]`, require:
  - fresh edge at worst price **`>= 0.50`**
  - seconds to expiry **`>= 420`**
- Keep the global edge threshold at **`0.45`** outside the near-min band.

## Force-Exit Decision

All `soft_force_exit` closes in run `20260527T143533Z` were losers. Soft exits now defer when the bid is below **`0.15`**, allowing the position to reach the hard force-exit window instead of fire-selling immediately into very weak bids.

Hard force exit timing and exit confirmation retry behavior are unchanged.

## Skip-Count Interpretation (run `20260527T143533Z`)

| Skip reason | Count | Interpretation |
|---|---:|---|
| `no_new_entry_window` | 124 | Expected: `no_new_entry_before_expiry_seconds=300` blocks late entries |
| `below_edge_threshold` | 78 | Expected: `edge_threshold=0.45` is strict |
| `round_already_filled` | 54 | Expected: one fill per round cap |
| `entry_price_below_min` | 18 | Will increase with `min_entry_price=0.35` |
| `fresh_edge_below_threshold` | 15 | Near-min gating will split some of this into `near_min_entry_*` skips |

The skip mix is intentionally conservative. It is hiding some executable signals, but that is acceptable until account-cash-flow PnL reconciliation passes.

## Next Validation Run

1. Run a capped BTC-15M session with the new defaults (same small size as `20260527T143533Z`).
2. Export Polymarket account history.
3. Run `reconcile_polymarket_cashflows.py` and `reconcile_stale_execution_positions.py`.
4. Append the runtime summary to issue #84.
5. Only if account cash PnL matches history and no rows remain `open` / pending, reconsider larger size or promotion evidence.
