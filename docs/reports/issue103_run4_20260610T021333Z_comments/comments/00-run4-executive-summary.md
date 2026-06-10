### Run `20260610T021333Z` — take-profit paper (executive summary)

**Stop:** `stop_max_runtime` (180min) @ 2026-06-10T05:22:14Z  
**Window:** 2026-06-10T02:22:09Z → 2026-06-10T05:22:14Z  
**Fills / closes:** 16 fills, 5 mid-round exits (4 take-profit, 1 reversal)  
**PnL:** log `realized_pnl_usdc=+0.535` | gamma-reconciled total **-2.90 USDC** (16 bets, 7W/9L)

**Take-profit config:** `convergence_take_profit_enabled=true`, `take_profit_hold_edge=0.03`, `force_exit_seconds=180`, hysteresis=2.

**Key findings (no executor code changed in this run):**
1. Take-profit exits worked when triggered: 4 exits **+1.39 USDC** combined.
2. 9 positions hit `exit_pending_settlement`; only 1 `paper_settlement_resolved` in log — Gamma reconcile required for true PnL.
3. Most held-to-settlement bets **did receive signals** (v7 evaluated 1–35×); exits blocked because `hold_edge` stayed above take-profit threshold or only `REDUCE` fired.
4. Bet #13 (`1781065800` UP): 6× `convergence_force_exit_before_expiry` recommended but `v7_settlement_position_exit_skipped: missing_bid` (dust size after repeated reduces).
5. Low entry price (<0.30): 5 bets, **-3.15 USDC** combined.

**Artifacts:**
- Log: `data/logs/xgboost-v7-paper-shadow-20260610T021333Z-event5s-30round-takeprofit/phase4-20260610T022208Z.jsonl`
- Summary: `logs/xgboost-v7-paper-shadow/phase4-20260610T021333Z-summary.json`
- Gamma reconcile: `logs/xgboost-v7-paper-shadow/phase4-20260610T021333Z-gamma-reconcile.json`
- Per-bet PnL: `logs/xgboost-v7-paper-shadow/v7_run4_20260610T021333Z_per_bet_pnl.json`

Per-round detail in the following comments (chronological).
