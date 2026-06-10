## xgboost-v7 paper shadow — four-run case study

Per-round detail follows in chronological issue comments (issue #100 format).

| run | stop | window | rounds | fills | realized_pnl | reconciled_pnl | status |
|---|---|---|---:|---:|---:|---:|---|
| `20260608T055415Z` | stop_max_runtime (180min) | 2026-06-08T05:54:16.165000+00:00 → 2026-06-08T08:54:20.301000+00:00 | 13 | 19 | -0.3126 | 1.3400 | LIFECYCLE_INCOMPLETE |
| `20260608T133724Z` | stop_daily_loss_limit (-3 USDC) | 2026-06-08T13:37:24.653000+00:00 → 2026-06-08T16:05:09.101000+00:00 | 9 | 10 | -3.0004 | -4.1557 | LIFECYCLE_INCOMPLETE |
| `20260609T103055Z` | stop_daily_loss_limit (-3 USDC) | 2026-06-09T10:30:56.354000+00:00 → 2026-06-09T13:06:12.440000+00:00 | 11 | 13 | -3.1468 | -3.5348 | LIFECYCLE_INCOMPLETE |
| `20260610T021333Z` | stop_max_runtime (180min) | 2026-06-10T02:22:09.092000+00:00 → 2026-06-10T05:22:14.108000+00:00 | 12 | 16 | 0.5350 | -2.8965 | LIFECYCLE_INCOMPLETE |

**Shared config (runs 1–3):** BTC-15M only, `entry_gate_mode=v7-pnl`, settlement conf=0.75, edge=0.04, max_signal_age=30s, PM enabled + paper_execute, re-entry allowed, model `20260608Tevent-5s-v1`.

**Run 4 delta:** `convergence_take_profit_enabled=true`, `take_profit_hold_edge=0.03`, `take_profit_force_exit_seconds=180`, `take_profit_hysteresis_bars=2`.

**Artifacts:**
- `20260608T055415Z` log: `/Users/tcscoder/Workspaces/BiGan/data/logs/xgboost-v7-paper-shadow-20260608T055415Z-event5s-30round/phase4-20260608T055415Z.jsonl`
- `20260608T133724Z` log: `/Users/tcscoder/Workspaces/BiGan/data/logs/xgboost-v7-paper-shadow-20260608T133721Z-event5s-30round/phase4-20260608T133724Z.jsonl`
- `20260609T103055Z` log: `/Users/tcscoder/Workspaces/BiGan/data/logs/xgboost-v7-paper-shadow-20260609T103055Z-event5s-30round/phase4-20260609T103055Z.jsonl`
- `20260610T021333Z` log: `/Users/tcscoder/Workspaces/BiGan/data/logs/xgboost-v7-paper-shadow-20260610T021333Z-event5s-30round-takeprofit/phase4-20260610T022208Z.jsonl`