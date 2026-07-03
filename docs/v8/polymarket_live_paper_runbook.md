# v8 Polymarket Live Paper Runbook

This runbook covers the paper-only BTC UP/DOWN live operator for #134.

## Safety Boundary

Allowed:

- public Polymarket metadata, order books, and trades
- public Binance BTCUSDT reference ticks and candles
- trained-model scoring
- EV paper decisions
- paper ledger and settlement evidence
- GitHub evidence comment generation

Forbidden:

- real orders
- wallet signing
- private keys
- CLOB write APIs
- real capital
- automatic live deployment

Every JSON/JSONL artifact written by the operator carries paper-only safety
fields, including `paper_only=true`, `capital_at_risk=false`,
`broker_exchange_write_enabled=false`, `live_exchange_write_enabled=false`,
`polymarket_write_enabled=false`, and `wallet_signing_enabled=false`.

## Mocked-Live Smoke

```bash
PYTHONPATH=src python examples/v8/run_polymarket_live_paper.py \
  --run-id polymarket_live_smoke_001 \
  --output-dir /tmp/polymarket-live-smoke \
  --mode dry-run \
  --mock-live \
  --duration-seconds 300 \
  --overwrite-existing
```

The mocked-live path is deterministic and is the only path used by CI. It emits
BTC 5m, 15m, and 1h markets, read-only UP/DOWN books, Binance reference ticks,
model predictions, EV decisions, paper ledger events, settlement events, PnL,
observability, and GitHub comment payload artifacts.

## Operator Artifacts

- `live_market_metadata.jsonl`
- `live_token_orderbooks.jsonl`
- `live_token_trades.jsonl`
- `live_btc_reference_ticks.jsonl`
- `live_btc_reference_candles.jsonl`
- `polymarket_model_predictions.jsonl`
- `polymarket_ev_decisions.jsonl`
- `polymarket_position_ledger.jsonl`
- `polymarket_settlement_events.jsonl`
- `polymarket_pnl_breakdown.json`
- `polymarket_live_operator_manifest.json`
- `paper_observability_report.json`
- `paper_operator_summary.md`
- `github_paper_comment_payload.json`
- `github_paper_comment.md`

## Fail-Closed Conditions

The operator records `operator_recommendation=blocked_fail_closed` and keeps
`capital_deployment_allowed=false` and `live_deployment_allowed=false` when it
detects:

- missing market settlement rule
- missing UP/DOWN token book
- stale order book
- stale BTC reference price
- missing reference candle
- model manifest mismatch
- delayed settlement with unresolved positions
- write-capable or wallet-enabled feed payloads

## STOP

Set `--stop-requested` to exercise the STOP path. The operator writes artifacts
with `operator_status=operator_stopped` and
`operator_recommendation=stop_paper_run`; open positions remain unresolved unless
an explicit paper settlement path has already run.

## Manual Evidence

Issue #134 still requires a manual live paper evidence comment before closure.
The mocked-live CI path proves the operator contract and fail-closed behavior,
but it intentionally records:

- `live_polymarket_data=false`
- `live_binance_reference_data=false`
- `deterministic_replay=true`
