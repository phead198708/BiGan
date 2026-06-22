# v8 Polymarket BTC 15m Adapter

This adapter maps deterministic Polymarket BTC 15m UP/DOWN binary-market
fixtures into v8 paper-only contracts. It is a read-only, paper-only bridge
between Polymarket market structure and the existing v8 Phase 0 / Phase 4 /
paper harness / observability stack.

It does not place orders, sign wallet messages, read private keys, call CLOB
write APIs, allocate capital, or promote live deployment.

## Safety Boundary

Every contract and artifact preserves:

```text
paper_only=true
capital_at_risk=false
broker_exchange_write_enabled=false
live_exchange_write_enabled=false
polymarket_write_enabled=false
wallet_signing_enabled=false
```

Any attempt to enable Polymarket writes or wallet signing fails before artifacts
are accepted.

## Market Mapping

The first supported market family is:

```text
market_family=btc_15m_up_down
base_asset=BTC
quote_asset=USD
horizon_ms=900000
```

The adapter requires exactly two outcomes:

```text
UP token
DOWN token
```

It rejects missing UP/DOWN tokens, non-binary markets, non-BTC markets,
non-15-minute windows, unknown settlement rules, and unsafe metadata.

## Label Semantics

The normalized settlement rule is:

```text
btc_reference_price_end_gt_start_up_else_down
```

The adapter records both raw settlement metadata hash and explicit reference
prices:

```text
reference_price_start
reference_price_end
market_start_ts
market_end_ts
horizon_ms
settlement_rule
raw_settlement_metadata_hash
```

UP wins when the BTC reference price at market end is greater than the reference
price at market start. DOWN wins otherwise. Each label stores token entry price,
binary payout exit price, gross return, execution costs, and net return.

## Feature Semantics

Feature rows are causal. Each row carries v8 `FeatureVector` provenance, and the
adapter enforces:

```text
feature_cutoff_ts <= decision_ts
max_input_ts <= decision_ts
available_at_ts <= decision_ts
```

Minimum Polymarket features include BTC returns and volatility, UP/DOWN token
prices, spread, liquidity depth, liquidity imbalance, market age, and time to
close.

## Paper Decision Mapping

The adapter maps v8 policy signal probability/confidence into paper-only
Polymarket decisions:

```text
positive UP edge -> selected_outcome=UP
positive DOWN edge -> selected_outcome=DOWN
low confidence / negative edge / missing price / closed market -> NO_TRADE
```

No real order fields are emitted. Trade decisions are converted into existing
Phase 4 `AdaptiveDecision` rows, then passed through the existing v8 paper
harness, Phase 5 safety layer, Phase 6 CI/CD evidence, observability, and
GitHub comment payload generation.

The deterministic example runner uses synthetic fixture policy signals:

```text
policy_signal_source=synthetic_fixture
trained_model_used=false
```

This validates adapter plumbing and paper-only evidence generation. It is not a
profitability claim and is not a production model inference path. A future
Polymarket-specific policy-training ticket should replace the fixture signal
source with a trained BTC 15m UP/DOWN model.

## Paper Summary Semantics

For BTC 15m runs, the generated `paper_run_summary.json` uses the market window:

```text
started_at = market_start_ts converted to UTC
ended_at = market_end_ts converted to UTC
duration_seconds = 900
configured_duration_seconds = 900
```

The summary, bundle manifest, observability report, operator summary, and
GitHub comment payload all expose the Polymarket-specific safety flags:

```text
polymarket_write_enabled=false
wallet_signing_enabled=false
```

For Polymarket runs, missing or enabled write/wallet fields are critical
paper-boundary alerts.

## Deterministic Run

```bash
PYTHONPATH=src python examples/v8/run_polymarket_btc15m_paper.py \
  --run-id polymarket_btc15m_smoke_001 \
  --output-dir examples/v8/polymarket_runs \
  --repo phead198708/BiGan \
  --issue-number 130 \
  --mode dry-run \
  --overwrite-existing
```

The run writes:

```text
adapter/polymarket_market_manifest.json
adapter/polymarket_token_snapshots.jsonl
adapter/polymarket_feature_rows.jsonl
adapter/polymarket_label_rows.jsonl
adapter/polymarket_paper_decisions.jsonl
adapter/polymarket_adapter_summary.json
paper_run/
observability/
github_comment/
polymarket_pipeline_summary.json
```

Use `--mode gh-command` to write a `gh issue comment` command. Use
`--mode direct-comment` only when you intentionally want to post to GitHub.

## CI Validation

```bash
PYTHONPATH=src python -m pytest \
  tests/v8/test_polymarket_contracts.py \
  tests/v8/test_polymarket_btc15m_adapter.py \
  tests/v8/test_polymarket_paper_pipeline.py -q
```

The tests use deterministic mocked markets and do not depend on live
Polymarket availability.
