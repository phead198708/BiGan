# v8 Polymarket BTC Policy Training

This document describes the deterministic Phase 3 Polymarket policy training path
for BTC Up/Down markets. It consumes the Phase 2 historical corpus, trains an
offline action-value policy model, keeps `P(UP)` as an auxiliary calibration
head, converts policy outputs into paper-only decisions, and replays those
decisions through Phase 1 ledger and settlement primitives.

## Safety Boundary

The training path is offline and paper-only:

- no real orders
- no wallet signing
- no CLOB write payloads
- no private keys
- no capital at risk
- no automatic promotion to live trading

Artifacts preserve:

- `paper_only=true`
- `capital_at_risk=false`
- `polymarket_write_enabled=false`
- `wallet_signing_enabled=false`

## Inputs

The runner consumes a Phase 2 corpus directory containing:

- `polymarket_corpus_manifest.json`
- `polymarket_market_metadata.jsonl`
- `polymarket_resolution_events.jsonl`
- `polymarket_feature_rows.jsonl`
- `polymarket_label_rows.jsonl`

Training examples are built from causal feature rows and all action-level labels
for each `(market_id, decision_ts)` state:

```text
NO_TRADE
BUY_UP_HOLD_TO_SETTLEMENT
BUY_DOWN_HOLD_TO_SETTLEMENT
BUY_UP_SELL_BEFORE_CLOSE
BUY_DOWN_SELL_BEFORE_CLOSE
```

The primary supervised target is cost-aware action value:

```text
primary_policy_target = action_expected_net_return
action_return_targets[action] = total_net_return
best_policy_action = argmax(action_return_targets)
best_action_margin = best_return - second_best_return
```

The auxiliary outcome head remains:

```text
resolved_outcome=UP             -> target_up_probability=1.0
resolved_outcome=DOWN           -> target_up_probability=0.0
resolved_outcome=UNKNOWN_50_50  -> target_up_probability=0.5
```

This is supervised from Phase 2 net-return labels, not direct live PnL
optimization. The manifest keeps `direct_pnl_optimization=false`.

## Temporal Splits

The dataset is split by unique `decision_ts`, not by row count. All markets that
share the same `decision_ts` remain in the same partition, and the runner
enforces strict timestamp separation:

```text
max(train.decision_ts) < min(validation.decision_ts)
max(validation.decision_ts) < min(shadow.decision_ts)
```

Dataset profiles and model manifests record:

- `train_min_ts` / `train_max_ts`
- `validation_min_ts` / `validation_max_ts`
- `shadow_min_ts` / `shadow_max_ts`
- `strict_temporal_separation=true`

## Model Output

The model writes trained-model predictions with:

- `p_up_auxiliary`
- `estimated_up_probability`
- `expected_return_by_action`
- `best_policy_action`
- `best_action_expected_return`
- `second_best_action_expected_return`
- `best_action_margin`
- `policy_confidence`
- `confidence`
- `score`
- `calibration_bucket`
- `model_version`
- `feature_schema_hash`
- `training_corpus_hash`
- `action_value_model_family`
- `feature_conditioned_action_value_model_enabled`

The implementation uses a deterministic feature-conditioned action-return model:

```text
action_value_model_family=feature_conditioned_action_return_model
fallback_action_value_model_family=market_family_mean_baseline
```

Each action has a family/global mean fallback plus point-in-time feature residual
coefficients. This keeps the model lightweight and auditable while ensuring
different states inside the same market family can produce different expected
returns by action.

## Evaluation

Validation reports include:

- `logloss`
- `brier_score`
- `calibration_error`
- `auc`
- `accuracy_by_threshold`
- `sample_count`
- `market_count`

Metrics are reported by market family:

- `btc_updown_5m`
- `btc_updown_15m`
- `btc_updown_1h`

Metrics are also reported by time-to-close bucket:

- `0-30s`
- `30-60s`
- `1-3m`
- `3-5m`
- `5-15m`
- `15m+`

Primary calibration is split-specific. The default primary calibration split is
`validation`, while `train`, `validation`, and `shadow` calibration sections are
all written for audit. Validation by-family and time-to-close bucket metrics use
only the selected out-of-sample validation split, never train rows.

## Policy Execution

The execution layer consumes action-value outputs first:

```text
expected_return_by_action
best_policy_action
best_action_expected_return
best_action_margin
policy_confidence
```

If no action-value head is available, the compatibility fallback turns `P(UP)`
into paper actions:

```text
EV_BUY_UP   = P(UP)     - ask_up   - costs
EV_BUY_DOWN = (1-P(UP)) - ask_down - costs
```

BUY decisions use executable ask prices. SELL decisions use executable bid prices.
Low confidence predictions become `NO_TRADE`. Existing paper positions become
`HOLD` unless sell EV deterioration triggers a bid-side exit.

If the selected entry label ends with `SELL_BEFORE_CLOSE`, the decision records:

- `entry_policy_action`
- `intended_exit_policy=sell_before_close`
- `planned_exit_before_ts`
- `policy_exit_reason`

The sequential paper decision builder preserves this state and emits a bid-side
`SELL_UP` or `SELL_DOWN` once the configured exit window is reached. Labels that
end with `HOLD_TO_SETTLEMENT` keep `intended_exit_policy=hold_to_settlement`.

Allowed actions are:

- `BUY_UP`
- `BUY_DOWN`
- `SELL_UP`
- `SELL_DOWN`
- `HOLD`
- `NO_TRADE`

Every EV decision records:

- `trained_model_used=true`
- `policy_signal_source=trained_model`
- `synthetic_fixture_signal_used=false`
- `action_value_head_used`
- `probability_ev_fallback_used`
- `entry_policy_action`
- `intended_exit_policy`
- `planned_exit_before_ts`
- `policy_exit_reason`

## Paper Replay

Paper replay uses Phase 1 Polymarket primitives:

- `PolymarketPositionLedger`
- `build_btc_updown_resolution_rule`
- `resolve_polymarket_rule`

Replay reports include:

- `calibration_split`
- `replay_split`
- `out_of_sample_replay`
- `trade_count`
- `no_trade_count`
- `settled_position_count`
- `realized_trade_pnl`
- `settlement_pnl`
- `total_polymarket_pnl`
- `max_drawdown`
- `calibration_error`
- `outcome_calibration_error`
- `action_value_policy_metrics`
- `intended_exit_policy_counts`
- `planned_sell_before_close_exit_count`
- `critical_alert_count`

This replay is deterministic and does not imply expected production profitability.
By default, EV replay is built only from `shadow` predictions. Full predictions
are still written as a debug artifact, alongside split-specific prediction sets.

## Artifacts

The runner writes:

- `polymarket_policy_training_config.json`
- `polymarket_policy_dataset_profile.json`
- `polymarket_policy_model.json`
- `polymarket_policy_model_manifest.json`
- `polymarket_policy_calibration_report.json`
- `polymarket_policy_validation_report.json`
- `polymarket_ev_threshold_report.json`
- `polymarket_policy_replay_report.json`
- `polymarket_policy_predictions.jsonl`
- `polymarket_policy_train_predictions.jsonl`
- `polymarket_policy_validation_predictions.jsonl`
- `polymarket_policy_shadow_predictions.jsonl`
- `polymarket_ev_decisions.jsonl`
- `polymarket_policy_training_summary.md`

The model manifest records:

- schema and model version
- market families
- training corpus hash
- feature schema hash
- label schema hash
- model SHA-256
- train / validation / shadow row counts
- train / validation / shadow timestamp ranges
- primary calibration split
- replay split
- `out_of_sample_replay=true`
- `primary_policy_target=action_expected_net_return`
- `outcome_probability_head_enabled=true`
- `action_value_head_enabled=true`
- `action_value_model_family=feature_conditioned_action_return_model`
- `feature_conditioned_action_value_model_enabled=true`
- action-label coverage by action
- paper-only safety flags

## Local Smoke

```bash
PYTHONPATH=src python examples/v8/train_polymarket_btc_policy.py \
  --output-dir /tmp/bigan-v8-polymarket-policy \
  --overwrite-existing
```

The default smoke path generates deterministic Phase 2 fixture corpus data first,
then trains and replays the policy model.

For real public-data recorder bundles, use
`examples/v8/run_polymarket_real_corpus_retraining_gate.py` instead. That gate
rejects mocked/synthetic recorder bundles before training and augments the model
manifest with real-history provenance fields for #134.
