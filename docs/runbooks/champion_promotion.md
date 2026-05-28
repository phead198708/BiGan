# Champion Model Promotion Process

Local repo copy of the user-provided attachment at `/Users/tcscoder/Downloads/champion-promotion.md`.
Use this process for `xgboost-v4` promotion evidence. Earlier first-champion bootstrap or cutover evidence is not sufficient for the active seven-day multi-market objective unless every gate below has fresh passing evidence.

## Overview

A new model goes through five sequential gates before becoming champion. Skipping a gate is not allowed. A failure at any stage sends the model back for fixes.

```text
Train -> Offline Eval -> Backtest -> Shadow -> Bootstrap Decision -> Champion
```

## Stage 1: Offline Evaluation

Train the challenger model and validate it on a time-split holdout set, never a random split.

Time split strategy:

- Train: past 5 days.
- Validation: next 1 day.
- Test: last 1 day.

Must pass:

| Metric | Threshold |
|---|---|
| AUC | Greater than current champion |
| Brier score | Lower than current champion |
| Platt ECE | Less than 0.05 |
| bucket calibration | high-up realized-up rate greater than 0.55 and high-down realized-down rate greater than 0.55 |
| family realized return | each required family has positive average realized return on the validation/test evidence bundle |
| global ECE regression | lower than the current production baseline ECE, currently `0.4784` for the observed v4 online baseline |
| execution subset ECE | less than `0.08` on rows matching the live execution region |

Rules:

- If the challenger does not beat the champion on both AUC and Brier, do not proceed.
- For v5 and later, prefer family-aware calibration over a single global calibrator. Fit separate calibrators by `(underlying_id, horizon_minutes)`, compare Platt, isotonic, temperature, and beta calibration, then select the lowest validation ECE per family.
- Mark the Phase 4/live execution subset explicitly. The execution subset is the evidence closest to real trading behavior, so calibration reports must include `execution_subset_metrics` with raw and calibrated Brier/ECE. Use `sample_weights` to up-weight executed rows, and apply extra loss weighting only when the realized PnL is from an account-cash-flow reconciled ledger.
- Search execution weighting before final v5 promotion. Start with `EXECUTION_WEIGHT` in `[1.0, 2.0, 3.0, 5.0]`, preserve a held-out validation split, and choose the candidate with the lowest execution-subset ECE rather than only the lowest global ECE.
- Apply probability clipping only as a documented post-calibration guard; serving hotfix clipping does not replace proper calibration evidence. For v5, search family-specific clipping bounds from `[0.03, 0.05, 0.08, 0.10] x [0.90, 0.92, 0.95, 0.97]` and choose by execution-subset Brier score.
- Document results in `rerun_report.md`.

## Stage 2: Cost-Adjusted Backtest

Run a backtest on the holdout period using realistic costs.

Must pass:

| Metric | Threshold |
|---|---|
| cost-adjusted `net_pnl` | Greater than champion baseline over the same period |
| `max_drawdown` | Less than 20% |
| Sharpe ratio | Greater than champion, or justified by a large enough Brier gap |
| turnover | Within a reasonable range |
| holdout setup | Matching dataset, dataset version, warehouse, outcome side, threshold grid, and hold duration |
| `fee_bps` / `slippage_bps` / `latency_ms` | Matching candidate/champion execution assumptions, with non-zero fee and slippage |

Rules:

- A lower Sharpe than the champion is allowed only if the Brier gap is large enough to justify it, as shown by `lower_sharpe_allowed_brier_gap` in bootstrap decision output.
- Missing any metric above is an automatic fail.
- Do not compare a candidate backtest against a champion baseline from a different holdout window, warehouse, outcome side, threshold grid, or hold duration.
- Do not compare a candidate backtest against a cheaper or differently delayed champion baseline.

## Stage 3: Shadow Evaluation

Deploy the challenger in shadow mode alongside the live champion. It generates predictions but does not execute real trades.

Must pass all items:

| Item | Pass Condition |
|---|---|
| `prediction_distribution_stability` | Probability mean drift vs offline validation less than 0.05; std change less than 20% |
| `edge_trigger_rate` | Edge >= 0.30 trigger rate is reasonable, not zero and not anomalously high |
| `simulated_pnl` | Greater than champion baseline over the same shadow period, with champion/challenger net PnL and trade counts recorded |
| `prediction_latency` | p95 less than 50 ms |
| `schema_error_rate` | Exactly 0.0 |

Rules:

- Run shadow for at least one full trading session before evaluating. For this 24/7 crypto-market workflow, the fail-closed audit and runner treat that as at least `86400` seconds of contiguous shadow evidence.
- If any item fails, fix the root cause and restart shadow from scratch. Do not patch mid-shadow.

## Stage 4: Bootstrap Decision

Feed offline evaluation, backtest, and shadow results into the bootstrap decision system.

Decision outputs:

| Decision | Meaning |
|---|---|
| `PROMOTE_CHAMPION` | All hard gates passed; proceed to cutover |
| `KEEP_BASELINE_TEMPORARILY` | One or more hard gates failed; fix and retry |

Full promotion checklist:

- [ ] Beats or justifies replacing current champion.
- [ ] Calibration acceptable.
- [ ] Backtest acceptable.
- [ ] Serving readiness acceptable.
- [ ] Rollback / fallback available.
- [ ] Schema stable.
- [ ] Simple enough for production.

If the decision is `KEEP_BASELINE_TEMPORARILY`, the output lists exactly which items failed. Fix only those, then rerun from the earliest failed stage.

For v5 and later, configure the bootstrap calibration rules with:

- `max_global_ece=0.4784` until a newer production baseline supersedes the observed v4 ECE.
- `max_execution_subset_ece=0.08`.
- `min_high_up_realized_up_rate=0.55`.
- `min_high_down_realized_down_rate=0.55`.
- `require_positive_avg_return_by_family=True`.

## Stage 5: Champion Cutover

Once the bootstrap decision is `PROMOTE_CHAMPION`, execute cutover in this order. Do not skip steps.

Pre-cutover checks:

```bash
# 1. Confirm shadow evaluation report exists and all items passed.
cat mlops/shadow_evaluation_report.md

# 2. Confirm bootstrap decision = PROMOTE.
grep "Recommended Action" bootstrap_decision.md

# 3. Confirm serving readiness JSON: ready = true.
cat mlops/serving_readiness.json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); assert d.get('ready') == True; print('OK')"

# 4. Confirm fallback model is available.
python3 -c \
  "from bigan.mlops.registry import get_model; m = get_model('xgboost-v3'); print(f'fallback: {m is not None}')"
```

Cutover steps:

1. Write the new model to the model registry with model id, promoter, reason, and promotion metrics: AUC, Brier, net PnL, delta vs baseline, max drawdown, Sharpe, edge trigger rate, shadow p95 latency, and schema error rate.
2. Update the active champion with the previous champion recorded as fallback.
3. Run an inference smoke test and require the new model id, `error_rate == 0.0`, and p95 latency less than 50 ms.
4. Register a drift monitoring baseline with offline validation probability mean/std, edge trigger rate, and alert thresholds.

Post-cutover verification:

```bash
# Confirm live predictions are coming from the new champion.
python3 -c \
  "from bigan.mlops.deployments import get_active_champion; c = get_active_champion(); print(c['model_id'])"

# Confirm drift monitor has the new baseline loaded.
python3 -c \
  "from bigan.monitoring.drift import get_baseline; print(get_baseline('xgboost-v4'))"
```

Required #54 live-monitoring checks:

- `prob_up_15m` drift is evaluated on rolling 1h and 6h live prediction windows against the offline validation baseline.
- Create a `prediction_drift` incident if probability mean shift exceeds 0.05.
- Create a `prediction_drift` incident if probability std changes by more than 20%.
- Create a `prediction_drift` incident if `edge >= 0.30` trigger rate is 0 for more than 2 hours.
- After round settlement, update label hit-rate monitoring from settled outcomes and create a `label_shift` incident if positive `label_profit_up_15m` rate stays below 0.50 for 50 consecutive samples.
- Alert output must be written to the monitoring incident catalog before promotion closure evidence is accepted.

## GitHub Closure

Only close promotion-related GitHub issues after Stage 5 evidence exists and the user confirms issue closure. The original attachment specifically closes shadow and cutover issues after successful promotion; for the active objective, do not close #54/#55/#56/#57/#58/#64/#65 while seven-day data, retraining, or promotion evidence remains incomplete.

The fail-closed Stage 5 audit requires cutover JSON to embed closure evidence for the attachment's
post-cutover issues:

```json
[
  {
    "issue": 52,
    "repo": "phead198708/BiGan",
    "state": "closed",
    "comment": "Shadow PASS. Bootstrap decision: PROMOTE_CHAMPION."
  },
  {
    "issue": 53,
    "repo": "phead198708/BiGan",
    "state": "closed",
    "comment": "Cutover complete. New champion: xgboost-v4."
  }
]
```

Pass this file to `champion-cutover-report-v1` with `--github-issue-closures-path` so the final
`champion-promotion-audit` can verify the closure step instead of relying on a manual note.

## Rollback Triggers

Immediately execute [model_rollback.md](model_rollback.md) if any of these occur after cutover:

| Condition | Threshold |
|---|---|
| p95 prediction latency | Greater than 50 ms for 5 or more consecutive minutes |
| `schema_error_rate` | Greater than 0 |
| `edge_trigger_rate` | 0 for more than 2 hours |
| simulated PnL | Drops below fallback baseline |

## Phase 4 Live Execution Gate

Bounded BTC-15M champion-signal runs are diagnostic only until account-cash-flow PnL reconciliation passes. See [phase4_execution_validation.md](phase4_execution_validation.md).

- Do not use Phase 4 `realized_pnl_usdc` for promotion, capital sizing, or execution-subset calibration weighting.
- Require matched Polymarket account-history reconciliation with both theoretical and account PnL reported.
- Keep runtime summaries in GitHub issue #84; track engineering follow-ups in issue #85.

## Active XGBoost-v4 Note

For the current `xgboost-v4` objective, Stage 1 is blocked until `live-collection-readiness` reports seven-day feature and label spans for all required families, fresh raw/processed progress, no recent invalid gzip files, no unrecovered fatal log errors, and a passing raw quarantine clean-window check. Do not rely on earlier BTC-only, one-day, smoke, first-champion, or preliminary cutover artifacts as final promotion proof. The same-dataset family metrics must also show at least one newly added ETH market family with usable test signal (`roc_auc > 0.50` and finite Brier), not just non-empty rows. Once readiness passes, follow [xgboost_v4_post_readiness.md](xgboost_v4_post_readiness.md) to regenerate same-dataset offline evaluation, direct model backtests, shadow, bootstrap, and cutover evidence.

If `raw_segment_quarantine.quarantined_count > 0`, promotion is fail-closed until
`collection_readiness.quarantine_clean_window.meets_target` is true. The clean-window ETA is seven
days after the latest quarantined raw segment, and `collection_readiness.estimated_ready_at` must
include that ETA when it is later than the feature/label span ETA. This prevents a preserved corrupt
segment from being silently ignored while still allowing a clean seven-day corpus to form after the
incident. The status artifact records `raw_segment_quarantine.latest_quarantined_segment.gzip_probe`
with `gzip_valid`, decompression error text, and readable prefix byte/line counts; a readable prefix
is diagnostic evidence only and must not be treated as a clean promotion segment.
