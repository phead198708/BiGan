"""Generate #159 v8 future unseen holdout raw input from public read-only rows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.contracts import canonical_json_sha256  # noqa: E402
from bigan.v8.polymarket.corpus.builder import (  # noqa: E402
    _normalize_book_snapshots,
    _normalize_candles,
    _normalize_markets,
    _normalize_trades,
)
from bigan.v8.polymarket.corpus.contracts import (  # noqa: E402
    PolymarketCorpusBuildConfig,
    safety_fields,
)
from bigan.v8.polymarket.corpus.features import (  # noqa: E402
    build_polymarket_corpus_feature_rows,
)
from bigan.v8.polymarket.recorder.contracts import (  # noqa: E402
    PolymarketRealCorpusRecorderConfig,
)
from bigan.v8.polymarket.recorder.public_provider import (  # noqa: E402
    DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    PolymarketPublicHTTPRealCorpusProvider,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (  # noqa: E402
    O_MODEL_PREDICTED_VARIANT,
    O_REPORT_ONLY_EVALUATION_FIELDS,
    O_REQUIRED_DECISION_ACTION_FAMILIES,
    O_V8_EXECUTION_MAX_BOOK_STALENESS_MS,
    O_V8_EXECUTION_MIN_HTS_TIME_TO_CLOSE_SECONDS,
    _apply_o_shadow_ranking_correction,
    _attach_decision_time_feature_fields,
    _deployable_model_features,
    _dot,
    _normalize_action_row,
    _read_json,
    _side_from_action,
    _v8_action_rank_handoff_action_entry,
    _v8_compact_runtime_state,
    _v8_execution_guard_config,
    _v8_future_holdout_prior_reference_summary,
    _v8_initial_runtime_state,
)

FORBIDDEN_HOLDOUT_ROW_FIELDS = {
    *O_REPORT_ONLY_EVALUATION_FIELDS,
    "oracle_executable_best_action",
    "oracle_action",
    "realized_replay_return_report_only",
    "regret_report_only",
    "realized_pnl",
    "realized_trade_pnl",
    "settlement_pnl",
    "total_polymarket_pnl",
    "future_return",
    "settlement_label",
    "settlement_outcome",
}


def run_generate_o_v8_future_unseen_holdout_raw_input(
    *,
    m2_candidate_report_path: Path | str,
    collection_plan_path: Path | str,
    source_ranking_objective_report_path: Path | str,
    source_action_rank_report_path: Path | str,
    source_simulated_replay_report_path: Path | str,
    output_path: Path | str,
    market_family: str = "btc_updown_5m",
    max_markets: int = 1,
    public_provider_timeout_seconds: float = 25.0,
    public_provider_http_timeout_seconds: float = 10.0,
    orderbook_snapshot_interval_seconds: float = 1.0,
    clob_ws_url: str = DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    min_selected_time_to_close_seconds: float = (
        O_V8_EXECUTION_MIN_HTS_TIME_TO_CLOSE_SECONDS
    ),
    max_selected_book_staleness_ms: float = O_V8_EXECUTION_MAX_BOOK_STALENESS_MS,
    max_collection_attempts: int = 12,
    collection_retry_sleep_seconds: float = 10.0,
) -> dict[str, Any]:
    """Collect public decision-time rows and write an outcome-free #159 manifest."""

    m2_report_path = Path(m2_candidate_report_path).expanduser().resolve()
    collection_plan_path = Path(collection_plan_path).expanduser().resolve()
    source_ranking_objective_report_path = (
        Path(source_ranking_objective_report_path).expanduser().resolve()
    )
    source_action_rank_report_path = (
        Path(source_action_rank_report_path).expanduser().resolve()
    )
    source_simulated_replay_report_path = (
        Path(source_simulated_replay_report_path).expanduser().resolve()
    )
    output_path = Path(output_path).expanduser().resolve()

    m2_report = _read_json(m2_report_path)
    collection_plan = _read_json(collection_plan_path)
    source_ranking_objective_report = _read_json(source_ranking_objective_report_path)
    source_action_rank_report = _read_json(source_action_rank_report_path)
    source_simulated_replay_report = _read_json(source_simulated_replay_report_path)
    prior = _v8_future_holdout_prior_reference_summary(
        m2_report=m2_report,
        action_rank_handoff_report=source_action_rank_report,
        simulated_order_replay_report=source_simulated_replay_report,
    )
    max_prior_decision_ts = int(prior["max_prior_decision_ts"])
    collection_plan_created_ts = int(collection_plan["collection_plan_created_ts"])

    model_training_summary = dict(
        source_ranking_objective_report["o_model_training_summary"]
    )
    model_identity = {
        "model_sha256": source_action_rank_report.get("model_sha256"),
        "feature_schema_hash": source_action_rank_report.get("feature_schema_hash"),
        "split_hash": source_action_rank_report.get("split_hash"),
    }
    guard_config = _v8_execution_guard_config()
    initial_runtime_state = _v8_initial_runtime_state(guard_config)
    attempts: list[dict[str, Any]] = []
    public_collection_summary: dict[str, Any] = {}
    raw_manifest_created_ts = int(time.time() * 1000)
    future_feature_rows: list[dict[str, Any]] = []
    holdout_decision_rows: list[dict[str, Any]] = []
    attempt_count = max(1, int(max_collection_attempts))
    for attempt_index in range(1, attempt_count + 1):
        provider = PolymarketPublicHTTPRealCorpusProvider(
            max_markets=max_markets,
            timeout_seconds=public_provider_timeout_seconds,
            http_timeout_seconds=public_provider_http_timeout_seconds,
            orderbook_snapshot_interval_seconds=orderbook_snapshot_interval_seconds,
            clob_ws_url=clob_ws_url,
            seed_rest_orderbooks_before_stream=True,
        )
        recorder_config = PolymarketRealCorpusRecorderConfig(
            run_id=(
                "o-v8-future-unseen-holdout-raw-input-generation-"
                f"attempt-{attempt_index:02d}"
            ),
            output_dir=output_path.parent,
            market_families=(market_family,),
            build_phase2_corpus=False,
            mock_public_data=False,
        )
        market_rows = provider.market_rows(recorder_config)
        orderbook_rows = provider.orderbook_rows(market_rows, recorder_config)
        trade_rows = provider.trade_rows(market_rows, recorder_config)
        btc_candle_rows = provider.btc_feature_candle_rows(market_rows, recorder_config)

        corpus_config = PolymarketCorpusBuildConfig(
            input_dir=output_path.parent,
            output_dir=output_path.parent / "_future_holdout_feature_build",
            market_families=(market_family,),  # type: ignore[arg-type]
            sample_interval_seconds={market_family: 1},
            min_time_to_close_seconds=0,
            overwrite_existing=True,
        )
        markets = _normalize_markets(market_rows, corpus_config)
        book_snapshots = _normalize_book_snapshots(orderbook_rows, markets)
        trades = _normalize_trades(trade_rows, markets)
        candles = _normalize_candles(btc_candle_rows)
        feature_rows = build_polymarket_corpus_feature_rows(
            markets=markets,
            book_snapshots=book_snapshots,
            trades=trades,
            btc_candles=candles,
            config=corpus_config,
        )
        raw_manifest_created_ts = int(time.time() * 1000)
        future_feature_rows = [
            row.to_dict()
            for row in feature_rows
            if int(row.decision_ts) > max_prior_decision_ts
            and int(row.decision_ts) > collection_plan_created_ts
            and int(row.decision_ts) <= raw_manifest_created_ts
            and int(row.available_at_ts) <= raw_manifest_created_ts
            and row.market_id not in set(prior["prior_market_ids"])
        ]
        future_feature_rows.sort(
            key=lambda row: (int(row["decision_ts"]), row["market_id"])
        )
        scored_rows = _score_future_feature_rows(
            feature_rows=future_feature_rows,
            feature_source_path=output_path,
            model_training_summary=model_training_summary,
            model_identity=model_identity,
            guard_config=guard_config,
            initial_runtime_state=initial_runtime_state,
        )
        quality_rows, rejected_rows = _filter_runtime_quality_rows(
            scored_rows,
            min_selected_time_to_close_seconds=min_selected_time_to_close_seconds,
            max_selected_book_staleness_ms=max_selected_book_staleness_ms,
        )
        attempt_summary = _collection_attempt_summary(
            attempt_index=attempt_index,
            market_rows=market_rows,
            orderbook_rows=orderbook_rows,
            trade_rows=trade_rows,
            btc_candle_rows=btc_candle_rows,
            feature_rows=feature_rows,
            future_feature_rows=future_feature_rows,
            scored_rows=scored_rows,
            quality_rows=quality_rows,
            rejected_rows=rejected_rows,
        )
        attempts.append(attempt_summary)
        public_collection_summary = {
            "market_row_count": len(market_rows),
            "orderbook_row_count": len(orderbook_rows),
            "trade_row_count": len(trade_rows),
            "btc_candle_row_count": len(btc_candle_rows),
            "feature_row_count": len(feature_rows),
            "future_disjoint_feature_row_count": len(future_feature_rows),
            "candidate_handoff_row_count": len(scored_rows),
            "runtime_quality_selected_row_count": len(quality_rows),
            "runtime_quality_rejected_row_count": len(rejected_rows),
        }
        if quality_rows:
            holdout_decision_rows = quality_rows
            break
        holdout_decision_rows = rejected_rows if rejected_rows else scored_rows
        if attempt_index < attempt_count:
            time.sleep(max(0.0, float(collection_retry_sleep_seconds)))

    payload = {
        "schema_version": "bigan-v8-o-future-unseen-holdout-raw-input-v1",
        "report_type": "o_v8_future_unseen_holdout_raw_input",
        "generation_source": "read_only_public_provider_decision_time_features",
        "m2_candidate_report_path": str(m2_report_path),
        "collection_plan_path": str(collection_plan_path),
        "source_ranking_objective_report_path": str(
            source_ranking_objective_report_path
        ),
        "source_action_rank_report_path": str(source_action_rank_report_path),
        "source_simulated_replay_report_path": str(source_simulated_replay_report_path),
        "market_family": market_family,
        "window_start_ts": min(
            (int(row["decision_ts"]) for row in holdout_decision_rows),
            default=None,
        ),
        "window_end_ts": max(
            (int(row["decision_ts"]) for row in holdout_decision_rows),
            default=None,
        ),
        "raw_manifest_created_ts": raw_manifest_created_ts,
        "input_freeze_created_ts": collection_plan_created_ts,
        "max_prior_decision_ts": max_prior_decision_ts,
        "collection_plan_created_ts": collection_plan_created_ts,
        "prior_reference_hash": prior["prior_reference_hash"],
        "public_collection_summary": public_collection_summary,
        "runtime_input_quality_policy": {
            "collect_earlier_in_market_window": True,
            "min_selected_time_to_close_seconds": min_selected_time_to_close_seconds,
            "max_selected_book_staleness_ms": max_selected_book_staleness_ms,
            "max_collection_attempts": attempt_count,
            "collection_retry_sleep_seconds": collection_retry_sleep_seconds,
            "simulated_exposure_state_added": True,
            "time_to_close_written_to_microstructure_snapshot": True,
            "forbidden_field_list_aligned_with_evaluator": True,
            "outcome_fields_stripped_recursively": True,
        },
        "runtime_input_quality_collection_attempts": attempts,
        "runtime_field_input_quality_rules_applied": bool(holdout_decision_rows),
        "runtime_field_input_quality_rule_counts": _runtime_input_quality_rule_counts(
            holdout_decision_rows
        ),
        "holdout_decision_rows": holdout_decision_rows,
        "future_outcome_evaluation_generated": False,
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **safety_fields(),
    }
    payload["o_v8_future_unseen_holdout_raw_input_id"] = canonical_json_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _score_future_feature_rows(
    *,
    feature_rows: list[dict[str, Any]],
    feature_source_path: Path,
    model_training_summary: dict[str, Any],
    model_identity: dict[str, Any],
    guard_config: dict[str, Any],
    initial_runtime_state: dict[str, Any],
) -> list[dict[str, Any]]:
    action_rows = []
    for feature_row in feature_rows:
        p_up = _market_implied_p_up(feature_row)
        for action in O_REQUIRED_DECISION_ACTION_FAMILIES:
            base = {
                **model_identity,
                "source_report_path": str(feature_source_path),
                "market_id": feature_row["market_id"],
                "decision_ts": int(feature_row["decision_ts"]),
                "slug": feature_row.get("slug"),
                "split": "future_unseen_holdout",
                "action": action,
                "selected_side": _side_from_action(action),
                "p_up": p_up,
                "raw_calibrated_action_score": 0.0,
                "best_action_margin": 0.0,
                "label_candidate_available": False,
                "source_score_available": False,
            }
            enriched = _attach_decision_time_feature_fields(base, feature_row)
            action_rows.append(_normalize_action_row(enriched))

    feature_names = tuple(model_training_summary["feature_names"])
    coefficients = [
        float(model_training_summary["coefficients_by_feature"][name])
        for name in feature_names
    ]
    raw_scored_rows = [
        {
            **row,
            "o_raw_ridge_model_score": _dot(
                coefficients,
                _deployable_model_features(row, feature_names),
            ),
        }
        for row in action_rows
    ]
    scored_rows = _apply_o_shadow_ranking_correction(
        rows=raw_scored_rows,
        deployable_available=True,
        ranking_correction=model_training_summary["ranking_correction_config"],
    )
    scored_rows = [
        {
            **row,
            "variant_scores": {
                O_MODEL_PREDICTED_VARIANT: float(
                    row.get("o_model_predicted_score") or 0.0
                ),
            },
        }
        for row in scored_rows
    ]
    high_score_threshold = float(
        model_training_summary["ranking_correction_config"]["high_score_calibration"][
            "high_score_threshold"
        ]
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[str(row["decision_group_id"])].append(row)

    selected_rows = []
    for decision_group_id, group_rows in sorted(grouped.items()):
        ranked = sorted(
            group_rows,
            key=lambda row: float(row.get("o_model_predicted_score") or 0.0),
            reverse=True,
        )
        selected = ranked[0]
        selected_rows.append(
            {
                **_outcome_free_action_entry(
                    selected,
                    rank=1,
                    high_score_threshold=high_score_threshold,
                    model_identity=model_identity,
                    guard_config=guard_config,
                    initial_runtime_state=initial_runtime_state,
                ),
                "decision_group_id": decision_group_id,
                "full_5_action_ranking": [
                    _outcome_free_action_entry(
                        row,
                        rank=rank,
                        high_score_threshold=high_score_threshold,
                        model_identity=model_identity,
                        guard_config=guard_config,
                        initial_runtime_state=initial_runtime_state,
                    )
                    for rank, row in enumerate(ranked, start=1)
                ],
            }
        )
    return selected_rows


def _outcome_free_action_entry(
    row: dict[str, Any],
    *,
    rank: int,
    high_score_threshold: float,
    model_identity: dict[str, Any],
    guard_config: dict[str, Any],
    initial_runtime_state: dict[str, Any],
) -> dict[str, Any]:
    entry = _v8_action_rank_handoff_action_entry(
        row,
        rank=rank,
        high_score_threshold=high_score_threshold,
    )
    entry.update(model_identity)
    return _prepare_runtime_quality_action_entry(
        entry,
        guard_config=guard_config,
        initial_runtime_state=initial_runtime_state,
    )


def _prepare_runtime_quality_action_entry(
    entry: dict[str, Any],
    *,
    guard_config: dict[str, Any],
    initial_runtime_state: dict[str, Any],
) -> dict[str, Any]:
    prepared = _strip_forbidden_holdout_fields(dict(entry))
    microstructure = dict(prepared.get("microstructure_snapshot") or {})
    backfill_source = dict(
        (prepared.get("runtime_field_backfill_sources") or {}).get(
            "microstructure_snapshot.time_to_close_seconds"
        )
        or {}
    )
    rules_applied: list[dict[str, Any]] = []
    time_to_close = _safe_optional_float(microstructure.get("time_to_close_seconds"))
    backfill_value = _safe_optional_float(backfill_source.get("value"))
    backfill_valid = bool(backfill_source.get("provenance_valid") is True)
    if time_to_close is None and backfill_valid and backfill_value is not None:
        microstructure["time_to_close_seconds"] = backfill_value
        rules_applied.append(
            {
                "deterministic_rule_id": (
                    backfill_source.get("deterministic_rule_id")
                    or "write_time_to_close_from_decision_time_backfill_source"
                ),
                "field": "microstructure_snapshot.time_to_close_seconds",
                "source_field_name": backfill_source.get("source_field_name"),
                "source_timestamp": backfill_source.get("source_timestamp"),
                "max_input_ts": backfill_source.get("max_input_ts"),
                "decision_ts": backfill_source.get("decision_ts"),
                "provenance_valid": True,
                "applied_value": backfill_value,
            }
        )
    prepared["microstructure_snapshot"] = microstructure
    prepared["runtime_exposure_state"] = _v8_compact_runtime_state(
        initial_runtime_state
    )
    prepared["configured_execution_limits"] = {
        "max_order_size": guard_config["max_order_size"],
        "max_total_exposure": guard_config["max_total_exposure"],
        "max_market_exposure": guard_config["max_market_exposure"],
        "max_side_exposure": guard_config["max_side_exposure"],
        "max_spread_bps": guard_config["max_spread_bps"],
        "max_book_staleness_ms": guard_config["max_book_staleness_ms"],
        "min_queue_fill": guard_config["min_queue_fill"],
        "min_time_to_close_seconds": guard_config["min_time_to_close_seconds"],
        "min_hts_time_to_close_seconds": guard_config[
            "min_hts_time_to_close_seconds"
        ],
    }
    rules_applied.append(
        {
            "deterministic_rule_id": "attach_initial_simulated_runtime_exposure_state",
            "field": "runtime_exposure_state",
            "source_field_name": "deterministic_v8_execution_layer_runtime_guard",
            "source_timestamp": prepared.get("decision_ts"),
            "max_input_ts": prepared.get("decision_ts"),
            "decision_ts": prepared.get("decision_ts"),
            "provenance_valid": True,
            "applied_value": "simulated_initial_ledger",
        }
    )
    prepared["runtime_input_quality_rules_applied"] = True
    prepared["runtime_input_quality_applied_rows"] = rules_applied
    return _strip_forbidden_holdout_fields(prepared)


def _strip_forbidden_holdout_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_forbidden_holdout_fields(child)
            for key, child in value.items()
            if key not in FORBIDDEN_HOLDOUT_ROW_FIELDS
        }
    if isinstance(value, list):
        return [_strip_forbidden_holdout_fields(child) for child in value]
    return value


def _filter_runtime_quality_rows(
    rows: list[dict[str, Any]],
    *,
    min_selected_time_to_close_seconds: float,
    max_selected_book_staleness_ms: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = []
    rejected = []
    for row in rows:
        reasons = _runtime_quality_rejection_reasons(
            row,
            min_selected_time_to_close_seconds=min_selected_time_to_close_seconds,
            max_selected_book_staleness_ms=max_selected_book_staleness_ms,
        )
        annotated = {
            **row,
            "runtime_input_quality_passed": not reasons,
            "runtime_input_quality_reason_codes": reasons,
        }
        if reasons:
            rejected.append(annotated)
        else:
            accepted.append(annotated)
    return accepted, rejected


def _runtime_quality_rejection_reasons(
    row: dict[str, Any],
    *,
    min_selected_time_to_close_seconds: float,
    max_selected_book_staleness_ms: float,
) -> list[str]:
    microstructure = dict(row.get("microstructure_snapshot") or {})
    time_to_close = _safe_optional_float(microstructure.get("time_to_close_seconds"))
    staleness = _safe_optional_float(microstructure.get("book_staleness_ms"))
    spread = _safe_optional_float(microstructure.get("spread_bps"))
    queue = _safe_optional_float(microstructure.get("queue_fill_proxy"))
    reference_provenance = dict(row.get("reference_price_feature_provenance") or {})
    reasons = []
    if time_to_close is None:
        reasons.append("runtime_quality_time_to_close_missing")
    elif time_to_close < float(min_selected_time_to_close_seconds):
        reasons.append("runtime_quality_time_to_close_below_execution_threshold")
    if staleness is None:
        reasons.append("runtime_quality_book_staleness_missing")
    elif staleness > float(max_selected_book_staleness_ms):
        reasons.append("runtime_quality_book_stale")
    if spread is None:
        reasons.append("runtime_quality_spread_missing")
    if queue is None:
        reasons.append("runtime_quality_queue_fill_missing")
    if reference_provenance.get("provenance_valid") is not True:
        reasons.append("runtime_quality_reference_provenance_invalid")
    if not row.get("runtime_exposure_state"):
        reasons.append("runtime_quality_exposure_state_missing")
    return sorted(set(reasons))


def _collection_attempt_summary(
    *,
    attempt_index: int,
    market_rows: list[Any],
    orderbook_rows: list[Any],
    trade_rows: list[Any],
    btc_candle_rows: list[Any],
    feature_rows: list[Any],
    future_feature_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rejected_reason_counts = defaultdict(int)
    for row in rejected_rows:
        for reason in row.get("runtime_input_quality_reason_codes", []):
            rejected_reason_counts[str(reason)] += 1
    selected_time_to_close_values = [
        value
        for value in (
            _safe_optional_float(
                (row.get("microstructure_snapshot") or {}).get(
                    "time_to_close_seconds"
                )
            )
            for row in scored_rows
        )
        if value is not None
    ]
    selected_staleness_values = [
        value
        for value in (
            _safe_optional_float(
                (row.get("microstructure_snapshot") or {}).get("book_staleness_ms")
            )
            for row in scored_rows
        )
        if value is not None
    ]
    return {
        "attempt_index": attempt_index,
        "market_row_count": len(market_rows),
        "orderbook_row_count": len(orderbook_rows),
        "trade_row_count": len(trade_rows),
        "btc_candle_row_count": len(btc_candle_rows),
        "feature_row_count": len(feature_rows),
        "future_disjoint_feature_row_count": len(future_feature_rows),
        "candidate_handoff_row_count": len(scored_rows),
        "runtime_quality_selected_row_count": len(quality_rows),
        "runtime_quality_rejected_row_count": len(rejected_rows),
        "runtime_quality_rejected_reason_counts": dict(
            sorted(rejected_reason_counts.items())
        ),
        "max_selected_time_to_close_seconds": max(
            selected_time_to_close_values,
            default=None,
        ),
        "min_selected_book_staleness_ms": min(selected_staleness_values, default=None),
    }


def _runtime_input_quality_rule_counts(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for applied in row.get("runtime_input_quality_applied_rows", []):
            counts[str(applied.get("deterministic_rule_id") or "UNKNOWN")] += 1
        for candidate in row.get("full_5_action_ranking", []):
            for applied in candidate.get("runtime_input_quality_applied_rows", []):
                counts[str(applied.get("deterministic_rule_id") or "UNKNOWN")] += 1
    return dict(sorted(counts.items()))


def _market_implied_p_up(feature_row: dict[str, Any]) -> float:
    features = dict(feature_row.get("features") or {})
    up_mid = _safe_float(features.get("up_mid"))
    down_mid = _safe_float(features.get("down_mid"))
    total = up_mid + down_mid
    if total > 0.0:
        return max(0.0, min(1.0, up_mid / total))
    return 0.5


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-candidate-report", required=True)
    parser.add_argument("--collection-plan", required=True)
    parser.add_argument("--source-ranking-objective-report", required=True)
    parser.add_argument("--source-action-rank-report", required=True)
    parser.add_argument("--source-simulated-replay-report", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--market-family", default="btc_updown_5m")
    parser.add_argument("--max-markets", type=int, default=1)
    parser.add_argument("--public-provider-timeout-seconds", type=float, default=25.0)
    parser.add_argument(
        "--public-provider-http-timeout-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument("--orderbook-snapshot-interval-seconds", type=float, default=1.0)
    parser.add_argument("--clob-ws-url", default=DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL)
    parser.add_argument(
        "--min-selected-time-to-close-seconds",
        type=float,
        default=O_V8_EXECUTION_MIN_HTS_TIME_TO_CLOSE_SECONDS,
    )
    parser.add_argument(
        "--max-selected-book-staleness-ms",
        type=float,
        default=O_V8_EXECUTION_MAX_BOOK_STALENESS_MS,
    )
    parser.add_argument("--max-collection-attempts", type=int, default=12)
    parser.add_argument("--collection-retry-sleep-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    result = run_generate_o_v8_future_unseen_holdout_raw_input(
        m2_candidate_report_path=args.m2_candidate_report,
        collection_plan_path=args.collection_plan,
        source_ranking_objective_report_path=args.source_ranking_objective_report,
        source_action_rank_report_path=args.source_action_rank_report,
        source_simulated_replay_report_path=args.source_simulated_replay_report,
        output_path=args.output_path,
        market_family=args.market_family,
        max_markets=args.max_markets,
        public_provider_timeout_seconds=args.public_provider_timeout_seconds,
        public_provider_http_timeout_seconds=args.public_provider_http_timeout_seconds,
        orderbook_snapshot_interval_seconds=args.orderbook_snapshot_interval_seconds,
        clob_ws_url=args.clob_ws_url,
        min_selected_time_to_close_seconds=args.min_selected_time_to_close_seconds,
        max_selected_book_staleness_ms=args.max_selected_book_staleness_ms,
        max_collection_attempts=args.max_collection_attempts,
        collection_retry_sleep_seconds=args.collection_retry_sleep_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
