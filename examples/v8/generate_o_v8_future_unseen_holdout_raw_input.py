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
    O_REQUIRED_DECISION_ACTION_FAMILIES,
    _apply_o_shadow_ranking_correction,
    _attach_decision_time_feature_fields,
    _deployable_model_features,
    _dot,
    _normalize_action_row,
    _read_json,
    _side_from_action,
    _v8_action_rank_handoff_action_entry,
    _v8_future_holdout_prior_reference_summary,
)

FORBIDDEN_HOLDOUT_ROW_FIELDS = {
    "oracle_executable_best_action",
    "realized_replay_return_report_only",
    "regret_report_only",
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

    provider = PolymarketPublicHTTPRealCorpusProvider(
        max_markets=max_markets,
        timeout_seconds=public_provider_timeout_seconds,
        http_timeout_seconds=public_provider_http_timeout_seconds,
        orderbook_snapshot_interval_seconds=orderbook_snapshot_interval_seconds,
        clob_ws_url=clob_ws_url,
        seed_rest_orderbooks_before_stream=True,
    )
    recorder_config = PolymarketRealCorpusRecorderConfig(
        run_id="o-v8-future-unseen-holdout-raw-input-generation",
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
    future_feature_rows.sort(key=lambda row: (int(row["decision_ts"]), row["market_id"]))

    model_training_summary = dict(
        source_ranking_objective_report["o_model_training_summary"]
    )
    model_identity = {
        "model_sha256": source_action_rank_report.get("model_sha256"),
        "feature_schema_hash": source_action_rank_report.get("feature_schema_hash"),
        "split_hash": source_action_rank_report.get("split_hash"),
    }
    holdout_decision_rows = _score_future_feature_rows(
        feature_rows=future_feature_rows,
        feature_source_path=output_path,
        model_training_summary=model_training_summary,
        model_identity=model_identity,
    )

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
        "public_collection_summary": {
            "market_row_count": len(market_rows),
            "orderbook_row_count": len(orderbook_rows),
            "trade_row_count": len(trade_rows),
            "btc_candle_row_count": len(btc_candle_rows),
            "feature_row_count": len(feature_rows),
            "future_disjoint_feature_row_count": len(future_feature_rows),
        },
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
                ),
                "decision_group_id": decision_group_id,
                "full_5_action_ranking": [
                    _outcome_free_action_entry(
                        row,
                        rank=rank,
                        high_score_threshold=high_score_threshold,
                        model_identity=model_identity,
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
) -> dict[str, Any]:
    entry = _v8_action_rank_handoff_action_entry(
        row,
        rank=rank,
        high_score_threshold=high_score_threshold,
    )
    entry.update(model_identity)
    for field in FORBIDDEN_HOLDOUT_ROW_FIELDS:
        entry.pop(field, None)
    return entry


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
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
