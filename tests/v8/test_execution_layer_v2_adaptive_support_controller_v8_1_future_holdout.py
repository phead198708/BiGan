from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_adaptive_support_controller_v8_1 as v81,
)
from bigan.v8.polymarket.training import (
    execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_pipeline as v81_future_pipeline,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout import (
    COMPLETE_CANARY_BATCH_LATEST_MARKET_CLOSE_TS,
    EXACT_MARKET_COUNT,
    FROZEN_PLAN_SHA256,
    MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
    SCAN_CAP,
    STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE,
    build_adaptive_support_controller_v8_1_future_pnl_gate,
    build_adaptive_support_controller_v8_1_target_free_freeze_report,
    materialize_adaptive_support_controller_v8_1_runtime_decisions,
    select_adaptive_support_controller_v8_1_future_holdout_window,
    validate_adaptive_support_controller_v8_1_future_holdout_plan,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_pipeline import (
    AdaptiveSupportControllerV81FutureFreezeConfig,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text())


def test_v8_1_future_holdout_plan_is_frozen_before_collection() -> None:
    plan = _plan()
    validate_adaptive_support_controller_v8_1_future_holdout_plan(plan)
    assert hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() == FROZEN_PLAN_SHA256
    assert plan["collection"]["exact_quality_valid_market_count"] == EXACT_MARKET_COUNT
    assert plan["collection"]["maximum_attempted_market_count"] == SCAN_CAP
    assert (
        plan["collection"]["strictly_later_minimum_market_start_ts_exclusive"]
        == STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE
    )
    assert (
        plan["collection"]["complete_target_free_canary_batch_latest_market_close_ts"]
        == COMPLETE_CANARY_BATCH_LATEST_MARKET_CLOSE_TS
    )
    assert (
        plan["target_free_decision_freeze"][
            "minimum_candidate_guard_accepted_unique_market_count"
        ]
        == MINIMUM_GUARD_ACCEPTED_MARKET_COUNT
    )
    assert plan["target_free_decision_freeze"]["side_quota_enabled"] is False


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("collection", "exact_quality_valid_market_count", 119),
        (
            "target_free_decision_freeze",
            "minimum_candidate_guard_accepted_unique_market_count",
            39,
        ),
        (
            "single_use_future_pnl_gate",
            "candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive",
            -0.01,
        ),
        ("safety", "capital_at_risk", True),
    ],
)
def test_v8_1_future_holdout_plan_rejects_gate_or_safety_drift(
    section: str,
    field: str,
    value: object,
) -> None:
    plan = copy.deepcopy(_plan())
    plan[section][field] = value
    with pytest.raises(ValueError, match="future holdout plan drifted"):
        validate_adaptive_support_controller_v8_1_future_holdout_plan(plan)


def test_v8_1_future_holdout_lineage_binds_completed_canary() -> None:
    plan = _plan()
    lineage = plan["lineage"]
    assert (
        lineage["target_free_canary_manifest_sha256"]
        == "225f243a73654c97042032ed77493e83dc44dfdf68f5a14bb5d25b47d2aee6c5"
    )
    assert (
        lineage["target_free_canary_batch_index_sha256"]
        == "d042f1b534956dd8bddfaa3a8269f3a00219d2873f558015a56603f6e8d57ebb"
    )
    assert (
        lineage["target_free_canary_batch_last_entry_sha256"]
        == "09ac313199e170e68900343426dafd4c05a7745053c7079c7fcab4465340fdc3"
    )
    assert plan["single_use_future_pnl_gate"]["equality_passes_noninferiority"]
    assert plan["safety"]["paper_only"] is True
    assert plan["safety"]["capital_at_risk"] is False
    assert plan["safety"]["polymarket_write_enabled"] is False
    assert plan["safety"]["wallet_signing_enabled"] is False
    assert plan["safety"]["#134_resume_allowed"] is False
    assert plan["safety"]["#146_start_allowed"] is False
    assert (
        hashlib.sha256(Path(v81.__file__).read_bytes()).hexdigest()
        == lineage["candidate_decision_policy_source_sha256"]
    )


def test_v8_1_future_freeze_config_requires_aligned_batch_pins(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"input-{index}.json" for index in range(7)]
    config = AdaptiveSupportControllerV81FutureFreezeConfig(
        run_id="future-freeze",
        output_dir=tmp_path,
        plan_path=paths[0],
        expected_plan_sha256="a" * 64,
        collector_protocol_path=paths[1],
        expected_collector_protocol_sha256="b" * 64,
        collector_index_path=paths[2],
        expected_collector_index_sha256="c" * 64,
        historical_manifest_path=paths[3],
        expected_historical_manifest_sha256="d" * 64,
        prior_canary_index_path=paths[4],
        expected_prior_canary_index_sha256="e" * 64,
        prior_canary_manifest_path=paths[5],
        expected_prior_canary_manifest_sha256="f" * 64,
        development_batch_manifest_paths=(paths[6],),
        expected_development_batch_manifest_sha256s=("1" * 64,),
        v6_2_batch_manifest_paths=(paths[6],),
        expected_v6_2_batch_manifest_sha256s=("2" * 64,),
        implementation_commit="3" * 40,
        stage_started_ts=1,
    )
    assert config.run_id == "future-freeze"
    with pytest.raises(ValueError, match="must be nonempty and aligned"):
        AdaptiveSupportControllerV81FutureFreezeConfig(
            run_id="future-freeze",
            output_dir=tmp_path,
            plan_path=paths[0],
            expected_plan_sha256="a" * 64,
            collector_protocol_path=paths[1],
            expected_collector_protocol_sha256="b" * 64,
            collector_index_path=paths[2],
            expected_collector_index_sha256="c" * 64,
            historical_manifest_path=paths[3],
            expected_historical_manifest_sha256="d" * 64,
            prior_canary_index_path=paths[4],
            expected_prior_canary_index_sha256="e" * 64,
            prior_canary_manifest_path=paths[5],
            expected_prior_canary_manifest_sha256="f" * 64,
            development_batch_manifest_paths=(paths[6],),
            expected_development_batch_manifest_sha256s=(),
            v6_2_batch_manifest_paths=(paths[6],),
            expected_v6_2_batch_manifest_sha256s=("2" * 64,),
            implementation_commit="3" * 40,
            stage_started_ts=1,
        )


def test_v8_1_prior_identity_registry_uses_target_free_lineage(
    tmp_path: Path,
) -> None:
    def write_json(path: Path, payload: object) -> dict[str, str]:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def write_jsonl(path: Path, rows: list[dict]) -> dict[str, str]:
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    rank_lineage = write_jsonl(
        tmp_path / "rank-lineage.jsonl",
        [{"market_id": "rank-market", "score_uses_target_outcome_or_pnl": False}],
    )
    selected = write_jsonl(
        tmp_path / "selected-index.jsonl",
        [
            {
                "market_id": "selected-market",
                "slug": "selected-slug",
                "decision_id": "selected-decision",
                "source_row_hash": "selected-source",
            }
        ],
    )
    five_action = write_jsonl(
        tmp_path / "five-action.jsonl",
        [
            {
                "market_id": "selected-market",
                "market_slug": "selected-slug",
                "decision_id": "action-decision",
                "source_feature_row_sha256": "action-source",
                "outcome_fields_used_as_decision_input": False,
                "target_used_as_decision_input": False,
            }
        ],
    )
    target_free_manifest = write_json(
        tmp_path / "target-free-manifest.json",
        {
            "target_free_freeze_passed": True,
            "labels_outcomes_resolution_or_pnl_opened": False,
            "settlement_provider_called": False,
            "source_score_mutated": False,
            "selected_window_rows": selected,
            "target_free_five_action_rows": five_action,
        },
    )
    legacy_targets = write_jsonl(
        tmp_path / "legacy-targets.jsonl",
        [{"market_id": "must-not-open", "settlement_pnl": 1.0}],
    )
    prior_canary_index = tmp_path / "prior-canary-index.jsonl"
    prior_canary_index.write_text("", encoding="utf-8")

    result = v81_future_pipeline._prior_reference_sets(
        historical={
            "rank_lineage_rows": rank_lineage,
            "consumed_stream_target_free_freeze_manifest": (
                target_free_manifest
            ),
            "seed_runtime_target_rows": legacy_targets,
            "consumed_stream_five_action_rows": legacy_targets,
        },
        prior_canary_index_path=prior_canary_index,
    )

    assert result["market_ids"] == {"rank-market", "selected-market"}
    assert result["slugs"] == {"selected-slug"}
    assert result["decision_ids"] == {
        "selected-decision",
        "action-decision",
    }
    assert result["source_row_hashes"] == {
        "selected-source",
        "action-source",
    }
    assert result["source_row_counts"] == {
        "historical_rows": 3,
        "historical_rank_lineage_rows": 1,
        "historical_target_free_selected_index_rows": 1,
        "historical_target_free_five_action_rows": 1,
        "target_free_canary_index": 0,
    }


def _index_row(
    sequence: int,
    *,
    valid: bool = True,
    market_id: str | None = None,
    market_start_ts: int | None = None,
) -> dict:
    start = market_start_ts or (
        STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE + sequence * 300_000
    )
    market = market_id or f"market-{sequence:03d}"
    return {
        "sequence": sequence,
        "run_id": f"round-{sequence:03d}",
        "scheduled_round_start_ts": start,
        "market_start_ts": start,
        "market_end_ts": start + 300_000,
        "market_id": market,
        "slug": f"slug-{sequence:03d}",
        "decision_id": f"decision-{sequence:03d}",
        "source_row_hash": f"source-{sequence:03d}",
        "capture_quality_valid": valid,
    }


def _action_rows(market_ids: list[str]) -> list[dict]:
    actions = [
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "NO_TRADE",
    ]
    output = []
    for index, market_id in enumerate(market_ids, start=1):
        decision_ts = 1_900_000_000_000 + index
        for action in actions:
            output.append(
                {
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "max_input_ts": decision_ts,
                    "market_close_ts": decision_ts + 60_000,
                    "action": action,
                    "decision_id": f"{market_id}-{action}",
                    "microstructure_snapshot": {"selected_execution_price": 0.5},
                }
            )
    return output


def _guard_rows(market_ids: list[str], *, accepted_count: int) -> list[dict]:
    return [
        {
            "market_id": market_id,
            "decision_ts": 1_900_000_000_000 + index,
            "selected_action": "BUY_UP_SELL_BEFORE_CLOSE",
            "selected_side": "UP",
            "execution_guard_order_allowed": index <= accepted_count,
            "execution_blocking_reason_codes": (
                [] if index <= accepted_count else ["rank_abstention"]
            ),
            "source_score_mutated": False,
            "labels_outcomes_or_pnl_opened": False,
            "current_guard_result_used_for_own_controller_decision": False,
            "current_guard_result_added_after_decision_freeze": True,
        }
        for index, market_id in enumerate(market_ids, start=1)
    ]


def test_v8_1_future_window_is_earliest_exact_120_and_disjoint() -> None:
    rows = [
        _index_row(
            1,
            market_start_ts=STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE,
        ),
        _index_row(2, valid=False),
        _index_row(3, market_id="prior-market"),
        *[_index_row(index) for index in range(4, 124)],
    ]
    selected, attempted, summary = (
        select_adaptive_support_controller_v8_1_future_holdout_window(
            rows,
            plan=_plan(),
            prior_market_ids={"prior-market"},
            prior_slugs=set(),
            prior_decision_ids=set(),
            prior_source_row_hashes=set(),
        )
    )
    assert len(selected) == 120
    assert len(attempted) == 123
    assert selected[0]["sequence"] == 4
    assert selected[-1]["sequence"] == 123
    assert summary["exact_window_ready"] is True
    assert summary["exclusion_reason_distribution"] == {
        "capture_quality_invalid": 1,
        "market_start_not_strictly_later": 1,
        "prior_market_id_overlap": 1,
        "scheduled_round_not_strictly_later": 1,
    }


def test_v8_1_target_free_freeze_preserves_support_and_safety() -> None:
    selected = [_index_row(index) for index in range(1, 121)]
    market_ids = [row["market_id"] for row in selected]
    actions = _action_rows(market_ids)
    candidate = _guard_rows(market_ids, accepted_count=40)
    baseline = _guard_rows(market_ids, accepted_count=120)
    report = build_adaptive_support_controller_v8_1_target_free_freeze_report(
        selected,
        attempted_rows=selected,
        action_rows=actions,
        candidate_guard_rows=candidate,
        baseline_guard_rows=baseline,
        selection_summary={"exact_window_ready": True},
        plan=_plan(),
        stage_started_ts=max(row["market_end_ts"] for row in selected) + 1,
        collector_index_sha256="a" * 64,
    )
    assert report["target_free_freeze_passed"] is True
    assert report["candidate_guard_accepted_market_count"] == 40
    assert report["side_quota_enabled"] is False
    assert report["future_target_access_allowed"] is True
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False

    runtime = materialize_adaptive_support_controller_v8_1_runtime_decisions(
        candidate,
        action_rows=actions,
    )
    assert len(runtime) == 40
    assert all(row["target_used_as_decision_time_input"] is False for row in runtime)


def _target_row(market_id: str, *, pnl: float) -> dict:
    return {
        "market_id": market_id,
        "decision_ts": 1_900_000_000_000,
        "max_input_ts": 1_900_000_000_000,
        "side": "UP",
        "action": "BUY_UP_SELL_BEFORE_CLOSE",
        "runtime_policy_after_cost_net_pnl_at_frozen_size": pnl,
        "target_available_only_post_exit_or_official_resolution": True,
        "target_used_as_decision_time_input": False,
    }


def test_v8_1_future_gate_accepts_equal_positive_noninferiority() -> None:
    market_ids = [f"market-{index:03d}" for index in range(1, 121)]
    candidate = [_target_row(market_id, pnl=0.01) for market_id in market_ids[:40]]
    baseline = copy.deepcopy(candidate)
    report = build_adaptive_support_controller_v8_1_future_pnl_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=market_ids,
        settled_market_ids=market_ids,
        plan=_plan(),
        target_free_freeze_sha256="b" * 64,
    )
    assert report["future_pnl_gate_passed"] is True
    assert report["candidate_after_cost_pnl"] > 0.0
    assert report["candidate_minus_v6_7_after_cost_pnl"] == 0.0
    assert report["equality_passes_noninferiority"] is True
    assert report["promotion_discussion_evidence_available"] is True
    assert report["promotion_evidence_eligible"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False


def test_v8_1_future_gate_fails_closed_when_candidate_is_inferior() -> None:
    market_ids = [f"market-{index:03d}" for index in range(1, 121)]
    candidate = [_target_row(market_id, pnl=0.005) for market_id in market_ids[:40]]
    baseline = [_target_row(market_id, pnl=0.01) for market_id in market_ids[:40]]
    report = build_adaptive_support_controller_v8_1_future_pnl_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=market_ids,
        settled_market_ids=market_ids,
        plan=_plan(),
        target_free_freeze_sha256="c" * 64,
    )
    assert report["future_pnl_gate_passed"] is False
    assert "candidate_total_pnl_inferior_to_v6_7" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]
    assert report["promotion_discussion_evidence_available"] is False
