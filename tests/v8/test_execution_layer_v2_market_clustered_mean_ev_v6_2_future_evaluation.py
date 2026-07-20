from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation as subject
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    _evaluation_only_settled_corpus_if_safe,
)
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation import (
    MarketClusteredMeanEVV62FutureFreezeConfig,
    _bound_single_use_claim_path,
    _claim_single_use,
    _select_exact_future_index_rows,
    _target_free_support,
    _validate_collection_profile,
    _validate_exact_feature_action_grid,
    _validate_freeze_manifest_for_target_access,
    build_market_clustered_mean_ev_v6_2_side_only_gate,
    freeze_market_clustered_mean_ev_v6_2_future_predictions,
    validate_market_clustered_mean_ev_v6_2_future_profile,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation_v1.json"
)
COLLECTION_PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_market_clustered_mean_ev_v6_2_future_holdout_v1.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _evaluation_only_feature_row() -> dict:
    return {
        "market_id": "market-1",
        "condition_id": "condition-1",
        "slug": "btc-updown-5m-1",
        "market_family": "btc_updown_5m",
        "horizon_ms": 300_000,
        "decision_ts": 200,
        "feature_cutoff_ts": 200,
        "max_input_ts": 199,
        "available_at_ts": 200,
        "features": {"execution_price": 0.5},
        "feature_provenance": {"source": "frozen"},
    }


def _evaluation_only_corpus(tmp_path: Path, *, feature_row: dict) -> tuple[Path, dict]:
    copied = tmp_path / "copy"
    corpus = copied / "phase2_corpus"
    corpus.mkdir(parents=True)
    (corpus / "polymarket_feature_rows.jsonl").write_text(
        json.dumps(feature_row, sort_keys=True) + "\n", encoding="utf-8"
    )
    (corpus / "polymarket_label_rows.jsonl").write_text("{}\n", encoding="utf-8")
    (corpus / "polymarket_resolution_events.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    (corpus / "polymarket_corpus_manifest.json").write_text(
        json.dumps(
            {
                "sell_before_close_label_gate_passed": True,
                "label_row_count": 1,
                "chainlink_decision_time_feature_integration": {
                    "timestamp_causality_violation_count": 0
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (copied / "pending_round_finalization_manifest.json").write_text(
        json.dumps({"phase2_corpus_dir": str(corpus.resolve())}) + "\n",
        encoding="utf-8",
    )
    report = {
        "finalization_status": "blocked_fail_closed",
        "pending_resolution": False,
        "resolution_provider_called": True,
        "phase2_corpus_built": True,
        "phase2_error": (
            "Chainlink decision-time feature integration failed: "
            "chainlink_feature_builder_integration_failed"
        ),
        "raw_resolution_count": 1,
        "reject_reason_counts": {},
        "chainlink_corpus_evidence": {
            "reason_codes": [
                "chainlink_feature_builder_integration_failed",
                "chainlink_feature_builder_integration_still_required",
            ]
        },
    }
    return copied, report


def test_profile_freezes_exact_window_support_and_side_only_gate() -> None:
    profile = _profile()
    validate_market_clustered_mean_ev_v6_2_future_profile(profile)
    assert profile["window"]["quality_valid_market_count"] == 200
    assert profile["window"]["maximum_index_scan_count"] == 240
    assert profile["support_and_pnl_gates"][
        "minimum_guard_accepted_unique_market_count"
    ] == 120
    assert profile["support_and_pnl_gates"]["minimum_supported_side_market_count"] == 17
    assert profile["support_and_pnl_gates"]["pnl_hard_gate_aggregation"] == (
        "selected_side_buy_up_buy_down_only"
    )
    assert profile["access_sequence"]["future_result_driven_rerun_allowed"] is False
    assert profile["safety"]["promotion_evidence_eligible"] is False


def test_evaluation_only_settlement_fallback_requires_frozen_feature_equality(
    tmp_path: Path,
) -> None:
    frozen = _evaluation_only_feature_row()
    copied, report = _evaluation_only_corpus(tmp_path, feature_row=frozen)

    corpus, reasons = _evaluation_only_settled_corpus_if_safe(
        copied_run_dir=copied,
        report=report,
        frozen_feature_rows=[frozen],
    )

    assert corpus == (copied / "phase2_corpus").resolve()
    assert reasons == []


def test_evaluation_only_settlement_fallback_fails_closed_on_feature_drift(
    tmp_path: Path,
) -> None:
    frozen = _evaluation_only_feature_row()
    settled = copy.deepcopy(frozen)
    settled["features"]["execution_price"] = 0.51
    copied, report = _evaluation_only_corpus(tmp_path, feature_row=settled)

    corpus, reasons = _evaluation_only_settled_corpus_if_safe(
        copied_run_dir=copied,
        report=report,
        frozen_feature_rows=[frozen],
    )

    assert corpus is None
    assert reasons == ["evaluation_only_frozen_feature_payload_mismatch"]


@pytest.mark.parametrize(
    ("path", "replacement", "reason"),
    [
        (("window", "candidate_freeze_created_ts_exclusive"), 1, "window_contract"),
        (("window", "all_markets_closed_before_target_access"), False, "window_contract"),
        (
            ("support_and_pnl_gates", "minimum_guard_accepted_bet_count"),
            119,
            "support_and_pnl_gate_contract",
        ),
        (
            (
                "support_and_pnl_gates",
                "accepted_bet_total_post_cost_pnl_minimum_exclusive",
            ),
            -1.0,
            "support_and_pnl_gate_contract",
        ),
        (
            ("support_and_pnl_gates", "bootstrap_seed"),
            1,
            "support_and_pnl_gate_contract",
        ),
        (
            ("access_sequence", "future_result_driven_rerun_allowed"),
            True,
            "access_sequence_contract",
        ),
        (("candidate_model_sha256",), "0" * 64, "frozen_lineage"),
    ],
)
def test_profile_rejects_any_frozen_semantic_or_lineage_mutation(
    path: tuple[str, ...], replacement: object, reason: str
) -> None:
    profile = copy.deepcopy(_profile())
    destination = profile
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = replacement
    with pytest.raises(ValueError, match=reason):
        validate_market_clustered_mean_ev_v6_2_future_profile(profile)


def test_collection_profile_is_bound_to_candidate_and_exact_window() -> None:
    collection = json.loads(COLLECTION_PROFILE_PATH.read_text(encoding="utf-8"))
    candidate_sha256 = _profile()["candidate_manifest_sha256"]
    _validate_collection_profile(collection, candidate_sha256=candidate_sha256)
    collection["candidate"]["candidate_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="candidate_contract"):
        _validate_collection_profile(collection, candidate_sha256=candidate_sha256)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "reason"),
    [
        ("collection", "attempt_scan_cap", 241, "collection_contract"),
        (
            "target_free_support",
            "minimum_guard_accepted_unique_market_count",
            119,
            "support_contract",
        ),
        (
            "early_stop",
            "threshold_or_model_change_after_canary_output_allowed",
            True,
            "early_stop_contract",
        ),
        (
            "future_evaluation",
            "single_evaluation_allowed",
            False,
            "evaluation_contract",
        ),
    ],
)
def test_collection_profile_rejects_frozen_contract_mutation(
    section: str, field: str, replacement: object, reason: str
) -> None:
    collection = json.loads(COLLECTION_PROFILE_PATH.read_text(encoding="utf-8"))
    collection[section][field] = replacement
    with pytest.raises(ValueError, match=reason):
        _validate_collection_profile(
            collection,
            candidate_sha256=_profile()["candidate_manifest_sha256"],
        )


def test_replacement_evaluation_hash_fails_before_prediction_or_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    with pytest.raises(ValueError, match="preregistered frozen #212 profile"):
        freeze_market_clustered_mean_ev_v6_2_future_predictions(
            MarketClusteredMeanEVV62FutureFreezeConfig(
                run_id="must-not-start",
                output_dir=output_dir,
                evaluation_profile_path=tmp_path / "replacement-profile.json",
                expected_evaluation_profile_sha256="0" * 64,
                collection_profile_path=tmp_path / "collection-profile.json",
                expected_collection_profile_sha256=(
                    subject.FROZEN_COLLECTION_PROFILE_SHA256
                ),
                candidate_manifest_path=tmp_path / "candidate-manifest.json",
                expected_candidate_manifest_sha256=(
                    subject.FROZEN_CANDIDATE_MANIFEST_SHA256
                ),
                cumulative_canary_manifest_path=tmp_path / "cumulative.json",
                expected_cumulative_canary_manifest_sha256="1" * 64,
                collector_index_path=tmp_path / "index.jsonl",
                expected_collector_index_sha256="2" * 64,
                builder_git_commit="a" * 40,
                decision_freeze_created_ts=1,
            )
        )
    assert not output_dir.exists()


def test_exact_window_uses_earliest_200_quality_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "_prior_market_reference", lambda candidate: (set(), "a" * 64))
    rows = [_index_row(sequence) for sequence in range(313, 553)]
    selected, attempted = _select_exact_future_index_rows(
        rows,
        profile=_profile(),
        candidate={},
    )
    assert len(selected) == 200
    assert selected[0]["sequence"] == 313
    assert selected[-1]["sequence"] == 512
    assert len(attempted) == 200


def test_exact_window_scans_invalid_rows_but_never_selects_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_prior_market_reference", lambda candidate: (set(), "a" * 64))
    rows = [_index_row(sequence) for sequence in range(313, 553)]
    rows[0]["capture_quality_valid"] = False
    rows[0]["capture_quality_reason_codes"] = ["market_id_missing"]
    rows[0]["market_id"] = ""
    rows[0]["market_start_ts"] = 0
    selected, attempted = _select_exact_future_index_rows(
        rows,
        profile=_profile(),
        candidate={},
    )
    assert selected[0]["sequence"] == 314
    assert selected[-1]["sequence"] == 513
    assert len(attempted) == 201


def test_exact_window_still_rejects_invalid_attempt_before_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_prior_market_reference", lambda candidate: (set(), "a" * 64))
    rows = [_index_row(sequence) for sequence in range(313, 553)]
    rows[0].update(
        {
            "capture_quality_valid": False,
            "market_id": "",
            "market_start_ts": 0,
            "scheduled_round_start_ts": 1,
        }
    )
    with pytest.raises(ValueError, match="scheduled strictly later"):
        _select_exact_future_index_rows(rows, profile=_profile(), candidate={})


def test_exact_window_rejects_resolution_or_target_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_prior_market_reference", lambda candidate: (set(), "a" * 64))
    rows = [_index_row(sequence) for sequence in range(313, 553)]
    rows[5]["raw_resolution_row_count"] = 1
    with pytest.raises(ValueError, match="resolution"):
        _select_exact_future_index_rows(rows, profile=_profile(), candidate={})
    rows[5]["raw_resolution_row_count"] = 0
    rows[5]["labels_outcomes_or_pnl_opened"] = True
    with pytest.raises(ValueError, match="target"):
        _select_exact_future_index_rows(rows, profile=_profile(), candidate={})


def test_target_free_support_is_unique_market_and_side_scoped() -> None:
    replay = [
        {
            "market_id": f"market-{index}",
            "selected_side": "UP" if index < 60 else "DOWN",
            "execution_guard_order_allowed": True,
        }
        for index in range(120)
    ]
    support = _target_free_support(replay, profile=_profile())
    assert support["target_free_support_gate_passed"] is True
    assert support["guard_accepted_unique_market_count"] == 120
    assert support["guard_accepted_unique_market_count_by_side"] == {
        "UP": 60,
        "DOWN": 60,
    }
    replay.extend([dict(replay[0]) for _ in range(100)])
    assert _target_free_support(replay, profile=_profile())[
        "guard_accepted_unique_market_count"
    ] == 120


def test_exact_materialized_grid_requires_five_actions_and_causal_features() -> None:
    selected = [
        {
            "market_id": "market-313",
            "market_start_ts": 1_784_471_300_000,
        }
    ]
    feature_rows = [
        {
            "market_id": "market-313",
            "decision_ts": 1_784_471_400_000,
            "max_input_ts": 1_784_471_399_999,
        }
    ]
    actions = [
        {
            **feature_rows[0],
            "action": action,
        }
        for action in sorted(subject.EXPECTED_ACTIONS)
    ]
    candidate = {"future_collection_minimum_created_ts_exclusive": 1_784_470_529_364}
    _validate_exact_feature_action_grid(
        feature_rows,
        actions,
        selected_rows=selected,
        candidate=candidate,
    )
    actions[0]["max_input_ts"] = actions[0]["decision_ts"] + 1
    with pytest.raises(ValueError, match="causality"):
        _validate_exact_feature_action_grid(
            feature_rows,
            actions,
            selected_rows=selected,
            candidate=candidate,
        )


def test_exact_200_freeze_materializes_raw_features_before_target_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_attempt = {
        **_index_row(313),
        "capture_quality_valid": False,
        "capture_quality_reason_codes": ["market_id_missing"],
        "market_id": "",
        "market_start_ts": 0,
        "market_end_ts": 0,
    }
    selected_rows = [
        invalid_attempt,
        *[_synthetic_selected_row(tmp_path, index) for index in range(1, 201)],
    ]
    bundle = _synthetic_freeze_bundle(tmp_path)
    monkeypatch.setattr(
        subject,
        "FROZEN_EVALUATION_PROFILE_SHA256",
        _sha256(bundle["evaluation_profile"]),
    )
    monkeypatch.setattr(
        subject,
        "FROZEN_COLLECTION_PROFILE_SHA256",
        _sha256(bundle["collection_profile"]),
    )
    monkeypatch.setattr(
        subject,
        "FROZEN_CANDIDATE_MANIFEST_SHA256",
        _sha256(bundle["candidate_manifest"]),
    )
    monkeypatch.setattr(
        subject,
        "FROZEN_CANDIDATE_MODEL_SHA256",
        _sha256(bundle["model"]),
    )
    monkeypatch.setattr(
        subject,
        "FROZEN_CANDIDATE_CALIBRATION_SHA256",
        _sha256(bundle["calibration"]),
    )
    monkeypatch.setattr(
        subject,
        "FROZEN_MATCHED_V5_MANIFEST_SHA256",
        _sha256(bundle["v5_manifest"]),
    )
    monkeypatch.setattr(
        subject,
        "FROZEN_FEATURE_CONTRACT_SHA256",
        _sha256(bundle["feature_contract"]),
    )
    monkeypatch.setattr(
        subject,
        "load_and_validate_persistent_outcome_blind_index",
        lambda _: selected_rows,
    )
    monkeypatch.setattr(subject, "_prior_market_reference", lambda _: (set(), "a" * 64))
    monkeypatch.setattr(subject.xgb.Booster, "load_model", lambda self, path: None)

    def fake_raw_predictions(booster, rows, *, feature_columns):
        output = []
        for row in rows:
            market_index = int(str(row["market_id"]).rsplit("-", 1)[-1])
            selected_action = (
                "BUY_UP_SELL_BEFORE_CLOSE"
                if market_index % 2 == 0
                else "BUY_DOWN_SELL_BEFORE_CLOSE"
            )
            output.append(
                    {
                        **row,
                        "raw_direct_predicted_net_return": (
                            0.10 if row["action"] == selected_action else 0.0
                        ),
                        "ranking_score_source": "synthetic_raw_model_score",
                    }
                )
        return output

    monkeypatch.setattr(subject, "_raw_target_stripped_predictions", fake_raw_predictions)
    monkeypatch.setattr(
        subject,
        "attach_frozen_execution_compatibility",
        lambda rows: [
            {
                **row,
                "guard_compatible_before_ranking": row["action"] != "NO_TRADE",
            }
            for row in rows
        ],
    )
    monkeypatch.setattr(
        subject,
        "apply_market_clustered_mean_ev_scores",
        lambda rows, *, calibration_artifact: [
            {
                **row,
                "scoring_lineage": "v6_2",
                "mean_ev_lower_confidence_bound": row[
                    "raw_direct_predicted_net_return"
                ],
                "raw_pairwise_rank_score": row[
                    "raw_direct_predicted_net_return"
                ],
                "pairwise_group_normalized_rank_score": row[
                    "raw_direct_predicted_net_return"
                ],
                "action_advantage_lcb_score_bucket": "synthetic_test_bucket",
                "action_advantage_lcb_estimate_source": "synthetic_test_source",
            }
            for row in rows
        ],
    )
    monkeypatch.setattr(
        subject,
        "apply_conformal_scores",
        lambda rows, *, calibration_artifact, profile: [
            {
                **row,
                "scoring_lineage": "matched_v5",
                "mean_ev_lower_confidence_bound": -1.0,
            }
            for row in rows
        ],
    )

    def fake_replay(rows, **kwargs):
        required_replay_fields = {
            "raw_pairwise_rank_score",
            "pairwise_group_normalized_rank_score",
            "action_advantage_lcb_score_bucket",
            "action_advantage_lcb_estimate_source",
        }
        assert all(required_replay_fields <= set(row) for row in rows)
        if rows and rows[0]["scoring_lineage"] == "matched_v5":
            return []
        grouped: dict[tuple[str, int], list[dict]] = {}
        for row in rows:
            grouped.setdefault((row["market_id"], int(row["decision_ts"])), []).append(row)
        output = []
        for group in grouped.values():
            selected = max(group, key=lambda row: float(row["mean_ev_lower_confidence_bound"]))
            output.append(
                {
                    **selected,
                    "selected_side": selected["side"],
                    "executed_action": selected["action"],
                    "execution_guard_order_allowed": True,
                    "execution_blocking_reason_codes": [],
                }
            )
        return output

    monkeypatch.setattr(subject, "_outcome_blind_acceptance_replay", fake_replay)
    result = freeze_market_clustered_mean_ev_v6_2_future_predictions(
        MarketClusteredMeanEVV62FutureFreezeConfig(
            run_id="exact-200-integration",
            output_dir=tmp_path / "runs",
            evaluation_profile_path=bundle["evaluation_profile"],
            expected_evaluation_profile_sha256=_sha256(bundle["evaluation_profile"]),
            collection_profile_path=bundle["collection_profile"],
            expected_collection_profile_sha256=_sha256(bundle["collection_profile"]),
            candidate_manifest_path=bundle["candidate_manifest"],
            expected_candidate_manifest_sha256=_sha256(bundle["candidate_manifest"]),
            cumulative_canary_manifest_path=bundle["cumulative_manifest"],
            expected_cumulative_canary_manifest_sha256=_sha256(
                bundle["cumulative_manifest"]
            ),
            collector_index_path=bundle["collector_index"],
            expected_collector_index_sha256=_sha256(bundle["collector_index"]),
            builder_git_commit="a" * 40,
            decision_freeze_created_ts=max(row["market_end_ts"] for row in selected_rows)
            + 1,
        )
    )

    report = result["report"]
    manifest = result["manifest"]
    assert report["selected_market_count"] == 200
    assert report["attempted_index_row_count"] == 201
    assert report["quality_invalid_attempt_count"] == 1
    assert report["quality_invalid_attempt_sequences"] == [313]
    assert report["quality_invalid_attempt_reason_distribution"] == {
        "market_id_missing": 1
    }
    assert (
        report[
            "quality_invalid_attempts_excluded_before_selected_market_identity_checks"
        ]
        is True
    )
    assert report["candidate_guard_accepted_unique_market_count"] == 200
    assert report["candidate_guard_accepted_unique_market_count_by_side"] == {
        "UP": 100,
        "DOWN": 100,
    }
    assert report["target_free_support_gate_passed"] is True
    assert report["future_target_access_allowed"] is True
    assert len(manifest["opened_raw_feature_artifacts"]) == 200
    assert manifest["labels_outcomes_or_pnl_opened"] is False
    assert manifest["resolution_artifact_opened"] is False
    assert not _bound_single_use_claim_path(result["manifest_path"]).exists()


def test_support_failure_keeps_target_access_fail_closed() -> None:
    manifest = {
        "schema_version": f"{subject.SCHEMA_PREFIX}-prediction-freeze-manifest-v1",
        "decision_freeze_written_before_target_access": True,
        "future_target_access_allowed": False,
        "labels_outcomes_or_pnl_opened": False,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        **subject._blocked_safety_fields(),
    }
    with pytest.raises(ValueError, match="support"):
        _validate_freeze_manifest_for_target_access(manifest)


def test_side_only_gate_passes_positive_two_sided_market_grouped_evidence() -> None:
    markets = [f"market-{index:03d}" for index in range(200)]
    candidate = [
        _evaluation_row(
            market_id=market,
            side="UP" if index < 60 else "DOWN",
            pnl=0.02,
        )
        for index, market in enumerate(markets[:120])
    ]
    gate = build_market_clustered_mean_ev_v6_2_side_only_gate(
        candidate,
        matched_v5_rows=[],
        evaluation_market_ids=markets,
        profile=_profile(),
        decision_freeze_sha256="a" * 64,
    )
    assert gate["future_gate_passed"] is True
    assert gate["candidate_post_cost_net_pnl"] == pytest.approx(2.4)
    assert gate["matched_v5_post_cost_net_pnl"] == 0.0
    assert gate["accepted_side_metrics"]["UP"]["accepted_bet_net_pnl_sum"] > 0.0
    assert gate["accepted_side_metrics"]["DOWN"]["accepted_bet_net_pnl_sum"] > 0.0
    assert gate["promotion_evidence_eligible"] is False
    assert gate["#134_resume_allowed"] is False
    assert gate["#146_start_allowed"] is False


def test_side_only_gate_fails_when_one_side_loses_even_if_total_is_positive() -> None:
    markets = [f"market-{index:03d}" for index in range(200)]
    candidate = [
        _evaluation_row(
            market_id=market,
            side="UP" if index < 20 else "DOWN",
            pnl=-0.01 if index < 20 else 0.03,
        )
        for index, market in enumerate(markets[:120])
    ]
    gate = build_market_clustered_mean_ev_v6_2_side_only_gate(
        candidate,
        matched_v5_rows=[],
        evaluation_market_ids=markets,
        profile=_profile(),
        decision_freeze_sha256="b" * 64,
    )
    assert gate["candidate_post_cost_net_pnl"] > 0.0
    assert gate["future_gate_passed"] is False
    assert "supported_side_post_cost_pnl_gate_failed" in gate[
        "future_gate_blocking_reason_codes"
    ]


def test_action_and_family_metrics_are_diagnostic_only() -> None:
    markets = [f"market-{index:03d}" for index in range(200)]
    candidate = [
        _evaluation_row(
            market_id=market,
            side="UP" if index < 60 else "DOWN",
            pnl=0.02,
            action=(
                "BUY_UP_HOLD_TO_SETTLEMENT"
                if index == 0
                else (
                    "BUY_UP_SELL_BEFORE_CLOSE"
                    if index < 60
                    else "BUY_DOWN_SELL_BEFORE_CLOSE"
                )
            ),
        )
        for index, market in enumerate(markets[:120])
    ]
    gate = build_market_clustered_mean_ev_v6_2_side_only_gate(
        candidate,
        matched_v5_rows=[],
        evaluation_market_ids=markets,
        profile=_profile(),
        decision_freeze_sha256="c" * 64,
    )
    assert gate["future_gate_passed"] is True
    assert gate["accepted_action_metrics"]["BUY_UP_HOLD_TO_SETTLEMENT"][
        "diagnostic_only"
    ] is True
    assert gate["accepted_action_family_metrics"]["HOLD_TO_SETTLEMENT"][
        "diagnostic_only"
    ] is True


def test_single_use_claim_is_atomic_and_cannot_be_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "single-use.json"
    _claim_single_use(path, {"claim_id": "first"})
    assert json.loads(path.read_text(encoding="utf-8"))["claim_id"] == "first"
    with pytest.raises(ValueError, match="already consumed"):
        _claim_single_use(path, {"claim_id": "second"})
    assert json.loads(path.read_text(encoding="utf-8"))["claim_id"] == "first"


def test_single_use_claim_path_is_deterministically_bound_to_freeze(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze" / "v6_2_future_prediction_freeze_manifest.json"
    assert _bound_single_use_claim_path(freeze) == (
        freeze.parent.resolve() / subject.SINGLE_USE_CLAIM_FILENAME
    )


def _index_row(sequence: int) -> dict:
    market_start_ts = 1_784_471_000_000 + sequence * 300_000
    return {
        "sequence": sequence,
        "market_id": f"market-{sequence}",
        "market_start_ts": market_start_ts,
        "scheduled_round_start_ts": market_start_ts,
        "capture_quality_valid": True,
        "raw_resolution_row_count": 0,
        "labels_outcomes_or_pnl_opened": False,
    }


def _evaluation_row(
    *,
    market_id: str,
    side: str,
    pnl: float,
    action: str | None = None,
) -> dict:
    selected_action = action or f"BUY_{side}_SELL_BEFORE_CLOSE"
    return {
        "market_id": market_id,
        "execution_guard_order_allowed": True,
        "accepted_bet_net_pnl": pnl,
        "selected_side": side,
        "executed_action": selected_action,
        "settlement_resolved": True,
        "target_joined_after_decision_freeze": True,
        "target_used_as_decision_input": False,
        "forbidden_outcome_field_used_for_decision": False,
        "feature_causality_violation": False,
        "provenance_violation": False,
        "runtime_state_violation": False,
    }


def _synthetic_freeze_bundle(tmp_path: Path) -> dict[str, Path]:
    feature_contract = Path(
        "examples/v8/polymarket_configs/"
        "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
    ).resolve()
    model = tmp_path / "model.json"
    model.write_text("{}\n", encoding="utf-8")
    calibration = tmp_path / "v6-2-calibration.json"
    _write_json(
        calibration,
        {
            "frozen": True,
            "sides": {
                "UP": {
                    "mean_residual": 0.0,
                    "mean_residual_upper_confidence_bound": 0.0,
                },
                "DOWN": {
                    "mean_residual": 0.0,
                    "mean_residual_upper_confidence_bound": 0.0,
                },
            },
        },
    )
    v5_calibration = tmp_path / "v5-calibration.json"
    v5_profile = tmp_path / "v5-profile.json"
    _write_json(v5_calibration, {})
    _write_json(v5_profile, {})
    v5_manifest = tmp_path / "v5-manifest.json"
    _write_json(
        v5_manifest,
        {
            "calibration_artifact": _descriptor(v5_calibration),
            "fit_profile": _descriptor(v5_profile),
        },
    )
    pre_audit = tmp_path / "pre-audit.json"
    _write_json(
        pre_audit,
        {
            "feature_contract": _descriptor(feature_contract),
            "v5_freeze_manifest": _descriptor(v5_manifest),
        },
    )
    candidate_manifest = tmp_path / "candidate-manifest.json"
    _write_json(
        candidate_manifest,
        {
            "candidate_name": "market_clustered_mean_ev_v6_2",
            "target_free_actionability_gate_passed": True,
            "research_actionability_candidate_frozen": True,
            "new_strictly_later_future_holdout_required": True,
            "future_collection_minimum_created_ts_exclusive": 1_784_470_529_364,
            "target_free_labels_outcomes_settlement_targets_or_pnl_opened": False,
            "pre_target_access_audit": _descriptor(pre_audit),
            "source_model": _descriptor(model),
            "market_clustered_mean_risk_calibration": _descriptor(calibration),
        },
    )
    collection_profile = tmp_path / "collection-profile.json"
    collection = json.loads(COLLECTION_PROFILE_PATH.read_text(encoding="utf-8"))
    collection["candidate"].update(
        {
            "candidate_manifest_sha256": _sha256(candidate_manifest),
            "model_sha256": _sha256(model),
            "calibration_sha256": _sha256(calibration),
        }
    )
    _write_json(collection_profile, collection)
    evaluation_profile = tmp_path / "evaluation-profile.json"
    evaluation = _profile()
    evaluation.update(
        {
            "collection_profile_sha256": _sha256(collection_profile),
            "candidate_manifest_sha256": _sha256(candidate_manifest),
            "candidate_model_sha256": _sha256(model),
            "candidate_calibration_sha256": _sha256(calibration),
            "matched_v5_manifest_sha256": _sha256(v5_manifest),
            "feature_contract_sha256": _sha256(feature_contract),
        }
    )
    _write_json(evaluation_profile, evaluation)
    batch_report = tmp_path / "batch-report.json"
    _write_json(
        batch_report,
        {"candidate_manifest_sha256": _sha256(candidate_manifest)},
    )
    cumulative_report = tmp_path / "cumulative-report.json"
    _write_json(
        cumulative_report,
        {
            "future_holdout_collection_complete": True,
            "quality_valid_market_count": 200,
            "attempted_market_count": 200,
            "target_free_terminal_blocked": False,
        },
    )
    cumulative_manifest = tmp_path / "cumulative-manifest.json"
    _write_json(
        cumulative_manifest,
        {
            "future_holdout_collection_complete": True,
            "target_free_terminal_blocked": False,
            "labels_outcomes_or_pnl_opened": False,
            "report": _descriptor(cumulative_report),
            "batch_reports": [_descriptor(batch_report)],
        },
    )
    collector_index = tmp_path / "collector-index.jsonl"
    collector_index.write_text("{}\n", encoding="utf-8")
    return {
        "candidate_manifest": candidate_manifest,
        "model": model,
        "calibration": calibration,
        "v5_manifest": v5_manifest,
        "feature_contract": feature_contract,
        "collection_profile": collection_profile,
        "evaluation_profile": evaluation_profile,
        "cumulative_manifest": cumulative_manifest,
        "collector_index": collector_index,
    }


def _synthetic_selected_row(tmp_path: Path, index: int) -> dict:
    start = 1_784_472_000_000 + index * 300_000
    end = start + 300_000
    market_id = f"future-market-{index:03d}"
    raw_dir = tmp_path / "raw" / market_id
    raw_dir.mkdir(parents=True)
    payloads: dict[str, list[dict]] = {
        "raw_polymarket_markets.jsonl": [
            {
                "market_id": market_id,
                "condition_id": f"condition-{index:03d}",
                "slug": f"btc-updown-5m-{start // 1000}",
                "market_family": "btc_updown_5m",
                "horizon_ms": 300_000,
                "market_start_ts": start,
                "market_end_ts": end,
                "settlement_ts": end,
                "up_token_id": f"up-token-{index:03d}",
                "down_token_id": f"down-token-{index:03d}",
                "reference_price_source": "polymarket_rtds_chainlink",
                "settlement_rule": "UP if end reference is at least start reference",
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        ],
        "raw_polymarket_orderbooks.jsonl": [],
        "raw_polymarket_trades.jsonl": [],
        "raw_binance_btcusdt_klines.jsonl": [],
        "raw_polymarket_chainlink_prices.jsonl": [],
    }
    for offset in range(-15, 5):
        ts = start + offset * 60_000
        payloads["raw_binance_btcusdt_klines.jsonl"].append(
            {
                "ts": ts,
                "available_at_ts": ts + 60_000,
                "open_price": 100_000.0 + offset,
                "high_price": 100_010.0 + offset,
                "low_price": 99_990.0 + offset,
                "close_price": 100_001.0 + offset,
                "volume": 1.0,
                "timeframe_ms": 60_000,
                "source": "binance_btcusdt",
            }
        )
    for offset in range(5):
        ts = start + offset * 60_000
        for outcome, bid, ask in (("UP", 0.54, 0.56), ("DOWN", 0.44, 0.46)):
            payloads["raw_polymarket_orderbooks.jsonl"].append(
                {
                    "market_id": market_id,
                    "token_id": f"{outcome.lower()}-token-{index:03d}",
                    "outcome": outcome,
                    "ts": ts,
                    "available_at_ts": ts,
                    "bid_price": bid,
                    "ask_price": ask,
                    "mid_price": (bid + ask) / 2.0,
                    "bid_size": 100.0,
                    "ask_size": 100.0,
                    "liquidity_depth": 200.0,
                    "paper_only": True,
                    "capital_at_risk": False,
                    "polymarket_write_enabled": False,
                    "wallet_signing_enabled": False,
                }
            )
        payloads["raw_polymarket_chainlink_prices.jsonl"].append(
            {
                "source_ts": ts,
                "available_at_ts": ts,
                "price": 100_000.0 + offset,
                "source_type": "polymarket_rtds_chainlink",
                "symbol": "BTC/USD",
                "read_only": True,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        )
    descriptors = {}
    for filename, rows in payloads.items():
        path = raw_dir / filename
        _write_jsonl(path, rows)
        descriptors[filename] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "row_count": len(rows),
        }
    return {
        "sequence": 313 + index,
        "scheduled_round_start_ts": start,
        "market_start_ts": start,
        "market_end_ts": end,
        "market_id": market_id,
        "entry_sha256": hashlib.sha256(market_id.encode()).hexdigest(),
        "capture_quality_valid": True,
        "labels_outcomes_or_pnl_opened": False,
        "raw_resolution_row_count": 0,
        "raw_artifacts": descriptors,
    }


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
