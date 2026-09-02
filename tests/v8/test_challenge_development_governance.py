from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_governance import (
    _v8_1_feature_bridge,
    audit_capture,
    build_lane_health_summary,
    build_training_readiness,
    run_transfer_diagnostic_if_ready,
    validate_training_collector_cap_consistency,
    validate_training_protocol,
    validate_transfer_protocol,
    verify_legacy_v7_artifact_bundle,
    verify_repository_artifact_registry,
)
from bigan.v8.polymarket.challenge_development_lane import SAFETY, sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples" / "v8" / "polymarket_configs"
TRANSFER = CONFIG / "challenge_model_15m_transfer_diagnostic_protocol.json"
TRAINING = CONFIG / "challenge_model_15m_training_protocol_preregistration.json"
LANE_PROTOCOL = CONFIG / "challenge_model_development_lane_15m_protocol.json"


def test_frozen_protocols_preserve_development_only_gates() -> None:
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    lane = json.loads(LANE_PROTOCOL.read_text(encoding="utf-8"))
    validate_transfer_protocol(transfer, verify_artifact_bytes=True)
    validate_training_protocol(training)
    consistency = validate_training_collector_cap_consistency(training, lane)
    registry = verify_repository_artifact_registry(transfer)
    bundle = verify_legacy_v7_artifact_bundle(transfer["legacy_v7_selected"])
    assert transfer["promotion_evidence_eligible"] is False
    assert training["readiness_gate"]["minimum_quality_valid_outcome_finalized_market_count"] == 100
    assert consistency == {
        "minimum_quality_valid_outcome_finalized_market_count": 100,
        "maximum_capture_attempts_without_additional_permission": 120,
        "maximum_quality_valid_market_count_per_attempt": 1,
    }
    assert training["representation"]["missing_encoded_as_numeric_zero_allowed"] is False
    assert training["safety"] == SAFETY
    assert registry["cwd_independent_resolution"] is True
    assert bundle["synthetic_prediction_finite"] is True
    assert bundle["settlement_prediction_shape"] == [1, 3]
    assert bundle["settlement_residual_prediction_shape"] == [1]
    bundle_descriptor = transfer["legacy_v7_selected"]["artifact_bundle"]
    bundle_parent = Path(bundle_descriptor["path"]).parent
    assert bundle_parent.name == bundle_descriptor["bundle_sha256"]
    for descriptor in (
        transfer["legacy_v7_selected"]["model_artifact"],
        transfer["legacy_v7_selected"]["settlement_model"],
        transfer["legacy_v7_selected"]["settlement_residual_model"],
    ):
        assert not Path(descriptor["path"]).is_absolute()
        assert Path(descriptor["path"]).parent == bundle_parent
    v81_descriptor = transfer["v8_1_unit_controller"]["model_artifact"]
    assert not Path(v81_descriptor["path"]).is_absolute()
    assert f"sha256/{v81_descriptor['sha256']}/" in v81_descriptor["path"]


def test_repository_artifact_bundle_is_portable_from_fresh_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone_root = tmp_path / "fresh-clone"
    destination = clone_root / "examples" / "v8" / "polymarket_artifacts"
    shutil.copytree(ROOT / "examples" / "v8" / "polymarket_artifacts", destination)
    outside_cwd = tmp_path / "unrelated-cwd"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    validate_transfer_protocol(
        transfer,
        verify_artifact_bytes=True,
        repository_root=clone_root,
    )
    registry = verify_repository_artifact_registry(
        transfer,
        repository_root=clone_root,
    )
    bundle = verify_legacy_v7_artifact_bundle(
        transfer["legacy_v7_selected"],
        repository_root=clone_root,
    )
    assert Path(registry["registry_path"]).is_relative_to(clone_root)
    assert Path(bundle["bundle_manifest_path"]).is_relative_to(clone_root)
    assert bundle["metadata_sha256"] == transfer["legacy_v7_selected"][
        "model_artifact"
    ]["sha256"]
    assert bundle["settlement_model_sha256"] == transfer["legacy_v7_selected"][
        "settlement_model"
    ]["sha256"]
    assert bundle["settlement_residual_model_sha256"] == transfer[
        "legacy_v7_selected"
    ]["settlement_residual_model"]["sha256"]
    assert bundle["synthetic_prediction_finite"] is True


def test_repository_artifact_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    clone_root = tmp_path / "fresh-clone"
    destination = clone_root / "examples" / "v8" / "polymarket_artifacts"
    shutil.copytree(ROOT / "examples" / "v8" / "polymarket_artifacts", destination)
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    residual_path = clone_root / transfer["legacy_v7_selected"][
        "settlement_residual_model"
    ]["path"]
    residual_path.write_bytes(residual_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes do not match pinned SHA-256"):
        verify_legacy_v7_artifact_bundle(
            transfer["legacy_v7_selected"],
            repository_root=clone_root,
        )


def test_readiness_policy_rejects_unreachable_minimum_and_complement_proxy() -> None:
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    lane = json.loads(LANE_PROTOCOL.read_text(encoding="utf-8"))
    inconsistent = copy.deepcopy(training)
    inconsistent["readiness_gate"]["minimum_quality_valid_outcome_finalized_market_count"] = 121
    with pytest.raises(ValueError, match="15m training protocol invalid"):
        validate_training_collector_cap_consistency(inconsistent, lane)

    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    complement = copy.deepcopy(transfer)
    complement["legacy_v7_selected"]["complement_quote_proxy_allowed"] = True
    with pytest.raises(ValueError, match="15m transfer protocol invalid"):
        validate_transfer_protocol(complement, verify_artifact_bytes=False)


def test_v8_1_bridge_is_handicapped_and_uses_900_second_horizon() -> None:
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    v81 = transfer["v8_1_unit_controller"]
    assert v81["bridge_handicapped_not_native"] is True
    assert v81["native_action_score_coverage"] == 0.0
    assert v81["time_normalized_feature_audit"] == [
        {
            "feature": "late_window_pressure",
            "denominator_source": "feature_row.horizon_ms",
            "required_market_horizon_seconds": 900,
            "hardcoded_300_seconds_allowed": False,
        }
    ]
    row = _transfer_feature_row("market-bridge", 900_000, time_to_close_seconds=450.0)
    bridge = _v8_1_feature_bridge(row, side="UP")
    assert bridge["action_score_available"] == 0.0
    assert math.isnan(bridge["action_score"])
    assert math.isnan(bridge["action_score_margin"])
    assert bridge["late_window_pressure"] == pytest.approx(0.5)
    row["features"]["time_to_close_seconds"] = 900.0
    assert _v8_1_feature_bridge(row, side="UP")["late_window_pressure"] == 0.0
    row["features"]["time_to_close_seconds"] = 0.0
    assert _v8_1_feature_bridge(row, side="UP")["late_window_pressure"] == 1.0

    row["horizon_ms"] = 300_000
    with pytest.raises(ValueError, match="900-second"):
        _v8_1_feature_bridge(row, side="UP")


def test_capture_audit_counts_true_paired_asks_and_causal_streams(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    corpus = run_dir / "phase2_corpus"
    corpus.mkdir(parents=True)
    rows = [_feature_row(1000), _feature_row(2000)]
    _write_jsonl(corpus / "polymarket_feature_rows.jsonl", rows)
    (run_dir / "pending_round_capture_report.json").write_text(
        json.dumps(
            {
                "resolution_provider_called": False,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
                "live_exchange_write_enabled": False,
                "broker_exchange_write_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    audit = audit_capture(_capture(run_dir))
    assert audit["quality_valid"] is True
    assert audit["paired_executable_ask_decision_count"] == 2
    assert audit["book_causal_complete_decision_count"] == 2
    assert audit["chainlink_causal_complete_decision_count"] == 2
    assert audit["trade_tape_causal_complete_decision_count"] == 2

    rows[0]["features"]["down_ask"] = None
    _write_jsonl(corpus / "polymarket_feature_rows.jsonl", rows)
    failed = audit_capture(_capture(run_dir))
    assert failed["quality_valid"] is False
    assert "paired_executable_ask_coverage_failed" in failed["exclusion_reason_codes"]


def test_health_and_training_gate_remain_closed_below_threshold(
    tmp_path: Path,
) -> None:
    lane = tmp_path / "lane"
    run_dir = lane / "capture"
    corpus = run_dir / "phase2_corpus"
    corpus.mkdir(parents=True)
    _write_jsonl(
        corpus / "polymarket_feature_rows.jsonl",
        [_feature_row(1000), _feature_row(2000)],
    )
    (run_dir / "pending_round_capture_report.json").write_text(
        json.dumps(
            {
                "resolution_provider_called": False,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
                "live_exchange_write_enabled": False,
                "broker_exchange_write_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    batch_summary = lane / "batch_summary.json"
    batch_summary.write_text(
        json.dumps({"captures": [_capture(run_dir)]}),
        encoding="utf-8",
    )
    _write_jsonl(
        lane / "outcome_blind_capture_batch_index.jsonl",
        [
            {
                "batch_summary_path": str(batch_summary),
                "collected_at": "2026-07-27T00:00:00+00:00",
            }
        ],
    )
    manifest = lane / "development_corpus" / "polymarket_corpus_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    _write_jsonl(
        lane / "finalized_development_corpus_index.jsonl",
        [
            {
                "run_id": "run-1",
                "exported_corpus_manifest_path": str(manifest),
            }
        ],
    )
    health = build_lane_health_summary(
        lane_root=lane,
        date_utc="2026-07-27",
        write=False,
    )
    assert health["cumulative"]["attempted_market_count"] == 1
    assert health["cumulative"]["quality_valid_outcome_finalized_market_count"] == 1
    assert health["cumulative"]["paired_up_down_executable_ask"]["coverage"] == 1.0

    readiness = build_training_readiness(
        lane_root=lane,
        training_protocol_path=TRAINING,
        expected_training_protocol_sha256=sha256_file(TRAINING),
        transfer_protocol_path=TRANSFER,
        expected_transfer_protocol_sha256=sha256_file(TRANSFER),
        write=False,
    )
    assert readiness["training_start_allowed"] is False
    assert readiness["attempt_120_authorized"] is True
    assert readiness["attempt_121_authorized"] is False
    assert (
        "quality_valid_outcome_finalized_market_count_at_least_100"
        in readiness["blocking_reason_codes"]
    )
    assert readiness["metrics"]["readiness_count_reachable_under_collector_cap"] is True
    waiting = run_transfer_diagnostic_if_ready(
        lane_root=lane,
        protocol_path=TRANSFER,
        expected_protocol_sha256=sha256_file(TRANSFER),
    )
    assert waiting["transfer_diagnostic_started"] is False
    assert waiting["required_market_count"] == 40


def test_synthetic_40_market_transfer_keeps_lifecycles_and_bridge_limits_separate(
    tmp_path: Path,
) -> None:
    lane = _synthetic_transfer_lane(tmp_path, market_count=40)
    result = run_transfer_diagnostic_if_ready(
        lane_root=lane,
        protocol_path=TRANSFER,
        expected_protocol_sha256=sha256_file(TRANSFER),
    )
    assert result["status"] == "transfer_diagnostic_complete"
    assert result["market_count"] == 40

    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    panels = report["policy_panels"]
    v7 = panels["legacy_v7_hold_to_settlement"]
    v81 = panels["v8_1_sell_before_close_bridge"]
    assert v7["lifecycle_policy"] == "HOLD_TO_SETTLEMENT"
    assert v81["lifecycle_policy"] == "SELL_BEFORE_CLOSE"
    assert v81["transfer_status"] == "bridge_handicapped_not_native"
    assert v81["bridge_handicapped_not_native"] is True
    assert v81["native_action_score_coverage"] == 0.0
    assert v81["bridge_limitations"]
    coverage = v81["native_feature_coverage"]
    assert 0.0 < coverage["coverage"] < 1.0
    assert coverage["native_feature_count"] + len(
        coverage["forced_missing_feature_names"]
    ) + len(coverage["bridged_or_assumed_feature_names"]) == coverage[
        "model_feature_count"
    ]
    assert v81["metrics"]["source_action_score_forced_missing_as_xgboost_nan"] is True
    assert v81["metrics"]["accepted_count"] == v81["metrics"]["accepted_market_count"]
    assert "unit_net_pnl_mean" in v81["metrics"]
    assert "unit_net_pnl_bootstrap_interval" in v81["metrics"]
    assert "weak_evidence_status" in v81["weak_evidence"]
    assert v81["weak_evidence"]["promotion_claim_made"] is False
    serialized_bridge_panel = json.dumps(v81, sort_keys=True).lower()
    assert "full_retrain" not in serialized_bridge_panel
    assert '"winner"' not in serialized_bridge_panel
    assert report["cross_policy_comparison"]["superiority_claim_made"] is False
    assert report["cross_policy_comparison"]["superiority_claim_allowed"] is False
    assert "judgements" not in report
    assert "legacy_v7_selected" not in report
    assert "v8_1_unit_controller" not in report

    output = lane / "transfer_diagnostic"
    for path in (
        output / "legacy_v7_hold_to_settlement_market_rows.jsonl",
        output / "v8_1_sell_before_close_bridge_market_rows.jsonl",
    ):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 40
        assert all(row["complement_quote_proxy_used"] is False for row in rows)


def _synthetic_transfer_lane(tmp_path: Path, *, market_count: int) -> Path:
    lane = tmp_path / "synthetic-lane"
    captures = []
    finalized = []
    for index in range(market_count):
        run_id = f"run-{index:03d}"
        market_id = f"market-{index:03d}"
        start_ts = 10_000_000 + index * 2_000_000
        decision_rows = [
            _transfer_feature_row(
                market_id,
                start_ts + 300_000,
                time_to_close_seconds=600.0,
            ),
            _transfer_feature_row(
                market_id,
                start_ts + 600_000,
                time_to_close_seconds=300.0,
            ),
        ]
        run_dir = lane / "captures" / run_id
        capture_corpus = run_dir / "phase2_corpus"
        capture_corpus.mkdir(parents=True)
        _write_jsonl(
            capture_corpus / "polymarket_feature_rows.jsonl",
            decision_rows,
        )
        (run_dir / "pending_round_capture_report.json").write_text(
            json.dumps(_safe_capture_report()),
            encoding="utf-8",
        )
        provider_raw = run_dir / "provider_raw"
        _write_jsonl(provider_raw / "raw_polymarket_trades.jsonl", [])
        _write_jsonl(
            provider_raw / "raw_polymarket_orderbooks.jsonl",
            _synthetic_orderbooks(decision_rows),
        )
        captures.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "round_index": index + 1,
                "scheduled_round_start_ts": start_ts,
                "capture_start_boundary_validation_passed": True,
                "raw_polymarket_market_count": 1,
                "provider_raw_orderbook_snapshot_count": 8,
                "orderbook_full_window_coverage_passed": True,
                "raw_btc_candle_row_count": 10,
                "raw_chainlink_price_row_count": 10,
                "chainlink_rtds_price_stream_fresh": True,
                "reject_reason_counts": {},
            }
        )

        finalized_dir = lane / "development_corpus" / run_id
        finalized_dir.mkdir(parents=True)
        _write_jsonl(
            finalized_dir / "polymarket_feature_rows.jsonl",
            decision_rows,
        )
        _write_jsonl(
            finalized_dir / "polymarket_label_rows.jsonl",
            _synthetic_labels(decision_rows, resolved_outcome=("UP" if index % 2 == 0 else "DOWN")),
        )
        manifest = finalized_dir / "polymarket_corpus_manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        finalized.append(
            {
                "run_id": run_id,
                "exported_corpus_manifest_path": str(manifest),
            }
        )

    summary = lane / "synthetic_batch_summary.json"
    summary.write_text(json.dumps({"captures": captures}), encoding="utf-8")
    _write_jsonl(
        lane / "outcome_blind_capture_batch_index.jsonl",
        [
            {
                "batch_summary_path": str(summary),
                "collected_at": "2026-07-28T00:00:00+00:00",
            }
        ],
    )
    _write_jsonl(lane / "finalized_development_corpus_index.jsonl", finalized)
    return lane


def _safe_capture_report() -> dict:
    return {
        "resolution_provider_called": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "live_exchange_write_enabled": False,
        "broker_exchange_write_enabled": False,
    }


def _transfer_feature_row(
    market_id: str,
    decision_ts: int,
    *,
    time_to_close_seconds: float,
) -> dict:
    provenance = {
        name: {
            "available_at_ts": decision_ts,
            "input_end_ts": decision_ts,
        }
        for name in ("up_ask", "down_ask")
    }
    provenance.update(
        {
            name: {
                "available_at_ts": decision_ts,
                "max_input_ts": decision_ts,
                "provenance_valid": True,
            }
            for name in (
                "chainlink_price_at_decision",
                "chainlink_reference_price_at_market_start",
            )
        }
    )
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "available_at_ts": decision_ts,
        "max_input_ts": decision_ts,
        "horizon_ms": 900_000,
        "features": {
            "up_ask": 0.51,
            "up_bid": 0.49,
            "up_mid": 0.50,
            "up_ask_size": 50.0,
            "up_bid_size": 50.0,
            "up_spread_bps": 400.0,
            "up_queue_fill_probability_proxy": 0.8,
            "up_book_staleness_ms": 100.0,
            "down_ask": 0.52,
            "down_bid": 0.50,
            "down_mid": 0.51,
            "down_ask_size": 50.0,
            "down_bid_size": 50.0,
            "down_spread_bps": 392.156862745098,
            "down_queue_fill_probability_proxy": 0.8,
            "down_book_staleness_ms": 100.0,
            "btc_return_30s": 0.001,
            "btc_return_1m": 0.001,
            "reference_price_to_beat_distance_at_decision": 0.001,
            "time_to_close_seconds": time_to_close_seconds,
            "recent_trade_volume_coverage_complete": 1,
            "trade_tape_available_at_ts": decision_ts,
            "trade_tape_max_causal_input_ts": decision_ts,
            "trade_tape_provider_timeout": 0,
            "trade_tape_truncated": 0,
            "trade_tape_censored": 0,
            "trade_tape_historical_backfill": 0,
        },
        "feature_provenance": provenance,
    }


def _synthetic_orderbooks(feature_rows: list[dict]) -> list[dict]:
    rows = []
    for feature_row in feature_rows:
        decision_ts = int(feature_row["decision_ts"])
        for side, mid in (("UP", 0.50), ("DOWN", 0.51)):
            rows.extend(
                [
                    {
                        "outcome": side,
                        "ts": decision_ts - 900_000,
                        "available_at_ts": decision_ts - 900_000,
                        "mid_price": mid - 0.01,
                    },
                    {
                        "outcome": side,
                        "ts": decision_ts,
                        "available_at_ts": decision_ts,
                        "mid_price": mid,
                    },
                ]
            )
    return rows


def _synthetic_labels(
    feature_rows: list[dict],
    *,
    resolved_outcome: str,
) -> list[dict]:
    labels = []
    for feature_row in feature_rows:
        features = feature_row["features"]
        for side in ("UP", "DOWN"):
            prefix = side.lower()
            entry_mid = float(features[f"{prefix}_mid"])
            entry_ask = float(features[f"{prefix}_ask"])
            payout = 1.0 if resolved_outcome == side else 0.0
            common = {
                "market_id": feature_row["market_id"],
                "decision_ts": feature_row["decision_ts"],
                "entry_mid": entry_mid,
                "entry_ask": entry_ask,
                "resolved_outcome": resolved_outcome,
                "fees": 0.001,
                "slippage": 0.0,
                "liquidity_impact": 0.0,
            }
            labels.append(
                {
                    **common,
                    "action": f"BUY_{side}_HOLD_TO_SETTLEMENT",
                    "settlement_payout": payout,
                    "total_net_pnl_per_notional": payout - entry_ask - 0.001,
                }
            )
            labels.append(
                {
                    **common,
                    "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
                    "exit_bid": 0.54,
                    "exit_ask": 0.56,
                    "total_net_pnl_per_notional": 0.54 - entry_ask - 0.001,
                }
            )
    return labels


def _capture(run_dir: Path) -> dict:
    return {
        "run_id": "run-1",
        "run_dir": str(run_dir),
        "round_index": 1,
        "scheduled_round_start_ts": 1,
        "capture_start_boundary_validation_passed": True,
        "raw_polymarket_market_count": 1,
        "provider_raw_orderbook_snapshot_count": 100,
        "orderbook_full_window_coverage_passed": True,
        "raw_btc_candle_row_count": 10,
        "raw_chainlink_price_row_count": 10,
        "chainlink_rtds_price_stream_fresh": True,
        "reject_reason_counts": {},
    }


def _feature_row(decision_ts: int) -> dict:
    provenance = {
        name: {
            "available_at_ts": decision_ts,
            "input_end_ts": decision_ts,
        }
        for name in ("up_ask", "down_ask")
    }
    provenance.update(
        {
            name: {
                "available_at_ts": decision_ts,
                "max_input_ts": decision_ts,
                "provenance_valid": True,
            }
            for name in (
                "chainlink_price_at_decision",
                "chainlink_reference_price_at_market_start",
            )
        }
    )
    return {
        "market_id": "market-1",
        "decision_ts": decision_ts,
        "available_at_ts": decision_ts,
        "max_input_ts": decision_ts,
        "features": {
            "up_ask": 0.51,
            "down_ask": 0.50,
            "recent_trade_volume_coverage_complete": 1,
            "trade_tape_available_at_ts": decision_ts,
            "trade_tape_max_causal_input_ts": decision_ts,
            "trade_tape_provider_timeout": 0,
            "trade_tape_truncated": 0,
            "trade_tape_censored": 0,
            "trade_tape_historical_backfill": 0,
        },
        "feature_provenance": provenance,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
