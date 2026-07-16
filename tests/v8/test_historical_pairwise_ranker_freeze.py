"""Tests for the historical-train-only pairwise ranker freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (
    _load_corpus_action_rows,
    _train_pairwise_ranker,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)
from bigan.v8.polymarket.training.historical_pairwise_ranker_freeze import (
    HistoricalPairwiseRankerFreezeConfig,
    freeze_historical_pairwise_ranker,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_v1.json"
)
FEATURE_CONTRACT_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)


def test_historical_corpus_materialization_requires_complete_cost_aware_grid(
    tmp_path: Path,
) -> None:
    corpus_dir, role_row = _build_training_corpus(tmp_path, market_id="market-a")
    feature_columns = tuple(_load_json(FEATURE_CONTRACT_PATH)["feature_columns"])

    rows, audit = _load_corpus_action_rows(
        corpus_dir,
        role_row=role_row,
        feature_columns=feature_columns,
    )

    assert len(rows) == 5
    assert audit["blocking_reason_codes"] == []
    assert all(row["target_used_as_decision_input"] is False for row in rows)
    assert all(row["outcome_fields_used_as_decision_input"] is False for row in rows)

    labels_path = corpus_dir / "polymarket_label_rows.jsonl"
    labels = _jsonl(labels_path)[:-1]
    _write_jsonl(labels_path, labels)
    _refresh_manifest_hash(corpus_dir, "label_rows", labels_path)
    _, incomplete = _load_corpus_action_rows(
        corpus_dir,
        role_row=_role_row(corpus_dir, "market-a"),
        feature_columns=feature_columns,
    )
    assert "incomplete_5_action_label_grid" in incomplete["blocking_reason_codes"]


def test_historical_corpus_materialization_rejects_missing_cost_components(
    tmp_path: Path,
) -> None:
    corpus_dir, _ = _build_training_corpus(tmp_path, market_id="market-a")
    labels_path = corpus_dir / "polymarket_label_rows.jsonl"
    labels = _jsonl(labels_path)
    del labels[0]["fees"]
    _write_jsonl(labels_path, labels)
    _refresh_manifest_hash(corpus_dir, "label_rows", labels_path)

    _, audit = _load_corpus_action_rows(
        corpus_dir,
        role_row=_role_row(corpus_dir, "market-a"),
        feature_columns=tuple(_load_json(FEATURE_CONTRACT_PATH)["feature_columns"]),
    )

    assert "cost_aware_label_contract_violation" in audit["blocking_reason_codes"]


def test_historical_corpus_materialization_rejects_feature_leakage(
    tmp_path: Path,
) -> None:
    corpus_dir, _ = _build_training_corpus(tmp_path, market_id="market-a")
    feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
    features = _jsonl(feature_path)
    features[0]["max_input_ts"] = features[0]["decision_ts"] + 1
    features[0]["features"]["settlement_pnl"] = 1.0
    _write_jsonl(feature_path, features)
    _refresh_manifest_hash(corpus_dir, "feature_rows", feature_path)

    _, audit = _load_corpus_action_rows(
        corpus_dir,
        role_row=_role_row(corpus_dir, "market-a"),
        feature_columns=tuple(_load_json(FEATURE_CONTRACT_PATH)["feature_columns"]),
    )

    assert (
        "feature_timestamp_or_field_causality_violation"
        in audit["blocking_reason_codes"]
    )


def test_pairwise_ranker_model_hash_is_deterministic(tmp_path: Path) -> None:
    rows = [
        row
        for market_index in range(90)
        for row in _synthetic_action_rows(
            market_id=f"market-{market_index:03d}",
            decision_ts=1_000 + market_index,
        )
    ]
    protocol = dict(_load_json(PROTOCOL_PATH)["cross_fit_protocol"])
    model_protocol = {
        key: protocol[key]
        for key in (
            "objective",
            "eval_metric",
            "num_boost_round",
            "max_depth",
            "eta",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "lambda",
            "alpha",
            "seed",
            "nthread",
            "verbosity",
        )
    }
    model_protocol["num_boost_round"] = 3
    first = _train_pairwise_ranker(
        rows,
        feature_columns=("execution_price",),
        model_protocol=model_protocol,
    )
    second = _train_pairwise_ranker(
        rows,
        feature_columns=("execution_price",),
        model_protocol=model_protocol,
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first.save_model(first_path)
    second.save_model(second_path)

    assert _sha256(first_path) == _sha256(second_path)


def test_historical_ranker_freeze_generates_no_calibration_or_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _registry_fixture(tmp_path)
    action_rows = [
        row
        for market_index in range(90)
        for row in _synthetic_action_rows(
            market_id=f"market-{market_index:03d}",
            decision_ts=1_000 + market_index,
        )
    ]

    def fake_materialize(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], list[dict]]:
        del args, kwargs
        return {"development_train": action_rows}, [
            {
                "feature_causality_violation_count": 0,
                "blocking_reason_codes": [],
            }
            for _ in range(90)
        ]

    oof_rows = [
        {
            "market_id": row["market_id"],
            "decision_ts": row["decision_ts"],
            "action": row["action"],
            "action_row_sha256": row["action_row_sha256"],
            "oof_raw_prediction": 0.1,
            "target_net_pnl_per_contract": row["target_net_pnl_per_contract"],
        }
        for row in action_rows
        if int(str(row["market_id"]).split("-")[-1]) >= 15
    ]

    def fake_cross_fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "method": "fixture",
            "fold_count": 5,
            "market_count": 90,
            "decision_group_count": 90,
            "oof_market_count": 75,
            "oof_decision_group_count": 75,
            "oof_prediction_count": len(oof_rows),
            "future_market_label_access_violation_count": 0,
            "fold_reports": [
                {
                    "fold_index": index,
                    "training_market_ids_sha256": str(index) * 64,
                    "validation_market_ids_sha256": str(index + 1) * 64,
                    "training_max_decision_ts": index * 100,
                    "validation_min_decision_ts": index * 100 + 1,
                    "training_strictly_precedes_validation": True,
                }
                for index in range(1, 6)
            ],
            "ranking_metrics": {},
            "oof_predictions": oof_rows,
        }

    class FakeBooster:
        def save_model(self, path: Path) -> None:
            path.write_text('{"fixture_model":true}\n', encoding="utf-8")

    monkeypatch.setattr(
        "bigan.v8.polymarket.training.historical_pairwise_ranker_freeze."
        "_materialize_role_action_rows",
        fake_materialize,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.historical_pairwise_ranker_freeze."
        "_cross_fit_training_predictions",
        fake_cross_fit,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.historical_pairwise_ranker_freeze."
        "_train_pairwise_ranker",
        lambda *args, **kwargs: FakeBooster(),
    )

    result = freeze_historical_pairwise_ranker(
        _config(tmp_path, fixture)
    )

    manifest = result["freeze_manifest"]
    assert manifest["historical_training_complete"] is True
    assert manifest["fresh_calibration_required"] is True
    assert manifest["rank_scores_execution_eligible"] is False
    assert manifest["action_advantage_lcb_artifact_created"] is False
    assert manifest["calibrated_expected_net_return_available"] is False
    assert manifest["policy_replay_attempted"] is False
    assert manifest["accepted_bet_evaluation_attempted"] is False
    assert manifest["source_model_candidate_eligible"] is False
    assert manifest["freeze_ready"] is False
    assert manifest["promotion_evidence_eligible"] is False
    assert manifest["v8_execution_handoff_allowed"] is False
    names = {path.name for path in result["run_dir"].iterdir()}
    assert not any("calibration_artifact" in name for name in names)
    assert not any("replay" in name for name in names)


def test_registry_hash_drift_blocks_before_fit(tmp_path: Path) -> None:
    fixture = _registry_fixture(tmp_path)
    fixture["rows_path"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="registry rows SHA-256 mismatch"):
        freeze_historical_pairwise_ranker(_config(tmp_path, fixture))


def _config(
    tmp_path: Path,
    fixture: dict[str, Path | str],
) -> HistoricalPairwiseRankerFreezeConfig:
    return HistoricalPairwiseRankerFreezeConfig(
        run_id="historical-ranker",
        output_dir=tmp_path / "runs",
        registry_descriptor_path=fixture["descriptor_path"],
        expected_registry_descriptor_sha256=str(fixture["descriptor_sha256"]),
        registry_manifest_path=fixture["manifest_path"],
        expected_registry_manifest_sha256=str(fixture["manifest_sha256"]),
        registry_report_path=fixture["report_path"],
        expected_registry_report_sha256=str(fixture["report_sha256"]),
        registry_rows_path=fixture["rows_path"],
        expected_registry_rows_sha256=str(fixture["rows_sha256"]),
        protocol_path=PROTOCOL_PATH,
        expected_protocol_sha256=_sha256(PROTOCOL_PATH),
        feature_contract_path=FEATURE_CONTRACT_PATH,
        expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
    )


def _registry_fixture(tmp_path: Path) -> dict[str, Path | str]:
    input_dir = tmp_path / "registry"
    input_dir.mkdir()
    corpus_manifest = input_dir / "corpus-manifest.json"
    _write_json(corpus_manifest, {"fixture": True})
    rows = []
    for index in range(90):
        market_id = f"market-{index:03d}"
        row = {
            "selection_rank": index + 1,
            "role": "historical_development_train",
            "market_id": market_id,
            "round_slug": f"btc-updown-5m-{index}",
            "corpus_id": f"corpus-{index}",
            "corpus_dir": str(input_dir),
            "minimum_decision_ts": 1_000 + index,
            "maximum_decision_ts": 1_000 + index,
            "maximum_feature_input_ts": 1_000 + index,
            "strictly_before_boundary": True,
            "artifact_pins": {
                "polymarket_corpus_manifest.json": {
                    "path": str(corpus_manifest.resolve()),
                    "sha256": _sha256(corpus_manifest),
                }
            },
            "fresh_calibration_eligible": False,
            "fresh_confirmatory_eligible": False,
            "labels_or_outcomes_used_for_selection": False,
            "outcome_values_loaded": False,
            "pnl_values_loaded": False,
        }
        row["registry_row_id"] = canonical_json_sha256(row)
        rows.append(row)
    rows_path = input_dir / "registry_rows.jsonl"
    _write_jsonl(rows_path, rows)
    market_ids_hash = canonical_json_sha256([row["market_id"] for row in rows])
    report = {
        "selected_market_count": 90,
        "selected_market_ids_sha256": market_ids_hash,
        "all_selected_strictly_before_boundary": True,
        "duplicate_selected_market_count": 0,
        "forbidden_evidence_access_audit": {
            "selection_uses_only_compatibility_time_and_identity": True,
            "label_rows_semantic_content_parsed": False,
            "resolution_rows_semantic_content_parsed": False,
            "outcome_values_loaded": False,
            "pnl_values_loaded": False,
            "oracle_values_loaded": False,
            "oof_metrics_loaded": False,
            "validation_metrics_loaded": False,
            "confirmatory_metrics_loaded": False,
        },
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = input_dir / "registry_report.json"
    _write_json(report_path, report)
    manifest = {
        "selected_market_count": 90,
        "selected_market_ids_sha256": market_ids_hash,
        "registry_rows": _descriptor(rows_path),
        "registry_report": _descriptor(report_path),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = input_dir / "registry_manifest.json"
    _write_json(manifest_path, manifest)
    descriptor = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "registry_rows_sha256": _sha256(rows_path),
        "registry_report_sha256": _sha256(report_path),
        "selected_market_ids_sha256": market_ids_hash,
    }
    descriptor["descriptor_id"] = canonical_json_sha256(descriptor)
    descriptor_path = input_dir / "registry_descriptor.json"
    _write_json(descriptor_path, descriptor)
    return {
        "descriptor_path": descriptor_path,
        "descriptor_sha256": _sha256(descriptor_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
        "rows_path": rows_path,
        "rows_sha256": _sha256(rows_path),
    }


def _build_training_corpus(
    tmp_path: Path,
    *,
    market_id: str,
) -> tuple[Path, dict[str, Any]]:
    corpus_dir = tmp_path / market_id
    corpus_dir.mkdir()
    decision_ts = 1_000_000
    feature_row = {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts,
        "feature_provenance": {
            "reference_price_to_beat_distance_at_decision": {
                "provenance_valid": True,
                "max_input_ts": decision_ts,
            }
        },
        "features": _feature_values(),
    }
    feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
    _write_jsonl(feature_path, [feature_row])
    labels = []
    for index, action in enumerate(REQUIRED_ACTIONS):
        labels.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "action": action,
                "fees": 0.001,
                "slippage": 0.001,
                "liquidity_impact": 0.001,
                "total_net_pnl_per_notional": 0.04 - index * 0.01,
                "paper_only": True,
                "capital_at_risk": False,
            }
        )
    label_path = corpus_dir / "polymarket_label_rows.jsonl"
    _write_jsonl(label_path, labels)
    metadata_path = corpus_dir / "polymarket_market_metadata.jsonl"
    _write_jsonl(
        metadata_path,
        [
            {
                "market_id": market_id,
                "condition_id": market_id,
                "slug": f"btc-updown-5m-{decision_ts // 1000}",
                "market_end_ts": decision_ts + 300_000,
            }
        ],
    )
    resolution_path = corpus_dir / "polymarket_resolution_events.jsonl"
    _write_jsonl(
        resolution_path,
        [{"market_id": market_id, "resolved_outcome": "UP"}],
    )
    manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    _write_json(
        manifest_path,
        {
            "normalized_artifact_hashes": {
                "feature_rows": _sha256(feature_path),
                "label_rows": _sha256(label_path),
                "market_metadata": _sha256(metadata_path),
                "resolution_events": _sha256(resolution_path),
            }
        },
    )
    return corpus_dir, _role_row(corpus_dir, market_id)


def _refresh_manifest_hash(corpus_dir: Path, key: str, path: Path) -> None:
    manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    manifest = _load_json(manifest_path)
    manifest["normalized_artifact_hashes"][key] = _sha256(path)
    _write_json(manifest_path, manifest)


def _role_row(corpus_dir: Path, market_id: str) -> dict[str, Any]:
    manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    return {
        "role": "development_train",
        "selection_rank": 1,
        "market_id": market_id,
        "source_corpus_dir": str(corpus_dir),
        "corpus_manifest": _descriptor(manifest_path),
    }


def _feature_values() -> dict[str, float]:
    values = {
        "btc_return_10s": 0.0,
        "btc_return_30s": 0.0,
        "btc_return_1m": 0.0,
        "btc_return_5m": 0.0,
        "btc_return_15m": 0.0,
        "btc_volatility_1m": 0.01,
        "btc_volatility_5m": 0.01,
        "btc_volatility_15m": 0.01,
        "reference_price_to_beat_distance_at_decision": 0.001,
        "time_to_close_seconds": 180.0,
        "market_age_seconds": 120.0,
        "combined_spread_bps": 200.0,
        "liquidity_imbalance": 0.0,
        "recent_up_trade_volume": 1.0,
        "recent_down_trade_volume": 1.0,
        "up_mid": 0.55,
        "down_mid": 0.45,
    }
    for side, bid, ask in (("up", 0.54, 0.56), ("down", 0.44, 0.46)):
        values.update(
            {
                f"{side}_bid": bid,
                f"{side}_ask": ask,
                f"{side}_spread_bps": 200.0,
                f"{side}_queue_fill_probability_proxy": 0.8,
                f"{side}_book_staleness_ms": 100.0,
                f"{side}_liquidity_depth": 100.0,
                f"{side}_executable_ask_notional": 50.0,
                f"{side}_executable_bid_notional": 50.0,
                f"{side}_recent_book_update_count_1m": 10.0,
                f"{side}_recent_spread_stability_1m": 1.0,
                f"{side}_recent_bid_depth_volatility_1m": 0.1,
            }
        )
    return values


def _synthetic_action_rows(*, market_id: str, decision_ts: int) -> list[dict[str, Any]]:
    rows = []
    for index, action in enumerate(REQUIRED_ACTIONS):
        side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
        family = (
            "HOLD_TO_SETTLEMENT"
            if "HOLD_TO_SETTLEMENT" in action
            else "SELL_BEFORE_CLOSE"
            if "SELL_BEFORE_CLOSE" in action
            else "NO_TRADE"
        )
        row = {
            "market_id": market_id,
            "decision_ts": decision_ts,
            "max_input_ts": decision_ts,
            "action": action,
            "side": side,
            "action_family": family,
            "decision_time_features": {
                "execution_price": 0.1 * index,
            },
            "target_net_pnl_per_contract": 0.04 - index * 0.01,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
        }
        row["action_row_sha256"] = canonical_json_sha256(row)
        rows.append(row)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
