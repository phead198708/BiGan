from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_attempt_002_collection import (
    Attempt002CollectionConfig,
    ChallengeAttempt002CollectionError,
    preflight_attempt_002_collection,
    run_attempt_002_collection,
    summarize_attempt_002_collection,
)
from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/v8/polymarket_configs"
PROTOCOL_PATH = CONFIG_DIR / "challenge_attempt_002_preregistration.json"
COLLECTOR_PROTOCOL_PATH = (
    CONFIG_DIR / "execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)
FEATURE_CONTRACT_PATH = (
    CONFIG_DIR
    / "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _authorization(path: Path) -> None:
    protocol = _json(PROTOCOL_PATH)
    _write_json(
        path,
        {
            "schema_version": (
                "bigan-v8-challenge-attempt-002-operator-authorization-v1"
            ),
            "attempt_id": protocol["attempt_id"],
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "authorization_scope": (
                "outcome_blind_collection_of_exact_120_market_window_only"
            ),
            "exact_quality_valid_market_count": 120,
            "collection_authorized": True,
            "authorized_at": "2026-07-27T03:00:00Z",
            "authorization_source": "synthetic_test_fixture_only",
            "target_access_before_decision_freeze_authorized": False,
            "outcomes_during_collection_authorized": False,
            "paper_allowed": False,
            "live_allowed": False,
            "write_allowed": False,
            "wallet_allowed": False,
            "handoff_allowed": False,
            "promotion_allowed": False,
            "capital_at_risk": False,
        },
    )


def _config(
    tmp_path: Path,
    *,
    authorization_path: Path,
    authorization_sha256: str,
) -> Attempt002CollectionConfig:
    protocol = _json(PROTOCOL_PATH)
    service_root = (
        tmp_path / str(protocol["future_window"]["service_root"])
    )
    return Attempt002CollectionConfig(
        repository_root=tmp_path,
        protocol_path=PROTOCOL_PATH,
        expected_protocol_sha256=_sha256(PROTOCOL_PATH),
        operator_authorization_path=authorization_path,
        expected_operator_authorization_sha256=authorization_sha256,
        collector_protocol_path=COLLECTOR_PROTOCOL_PATH,
        expected_collector_protocol_sha256=_sha256(
            COLLECTOR_PROTOCOL_PATH
        ),
        feature_contract_path=FEATURE_CONTRACT_PATH,
        expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
        service_root=service_root,
        implementation_commit="a" * 40,
        run_id="attempt-002-test",
        failure_backoff_seconds=0.0,
    )


def _row(index: int, *, valid: bool, boundary: int) -> dict:
    start = boundary + (index + 1) * 300_000
    return {
        "sequence": index + 1,
        "scheduled_round_start_ts": start,
        "market_start_ts": start,
        "market_id": f"market-{index:03d}",
        "slug": f"slug-{index:03d}",
        "decision_id": f"{index + 1:064x}",
        "source_row_hash": f"{index + 1000:064x}",
        "capture_quality_valid": valid,
        "capture_quality_reason_codes": (
            [] if valid else ["provider_coverage_incomplete"]
        ),
        "duplicate_identity_reason_codes": [],
        "labels_outcomes_or_pnl_opened": False,
    }


def test_missing_authorization_fails_before_service_root_creation(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-authorization.json"
    config = _config(
        tmp_path,
        authorization_path=missing,
        authorization_sha256="0" * 64,
    )

    with pytest.raises(
        ChallengeAttempt002CollectionError,
        match="operator authorization is missing",
    ):
        preflight_attempt_002_collection(config)

    assert not config.service_root.exists()


def test_valid_preflight_is_read_only_and_pins_exact_limits(
    tmp_path: Path,
) -> None:
    authorization = tmp_path / "authorization.json"
    _authorization(authorization)
    config = _config(
        tmp_path,
        authorization_path=authorization,
        authorization_sha256=_sha256(authorization),
    )

    report = preflight_attempt_002_collection(config)

    assert report["exact_quality_valid_market_target"] == 120
    assert report["maximum_attempted_market_count"] == 180
    assert report["bounded_batch_market_count"] == 12
    assert report["maximum_batch_count"] == 15
    assert report["network_or_collection_invoked"] is False
    assert report["service_root_created"] is False
    assert report["safety"] == SAFE_FALSES
    assert not config.service_root.exists()


def test_collection_summary_selects_earliest_exact_120() -> None:
    boundary = 1_000_000
    rows = [_row(index, valid=True, boundary=boundary) for index in range(130)]

    report = summarize_attempt_002_collection(
        rows,
        boundary_exclusive=boundary,
    )

    assert report["attempted_market_count"] == 130
    assert report["quality_valid_market_count"] == 120
    assert report["target_reached"] is True
    assert report["selected_market_ids"] == [
        f"market-{index:03d}" for index in range(120)
    ]
    assert report["outcomes_resolution_labels_or_pnl_opened"] is False

    rows[0]["labels_outcomes_or_pnl_opened"] = True
    with pytest.raises(
        ChallengeAttempt002CollectionError,
        match="opened outcomes",
    ):
        summarize_attempt_002_collection(
            rows,
            boundary_exclusive=boundary,
        )


def test_supervisor_stops_after_ten_full_valid_batches(
    tmp_path: Path,
) -> None:
    authorization = tmp_path / "authorization.json"
    _authorization(authorization)
    config = _config(
        tmp_path,
        authorization_path=authorization,
        authorization_sha256=_sha256(authorization),
    )
    boundary = _json(PROTOCOL_PATH)[
        "preregistration_freeze_created_ts"
    ]
    rows: list[dict] = []

    def run_batch(**kwargs):
        del kwargs
        batch_number = len(rows) // 12 + 1
        rows.extend(
            _row(index, valid=True, boundary=boundary)
            for index in range(len(rows), len(rows) + 12)
        )
        canary = config.service_root / f"canary-{batch_number:03d}.json"
        _write_json(
            canary,
            {
                "development_data_canary_passed": True,
                "provider_health_validation_error_count": 0,
                "provider_health_diagnostics": {
                    "feature_completeness_report": {
                        "feature_row_count": 12,
                        "complete_feature_row_count": 12,
                        "incomplete_feature_row_count": 0,
                        "provider_health_bucket_counts": {"healthy": 12},
                    },
                    "missing_versus_zero_audit_report": {
                        "up_positive": 12,
                        "down_positive": 12,
                    },
                },
            },
        )
        return {
            "last_batch_canary_report_path": str(canary),
            "last_batch_canary_report_sha256": _sha256(canary),
            "labels_outcomes_or_pnl_opened": False,
        }

    result = run_attempt_002_collection(
        config,
        run_batch=run_batch,
        load_index=lambda path: list(rows),
        process_id=4242,
    )

    assert result["status"] == "quality_valid_target_reached"
    assert result["collector_pid"] is None
    assert result["batch_count"] == 10
    assert result["attempted_market_count"] == 120
    assert result["quality_valid_market_count"] == 120
    assert len(result["batch_reports"]) == 10
    assert result["outcomes_resolution_labels_or_pnl_opened"] is False
    assert result["safety"] == SAFE_FALSES
    final_batch = _json(
        Path(str(result["batch_reports"][-1]["path"]))
    )
    assert final_batch["batch_exclusion_reason_distribution"] == {}
    assert "cumulative attempted / quality-valid / remaining" in (
        final_batch["github_comment_markdown"]
    )
    assert "outcomes, resolution labels, and PnL opened: `false`" in (
        final_batch["github_comment_markdown"]
    )


def test_supervisor_fails_closed_at_180_attempts(
    tmp_path: Path,
) -> None:
    authorization = tmp_path / "authorization.json"
    _authorization(authorization)
    config = _config(
        tmp_path,
        authorization_path=authorization,
        authorization_sha256=_sha256(authorization),
    )
    boundary = _json(PROTOCOL_PATH)[
        "preregistration_freeze_created_ts"
    ]
    rows: list[dict] = []

    def run_batch(**kwargs):
        del kwargs
        batch_number = len(rows) // 12 + 1
        rows.extend(
            _row(index, valid=False, boundary=boundary)
            for index in range(len(rows), len(rows) + 12)
        )
        canary = config.service_root / f"canary-{batch_number:03d}.json"
        _write_json(
            canary,
            {
                "development_data_canary_passed": True,
                "provider_health_validation_error_count": 0,
                "provider_health_diagnostics": {
                    "feature_completeness_report": {
                        "feature_row_count": 12,
                        "complete_feature_row_count": 12,
                        "incomplete_feature_row_count": 0,
                        "provider_health_bucket_counts": {"healthy": 12},
                    },
                    "missing_versus_zero_audit_report": {},
                },
            },
        )
        return {
            "last_batch_canary_report_path": str(canary),
            "last_batch_canary_report_sha256": _sha256(canary),
            "labels_outcomes_or_pnl_opened": False,
        }

    result = run_attempt_002_collection(
        config,
        run_batch=run_batch,
        load_index=lambda path: list(rows),
        process_id=4242,
    )

    assert result["status"] == "attempt_cap_exhausted_fail_closed"
    assert result["fail_closed"] is True
    assert result["batch_count"] == 15
    assert result["attempted_market_count"] == 180
    assert result["quality_valid_market_count"] == 0
    assert result["exclusion_reason_distribution"][
        "provider_coverage_incomplete"
    ] == 180
    assert result["safety"] == SAFE_FALSES
