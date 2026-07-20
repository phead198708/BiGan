from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_sbc_exit_reliability_v6_3 import (
    SBCExitReliabilityV63AuditConfig,
    run_sbc_exit_reliability_v6_3_audit,
    validate_sbc_exit_reliability_v6_3_profile,
)


def test_profile_freezes_support_access_and_safety() -> None:
    profile = _profile()
    validate_sbc_exit_reliability_v6_3_profile(profile)
    assert profile["audit_support_gates"]["minimum_eligible_market_count"] == 180
    assert profile["access_sequence"]["schema_inspection_contamination_is_excluded"]
    assert profile["safety"]["source_model_candidate_eligible"] is False


def test_freeze_excludes_schema_inspection_and_does_not_open_label_content(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    frozen = _run_freeze(fixture, tmp_path)
    manifest = frozen["manifest"]
    assert manifest["source_market_count"] == 3
    assert manifest["eligible_market_count"] == 2
    assert manifest["label_file_content_opened"] is False
    assert manifest["schema_inspection_contaminated_market_ids"] == ["market-0"]
    assert manifest["source_model_candidate_eligible"] is False


def test_audit_reports_both_side_support_and_decision_time_causality(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    frozen = _run_freeze(fixture, tmp_path)
    audited = _run_audit(fixture, frozen, tmp_path)
    labels = audited["label_audit"]
    features = audited["feature_coverage"]
    assert labels["target_counts_by_side"] == {
        "UP": {"1": 2, "0": 2},
        "DOWN": {"1": 2, "0": 2},
    }
    assert labels["label_causality_violation_count"] == 0
    assert features["feature_causality_violation_count"] == 0
    assert features["prohibited_feature_field_count"] == 0
    assert audited["manifest"]["source_model_candidate_eligible"] is False


def test_audit_requires_exact_frozen_lineage_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    frozen = _run_freeze(fixture, tmp_path)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        run_sbc_exit_reliability_v6_3_audit(
            _audit_config(
                fixture,
                tmp_path,
                lineage=frozen["lineage_manifest_path"],
                lineage_sha="f" * 64,
            )
        )


def test_executable_exit_must_be_strictly_after_decision(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, causality_violation=True)
    frozen = _run_freeze(fixture, tmp_path)
    audited = _run_audit(fixture, frozen, tmp_path)
    assert audited["label_audit"]["label_causality_violation_count"] == 1
    assert audited["label_audit"]["label_audit_gate_passed"] is False


def _profile() -> dict:
    path = Path(
        "examples/v8/polymarket_configs/"
        "execution_layer_v2_sbc_exit_reliability_v6_3_profile.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture(tmp_path: Path, *, causality_violation: bool = False) -> dict:
    root = tmp_path / "corpus"
    root.mkdir()
    profile = copy.deepcopy(_profile())
    profile["source_contract"].update(
        {
            "source_market_count": 3,
            "eligible_corpus_root": str(root),
            "eligible_max_decision_ts_exclusive": 10_000,
            "role_market_counts_before_exclusion": {
                "development_train": 2,
                "development_calibration": 1,
                "confirmatory_validation": 0,
            },
            "role_market_counts_after_exclusion": {
                "development_train": 1,
                "development_calibration": 1,
                "confirmatory_validation": 0,
            },
            "schema_inspection_contaminated_market_ids": ["market-0"],
            "schema_inspection_contaminated_slugs": ["market-0"],
        }
    )
    profile["audit_support_gates"].update(
        {
            "minimum_eligible_market_count": 2,
            "minimum_role_market_counts": {
                "development_train": 1,
                "development_calibration": 1,
                "confirmatory_validation": 0,
            },
            "minimum_sell_before_close_label_count_per_side": 2,
            "minimum_positive_exit_label_count_per_side": 1,
            "minimum_negative_exit_label_count_per_side": 1,
            "minimum_markets_with_both_target_classes_per_side": 1,
            "minimum_raw_exit_window_snapshot_coverage_rate": 0.5,
        }
    )
    source_dirs = []
    role_rows = []
    for index, role in enumerate(
        ("development_train", "development_train", "development_calibration")
    ):
        source = root / f"market-{index}"
        source.mkdir()
        source_dirs.append(source)
        features = [_feature_row(f"market-{index}", 1000), _feature_row(f"market-{index}", 2000)]
        labels = []
        for decision_ts in (1000, 2000):
            for side in ("UP", "DOWN"):
                labels.append(
                    _label_row(
                        market_id=f"market-{index}",
                        decision_ts=decision_ts,
                        side=side,
                        executable=decision_ts == 1000,
                        causality_violation=(
                            causality_violation
                            and index == 1
                            and decision_ts == 1000
                            and side == "UP"
                        ),
                    )
                )
        _write_jsonl(source / "polymarket_feature_rows.jsonl", features)
        _write_jsonl(source / "polymarket_label_rows.jsonl", labels)
        _write_jsonl(source / "polymarket_token_book_snapshots.jsonl", [{"ts": 1}])
        _write_jsonl(source / "raw_polymarket_orderbooks.jsonl", [{"ts": 1}])
        corpus = {
            "sell_before_close_label_schema_version": (
                "bigan-v8-polymarket-sell-before-close-executable-exit-v1"
            ),
            "sell_before_close_label_gate_passed": True,
            "normalized_artifact_hashes": {
                "feature_rows": _sha(source / "polymarket_feature_rows.jsonl"),
                "label_rows": _sha(source / "polymarket_label_rows.jsonl"),
                "token_book_snapshots": _sha(
                    source / "polymarket_token_book_snapshots.jsonl"
                ),
            },
            "raw_artifact_hashes": {
                "raw_polymarket_orderbooks.jsonl": _sha(
                    source / "raw_polymarket_orderbooks.jsonl"
                )
            },
        }
        _write_json(source / "polymarket_corpus_manifest.json", corpus)
        role_rows.append(
            {
                "market_id": f"market-{index}",
                "role": role,
                "source_corpus_dir": str(source),
                "minimum_decision_ts": 1000,
                "maximum_decision_ts": 2000,
                "decision_row_count": 2,
                "training_sampled_orderbook_row_count": 1,
                "provider_raw_orderbook_snapshot_count": 1,
                "labels_or_outcomes_opened_for_role_assignment": False,
                "feature_rows": _descriptor(source / "polymarket_feature_rows.jsonl"),
                "corpus_manifest": _descriptor(source / "polymarket_corpus_manifest.json"),
            }
        )
    role_rows_path = tmp_path / "role_rows.jsonl"
    _write_jsonl(role_rows_path, role_rows)
    role_manifest_path = tmp_path / "role_manifest.json"
    _write_json(
        role_manifest_path,
        {"selected_rows": _descriptor(role_rows_path)},
    )
    profile["source_lineage"]["role_assignment_manifest_sha256"] = _sha(
        role_manifest_path
    )
    profile["source_lineage"]["role_assignment_rows_sha256"] = _sha(role_rows_path)
    profile_path = tmp_path / "profile.json"
    _write_json(profile_path, profile)
    return {
        "profile_path": profile_path,
        "role_manifest_path": role_manifest_path,
        "source_dirs": source_dirs,
    }


def _run_freeze(fixture: dict, tmp_path: Path) -> dict:
    return run_sbc_exit_reliability_v6_3_audit(
        SBCExitReliabilityV63AuditConfig(
            stage="freeze_lineage",
            run_id="freeze",
            output_dir=tmp_path / "runs",
            profile_path=fixture["profile_path"],
            expected_profile_sha256=_sha(fixture["profile_path"]),
            role_assignment_manifest_path=fixture["role_manifest_path"],
            implementation_commit="a" * 40,
            overwrite_existing=True,
        )
    )


def _run_audit(fixture: dict, frozen: dict, tmp_path: Path) -> dict:
    return run_sbc_exit_reliability_v6_3_audit(
        _audit_config(
            fixture,
            tmp_path,
            lineage=frozen["lineage_manifest_path"],
            lineage_sha=_sha(frozen["lineage_manifest_path"]),
        )
    )


def _audit_config(
    fixture: dict, tmp_path: Path, *, lineage: Path, lineage_sha: str
) -> SBCExitReliabilityV63AuditConfig:
    return SBCExitReliabilityV63AuditConfig(
        stage="audit_labels",
        run_id="audit",
        output_dir=tmp_path / "runs",
        profile_path=fixture["profile_path"],
        expected_profile_sha256=_sha(fixture["profile_path"]),
        role_assignment_manifest_path=fixture["role_manifest_path"],
        implementation_commit="a" * 40,
        lineage_manifest_path=lineage,
        expected_lineage_manifest_sha256=lineage_sha,
        overwrite_existing=True,
    )


def _feature_row(market_id: str, decision_ts: int) -> dict:
    profile = _profile()
    payload = dict.fromkeys(
        profile["feature_contract"]["common_required_features"], 1.0
    )
    for side in ("up", "down"):
        for suffix in profile["feature_contract"]["side_required_feature_suffixes"]:
            payload[f"{side}_{suffix}"] = 1.0
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts,
        "features": payload,
    }


def _label_row(
    *,
    market_id: str,
    decision_ts: int,
    side: str,
    executable: bool,
    causality_violation: bool = False,
) -> dict:
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "sell_before_close_execution_class": (
            "realizable_sell_before_close"
            if executable
            else "non_executable_sell_before_close"
        ),
        "label_uses_executable_exit_path": executable,
        "sell_before_close_exit_path": {
            "candidate_exit_snapshot_count": 1 if executable else 0,
            "best_executable_exit_ts": (
                decision_ts if causality_violation else decision_ts + 1
            )
            if executable
            else 0,
            "exit_path_reason_codes": [
                "executable_intraround_exit_found"
                if executable
                else "no_candidate_exit_snapshot"
            ],
        },
        "paper_only": True,
        "capital_at_risk": False,
    }


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
