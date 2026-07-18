from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training import (
    execution_layer_v2_pairwise_supplemental_support as supplemental,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_supplemental_support import (
    PairwiseSupplementalSupportFreezeConfig,
    PairwiseSupplementalSupportGateConfig,
    create_pairwise_supplemental_support_freeze,
    run_pairwise_supplemental_support_gate,
)


def test_successor_freeze_is_fixed_outcome_blind_and_immutable(
    tmp_path: Path,
) -> None:
    fixture = _parent_fixture(tmp_path)
    source_bytes = {
        path: path.read_bytes() for path in fixture["source_paths"]
    }

    result = _create_freeze(tmp_path, fixture)

    freeze = result["freeze"]
    report = result["report"]
    assert report["status"] == "SUPPLEMENTAL_SUPPORT_FREEZE_READY"
    assert report["parent_selected_market_count"] == 191
    assert report["required_supplemental_valid_market_count"] == 4
    assert report["supplemental_capture_attempt_count"] == 20
    assert report["maximum_total_capture_attempt_count"] == 280
    assert freeze["initial_capture_attempt_count"] == 280
    assert freeze["maximum_total_capture_attempt_count"] == 280
    assert freeze["supplemental_collection_stop_early_allowed"] is False
    assert freeze["supplemental_dynamic_extension_allowed"] is False
    assert freeze["labels_or_outcomes_opened_for_support_planning"] is False
    assert freeze["execution_thresholds_mutated"] is False
    assert freeze["collector_contract_mutated"] is False
    assert freeze["paper_only"] is True
    assert freeze["capital_at_risk"] is False
    assert freeze["v8_execution_handoff_allowed"] is False
    assert all(path.read_bytes() == value for path, value in source_bytes.items())


def test_parent_target_access_blocks_successor_freeze(tmp_path: Path) -> None:
    fixture = _parent_fixture(tmp_path)
    report = _load(fixture["support_report"])
    report["labels_or_outcomes_opened_for_continuation"] = True
    _write(fixture["support_report"], report)
    support_manifest = _load(fixture["support_manifest"])
    support_manifest["support_gate_report"] = _descriptor(
        fixture["support_report"]
    )
    _write(fixture["support_manifest"], support_manifest)

    with pytest.raises(ValueError, match="parent_support_opened_targets"):
        create_pairwise_supplemental_support_freeze(
            _freeze_config(tmp_path, fixture)
        )


def test_parent_hash_mismatch_fails_before_freeze(tmp_path: Path) -> None:
    fixture = _parent_fixture(tmp_path)
    config = _freeze_config(tmp_path, fixture)
    config = PairwiseSupplementalSupportFreezeConfig(
        **{
            field: getattr(config, field)
            for field in config.__dataclass_fields__
            if field != "parent_support_report_sha256"
        },
        parent_support_report_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        create_pairwise_supplemental_support_freeze(config)


def test_gate_rejects_less_than_fixed_twenty_attempts(
    tmp_path: Path,
) -> None:
    fixture = _parent_fixture(tmp_path)
    freeze = _create_freeze(tmp_path, fixture)
    batch = _supplemental_batch(
        tmp_path,
        freeze["freeze"],
        capture_count=19,
    )

    result = _run_gate(tmp_path, freeze, batch)

    assert result["report"]["supplemental_support_target_ready"] is False
    assert "supplemental_capture_attempt_count_mismatch" in result[
        "report"
    ]["blocking_reason_codes"]
    assert result["report"]["continuation_allowed"] is False


def test_gate_rejects_nonfuture_supplemental_capture(
    tmp_path: Path,
) -> None:
    fixture = _parent_fixture(tmp_path)
    freeze = _create_freeze(tmp_path, fixture)
    batch = _supplemental_batch(
        tmp_path,
        freeze["freeze"],
        scheduled_start=(
            freeze["freeze"]["supplemental_minimum_collection_decision_ts"]
            - 1
        ),
    )

    result = _run_gate(tmp_path, freeze, batch)

    assert "supplemental_capture_not_strictly_later" in result[
        "report"
    ]["blocking_reason_codes"]
    assert result["core_result"] is None


def test_gate_rejects_forbidden_supplemental_outcome_field(
    tmp_path: Path,
) -> None:
    fixture = _parent_fixture(tmp_path)
    freeze = _create_freeze(tmp_path, fixture)
    batch = _supplemental_batch(tmp_path, freeze["freeze"])
    payload = _load(batch)
    payload["captures"][0]["settlement_pnl"] = 1.0
    _write(batch, payload)

    result = _run_gate(tmp_path, freeze, batch)

    assert "batch_progress_forbidden_outcome_fields_present" in result[
        "report"
    ]["blocking_reason_codes"]
    assert result["core_result"] is None


def test_gate_accepts_exact_twenty_and_frozen_role_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _parent_fixture(tmp_path)
    freeze = _create_freeze(tmp_path, fixture)
    batch = _supplemental_batch(tmp_path, freeze["freeze"])
    fake = _fake_core_result(tmp_path, selected_count=195, ready=True)
    monkeypatch.setattr(
        supplemental,
        "run_pairwise_precollection_support_gate",
        lambda config: fake,
    )

    result = _run_gate(tmp_path, freeze, batch)

    report = result["report"]
    assert report["status"] == (
        "OUTCOME_BLIND_SUPPLEMENTAL_SUPPORT_TARGET_READY"
    )
    assert report["supplemental_support_target_ready"] is True
    assert report["combined_capture_attempt_count"] == 280
    assert report["selected_market_count"] == 195
    assert report["new_selected_market_count"] == 4
    assert report["role_market_counts"] == {
        "confirmatory_validation": 60,
        "development_calibration": 45,
        "development_train": 90,
    }
    assert report["continuation_allowed"] is False
    assert report["labels_or_outcomes_opened_for_support_gate"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False


def test_gate_remains_fail_closed_when_core_support_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _parent_fixture(tmp_path)
    freeze = _create_freeze(tmp_path, fixture)
    batch = _supplemental_batch(tmp_path, freeze["freeze"])
    fake = _fake_core_result(tmp_path, selected_count=194, ready=False)
    monkeypatch.setattr(
        supplemental,
        "run_pairwise_precollection_support_gate",
        lambda config: fake,
    )

    result = _run_gate(tmp_path, freeze, batch)

    report = result["report"]
    assert report["supplemental_support_target_ready"] is False
    assert "insufficient_support_at_frozen_maximum" in report[
        "blocking_reason_codes"
    ]
    assert "successor_target_market_count_not_reached" in report[
        "blocking_reason_codes"
    ]
    assert report["continuation_allowed"] is False
    assert report["v8_execution_handoff_allowed"] is False


def _create_freeze(tmp_path: Path, fixture: dict) -> dict:
    return create_pairwise_supplemental_support_freeze(
        _freeze_config(tmp_path, fixture)
    )


def _freeze_config(
    tmp_path: Path,
    fixture: dict,
) -> PairwiseSupplementalSupportFreezeConfig:
    return PairwiseSupplementalSupportFreezeConfig(
        run_id="successor-freeze",
        output_dir=tmp_path / "runs",
        freeze_created_ts=10_000_000,
        parent_precollection_freeze_path=fixture["parent_freeze"],
        parent_precollection_freeze_sha256=_sha(fixture["parent_freeze"]),
        parent_terminal_reconciliation_report_path=fixture[
            "terminal_report"
        ],
        parent_terminal_reconciliation_report_sha256=_sha(
            fixture["terminal_report"]
        ),
        parent_terminal_reconciliation_manifest_path=fixture[
            "terminal_manifest"
        ],
        parent_terminal_reconciliation_manifest_sha256=_sha(
            fixture["terminal_manifest"]
        ),
        parent_support_report_path=fixture["support_report"],
        parent_support_report_sha256=_sha(fixture["support_report"]),
        parent_support_manifest_path=fixture["support_manifest"],
        parent_support_manifest_sha256=_sha(
            fixture["support_manifest"]
        ),
        successor_freeze_builder_git_commit="a" * 40,
    )


def _run_gate(
    tmp_path: Path,
    freeze: dict,
    batch: Path,
) -> dict:
    return run_pairwise_supplemental_support_gate(
        PairwiseSupplementalSupportGateConfig(
            run_id=f"gate-{batch.stem}",
            output_dir=tmp_path / "gate-runs",
            successor_freeze_path=freeze["freeze_path"],
            successor_freeze_sha256=freeze["freeze_sha256"],
            supplemental_batch_progress_pins=((batch, _sha(batch)),),
            training_corpus_root=tmp_path / "training",
        )
    )


def _parent_fixture(tmp_path: Path) -> dict:
    batches: list[Path] = []
    capture_index = 0
    for batch_index in range(4):
        captures = []
        for _ in range(65):
            capture_index += 1
            captures.append(
                {
                    "run_id": f"parent-run-{capture_index:03d}",
                    "scheduled_round_start_ts": capture_index * 1_000,
                }
            )
        path = tmp_path / f"parent_batch_{batch_index + 1}.json"
        _write(
            path,
            {
                "batch_id": f"parent-batch-{batch_index + 1}",
                "capture_count": len(captures),
                "captures": captures,
                "error_count": 0,
                "errors": [],
                **_safe_fields(),
            },
        )
        batches.append(path)
    parent_freeze = tmp_path / "parent_freeze.json"
    _write(
        parent_freeze,
        {
            "schema_version": "parent-freeze-v1",
            "run_id": "parent-freeze",
            "initial_capture_attempt_count": 210,
            "maximum_total_capture_attempt_count": 260,
            "target_valid_market_count": 195,
            "minimum_collection_decision_ts": 1,
            "collector_contract": {"training_corpus_root": str(tmp_path / "training")},
            "collection_output_dir": str(tmp_path / "parent-collection"),
            "collection_batch_id_prefix": "parent",
            "precollection_freeze_id": "parent",
            **_safe_fields(),
        },
    )
    terminal_report = tmp_path / "terminal_report.json"
    _write(
        terminal_report,
        {
            "status": "TERMINAL_RECONCILIATION_READY",
            "source_capture_count": 260,
            "labels_or_outcomes_opened_for_reconciliation": False,
            **_safe_fields(),
        },
    )
    terminal_manifest = tmp_path / "terminal_manifest.json"
    _write(
        terminal_manifest,
        {
            "report": _descriptor(terminal_report),
            **_safe_fields(),
        },
    )
    selected_rows = tmp_path / "selected_rows.jsonl"
    selected_payload = [
        {
            "market_id": f"market-{index:03d}",
            "selection_rank": index,
        }
        for index in range(1, 192)
    ]
    selected_rows.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in selected_payload
        ),
        encoding="utf-8",
    )
    role_report = tmp_path / "role_report.json"
    _write(
        role_report,
        {
            "role_assignment_ready": False,
            "selected_market_count": 191,
            "labels_or_outcomes_opened_for_role_assignment": False,
            **_safe_fields(),
        },
    )
    role_manifest = tmp_path / "role_manifest.json"
    _write(
        role_manifest,
        {
            "report": _descriptor(role_report),
            "selected_rows": _descriptor(selected_rows),
            "selected_market_ids_sha256": canonical_json_sha256(
                sorted(row["market_id"] for row in selected_payload)
            ),
            "labels_or_outcomes_opened_for_role_assignment": False,
            **_safe_fields(),
        },
    )
    support_report = tmp_path / "support_report.json"
    _write(
        support_report,
        {
            "status": "BLOCKED_INSUFFICIENT_SUPPORT_AT_FROZEN_MAXIMUM",
            "attempted_capture_count": 260,
            "selected_market_count": 191,
            "continuation_allowed": False,
            "labels_or_outcomes_opened_for_continuation": False,
            "settlement_pnl_opened_for_continuation": False,
            **_safe_fields(),
        },
    )
    support_manifest = tmp_path / "support_manifest.json"
    _write(
        support_manifest,
        {
            "support_gate_report": _descriptor(support_report),
            "role_assignment_report": _descriptor(role_report),
            "role_assignment_manifest": _descriptor(role_manifest),
            "batch_progress_inputs": [
                _descriptor(path) for path in batches
            ],
            **_safe_fields(),
        },
    )
    return {
        "parent_freeze": parent_freeze,
        "terminal_report": terminal_report,
        "terminal_manifest": terminal_manifest,
        "support_report": support_report,
        "support_manifest": support_manifest,
        "source_paths": [
            parent_freeze,
            terminal_report,
            terminal_manifest,
            support_report,
            support_manifest,
            role_report,
            role_manifest,
            selected_rows,
            *batches,
        ],
    }


def _supplemental_batch(
    tmp_path: Path,
    freeze: dict,
    *,
    capture_count: int = 20,
    scheduled_start: int | None = None,
) -> Path:
    first_ts = scheduled_start or int(
        freeze["supplemental_minimum_collection_decision_ts"]
    )
    captures = [
        {
            "run_id": f"supplemental-run-{index:02d}",
            "scheduled_round_start_ts": first_ts + index * 300_000,
        }
        for index in range(capture_count)
    ]
    path = tmp_path / f"supplemental_batch_{capture_count}.json"
    _write(
        path,
        {
            "batch_id": "supplemental-batch-01",
            "capture_count": capture_count,
            "captures": captures,
            "error_count": 0,
            "errors": [],
            **_safe_fields(),
        },
    )
    return path


def _fake_core_result(
    tmp_path: Path,
    *,
    selected_count: int,
    ready: bool,
) -> dict:
    suffix = "ready" if ready else "blocked"
    report_path = tmp_path / f"core_report_{suffix}.json"
    manifest_path = tmp_path / f"core_manifest_{suffix}.json"
    _write(report_path, {"ready": ready})
    _write(manifest_path, {"ready": ready})
    role_counts = {
        "confirmatory_validation": 60 if ready else 59,
        "development_calibration": 45,
        "development_train": 90,
    }
    return {
        "report_path": report_path,
        "manifest_path": manifest_path,
        "report": {
            "status": (
                "OUTCOME_BLIND_SUPPORT_TARGET_READY"
                if ready
                else "BLOCKED_INSUFFICIENT_SUPPORT_AT_FROZEN_MAXIMUM"
            ),
            "selected_market_count": selected_count,
            "continuation_allowed": False,
            "blocking_reason_codes": (
                [] if ready else ["insufficient_support_at_frozen_maximum"]
            ),
        },
        "role_assignment_result": {
            "report": {"role_market_counts": role_counts}
        },
    }


def _safe_fields() -> dict:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
