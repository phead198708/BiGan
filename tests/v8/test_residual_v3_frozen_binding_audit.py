from __future__ import annotations

from copy import deepcopy

import pytest

from bigan.v8.polymarket.cost_aware_residual import _load_json
from bigan.v8.polymarket.cost_aware_residual_v3 import DEFAULT_CONFIG_DIR
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT
from bigan.v8.polymarket.residual_v3_frozen_binding_audit import (
    CHALLENGER_EXPECTED_IMPLEMENTATION,
    CHALLENGER_PROTOCOL_PATH,
    GATE_EXPECTED_IMPLEMENTATION,
    PRIMARY_EXPECTED_IMPLEMENTATION,
    PRIMARY_PROTOCOL_PATH,
    build_residual_v3_frozen_binding_audit,
    require_executing_module_descriptor,
    verify_residual_v3_frozen_binding_audit,
)


def test_exact_committed_candidate_descriptors_bind_to_executing_modules() -> None:
    primary = _load_json(PRIMARY_PROTOCOL_PATH)
    challenger = _load_json(CHALLENGER_PROTOCOL_PATH)
    assert require_executing_module_descriptor(
        primary,
        executing_module_path=REPO_ROOT / PRIMARY_EXPECTED_IMPLEMENTATION["path"],
        expected_sha256=PRIMARY_EXPECTED_IMPLEMENTATION["sha256"],
    ) == dict(PRIMARY_EXPECTED_IMPLEMENTATION)
    assert require_executing_module_descriptor(
        challenger,
        executing_module_path=REPO_ROOT / CHALLENGER_EXPECTED_IMPLEMENTATION["path"],
        expected_sha256=CHALLENGER_EXPECTED_IMPLEMENTATION["sha256"],
    ) == dict(CHALLENGER_EXPECTED_IMPLEMENTATION)


@pytest.mark.parametrize(
    ("protocol_path", "expected_implementation"),
    [
        (PRIMARY_PROTOCOL_PATH, PRIMARY_EXPECTED_IMPLEMENTATION),
        (CHALLENGER_PROTOCOL_PATH, CHALLENGER_EXPECTED_IMPLEMENTATION),
    ],
)
def test_valid_unrelated_repository_file_swap_is_rejected(
    protocol_path, expected_implementation
) -> None:
    protocol = deepcopy(_load_json(protocol_path))
    protocol["inputs"]["candidate_implementation"] = dict(
        GATE_EXPECTED_IMPLEMENTATION
    )
    with pytest.raises(
        ValueError,
        match="does not identify executing module",
    ):
        require_executing_module_descriptor(
            protocol,
            executing_module_path=REPO_ROOT / expected_implementation["path"],
            expected_sha256=expected_implementation["sha256"],
        )


def test_future_lineage_guard_rejects_path_correct_but_sha_wrong() -> None:
    protocol = deepcopy(_load_json(PRIMARY_PROTOCOL_PATH))
    protocol["inputs"]["candidate_implementation"]["sha256"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="does not identify executing module",
    ):
        require_executing_module_descriptor(
            protocol,
            executing_module_path=REPO_ROOT / PRIMARY_EXPECTED_IMPLEMENTATION["path"],
            expected_sha256=PRIMARY_EXPECTED_IMPLEMENTATION["sha256"],
        )


def test_additive_audit_rebuilds_reports_and_verifies_frozen_inventory() -> None:
    before = {
        path.name: path.read_text(encoding="utf-8").strip()
        for path in DEFAULT_CONFIG_DIR.glob("*.sha256")
    }
    audit = build_residual_v3_frozen_binding_audit()
    after = {
        path.name: path.read_text(encoding="utf-8").strip()
        for path in DEFAULT_CONFIG_DIR.glob("*.sha256")
    }
    assert before == after
    assert audit["audit_passed"] is True
    assert audit["frozen_bytes_changed"] == {
        "candidate_implementations": False,
        "protocols": False,
        "oof_artifacts": False,
        "terminal_review": False,
    }
    assert audit["sidecar_audit"]["frozen_sidecar_count"] == 20
    assert audit["sidecar_audit"]["all_sidecars_verified"] is True
    for role in ("primary", "challenger"):
        candidate = audit["candidate_binding"][role]
        assert candidate["executing_module_descriptor_match"] is True
        assert candidate["protocol_descriptor_match"] is True
        assert candidate["manifest_descriptor_match"] is True
        assert candidate["market_result_count"] == 600
        assert candidate["fold_audit_count"] == 6
        assert candidate["report_rebuilt_independently"] is True
        assert candidate["report_markdown_rebuilt"] is True
        assert candidate["all_gates_passed"] is False
        assert candidate["candidate_freeze_allowed"] is False
        assert candidate["safety"] == SAFETY
    assert audit["terminal_review_reconciliation"]["verification_passed"] is True
    assert audit["future_lineage_validator_requirement"] == {
        "mandatory_before_preregistration_and_evaluation": True,
        "guard": "require_executing_module_descriptor",
        "exact_repository_relative_path_required": True,
        "exact_sha256_required": True,
        "valid_but_unrelated_repository_file_must_be_rejected": True,
        "regression_test_required": True,
    }
    assert all(value is False for value in audit["safety"].values())


def test_frozen_binding_audit_reproduces_and_keeps_all_permissions_false() -> None:
    result = verify_residual_v3_frozen_binding_audit()
    assert result["verification_passed"] is True
    assert result["audit_passed"] is True
    assert result["primary_report_rebuilt"] is True
    assert result["challenger_report_rebuilt"] is True
    assert result["frozen_sidecar_count"] == 20
    assert result["audit_sha256"] == (
        "7f1e336642cebdd4fba76a68671ba33cbcc3ed896e2281fb835be7e84721bb85"
    )
    assert all(value is False for value in result["safety"].values())
