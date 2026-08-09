"""Additive frozen-artifact binding audit for the terminal residual v3 lineage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.cost_aware_residual import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _verified_json,
    _verify_descriptor,
)
from bigan.v8.polymarket.cost_aware_residual_v3 import (
    DEFAULT_CONFIG_DIR,
    _build_report,
    render_v3_markdown,
)
from bigan.v8.polymarket.cost_aware_residual_v3_logit import (
    _build_logit_report,
    render_logit_challenger_markdown,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import _write_new_frozen_json
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.moe_terminal_diagnostic import _assert_semantically_equal
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT
from bigan.v8.polymarket.residual_v3_terminal_review import (
    verify_residual_v3_terminal_review,
)

SCHEMA_VERSION = "bigan-btc-15m-residual-v3-frozen-binding-audit-v1"
CREATED_AT = "2026-08-09T15:45:49Z"
DEFAULT_AUDIT_PATH = (
    DEFAULT_CONFIG_DIR / "residual_v3_frozen_artifact_binding_audit.json"
)

PRIMARY_PROTOCOL_PATH = DEFAULT_CONFIG_DIR / "residual_v3_primary_slot_001_protocol.json"
PRIMARY_RUN_DIR = DEFAULT_CONFIG_DIR / "residual_v3_primary_slot_001_oof"
PRIMARY_MANIFEST_PATH = PRIMARY_RUN_DIR / "residual_v3_oof_manifest.json"
PRIMARY_EXPECTED_IMPLEMENTATION = {
    "path": "src/bigan/v8/polymarket/cost_aware_residual_v3.py",
    "sha256": "194be7580988384d2b8e7963f2bcc09531a51154db96a69691d56e40968837f2",
}
PRIMARY_EXPECTED_PROTOCOL_SHA256 = (
    "49550a4f924c82060aa3e56fd1dbe6f691c96ad83c39b431fe0be86f7df9b3db"
)

CHALLENGER_PROTOCOL_PATH = (
    DEFAULT_CONFIG_DIR / "residual_v3_challenger_slot_002_protocol.json"
)
CHALLENGER_RUN_DIR = DEFAULT_CONFIG_DIR / "residual_v3_challenger_slot_002_oof"
CHALLENGER_MANIFEST_PATH = (
    CHALLENGER_RUN_DIR / "residual_v3_logit_oof_manifest.json"
)
CHALLENGER_EXPECTED_IMPLEMENTATION = {
    "path": "src/bigan/v8/polymarket/cost_aware_residual_v3_logit.py",
    "sha256": "d9d0128a24acf16d483301e579e70fba3788e8056901343f68706fa28da76e1d",
}
CHALLENGER_EXPECTED_PROTOCOL_SHA256 = (
    "acaa85b576a90a9ae20e39a53800dcef858240710312b32187d180392ac646a8"
)

GATE_EXPECTED_IMPLEMENTATION = {
    "path": "src/bigan/v8/polymarket/cost_aware_residual.py",
    "sha256": "491a329f708a16d5aecdd952552cbff3fa13d8f7446bfe3e0c78fade3b36f78c",
}

FROZEN_SIDECAR_TARGETS = (
    DEFAULT_CONFIG_DIR / "development_data_registry.json",
    DEFAULT_CONFIG_DIR / "lineage_authorization.json",
    PRIMARY_PROTOCOL_PATH,
    PRIMARY_RUN_DIR / "residual_v3_development_dataset_rows.jsonl",
    PRIMARY_RUN_DIR / "residual_v3_oof_predictions.jsonl",
    PRIMARY_RUN_DIR / "residual_v3_oof_fold_audits.jsonl",
    PRIMARY_RUN_DIR / "residual_v3_oof_market_results.jsonl",
    PRIMARY_RUN_DIR / "residual_v3_oof_report.json",
    PRIMARY_RUN_DIR / "residual_v3_oof_report.md",
    PRIMARY_MANIFEST_PATH,
    CHALLENGER_PROTOCOL_PATH,
    CHALLENGER_RUN_DIR / "residual_v3_logit_development_dataset_rows.jsonl",
    CHALLENGER_RUN_DIR / "residual_v3_logit_oof_predictions.jsonl",
    CHALLENGER_RUN_DIR / "residual_v3_logit_oof_fold_audits.jsonl",
    CHALLENGER_RUN_DIR / "residual_v3_logit_oof_market_results.jsonl",
    CHALLENGER_RUN_DIR / "residual_v3_logit_oof_report.json",
    CHALLENGER_RUN_DIR / "residual_v3_logit_oof_report.md",
    CHALLENGER_MANIFEST_PATH,
    DEFAULT_CONFIG_DIR / "residual_v3_development_terminal_review.json",
    DEFAULT_CONFIG_DIR / "residual_v3_development_terminal_review.md",
)


def require_executing_module_descriptor(
    protocol: Mapping[str, Any],
    *,
    executing_module_path: Path | str,
    expected_sha256: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, str]:
    """Bind a protocol descriptor to the exact module its runner executes."""

    root = Path(repository_root).resolve()
    module = Path(executing_module_path).resolve()
    if not module.is_relative_to(root) or not module.is_file():
        raise ValueError("candidate executing module must be repository-local")
    if sha256_file(module) != expected_sha256:
        raise ValueError("candidate executing module SHA does not match frozen SHA")
    expected = {
        "path": module.relative_to(root).as_posix(),
        "sha256": expected_sha256,
    }
    try:
        declared = dict(protocol["inputs"]["candidate_implementation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate implementation descriptor unavailable") from exc
    if declared != expected:
        raise ValueError(
            "candidate implementation descriptor does not identify executing module"
        )
    resolved = _verify_descriptor(declared, repository_root=root)
    if resolved != module:
        raise ValueError("candidate implementation resolved to another module")
    return expected


def build_residual_v3_frozen_binding_audit(
    *, repository_root: Path | str = REPO_ROOT
) -> dict[str, Any]:
    """Rebuild both reports and verify exact implementation and sidecar bindings."""

    root = Path(repository_root).resolve()
    primary = _audit_candidate(
        role="primary",
        protocol_path=PRIMARY_PROTOCOL_PATH,
        manifest_path=PRIMARY_MANIFEST_PATH,
        expected_protocol_sha256=PRIMARY_EXPECTED_PROTOCOL_SHA256,
        expected_implementation=PRIMARY_EXPECTED_IMPLEMENTATION,
        executing_module_path=root / PRIMARY_EXPECTED_IMPLEMENTATION["path"],
        report_builder=_build_report,
        markdown_renderer=render_v3_markdown,
        repository_root=root,
    )
    challenger = _audit_candidate(
        role="challenger",
        protocol_path=CHALLENGER_PROTOCOL_PATH,
        manifest_path=CHALLENGER_MANIFEST_PATH,
        expected_protocol_sha256=CHALLENGER_EXPECTED_PROTOCOL_SHA256,
        expected_implementation=CHALLENGER_EXPECTED_IMPLEMENTATION,
        executing_module_path=root / CHALLENGER_EXPECTED_IMPLEMENTATION["path"],
        report_builder=_build_logit_report,
        markdown_renderer=render_logit_challenger_markdown,
        repository_root=root,
    )
    sidecars = _audit_frozen_sidecars(root)
    terminal = verify_residual_v3_terminal_review(repository_root=root)
    audit_implementation = _descriptor(Path(__file__), root)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": CREATED_AT,
        "lineage_id": "BTC-15M-cost-aware-market-residual-v3",
        "role": "additive_post_freeze_binding_and_reconciliation_audit",
        "audit_passed": True,
        "review_id": 4891741652,
        "review_url": (
            "https://github.com/phead198708/BiGan/pull/268"
            "#pullrequestreview-4891741652"
        ),
        "frozen_bytes_changed": {
            "candidate_implementations": False,
            "protocols": False,
            "oof_artifacts": False,
            "terminal_review": False,
        },
        "candidate_binding": {
            "primary": primary,
            "challenger": challenger,
        },
        "sidecar_audit": sidecars,
        "terminal_review_reconciliation": {
            "verification_passed": terminal["verification_passed"],
            "review_sha256": terminal["review_sha256"],
            "candidate_freeze_allowed": terminal["candidate_freeze_allowed"],
            "live_shadow_start_allowed": terminal["live_shadow_start_allowed"],
            "fresh_collection_authorized": terminal["fresh_collection_authorized"],
        },
        "future_lineage_validator_requirement": {
            "mandatory_before_preregistration_and_evaluation": True,
            "guard": "require_executing_module_descriptor",
            "exact_repository_relative_path_required": True,
            "exact_sha256_required": True,
            "valid_but_unrelated_repository_file_must_be_rejected": True,
            "regression_test_required": True,
        },
        "audit_implementation": audit_implementation,
        "state": {
            "candidate_selected": None,
            "candidate_freeze_allowed": False,
            "live_shadow_start_allowed": False,
            "fresh_collection_authorized": False,
            "promotion_authorized": False,
        },
        "safety": dict(SAFETY),
    }


def generate_residual_v3_frozen_binding_audit(
    *,
    output_path: Path | str = DEFAULT_AUDIT_PATH,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Write the additive audit once, with a SHA sidecar."""

    root = Path(repository_root).resolve()
    output = Path(output_path).resolve()
    if output.exists() or output.with_suffix(".sha256").exists():
        raise FileExistsError("residual v3 frozen binding audit already exists")
    report = build_residual_v3_frozen_binding_audit(repository_root=root)
    artifact = _write_new_frozen_json(output, report)
    return {
        "audit": _descriptor(Path(artifact["path"]), root),
        "audit_passed": True,
        "safety": dict(SAFETY),
    }


def verify_residual_v3_frozen_binding_audit(
    *,
    audit_path: Path | str = DEFAULT_AUDIT_PATH,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify the audit sidecar and reproduce every audit conclusion."""

    root = Path(repository_root).resolve()
    audit_file = Path(audit_path).resolve()
    frozen = _verified_json(audit_file)
    rebuilt = build_residual_v3_frozen_binding_audit(repository_root=root)
    _assert_semantically_equal(rebuilt, frozen, path="residual_v3_frozen_binding_audit")
    return {
        "verification_passed": True,
        "audit_passed": True,
        "audit_sha256": sha256_file(audit_file),
        "primary_report_rebuilt": True,
        "challenger_report_rebuilt": True,
        "frozen_sidecar_count": len(FROZEN_SIDECAR_TARGETS),
        "safety": dict(SAFETY),
    }


def _audit_candidate(
    *,
    role: str,
    protocol_path: Path,
    manifest_path: Path,
    expected_protocol_sha256: str,
    expected_implementation: Mapping[str, str],
    executing_module_path: Path,
    report_builder: Any,
    markdown_renderer: Any,
    repository_root: Path,
) -> dict[str, Any]:
    protocol_file = protocol_path.resolve()
    manifest_file = manifest_path.resolve()
    protocol = _verified_json(protocol_file)
    manifest = _verified_json(manifest_file)
    if sha256_file(protocol_file) != expected_protocol_sha256:
        raise ValueError(f"{role} protocol SHA is not the frozen expected SHA")
    exact = require_executing_module_descriptor(
        protocol,
        executing_module_path=executing_module_path,
        expected_sha256=str(expected_implementation["sha256"]),
        repository_root=repository_root,
    )
    if exact != dict(expected_implementation):
        raise ValueError(f"{role} exact implementation descriptor mismatch")
    if dict(manifest.get("candidate_implementation") or {}) != exact:
        raise ValueError(f"{role} manifest implementation binding mismatch")
    if dict(manifest.get("protocol") or {}) != _descriptor(
        protocol_file, repository_root
    ):
        raise ValueError(f"{role} manifest protocol binding mismatch")
    if dict(protocol["inputs"]["gate_implementation"]) != GATE_EXPECTED_IMPLEMENTATION:
        raise ValueError(f"{role} gate implementation binding mismatch")
    if dict(manifest.get("immutable_gate_implementation") or {}) != (
        GATE_EXPECTED_IMPLEMENTATION
    ):
        raise ValueError(f"{role} manifest gate binding mismatch")
    artifacts = {
        name: _verify_descriptor(dict(descriptor), repository_root=repository_root)
        for name, descriptor in dict(manifest.get("artifacts") or {}).items()
    }
    if set(artifacts) != {
        "dataset_rows",
        "predictions",
        "fold_audits",
        "market_results",
        "report",
        "report_markdown",
    }:
        raise ValueError(f"{role} manifest artifact set mismatch")
    markets = _load_jsonl(artifacts["market_results"])
    folds = _load_jsonl(artifacts["fold_audits"])
    if len(markets) != 600 or len(folds) != 6:
        raise ValueError(f"{role} OOF report rebuild population mismatch")
    rebuilt = report_builder(
        protocol=protocol,
        protocol_sha256=expected_protocol_sha256,
        source_commit=str(manifest["source_commit"]),
        market_results=markets,
        fold_audits=folds,
    )
    frozen = _load_json(artifacts["report"])
    _assert_semantically_equal(rebuilt, frozen, path=f"{role}_oof_report")
    rendered = markdown_renderer(rebuilt)
    if rendered != artifacts["report_markdown"].read_text(encoding="utf-8"):
        raise ValueError(f"{role} report Markdown does not reproduce")
    return {
        "exact_candidate_implementation_descriptor": exact,
        "executing_module_descriptor_match": True,
        "protocol_descriptor_match": True,
        "manifest_descriptor_match": True,
        "gate_descriptor_match": True,
        "market_result_count": len(markets),
        "fold_audit_count": len(folds),
        "report_rebuilt_independently": True,
        "report_semantic_sha256": canonical_json_sha256(rebuilt),
        "frozen_report_sha256": sha256_file(artifacts["report"]),
        "report_markdown_rebuilt": True,
        "all_gates_passed": bool(frozen["all_gates_passed"]),
        "candidate_freeze_allowed": bool(frozen["candidate_freeze_allowed"]),
        "safety": dict(SAFETY),
    }


def _audit_frozen_sidecars(repository_root: Path) -> dict[str, Any]:
    entries = []
    expected_sidecars = set()
    for target in FROZEN_SIDECAR_TARGETS:
        resolved = target.resolve()
        if not resolved.is_relative_to(repository_root) or not resolved.is_file():
            raise ValueError("frozen sidecar target unavailable")
        sidecar = (
            Path(str(resolved) + ".sha256")
            if resolved.suffix == ".md"
            else resolved.with_suffix(".sha256")
        )
        expected_sidecars.add(sidecar)
        if not sidecar.is_file():
            raise ValueError("frozen sidecar unavailable")
        expected = sidecar.read_text(encoding="utf-8").strip()
        actual = sha256_file(resolved)
        if expected != actual:
            raise ValueError("frozen sidecar SHA mismatch")
        entries.append(
            {
                "target_path": resolved.relative_to(repository_root).as_posix(),
                "sidecar_path": sidecar.relative_to(repository_root).as_posix(),
                "sha256": actual,
            }
        )
    discovered = set(DEFAULT_CONFIG_DIR.rglob("*.sha256"))
    allowed_additive = {DEFAULT_AUDIT_PATH.with_suffix(".sha256").resolve()}
    if discovered - expected_sidecars - allowed_additive:
        raise ValueError("unexpected residual v3 sidecar outside frozen inventory")
    if len(expected_sidecars) != 20:
        raise ValueError("frozen residual v3 sidecar inventory is not exactly 20")
    return {
        "frozen_sidecar_count": len(entries),
        "all_sidecars_verified": True,
        "inventory_sha256": canonical_json_sha256(entries),
        "entries": entries,
    }


__all__ = [
    "build_residual_v3_frozen_binding_audit",
    "generate_residual_v3_frozen_binding_audit",
    "require_executing_module_descriptor",
    "verify_residual_v3_frozen_binding_audit",
]
