"""Cross-platform semantic verification for frozen residual OOF reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.cost_aware_residual import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROTOCOL,
    MANIFEST_SCHEMA_VERSION,
    _descriptor,
    _load_json,
    _load_jsonl,
    _validate_frozen_population,
    _verified_json,
    _verify_descriptor,
    build_residual_oof_report,
    render_residual_oof_markdown,
    validate_residual_oof_protocol,
)
from bigan.v8.polymarket.cost_aware_residual_quantile import (
    DEFAULT_CHALLENGER_OUTPUT_DIR,
    DEFAULT_CHALLENGER_PROTOCOL,
    render_quantile_challenger_markdown,
    validate_quantile_challenger_protocol,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.moe_terminal_diagnostic import _assert_semantically_equal
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT


def verify_primary_oof_cross_platform(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify the primary report without altering its hash-pinned implementation."""

    return _verify(
        protocol_path=protocol_path,
        output_dir=output_dir,
        repository_root=repository_root,
        challenger=False,
    )


def verify_challenger_oof_cross_platform(
    *,
    protocol_path: Path | str = DEFAULT_CHALLENGER_PROTOCOL,
    output_dir: Path | str = DEFAULT_CHALLENGER_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify the challenger report without altering its hash-pinned implementation."""

    return _verify(
        protocol_path=protocol_path,
        output_dir=output_dir,
        repository_root=repository_root,
        challenger=True,
    )


def _verify(
    *,
    protocol_path: Path | str,
    output_dir: Path | str,
    repository_root: Path | str,
    challenger: bool,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    protocol = _verified_json(protocol_file)
    if challenger:
        validate_quantile_challenger_protocol(protocol, repository_root=root)
    else:
        validate_residual_oof_protocol(protocol, repository_root=root)
    manifest = _verified_json(output / "residual_oof_manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("residual cross-platform manifest schema mismatch")
    if manifest.get("protocol") != _descriptor(protocol_file, root):
        raise ValueError("residual cross-platform protocol binding mismatch")
    artifact_paths = {
        name: _verify_descriptor(descriptor, repository_root=root)
        for name, descriptor in dict(manifest["artifacts"]).items()
    }
    predictions = _load_jsonl(artifact_paths["predictions"])
    folds = _load_jsonl(artifact_paths["fold_audits"])
    markets = _load_jsonl(artifact_paths["market_results"])
    _validate_frozen_population(
        predictions=predictions,
        fold_audits=folds,
        market_results=markets,
        protocol=protocol,
    )
    rebuilt = build_residual_oof_report(
        protocol=protocol,
        protocol_sha256=sha256_file(protocol_file),
        source_commit=str(manifest["source_commit"]),
        market_results=markets,
        fold_audits=folds,
    )
    if challenger:
        rebuilt["candidate_budget_exhausted"] = True
        rebuilt["additional_candidate_allowed"] = False
        rebuilt["structural_change"] = dict(protocol["structural_change"])
    frozen = _load_json(artifact_paths["report"])
    _assert_semantically_equal(
        rebuilt,
        frozen,
        path="residual_oof_report",
    )
    expected_markdown = (
        render_quantile_challenger_markdown(rebuilt)
        if challenger
        else render_residual_oof_markdown(rebuilt)
    )
    if expected_markdown != artifact_paths["report_markdown"].read_text(
        encoding="utf-8"
    ):
        raise ValueError("residual cross-platform markdown does not reproduce")
    return {
        "verification_passed": True,
        "cross_platform_float_tolerance": {
            "relative": 1e-12,
            "absolute": 1e-12,
            "non_numeric_fields_require_exact_equality": True,
        },
        "all_gates_passed": bool(frozen["all_gates_passed"]),
        "failed_gates": list(frozen["failed_gates"]),
        "candidate_budget_exhausted": challenger,
        "oof_market_count": len(markets),
        "manifest_sha256": sha256_file(output / "residual_oof_manifest.json"),
        "safety": dict(SAFETY),
    }


__all__ = [
    "verify_challenger_oof_cross_platform",
    "verify_primary_oof_cross_platform",
]
