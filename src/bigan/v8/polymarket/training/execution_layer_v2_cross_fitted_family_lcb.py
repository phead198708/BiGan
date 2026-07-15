"""Precollection freeze contract for the #172 cross-fitted family LCB candidate."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields

SCHEMA_PREFIX = "bigan-v8-execution-layer-v2-cross-fitted-family-lcb"
PROTOCOL_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-protocol-v1"
FORBIDDEN_REGISTRY_FIELDS = {
    "accepted_bet_net_pnl",
    "evaluation_target_net_pnl_per_contract_by_action",
    "evaluation_target_net_return_after_cost_by_action",
    "future_return",
    "gross_pnl",
    "net_pnl",
    "oracle_action",
    "realized_pnl",
    "resolved_outcome",
    "settlement_pnl",
    "settlement_return",
    "target_net_return_after_cost",
    "total_net_pnl_per_notional",
}


@dataclass(frozen=True, slots=True)
class CrossFittedFamilyLCBPrecollectionFreezeConfig:
    """Hash-pinned inputs for freezing collection roles before data arrives."""

    run_id: str
    output_dir: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    git_commit: str
    prior_market_registry_pins: tuple[tuple[Path | str, str], ...]
    prior_evidence_artifact_pins: tuple[tuple[Path | str, str], ...]
    expected_prior_unique_market_count: int = 95

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_protocol_sha256, name="protocol SHA-256")
        if len(self.git_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.git_commit.lower()
        ):
            raise ValueError("git_commit must be a 40-character hex digest")
        if self.expected_prior_unique_market_count < 1:
            raise ValueError("expected_prior_unique_market_count must be positive")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "protocol_path", Path(self.protocol_path))
        object.__setattr__(
            self,
            "prior_market_registry_pins",
            _normalize_pins(self.prior_market_registry_pins, name="market registry"),
        )
        object.__setattr__(
            self,
            "prior_evidence_artifact_pins",
            _normalize_pins(self.prior_evidence_artifact_pins, name="prior evidence"),
        )


def validate_cross_fitted_family_lcb_protocol(protocol: dict[str, Any]) -> None:
    """Fail closed on any drift in the precollection protocol."""

    roles = dict(protocol.get("role_assignment") or {})
    collector = dict(protocol.get("collector_contract") or {})
    cross_fit = dict(protocol.get("cross_fit_protocol") or {})
    conformal = dict(protocol.get("conformal_lcb_protocol") or {})
    confirmatory = dict(protocol.get("confirmatory_validation_gates") or {})
    safety = dict(protocol.get("safety") or {})
    role_total = sum(
        int(roles.get(name) or 0)
        for name in (
            "development_train_market_count",
            "development_calibration_market_count",
            "confirmatory_validation_market_count",
        )
    )
    checks = {
        "schema_version": protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION,
        "candidate_name": protocol.get("candidate_name")
        == "market_grouped_cross_fitted_family_lcb_v1",
        "frozen": protocol.get("frozen") is True,
        "decision_time_safe": protocol.get("decision_time_safe") is True,
        "no_prior_validation_tuning": protocol.get(
            "uses_prior_validation_or_future_labels_for_tuning"
        )
        is False,
        "role_method": roles.get("method")
        == "earliest_capture_quality_valid_unique_markets_chronological_v1",
        "role_total": role_total == int(roles.get("target_valid_market_count") or 0)
        == 90,
        "outcome_blind_roles": roles.get("outcome_blind_role_assignment") is True,
        "bounded_collection": int(roles.get("initial_capture_attempt_count") or 0)
        >= 90
        and int(roles.get("maximum_total_capture_attempt_count") or 0)
        >= int(roles.get("initial_capture_attempt_count") or 0),
        "ws_first": collector.get("orderbook_source_priority")
        == "clob_websocket_primary_rest_fallback",
        "full_round_ws_collection_window": float(
            collector.get("public_provider_timeout_seconds") or 0.0
        )
        >= 300.0
        and float(collector.get("public_provider_timeout_seconds") or 0.0)
        > float(collector.get("public_provider_http_timeout_seconds") or 0.0),
        "external_training_root": collector.get("training_corpus_root")
        == "/Volumes/PHILIPS/v8",
        "raw_evidence": collector.get("per_round_raw_evidence_required") is True,
        "async_settlement": collector.get("asynchronous_settlement_required") is True,
        "cross_fit": int(cross_fit.get("fold_count") or 0) == 5
        and cross_fit.get("group_key") == "market_id"
        and cross_fit.get("fit_split") == "development_train_only",
        "deterministic_model": cross_fit.get("objective") == "reg:squarederror"
        and cross_fit.get("nthread") == 1
        and isinstance(cross_fit.get("seed"), int),
        "calibration_only_lcb": conformal.get("source_split")
        == "development_calibration_only"
        and conformal.get("affine_calibration_enabled") is False
        and 0.5 < float(conformal.get("one_sided_quantile") or 0.0) < 1.0,
        "confirmatory_support": int(
            confirmatory.get("required_unique_market_count") or 0
        )
        == 30
        and int(confirmatory.get("minimum_accepted_bet_count") or 0) >= 15,
        "safety": safety.get("paper_only") is True
        and safety.get("capital_at_risk") is False
        and safety.get("polymarket_write_enabled") is False
        and safety.get("wallet_signing_enabled") is False
        and safety.get("source_model_candidate_eligible") is False
        and safety.get("freeze_ready") is False
        and safety.get("promotion_evidence_eligible") is False
        and safety.get("v8_execution_handoff_allowed") is False
        and safety.get("#134_resume_allowed") is False
        and safety.get("#146_start_allowed") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid cross-fitted family LCB protocol: " + ", ".join(failed))


def freeze_cross_fitted_family_lcb_precollection(
    config: CrossFittedFamilyLCBPrecollectionFreezeConfig,
) -> dict[str, Any]:
    """Freeze data roles, exclusions, and collector/model contracts before collection."""

    protocol_path = config.protocol_path.resolve()
    _verify_pin(protocol_path, config.expected_protocol_sha256, name="protocol")
    protocol = _load_json(protocol_path)
    validate_cross_fitted_family_lcb_protocol(protocol)

    registry_descriptors = []
    prior_market_ids: set[str] = set()
    prior_decision_timestamps: list[int] = []
    for path, expected_sha256 in config.prior_market_registry_pins:
        resolved = path.resolve()
        _verify_pin(resolved, expected_sha256, name="prior market registry")
        payload = _load_json_or_jsonl(resolved)
        forbidden = sorted(_find_fields(payload, FORBIDDEN_REGISTRY_FIELDS))
        if forbidden:
            raise ValueError(
                "prior market registry contains forbidden outcome fields: "
                + ", ".join(forbidden)
            )
        prior_market_ids.update(_extract_market_ids(payload))
        prior_decision_timestamps.extend(_extract_decision_timestamps(payload))
        registry_descriptors.append(_descriptor(resolved))
    if "" in prior_market_ids or len(prior_market_ids) != config.expected_prior_unique_market_count:
        raise ValueError(
            "prior unique market count mismatch: "
            f"expected {config.expected_prior_unique_market_count}, got {len(prior_market_ids)}"
        )
    if not prior_decision_timestamps or any(value <= 0 for value in prior_decision_timestamps):
        raise ValueError("prior decision-time registry is incomplete")

    evidence_descriptors = []
    for path, expected_sha256 in config.prior_evidence_artifact_pins:
        resolved = path.resolve()
        _verify_pin(resolved, expected_sha256, name="prior evidence artifact")
        evidence_descriptors.append(_descriptor(resolved))

    created_ts = int(time.time() * 1000)
    max_prior_decision_ts = max(prior_decision_timestamps)
    roles = dict(protocol["role_assignment"])
    collector = dict(protocol["collector_contract"])
    output_dir = config.output_dir / config.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    exclusion_registry = {
        "schema_version": f"{SCHEMA_PREFIX}-prior-exclusion-registry-v1",
        "run_id": config.run_id,
        "prior_market_registry_sources": registry_descriptors,
        "prior_evidence_artifacts": evidence_descriptors,
        "prior_unique_market_count": len(prior_market_ids),
        "prior_market_ids": sorted(prior_market_ids),
        "prior_market_ids_sha256": canonical_json_sha256(sorted(prior_market_ids)),
        "maximum_prior_decision_ts": max_prior_decision_ts,
        "prior_outcome_or_pnl_values_loaded": False,
        "prior_validation_or_future_evidence_used_for_tuning": False,
        **_blocked_safety_fields(),
    }
    exclusion_registry["exclusion_registry_id"] = canonical_json_sha256(
        exclusion_registry
    )
    exclusion_path = output_dir / "prior_evidence_exclusion_registry.json"
    _write_json(exclusion_path, exclusion_registry)

    role_plan = [
        {
            "role": "development_train",
            "valid_market_rank_start": 1,
            "valid_market_rank_end": int(roles["development_train_market_count"]),
        },
        {
            "role": "development_calibration",
            "valid_market_rank_start": int(roles["development_train_market_count"]) + 1,
            "valid_market_rank_end": int(roles["development_train_market_count"])
            + int(roles["development_calibration_market_count"]),
        },
        {
            "role": "confirmatory_validation",
            "valid_market_rank_start": int(roles["development_train_market_count"])
            + int(roles["development_calibration_market_count"])
            + 1,
            "valid_market_rank_end": int(roles["target_valid_market_count"]),
        },
    ]
    batch_id_prefix = f"issue172-{config.run_id}"
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-precollection-role-freeze-v1",
        "run_id": config.run_id,
        "freeze_created_ts": created_ts,
        "git_commit": config.git_commit.lower(),
        "protocol": _descriptor(protocol_path),
        "prior_evidence_exclusion_registry": _descriptor(exclusion_path),
        "candidate_name": protocol["candidate_name"],
        "role_assignment_method": roles["method"],
        "role_plan": role_plan,
        "target_valid_market_count": int(roles["target_valid_market_count"]),
        "initial_capture_attempt_count": int(roles["initial_capture_attempt_count"]),
        "maximum_total_capture_attempt_count": int(
            roles["maximum_total_capture_attempt_count"]
        ),
        "collection_batch_id_prefix": batch_id_prefix,
        "collection_output_dir": str((output_dir / "collection").resolve()),
        "collector_contract": collector,
        "collector_contract_sha256": canonical_json_sha256(collector),
        "cross_fit_protocol": protocol["cross_fit_protocol"],
        "cross_fit_protocol_sha256": canonical_json_sha256(
            protocol["cross_fit_protocol"]
        ),
        "conformal_lcb_protocol": protocol["conformal_lcb_protocol"],
        "conformal_lcb_protocol_sha256": canonical_json_sha256(
            protocol["conformal_lcb_protocol"]
        ),
        "frozen_execution_contract": protocol["frozen_execution_contract"],
        "frozen_execution_contract_sha256": canonical_json_sha256(
            protocol["frozen_execution_contract"]
        ),
        "minimum_collection_decision_ts": max(max_prior_decision_ts + 1, created_ts + 1),
        "collection_must_be_strictly_later": True,
        "new_market_ids_must_be_disjoint": True,
        "role_assignment_outcome_blind": True,
        "settlement_labels_available_only_after_round_close": True,
        "collection_started": False,
        "model_fit_started": False,
        "confirmatory_validation_started": False,
        "future_holdout_started": False,
        **_blocked_safety_fields(),
    }
    manifest["precollection_freeze_id"] = canonical_json_sha256(manifest)
    manifest_path = output_dir / "precollection_role_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    markdown_path = output_dir / "precollection_role_freeze_manifest.md"
    _write_text(markdown_path, _freeze_markdown(manifest))
    descriptor = {
        "schema_version": f"{SCHEMA_PREFIX}-precollection-role-freeze-descriptor-v1",
        "manifest": _descriptor(manifest_path),
        "markdown": _descriptor(markdown_path),
        "precollection_freeze_id": manifest["precollection_freeze_id"],
        "collection_started": False,
        **_blocked_safety_fields(),
    }
    descriptor_path = output_dir / "precollection_role_freeze_descriptor.json"
    _write_json(descriptor_path, descriptor)
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "descriptor_path": descriptor_path,
        "descriptor_sha256": _sha256_file(descriptor_path),
        "manifest": manifest,
    }


def _extract_market_ids(payload: Any) -> set[str]:
    market_ids: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "market_id" and isinstance(value, str):
                market_ids.add(value)
            elif key == "market_ids" and isinstance(value, list):
                market_ids.update(str(item) for item in value)
            else:
                market_ids.update(_extract_market_ids(value))
    elif isinstance(payload, list):
        for value in payload:
            market_ids.update(_extract_market_ids(value))
    return market_ids


def _extract_decision_timestamps(payload: Any) -> list[int]:
    timestamps: list[int] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {
                "decision_ts",
                "minimum_decision_ts",
                "maximum_decision_ts",
                "maximum_prior_decision_ts",
            } and isinstance(value, (int, float)):
                timestamps.append(int(value))
            else:
                timestamps.extend(_extract_decision_timestamps(value))
    elif isinstance(payload, list):
        for value in payload:
            timestamps.extend(_extract_decision_timestamps(value))
    return timestamps


def _find_fields(payload: Any, forbidden: set[str], prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in forbidden:
                found.add(path)
            found.update(_find_fields(value, forbidden, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.update(_find_fields(value, forbidden, f"{prefix}[{index}]"))
    return found


def _normalize_pins(
    pins: tuple[tuple[Path | str, str], ...],
    *,
    name: str,
) -> tuple[tuple[Path, str], ...]:
    if not pins:
        raise ValueError(f"at least one {name} pin is required")
    normalized = []
    for path, digest in pins:
        _require_sha256(digest, name=f"{name} SHA-256")
        normalized.append((Path(path), digest.lower()))
    return tuple(normalized)


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }


def _freeze_markdown(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #172 Precollection Role Freeze",
            "",
            f"- candidate: `{manifest['candidate_name']}`",
            f"- target valid markets: `{manifest['target_valid_market_count']}`",
            "- roles: `40 train / 20 calibration / 30 confirmatory`",
            f"- prior excluded markets: `{len(_load_json(Path(manifest['prior_evidence_exclusion_registry']['path']))['prior_market_ids'])}`",
            "- role assignment uses outcomes: `false`",
            "- model fit started: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _verify_pin(path: Path, expected_sha256: str, *, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    _require_sha256(expected_sha256, name=f"{name} SHA-256")
    if _sha256_file(path) != expected_sha256.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    return _load_json(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
