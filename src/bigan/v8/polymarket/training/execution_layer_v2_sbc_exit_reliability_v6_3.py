"""Pre-target lineage freeze and label-support audit for #223 v6.3."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256

PROFILE_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-profile-v1"
LINEAGE_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-lineage-v1"
LINEAGE_ROWS_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-lineage-rows-v1"
LABEL_AUDIT_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-label-audit-v1"
FEATURE_AUDIT_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-feature-audit-v1"
AUDIT_MANIFEST_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-audit-manifest-v1"
CANDIDATE_NAME = "sbc_exit_reliability_v6_3"
ROLES = ("development_train", "development_calibration", "confirmatory_validation")
SIDES = ("UP", "DOWN")
SBC_ACTIONS = {
    "UP": "BUY_UP_SELL_BEFORE_CLOSE",
    "DOWN": "BUY_DOWN_SELL_BEFORE_CLOSE",
}


@dataclass(frozen=True, slots=True)
class SBCExitReliabilityV63AuditConfig:
    """Pinned two-stage #223 audit configuration."""

    stage: Literal["freeze_lineage", "audit_labels"]
    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    role_assignment_manifest_path: Path | str
    implementation_commit: str
    lineage_manifest_path: Path | str | None = None
    expected_lineage_manifest_sha256: str | None = None
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if self.stage not in {"freeze_lineage", "audit_labels"}:
            raise ValueError("stage must be freeze_lineage or audit_labels")
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_profile_sha256, "expected_profile_sha256")
        _require_git_sha(self.implementation_commit)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "profile_path", Path(self.profile_path))
        object.__setattr__(
            self, "role_assignment_manifest_path", Path(self.role_assignment_manifest_path)
        )
        if self.lineage_manifest_path is not None:
            object.__setattr__(self, "lineage_manifest_path", Path(self.lineage_manifest_path))
        if self.stage == "audit_labels":
            if self.lineage_manifest_path is None:
                raise ValueError("lineage_manifest_path is required for audit_labels")
            _require_sha256(
                str(self.expected_lineage_manifest_sha256 or ""),
                "expected_lineage_manifest_sha256",
            )


def validate_sbc_exit_reliability_v6_3_profile(profile: dict[str, Any]) -> None:
    """Reject any drift from the #223 preregistered audit contract."""

    source = dict(profile.get("source_contract") or {})
    label = dict(profile.get("label_contract") or {})
    features = dict(profile.get("feature_contract") or {})
    gates = dict(profile.get("audit_support_gates") or {})
    access = dict(profile.get("access_sequence") or {})
    lineage = dict(profile.get("source_lineage") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 223,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "lineage": set(lineage)
        == {
            "role_assignment_manifest_sha256",
            "role_assignment_rows_sha256",
            "v5_freeze_manifest_sha256",
        }
        and all(_is_sha256(str(value)) for value in lineage.values()),
        "source": _valid_source_contract(source),
        "label": label.get("target")
        == "later_executable_sell_before_close_exit_available"
        and label.get("positive_execution_class") == "realizable_sell_before_close"
        and label.get("future_intraround_orderbooks_allowed_in_label_stage_only") is True
        and label.get("fixed_terminal_bid_only_labels_allowed") is False
        and label.get("outcome_or_settlement_required_for_target") is False,
        "features": bool(features.get("common_required_features"))
        and bool(features.get("side_required_feature_suffixes"))
        and bool(features.get("prohibited_decision_input_tokens")),
        "gates": _valid_support_gates(gates),
        "access": access
        == {
            "lineage_manifest_must_be_written_before_label_content_access": True,
            "label_content_hashing_before_access_is_allowed": True,
            "audit_requires_exact_lineage_manifest_hash": True,
            "fit_allowed_only_after_audit_gate_passes": True,
            "schema_inspection_contamination_is_excluded": True,
        },
        "safety": profile.get("safety") == _blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#223 profile invalid: " + ", ".join(blockers))


def run_sbc_exit_reliability_v6_3_audit(
    config: SBCExitReliabilityV63AuditConfig,
) -> dict[str, Any]:
    """Freeze lineage without labels, or audit labels against that exact freeze."""

    profile_path = config.profile_path.resolve()
    role_manifest_path = config.role_assignment_manifest_path.resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#223 profile")
    profile = _load_json(profile_path)
    validate_sbc_exit_reliability_v6_3_profile(profile)
    expected_role_hash = str(profile["source_lineage"]["role_assignment_manifest_sha256"])
    _verify_pin(role_manifest_path, expected_role_hash, "role assignment manifest")

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    if config.stage == "freeze_lineage":
        return _freeze_lineage(
            config=config,
            profile=profile,
            profile_path=profile_path,
            role_manifest_path=role_manifest_path,
            run_dir=run_dir,
        )
    return _audit_labels(
        config=config,
        profile=profile,
        profile_path=profile_path,
        role_manifest_path=role_manifest_path,
        run_dir=run_dir,
    )


def _freeze_lineage(
    *,
    config: SBCExitReliabilityV63AuditConfig,
    profile: dict[str, Any],
    profile_path: Path,
    role_manifest_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    role_manifest = _load_json(role_manifest_path)
    role_rows_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"), "role assignment selected rows"
    )
    expected_rows_hash = str(profile["source_lineage"]["role_assignment_rows_sha256"])
    if role_rows_descriptor["sha256"] != expected_rows_hash:
        raise ValueError("role assignment rows hash does not match #223 profile")
    role_rows = _load_jsonl(Path(role_rows_descriptor["path"]), label_content=False)
    _validate_role_rows(role_rows, profile=profile)
    contaminated = {
        str(value)
        for value in profile["source_contract"][
            "schema_inspection_contaminated_market_ids"
        ]
    }
    lineage_rows = [
        _freeze_source_row(row, profile=profile, contaminated=contaminated)
        for row in sorted(
            role_rows,
            key=lambda item: (int(item["minimum_decision_ts"]), str(item["market_id"])),
        )
    ]
    rows_path = run_dir / "pre_target_access_exit_reliability_lineage_rows.jsonl"
    _write_jsonl(rows_path, lineage_rows)
    eligible_rows = [row for row in lineage_rows if row["eligible_for_exit_reliability"]]
    excluded_rows = [row for row in lineage_rows if not row["eligible_for_exit_reliability"]]
    role_counts = Counter(str(row["role"]) for row in eligible_rows)
    manifest = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "role_assignment_manifest": _descriptor(role_manifest_path),
        "role_assignment_rows": role_rows_descriptor,
        "lineage_rows": _descriptor(rows_path),
        "source_market_count": len(lineage_rows),
        "eligible_market_count": len(eligible_rows),
        "excluded_market_count": len(excluded_rows),
        "eligible_role_market_counts": dict(sorted(role_counts.items())),
        "eligible_minimum_decision_ts": min(
            int(row["minimum_decision_ts"]) for row in eligible_rows
        ),
        "eligible_maximum_decision_ts": max(
            int(row["maximum_decision_ts"]) for row in eligible_rows
        ),
        "schema_inspection_contamination_excluded": True,
        "schema_inspection_contaminated_market_ids": sorted(contaminated),
        "excluded_rows": [
            {
                "market_id": row["market_id"],
                "slug": row["slug"],
                "reason_codes": row["exclusion_reason_codes"],
            }
            for row in excluded_rows
        ],
        "explicitly_excluded_lineages": profile["explicitly_excluded_lineages"],
        "label_file_content_opened": False,
        "label_file_bytes_hashed_only": True,
        "corpus_manifest_content_opened": False,
        "outcome_resolution_or_pnl_opened": False,
        "pre_target_access_lineage_frozen": True,
        "pre_target_access_validation_passed": True,
        "fit_started": False,
        **_blocked_safety_fields(),
    }
    manifest["lineage_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "pre_target_access_exit_reliability_lineage_manifest.json"
    _write_json(manifest_path, manifest)
    markdown_path = run_dir / "pre_target_access_exit_reliability_lineage_manifest.md"
    _write_text(markdown_path, _lineage_markdown(manifest))
    return {
        "run_dir": run_dir,
        "lineage_manifest_path": manifest_path,
        "lineage_manifest_sha256": _sha256_file(manifest_path),
        "lineage_rows_path": rows_path,
        "lineage_rows_sha256": _sha256_file(rows_path),
        "manifest": manifest,
    }


def _audit_labels(
    *,
    config: SBCExitReliabilityV63AuditConfig,
    profile: dict[str, Any],
    profile_path: Path,
    role_manifest_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    lineage_path = Path(config.lineage_manifest_path).resolve()  # type: ignore[arg-type]
    _verify_pin(
        lineage_path,
        str(config.expected_lineage_manifest_sha256),
        "#223 pre-target lineage manifest",
    )
    lineage = _load_json(lineage_path)
    _validate_frozen_lineage(
        lineage,
        profile_path=profile_path,
        role_manifest_path=role_manifest_path,
    )
    rows_descriptor = _verified_descriptor(lineage.get("lineage_rows"), "lineage rows")
    lineage_rows = _load_jsonl(Path(rows_descriptor["path"]), label_content=False)
    eligible_rows = [row for row in lineage_rows if row["eligible_for_exit_reliability"]]
    label_stats: dict[str, Any] = {
        "side_counts": Counter(),
        "target_counts_by_side": {side: Counter() for side in SIDES},
        "target_counts_by_role_and_side": defaultdict(Counter),
        "markets_with_classes_by_side": {side: defaultdict(set) for side in SIDES},
        "execution_class_counts": Counter(),
        "exit_reason_counts": Counter(),
        "candidate_snapshot_count_distribution": Counter(),
        "raw_exit_window_snapshot_row_count": 0,
        "label_causality_violations": [],
        "invalid_label_rows": [],
    }
    feature_stats: dict[str, Any] = {
        "feature_row_count": 0,
        "side_feature_row_count": 0,
        "coverage_counts": Counter(),
        "missing_counts": Counter(),
        "feature_causality_violations": [],
        "prohibited_feature_fields": Counter(),
    }
    for source in eligible_rows:
        _audit_source_labels(source, profile=profile, stats=label_stats)
        _audit_source_features(source, profile=profile, stats=feature_stats)
    label_report = _build_label_audit_report(
        lineage=lineage,
        lineage_path=lineage_path,
        profile=profile,
        stats=label_stats,
        eligible_rows=eligible_rows,
    )
    feature_report = _build_feature_audit_report(
        lineage=lineage,
        lineage_path=lineage_path,
        profile=profile,
        stats=feature_stats,
        eligible_rows=eligible_rows,
    )
    audit_gate_passed = bool(
        label_report["label_audit_gate_passed"]
        and feature_report["feature_coverage_gate_passed"]
    )
    label_path = run_dir / "sbc_exit_reliability_label_audit.json"
    feature_path = run_dir / "sbc_exit_reliability_feature_coverage_report.json"
    _write_json(label_path, label_report)
    _write_json(feature_path, feature_report)
    _write_text(
        run_dir / "sbc_exit_reliability_label_audit.md",
        _label_audit_markdown(label_report),
    )
    _write_text(
        run_dir / "sbc_exit_reliability_feature_coverage_report.md",
        _feature_audit_markdown(feature_report),
    )
    manifest = {
        "schema_version": AUDIT_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "pre_target_access_lineage_manifest": _descriptor(lineage_path),
        "label_audit": _descriptor(label_path),
        "feature_coverage_report": _descriptor(feature_path),
        "eligible_market_count": len(eligible_rows),
        "audit_gate_passed": audit_gate_passed,
        "fit_allowed": audit_gate_passed,
        "fit_started": False,
        "blocking_reason_codes": sorted(
            set(label_report["label_audit_reason_codes"])
            | set(feature_report["feature_coverage_reason_codes"])
        ),
        "labels_opened_only_after_exact_lineage_hash_verification": True,
        "outcomes_settlement_and_pnl_not_required_for_exit_availability_target": True,
        "future_intraround_books_used_in_label_stage_only": True,
        **_blocked_safety_fields(),
    }
    manifest["audit_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_3_exit_reliability_audit_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "label_audit_path": label_path,
        "feature_coverage_path": feature_path,
        "manifest": manifest,
        "label_audit": label_report,
        "feature_coverage": feature_report,
    }


def _validate_role_rows(rows: list[dict[str, Any]], *, profile: dict[str, Any]) -> None:
    source = profile["source_contract"]
    if len(rows) != int(source["source_market_count"]):
        raise ValueError("role assignment market count mismatch")
    if len({str(row.get("market_id")) for row in rows}) != len(rows):
        raise ValueError("role assignment markets must be unique")
    counts = Counter(str(row.get("role")) for row in rows)
    observed_counts = {role: counts[role] for role in ROLES}
    if observed_counts != source["role_market_counts_before_exclusion"]:
        raise ValueError("role assignment counts mismatch")
    if any(row.get("labels_or_outcomes_opened_for_role_assignment") is not False for row in rows):
        raise ValueError("role assignment opened labels or outcomes")


def _freeze_source_row(
    row: dict[str, Any], *, profile: dict[str, Any], contaminated: set[str]
) -> dict[str, Any]:
    market_id = str(row["market_id"])
    source_dir = Path(str(row["source_corpus_dir"])).resolve()
    allowed_root = Path(profile["source_contract"]["eligible_corpus_root"]).resolve()
    if not source_dir.is_relative_to(allowed_root):
        raise ValueError(f"source corpus outside frozen root: {source_dir}")
    feature_descriptor = _verified_descriptor(row.get("feature_rows"), "feature rows")
    corpus_descriptor = _verified_descriptor(row.get("corpus_manifest"), "corpus manifest")
    label_path = source_dir / "polymarket_label_rows.jsonl"
    snapshot_path = source_dir / "polymarket_token_book_snapshots.jsonl"
    raw_orderbook_path = source_dir / "raw_polymarket_orderbooks.jsonl"
    raw_orderbook_descriptor: dict[str, Any]
    if raw_orderbook_path.is_file():
        raw_orderbook_descriptor = {
            **_descriptor(raw_orderbook_path),
            "file_present": True,
            "verification_source": "physical_file",
        }
    else:
        raw_orderbook_descriptor = {
            "path": str(raw_orderbook_path),
            "sha256": None,
            "file_present": False,
            "verification_source": "deferred_until_post_freeze_corpus_manifest_access",
        }
    excluded = market_id in contaminated
    maximum_decision_ts = int(row["maximum_decision_ts"])
    if maximum_decision_ts >= int(
        profile["source_contract"]["eligible_max_decision_ts_exclusive"]
    ):
        raise ValueError("historical source overlaps excluded future lineage")
    return {
        "schema_version": LINEAGE_ROWS_SCHEMA_VERSION,
        "market_id": market_id,
        "slug": source_dir.name,
        "role": str(row["role"]),
        "source_corpus_dir": str(source_dir),
        "minimum_decision_ts": int(row["minimum_decision_ts"]),
        "maximum_decision_ts": maximum_decision_ts,
        "decision_row_count": int(row["decision_row_count"]),
        "training_sampled_orderbook_row_count": int(
            row["training_sampled_orderbook_row_count"]
        ),
        "provider_raw_orderbook_snapshot_count": int(
            row["provider_raw_orderbook_snapshot_count"]
        ),
        "feature_rows": feature_descriptor,
        "label_rows": _descriptor(label_path),
        "token_book_snapshots": _descriptor(snapshot_path),
        "raw_orderbooks": raw_orderbook_descriptor,
        "corpus_manifest": corpus_descriptor,
        "corpus_manifest_content_opened_during_lineage_freeze": False,
        "eligible_for_exit_reliability": not excluded,
        "exclusion_reason_codes": (
            ["pre_freeze_schema_inspection_contamination"] if excluded else []
        ),
        "label_content_opened_during_lineage_freeze": False,
    }


def _audit_source_labels(
    source: dict[str, Any], *, profile: dict[str, Any], stats: dict[str, Any]
) -> None:
    corpus_descriptor = _verified_descriptor(source.get("corpus_manifest"), "corpus manifest")
    corpus = _load_json(Path(corpus_descriptor["path"]))
    normalized_hashes = dict(corpus.get("normalized_artifact_hashes") or {})
    raw_hashes = dict(corpus.get("raw_artifact_hashes") or {})
    label_descriptor = _verified_descriptor(source.get("label_rows"), "label rows")
    feature_descriptor = _verified_descriptor(source.get("feature_rows"), "feature rows")
    snapshot_descriptor = _verified_descriptor(
        source.get("token_book_snapshots"), "token book snapshots"
    )
    _require_matching_hash(
        feature_descriptor["sha256"], normalized_hashes.get("feature_rows"), "feature"
    )
    _require_matching_hash(
        label_descriptor["sha256"], normalized_hashes.get("label_rows"), "label"
    )
    _require_matching_hash(
        snapshot_descriptor["sha256"],
        normalized_hashes.get("token_book_snapshots"),
        "snapshot",
    )
    raw_orderbook_hash = str(
        raw_hashes.get("raw_polymarket_orderbooks.jsonl") or ""
    )
    _require_sha256(raw_orderbook_hash, "raw orderbook inherited sha256")
    raw_descriptor = dict(source.get("raw_orderbooks") or {})
    if raw_descriptor.get("file_present") is True:
        _require_matching_hash(
            _sha256_file(Path(str(raw_descriptor["path"]))),
            raw_orderbook_hash,
            "raw orderbook",
        )
    labels = _load_jsonl(Path(label_descriptor["path"]), label_content=True)
    expected_schema = profile["label_contract"]["required_label_schema_version"]
    sbc_rows = [row for row in labels if str(row.get("action")) in SBC_ACTIONS.values()]
    expected_count = int(source["decision_row_count"]) * len(SIDES)
    if len(sbc_rows) != expected_count:
        stats["invalid_label_rows"].append(
            {"market_id": source["market_id"], "reason": "sbc_label_count_mismatch"}
        )
    for row in sbc_rows:
        side = "UP" if str(row.get("action")) == SBC_ACTIONS["UP"] else "DOWN"
        path = dict(row.get("sell_before_close_exit_path") or {})
        execution_class = str(row.get("sell_before_close_execution_class") or "")
        uses_path = row.get("label_uses_executable_exit_path") is True
        target = int(execution_class == "realizable_sell_before_close" and uses_path)
        market_id = str(source["market_id"])
        role = str(source["role"])
        stats["side_counts"][side] += 1
        stats["target_counts_by_side"][side][str(target)] += 1
        stats["target_counts_by_role_and_side"][(role, side)][str(target)] += 1
        stats["markets_with_classes_by_side"][side][str(target)].add(market_id)
        stats["execution_class_counts"][execution_class] += 1
        for reason in path.get("exit_path_reason_codes") or []:
            stats["exit_reason_counts"][str(reason)] += 1
        candidate_count = int(path.get("candidate_exit_snapshot_count") or 0)
        stats["candidate_snapshot_count_distribution"][str(candidate_count)] += 1
        if candidate_count > 0:
            stats["raw_exit_window_snapshot_row_count"] += 1
        decision_ts = int(row.get("decision_ts") or 0)
        best_exit_ts = int(path.get("best_executable_exit_ts") or 0)
        if target and best_exit_ts <= decision_ts:
            stats["label_causality_violations"].append(
                {
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "best_executable_exit_ts": best_exit_ts,
                    "reason": "executable_exit_not_strictly_after_decision",
                }
            )
        if corpus.get("sell_before_close_label_schema_version") != expected_schema:
            stats["invalid_label_rows"].append(
                {"market_id": market_id, "reason": "label_schema_version_mismatch"}
            )
        if corpus.get("sell_before_close_label_gate_passed") is not True:
            stats["invalid_label_rows"].append(
                {"market_id": market_id, "reason": "source_label_gate_not_passed"}
            )
        if row.get("paper_only") is not True or row.get("capital_at_risk") is not False:
            stats["invalid_label_rows"].append(
                {"market_id": market_id, "reason": "label_safety_contract_invalid"}
            )


def _audit_source_features(
    source: dict[str, Any], *, profile: dict[str, Any], stats: dict[str, Any]
) -> None:
    descriptor = _verified_descriptor(source.get("feature_rows"), "feature rows")
    features = _load_jsonl(Path(descriptor["path"]), label_content=False)
    common = tuple(profile["feature_contract"]["common_required_features"])
    side_suffixes = tuple(profile["feature_contract"]["side_required_feature_suffixes"])
    prohibited = tuple(profile["feature_contract"]["prohibited_decision_input_tokens"])
    for row in features:
        stats["feature_row_count"] += 1
        decision_ts = int(row.get("decision_ts") or 0)
        max_input_ts = int(row.get("max_input_ts") or 0)
        if max_input_ts > decision_ts:
            stats["feature_causality_violations"].append(
                {
                    "market_id": source["market_id"],
                    "decision_ts": decision_ts,
                    "max_input_ts": max_input_ts,
                }
            )
        payload = dict(row.get("features") or {})
        for name in common:
            _count_feature(name, payload.get(name), stats)
        for side in SIDES:
            stats["side_feature_row_count"] += 1
            prefix = side.lower()
            for suffix in side_suffixes:
                name = f"{prefix}_{suffix}"
                _count_feature(name, payload.get(name), stats)
        for field in _flatten_keys(payload):
            lower = field.lower()
            if any(token in lower for token in prohibited):
                stats["prohibited_feature_fields"][field] += 1


def _build_label_audit_report(
    *,
    lineage: dict[str, Any],
    lineage_path: Path,
    profile: dict[str, Any],
    stats: dict[str, Any],
    eligible_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gates = profile["audit_support_gates"]
    market_classes = {
        side: {
            target: len(markets)
            for target, markets in stats["markets_with_classes_by_side"][side].items()
        }
        for side in SIDES
    }
    both_class_markets = {
        side: len(
            stats["markets_with_classes_by_side"][side]["1"]
            & stats["markets_with_classes_by_side"][side]["0"]
        )
        for side in SIDES
    }
    total_labels = sum(stats["side_counts"].values())
    snapshot_coverage = (
        stats["raw_exit_window_snapshot_row_count"] / total_labels if total_labels else 0.0
    )
    role_counts = Counter(str(row["role"]) for row in eligible_rows)
    checks = {
        "eligible_market_support": len(eligible_rows)
        >= int(gates["minimum_eligible_market_count"]),
        "role_market_support": all(
            role_counts[role] >= int(minimum)
            for role, minimum in gates["minimum_role_market_counts"].items()
        ),
        "side_label_support": all(
            stats["side_counts"][side]
            >= int(gates["minimum_sell_before_close_label_count_per_side"])
            for side in SIDES
        ),
        "positive_target_support": all(
            stats["target_counts_by_side"][side]["1"]
            >= int(gates["minimum_positive_exit_label_count_per_side"])
            for side in SIDES
        ),
        "negative_target_support": all(
            stats["target_counts_by_side"][side]["0"]
            >= int(gates["minimum_negative_exit_label_count_per_side"])
            for side in SIDES
        ),
        "both_class_market_support": all(
            both_class_markets[side]
            >= int(gates["minimum_markets_with_both_target_classes_per_side"])
            for side in SIDES
        ),
        "raw_exit_window_snapshot_coverage": snapshot_coverage
        >= float(gates["minimum_raw_exit_window_snapshot_coverage_rate"]),
        "label_causality": len(stats["label_causality_violations"])
        <= int(gates["maximum_label_causality_violation_count"]),
        "label_contract": not stats["invalid_label_rows"],
    }
    reasons = [f"{name}_failed" for name, passed in checks.items() if not passed]
    report = {
        "schema_version": LABEL_AUDIT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "pre_target_access_lineage_manifest": _descriptor(lineage_path),
        "lineage_manifest_id": lineage["lineage_manifest_id"],
        "eligible_market_count": len(eligible_rows),
        "eligible_role_market_counts": dict(sorted(role_counts.items())),
        "sell_before_close_label_count": total_labels,
        "sell_before_close_label_count_by_side": dict(stats["side_counts"]),
        "target_counts_by_side": {
            side: dict(stats["target_counts_by_side"][side]) for side in SIDES
        },
        "target_counts_by_role_and_side": {
            f"{role}:{side}": dict(counts)
            for (role, side), counts in sorted(
                stats["target_counts_by_role_and_side"].items()
            )
        },
        "market_target_class_coverage_by_side": market_classes,
        "markets_with_both_target_classes_by_side": both_class_markets,
        "sell_before_close_execution_class_counts": dict(
            stats["execution_class_counts"]
        ),
        "exit_path_reason_code_counts": dict(stats["exit_reason_counts"]),
        "candidate_exit_snapshot_count_distribution": dict(
            stats["candidate_snapshot_count_distribution"]
        ),
        "raw_exit_window_snapshot_coverage_count": stats[
            "raw_exit_window_snapshot_row_count"
        ],
        "raw_exit_window_snapshot_coverage_rate": snapshot_coverage,
        "label_causality_violation_count": len(stats["label_causality_violations"]),
        "label_causality_violations": stats["label_causality_violations"][:25],
        "invalid_label_row_count": len(stats["invalid_label_rows"]),
        "invalid_label_rows": stats["invalid_label_rows"][:25],
        "market_disjoint_split_feasible": len(eligible_rows)
        == len({str(row["market_id"]) for row in eligible_rows}),
        "support_gate_checks": checks,
        "label_audit_gate_passed": all(checks.values()),
        "label_audit_reason_codes": reasons,
        "target_uses_outcome_settlement_or_pnl": False,
        "future_intraround_books_used_in_label_stage_only": True,
        "fit_started": False,
        **_blocked_safety_fields(),
    }
    report["label_audit_id"] = canonical_json_sha256(report)
    return report


def _build_feature_audit_report(
    *,
    lineage: dict[str, Any],
    lineage_path: Path,
    profile: dict[str, Any],
    stats: dict[str, Any],
    eligible_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gates = profile["audit_support_gates"]
    expected_feature_rows = sum(int(row["decision_row_count"]) for row in eligible_rows)
    required_names = [
        *profile["feature_contract"]["common_required_features"],
        *[
            f"{side.lower()}_{suffix}"
            for side in SIDES
            for suffix in profile["feature_contract"]["side_required_feature_suffixes"]
        ],
    ]
    coverage = {
        name: {
            "available_count": int(stats["coverage_counts"][name]),
            "missing_count": int(stats["missing_counts"][name]),
        }
        for name in required_names
    }
    checks = {
        "feature_row_count": stats["feature_row_count"] == expected_feature_rows,
        "required_feature_coverage": all(
            item["missing_count"] == 0 for item in coverage.values()
        ),
        "feature_causality": len(stats["feature_causality_violations"])
        <= int(gates["maximum_feature_causality_violation_count"]),
        "prohibited_feature_fields": sum(stats["prohibited_feature_fields"].values())
        <= int(gates["maximum_prohibited_feature_field_count"]),
    }
    reasons = [f"{name}_failed" for name, passed in checks.items() if not passed]
    report = {
        "schema_version": FEATURE_AUDIT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "pre_target_access_lineage_manifest": _descriptor(lineage_path),
        "lineage_manifest_id": lineage["lineage_manifest_id"],
        "eligible_market_count": len(eligible_rows),
        "feature_row_count": stats["feature_row_count"],
        "expected_feature_row_count": expected_feature_rows,
        "side_feature_row_count": stats["side_feature_row_count"],
        "required_feature_coverage": coverage,
        "composition_stage_feature_sources": profile["feature_contract"][
            "composition_stage_features"
        ],
        "feature_causality_violation_count": len(stats["feature_causality_violations"]),
        "feature_causality_violations": stats["feature_causality_violations"][:25],
        "prohibited_feature_field_count": sum(
            stats["prohibited_feature_fields"].values()
        ),
        "prohibited_feature_field_distribution": dict(
            stats["prohibited_feature_fields"]
        ),
        "feature_coverage_gate_checks": checks,
        "feature_coverage_gate_passed": all(checks.values()),
        "feature_coverage_reason_codes": reasons,
        "decision_time_inputs_only": True,
        "fit_started": False,
        **_blocked_safety_fields(),
    }
    report["feature_coverage_report_id"] = canonical_json_sha256(report)
    return report


def _validate_frozen_lineage(
    lineage: dict[str, Any], *, profile_path: Path, role_manifest_path: Path
) -> None:
    checks = {
        "schema": lineage.get("schema_version") == LINEAGE_SCHEMA_VERSION,
        "candidate": lineage.get("candidate_name") == CANDIDATE_NAME,
        "profile": lineage.get("profile") == _descriptor(profile_path),
        "role_manifest": lineage.get("role_assignment_manifest")
        == _descriptor(role_manifest_path),
        "frozen": lineage.get("pre_target_access_lineage_frozen") is True,
        "validated": lineage.get("pre_target_access_validation_passed") is True,
        "labels_closed": lineage.get("label_file_content_opened") is False,
        "corpus_manifest_closed": lineage.get("corpus_manifest_content_opened") is False,
        "targets_closed": lineage.get("outcome_resolution_or_pnl_opened") is False,
        "contamination_excluded": lineage.get(
            "schema_inspection_contamination_excluded"
        )
        is True,
        "fit_not_started": lineage.get("fit_started") is False,
        "safety": all(lineage.get(key) == value for key, value in _blocked_safety_fields().items()),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#223 frozen lineage invalid: " + ", ".join(blockers))
    _verified_descriptor(lineage.get("lineage_rows"), "lineage rows")


def _valid_source_contract(source: dict[str, Any]) -> bool:
    before = dict(source.get("role_market_counts_before_exclusion") or {})
    after = dict(source.get("role_market_counts_after_exclusion") or {})
    contaminated = list(source.get("schema_inspection_contaminated_market_ids") or [])
    return (
        set(before) == set(ROLES)
        and set(after) == set(ROLES)
        and all(isinstance(value, int) and value >= 0 for value in before.values())
        and all(isinstance(value, int) and value >= 0 for value in after.values())
        and sum(before.values()) == int(source.get("source_market_count") or 0)
        and sum(after.values()) == sum(before.values()) - len(contaminated)
        and len(contaminated) == 1
        and bool(str(source.get("eligible_corpus_root") or ""))
        and int(source.get("eligible_max_decision_ts_exclusive") or 0) > 0
    )


def _valid_support_gates(gates: dict[str, Any]) -> bool:
    return (
        int(gates.get("minimum_eligible_market_count", -1)) > 0
        and set(gates.get("minimum_role_market_counts") or {}) == set(ROLES)
        and all(
            int(value) >= 0
            for value in dict(gates["minimum_role_market_counts"]).values()
        )
        and int(gates.get("minimum_sell_before_close_label_count_per_side", -1)) > 0
        and int(gates.get("minimum_positive_exit_label_count_per_side", -1)) > 0
        and int(gates.get("minimum_negative_exit_label_count_per_side", -1)) > 0
        and int(gates.get("minimum_markets_with_both_target_classes_per_side", -1)) > 0
        and 0.0
        < float(gates.get("minimum_raw_exit_window_snapshot_coverage_rate", 0.0))
        <= 1.0
        and int(gates.get("maximum_feature_causality_violation_count", -1)) == 0
        and int(gates.get("maximum_label_causality_violation_count", -1)) == 0
        and int(gates.get("maximum_prohibited_feature_field_count", -1)) == 0
    )


def _count_feature(name: str, value: Any, stats: dict[str, Any]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        stats["missing_counts"][name] += 1
        return
    if not math.isfinite(float(value)):
        stats["missing_counts"][name] += 1
        return
    stats["coverage_counts"][name] += 1


def _flatten_keys(payload: dict[str, Any], prefix: str = "") -> list[str]:
    names = []
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        names.append(name)
        if isinstance(value, dict):
            names.extend(_flatten_keys(value, name))
    return names


def _lineage_markdown(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.3 Pre-Target Exit-Reliability Lineage",
            "",
            f"- source_market_count: `{manifest['source_market_count']}`",
            f"- eligible_market_count: `{manifest['eligible_market_count']}`",
            f"- excluded_market_count: `{manifest['excluded_market_count']}`",
            f"- eligible_role_market_counts: `{json.dumps(manifest['eligible_role_market_counts'], sort_keys=True)}`",
            "- label_file_content_opened: `false`",
            "- corpus_manifest_content_opened: `false`",
            "- label_file_bytes_hashed_only: `true`",
            "- schema_inspection_contamination_excluded: `true`",
            "- fit_started: `false`",
            "- paper/live/promotion: `blocked`",
            "",
        ]
    )


def _label_audit_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.3 SBC Exit-Reliability Label Audit",
            "",
            f"- eligible_market_count: `{report['eligible_market_count']}`",
            f"- sell_before_close_label_count: `{report['sell_before_close_label_count']}`",
            f"- target_counts_by_side: `{json.dumps(report['target_counts_by_side'], sort_keys=True)}`",
            f"- markets_with_both_target_classes_by_side: `{json.dumps(report['markets_with_both_target_classes_by_side'], sort_keys=True)}`",
            f"- raw_exit_window_snapshot_coverage_rate: `{report['raw_exit_window_snapshot_coverage_rate']}`",
            f"- label_causality_violation_count: `{report['label_causality_violation_count']}`",
            f"- label_audit_gate_passed: `{str(report['label_audit_gate_passed']).lower()}`",
            f"- reason_codes: `{json.dumps(report['label_audit_reason_codes'])}`",
            "- fit_started: `false`",
            "",
        ]
    )


def _feature_audit_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.3 SBC Exit-Reliability Feature Coverage",
            "",
            f"- feature_row_count: `{report['feature_row_count']}`",
            f"- feature_causality_violation_count: `{report['feature_causality_violation_count']}`",
            f"- prohibited_feature_field_count: `{report['prohibited_feature_field_count']}`",
            f"- feature_coverage_gate_passed: `{str(report['feature_coverage_gate_passed']).lower()}`",
            f"- reason_codes: `{json.dumps(report['feature_coverage_reason_codes'])}`",
            "- decision_time_inputs_only: `true`",
            "- fit_started: `false`",
            "",
        ]
    )


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _verified_descriptor(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor missing")
    path = Path(str(value.get("path") or "")).resolve()
    expected = str(value.get("sha256") or "")
    _require_sha256(expected, f"{name} sha256")
    _verify_pin(path, expected, name)
    return {"path": str(path), "sha256": expected}


def _descriptor(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_jsonl(path: Path, *, label_content: bool) -> list[dict[str, Any]]:
    if "label" in path.name.lower() and not label_content:
        raise ValueError(f"label content access forbidden in lineage-freeze stage: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL object required: {path}")
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_pin(path: Path, expected: str, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} missing: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} sha256 mismatch")


def _require_matching_hash(observed: str, expected: Any, name: str) -> None:
    if observed != str(expected or ""):
        raise ValueError(f"{name} hash mismatch")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _require_sha256(value: str, name: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _require_git_sha(value: str) -> None:
    if len(value) != 40 or not all(char in "0123456789abcdef" for char in value):
        raise ValueError("implementation_commit must be a Git SHA-1")
