"""Freeze and build the preregistered #224 runtime-aligned SBC target corpus."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_paper_canary import (
    SELL_BEFORE_CLOSE_EXIT_RULE_ID,
    SELL_BEFORE_CLOSE_EXIT_WINDOW_SECONDS,
    _paper_position_lifecycle,
    _sell_before_close_exit_blockers,
    _sell_before_close_exit_candidates,
)
from bigan.v8.polymarket.training.execution_layer_v2_sbc_exit_reliability_v6_3_fit import (
    SBC_ACTIONS,
    _materialize_rows,
)
from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    O_V8_PAPER_FRESH_EXIT_EDGE_THRESHOLD,
    O_V8_PAPER_FRESH_EXIT_FORCE_SECONDS_TO_CLOSE,
    O_V8_PAPER_FRESH_EXIT_PROFIT_TARGET,
    O_V8_PAPER_FRESH_EXIT_THRESHOLD_PROFILE_NAME,
    _fresh_paper_adapter_sell_decision,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_execution_guard_config,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-runtime-aligned-sbc-net-return-v6-4-profile-v1"
LINEAGE_ROWS_SCHEMA_VERSION = "bigan-v8-runtime-aligned-sbc-net-return-v6-4-lineage-row-v1"
LINEAGE_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-runtime-aligned-sbc-net-return-v6-4-lineage-manifest-v1"
)
TARGET_ROW_SCHEMA_VERSION = "bigan-v8-runtime-aligned-sbc-net-return-v6-4-target-row-v1"
TARGET_REPORT_SCHEMA_VERSION = (
    "bigan-v8-runtime-aligned-sbc-net-return-v6-4-target-report-v1"
)
TARGET_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-runtime-aligned-sbc-net-return-v6-4-target-manifest-v1"
)
CANDIDATE_NAME = "runtime_aligned_sbc_net_return_v6_4"
DEVELOPMENT_ROLES = ("development_train", "development_calibration")
SIDES = ("UP", "DOWN")
FORBIDDEN_FEATURE_TOKENS = (
    "outcome",
    "settlement",
    "resolution",
    "target",
    "realized",
    "pnl",
    "oracle",
    "future_return",
)


@dataclass(frozen=True, slots=True)
class RuntimeAlignedSBCNetReturnV64Config:
    """Pinned inputs for either lineage freeze or target materialization."""

    stage: Literal["freeze_lineage", "build_labels"]
    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    issue_223_lineage_manifest_path: Path | str
    v6_2_historical_manifest_path: Path | str
    implementation_commit: str
    external_corpus_dir: Path | str | None = None
    lineage_freeze_manifest_path: Path | str | None = None
    expected_lineage_freeze_manifest_sha256: str | None = None
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if self.stage not in {"freeze_lineage", "build_labels"}:
            raise ValueError("unsupported v6.4 stage")
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_profile_sha256, "expected_profile_sha256")
        _require_git_sha(self.implementation_commit)
        for name in (
            "output_dir",
            "profile_path",
            "issue_223_lineage_manifest_path",
            "v6_2_historical_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.external_corpus_dir is not None:
            object.__setattr__(self, "external_corpus_dir", Path(self.external_corpus_dir))
        if self.lineage_freeze_manifest_path is not None:
            object.__setattr__(
                self,
                "lineage_freeze_manifest_path",
                Path(self.lineage_freeze_manifest_path),
            )
        if self.stage == "build_labels":
            if self.external_corpus_dir is None or self.lineage_freeze_manifest_path is None:
                raise ValueError("build_labels requires external corpus and lineage freeze")
            _require_sha256(
                str(self.expected_lineage_freeze_manifest_sha256 or ""),
                "expected_lineage_freeze_manifest_sha256",
            )


def validate_runtime_aligned_sbc_net_return_v6_4_profile(
    profile: dict[str, Any],
) -> None:
    """Validate the immutable #224 target and access contract."""

    lineage = dict(profile.get("source_lineage") or {})
    roles = dict(profile.get("development_roles") or {})
    runtime = dict(profile.get("runtime_policy_contract") or {})
    target = dict(profile.get("target_contract") or {})
    support = dict(profile.get("support_gates") or {})
    access = dict(profile.get("access_policy") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 224,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "lineage": len(lineage) == 5
        and all(_is_sha256(str(value)) for value in lineage.values()),
        "roles": roles
        == {
            "model_fit": "development_train",
            "model_fit_market_count": 89,
            "conformal_calibration": "development_calibration",
            "conformal_calibration_market_count": 45,
            "total_market_count": 134,
            "historical_oof_role_excluded": "confirmatory_validation",
            "historical_oof_market_count_excluded": 60,
            "market_disjoint": True,
            "chronological": True,
        },
        "runtime": runtime.get("paper_position_size") == 0.2
        and runtime.get("intended_exit_policy") == "sell_before_close"
        and runtime.get("exit_policy_source")
        == O_V8_PAPER_FRESH_EXIT_THRESHOLD_PROFILE_NAME
        and runtime.get("exit_rule_id") == SELL_BEFORE_CLOSE_EXIT_RULE_ID
        and runtime.get("exit_force_seconds_to_close")
        == O_V8_PAPER_FRESH_EXIT_FORCE_SECONDS_TO_CLOSE
        and runtime.get("exit_edge_threshold") == O_V8_PAPER_FRESH_EXIT_EDGE_THRESHOLD
        and runtime.get("exit_profit_target") == O_V8_PAPER_FRESH_EXIT_PROFIT_TARGET
        and all(
            _is_sha256(str(value))
            for value in dict(runtime.get("source_function_sha256") or {}).values()
        ),
        "target": target.get("primary_target")
        == "runtime_policy_after_cost_net_pnl_per_contract"
        and target.get("cost_fields_subtracted_exactly_once")
        == ["fees", "slippage", "liquidity_impact"]
        and target.get("future_fields_allowed_only_in_target_stage") is True
        and target.get("target_used_as_decision_time_input") is False
        and target.get("settlement_timestamp_required") is False,
        "support": int(support.get("expected_rows_per_market") or 0) == 8
        and all(
            int(support.get(name) or 0) > 0
            for name in (
                "minimum_train_residual_rows_per_side",
                "minimum_calibration_residual_rows_per_side",
                "minimum_train_closed_rows_per_side",
                "minimum_calibration_closed_rows_per_side",
                "minimum_unique_markets_per_role_and_side",
            )
        ),
        "external_root": Path(
            str(profile.get("external_training_corpus_root") or "")
        ).resolve()
        == Path("/Volumes/PHILIPS/v8").resolve(),
        "access": access.get("freeze_lineage_before_target_content_access") is True
        and access.get("fit_stage_roles_only") == list(DEVELOPMENT_ROLES)
        and all(
            access.get(name) is False
            for name in (
                "issue_223_oof_opened",
                "issue_212_future_outcomes_opened",
                "issue_221_paper_outcomes_opened",
                "issue_192_prefreeze_rows_opened",
                "new_future_holdout_outcomes_opened",
            )
        ),
        "safety": profile.get("safety") == _blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#224 profile invalid: " + ", ".join(blockers))


def runtime_policy_source_hashes() -> dict[str, str]:
    """Return content hashes for every frozen runtime-policy function."""

    functions = {
        "paper_position_lifecycle": _paper_position_lifecycle,
        "sell_before_close_exit_candidates": _sell_before_close_exit_candidates,
        "sell_before_close_exit_blockers": _sell_before_close_exit_blockers,
        "fresh_paper_adapter_sell_decision": _fresh_paper_adapter_sell_decision,
        "v8_execution_guard_config": _v8_execution_guard_config,
    }
    return {
        name: hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()
        for name, function in functions.items()
    }


def compute_runtime_policy_after_cost_target(
    *,
    selected_side: str,
    entry_price: float,
    exit_price: float | None,
    resolved_outcome: str,
    fees: float,
    slippage: float,
    liquidity_impact: float,
    paper_position_size: float,
) -> dict[str, Any]:
    """Compute one post-decision target with the frozen cost fields once."""

    if selected_side not in SIDES or resolved_outcome not in SIDES:
        raise ValueError("runtime target requires UP/DOWN side and official outcome")
    values = (entry_price, fees, slippage, liquidity_impact, paper_position_size)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("runtime target values must be finite")
    if not 0.0 < entry_price < 1.0 or paper_position_size <= 0.0:
        raise ValueError("runtime target entry price/size invalid")
    if min(fees, slippage, liquidity_impact) < 0.0:
        raise ValueError("runtime target costs must be non-negative")
    closed = exit_price is not None
    terminal_value = float(exit_price) if closed else float(selected_side == resolved_outcome)
    if not 0.0 <= terminal_value <= 1.0:
        raise ValueError("runtime target terminal value invalid")
    execution_cost = fees + slippage + liquidity_impact
    gross_pnl = terminal_value - entry_price
    net_pnl = gross_pnl - execution_cost
    gross_return = terminal_value / entry_price - 1.0
    net_return = gross_return - execution_cost
    return {
        "position_lifecycle_class": (
            "closed_before_settlement" if closed else "settlement_residual"
        ),
        "terminal_value_source": (
            "first_causal_guard_quality_sell_position_executable_bid"
            if closed
            else "official_read_only_resolved_outcome_payout"
        ),
        "terminal_value_per_contract": terminal_value,
        "gross_pnl_per_contract": gross_pnl,
        "execution_cost_per_contract": execution_cost,
        "runtime_policy_after_cost_net_pnl_per_contract": net_pnl,
        "runtime_policy_after_cost_net_return": net_return,
        "runtime_policy_after_cost_net_pnl_at_frozen_size": (
            net_pnl * paper_position_size
        ),
        "cost_fields_subtracted_exactly_once": True,
        "settlement_timestamp_used": False,
    }


def run_runtime_aligned_sbc_net_return_v6_4(
    config: RuntimeAlignedSBCNetReturnV64Config,
) -> dict[str, Any]:
    """Run the lineage freeze or the target materialization stage."""

    profile_path = Path(config.profile_path).resolve()
    lineage_path = Path(config.issue_223_lineage_manifest_path).resolve()
    historical_path = Path(config.v6_2_historical_manifest_path).resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#224 profile")
    profile = _load_json(profile_path)
    validate_runtime_aligned_sbc_net_return_v6_4_profile(profile)
    _verify_pin(
        lineage_path,
        profile["source_lineage"]["issue_223_pre_target_lineage_manifest_sha256"],
        "#223 pre-target lineage manifest",
    )
    _verify_pin(
        historical_path,
        profile["source_lineage"]["v6_2_historical_manifest_sha256"],
        "v6.2 historical manifest",
    )
    run_dir = Path(config.output_dir).resolve() / config.run_id
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
            issue_223_lineage_path=lineage_path,
            historical_path=historical_path,
            run_dir=run_dir,
        )
    return _build_labels(
        config=config,
        profile=profile,
        profile_path=profile_path,
        issue_223_lineage_path=lineage_path,
        historical_path=historical_path,
        run_dir=run_dir,
    )


def _freeze_lineage(
    *,
    config: RuntimeAlignedSBCNetReturnV64Config,
    profile: dict[str, Any],
    profile_path: Path,
    issue_223_lineage_path: Path,
    historical_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    issue_223_lineage = _load_json(issue_223_lineage_path)
    source_rows_descriptor = _verified_descriptor(
        issue_223_lineage.get("lineage_rows"), "#223 lineage rows"
    )
    if source_rows_descriptor["sha256"] != profile["source_lineage"][
        "issue_223_lineage_rows_sha256"
    ]:
        raise ValueError("#223 lineage rows hash mismatch")
    source_rows = _load_jsonl(Path(source_rows_descriptor["path"]))
    selected = [
        row
        for row in source_rows
        if row.get("eligible_for_exit_reliability") is True
        and str(row.get("role")) in DEVELOPMENT_ROLES
    ]
    counts = Counter(str(row["role"]) for row in selected)
    expected = profile["development_roles"]
    if counts != Counter(
        {
            "development_train": int(expected["model_fit_market_count"]),
            "development_calibration": int(
                expected["conformal_calibration_market_count"]
            ),
        }
    ):
        raise ValueError("#224 development role count mismatch")
    frozen_rows = []
    for row in sorted(
        selected,
        key=lambda value: (int(value["minimum_decision_ts"]), str(value["market_id"])),
    ):
        artifacts = {
            name: _verified_descriptor(row.get(name), name)
            for name in (
                "feature_rows",
                "label_rows",
                "token_book_snapshots",
                "corpus_manifest",
            )
        }
        frozen = {
            "schema_version": LINEAGE_ROWS_SCHEMA_VERSION,
            "market_id": str(row["market_id"]),
            "slug": str(row["slug"]),
            "role": str(row["role"]),
            "minimum_decision_ts": int(row["minimum_decision_ts"]),
            "maximum_decision_ts": int(row["maximum_decision_ts"]),
            "decision_row_count": int(row["decision_row_count"]),
            "source_corpus_dir": str(row["source_corpus_dir"]),
            **artifacts,
            "target_file_content_opened_during_lineage_freeze": False,
            "outcome_resolution_or_pnl_opened_during_lineage_freeze": False,
        }
        frozen["lineage_row_id"] = canonical_json_sha256(frozen)
        frozen_rows.append(frozen)
    row_path = run_dir / "v6_4_runtime_policy_lineage_rows.jsonl"
    _write_jsonl(row_path, frozen_rows)
    observed_hashes = runtime_policy_source_hashes()
    expected_hashes = profile["runtime_policy_contract"]["source_function_sha256"]
    hashes_verified = observed_hashes == expected_hashes
    if not hashes_verified:
        raise ValueError("frozen runtime-policy source hash mismatch")
    manifest = {
        "schema_version": LINEAGE_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "issue_223_pre_target_lineage_manifest": _descriptor(
            issue_223_lineage_path
        ),
        "v6_2_historical_manifest": _descriptor(historical_path),
        "lineage_rows": _descriptor(row_path),
        "market_count": len(frozen_rows),
        "market_count_by_role": dict(sorted(counts.items())),
        "runtime_policy_source_hashes": observed_hashes,
        "runtime_policy_source_hashes_verified": hashes_verified,
        "target_file_content_opened": False,
        "outcome_resolution_or_pnl_opened": False,
        "historical_oof_market_count_included": 0,
        "issue_212_or_221_market_count_included": 0,
        "lineage_freeze_passed": len(frozen_rows) == 134 and hashes_verified,
        **_blocked_safety_fields(),
    }
    manifest["lineage_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_4_runtime_policy_lineage_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "lineage_manifest_path": manifest_path,
        "lineage_manifest_sha256": _sha256_file(manifest_path),
        "manifest": manifest,
    }


def _build_labels(
    *,
    config: RuntimeAlignedSBCNetReturnV64Config,
    profile: dict[str, Any],
    profile_path: Path,
    issue_223_lineage_path: Path,
    historical_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    freeze_path = Path(config.lineage_freeze_manifest_path).resolve()  # type: ignore[arg-type]
    _verify_pin(
        freeze_path,
        str(config.expected_lineage_freeze_manifest_sha256),
        "#224 lineage freeze manifest",
    )
    freeze = _load_json(freeze_path)
    if freeze.get("lineage_freeze_passed") is not True:
        raise ValueError("#224 lineage freeze did not pass")
    if freeze.get("runtime_policy_source_hashes") != runtime_policy_source_hashes():
        raise ValueError("runtime-policy source changed after lineage freeze")
    lineage_rows = _load_jsonl(
        Path(_verified_descriptor(freeze.get("lineage_rows"), "lineage rows")["path"])
    )
    historical = _load_json(historical_path)
    replay_descriptor = _verified_descriptor(
        historical.get("candidate_target_free_guard_replay"),
        "v6.2 target-free replay",
    )
    if replay_descriptor["sha256"] != profile["source_lineage"][
        "v6_2_target_free_guard_replay_sha256"
    ]:
        raise ValueError("v6.2 target-free replay hash mismatch")
    replay_rows = _load_jsonl(Path(replay_descriptor["path"]))
    external_dir = Path(config.external_corpus_dir).resolve()  # type: ignore[arg-type]
    _validate_external_corpus_dir(external_dir, profile=profile)
    if external_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"external corpus directory exists: {external_dir}")
        shutil.rmtree(external_dir)
    external_dir.mkdir(parents=True)

    output_rows = []
    lifecycle_reason_counts: Counter[str] = Counter()
    for source in lineage_rows:
        role = str(source["role"])
        if role not in DEVELOPMENT_ROLES:
            raise ValueError("non-development role entered #224 label stage")
        feature_rows = _load_jsonl(
            Path(_verified_descriptor(source["feature_rows"], "feature rows")["path"])
        )
        label_rows = _load_jsonl(
            Path(_verified_descriptor(source["label_rows"], "label rows")["path"])
        )
        decision_rows = _materialize_rows(
            [source],
            replay_rows=replay_rows,
            roles={role},
            include_targets=False,
        )
        rows, reasons = _market_runtime_target_rows(
            source=source,
            feature_rows=feature_rows,
            label_rows=label_rows,
            decision_rows=decision_rows,
            profile=profile,
            run_id=config.run_id,
        )
        output_rows.extend(rows)
        lifecycle_reason_counts.update(reasons)
    output_rows.sort(
        key=lambda row: (row["decision_ts"], row["market_id"], row["action"])
    )
    corpus_path = external_dir / "runtime_aligned_sbc_net_return_rows.jsonl"
    _write_jsonl(corpus_path, output_rows)
    report = _target_report(
        rows=output_rows,
        lineage_rows=lineage_rows,
        lifecycle_reason_counts=lifecycle_reason_counts,
        profile=profile,
        corpus_path=corpus_path,
        freeze_path=freeze_path,
    )
    report_path = run_dir / "v6_4_runtime_aligned_target_report.json"
    _write_json(report_path, report)
    _write_text(
        run_dir / "v6_4_runtime_aligned_target_report.md",
        _target_report_markdown(report),
    )
    corpus_manifest = {
        "schema_version": TARGET_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "profile": _descriptor(profile_path),
        "lineage_freeze_manifest": _descriptor(freeze_path),
        "runtime_aligned_rows": _descriptor(corpus_path),
        "target_report": _descriptor(report_path),
        "target_corpus_gate_passed": report["target_corpus_gate_passed"],
        "target_corpus_gate_reason_codes": report[
            "target_corpus_gate_reason_codes"
        ],
        "fit_allowed": report["target_corpus_gate_passed"],
        "historical_oof_opened": False,
        "issue_212_future_outcomes_opened": False,
        "issue_221_paper_outcomes_opened": False,
        **_blocked_safety_fields(),
    }
    corpus_manifest["target_manifest_id"] = canonical_json_sha256(corpus_manifest)
    external_manifest_path = external_dir / "runtime_aligned_corpus_manifest.json"
    _write_json(external_manifest_path, corpus_manifest)
    run_manifest = {
        **corpus_manifest,
        "runtime_aligned_rows": _descriptor(corpus_path),
        "external_corpus_manifest": _descriptor(external_manifest_path),
    }
    run_manifest["target_manifest_id"] = canonical_json_sha256(run_manifest)
    run_manifest_path = run_dir / "v6_4_runtime_aligned_target_manifest.json"
    _write_json(run_manifest_path, run_manifest)
    return {
        "run_dir": run_dir,
        "external_corpus_dir": external_dir,
        "target_manifest_path": run_manifest_path,
        "target_manifest_sha256": _sha256_file(run_manifest_path),
        "report": report,
        "manifest": run_manifest,
    }


def _market_runtime_target_rows(
    *,
    source: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    run_id: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    decision_map = {
        (int(row["decision_ts"]), str(row["action"])): row for row in decision_rows
    }
    label_map = {
        (int(row["decision_ts"]), str(row["action"])): row
        for row in label_rows
        if str(row.get("action")) in SBC_ACTIONS.values()
    }
    size = float(profile["runtime_policy_contract"]["paper_position_size"])
    fills = []
    for key, label in sorted(label_map.items()):
        decision = decision_map.get(key)
        if decision is None:
            raise ValueError("runtime target decision-time feature row missing")
        side = str(decision["side"])
        entry_price = float(label["entry_ask"])
        decision_ts = int(decision["decision_ts"])
        time_to_close = float(decision["features"]["time_to_close_seconds"])
        market_close_ts = decision_ts + int(round(time_to_close * 1000.0))
        fill_id = f"runtime-target-{source['market_id']}-{decision_ts}-{side}"
        fills.append(
            {
                "paper_fill_id": fill_id,
                "paper_intent_id": f"intent-{fill_id}",
                "market_id": str(source["market_id"]),
                "selected_side": side,
                "executed_action": str(decision["action"]),
                "selected_action_family": "SELL_BEFORE_CLOSE",
                "intended_exit_policy": "sell_before_close",
                "paper_fill_price": entry_price,
                "filled_size": size,
                "decision_ts": decision_ts,
                "market_close_ts": market_close_ts,
                "planned_exit_before_ts": market_close_ts
                - int(SELL_BEFORE_CLOSE_EXIT_WINDOW_SECONDS * 1000.0),
            }
        )
    lifecycle = _paper_position_lifecycle(
        run_id=f"{run_id}-{source['market_id']}",
        feature_rows=feature_rows,
        entry_fills=fills,
    )
    positions = {
        str(row["position_id"]): row for row in lifecycle["positions"]["positions"]
    }
    reasons: Counter[str] = Counter()
    rows = []
    for fill in fills:
        key = (int(fill["decision_ts"]), str(fill["executed_action"]))
        decision = decision_map[key]
        label = label_map[key]
        position = positions.get(str(fill["paper_fill_id"]))
        if position is None:
            raise ValueError("runtime target position lifecycle row missing")
        resolved_outcome = str(label.get("resolved_outcome") or "")
        target = compute_runtime_policy_after_cost_target(
            selected_side=str(fill["selected_side"]),
            entry_price=float(fill["paper_fill_price"]),
            exit_price=(
                None
                if position.get("exit_price") is None
                else float(position["exit_price"])
            ),
            resolved_outcome=resolved_outcome,
            fees=float(label["fees"]),
            slippage=float(label["slippage"]),
            liquidity_impact=float(label["liquidity_impact"]),
            paper_position_size=float(fill["filled_size"]),
        )
        reason_codes = sorted(set(position.get("residual_reason_codes") or []))
        reasons.update(reason_codes)
        row = {
            "schema_version": TARGET_ROW_SCHEMA_VERSION,
            "market_id": str(fill["market_id"]),
            "slug": str(source["slug"]),
            "role": str(source["role"]),
            "decision_ts": int(fill["decision_ts"]),
            "market_close_ts": int(fill["market_close_ts"]),
            "max_input_ts": int(decision["max_input_ts"]),
            "side": str(fill["selected_side"]),
            "action": str(fill["executed_action"]),
            "features": decision["features"],
            "entry_price": float(fill["paper_fill_price"]),
            "paper_position_size": float(fill["filled_size"]),
            "exit_decision_ts": position.get("exit_decision_ts"),
            "exit_price": position.get("exit_price"),
            "resolved_outcome": resolved_outcome,
            "fees": float(label["fees"]),
            "slippage": float(label["slippage"]),
            "liquidity_impact": float(label["liquidity_impact"]),
            "runtime_policy_residual_reason_codes": reason_codes,
            **target,
            "target_available_only_post_exit_or_official_resolution": True,
            "target_used_as_decision_time_input": False,
            "runtime_policy_source_hashes_verified": True,
            "historical_oof_row": False,
            **_blocked_safety_fields(),
        }
        if row["max_input_ts"] > row["decision_ts"]:
            raise ValueError("runtime target feature causality violation")
        if row["exit_decision_ts"] is not None and not (
            row["decision_ts"] < int(row["exit_decision_ts"]) < row["market_close_ts"]
        ):
            raise ValueError("runtime target exit timestamp causality violation")
        row["target_row_id"] = canonical_json_sha256(row)
        rows.append(row)
    return rows, reasons


def _target_report(
    *,
    rows: list[dict[str, Any]],
    lineage_rows: list[dict[str, Any]],
    lifecycle_reason_counts: Counter[str],
    profile: dict[str, Any],
    corpus_path: Path,
    freeze_path: Path,
) -> dict[str, Any]:
    expected_rows = len(lineage_rows) * int(
        profile["support_gates"]["expected_rows_per_market"]
    )
    causality_violations = sum(row["max_input_ts"] > row["decision_ts"] for row in rows)
    exit_causality_violations = sum(
        row["exit_decision_ts"] is not None
        and not (
            row["decision_ts"]
            < int(row["exit_decision_ts"])
            < row["market_close_ts"]
        )
        for row in rows
    )
    forbidden = sorted(
        {
            key
            for row in rows
            for key in _flatten_keys(row["features"])
            if any(token in key.lower() for token in FORBIDDEN_FEATURE_TOKENS)
        }
    )
    identity_count = len({str(row["target_row_id"]) for row in rows})
    finite_targets = all(
        math.isfinite(float(row["runtime_policy_after_cost_net_pnl_per_contract"]))
        for row in rows
    )
    support_rows = []
    support = profile["support_gates"]
    support_passed = True
    for role in DEVELOPMENT_ROLES:
        for side in SIDES:
            selected = [
                row for row in rows if row["role"] == role and row["side"] == side
            ]
            closed = sum(
                row["position_lifecycle_class"] == "closed_before_settlement"
                for row in selected
            )
            residual = sum(
                row["position_lifecycle_class"] == "settlement_residual"
                for row in selected
            )
            positive = sum(
                float(row["runtime_policy_after_cost_net_pnl_per_contract"]) > 0.0
                for row in selected
            )
            negative = sum(
                float(row["runtime_policy_after_cost_net_pnl_per_contract"]) < 0.0
                for row in selected
            )
            markets = len({str(row["market_id"]) for row in selected})
            role_prefix = "train" if role == "development_train" else "calibration"
            checks = {
                "closed_support": closed
                >= int(support[f"minimum_{role_prefix}_closed_rows_per_side"]),
                "residual_support": residual
                >= int(support[f"minimum_{role_prefix}_residual_rows_per_side"]),
                "market_support": markets
                >= int(support["minimum_unique_markets_per_role_and_side"]),
                "target_sign_support": positive > 0 and negative > 0,
            }
            support_passed = support_passed and all(checks.values())
            support_rows.append(
                {
                    "role": role,
                    "side": side,
                    "row_count": len(selected),
                    "market_count": markets,
                    "closed_before_settlement_count": closed,
                    "settlement_residual_count": residual,
                    "positive_target_count": positive,
                    "negative_target_count": negative,
                    "target_sum": sum(
                        float(
                            row[
                                "runtime_policy_after_cost_net_pnl_per_contract"
                            ]
                        )
                        for row in selected
                    ),
                    "support_checks": checks,
                    "support_passed": all(checks.values()),
                }
            )
    checks = {
        "row_count": len(rows) == expected_rows,
        "unique_identity": identity_count == len(rows),
        "feature_causality": causality_violations == 0,
        "exit_causality": exit_causality_violations == 0,
        "forbidden_feature_fields": not forbidden,
        "finite_targets": finite_targets,
        "support": support_passed,
        "runtime_policy_source_hashes": (
            runtime_policy_source_hashes()
            == profile["runtime_policy_contract"]["source_function_sha256"]
        ),
    }
    reasons = [f"{name}_gate_failed" for name, passed in checks.items() if not passed]
    report = {
        "schema_version": TARGET_REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "lineage_freeze_manifest": _descriptor(freeze_path),
        "runtime_aligned_rows": _descriptor(corpus_path),
        "market_count": len({str(row["market_id"]) for row in rows}),
        "market_count_by_role": dict(
            sorted(Counter(str(row["role"]) for row in lineage_rows).items())
        ),
        "target_row_count": len(rows),
        "expected_target_row_count": expected_rows,
        "position_lifecycle_class_counts": dict(
            sorted(Counter(row["position_lifecycle_class"] for row in rows).items())
        ),
        "runtime_policy_residual_reason_distribution": dict(
            sorted(lifecycle_reason_counts.items())
        ),
        "support_by_role_and_side": support_rows,
        "feature_causality_violation_count": causality_violations,
        "exit_causality_violation_count": exit_causality_violations,
        "forbidden_decision_time_feature_fields": forbidden,
        "runtime_policy_source_hashes": runtime_policy_source_hashes(),
        "runtime_policy_source_hashes_verified": checks[
            "runtime_policy_source_hashes"
        ],
        "cost_fields_subtracted_exactly_once": [
            "fees",
            "slippage",
            "liquidity_impact",
        ],
        "settlement_timestamp_used": False,
        "historical_oof_opened": False,
        "issue_212_future_outcomes_opened": False,
        "issue_221_paper_outcomes_opened": False,
        "target_corpus_gate_checks": checks,
        "target_corpus_gate_passed": all(checks.values()),
        "target_corpus_gate_reason_codes": reasons,
        **_blocked_safety_fields(),
    }
    report["target_report_id"] = canonical_json_sha256(report)
    return report


def _validate_external_corpus_dir(path: Path, *, profile: dict[str, Any]) -> None:
    root = Path(profile["external_training_corpus_root"]).resolve()
    if path == root or root not in path.parents:
        raise ValueError("direct training corpus must be below /Volumes/PHILIPS/v8")


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
    _verify_pin(path, expected, name)
    return {"path": str(path), "sha256": expected}


def _descriptor(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _verify_pin(path: Path, expected: str, name: str) -> None:
    _require_sha256(expected, f"{name} SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: {actual} != {expected}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: str, name: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _require_git_sha(value: str) -> None:
    if len(value) != 40 or not all(
        character in "0123456789abcdef" for character in value
    ):
        raise ValueError("implementation_commit must be a 40-character git SHA")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL object required: {path}")
                rows.append(value)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _flatten_keys(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        output = []
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.append(name)
            output.extend(_flatten_keys(child, name))
        return output
    if isinstance(value, list):
        return [name for child in value for name in _flatten_keys(child, prefix)]
    return []


def _target_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v6.4 Runtime-Aligned SBC Target Corpus",
        "",
        f"- market_count: `{report['market_count']}`",
        f"- target_row_count: `{report['target_row_count']}`",
        f"- position_lifecycle_class_counts: `{json.dumps(report['position_lifecycle_class_counts'], sort_keys=True)}`",
        f"- feature_causality_violation_count: `{report['feature_causality_violation_count']}`",
        f"- exit_causality_violation_count: `{report['exit_causality_violation_count']}`",
        f"- target_corpus_gate_passed: `{str(report['target_corpus_gate_passed']).lower()}`",
        f"- target_corpus_gate_reason_codes: `{json.dumps(report['target_corpus_gate_reason_codes'])}`",
        "",
        "## Role And Side Support",
        "",
        "| Role | Side | Rows | Closed | Residual | Positive | Negative | Passed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["support_by_role_and_side"]:
        lines.append(
            "| {role} | {side} | {row_count} | {closed_before_settlement_count} | "
            "{settlement_residual_count} | {positive_target_count} | "
            "{negative_target_count} | {support_passed} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Outcome, settlement, and PnL fields are target-stage only and are never included in the decision-time feature map.",
            "Historical #223 OOF and #212/#221 future or paper outcomes are excluded.",
        ]
    )
    return "\n".join(lines)
