"""Historical paper-only corpus builder for regime-conditioned EV v2."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_policy_replay import (
    _safety_report_fields,
    _sha256_file,
    _write_json,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev import (
    CURRENT_75_ROW_REPLAY_RUN_ID,
    LATEST_ONE_HOUR_RECONCILED_RUN_ID,
    REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev_calibration import (
    V2_REQUIRED_VALIDATION_ACTION_FAMILIES,
    V2_REQUIRED_VALIDATION_RESOLVED_OUTCOMES,
    V2_REQUIRED_VALIDATION_SIDES,
    regime_conditioned_ev_v2_calibration_row_identity,
    validate_regime_conditioned_ev_v2_calibration_rows,
)

REGIME_CONDITIONED_EV_V2_CORPUS_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-regime-conditioned-ev-v2-corpus-manifest-v1"
)
REGIME_CONDITIONED_EV_V2_CORPUS_QUALITY_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-regime-conditioned-ev-v2-corpus-quality-v1"
)
SOURCE_MANIFEST_NAMES = {
    "clob_settlement_reconciliation_manifest.json",
    "one_hour_remap_paper_goal_manifest.json",
    "execution_layer_v2_historical_calibration_source_manifest.json",
}
PROHIBITED_RUN_IDS = {
    CURRENT_75_ROW_REPLAY_RUN_ID,
    LATEST_ONE_HOUR_RECONCILED_RUN_ID,
    LATEST_ONE_HOUR_RECONCILED_RUN_ID.removesuffix(
        "-clob-settlement-reconciled"
    ),
}
REQUIRED_ARTIFACT_ALIASES = {
    "paper_intent_log": (
        "paper_intent_log",
        "one_hour_paper_intent_log.jsonl",
    ),
    "paper_fill_log": (
        "paper_fill_log",
        "one_hour_paper_fill_log.jsonl",
    ),
    "settlement_rows": (
        "settlement_evaluation_rows",
        "settlement_evaluation_rows.jsonl",
        "settlement_pnl_rows",
        "settlement_pnl_rows.jsonl",
    ),
}
SIGNAL_TRACE_ALIASES = (
    "signal_trace",
    "fresh_signal_trace_report",
    "signal_trace_report",
    "incremental_fresh_loop/o_v8_paper_fresh_signal_trace.json",
    "o_v8_paper_fresh_signal_trace.json",
)
PAPER_FRESH_LOOP_MANIFEST_ALIASES = (
    "paper_fresh_loop_manifest",
    "o_v8_paper_fresh_loop_manifest.json",
)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2RegimeConditionedEVCorpusConfig:
    run_id: str
    source_roots: tuple[Path | str, ...]
    output_dir: Path | str
    existing_corpus_manifest: Path | str | None = None
    probability_price_tolerance: float = 1e-9
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.source_roots:
            raise ValueError("at least one source root is required")
        roots = tuple(Path(root).expanduser().resolve() for root in self.source_roots)
        object.__setattr__(self, "source_roots", roots)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.existing_corpus_manifest is not None:
            object.__setattr__(
                self,
                "existing_corpus_manifest",
                Path(self.existing_corpus_manifest),
            )
        if (
            not math.isfinite(self.probability_price_tolerance)
            or self.probability_price_tolerance < 0.0
        ):
            raise ValueError(
                "probability_price_tolerance must be finite and non-negative"
            )

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_roots"] = [str(root) for root in self.source_roots]
        payload["output_dir"] = str(self.output_dir)
        if self.existing_corpus_manifest is not None:
            payload["existing_corpus_manifest"] = str(
                self.existing_corpus_manifest
            )
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2RegimeConditionedEVCorpusResult:
    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    quality_report: dict[str, Any]
    manifest: dict[str, Any]


def run_execution_layer_v2_regime_conditioned_ev_corpus_builder(
    config: ExecutionLayerV2RegimeConditionedEVCorpusConfig,
) -> ExecutionLayerV2RegimeConditionedEVCorpusResult:
    """Build a deterministic corpus from immutable completed paper runs."""

    for root in config.source_roots:
        if not root.exists():
            raise FileNotFoundError(f"source root not found: {root}")
    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"corpus output exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    manifest_paths = _discover_source_manifests(config.source_roots)
    source_reports: list[dict[str, Any]] = []
    candidate_envelopes: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        report, envelopes = _ingest_source_manifest(manifest_path)
        source_reports.append(report)
        candidate_envelopes.extend(envelopes)

    full_rows, dedup_report = _deduplicate_envelopes(candidate_envelopes)
    validated_rows, invalid_rows, validator_excluded_rows = (
        validate_regime_conditioned_ev_v2_calibration_rows(
            full_rows,
            source_root=run_dir,
            probability_price_tolerance=config.probability_price_tolerance,
        )
    )
    validated_rows = _sort_rows(
        [_calibration_schema_row(row) for row in validated_rows]
    )
    existing_state = _load_existing_corpus(config.existing_corpus_manifest)
    incremental_rows, incremental_report = _incremental_rows(
        existing_state,
        validated_rows,
    )
    output_rows = incremental_rows if existing_state["present"] else validated_rows
    output_rows = _sort_rows(output_rows)

    artifact_paths = {
        "corpus_rows": run_dir
        / "execution_layer_v2_regime_conditioned_ev_v2_corpus_rows.jsonl",
        "corpus_manifest": run_dir
        / "execution_layer_v2_regime_conditioned_ev_v2_corpus_manifest.json",
        "corpus_quality_report": run_dir
        / "execution_layer_v2_regime_conditioned_ev_v2_corpus_quality_report.json",
        "corpus_quality_summary": run_dir
        / "execution_layer_v2_regime_conditioned_ev_v2_corpus_quality_report.md",
    }
    _write_jsonl(artifact_paths["corpus_rows"], output_rows)
    corpus_sha256 = _sha256_file(artifact_paths["corpus_rows"])
    full_rebuild_sha256 = _rows_sha256(validated_rows)
    incremental_sha256 = _rows_sha256(output_rows)
    incremental_hash_match = incremental_sha256 == full_rebuild_sha256
    quality_report = _build_quality_report(
        config,
        source_reports=source_reports,
        discovered_manifest_paths=manifest_paths,
        rows=output_rows,
        invalid_rows=invalid_rows,
        validator_excluded_rows=validator_excluded_rows,
        dedup_report=dedup_report,
        incremental_report=incremental_report,
        incremental_full_rebuild_hash_match=incremental_hash_match,
    )
    _write_json(artifact_paths["corpus_quality_report"], quality_report)
    _write_text(
        artifact_paths["corpus_quality_summary"],
        execution_layer_v2_regime_conditioned_ev_corpus_quality_to_markdown(
            quality_report
        ),
    )
    artifact_hashes = {
        "corpus_rows": corpus_sha256,
        "corpus_quality_report": _sha256_file(
            artifact_paths["corpus_quality_report"]
        ),
        "corpus_quality_summary": _sha256_file(
            artifact_paths["corpus_quality_summary"]
        ),
    }
    manifest = _build_corpus_manifest(
        config,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        quality_report=quality_report,
        source_reports=source_reports,
    )
    _write_json(artifact_paths["corpus_manifest"], manifest)
    artifact_hashes["corpus_manifest"] = _sha256_file(
        artifact_paths["corpus_manifest"]
    )
    return ExecutionLayerV2RegimeConditionedEVCorpusResult(
        output_dir=run_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        quality_report=quality_report,
        manifest=manifest,
    )


def _discover_source_manifests(roots: tuple[Path, ...]) -> list[Path]:
    discovered = {
        path.resolve()
        for root in roots
        for path in root.rglob("*manifest.json")
        if path.name in SOURCE_MANIFEST_NAMES
    }
    return sorted(discovered, key=str)


def _ingest_source_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _source_report(
            manifest_path,
            source_run_id="",
            included=False,
            reasons=["source_manifest_invalid_json"],
        ), []
    if not isinstance(manifest, dict):
        return _source_report(
            manifest_path,
            source_run_id="",
            included=False,
            reasons=["source_manifest_not_object"],
        ), []
    source_run_id = str(manifest.get("source_run_id") or manifest.get("run_id") or "")
    reasons = _source_manifest_blocking_reasons(manifest, source_run_id)
    source_manifest_artifact = {
        "logical_name": "source_manifest",
        "manifest_key": "__self__",
        "path": manifest_path.resolve(),
        "sha256": _sha256_file(manifest_path),
    }
    resolved_artifacts: dict[str, dict[str, Any]] = {
        "source_manifest": source_manifest_artifact,
    }
    if not reasons:
        for artifact_name, aliases in REQUIRED_ARTIFACT_ALIASES.items():
            resolved, artifact_reasons = _resolve_and_verify_artifact(
                manifest_path,
                manifest,
                aliases,
            )
            reasons.extend(artifact_reasons)
            if resolved is not None:
                resolved_artifacts[artifact_name] = resolved
        trace_artifacts, trace_reasons = _resolve_signal_trace_artifacts(
            manifest_path=manifest_path,
            manifest=manifest,
            source_manifest_artifact=source_manifest_artifact,
        )
        reasons.extend(trace_reasons)
        resolved_artifacts.update(trace_artifacts)
    if reasons:
        return _source_report(
            manifest_path,
            source_run_id=source_run_id,
            included=False,
            reasons=sorted(set(reasons)),
            resolved_artifacts=resolved_artifacts,
        ), []
    try:
        trace_rows = _load_rows(
            resolved_artifacts["signal_trace"]["path"],
            list_keys=("trace_rows", "decision_rows", "rows"),
        )
        intent_rows = _load_rows(resolved_artifacts["paper_intent_log"]["path"])
        fill_rows = _load_rows(resolved_artifacts["paper_fill_log"]["path"])
        settlement_rows = _load_rows(resolved_artifacts["settlement_rows"]["path"])
    except (OSError, ValueError, json.JSONDecodeError):
        return _source_report(
            manifest_path,
            source_run_id=source_run_id,
            included=False,
            reasons=["source_artifact_rows_invalid"],
            resolved_artifacts=resolved_artifacts,
        ), []
    envelopes, row_reasons = _join_source_rows(
        source_run_id=source_run_id,
        source_manifest_path=manifest_path,
        trace_rows=trace_rows,
        intent_rows=intent_rows,
        fill_rows=fill_rows,
        settlement_rows=settlement_rows,
        artifacts=resolved_artifacts,
    )
    report = _source_report(
        manifest_path,
        source_run_id=source_run_id,
        included=True,
        reasons=[],
        resolved_artifacts=resolved_artifacts,
        candidate_row_count=len(envelopes),
        source_fill_row_count=len(fill_rows),
        row_excluded_count=max(0, len(fill_rows) - len(envelopes)),
        row_exclusion_reason_distribution=row_reasons,
    )
    return report, envelopes


def _source_manifest_blocking_reasons(
    manifest: dict[str, Any], source_run_id: str
) -> list[str]:
    reasons: list[str] = []
    manifest_run_id = str(manifest.get("run_id") or "")
    identifiers = {source_run_id, manifest_run_id}
    if identifiers & PROHIBITED_RUN_IDS:
        reasons.append("prohibited_source_run")
    if any(_future_or_holdout_run_id(run_id) for run_id in identifiers):
        reasons.append("future_shadow_or_holdout_run_excluded")
    if bool(manifest.get("synthetic_fixture")) or any(
        "pytest-fixture" in run_id.lower() or "synthetic" in run_id.lower()
        for run_id in identifiers
    ):
        reasons.append("synthetic_fixture_source_excluded")
    if not source_run_id:
        reasons.append("source_run_id_missing")
    required_safety = {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    for field_name, expected in required_safety.items():
        if manifest.get(field_name) is not expected:
            reasons.append(f"source_manifest_safety_flag_invalid:{field_name}")
    completed = bool(
        manifest.get("completed") is True
        or manifest.get("reconciled_final_goal_success") is True
        or manifest.get("final_goal_success") is True
        or "final_goal_success" in manifest
    )
    if not completed:
        reasons.append("source_run_not_completed")
    if manifest.get("provider_fail_fast_stop_triggered") is True:
        reasons.append("source_run_provider_fail_fast_incomplete")
    if int(manifest.get("unresolved_settlement_count") or 0) != 0:
        reasons.append("source_run_has_unresolved_settlements")
    return reasons


def _future_or_holdout_run_id(run_id: str) -> bool:
    lowered = run_id.lower().replace("_", "-")
    return "forward-shadow" in lowered or "holdout" in lowered


def _resolve_and_verify_artifact(
    manifest_path: Path,
    manifest: dict[str, Any],
    aliases: tuple[str, ...],
) -> tuple[dict[str, Any] | None, list[str]]:
    artifact_paths = manifest.get("artifact_paths")
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_paths, dict) or not isinstance(artifact_hashes, dict):
        return None, ["source_manifest_artifact_map_missing"]
    selected_key = next(
        (
            key
            for key, value in artifact_paths.items()
            if key in aliases or Path(str(value)).name in aliases
        ),
        None,
    )
    if selected_key is None:
        return None, [f"required_source_artifact_missing:{aliases[0]}"]
    path = _resolve_source_artifact_path(
        manifest_path, str(artifact_paths[selected_key])
    )
    if path is None:
        return None, [f"source_artifact_not_found:{aliases[0]}"]
    expected_hash = artifact_hashes.get(selected_key)
    if expected_hash is None:
        expected_hash = artifact_hashes.get(path.name)
    if not _is_sha256(expected_hash):
        return None, [f"source_artifact_hash_missing_or_invalid:{aliases[0]}"]
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        return None, [f"source_artifact_hash_mismatch:{aliases[0]}"]
    return {
        "logical_name": aliases[0],
        "manifest_key": selected_key,
        "path": path,
        "sha256": actual_hash,
    }, []


def _resolve_signal_trace_artifacts(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    source_manifest_artifact: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    direct_trace, direct_reasons = _resolve_and_verify_artifact(
        manifest_path,
        manifest,
        SIGNAL_TRACE_ALIASES,
    )
    if direct_trace is not None:
        return {
            "signal_trace": direct_trace,
            "trace_manifest": source_manifest_artifact,
        }, []
    if direct_reasons != ["required_source_artifact_missing:signal_trace"]:
        return {}, direct_reasons

    trace_manifest, trace_manifest_reasons = _resolve_and_verify_artifact(
        manifest_path,
        manifest,
        PAPER_FRESH_LOOP_MANIFEST_ALIASES,
    )
    if trace_manifest is None:
        if trace_manifest_reasons == [
            "required_source_artifact_missing:paper_fresh_loop_manifest"
        ]:
            return {}, direct_reasons
        return {}, trace_manifest_reasons
    try:
        trace_manifest_payload = json.loads(
            trace_manifest["path"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}, ["trace_manifest_invalid_json"]
    if not isinstance(trace_manifest_payload, dict):
        return {}, ["trace_manifest_not_object"]
    nested_trace, nested_trace_reasons = _resolve_and_verify_artifact(
        trace_manifest["path"],
        trace_manifest_payload,
        SIGNAL_TRACE_ALIASES,
    )
    if nested_trace is None:
        return {"trace_manifest": trace_manifest}, nested_trace_reasons
    return {
        "signal_trace": nested_trace,
        "trace_manifest": trace_manifest,
    }, []


def _resolve_source_artifact_path(
    manifest_path: Path, path_text: str
) -> Path | None:
    raw = Path(path_text).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, manifest_path.parent / raw]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def _load_rows(path: Path, *, list_keys: tuple[str, ...] = ("rows",)) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows: Any = payload
        if isinstance(payload, dict):
            rows = next(
                (payload[key] for key in list_keys if isinstance(payload.get(key), list)),
                None,
            )
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"artifact does not contain object rows: {path}")
    return rows


def _join_source_rows(
    *,
    source_run_id: str,
    source_manifest_path: Path,
    trace_rows: list[dict[str, Any]],
    intent_rows: list[dict[str, Any]],
    fill_rows: list[dict[str, Any]],
    settlement_rows: list[dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    traces = _index_rows(trace_rows, _intent_id)
    intents = _index_rows(intent_rows, _intent_id)
    settlements = _index_rows(settlement_rows, _fill_id)
    envelopes: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    market_history: dict[str, list[dict[str, Any]]] = {}
    sorted_fills = sorted(
        fill_rows,
        key=lambda row: (
            _number(row.get("decision_ts")) or float("inf"),
            _fill_id(row),
        ),
    )
    for fill in sorted_fills:
        intent_id = _intent_id(fill)
        fill_id = _fill_id(fill)
        intent = _single_indexed_row(intents, intent_id)
        trace = _single_indexed_row(traces, intent_id)
        settlement = _single_indexed_row(settlements, fill_id)
        missing = []
        if not intent_id or not fill_id:
            missing.append("fill_identity_missing")
        if intent is None:
            missing.append("matching_intent_missing_or_ambiguous")
        if trace is None:
            missing.append("matching_trace_missing_or_ambiguous")
        if settlement is None:
            missing.append("matching_settlement_missing_or_ambiguous")
        if missing:
            reasons.update(missing)
            continue
        row, row_reasons = _calibration_row_from_join(
            source_run_id=source_run_id,
            trace=trace,
            intent=intent,
            fill=fill,
            settlement=settlement,
            artifacts=artifacts,
            market_history=market_history,
        )
        if row_reasons:
            reasons.update(row_reasons)
            continue
        envelopes.append(
            {
                "row": row,
                "source_manifest_path": str(source_manifest_path),
                "source_manifest_sha256": _sha256_file(source_manifest_path),
            }
        )
        market_history.setdefault(row["market_id"], []).append(row)
    return envelopes, dict(sorted(reasons.items()))


def _calibration_row_from_join(
    *,
    source_run_id: str,
    trace: dict[str, Any],
    intent: dict[str, Any],
    fill: dict[str, Any],
    settlement: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    market_history: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    intent_id = _intent_id(intent)
    fill_id = _fill_id(fill)
    market_id = str(intent.get("market_id") or fill.get("market_id") or "")
    decision_ts = _number(intent.get("decision_ts") or trace.get("decision_ts"))
    selected_action = str(
        intent.get("execution_guarded_action") or intent.get("selected_action") or ""
    )
    selected_side = str(
        intent.get("execution_guarded_side") or intent.get("selected_side") or ""
    ).upper()
    action_family = str(
        intent.get("execution_guarded_family") or intent.get("action_family") or ""
    )
    settlement_status = str(
        settlement.get("resolution_status")
        or settlement.get("settlement_status")
        or ""
    ).lower()
    resolved_outcome = str(settlement.get("resolved_outcome") or "").upper()
    if settlement_status not in {"normal", "resolved", "settled"}:
        reasons.append("ambiguous_or_unresolved_settlement")
    if resolved_outcome not in V2_REQUIRED_VALIDATION_RESOLVED_OUTCOMES:
        reasons.append("resolved_outcome_missing_or_invalid")
    outcome_observed_at_ts = _first_number(
        settlement,
        fields=(
            "outcome_observed_at_ts",
            "resolution_observed_at_ts",
            "target_observed_at_ts",
        ),
    )
    observation_time_source = str(
        settlement.get("outcome_observation_time_source")
        or (
            "artifact_recorded"
            if outcome_observed_at_ts is not None
            else "not_recorded_historical"
        )
    )
    market_close_ts = _number(
        trace.get("market_end_ts")
        or intent.get("market_end_ts")
        or settlement.get("market_close_ts")
    )
    max_input_ts = _decision_feature_max_input_ts(trace, intent)
    if not market_id:
        reasons.append("market_id_missing")
    if decision_ts is None:
        reasons.append("decision_ts_missing")
    if market_close_ts is None:
        reasons.append("market_close_ts_missing")
    if max_input_ts is None:
        reasons.append("decision_time_max_input_ts_missing")
    canonical_score = _canonical_score_for_action(trace, intent, selected_action)
    action_margin = _first_number(trace, intent, fields=("action_score_margin", "score_margin"))
    momentum = _first_number(trace, intent, fill, fields=("btc_momentum",))
    reference_distance = _first_number(
        trace,
        intent,
        fill,
        fields=("reference_price_to_beat_distance_at_decision",),
    )
    p_up = _first_number(intent, trace, fields=("p_up",))
    p_down = _first_number(intent, trace, fields=("p_down",))
    selected_probability = p_down if selected_side == "DOWN" else p_up
    execution_price = _first_number(
        intent,
        fields=("entry_ask", "paper_limit_price", "execution_price"),
    )
    spread = _first_number(intent, trace, fields=("spread_bps",))
    staleness = _first_number(intent, trace, fields=("book_staleness_ms",))
    queue = _first_number(intent, trace, fields=("queue_fill_proxy",))
    time_to_close = _first_number(intent, trace, fields=("time_to_close_seconds",))
    required_values = {
        "canonical_o_action_score": canonical_score,
        "action_score_margin": action_margin,
        "btc_momentum": momentum,
        "reference_price_to_beat_distance_at_decision": reference_distance,
        "selected_side_probability": selected_probability,
        "execution_price": execution_price,
        "spread_bps": spread,
        "book_staleness_ms": staleness,
        "queue_fill_proxy": queue,
        "time_to_close_seconds": time_to_close,
    }
    reasons.extend(
        f"decision_time_feature_missing:{field_name}"
        for field_name, value in required_values.items()
        if value is None
    )
    target = _number(settlement.get("settlement_pnl"))
    if target is None:
        reasons.append("settlement_net_return_target_missing")
    pre_state = intent.get("pre_decision_exposure_state")
    if not isinstance(pre_state, dict):
        reasons.append("pre_entry_exposure_state_missing")
        pre_state = {}
    market_exposure = pre_state.get("current_market_exposure_by_market_id")
    if not isinstance(market_exposure, dict):
        reasons.append("pre_entry_market_exposure_map_missing")
        market_exposure = {}
    prior_rows = market_history.get(market_id, [])
    cumulative_exposure = _number(market_exposure.get(market_id))
    if cumulative_exposure is None and prior_rows:
        reasons.append("pre_entry_market_exposure_value_missing_for_reentry")
    elif cumulative_exposure is None:
        cumulative_exposure = 0.0
    if reasons:
        return {}, sorted(set(reasons))
    same_side_reentry = any(row["selected_side"] == selected_side for row in prior_rows)
    side_flip = any(row["selected_side"] != selected_side for row in prior_rows)
    trace_id = str(
        trace.get("o_v8_paper_fresh_signal_trace_row_hash")
        or trace.get("trace_row_id")
        or canonical_json_sha256(trace)
    )
    settlement_id = str(
        settlement.get("settlement_evaluation_row_hash")
        or settlement.get("settlement_pnl_row_hash")
        or settlement.get("settlement_row_id")
        or canonical_json_sha256(settlement)
    )
    row_identity = regime_conditioned_ev_v2_calibration_row_identity(
        source_run_id=source_run_id,
        market_id=market_id,
        decision_ts=decision_ts,
        selected_action=selected_action,
        source_intent_id=intent_id,
        source_fill_id=fill_id,
    )
    settlement_source = _canonical_settlement_source(
        str(settlement.get("resolution_source_type") or "")
    )
    if settlement_source is None:
        return {}, ["settlement_source_not_approved_read_only"]
    settlement_artifact = artifacts["settlement_rows"]
    row = {
        "source_run_id": source_run_id,
        "source_intent_id": intent_id,
        "source_fill_id": fill_id,
        "row_identity": row_identity,
        "source_lineage": {
            "source_manifest_path": str(artifacts["source_manifest"]["path"]),
            "source_manifest_sha256": artifacts["source_manifest"]["sha256"],
            "trace_manifest_path": str(artifacts["trace_manifest"]["path"]),
            "trace_manifest_sha256": artifacts["trace_manifest"]["sha256"],
            "trace_artifact_path": str(artifacts["signal_trace"]["path"]),
            "trace_artifact_sha256": artifacts["signal_trace"]["sha256"],
            "trace_row_id": trace_id,
            "intent_artifact_path": str(artifacts["paper_intent_log"]["path"]),
            "intent_artifact_sha256": artifacts["paper_intent_log"]["sha256"],
            "fill_artifact_path": str(artifacts["paper_fill_log"]["path"]),
            "fill_artifact_sha256": artifacts["paper_fill_log"]["sha256"],
            "settlement_artifact_path": str(settlement_artifact["path"]),
            "settlement_artifact_sha256": settlement_artifact["sha256"],
            "settlement_row_id": settlement_id,
        },
        "market_id": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": max_input_ts,
        "market_close_ts": market_close_ts,
        "selected_side": selected_side,
        "selected_action": selected_action,
        "action_family": action_family,
        "decision_time_features": {
            **required_values,
            "selected_side_probability_minus_execution_price": (
                selected_probability - execution_price
            ),
            "entry_index_within_market": len(prior_rows) + 1,
            "cumulative_market_exposure_before_entry": cumulative_exposure,
            "same_side_reentry": int(same_side_reentry),
            "side_flip": int(side_flip),
        },
        "target_net_return_after_cost": target,
        "target_provenance": {
            "source_type": settlement_source,
            "source_artifact_path": str(settlement_artifact["path"]),
            "source_artifact_sha256": settlement_artifact["sha256"],
            "resolution_status": "resolved",
            "resolved_outcome": resolved_outcome,
            "outcome_observed_after_market_close": True,
            "outcome_observation_time_source": observation_time_source,
            "outcome_observed_at_ts": outcome_observed_at_ts,
        },
    }
    return row, []


def _canonical_score_for_action(
    trace: dict[str, Any], intent: dict[str, Any], selected_action: str
) -> float | None:
    direct = _first_number(
        trace,
        intent,
        fields=("canonical_o_action_score", "selected_action_score"),
    )
    if direct is not None:
        return direct
    canonical_action = str(
        trace.get("canonical_selected_action")
        or intent.get("source_selected_action")
        or ""
    )
    if canonical_action != selected_action:
        return None
    return _first_number(
        trace,
        intent,
        fields=("canonical_corrected_score", "source_model_score"),
    )


def _decision_feature_max_input_ts(
    trace: dict[str, Any], intent: dict[str, Any]
) -> float | None:
    values: list[float] = []
    for payload in (trace, intent):
        explicit = _number(payload.get("decision_time_regime_feature_max_input_ts"))
        if explicit is not None:
            values.append(explicit)
        for value in payload.values():
            if isinstance(value, dict):
                nested = _number(value.get("max_input_ts"))
                if nested is not None:
                    values.append(nested)
    return max(values) if values else None


def _canonical_settlement_source(raw_source: str) -> str | None:
    lowered = raw_source.lower()
    if "clob" in lowered:
        return "polymarket_clob_read_only_settlement"
    if "gamma" in lowered:
        return "polymarket_gamma_read_only_settlement"
    if raw_source == "paper_ledger_read_only_settlement_reconciliation":
        return raw_source
    return None


def _index_rows(
    rows: list[dict[str, Any]], key_function
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = key_function(row)
        if key:
            result.setdefault(key, []).append(row)
    return result


def _single_indexed_row(
    index: dict[str, list[dict[str, Any]]], key: str
) -> dict[str, Any] | None:
    rows = index.get(key, [])
    return rows[0] if len(rows) == 1 else None


def _intent_id(row: dict[str, Any]) -> str:
    return str(
        row.get("paper_fresh_order_intent_id")
        or row.get("paper_intent_id")
        or row.get("intent_id")
        or ""
    )


def _fill_id(row: dict[str, Any]) -> str:
    return str(
        row.get("paper_fresh_fill_id")
        or row.get("paper_fill_id")
        or row.get("fill_id")
        or ""
    )


def _first_number(*payloads: dict[str, Any], fields: tuple[str, ...]) -> float | None:
    for payload in payloads:
        for field_name in fields:
            value = _number(payload.get(field_name))
            if value is not None:
                return value
    return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _deduplicate_envelopes(
    envelopes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_identity: dict[str, dict[str, Any]] = {}
    conflict_identities: set[str] = set()
    exact_duplicates: list[dict[str, Any]] = []
    supplemental_fill_duplicates: list[dict[str, Any]] = []
    fill_origins: dict[tuple[str, str], set[str]] = {}
    for envelope in sorted(
        envelopes,
        key=lambda item: (
            item["row"]["row_identity"],
            item["source_manifest_path"],
        ),
    ):
        row = envelope["row"]
        identity = row["row_identity"]
        row_hash = _economic_row_hash(row)
        existing = by_identity.get(identity)
        if existing is None:
            by_identity[identity] = envelope
        elif _economic_row_hash(existing["row"]) == row_hash:
            exact_duplicates.append(
                {
                    "row_identity": identity,
                    "source_manifest_path": envelope["source_manifest_path"],
                }
            )
        else:
            conflict_identities.add(identity)
        fill_key = (row["source_run_id"], row["source_fill_id"])
        origins = fill_origins.setdefault(fill_key, set())
        origins.add(envelope["source_manifest_path"])
        if len(origins) > 1:
            supplemental_fill_duplicates.append(
                {
                    "source_run_id": fill_key[0],
                    "source_fill_id": fill_key[1],
                    "source_manifest_paths": sorted(origins),
                }
            )
    rows = [
        envelope["row"]
        for identity, envelope in sorted(by_identity.items())
        if identity not in conflict_identities
    ]
    legitimate_repeated_entries = _legitimate_repeated_entry_count(rows)
    return rows, {
        "candidate_row_count": len(envelopes),
        "deduplicated_row_count": len(rows),
        "exact_duplicate_count": len(exact_duplicates),
        "exact_duplicates": exact_duplicates,
        "supplemental_duplicate_fill_count": len(supplemental_fill_duplicates),
        "supplemental_duplicate_fills": supplemental_fill_duplicates,
        "conflicting_identity_count": len(conflict_identities),
        "conflicting_row_identities": sorted(conflict_identities),
        "legitimate_repeated_entry_count": legitimate_repeated_entries,
    }


def _economic_row_hash(row: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(row))
    payload.pop("source_lineage", None)
    target_provenance = payload.get("target_provenance")
    if isinstance(target_provenance, dict):
        target_provenance.pop("source_artifact_path", None)
        target_provenance.pop("source_artifact_sha256", None)
    return canonical_json_sha256(payload)


def _legitimate_repeated_entry_count(rows: list[dict[str, Any]]) -> int:
    counts = Counter((row["source_run_id"], row["market_id"]) for row in rows)
    return sum(max(0, count - 1) for count in counts.values())


def _load_existing_corpus(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"present": False, "rows": [], "manifest": {}}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"existing corpus manifest not found: {resolved}")
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    artifact_paths = manifest.get("artifact_paths", {})
    artifact_hashes = manifest.get("artifact_hashes", {})
    rows_path_text = artifact_paths.get("corpus_rows")
    expected_hash = artifact_hashes.get("corpus_rows")
    if not rows_path_text or not _is_sha256(expected_hash):
        raise ValueError("existing corpus manifest row lineage is incomplete")
    rows_path = _resolve_source_artifact_path(resolved, str(rows_path_text))
    if rows_path is None or _sha256_file(rows_path) != expected_hash:
        raise ValueError("existing corpus rows hash mismatch")
    rows = _load_rows(rows_path)
    return {"present": True, "rows": rows, "manifest": manifest}


def _incremental_rows(
    existing_state: dict[str, Any], full_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not existing_state["present"]:
        return full_rows, {
            "incremental_mode": False,
            "existing_row_count": 0,
            "appended_row_count": len(full_rows),
            "existing_rows_preserved": True,
        }
    existing_rows = existing_state["rows"]
    full_by_identity = {row["row_identity"]: row for row in full_rows}
    existing_preserved = all(
        row.get("row_identity") in full_by_identity
        and canonical_json_sha256(row)
        == canonical_json_sha256(full_by_identity[row["row_identity"]])
        for row in existing_rows
    )
    existing_identities = {row.get("row_identity") for row in existing_rows}
    new_rows = [
        row for row in full_rows if row["row_identity"] not in existing_identities
    ]
    combined = _sort_rows([*existing_rows, *new_rows])
    return combined, {
        "incremental_mode": True,
        "existing_row_count": len(existing_rows),
        "appended_row_count": len(new_rows),
        "existing_rows_preserved": existing_preserved,
    }


def _build_quality_report(
    config: ExecutionLayerV2RegimeConditionedEVCorpusConfig,
    *,
    source_reports: list[dict[str, Any]],
    discovered_manifest_paths: list[Path],
    rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    validator_excluded_rows: list[dict[str, Any]],
    dedup_report: dict[str, Any],
    incremental_report: dict[str, Any],
    incremental_full_rebuild_hash_match: bool,
) -> dict[str, Any]:
    included_sources = [report for report in source_reports if report["included"]]
    excluded_sources = [report for report in source_reports if not report["included"]]
    exclusion_reasons: Counter[str] = Counter()
    row_exclusion_reasons: Counter[str] = Counter()
    for report in excluded_sources:
        exclusion_reasons.update(report["blocking_reason_codes"])
    for report in source_reports:
        row_exclusion_reasons.update(report["row_exclusion_reason_distribution"])
    for invalid in invalid_rows:
        row_exclusion_reasons.update(invalid["reason_codes"])
    row_exclusion_reasons.update(
        "validator_excluded_source_run" for _ in validator_excluded_rows
    )
    coverage = _coverage(rows)
    unique_market_count = len({row["market_id"] for row in rows})
    violations = len(invalid_rows) + dedup_report["conflicting_identity_count"]
    smoke = bool(
        len(rows) >= 150
        and unique_market_count >= 30
        and _required_coverage_present(coverage)
        and violations == 0
        and incremental_full_rebuild_hash_match
    )
    initial = bool(
        len(rows) >= 1_000
        and unique_market_count >= 200
        and _required_coverage_present(coverage)
        and _chronological_span_sufficient(rows)
        and violations == 0
        and incremental_full_rebuild_hash_match
    )
    preferred = bool(
        len(rows) >= 3_000
        and unique_market_count >= 500
        and _required_coverage_present(coverage)
        and all(
            coverage["by_anchor_regime"].get(regime, 0) > 0
            for regime in ("UP", "DOWN", "MIXED_OR_NEUTRAL")
        )
        and violations == 0
        and incremental_full_rebuild_hash_match
    )
    readiness_reasons = []
    if not smoke:
        readiness_reasons.append("minimum_protocol_smoke_not_met")
    if not initial:
        readiness_reasons.append("initial_real_calibration_candidate_not_met")
    if dedup_report["conflicting_identity_count"]:
        readiness_reasons.append("conflicting_duplicate_rows_present")
    if invalid_rows:
        readiness_reasons.append("invalid_calibration_rows_present")
    if not incremental_report["existing_rows_preserved"]:
        readiness_reasons.append("existing_incremental_rows_not_preserved")
    if not incremental_full_rebuild_hash_match:
        readiness_reasons.append("incremental_full_rebuild_hash_mismatch")
    report = {
        "schema_version": REGIME_CONDITIONED_EV_V2_CORPUS_QUALITY_SCHEMA_VERSION,
        "run_id": config.run_id,
        "source_manifest_discovered_count": len(discovered_manifest_paths),
        "source_manifest_paths": [str(path) for path in discovered_manifest_paths],
        "source_run_included_count": len(included_sources),
        "source_run_excluded_count": len(excluded_sources),
        "source_run_ids_included": sorted(
            {report["source_run_id"] for report in included_sources}
        ),
        "source_run_ids_excluded": sorted(
            {report["source_run_id"] for report in excluded_sources}
        ),
        "source_exclusion_reason_distribution": dict(
            sorted(exclusion_reasons.items())
        ),
        "source_ingestion_reports": source_reports,
        "eligible_row_count": len(rows),
        "source_fill_row_count": sum(
            report["source_fill_row_count"] for report in included_sources
        ),
        "excluded_row_count": max(
            0,
            sum(report["source_fill_row_count"] for report in included_sources)
            - len(rows),
        ),
        "invalid_row_count": len(invalid_rows),
        "target_observation_time_contract": {
            "exact_settlement_timestamp_required": False,
            "resolved_official_outcome_required": True,
            "historical_missing_outcome_observation_timestamp_allowed": True,
            "recorded_outcome_observation_timestamp_must_follow_market_close": True,
        },
        "row_exclusion_reason_distribution": dict(
            sorted(row_exclusion_reasons.items())
        ),
        "unique_market_count": unique_market_count,
        "decision_time_start": min(
            (row["decision_ts"] for row in rows), default=None
        ),
        "decision_time_end": max((row["decision_ts"] for row in rows), default=None),
        "coverage": coverage,
        "feature_coverage": _feature_coverage(rows),
        "provenance_coverage": {
            "row_count": len(rows),
            "verified_row_count": len(rows),
            "violation_count": len(invalid_rows),
        },
        "deduplication": dedup_report,
        "incremental_build": incremental_report,
        "incremental_full_rebuild_hash_match": incremental_full_rebuild_hash_match,
        "minimum_protocol_smoke_passed": smoke,
        "initial_real_calibration_candidate_passed": initial,
        "preferred_robust_corpus_passed": preferred,
        "existing_v2_calibration_protocol_run_allowed": initial,
        "corpus_ready": smoke and not readiness_reasons,
        "readiness_blocking_reason_codes": sorted(set(readiness_reasons)),
        "real_frozen_artifact_created": False,
        "future_shadow_run_started": False,
        "diagnostic_only": True,
        "production_gate_implemented": False,
        **_safety_report_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "by_side": dict(sorted(Counter(row["selected_side"] for row in rows).items())),
        "by_action": dict(sorted(Counter(row["selected_action"] for row in rows).items())),
        "by_action_family": dict(
            sorted(Counter(row["action_family"] for row in rows).items())
        ),
        "by_resolved_outcome": dict(
            sorted(
                Counter(
                    row["target_provenance"]["resolved_outcome"] for row in rows
                ).items()
            )
        ),
        "by_anchor_regime": dict(
            sorted(Counter(_anchor_regime(row) for row in rows).items())
        ),
    }


def _anchor_regime(row: dict[str, Any]) -> str:
    features = row["decision_time_features"]
    momentum = float(features["btc_momentum"])
    reference_distance = float(
        features["reference_price_to_beat_distance_at_decision"]
    )
    if momentum > 0.0 and reference_distance > 0.0:
        return "UP"
    if momentum < 0.0 and reference_distance < 0.0:
        return "DOWN"
    return "MIXED_OR_NEUTRAL"


def _feature_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        feature: {
            "available_count": sum(
                row["decision_time_features"].get(feature) is not None for row in rows
            ),
            "missing_count": sum(
                row["decision_time_features"].get(feature) is None for row in rows
            ),
        }
        for features in REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS.values()
        for feature in features
    }


def _required_coverage_present(coverage: dict[str, dict[str, int]]) -> bool:
    return bool(
        all(coverage["by_side"].get(value, 0) > 0 for value in V2_REQUIRED_VALIDATION_SIDES)
        and all(
            coverage["by_action_family"].get(value, 0) > 0
            for value in V2_REQUIRED_VALIDATION_ACTION_FAMILIES
        )
        and all(
            coverage["by_resolved_outcome"].get(value, 0) > 0
            for value in V2_REQUIRED_VALIDATION_RESOLVED_OUTCOMES
        )
    )


def _chronological_span_sufficient(rows: list[dict[str, Any]]) -> bool:
    return bool(
        len({row["market_id"] for row in rows}) >= 2
        and len({row["decision_ts"] for row in rows}) >= 2
        and min(row["decision_ts"] for row in rows)
        < max(row["decision_ts"] for row in rows)
    )


def _source_report(
    manifest_path: Path,
    *,
    source_run_id: str,
    included: bool,
    reasons: list[str],
    resolved_artifacts: dict[str, dict[str, Any]] | None = None,
    candidate_row_count: int = 0,
    source_fill_row_count: int = 0,
    row_excluded_count: int = 0,
    row_exclusion_reason_distribution: dict[str, int] | None = None,
) -> dict[str, Any]:
    artifacts = resolved_artifacts or {}
    source_manifest = artifacts.get("source_manifest")
    trace_manifest = artifacts.get("trace_manifest")
    signal_trace = artifacts.get("signal_trace")
    trace_chain_verified = bool(
        source_manifest is not None
        and trace_manifest is not None
        and signal_trace is not None
    )
    trace_resolution_mode = None
    if trace_chain_verified:
        trace_resolution_mode = (
            "direct_manifest"
            if source_manifest["path"] == trace_manifest["path"]
            else "nested_manifest_chain"
        )
    return {
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": (
            _sha256_file(manifest_path) if manifest_path.is_file() else None
        ),
        "source_run_id": source_run_id,
        "included": included,
        "blocking_reason_codes": reasons,
        "signal_trace_resolution_mode": trace_resolution_mode,
        "signal_trace_manifest_chain_verified": trace_chain_verified,
        "resolved_artifacts": {
            name: {
                "path": str(artifact["path"]),
                "sha256": artifact["sha256"],
            }
            for name, artifact in sorted(artifacts.items())
        },
        "candidate_row_count": candidate_row_count,
        "source_fill_row_count": source_fill_row_count,
        "row_excluded_count": row_excluded_count,
        "row_exclusion_reason_distribution": (
            row_exclusion_reason_distribution or {}
        ),
    }


def _build_corpus_manifest(
    config: ExecutionLayerV2RegimeConditionedEVCorpusConfig,
    *,
    artifact_paths: dict[str, Path],
    artifact_hashes: dict[str, str],
    quality_report: dict[str, Any],
    source_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    builder_config = config.to_dict()
    manifest = {
        "schema_version": REGIME_CONDITIONED_EV_V2_CORPUS_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "builder_config": builder_config,
        "builder_config_sha256": canonical_json_sha256(builder_config),
        "corpus_sha256": artifact_hashes["corpus_rows"],
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_hashes": dict(artifact_hashes),
        "source_run_ids": quality_report["source_run_ids_included"],
        "source_artifact_paths_and_hashes": [
            {
                "source_run_id": report["source_run_id"],
                "source_manifest_path": report["source_manifest_path"],
                "source_manifest_sha256": report["source_manifest_sha256"],
                "resolved_artifacts": report["resolved_artifacts"],
            }
            for report in source_reports
        ],
        "included_row_count": quality_report["eligible_row_count"],
        "excluded_row_count": quality_report["excluded_row_count"],
        "exclusion_reason_distribution": quality_report[
            "row_exclusion_reason_distribution"
        ],
        "unique_market_count": quality_report["unique_market_count"],
        "decision_time_start": quality_report["decision_time_start"],
        "decision_time_end": quality_report["decision_time_end"],
        "coverage": quality_report["coverage"],
        "feature_coverage": quality_report["feature_coverage"],
        "provenance_coverage": quality_report["provenance_coverage"],
        "target_observation_time_contract": quality_report[
            "target_observation_time_contract"
        ],
        "duplicate_conflict_summary": quality_report["deduplication"],
        "readiness_milestones": {
            "minimum_protocol_smoke_passed": quality_report[
                "minimum_protocol_smoke_passed"
            ],
            "initial_real_calibration_candidate_passed": quality_report[
                "initial_real_calibration_candidate_passed"
            ],
            "preferred_robust_corpus_passed": quality_report[
                "preferred_robust_corpus_passed"
            ],
        },
        "corpus_ready": quality_report["corpus_ready"],
        "real_frozen_artifact_created": False,
        "future_shadow_run_started": False,
        "diagnostic_only": True,
        **_safety_report_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    return manifest


def execution_layer_v2_regime_conditioned_ev_corpus_quality_to_markdown(
    report: dict[str, Any],
) -> str:
    reasons = report["readiness_blocking_reason_codes"] or ["none"]
    return "\n".join(
        [
            "# Regime-Conditioned EV v2 Historical Corpus Quality",
            "",
            f"- eligible rows: `{report['eligible_row_count']}`",
            f"- unique markets: `{report['unique_market_count']}`",
            f"- included source runs: `{report['source_run_included_count']}`",
            f"- excluded source runs: `{report['source_run_excluded_count']}`",
            f"- duplicate conflicts: `{report['deduplication']['conflicting_identity_count']}`",
            f"- incremental/full hash match: `{report['incremental_full_rebuild_hash_match']}`",
            f"- minimum protocol smoke: `{report['minimum_protocol_smoke_passed']}`",
            f"- initial real calibration candidate: `{report['initial_real_calibration_candidate_passed']}`",
            f"- preferred robust corpus: `{report['preferred_robust_corpus_passed']}`",
            "- exact settlement timestamp required: `false`",
            "- resolved official outcome required: `true`",
            "- real frozen artifact created: `false`",
            "- future shadow started: `false`",
            "",
            "## Readiness Blocking Reasons",
            "",
            *[f"- `{reason}`" for reason in reasons],
            "",
        ]
    )


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row["decision_ts"],
            row["market_id"],
            row["row_identity"],
        ),
    )


def _calibration_schema_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.pop("row_index", None)
    payload.pop("resolved_outcome", None)
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    return canonical_json_sha256(_sort_rows(rows))


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


__all__ = [
    "REGIME_CONDITIONED_EV_V2_CORPUS_MANIFEST_SCHEMA_VERSION",
    "REGIME_CONDITIONED_EV_V2_CORPUS_QUALITY_SCHEMA_VERSION",
    "ExecutionLayerV2RegimeConditionedEVCorpusConfig",
    "ExecutionLayerV2RegimeConditionedEVCorpusResult",
    "execution_layer_v2_regime_conditioned_ev_corpus_quality_to_markdown",
    "run_execution_layer_v2_regime_conditioned_ev_corpus_builder",
]
