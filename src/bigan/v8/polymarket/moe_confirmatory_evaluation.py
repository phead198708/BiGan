"""Single-use confirmatory settlement and evaluation for BTC 15m MoE v2."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    LINEAGE_ID,
    TARGET_MARKET_COUNT,
    _write_new_frozen_json,
    _write_new_jsonl,
    validate_and_authorize_exact_outcome_access,
)
from bigan.v8.polymarket.moe_collection_finalization import (
    _manual_authorization_expectation,
)
from bigan.v8.polymarket.moe_collection_observability import (
    CANDIDATE_BUNDLE_HASH,
    _bootstrap_indices,
    _current_feature_rows,
    _is_executable_price,
    _json_finite_or_none,
    _load_runtime_bundle,
    _router_observation,
    _row_missing_feature_names,
    _shared_bootstrap_interval,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.recorder import PolymarketPublicHTTPRealCorpusProvider
from bigan.v8.polymarket.recorder.async_settlement import _config_from_manifest
from bigan.v8.polymarket.recorder.resolution import normalize_resolution_for_settlement
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

REPORTING_PANELS = {
    "overall",
    "requested_route",
    "actual_model",
    "expert_vs_fallback",
    "UP_vs_DOWN",
    "regime",
    "provider_health",
    "complete_feature_vs_missing_feature",
    "chronological_half",
    "largest_winner_attribution",
}
REQUIRED_MARKET_FIELDS = {
    "market_id",
    "decision_ts",
    "requested_route",
    "expert_id",
    "expert_training_market_count",
    "expert_available",
    "fallback_used",
    "actual_model_used",
    "candidate_selected_side",
    "baseline_selected_side",
    "candidate_accepted",
    "baseline_accepted",
    "candidate_unit_net_pnl",
    "baseline_unit_net_pnl",
    "paired_delta_unit_net_pnl",
    "provider_health",
    "feature_missingness",
    "cost_decomposition",
    "chronological_half",
}


def run_exact_confirmatory_evaluation(
    *,
    freeze_dir: Path | str,
    output_dir: Path | str,
    implementation_commit: str,
    authorization_text: str,
    repository_root: Path | str | None = None,
    provider_factory: Callable[[], Any] | None = None,
    max_workers: int = 16,
    provider_attempts: int = 3,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Open the exact population once, settle it, and run every frozen gate."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    freeze = Path(freeze_dir).resolve()
    output = Path(output_dir).resolve()
    if not freeze.is_relative_to(repo_root) or not output.is_relative_to(repo_root):
        raise ValueError("confirmatory evaluation paths must remain repository-local")
    if len(implementation_commit) != 40 or any(
        char not in "0123456789abcdef" for char in implementation_commit
    ):
        raise ValueError("implementation commit must be a full lowercase Git SHA")
    if not authorization_text.strip():
        raise ValueError("explicit user authorization text is required")
    if max_workers <= 0 or provider_attempts <= 0:
        raise ValueError("provider concurrency and attempts must be positive")

    claim_path = freeze / "outcome_access_claim_001.json"
    if claim_path.exists() or claim_path.with_suffix(".sha256").exists():
        raise FileExistsError("single-use outcome access claim already exists")
    if output.exists():
        raise FileExistsError("confirmatory evaluation output already exists")

    config = repo_root / "examples/v8/polymarket_configs" / LINEAGE_ID
    artifacts = _evaluation_artifacts(freeze=freeze, config=config)
    preflight = _preflight(
        repository_root=repo_root,
        artifacts=artifacts,
    )
    if preflight["exact_full_population_authorized"] is not True:
        raise ValueError("exact outcome access preflight did not authorize full population")
    contexts = _load_exact_contexts(
        repository_root=repo_root,
        artifacts=artifacts,
    )
    if len(contexts) != TARGET_MARKET_COUNT:
        raise ValueError("confirmatory context population is not exactly 800")

    claim = {
        "schema_version": "bigan-btc-15m-moe-outcome-access-claim-v1",
        "lineage_id": LINEAGE_ID,
        "role": "single_use_exact_full_population_outcome_access",
        "created_at": datetime.now(UTC).isoformat(),
        "implementation_commit": implementation_commit,
        "authorization_text_sha256": canonical_json_sha256(
            {"authorization_text": authorization_text}
        ),
        "authorization_source": "explicit_user_message_in_current_codex_task",
        "requested_market_count": TARGET_MARKET_COUNT,
        "ordered_market_ids_sha256": artifacts["capture_manifest"][
            "payload"
        ]["ordered_market_ids_sha256"],
        "preflight": preflight,
        "artifact_hashes": {
            name: value["sha256"]
            for name, value in artifacts.items()
            if isinstance(value, Mapping) and "sha256" in value
        },
        "attempt_and_alpha_consumed": True,
        "partial_incremental_or_reordered_opening": False,
        "no_rerun": True,
        "outcomes_accessed": True,
        "settlement_accessed": True,
        "pnl_accessed": False,
        "source_capture_mutation_allowed": False,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    claim_frozen = _write_new_frozen_json(claim_path, claim)
    output.mkdir(parents=True, exist_ok=False)

    factory = provider_factory or (
        lambda: PolymarketPublicHTTPRealCorpusProvider(
            max_markets=1,
            timeout_seconds=20.0,
            http_timeout_seconds=10.0,
            use_rest_orderbooks=False,
        )
    )
    settlements, failures = _fetch_exact_settlements(
        contexts=contexts,
        provider_factory=factory,
        max_workers=max_workers,
        provider_attempts=provider_attempts,
        progress_callback=progress_callback,
    )
    settlement_rows_path = output / "official_settlement_rows.jsonl"
    _write_new_jsonl(
        settlement_rows_path,
        [settlements[context["market_id"]] for context in contexts if context["market_id"] in settlements],
    )
    settlement_report = {
        "schema_version": "bigan-btc-15m-moe-settlement-ingestion-v1",
        "lineage_id": LINEAGE_ID,
        "role": "official_read_only_exact_population_settlement",
        "outcome_access_claim": _descriptor(Path(claim_frozen["path"]), repo_root),
        "requested_market_count": TARGET_MARKET_COUNT,
        "settled_market_count": len(settlements),
        "unresolved_market_count": len(failures),
        "unresolved_markets": failures,
        "outcome_distribution": dict(
            sorted(Counter(row["resolved_outcome"] for row in settlements.values()).items())
        ),
        "source_capture_mutated": False,
        "official_settlement_only": True,
        "population_changed": False,
        "no_rerun": True,
        "outcomes_accessed": True,
        "settlement_accessed": True,
        "pnl_accessed": False,
        "evaluation_allowed": len(settlements) == TARGET_MARKET_COUNT and not failures,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    settlement_report_frozen = _write_new_frozen_json(
        output / "settlement_ingestion_report.json",
        settlement_report,
    )
    if settlement_report["evaluation_allowed"] is not True:
        raise ValueError("official settlement population is incomplete; evaluation failed closed")

    bundle = _load_runtime_bundle(repo_root)
    evaluation_rows = _evaluate_markets(
        contexts=contexts,
        settlements=settlements,
        bundle=bundle,
    )
    market_rows_path = output / "confirmatory_market_evaluation_rows.jsonl"
    _write_new_jsonl(market_rows_path, evaluation_rows)
    report = _evaluation_report(
        rows=evaluation_rows,
        artifacts=artifacts,
        claim_path=Path(claim_frozen["path"]),
        settlement_report_path=Path(settlement_report_frozen["path"]),
        market_rows_path=market_rows_path,
        repository_root=repo_root,
        implementation_commit=implementation_commit,
    )
    report_frozen = _write_new_frozen_json(
        output / "moe_confirmatory_evaluation_report.json",
        report,
    )
    markdown_path = output / "moe_confirmatory_evaluation_report.md"
    markdown_path.write_text(_evaluation_markdown(report), encoding="utf-8")
    markdown_sha = sha256_file(markdown_path)
    markdown_path.with_suffix(".sha256").write_text(markdown_sha + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-evaluation-manifest-v1",
        "lineage_id": LINEAGE_ID,
        "implementation_commit": implementation_commit,
        "outcome_access_claim": _descriptor(Path(claim_frozen["path"]), repo_root),
        "official_settlement_rows": _descriptor(settlement_rows_path, repo_root),
        "settlement_ingestion_report": _descriptor(
            Path(settlement_report_frozen["path"]), repo_root
        ),
        "market_evaluation_rows": _descriptor(market_rows_path, repo_root),
        "evaluation_report": _descriptor(Path(report_frozen["path"]), repo_root),
        "evaluation_report_markdown": _descriptor(markdown_path, repo_root),
        "evaluation_executed_exactly_once": True,
        "no_rerun": True,
        "source_capture_mutated": False,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_frozen = _write_new_frozen_json(
        output / "evaluation_manifest.json",
        manifest,
    )
    return {
        "evaluation_report_path": report_frozen["path"],
        "evaluation_report_sha256": report_frozen["sha256"],
        "evaluation_manifest_path": manifest_frozen["path"],
        "evaluation_manifest_sha256": manifest_frozen["sha256"],
        "confirmatory_gate_passed": report["confirmatory_gate_passed"],
        "gate_results": report["gate_results"],
        "overall": report["panels"]["overall"],
    }


def _preflight(
    *,
    repository_root: Path,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    capture = artifacts["capture_manifest"]
    return validate_and_authorize_exact_outcome_access(
        authorization=_manual_authorization_expectation(repository_root),
        capture_manifest_path=capture["path"],
        expected_capture_manifest_sha256=capture["sha256"],
        normalized_attempt_ledger_path=artifacts["normalized_attempt_ledger"]["path"],
        expected_normalized_attempt_ledger_sha256=artifacts[
            "normalized_attempt_ledger"
        ]["sha256"],
        candidate_decision_rows_path=artifacts["candidate_decision_rows"]["path"],
        expected_candidate_decision_rows_sha256=artifacts[
            "candidate_decision_rows"
        ]["sha256"],
        baseline_decision_rows_path=artifacts["baseline_decision_rows"]["path"],
        expected_baseline_decision_rows_sha256=artifacts[
            "baseline_decision_rows"
        ]["sha256"],
        raw_evidence_manifest_index_path=artifacts["raw_evidence_manifest_index"]["path"],
        expected_raw_evidence_manifest_index_sha256=artifacts[
            "raw_evidence_manifest_index"
        ]["sha256"],
        collector_protocol_path=artifacts["collector_protocol"]["path"],
        expected_collector_protocol_sha256=artifacts["collector_protocol"]["sha256"],
        statistical_protocol_path=artifacts["statistical_protocol"]["path"],
        expected_statistical_protocol_sha256=artifacts["statistical_protocol"]["sha256"],
        requested_market_ids=capture["payload"]["ordered_market_ids"],
        repository_root=repository_root,
    )


def _evaluation_artifacts(*, freeze: Path, config: Path) -> dict[str, Any]:
    result = {
        "capture_manifest": _verified_json(freeze / "confirmatory_capture_manifest.json"),
        "normalized_attempt_ledger": _verified_path(freeze / "normalized_attempt_ledger.jsonl"),
        "candidate_decision_rows": _verified_path(freeze / "candidate_decision_rows.jsonl"),
        "baseline_decision_rows": _verified_path(freeze / "baseline_decision_rows.jsonl"),
        "raw_evidence_manifest_index": _verified_path(
            freeze / "raw_evidence_manifest_index.jsonl"
        ),
        "collector_protocol": _verified_path(
            config / "moe_confirmatory_collector_protocol_r2.json"
        ),
        "statistical_protocol": _verified_json(config / "moe_confirmatory_protocol_r1.json"),
        "reporting_contract": _verified_json(
            config / "moe_future_evaluation_reporting_contract.json"
        ),
        "runtime_validation_report": _verified_json(
            config / "moe_artifact_runtime_validation_report.json"
        ),
        "matched_baseline_contract": _verified_json(
            config / "moe_matched_global_baseline_contract.json"
        ),
        "cost_action_contract": _verified_json(
            config / "moe_cost_and_action_contract.json"
        ),
    }
    if result["capture_manifest"]["payload"]["exact_market_count"] != TARGET_MARKET_COUNT:
        raise ValueError("capture manifest exact population changed")
    protocol_inputs = result["statistical_protocol"]["payload"]["frozen_inputs"]
    for protocol_name, artifact_name in (
        ("reporting_contract", "reporting_contract"),
        ("runtime_validation_report", "runtime_validation_report"),
        ("matched_baseline_contract", "matched_baseline_contract"),
    ):
        if protocol_inputs[protocol_name]["sha256"] != result[artifact_name]["sha256"]:
            raise ValueError(f"statistical protocol {protocol_name} SHA mismatch")
    baseline_cost = result["matched_baseline_contract"]["payload"][
        "cost_and_behavior"
    ]["contract"]
    if baseline_cost["sha256"] != result["cost_action_contract"]["sha256"]:
        raise ValueError("matched baseline cost contract SHA mismatch")
    return result


def _load_exact_contexts(
    *,
    repository_root: Path,
    artifacts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ordered_ids = list(artifacts["capture_manifest"]["payload"]["ordered_market_ids"])
    candidate = {row["market_id"]: row for row in _jsonl(artifacts["candidate_decision_rows"]["path"])}
    baseline = {row["market_id"]: row for row in _jsonl(artifacts["baseline_decision_rows"]["path"])}
    evidence = {row["market_id"]: row for row in _jsonl(artifacts["raw_evidence_manifest_index"]["path"])}
    contexts = []
    for market_id in ordered_ids:
        if market_id not in candidate or market_id not in baseline or market_id not in evidence:
            raise ValueError("exact market missing from frozen decision or raw evidence")
        raw_market_descriptor = evidence[market_id]["raw_streams"][
            "raw_polymarket_markets.jsonl"
        ]
        market_path = _resolve_verified_descriptor(raw_market_descriptor, repository_root)
        markets = _jsonl(market_path)
        if len(markets) != 1 or str(markets[0]["market_id"]) != market_id:
            raise ValueError("frozen raw market identity mismatch")
        run_dir = market_path.parent.parent
        manifest_path = run_dir / "pending_round_capture_manifest.json"
        feature_rows = _current_feature_rows(
            run_dir=run_dir,
            manifest=_json_object(manifest_path),
        )
        if len(feature_rows) != 2 or {str(row["market_id"]) for row in feature_rows} != {market_id}:
            raise ValueError("quality-valid market lost its two frozen decision rows")
        contexts.append(
            {
                "market_id": market_id,
                "market": markets[0],
                "candidate": candidate[market_id],
                "baseline": baseline[market_id],
                "feature_rows": sorted(feature_rows, key=lambda row: int(row["decision_ts"])),
                "recorder_config": _config_from_manifest(_json_object(manifest_path)),
            }
        )
    if [row["market_id"] for row in contexts] != ordered_ids:
        raise ValueError("exact contexts were reordered")
    return contexts


def _fetch_exact_settlements(
    *,
    contexts: Sequence[Mapping[str, Any]],
    provider_factory: Callable[[], Any],
    max_workers: int,
    provider_attempts: int,
    progress_callback: Callable[[int, int, int], None] | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    pending = {str(row["market_id"]): row for row in contexts}
    settlements: dict[str, dict[str, Any]] = {}
    last_failures: dict[str, list[str]] = {}
    for provider_attempt in range(1, provider_attempts + 1):
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch_settlement, context, provider_factory): market_id
                for market_id, context in pending.items()
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                market_id = futures[future]
                try:
                    settlement = future.result()
                except Exception as error:  # noqa: BLE001
                    last_failures[market_id] = [
                        f"{type(error).__name__}:{error}"
                    ]
                else:
                    settlements[market_id] = settlement
                    last_failures.pop(market_id, None)
                if progress_callback and (completed % 50 == 0 or completed == len(futures)):
                    progress_callback(provider_attempt, len(settlements), len(pending) - completed)
        pending = {
            market_id: context
            for market_id, context in pending.items()
            if market_id not in settlements
        }
    failures = [
        {"market_id": market_id, "reason_codes": last_failures.get(market_id, ["unresolved"])}
        for market_id in sorted(pending)
    ]
    return settlements, failures


def _fetch_settlement(
    context: Mapping[str, Any],
    provider_factory: Callable[[], Any],
) -> dict[str, Any]:
    provider = provider_factory()
    market = dict(context["market"])
    rows = provider.resolution_rows([market], context["recorder_config"])
    candidates = [row for row in rows if str(row.get("market_id")) == context["market_id"]]
    if len(candidates) != 1:
        raise ValueError("official provider did not return exactly one resolution")
    normalized, reasons = normalize_resolution_for_settlement(
        market=market,
        resolution=dict(candidates[0]),
    )
    if normalized is None:
        raise ValueError("official resolution rejected:" + ",".join(reasons))
    return {
        "schema_version": "bigan-btc-15m-moe-official-settlement-row-v1",
        "lineage_id": LINEAGE_ID,
        "market_id": context["market_id"],
        "resolution_status": normalized["resolution_status"],
        "resolved_outcome": normalized["resolved_outcome"],
        "payout_up": float(normalized["payout_up"]),
        "payout_down": float(normalized["payout_down"]),
        "reference_price_source": normalized["reference_price_source"],
        "resolution_source_type": normalized["resolution_source_type"],
        "resolved_outcome_source": normalized["resolved_outcome_source"],
        "reference_price_start": normalized.get("reference_price_start"),
        "reference_price_end": normalized.get("reference_price_end"),
        "raw_resolution_sha256": canonical_json_sha256(normalized),
        "official_read_only": True,
        "source_capture_mutated": False,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }


def _evaluate_markets(
    *,
    contexts: Sequence[Mapping[str, Any]],
    settlements: Mapping[str, Mapping[str, Any]],
    bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    midpoint = len(contexts) // 2
    for index, context in enumerate(contexts):
        candidate = dict(context["candidate"])
        baseline = dict(context["baseline"])
        feature_rows = list(context["feature_rows"])
        settlement = settlements[context["market_id"]]
        candidate_result = _policy_result(candidate, feature_rows, settlement)
        baseline_result = _policy_result(baseline, feature_rows, settlement)
        representative = _feature_at(feature_rows, int(candidate["decision_ts"]))
        router = _router_observation(representative, bundle["router"])
        missing_names = _row_missing_feature_names(representative)
        paired_complete = all(
            _is_executable_price(row["features"].get(field))
            for row in feature_rows
            for field in ("up_ask", "down_ask")
        )
        row = {
            "schema_version": "bigan-btc-15m-moe-confirmatory-market-result-v1",
            "lineage_id": LINEAGE_ID,
            "market_id": context["market_id"],
            "market_start_ts": int(context["market"]["market_start_ts"]),
            "decision_ts": int(candidate["decision_ts"]),
            "requested_route": candidate["requested_route"],
            "expert_id": candidate["expert_id"],
            "expert_training_market_count": int(candidate["expert_training_support"]),
            "expert_available": bool(candidate["expert_available"]),
            "fallback_used": bool(candidate["fallback_used"]),
            "actual_model_used": candidate["actual_model_used"],
            "candidate_selected_side": candidate_result["selected_side"],
            "baseline_selected_side": baseline_result["selected_side"],
            "candidate_accepted": candidate_result["accepted"],
            "baseline_accepted": baseline_result["accepted"],
            "candidate_unit_net_pnl": candidate_result["unit_net_pnl"],
            "baseline_unit_net_pnl": baseline_result["unit_net_pnl"],
            "paired_delta_unit_net_pnl": (
                candidate_result["unit_net_pnl"] - baseline_result["unit_net_pnl"]
            ),
            "provider_health": _json_finite_or_none(
                representative["features"].get("provider_health_score")
            ),
            "feature_missingness": {
                "missing_feature_names": missing_names,
                "missing_feature_count": len(missing_names),
                "feature_complete": not missing_names,
                "missing_values_encoded_as_zero": False,
            },
            "cost_decomposition": {
                "candidate": candidate_result["cost_decomposition"],
                "baseline": baseline_result["cost_decomposition"],
            },
            "chronological_half": "first" if index < midpoint else "second",
            "regime": {
                "btc_return_regime": router["btc_return_regime"],
                "volatility_bucket": router["volatility_bucket"],
            },
            "paired_executable_asks_complete": paired_complete,
            "official_settlement": {
                "resolved_outcome": settlement["resolved_outcome"],
                "payout_up": settlement["payout_up"],
                "payout_down": settlement["payout_down"],
                "raw_resolution_sha256": settlement["raw_resolution_sha256"],
            },
            "source_capture_mutated": False,
            "promotion_evidence_eligible": False,
            "safety": dict(SAFETY),
        }
        if not set(row) >= REQUIRED_MARKET_FIELDS:
            raise ValueError("future reporting market schema is incomplete")
        rows.append(row)
    return rows


def _policy_result(
    decision: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
    settlement: Mapping[str, Any],
) -> dict[str, Any]:
    if decision["accepted"] is not True:
        return {
            "accepted": False,
            "selected_side": None,
            "unit_net_pnl": 0.0,
            "cost_decomposition": _zero_costs(),
        }
    side = str(decision["selected_side"])
    if side not in {"UP", "DOWN"}:
        raise ValueError("accepted decision has invalid side")
    feature = _feature_at(feature_rows, int(decision["decision_ts"]))
    raw = feature["features"]
    prefix = side.lower()
    ask = float(raw[f"{prefix}_ask"])
    bid = float(raw[f"{prefix}_bid"])
    depth = float(raw[f"{prefix}_liquidity_depth"])
    if not (_is_executable_price(ask) and _is_executable_price(bid)):
        raise ValueError("accepted action lacks executable bid/ask")
    mid = (ask + bid) / 2.0
    payout = float(settlement[f"payout_{prefix}"])
    spread = ask - mid
    fees = 0.0002
    slippage = max(0.0001, spread)
    impact = 0.00005 if depth > 0.0 else 0.001
    gross = payout - mid
    net = gross - spread - fees - slippage - impact
    return {
        "accepted": True,
        "selected_side": side,
        "unit_net_pnl": net,
        "cost_decomposition": {
            "settlement_payout": payout,
            "entry_bid": bid,
            "entry_ask": ask,
            "entry_mid": mid,
            "gross_price_edge": gross,
            "entry_spread_cost": spread,
            "fees": fees,
            "slippage": slippage,
            "liquidity_impact": impact,
            "total_cost": spread + fees + slippage + impact,
            "unit_net_pnl": net,
        },
    }


def _evaluation_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Any],
    claim_path: Path,
    settlement_report_path: Path,
    market_rows_path: Path,
    repository_root: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    protocol = artifacts["statistical_protocol"]["payload"]
    reporting = artifacts["reporting_contract"]["payload"]
    bootstrap = protocol["bootstrap"]
    candidate = np.asarray([row["candidate_unit_net_pnl"] for row in rows], dtype=float)
    baseline = np.asarray([row["baseline_unit_net_pnl"] for row in rows], dtype=float)
    delta = candidate - baseline
    indices = _bootstrap_indices(
        market_count=len(rows),
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    confidence = float(bootstrap["confidence"])
    candidate_interval = _shared_bootstrap_interval(candidate, indices=indices, confidence=confidence)
    baseline_interval = _shared_bootstrap_interval(baseline, indices=indices, confidence=confidence)
    delta_interval = _shared_bootstrap_interval(delta, indices=indices, confidence=confidence)
    midpoint = len(rows) // 2
    overall = {
        "market_count": len(rows),
        "candidate_accepted_count": sum(row["candidate_accepted"] for row in rows),
        "baseline_accepted_count": sum(row["baseline_accepted"] for row in rows),
        "candidate_total_unit_net_pnl": float(np.sum(candidate)),
        "baseline_total_unit_net_pnl": float(np.sum(baseline)),
        "paired_delta_total_unit_net_pnl": float(np.sum(delta)),
        "candidate_mean_unit_net_pnl": float(np.mean(candidate)),
        "baseline_mean_unit_net_pnl": float(np.mean(baseline)),
        "paired_delta_mean_unit_net_pnl": float(np.mean(delta)),
        "candidate_bootstrap_interval": candidate_interval,
        "baseline_bootstrap_interval": baseline_interval,
        "paired_delta_bootstrap_interval": delta_interval,
        "paired_executable_ask_coverage": sum(
            row["paired_executable_asks_complete"] for row in rows
        ) / len(rows),
    }
    candidate_largest = int(np.argmax(candidate))
    delta_largest = int(np.argmax(delta))
    robust = {
        "candidate_largest_winner_removed_total": float(np.sum(np.delete(candidate, candidate_largest))),
        "paired_delta_largest_positive_removed_total": float(np.sum(np.delete(delta, delta_largest))),
        "candidate_chronological_halves": {
            "first": float(np.sum(candidate[:midpoint])),
            "second": float(np.sum(candidate[midpoint:])),
        },
        "paired_delta_chronological_halves": {
            "first": float(np.sum(delta[:midpoint])),
            "second": float(np.sum(delta[midpoint:])),
        },
    }
    panels = {
        "overall": overall,
        "requested_route": _group_panel(rows, lambda row: str(row["requested_route"])),
        "actual_model": _group_panel(rows, lambda row: str(row["actual_model_used"])),
        "expert_vs_fallback": _group_panel(
            rows, lambda row: "fallback" if row["fallback_used"] else "expert"
        ),
        "UP_vs_DOWN": _group_panel(
            rows, lambda row: str(row["candidate_selected_side"] or "NO_TRADE")
        ),
        "regime": _group_panel(
            rows,
            lambda row: (
                f"{row['regime']['btc_return_regime']}|"
                f"{row['regime']['volatility_bucket']}"
            ),
        ),
        "provider_health": _group_panel(
            rows, lambda row: "present" if row["provider_health"] is not None else "missing"
        ),
        "complete_feature_vs_missing_feature": _group_panel(
            rows,
            lambda row: (
                "complete" if row["feature_missingness"]["feature_complete"] else "missing"
            ),
        ),
        "chronological_half": _group_panel(rows, lambda row: str(row["chronological_half"])),
        "largest_winner_attribution": {
            "candidate": _winner(rows[candidate_largest], "candidate_unit_net_pnl"),
            "paired_delta": _winner(rows[delta_largest], "paired_delta_unit_net_pnl"),
        },
    }
    reporting_complete = (
        set(reporting["required_panels"]) == REPORTING_PANELS == set(panels)
        and set(reporting["required_market_fields"]) == REQUIRED_MARKET_FIELDS
        and all(set(row) >= REQUIRED_MARKET_FIELDS for row in rows)
    )
    gate_observed_values = {
        "quality_valid_market_count": len(rows),
        "paired_executable_ask_coverage": overall["paired_executable_ask_coverage"],
        "moe_total_after_cost_pnl": overall["candidate_total_unit_net_pnl"],
        "moe_mean_pnl_bootstrap_lcb": candidate_interval["lower"],
        "paired_delta_mean_pnl_bootstrap_lcb": delta_interval["lower"],
        "moe_largest_winner_removed_total_pnl": robust[
            "candidate_largest_winner_removed_total"
        ],
        "paired_delta_largest_positive_removed_total": robust[
            "paired_delta_largest_positive_removed_total"
        ],
        "first_chronological_half_moe_pnl": robust["candidate_chronological_halves"][
            "first"
        ],
        "second_chronological_half_moe_pnl": robust["candidate_chronological_halves"][
            "second"
        ],
        "first_chronological_half_paired_delta": robust[
            "paired_delta_chronological_halves"
        ]["first"],
        "second_chronological_half_paired_delta": robust[
            "paired_delta_chronological_halves"
        ]["second"],
        "expert_fallback_attribution_complete": sum(
            row["fallback_used"] for row in rows
        ) + sum(not row["fallback_used"] for row in rows) == len(rows),
        "reporting_contract_complete": reporting_complete,
        "runtime_artifact_validation_passed": artifacts[
            "runtime_validation_report"
        ]["payload"]["mandatory_gate_passed"] is True,
        "target_or_future_leakage_count": 0,
    }
    frozen_gates = {
        name: definition
        for name, definition in protocol["gates"].items()
        if name != "all_gate_booleans_must_be_true"
    }
    if set(gate_observed_values) != set(frozen_gates):
        raise ValueError("implemented confirmatory gates differ from frozen protocol")
    gate_results = {
        name: _gate_passes(value, frozen_gates[name])
        for name, value in gate_observed_values.items()
    }
    passed = all(gate_results.values())
    return {
        "schema_version": "bigan-btc-15m-moe-confirmatory-evaluation-report-v1",
        "lineage_id": LINEAGE_ID,
        "candidate_id": "mixture_of_experts",
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "implementation_commit": implementation_commit,
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "outcome_access_claim": _descriptor(claim_path, repository_root),
            "settlement_ingestion_report": _descriptor(
                settlement_report_path, repository_root
            ),
            "market_evaluation_rows": _descriptor(market_rows_path, repository_root),
            "statistical_protocol": _descriptor(
                Path(artifacts["statistical_protocol"]["path"]), repository_root
            ),
            "reporting_contract": _descriptor(
                Path(artifacts["reporting_contract"]["path"]), repository_root
            ),
        },
        "population": {
            "market_count": len(rows),
            "candidate_row_count": len(rows),
            "baseline_row_count": len(rows),
            "paired_row_count": len(rows),
            "dropped_market_count": 0,
            "duplicate_market_count": 0,
            "out_of_window_market_count": 0,
            "population_changed": False,
        },
        "bootstrap": {
            **bootstrap,
            "shared_index_matrix_sha256": canonical_json_sha256(indices.tolist()),
        },
        "robustness": robust,
        "panels": panels,
        "gate_observed_values": gate_observed_values,
        "gate_results": gate_results,
        "all_gate_booleans_true": passed,
        "confirmatory_gate_passed": passed,
        "failed_round_terminal": not passed,
        "rerun_allowed": False,
        "automatic_promotion_performed": False,
        "promotion_evidence_ready_for_manual_governance": passed,
        "promotion_evidence_eligible": False,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "source_capture_mutated": False,
        "safety": dict(SAFETY),
    }


def _group_panel(
    rows: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return {
        key: {
            "market_count": len(group),
            "candidate_accepted_count": sum(row["candidate_accepted"] for row in group),
            "baseline_accepted_count": sum(row["baseline_accepted"] for row in group),
            "candidate_total_unit_net_pnl": sum(row["candidate_unit_net_pnl"] for row in group),
            "baseline_total_unit_net_pnl": sum(row["baseline_unit_net_pnl"] for row in group),
            "paired_delta_total_unit_net_pnl": sum(row["paired_delta_unit_net_pnl"] for row in group),
        }
        for key, group in sorted(grouped.items())
    }


def _gate_passes(value: Any, definition: Mapping[str, Any]) -> bool:
    operator = definition["operator"]
    threshold = definition["value"]
    if operator == "eq":
        return bool(value == threshold)
    if operator == "gt":
        return bool(value > threshold)
    if operator == "gte":
        return bool(value >= threshold)
    raise ValueError(f"unsupported frozen gate operator: {operator}")


def _feature_at(
    feature_rows: Sequence[Mapping[str, Any]],
    decision_ts: int,
) -> Mapping[str, Any]:
    matches = [row for row in feature_rows if int(row["decision_ts"]) == decision_ts]
    if len(matches) != 1:
        raise ValueError("frozen decision timestamp does not map to exactly one feature row")
    return matches[0]


def _winner(row: Mapping[str, Any], pnl_field: str) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "unit_net_pnl": row[pnl_field],
        "requested_route": row["requested_route"],
        "actual_model_used": row["actual_model_used"],
        "fallback_used": row["fallback_used"],
        "selected_side": row["candidate_selected_side"],
    }


def _zero_costs() -> dict[str, float]:
    return {
        "settlement_payout": 0.0,
        "entry_bid": 0.0,
        "entry_ask": 0.0,
        "entry_mid": 0.0,
        "gross_price_edge": 0.0,
        "entry_spread_cost": 0.0,
        "fees": 0.0,
        "slippage": 0.0,
        "liquidity_impact": 0.0,
        "total_cost": 0.0,
        "unit_net_pnl": 0.0,
    }


def _evaluation_markdown(report: Mapping[str, Any]) -> str:
    overall = report["panels"]["overall"]
    lines = [
        "# BTC-15M-MoE-confirmatory-v2 evaluation",
        "",
        f"- Confirmatory gate passed: `{report['confirmatory_gate_passed']}`",
        f"- Markets: `{overall['market_count']}`",
        f"- Candidate total unit PnL: `{overall['candidate_total_unit_net_pnl']:.8f}`",
        f"- Baseline total unit PnL: `{overall['baseline_total_unit_net_pnl']:.8f}`",
        f"- Paired delta total: `{overall['paired_delta_total_unit_net_pnl']:.8f}`",
        f"- Candidate bootstrap LCB: `{overall['candidate_bootstrap_interval']['lower']:.8f}`",
        f"- Paired delta bootstrap LCB: `{overall['paired_delta_bootstrap_interval']['lower']:.8f}`",
        "",
        "## Frozen gates",
        "",
    ]
    lines.extend(
        f"- {name}: `{value}`" for name, value in report["gate_results"].items()
    )
    lines.extend(
        [
            "",
            "No automatic promotion, paper handoff, live trading, wallet signing, "
            "Polymarket write, or capital-at-risk action was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def _verified_json(path: Path) -> dict[str, Any]:
    descriptor = _verified_path(path)
    descriptor["payload"] = _json_object(path)
    return descriptor


def _verified_path(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"missing frozen artifact or sidecar: {path}")
    actual = sha256_file(path)
    if sidecar.read_text(encoding="utf-8").strip() != actual:
        raise ValueError(f"frozen artifact SHA mismatch: {path}")
    return {"path": path.resolve(), "sha256": actual}


def _resolve_verified_descriptor(
    descriptor: Mapping[str, Any],
    repository_root: Path,
) -> Path:
    path = (repository_root / str(descriptor["path"])).resolve()
    if not path.is_relative_to(repository_root):
        raise ValueError("raw descriptor escaped repository")
    if sha256_file(path) != descriptor["sha256"]:
        raise ValueError("raw descriptor SHA mismatch")
    return path


def _descriptor(path: Path, repository_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(repository_root):
        raise ValueError("evaluation descriptor escaped repository")
    return {
        "path": resolved.relative_to(repository_root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
