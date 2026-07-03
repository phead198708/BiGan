"""Fresh public-data paper-only loop for the v8 O candidate."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus.contracts import BTC_UPDOWN_MARKET_HORIZONS_MS
from bigan.v8.polymarket.recorder.contracts import PolymarketRealCorpusRecorderConfig
from bigan.v8.polymarket.recorder.public_provider import (
    PolymarketPublicHTTPRealCorpusProvider,
    RealCorpusPublicProviderError,
)
from bigan.v8.polymarket.training.contracts import (
    POLYMARKET_POLICY_TRAINING_PHASE,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.o_v8_paper_candidate_unlock import (
    _sha256_file as _sha256_file_existing,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    O_REQUIRED_DECISION_ACTION_FAMILIES,
    _action_family,
    _side_from_action,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
)

O_V8_PAPER_FRESH_LOOP_RUN_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-loop-run-v1"
)
O_V8_PAPER_FRESH_FILL_SIMULATION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-fill-simulation-v1"
)
O_V8_PAPER_FRESH_RUNTIME_SAFETY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-runtime-safety-v1"
)
O_V8_PAPER_FRESH_MONITORING_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-monitoring-v1"
)
O_V8_PAPER_FRESH_CUMULATIVE_MONITORING_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-cumulative-monitoring-v1"
)
O_V8_PAPER_FRESH_LOOP_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-loop-manifest-v1"
)

PINNED_ISSUE_160_RUN_ID = "o-v8-paper-candidate-unlock-20260703T073000Z"
PINNED_ISSUE_160_MANIFEST_SHA256 = (
    "a7bbe5c6128e4471ee48ea0765d4305acb0d0c5722226b7556c9fd4a8f648815"
)
O_V8_PAPER_FRESH_FORBIDDEN_PUBLIC_DATA_FIELDS: tuple[str, ...] = (
    "realized_pnl",
    "realized_trade_pnl",
    "settlement_pnl",
    "settlement_label",
    "oracle_action",
    "oracle_side",
    "future_return",
    "future_price",
    "future_outcome",
    "total_polymarket_pnl",
)
O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER = "read_only_public_provider"
O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE = "snapshot_fixture"
O_V8_PAPER_FRESH_PUBLIC_DATA_SOURCES = (
    O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
    O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE,
)

_FALSE_SAFETY_FIELDS = (
    "capital_at_risk",
    "polymarket_write_enabled",
    "wallet_signing_enabled",
    "v8_execution_handoff_allowed",
    "source_model_candidate_eligible",
    "freeze_ready",
    "promotion_evidence_eligible",
    "#134_resume_allowed",
    "#146_start_allowed",
)


@dataclass(frozen=True, slots=True)
class PolymarketOV8PaperFreshLoopConfig:
    """Configuration for one fresh public-data paper-only loop run."""

    run_id: str
    output_dir: Path | str
    paper_candidate_unlock_dir: Path | str
    loop_mode: Literal["single_cycle", "bounded_recurring"] = "single_cycle"
    max_cycles: int = 1
    sleep_seconds: float = 0.0
    public_data_cycles: tuple[tuple[dict[str, Any], ...], ...] | None = None
    public_data_source: str = O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
    public_provider: Any | None = None
    expected_paper_candidate_unlock_manifest_sha256: str | None = (
        PINNED_ISSUE_160_MANIFEST_SHA256
    )
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.loop_mode not in {"single_cycle", "bounded_recurring"}:
            raise ValueError("loop_mode must be single_cycle or bounded_recurring")
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        if self.loop_mode == "single_cycle" and self.max_cycles != 1:
            raise ValueError("single_cycle mode requires max_cycles=1")
        if self.sleep_seconds < 0.0:
            raise ValueError("sleep_seconds must be non-negative")
        if self.public_data_source not in O_V8_PAPER_FRESH_PUBLIC_DATA_SOURCES:
            raise ValueError(
                "public_data_source must be read_only_public_provider or snapshot_fixture"
            )
        if (
            self.public_data_source == O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE
            and self.public_data_cycles is None
        ):
            raise ValueError("snapshot_fixture mode requires public_data_cycles")
        if self.paper_only is not True:
            raise ValueError("paper_only must be true")
        if self.capital_at_risk is not False:
            raise ValueError("capital_at_risk must be false")
        if self.polymarket_write_enabled is not False:
            raise ValueError("polymarket_write_enabled must be false")
        if self.wallet_signing_enabled is not False:
            raise ValueError("wallet_signing_enabled must be false")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self, "paper_candidate_unlock_dir", Path(self.paper_candidate_unlock_dir)
        )


@dataclass(frozen=True, slots=True)
class PolymarketOV8PaperFreshLoopResult:
    """Generated fresh paper loop bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    fresh_loop_run_report: dict[str, Any]
    fill_simulation_report: dict[str, Any]
    runtime_safety_report: dict[str, Any]
    monitoring_report: dict[str, Any]
    cumulative_monitoring_report: dict[str, Any]
    manifest: dict[str, Any]


def run_polymarket_o_v8_paper_fresh_loop(
    config: PolymarketOV8PaperFreshLoopConfig,
) -> PolymarketOV8PaperFreshLoopResult:
    """Run a bounded paper-only loop over fresh public read-only rows."""

    output_dir = Path(config.output_dir) / config.run_id
    if output_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"paper fresh loop output_dir already exists: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    unlock_evidence = _verify_paper_candidate_unlock(config)
    unlock_verified = bool(unlock_evidence["paper_candidate_unlock_verified"])
    public_data = _resolve_public_data_cycles(config, unlock_evidence)
    public_cycles = public_data["public_data_cycles"]
    public_data_collection_report = public_data["public_data_collection_report"]
    execution_result = _execute_fresh_public_cycles(
        config=config,
        public_cycles=public_cycles,
        public_data_source=public_data_collection_report["public_data_source"],
        unlock_verified=unlock_verified,
    )
    intents = execution_result["paper_order_intents"]
    fills = _fresh_paper_fills_from_intents(intents)
    ledger_rows = _fresh_paper_ledger_from_fills(fills)

    run_report = _fresh_loop_run_report(
        config=config,
        unlock_evidence=unlock_evidence,
        public_data_collection_report=public_data_collection_report,
        public_cycles=public_cycles,
        execution_result=execution_result,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )
    fill_report = _fresh_fill_simulation_report(config=config, fills=fills)
    safety_report = _fresh_runtime_safety_report(
        config=config,
        run_report=run_report,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )
    monitoring_report = _fresh_monitoring_report(
        config=config,
        run_report=run_report,
        execution_result=execution_result,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )
    cumulative_report = _fresh_cumulative_monitoring_report(
        config=config,
        run_report=run_report,
        monitoring_report=monitoring_report,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )

    artifact_paths = {
        "fresh_loop_run_report": output_dir / "o_v8_paper_fresh_loop_run_report.json",
        "fresh_loop_run_summary": output_dir / "o_v8_paper_fresh_loop_run_report.md",
        "fresh_order_intent_log": output_dir
        / "o_v8_paper_fresh_order_intent_log.jsonl",
        "fresh_fill_simulation_report": output_dir
        / "o_v8_paper_fresh_fill_simulation_report.json",
        "fresh_fill_simulation_summary": output_dir
        / "o_v8_paper_fresh_fill_simulation_report.md",
        "fresh_runtime_safety_report": output_dir
        / "o_v8_paper_fresh_runtime_safety_report.json",
        "fresh_runtime_safety_summary": output_dir
        / "o_v8_paper_fresh_runtime_safety_report.md",
        "fresh_monitoring_report": output_dir
        / "o_v8_paper_fresh_monitoring_report.json",
        "fresh_monitoring_summary": output_dir
        / "o_v8_paper_fresh_monitoring_report.md",
        "fresh_cumulative_monitoring_report": output_dir
        / "o_v8_paper_fresh_cumulative_monitoring_report.json",
        "fresh_cumulative_monitoring_summary": output_dir
        / "o_v8_paper_fresh_cumulative_monitoring_report.md",
        "manifest": output_dir / "o_v8_paper_fresh_loop_manifest.json",
    }
    _write_json(artifact_paths["fresh_loop_run_report"], run_report)
    _write_text(artifact_paths["fresh_loop_run_summary"], _fresh_loop_run_md(run_report))
    _write_jsonl(artifact_paths["fresh_order_intent_log"], intents)
    _write_json(artifact_paths["fresh_fill_simulation_report"], fill_report)
    _write_text(
        artifact_paths["fresh_fill_simulation_summary"],
        _fresh_fill_simulation_md(fill_report),
    )
    _write_json(artifact_paths["fresh_runtime_safety_report"], safety_report)
    _write_text(
        artifact_paths["fresh_runtime_safety_summary"],
        _fresh_runtime_safety_md(safety_report),
    )
    _write_json(artifact_paths["fresh_monitoring_report"], monitoring_report)
    _write_text(
        artifact_paths["fresh_monitoring_summary"],
        _fresh_monitoring_md(monitoring_report),
    )
    _write_json(
        artifact_paths["fresh_cumulative_monitoring_report"],
        cumulative_report,
    )
    _write_text(
        artifact_paths["fresh_cumulative_monitoring_summary"],
        _fresh_cumulative_monitoring_md(cumulative_report),
    )

    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
        if name != "manifest"
    }
    manifest = _fresh_loop_manifest(
        config=config,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        unlock_evidence=unlock_evidence,
        public_data_collection_report=public_data_collection_report,
        run_report=run_report,
        fill_report=fill_report,
        safety_report=safety_report,
        monitoring_report=monitoring_report,
        cumulative_report=cumulative_report,
    )
    _write_json(artifact_paths["manifest"], manifest)
    artifact_hashes["manifest"] = _sha256_file(artifact_paths["manifest"])

    return PolymarketOV8PaperFreshLoopResult(
        output_dir=output_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        fresh_loop_run_report=run_report,
        fill_simulation_report=fill_report,
        runtime_safety_report=safety_report,
        monitoring_report=monitoring_report,
        cumulative_monitoring_report=cumulative_report,
        manifest=manifest,
    )


def _verify_paper_candidate_unlock(
    config: PolymarketOV8PaperFreshLoopConfig,
) -> dict[str, Any]:
    unlock_dir = Path(config.paper_candidate_unlock_dir)
    manifest_path = unlock_dir / "o_v8_paper_candidate_unlock_manifest.json"
    observed_manifest_sha = _sha256_file(manifest_path) if manifest_path.exists() else ""
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    expected_manifest_sha = config.expected_paper_candidate_unlock_manifest_sha256
    manifest_hash_passed = (
        expected_manifest_sha is None or observed_manifest_sha == expected_manifest_sha
    )
    artifact_hash_rows: list[dict[str, Any]] = []
    artifact_hashes_match = True
    for name, expected_hash in sorted((manifest.get("artifact_hashes") or {}).items()):
        artifact_path = _resolve_unlock_artifact_path(
            unlock_dir,
            (manifest.get("artifact_paths") or {}).get(name) or "",
        )
        observed_hash = _sha256_file(artifact_path) if artifact_path.exists() else ""
        passed = observed_hash == expected_hash
        artifact_hashes_match = artifact_hashes_match and passed
        artifact_hash_rows.append(
            {
                "artifact_name": name,
                "artifact_path": str(artifact_path),
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "passed": passed,
            }
        )
    required_flags = {
        "paper_candidate_allowed": True,
        "paper_internal_execution_loop_enabled": True,
        "v8_paper_internal_handoff_allowed": True,
        "v8_execution_handoff_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    flag_rows = {
        field_name: {
            "expected": expected,
            "observed": manifest.get(field_name),
            "passed": manifest.get(field_name) is expected,
        }
        for field_name, expected in required_flags.items()
    }
    blocking_reason_codes = []
    if not manifest_path.exists():
        blocking_reason_codes.append("paper_candidate_unlock_manifest_missing")
    if not manifest_hash_passed:
        blocking_reason_codes.append("paper_candidate_unlock_manifest_hash_mismatch")
    if not artifact_hashes_match:
        blocking_reason_codes.append("paper_candidate_unlock_artifact_hash_mismatch")
    if any(row["passed"] is not True for row in flag_rows.values()):
        blocking_reason_codes.append("paper_candidate_unlock_safety_flags_invalid")
    return {
        "paper_candidate_unlock_dir": str(unlock_dir),
        "paper_candidate_unlock_manifest_path": str(manifest_path),
        "expected_manifest_sha256": expected_manifest_sha,
        "observed_manifest_sha256": observed_manifest_sha,
        "manifest_hash_passed": manifest_hash_passed,
        "artifact_hash_rows": artifact_hash_rows,
        "artifact_hashes_match": artifact_hashes_match,
        "required_flag_checks": flag_rows,
        "paper_candidate_unlock_verified": blocking_reason_codes == [],
        "paper_candidate_unlock_blocking_reason_codes": sorted(blocking_reason_codes),
        "unlock_manifest": manifest,
    }


def _resolve_public_data_cycles(
    config: PolymarketOV8PaperFreshLoopConfig,
    unlock_evidence: dict[str, Any],
) -> dict[str, Any]:
    if config.public_data_cycles is not None:
        cycles = [
            [dict(row) for row in cycle] for cycle in config.public_data_cycles
        ]
        return {
            "public_data_cycles": cycles,
            "public_data_collection_report": _snapshot_fixture_collection_report(
                config=config,
                cycles=cycles,
                unlock_evidence=unlock_evidence,
            ),
        }
    return _collect_read_only_public_provider_cycles(
        config=config,
        unlock_evidence=unlock_evidence,
    )


def _snapshot_fixture_collection_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    cycles: list[list[dict[str, Any]]],
    unlock_evidence: dict[str, Any],
) -> dict[str, Any]:
    row_count = sum(len(cycle) for cycle in cycles)
    return {
        "public_data_source": O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE,
        "public_data_collection_mode": "offline_snapshot_fixture",
        "public_provider_class": None,
        "public_provider_read_only": True,
        "paper_fresh_provider_collection_failed": False,
        "public_data_collection_reason_codes": [],
        "public_data_cycle_count": len(cycles),
        "public_data_row_count": row_count,
        "public_market_count": len(
            {str(row.get("market_id")) for cycle in cycles for row in cycle}
        ),
        "public_orderbook_row_count": None,
        "public_trade_row_count": None,
        "public_btc_feature_candle_row_count": None,
        "public_feature_row_count": row_count,
        "frozen_o_action_rank_reference_source": "issue_160_paper_candidate_unlock_manifest",
        "frozen_o_action_rank_reference_sha256": unlock_evidence[
            "observed_manifest_sha256"
        ],
        "scoring_rule_id": "snapshot_fixture_pre_scored_rows",
        "uses_paper_intent_logs_as_fresh_public_data": False,
        "uses_validation_outcomes_for_tuning": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "thresholds_tuned": False,
        "forbidden_outcome_fields_used": [],
    }


def _collect_read_only_public_provider_cycles(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    unlock_evidence: dict[str, Any],
) -> dict[str, Any]:
    provider = config.public_provider or PolymarketPublicHTTPRealCorpusProvider()
    provider_class = provider.__class__.__name__
    provider_safety = _public_provider_safety(provider)
    base_report = {
        "public_data_source": O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
        "public_data_collection_mode": "read_only_public_provider_live_snapshot",
        "public_provider_class": provider_class,
        **provider_safety,
        "frozen_o_action_rank_reference_source": "issue_160_paper_candidate_unlock_manifest",
        "frozen_o_action_rank_reference_sha256": unlock_evidence[
            "observed_manifest_sha256"
        ],
        "scoring_rule_id": "frozen_v8_o_public_provider_feature_score_v1",
        "uses_paper_intent_logs_as_fresh_public_data": False,
        "uses_validation_outcomes_for_tuning": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "thresholds_tuned": False,
        "forbidden_outcome_fields_used": [],
    }
    if not provider_safety["public_provider_safety_passed"]:
        report = {
            **base_report,
            "paper_fresh_provider_collection_failed": True,
            "public_data_collection_reason_codes": [
                "read_only_public_provider_safety_flags_invalid"
            ],
            "public_data_cycle_count": config.max_cycles,
            "public_data_row_count": 0,
            "public_market_count": 0,
            "public_orderbook_row_count": 0,
            "public_trade_row_count": 0,
            "public_btc_feature_candle_row_count": 0,
            "public_feature_row_count": 0,
            "provider_exception_type": None,
            "provider_exception_message": None,
        }
        return {
            "public_data_cycles": [[] for _ in range(config.max_cycles)],
            "public_data_collection_report": report,
        }

    recorder_config = PolymarketRealCorpusRecorderConfig(
        run_id=f"{config.run_id}-fresh-public-provider",
        output_dir=Path(config.output_dir) / config.run_id / "_public_provider_input",
        market_families=tuple(BTC_UPDOWN_MARKET_HORIZONS_MS),
        mock_public_data=False,
        build_phase2_corpus=False,
    )
    try:
        markets = provider.market_rows(recorder_config)
        orderbooks = provider.orderbook_rows(markets, recorder_config)
        trades = provider.trade_rows(markets, recorder_config)
        btc_candles = provider.btc_feature_candle_rows(markets, recorder_config)
        rows = _fresh_public_rows_from_provider_payloads(
            run_id=config.run_id,
            markets=markets,
            orderbooks=orderbooks,
            trades=trades,
            btc_candles=btc_candles,
        )
        reason_codes: list[str] = []
        collection_failed = False
        if not rows:
            collection_failed = True
            reason_codes.append("read_only_public_provider_no_decision_feature_rows")
    except RealCorpusPublicProviderError as exc:
        markets = []
        orderbooks = []
        trades = []
        btc_candles = []
        rows = []
        collection_failed = True
        reason_codes = list(exc.reason_codes) or [
            "read_only_public_provider_collection_failed"
        ]
        exception_type = exc.__class__.__name__
        exception_message = str(exc)
    except Exception as exc:  # noqa: BLE001
        markets = []
        orderbooks = []
        trades = []
        btc_candles = []
        rows = []
        collection_failed = True
        reason_codes = ["read_only_public_provider_collection_failed"]
        exception_type = exc.__class__.__name__
        exception_message = str(exc)
    else:
        exception_type = None
        exception_message = None

    cycles = _partition_public_rows(rows, config.max_cycles)
    report = {
        **base_report,
        "paper_fresh_provider_collection_failed": collection_failed,
        "public_data_collection_reason_codes": sorted(set(reason_codes)),
        "public_data_cycle_count": len(cycles),
        "public_data_row_count": len(rows),
        "public_market_count": len(markets),
        "public_orderbook_row_count": len(orderbooks),
        "public_trade_row_count": len(trades),
        "public_btc_feature_candle_row_count": len(btc_candles),
        "public_feature_row_count": len(rows),
        "provider_exception_type": exception_type,
        "provider_exception_message": exception_message,
    }
    return {
        "public_data_cycles": cycles,
        "public_data_collection_report": report,
    }


def _public_provider_safety(provider: Any) -> dict[str, Any]:
    required = {
        "read_only": True,
        "write_capable": False,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    checks = {
        field_name: {
            "expected": expected,
            "observed": getattr(provider, field_name, None),
            "passed": getattr(provider, field_name, None) is expected,
        }
        for field_name, expected in required.items()
    }
    return {
        "public_provider_read_only": checks["read_only"]["passed"],
        "public_provider_safety_passed": all(
            row["passed"] is True for row in checks.values()
        ),
        "public_provider_safety_checks": checks,
    }


def _fresh_public_rows_from_provider_payloads(
    *,
    run_id: str,
    markets: list[dict[str, Any]],
    orderbooks: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    btc_candles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    del trades
    markets_by_id = {str(market.get("market_id")): dict(market) for market in markets}
    books_by_market = _latest_public_books_by_market(orderbooks)
    candles = sorted(btc_candles, key=lambda row: int(row.get("available_at_ts") or row.get("ts") or 0))
    rows: list[dict[str, Any]] = []
    for market_id, pair in sorted(books_by_market.items()):
        market = markets_by_id.get(market_id)
        if market is None or "UP" not in pair or "DOWN" not in pair:
            continue
        up = pair["UP"]
        down = pair["DOWN"]
        decision_ts = max(
            _book_available_at(up),
            _book_available_at(down),
            int(market.get("market_start_ts") or 0),
        )
        market_end_ts = int(market.get("market_end_ts") or 0)
        if decision_ts <= 0 or market_end_ts <= decision_ts:
            continue
        candle = _latest_public_btc_candle(candles, decision_ts)
        if candle is None:
            continue
        rows.append(
            _fresh_public_row_from_provider_feature_context(
                run_id=run_id,
                row_index=len(rows),
                market=market,
                up=up,
                down=down,
                candle=candle,
                decision_ts=decision_ts,
            )
        )
    return rows


def _latest_public_books_by_market(
    orderbooks: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in orderbooks:
        market_id = str(row.get("market_id") or "")
        outcome = str(row.get("outcome") or "").upper()
        if market_id == "" or outcome not in {"UP", "DOWN"}:
            continue
        previous = grouped.setdefault(market_id, {}).get(outcome)
        if previous is None or _book_available_at(row) >= _book_available_at(previous):
            grouped[market_id][outcome] = dict(row)
    return grouped


def _latest_public_btc_candle(
    candles: list[dict[str, Any]],
    decision_ts: int,
) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for row in candles:
        available_at = int(row.get("available_at_ts") or row.get("ts") or 0)
        if available_at <= decision_ts:
            latest = dict(row)
        if available_at > decision_ts:
            break
    return latest


def _fresh_public_row_from_provider_feature_context(
    *,
    run_id: str,
    row_index: int,
    market: dict[str, Any],
    up: dict[str, Any],
    down: dict[str, Any],
    candle: dict[str, Any],
    decision_ts: int,
) -> dict[str, Any]:
    p_up = _public_p_up(up=up, down=down)
    p_down = 1.0 - p_up
    scores = _provider_action_scores(p_up=p_up, p_down=p_down, up=up, down=down)
    ranking = _provider_full_action_ranking(
        scores=scores,
        p_up=p_up,
        up=up,
        down=down,
        market=market,
        decision_ts=decision_ts,
    )
    selected = ranking[0]
    selected_action = str(selected["selected_action"])
    selected_side = _side_from_action(selected_action)
    reference_provenance = _provider_reference_price_provenance(
        market=market,
        candle=candle,
        decision_ts=decision_ts,
    )
    max_input_ts = max(
        _book_available_at(up),
        _book_available_at(down),
        int(candle.get("available_at_ts") or candle.get("ts") or 0),
        int(reference_provenance.get("max_input_ts") or 0),
    )
    return {
        "decision_group_id": (
            f"{run_id}|read-only-public-provider|{market.get('market_id')}|"
            f"{decision_ts}"
        ),
        "market_id": str(market.get("market_id") or ""),
        "condition_id": str(market.get("condition_id") or ""),
        "slug": str(market.get("slug") or ""),
        "market_family": str(market.get("market_family") or ""),
        "decision_ts": decision_ts,
        "selected_action": selected_action,
        "selected_side": selected_side,
        "selected_action_family": _action_family(selected_action),
        "corrected_model_score": _float(selected.get("corrected_model_score")),
        "raw_model_score": _float(selected.get("raw_model_score")),
        "high_score_flag": _float(selected.get("corrected_model_score")) >= 0.02,
        "p_up": p_up,
        "p_down": p_down,
        "p_up_action_disagreement": _p_up_action_disagreement(
            action=selected_action,
            p_up=p_up,
        ),
        "microstructure_snapshot": _provider_microstructure_for_action(
            action=selected_action,
            up=up,
            down=down,
            market=market,
            decision_ts=decision_ts,
        ),
        "reference_price_feature_provenance": reference_provenance,
        "decision_time_feature_max_input_ts": max_input_ts,
        "full_5_action_ranking": ranking,
        "score_components": {
            "scoring_rule_id": "frozen_v8_o_public_provider_feature_score_v1",
            "p_up": p_up,
            "p_down": p_down,
            "btc_mid_price": _float(candle.get("close_price")),
            "reference_price_to_beat": _float(
                market.get("reference_price_start")
                if market.get("reference_price_start") is not None
                else market.get("reference_price_at_start")
            ),
            "max_input_ts": max_input_ts,
        },
        "public_data_source": O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
        "public_provider_row_index": row_index,
        "public_provider_feature_builder_rule_id": (
            "public_provider_market_orderbook_trade_btc_to_decision_features_v1"
        ),
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _provider_action_scores(
    *,
    p_up: float,
    p_down: float,
    up: dict[str, Any],
    down: dict[str, Any],
) -> dict[str, float]:
    up_ask = _float(up.get("ask_price"))
    down_ask = _float(down.get("ask_price"))
    up_bid = _float(up.get("bid_price"))
    down_bid = _float(down.get("bid_price"))
    up_spread_penalty = _provider_spread_bps(up) / 10_000.0
    down_spread_penalty = _provider_spread_bps(down) / 10_000.0
    return {
        "BUY_UP_HOLD_TO_SETTLEMENT": p_up - up_ask - up_spread_penalty,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": p_down - down_ask - down_spread_penalty,
        "BUY_UP_SELL_BEFORE_CLOSE": up_bid - up_ask - up_spread_penalty,
        "BUY_DOWN_SELL_BEFORE_CLOSE": down_bid - down_ask - down_spread_penalty,
        "NO_TRADE": 0.0,
    }


def _provider_full_action_ranking(
    *,
    scores: dict[str, float],
    p_up: float,
    up: dict[str, Any],
    down: dict[str, Any],
    market: dict[str, Any],
    decision_ts: int,
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for action in O_REQUIRED_DECISION_ACTION_FAMILIES:
        side = _side_from_action(action)
        score = _float(scores.get(action))
        ranking.append(
            {
                "selected_action": action,
                "selected_side": side,
                "selected_action_family": _action_family(action),
                "corrected_model_score": score,
                "raw_model_score": score,
                "p_up_action_disagreement": _p_up_action_disagreement(
                    action=action,
                    p_up=p_up,
                ),
                "microstructure_snapshot": _provider_microstructure_for_action(
                    action=action,
                    up=up,
                    down=down,
                    market=market,
                    decision_ts=decision_ts,
                ),
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            _float(row.get("corrected_model_score")),
            1 if row.get("selected_action") != "NO_TRADE" else 0,
            str(row.get("selected_action")),
        ),
        reverse=True,
    )


def _provider_microstructure_for_action(
    *,
    action: str,
    up: dict[str, Any],
    down: dict[str, Any],
    market: dict[str, Any],
    decision_ts: int,
) -> dict[str, Any]:
    if action == "NO_TRADE":
        return {}
    book = up if _side_from_action(action) == "UP" else down
    return {
        "entry_ask": _float(book.get("ask_price")),
        "executable_exit_bid_proxy": _float(book.get("bid_price")),
        "spread_bps": _provider_spread_bps(book),
        "book_staleness_ms": max(0, decision_ts - _book_available_at(book)),
        "queue_fill_proxy": _provider_queue_fill_proxy(book),
        "time_to_close_seconds": max(
            0.0,
            (int(market.get("market_end_ts") or decision_ts) - decision_ts)
            / 1000.0,
        ),
    }


def _provider_reference_price_provenance(
    *,
    market: dict[str, Any],
    candle: dict[str, Any],
    decision_ts: int,
) -> dict[str, Any]:
    reference_ts = int(market.get("market_start_ts") or 0)
    candle_ts = int(candle.get("available_at_ts") or candle.get("ts") or 0)
    max_input_ts = max(reference_ts, candle_ts)
    return {
        "provenance_valid": max_input_ts <= decision_ts,
        "decision_ts": decision_ts,
        "max_input_ts": max_input_ts,
        "source_fields_used": [
            "raw_polymarket_markets.reference_price_start",
            "raw_btc_feature_candles.close_price",
        ],
        "source_field_name": "read_only_public_provider_reference_and_btc_candle",
        "source_timestamp": max_input_ts,
    }


def _partition_public_rows(
    rows: list[dict[str, Any]],
    max_cycles: int,
) -> list[list[dict[str, Any]]]:
    cycles: list[list[dict[str, Any]]] = [[] for _ in range(max_cycles)]
    for index, row in enumerate(rows):
        cycles[index % max_cycles].append(row)
    return cycles


def _public_p_up(*, up: dict[str, Any], down: dict[str, Any]) -> float:
    up_mid = _float(up.get("mid_price"))
    down_mid = _float(down.get("mid_price"))
    total = up_mid + down_mid
    if total <= 0.0:
        return 0.5
    return max(0.0, min(1.0, up_mid / total))


def _provider_spread_bps(book: dict[str, Any]) -> float:
    bid = _float(book.get("bid_price"))
    ask = _float(book.get("ask_price"))
    mid = _float(book.get("mid_price")) or (bid + ask) / 2.0
    if mid <= 0.0:
        return 10_000.0
    return max(0.0, (ask - bid) / mid * 10_000.0)


def _provider_queue_fill_proxy(book: dict[str, Any]) -> float:
    bid_notional = _float(book.get("bid_price")) * _float(book.get("bid_size"))
    ask_notional = _float(book.get("ask_price")) * _float(book.get("ask_size"))
    depth_score = min(1.0, _float(book.get("liquidity_depth")) / 2.0)
    notional_score = min(1.0, max(bid_notional, ask_notional))
    spread_score = max(0.0, 1.0 - _provider_spread_bps(book) / 2_000.0)
    return max(
        0.0,
        min(1.0, 0.55 * notional_score + 0.35 * depth_score + 0.10 * spread_score),
    )


def _book_available_at(book: dict[str, Any]) -> int:
    return int(book.get("available_at_ts") or book.get("ts") or 0)


def _p_up_action_disagreement(*, action: str, p_up: float) -> bool:
    side = _side_from_action(action)
    if side == "UP":
        return p_up < 0.50
    if side == "DOWN":
        return p_up > 0.50
    return False


def _execute_fresh_public_cycles(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    public_cycles: list[list[dict[str, Any]]],
    public_data_source: str,
    unlock_verified: bool,
) -> dict[str, Any]:
    runtime_state = _initial_fresh_runtime_state()
    guard_config = _v8_execution_guard_config()
    all_guard_rows: list[dict[str, Any]] = []
    all_intents: list[dict[str, Any]] = []
    cycle_reports: list[dict[str, Any]] = []
    cycle_failure_count = 0

    for cycle_index in range(config.max_cycles):
        cycle_rows = list(public_cycles[cycle_index] if cycle_index < len(public_cycles) else [])
        cycle_id = f"{config.run_id}-cycle-{cycle_index + 1:06d}"
        cycle_guard_rows: list[dict[str, Any]] = []
        cycle_intents: list[dict[str, Any]] = []
        cycle_forbidden_rows = _rows_with_forbidden_fields(cycle_rows)
        cycle_failed = not unlock_verified or bool(cycle_forbidden_rows)
        if not cycle_failed:
            for row_index, public_row in enumerate(cycle_rows):
                guard_input = _guard_input_from_public_row(
                    public_row=public_row,
                    cycle_id=cycle_id,
                    row_index=row_index,
                )
                pre_state = _compact_runtime_state(runtime_state)
                guard_row = _v8_execution_guard_decision(
                    guard_input,
                    guard_config=guard_config,
                    runtime_state=runtime_state,
                    runtime_mode="simulated_runtime_state",
                )
                guard_row["cycle_id"] = cycle_id
                guard_row["public_data_source"] = public_data_source
                guard_row["pre_decision_exposure_state"] = pre_state
                if guard_row.get("order_allowed") is True:
                    guard_row["simulated_order_id"] = (
                        f"{config.run_id}-fresh-sim-{len(all_intents) + 1:06d}"
                    )
                    _apply_guard_row_to_runtime_state(runtime_state, guard_row)
                else:
                    runtime_state["blocked_simulated_order_count"] = int(
                        runtime_state["blocked_simulated_order_count"]
                    ) + 1
                    guard_row["simulated_order_id"] = None
                guard_row["post_decision_exposure_state"] = _compact_runtime_state(
                    runtime_state
                )
                cycle_guard_rows.append(guard_row)
                if guard_row.get("order_allowed") is True:
                    intent = _fresh_order_intent_from_guard_row(
                        config=config,
                        cycle_id=cycle_id,
                        guard_row=guard_row,
                        intent_index=len(all_intents) + 1,
                    )
                    cycle_intents.append(intent)
                    all_intents.append(intent)
        else:
            cycle_failure_count += 1
        cycle_reports.append(
            _cycle_monitoring_row(
                cycle_id=cycle_id,
                cycle_index=cycle_index,
                public_rows=cycle_rows,
                guard_rows=cycle_guard_rows,
                intents=cycle_intents,
                cycle_failed=cycle_failed,
                cycle_forbidden_rows=cycle_forbidden_rows,
                public_data_source=public_data_source,
                runtime_state=runtime_state,
            )
        )
        all_guard_rows.extend(cycle_guard_rows)
        if cycle_index < config.max_cycles - 1 and config.sleep_seconds > 0.0:
            time.sleep(config.sleep_seconds)

    return {
        "guard_config": guard_config,
        "cycle_monitoring_rows": cycle_reports,
        "guard_decision_rows": all_guard_rows,
        "paper_order_intents": all_intents,
        "final_runtime_state": _compact_runtime_state(runtime_state),
        "cycle_failure_count": cycle_failure_count,
    }


def _fresh_loop_run_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    unlock_evidence: dict[str, Any],
    public_data_collection_report: dict[str, Any],
    public_cycles: list[list[dict[str, Any]]],
    execution_result: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    guard_rows = execution_result["guard_decision_rows"]
    blockers = list(unlock_evidence["paper_candidate_unlock_blocking_reason_codes"])
    if execution_result["cycle_failure_count"]:
        blockers.append("paper_fresh_public_data_cycle_failed")
    if public_data_collection_report["paper_fresh_provider_collection_failed"]:
        blockers.append("paper_fresh_public_provider_collection_failed")
        blockers.extend(public_data_collection_report["public_data_collection_reason_codes"])
    report = {
        "schema_version": O_V8_PAPER_FRESH_LOOP_RUN_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_loop_run",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_loop_enabled": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "paper_fresh_loop_mode": config.loop_mode,
        "paper_fresh_loop_cycle_count": len(public_cycles),
        "paper_fresh_loop_max_cycles": config.max_cycles,
        "paper_fresh_loop_sleep_seconds": config.sleep_seconds,
        "paper_fresh_loop_public_data_source": public_data_collection_report[
            "public_data_source"
        ],
        "public_data_collection_report": public_data_collection_report,
        "paper_fresh_provider_collection_failed": public_data_collection_report[
            "paper_fresh_provider_collection_failed"
        ],
        "public_data_collection_reason_codes": public_data_collection_report[
            "public_data_collection_reason_codes"
        ],
        "uses_paper_intent_logs_as_fresh_public_data": public_data_collection_report[
            "uses_paper_intent_logs_as_fresh_public_data"
        ],
        "paper_candidate_unlock_verified": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "paper_candidate_unlock_manifest_sha256": unlock_evidence[
            "observed_manifest_sha256"
        ],
        "paper_candidate_unlock_blocking_reason_codes": unlock_evidence[
            "paper_candidate_unlock_blocking_reason_codes"
        ],
        "paper_fresh_loop_blocking_reason_codes": sorted(set(blockers)),
        "public_data_cycle_input_count": sum(len(cycle) for cycle in public_cycles),
        "candidate_decision_count": len(guard_rows),
        "guard_allowed_decision_count": sum(
            1 for row in guard_rows if row.get("order_allowed") is True
        ),
        "guard_blocked_decision_count": sum(
            1 for row in guard_rows if row.get("order_allowed") is not True
        ),
        "paper_fresh_order_intent_count": len(intents),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_ledger_entry_count": len(ledger_rows),
        "runtime_field_missing_count": sum(
            len(row.get("missing_runtime_field_codes") or []) for row in guard_rows
        ),
        "provenance_violation_count": sum(
            len(row.get("runtime_field_backfill_provenance_violations") or [])
            for row in guard_rows
        ),
        "p_up_disagreement_count": sum(
            1 for row in guard_rows if row.get("p_up_action_disagreement") is True
        ),
        "block_reason_distribution": _counter_from_rows(
            guard_rows, "execution_blocking_reason_codes"
        ),
        "action_distribution": Counter(
            str(row.get("execution_guarded_action")) for row in guard_rows
        ),
        "side_distribution": Counter(
            str(row.get("execution_guarded_side")) for row in guard_rows
        ),
        "family_distribution": Counter(
            str(row.get("execution_guarded_family")) for row in guard_rows
        ),
        "final_runtime_state": execution_result["final_runtime_state"],
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_paper_internal_handoff_allowed": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_loop_run_report_id")


def _fresh_fill_simulation_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    fills: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "schema_version": O_V8_PAPER_FRESH_FILL_SIMULATION_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_fill_simulation",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_fill_simulation_enabled": bool(fills),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_filled_size_sum": sum(_float(row.get("filled_size")) for row in fills),
        "paper_fresh_total_synthetic_execution_cost": sum(
            _float(row.get("total_execution_cost")) for row in fills
        ),
        "fill_simulation_rule_ids": sorted(
            {str(row.get("fill_simulation_rule_id")) for row in fills}
        ),
        "outcome_pnl_used": False,
        "realized_pnl_used": False,
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "capital_at_risk": False,
        "paper_only": True,
    }
    return _with_report_id(report, "o_v8_paper_fresh_fill_simulation_report_id")


def _fresh_runtime_safety_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    run_report: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [*intents, *fills, *ledger_rows]
    safety_checks = {
        "paper_only_true": _check(
            passed=config.paper_only is True
            and all(row.get("paper_only") is True for row in rows),
            reason_code="paper_fresh_runtime_not_paper_only",
            observed=True,
            required=True,
        ),
        "capital_at_risk_false": _check(
            passed=config.capital_at_risk is False
            and all(row.get("capital_at_risk") is False for row in rows),
            reason_code="paper_fresh_runtime_capital_at_risk",
            observed=False,
            required=False,
        ),
        "polymarket_writes_disabled": _check(
            passed=config.polymarket_write_enabled is False
            and all(row.get("polymarket_write_enabled") is False for row in rows),
            reason_code="paper_fresh_runtime_polymarket_write_enabled",
            observed=False,
            required=False,
        ),
        "wallet_signing_disabled": _check(
            passed=config.wallet_signing_enabled is False
            and all(row.get("wallet_signing_enabled") is False for row in rows),
            reason_code="paper_fresh_runtime_wallet_signing_enabled",
            observed=False,
            required=False,
        ),
        "ledger_updates_only_accepted_intents": _check(
            passed=len(ledger_rows) == len(intents)
            and {row["paper_fresh_order_intent_id"] for row in ledger_rows}
            == {row["paper_fresh_order_intent_id"] for row in intents},
            reason_code="paper_fresh_ledger_updates_unaccepted_intents",
            observed={
                "intent_count": len(intents),
                "ledger_entry_count": len(ledger_rows),
            },
            required="ledger ids equal accepted fresh intent ids",
        ),
        "live_handoff_remains_blocked": _check(
            passed=run_report["v8_execution_handoff_allowed"] is False
            and run_report["#134_resume_allowed"] is False
            and run_report["#146_start_allowed"] is False,
            reason_code="paper_fresh_live_handoff_unexpectedly_unlocked",
            observed={
                "v8_execution_handoff_allowed": run_report[
                    "v8_execution_handoff_allowed"
                ],
                "#134_resume_allowed": run_report["#134_resume_allowed"],
                "#146_start_allowed": run_report["#146_start_allowed"],
            },
            required=False,
        ),
        "no_threshold_tuning_or_forbidden_outcomes": _check(
            passed=run_report["thresholds_tuned"] is False
            and run_report["forbidden_outcome_fields_used"] == [],
            reason_code="paper_fresh_threshold_or_forbidden_outcome_usage",
            observed={
                "thresholds_tuned": run_report["thresholds_tuned"],
                "forbidden_outcome_fields_used": run_report[
                    "forbidden_outcome_fields_used"
                ],
            },
            required={"thresholds_tuned": False, "forbidden_outcome_fields_used": []},
        ),
    }
    blockers = _blocking_reason_codes(safety_checks)
    report = {
        "schema_version": O_V8_PAPER_FRESH_RUNTIME_SAFETY_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_runtime_safety",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_runtime_safety_checks": safety_checks,
        "paper_fresh_runtime_safety_blocking_reason_codes": blockers,
        "paper_fresh_runtime_safety_passed": blockers == [],
        "paper_fresh_loop_enabled": run_report["paper_fresh_loop_enabled"],
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "paper_fresh_order_intent_count": len(intents),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_ledger_entry_count": len(ledger_rows),
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_runtime_safety_report_id")


def _fresh_monitoring_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    run_report: dict[str, Any],
    execution_result: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "schema_version": O_V8_PAPER_FRESH_MONITORING_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_monitoring",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_monitoring_passed": run_report[
            "paper_fresh_loop_blocking_reason_codes"
        ]
        == [],
        "cycle_monitoring_reports": execution_result["cycle_monitoring_rows"],
        "cycle_count": len(execution_result["cycle_monitoring_rows"]),
        "cycle_failure_count": execution_result["cycle_failure_count"],
        "candidate_decision_count": run_report["candidate_decision_count"],
        "guard_allowed_decision_count": run_report["guard_allowed_decision_count"],
        "guard_blocked_decision_count": run_report["guard_blocked_decision_count"],
        "paper_fresh_order_intent_count": len(intents),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_ledger_entry_count": len(ledger_rows),
        "block_reason_distribution": run_report["block_reason_distribution"],
        "action_distribution": run_report["action_distribution"],
        "side_distribution": run_report["side_distribution"],
        "family_distribution": run_report["family_distribution"],
        "final_runtime_state": run_report["final_runtime_state"],
        "safety_flags": compact_safety_fields(),
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_monitoring_report_id")


def _fresh_cumulative_monitoring_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    run_report: dict[str, Any],
    monitoring_report: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    exposure_by_market: dict[str, float] = defaultdict(float)
    exposure_by_side: dict[str, float] = defaultdict(float)
    for intent in intents:
        size = _float(intent.get("paper_fresh_order_size"))
        exposure_by_market[str(intent.get("market_id"))] += size
        exposure_by_side[str(intent.get("execution_guarded_side"))] += size
    report = {
        "schema_version": O_V8_PAPER_FRESH_CUMULATIVE_MONITORING_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_cumulative_monitoring",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "total_cycles": monitoring_report["cycle_count"],
        "total_paper_intents": len(intents),
        "total_paper_fills": len(fills),
        "total_blocked_decisions": run_report["guard_blocked_decision_count"],
        "cumulative_block_reason_distribution": run_report[
            "block_reason_distribution"
        ],
        "cumulative_action_distribution": run_report["action_distribution"],
        "cumulative_side_distribution": run_report["side_distribution"],
        "cumulative_family_distribution": run_report["family_distribution"],
        "cumulative_simulated_exposure_by_market": dict(sorted(exposure_by_market.items())),
        "cumulative_simulated_exposure_by_side": dict(sorted(exposure_by_side.items())),
        "cycle_failure_count": monitoring_report["cycle_failure_count"],
        "safety_violation_count": 0
        if run_report["paper_fresh_loop_blocking_reason_codes"] == []
        else len(run_report["paper_fresh_loop_blocking_reason_codes"]),
        "thresholds_tuned": False,
        "forbidden_outcome_fields_used": [],
        "paper_fresh_monitoring_passed": monitoring_report[
            "paper_fresh_monitoring_passed"
        ],
        "ledger_updates_only_accepted_intents": len(ledger_rows) == len(intents),
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(
        report, "o_v8_paper_fresh_cumulative_monitoring_report_id"
    )


def _fresh_loop_manifest(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    artifact_paths: dict[str, Path],
    artifact_hashes: dict[str, str],
    unlock_evidence: dict[str, Any],
    public_data_collection_report: dict[str, Any],
    run_report: dict[str, Any],
    fill_report: dict[str, Any],
    safety_report: dict[str, Any],
    monitoring_report: dict[str, Any],
    cumulative_report: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": O_V8_PAPER_FRESH_LOOP_MANIFEST_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_loop_manifest",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "artifact_paths": {
            name: str(path) for name, path in sorted(artifact_paths.items())
        },
        "artifact_hashes": dict(artifact_hashes),
        "paper_candidate_unlock_dir": unlock_evidence["paper_candidate_unlock_dir"],
        "paper_candidate_unlock_manifest_sha256": unlock_evidence[
            "observed_manifest_sha256"
        ],
        "paper_candidate_unlock_verified": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "fresh_loop_run_report_id": run_report[
            "o_v8_paper_fresh_loop_run_report_id"
        ],
        "fresh_fill_simulation_report_id": fill_report[
            "o_v8_paper_fresh_fill_simulation_report_id"
        ],
        "fresh_runtime_safety_report_id": safety_report[
            "o_v8_paper_fresh_runtime_safety_report_id"
        ],
        "fresh_monitoring_report_id": monitoring_report[
            "o_v8_paper_fresh_monitoring_report_id"
        ],
        "fresh_cumulative_monitoring_report_id": cumulative_report[
            "o_v8_paper_fresh_cumulative_monitoring_report_id"
        ],
        "paper_fresh_loop_enabled": run_report["paper_fresh_loop_enabled"],
        "paper_fresh_loop_mode": run_report["paper_fresh_loop_mode"],
        "paper_fresh_loop_cycle_count": run_report["paper_fresh_loop_cycle_count"],
        "paper_fresh_loop_max_cycles": run_report["paper_fresh_loop_max_cycles"],
        "paper_fresh_loop_sleep_seconds": run_report[
            "paper_fresh_loop_sleep_seconds"
        ],
        "paper_fresh_loop_public_data_source": run_report[
            "paper_fresh_loop_public_data_source"
        ],
        "public_data_collection_report": public_data_collection_report,
        "paper_fresh_provider_collection_failed": run_report[
            "paper_fresh_provider_collection_failed"
        ],
        "public_data_collection_reason_codes": run_report[
            "public_data_collection_reason_codes"
        ],
        "uses_paper_intent_logs_as_fresh_public_data": run_report[
            "uses_paper_intent_logs_as_fresh_public_data"
        ],
        "paper_fresh_order_intent_count": run_report[
            "paper_fresh_order_intent_count"
        ],
        "paper_fresh_fill_count": run_report["paper_fresh_fill_count"],
        "paper_fresh_ledger_entry_count": run_report[
            "paper_fresh_ledger_entry_count"
        ],
        "paper_fresh_monitoring_passed": monitoring_report[
            "paper_fresh_monitoring_passed"
        ],
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(manifest, "o_v8_paper_fresh_loop_manifest_id")


def _guard_input_from_public_row(
    *,
    public_row: dict[str, Any],
    cycle_id: str,
    row_index: int,
) -> dict[str, Any]:
    action = str(public_row.get("selected_action") or public_row.get("action") or "")
    side = str(public_row.get("selected_side") or _side_from_action(action))
    family = str(public_row.get("selected_action_family") or _action_family(action))
    score = _float(
        public_row.get("corrected_model_score")
        if public_row.get("corrected_model_score") is not None
        else public_row.get("source_model_score")
    )
    decision_ts = int(public_row.get("decision_ts") or 0)
    ranking = list(public_row.get("full_5_action_ranking") or [])
    if not ranking:
        ranking = [
            {
                "selected_action": candidate,
                "corrected_model_score": score if candidate == action else score - 0.1,
                "raw_model_score": _float(public_row.get("raw_model_score")),
            }
            for candidate in O_REQUIRED_DECISION_ACTION_FAMILIES
        ]
    return {
        "decision_group_id": public_row.get("decision_group_id")
        or f"{cycle_id}|{public_row.get('market_id')}|{decision_ts}|{row_index}",
        "market_id": public_row.get("market_id"),
        "decision_ts": decision_ts,
        "selected_action": action,
        "selected_side": side,
        "selected_action_family": family,
        "full_5_action_ranking": ranking,
        "corrected_model_score": score,
        "raw_model_score": _float(public_row.get("raw_model_score")),
        "score_components": dict(public_row.get("score_components") or {}),
        "high_score_flag": bool(public_row.get("high_score_flag", True)),
        "p_up": _float(public_row.get("p_up")),
        "p_down": _float(public_row.get("p_down")),
        "p_up_action_disagreement": bool(public_row.get("p_up_action_disagreement")),
        "microstructure_snapshot": dict(public_row.get("microstructure_snapshot") or {}),
        "reference_price_feature_provenance": dict(
            public_row.get("reference_price_feature_provenance")
            or {
                "provenance_valid": True,
                "decision_ts": decision_ts,
                "max_input_ts": decision_ts,
                "source_fields_used": ["fresh_public_provider_fixture"],
            }
        ),
        "decision_time_feature_max_input_ts": public_row.get(
            "decision_time_feature_max_input_ts", decision_ts
        ),
    }

def _resolve_unlock_artifact_path(unlock_dir: Path, raw_path: str) -> Path:
    artifact_path = Path(raw_path)
    if artifact_path.is_absolute() or artifact_path.exists():
        return artifact_path
    return unlock_dir / artifact_path


def _fresh_order_intent_from_guard_row(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    cycle_id: str,
    guard_row: dict[str, Any],
    intent_index: int,
) -> dict[str, Any]:
    micro = dict(guard_row.get("microstructure_snapshot") or {})
    intent = {
        "paper_fresh_order_intent_id": f"{config.run_id}-fresh-intent-{intent_index:06d}",
        "cycle_id": cycle_id,
        "simulated_order_id": guard_row.get("simulated_order_id"),
        "decision_group_id": guard_row.get("decision_group_id"),
        "market_id": guard_row.get("market_id"),
        "decision_ts": guard_row.get("decision_ts"),
        "source_selected_action": guard_row.get("source_selected_action"),
        "source_selected_family": guard_row.get("source_selected_family"),
        "source_selected_side": guard_row.get("source_selected_side"),
        "execution_guarded_action": guard_row.get("execution_guarded_action"),
        "execution_guarded_family": guard_row.get("execution_guarded_family"),
        "execution_guarded_side": guard_row.get("execution_guarded_side"),
        "source_model_score": _float(guard_row.get("source_model_score")),
        "execution_guarded_score": _float(guard_row.get("execution_guarded_score")),
        "p_up": _float(guard_row.get("p_up")),
        "p_down": _float(guard_row.get("p_down")),
        "p_up_action_disagreement": bool(guard_row.get("p_up_action_disagreement")),
        "order_origin": "fresh_public_guard_allowed_action",
        "paper_fresh_order_size": _float(guard_row.get("proposed_order_size")),
        "paper_limit_price": _fill_price_from_microstructure(micro),
        "spread_bps": _float(micro.get("spread_bps")),
        "book_staleness_ms": _float(micro.get("book_staleness_ms")),
        "queue_fill_proxy": _float(micro.get("queue_fill_proxy")),
        "time_to_close_seconds": _float(micro.get("time_to_close_seconds")),
        "entry_ask": _float(micro.get("entry_ask")),
        "executable_exit_bid_proxy": _float(micro.get("executable_exit_bid_proxy")),
        "pre_decision_exposure_state": guard_row.get("pre_decision_exposure_state"),
        "post_decision_exposure_state": guard_row.get("post_decision_exposure_state"),
        "execution_guard_reason_codes": guard_row.get("execution_guard_reason_codes", []),
        "execution_blocking_reason_codes": guard_row.get(
            "execution_blocking_reason_codes", []
        ),
        "sizing_reason_codes": guard_row.get("sizing_reason_codes", []),
        "paper_fresh_order_intent_status": "accepted_for_fresh_paper_loop",
        "order_intent_contract": "fresh_public_local_paper_intent_no_exchange_write_v1",
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    intent["paper_fresh_order_intent_hash"] = canonical_json_sha256(intent)
    return intent


def _fresh_paper_fills_from_intents(
    intents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fills: list[dict[str, Any]] = []
    for index, intent in enumerate(intents, start=1):
        size = _float(intent.get("paper_fresh_order_size"))
        fill_price = _float(intent.get("paper_limit_price"))
        spread_cost = size * _float(intent.get("spread_bps")) / 10_000.0
        fill = {
            "paper_fresh_fill_id": f"fresh-paper-fill-{index:06d}",
            "paper_fresh_order_intent_id": intent["paper_fresh_order_intent_id"],
            "cycle_id": intent.get("cycle_id"),
            "simulated_order_id": intent.get("simulated_order_id"),
            "market_id": intent.get("market_id"),
            "decision_ts": intent.get("decision_ts"),
            "execution_guarded_action": intent.get("execution_guarded_action"),
            "execution_guarded_family": intent.get("execution_guarded_family"),
            "execution_guarded_side": intent.get("execution_guarded_side"),
            "fill_simulation_status": "paper_fresh_filled",
            "fill_simulation_rule_id": "fresh_deterministic_queue_fill_proxy_v1",
            "requested_size": size,
            "filled_size": size,
            "fill_probability": _float(intent.get("queue_fill_proxy")),
            "paper_fill_price": fill_price,
            "spread_cost": spread_cost,
            "fee_cost": 0.0,
            "slippage_cost": 0.0,
            "liquidity_impact_cost": 0.0,
            "total_execution_cost": spread_cost,
            "outcome_pnl_used": False,
            "realized_pnl_used": False,
            "synthetic_paper_cash_delta": -(size * fill_price + spread_cost),
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        fill["paper_fresh_fill_hash"] = canonical_json_sha256(fill)
        fills.append(fill)
    return fills


def _fresh_paper_ledger_from_fills(
    fills: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cash = 10_000.0
    exposure_by_market: dict[str, float] = defaultdict(float)
    exposure_by_side: dict[str, float] = defaultdict(float)
    ledger_rows: list[dict[str, Any]] = []
    for index, fill in enumerate(fills, start=1):
        market_id = str(fill.get("market_id"))
        side = str(fill.get("execution_guarded_side"))
        size = _float(fill.get("filled_size"))
        cash_before = cash
        cash += _float(fill.get("synthetic_paper_cash_delta"))
        exposure_by_market[market_id] += size
        exposure_by_side[side] += size
        row = {
            "paper_fresh_ledger_entry_id": f"fresh-paper-ledger-{index:06d}",
            "paper_fresh_fill_id": fill["paper_fresh_fill_id"],
            "paper_fresh_order_intent_id": fill["paper_fresh_order_intent_id"],
            "cycle_id": fill.get("cycle_id"),
            "market_id": market_id,
            "decision_ts": fill.get("decision_ts"),
            "execution_guarded_action": fill.get("execution_guarded_action"),
            "execution_guarded_side": side,
            "cash_before": cash_before,
            "cash_after": cash,
            "synthetic_position_after": exposure_by_market[market_id],
            "total_exposure_after": sum(exposure_by_market.values()),
            "side_exposure_after": exposure_by_side[side],
            "outcome_pnl_used": False,
            "realized_pnl_used": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        row["paper_fresh_ledger_entry_hash"] = canonical_json_sha256(row)
        ledger_rows.append(row)
    return ledger_rows


def _cycle_monitoring_row(
    *,
    cycle_id: str,
    cycle_index: int,
    public_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    cycle_failed: bool,
    cycle_forbidden_rows: list[dict[str, Any]],
    public_data_source: str,
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    unique_markets = sorted({str(row.get("market_id")) for row in public_rows})
    return {
        "cycle_id": cycle_id,
        "cycle_index": cycle_index + 1,
        "cycle_failed": cycle_failed,
        "cycle_failure_reason_codes": [
            "paper_candidate_unlock_not_verified"
        ]
        if cycle_failed and not cycle_forbidden_rows
        else (["fresh_public_data_forbidden_outcome_fields_present"] if cycle_failed else []),
        "public_data_source": public_data_source,
        "public_data_freshness": "read_only_public_provider_snapshot"
        if public_data_source == O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
        else "offline_snapshot_fixture",
        "market_count": len(public_rows),
        "unique_market_count": len(unique_markets),
        "unique_market_ids": unique_markets,
        "candidate_decision_count": len(guard_rows),
        "guard_allowed_paper_intent_count": len(intents),
        "guard_blocked_decision_count": sum(
            1 for row in guard_rows if row.get("order_allowed") is not True
        ),
        "block_reason_distribution": _counter_from_rows(
            guard_rows, "execution_blocking_reason_codes"
        ),
        "action_distribution": Counter(
            str(row.get("execution_guarded_action")) for row in guard_rows
        ),
        "side_distribution": Counter(
            str(row.get("execution_guarded_side")) for row in guard_rows
        ),
        "family_distribution": Counter(
            str(row.get("execution_guarded_family")) for row in guard_rows
        ),
        "runtime_field_missing_count": sum(
            len(row.get("missing_runtime_field_codes") or []) for row in guard_rows
        ),
        "provenance_violation_count": sum(
            len(row.get("runtime_field_backfill_provenance_violations") or [])
            for row in guard_rows
        ),
        "p_up_disagreement_count": sum(
            1 for row in guard_rows if row.get("p_up_action_disagreement") is True
        ),
        "exposure_state_after": _compact_runtime_state(runtime_state),
        "forbidden_public_data_rows": cycle_forbidden_rows,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _rows_with_forbidden_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        present = sorted(
            field_name
            for field_name in O_V8_PAPER_FRESH_FORBIDDEN_PUBLIC_DATA_FIELDS
            if field_name in row
        )
        if present:
            failures.append(
                {
                    "row_index": index,
                    "market_id": row.get("market_id"),
                    "decision_ts": row.get("decision_ts"),
                    "forbidden_fields": present,
                }
            )
    return failures


def _initial_fresh_runtime_state() -> dict[str, Any]:
    return {
        "risk_state_source": "fresh_public_paper_simulated_ledger",
        "runtime_state_validation_passed": True,
        "current_total_exposure": 0.0,
        "current_side_exposure_by_side": {"UP": 0.0, "DOWN": 0.0, "NONE": 0.0},
        "current_market_exposure_by_market_id": {},
        "open_position_by_market_id": {},
        "open_position_by_market_side": {},
        "cooldown_state": {},
        "executed_simulated_order_count": 0,
        "blocked_simulated_order_count": 0,
    }


def _apply_guard_row_to_runtime_state(
    runtime_state: dict[str, Any],
    guard_row: dict[str, Any],
) -> None:
    market_id = str(guard_row.get("market_id"))
    side = str(guard_row.get("execution_guarded_side"))
    size = _float(guard_row.get("proposed_order_size"))
    market_exposure = runtime_state["current_market_exposure_by_market_id"]
    side_exposure = runtime_state["current_side_exposure_by_side"]
    market_exposure[market_id] = _float(market_exposure.get(market_id)) + size
    side_exposure[side] = _float(side_exposure.get(side)) + size
    runtime_state["current_total_exposure"] = _float(
        runtime_state.get("current_total_exposure")
    ) + size
    position = {
        "market_id": market_id,
        "side": side,
        "action": guard_row.get("execution_guarded_action"),
        "notional": size,
        "simulated_order_id": guard_row.get("simulated_order_id"),
    }
    runtime_state["open_position_by_market_id"][market_id] = position
    runtime_state["open_position_by_market_side"][f"{market_id}|{side}"] = position
    runtime_state["executed_simulated_order_count"] = int(
        runtime_state.get("executed_simulated_order_count") or 0
    ) + 1


def _compact_runtime_state(runtime_state: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(runtime_state, sort_keys=True))


def _fill_price_from_microstructure(micro: dict[str, Any]) -> float:
    entry_ask = _float(micro.get("entry_ask"))
    if entry_ask > 0.0:
        return entry_ask
    exit_bid = _float(micro.get("executable_exit_bid_proxy"))
    return exit_bid if exit_bid > 0.0 else 1.0


def _counter_from_rows(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(field_name)
        if isinstance(value, list):
            for item in value:
                counter[str(item)] += 1
        elif value is not None:
            counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _check(
    *,
    passed: bool,
    reason_code: str,
    observed: Any,
    required: Any,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "reason_code": reason_code,
        "observed": observed,
        "required": required,
    }


def _blocking_reason_codes(checks: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        str(check["reason_code"])
        for check in checks.values()
        if check.get("passed") is not True
    )


def _with_report_id(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    report = dict(payload)
    report[field_name] = canonical_json_sha256(report)
    return report


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fresh_loop_run_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Loop Run",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_loop_enabled: `{str(report['paper_fresh_loop_enabled']).lower()}`",
            f"- paper_fresh_loop_mode: `{report['paper_fresh_loop_mode']}`",
            f"- public_data_source: `{report['paper_fresh_loop_public_data_source']}`",
            f"- provider_collection_failed: `{str(report['paper_fresh_provider_collection_failed']).lower()}`",
            f"- paper_fresh_loop_cycle_count: `{report['paper_fresh_loop_cycle_count']}`",
            f"- paper_fresh_order_intent_count: `{report['paper_fresh_order_intent_count']}`",
            f"- paper_fresh_fill_count: `{report['paper_fresh_fill_count']}`",
            f"- paper_fresh_ledger_entry_count: `{report['paper_fresh_ledger_entry_count']}`",
            f"- v8_paper_internal_handoff_allowed: `{str(report['v8_paper_internal_handoff_allowed']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["paper_fresh_loop_blocking_reason_codes"]),
            "",
        ]
    )


def _fresh_fill_simulation_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Fill Simulation",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_fill_count: `{report['paper_fresh_fill_count']}`",
            f"- paper_fresh_filled_size_sum: `{report['paper_fresh_filled_size_sum']}`",
            f"- total_synthetic_execution_cost: `{report['paper_fresh_total_synthetic_execution_cost']}`",
            f"- outcome_pnl_used: `{str(report['outcome_pnl_used']).lower()}`",
            f"- realized_pnl_used: `{str(report['realized_pnl_used']).lower()}`",
            "",
        ]
    )


def _fresh_runtime_safety_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Runtime Safety",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_runtime_safety_passed: `{str(report['paper_fresh_runtime_safety_passed']).lower()}`",
            f"- v8_paper_internal_handoff_allowed: `{str(report['v8_paper_internal_handoff_allowed']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- paper_only: `{str(report['paper_only']).lower()}`",
            f"- capital_at_risk: `{str(report['capital_at_risk']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["paper_fresh_runtime_safety_blocking_reason_codes"]),
            "",
        ]
    )


def _fresh_monitoring_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Monitoring",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_monitoring_passed: `{str(report['paper_fresh_monitoring_passed']).lower()}`",
            f"- cycle_count: `{report['cycle_count']}`",
            f"- cycle_failure_count: `{report['cycle_failure_count']}`",
            f"- candidate_decision_count: `{report['candidate_decision_count']}`",
            f"- guard_allowed_decision_count: `{report['guard_allowed_decision_count']}`",
            f"- guard_blocked_decision_count: `{report['guard_blocked_decision_count']}`",
            "",
        ]
    )


def _fresh_cumulative_monitoring_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Cumulative Monitoring",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- total_cycles: `{report['total_cycles']}`",
            f"- total_paper_intents: `{report['total_paper_intents']}`",
            f"- total_paper_fills: `{report['total_paper_fills']}`",
            f"- total_blocked_decisions: `{report['total_blocked_decisions']}`",
            f"- cycle_failure_count: `{report['cycle_failure_count']}`",
            f"- safety_violation_count: `{report['safety_violation_count']}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
        ]
    )


def _markdown_list(rows: list[str]) -> list[str]:
    return ["- none"] if not rows else [f"- `{row}`" for row in rows]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return _sha256_file_existing(path) if path.exists() else ""
