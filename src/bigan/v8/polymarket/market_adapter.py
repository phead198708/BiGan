"""BTC 15m UP/DOWN Polymarket adapter and artifact runner."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bigan.v8.paper import (
    GitHubCommentDeliveryConfig,
    PaperHarnessConfig,
    deliver_github_paper_comment,
    run_paper_trading_harness,
    summarize_paper_run,
)
from bigan.v8.phase0 import MarketData
from bigan.v8.polymarket.contracts import (
    POLYMARKET_ADAPTER_SCHEMA_VERSION,
    POLYMARKET_BTC15M_HORIZON_MS,
    POLYMARKET_BTC15M_MARKET_FAMILY,
    POLYMARKET_SOURCE,
    PolymarketAdapterError,
    PolymarketBinaryDecision,
    PolymarketBinaryMarket,
    PolymarketFeatureRow,
    PolymarketLabelRow,
    PolymarketTokenSnapshot,
    canonical_json_sha256,
)
from bigan.v8.polymarket.features import build_polymarket_feature_rows
from bigan.v8.polymarket.labels import build_polymarket_label_rows
from bigan.v8.polymarket.paper_decision import (
    PolymarketPolicySignal,
    build_polymarket_paper_decisions,
    polymarket_decisions_to_phase4,
)

DEFAULT_POLYMARKET_ADAPTER_CREATED_AT = "2026-06-22T07:00:00Z"
POLYMARKET_ADAPTER_PHASE = "polymarket_btc15m_adapter"
POLICY_SIGNAL_SOURCE_SYNTHETIC_FIXTURE = "synthetic_fixture"


@dataclass(frozen=True, slots=True)
class PolymarketAdapterRunConfig:
    """Configuration for one deterministic Polymarket BTC 15m adapter run."""

    run_id: str
    output_dir: Path | str
    repo_full_name: str = "phead198708/BiGan"
    issue_number: int = 130
    comment_post_mode: str = "dry_run"
    created_at: str = DEFAULT_POLYMARKET_ADAPTER_CREATED_AT
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.comment_post_mode not in ("dry_run", "gh_command", "direct_comment"):
            raise ValueError(
                "comment_post_mode must be dry_run, gh_command, or direct_comment"
            )
        if not self.repo_full_name.strip() or "/" not in self.repo_full_name:
            raise ValueError("repo_full_name must be owner/repo")
        if self.issue_number <= 0:
            raise ValueError("issue_number must be positive")
        _assert_safe_flags(self)

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    @property
    def adapter_dir(self) -> Path:
        return self.run_dir / "adapter"

    @property
    def paper_run_dir(self) -> Path:
        return self.run_dir / "paper_run"

    @property
    def observability_dir(self) -> Path:
        return self.run_dir / "observability"

    @property
    def github_comment_dir(self) -> Path:
        return self.run_dir / "github_comment"


@dataclass(frozen=True, slots=True)
class PolymarketAdapterRunResult:
    """Output handles for one deterministic Polymarket adapter paper run."""

    run_dir: Path
    adapter_dir: Path
    paper_run_dir: Path
    observability_dir: Path
    github_comment_dir: Path
    market: PolymarketBinaryMarket
    snapshots: tuple[PolymarketTokenSnapshot, ...]
    feature_rows: tuple[PolymarketFeatureRow, ...]
    label_rows: tuple[PolymarketLabelRow, ...]
    decisions: tuple[PolymarketBinaryDecision, ...]
    adapter_summary: dict[str, Any]
    console_summary: dict[str, Any]


def normalize_btc15m_binary_market(raw: dict[str, Any]) -> PolymarketBinaryMarket:
    """Normalize raw mocked or public-style metadata into the strict contract."""

    outcomes = _extract_outcomes(raw)
    if len(outcomes) != 2:
        raise PolymarketAdapterError("non_binary_market")
    up = _single_outcome(outcomes, "UP", "missing_up_token")
    down = _single_outcome(outcomes, "DOWN", "missing_down_token")
    title = str(raw.get("title") or raw.get("question") or "")
    slug = str(raw.get("slug") or "")
    if "btc" not in f"{title} {slug}".lower() and "bitcoin" not in (
        f"{title} {slug}".lower()
    ):
        raise PolymarketAdapterError("non_btc_market")
    start_ts = _required_int(raw, "market_start_ts", "start_ts")
    end_ts = _required_int(raw, "market_end_ts", "end_ts")
    horizon_ms = end_ts - start_ts
    if horizon_ms != POLYMARKET_BTC15M_HORIZON_MS:
        raise PolymarketAdapterError("non_15m_market")
    settlement_rule = _normalize_settlement_rule(str(raw.get("settlement_rule") or ""))
    reference_price_start = _required_float(
        raw,
        "reference_price_at_start",
        "reference_price_start",
    )
    return PolymarketBinaryMarket(
        market_id=str(raw.get("market_id") or raw.get("id") or ""),
        condition_id=str(raw.get("condition_id") or ""),
        slug=slug,
        title=title,
        market_family=POLYMARKET_BTC15M_MARKET_FAMILY,
        base_asset="BTC",
        quote_asset="USD",
        outcome_up=str(up.get("name") or up.get("outcome") or "UP").upper(),
        outcome_down=str(down.get("name") or down.get("outcome") or "DOWN").upper(),
        up_token_id=str(up.get("token_id") or up.get("tokenId") or ""),
        down_token_id=str(down.get("token_id") or down.get("tokenId") or ""),
        market_start_ts=start_ts,
        market_end_ts=end_ts,
        settlement_ts=int(raw.get("settlement_ts") or end_ts),
        horizon_ms=horizon_ms,
        reference_price_source=str(
            raw.get("reference_price_source") or "coinbase_btc_usd"
        ),
        reference_price_at_start=reference_price_start,
        settlement_rule=settlement_rule,
        status=str(raw.get("status") or "open").lower(),  # type: ignore[arg-type]
        paper_only=raw.get("paper_only", True) is True,
        capital_at_risk=raw.get("capital_at_risk", False) is True,
        broker_exchange_write_enabled=(
            raw.get("broker_exchange_write_enabled", False) is True
        ),
        live_exchange_write_enabled=(
            raw.get("live_exchange_write_enabled", False) is True
        ),
        polymarket_write_enabled=raw.get("polymarket_write_enabled", False) is True,
        wallet_signing_enabled=raw.get("wallet_signing_enabled", False) is True,
    )


def normalize_token_snapshots(
    *,
    market: PolymarketBinaryMarket,
    rows: list[dict[str, Any]],
) -> tuple[PolymarketTokenSnapshot, ...]:
    """Normalize public CLOB-style token snapshots without creating write paths."""

    snapshots: list[PolymarketTokenSnapshot] = []
    for row in rows:
        outcome = _snapshot_outcome(market, row)
        bid = _required_float(row, "bid_price", "bid")
        ask = _required_float(row, "ask_price", "ask")
        mid = float(row.get("mid_price") or (bid + ask) / 2.0)
        spread_bps = float(row.get("spread_bps") or ((ask - bid) / mid) * 10_000)
        token_id = market.token_id_for_outcome(outcome)
        snapshots.append(
            PolymarketTokenSnapshot(
                market_id=market.market_id,
                token_id=str(row.get("token_id") or token_id),
                outcome=outcome,
                ts=_required_int(row, "ts"),
                bid_price=bid,
                ask_price=ask,
                mid_price=mid,
                last_price=float(row.get("last_price") or mid),
                spread_bps=spread_bps,
                volume=float(row.get("volume") or 0.0),
                liquidity_depth=float(row.get("liquidity_depth") or 0.0),
                trade_count=int(row.get("trade_count") or 0),
                source=str(row.get("source") or POLYMARKET_SOURCE),
                read_only=row.get("read_only", True) is True,
                write_capable=row.get("write_capable", False) is True,
                paper_only=row.get("paper_only", True) is True,
                capital_at_risk=row.get("capital_at_risk", False) is True,
                broker_exchange_write_enabled=(
                    row.get("broker_exchange_write_enabled", False) is True
                ),
                live_exchange_write_enabled=(
                    row.get("live_exchange_write_enabled", False) is True
                ),
                polymarket_write_enabled=(
                    row.get("polymarket_write_enabled", False) is True
                ),
                wallet_signing_enabled=(
                    row.get("wallet_signing_enabled", False) is True
                ),
            )
        )
    _assert_snapshot_coverage(market, snapshots)
    return tuple(sorted(snapshots, key=lambda item: (item.ts, item.outcome)))


def run_polymarket_btc15m_paper_pipeline(
    *,
    config: PolymarketAdapterRunConfig,
    market_payload: dict[str, Any] | None = None,
    token_snapshot_rows: list[dict[str, Any]] | None = None,
    btc_market_rows: list[dict[str, Any]] | None = None,
) -> PolymarketAdapterRunResult:
    """Run deterministic mocked BTC 15m adapter through v8 paper evidence."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"polymarket run_dir already exists: {run_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(run_dir)
    config.adapter_dir.mkdir(parents=True)

    market = normalize_btc15m_binary_market(
        market_payload or synthetic_btc15m_market_payload()
    )
    snapshots = normalize_token_snapshots(
        market=market,
        rows=token_snapshot_rows or synthetic_token_snapshot_rows(market),
    )
    btc_rows = _market_data_rows(btc_market_rows or synthetic_btc_market_rows(market))
    feature_rows = build_polymarket_feature_rows(
        market=market,
        token_snapshots=snapshots,
        btc_market_data=btc_rows,
    )
    label_rows = build_polymarket_label_rows(
        market=market,
        token_snapshots=snapshots,
        reference_price_end=btc_rows[-1].effective_mid_price,
    )
    decisions = build_polymarket_paper_decisions(
        market=market,
        feature_rows=feature_rows,
        token_snapshots=snapshots,
        policy_signals=_default_policy_signals(feature_rows),
    )
    phase4_decisions = polymarket_decisions_to_phase4(
        decisions=decisions,
        labels=label_rows,
    )
    adapter_paths = _write_adapter_artifacts(
        adapter_dir=config.adapter_dir,
        market=market,
        snapshots=snapshots,
        feature_rows=feature_rows,
        label_rows=label_rows,
        decisions=decisions,
        created_at=config.created_at,
    )
    paper_result = run_paper_trading_harness(
        decisions=phase4_decisions,
        config=PaperHarnessConfig(
            run_id=config.run_id,
            candidate_run_id=f"{config.run_id}-polymarket-candidate",
            model_sha256=_sha256_text("polymarket-btc15m-fixture-model"),
            policy_dataset_hash=_sha256_file(adapter_paths["feature_rows"]),
            split_hash=_sha256_text("polymarket-btc15m-fixture-split"),
            upstream_training_report_sha256=_sha256_text(
                "polymarket-btc15m-training-report"
            ),
            upstream_validation_report_sha256=_sha256_text(
                "polymarket-btc15m-validation-report"
            ),
            output_dir=config.paper_run_dir,
            created_at=config.created_at,
            overwrite_existing=True,
            broker_write_enabled=False,
            paper_only=True,
            capital_at_risk=False,
        ),
    )
    _write_observability_inputs(
        run_dir=config.paper_run_dir,
        market=market,
        feature_rows=feature_rows,
        decisions=decisions,
        paper_result=paper_result,
        created_at=config.created_at,
    )
    observability_result = summarize_paper_run(
        run_dir=config.paper_run_dir,
        output_dir=config.observability_dir,
        created_at=config.created_at,
    )
    comment_result = deliver_github_paper_comment(
        observability_dir=config.observability_dir,
        config=GitHubCommentDeliveryConfig(
            repo_full_name=config.repo_full_name,
            issue_number=config.issue_number,
            output_dir=config.github_comment_dir,
            post_mode=config.comment_post_mode,  # type: ignore[arg-type]
            created_at=config.created_at,
        ),
    )
    adapter_summary = _adapter_summary(
        config=config,
        market=market,
        snapshots=snapshots,
        feature_rows=feature_rows,
        label_rows=label_rows,
        decisions=decisions,
        adapter_paths=adapter_paths,
        paper_summary_path=config.paper_run_dir / "paper_run_summary.json",
        observability_report_path=(
            observability_result.artifact_paths["observability_report"]
        ),
        github_comment_payload_path=comment_result.artifact_paths["payload"],
    )
    _write_json(config.adapter_dir / "polymarket_adapter_summary.json", adapter_summary)
    console_summary = {
        "run_id": config.run_id,
        "run_dir": str(config.run_dir),
        "adapter_summary_path": str(
            config.adapter_dir / "polymarket_adapter_summary.json"
        ),
        "paper_run_summary_path": str(config.paper_run_dir / "paper_run_summary.json"),
        "observability_report_path": str(
            observability_result.artifact_paths["observability_report"]
        ),
        "github_comment_payload_path": str(comment_result.artifact_paths["payload"]),
        "phase6_deployment_status": (
            paper_result.phase6_result.report.deployment_status
        ),
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "policy_signal_source": POLICY_SIGNAL_SOURCE_SYNTHETIC_FIXTURE,
        "trained_model_used": False,
    }
    _write_json(config.run_dir / "polymarket_pipeline_summary.json", console_summary)
    return PolymarketAdapterRunResult(
        run_dir=config.run_dir,
        adapter_dir=config.adapter_dir,
        paper_run_dir=config.paper_run_dir,
        observability_dir=config.observability_dir,
        github_comment_dir=config.github_comment_dir,
        market=market,
        snapshots=snapshots,
        feature_rows=feature_rows,
        label_rows=label_rows,
        decisions=decisions,
        adapter_summary=adapter_summary,
        console_summary=console_summary,
    )


def synthetic_btc15m_market_payload() -> dict[str, Any]:
    start_ts = 1_780_000_000_000
    end_ts = start_ts + POLYMARKET_BTC15M_HORIZON_MS
    return {
        "market_id": "pm-btc-15m-1780000000",
        "condition_id": "0xconditionbtc15m0001",
        "slug": "bitcoin-up-or-down-june-22-1780000000",
        "title": "Bitcoin Up or Down - 15m",
        "market_start_ts": start_ts,
        "market_end_ts": end_ts,
        "settlement_ts": end_ts,
        "reference_price_source": "coinbase_btc_usd",
        "reference_price_at_start": 65_000.0,
        "settlement_rule": (
            "UP wins if the BTC reference price at market end is greater than "
            "the BTC reference price at market start; otherwise DOWN wins."
        ),
        "status": "open",
        "outcomes": [
            {"name": "UP", "token_id": "token-up-001"},
            {"name": "DOWN", "token_id": "token-down-001"},
        ],
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def synthetic_btc_market_rows(
    market: PolymarketBinaryMarket | None = None,
) -> list[dict[str, Any]]:
    resolved_market = market or normalize_btc15m_binary_market(
        synthetic_btc15m_market_payload()
    )
    rows = []
    for index in range(16):
        ts = resolved_market.market_start_ts + index * 60_000
        mid = 65_000.0 + index * 8.0 + (index % 3) * 1.5
        rows.append(
            {
                "ts": ts,
                "available_at_ts": ts,
                "source": "coinbase_btc_usd_fixture",
                "instrument_id": "BTC-USD",
                "bid_price": mid - 0.5,
                "ask_price": mid + 0.5,
                "mid_price": mid,
                "last_price": mid,
                "volume": 100.0 + index,
                "trade_count": 10 + index,
                "liquidity_depth": 10_000.0 + 50.0 * index,
                "timeframe_ms": 60_000,
                "sequence": index,
            }
        )
    return rows


def synthetic_token_snapshot_rows(market: PolymarketBinaryMarket) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(6):
        ts = market.market_start_ts + index * 60_000
        up_mid = 0.51 + index * 0.012
        down_mid = 1.0 - up_mid
        for outcome, mid, depth in (
            ("UP", up_mid, 2_000.0 + 50.0 * index),
            ("DOWN", down_mid, 1_900.0 - 20.0 * index),
        ):
            rows.append(
                {
                    "ts": ts,
                    "market_id": market.market_id,
                    "outcome": outcome,
                    "token_id": market.token_id_for_outcome(outcome),  # type: ignore[arg-type]
                    "bid_price": mid - 0.01,
                    "ask_price": mid + 0.01,
                    "last_price": mid,
                    "volume": 500.0 + index,
                    "liquidity_depth": depth,
                    "trade_count": 4 + index,
                    "source": POLYMARKET_SOURCE,
                    "read_only": True,
                    "write_capable": False,
                    "paper_only": True,
                    "capital_at_risk": False,
                    "polymarket_write_enabled": False,
                    "wallet_signing_enabled": False,
                }
            )
    return rows


def _write_adapter_artifacts(
    *,
    adapter_dir: Path,
    market: PolymarketBinaryMarket,
    snapshots: tuple[PolymarketTokenSnapshot, ...],
    feature_rows: tuple[PolymarketFeatureRow, ...],
    label_rows: tuple[PolymarketLabelRow, ...],
    decisions: tuple[PolymarketBinaryDecision, ...],
    created_at: str,
) -> dict[str, Path]:
    paths = {
        "market_manifest": adapter_dir / "polymarket_market_manifest.json",
        "token_snapshots": adapter_dir / "polymarket_token_snapshots.jsonl",
        "feature_rows": adapter_dir / "polymarket_feature_rows.jsonl",
        "label_rows": adapter_dir / "polymarket_label_rows.jsonl",
        "paper_decisions": adapter_dir / "polymarket_paper_decisions.jsonl",
    }
    _write_json(
        paths["market_manifest"],
        {
            "schema_version": POLYMARKET_ADAPTER_SCHEMA_VERSION,
            "phase": POLYMARKET_ADAPTER_PHASE,
            "created_at": created_at,
            **market.to_dict(),
        },
    )
    _write_jsonl(paths["token_snapshots"], [row.to_dict() for row in snapshots])
    _write_jsonl(paths["feature_rows"], [row.to_dict() for row in feature_rows])
    _write_jsonl(paths["label_rows"], [row.to_dict() for row in label_rows])
    _write_jsonl(paths["paper_decisions"], [row.to_dict() for row in decisions])
    return paths


def _write_observability_inputs(
    *,
    run_dir: Path,
    market: PolymarketBinaryMarket,
    feature_rows: tuple[PolymarketFeatureRow, ...],
    decisions: tuple[PolymarketBinaryDecision, ...],
    paper_result: Any,
    created_at: str,
) -> None:
    if paper_result.phase5_result.report_path is None:
        raise PolymarketAdapterError("phase5_report_missing")
    if paper_result.phase6_result.report_path is None:
        raise PolymarketAdapterError("phase6_report_missing")
    _augment_report_safety_flags(paper_result.phase5_result.report_path)
    _augment_report_safety_flags(paper_result.phase6_result.report_path)
    phase5_report = paper_result.phase5_result.report
    safety_action = phase5_report.safety_action
    phase6_report = paper_result.phase6_result.report
    feed_count = len(feature_rows)
    heartbeat_rows = [
        {
            "run_id": market.market_id,
            "heartbeat_ts": row.decision_ts,
            "last_feed_event_ts": row.decision_ts,
            "last_feed_sequence": index,
            "feed_event_count": index + 1,
            "paper_only": True,
            "capital_at_risk": False,
            "broker_exchange_write_enabled": False,
            "live_exchange_write_enabled": False,
        }
        for index, row in enumerate(feature_rows)
    ]
    periodic_rows = heartbeat_rows[:: max(1, len(heartbeat_rows) // 2)] or heartbeat_rows
    _write_jsonl(run_dir / "paper_soak_heartbeat.jsonl", heartbeat_rows)
    _write_jsonl(run_dir / "paper_soak_periodic_summaries.jsonl", periodic_rows)
    feed_report = {
        "schema_version": POLYMARKET_ADAPTER_SCHEMA_VERSION,
        "run_id": market.market_id,
        "feed_event_count": feed_count,
        "first_event_ts": feature_rows[0].decision_ts if feature_rows else None,
        "last_event_ts": feature_rows[-1].decision_ts if feature_rows else None,
        "feed_gap_count": 0,
        "max_feed_gap_seconds": 60.0,
        "feed_late_event_count": 0,
        "feed_out_of_order_count": 0,
        "provider_disconnect_count": 0,
        "provider_reconnect_count": 0,
        "provider_error_count": 0,
        "stale_event_count": 0,
        "empty_response_count": 0,
        "rate_limit_count": 0,
        "last_successful_receive_ts": (
            feature_rows[-1].decision_ts if feature_rows else None
        ),
        "feed_health_passed": True,
        "feed_health_reason_codes": [],
        "acceptance": {"passed": True, "reason_codes": []},
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
    }
    _write_json(run_dir / "feed_health_report.json", feed_report)
    fills = paper_result.fills
    total_execution_cost = sum(fill.total_execution_cost for fill in fills)
    cumulative_net_return = sum(fill.net_return for fill in fills)
    artifact_paths = dict(paper_result.artifact_paths)
    artifact_paths.update(
        {
            "feed_health_report": run_dir / "feed_health_report.json",
            "paper_soak_heartbeat": run_dir / "paper_soak_heartbeat.jsonl",
            "paper_soak_periodic_summaries": (
                run_dir / "paper_soak_periodic_summaries.jsonl"
            ),
        }
    )
    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
        if path.exists() and name != "paper_bundle_manifest"
    }
    summary = {
        "schema_version": POLYMARKET_ADAPTER_SCHEMA_VERSION,
        "run_id": paper_result.paper_report.run_id,
        "started_at": _ts_to_utc(market.market_start_ts),
        "ended_at": _ts_to_utc(market.market_end_ts),
        "duration_seconds": market.horizon_ms // 1000,
        "configured_duration_seconds": POLYMARKET_BTC15M_HORIZON_MS // 1000,
        "stop_reason": "fixture_complete",
        "feed_event_count": feed_count,
        "feed_gap_count": 0,
        "max_feed_gap_seconds": 60.0,
        "feed_late_event_count": 0,
        "feed_out_of_order_count": 0,
        "provider_disconnect_count": 0,
        "provider_reconnect_count": 0,
        "provider_error_count": 0,
        "stale_event_count": 0,
        "empty_response_count": 0,
        "rate_limit_count": 0,
        "last_successful_receive_ts": feed_report["last_successful_receive_ts"],
        "feed_health_passed": True,
        "feed_health_reason_codes": [],
        "heartbeat_count": len(heartbeat_rows),
        "periodic_summary_count": len(periodic_rows),
        "row_count": len(paper_result.orders),
        "order_count": len(paper_result.orders),
        "fill_count": len(fills),
        "fill_rate": len(fills) / len(paper_result.orders)
        if paper_result.orders
        else 0.0,
        "ledger_entry_count": len(paper_result.ledger_entries),
        "final_position_count": len(paper_result.positions),
        "mean_net_return": paper_result.paper_report.mean_net_return,
        "cumulative_net_return": cumulative_net_return,
        "max_drawdown": paper_result.paper_report.max_drawdown,
        "total_execution_cost": total_execution_cost,
        "mean_execution_cost": total_execution_cost / len(fills) if fills else 0.0,
        "shadow_live_correlation": phase5_report.drift_metrics[
            "shadow_live_correlation"
        ],
        "pnl_drift": phase5_report.drift_metrics["mean_pnl_drift"],
        "cost_drift_ratio": phase5_report.drift_metrics["cost_drift_ratio"],
        "regime_mismatch_rate": phase5_report.drift_metrics["regime_mismatch_rate"],
        "phase5_passed": paper_result.phase5_result.passed,
        "phase5_kill_switch_triggered": safety_action["kill_switch_triggered"],
        "phase5_reason_codes": list(safety_action["reason_codes"]),
        "rollback_model_id": safety_action["rollback_model_id"],
        "phase6_candidate_identity_verified": (
            phase6_report.candidate_identity_verified
        ),
        "phase6_deployment_status": phase6_report.deployment_status,
        "feed_mode": "polymarket-mocked-btc15m",
        "real_live_data": False,
        "deterministic_replay": True,
        "provider_name": "mocked_polymarket",
        "instrument_id": market.slug,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "policy_signal_source": POLICY_SIGNAL_SOURCE_SYNTHETIC_FIXTURE,
        "trained_model_used": False,
        "artifact_hashes": artifact_hashes,
    }
    _write_json(run_dir / "paper_run_summary.json", summary)
    artifact_paths["paper_run_summary"] = run_dir / "paper_run_summary.json"
    _write_json(
        run_dir / "paper_bundle_manifest.json",
        {
            **paper_result.bundle_manifest,
            "schema_version": "bigan-v8-polymarket-paper-bundle-v1",
            "feed_health_passed": True,
            "paper_only": True,
            "capital_at_risk": False,
            "broker_exchange_write_enabled": False,
            "live_exchange_write_enabled": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "policy_signal_source": POLICY_SIGNAL_SOURCE_SYNTHETIC_FIXTURE,
            "trained_model_used": False,
            "phase6_deployment_status": phase6_report.deployment_status,
            "artifacts": {
                name: {
                    "path": path.name,
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in sorted(artifact_paths.items())
                if path.exists() and name != "paper_bundle_manifest"
            },
        },
    )


def _adapter_summary(
    *,
    config: PolymarketAdapterRunConfig,
    market: PolymarketBinaryMarket,
    snapshots: tuple[PolymarketTokenSnapshot, ...],
    feature_rows: tuple[PolymarketFeatureRow, ...],
    label_rows: tuple[PolymarketLabelRow, ...],
    decisions: tuple[PolymarketBinaryDecision, ...],
    adapter_paths: dict[str, Path],
    paper_summary_path: Path,
    observability_report_path: Path,
    github_comment_payload_path: Path,
) -> dict[str, Any]:
    hashes = {
        name: _sha256_file(path)
        for name, path in sorted(adapter_paths.items())
        if path.exists()
    }
    hashes.update(
        {
            "paper_run_summary": _sha256_file(paper_summary_path),
            "observability_report": _sha256_file(observability_report_path),
            "github_comment_payload": _sha256_file(github_comment_payload_path),
        }
    )
    return {
        "schema_version": POLYMARKET_ADAPTER_SCHEMA_VERSION,
        "phase": POLYMARKET_ADAPTER_PHASE,
        "run_id": config.run_id,
        "created_at": config.created_at,
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "slug": market.slug,
        "market_family": market.market_family,
        "horizon_ms": market.horizon_ms,
        "up_token_id": market.up_token_id,
        "down_token_id": market.down_token_id,
        "settlement_rule": market.settlement_rule,
        "snapshot_count": len(snapshots),
        "feature_row_count": len(feature_rows),
        "label_row_count": len(label_rows),
        "decision_count": len(decisions),
        "trade_decision_count": sum(
            1 for decision in decisions if decision.selected_outcome != "NO_TRADE"
        ),
        "no_trade_decision_count": sum(
            1 for decision in decisions if decision.selected_outcome == "NO_TRADE"
        ),
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "policy_signal_source": POLICY_SIGNAL_SOURCE_SYNTHETIC_FIXTURE,
        "trained_model_used": False,
        "artifact_hashes": hashes,
        "artifact_paths": {
            name: str(path) for name, path in sorted(adapter_paths.items())
        },
    }


def _default_policy_signals(
    feature_rows: tuple[PolymarketFeatureRow, ...],
) -> tuple[PolymarketPolicySignal, ...]:
    signals: list[PolymarketPolicySignal] = []
    for index, row in enumerate(feature_rows):
        up_mid = float(row.features["up_token_mid_price"] or 0.0)
        estimated_probability = min(0.95, up_mid + 0.08 + index * 0.002)
        signals.append(
            PolymarketPolicySignal(
                decision_ts=row.decision_ts,
                action=0.72,
                confidence=0.82,
                score=0.78 + index * 0.005,
                estimated_up_probability=estimated_probability,
            )
        )
    return tuple(signals)


def _extract_outcomes(raw: dict[str, Any]) -> list[dict[str, Any]]:
    raw_outcomes = raw.get("outcomes") or raw.get("tokens")
    if not isinstance(raw_outcomes, list):
        raise PolymarketAdapterError("missing_outcomes")
    return [dict(item) for item in raw_outcomes]


def _single_outcome(
    outcomes: list[dict[str, Any]],
    target: str,
    missing_code: str,
) -> dict[str, Any]:
    matches = [
        outcome
        for outcome in outcomes
        if str(outcome.get("name") or outcome.get("outcome") or "").upper() == target
    ]
    if len(matches) != 1:
        raise PolymarketAdapterError(missing_code)
    if not str(matches[0].get("token_id") or matches[0].get("tokenId") or "").strip():
        raise PolymarketAdapterError(missing_code)
    return matches[0]


def _snapshot_outcome(
    market: PolymarketBinaryMarket,
    row: dict[str, Any],
) -> str:
    token_id = str(row.get("token_id") or row.get("tokenId") or "")
    if token_id == market.up_token_id:
        return "UP"
    if token_id == market.down_token_id:
        return "DOWN"
    outcome = str(row.get("outcome") or "").upper()
    if outcome in {"UP", "DOWN"}:
        return outcome
    raise PolymarketAdapterError("unknown_token_outcome")


def _normalize_settlement_rule(raw_rule: str) -> str:
    normalized = " ".join(raw_rule.lower().split())
    if (
        ("greater than" in normalized or ">" in normalized or "higher" in normalized)
        and "start" in normalized
        and "end" in normalized
    ):
        return "btc_reference_price_end_gt_start_up_else_down"
    raise PolymarketAdapterError("unknown_settlement_rule")


def _assert_snapshot_coverage(
    market: PolymarketBinaryMarket,
    snapshots: list[PolymarketTokenSnapshot],
) -> None:
    outcomes = {snapshot.outcome for snapshot in snapshots}
    if "UP" not in outcomes:
        raise PolymarketAdapterError("missing_up_snapshot")
    if "DOWN" not in outcomes:
        raise PolymarketAdapterError("missing_down_snapshot")
    invalid_tokens = [
        snapshot.token_id
        for snapshot in snapshots
        if snapshot.token_id
        not in {market.up_token_id, market.down_token_id}
    ]
    if invalid_tokens:
        raise PolymarketAdapterError("unknown_token_id")


def _market_data_rows(rows: list[dict[str, Any]]) -> tuple[MarketData, ...]:
    return tuple(MarketData(**row) for row in sorted(rows, key=lambda item: item["ts"]))


def _required_int(row: dict[str, Any], *names: str) -> int:
    for name in names:
        if row.get(name) is not None:
            return int(row[name])
    raise PolymarketAdapterError(f"missing_{names[0]}")


def _required_float(row: dict[str, Any], *names: str) -> float:
    for name in names:
        if row.get(name) is not None:
            return float(row[name])
    raise PolymarketAdapterError(f"missing_{names[0]}")


def _assert_safe_flags(config: PolymarketAdapterRunConfig) -> None:
    if config.paper_only is not True:
        raise ValueError("paper_only must be true")
    if config.capital_at_risk is not False:
        raise ValueError("capital_at_risk must be false")
    if config.broker_exchange_write_enabled:
        raise ValueError("broker/exchange writes are forbidden")
    if config.live_exchange_write_enabled:
        raise ValueError("live exchange writes are forbidden")
    if config.polymarket_write_enabled:
        raise ValueError("polymarket writes are forbidden")
    if config.wallet_signing_enabled:
        raise ValueError("wallet signing is forbidden")


def _ts_to_utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _augment_report_safety_flags(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "paper_only": True,
            "capital_at_risk": False,
            "broker_exchange_write_enabled": False,
            "live_exchange_write_enabled": False,
        }
    )
    _write_json(path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                _json_ready(row),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_ready(value.model_dump())
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def metadata_hash(payload: dict[str, Any]) -> str:
    return canonical_json_sha256(payload)
