"""Settlement-aware label construction for Polymarket BTC corpus rows."""

from __future__ import annotations

from collections import Counter
from typing import Any

from bigan.v8.polymarket.corpus.contracts import (
    POLYMARKET_SELL_BEFORE_CLOSE_LABEL_REDESIGN_REPORT_SCHEMA_VERSION,
    POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION,
    CorpusLabelAction,
    PolymarketCorpusBookSnapshot,
    PolymarketCorpusBuildConfig,
    PolymarketCorpusFeatureRow,
    PolymarketCorpusLabelRow,
    PolymarketCorpusMarket,
    PolymarketCorpusResolutionEvent,
    safety_fields,
    stable_hash,
)
from bigan.v8.polymarket.rules import PolymarketResolutionRule


def build_polymarket_corpus_label_rows(
    *,
    markets: tuple[PolymarketCorpusMarket, ...],
    rules: dict[str, PolymarketResolutionRule],
    book_snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    resolution_events: dict[str, PolymarketCorpusResolutionEvent],
    feature_rows: tuple[PolymarketCorpusFeatureRow, ...],
    config: PolymarketCorpusBuildConfig,
) -> tuple[PolymarketCorpusLabelRow, ...]:
    """Build future-aware labels from feature rows and Phase 1 rule semantics."""

    market_by_id = {market.market_id: market for market in markets}
    snapshots_by_market = _snapshots_by_market(book_snapshots)
    rows: list[PolymarketCorpusLabelRow] = []
    for feature in sorted(feature_rows, key=lambda item: (item.decision_ts, item.market_id)):
        market = market_by_id[feature.market_id]
        rule = rules[market.market_id]
        resolution = resolution_events[market.market_id]
        actions: list[CorpusLabelAction] = ["NO_TRADE"]
        if config.include_settlement_labels:
            actions.extend(
                [
                    "BUY_UP_HOLD_TO_SETTLEMENT",
                    "BUY_DOWN_HOLD_TO_SETTLEMENT",
                ]
            )
        if config.include_trade_labels:
            actions.extend(
                [
                    "BUY_UP_SELL_BEFORE_CLOSE",
                    "BUY_DOWN_SELL_BEFORE_CLOSE",
                ]
            )
        for action in actions:
            rows.append(
                _label_for_action(
                    market=market,
                    rule=rule,
                    resolution=resolution,
                    snapshots=snapshots_by_market[market.market_id],
                    feature=feature,
                    action=action,
                    config=config,
                )
            )
    if not rows:
        raise ValueError("no Polymarket corpus labels")
    return tuple(sorted(rows, key=lambda item: (item.decision_ts, item.market_id, item.action)))


def build_sell_before_close_label_redesign_report(
    *,
    label_rows: tuple[PolymarketCorpusLabelRow, ...],
    config: PolymarketCorpusBuildConfig,
) -> dict[str, Any]:
    """Summarize executable sell-before-close label quality."""

    sell_rows = [row for row in label_rows if row.action.endswith("SELL_BEFORE_CLOSE")]
    class_counts = Counter(row.sell_before_close_execution_class for row in sell_rows)
    side_counts = Counter(row.outcome for row in sell_rows)
    reason_counts: Counter[str] = Counter()
    leakage_failures = []
    executable_rows = []
    theoretical_rows = []
    for row in sell_rows:
        exit_path = row.sell_before_close_exit_path or {}
        for reason_code in _exit_path_reason_codes(exit_path):
            reason_counts[str(reason_code)] += 1
        if exit_path.get("label_source") == "fixed_terminal_bid_only":
            leakage_failures.append(
                {
                    "market_id": row.market_id,
                    "decision_ts": row.decision_ts,
                    "action": row.action,
                    "reason": "fixed_terminal_bid_only_label",
                }
            )
        if row.sell_before_close_execution_class == "realizable_sell_before_close":
            executable_rows.append(row)
        elif row.sell_before_close_execution_class == "theoretical_sell_before_close":
            theoretical_rows.append(row)
    reason_codes = _label_gate_reason_codes(
        sell_rows=sell_rows,
        leakage_failures=leakage_failures,
    )
    report = {
        "schema_version": POLYMARKET_SELL_BEFORE_CLOSE_LABEL_REDESIGN_REPORT_SCHEMA_VERSION,
        "sell_before_close_label_schema_version": (
            POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION
        ),
        "fixed_terminal_bid_only_labels_allowed": False,
        "fixed_terminal_bid_only_labels_accepted": len(leakage_failures) == 0,
        "uses_intraround_exit_opportunity_model": True,
        "uses_queue_fill_probability_model": True,
        "uses_executable_liquidity_features": True,
        "sell_before_close_label_count": len(sell_rows),
        "sell_before_close_execution_class_counts": dict(sorted(class_counts.items())),
        "sell_before_close_side_counts": dict(sorted(side_counts.items())),
        "exit_path_reason_code_counts": dict(sorted(reason_counts.items())),
        "realizable_sell_before_close_count": len(executable_rows),
        "theoretical_sell_before_close_count": len(theoretical_rows),
        "non_executable_sell_before_close_count": class_counts[
            "non_executable_sell_before_close"
        ],
        "average_queue_fill_probability": _mean(
            [row.queue_fill_probability_estimate for row in sell_rows]
        ),
        "average_executable_liquidity_notional": _mean(
            [row.executable_liquidity_notional for row in sell_rows]
        ),
        "average_theoretical_terminal_bid_return": _mean(
            [row.theoretical_terminal_bid_return for row in sell_rows]
        ),
        "average_realized_executable_sell_before_close_return": _mean(
            [row.realized_executable_sell_before_close_return for row in sell_rows]
        ),
        "average_execution_gap_return": _mean(
            [row.execution_gap_return for row in sell_rows]
        ),
        "min_exit_notional": float(config.sell_before_close_min_exit_notional),
        "min_queue_fill_probability": float(
            config.sell_before_close_min_queue_fill_probability
        ),
        "exit_buffer_seconds": int(config.sell_before_close_exit_buffer_seconds),
        "label_gate_passed": not reason_codes,
        "label_gate_reason_codes": reason_codes,
        "theoretical_sell_before_close_rows": (
            _theoretical_sell_before_close_rows(
                theoretical_rows,
                config=config,
            )
        ),
        "top_execution_gap_examples": _top_execution_gap_examples(sell_rows),
        "leakage_failures": leakage_failures,
        **safety_fields(),
    }
    report["sell_before_close_label_redesign_report_id"] = stable_hash(report)
    return report


def sell_before_close_label_redesign_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown summary of the sell-before-close redesign report."""

    lines = [
        "# SELL_BEFORE_CLOSE Label Redesign Report",
        "",
        f"- schema_version: `{report['schema_version']}`",
        "- fixed_terminal_bid_only_labels_allowed: "
        f"`{str(report['fixed_terminal_bid_only_labels_allowed']).lower()}`",
        "- fixed_terminal_bid_only_labels_accepted: "
        f"`{str(report['fixed_terminal_bid_only_labels_accepted']).lower()}`",
        "- uses_intraround_exit_opportunity_model: "
        f"`{str(report['uses_intraround_exit_opportunity_model']).lower()}`",
        "- uses_queue_fill_probability_model: "
        f"`{str(report['uses_queue_fill_probability_model']).lower()}`",
        "",
        "## Execution Classes",
        "",
    ]
    for class_name, count in report["sell_before_close_execution_class_counts"].items():
        lines.append(f"- {class_name}: {count}")
    lines.extend(
        [
            "",
            "## Top Execution Gaps",
            "",
        ]
    )
    for row in report["top_execution_gap_examples"][:10]:
        lines.append(
            "- "
            f"{row['decision_ts']} {row['action']} "
            f"{row['sell_before_close_execution_class']} "
            f"theoretical={row['theoretical_terminal_bid_return']} "
            f"executable={row['realized_executable_sell_before_close_return']} "
            f"gap={row['execution_gap_return']}"
        )
    if report.get("theoretical_sell_before_close_rows"):
        lines.extend(
            [
                "",
                "## Theoretical Positive Rows",
                "",
            ]
        )
        for row in report["theoretical_sell_before_close_rows"]:
            lines.append(
                "- "
                f"{row['decision_ts']} {row['action']} {row['outcome']} "
                f"entry_ask={row['entry_ask']} "
                f"terminal_bid={row['terminal_bid']} "
                f"best_candidate_bid={row['best_candidate_bid']} "
                f"reason={','.join(row['exit_path_reason_codes'])}"
            )
    lines.extend(
        [
            "",
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )
    return "\n".join(lines)


def _label_for_action(
    *,
    market: PolymarketCorpusMarket,
    rule: PolymarketResolutionRule,
    resolution: PolymarketCorpusResolutionEvent,
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    feature: PolymarketCorpusFeatureRow,
    action: CorpusLabelAction,
    config: PolymarketCorpusBuildConfig,
) -> PolymarketCorpusLabelRow:
    if action == "NO_TRADE":
        return PolymarketCorpusLabelRow(
            market_id=market.market_id,
            condition_id=market.condition_id,
            slug=market.slug,
            market_family=market.market_family,
            horizon_ms=market.horizon_ms,
            decision_ts=feature.decision_ts,
            action=action,
            outcome="NONE",
            entry_bid=0.0,
            entry_ask=0.0,
            entry_mid=0.0,
            exit_bid=0.0,
            exit_ask=0.0,
            settlement_payout=0.0,
            realized_trade_return=0.0,
            settlement_return=0.0,
            total_net_return=0.0,
            total_net_pnl_per_notional=0.0,
            fees=0.0,
            slippage=0.0,
            liquidity_impact=0.0,
            is_positive=False,
            resolved_outcome=resolution.resolved_outcome,
            resolution_status=resolution.resolution_status,
            comparator=rule.comparator,
            tie_breaker=rule.tie_breaker,
            resolution_rule_sha256=rule.raw_rule_sha256,
            raw_resolution_sha256=resolution.raw_resolution_sha256,
        )
    outcome = "UP" if "_UP_" in action else "DOWN"
    entry = _last_snapshot(snapshots=snapshots, outcome=outcome, decision_ts=feature.decision_ts)
    if entry is None:
        raise ValueError("missing entry snapshot for label")
    fees = 0.0002
    slippage = max(0.0001, (entry.ask_price - entry.bid_price) / 2.0)
    liquidity_impact = 0.00005 if entry.liquidity_depth > 0.0 else 0.001
    exit_path: dict[str, Any] | None = None
    execution_class = "not_applicable"
    label_uses_executable_exit_path = False
    theoretical_terminal_bid_return = 0.0
    realized_executable_sell_before_close_return = 0.0
    execution_gap_return = 0.0
    queue_fill_probability_estimate = 0.0
    executable_liquidity_notional = 0.0
    if action.endswith("HOLD_TO_SETTLEMENT"):
        payout = resolution.payout_up if outcome == "UP" else resolution.payout_down
        realized_trade_return = 0.0
        settlement_return = payout / entry.ask_price - 1.0
        gross_pnl_per_notional = payout - entry.ask_price
        exit_bid = 0.0
        exit_ask = 0.0
    else:
        exit_path = _sell_before_close_exit_path(
            market=market,
            snapshots=snapshots,
            outcome=outcome,
            entry=entry,
            decision_ts=feature.decision_ts,
            config=config,
        )
        execution_class = exit_path["exit_path_quality"]
        label_uses_executable_exit_path = bool(
            exit_path["label_uses_executable_exit_path"]
        )
        theoretical_terminal_bid_return = float(
            exit_path["theoretical_terminal_bid_return"]
        )
        realized_executable_sell_before_close_return = float(
            exit_path["realized_executable_sell_before_close_return"]
        )
        execution_gap_return = float(exit_path["execution_gap_return"])
        queue_fill_probability_estimate = float(
            exit_path["queue_fill_probability_estimate"]
        )
        executable_liquidity_notional = float(
            exit_path["executable_liquidity_notional"]
        )
        payout = 0.0
        exit_bid = float(exit_path["best_executable_exit_price"])
        exit_ask = float(exit_path["best_executable_exit_ask"])
        realized_trade_return = realized_executable_sell_before_close_return
        settlement_return = 0.0
        gross_pnl_per_notional = (
            exit_bid - entry.ask_price
            if label_uses_executable_exit_path
            else -entry.ask_price
        )
    total_net_return = (
        realized_trade_return
        + settlement_return
        - fees
        - slippage
        - liquidity_impact
    )
    total_net_pnl_per_notional = (
        gross_pnl_per_notional
        - fees
        - slippage
        - liquidity_impact
    )
    return PolymarketCorpusLabelRow(
        market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        market_family=market.market_family,
        horizon_ms=market.horizon_ms,
        decision_ts=feature.decision_ts,
        action=action,
        outcome=outcome,
        entry_bid=entry.bid_price,
        entry_ask=entry.ask_price,
        entry_mid=entry.mid_price,
        exit_bid=exit_bid,
        exit_ask=exit_ask,
        settlement_payout=payout,
        realized_trade_return=realized_trade_return,
        settlement_return=settlement_return,
        total_net_return=total_net_return,
        total_net_pnl_per_notional=total_net_pnl_per_notional,
        fees=fees,
        slippage=slippage,
        liquidity_impact=liquidity_impact,
        is_positive=total_net_return > 0.0,
        resolved_outcome=resolution.resolved_outcome,
        resolution_status=resolution.resolution_status,
        comparator=rule.comparator,
        tie_breaker=rule.tie_breaker,
        resolution_rule_sha256=rule.raw_rule_sha256,
        raw_resolution_sha256=resolution.raw_resolution_sha256,
        sell_before_close_execution_class=execution_class,  # type: ignore[arg-type]
        sell_before_close_exit_path=exit_path,
        label_uses_executable_exit_path=label_uses_executable_exit_path,
        theoretical_terminal_bid_return=theoretical_terminal_bid_return,
        realized_executable_sell_before_close_return=(
            realized_executable_sell_before_close_return
        ),
        execution_gap_return=execution_gap_return,
        queue_fill_probability_estimate=queue_fill_probability_estimate,
        executable_liquidity_notional=executable_liquidity_notional,
    )


def _sell_before_close_exit_path(
    *,
    market: PolymarketCorpusMarket,
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    outcome: str,
    entry: PolymarketCorpusBookSnapshot,
    decision_ts: int,
    config: PolymarketCorpusBuildConfig,
) -> dict[str, Any]:
    exit_window_start_ts = decision_ts + 1
    exit_window_end_ts = (
        market.market_end_ts - int(config.sell_before_close_exit_buffer_seconds) * 1000
    )
    candidates = [
        snapshot
        for snapshot in snapshots
        if snapshot.outcome == outcome
        and exit_window_start_ts <= snapshot.ts <= exit_window_end_ts
        and snapshot.available_at_ts <= market.market_end_ts
    ]
    terminal_snapshot = candidates[-1] if candidates else None
    theoretical_terminal_bid_return = (
        terminal_snapshot.bid_price / entry.ask_price - 1.0
        if terminal_snapshot is not None
        else -1.0
    )
    scored_candidates = [
        _score_exit_candidate(
            snapshot=snapshot,
            entry=entry,
            min_exit_notional=float(config.sell_before_close_min_exit_notional),
        )
        for snapshot in candidates
    ]
    terminal_candidate = scored_candidates[-1] if scored_candidates else None
    executable_candidates = [
        candidate
        for candidate in scored_candidates
        if candidate["executable_liquidity_notional"]
        >= float(config.sell_before_close_min_exit_notional)
        and candidate["queue_fill_probability_estimate"]
        >= float(config.sell_before_close_min_queue_fill_probability)
    ]
    best_candidate = max(
        scored_candidates,
        key=lambda item: (
            float(item["bid_price"]),
            float(item["queue_fill_probability_estimate"]),
            int(item["ts"]),
        ),
        default=None,
    )
    best = (
        max(
            executable_candidates,
            key=lambda item: (
                float(item["realized_executable_sell_before_close_return"]),
                float(item["queue_fill_probability_estimate"]),
                int(item["ts"]),
            ),
        )
        if executable_candidates
        else None
    )
    if best is not None:
        exit_path_quality = "realizable_sell_before_close"
        reason_codes = ("executable_intraround_exit_found",)
        best_exit_ts = int(best["ts"])
        best_exit_price = float(best["bid_price"])
        best_exit_ask = float(best["ask_price"])
        best_exit_size = float(best["bid_size"])
        queue_fill_probability = float(best["queue_fill_probability_estimate"])
        executable_liquidity_notional = float(best["executable_liquidity_notional"])
        realized_executable_return = float(
            best["realized_executable_sell_before_close_return"]
        )
        label_uses_executable_exit_path = True
    else:
        exit_path_quality = (
            "theoretical_sell_before_close"
            if theoretical_terminal_bid_return > 0.0
            else "non_executable_sell_before_close"
        )
        reason_codes = _non_executable_exit_path_reason_codes(
            theoretical_terminal_bid_return=theoretical_terminal_bid_return,
            candidate_count=len(scored_candidates),
            best_candidate=best_candidate,
            min_exit_notional=float(config.sell_before_close_min_exit_notional),
            min_queue_fill_probability=float(
                config.sell_before_close_min_queue_fill_probability
            ),
        )
        best_exit_ts = int(best_candidate["ts"]) if best_candidate else 0
        best_exit_price = 0.0
        best_exit_ask = 0.0
        best_exit_size = 0.0
        queue_fill_probability = (
            float(best_candidate["queue_fill_probability_estimate"])
            if best_candidate
            else 0.0
        )
        executable_liquidity_notional = (
            float(best_candidate["executable_liquidity_notional"])
            if best_candidate
            else 0.0
        )
        realized_executable_return = -1.0
        label_uses_executable_exit_path = False
    return {
        "label_source": "intraround_executable_exit_path",
        "entry_ts": entry.ts,
        "entry_side": outcome,
        "entry_price": entry.ask_price,
        "entry_size_notional": float(config.sell_before_close_entry_notional),
        "exit_window_start_ts": exit_window_start_ts,
        "exit_window_end_ts": exit_window_end_ts,
        "candidate_exit_snapshot_count": len(candidates),
        "candidate_exit_snapshots": tuple(scored_candidates),
        "terminal_bid": (
            float(terminal_candidate["bid_price"]) if terminal_candidate else 0.0
        ),
        "terminal_ask": (
            float(terminal_candidate["ask_price"]) if terminal_candidate else 0.0
        ),
        "terminal_ts": int(terminal_candidate["ts"]) if terminal_candidate else 0,
        "best_candidate_bid": (
            float(best_candidate["bid_price"]) if best_candidate else 0.0
        ),
        "best_candidate_ask": (
            float(best_candidate["ask_price"]) if best_candidate else 0.0
        ),
        "best_candidate_ts": int(best_candidate["ts"]) if best_candidate else 0,
        "best_candidate_queue_fill_probability_estimate": (
            float(best_candidate["queue_fill_probability_estimate"])
            if best_candidate
            else 0.0
        ),
        "best_candidate_executable_liquidity_notional": (
            float(best_candidate["executable_liquidity_notional"])
            if best_candidate
            else 0.0
        ),
        "best_executable_exit_ts": best_exit_ts,
        "best_executable_exit_price": best_exit_price,
        "best_executable_exit_ask": best_exit_ask,
        "best_executable_exit_size": best_exit_size,
        "queue_fill_probability_estimate": queue_fill_probability,
        "executable_liquidity_notional": executable_liquidity_notional,
        "exit_path_quality": exit_path_quality,
        "exit_path_reason_codes": reason_codes,
        "label_uses_executable_exit_path": label_uses_executable_exit_path,
        "realized_executable_sell_before_close_return": realized_executable_return,
        "theoretical_terminal_bid_return": theoretical_terminal_bid_return,
        "execution_gap_return": (
            theoretical_terminal_bid_return - realized_executable_return
        ),
    }


def _score_exit_candidate(
    *,
    snapshot: PolymarketCorpusBookSnapshot,
    entry: PolymarketCorpusBookSnapshot,
    min_exit_notional: float,
) -> dict[str, Any]:
    executable_liquidity_notional = snapshot.bid_price * snapshot.bid_size
    size_score = min(1.0, executable_liquidity_notional / min_exit_notional)
    depth_score = min(1.0, snapshot.liquidity_depth / (min_exit_notional * 2.0))
    spread = snapshot.ask_price - snapshot.bid_price
    spread_score = max(0.0, 1.0 - spread / max(snapshot.mid_price, 0.01))
    queue_fill_probability = max(
        0.0,
        min(1.0, 0.60 * size_score + 0.30 * depth_score + 0.10 * spread_score),
    )
    return {
        "ts": snapshot.ts,
        "available_at_ts": snapshot.available_at_ts,
        "bid_price": snapshot.bid_price,
        "ask_price": snapshot.ask_price,
        "bid_size": snapshot.bid_size,
        "liquidity_depth": snapshot.liquidity_depth,
        "executable_liquidity_notional": executable_liquidity_notional,
        "queue_fill_probability_estimate": queue_fill_probability,
        "realized_executable_sell_before_close_return": (
            snapshot.bid_price / entry.ask_price - 1.0
        ),
    }


def _label_gate_reason_codes(
    *,
    sell_rows: list[PolymarketCorpusLabelRow],
    leakage_failures: list[dict[str, Any]],
) -> list[str]:
    reason_codes = set()
    if leakage_failures:
        reason_codes.add("fixed_terminal_bid_only_label_detected")
    if any(not row.sell_before_close_exit_path for row in sell_rows):
        reason_codes.add("missing_sell_before_close_exit_path")
    if any(
        row.sell_before_close_execution_class == "realizable_sell_before_close"
        and row.queue_fill_probability_estimate <= 0.0
        for row in sell_rows
    ):
        reason_codes.add("missing_queue_fill_probability")
    if any(
        row.theoretical_terminal_bid_return > 0.0
        and not row.label_uses_executable_exit_path
        for row in sell_rows
    ):
        reason_codes.add("positive_theoretical_return_without_executable_exit")
    return sorted(reason_codes)


def _non_executable_exit_path_reason_codes(
    *,
    theoretical_terminal_bid_return: float,
    candidate_count: int,
    best_candidate: dict[str, Any] | None,
    min_exit_notional: float,
    min_queue_fill_probability: float,
) -> tuple[str, ...]:
    reason_codes: list[str] = [
        (
            "terminal_bid_positive_but_not_executable"
            if theoretical_terminal_bid_return > 0.0
            else "no_executable_intraround_exit"
        )
    ]
    if candidate_count == 0:
        reason_codes.append("no_candidate_exit_snapshot")
        return tuple(reason_codes)
    if candidate_count < 2:
        reason_codes.append("sparse_exit_snapshot_sampling")
    if best_candidate is None:
        reason_codes.append("missing_best_exit_candidate")
        return tuple(reason_codes)
    if float(best_candidate["executable_liquidity_notional"]) < min_exit_notional:
        reason_codes.append("min_exit_notional_not_met")
    if (
        float(best_candidate["queue_fill_probability_estimate"])
        < min_queue_fill_probability
    ):
        reason_codes.append("queue_fill_probability_below_threshold")
    return tuple(reason_codes)


def _exit_path_reason_codes(exit_path: dict[str, Any]) -> tuple[str, ...]:
    raw_reason_codes = exit_path.get("exit_path_reason_codes", ())
    if isinstance(raw_reason_codes, str):
        return (raw_reason_codes,)
    return tuple(str(reason_code) for reason_code in raw_reason_codes)


def _top_execution_gap_examples(
    sell_rows: list[PolymarketCorpusLabelRow],
) -> list[dict[str, Any]]:
    ranked = sorted(
        sell_rows,
        key=lambda row: (
            -float(row.execution_gap_return),
            str(row.market_id),
            int(row.decision_ts),
            str(row.action),
        ),
    )
    return [
        {
            "market_id": row.market_id,
            "condition_id": row.condition_id,
            "slug": row.slug,
            "decision_ts": row.decision_ts,
            "action": row.action,
            "outcome": row.outcome,
            "entry_ask": row.entry_ask,
            "terminal_bid": _exit_path_float(row, "terminal_bid"),
            "best_candidate_bid": _exit_path_float(row, "best_candidate_bid"),
            "best_candidate_ts": _exit_path_int(row, "best_candidate_ts"),
            "candidate_exit_snapshot_count": _exit_path_int(
                row,
                "candidate_exit_snapshot_count",
            ),
            "sell_before_close_execution_class": row.sell_before_close_execution_class,
            "label_uses_executable_exit_path": row.label_uses_executable_exit_path,
            "theoretical_terminal_bid_return": row.theoretical_terminal_bid_return,
            "realized_executable_sell_before_close_return": (
                row.realized_executable_sell_before_close_return
            ),
            "execution_gap_return": row.execution_gap_return,
            "queue_fill_probability_estimate": row.queue_fill_probability_estimate,
            "executable_liquidity_notional": row.executable_liquidity_notional,
            "exit_path_reason_codes": _exit_path_reason_codes(
                row.sell_before_close_exit_path or {}
            ),
        }
        for row in ranked[:20]
    ]


def _theoretical_sell_before_close_rows(
    rows: list[PolymarketCorpusLabelRow],
    *,
    config: PolymarketCorpusBuildConfig,
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            str(row.market_id),
            int(row.decision_ts),
            str(row.action),
        ),
    )
    return [
        {
            "market_id": row.market_id,
            "condition_id": row.condition_id,
            "slug": row.slug,
            "decision_ts": row.decision_ts,
            "action": row.action,
            "outcome": row.outcome,
            "entry_ask": row.entry_ask,
            "terminal_bid": _exit_path_float(row, "terminal_bid"),
            "theoretical_terminal_bid_return": row.theoretical_terminal_bid_return,
            "best_candidate_bid": _exit_path_float(row, "best_candidate_bid"),
            "best_candidate_ts": _exit_path_int(row, "best_candidate_ts"),
            "candidate_exit_snapshot_count": _exit_path_int(
                row,
                "candidate_exit_snapshot_count",
            ),
            "queue_fill_probability_estimate": row.queue_fill_probability_estimate,
            "executable_liquidity_notional": row.executable_liquidity_notional,
            "min_exit_notional": float(config.sell_before_close_min_exit_notional),
            "min_queue_fill_probability": float(
                config.sell_before_close_min_queue_fill_probability
            ),
            "min_exit_notional_met": (
                row.executable_liquidity_notional
                >= float(config.sell_before_close_min_exit_notional)
            ),
            "min_queue_fill_probability_met": (
                row.queue_fill_probability_estimate
                >= float(config.sell_before_close_min_queue_fill_probability)
            ),
            "exit_path_reason_code": _primary_exit_path_reason_code(row),
            "exit_path_reason_codes": _exit_path_reason_codes(
                row.sell_before_close_exit_path or {}
            ),
        }
        for row in ranked
    ]


def _primary_exit_path_reason_code(row: PolymarketCorpusLabelRow) -> str:
    reason_codes = _exit_path_reason_codes(row.sell_before_close_exit_path or {})
    return reason_codes[0] if reason_codes else "unknown"


def _exit_path_float(row: PolymarketCorpusLabelRow, field_name: str) -> float:
    return float((row.sell_before_close_exit_path or {}).get(field_name, 0.0))


def _exit_path_int(row: PolymarketCorpusLabelRow, field_name: str) -> int:
    return int((row.sell_before_close_exit_path or {}).get(field_name, 0))


def _mean(values: list[float]) -> float:
    return 0.0 if not values else sum(float(value) for value in values) / len(values)


def _snapshots_by_market(
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
) -> dict[str, tuple[PolymarketCorpusBookSnapshot, ...]]:
    grouped: dict[str, list[PolymarketCorpusBookSnapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault(snapshot.market_id, []).append(snapshot)
    return {
        key: tuple(sorted(value, key=lambda item: (item.ts, item.outcome)))
        for key, value in grouped.items()
    }


def _last_snapshot(
    *,
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    outcome: str,
    decision_ts: int,
) -> PolymarketCorpusBookSnapshot | None:
    eligible = [
        snapshot
        for snapshot in snapshots
        if snapshot.outcome == outcome
        and snapshot.ts <= decision_ts
        and snapshot.available_at_ts <= decision_ts
    ]
    return eligible[-1] if eligible else None
