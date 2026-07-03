#!/usr/bin/env python3
"""Replay xgboost-v7 raw-side entry gate impact across paper-shadow logs.

The report has two complementary views:

1. Actual fill ledger: remove already-filled positions whose entry raw p_side is
   below the proposed threshold. This uses event-log PnL and does not simulate
   replacement trades.
2. Signal path replay: sequentially opens hypothetical one-notional positions
   from logged signals that approximately pass the run's entry gate. This allows
   later signals to replace skipped ones, but the PnL is path-derived proxy PnL.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any


POSITION_SKIP_REASONS = {
    "max_combined_concurrent_positions",
    "max_concurrent_positions",
}


@dataclass(frozen=True)
class SignalCandidate:
    run_id: str
    path: str
    line_number: int
    event: str
    event_id: str
    round_slug: str
    side: str
    created_at_ms: int
    round_end_ts_ms: int
    signal_price: float
    entry_price: float
    model_value: float
    edge: float
    raw_p_side: float | None
    seconds_to_expiry: float | None
    signal_age_seconds: float | None
    skip_reason: str | None
    entry_filled: bool
    logged_gate_passed: bool
    approx_gate_passed: bool
    approx_gate_reasons: tuple[str, ...]


@dataclass(frozen=True)
class LabeledCandidate:
    signal: SignalCandidate
    future_count: int
    future_max_price: float
    future_max_ts_ms: int
    future_last_price: float
    future_last_ts_ms: int


@dataclass(frozen=True)
class ReplayTrade:
    signal: SignalCandidate
    exit_ts_ms: int
    exit_price: float
    pnl_proxy_usdc: float
    exit_reason: str


@dataclass(frozen=True)
class ActualEntry:
    run_id: str
    path: str
    position_id: str
    event_id: str
    round_slug: str
    side: str
    opened_at_ms: int
    entry_price: float
    model_value: float | None
    raw_p_side: float | None
    realized_pnl_usdc: float


def main() -> int:
    args = _parse_args()
    paths = _discover_paths(args.jsonl)
    reports = [
        _run_report(
            path,
            baseline_raw_threshold=args.baseline_raw_threshold,
            proposed_raw_threshold=args.proposed_raw_threshold,
            take_profit_delta=args.take_profit_delta,
        )
        for path in paths
    ]
    reports = [report for report in reports if report is not None]
    report = {
        "config": {
            "jsonl": args.jsonl,
            "baseline_raw_threshold": args.baseline_raw_threshold,
            "proposed_raw_threshold": args.proposed_raw_threshold,
            "take_profit_delta": args.take_profit_delta,
        },
        "totals": _aggregate(reports),
        "current_price_policy_totals": _aggregate(
            [
                report
                for report in reports
                if _float((report.get("run_config") or {}).get("min_entry_price")) is not None
                and float((report.get("run_config") or {}).get("min_entry_price")) >= 0.40 - 1e-12
            ]
        ),
        "runs": reports,
        "blocked_actual_entries": _blocked_actual_entries(reports, limit=args.blocked_limit),
    }
    if args.output_json_path:
        output = Path(args.output_json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_path:
        output = Path(args.report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "runs": len(reports),
                "totals": report["totals"],
                "current_price_policy_totals": report["current_price_policy_totals"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        action="append",
        default=[],
        help="phase4 JSONL path or glob. Defaults to all xgboost-v7 paper-shadow logs.",
    )
    parser.add_argument("--baseline-raw-threshold", type=float, default=0.50)
    parser.add_argument("--proposed-raw-threshold", type=float, default=0.60)
    parser.add_argument("--take-profit-delta", type=float, default=0.10)
    parser.add_argument("--blocked-limit", type=int, default=30)
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    return parser.parse_args()


def _discover_paths(patterns: list[str]) -> list[Path]:
    if not patterns:
        patterns = ["data/logs/xgboost-v7-paper-shadow*/phase4-*.jsonl"]
    paths: list[Path] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(Path(item) for item in expanded)
        else:
            paths.append(Path(pattern))
    result = []
    for path in sorted(set(paths)):
        name = path.name
        if not path.exists():
            continue
        if "summary" in name or "backfill" in name:
            continue
        result.append(path)
    return result


def _run_report(
    path: Path,
    *,
    baseline_raw_threshold: float,
    proposed_raw_threshold: float,
    take_profit_delta: float,
) -> dict[str, Any] | None:
    payloads: list[dict[str, Any]] = []
    run_config: dict[str, Any] = {}
    inline_summary: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["_line_number"] = line_number
            payloads.append(payload)
            if payload.get("event") == "phase4_started":
                run_config = payload.get("config") or {}
            if payload.get("event") == "phase4_summary":
                inline_summary = payload
    if not run_config:
        run_config = _summary_file(path).get("run_config") or {}
    summary = inline_summary or _summary_file(path)
    run_id = _run_id(path)
    signals = _dedup_candidates(
        [
            candidate
            for payload in payloads
            if (candidate := _candidate_from_payload(payload, path=path, run_id=run_id, config=run_config))
            is not None
        ]
    )
    labeled = _label_candidates(signals)
    actual_entries = _actual_entries(payloads, path=path, run_id=run_id)
    if not signals and not actual_entries:
        return None
    actual = _actual_ledger_summary(actual_entries, proposed_raw_threshold=proposed_raw_threshold)
    logged_baseline = _replay(
        labeled,
        gate_source="logged",
        raw_threshold=baseline_raw_threshold,
        take_profit_delta=take_profit_delta,
    )
    logged_proposed = _replay(
        labeled,
        gate_source="logged",
        raw_threshold=proposed_raw_threshold,
        take_profit_delta=take_profit_delta,
    )
    approx_baseline = _replay(
        labeled,
        gate_source="approx",
        raw_threshold=baseline_raw_threshold,
        take_profit_delta=take_profit_delta,
    )
    approx_proposed = _replay(
        labeled,
        gate_source="approx",
        raw_threshold=proposed_raw_threshold,
        take_profit_delta=take_profit_delta,
    )
    return {
        "run_id": run_id,
        "path": str(path),
        "summary_path": str(path.with_name(path.stem + "-summary.json"))
        if path.with_name(path.stem + "-summary.json").exists()
        else "",
        "status": summary.get("status"),
        "lifecycle_complete": summary.get("lifecycle_complete"),
        "run_config": {
            "min_entry_price": run_config.get("min_entry_price"),
            "settlement_edge_threshold": run_config.get("settlement_edge_threshold"),
            "near_min_fresh_edge_threshold": run_config.get("near_min_fresh_edge_threshold"),
            "no_new_entry_before_expiry_seconds": run_config.get("no_new_entry_before_expiry_seconds"),
            "max_signal_age_seconds": run_config.get("max_signal_age_seconds"),
            "signal_source": run_config.get("signal_source"),
        },
        "summary_realized_pnl_usdc": _float(summary.get("realized_pnl_usdc")),
        "signal_count": len(signals),
        "labeled_signal_count": len(labeled),
        "actual": actual,
        "logged_gate_replay": _replay_pair_summary(logged_baseline, logged_proposed),
        "approx_gate_replay": _replay_pair_summary(approx_baseline, approx_proposed),
        "skip_reasons": dict(Counter(signal.skip_reason for signal in signals if signal.skip_reason)),
    }


def _candidate_from_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    run_id: str,
    config: dict[str, Any],
) -> SignalCandidate | None:
    signal = payload.get("signal")
    if not isinstance(signal, dict):
        return None
    side = str(signal.get("selected_side") or signal.get("outcome_side") or "").upper()
    if side not in {"UP", "DOWN"}:
        return None
    signal_price = _float(signal.get("polymarket_price"))
    model_value = _float(signal.get("model_probability") or signal.get("token_expected_win_probability"))
    if signal_price is None or model_value is None:
        return None
    entry_price = _float(payload.get("worst_price"))
    if entry_price is None:
        entry_price = _float(payload.get("ask"))
    if entry_price is None:
        entry_price = signal_price
    created_at_ms = _millis(signal.get("created_at") or signal.get("ts"))
    if created_at_ms <= 0:
        return None
    round_end_ts_ms = _millis(signal.get("round_end_ts"))
    if round_end_ts_ms <= 0:
        round_end_ts_ms = _round_end_ms(str(signal.get("round_slug") or ""))
    raw_p_side, _ = _raw_probabilities(signal, side)
    seconds_to_expiry = _float(payload.get("seconds_to_expiry"))
    if seconds_to_expiry is None and round_end_ts_ms > 0:
        seconds_to_expiry = max(0.0, (round_end_ts_ms - created_at_ms) / 1000.0)
    signal_age_seconds = _float(payload.get("signal_age_seconds"))
    gate = payload.get("gate_evaluation") or {}
    if signal_age_seconds is None:
        signal_age_seconds = _float(gate.get("signal_age_seconds"))
    event = str(payload.get("event") or "")
    skip_reason = str(payload.get("reason") or "") if event == "entry_skipped" else None
    logged_gate_passed = event == "paper_entry_filled" or (
        event == "entry_gate_evaluated" and _entry_gate_passed(payload)
    )
    approx_reasons = _approx_gate_reasons(
        payload,
        signal,
        entry_price=entry_price,
        model_value=model_value,
        raw_p_side=raw_p_side,
        seconds_to_expiry=seconds_to_expiry,
        signal_age_seconds=signal_age_seconds,
        config=config,
    )
    return SignalCandidate(
        run_id=run_id,
        path=str(path),
        line_number=int(payload.get("_line_number") or 0),
        event=event,
        event_id=str(signal.get("event_id") or ""),
        round_slug=str(signal.get("round_slug") or ""),
        side=side,
        created_at_ms=created_at_ms,
        round_end_ts_ms=round_end_ts_ms,
        signal_price=signal_price,
        entry_price=entry_price,
        model_value=model_value,
        edge=model_value - entry_price,
        raw_p_side=raw_p_side,
        seconds_to_expiry=seconds_to_expiry,
        signal_age_seconds=signal_age_seconds,
        skip_reason=skip_reason,
        entry_filled=event == "paper_entry_filled",
        logged_gate_passed=logged_gate_passed,
        approx_gate_passed=not approx_reasons,
        approx_gate_reasons=tuple(approx_reasons),
    )


def _entry_gate_passed(payload: dict[str, Any]) -> bool:
    gate = payload.get("gate_evaluation") or {}
    if gate.get("settlement_gate_passed") is False:
        return False
    raw_gate = gate.get("v7_raw_side_agreement") or {}
    if raw_gate.get("skip_reason"):
        return False
    return True


def _approx_gate_reasons(
    payload: dict[str, Any],
    signal: dict[str, Any],
    *,
    entry_price: float,
    model_value: float,
    raw_p_side: float | None,
    seconds_to_expiry: float | None,
    signal_age_seconds: float | None,
    config: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    min_entry_price = _float(config.get("min_entry_price"))
    if min_entry_price is not None and entry_price < min_entry_price - 1e-12:
        reasons.append("entry_price_below_min")
    max_signal_age = _float(config.get("max_signal_age_seconds"))
    if max_signal_age is not None and signal_age_seconds is not None and signal_age_seconds > max_signal_age:
        reasons.append("signal_age_above_threshold")
    min_seconds = _float(config.get("min_seconds_to_expiry"))
    if min_seconds is not None and seconds_to_expiry is not None and seconds_to_expiry < min_seconds:
        reasons.append("seconds_to_expiry_below_min")
    max_seconds = _float(config.get("max_seconds_to_expiry"))
    if max_seconds is not None and seconds_to_expiry is not None and seconds_to_expiry > max_seconds:
        reasons.append("seconds_to_expiry_above_max")
    no_new_entry = _float(config.get("no_new_entry_before_expiry_seconds"))
    if no_new_entry is not None and seconds_to_expiry is not None and seconds_to_expiry <= no_new_entry:
        reasons.append("no_new_entry_window")
    edge_threshold = _float(config.get("settlement_edge_threshold"))
    fresh_edge = _float(payload.get("fresh_edge_at_worst"))
    if fresh_edge is None:
        fresh_edge = model_value - entry_price
    near_min_threshold = _float(config.get("near_min_fresh_edge_threshold"))
    near_min_band = _float(config.get("near_min_price_band")) or 0.0
    near_min_seconds = _float(config.get("near_min_seconds_to_expiry"))
    if (
        near_min_threshold is not None
        and min_entry_price is not None
        and seconds_to_expiry is not None
        and seconds_to_expiry >= (near_min_seconds or 0.0)
        and entry_price <= min_entry_price + near_min_band + 1e-12
    ):
        edge_threshold = max(edge_threshold or 0.0, near_min_threshold)
    if edge_threshold is not None and fresh_edge < edge_threshold - 1e-12:
        reasons.append("fresh_edge_below_threshold")
    current_raw_min = _current_raw_min_probability(payload)
    if current_raw_min is not None and (raw_p_side is None or raw_p_side < current_raw_min - 1e-12):
        reasons.append("raw_p_side_below_current_min")
    # Preserve explicit hard skips from the log for non-position causes; the raw
    # threshold itself is varied by the replay layer below.
    if payload.get("event") == "entry_skipped":
        reason = str(payload.get("reason") or "")
        if reason and reason not in POSITION_SKIP_REASONS and reason != "v7_raw_side_probability_below_threshold":
            reasons.append(reason)
    return sorted(set(reasons))


def _current_raw_min_probability(payload: dict[str, Any]) -> float | None:
    gate = payload.get("gate_evaluation") or {}
    raw_gate = gate.get("v7_raw_side_agreement") or {}
    value = _float(raw_gate.get("min_probability"))
    return value


def _dedup_candidates(candidates: list[SignalCandidate]) -> list[SignalCandidate]:
    by_key: dict[tuple[str, str], SignalCandidate] = {}
    for candidate in candidates:
        key = (candidate.run_id, candidate.event_id)
        current = by_key.get(key)
        if current is None:
            by_key[key] = candidate
            continue
        by_key[key] = _merge_candidate(current, candidate)
    return sorted(by_key.values(), key=lambda item: (item.created_at_ms, item.event_id))


def _merge_candidate(left: SignalCandidate, right: SignalCandidate) -> SignalCandidate:
    base = left if left.created_at_ms <= right.created_at_ms else right
    other = right if base is left else left
    skip_reason = base.skip_reason or other.skip_reason
    approx_reasons = tuple(sorted(set(base.approx_gate_reasons) & set(other.approx_gate_reasons)))
    if not approx_reasons and (base.approx_gate_passed or other.approx_gate_passed):
        approx_passed = True
    else:
        approx_reasons = tuple(sorted(set(base.approx_gate_reasons) | set(other.approx_gate_reasons)))
        approx_passed = not approx_reasons
    return SignalCandidate(
        **{
            **asdict(base),
            "skip_reason": skip_reason,
            "entry_filled": base.entry_filled or other.entry_filled,
            "logged_gate_passed": base.logged_gate_passed or other.logged_gate_passed,
            "approx_gate_passed": approx_passed,
            "approx_gate_reasons": approx_reasons,
        }
    )


def _label_candidates(candidates: list[SignalCandidate]) -> list[LabeledCandidate]:
    by_path: dict[tuple[str, str, str], list[SignalCandidate]] = defaultdict(list)
    for signal in candidates:
        by_path[(signal.run_id, signal.round_slug, signal.side)].append(signal)
    for rows in by_path.values():
        rows.sort(key=lambda item: (item.created_at_ms, item.event_id))
    labeled: list[LabeledCandidate] = []
    for signal in candidates:
        future = [
            item
            for item in by_path[(signal.run_id, signal.round_slug, signal.side)]
            if item.created_at_ms > signal.created_at_ms
            and (signal.round_end_ts_ms <= 0 or item.created_at_ms <= signal.round_end_ts_ms)
        ]
        if not future:
            continue
        max_row = max(future, key=lambda item: item.signal_price)
        last_row = future[-1]
        labeled.append(
            LabeledCandidate(
                signal=signal,
                future_count=len(future),
                future_max_price=max_row.signal_price,
                future_max_ts_ms=max_row.created_at_ms,
                future_last_price=last_row.signal_price,
                future_last_ts_ms=last_row.created_at_ms,
            )
        )
    return labeled


def _actual_entries(payloads: list[dict[str, Any]], *, path: Path, run_id: str) -> list[ActualEntry]:
    entries: dict[str, dict[str, Any]] = {}
    pnl_by_position: dict[str, float] = defaultdict(float)
    for payload in payloads:
        event = payload.get("event")
        position = payload.get("position") or {}
        position_id = str(position.get("event_id") or "")
        if event == "paper_entry_filled" and position_id:
            signal = payload.get("signal") or {}
            side = str(position.get("side") or signal.get("selected_side") or signal.get("outcome_side") or "").upper()
            raw_p_side, _ = _raw_probabilities(signal, side)
            entries[position_id] = {
                "event_id": str(signal.get("event_id") or position.get("entry_signal_event_id") or ""),
                "round_slug": str(position.get("round_slug") or signal.get("round_slug") or ""),
                "side": side,
                "opened_at_ms": _millis(position.get("opened_at") or signal.get("created_at")),
                "entry_price": _float(position.get("entry_price") or position.get("fill_price")) or 0.0,
                "model_value": _float(signal.get("model_probability") or signal.get("token_expected_win_probability")),
                "raw_p_side": raw_p_side,
            }
            pnl_by_position.setdefault(position_id, 0.0)
        elif event == "paper_v7_settlement_position_reduced" and position_id:
            pnl_by_position[position_id] += _float(payload.get("realized_pnl_delta")) or 0.0
        elif event in {"paper_exit_filled", "paper_settlement_resolved"} and position_id:
            pnl_by_position[position_id] += _float(payload.get("realized_pnl")) or 0.0
    result: list[ActualEntry] = []
    for position_id, entry in entries.items():
        result.append(
            ActualEntry(
                run_id=run_id,
                path=str(path),
                position_id=position_id,
                event_id=entry["event_id"],
                round_slug=entry["round_slug"],
                side=entry["side"],
                opened_at_ms=entry["opened_at_ms"],
                entry_price=entry["entry_price"],
                model_value=entry["model_value"],
                raw_p_side=entry["raw_p_side"],
                realized_pnl_usdc=pnl_by_position.get(position_id, 0.0),
            )
        )
    return sorted(result, key=lambda item: (item.opened_at_ms, item.position_id))


def _actual_ledger_summary(entries: list[ActualEntry], *, proposed_raw_threshold: float) -> dict[str, Any]:
    kept = [entry for entry in entries if entry.raw_p_side is not None and entry.raw_p_side >= proposed_raw_threshold]
    skipped = [entry for entry in entries if entry not in kept]
    total_pnl = sum(entry.realized_pnl_usdc for entry in entries)
    kept_pnl = sum(entry.realized_pnl_usdc for entry in kept)
    skipped_pnl = sum(entry.realized_pnl_usdc for entry in skipped)
    return {
        "entry_count": len(entries),
        "event_log_pnl_usdc": total_pnl,
        "kept_entry_count": len(kept),
        "kept_event_log_pnl_usdc": kept_pnl,
        "skipped_entry_count": len(skipped),
        "skipped_event_log_pnl_usdc": skipped_pnl,
        "delta_vs_actual_event_log_pnl_usdc": kept_pnl - total_pnl,
        "skipped_entries": [asdict(entry) for entry in skipped],
    }


def _replay(
    rows: list[LabeledCandidate],
    *,
    gate_source: str,
    raw_threshold: float,
    take_profit_delta: float,
) -> dict[str, Any]:
    selected: list[ReplayTrade] = []
    skipped = Counter()
    open_until_ms = -1
    for row in sorted(rows, key=lambda item: (item.signal.created_at_ms, item.signal.event_id)):
        signal = row.signal
        gate_passed = signal.logged_gate_passed if gate_source == "logged" else signal.approx_gate_passed
        if not gate_passed:
            if gate_source == "approx":
                for reason in signal.approx_gate_reasons or ("approx_gate_not_passed",):
                    skipped[reason] += 1
            else:
                skipped["logged_gate_not_passed"] += 1
            continue
        if signal.raw_p_side is None or signal.raw_p_side < raw_threshold - 1e-12:
            skipped["raw_p_side_below_threshold"] += 1
            continue
        if signal.created_at_ms < open_until_ms:
            skipped["sim_max_combined_concurrent_positions"] += 1
            continue
        if row.future_max_price >= signal.entry_price + take_profit_delta - 1e-12:
            exit_price = signal.entry_price + take_profit_delta
            exit_ts_ms = row.future_max_ts_ms
            exit_reason = f"tp_{take_profit_delta:.2f}"
        else:
            exit_price = row.future_last_price
            exit_ts_ms = max(row.future_last_ts_ms, signal.round_end_ts_ms)
            exit_reason = "last_observed"
        pnl_proxy = (exit_price - signal.entry_price) / signal.entry_price if signal.entry_price > 0 else 0.0
        selected.append(
            ReplayTrade(
                signal=signal,
                exit_ts_ms=exit_ts_ms,
                exit_price=exit_price,
                pnl_proxy_usdc=pnl_proxy,
                exit_reason=exit_reason,
            )
        )
        open_until_ms = exit_ts_ms
    return {
        "gate_source": gate_source,
        "raw_threshold": raw_threshold,
        "trade_count": len(selected),
        "pnl_proxy_usdc": sum(item.pnl_proxy_usdc for item in selected),
        "tp_exit_count": sum(1 for item in selected if item.exit_reason.startswith("tp_")),
        "last_exit_count": sum(1 for item in selected if item.exit_reason == "last_observed"),
        "entry_filled_overlap": sum(1 for item in selected if item.signal.entry_filled),
        "hit_5c_rate": _rate(item.exit_price >= item.signal.entry_price + 0.05 - 1e-12 for item in selected),
        "hit_10c_rate": _rate(item.exit_price >= item.signal.entry_price + 0.10 - 1e-12 for item in selected),
        "skipped": dict(sorted(skipped.items())),
        "trades": [_trade_dict(item) for item in selected],
    }


def _replay_pair_summary(baseline: dict[str, Any], proposed: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": _compact_replay(baseline),
        "proposed": _compact_replay(proposed),
        "delta": {
            "trade_count": proposed["trade_count"] - baseline["trade_count"],
            "pnl_proxy_usdc": proposed["pnl_proxy_usdc"] - baseline["pnl_proxy_usdc"],
            "tp_exit_count": proposed["tp_exit_count"] - baseline["tp_exit_count"],
        },
        "proposed_trades": proposed["trades"],
    }


def _compact_replay(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_threshold": item["raw_threshold"],
        "trade_count": item["trade_count"],
        "pnl_proxy_usdc": item["pnl_proxy_usdc"],
        "tp_exit_count": item["tp_exit_count"],
        "last_exit_count": item["last_exit_count"],
        "entry_filled_overlap": item["entry_filled_overlap"],
        "hit_5c_rate": item["hit_5c_rate"],
        "hit_10c_rate": item["hit_10c_rate"],
        "skipped": item["skipped"],
    }


def _trade_dict(trade: ReplayTrade) -> dict[str, Any]:
    return {
        "run_id": trade.signal.run_id,
        "round_slug": trade.signal.round_slug,
        "side": trade.signal.side,
        "event_id": trade.signal.event_id,
        "created_at_ms": trade.signal.created_at_ms,
        "entry_price": trade.signal.entry_price,
        "signal_price": trade.signal.signal_price,
        "model_value": trade.signal.model_value,
        "edge": trade.signal.edge,
        "raw_p_side": trade.signal.raw_p_side,
        "entry_filled": trade.signal.entry_filled,
        "exit_ts_ms": trade.exit_ts_ms,
        "exit_price": trade.exit_price,
        "exit_reason": trade.exit_reason,
        "pnl_proxy_usdc": trade.pnl_proxy_usdc,
    }


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {}
    actual = [report["actual"] for report in reports]
    logged = [report["logged_gate_replay"] for report in reports]
    approx = [report["approx_gate_replay"] for report in reports]
    skipped_entries = [
        entry
        for item in actual
        for entry in item["skipped_entries"]
    ]
    skipped_losses = [entry for entry in skipped_entries if entry["realized_pnl_usdc"] < -1e-12]
    skipped_wins = [entry for entry in skipped_entries if entry["realized_pnl_usdc"] > 1e-12]
    skipped_flat = [
        entry
        for entry in skipped_entries
        if abs(entry["realized_pnl_usdc"]) <= 1e-12
    ]
    return {
        "run_count": len(reports),
        "actual_entry_count": sum(item["entry_count"] for item in actual),
        "actual_event_log_pnl_usdc": sum(item["event_log_pnl_usdc"] for item in actual),
        "actual_kept_entry_count": sum(item["kept_entry_count"] for item in actual),
        "actual_kept_event_log_pnl_usdc": sum(item["kept_event_log_pnl_usdc"] for item in actual),
        "actual_skipped_entry_count": sum(item["skipped_entry_count"] for item in actual),
        "actual_skipped_event_log_pnl_usdc": sum(item["skipped_event_log_pnl_usdc"] for item in actual),
        "actual_skipped_loss_count": len(skipped_losses),
        "actual_skipped_loss_pnl_usdc": sum(entry["realized_pnl_usdc"] for entry in skipped_losses),
        "actual_skipped_win_count": len(skipped_wins),
        "actual_skipped_win_pnl_usdc": sum(entry["realized_pnl_usdc"] for entry in skipped_wins),
        "actual_skipped_flat_count": len(skipped_flat),
        "actual_delta_vs_event_log_pnl_usdc": sum(
            item["delta_vs_actual_event_log_pnl_usdc"] for item in actual
        ),
        "logged_baseline_trade_count": sum(item["baseline"]["trade_count"] for item in logged),
        "logged_proposed_trade_count": sum(item["proposed"]["trade_count"] for item in logged),
        "logged_baseline_pnl_proxy_usdc": sum(item["baseline"]["pnl_proxy_usdc"] for item in logged),
        "logged_proposed_pnl_proxy_usdc": sum(item["proposed"]["pnl_proxy_usdc"] for item in logged),
        "logged_delta_pnl_proxy_usdc": sum(item["delta"]["pnl_proxy_usdc"] for item in logged),
        "approx_baseline_trade_count": sum(item["baseline"]["trade_count"] for item in approx),
        "approx_proposed_trade_count": sum(item["proposed"]["trade_count"] for item in approx),
        "approx_baseline_pnl_proxy_usdc": sum(item["baseline"]["pnl_proxy_usdc"] for item in approx),
        "approx_proposed_pnl_proxy_usdc": sum(item["proposed"]["pnl_proxy_usdc"] for item in approx),
        "approx_delta_pnl_proxy_usdc": sum(item["delta"]["pnl_proxy_usdc"] for item in approx),
    }


def _blocked_actual_entries(reports: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for report in reports:
        for entry in report["actual"]["skipped_entries"]:
            entries.append(entry)
    entries.sort(key=lambda item: item["realized_pnl_usdc"])
    return entries[:limit]


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# v7 raw-side gate impact replay",
        "",
        "## Scope",
        "",
        f"- Baseline raw threshold: `{report['config']['baseline_raw_threshold']}`",
        f"- Proposed raw threshold: `{report['config']['proposed_raw_threshold']}`",
        f"- Take-profit proxy delta: `{report['config']['take_profit_delta']}`",
        "- Actual ledger view removes filled entries below the proposed threshold and does not model replacements.",
        "- Approx replay view replays logged signal paths with approximate entry gates, so its PnL is proxy PnL.",
        "",
        "## Totals",
        "",
        _totals_table("All discovered v7 runs", report["totals"]),
        "",
        _totals_table("Current price-policy runs (min_entry_price >= 0.40)", report["current_price_policy_totals"]),
        "",
        "## Bet Count And Opportunity Cost",
        "",
        "|Scope|Actual entries|Kept @ raw60|Reduction|Avg PnL / bet before|Avg PnL / kept bet|Avg PnL / blocked bet|Blocked losses|Blocked wins|Approx trades|Approx reduction|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _count_impact_row("All discovered v7 runs", report["totals"]),
        _count_impact_row(
            "Current price-policy runs",
            report["current_price_policy_totals"],
        ),
        "",
        "## Per Run",
        "",
        "|Run|Status|Min price|Actual entries|Kept @ raw60|Reduction|Actual PnL|Actual delta|Blocked loss|Blocked win|Approx trades|Approx reduction|Approx delta|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["runs"]:
        actual = item["actual"]
        approx = item["approx_gate_replay"]
        blocked_losses = [
            entry
            for entry in actual["skipped_entries"]
            if entry["realized_pnl_usdc"] < -1e-12
        ]
        blocked_wins = [
            entry
            for entry in actual["skipped_entries"]
            if entry["realized_pnl_usdc"] > 1e-12
        ]
        actual_reduction = (
            actual["skipped_entry_count"] / actual["entry_count"]
            if actual["entry_count"]
            else 0.0
        )
        approx_reduction = (
            (approx["baseline"]["trade_count"] - approx["proposed"]["trade_count"])
            / approx["baseline"]["trade_count"]
            if approx["baseline"]["trade_count"]
            else 0.0
        )
        lines.append(
            "|{run}|{status}|{min_price}|{entries}|{kept}|{actual_reduction}|{actual_pnl}|{actual_delta}|{blocked_loss}|{blocked_win}|{trades}|{approx_reduction}|{delta}|".format(
                run=item["run_id"],
                status=item.get("status") or "",
                min_price=_fmt((item.get("run_config") or {}).get("min_entry_price")),
                entries=actual["entry_count"],
                kept=actual["kept_entry_count"],
                actual_reduction=_pct(actual_reduction),
                actual_pnl=_fmt(actual["event_log_pnl_usdc"]),
                actual_delta=_fmt(actual["delta_vs_actual_event_log_pnl_usdc"]),
                blocked_loss=_fmt(sum(entry["realized_pnl_usdc"] for entry in blocked_losses)),
                blocked_win=_fmt(sum(entry["realized_pnl_usdc"] for entry in blocked_wins)),
                trades=f"{approx['baseline']['trade_count']} -> {approx['proposed']['trade_count']}",
                approx_reduction=_pct(approx_reduction),
                delta=_fmt(approx["delta"]["pnl_proxy_usdc"]),
            )
        )
    lines.extend(
        [
            "",
            "## Worst Actual Entries Blocked By raw60",
            "",
            "|Run|Round|Side|Entry|Raw p_side|Actual PnL|Event ID|",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for entry in report["blocked_actual_entries"]:
        lines.append(
            "|{run}|{round}|{side}|{entry_price}|{raw}|{pnl}|`{event}`|".format(
                run=entry["run_id"],
                round=entry["round_slug"],
                side=entry["side"],
                entry_price=_fmt(entry["entry_price"]),
                raw=_fmt(entry["raw_p_side"]),
                pnl=_fmt(entry["realized_pnl_usdc"]),
                event=entry["event_id"],
            )
        )
    return "\n".join(lines) + "\n"


def _totals_table(title: str, totals: dict[str, Any]) -> str:
    if not totals:
        return f"**{title}**: no eligible runs."
    return (
        f"**{title}**: runs `{totals['run_count']}`, actual entries `{totals['actual_entry_count']}`, "
        f"actual event-log PnL `{_fmt(totals['actual_event_log_pnl_usdc'])}` -> "
        f"`{_fmt(totals['actual_kept_event_log_pnl_usdc'])}` after raw60 "
        f"(delta `{_fmt(totals['actual_delta_vs_event_log_pnl_usdc'])}`, skipped entries "
        f"`{totals['actual_skipped_entry_count']}`, skipped loss "
        f"`{totals['actual_skipped_loss_count']}` / `{_fmt(totals['actual_skipped_loss_pnl_usdc'])}`, "
        f"skipped win `{totals['actual_skipped_win_count']}` / "
        f"`{_fmt(totals['actual_skipped_win_pnl_usdc'])}`); approx replay trades "
        f"`{totals['approx_baseline_trade_count']} -> {totals['approx_proposed_trade_count']}`, "
        f"approx PnL `{_fmt(totals['approx_baseline_pnl_proxy_usdc'])}` -> "
        f"`{_fmt(totals['approx_proposed_pnl_proxy_usdc'])}` "
        f"(delta `{_fmt(totals['approx_delta_pnl_proxy_usdc'])}`)."
    )


def _count_impact_row(title: str, totals: dict[str, Any]) -> str:
    if not totals:
        return f"|{title}|0|0|0.0%|n/a|n/a|n/a|0 / 0.0000|0 / 0.0000|0 -> 0|0.0%|"
    actual_entries = totals["actual_entry_count"]
    kept_entries = totals["actual_kept_entry_count"]
    skipped_entries = totals["actual_skipped_entry_count"]
    approx_base = totals["approx_baseline_trade_count"]
    approx_prop = totals["approx_proposed_trade_count"]
    return (
        f"|{title}|{actual_entries}|{kept_entries}|"
        f"{_pct(skipped_entries / actual_entries if actual_entries else 0.0)}|"
        f"{_fmt(totals['actual_event_log_pnl_usdc'] / actual_entries if actual_entries else None)}|"
        f"{_fmt(totals['actual_kept_event_log_pnl_usdc'] / kept_entries if kept_entries else None)}|"
        f"{_fmt(totals['actual_skipped_event_log_pnl_usdc'] / skipped_entries if skipped_entries else None)}|"
        f"{totals['actual_skipped_loss_count']} / {_fmt(totals['actual_skipped_loss_pnl_usdc'])}|"
        f"{totals['actual_skipped_win_count']} / {_fmt(totals['actual_skipped_win_pnl_usdc'])}|"
        f"{approx_base} -> {approx_prop}|"
        f"{_pct((approx_base - approx_prop) / approx_base if approx_base else 0.0)}|"
    )


def _summary_file(path: Path) -> dict[str, Any]:
    summary_path = path.with_name(path.stem + "-summary.json")
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _raw_probabilities(signal: dict[str, Any], side: str) -> tuple[float | None, float | None]:
    p_up = _float(signal.get("p_up"))
    p_down = _float(signal.get("p_down"))
    if side == "UP":
        return p_up, p_down
    return p_down, p_up


def _run_id(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("xgboost-v7-paper-shadow"):
            return part
    return path.parent.name


def _round_end_ms(round_slug: str) -> int:
    match = re.search(r"-(\d+)$", round_slug)
    return (int(match.group(1)) + 900) * 1000 if match else 0


def _millis(value: Any) -> int:
    try:
        if value is None:
            return 0
        number = float(value)
        if math.isnan(number):
            return 0
        if number < 10_000_000_000:
            number *= 1000
        return int(number)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        return None if math.isnan(result) else result
    except (TypeError, ValueError):
        return None


def _rate(values: Any) -> float | None:
    cleaned = [bool(item) for item in values if item is not None]
    return sum(1 for item in cleaned if item) / len(cleaned) if cleaned else None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.4f}"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
