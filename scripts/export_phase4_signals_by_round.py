#!/usr/bin/env python3
"""Export Phase 4 executor signals grouped by BTC-15M round slug.

Builds a per-round inventory from ``signal_batch_received`` (deduped by
``event_id``) merged with ``entry_gate_evaluated``, ``entry_skipped``,
``paper_entry_filled``, and ``paper_settlement_resolved`` events.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class SignalRecord:
    event_id: str
    round_slug: str = ""
    side: str = ""
    signal_utc: str = ""
    created_utc: str = ""
    batch_utc: str = ""
    v6_joint_side: str = ""
    p_up: float | None = None
    p_down: float | None = None
    p_neutral: float | None = None
    p_vol_up: float | None = None
    p_vol_down: float | None = None
    token_probability: float | None = None
    market_implied_prob: float | None = None
    edge: float | None = None
    eval_count: int = 0
    settlement_gate_passed: bool = False
    volatility_gate_passed: bool = False
    worst_price_min: float | None = None
    settlement_edge_max: float | None = None
    volatility_score_max: float | None = None
    seconds_to_expiry_max: float | None = None
    action: str = "received"
    skip_reasons: str = ""
    signal_age_seconds_max: float | None = None
    fill_side: str = ""
    fill_price: float | None = None
    settlement_result: str = ""
    realized_pnl: float | None = None

    def merge_signal(self, signal: dict[str, Any], *, batch_utc: str | None = None) -> None:
        if batch_utc:
            self.batch_utc = batch_utc
        self.round_slug = str(signal.get("round_slug") or self.round_slug)
        self.side = str(signal.get("outcome_side") or self.side).upper()
        self.signal_utc = _ms_to_iso(signal.get("ts")) or self.signal_utc
        self.created_utc = _ms_to_iso(signal.get("created_at")) or self.created_utc
        self.v6_joint_side = str(signal.get("v6_joint_side") or self.v6_joint_side)
        self.p_up = _optional_float(signal.get("p_up"), self.p_up)
        self.p_down = _optional_float(signal.get("p_down"), self.p_down)
        self.p_neutral = _optional_float(signal.get("p_neutral"), self.p_neutral)
        self.p_vol_up = _optional_float(signal.get("p_vol_up"), self.p_vol_up)
        self.p_vol_down = _optional_float(signal.get("p_vol_down"), self.p_vol_down)
        self.token_probability = _optional_float(signal.get("token_probability"), self.token_probability)
        self.market_implied_prob = _optional_float(
            signal.get("market_implied_prob"),
            self.market_implied_prob,
        )
        self.edge = _optional_float(signal.get("edge"), self.edge)

    def merge_gate_eval(
        self,
        *,
        gate_evaluation: dict[str, Any],
        worst_price: float | None,
        seconds_to_expiry: float | None,
        fresh_edge_at_worst: float | None,
    ) -> None:
        self.eval_count += 1
        settlement_edge = _optional_float(gate_evaluation.get("settlement_edge"))
        volatility_score = _optional_float(gate_evaluation.get("volatility_score"))
        if worst_price is not None:
            self.worst_price_min = (
                worst_price
                if self.worst_price_min is None
                else min(self.worst_price_min, worst_price)
            )
        if settlement_edge is not None:
            self.settlement_edge_max = (
                settlement_edge
                if self.settlement_edge_max is None
                else max(self.settlement_edge_max, settlement_edge)
            )
        elif fresh_edge_at_worst is not None:
            self.settlement_edge_max = (
                fresh_edge_at_worst
                if self.settlement_edge_max is None
                else max(self.settlement_edge_max, fresh_edge_at_worst)
            )
        if volatility_score is not None:
            self.volatility_score_max = (
                volatility_score
                if self.volatility_score_max is None
                else max(self.volatility_score_max, volatility_score)
            )
        if seconds_to_expiry is not None:
            self.seconds_to_expiry_max = (
                seconds_to_expiry
                if self.seconds_to_expiry_max is None
                else max(self.seconds_to_expiry_max, seconds_to_expiry)
            )
        if gate_evaluation.get("settlement_gate_passed"):
            self.settlement_gate_passed = True
        if gate_evaluation.get("volatility_gate_passed"):
            self.volatility_gate_passed = True
        if not gate_evaluation.get("settlement_gate_passed"):
            if gate_evaluation.get("settlement_confidence_passed") is False:
                self.add_skip(sleeve="settlement", reason="settlement_confidence_below_min")
            elif gate_evaluation.get("signal_freshness_passed") is False:
                self.add_skip(sleeve="settlement", reason="signal_freshness_failed")
            else:
                self.add_skip(sleeve="settlement", reason="settlement_edge_below_threshold")
        if not gate_evaluation.get("volatility_gate_passed") and gate_evaluation.get(
            "volatility_live_entry_enabled"
        ):
            self.add_skip(sleeve="volatility", reason="volatility_gate_miss")

    def add_skip(self, *, sleeve: str, reason: str) -> None:
        prefix = _skip_prefix(sleeve)
        token = f"{prefix}:{reason}"
        existing = [part for part in self.skip_reasons.split(";") if part]
        if token not in existing:
            existing.append(token)
        self.skip_reasons = ";".join(existing)
        self.action = "skipped"

    def to_csv_row(self, idx: int) -> dict[str, Any]:
        return {
            "idx": idx,
            "signal_utc": self.signal_utc,
            "created_utc": self.created_utc,
            "batch_utc": self.batch_utc,
            "round": self.round_slug,
            "side": self.side,
            "v6_joint_side": self.v6_joint_side,
            "p_up": _fmt(self.p_up),
            "p_down": _fmt(self.p_down),
            "p_neutral": _fmt(self.p_neutral),
            "p_vol_up": _fmt(self.p_vol_up),
            "p_vol_down": _fmt(self.p_vol_down),
            "token_probability": _fmt(self.token_probability),
            "market_implied_prob": _fmt(self.market_implied_prob),
            "edge": _fmt(self.edge, digits=4),
            "eval_count": self.eval_count,
            "settlement_gate_passed": self.settlement_gate_passed,
            "volatility_gate_passed": self.volatility_gate_passed,
            "worst_price_min": _fmt(self.worst_price_min),
            "settlement_edge_max": _fmt(self.settlement_edge_max),
            "volatility_score_max": _fmt(self.volatility_score_max),
            "seconds_to_expiry_max": _fmt(self.seconds_to_expiry_max, digits=1),
            "action": self.action,
            "skip_reasons": self.skip_reasons,
            "signal_age_seconds_max": _fmt(self.signal_age_seconds_max, digits=1),
            "no_fill_reason": _no_fill_reason(self),
            "fill_side": self.fill_side,
            "fill_price": _fmt(self.fill_price),
            "settlement_result": self.settlement_result,
            "realized_pnl": _fmt(self.realized_pnl),
            "event_id": self.event_id,
        }


@dataclass
class RunSummary:
    log_path: Path
    summary_path: Path | None = None
    started_at: str = ""
    finished_at: str = ""
    status: str = ""
    observed_round_count: int = 0
    processed_event_count: int = 0
    rows_filtered: dict[str, int] = field(default_factory=dict)
    observed_round_slugs: list[str] = field(default_factory=list)
    executor_skip_totals: dict[str, int] = field(default_factory=dict)


CSV_COLUMNS = [
    "idx",
    "signal_utc",
    "created_utc",
    "batch_utc",
    "round",
    "side",
    "v6_joint_side",
    "p_up",
    "p_down",
    "p_neutral",
    "p_vol_up",
    "p_vol_down",
    "token_probability",
    "market_implied_prob",
    "edge",
    "eval_count",
    "settlement_gate_passed",
    "volatility_gate_passed",
    "worst_price_min",
    "settlement_edge_max",
    "volatility_score_max",
    "seconds_to_expiry_max",
    "action",
    "skip_reasons",
    "signal_age_seconds_max",
    "no_fill_reason",
    "fill_side",
    "fill_price",
    "settlement_result",
    "realized_pnl",
    "event_id",
]


def main() -> int:
    args = _parse_args()
    log_path = Path(args.phase4_jsonl_path)
    summary_path = Path(args.summary_json_path) if args.summary_json_path else _default_summary_path(log_path)
    records, run_summary, filter_totals = load_phase4_signal_records(log_path, summary_path=summary_path)
    observed = set(run_summary.observed_round_slugs)
    if args.observed_rounds_only:
        records = [row for row in records if row.round_slug in observed]

    by_round: dict[str, list[SignalRecord]] = defaultdict(list)
    for row in records:
        by_round[row.round_slug].append(row)

    round_order = sorted(
        by_round,
        key=lambda slug: (_round_end_ts(slug) or 0, slug),
    )
    if args.observed_rounds_only:
        round_order = [slug for slug in round_order if slug in observed]
        round_order.extend(
            slug
            for slug in sorted(observed, key=lambda slug: (_round_end_ts(slug) or 0, slug))
            if slug not in round_order
        )

    output_dir = Path(args.output_dir) if args.output_dir else log_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = log_path.stem
    markdown_path = output_dir / f"{stem}-signals-by-round.md"
    csv_dir = output_dir / f"{stem}-signals-by-round"
    if args.write_csv:
        csv_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown_report(
        run_summary=run_summary,
        filter_totals=filter_totals,
        by_round=by_round,
        round_order=round_order,
        observed=observed,
    )
    markdown_path.write_text(markdown, encoding="utf-8")

    if args.write_csv:
        for slug in round_order:
            rows = sorted(
                by_round.get(slug, []),
                key=lambda row: (row.signal_utc, row.event_id),
            )
            _write_round_csv(csv_dir / f"{slug}.csv", rows)

    print(
        json.dumps(
            {
                "markdown_path": str(markdown_path),
                "round_count": len(round_order),
                "signal_count": sum(len(by_round[slug]) for slug in round_order),
                "csv_dir": str(csv_dir) if args.write_csv else None,
            },
            indent=2,
        )
    )
    return 0


def load_phase4_signal_records(
    log_path: Path,
    *,
    summary_path: Path | None,
) -> tuple[list[SignalRecord], RunSummary, dict[str, int]]:
    records_by_id: dict[str, SignalRecord] = {}
    filter_totals: dict[str, int] = defaultdict(int)
    run_summary = RunSummary(log_path=log_path, summary_path=summary_path)

    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            event = payload.get("event")
            if event == "signal_rows_filtered":
                for reason, count in (payload.get("filter_reasons") or {}).items():
                    filter_totals[str(reason)] += int(count)
                continue
            if event == "signal_batch_received":
                batch_utc = str(payload.get("ts") or "")
                for summary in payload.get("signals") or []:
                    if not isinstance(summary, dict):
                        continue
                    event_id = str(summary.get("event_id") or "")
                    if not event_id:
                        continue
                    record = records_by_id.setdefault(event_id, SignalRecord(event_id=event_id))
                    record.round_slug = str(summary.get("round_slug") or record.round_slug)
                    record.side = str(summary.get("side") or record.side).upper()
                    record.batch_utc = batch_utc
                    record.edge = _optional_float(summary.get("edge"), record.edge)
                    timestamps = summary.get("timestamps") or {}
                    record.signal_utc = str(timestamps.get("event_ts") or record.signal_utc)
                    record.created_utc = str(timestamps.get("signal_created_at") or record.created_utc)
                continue
            signal = payload.get("signal")
            if not isinstance(signal, dict):
                signal = None
            if signal is not None:
                event_id = str(signal.get("event_id") or "")
                if event_id:
                    record = records_by_id.setdefault(event_id, SignalRecord(event_id=event_id))
                    record.merge_signal(signal)
            if event == "entry_gate_evaluated" and signal is not None:
                records_by_id[event_id].merge_gate_eval(
                    gate_evaluation=dict(payload.get("gate_evaluation") or {}),
                    worst_price=_optional_float(payload.get("worst_price")),
                    seconds_to_expiry=_optional_float(payload.get("seconds_to_expiry")),
                    fresh_edge_at_worst=_optional_float(payload.get("fresh_edge_at_worst")),
                )
            elif event == "entry_skipped" and signal is not None:
                record = records_by_id[event_id]
                age = _optional_float(payload.get("signal_age_seconds"))
                if age is not None:
                    record.signal_age_seconds_max = (
                        age
                        if record.signal_age_seconds_max is None
                        else max(record.signal_age_seconds_max, age)
                    )
                record.add_skip(
                    sleeve=str(payload.get("sleeve") or ""),
                    reason=str(payload.get("reason") or "unknown"),
                )
            elif event == "paper_entry_filled":
                position = payload.get("position") or {}
                event_id = str(position.get("entry_signal_event_id") or "")
                if event_id:
                    record = records_by_id.setdefault(event_id, SignalRecord(event_id=event_id))
                    record.action = "filled"
                    record.fill_side = str(position.get("side") or record.fill_side).upper()
                    record.fill_price = _optional_float(position.get("fill_price"), record.fill_price)
                    record.round_slug = str(position.get("round_slug") or record.round_slug)
                    if signal is not None:
                        record.merge_signal(signal)
                    elif not record.signal_utc:
                        record.signal_utc = _ms_to_iso(position.get("entry_signal_ts")) or record.signal_utc
                        record.created_utc = (
                            _ms_to_iso(position.get("entry_signal_created_at")) or record.created_utc
                        )
            elif event == "paper_settlement_resolved":
                position = payload.get("position") or {}
                event_id = str(position.get("entry_signal_event_id") or "")
                if event_id and event_id in records_by_id:
                    record = records_by_id[event_id]
                    record.settlement_result = str(payload.get("settlement_result") or "")
                    record.realized_pnl = _optional_float(payload.get("realized_pnl"), record.realized_pnl)
            elif event == "phase4_summary":
                run_summary.started_at = str(payload.get("started_at") or run_summary.started_at)
                run_summary.finished_at = str(payload.get("finished_at") or run_summary.finished_at)
                run_summary.status = str(payload.get("status") or run_summary.status)
                run_summary.observed_round_count = int(payload.get("observed_round_count") or 0)
                run_summary.processed_event_count = int(payload.get("processed_event_count") or 0)

    if summary_path and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_summary.started_at = str(summary.get("started_at") or run_summary.started_at)
        run_summary.finished_at = str(summary.get("finished_at") or run_summary.finished_at)
        run_summary.status = str(summary.get("status") or run_summary.status)
        run_summary.observed_round_count = int(summary.get("observed_round_count") or 0)
        run_summary.processed_event_count = int(summary.get("processed_event_count") or 0)
        skipped = summary.get("skipped") or {}
        if isinstance(skipped, dict):
            run_summary.executor_skip_totals = {
                str(reason): int(count) for reason, count in skipped.items()
            }
        balances = summary.get("volatility_budget_balances") or {}
        if isinstance(balances, dict):
            run_summary.observed_round_slugs = sorted(
                balances,
                key=lambda slug: (_round_end_ts(slug) or 0, slug),
            )

    records = sorted(
        records_by_id.values(),
        key=lambda row: (row.signal_utc or row.batch_utc, row.event_id),
    )
    return records, run_summary, dict(filter_totals)


def _no_fill_reason(row: SignalRecord) -> str:
    if row.action == "filled":
        return ""
    if row.skip_reasons:
        return row.skip_reasons
    if row.eval_count > 0 and not row.settlement_gate_passed:
        return "gate_evaluated_never_passed_settlement"
    if row.eval_count > 0:
        return "gate_passed_no_fill_logged"
    return "batch_only_no_entry_attempt"


def _no_fill_rollup(records: list[SignalRecord]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for row in records:
        if row.action == "filled":
            continue
        reason = _no_fill_reason(row)
        for part in reason.split(";"):
            if part:
                totals[part] += 1
    return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))


def render_markdown_report(
    *,
    run_summary: RunSummary,
    filter_totals: dict[str, int],
    by_round: dict[str, list[SignalRecord]],
    round_order: list[str],
    observed: set[str],
) -> str:
    lines = [
        f"# Phase 4 signals by round - `{run_summary.log_path.name}`",
        "",
        "## Run summary",
        "",
        f"- Log: `{run_summary.log_path}`",
        f"- Summary: `{run_summary.summary_path}`" if run_summary.summary_path else "- Summary: (not found)",
        f"- Status: `{run_summary.status or 'unknown'}`",
        f"- Window: `{run_summary.started_at}` -> `{run_summary.finished_at}`",
        f"- Unique signals (deduped): `{run_summary.processed_event_count or sum(len(v) for v in by_round.values())}`",
        f"- Lifecycle observed rounds: `{run_summary.observed_round_count or len(observed)}`",
        f"- Pre-batch filtered rows: `{sum(filter_totals.values())}` (`{filter_totals}`)",
        "",
        "## No-fill reasons",
        "",
        "Per-signal `no_fill_reason` is derived from `entry_skipped`, gate evaluation failures, or "
        "`batch_only_no_entry_attempt` when the signal appeared in a batch but never reached entry logic.",
        "",
    ]
    all_records = [row for slug in round_order for row in by_round.get(slug, [])]
    rollup = _no_fill_rollup(all_records)
    if rollup:
        lines.append("| reason | signal_count |")
        lines.append("| --- | --- |")
        for reason, count in rollup.items():
            lines.append(f"| `{reason}` | {count} |")
        lines.append("")
    if run_summary.executor_skip_totals:
        lines.extend(
            [
                "### Executor skip totals (phase4_summary)",
                "",
                "| reason | count |",
                "| --- | --- |",
            ]
        )
        for reason, count in sorted(
            run_summary.executor_skip_totals.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"| `{reason}` | {count} |")
        lines.append("")
    lines.extend(
        [
            "Each section lists every deduped `event_id` whose `round_slug` matches that 15m market.",
            "",
        ]
    )

    for slug in round_order:
        rows = sorted(by_round.get(slug, []), key=lambda row: (row.signal_utc, row.event_id))
        round_end = _round_end_iso(slug)
        filled = [row for row in rows if row.action == "filled"]
        settlement_pass = [row for row in rows if row.settlement_gate_passed]
        lines.extend(
            [
                f"## Round `{slug}`",
                "",
                f"- Round end (UTC): `{round_end}`",
                f"- Lifecycle observed: `{'yes' if slug in observed else 'no'}`",
                f"- Signals: `{len(rows)}` (settlement gate-pass: `{len(settlement_pass)}`, filled: `{len(filled)}`)",
                "",
            ]
        )
        if filled:
            lines.append("### Filled")
            lines.append("")
            lines.append(
                "| event_id | side | fill_price | settlement | realized_pnl |",
            )
            lines.append("| --- | --- | --- | --- | --- |")
            for row in filled:
                lines.append(
                    f"| `{row.event_id}` | {row.fill_side or row.side} | "
                    f"{_fmt(row.fill_price)} | {row.settlement_result} | {_fmt(row.realized_pnl)} |"
                )
            lines.append("")

        lines.append("### Signals")
        lines.append("")
        lines.append(
            "| idx | signal_utc | side | p_up | p_down | edge | settlement_pass | action | "
            "no_fill_reason | age_s | event_id |",
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for idx, row in enumerate(rows, start=1):
            lines.append(
                f"| {idx} | {row.signal_utc} | {row.side} | {_fmt(row.p_up)} | {_fmt(row.p_down)} | "
                f"{_fmt(row.edge, digits=4)} | {row.settlement_gate_passed} | {row.action} | "
                f"{_no_fill_reason(row) or '-'} | {_fmt(row.signal_age_seconds_max, digits=1)} | "
                f"`{row.event_id}` |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_round_csv(path: Path, rows: list[SignalRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow(row.to_csv_row(idx))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase4_jsonl_path",
        help="Path to phase4 execution jsonl (e.g. logs/xgboost-v6-paper-shadow/phase4-*.jsonl)",
    )
    parser.add_argument(
        "--summary-json-path",
        default="",
        help="Optional phase4 summary JSON; defaults to <log-stem>-summary.json beside the log",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for markdown/csv output (default: same directory as the log)",
    )
    parser.add_argument(
        "--observed-rounds-only",
        action="store_true",
        help="Only emit sections for lifecycle-observed rounds from summary volatility balances",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write one CSV per round under <stem>-signals-by-round/",
    )
    return parser.parse_args()


def _default_summary_path(log_path: Path) -> Path | None:
    candidate = log_path.with_name(f"{log_path.stem}-summary.json")
    return candidate if candidate.exists() else None


def _round_end_ts(round_slug: str) -> int | None:
    try:
        start_ts = int(round_slug.rsplit("-", 1)[-1]) * 1000
    except ValueError:
        return None
    if "updown-15m-" in round_slug:
        return start_ts + 15 * 60_000
    return start_ts


def _round_end_iso(round_slug: str) -> str:
    end_ms = _round_end_ts(round_slug)
    if end_ms is None:
        return ""
    return datetime.fromtimestamp(end_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ms_to_iso(value: Any) -> str:
    if value in (None, "", 0):
        return ""
    if isinstance(value, str) and "T" in value:
        return value.replace("+00:00", "Z") if value.endswith("+00:00") else value
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return ""
    if ms < 1_000_000_000_000:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _optional_float(value: Any, current: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return current
    try:
        return float(value)
    except (TypeError, ValueError):
        return current


def _fmt(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _skip_prefix(sleeve: str) -> str:
    sleeve = str(sleeve or "").strip().lower()
    if sleeve == "settlement":
        return "settlement"
    if sleeve == "volatility":
        return "volatility"
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
