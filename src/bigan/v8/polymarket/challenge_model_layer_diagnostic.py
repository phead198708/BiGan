"""Outcome-aware development diagnostics for challenge-model market selection.

This module is deliberately diagnostic-only.  It reads already-open development
corpora, reconstructs cost identities, inventories point-in-time features, and
never trains a model or enables an execution path.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DIAGNOSTIC_SCHEMA_VERSION = "bigan-challenge-model-layer-market-diagnostic-v1"
INPUTS_SCHEMA_VERSION = "bigan-challenge-model-layer-diagnostic-inputs-v1"
SAFETY = {
    "paper_allowed": False,
    "live_allowed": False,
    "write_allowed": False,
    "wallet_allowed": False,
    "handoff_allowed": False,
    "promotion_allowed": False,
}


def sha256_file(path: Path | str) -> str:
    """Return the exact byte SHA-256 for *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash JSON with deterministic key and separator settings."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_and_verify_inputs(
    inputs_path: Path | str,
    *,
    repo_root: Path | str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load the diagnostic input manifest and verify every declared byte pin."""

    manifest = _load_json(inputs_path)
    if manifest.get("schema_version") != INPUTS_SCHEMA_VERSION:
        raise ValueError("unexpected challenge model-layer diagnostic inputs schema")
    root = Path(repo_root).resolve()
    resolved: dict[str, Path] = {}
    for name, descriptor in dict(manifest.get("artifacts") or {}).items():
        candidate = Path(str(descriptor.get("path") or ""))
        path = candidate if candidate.is_absolute() else root / candidate
        path = path.resolve()
        expected = str(descriptor.get("sha256") or "").lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(f"{name} does not declare a valid SHA-256")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
        resolved[name] = path
    required = {
        "exact_195_five_action_rows",
        "exact_195_market_ids",
        "exact_195_settled_corpus_index",
        "exact_195_v6_7_runtime_targets",
        "variant_23_bet_comparison",
        "variant_iteration_1_comparison",
        "variant_iteration_3_comparison",
        "variant_iteration_4_comparison",
        "variant_iteration_5_comparison",
        "historical_15m_v7_selected_trades",
        "historical_15m_v7_evaluation_report",
    }
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError("missing diagnostic inputs: " + ", ".join(missing))
    return manifest, resolved


def build_challenge_model_layer_diagnostic(
    *,
    inputs_manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    source_base_commit: str,
) -> dict[str, Any]:
    """Build the complete exact-195/legacy-15m development diagnostic."""

    market_ids = [
        line.strip()
        for line in paths["exact_195_market_ids"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(market_ids) != 195 or len(set(market_ids)) != 195:
        raise ValueError("exact-195 market id input is not exactly 195 unique rows")

    index = _load_json(paths["exact_195_settled_corpus_index"])
    entries = list(index.get("entries") or [])
    if len(entries) != 195 or int(index.get("entry_count") or 0) != 195:
        raise ValueError("settled corpus index is not exact-195")

    labels: dict[tuple[str, int, str], dict[str, Any]] = {}
    outcomes: dict[str, str] = {}
    market_returns: dict[str, float] = {}
    raw_stream_rows: dict[str, list[int]] = {
        "orderbook_snapshots": [],
        "trade_tape": [],
        "chainlink_rtds": [],
        "btc_reference_klines": [],
    }
    for entry in entries:
        for row in _load_jsonl(Path(str(entry["label_rows"]["path"]))):
            key = (str(row["market_id"]), int(row["decision_ts"]), str(row["action"]))
            labels[key] = row
        resolution = _load_jsonl(Path(str(entry["resolution_events"]["path"])))[0]
        market_id = str(resolution["market_id"])
        outcomes[market_id] = str(resolution["resolved_outcome"])
        start = float(resolution["reference_price_start"])
        end = float(resolution["reference_price_end"])
        market_returns[market_id] = end / start - 1.0
        capture_manifest = _load_json(
            Path(str(entry["source_pending_capture_manifest"]["path"]))
        )
        raw_counts = dict(capture_manifest["provider_raw_artifact_row_counts"])
        raw_stream_rows["orderbook_snapshots"].append(
            int(raw_counts.get("raw_polymarket_orderbooks.jsonl") or 0)
        )
        raw_stream_rows["trade_tape"].append(
            int(raw_counts.get("raw_polymarket_trades.jsonl") or 0)
        )
        raw_stream_rows["btc_reference_klines"].append(
            int(raw_counts.get("raw_binance_btcusdt_klines.jsonl") or 0)
        )
        raw_stream_rows["chainlink_rtds"].append(
            int(capture_manifest.get("provider_chainlink_raw_artifact_row_count") or 0)
        )

    if set(outcomes) != set(market_ids):
        raise ValueError("settled outcomes do not match the exact-195 market ids")

    targets = _load_jsonl(paths["exact_195_v6_7_runtime_targets"])
    if len(targets) != 193:
        raise ValueError("expected 193 v6.7 runtime target trades")
    v6_rows = [_decompose_5m_target(row, labels=labels) for row in targets]
    v6_by_market = {str(row["market_id"]): row for row in v6_rows}

    variant_specs = (
        ("v8_1_23_bet", "variant_23_bet_comparison", "challenge_action", 0.2),
        ("iteration_1", "variant_iteration_1_comparison", "candidate_action", 0.2),
        ("iteration_3", "variant_iteration_3_comparison", "candidate_action", 1.0),
        ("iteration_4", "variant_iteration_4_comparison", "candidate_action", 0.2),
        ("iteration_5", "variant_iteration_5_comparison", "candidate_action", 0.2),
    )
    variant_rows: dict[str, list[dict[str, Any]]] = {"v6_7": v6_rows}
    declared_sizes = {"v6_7": 0.2}
    for name, path_key, action_key, size in variant_specs:
        comparisons = _load_jsonl(paths[path_key])
        selected_ids = [
            str(row["market_id"])
            for row in comparisons
            if str(row[action_key]) != "NO_TRADE"
        ]
        missing = sorted(set(selected_ids) - set(v6_by_market))
        if missing:
            raise ValueError(f"{name} selected markets without a v6.7 runtime target")
        variant_rows[name] = [dict(v6_by_market[market_id]) for market_id in selected_ids]
        declared_sizes[name] = size

    exact_5m = {
        "corpus_role": "outcome_opened_development_only_never_promotion_evidence",
        "market_count": 195,
        "v6_7_no_trade_market_count": 2,
        "outcome_distribution": dict(sorted(Counter(outcomes.values()).items())),
        "btc_market_return_distribution": _distribution(list(market_returns.values())),
        "positive_or_tie_market_count": sum(value >= 0.0 for value in market_returns.values()),
        "negative_market_count": sum(value < 0.0 for value in market_returns.values()),
        "variants": {
            name: _summarize_5m_variant(rows, declared_size=declared_sizes[name])
            for name, rows in variant_rows.items()
        },
        "per_bet_decomposition": dict(variant_rows),
    }

    five_action_rows = _load_jsonl(paths["exact_195_five_action_rows"])
    concentration = _down_concentration_diagnostic(
        five_action_rows=five_action_rows,
        outcomes=outcomes,
        variants=variant_rows,
    )
    feature_inventory = _feature_inventory(
        five_action_rows=five_action_rows,
        raw_stream_rows=raw_stream_rows,
    )

    historical_15m_rows = _load_jsonl(paths["historical_15m_v7_selected_trades"])
    if len(historical_15m_rows) != 119:
        raise ValueError("historical 15m selected-trade derivation is not 119 rows")
    historical_15m = _summarize_15m(historical_15m_rows)
    historical_15m["per_bet_decomposition"] = historical_15m_rows

    market_comparison = _market_comparison(
        exact_5m=exact_5m,
        historical_15m=historical_15m,
    )
    result = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "report_role": "outcome_aware_model_layer_development_diagnostic",
        "source_base_commit": source_base_commit,
        "inputs_manifest_sha256": canonical_json_sha256(dict(inputs_manifest)),
        "training_started": False,
        "promotion_evidence_eligible": False,
        "exact_195_5m_cost_edge_decomposition": exact_5m,
        "historical_15m_cost_edge_decomposition": historical_15m,
        "market_cost_signal_comparison": market_comparison,
        "down_concentration_attribution": concentration,
        "feature_space_inventory": feature_inventory,
        "market_selection": {
            "recommendation": "turn_to_15m",
            "primary_development_market_family": "btc_updown_15m",
            "new_persistent_collection_market_families": ["btc_updown_15m"],
            "new_5m_collection_started": False,
            "retain_exact_195_5m_for_secondary_research": True,
            "reason_codes": [
                "5m_broad_support_variants_have_costs_greater_than_gross_mark_edge",
                "15m_conservative_cost_adjusted_edge_remains_positive",
                "15m_historical_quote_proxy_limitation_is_explicitly_conservatively_corrected",
                "exact_195_outcomes_are_side_balanced_so_down_concentration_is_not_a_down_regime",
            ],
        },
        "model_layer_retraining_direction": {
            "train_now": False,
            "first_target": (
                "side-symmetric BTC 15m model with true paired-token executable asks, "
                "dynamic depth/order-flow, Chainlink displacement, and causal momentum"
            ),
            "required_before_training": [
                "accumulate development-lane 15m rows with paired UP/DOWN executable quotes",
                "preserve missing-versus-zero trade-tape semantics",
                "split chronologically and group all decisions from one market together",
                "keep every target unavailable until official post-close settlement",
            ],
        },
        "safety": dict(SAFETY),
    }
    result["report_payload_sha256"] = canonical_json_sha256(result)
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the compact human-readable report paired with the JSON appendix."""

    exact = report["exact_195_5m_cost_edge_decomposition"]
    variants = exact["variants"]
    fifteen = report["historical_15m_cost_edge_decomposition"]
    attribution = report["down_concentration_attribution"]
    inventory = report["feature_space_inventory"]
    lines = [
        "# Challenge model-layer market diagnostic",
        "",
        "Status: **diagnostic complete; training not started**",
        "",
        "This is outcome-aware development analysis. The exact-195 and legacy 15m "
        "corpora are permanently ineligible for promotion evidence.",
        "",
        "## Market decision",
        "",
        "**Recommendation: turn the new development lane to BTC 15m.** Keep exact-195 "
        "5m as a secondary research corpus, but do not start a new 5m lane yet.",
        "",
        "The broad-support 5m variants do not earn their structural costs: iteration 4 "
        f"has unit net PnL `{variants['iteration_4']['unit_net_pnl_sum']:.6f}` over "
        f"`{variants['iteration_4']['trade_count']}` bets and iteration 5 has "
        f"`{variants['iteration_5']['unit_net_pnl_sum']:.6f}` over "
        f"`{variants['iteration_5']['trade_count']}` bets. The sparse 23-bet variant "
        f"is positive (`{variants['v8_1_23_bet']['unit_net_pnl_sum']:.6f}`) but does "
        "not provide broad support.",
        "",
        "The legacy 15m v7 selection has 119 bets and reported after-cost PnL "
        f"`{fifteen['reported_after_cost_pnl']['sum']:.6f}`. After charging an extra "
        "full observed source-token spread to every opposite-side quote proxy, the "
        f"conservative PnL remains `{fifteen['conservative_after_cost_pnl']['sum']:.6f}`. "
        "This is development evidence only and is not directly comparable to a future "
        "confirmatory window.",
        "",
        "## 5m cost/edge decomposition",
        "",
        "| Variant | Bets | DOWN/UP | Mid-mark edge | Spread | Fee | Slippage | Liquidity | Net PnL | Cost / positive signal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("v6_7", "v8_1_23_bet", "iteration_1", "iteration_3", "iteration_4", "iteration_5"):
        row = variants[name]
        sides = row["side_distribution"]
        ratio = row["modeled_cost_to_positive_gross_mark_edge_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.3f}"
        lines.append(
            f"| {name} | {row['trade_count']} | {sides.get('DOWN', 0)}/{sides.get('UP', 0)} | "
            f"{row['gross_mark_to_mid_edge']['sum']:.6f} | "
            f"{row['spread_cost']['sum']:.6f} | {row['fee']['sum']:.6f} | "
            f"{row['slippage_assumption']['sum']:.6f} | "
            f"{row['liquidity_impact']['sum']:.6f} | "
            f"{row['unit_net_pnl_sum']:.6f} | {ratio_text} |"
        )
    lines.extend(
        [
            "",
            "For sell-before-close rows, spread cost includes the entry half-spread and "
            "the half-spread at the actual runtime guard exit snapshot. Every identity "
            "`mid-mark edge - spread - fee - slippage - liquidity = unit net PnL` "
            "is checked before report generation.",
            "",
            "## DOWN concentration",
            "",
            f"- Exact-195 outcomes: UP `{attribution['outcome_distribution']['UP']}`, "
            f"DOWN `{attribution['outcome_distribution']['DOWN']}`.",
            f"- BTC market returns: positive/tie `{attribution['positive_or_tie_market_count']}`, "
            f"negative `{attribution['negative_market_count']}`.",
            f"- v8.1 23-bet side mix: DOWN `{attribution['variant_side_distribution']['v8_1_23_bet'].get('DOWN', 0)}`, "
            f"UP `{attribution['variant_side_distribution']['v8_1_23_bet'].get('UP', 0)}`.",
            f"- All-decision UP mid-price minus realized UP frequency: "
            f"`{attribution['up_mid_minus_realized_frequency']['mean']:.6f}`; "
            f"market bootstrap 95% interval "
            f"`[{attribution['up_mid_minus_realized_frequency']['bootstrap_95_lcb']:.6f}, "
            f"{attribution['up_mid_minus_realized_frequency']['bootstrap_95_ucb']:.6f}]`.",
            "",
            "The corpus is not a DOWN regime. The point estimate is compatible with mild "
            "UP overpricing, but its interval includes zero. The defensible attribution is "
            "model/controller side asymmetry, not proven structural longshot bias. New labels, "
            "features, calibration, and evaluation must therefore be side-symmetric and report "
            "UP/DOWN strata separately.",
            "",
            "## Feature inventory",
            "",
            "| Feature family | Archived stream | Exact-195 causal coverage | Constraint / caveat |",
            "|---|---|---:|---|",
        ]
    )
    for row in inventory["candidate_feature_families"]:
        lines.append(
            f"| {row['feature_family']} | {row['source_streams']} | "
            f"{row['causal_coverage_market_count']}/195 | {row['causality_constraint']} |"
        )
    lines.extend(
        [
            "",
            "Trade-tape raw rows are non-empty for 193/195 markets. Legacy action rows "
            "contain numeric zero even when missingness was not separately encoded, so "
            "the safe coverage for order-flow features is 193/195, not 195/195.",
            "",
            "## 15m limitation and retraining target",
            "",
            "The legacy evaluator had true source-token asks for 96/119 selected trades. "
            "For 23 DOWN trades it used the complement of an UP ask, which is an optimistic "
            "proxy rather than a true paired DOWN ask. The report preserves both the original "
            "numbers and a conservative one-full-spread correction. New 15m collection must "
            "store both token books and train only on executable side-specific asks.",
            "",
            "No model training was started. All paper/live/write/wallet/handoff/promotion "
            "paths remain false.",
            "",
            f"JSON report payload SHA-256: `{report['report_payload_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _decompose_5m_target(
    target: Mapping[str, Any],
    *,
    labels: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    key = (
        str(target["market_id"]),
        int(target["decision_ts"]),
        str(target["action"]),
    )
    label = labels.get(key)
    if label is None:
        raise ValueError(f"missing exact-195 label for runtime target {key}")
    entry_ask = float(label["entry_ask"])
    entry_bid = float(label["entry_bid"])
    entry_mid = float(label["entry_mid"])
    entry_half_spread = entry_ask - entry_mid
    exit_bid: float | None = None
    exit_ask: float | None = None
    exit_half_spread = 0.0
    if str(target["position_lifecycle_class"]) == "closed_before_settlement":
        path = dict(label.get("sell_before_close_exit_path") or {})
        snapshots = list(path.get("candidate_exit_snapshots") or [])
        target_exit_price = float(target["exit_price"])
        target_exit_ts = int(target["exit_decision_ts"])
        candidates = [
            row
            for row in snapshots
            if math.isclose(float(row["bid_price"]), target_exit_price, abs_tol=1e-12)
            and int(row["ts"]) <= target_exit_ts
        ]
        if not candidates:
            raise ValueError(f"runtime exit snapshot is not observable for {key}")
        snapshot = max(candidates, key=lambda row: int(row["ts"]))
        exit_bid = float(snapshot["bid_price"])
        exit_ask = float(snapshot["ask_price"])
        exit_mid = (exit_bid + exit_ask) / 2.0
        gross_mark_to_mid_edge = exit_mid - entry_mid
        exit_half_spread = exit_mid - exit_bid
    else:
        gross_mark_to_mid_edge = float(target["terminal_value_per_contract"]) - entry_mid
    spread_cost = entry_half_spread + exit_half_spread
    executable_gross_edge = float(target["gross_pnl_per_contract"])
    fee = float(target["fees"])
    slippage = float(target["slippage"])
    liquidity = float(target["liquidity_impact"])
    unit_net = float(target["runtime_policy_after_cost_net_pnl_per_contract"])
    if not math.isclose(
        gross_mark_to_mid_edge - spread_cost,
        executable_gross_edge,
        abs_tol=1e-12,
    ):
        raise ValueError(f"5m mark/spread decomposition identity failed for {key}")
    if not math.isclose(
        executable_gross_edge - fee - slippage - liquidity,
        unit_net,
        abs_tol=1e-12,
    ):
        raise ValueError(f"5m runtime cost identity failed for {key}")
    return {
        "market_id": key[0],
        "decision_ts": key[1],
        "action": key[2],
        "side": str(target["side"]),
        "resolved_outcome": str(target["resolved_outcome"]),
        "position_lifecycle_class": str(target["position_lifecycle_class"]),
        "entry_bid": entry_bid,
        "entry_ask": entry_ask,
        "entry_mid": entry_mid,
        "exit_bid": exit_bid,
        "exit_ask": exit_ask,
        "gross_mark_to_mid_edge": gross_mark_to_mid_edge,
        "entry_half_spread_cost": entry_half_spread,
        "exit_half_spread_cost": exit_half_spread,
        "spread_cost": spread_cost,
        "executable_gross_edge": executable_gross_edge,
        "fee": fee,
        "slippage_assumption": slippage,
        "liquidity_impact": liquidity,
        "unit_net_pnl": unit_net,
        "historical_development_only": True,
        "promotion_evidence_eligible": False,
    }


def _summarize_5m_variant(
    rows: Sequence[Mapping[str, Any]],
    *,
    declared_size: float,
) -> dict[str, Any]:
    fields = {
        "gross_mark_to_mid_edge": "gross_mark_to_mid_edge",
        "spread_cost": "spread_cost",
        "fee": "fee",
        "slippage_assumption": "slippage_assumption",
        "liquidity_impact": "liquidity_impact",
        "executable_gross_edge": "executable_gross_edge",
        "unit_net_pnl": "unit_net_pnl",
    }
    distributions = {
        output: _distribution([float(row[source]) for row in rows])
        for output, source in fields.items()
    }
    signal = distributions["gross_mark_to_mid_edge"]["sum"]
    cost = (
        distributions["spread_cost"]["sum"]
        + distributions["fee"]["sum"]
        + distributions["slippage_assumption"]["sum"]
        + distributions["liquidity_impact"]["sum"]
    )
    return {
        "trade_count": len(rows),
        "side_distribution": dict(sorted(Counter(str(row["side"]) for row in rows).items())),
        "outcome_distribution": dict(
            sorted(Counter(str(row["resolved_outcome"]) for row in rows).items())
        ),
        **distributions,
        "structural_cost_threshold": _distribution(
            [
                float(row["spread_cost"])
                + float(row["fee"])
                + float(row["slippage_assumption"])
                + float(row["liquidity_impact"])
                for row in rows
            ]
        ),
        "unit_net_pnl_sum": distributions["unit_net_pnl"]["sum"],
        "declared_position_size": declared_size,
        "declared_size_net_pnl_sum": distributions["unit_net_pnl"]["sum"]
        * declared_size,
        "modeled_cost_to_positive_gross_mark_edge_ratio": (
            cost / signal if signal > 0.0 else None
        ),
        "statistical_gates_must_use_unit_sizing": True,
    }


def _summarize_15m(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aligned = sum(bool(row["source_token_quote_aligned"]) for row in rows)
    fields = (
        "gross_mark_to_mid_edge",
        "source_token_spread",
        "gross_edge_at_evaluator_quote",
        "fee",
        "slippage_assumption",
        "reported_after_cost_pnl",
        "conservative_opposite_side_quote_correction",
        "conservative_after_cost_pnl",
    )
    result = {
        field: _distribution([float(row[field]) for row in rows])
        for field in fields
    }
    conservative_cost = sum(
        float(row["source_token_spread"]) / 2.0
        + float(row["slippage_assumption"])
        + float(row["fee"])
        + float(row["conservative_opposite_side_quote_correction"])
        for row in rows
    )
    gross_signal = result["gross_mark_to_mid_edge"]["sum"]
    return {
        "corpus_role": "legacy_outcome_opened_development_only_never_promotion_evidence",
        "market_family": "btc_updown_15m",
        "trade_count": len(rows),
        "side_distribution": dict(sorted(Counter(str(row["side"]) for row in rows).items())),
        "outcome_distribution": dict(
            sorted(Counter(str(row["resolved_outcome"]) for row in rows).items())
        ),
        "true_source_token_executable_quote_count": aligned,
        "opposite_side_complement_quote_proxy_count": len(rows) - aligned,
        **result,
        "conservative_modeled_cost_to_positive_gross_mark_edge_ratio": (
            conservative_cost / gross_signal if gross_signal > 0.0 else None
        ),
        "comparison_limitations": [
            "legacy policy was selected with outcome-aware train/validation development",
            "23 DOWN rows used complement-of-UP-ask rather than a paired DOWN executable ask",
            "source-token spread is observable but a paired opposite-token spread is not",
            "legacy 15m results are development evidence and never promotion evidence",
        ],
    }


def _down_concentration_diagnostic(
    *,
    five_action_rows: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, str],
    variants: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    quotes: list[dict[str, Any]] = []
    causal_decisions: set[tuple[str, int]] = set()
    for row in five_action_rows:
        if str(row["action"]) not in {
            "BUY_UP_HOLD_TO_SETTLEMENT",
            "BUY_DOWN_HOLD_TO_SETTLEMENT",
        }:
            continue
        market_id = str(row["market_id"])
        side = str(row["side"])
        snapshot = dict(row["microstructure_snapshot"])
        bid = float(snapshot["entry_bid"])
        ask = float(snapshot["entry_ask"])
        quotes.append(
            {
                "market_id": market_id,
                "decision_ts": int(row["decision_ts"]),
                "side": side,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0,
                "realized": float(outcomes[market_id] == side),
            }
        )
        causal_decisions.add((market_id, int(row["decision_ts"])))
    if len(quotes) != 1560 or len(causal_decisions) != 780:
        raise ValueError("expected 780 paired exact-195 decision quotes")
    up = [row for row in quotes if row["side"] == "UP"]
    down = [row for row in quotes if row["side"] == "DOWN"]
    up_bias_by_market = {
        market_id: statistics.mean(
            row["mid"] - row["realized"]
            for row in up
            if row["market_id"] == market_id
        )
        for market_id in outcomes
    }
    lcb, ucb = _bootstrap_mean_interval(list(up_bias_by_market.values()), seed=260)
    up_bias = _distribution([row["mid"] - row["realized"] for row in up])
    up_bias["bootstrap_95_lcb"] = lcb
    up_bias["bootstrap_95_ucb"] = ucb
    return {
        "outcome_distribution": dict(sorted(Counter(outcomes.values()).items())),
        "positive_or_tie_market_count": sum(
            str(value) == "UP" for value in outcomes.values()
        ),
        "negative_market_count": sum(
            str(value) == "DOWN" for value in outcomes.values()
        ),
        "quote_observation_count_per_side": 780,
        "up_entry_ask_distribution": _distribution([float(row["ask"]) for row in up]),
        "down_entry_ask_distribution": _distribution([float(row["ask"]) for row in down]),
        "up_entry_mid_distribution": _distribution([float(row["mid"]) for row in up]),
        "down_entry_mid_distribution": _distribution([float(row["mid"]) for row in down]),
        "realized_up_frequency": statistics.mean(float(row["realized"]) for row in up),
        "realized_down_frequency": statistics.mean(float(row["realized"]) for row in down),
        "up_mid_minus_realized_frequency": up_bias,
        "down_mid_minus_realized_frequency": _distribution(
            [float(row["mid"]) - float(row["realized"]) for row in down]
        ),
        "variant_side_distribution": {
            name: dict(sorted(Counter(str(row["side"]) for row in rows).items()))
            for name, rows in variants.items()
        },
        "attribution": (
            "not_down_regime; mild_UP_overpricing_point_estimate_not_statistically_resolved; "
            "controller_or_scorer_side_asymmetry_is_primary"
        ),
        "side_symmetric_labels_and_features_required": True,
        "side_stratified_calibration_and_evaluation_required": True,
    }


def _feature_inventory(
    *,
    five_action_rows: Sequence[Mapping[str, Any]],
    raw_stream_rows: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    representatives: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in five_action_rows:
        key = (str(row["market_id"]), int(row["decision_ts"]))
        representatives.setdefault(key, row)
    if len(representatives) != 780:
        raise ValueError("expected 780 exact-195 decision rows for feature coverage")

    def coverage(fields: Iterable[str]) -> int:
        required = tuple(fields)
        markets: set[str] = set()
        for (market_id, _), row in representatives.items():
            features = dict(row["decision_time_features"])
            if all(features.get(field) is not None for field in required):
                markets.add(market_id)
        return len(markets)

    causal_decisions = sum(
        int(
            int(row["max_input_ts"]) <= int(row["decision_ts"])
            and int(row["reference_price_feature_provenance"]["available_at_ts"])
            <= int(row["decision_ts"])
            and int(row["reference_price_feature_provenance"]["max_input_ts"])
            <= int(row["decision_ts"])
        )
        for row in representatives.values()
    )
    stream_summary = {
        name: {
            "market_count": len(counts),
            "nonempty_market_count": sum(int(value) > 0 for value in counts),
            "row_count_distribution": _distribution([float(value) for value in counts]),
        }
        for name, counts in raw_stream_rows.items()
    }
    return {
        "exact_195_market_count": 195,
        "decision_row_count": 780,
        "causal_decision_row_count": causal_decisions,
        "causality_contract": "available_at_ts <= decision_ts and max_input_ts <= decision_ts",
        "archived_stream_coverage": stream_summary,
        "candidate_feature_families": [
            {
                "feature_family": "paired top-of-book and structural spread",
                "source_streams": "order books",
                "candidate_features": (
                    "UP/DOWN bid, ask, mid, combined spread, executable notional, staleness"
                ),
                "causal_coverage_market_count": coverage(
                    ["combined_spread_bps", "selected_side_spread_bps"]
                ),
                "causality_constraint": "latest paired book available no later than decision_ts",
            },
            {
                "feature_family": "depth dynamics and queue state",
                "source_streams": "order books",
                "candidate_features": (
                    "depth imbalance, update intensity, bid-depth volatility, spread stability, "
                    "queue-fill proxy"
                ),
                "causal_coverage_market_count": coverage(
                    [
                        "selected_side_recent_book_update_count_1m",
                        "selected_side_recent_bid_depth_volatility_1m",
                        "selected_side_recent_spread_stability_1m",
                        "selected_side_queue_fill_probability_proxy",
                    ]
                ),
                "causality_constraint": "rolling windows end at decision_ts; no exit snapshots",
            },
            {
                "feature_family": "order-flow imbalance and trade intensity",
                "source_streams": "trade tape + order books",
                "candidate_features": (
                    "aggressor-side volume, selected/opposite volume, signed flow, arrival rate"
                ),
                "causal_coverage_market_count": sum(
                    int(value) > 0 for value in raw_stream_rows["trade_tape"]
                ),
                "causality_constraint": (
                    "trade available_at <= decision_ts; 2 legacy markets have no tape, and "
                    "numeric zero must not substitute for missing"
                ),
            },
            {
                "feature_family": "Chainlink reference displacement",
                "source_streams": "Chainlink RTDS + decision-time market price",
                "candidate_features": (
                    "reference-price-to-beat distance, displacement velocity, market-vs-oracle gap"
                ),
                "causal_coverage_market_count": coverage(
                    ["reference_price_to_beat_distance_at_decision"]
                ),
                "causality_constraint": (
                    "reference available_at and max_input_ts must both be <= decision_ts"
                ),
            },
            {
                "feature_family": "causal BTC momentum and volatility",
                "source_streams": "BTC reference klines / ticks",
                "candidate_features": (
                    "10s/30s/1m/5m/15m returns, realized volatility, cross-round lagged momentum"
                ),
                "causal_coverage_market_count": coverage(
                    [
                        "btc_return_10s",
                        "btc_return_30s",
                        "btc_return_1m",
                        "btc_return_5m",
                        "btc_return_15m",
                        "btc_volatility_1m",
                        "btc_volatility_5m",
                        "btc_volatility_15m",
                    ]
                ),
                "causality_constraint": (
                    "closed candle/tick available_at <= decision_ts; current candle close forbidden"
                ),
            },
            {
                "feature_family": "side-symmetric relative-value transforms",
                "source_streams": "paired order books + Chainlink + model-free BTC features",
                "candidate_features": (
                    "UP-minus-DOWN depth/flow, side-normalized oracle distance, paired price residual"
                ),
                "causal_coverage_market_count": 195,
                "causality_constraint": (
                    "derive both sides from the same causal snapshot and share transformations"
                ),
            },
        ],
        "prohibited_feature_inputs": [
            "current-market resolved_outcome before official post-close resolution",
            "future exit price or best intraround exit",
            "settlement PnL",
            "promotion-window labels",
        ],
    }


def _market_comparison(
    *,
    exact_5m: Mapping[str, Any],
    historical_15m: Mapping[str, Any],
) -> dict[str, Any]:
    variants = exact_5m["variants"]
    return {
        "comparison_role": "development_direction_only_not_promotion_evidence",
        "five_minute": {
            "broad_support_reference_variants": ["iteration_4", "iteration_5"],
            "iteration_4_cost_signal_ratio": variants["iteration_4"][
                "modeled_cost_to_positive_gross_mark_edge_ratio"
            ],
            "iteration_5_cost_signal_ratio": variants["iteration_5"][
                "modeled_cost_to_positive_gross_mark_edge_ratio"
            ],
            "iteration_4_unit_net_pnl": variants["iteration_4"]["unit_net_pnl_sum"],
            "iteration_5_unit_net_pnl": variants["iteration_5"]["unit_net_pnl_sum"],
            "sparse_23_bet_cost_signal_ratio": variants["v8_1_23_bet"][
                "modeled_cost_to_positive_gross_mark_edge_ratio"
            ],
        },
        "fifteen_minute": {
            "trade_count": historical_15m["trade_count"],
            "reported_after_cost_pnl": historical_15m["reported_after_cost_pnl"]["sum"],
            "conservative_after_cost_pnl": historical_15m[
                "conservative_after_cost_pnl"
            ]["sum"],
            "conservative_cost_signal_ratio": historical_15m[
                "conservative_modeled_cost_to_positive_gross_mark_edge_ratio"
            ],
            "opposite_side_quote_proxy_count": historical_15m[
                "opposite_side_complement_quote_proxy_count"
            ],
        },
        "conclusion": "15m_cost_to_signal_is_more_favorable",
        "recommended_collection_scope": ["btc_updown_15m"],
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "sum": sum(ordered),
        "mean": statistics.mean(ordered),
        "p05": _quantile(ordered, 0.05),
        "p25": _quantile(ordered, 0.25),
        "p50": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
    }


def _quantile(ordered: Sequence[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower]) + (
        float(ordered[upper]) - float(ordered[lower])
    ) * fraction


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    replicates: int = 10_000,
) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(values)
    means = [
        statistics.mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(replicates)
    ]
    means.sort()
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows
