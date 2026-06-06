from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SCRIPT = REPO_ROOT / "scripts" / "champion_signal_bridge.py"
EXECUTOR_SCRIPT = REPO_ROOT / "scripts" / "polymarket_phase4_live_champion_executor.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bridge = _load_script(BRIDGE_SCRIPT, "champion_signal_bridge")
executor = _load_script(EXECUTOR_SCRIPT, "polymarket_phase4_live_champion_executor")


def test_bridge_signal_from_prediction_event_row() -> None:
    snapshot = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
        "source_symbol": "token-up",
        "market_implied_prob": 0.47,
    }

    signal = bridge._bridge_signal_from_row(
        ("pred-1", 1_779_774_400_000, 1_779_774_410_000, 0.98, json.dumps(snapshot)),
        model_version="xgboost-v4",
    )

    assert signal is not None
    assert signal.event_id == "pred-1"
    assert signal.round_slug == "btc-updown-15m-1779774300"
    assert signal.round_end_ts == 1_779_775_200_000
    assert signal.outcome_side == "UP"
    assert signal.token_id == "token-up"
    assert signal.token_probability == 0.98
    assert signal.edge == 0.51
    assert signal.bridged_at > 0
    assert signal.opposite_token_id == ""


def test_bridge_signals_include_opposite_token_id(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.duckdb"
    ts = 1_779_774_400_000
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE prediction_events (
                event_id VARCHAR,
                ts BIGINT,
                created_at BIGINT,
                model_version VARCHAR,
                prob_up_15m DOUBLE,
                feature_snapshot_json VARCHAR
            )
            """
        )
        conn.executemany(
            "INSERT INTO prediction_events VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "pred-up",
                    ts,
                    ts + 1_000,
                    "xgboost-v4",
                    0.98,
                    json.dumps(
                        {
                            "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
                            "source_symbol": "token-up",
                            "market_implied_prob": 0.47,
                        }
                    ),
                ),
                (
                    "pred-down",
                    ts,
                    ts + 1_001,
                    "xgboost-v4",
                    0.98,
                    json.dumps(
                        {
                            "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:DOWN",
                            "source_symbol": "token-down",
                            "market_implied_prob": 0.53,
                        }
                    ),
                ),
            ],
        )

    signals = bridge._read_bridge_signals_after(
        str(db_path),
        model_version="xgboost-v4",
        after_created_at=0,
        after_event_id="",
        limit=10,
    )

    by_side = {signal.outcome_side: signal for signal in signals}
    assert by_side["UP"].opposite_token_id == "token-down"
    assert by_side["DOWN"].opposite_token_id == "token-up"


def test_bridge_outcome_side_filter_keeps_up_only_predictions(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.duckdb"
    ts = 1_779_774_400_000
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE prediction_events (
                event_id VARCHAR,
                ts BIGINT,
                created_at BIGINT,
                model_version VARCHAR,
                prob_up_15m DOUBLE,
                feature_snapshot_json VARCHAR
            )
            """
        )
        conn.executemany(
            "INSERT INTO prediction_events VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "pred-up",
                    ts,
                    ts + 1_000,
                    "xgboost-v5",
                    0.90,
                    json.dumps(
                        {
                            "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
                            "source_symbol": "token-up",
                            "market_implied_prob": 0.47,
                        }
                    ),
                ),
                (
                    "pred-down",
                    ts,
                    ts + 1_001,
                    "xgboost-v5",
                    0.10,
                    json.dumps(
                        {
                            "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:DOWN",
                            "source_symbol": "token-down",
                            "market_implied_prob": 0.53,
                        }
                    ),
                ),
            ],
        )

    signals = bridge._read_bridge_signals_after(
        str(db_path),
        model_version="xgboost-v5",
        after_created_at=0,
        after_event_id="",
        limit=10,
        allowed_outcome_sides=frozenset({"UP"}),
    )

    assert [signal.outcome_side for signal in signals] == ["UP"]
    assert signals[0].opposite_token_id == "token-down"


def test_bridge_forwards_multiple_15m_families(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.duckdb"
    ts = 1_779_774_400_000
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE prediction_events (
                event_id VARCHAR,
                ts BIGINT,
                created_at BIGINT,
                model_version VARCHAR,
                prob_up_15m DOUBLE,
                feature_snapshot_json VARCHAR
            )
            """
        )
        conn.executemany(
            "INSERT INTO prediction_events VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "btc-up",
                    ts,
                    ts + 1_000,
                    "xgboost-v5",
                    0.90,
                    json.dumps(
                        {
                            "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
                            "source_symbol": "btc-token-up",
                            "market_implied_prob": 0.50,
                        }
                    ),
                ),
                (
                    "eth-up",
                    ts,
                    ts + 1_002,
                    "xgboost-v5",
                    0.90,
                    json.dumps(
                        {
                            "canonical_symbol": "ETH-15M:eth-updown-15m-1779774300:UP",
                            "source_symbol": "eth-token-up",
                            "market_implied_prob": 0.50,
                        }
                    ),
                ),
                (
                    "eth-down",
                    ts,
                    ts + 1_003,
                    "xgboost-v5",
                    0.90,
                    json.dumps(
                        {
                            "canonical_symbol": "ETH-15M:eth-updown-15m-1779774300:DOWN",
                            "source_symbol": "eth-token-down",
                            "market_implied_prob": 0.50,
                        }
                    ),
                ),
                (
                    "eth-5m-up",
                    ts,
                    ts + 1_004,
                    "xgboost-v5",
                    0.90,
                    json.dumps(
                        {
                            "canonical_symbol": "ETH-5M:eth-updown-5m-1779774300:UP",
                            "source_symbol": "eth-5m-token-up",
                            "market_implied_prob": 0.50,
                        }
                    ),
                ),
            ],
        )

    signals = bridge._read_bridge_signals_after(
        str(db_path),
        model_version="xgboost-v5",
        after_created_at=0,
        after_event_id="",
        limit=10,
        allowed_families=frozenset({"BTC-15M", "ETH-15M"}),
    )

    families = {signal.canonical_symbol.split(":", 1)[0] for signal in signals}
    assert families == {"BTC-15M", "ETH-15M"}
    # ETH-5M is excluded by the allow-set.
    assert all(not s.canonical_symbol.startswith("ETH-5M") for s in signals)
    # ETH opposite-token reconstruction uses the ETH family, not a hardcoded BTC prefix.
    eth_up = next(s for s in signals if s.canonical_symbol.startswith("ETH-15M") and s.outcome_side == "UP")
    assert eth_up.opposite_token_id == "eth-token-down"


def test_bridge_skips_post_expiry_degenerate_signal() -> None:
    snapshot = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-1779755400:UP",
        "source_symbol": "token-up",
        "market_implied_prob": 1.0,
        "features": {
            "market_implied_prob": 1.0,
            "spread": 1.0,
            "tick_spread": 1.0,
            "liquidity_bucket": 0.0,
        },
    }

    signal = bridge._bridge_signal_from_row(
        ("pred-dirty", 1_779_756_360_000, 1_779_756_361_000, 0.99, json.dumps(snapshot)),
        model_version="xgboost-v4",
    )

    assert signal is None


def test_bridge_skips_future_round_before_start() -> None:
    snapshot = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
        "source_symbol": "token-up",
        "market_implied_prob": 0.47,
    }

    signal = bridge._bridge_signal_from_row(
        ("pred-future", 1_779_773_900_000, 1_779_773_910_000, 0.98, json.dumps(snapshot)),
        model_version="xgboost-v4",
    )

    assert signal is None


def test_bridge_v6_settlement_gate_emits_down_side_signal_without_volatility_pass() -> None:
    from bigan.execution.v6_gate import V6JointGateConfig

    snapshot = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
        "source_symbol": "token-up",
        "market_implied_prob": 0.40,
        "p_up": 0.10,
        "p_down": 0.85,
        "p_neutral": 0.05,
        "p_vol_up": 0.10,
        "p_vol_down": 0.10,
    }
    config = V6JointGateConfig(
        settlement_threshold=0.50,
        neutral_cap=0.25,
        volatility_threshold=0.50,
        round_trip_cost=0.04,
        ev_margin=0.01,
        gain_priors=(("up", 0.30), ("down", 0.30)),
    )
    signal = bridge._bridge_signal_from_row(
        ("pred-v6", 1_779_774_400_000, 1_779_774_410_000, 0.10, json.dumps(snapshot)),
        model_version="xgboost-v6",
        v6_joint_config=config,
        opposite_token_id="token-down",
    )
    assert signal is not None
    assert signal.outcome_side == "DOWN"
    assert signal.token_id == "token-down"
    assert signal.v6_joint_side == "DOWN"
    assert signal.p_vol_down == pytest.approx(0.10)


def test_bridge_v6_emits_volatility_only_signal_without_settlement_side() -> None:
    from bigan.execution.v6_gate import V6JointGateConfig

    snapshot = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
        "source_symbol": "token-up",
        "market_implied_prob": 0.40,
        "p_up": 0.41,
        "p_down": 0.47,
        "p_neutral": 0.12,
        "p_vol_up": 0.56,
        "p_vol_down": 0.62,
    }
    config = V6JointGateConfig(
        settlement_threshold=0.50,
        neutral_cap=0.25,
        volatility_threshold=0.60,
    )
    signal = bridge._bridge_signal_from_row(
        ("pred-v6-vol", 1_779_774_400_000, 1_779_774_410_000, 0.41, json.dumps(snapshot)),
        model_version="xgboost-v6",
        v6_joint_config=config,
        opposite_token_id="token-down",
    )

    assert signal is not None
    assert signal.outcome_side == "DOWN"
    assert signal.token_id == "token-down"
    assert signal.token_probability == pytest.approx(0.62)
    assert signal.v6_joint_side is None


def test_executor_reads_bridged_signal_jsonl(tmp_path: Path) -> None:
    queue = tmp_path / "signals.jsonl"
    payload = {
        "event_id": "pred-1",
        "ts": 1_779_773_900_000,
        "created_at": 1_779_773_910_000,
        "model_version": "xgboost-v4",
        "prob_up_15m": 0.98,
        "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
        "token_id": "token-up",
        "outcome_side": "UP",
        "round_slug": "btc-updown-15m-1779774300",
        "round_end_ts": 1_779_775_200_000,
        "market_implied_prob": 0.47,
        "token_probability": 0.98,
        "edge": 0.51,
        "bridged_at": 1_779_773_912_000,
        "opposite_token_id": "token-down",
    }
    queue.write_text(json.dumps(payload) + "\nnot-json\n\n", encoding="utf-8")

    events, cursor, signature = executor._read_signal_jsonl_after(
        queue,
        after_line_number=0,
        model_version="xgboost-v4",
        limit=10,
    )

    assert cursor == 3
    assert signature
    assert len(events) == 1
    assert events[0].event_id == "pred-1"
    assert events[0].edge == 0.51
    assert events[0].bridged_at == 1_779_773_912_000
    assert events[0].opposite_token_id == "token-down"


def test_executor_trusts_executor_ready_v6_volatility_jsonl_payload(tmp_path: Path) -> None:
    from bigan.execution.v6_gate import V6JointGateConfig

    queue = tmp_path / "signals.jsonl"
    payload = {
        "event_id": "pred-v6-vol",
        "ts": 1_779_774_400_000,
        "created_at": 1_779_774_410_000,
        "model_version": "xgboost-v6",
        "prob_up_15m": 0.41,
        "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:DOWN",
        "token_id": "token-down",
        "outcome_side": "DOWN",
        "round_slug": "btc-updown-15m-1779774300",
        "round_end_ts": 1_779_775_200_000,
        "market_implied_prob": 0.60,
        "token_probability": 0.62,
        "edge": 0.02,
        "bridged_at": 1_779_774_411_000,
        "opposite_token_id": "token-up",
        "p_up": 0.41,
        "p_down": 0.47,
        "p_neutral": 0.12,
        "p_vol_up": 0.56,
        "p_vol_down": 0.62,
        "v6_joint_side": None,
    }
    queue.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    events, _cursor, _signature = executor._read_signal_jsonl_after(
        queue,
        after_line_number=0,
        model_version="xgboost-v6",
        limit=10,
        v6_joint_config=V6JointGateConfig(volatility_threshold=0.60),
        entry_gate_mode="v6-joint",
    )

    assert len(events) == 1
    assert events[0].outcome_side == "DOWN"
    assert events[0].token_id == "token-down"
    assert events[0].token_probability == pytest.approx(0.62)
    assert events[0].p_vol_down == pytest.approx(0.62)
    assert events[0].v6_joint_side is None


def test_executor_reads_executor_ready_v7_pnl_jsonl_payload(tmp_path: Path) -> None:
    queue = tmp_path / "signals.jsonl"
    base_payload = {
        "event_id": "pred-v7-low",
        "ts": 1_779_774_400_000,
        "created_at": 1_779_774_410_000,
        "model_version": "xgboost-v7",
        "prob_up_15m": 0.76,
        "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
        "token_id": "token-up",
        "outcome_side": "UP",
        "round_slug": "btc-updown-15m-1779774300",
        "round_end_ts": 1_779_775_200_000,
        "market_implied_prob": 0.70,
        "token_probability": 0.76,
        "edge": 0.03,
        "bridged_at": 1_779_774_411_000,
        "opposite_token_id": "token-down",
        "p_up": 0.76,
        "p_down": 0.18,
        "p_neutral": 0.06,
        "settlement_residual": 0.12,
        "expected_edge_up": 0.03,
        "expected_edge_down": -0.12,
        "residual_expected_edge_up": 0.02,
        "residual_expected_edge_down": -0.09,
        "selected_side": "UP",
        "selected_expected_edge": 0.03,
        "entry_worst_price": 0.73,
        "should_enter_settlement": False,
    }
    stronger_payload = {
        **base_payload,
        "event_id": "pred-v7-high",
        "created_at": 1_779_774_412_000,
        "prob_up_15m": 0.84,
        "token_probability": 0.84,
        "edge": 0.12,
        "p_up": 0.84,
        "p_down": 0.10,
        "expected_edge_up": 0.12,
        "selected_expected_edge": 0.12,
        "entry_worst_price": 0.72,
        "should_enter_settlement": True,
    }
    queue.write_text(
        json.dumps(base_payload) + "\n" + json.dumps(stronger_payload) + "\n",
        encoding="utf-8",
    )

    events, cursor, _signature = executor._read_signal_jsonl_after(
        queue,
        after_line_number=0,
        model_version="xgboost-v7",
        limit=10,
        entry_gate_mode="v7-pnl",
    )

    assert cursor == 2
    assert len(events) == 1
    assert events[0].event_id == "pred-v7-high"
    assert events[0].selected_side == "UP"
    assert events[0].selected_expected_edge == pytest.approx(0.12)
    assert events[0].entry_worst_price == pytest.approx(0.72)
    assert events[0].should_enter_settlement is True
    assert events[0].settlement_residual == pytest.approx(0.12)


def test_v7_pnl_jsonl_selection_ignores_stale_high_edge_signal(tmp_path: Path) -> None:
    queue = tmp_path / "signals.jsonl"
    stale_high_edge = {
        "event_id": "pred-v7-stale-high",
        "ts": 1_779_774_000_000,
        "created_at": 1_779_774_010_000,
        "model_version": "xgboost-v7",
        "prob_up_15m": 0.85,
        "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
        "token_id": "token-up",
        "outcome_side": "UP",
        "round_slug": "btc-updown-15m-1779774300",
        "round_end_ts": 1_779_775_200_000,
        "market_implied_prob": 0.55,
        "token_probability": 0.85,
        "edge": 0.30,
        "bridged_at": 1_779_774_011_000,
        "opposite_token_id": "token-down",
        "p_up": 0.85,
        "p_down": 0.10,
        "p_neutral": 0.05,
        "selected_side": "UP",
        "selected_expected_edge": 0.30,
    }
    fresh_lower_edge = {
        **stale_high_edge,
        "event_id": "pred-v7-fresh-lower",
        "ts": 1_779_774_500_000,
        "created_at": 1_779_774_501_000,
        "prob_up_15m": 0.77,
        "token_probability": 0.77,
        "edge": 0.08,
        "bridged_at": 1_779_774_502_000,
        "p_up": 0.77,
        "p_down": 0.18,
        "selected_expected_edge": 0.08,
    }
    queue.write_text(
        json.dumps(stale_high_edge) + "\n" + json.dumps(fresh_lower_edge) + "\n",
        encoding="utf-8",
    )

    events, cursor, _signature = executor._read_signal_jsonl_after(
        queue,
        after_line_number=0,
        model_version="xgboost-v7",
        limit=10,
        entry_gate_mode="v7-pnl",
        selection_now_ms=1_779_774_530_000,
        max_signal_age_seconds=120,
    )

    assert cursor == 2
    assert len(events) == 1
    assert events[0].event_id == "pred-v7-fresh-lower"
    assert events[0].selected_expected_edge == pytest.approx(0.08)


def test_executor_reads_db_signal_with_opposite_token_id(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.duckdb"
    ts = 1_779_774_400_000
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE prediction_events (
                event_id VARCHAR,
                ts BIGINT,
                created_at BIGINT,
                model_version VARCHAR,
                prob_up_15m DOUBLE,
                feature_snapshot_json VARCHAR
            )
            """
        )
        conn.executemany(
            "INSERT INTO prediction_events VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "pred-up",
                    ts,
                    ts + 1_000,
                    "xgboost-v4",
                    0.98,
                    json.dumps(
                        {
                            "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
                            "source_symbol": "token-up",
                            "market_implied_prob": 0.47,
                        }
                    ),
                ),
                (
                    "pred-down",
                    ts,
                    ts + 1_001,
                    "xgboost-v4",
                    0.98,
                    json.dumps(
                        {
                            "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:DOWN",
                            "source_symbol": "token-down",
                            "market_implied_prob": 0.53,
                        }
                    ),
                ),
            ],
        )

    events = executor._read_events_after(
        str(db_path),
        model_version="xgboost-v4",
        after_created_at=0,
        after_event_id="",
        limit=10,
    )

    assert len(events) == 1
    assert events[0].outcome_side == "UP"
    assert events[0].opposite_token_id == "token-down"


def test_executor_duckdb_cursor_advances_past_v6_settlement_gate_misses(tmp_path: Path) -> None:
    from bigan.execution.v6_gate import V6JointGateConfig

    db_path = tmp_path / "catalog.duckdb"
    ts = 1_779_774_400_000

    def snapshot(
        *,
        side: str,
        source_symbol: str,
        p_up: float,
        p_down: float,
        p_vol_up: float,
        p_vol_down: float,
    ) -> str:
        return json.dumps(
            {
                "canonical_symbol": f"BTC-15M:btc-updown-15m-1779774300:{side}",
                "source_symbol": source_symbol,
                "market_implied_prob": 0.50,
                "p_up": p_up,
                "p_down": p_down,
                "p_neutral": 0.05,
                "p_vol_up": p_vol_up,
                "p_vol_down": p_vol_down,
            }
        )

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE prediction_events (
                event_id VARCHAR,
                ts BIGINT,
                created_at BIGINT,
                model_version VARCHAR,
                prob_up_15m DOUBLE,
                feature_snapshot_json VARCHAR
            )
            """
        )
        conn.executemany(
            "INSERT INTO prediction_events VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "pred-miss-up",
                    ts,
                    ts + 1_000,
                    "xgboost-v6",
                    0.40,
                    snapshot(
                        side="UP",
                        source_symbol="token-up",
                        p_up=0.40,
                        p_down=0.45,
                        p_vol_up=0.10,
                        p_vol_down=0.10,
                    ),
                ),
                (
                    "pred-miss-down",
                    ts,
                    ts + 1_001,
                    "xgboost-v6",
                    0.40,
                    snapshot(
                        side="DOWN",
                        source_symbol="token-down",
                        p_up=0.40,
                        p_down=0.45,
                        p_vol_up=0.10,
                        p_vol_down=0.10,
                    ),
                ),
                (
                    "pred-pass-down",
                    ts,
                    ts + 1_002,
                    "xgboost-v6",
                    0.10,
                    snapshot(
                        side="DOWN",
                        source_symbol="token-down",
                        p_up=0.85,
                        p_down=0.10,
                        p_vol_up=0.10,
                        p_vol_down=0.10,
                    ),
                ),
                (
                    "pred-pass-up-row",
                    ts,
                    ts + 1_003,
                    "xgboost-v6",
                    0.10,
                    snapshot(
                        side="UP",
                        source_symbol="token-up",
                        p_up=0.10,
                        p_down=0.85,
                        p_vol_up=0.10,
                        p_vol_down=0.90,
                    ),
                ),
            ],
        )
    config = V6JointGateConfig(
        settlement_threshold=0.50,
        neutral_cap=0.25,
        volatility_threshold=0.50,
        round_trip_cost=0.04,
        ev_margin=0.01,
        gain_priors=(("up", 0.30), ("down", 0.30)),
    )

    first = executor._read_event_batch_after(
        str(db_path),
        model_version="xgboost-v6",
        after_created_at=0,
        after_event_id="",
        limit=2,
        v6_joint_config=config,
    )
    second = executor._read_event_batch_after(
        str(db_path),
        model_version="xgboost-v6",
        after_created_at=first.cursor_created_at,
        after_event_id=first.cursor_event_id,
        limit=10,
        v6_joint_config=config,
    )

    assert first.events == []
    assert first.rows_scanned == 2
    assert first.rows_filtered == 2
    assert first.cursor_event_id == "pred-miss-down"
    assert first.filter_reasons == {"v6_settlement_gate_miss": 2}
    assert len(second.events) == 1
    assert second.events[0].outcome_side == "DOWN"
    assert second.events[0].token_id == "token-down"
    assert second.events[0].opposite_token_id == "token-up"
    assert second.events[0].p_vol_down == pytest.approx(0.10)


def test_executor_duckdb_keeps_v6_settlement_and_volatility_lanes_per_round(
    tmp_path: Path,
) -> None:
    from bigan.execution.v6_gate import V6JointGateConfig

    db_path = tmp_path / "catalog.duckdb"
    ts = 1_779_774_400_000

    def snapshot(
        *,
        side: str,
        source_symbol: str,
        p_up: float,
        p_down: float,
        p_vol_up: float,
        p_vol_down: float,
    ) -> str:
        return json.dumps(
            {
                "canonical_symbol": f"BTC-15M:btc-updown-15m-1779774300:{side}",
                "source_symbol": source_symbol,
                "market_implied_prob": 0.40 if side == "UP" else 0.60,
                "p_up": p_up,
                "p_down": p_down,
                "p_neutral": 0.05,
                "p_vol_up": p_vol_up,
                "p_vol_down": p_vol_down,
            }
        )

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE prediction_events (
                event_id VARCHAR,
                ts BIGINT,
                created_at BIGINT,
                model_version VARCHAR,
                prob_up_15m DOUBLE,
                feature_snapshot_json VARCHAR
            )
            """
        )
        conn.executemany(
            "INSERT INTO prediction_events VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "pred-vol-only",
                    ts,
                    ts + 1_000,
                    "xgboost-v6",
                    0.41,
                    snapshot(
                        side="UP",
                        source_symbol="token-up",
                        p_up=0.41,
                        p_down=0.47,
                        p_vol_up=0.56,
                        p_vol_down=0.62,
                    ),
                ),
                (
                    "pred-settlement",
                    ts,
                    ts + 1_001,
                    "xgboost-v6",
                    0.85,
                    snapshot(
                        side="DOWN",
                        source_symbol="token-down",
                        p_up=0.85,
                        p_down=0.10,
                        p_vol_up=0.10,
                        p_vol_down=0.10,
                    ),
                ),
            ],
        )
    config = V6JointGateConfig(
        settlement_threshold=0.50,
        neutral_cap=0.25,
        volatility_threshold=0.60,
    )

    batch = executor._read_event_batch_after(
        str(db_path),
        model_version="xgboost-v6",
        after_created_at=0,
        after_event_id="",
        limit=10,
        v6_joint_config=config,
    )

    by_id = {event.event_id: event for event in batch.events}
    assert set(by_id) == {"pred-vol-only", "pred-settlement"}
    assert by_id["pred-vol-only"].outcome_side == "DOWN"
    assert by_id["pred-vol-only"].token_id == "token-down"
    assert by_id["pred-vol-only"].token_probability == pytest.approx(0.62)
    assert by_id["pred-vol-only"].v6_joint_side is None
    assert by_id["pred-settlement"].outcome_side == "DOWN"
    assert by_id["pred-settlement"].p_up == pytest.approx(0.10)
    assert by_id["pred-settlement"].p_down == pytest.approx(0.85)
    assert by_id["pred-settlement"].v6_joint_side == "DOWN"


def test_executor_latency_helpers() -> None:
    event = executor.SignalEvent(
        event_id="pred-1",
        ts=1_000,
        created_at=2_000,
        prob_up_15m=0.98,
        canonical_symbol="BTC-15M:btc-updown-15m-1779774300:UP",
        token_id="token-up",
        outcome_side="UP",
        round_slug="btc-updown-15m-1779774300",
        round_end_ts=1_779_775_200_000,
        market_implied_prob=0.47,
        token_probability=0.98,
        edge=0.51,
        bridged_at=3_000,
    )

    assert executor._signal_latency_ms(event, 5_500) == {
        "event_ts_to_at_ms": 4_500,
        "signal_created_to_at_ms": 3_500,
        "bridge_to_at_ms": 2_500,
    }


def test_executor_event_family_filter() -> None:
    def _event(canonical_symbol: str):
        return executor.SignalEvent(
            event_id="pred-1",
            ts=1_000,
            created_at=2_000,
            prob_up_15m=0.98,
            canonical_symbol=canonical_symbol,
            token_id="token-up",
            outcome_side="UP",
            round_slug="round",
            round_end_ts=1_779_775_200_000,
            market_implied_prob=0.47,
            token_probability=0.98,
            edge=0.51,
        )

    btc_15m = _event("BTC-15M:btc-updown-15m-1779774300:UP")
    eth_15m = _event("ETH-15M:eth-updown-15m-1779774300:UP")
    eth_5m = _event("ETH-5M:eth-updown-5m-1779774300:UP")
    allowed = frozenset({"BTC-15M", "ETH-15M"})

    assert executor._event_family_allowed(btc_15m, allowed) is True
    assert executor._event_family_allowed(eth_15m, allowed) is True
    assert executor._event_family_allowed(eth_5m, allowed) is False
    # Empty allow-set trades everything (legacy behavior).
    assert executor._event_family_allowed(eth_5m, frozenset()) is True


def test_executor_starts_jsonl_cursor_at_tail(tmp_path: Path) -> None:
    queue = tmp_path / "signals.jsonl"
    queue.write_text("{}\n{}\n", encoding="utf-8")

    tail_cursor, tail_signature = executor._latest_signal_jsonl_cursor(queue, start="tail")
    assert tail_cursor == 2
    assert tail_signature
    assert executor._latest_signal_jsonl_cursor(queue, start="beginning") == (0, "")


def test_executor_resets_jsonl_cursor_after_queue_rotation(tmp_path: Path) -> None:
    queue = tmp_path / "signals.jsonl"
    queue.write_text(
        json.dumps(
            {
                "event_id": "pred-new-round",
                "ts": 1_000,
                "created_at": 2_000,
                "prob_up_15m": 0.91,
                "canonical_symbol": "BTC-15M:btc-updown-15m-1779775200:UP",
                "token_id": "token-up",
                "outcome_side": "UP",
                "round_slug": "btc-updown-15m-1779775200",
                "round_end_ts": 1_779_775_200_000,
                "market_implied_prob": 0.50,
                "token_probability": 0.91,
                "edge": 0.41,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events, cursor, signature = executor._read_signal_jsonl_after(
        queue,
        after_line_number=5,
        model_version="xgboost-v5",
        limit=10,
    )

    assert cursor == 1
    assert signature
    assert [event.event_id for event in events] == ["pred-new-round"]


def test_executor_exit_holds_when_orderbook_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(executor, "_now_ms", lambda: 10_000)
    log_path = tmp_path / "phase4.jsonl"
    signal = _signal(round_end_ts=300_000)
    position = _position()

    result = executor._maybe_exit(
        client=_UnavailableBookClient(),
        position_manager=object(),
        position=position,
        signal=signal,
        log_path=log_path,
        exit_edge_threshold=0.10,
        profit_target=0.15,
        sell_slippage=0.01,
    )

    assert result is None
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "exit_hold"
    assert row["reason"] == "orderbook_unavailable"
    assert row["token_id"] == "token-up"
    assert row["error_type"] == "RuntimeError"
    assert row["seconds_to_expiry"] == 290.0


def test_executor_shutdown_close_skips_unavailable_orderbook(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    position = _position(round_slug="round-without-timestamp")
    positions = {position.round_slug: position}

    closed, pending, settlement, pnl = executor._close_remaining_positions(
        client=_UnavailableBookClient(),
        position_manager=object(),
        positions=positions,
        log_path=log_path,
        sell_slippage=0.01,
    )

    assert closed == 0
    assert pending == 0
    assert settlement == 0
    assert pnl == 0.0
    assert position.round_slug in positions
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "shutdown_close_skipped"
    assert row["reason"] == "orderbook_unavailable"
    assert row["token_id"] == "token-up"


def test_executor_expired_exit_without_orderbook_marks_pending_settlement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor, "_now_ms", lambda: 200_000)
    log_path = tmp_path / "phase4.jsonl"
    signal = _signal(round_end_ts=100_000)
    position = _position()

    result = executor._maybe_exit(
        client=_UnavailableBookClient(),
        position_manager=object(),
        position=position,
        signal=signal,
        log_path=log_path,
        exit_edge_threshold=0.10,
        profit_target=0.15,
        sell_slippage=0.01,
    )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert result == executor.SellResult(status="pending_settlement")
    assert row["event"] == "exit_pending_settlement"
    assert row["reason"] == "expired_orderbook_unavailable"
    assert row["settlement_reconciliation_required"] is True
    assert row["position_assumed_closed_to_prevent_duplicate_sell"] is True
    assert row["seconds_to_expiry"] == -100.0


def test_executor_expired_exit_without_bid_marks_pending_settlement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor, "_now_ms", lambda: 200_000)
    log_path = tmp_path / "phase4.jsonl"
    signal = _signal(round_end_ts=100_000)
    position = _position()

    result = executor._maybe_exit(
        client=_MissingBidClient(),
        position_manager=object(),
        position=position,
        signal=signal,
        log_path=log_path,
        exit_edge_threshold=0.10,
        profit_target=0.15,
        sell_slippage=0.01,
    )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert result == executor.SellResult(status="pending_settlement")
    assert row["event"] == "exit_pending_settlement"
    assert row["reason"] == "expired_missing_bid"
    assert row["settlement_reconciliation_required"] is True


def test_executor_shutdown_expired_unavailable_orderbook_marks_pending_settlement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor, "_now_ms", lambda: 2_000_001)
    log_path = tmp_path / "phase4.jsonl"
    position = _position(round_slug="btc-updown-15m-1000")
    positions = {position.round_slug: position}

    closed, pending, settlement, pnl = executor._close_remaining_positions(
        client=_UnavailableBookClient(),
        position_manager=object(),
        positions=positions,
        log_path=log_path,
        sell_slippage=0.01,
    )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert closed == 0
    assert pending == 0
    assert settlement == 1
    assert pnl == 0.0
    assert positions == {}
    assert row["event"] == "exit_pending_settlement"
    assert row["reason"] == "shutdown_expired_orderbook_unavailable"
    assert row["seconds_to_expiry"] == -100.001


def test_executor_entry_time_window_prefers_no_new_entry_window() -> None:
    reason = executor._entry_time_window_skip_reason(
        240.0,
        no_new_entry_before_expiry_seconds=300.0,
        min_seconds_to_expiry=180.0,
        max_seconds_to_expiry=1200.0,
    )

    assert reason == "no_new_entry_window"


def test_executor_position_tick_hard_force_exits_without_new_signal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    now_ms = 1_810_000
    log_path = tmp_path / "phase4.jsonl"
    lifecycle = executor.RoundLifecycleState()
    position = _position(round_slug="btc-updown-15m-1000", size=1.960783)
    _store_open_position(lifecycle, position)
    client = _SellClient()
    manager = _PositionManager()

    closed, pending, settlement, pnl = executor._tick_open_positions(
        client=client,
        position_manager=manager,
        lifecycle=lifecycle,
        log_path=log_path,
        now_ms=now_ms,
        soft_force_exit_before_expiry_seconds=240.0,
        hard_force_exit_before_expiry_seconds=120.0,
        soft_force_exit_min_bid=0.15,
        exit_retry_seconds=10.0,
        max_exit_attempts_per_position=6,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    transition = next(row for row in rows if row["event"] == "position_lifecycle_transition")
    posted = next(row for row in rows if row["event"] == "exit_order_posted")
    filled = next(row for row in rows if row["event"] == "exit_filled")

    assert (closed, pending, settlement, pnl) == (1, 0, 0, 0.10)
    assert lifecycle.open_positions == {}
    assert transition["reason"] == "hard_force_exit"
    assert transition["lifecycle_state"] == "EXIT_PENDING"
    assert posted["reason"] == "hard_force_exit"
    assert posted["signal"] is None
    assert filled["reason"] == "hard_force_exit"
    assert client.created_orders == [{"token_id": "token-up", "side": "SELL", "amount": 1.96, "price": 0.59}]


def test_executor_position_tick_exits_under_min_fill_immediately(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    now_ms = 1_200_000
    log_path = tmp_path / "phase4.jsonl"
    lifecycle = executor.RoundLifecycleState()
    position = _position(round_slug="btc-updown-15m-1000", size=1.960783)
    position.lifecycle_state = "EXIT_REQUIRED"
    position.last_lifecycle_reason = "under_min_fill_exit"
    _store_open_position(lifecycle, position)
    client = _SellClient()

    closed, pending, settlement, pnl = executor._tick_open_positions(
        client=client,
        position_manager=_PositionManager(),
        lifecycle=lifecycle,
        log_path=log_path,
        now_ms=now_ms,
        soft_force_exit_before_expiry_seconds=240.0,
        hard_force_exit_before_expiry_seconds=120.0,
        soft_force_exit_min_bid=0.15,
        exit_retry_seconds=10.0,
        max_exit_attempts_per_position=6,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    transition = next(row for row in rows if row["event"] == "position_lifecycle_transition")
    filled = next(row for row in rows if row["event"] == "exit_filled")

    assert (closed, pending, settlement, pnl) == (1, 0, 0, 0.10)
    assert lifecycle.open_positions == {}
    assert transition["reason"] == "under_min_fill_exit"
    assert filled["reason"] == "under_min_fill_exit"


def test_executor_hard_force_exit_bypasses_soft_retry_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    now_ms = 1_810_000
    log_path = tmp_path / "phase4.jsonl"
    lifecycle = executor.RoundLifecycleState()
    position = _position(round_slug="btc-updown-15m-1000", size=1.960783)
    position.exit_attempt_count = 6
    position.last_exit_attempt_at = 1_700_000
    position.lifecycle_state = "EXIT_PENDING"
    position.last_lifecycle_reason = "soft_force_exit"
    _store_open_position(lifecycle, position)
    client = _SellClient()
    manager = _PositionManager()

    closed, pending, settlement, pnl = executor._tick_open_positions(
        client=client,
        position_manager=manager,
        lifecycle=lifecycle,
        log_path=log_path,
        now_ms=now_ms,
        soft_force_exit_before_expiry_seconds=240.0,
        hard_force_exit_before_expiry_seconds=120.0,
        soft_force_exit_min_bid=0.15,
        exit_retry_seconds=10.0,
        max_exit_attempts_per_position=6,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    transition = next(row for row in rows if row["event"] == "position_lifecycle_transition")
    filled = next(row for row in rows if row["event"] == "exit_filled")

    assert (closed, pending, settlement, pnl) == (1, 0, 0, 0.10)
    assert lifecycle.open_positions == {}
    assert transition["reason"] == "hard_force_exit"
    assert transition["lifecycle_state"] == "EXIT_PENDING"
    assert transition["exit_attempt_count"] == 7
    assert filled["reason"] == "hard_force_exit"
    assert client.created_orders == [{"token_id": "token-up", "side": "SELL", "amount": 1.96, "price": 0.59}]


def test_executor_position_tick_soft_force_exit_defers_weak_bid(
    tmp_path: Path,
) -> None:
    now_ms = 1_700_000
    log_path = tmp_path / "phase4.jsonl"
    lifecycle = executor.RoundLifecycleState()
    position = _position(round_slug="btc-updown-15m-1000")
    _store_open_position(lifecycle, position)

    result = executor._tick_open_positions(
        client=_WeakBidClient(),
        position_manager=_PositionManager(),
        lifecycle=lifecycle,
        log_path=log_path,
        now_ms=now_ms,
        soft_force_exit_before_expiry_seconds=240.0,
        hard_force_exit_before_expiry_seconds=120.0,
        soft_force_exit_min_bid=0.15,
        exit_retry_seconds=10.0,
        max_exit_attempts_per_position=6,
        sell_slippage=0.01,
    )

    hold = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert result == (0, 0, 0, 0.0)
    assert hold["event"] == "force_exit_hold"
    assert hold["reason"] == "soft_force_exit_bid_too_low"
    assert lifecycle.open_positions[executor._position_key(position.round_slug, position.sleeve)] == position


def test_executor_position_tick_soft_force_exit_retries_missing_bid(
    tmp_path: Path,
) -> None:
    now_ms = 1_700_000
    log_path = tmp_path / "phase4.jsonl"
    lifecycle = executor.RoundLifecycleState()
    position = _position(round_slug="btc-updown-15m-1000")
    _store_open_position(lifecycle, position)

    result = executor._tick_open_positions(
        client=_MissingBidClient(),
        position_manager=_PositionManager(),
        lifecycle=lifecycle,
        log_path=log_path,
        now_ms=now_ms,
        soft_force_exit_before_expiry_seconds=240.0,
        hard_force_exit_before_expiry_seconds=120.0,
        soft_force_exit_min_bid=0.15,
        exit_retry_seconds=10.0,
        max_exit_attempts_per_position=6,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    hold = rows[-1]

    assert result == (0, 0, 0, 0.0)
    assert position.lifecycle_state == "EXIT_PENDING"
    assert position.exit_attempt_count == 1
    assert hold["event"] == "force_exit_hold"
    assert hold["reason"] == "missing_bid"
    assert hold["exit_reason"] == "soft_force_exit"
    assert lifecycle.open_positions[executor._position_key(position.round_slug, position.sleeve)] == position


def test_executor_position_tick_expired_moves_to_settlement(
    tmp_path: Path,
) -> None:
    now_ms = 2_000_001
    log_path = tmp_path / "phase4.jsonl"
    lifecycle = executor.RoundLifecycleState()
    position = _position(round_slug="btc-updown-15m-1000")
    _store_open_position(lifecycle, position)

    result = executor._tick_open_positions(
        client=_MissingBidClient(),
        position_manager=_PositionManager(),
        lifecycle=lifecycle,
        log_path=log_path,
        now_ms=now_ms,
        soft_force_exit_before_expiry_seconds=240.0,
        hard_force_exit_before_expiry_seconds=120.0,
        soft_force_exit_min_bid=0.15,
        exit_retry_seconds=10.0,
        max_exit_attempts_per_position=6,
        sell_slippage=0.01,
    )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert result == (0, 0, 1, 0.0)
    assert lifecycle.open_positions == {}
    assert row["event"] == "exit_pending_settlement"
    assert row["reason"] == "expired_position_monitor"
    assert row["position"]["lifecycle_state"] == "AWAITING_SETTLEMENT"
    assert row["settlement_reconciliation_required"] is True


def test_executor_position_tick_resolves_expired_paper_settlement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now_ms = 2_000_001
    log_path = tmp_path / "phase4.jsonl"
    lifecycle = executor.RoundLifecycleState()
    position = _position(round_slug="btc-updown-15m-1000", sleeve="settlement")
    position.paper = True
    _store_open_position(lifecycle, position)
    manager = _SettlementPositionManager(realized_pnl=-1.0)

    monkeypatch.setattr(
        executor,
        "_fetch_paper_settlement_resolution",
        lambda _round_slug, *, config: executor.PaperSettlementResolution(
            result="DOWN",
            source="gamma_market",
            market={"slug": "btc-updown-15m-1000"},
        ),
    )

    result = executor._tick_open_positions(
        client=_MissingBidClient(),
        position_manager=manager,
        lifecycle=lifecycle,
        log_path=log_path,
        now_ms=now_ms,
        soft_force_exit_before_expiry_seconds=240.0,
        hard_force_exit_before_expiry_seconds=120.0,
        soft_force_exit_min_bid=0.15,
        exit_retry_seconds=10.0,
        max_exit_attempts_per_position=6,
        sell_slippage=0.01,
        paper_settlement_config=executor.PaperSettlementResolverConfig(enabled=True),
    )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert result == (0, 0, 0, -1.0)
    assert lifecycle.open_positions == {}
    assert manager.settled == [
        ("phase4-round-1-UP", "DOWN", 1_900_000),
    ]
    assert row["event"] == "paper_settlement_resolved"
    assert row["reason"] == "expired_position_monitor"
    assert row["settlement_result"] == "DOWN"
    assert row["realized_pnl"] == -1.0
    assert row["settlement_reconciliation_required"] is False


def test_executor_expired_paper_settlement_waits_when_unresolved_within_grace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now_ms = 2_000_001
    log_path = tmp_path / "phase4.jsonl"
    lifecycle = executor.RoundLifecycleState()
    position = _position(round_slug="btc-updown-15m-1000", sleeve="settlement")
    position.paper = True
    _store_open_position(lifecycle, position)
    manager = _SettlementPositionManager(realized_pnl=-1.0)

    monkeypatch.setattr(
        executor,
        "_fetch_paper_settlement_resolution",
        lambda _round_slug, *, config: executor.PaperSettlementResolution(
            result=None,
            source="gamma_market",
            error="market_not_resolved",
        ),
    )

    result = executor._tick_open_positions(
        client=_MissingBidClient(),
        position_manager=manager,
        lifecycle=lifecycle,
        log_path=log_path,
        now_ms=now_ms,
        soft_force_exit_before_expiry_seconds=240.0,
        hard_force_exit_before_expiry_seconds=120.0,
        soft_force_exit_min_bid=0.15,
        exit_retry_seconds=10.0,
        max_exit_attempts_per_position=6,
        sell_slippage=0.01,
        paper_settlement_config=executor.PaperSettlementResolverConfig(enabled=True),
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert result == (0, 0, 0, 0.0)
    assert lifecycle.open_positions[executor._position_key(position.round_slug, "settlement")] == position
    assert position.lifecycle_state == "AWAITING_SETTLEMENT"
    assert manager.settled == []
    assert rows[-2]["event"] == "paper_settlement_resolution_pending"
    assert rows[-2]["resolution_error"] == "market_not_resolved"
    assert rows[-1]["event"] == "paper_settlement_resolution_waiting"


def test_executor_expired_paper_settlement_falls_back_after_grace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now_ms = 2_000_001
    log_path = tmp_path / "phase4.jsonl"
    lifecycle = executor.RoundLifecycleState()
    position = _position(round_slug="btc-updown-15m-1000", sleeve="settlement")
    position.paper = True
    _store_open_position(lifecycle, position)
    manager = _SettlementPositionManager(realized_pnl=-1.0)

    monkeypatch.setattr(
        executor,
        "_fetch_paper_settlement_resolution",
        lambda _round_slug, *, config: executor.PaperSettlementResolution(
            result=None,
            source="gamma_market",
            error="market_not_resolved",
        ),
    )

    result = executor._tick_open_positions(
        client=_MissingBidClient(),
        position_manager=manager,
        lifecycle=lifecycle,
        log_path=log_path,
        now_ms=now_ms,
        soft_force_exit_before_expiry_seconds=240.0,
        hard_force_exit_before_expiry_seconds=120.0,
        soft_force_exit_min_bid=0.15,
        exit_retry_seconds=10.0,
        max_exit_attempts_per_position=6,
        sell_slippage=0.01,
        paper_settlement_config=executor.PaperSettlementResolverConfig(
            enabled=True,
            max_wait_after_expiry_seconds=0.0,
        ),
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert result == (0, 0, 1, 0.0)
    assert lifecycle.open_positions == {}
    assert manager.settled == []
    assert rows[-2]["event"] == "paper_settlement_resolution_pending"
    assert rows[-2]["resolution_error"] == "market_not_resolved"
    assert rows[-1]["event"] == "exit_pending_settlement"


def test_executor_gamma_market_winner_parses_outcome_price_strings() -> None:
    assert (
        executor._winning_outcome_from_gamma_market(
            {
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["0", "1"]',
            }
        )
        == "DOWN"
    )
    assert (
        executor._winning_outcome_from_gamma_market(
            {
                "outcomes": ["Up", "Down"],
                "outcomePrices": ["1", "0"],
            }
        )
        == "UP"
    )


def test_executor_gamma_event_payload_extracts_nested_market() -> None:
    payload = {
        "slug": "btc-updown-15m-1000",
        "markets": [
            {"slug": "other", "outcomes": '["Up", "Down"]'},
            {
                "slug": "btc-updown-15m-1000",
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["1", "0"]',
            },
        ],
    }

    market = executor._gamma_market_from_event_payload(payload, "btc-updown-15m-1000")

    assert market is not None
    assert market["slug"] == "btc-updown-15m-1000"
    assert executor._winning_outcome_from_gamma_market(market) == "UP"


def test_settlement_sleeve_holds_until_redeem_before_expiry(
    tmp_path: Path,
) -> None:
    now_ms = 1_700_000
    log_path = tmp_path / "phase4.jsonl"
    lifecycle = executor.RoundLifecycleState()
    position = _position(round_slug="btc-updown-15m-2000", sleeve="settlement")
    _store_open_position(lifecycle, position)

    result = executor._tick_open_positions(
        client=_SellClient(),
        position_manager=_PositionManager(),
        lifecycle=lifecycle,
        log_path=log_path,
        now_ms=now_ms,
        soft_force_exit_before_expiry_seconds=240.0,
        hard_force_exit_before_expiry_seconds=120.0,
        soft_force_exit_min_bid=0.15,
        exit_retry_seconds=10.0,
        max_exit_attempts_per_position=6,
        sell_slippage=0.01,
    )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert result == (0, 0, 0, 0.0)
    assert lifecycle.open_positions[executor._position_key(position.round_slug, position.sleeve)] == position
    assert row["event"] == "settlement_sleeve_hold"
    assert row["reason"] == "hold_to_settlement"
    assert row["position"]["sleeve"] == "settlement"


def test_round_lifecycle_only_confirmed_fill_locks_round() -> None:
    state = executor.RoundLifecycleState()
    signal = _signal()

    assert state.mark_event_seen(signal.event_id) is True
    assert state.mark_event_seen(signal.event_id) is False

    state.mark_entry_attempted(signal.event_id)
    state.mark_entry_result(signal, None)

    assert signal.event_id in state.attempted_entry_event_ids
    assert signal.round_slug not in state.filled_rounds
    assert signal.round_slug not in state.open_positions

    position = _position(sleeve="settlement")
    state.mark_entry_result(signal, position)

    assert signal.round_slug in state.filled_rounds
    assert state.position_event_ids == {position.event_id}
    assert state.open_positions[signal.round_slug] == position


def test_round_lifecycle_enforces_settlement_side_cap_after_confirmed_fill() -> None:
    state = executor.RoundLifecycleState()
    signal = _signal(side="DOWN")
    position = _position(sleeve="settlement")
    position.side = "DOWN"

    assert (
        executor._sleeve_side_cap_skip_reason(
            state,
            round_slug=signal.round_slug,
            sleeve="settlement",
            side="DOWN",
            max_filled_per_side_per_round=1,
        )
        is None
    )

    state.mark_entry_result(signal, position)

    assert state.filled_count_for_side(
        round_slug=signal.round_slug,
        sleeve="settlement",
        side="DOWN",
    ) == 1
    assert (
        executor._sleeve_side_cap_skip_reason(
            state,
            round_slug=signal.round_slug,
            sleeve="settlement",
            side="DOWN",
            max_filled_per_side_per_round=1,
        )
        == "settlement_side_cap"
    )
    assert (
        executor._sleeve_side_cap_skip_reason(
            state,
            round_slug=signal.round_slug,
            sleeve="settlement",
            side="UP",
            max_filled_per_side_per_round=1,
        )
        is None
    )


def test_volatility_same_side_reentry_is_budget_controlled_after_close() -> None:
    state = executor.RoundLifecycleState()
    budget = executor.VolatilitySleeveBudget(
        round_cap_usdc=1.0,
        per_bet_cap_usdc=1.0,
        min_order_size_usdc=0.05,
    )
    signal = _signal(side="DOWN")
    position = _position(sleeve="volatility")
    position.side = "DOWN"

    first = budget.next_entry_decision(signal.round_slug)
    assert first.allowed is True
    state.mark_entry_result(signal, position)
    assert state.has_open_sleeve(signal.round_slug, "volatility") is True

    state.mark_position_closed(signal.round_slug, "volatility")
    budget.apply_account_pnl(signal.round_slug, -0.20)

    repeat = budget.next_entry_decision(signal.round_slug)
    assert state.has_open_sleeve(signal.round_slug, "volatility") is False
    assert state.filled_count_for_side(
        round_slug=signal.round_slug,
        sleeve="volatility",
        side="DOWN",
    ) == 1
    assert repeat.allowed is True
    assert repeat.size_usdc == pytest.approx(0.80)


def test_executor_theoretical_pnl_scopes_to_current_run_positions(tmp_path: Path) -> None:
    manager = executor.PositionManager(tmp_path / "positions.duckdb")
    manager.open_position(
        "historical-winner",
        "BTC-15M:btc-updown-15m-1779805800:UP",
        "UP",
        0.04,
        25.0,
        "old-order",
    )
    manager.settle_position("historical-winner", "UP")
    manager.open_position(
        "current-loss",
        "BTC-15M:btc-updown-15m-1780239600:DOWN",
        "DOWN",
        0.50,
        2.0,
        "run-order-1",
    )
    manager.close_position("current-loss", 0.20)
    manager.open_position(
        "current-win",
        "BTC-15M:btc-updown-15m-1780240500:DOWN",
        "DOWN",
        0.40,
        2.5,
        "run-order-2",
    )
    manager.close_position("current-win", 0.65)

    all_history_pnl = executor._theoretical_pnl_from_positions(manager)
    current_run_pnl = executor._theoretical_pnl_from_positions(
        manager,
        event_ids={"current-loss", "current-win"},
    )

    assert round(all_history_pnl, 8) == 24.025
    assert round(current_run_pnl, 8) == 0.025


def test_round_lifecycle_caps_unique_observed_rounds() -> None:
    state = executor.RoundLifecycleState()

    assert state.mark_round_seen("round-1", max_rounds=2) is True
    assert state.mark_round_seen("round-1", max_rounds=2) is True
    assert state.mark_round_seen("round-2", max_rounds=2) is True

    assert state.max_rounds_reached(2) is True
    assert state.mark_round_seen("round-3", max_rounds=2) is False
    assert state.observed_rounds == ["round-1", "round-2"]
    assert state.observed_round_set == {"round-1", "round-2"}


def test_executor_fill_requires_confirmed_trade(monkeypatch) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    client = _TradeClient(
        [
            {
                "taker_order_id": "order-1",
                "side": "BUY",
                "status": "FAILED",
                "price": "0.50",
                "size": "2",
            },
            {
                "taker_order_id": "order-1",
                "side": "BUY",
                "status": "CONFIRMED",
                "price": "0.51",
                "size": "1.96",
            },
        ]
    )

    fill = executor._fill_for_order(client, "order-1", wanted_side="BUY")

    assert fill["status"] == "CONFIRMED"
    assert fill["price"] == "0.51"


def test_executor_fok_post_failure_does_not_create_position(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    signal = _signal()

    position = executor._try_entry(
        client=_BuyPostFailureClient(),
        position_manager=_PositionManager(),
        signal=signal,
        log_path=log_path,
        max_position_size_usdc=1.0,
        entry_policy=executor.Phase4EntryPolicy(min_entry_price=0.0, edge_threshold=0.45),
        seconds_to_expiry=600.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
    )

    assert position is None
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "entry_order_post_failed"
    assert row["error_type"] == "_FokKilledError"
    assert row["signal"]["event_id"] == signal.event_id
    assert row["worst_price"] == 0.52


def test_executor_skips_near_min_entry_without_strong_edge(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    signal = _signal()

    position = executor._try_entry(
        client=_NearMinAskClient(),
        position_manager=_PositionManager(),
        signal=signal,
        log_path=log_path,
        max_position_size_usdc=1.0,
        entry_policy=executor.Phase4EntryPolicy(
            min_entry_price=0.35,
            near_min_price_band=0.05,
            near_min_fresh_edge_threshold=0.50,
            near_min_seconds_to_expiry=420.0,
            edge_threshold=0.45,
        ),
        seconds_to_expiry=300.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
    )

    assert position is None
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "entry_skipped"
    assert row["reason"] == "near_min_entry_too_close_to_expiry"


def test_executor_skips_entry_below_min_price(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    signal = _signal()

    position = executor._try_entry(
        client=_CheapAskClient(),
        position_manager=_PositionManager(),
        signal=signal,
        log_path=log_path,
        max_position_size_usdc=1.0,
        entry_policy=executor.Phase4EntryPolicy(min_entry_price=0.30, edge_threshold=0.45),
        seconds_to_expiry=600.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
    )

    assert position is None
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "entry_skipped"
    assert row["reason"] == "entry_price_below_min"
    assert row["ask"] == 0.25
    assert row["worst_price"] == 0.27
    assert row["min_entry_price"] == 0.30


def test_executor_volatility_entry_requires_paper_mode(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"

    position = executor._try_entry(
        client=_VolatilityPaperClient(),
        position_manager=_OpenPositionManager(),
        signal=_signal(edge=0.10),
        log_path=log_path,
        max_position_size_usdc=1.0,
        entry_policy=executor.Phase4EntryPolicy(
            settlement_edge_threshold=0.45,
            volatility_min_entry_price=0.20,
            volatility_round_trip_cost=0.04,
            volatility_safety_margin=0.02,
        ),
        seconds_to_expiry=600.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
        sleeve="volatility",
        paper=False,
    )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert position is None
    assert row["event"] == "entry_skipped"
    assert row["reason"] == "volatility_live_disabled"
    assert row["gate_evaluation"]["expected_volatility_exit_gain"] == pytest.approx(0.18)


def test_v6_entry_policy_defaults_settlement_edge_to_cost_plus_margin() -> None:
    policy = executor._entry_policy_from_args(
        SimpleNamespace(
            entry_gate_mode="v6-joint",
            min_entry_price=0.35,
            near_min_price_band=0.05,
            near_min_fresh_edge_threshold=0.50,
            near_min_seconds_to_expiry=420.0,
            edge_threshold=0.45,
            settlement_edge_threshold=None,
            volatility_score_threshold=0.50,
            volatility_min_entry_price=0.20,
            volatility_min_seconds_to_expiry=420.0,
            volatility_round_trip_cost=0.04,
            volatility_safety_margin=0.02,
            enable_volatility_live_entries=False,
            v6_round_trip_cost=0.072,
            v6_ev_margin=0.01,
            v6_settlement_min_edge_after_cost=None,
        )
    )

    assert policy.effective_settlement_edge_threshold == pytest.approx(0.082)


def test_executor_volatility_paper_entry_creates_tagged_position(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    manager = _OpenPositionManager()

    position = executor._try_entry(
        client=_VolatilityPaperClient(),
        position_manager=manager,
        signal=_signal(edge=0.10),
        log_path=log_path,
        max_position_size_usdc=0.80,
        entry_policy=executor.Phase4EntryPolicy(
            settlement_edge_threshold=0.45,
            volatility_min_entry_price=0.20,
            volatility_round_trip_cost=0.04,
            volatility_safety_margin=0.02,
        ),
        seconds_to_expiry=600.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
        sleeve="volatility",
        paper=True,
    )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert position is not None
    assert position.paper is True
    assert position.sleeve == "volatility"
    assert position.fill_price == pytest.approx(0.52)
    assert manager.opened[0]["sleeve"] == "volatility"
    assert row["event"] == "paper_entry_filled"
    assert row["sleeve"] == "volatility"


def test_v6_joint_settlement_gate_logs_cost_edge_for_paper_fill(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    manager = _OpenPositionManager()

    position = executor._try_entry(
        client=_VolatilityPaperClient(),
        position_manager=manager,
        signal=_signal(
            edge=0.004,
            token_probability=0.81,
            p_up=0.81,
            p_down=0.15,
            p_neutral=0.04,
            p_vol_up=0.55,
            p_vol_down=0.45,
            v6_joint_side="UP",
        ),
        log_path=log_path,
        max_position_size_usdc=1.0,
        entry_policy=executor.Phase4EntryPolicy(
            settlement_edge_threshold=0.082,
            volatility_min_entry_price=0.20,
            volatility_round_trip_cost=0.04,
            volatility_safety_margin=0.02,
        ),
        seconds_to_expiry=600.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
        sleeve="settlement",
        paper=True,
        entry_gate_mode="v6-joint",
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    evaluated = next(row for row in rows if row["event"] == "entry_gate_evaluated")
    filled = next(row for row in rows if row["event"] == "paper_entry_filled")

    assert position is not None
    assert evaluated["raw_settlement_edge"] == pytest.approx(0.004)
    assert evaluated["fresh_edge_at_worst"] == pytest.approx(0.31)
    assert evaluated["gate_evaluation"]["settlement_edge"] == pytest.approx(0.31)
    assert evaluated["max_acceptable_price"] == pytest.approx(0.72)
    assert evaluated["order_limit_price"] == pytest.approx(0.52)
    assert evaluated["gate_evaluation"]["settlement_gate_passed"] is True
    assert position.fill_price == pytest.approx(0.50)
    assert filled["gate_evaluation"]["settlement_edge"] == pytest.approx(0.31)
    assert filled["gate_evaluation"]["settlement_gate_passed"] is True


def test_v6_joint_settlement_entry_ignores_legacy_min_entry_price(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    manager = _OpenPositionManager()

    position = executor._try_entry(
        client=_CheapAskClient(),
        position_manager=manager,
        signal=_signal(
            edge=0.49,
            token_probability=0.82,
            p_up=0.82,
            p_down=0.14,
            p_neutral=0.04,
            p_vol_up=0.10,
            p_vol_down=0.10,
            v6_joint_side="UP",
        ),
        log_path=log_path,
        max_position_size_usdc=1.0,
        entry_policy=executor.Phase4EntryPolicy(
            min_entry_price=0.35,
            settlement_edge_threshold=0.082,
            volatility_min_entry_price=0.20,
        ),
        seconds_to_expiry=600.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
        sleeve="settlement",
        paper=True,
        entry_gate_mode="v6-joint",
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    evaluated = next(row for row in rows if row["event"] == "entry_gate_evaluated")
    filled = next(row for row in rows if row["event"] == "paper_entry_filled")

    assert position is not None
    assert position.sleeve == "settlement"
    assert position.fill_price == pytest.approx(0.25)
    assert evaluated["worst_price"] == pytest.approx(0.25)
    assert evaluated["fresh_edge_at_worst"] == pytest.approx(0.57)
    assert evaluated["max_acceptable_price"] == pytest.approx(0.73)
    assert evaluated["order_limit_price"] == pytest.approx(0.27)
    assert filled["gate_evaluation"]["settlement_edge"] == pytest.approx(0.57)


def test_v6_joint_settlement_entry_skips_after_confidence_peak_drop(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"

    position = executor._try_entry(
        client=_VolatilityPaperClient(),
        position_manager=_OpenPositionManager(),
        signal=_signal(
            edge=0.16,
            token_probability=0.83,
            p_up=0.83,
            p_down=0.14,
            p_neutral=0.03,
            p_vol_up=0.10,
            p_vol_down=0.10,
            v6_joint_side="UP",
        ),
        log_path=log_path,
        max_position_size_usdc=1.0,
        entry_policy=executor.Phase4EntryPolicy(
            settlement_edge_threshold=0.082,
            settlement_min_confidence=0.80,
            settlement_peak_confidence_drop_tolerance=0.05,
        ),
        seconds_to_expiry=600.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
        sleeve="settlement",
        paper=True,
        entry_gate_mode="v6-joint",
        settlement_peak_confidence=0.94,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    skipped = rows[-1]

    assert position is None
    assert skipped["event"] == "entry_skipped"
    assert skipped["reason"] == "settlement_confidence_peak_drop"
    assert skipped["settlement_peak_confidence"] == pytest.approx(0.94)
    assert skipped["settlement_peak_confidence_drop_tolerance"] == pytest.approx(0.05)


def test_v7_pnl_entry_gate_uses_fresh_orderbook_edge(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"

    position = executor._try_entry(
        client=_V7ExpensiveAskClient(),
        position_manager=_OpenPositionManager(),
        signal=_signal(
            edge=0.20,
            token_probability=0.80,
            p_up=0.80,
            p_down=0.15,
            p_neutral=0.05,
            selected_side="UP",
            selected_expected_edge=0.20,
            entry_worst_price=0.60,
            should_enter_settlement=True,
        ),
        log_path=log_path,
        max_position_size_usdc=1.0,
        entry_policy=executor.Phase4EntryPolicy(
            settlement_edge_threshold=0.04,
            settlement_min_confidence=0.75,
        ),
        seconds_to_expiry=600.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
        sleeve="settlement",
        paper=True,
        entry_gate_mode="v7-pnl",
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    evaluated = next(row for row in rows if row["event"] == "entry_gate_evaluated")
    skipped = rows[-1]

    assert position is None
    assert evaluated["raw_settlement_edge"] == pytest.approx(0.20)
    assert evaluated["fresh_edge_at_worst"] == pytest.approx(0.02)
    assert evaluated["model_selected_expected_edge"] == pytest.approx(0.20)
    assert evaluated["model_entry_worst_price"] == pytest.approx(0.60)
    assert evaluated["entry_gate_mode"] == "v7-pnl"
    assert evaluated["gate_evaluation"]["settlement_edge"] == pytest.approx(0.02)
    assert skipped["event"] == "entry_skipped"
    assert skipped["reason"] == "fresh_edge_below_threshold"
    assert skipped["settlement_price_gate_mode"] == "v7_pnl_edge_only"


def test_v7_pnl_entry_gate_allows_fresh_positive_edge_paper_fill(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"

    position = executor._try_entry(
        client=_VolatilityPaperClient(),
        position_manager=_OpenPositionManager(),
        signal=_signal(
            edge=0.05,
            token_probability=0.80,
            p_up=0.80,
            p_down=0.15,
            p_neutral=0.05,
            selected_side="UP",
            selected_expected_edge=0.05,
            entry_worst_price=0.75,
            should_enter_settlement=True,
        ),
        log_path=log_path,
        max_position_size_usdc=1.0,
        entry_policy=executor.Phase4EntryPolicy(
            settlement_edge_threshold=0.04,
            settlement_min_confidence=0.75,
        ),
        seconds_to_expiry=600.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
        sleeve="settlement",
        paper=True,
        entry_gate_mode="v7-pnl",
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    evaluated = next(row for row in rows if row["event"] == "entry_gate_evaluated")
    filled = rows[-1]

    assert position is not None
    assert position.fill_price == pytest.approx(0.50)
    assert evaluated["fresh_edge_at_worst"] == pytest.approx(0.30)
    assert evaluated["gate_evaluation"]["settlement_gate_passed"] is True
    assert filled["event"] == "paper_entry_filled"


def test_executor_skips_complementary_entry_below_min_price(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    signal = _signal()

    position = executor._try_entry(
        client=_ComplementCheapClient(),
        position_manager=_OpenPositionManager(),
        signal=signal,
        log_path=log_path,
        max_position_size_usdc=1.0,
        entry_policy=executor.Phase4EntryPolicy(min_entry_price=0.35, edge_threshold=0.45),
        seconds_to_expiry=600.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
    )

    assert position is None
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "entry_skipped"
    assert row["reason"] == "complement_entry_price_below_min"
    assert row["complement_bid"] == 0.68
    assert row["complement_entry_price"] == 0.32
    assert row["opposite_token_id"] == "token-down"


def test_executor_under_min_actual_fill_requires_exit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    log_path = tmp_path / "phase4.jsonl"
    manager = _OpenPositionManager()

    position = executor._try_entry(
        client=_UnderMinFillClient(),
        position_manager=manager,
        signal=_signal(),
        log_path=log_path,
        max_position_size_usdc=1.0,
        entry_policy=executor.Phase4EntryPolicy(min_entry_price=0.35, edge_threshold=0.45),
        seconds_to_expiry=600.0,
        buy_slippage=0.02,
        monitoring_db_path=str(tmp_path / "catalog.duckdb"),
    )

    assert position is not None
    assert position.fill_price == 0.32
    assert position.lifecycle_state == "EXIT_REQUIRED"
    assert position.last_lifecycle_reason == "under_min_fill_exit"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    below_min = next(row for row in rows if row["event"] == "entry_fill_below_min")
    assert below_min["min_entry_price"] == 0.35
    assert below_min["exit_required"] is True
    assert manager.opened[0]["entry_price"] == 0.32


def test_executor_opposite_signal_can_exit_without_reverse_entry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(executor, "_now_ms", lambda: 10_000)
    log_path = tmp_path / "phase4.jsonl"
    position = _position(size=1.960783)
    signal = _signal(
        event_id="pred-strong-down",
        side="DOWN",
        token_id="token-down",
        edge=0.52,
        round_end_ts=300_000,
    )
    client = _SellClient()
    manager = _PositionManager()

    sell_result = executor._maybe_exit_opposite_correction(
        client=client,
        position_manager=manager,
        position=position,
        signal=signal,
        log_path=log_path,
        opposite_exit_edge_threshold=0.45,
        opposite_exit_min_seconds_to_expiry=120.0,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    posted = next(row for row in rows if row["event"] == "exit_order_posted")
    filled = next(row for row in rows if row["event"] == "exit_filled")

    assert sell_result == executor.SellResult(status="filled", realized_pnl=0.10)
    assert client.created_orders == [{"token_id": "token-up", "side": "SELL", "amount": 1.96, "price": 0.59}]
    assert manager.closed == [("phase4-round-1-UP", 0.60)]
    assert posted["reason"] == "opposite_side_exit_correction"
    assert posted["signal"]["event_id"] == "pred-strong-down"
    assert posted["sell_size"] == 1.96
    assert posted["dust_amount"] > 0
    assert filled["reason"] == "opposite_side_exit_correction"
    assert filled["account_cashflow_reconciliation_required"] is True


def test_settlement_reversal_exit_requires_hysteresis_then_sells(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(executor, "_now_ms", lambda: 10_000)
    log_path = tmp_path / "phase4.jsonl"
    position = _position(sleeve="settlement")
    position.paper = True
    position.entry_p_up = 0.91
    position.entry_p_down = 0.04
    signal = _signal(
        event_id="pred-strong-down",
        side="DOWN",
        token_id="token-down",
        token_probability=0.92,
        p_up=0.05,
        p_down=0.92,
        p_neutral=0.03,
        p_vol_up=0.10,
        p_vol_down=0.10,
        round_end_ts=300_000,
    )
    config = executor.SettlementExitConfig(
        allow_mid_round_exit=True,
        reversal_min_confidence=0.80,
        reversal_hysteresis_bars=2,
    )
    manager = _PositionManager()

    first = executor._maybe_settlement_signal_exit(
        client=_SellClient(),
        position_manager=manager,
        position=position,
        signal=signal,
        log_path=log_path,
        config=config,
        v6_joint_config=executor.V6JointGateConfig(settlement_threshold=0.50),
        signal_age_seconds=9.0,
        max_signal_age_seconds=180.0,
        seconds_to_expiry=290.0,
        opposite_exit_min_seconds_to_expiry=120.0,
        sell_slippage=0.01,
    )
    second = executor._maybe_settlement_signal_exit(
        client=_SellClient(),
        position_manager=manager,
        position=position,
        signal=signal,
        log_path=log_path,
        config=config,
        v6_joint_config=executor.V6JointGateConfig(settlement_threshold=0.50),
        signal_age_seconds=9.0,
        max_signal_age_seconds=180.0,
        seconds_to_expiry=290.0,
        opposite_exit_min_seconds_to_expiry=120.0,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert first is None
    assert second == (
        "settlement_reversal_exit",
        executor.SellResult(status="filled", realized_pnl=0.10, account_cash_pnl=0.10),
    )
    assert manager.closed == [("phase4-round-1-UP", 0.59)]
    assert any(
        row["event"] == "settlement_reversal_exit_skipped"
        and row["reason"] == "hysteresis_wait"
        for row in rows
    )
    assert rows[-1]["event"] == "settlement_reversal_exit_filled"


def test_settlement_price_stop_tracks_reversal_when_mid_round_exit_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(executor, "_now_ms", lambda: 10_000)
    log_path = tmp_path / "phase4.jsonl"
    position = _position(sleeve="settlement")
    position.paper = True
    signal = _signal(
        event_id="pred-strong-down",
        side="DOWN",
        token_id="token-down",
        token_probability=0.91,
        p_up=0.06,
        p_down=0.91,
        p_neutral=0.03,
        p_vol_up=0.10,
        p_vol_down=0.10,
        round_end_ts=300_000,
    )

    result = executor._maybe_settlement_signal_exit(
        client=_SellClient(),
        position_manager=_PositionManager(),
        position=position,
        signal=signal,
        log_path=log_path,
        config=executor.SettlementExitConfig(
            allow_mid_round_exit=False,
            price_stop_enabled=True,
            reversal_min_confidence=0.75,
            reversal_hysteresis_bars=2,
        ),
        v6_joint_config=executor.V6JointGateConfig(settlement_threshold=0.50),
        signal_age_seconds=9.0,
        max_signal_age_seconds=180.0,
        seconds_to_expiry=290.0,
        opposite_exit_min_seconds_to_expiry=120.0,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert result is None
    assert position.settlement_reversal_candidate_side == "DOWN"
    assert position.settlement_reversal_candidate_count == 1
    assert rows[-1]["event"] == "settlement_reversal_exit_skipped"
    assert rows[-1]["reason"] == "hysteresis_wait"


def test_settlement_confidence_decay_exit_sells_on_fresh_regime_shift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(executor, "_now_ms", lambda: 10_000)
    log_path = tmp_path / "phase4.jsonl"
    position = _position(sleeve="settlement")
    position.paper = True
    position.entry_p_up = 0.90
    position.entry_p_down = 0.05
    signal = _signal(
        event_id="pred-decay-down",
        side="DOWN",
        token_id="token-down",
        token_probability=0.82,
        p_up=0.16,
        p_down=0.82,
        p_neutral=0.02,
        p_vol_up=0.10,
        p_vol_down=0.10,
        round_end_ts=300_000,
    )

    result = executor._maybe_settlement_signal_exit(
        client=_SellClient(),
        position_manager=_PositionManager(),
        position=position,
        signal=signal,
        log_path=log_path,
        config=executor.SettlementExitConfig(
            confidence_decay_enabled=True,
            decay_hysteresis_bars=1,
        ),
        v6_joint_config=executor.V6JointGateConfig(settlement_threshold=0.50),
        signal_age_seconds=9.0,
        max_signal_age_seconds=180.0,
        seconds_to_expiry=290.0,
        opposite_exit_min_seconds_to_expiry=120.0,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert result == (
        "settlement_confidence_decay_exit",
        executor.SellResult(status="filled", realized_pnl=0.10, account_cash_pnl=0.10),
    )
    evaluated = next(
        row for row in rows if row["event"] == "settlement_confidence_decay_exit_evaluated"
    )
    assert evaluated["below_floor"] is True
    assert evaluated["below_delta"] is True
    assert evaluated["regime_shift"] is True
    assert evaluated["opposite_confidence_passed"] is True
    assert rows[-1]["event"] == "settlement_confidence_decay_exit_filled"


def test_settlement_confidence_decay_exit_ignores_weak_regime_shift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(executor, "_now_ms", lambda: 10_000)
    log_path = tmp_path / "phase4.jsonl"
    position = _position(sleeve="settlement")
    position.paper = True
    position.entry_p_up = 0.90
    position.entry_p_down = 0.05
    signal = _signal(
        event_id="pred-weak-down",
        side="DOWN",
        token_id="token-down",
        token_probability=0.58,
        p_up=0.40,
        p_down=0.58,
        p_neutral=0.02,
        p_vol_up=0.10,
        p_vol_down=0.10,
        round_end_ts=300_000,
    )

    result = executor._maybe_settlement_signal_exit(
        client=_SellClient(),
        position_manager=_PositionManager(),
        position=position,
        signal=signal,
        log_path=log_path,
        config=executor.SettlementExitConfig(confidence_decay_enabled=True),
        v6_joint_config=executor.V6JointGateConfig(settlement_threshold=0.50),
        signal_age_seconds=9.0,
        max_signal_age_seconds=180.0,
        seconds_to_expiry=290.0,
        opposite_exit_min_seconds_to_expiry=120.0,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    evaluated = next(
        row for row in rows if row["event"] == "settlement_confidence_decay_exit_evaluated"
    )

    assert result is None
    assert evaluated["below_floor"] is True
    assert evaluated["below_delta"] is True
    assert evaluated["regime_shift"] is True
    assert evaluated["opposite_confidence_passed"] is False
    assert evaluated["should_exit"] is False


def test_settlement_price_stop_waits_for_reversal_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    log_path = tmp_path / "phase4.jsonl"
    position = _position(sleeve="settlement")
    position.paper = True
    position.entry_price = 0.75
    position.fill_price = 0.75

    result = executor._maybe_settlement_price_stop_exit(
        client=_SellClient(),
        position_manager=_PositionManager(),
        position=position,
        log_path=log_path,
        seconds_to_expiry=300.0,
        config=executor.SettlementExitConfig(
            price_stop_enabled=True,
            stop_price_delta=0.10,
            stop_loss_usdc=0.50,
            stop_min_seconds_to_expiry=120.0,
        ),
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert result is None
    evaluated = next(row for row in rows if row["event"] == "settlement_stop_exit_evaluated")
    assert evaluated["price_breach"] is True
    assert evaluated["reversal_confirmation"]["confirmed"] is False
    assert rows[-1]["event"] == "settlement_stop_exit_skipped"
    assert rows[-1]["reason"] == "reversal_confirmation_required"


def test_settlement_price_stop_exit_sells_after_reversal_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    log_path = tmp_path / "phase4.jsonl"
    position = _position(sleeve="settlement")
    position.paper = True
    position.entry_price = 0.75
    position.fill_price = 0.75
    position.settlement_reversal_candidate_side = "DOWN"
    position.settlement_reversal_candidate_count = 2

    result = executor._maybe_settlement_price_stop_exit(
        client=_SellClient(),
        position_manager=_PositionManager(),
        position=position,
        log_path=log_path,
        seconds_to_expiry=300.0,
        config=executor.SettlementExitConfig(
            price_stop_enabled=True,
            stop_price_delta=0.10,
            stop_loss_usdc=0.50,
            stop_min_seconds_to_expiry=120.0,
            reversal_hysteresis_bars=2,
        ),
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert result == executor.SellResult(status="filled", realized_pnl=0.10, account_cash_pnl=0.10)
    evaluated = next(row for row in rows if row["event"] == "settlement_stop_exit_evaluated")
    assert evaluated["price_breach"] is True
    assert evaluated["reversal_confirmation"]["confirmed"] is True
    assert rows[-1]["event"] == "settlement_stop_exit_filled"


def test_settlement_price_stop_same_side_confirmation_veto_holds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(executor, "_now_ms", lambda: 5_000)
    log_path = tmp_path / "phase4.jsonl"
    position = _position(sleeve="settlement")
    position.paper = True
    position.entry_price = 0.75
    position.fill_price = 0.75
    position.settlement_reversal_candidate_side = "DOWN"
    position.settlement_reversal_candidate_count = 2
    signal = _signal(
        event_id="pred-confirm-up",
        side="UP",
        token_id="token-up",
        token_probability=0.86,
        p_up=0.86,
        p_down=0.10,
        p_neutral=0.04,
        p_vol_up=0.10,
        p_vol_down=0.10,
        created_at=3_000,
    )
    config = executor.SettlementExitConfig(
        price_stop_enabled=True,
        stop_price_delta=0.10,
        stop_loss_usdc=0.50,
        stop_min_seconds_to_expiry=120.0,
        price_stop_same_side_confirmation_veto_enabled=True,
        price_stop_same_side_confirmation_min_confidence=0.80,
        price_stop_same_side_confirmation_max_age_seconds=180.0,
    )
    client = _SellClient()
    manager = _PositionManager()

    recorded = executor._record_settlement_same_side_confirmation(
        position=position,
        signal=signal,
        log_path=log_path,
        config=config,
        signal_age_seconds=2.0,
        max_signal_age_seconds=180.0,
    )
    result = executor._maybe_settlement_price_stop_exit(
        client=client,
        position_manager=manager,
        position=position,
        log_path=log_path,
        seconds_to_expiry=300.0,
        config=config,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    skipped = next(
        row
        for row in rows
        if row["event"] == "settlement_stop_exit_skipped"
        and row["reason"] == "same_side_confirmation_veto"
    )
    assert recorded is True
    assert result is None
    assert manager.closed == []
    assert client.created_orders == []
    assert position.settlement_same_side_confirmation_event_id == "pred-confirm-up"
    assert skipped["same_side_confirmation"]["confidence"] == 0.86


def test_settlement_price_stop_same_side_confirmation_veto_expires(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(executor, "_now_ms", lambda: 10_000)
    log_path = tmp_path / "phase4.jsonl"
    position = _position(sleeve="settlement")
    position.paper = True
    position.entry_price = 0.75
    position.fill_price = 0.75
    position.settlement_reversal_candidate_side = "DOWN"
    position.settlement_reversal_candidate_count = 2
    signal = _signal(
        event_id="pred-confirm-up",
        side="UP",
        token_id="token-up",
        token_probability=0.86,
        p_up=0.86,
        p_down=0.10,
        p_neutral=0.04,
        p_vol_up=0.10,
        p_vol_down=0.10,
        created_at=3_000,
    )
    config = executor.SettlementExitConfig(
        price_stop_enabled=True,
        stop_price_delta=0.10,
        stop_loss_usdc=0.50,
        stop_min_seconds_to_expiry=120.0,
        price_stop_same_side_confirmation_veto_enabled=True,
        price_stop_same_side_confirmation_min_confidence=0.80,
        price_stop_same_side_confirmation_max_age_seconds=1.0,
    )
    manager = _PositionManager()

    recorded = executor._record_settlement_same_side_confirmation(
        position=position,
        signal=signal,
        log_path=log_path,
        config=config,
        signal_age_seconds=2.0,
        max_signal_age_seconds=180.0,
    )
    result = executor._maybe_settlement_price_stop_exit(
        client=_SellClient(),
        position_manager=manager,
        position=position,
        log_path=log_path,
        seconds_to_expiry=300.0,
        config=config,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert recorded is True
    assert result == executor.SellResult(status="filled", realized_pnl=0.10, account_cash_pnl=0.10)
    assert manager.closed == [("phase4-round-1-UP", 0.59)]
    assert rows[-1]["event"] == "settlement_stop_exit_filled"


def test_executor_matched_sell_without_trade_confirmation_is_pending(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    log_path = tmp_path / "phase4.jsonl"
    position = _position()
    signal = _signal(edge=-0.31)
    manager = _PositionManager()

    sell_result = executor._maybe_exit(
        client=_SellPendingClient(),
        position_manager=manager,
        position=position,
        signal=signal,
        log_path=log_path,
        exit_edge_threshold=0.10,
        profit_target=0.15,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    pending = next(row for row in rows if row["event"] == "exit_pending_confirmation")

    assert sell_result == executor.SellResult(status="pending_confirmation")
    assert manager.closed == []
    assert pending["position_assumed_closed_to_prevent_duplicate_sell"] is True
    assert pending["account_cashflow_reconciliation_required"] is True


def test_executor_retries_matched_sell_until_trade_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    log_path = tmp_path / "phase4.jsonl"
    position = _position()
    signal = _signal(edge=-0.31)
    manager = _PositionManager()
    client = _SellDelayedConfirmationClient()

    sell_result = executor._maybe_exit(
        client=client,
        position_manager=manager,
        position=position,
        signal=signal,
        log_path=log_path,
        exit_edge_threshold=0.10,
        profit_target=0.15,
        sell_slippage=0.01,
        exit_order_timeout_seconds=20.0,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    filled = next(row for row in rows if row["event"] == "exit_filled")

    assert sell_result == executor.SellResult(status="filled", realized_pnl=0.10)
    assert manager.closed == [("phase4-round-1-UP", 0.82)]
    assert client.trade_calls == 2
    assert filled["sell_order_id"] == "order-sell"
    assert filled["exit_price"] == 0.82


def test_executor_sell_result_prefers_cash_leg_account_pnl_for_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    db_path = tmp_path / "catalog.duckdb"
    log_path = tmp_path / "phase4.jsonl"
    position = _position()
    manager = executor.PositionManager(db_path)
    manager.open_position(
        position.event_id,
        "BTC-15M:round-1:UP",
        "UP",
        0.50,
        2.0,
        "order-buy",
        fill_price=0.50,
        sleeve=position.sleeve,
    )
    executor._persist_cash_leg(
        monitoring_db_path=str(db_path),
        event_id=position.event_id,
        round_slug=position.round_slug,
        action="BUY",
        fill={
            "price": "0.50",
            "size": "2.0",
            "usdcAmount": "1.05",
            "timestamp": 1,
        },
        order_id="order-buy",
        sleeve=position.sleeve,
    )

    sell_result = executor._sell_position(
        client=_SellDelayedConfirmationClient(),
        position_manager=manager,
        position=position,
        log_path=log_path,
        bid=0.82,
        sell_slippage=0.01,
        reason="test_budget_account_pnl",
        signal=None,
        monitoring_db_path=str(db_path),
    )

    filled = next(
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "exit_filled"
    )

    assert sell_result.realized_pnl == pytest.approx(0.64)
    assert sell_result.account_cash_pnl == pytest.approx(0.59)
    assert executor._cash_pnl_for_budget(sell_result) == pytest.approx(0.59)
    assert filled["account_cash_pnl"] == pytest.approx(0.59)


def test_executor_sell_post_failure_does_not_raise(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    log_path = tmp_path / "phase4.jsonl"
    position = _position()
    signal = _signal(edge=-0.31)

    sell_result = executor._maybe_exit(
        client=_SellPostFailureClient(),
        position_manager=_PositionManager(),
        position=position,
        signal=signal,
        log_path=log_path,
        exit_edge_threshold=0.10,
        profit_target=0.15,
        sell_slippage=0.01,
    )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert sell_result is None
    assert row["event"] == "exit_order_post_failed"
    assert row["error_type"] == "_FokKilledError"


def test_executor_opposite_signal_holds_near_expiry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor, "_now_ms", lambda: 10_000)
    log_path = tmp_path / "phase4.jsonl"
    position = _position()
    signal = _signal(
        event_id="pred-strong-down",
        side="DOWN",
        token_id="token-down",
        edge=0.55,
        round_end_ts=100_000,
    )

    pnl = executor._maybe_exit_opposite_correction(
        client=_SellClient(),
        position_manager=_PositionManager(),
        position=position,
        signal=signal,
        log_path=log_path,
        opposite_exit_edge_threshold=0.45,
        opposite_exit_min_seconds_to_expiry=120.0,
        sell_slippage=0.01,
    )

    assert pnl is None
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "opposite_exit_hold"
    assert row["reason"] == "insufficient_time_remaining"
    assert row["seconds_to_expiry"] == 90.0


class _UnavailableBookClient:
    def get_order_book(self, token_id: str):  # noqa: ANN201
        raise RuntimeError(f"No orderbook exists for the requested token id: {token_id}")


class _RestBookResponse:
    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "bids": [{"price": "0.44"}, {"price": "0.48"}],
                "asks": [{"price": "0.53"}, {"price": "0.51"}],
            }
        ).encode("utf-8")


def test_best_bid_ask_falls_back_to_rest_book(monkeypatch) -> None:
    requests = []

    def fake_urlopen(req, *, timeout):  # noqa: ANN001, ANN202
        requests.append((req.full_url, timeout, dict(req.header_items())))
        return _RestBookResponse()

    monkeypatch.setenv("POLYMARKET_ORDERBOOK_REST_FALLBACK", "true")
    monkeypatch.setenv("POLYMARKET_ORDERBOOK_REST_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("POLYMARKET_HOST", "https://clob.polymarket.test")
    monkeypatch.setattr(executor.request, "urlopen", fake_urlopen)

    assert executor._best_bid_ask(_UnavailableBookClient(), "token-up") == (0.48, 0.51)
    assert len(requests) == 1
    url, timeout, headers = requests[0]
    assert url == "https://clob.polymarket.test/book?token_id=token-up"
    assert timeout == 1.5
    assert headers["Accept"] == "application/json"


class _WeakBidClient:
    def get_order_book(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return {"bids": [{"price": "0.10"}], "asks": [{"price": "0.61"}]}


class _MissingBidClient:
    def get_order_book(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return {"bids": [], "asks": [{"price": "0.61"}]}


class _TradeClient:
    def __init__(self, trades):
        self.trades = trades

    def get_trades(self):  # noqa: ANN201
        return self.trades


class _FokKilledError(Exception):
    pass


class _BuyPostFailureClient:
    def get_order_book(self, token_id: str):  # noqa: ANN201
        if token_id == "token-up":
            return {"bids": [{"price": "0.49"}], "asks": [{"price": "0.50"}]}
        assert token_id == "token-down"
        return {"bids": [{"price": "0.50"}], "asks": [{"price": "0.51"}]}

    def get_tick_size(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return "0.01"

    def get_neg_risk(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return False

    def create_market_order(self, *, order_args, options):  # noqa: ANN001, ANN201, ARG002
        assert order_args.token_id == "token-up"
        assert order_args.amount == 1.0
        assert order_args.price == 0.52
        return {"order": "signed"}

    def post_order(self, order, order_type):  # noqa: ANN001, ANN201, ARG002
        raise _FokKilledError("order could not be fully filled")


class _CheapAskClient:
    def get_order_book(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return {"bids": [{"price": "0.24"}], "asks": [{"price": "0.25"}]}

    def get_tick_size(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return "0.01"

    def get_neg_risk(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return False


class _NearMinAskClient(_CheapAskClient):
    def get_order_book(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return {"bids": [{"price": "0.35"}], "asks": [{"price": "0.36"}]}


class _VolatilityPaperClient:
    def get_order_book(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return {"bids": [{"price": "0.70"}], "asks": [{"price": "0.50"}]}

    def get_tick_size(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return "0.01"

    def get_neg_risk(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return False


class _V7ExpensiveAskClient:
    def get_order_book(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return {"bids": [{"price": "0.76"}], "asks": [{"price": "0.78"}]}

    def get_tick_size(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return "0.01"

    def get_neg_risk(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return False


class _ComplementCheapClient:
    def get_order_book(self, token_id: str):  # noqa: ANN201
        if token_id == "token-up":
            return {"bids": [{"price": "0.38"}], "asks": [{"price": "0.39"}]}
        assert token_id == "token-down"
        return {"bids": [{"price": "0.68"}], "asks": [{"price": "0.69"}]}

    def get_tick_size(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return "0.01"

    def get_neg_risk(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return True


class _UnderMinFillClient(_ComplementCheapClient):
    def get_order_book(self, token_id: str):  # noqa: ANN201
        if token_id == "token-up":
            return {"bids": [{"price": "0.38"}], "asks": [{"price": "0.39"}]}
        assert token_id == "token-down"
        return {"bids": [{"price": "0.60"}], "asks": [{"price": "0.61"}]}

    def create_market_order(self, *, order_args, options):  # noqa: ANN001, ANN201, ARG002
        assert order_args.token_id == "token-up"
        assert order_args.price == 0.41
        return {"order": "signed"}

    def post_order(self, order, order_type):  # noqa: ANN001, ANN201, ARG002
        return {
            "success": True,
            "status": "matched",
            "orderID": "order-buy",
            "takingAmount": "3.125",
            "transactionsHashes": ["0xhash"],
        }

    def get_trades(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return [
            {
                "taker_order_id": "order-buy",
                "transaction_hash": "0xhash",
                "side": "BUY",
                "status": "MINED",
                "price": "0.32",
                "size": "3.125",
            }
        ]


class _SellClient:
    def __init__(self) -> None:
        self.created_orders = []

    def get_order_book(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return {"bids": [{"price": "0.60"}], "asks": [{"price": "0.61"}]}

    def get_tick_size(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return "0.01"

    def get_neg_risk(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return False

    def create_market_order(self, *, order_args, options):  # noqa: ANN001, ANN201, ARG002
        self.created_orders.append(
            {
                "token_id": order_args.token_id,
                "side": order_args.side,
                "amount": order_args.amount,
                "price": order_args.price,
            }
        )
        return {"order": "signed"}

    def post_order(self, order, order_type):  # noqa: ANN001, ANN201, ARG002
        return {"success": True, "status": "matched", "orderID": "order-sell"}

    def get_trades(self):  # noqa: ANN201
        return [
            {
                "taker_order_id": "order-sell",
                "side": "SELL",
                "status": "MINED",
                "price": "0.60",
                "size": "1.96",
            }
        ]


class _SellPendingClient(_SellClient):
    def get_trades(self):  # noqa: ANN201
        return []


class _SellDelayedConfirmationClient(_SellClient):
    def __init__(self) -> None:
        super().__init__()
        self.trade_calls = 0

    def get_trades(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.trade_calls += 1
        if self.trade_calls == 1:
            return []
        return [
            {
                "taker_order_id": "order-sell",
                "transaction_hash": "0xsell",
                "side": "SELL",
                "status": "MINED",
                "price": "0.82",
                "size": "2.00",
            }
        ]


class _SellPostFailureClient(_SellClient):
    def post_order(self, order, order_type):  # noqa: ANN001, ANN201, ARG002
        raise _FokKilledError("not enough balance / allowance")


class _PositionManager:
    def __init__(self) -> None:
        self.closed = []

    def close_position(self, event_id: str, exit_price: float):  # noqa: ANN201
        self.closed.append((event_id, exit_price))
        return SimpleNamespace(realized_pnl=0.10)


class _SettlementPositionManager(_PositionManager):
    def __init__(self, *, realized_pnl: float) -> None:
        super().__init__()
        self.realized_pnl = realized_pnl
        self.settled = []

    def settle_position(self, event_id: str, result: str, settlement_time=None):  # noqa: ANN001, ANN201
        self.settled.append((event_id, result, settlement_time))
        return SimpleNamespace(
            realized_pnl=self.realized_pnl,
            settlement_result=result,
            exit_price=1.0 if result == "UP" else 0.0,
        )


class _OpenPositionManager(_PositionManager):
    def __init__(self) -> None:
        super().__init__()
        self.opened = []

    def open_position(self, **kwargs):  # noqa: ANN003, ANN201
        self.opened.append(kwargs)
        return SimpleNamespace(**kwargs)


def _signal(
    *,
    event_id: str = "pred-1",
    side: str = "UP",
    token_id: str = "token-up",
    opposite_token_id: str = "token-down",
    edge: float = 0.51,
    token_probability: float = 0.98,
    p_up: float | None = None,
    p_down: float | None = None,
    p_neutral: float | None = None,
    p_vol_up: float | None = None,
    p_vol_down: float | None = None,
    v6_joint_side: str | None = None,
    selected_side: str | None = None,
    selected_expected_edge: float | None = None,
    entry_worst_price: float | None = None,
    should_enter_settlement: bool | None = None,
    round_end_ts: int = 1_779_775_200_000,
    created_at: int = 2_000,
):
    return executor.SignalEvent(
        event_id=event_id,
        ts=1_000,
        created_at=created_at,
        prob_up_15m=0.98,
        canonical_symbol=f"BTC-15M:round-1:{side}",
        token_id=token_id,
        outcome_side=side,
        round_slug="round-1",
        round_end_ts=round_end_ts,
        market_implied_prob=0.47,
        token_probability=token_probability,
        edge=edge,
        bridged_at=3_000,
        opposite_token_id=opposite_token_id,
        p_up=p_up,
        p_down=p_down,
        p_neutral=p_neutral,
        p_vol_up=p_vol_up,
        p_vol_down=p_vol_down,
        v6_joint_side=v6_joint_side,
        selected_side=selected_side,
        selected_expected_edge=selected_expected_edge,
        entry_worst_price=entry_worst_price,
        should_enter_settlement=should_enter_settlement,
    )


def _position(*, size: float = 2.0, round_slug: str = "round-1", sleeve: str = "volatility"):
    return executor.LivePosition(
        event_id="phase4-round-1-UP",
        round_slug=round_slug,
        side="UP",
        token_id="token-up",
        entry_price=0.50,
        fill_price=0.50,
        size=size,
        order_id="order-1",
        opened_at=4_000,
        entry_signal_event_id="pred-1",
        entry_signal_ts=1_000,
        entry_signal_created_at=2_000,
        entry_signal_bridged_at=3_000,
        entry_order_posted_at=3_500,
        sleeve=sleeve,
    )


def _store_open_position(
    lifecycle: executor.RoundLifecycleState,
    position: executor.LivePosition,
) -> None:
    lifecycle.open_positions[executor._position_key(position.round_slug, position.sleeve)] = position
