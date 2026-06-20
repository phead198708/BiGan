from __future__ import annotations

import json
from pathlib import Path

import pytest

import bigan.execution.signal_queue as signal_queue
from bigan.execution.signal_queue import (
    JsonlSignalSource,
    KafkaSignalSink,
    KafkaSignalSource,
    SignalCursor,
    append_event_driven_v7_signals_from_raw_queue,
    append_prediction_rows_as_signal_jsonl,
    append_prediction_rows_to_signal_sink,
    diagnose_prediction_rows_for_signal_bridge,
)
from bigan.execution.v6_gate import V6JointGateConfig
from bigan.features.low_latency import JsonlRawQueue


def test_append_prediction_rows_as_signal_jsonl_emits_executor_ready_v6_signal(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "signals.jsonl"
    rows = [
        _prediction_row(
            side="UP",
            token_id="token-up",
            market_implied_prob=0.30,
            p_up=0.90,
            p_down=0.05,
            p_neutral=0.05,
            p_vol_up=0.20,
            p_vol_down=0.10,
        ),
        _prediction_row(
            side="DOWN",
            token_id="token-down",
            market_implied_prob=0.70,
            p_up=0.45,
            p_down=0.45,
            p_neutral=0.10,
            p_vol_up=0.20,
            p_vol_down=0.10,
        ),
    ]

    written = append_prediction_rows_as_signal_jsonl(
        queue_path,
        rows,
        model_version="xgboost-v6",
        v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
        bridged_at=1_779_774_410_000,
    )

    assert written == 1
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["model_version"] == "xgboost-v6"
    assert payload["outcome_side"] == "UP"
    assert payload["token_id"] == "token-up"
    assert payload["opposite_token_id"] == "token-down"
    assert payload["p_up"] == 0.90
    assert payload["v6_joint_side"] == "UP"


def test_append_prediction_rows_as_signal_jsonl_rotates_to_current_round(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "signals.jsonl"
    old_round = "btc-updown-15m-1779774300"
    new_round = "btc-updown-15m-1779775200"

    assert (
        append_prediction_rows_as_signal_jsonl(
            queue_path,
            [
                _prediction_row(
                    side="UP",
                    token_id="old-up",
                    market_implied_prob=0.30,
                    p_up=0.90,
                    p_down=0.05,
                    p_neutral=0.05,
                    p_vol_up=0.20,
                    p_vol_down=0.10,
                    round_slug=old_round,
                    ts=1_779_774_400_000,
                )
            ],
            model_version="xgboost-v6",
            v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
            bridged_at=1_779_774_410_000,
        )
        == 1
    )
    assert (
        append_prediction_rows_as_signal_jsonl(
            queue_path,
            [
                _prediction_row(
                    side="UP",
                    token_id="old-up",
                    market_implied_prob=0.30,
                    p_up=0.90,
                    p_down=0.05,
                    p_neutral=0.05,
                    p_vol_up=0.20,
                    p_vol_down=0.10,
                    round_slug=old_round,
                    ts=1_779_774_400_000,
                ),
                _prediction_row(
                    side="UP",
                    token_id="new-up",
                    market_implied_prob=0.30,
                    p_up=0.91,
                    p_down=0.04,
                    p_neutral=0.05,
                    p_vol_up=0.20,
                    p_vol_down=0.10,
                    round_slug=new_round,
                    ts=1_779_775_300_000,
                ),
            ],
            model_version="xgboost-v6",
            v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
            bridged_at=1_779_775_310_000,
        )
        == 1
    )

    payloads = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [payload["round_slug"] for payload in payloads] == [new_round]
    assert payloads[0]["outcome_side"] == "UP"


def test_append_prediction_rows_as_signal_jsonl_prefers_nearest_active_round(
    tmp_path: Path,
    monkeypatch,
) -> None:
    queue_path = tmp_path / "signals.jsonl"
    near_round = "btc-updown-15m-1779774300"
    far_round = "btc-updown-15m-1779782400"
    monkeypatch.setattr(signal_queue, "_now_ms", lambda: 1_779_774_410_000)

    written = append_prediction_rows_as_signal_jsonl(
        queue_path,
        [
            _prediction_row(
                side="UP",
                token_id="near-up",
                market_implied_prob=0.30,
                p_up=0.90,
                p_down=0.05,
                p_neutral=0.05,
                p_vol_up=0.20,
                p_vol_down=0.10,
                round_slug=near_round,
                ts=1_779_774_400_000,
            ),
            _prediction_row(
                side="UP",
                token_id="far-up",
                market_implied_prob=0.30,
                p_up=0.91,
                p_down=0.04,
                p_neutral=0.05,
                p_vol_up=0.20,
                p_vol_down=0.10,
                round_slug=far_round,
                ts=1_779_774_400_000,
            ),
        ],
        model_version="xgboost-v6",
        v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
        bridged_at=1_779_774_410_000,
    )

    assert written == 1
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["round_slug"] == near_round
    assert payload["token_id"] == "near-up"


def test_append_prediction_rows_as_signal_jsonl_clears_when_no_active_round(
    tmp_path: Path,
    monkeypatch,
) -> None:
    queue_path = tmp_path / "signals.jsonl"
    queue_path.write_text(
        json.dumps({"round_slug": "btc-updown-15m-1779774300"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(signal_queue, "_now_ms", lambda: 1_779_776_500_000)

    written = append_prediction_rows_as_signal_jsonl(
        queue_path,
        [
            _prediction_row(
                side="UP",
                token_id="expired-up",
                market_implied_prob=0.30,
                p_up=0.90,
                p_down=0.05,
                p_neutral=0.05,
                p_vol_up=0.20,
                p_vol_down=0.10,
                round_slug="btc-updown-15m-1779774300",
                ts=1_779_774_400_000,
            )
        ],
        model_version="xgboost-v6",
        v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
        bridged_at=1_779_776_500_000,
    )

    assert written == 0
    assert queue_path.read_text(encoding="utf-8") == ""


def test_append_prediction_rows_as_signal_jsonl_deduplicates_current_round(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "signals.jsonl"
    row = _prediction_row(
        side="UP",
        token_id="token-up",
        market_implied_prob=0.30,
        p_up=0.90,
        p_down=0.05,
        p_neutral=0.05,
        p_vol_up=0.20,
        p_vol_down=0.10,
    )

    assert (
        append_prediction_rows_as_signal_jsonl(
            queue_path,
            [row, row],
            model_version="xgboost-v6",
            v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
            bridged_at=1_779_774_410_000,
        )
        == 1
    )
    assert (
        append_prediction_rows_as_signal_jsonl(
            queue_path,
            [row],
            model_version="xgboost-v6",
            v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
            bridged_at=1_779_774_420_000,
        )
        == 0
    )

    payloads = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(payloads) == 1


def test_append_prediction_rows_as_signal_jsonl_skips_stale_signals(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "signals.jsonl"

    written = append_prediction_rows_as_signal_jsonl(
        queue_path,
        [
            _prediction_row(
                side="UP",
                token_id="token-up",
                market_implied_prob=0.30,
                p_up=0.90,
                p_down=0.05,
                p_neutral=0.05,
                p_vol_up=0.20,
                p_vol_down=0.10,
            )
        ],
        model_version="xgboost-v6",
        v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
        bridged_at=1_779_774_700_000,
        max_event_age_seconds=60,
    )

    assert written == 0
    assert not queue_path.exists()


def test_signal_bridge_diagnostics_reports_writable_current_round_signals() -> None:
    report = diagnose_prediction_rows_for_signal_bridge(
        [
            _prediction_row(
                side="UP",
                token_id="token-up",
                market_implied_prob=0.30,
                p_up=0.90,
                p_down=0.05,
                p_neutral=0.05,
                p_vol_up=0.20,
                p_vol_down=0.10,
                ts=1_779_774_405_000,
            )
        ],
        model_version="xgboost-v6",
        v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
        bridged_at=1_779_774_410_000,
        max_event_age_seconds=30,
    )

    assert report.rows_read == 1
    assert report.signals_built == 1
    assert report.current_round_slug == "btc-updown-15m-1779774300"
    assert report.current_round_signal_count == 1
    assert report.fresh_current_round_signal_count == 1
    assert report.stale_current_round_signal_count == 0
    assert report.reason_counts == {"writable_current_round_signals": 1}
    assert report.round_counts == {"btc-updown-15m-1779774300": 1}
    assert report.max_signal_age_seconds == pytest.approx(5.0)


def test_signal_bridge_diagnostics_reports_stale_current_round_signals() -> None:
    report = diagnose_prediction_rows_for_signal_bridge(
        [
            _prediction_row(
                side="UP",
                token_id="token-up",
                market_implied_prob=0.30,
                p_up=0.90,
                p_down=0.05,
                p_neutral=0.05,
                p_vol_up=0.20,
                p_vol_down=0.10,
                ts=1_779_774_405_000,
            )
        ],
        model_version="xgboost-v6",
        v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
        bridged_at=1_779_774_700_000,
        max_event_age_seconds=30,
    )

    assert report.signals_built == 1
    assert report.current_round_slug == "btc-updown-15m-1779774300"
    assert report.current_round_signal_count == 1
    assert report.fresh_current_round_signal_count == 0
    assert report.stale_current_round_signal_count == 1
    assert report.reason_counts == {"current_round_signals_stale": 1}
    assert report.max_signal_age_seconds == pytest.approx(295.0)


def test_signal_bridge_diagnostics_reports_no_active_round() -> None:
    report = diagnose_prediction_rows_for_signal_bridge(
        [
            _prediction_row(
                side="UP",
                token_id="token-up",
                market_implied_prob=0.30,
                p_up=0.90,
                p_down=0.05,
                p_neutral=0.05,
                p_vol_up=0.20,
                p_vol_down=0.10,
                ts=1_779_774_405_000,
            )
        ],
        model_version="xgboost-v6",
        v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
        bridged_at=1_779_776_500_000,
        max_event_age_seconds=30,
    )

    assert report.signals_built == 1
    assert report.current_round_slug is None
    assert report.current_round_signal_count == 0
    assert report.active_signal_count == 0
    assert report.expired_signal_count == 1
    assert report.non_current_round_signal_count == 1
    assert report.reason_counts == {"no_active_round": 1}


def test_jsonl_signal_source_reads_after_tail_and_resets_after_rotation(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "signals.jsonl"
    queue_path.write_text(
        json.dumps({"event_id": "old-1", "round_slug": "round-a"}) + "\n"
        + json.dumps({"event_id": "old-2", "round_slug": "round-a"}) + "\n",
        encoding="utf-8",
    )
    source = JsonlSignalSource(queue_path)
    cursor = source.latest_cursor(start="tail")

    assert cursor.position == 2
    assert cursor.signature

    with queue_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_id": "old-3", "round_slug": "round-a"}) + "\n")
    batch = source.read_after(cursor)

    assert [payload["event_id"] for payload in batch.payloads] == ["old-3"]
    assert batch.cursor.position == 3
    assert batch.cursor.signature

    queue_path.write_text(
        json.dumps({"event_id": "new-1", "round_slug": "round-b"}) + "\n",
        encoding="utf-8",
    )
    rotated = source.read_after(batch.cursor)

    assert [payload["event_id"] for payload in rotated.payloads] == ["new-1"]
    assert rotated.cursor.position == 1
    assert rotated.cursor.signature


def test_jsonl_signal_source_preserves_cursor_when_queue_is_missing(
    tmp_path: Path,
) -> None:
    source = JsonlSignalSource(tmp_path / "missing.jsonl")
    cursor = SignalCursor(position=7, signature="known")

    batch = source.read_after(cursor)

    assert batch.payloads == []
    assert batch.cursor == cursor


def test_kafka_signal_sink_emits_executor_ready_current_round_payloads() -> None:
    producer = _FakeKafkaProducer()
    sink = KafkaSignalSink(
        bootstrap_servers="localhost:9092",
        topic="bigan.signals",
        producer=producer,
    )
    row = _prediction_row(
        side="UP",
        token_id="token-up",
        market_implied_prob=0.30,
        p_up=0.90,
        p_down=0.05,
        p_neutral=0.05,
        p_vol_up=0.20,
        p_vol_down=0.10,
    )

    written = append_prediction_rows_to_signal_sink(
        sink,
        [row, row],
        model_version="xgboost-v6",
        v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
        bridged_at=1_779_774_410_000,
    )

    assert written == 1
    assert producer.flush_calls == [5.0]
    assert len(producer.records) == 1
    record = producer.records[0]
    assert record["topic"] == "bigan.signals"
    assert record["key"].startswith(b"btc-updown-15m-1779774300|UP|token-up|")
    payload = json.loads(record["value"].decode("utf-8"))
    assert payload["round_slug"] == "btc-updown-15m-1779774300"
    assert payload["outcome_side"] == "UP"
    assert payload["token_id"] == "token-up"


def test_kafka_signal_source_polls_executor_ready_payloads() -> None:
    consumer = _FakeKafkaConsumer(
        [
            _FakeKafkaMessage(json.dumps({"event_id": "pred-1"}).encode("utf-8")),
            _FakeKafkaMessage(b"not-json"),
            None,
        ]
    )
    source = KafkaSignalSource(
        bootstrap_servers="localhost:9092",
        topic="bigan.signals",
        group_id="paper",
        consumer=consumer,
    )
    cursor = source.latest_cursor(start="beginning")

    batch = source.read_after(cursor)

    assert consumer.subscriptions == [["bigan.signals"]]
    assert batch.payloads == [{"event_id": "pred-1"}]
    assert batch.cursor.position == 2


def test_v6_signal_queue_uses_external_token_map_for_opposite_side(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "signals.jsonl"
    round_slug = "btc-updown-15m-1779774300"

    written = append_prediction_rows_as_signal_jsonl(
        queue_path,
        [
            _prediction_row(
                side="DOWN",
                token_id="token-down",
                market_implied_prob=0.01,
                p_up=0.20,
                p_down=0.75,
                p_neutral=0.05,
                p_vol_up=0.10,
                p_vol_down=0.10,
                round_slug=round_slug,
            )
        ],
        model_version="xgboost-v6",
        v6_joint_config=V6JointGateConfig(settlement_threshold=0.50),
        bridged_at=1_779_774_410_000,
        token_ids_by_market_side={
            ("BTC-15M", round_slug, "UP"): "token-up",
            ("BTC-15M", round_slug, "DOWN"): "token-down",
        },
    )

    assert written == 1
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["outcome_side"] == "UP"
    assert payload["token_id"] == "token-up"
    assert payload["opposite_token_id"] == "token-down"


def test_append_prediction_rows_as_signal_jsonl_clears_old_round_when_new_round_is_stale(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "signals.jsonl"
    old_round = "btc-updown-15m-1779774300"
    new_round = "btc-updown-15m-1779775200"

    assert (
        append_prediction_rows_as_signal_jsonl(
            queue_path,
            [
                _prediction_row(
                    side="UP",
                    token_id="old-up",
                    market_implied_prob=0.30,
                    p_up=0.90,
                    p_down=0.05,
                    p_neutral=0.05,
                    p_vol_up=0.20,
                    p_vol_down=0.10,
                    round_slug=old_round,
                    ts=1_779_774_400_000,
                )
            ],
            model_version="xgboost-v6",
            v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
            bridged_at=1_779_774_410_000,
            max_event_age_seconds=60,
        )
        == 1
    )
    assert (
        append_prediction_rows_as_signal_jsonl(
            queue_path,
            [
                _prediction_row(
                    side="UP",
                    token_id="new-up",
                    market_implied_prob=0.30,
                    p_up=0.91,
                    p_down=0.04,
                    p_neutral=0.05,
                    p_vol_up=0.20,
                    p_vol_down=0.10,
                    round_slug=new_round,
                    ts=1_779_775_300_000,
                )
            ],
            model_version="xgboost-v6",
            v6_joint_config=V6JointGateConfig(settlement_threshold=0.80),
            bridged_at=1_779_775_500_000,
            max_event_age_seconds=60,
        )
        == 0
    )

    assert queue_path.read_text(encoding="utf-8") == ""


def test_append_event_driven_v7_signals_reprices_base_signal_from_raw_queue(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base-signals.jsonl"
    event_path = tmp_path / "event-signals.jsonl"
    cursor_path = tmp_path / "event.cursor"
    raw_path = tmp_path / "raw.jsonl"
    round_slug = "btc-updown-15m-1779774300"
    base_path.write_text(
        json.dumps(
            _v7_base_signal(
                side="UP",
                round_slug=round_slug,
                ts=1_779_774_300_000,
                created_at=1_779_774_305_000,
                model_probability=0.87,
                market_implied_prob=0.50,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    queue = JsonlRawQueue(raw_path)
    queue.append(
        "raw_top_of_book",
        _raw_top_of_book(
            side="UP",
            round_slug=round_slug,
            ts=1_779_774_312_000,
            bid=0.43,
            ask=0.44,
        ),
        published_at_ms=1_779_774_312_250,
    )

    report = append_event_driven_v7_signals_from_raw_queue(
        event_path,
        base_signal_jsonl_path=base_path,
        raw_queue_path=raw_path,
        cursor_path=cursor_path,
        bucket_seconds=10,
        bridged_at=1_779_774_321_000,
        max_event_age_seconds=60,
    )

    assert report.signals_written == 1
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    assert payload["ts"] == 1_779_774_310_000
    assert payload["created_at"] == 1_779_774_312_250
    assert payload["market_implied_prob"] == 0.44
    assert payload["entry_worst_price"] == 0.44
    assert payload["token_probability"] == 0.87
    assert payload["model_probability"] == 0.87
    assert payload["polymarket_price"] == 0.44
    assert payload["mispricing_edge"] == pytest.approx(0.43)
    assert payload["edge"] == pytest.approx(0.43)
    assert payload["residual_expected_edge_up"] == pytest.approx(0.43)
    assert payload["residual_expected_edge_down"] is None
    assert payload["selected_side"] == "UP"
    assert payload["event_id"].endswith("event-driven-1779774310000-1779774312250")
    assert int(cursor_path.read_text(encoding="utf-8")) > 0


def test_append_event_driven_v7_signals_uses_cursor_to_avoid_duplicates(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base-signals.jsonl"
    event_path = tmp_path / "event-signals.jsonl"
    cursor_path = tmp_path / "event.cursor"
    raw_path = tmp_path / "raw.jsonl"
    round_slug = "btc-updown-15m-1779774300"
    base_path.write_text(
        json.dumps(_v7_base_signal(side="DOWN", round_slug=round_slug))
        + "\n",
        encoding="utf-8",
    )
    queue = JsonlRawQueue(raw_path)
    queue.append(
        "raw_top_of_book",
        _raw_top_of_book(
            side="DOWN",
            round_slug=round_slug,
            ts=1_779_774_315_000,
            bid=0.31,
            ask=0.32,
        ),
        published_at_ms=1_779_774_315_100,
    )

    first = append_event_driven_v7_signals_from_raw_queue(
        event_path,
        base_signal_jsonl_path=base_path,
        raw_queue_path=raw_path,
        cursor_path=cursor_path,
        bucket_seconds=5,
        bridged_at=1_779_774_321_000,
    )
    second = append_event_driven_v7_signals_from_raw_queue(
        event_path,
        base_signal_jsonl_path=base_path,
        raw_queue_path=raw_path,
        cursor_path=cursor_path,
        bucket_seconds=5,
        bridged_at=1_779_774_322_000,
    )

    assert first.signals_written == 1
    assert second.rows_read == 0
    assert second.signals_written == 0
    assert len(event_path.read_text(encoding="utf-8").splitlines()) == 1


def test_append_event_driven_v7_signals_emits_best_side_once_per_bucket(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base-signals.jsonl"
    event_path = tmp_path / "event-signals.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    round_slug = "btc-updown-15m-1779774300"
    base_path.write_text(
        json.dumps(
            _v7_base_signal(
                side="UP",
                round_slug=round_slug,
                model_probability=0.82,
                market_implied_prob=0.50,
            )
        )
        + "\n"
        + json.dumps(
            _v7_base_signal(
                side="DOWN",
                round_slug=round_slug,
                model_probability=0.74,
                market_implied_prob=0.50,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    queue = JsonlRawQueue(raw_path)
    queue.append(
        "raw_top_of_book",
        _raw_top_of_book(
            side="UP",
            round_slug=round_slug,
            ts=1_779_774_312_000,
            bid=0.60,
            ask=0.61,
        ),
        published_at_ms=1_779_774_312_100,
    )
    queue.append(
        "raw_top_of_book",
        _raw_top_of_book(
            side="DOWN",
            round_slug=round_slug,
            ts=1_779_774_312_500,
            bid=0.39,
            ask=0.40,
        ),
        published_at_ms=1_779_774_312_600,
    )

    report = append_event_driven_v7_signals_from_raw_queue(
        event_path,
        base_signal_jsonl_path=base_path,
        raw_queue_path=raw_path,
        bucket_seconds=10,
        bridged_at=1_779_774_321_000,
        max_event_age_seconds=60,
    )

    assert report.buckets_seen == 2
    assert report.buckets_emitted == 1
    assert report.signals_written == 1
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    assert payload["outcome_side"] == "DOWN"
    assert payload["model_probability"] == pytest.approx(0.74)
    assert payload["polymarket_price"] == pytest.approx(0.40)
    assert payload["mispricing_edge"] == pytest.approx(0.34)


def test_append_event_driven_v7_signals_does_not_append_opposite_side_for_existing_bucket(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base-signals.jsonl"
    event_path = tmp_path / "event-signals.jsonl"
    cursor_path = tmp_path / "event.cursor"
    raw_path = tmp_path / "raw.jsonl"
    round_slug = "btc-updown-15m-1779774300"
    base_path.write_text(
        json.dumps(
            _v7_base_signal(
                side="UP",
                round_slug=round_slug,
                model_probability=0.82,
                market_implied_prob=0.50,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    queue = JsonlRawQueue(raw_path)
    queue.append(
        "raw_top_of_book",
        _raw_top_of_book(
            side="UP",
            round_slug=round_slug,
            ts=1_779_774_312_000,
            bid=0.45,
            ask=0.46,
        ),
        published_at_ms=1_779_774_312_100,
    )

    first = append_event_driven_v7_signals_from_raw_queue(
        event_path,
        base_signal_jsonl_path=base_path,
        raw_queue_path=raw_path,
        cursor_path=cursor_path,
        bucket_seconds=10,
        bridged_at=1_779_774_321_000,
        max_event_age_seconds=60,
    )

    base_path.write_text(
        base_path.read_text(encoding="utf-8")
        + json.dumps(
            _v7_base_signal(
                side="DOWN",
                round_slug=round_slug,
                model_probability=0.90,
                market_implied_prob=0.50,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    queue.append(
        "raw_top_of_book",
        _raw_top_of_book(
            side="DOWN",
            round_slug=round_slug,
            ts=1_779_774_313_000,
            bid=0.35,
            ask=0.36,
        ),
        published_at_ms=1_779_774_313_100,
    )
    second = append_event_driven_v7_signals_from_raw_queue(
        event_path,
        base_signal_jsonl_path=base_path,
        raw_queue_path=raw_path,
        cursor_path=cursor_path,
        bucket_seconds=10,
        bridged_at=1_779_774_322_000,
        max_event_age_seconds=60,
    )

    assert first.signals_written == 1
    assert second.signals_written == 0
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["outcome_side"] == "UP"


class _FakeKafkaProducer:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self.flush_calls: list[float] = []

    def produce(self, topic: str, *, key: bytes, value: bytes) -> None:
        self.records.append({"topic": topic, "key": key, "value": value})

    def flush(self, timeout: float) -> None:
        self.flush_calls.append(timeout)


class _FakeKafkaMessage:
    def __init__(self, value: bytes, error: object | None = None) -> None:
        self._value = value
        self._error = error

    def value(self) -> bytes:
        return self._value

    def error(self) -> object | None:
        return self._error


class _FakeKafkaConsumer:
    def __init__(self, messages: list[_FakeKafkaMessage | None]) -> None:
        self.messages = list(messages)
        self.subscriptions: list[list[str]] = []

    def subscribe(self, topics: list[str]) -> None:
        self.subscriptions.append(topics)

    def poll(self, _timeout: float) -> _FakeKafkaMessage | None:
        if not self.messages:
            return None
        return self.messages.pop(0)


def _prediction_row(
    *,
    side: str,
    token_id: str,
    market_implied_prob: float,
    p_up: float,
    p_down: float,
    p_neutral: float,
    p_vol_up: float,
    p_vol_down: float,
    round_slug: str = "btc-updown-15m-1779774300",
    ts: int = 1_779_774_400_000,
) -> dict[str, object]:
    return {
        "ts": ts,
        "message_ts": ts,
        "prediction_ts": ts,
        "ingest_ts": 1_779_774_405_000,
        "source": "polymarket",
        "source_symbol": token_id,
        "source_market": round_slug,
        "canonical_symbol": f"BTC-15M:{round_slug}:{side}",
        "symbol": f"BTC-15M:{round_slug}:{side}",
        "feature_version": "features_15m_v1",
        "model_version": "xgboost-v6",
        "calibration_method": "family-aware temperature scaling",
        "prob_up_15m": p_up,
        "raw_prob_up_15m": p_up,
        "p_up": p_up,
        "p_down": p_down,
        "p_neutral": p_neutral,
        "p_vol_up": p_vol_up,
        "p_vol_down": p_vol_down,
        "market_implied_prob": market_implied_prob,
        "confidence_bucket": "high_up",
        "top_features_json": "[]",
        "feature_values_json": "{}",
    }


def _v7_base_signal(
    *,
    side: str,
    round_slug: str,
    ts: int = 1_779_774_300_000,
    created_at: int = 1_779_774_305_000,
    model_probability: float = 0.88,
    market_implied_prob: float = 0.50,
) -> dict[str, object]:
    token_id = f"token-{side.lower()}"
    opposite = "DOWN" if side == "UP" else "UP"
    token_probability = model_probability
    return {
        "event_id": f"base-{side.lower()}",
        "ts": ts,
        "created_at": created_at,
        "model_version": "xgboost-v7",
        "prob_up_15m": 0.60,
        "canonical_symbol": f"BTC-15M:{round_slug}:{side}",
        "token_id": token_id,
        "outcome_side": side,
        "round_slug": round_slug,
        "round_end_ts": (int(round_slug.rsplit("-", 1)[-1]) + 900) * 1000,
        "market_implied_prob": market_implied_prob,
        "token_probability": token_probability,
        "edge": 0.10,
        "bridged_at": created_at + 100,
        "opposite_token_id": f"token-{opposite.lower()}",
        "p_up": 0.60,
        "p_down": 0.40,
        "p_neutral": 0.0,
        "settlement_residual": 0.20,
        "model_probability": model_probability,
        "polymarket_price": market_implied_prob,
        "mispricing_edge": token_probability - market_implied_prob,
        "token_expected_win_probability": token_probability,
        "p_up_residual_adjusted": None,
        "p_down_residual_adjusted": None,
        "residual_expected_edge_up": token_probability - market_implied_prob if side == "UP" else None,
        "residual_expected_edge_down": token_probability - market_implied_prob if side == "DOWN" else None,
        "selected_side": side,
        "selected_expected_edge": token_probability - market_implied_prob,
        "entry_worst_price": market_implied_prob,
        "should_enter_settlement": True,
    }


def _raw_top_of_book(
    *,
    side: str,
    round_slug: str,
    ts: int,
    bid: float,
    ask: float,
) -> dict[str, object]:
    return {
        "ts": ts,
        "message_ts": ts,
        "capture_timestamp_ms": ts + 50,
        "ingest_ts": ts + 50,
        "source": "polymarket",
        "source_symbol": f"token-{side.lower()}",
        "source_market": round_slug,
        "canonical_symbol": f"BTC-15M:{round_slug}:{side}",
        "bid_price": bid,
        "ask_price": ask,
        "spread": ask - bid,
    }
