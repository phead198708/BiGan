from __future__ import annotations

import json
from pathlib import Path

from bigan.execution.signal_queue import append_prediction_rows_as_signal_jsonl
from bigan.execution.v6_gate import V6JointGateConfig


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
