"""Capability-gated residual-promotion micro-live executor tests."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
import tempfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.phase5 import compute_safe_parameters_sha256
from bigan.v8.phase6 import (
    CICDPipelineConfig,
    CICDStageEvidence,
    RollbackPlan,
    run_phase6_cicd_pipeline,
)
from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus import (
    write_deterministic_polymarket_corpus_fixtures,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    REQUIRED_GATE_NAMES,
)
from bigan.v8.polymarket.residual_promotion_micro_live_authorization import (
    AUTHORIZATION_SCHEMA_VERSION,
    HUMAN_ATTESTATION_SCHEMA_VERSION,
    MicroLiveAuthorizationError,
    authorization_capability_is_verified,
    verify_micro_live_authorization,
)
from bigan.v8.polymarket.residual_promotion_micro_live_executor import (
    PROVIDER_FEATURE_FILENAMES,
    SIGNAL_SCHEMA_VERSION,
    MicroLiveExecutionError,
    ProviderFeatureEvidenceError,
    build_provider_bound_feature_rows,
    create_micro_live_executor,
    verify_provider_feature_evidence,
)
from bigan.v8.polymarket.residual_promotion_micro_live_executor import (
    MicroLiveExecutor as _StrictMicroLiveExecutor,
)
from bigan.v8.polymarket.residual_promotion_release_readiness import (
    OPERATIONAL_ROLLBACK_SCHEMA_VERSION,
    PHASE6_AUTHORIZATION_SCHEMA_VERSION,
    SHADOW_SCHEMA_VERSION,
)
from bigan.v8.polymarket.residual_promotion_release_readiness_v7 import (
    CONTRACT_REPOSITORY_PATH,
    run_micro_live_preapproval_assessment_v7,
)
from bigan.v8.polymarket.residual_promotion_security_review import (
    ATTESTATION_SCHEMA_VERSION,
)
from bigan.v8.polymarket.residual_promotion_security_review_v2 import (
    CANDIDATE_BUNDLE_REPOSITORY_PATH,
    PROTOCOL_REPOSITORY_PATH,
    REPORT_SCHEMA_VERSION,
    REQUIRED_CONTROL_EVIDENCE_PATHS,
    REQUIRED_CONTROL_IDS,
    REQUIRED_SCOPE_COMPONENT_PATHS,
    SCOPE_SCHEMA_VERSION,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    load_residual_promotion_runtime,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
)
AUTHORIZATION_TEMPLATE_PATH = f"{CONFIG_PATH}/micro_live_authorization_template_v7.json"
AUTHORIZED_AT_TS_MS = 1_789_948_800_000
NOW_TS_MS = AUTHORIZED_AT_TS_MS + 301_000
SETTLEMENT_NOW_TS_MS = AUTHORIZED_AT_TS_MS + 901_000


def _market_identity_evidence(
    signal: dict[str, Any],
) -> dict[str, bytes]:
    gamma_raw = json.dumps(
        [
            {
                "conditionId": signal["market_id"],
                "slug": signal["slug"],
                "outcomes": ["Up", "Down"],
                "clobTokenIds": [signal["up_token_id"], signal["down_token_id"]],
            }
        ],
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    clob_raw = json.dumps(
        {
            "condition_id": signal["market_id"],
            "market_slug": signal["slug"],
            "tokens": [
                {"outcome": "Up", "token_id": signal["up_token_id"]},
                {"outcome": "Down", "token_id": signal["down_token_id"]},
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    signal["market_identity"]["raw_gamma_payload_sha256"] = hashlib.sha256(
        gamma_raw
    ).hexdigest()
    signal["market_identity"]["clob_revalidation_payload_sha256"] = hashlib.sha256(
        clob_raw
    ).hexdigest()
    return {
        "raw_gamma_payload": gamma_raw,
        "raw_clob_revalidation_payload": clob_raw,
    }


def _synthetic_provider_feature_evidence() -> dict[str, bytes]:
    market_id = "0x" + "1" * 64
    up_token_id = "67890"
    down_token_id = "12345"
    with tempfile.TemporaryDirectory() as raw_directory:
        raw_dir = Path(raw_directory)
        write_deterministic_polymarket_corpus_fixtures(raw_dir)

        def load_rows(name: str) -> list[dict[str, Any]]:
            return [
                json.loads(line)
                for line in (raw_dir / name).read_text(encoding="utf-8").splitlines()
                if line
            ]

        market = next(
            row
            for row in load_rows("raw_polymarket_markets.jsonl")
            if row["market_family"] == "btc_updown_15m"
        )
        original_market_id = str(market["market_id"])
        delta = AUTHORIZED_AT_TS_MS - int(market["market_start_ts"])
        market.update(
            {
                "market_id": market_id,
                "condition_id": market_id,
                "slug": f"btc-updown-15m-{AUTHORIZED_AT_TS_MS // 1_000}",
                "up_token_id": up_token_id,
                "down_token_id": down_token_id,
                "market_start_ts": AUTHORIZED_AT_TS_MS,
                "market_end_ts": AUTHORIZED_AT_TS_MS + 900_000,
                "settlement_ts": AUTHORIZED_AT_TS_MS + 900_000,
                "trade_collection_mode": "websocket",
                "trade_stream_started_at_ts": AUTHORIZED_AT_TS_MS,
                "trade_stream_ended_at_ts": AUTHORIZED_AT_TS_MS + 600_000,
                "trade_stream_continuity_passed": True,
                "trade_stream_timestamp_causality_violation_count": 0,
                "trade_api_collection_ts": AUTHORIZED_AT_TS_MS + 600_000,
                "trade_api_request_failed": False,
                "trade_rest_rows_truncated": False,
                "trade_full_round_coverage_complete": True,
                "trade_tape_censored": False,
                "trade_collection_reason_codes": [],
            }
        )
        orderbooks = []
        for row in load_rows("raw_polymarket_orderbooks.jsonl"):
            if row["market_id"] != original_market_id:
                continue
            row.update(
                {
                    "market_id": market_id,
                    "token_id": (
                        up_token_id if row["outcome"] == "UP" else down_token_id
                    ),
                    "ts": int(row["ts"]) + delta,
                    "available_at_ts": int(row["available_at_ts"]) + delta,
                }
            )
            if row["outcome"] == "UP":
                for price_field in ("ask_price", "bid_price", "mid_price"):
                    row[price_field] = round(float(row[price_field]) - 0.11, 2)
            orderbooks.append(row)
        trades = []
        for row in load_rows("raw_polymarket_trades.jsonl"):
            if row["market_id"] != original_market_id:
                continue
            row.update(
                {
                    "market_id": market_id,
                    "token_id": (
                        up_token_id if row["outcome"] == "UP" else down_token_id
                    ),
                    "ts": int(row["ts"]) + delta,
                    "available_at_ts": int(row["available_at_ts"]) + delta,
                }
            )
            trades.append(row)
        candles = load_rows("raw_binance_btcusdt_klines.jsonl")
        for row in candles:
            for field in ("ts", "close_time", "available_at_ts"):
                row[field] = int(row[field]) + delta

    def encode(rows: list[dict[str, Any]]) -> bytes:
        return "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ).encode()

    return {
        "raw_polymarket_markets.jsonl": encode([market]),
        "raw_polymarket_orderbooks.jsonl": encode(orderbooks),
        "raw_polymarket_trades.jsonl": encode(trades),
        "raw_binance_btcusdt_klines.jsonl": encode(candles),
        "raw_polymarket_chainlink_prices.jsonl": b"",
    }


def _causal_provider_feature_evidence(
    source: dict[str, bytes],
    *,
    decision_ts_ms: int,
) -> dict[str, bytes]:
    output: dict[str, bytes] = {}
    for name in PROVIDER_FEATURE_FILENAMES:
        rows = [
            json.loads(line)
            for line in source[name].decode().splitlines()
            if line
        ]
        if name == "raw_polymarket_markets.jsonl":
            market = rows[0]
            market["trade_stream_ended_at_ts"] = decision_ts_ms
            market["trade_api_collection_ts"] = decision_ts_ms
            market["trade_full_round_coverage_complete"] = None
        else:
            rows = [
                row
                for row in rows
                if int(row["available_at_ts"]) <= decision_ts_ms
            ]
        output[name] = "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ).encode()
    return output


def _base_feature_and_signal() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, bytes],
    dict[str, bytes],
    dict[int, dict[str, Any]],
    dict[int, dict[str, bytes]],
]:
    full_provider_evidence = _synthetic_provider_feature_evidence()
    provider_evidence_by_decision = {
        decision_ts: _causal_provider_feature_evidence(
            full_provider_evidence,
            decision_ts_ms=decision_ts,
        )
        for decision_ts in (
            AUTHORIZED_AT_TS_MS + 300_000,
            AUTHORIZED_AT_TS_MS + 600_000,
        )
    }
    feature_rows: dict[int, dict[str, Any]] = {}
    for decision_ts, evidence in provider_evidence_by_decision.items():
        feature_rows[decision_ts] = copy.deepcopy(
            next(
                row
                for row in build_provider_bound_feature_rows(evidence)
                if int(row["decision_ts"]) == decision_ts
            )
        )
    decision_ts_ms = AUTHORIZED_AT_TS_MS + 300_000
    feature_row = copy.deepcopy(feature_rows[decision_ts_ms])
    manifest_path = (
        REPO_ROOT
        / "examples/v8/polymarket_configs/"
        "BTC-15M-cost-aware-market-residual-promotion-v1/"
        "candidate_bundle/bundle_manifest.json"
    )
    runtime = load_residual_promotion_runtime(
        manifest_path=manifest_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        repository_root=REPO_ROOT,
    )
    projection = runtime.score_feature_row(
        feature_row,
        observed_at_ts=decision_ts_ms,
    )
    assert projection["model_scored"] is True
    assert projection["fail_closed"] is False
    raw = dict(feature_row["features"])
    signal = {
        "schema_version": SIGNAL_SCHEMA_VERSION,
        "lineage_id": "BTC-15M-cost-aware-market-residual-promotion-v1",
        "candidate_id": "residual-v4-challenger-carry-forward-final-fit-001",
        "candidate_bundle_sha256": "placeholder",
        "market_id": feature_row["market_id"],
        "slug": "btc-updown-15m-1789948800",
        "market_family": "BTC-15M",
        "decision_ts_ms": decision_ts_ms,
        "observed_at_ts_ms": decision_ts_ms,
        "action_values": projection["action_values"],
        "executable_asks": {
            "UP": str(raw["up_ask"]),
            "DOWN": str(raw["down_ask"]),
        },
        "up_token_id": "67890",
        "down_token_id": "12345",
        "market_identity": {
            "source_type": "gamma_primary_plus_live_clob_revalidation",
            "condition_id": feature_row["market_id"],
            "slug": "btc-updown-15m-1789948800",
            "market_family": "btc_updown_15m",
            "market_start_ts_ms": AUTHORIZED_AT_TS_MS,
            "market_end_ts_ms": AUTHORIZED_AT_TS_MS + 900_000,
            "up_token_id": "67890",
            "down_token_id": "12345",
            "gamma_fetched_at_ts_ms": AUTHORIZED_AT_TS_MS + 1_000,
            "clob_revalidated_at_ts_ms": decision_ts_ms,
            "raw_gamma_payload_sha256": "placeholder",
            "clob_revalidation_payload_sha256": "placeholder",
            "clob_revalidation_passed": True,
            "outcomes_accessed": False,
            "settlement_accessed": False,
            "pnl_accessed": False,
        },
        "selected_action": projection["selected_action"],
        "model_scored": projection["model_scored"],
        "fail_closed": projection["fail_closed"],
        "fail_closed_reasons": [],
        "decision_influenced_collection": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "safety": dict(SAFETY),
    }
    return (
        feature_row,
        signal,
        _market_identity_evidence(signal),
        provider_evidence_by_decision[decision_ts_ms],
        feature_rows,
        provider_evidence_by_decision,
    )


(
    BASE_FEATURE_ROW,
    BASE_SIGNAL_PAYLOAD,
    BASE_MARKET_IDENTITY_EVIDENCE,
    BASE_PROVIDER_FEATURE_EVIDENCE,
    BASE_PROVIDER_FEATURE_ROWS,
    BASE_PROVIDER_FEATURE_EVIDENCE_BY_DECISION,
) = _base_feature_and_signal()


def _provider_feature_evidence_for_signal(
    signal: dict[str, Any],
) -> dict[str, bytes]:
    target_start = int(str(signal["slug"]).rsplit("-", maxsplit=1)[1]) * 1_000
    decision_offset = int(signal["decision_ts_ms"]) - target_start
    template_decision_ts = AUTHORIZED_AT_TS_MS + decision_offset
    source = BASE_PROVIDER_FEATURE_EVIDENCE_BY_DECISION.get(
        template_decision_ts,
        BASE_PROVIDER_FEATURE_EVIDENCE,
    )
    delta = target_start - AUTHORIZED_AT_TS_MS
    output: dict[str, bytes] = {}
    for name in PROVIDER_FEATURE_FILENAMES:
        raw = source[name]
        if not raw:
            output[name] = b""
            continue
        rows = [json.loads(line) for line in raw.decode().splitlines() if line]
        for row in rows:
            if name == "raw_polymarket_markets.jsonl":
                row.update(
                    {
                        "market_id": signal["market_id"],
                        "condition_id": signal["market_id"],
                        "slug": signal["slug"],
                        "up_token_id": signal["up_token_id"],
                        "down_token_id": signal["down_token_id"],
                    }
                )
                for field in (
                    "market_start_ts",
                    "market_end_ts",
                    "settlement_ts",
                    "trade_stream_started_at_ts",
                    "trade_stream_ended_at_ts",
                    "trade_api_collection_ts",
                ):
                    row[field] = int(row[field]) + delta
            elif name in {
                "raw_polymarket_orderbooks.jsonl",
                "raw_polymarket_trades.jsonl",
            }:
                row["market_id"] = signal["market_id"]
                row["token_id"] = (
                    signal["up_token_id"]
                    if row["outcome"] == "UP"
                    else signal["down_token_id"]
                )
                row["ts"] = int(row["ts"]) + delta
                row["available_at_ts"] = int(row["available_at_ts"]) + delta
            elif name == "raw_binance_btcusdt_klines.jsonl":
                for field in ("ts", "close_time", "available_at_ts"):
                    row[field] = int(row[field]) + delta
        output[name] = "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ).encode()
    return output


def _order_identity(
    executor: MicroLiveExecutor,
    client_order_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = next(
        (
            dict(event["payload"])
            for event in executor.events
            if event["event_type"] == "ORDER_PREPARED"
            and event["payload"]["client_order_id"] == client_order_id
        ),
        {
            "client_order_id": client_order_id,
            "market_id": "0x" + "0" * 64,
            "token_id": "1",
            "slug": "btc-updown-15m-1789948800",
            "submitted_at_ts_ms": NOW_TS_MS,
            "signal_payload": {
                "up_token_id": "1",
                "down_token_id": "2",
                "market_identity": {"market_end_ts_ms": AUTHORIZED_AT_TS_MS + 900_000},
            },
        },
    )
    acknowledgement = next(
        (
            dict(event["payload"])
            for event in executor.events
            if event["event_type"] == "ORDER_ACKNOWLEDGED"
            and event["payload"]["client_order_id"] == client_order_id
        ),
        {"exchange_order_id": "exchange-missing"},
    )
    return prepared, acknowledgement


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _provider_evidence_with_market_patch(**changes: Any) -> dict[str, bytes]:
    evidence = dict(BASE_PROVIDER_FEATURE_EVIDENCE)
    market = json.loads(
        evidence["raw_polymarket_markets.jsonl"].decode().strip()
    )
    market.update(changes)
    evidence["raw_polymarket_markets.jsonl"] = _json_bytes(market) + b"\n"
    return evidence


class MicroLiveExecutor(_StrictMicroLiveExecutor):
    """Test adapter that materializes mutable fixtures as exact raw bytes."""

    def submit_signal(
        self,
        *,
        signal_payload: dict[str, Any],
        feature_row: dict[str, Any],
        now_ts_ms: int,
        operator_heartbeat_ts_ms: int,
        market_identity_evidence: dict[str, bytes] | None = None,
    ) -> dict[str, Any]:
        return super().submit_signal(
            raw_signal_payload=_json_bytes(signal_payload),
            raw_feature_row=_json_bytes(feature_row),
            provider_feature_evidence=_provider_feature_evidence_for_signal(
                signal_payload
            ),
            now_ts_ms=now_ts_ms,
            operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            market_identity_evidence=market_identity_evidence,
        )


def _record_fill(
    executor: MicroLiveExecutor,
    *,
    client_order_id: str,
    fill_id: str,
    now_ts_ms: int,
    quantity: str,
    price: str,
    fee_usd: str,
    transport_event_sha256: str,
    event_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del transport_event_sha256  # Opaque caller-provided hashes are no longer trusted.
    prepared, acknowledgement = _order_identity(executor, client_order_id)
    event = {
        "event_type": "FILL",
        "client_order_id": client_order_id,
        "exchange_order_id": acknowledgement["exchange_order_id"],
        "fill_id": fill_id,
        "market_id": prepared["market_id"],
        "token_id": prepared["token_id"],
        "quantity": quantity,
        "price": price,
        "fee_usd": fee_usd,
        "executed_at_ts_ms": now_ts_ms,
    }
    event.update(event_overrides or {})
    return executor.record_fill(
        client_order_id=client_order_id,
        fill_id=fill_id,
        now_ts_ms=now_ts_ms,
        quantity=quantity,
        price=price,
        fee_usd=fee_usd,
        raw_transport_event=_json_bytes(event),
    )


def _record_order_closed(
    executor: MicroLiveExecutor,
    *,
    client_order_id: str,
    status: str,
    now_ts_ms: int,
    transport_event_sha256: str,
    event_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del transport_event_sha256  # Opaque caller-provided hashes are no longer trusted.
    prepared, acknowledgement = _order_identity(executor, client_order_id)
    event = {
        "event_type": "ORDER_CLOSED",
        "client_order_id": client_order_id,
        "exchange_order_id": acknowledgement["exchange_order_id"],
        "market_id": prepared["market_id"],
        "token_id": prepared["token_id"],
        "status": status,
        "effective_at_ts_ms": now_ts_ms,
    }
    event.update(event_overrides or {})
    return executor.record_order_closed(
        client_order_id=client_order_id,
        status=status,
        now_ts_ms=now_ts_ms,
        raw_transport_event=_json_bytes(event),
    )


def _record_settlement(
    executor: MicroLiveExecutor,
    *,
    client_order_id: str,
    settlement_id: str,
    now_ts_ms: int,
    payout_per_token: str,
    official_settlement_sha256: str,
    event_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del official_settlement_sha256  # Exact official bytes now determine the hash.
    prepared, _ = _order_identity(executor, client_order_id)
    signal = dict(prepared["signal_payload"])
    winning_token_id = (
        prepared["token_id"]
        if payout_per_token == "1"
        else (
            signal["down_token_id"]
            if prepared["token_id"] == signal["up_token_id"]
            else signal["up_token_id"]
        )
    )
    event = {
        "event_type": "OFFICIAL_SETTLEMENT",
        "settlement_id": settlement_id,
        "market_id": prepared["market_id"],
        "slug": prepared["slug"],
        "winning_token_id": winning_token_id,
        "payout_per_token": payout_per_token,
        "finalized_at_ts_ms": now_ts_ms,
    }
    event.update(event_overrides or {})
    return executor.record_settlement(
        client_order_id=client_order_id,
        settlement_id=settlement_id,
        now_ts_ms=now_ts_ms,
        payout_per_token=payout_per_token,
        raw_official_settlement_event=_json_bytes(event),
    )


class FakeTransport:
    def __init__(
        self,
        *,
        fail_submit: bool = False,
        fail_cancel: bool = False,
        fail_lookup: bool = False,
        fixed_exchange_order_id: str | None = None,
        submit_status: str = "ACCEPTED",
        lookup_status: str = "ACCEPTED",
        cancel_lookup_status: str = "CANCELED",
        cancel_lookup_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.fail_submit = fail_submit
        self.fail_cancel = fail_cancel
        self.fail_lookup = fail_lookup
        self.fixed_exchange_order_id = fixed_exchange_order_id
        self.submit_status = submit_status
        self.lookup_status = lookup_status
        self.cancel_lookup_status = cancel_lookup_status
        self.cancel_lookup_overrides = dict(cancel_lookup_overrides or {})
        self.submit_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []
        self.lookup_calls: list[dict[str, Any]] = []

    def submit_order(self, request: dict[str, Any]) -> bytes:
        self.submit_calls.append(copy.deepcopy(request))
        if self.fail_submit:
            raise RuntimeError("synthetic transport timeout")
        return _json_bytes({
            "client_order_id": request["client_order_id"],
            "exchange_order_id": self.fixed_exchange_order_id
            or f"exchange-{request['client_order_id'][:12]}",
            "status": self.submit_status,
            "market_id": request["market_id"],
            "token_id": request["token_id"],
            "accepted_quantity": request["quantity"],
            "limit_price": request["limit_price"],
        })

    def cancel_order(self, request: dict[str, Any]) -> bytes:
        self.cancel_calls.append(copy.deepcopy(request))
        if self.fail_cancel:
            raise RuntimeError("synthetic cancel timeout")
        return _json_bytes({
            "client_order_id": request["client_order_id"],
            "exchange_order_id": request["exchange_order_id"],
            "status": "CANCELED",
        })

    def lookup_order(self, request: dict[str, Any]) -> bytes:
        self.lookup_calls.append(copy.deepcopy(request))
        if self.fail_lookup:
            raise RuntimeError("synthetic lookup timeout")
        submitted = next(
            row
            for row in reversed(self.submit_calls)
            if row["client_order_id"] == request["client_order_id"]
        )
        if request.get("lookup_purpose") == "cancel_reconciliation":
            status = self.cancel_lookup_status
            response = {
                "client_order_id": submitted["client_order_id"],
                "exchange_order_id": self.fixed_exchange_order_id
                or f"exchange-{submitted['client_order_id'][:12]}",
                "status": status,
                "market_id": submitted["market_id"],
                "token_id": submitted["token_id"],
                "accepted_quantity": submitted["quantity"],
                "limit_price": submitted["limit_price"],
                "observed_at_ts_ms": NOW_TS_MS + 2,
                "effective_at_ts_ms": (
                    NOW_TS_MS + 1
                    if status in {"CANCELED", "EXPIRED"}
                    else None
                ),
            }
            response.update(self.cancel_lookup_overrides)
            return _json_bytes(response)
        return _json_bytes({
            "client_order_id": submitted["client_order_id"],
            "exchange_order_id": self.fixed_exchange_order_id
            or f"exchange-{submitted['client_order_id'][:12]}",
            "status": self.lookup_status,
            "market_id": submitted["market_id"],
            "token_id": submitted["token_id"],
            "accepted_quantity": submitted["quantity"],
            "limit_price": submitted["limit_price"],
        })


@pytest.fixture(scope="module")
def authorized_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("micro-live-repository") / "repo"
    shutil.copytree(
        REPO_ROOT,
        root,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
    )
    evidence_root = root / "future_evidence"
    evidence_root.mkdir()
    contract = _json(root / CONTRACT_REPOSITORY_PATH)
    evidence = _complete_evidence(root, contract)
    descriptors = _write_evidence(evidence_root, evidence)
    assessment_path = evidence_root / "preapproval_assessment.json"
    assessment = run_micro_live_preapproval_assessment_v7(
        repository_root=root,
        contract_path=root / CONTRACT_REPOSITORY_PATH,
        expected_contract_sha256=sha256_file(root / CONTRACT_REPOSITORY_PATH),
        evidence_root=evidence_root,
        evidence_descriptors=descriptors,
        output_path=assessment_path,
        created_at="2026-09-20T00:00:00Z",
    )
    assert assessment["ready_to_request_micro_live_approval"] is True
    required = {
        "preapproval_assessment": _descriptor(evidence_root, assessment_path),
        "fresh_evaluation_manifest": descriptors["evaluation_manifest"],
        "phase6_release_manifest": descriptors["phase6_report"],
        "phase6_zero_capital_authorization": descriptors["phase6_authorization"],
        "operational_rollback_report": descriptors["operational_rollback"],
        "independent_security_review_report": descriptors["security_review"],
    }
    authorization = _authorization(root, evidence_root, required)
    return {
        "root": root,
        "evidence_root": evidence_root,
        "authorization": authorization,
        "now_ts_ms": NOW_TS_MS,
    }


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _descriptor(base: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(base).as_posix(), "sha256": sha256_file(path)}


def _repository_descriptor(root: Path, repository_path: str) -> dict[str, str]:
    return {"path": repository_path, "sha256": sha256_file(root / repository_path)}


def _closed(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }


def _complete_evidence(
    root: Path,
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    bundle_sha = dict(contract["candidate_bundle"])["sha256"]
    functional_sha = dict(contract["functional_rollback_drill"])["sha256"]
    population_sha = "a" * 64
    safe_parameters = {"action": "NO_TRADE", "capital_fraction": 0.0}
    evaluation_report = _closed(
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "production_evaluation": True,
            "population": {"passed": True, "paired_market_count": 2_500},
            "gate_results": dict.fromkeys(REQUIRED_GATE_NAMES, True),
            "all_gates_passed": True,
            "failed_gates": [],
            "lineage_terminalized": False,
            "failed_population_reuse_allowed": False,
            "phase6_required": True,
            "rollback_drill_required": True,
            "micro_live_go_no_go": "NO_GO_PENDING_PHASE6_AND_ROLLBACK_DRILL",
            "automatic_promotion_or_live_unlock": False,
        }
    )
    evaluation_manifest = _closed(
        {
            "lineage_id": contract["lineage_id"],
            "candidate_id": contract["candidate_id"],
            "evaluation_executed_exactly_once": True,
            "rerun_allowed": False,
            "fresh_population_reuse_allowed": False,
            "all_fresh_confirmation_gates_passed": True,
            "lineage_terminalized": False,
            "automatic_promotion_or_live_unlock": False,
            "micro_live_approval_granted": False,
            "population_manifest_sha256": population_sha,
            "settlement_ingestion_manifest": {
                "path": "settlement_ingestion_manifest.json",
                "sha256": "c" * 64,
            },
            "evaluation_report": {},
        }
    )
    shadow = _closed(
        {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "lineage_id": contract["lineage_id"],
            "candidate_id": contract["candidate_id"],
            "implementation": dict(contract["shadow_evidence_implementation"]),
            "cli": dict(contract["shadow_evidence_cli"]),
            "candidate_bundle_sha256": bundle_sha,
            "population_manifest_sha256": population_sha,
            "candidate_row_count": 2_500,
            "baseline_row_count": 2_500,
            "paired_row_count": 2_500,
            "zero_capital_read_only": True,
            "runtime_decision_parity_passed": True,
            "shadow_stability_passed": True,
            "monitoring_enabled": True,
            "kill_switch_wired": True,
            "collection_population_changed": False,
            "outcomes_accessed_during_collection": False,
        }
    )
    operational = _closed(
        {
            "schema_version": OPERATIONAL_ROLLBACK_SCHEMA_VERSION,
            "lineage_id": contract["lineage_id"],
            "candidate_id": contract["candidate_id"],
            "implementation": dict(contract["operational_rollback_evidence_implementation"]),
            "cli": dict(contract["operational_rollback_evidence_cli"]),
            "candidate_bundle_sha256": bundle_sha,
            "functional_rollback_report_sha256": functional_sha,
            "rollback_target": "NO_TRADE",
            "safe_parameters": safe_parameters,
            "safe_parameters_sha256": canonical_json_sha256(safe_parameters),
            "latency_measurements_ms": [75, 92, 88],
            "maximum_observed_latency_ms": 92.0,
            "rollback_drill_passed": True,
            "micro_live_authorized": False,
        }
    )
    security = _security_review(root, bundle_sha)
    evidence = {
        "evaluation_manifest": evaluation_manifest,
        "evaluation_report": evaluation_report,
        "shadow_stability": shadow,
        "operational_rollback": operational,
        "security_review": security,
    }
    phase6_authorization = _closed(
        {
            "schema_version": PHASE6_AUTHORIZATION_SCHEMA_VERSION,
            "lineage_id": contract["lineage_id"],
            "candidate_id": contract["candidate_id"],
            "authorization_scope": "post_confirmation_phase6_zero_capital_only",
            "candidate_bundle_sha256": bundle_sha,
            "supersedes_template": dict(contract["phase6_zero_capital_authorization_template"]),
            "fresh_evaluation_manifest_payload_sha256": "pending",
            "phase6_zero_capital_authorized": True,
            "requested_capital_fraction": 0.0,
            "rollout_step_index": 0,
            "explicit_human_zero_capital_approval_recorded": True,
            "authorization_record_executable": True,
            "collection_authorization_reused": False,
            "micro_live_authorized": False,
        }
    )
    evidence["phase6_authorization"] = phase6_authorization
    evidence["phase6_report"] = {}
    return evidence


def _write_evidence(
    evidence_root: Path,
    evidence: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    report_path = evidence_root / "evaluation_report.json"
    _write_json(report_path, evidence["evaluation_report"])
    evidence["evaluation_manifest"]["evaluation_report"] = _descriptor(
        evidence_root, report_path
    )
    evidence["phase6_authorization"]["fresh_evaluation_manifest_payload_sha256"] = (
        canonical_json_sha256(evidence["evaluation_manifest"])
    )
    contract = _json(
        evidence_root.parent / "repo" / CONTRACT_REPOSITORY_PATH
        if evidence_root.parent.name != "repo"
        else evidence_root.parent / CONTRACT_REPOSITORY_PATH
    )
    evidence["phase6_report"] = _phase6_report(
        contract,
        evidence["phase6_authorization"],
    )
    descriptors: dict[str, dict[str, str]] = {}
    for name, payload in evidence.items():
        path = evidence_root / f"{name}.json"
        _write_json(path, payload)
        descriptors[name] = _descriptor(evidence_root, path)
    return descriptors


def _phase6_report(
    contract: dict[str, Any],
    phase6_authorization: dict[str, Any],
) -> dict[str, Any]:
    identity = dict(contract["phase6_candidate_identity"])
    candidate_id = str(contract["candidate_id"])
    bundle_sha = str(identity["model_sha256"])
    authorization_sha = canonical_json_sha256(phase6_authorization)

    def stage(name: str, artifact_sha: str, metadata: dict[str, Any]) -> CICDStageEvidence:
        return CICDStageEvidence(
            stage=name,  # type: ignore[arg-type]
            passed=True,
            artifact_sha256=artifact_sha,
            report_sha256="f" * 64,
            run_id=f"{name}-001",
            metadata={
                "candidate_run_id": candidate_id,
                "model_sha256": identity["model_sha256"],
                "policy_dataset_hash": identity["policy_dataset_hash"],
                "split_hash": identity["split_hash"],
                **metadata,
            },
        )

    stages = (
        stage("training", bundle_sha, {"accepted_candidate_model": True, "deterministic_training": True}),
        stage(
            "validation",
            "1" * 64,
            {
                "oos_backtest_passed": True,
                "cost_stress_passed": True,
                "cost_stress_multipliers": [1.2, 1.5, 2.0],
            },
        ),
        stage(
            "shadow_deployment",
            "2" * 64,
            {"shadow_mode": True, "simulate_live_execution": True, "capital_at_risk": False},
        ),
        stage(
            "live_deployment",
            "3" * 64,
            {
                "staged_capital_rollout": True,
                "manual_approval_recorded": True,
                "zero_capital_authorization_sha256": authorization_sha,
                "rollout_capital_fractions": [0.0, 0.01, 0.05, 0.10],
                "rollout_step_index": 0,
                "requested_capital_fraction": 0.0,
                "capital_at_risk": False,
                "wallet_signing_allowed": False,
                "polymarket_write_allowed": False,
                "one_percent_micro_live_authorized": False,
            },
        ),
        stage(
            "monitoring",
            "4" * 64,
            {
                "performance_tracking_enabled": True,
                "risk_tracking_enabled": True,
                "kill_switch_wired": True,
                "feed_health_passed": True,
            },
        ),
    )
    safe_parameters = {"action": "NO_TRADE", "capital_fraction": 0.0}
    rollback = RollbackPlan(
        stable_model_id=candidate_id,
        stable_model_sha256=bundle_sha,
        safe_parameter_sha256=compute_safe_parameters_sha256(safe_parameters),
        safe_parameters=safe_parameters,
        rollback_artifact_sha256=dict(contract["functional_rollback_drill"])["sha256"],
        latency_measurements_ms=(75, 92, 88),
    )
    return run_phase6_cicd_pipeline(
        candidate_run_id=candidate_id,
        stage_evidence=stages,
        rollback_plan=rollback,
        config=CICDPipelineConfig(created_at="2026-09-20T00:00:00Z"),
    ).report.to_dict()


def _security_review(root: Path, bundle_sha: str) -> dict[str, Any]:
    reviewed_commit = "d" * 40
    config = root / CONFIG_PATH
    review_url = "https://github.com/phead198708/BiGan/pull/999#pullrequestreview-9001"
    attestation_path = config / "security_review_attestation_9001.json"
    _write_json(
        attestation_path,
        {
            "schema_version": ATTESTATION_SCHEMA_VERSION,
            "reviewer_github_login": "independent-reviewer",
            "review_id": 9001,
            "review_url": review_url,
            "reviewed_commit_sha": reviewed_commit,
            "independent_from_implementation": True,
            "authored_reviewed_bytes": False,
            "attestation_statement": "Independent exact-scope security review completed.",
        },
    )
    github_path = config / "security_review_github_payload_9001.json"
    _write_json(
        github_path,
        {
            "id": 9001,
            "html_url": review_url,
            "state": "APPROVED",
            "commit_id": reviewed_commit,
            "submitted_at": "2026-09-20T00:00:00Z",
            "user": {"login": "independent-reviewer"},
        },
    )
    components = [
        {
            "component_id": component_id,
            **_repository_descriptor(root, repository_path),
        }
        for component_id, repository_path in REQUIRED_SCOPE_COMPONENT_PATHS.items()
    ]
    controls = {
        control_id: {
            "status": "PASS",
            "evidence": [
                _repository_descriptor(root, repository_path)
                for repository_path in REQUIRED_CONTROL_EVIDENCE_PATHS[control_id]
            ],
            "notes": f"Independent evidence reviewed for {control_id}.",
        }
        for control_id in REQUIRED_CONTROL_IDS
    }
    return _closed(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "lineage_id": "BTC-15M-cost-aware-market-residual-promotion-v1",
            "candidate_id": "residual-v4-challenger-carry-forward-final-fit-001",
            "created_at": "2026-09-20T00:00:00Z",
            "security_review_protocol": _repository_descriptor(root, PROTOCOL_REPOSITORY_PATH),
            "candidate_bundle_sha256": bundle_sha,
            "reviewed_commit_sha": reviewed_commit,
            "reviewer": {
                "github_login": "independent-reviewer",
                "review_id": 9001,
                "review_url": review_url,
                "review_state": "APPROVED",
                "reviewed_commit_sha": reviewed_commit,
                "independent_from_implementation": True,
                "authored_reviewed_bytes": False,
                "attestation": _repository_descriptor(
                    root, attestation_path.relative_to(root).as_posix()
                ),
                "github_review_payload": _repository_descriptor(
                    root, github_path.relative_to(root).as_posix()
                ),
            },
            "implementation_author_logins": ["implementation-author"],
            "scope_manifest": {
                "schema_version": SCOPE_SCHEMA_VERSION,
                "reviewed_commit_sha": reviewed_commit,
                "components": components,
            },
            "controls": controls,
            "findings": {
                "open_counts": {"P0": 0, "P1": 0, "P2": 0, "P3": 0},
                "items": [],
            },
            "ci": {
                "run_url": "https://github.com/phead198708/BiGan/actions/runs/9001",
                "conclusion": "SUCCESS",
                "exact_head_sha": reviewed_commit,
            },
            "security_review_passed": True,
            "maximum_initial_capital_fraction": 0.01,
            "fresh_outcomes_accessed": False,
            "settlement_accessed": False,
            "pnl_accessed": False,
            "explicit_human_approval_recorded": False,
            "phase6_zero_capital_authorized": False,
            "micro_live_authorized": False,
        }
    )


def _authorization(
    root: Path,
    evidence_root: Path,
    required: dict[str, dict[str, str]],
    *,
    maximum_realized_loss_usd: str = "1.00",
) -> dict[str, Any]:
    evidence_payload_sha = canonical_json_sha256(
        {name: value["sha256"] for name, value in sorted(required.items())}
    )
    candidate_sha = sha256_file(root / CANDIDATE_BUNDLE_REPOSITORY_PATH)
    identity = {
        "lineage_id": "BTC-15M-cost-aware-market-residual-promotion-v1",
        "candidate_id": "residual-v4-challenger-carry-forward-final-fit-001",
        "candidate_bundle_sha256": candidate_sha,
        "evidence_payload_sha256": evidence_payload_sha,
        "capital_base_usd": "1000",
        "requested_initial_capital_fraction": "0.01",
        "maximum_notional_usd": "10.00",
        "maximum_realized_loss_usd": maximum_realized_loss_usd,
        "maximum_open_orders": 2,
        "market_allowlist": ["BTC-15M"],
        "allowed_actions": ["BUY_UP_HOLD", "BUY_DOWN_HOLD"],
        "authorized_at_ts_ms": AUTHORIZED_AT_TS_MS,
        "expires_at_ts_ms": AUTHORIZED_AT_TS_MS + 10_000_000,
        "maximum_signal_age_ms": 5_000,
        "maximum_operator_heartbeat_age_ms": 5_000,
        "approval_issue_number": 264,
    }
    authorization_id = canonical_json_sha256(identity)
    command = (
        "APPROVE BTC-15M-cost-aware-market-residual-promotion-v1 MICRO-LIVE "
        f"authorization_id={authorization_id} capital_base_usd=1000 "
        f"maximum_notional_usd=10.00 maximum_realized_loss_usd={maximum_realized_loss_usd} "
        "maximum_open_orders=2 "
        f"capital_fraction=0.01 expires_at_ts_ms={identity['expires_at_ts_ms']}"
    )
    comment_url = "https://github.com/phead198708/BiGan/issues/264#issuecomment-99001"
    github_path = evidence_root / "human_approval_github_payload.json"
    _write_json(
        github_path,
        {
            "id": 99001,
            "html_url": comment_url,
            "created_at": "2026-09-21T00:00:00Z",
            "body": command,
            "user": {"login": "phead198708"},
        },
    )
    attestation_path = evidence_root / "human_approval_attestation.json"
    _write_json(
        attestation_path,
        {
            "schema_version": HUMAN_ATTESTATION_SCHEMA_VERSION,
            "github_login": "phead198708",
            "issue_number": 264,
            "comment_id": 99001,
            "comment_url": comment_url,
            "authorization_id": authorization_id,
            "approved_at_ts_ms": AUTHORIZED_AT_TS_MS,
            "attestation_statement": command,
        },
    )
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "lineage_id": identity["lineage_id"],
        "candidate_id": identity["candidate_id"],
        "created_at": "2026-09-21T00:00:00Z",
        "authorization_id": authorization_id,
        "supersedes_template": _repository_descriptor(root, AUTHORIZATION_TEMPLATE_PATH),
        "candidate_bundle": _repository_descriptor(root, CANDIDATE_BUNDLE_REPOSITORY_PATH),
        "preapproval_contract": _repository_descriptor(root, CONTRACT_REPOSITORY_PATH),
        "required_evidence": required,
        "evidence_payload_sha256": evidence_payload_sha,
        "human_approval": {
            "github_login": "phead198708",
            "issue_number": 264,
            "comment_id": 99001,
            "comment_url": comment_url,
            "approved_at_ts_ms": AUTHORIZED_AT_TS_MS,
            "github_comment_payload": _descriptor(evidence_root, github_path),
            "attestation": _descriptor(evidence_root, attestation_path),
        },
        "capital_base_usd": identity["capital_base_usd"],
        "requested_initial_capital_fraction": identity[
            "requested_initial_capital_fraction"
        ],
        "maximum_notional_usd": identity["maximum_notional_usd"],
        "maximum_realized_loss_usd": identity["maximum_realized_loss_usd"],
        "maximum_open_orders": identity["maximum_open_orders"],
        "market_allowlist": identity["market_allowlist"],
        "allowed_actions": identity["allowed_actions"],
        "one_trade_maximum_per_market": True,
        "authorized_at_ts_ms": identity["authorized_at_ts_ms"],
        "expires_at_ts_ms": identity["expires_at_ts_ms"],
        "maximum_signal_age_ms": identity["maximum_signal_age_ms"],
        "maximum_operator_heartbeat_age_ms": identity[
            "maximum_operator_heartbeat_age_ms"
        ],
        "explicit_human_approval_recorded": True,
        "micro_live_authorized": True,
        "micro_live_started": False,
        "live_trading_allowed": True,
        "wallet_signing_allowed": True,
        "polymarket_write_allowed": True,
        "capital_at_risk": True,
        "automatic_launch_allowed": False,
        "capital_increase_allowed": False,
        "executable": True,
    }


def _verified(fixture: dict[str, Any]):
    return verify_micro_live_authorization(
        _json_bytes(fixture["authorization"]),
        repository_root=fixture["root"],
        evidence_root=fixture["evidence_root"],
        now_ts_ms=fixture["now_ts_ms"],
    )


def _signal(**overrides: Any) -> dict[str, Any]:
    signal_payload = copy.deepcopy(BASE_SIGNAL_PAYLOAD)
    feature_row = copy.deepcopy(BASE_FEATURE_ROW)
    payload = {
        "signal_payload": signal_payload,
        "feature_row": feature_row,
        "market_identity_evidence": copy.deepcopy(BASE_MARKET_IDENTITY_EVIDENCE),
        "now_ts_ms": NOW_TS_MS,
        "operator_heartbeat_ts_ms": NOW_TS_MS - 50,
    }
    for key, value in overrides.items():
        if key in signal_payload:
            signal_payload[key] = value
            if key == "market_id":
                feature_row["market_id"] = value
                feature_row["condition_id"] = value
                signal_payload["market_identity"]["condition_id"] = value
            elif key == "slug":
                start = int(str(value).rsplit("-", maxsplit=1)[1]) * 1_000
                signal_payload["market_identity"].update(
                    {
                        "slug": value,
                        "market_start_ts_ms": start,
                        "market_end_ts_ms": start + 900_000,
                    }
                )
            elif key in {"up_token_id", "down_token_id"}:
                signal_payload["market_identity"][key] = value
            elif key == "observed_at_ts_ms":
                signal_payload["market_identity"]["clob_revalidated_at_ts_ms"] = value
        else:
            payload[key] = value
    payload["market_identity_evidence"] = _market_identity_evidence(signal_payload)
    return payload


def _bind_signal_to_runtime(
    payload: dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    signal = payload["signal_payload"]
    feature_row = payload["feature_row"]
    result = runtime.score_feature_row(
        feature_row,
        observed_at_ts=signal["observed_at_ts_ms"],
    )
    for name in (
        "action_values",
        "selected_action",
        "model_scored",
        "fail_closed",
        "fail_closed_reasons",
    ):
        signal[name] = result[name]
    raw = feature_row["features"]
    signal["executable_asks"] = {
        "UP": str(raw["up_ask"]),
        "DOWN": str(raw["down_ask"]),
    }
    return payload


def test_current_template_cannot_create_executor(
    authorized_fixture: dict[str, Any],
) -> None:
    template = _json(authorized_fixture["root"] / AUTHORIZATION_TEMPLATE_PATH)
    transport = FakeTransport()
    with pytest.raises(MicroLiveAuthorizationError, match="schema is not exact"):
        create_micro_live_executor(
            raw_authorization=_json_bytes(template),
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
            transport=transport,
        )
    assert transport.submit_calls == []
    assert transport.cancel_calls == []


def test_executor_bundles_no_network_wallet_or_credential_adapter() -> None:
    paths = (
        REPO_ROOT
        / "src/bigan/v8/polymarket/residual_promotion_micro_live_authorization.py",
        REPO_ROOT / "src/bigan/v8/polymarket/residual_promotion_micro_live_executor.py",
    )
    forbidden_modules = {
        "eth_account",
        "httpx",
        "py_clob_client",
        "requests",
        "socket",
        "urllib",
        "web3",
    }
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name.split(".", maxsplit=1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (
                node.names
                if isinstance(node, ast.Import)
                else [ast.alias(name=str(node.module or ""))]
            )
        }
        assert imports.isdisjoint(forbidden_modules)
        assert "os.environ" not in source
        assert "getenv(" not in source


def test_valid_graph_creates_capability_but_does_not_auto_launch(
    authorized_fixture: dict[str, Any],
) -> None:
    transport = FakeTransport()
    executor = create_micro_live_executor(
        raw_authorization=_json_bytes(authorized_fixture["authorization"]),
        repository_root=authorized_fixture["root"],
        evidence_root=authorized_fixture["evidence_root"],
        now_ts_ms=NOW_TS_MS,
        transport=transport,
    )
    assert executor.events == ()
    assert executor.reconciliation_snapshot()["cash_usd"] == "10.00"
    assert transport.submit_calls == []


def test_authorization_requires_strict_raw_bytes_and_binds_exact_payload(
    authorized_fixture: dict[str, Any],
) -> None:
    authorization = authorized_fixture["authorization"]
    raw_authorization = _json_bytes(authorization)
    verified = _verified(authorized_fixture)
    assert verified.authorization_payload_sha256 == hashlib.sha256(
        raw_authorization
    ).hexdigest()

    with pytest.raises(MicroLiveAuthorizationError, match="raw bytes are invalid"):
        verify_micro_live_authorization(
            authorization,
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
        )

    duplicate_key = raw_authorization[:-1] + b',"executable":true}'
    with pytest.raises(MicroLiveAuthorizationError, match="strict JSON"):
        verify_micro_live_authorization(
            duplicate_key,
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
        )

    numeric_overflow = raw_authorization[:-1] + b',"ambiguous":1e400}'
    with pytest.raises(MicroLiveAuthorizationError, match="strict JSON"):
        verify_micro_live_authorization(
            numeric_overflow,
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
        )


def test_submit_is_idempotent_and_one_market_has_one_intent(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    signal = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    first = executor.submit_signal(**signal)
    replay = executor.submit_signal(**signal)
    assert first["status"] == "ORDER_ACKNOWLEDGED"
    assert replay["status"] == "IDEMPOTENT_REPLAY"
    assert replay["client_order_id"] == first["client_order_id"]
    assert len(transport.submit_calls) == 1
    audit_events = [
        event for event in executor.events if event["event_type"] == "SIGNAL_EVALUATED"
    ]
    assert [event["payload"]["disposition"] for event in audit_events] == [
        "EXECUTION_INTENT",
        "IDEMPOTENT_REPLAY",
    ]
    assert sum(event["event_type"] == "ORDER_PREPARED" for event in executor.events) == 1


def test_production_signal_boundary_requires_and_replays_exact_raw_bytes(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    payload = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    raw_signal = _json_bytes(payload["signal_payload"])
    raw_feature = _json_bytes(payload["feature_row"])
    executor = _StrictMicroLiveExecutor(verified, transport=FakeTransport())
    result = executor.submit_signal(
        raw_signal_payload=raw_signal,
        raw_feature_row=raw_feature,
        provider_feature_evidence=BASE_PROVIDER_FEATURE_EVIDENCE,
        now_ts_ms=payload["now_ts_ms"],
        operator_heartbeat_ts_ms=payload["operator_heartbeat_ts_ms"],
        market_identity_evidence=payload["market_identity_evidence"],
    )
    assert result["status"] == "ORDER_ACKNOWLEDGED"
    evaluated = next(
        event["payload"]
        for event in executor.events
        if event["event_type"] == "SIGNAL_EVALUATED"
    )
    assert evaluated["raw_signal_payload_sha256"] == hashlib.sha256(
        raw_signal
    ).hexdigest()
    assert evaluated["raw_signal_payload_json"].encode() == raw_signal
    assert evaluated["raw_feature_row_sha256"] == hashlib.sha256(
        raw_feature
    ).hexdigest()
    assert evaluated["raw_feature_row_json"].encode() == raw_feature
    provider_verification = verify_provider_feature_evidence(
        raw_evidence=BASE_PROVIDER_FEATURE_EVIDENCE,
        signal=payload["signal_payload"],
        feature_row=payload["feature_row"],
    )
    assert (
        evaluated["provider_feature_evidence_graph_sha256"]
        == provider_verification.evidence_graph_sha256
    )
    assert (
        evaluated["provider_feature_file_sha256"]
        == provider_verification.file_sha256
    )
    assert evaluated["raw_provider_feature_evidence_jsonl"] == {
        name: raw.decode()
        for name, raw in BASE_PROVIDER_FEATURE_EVIDENCE.items()
    }
    restored = _StrictMicroLiveExecutor.restore(
        authorization=verified,
        transport=FakeTransport(),
        raw_state=executor.export_state_bytes(),
    )
    assert restored.export_state_bytes() == executor.export_state_bytes()

    tampered_state = executor.export_state()
    evaluated_event = next(
        event
        for event in tampered_state["events"]
        if event["event_type"] == "SIGNAL_EVALUATED"
    )
    changed_raw_feature = json.loads(
        evaluated_event["payload"]["raw_feature_row_json"]
    )
    changed_raw_feature["benign_extra"] = True
    changed_raw_feature_bytes = _json_bytes(changed_raw_feature)
    evaluated_event["payload"]["raw_feature_row_json"] = (
        changed_raw_feature_bytes.decode()
    )
    evaluated_event["payload"]["raw_feature_row_sha256"] = hashlib.sha256(
        changed_raw_feature_bytes
    ).hexdigest()
    audit_core = {
        key: value
        for key, value in evaluated_event["payload"].items()
        if key != "decision_audit_sha256"
    }
    evaluated_event["payload"]["decision_audit_sha256"] = canonical_json_sha256(
        audit_core
    )
    previous = "GENESIS"
    for event in tampered_state["events"]:
        event["previous_event_sha256"] = previous
        event_core = {
            key: value for key, value in event.items() if key != "event_sha256"
        }
        event["event_sha256"] = canonical_json_sha256(event_core)
        previous = event["event_sha256"]
    state_core = {
        key: value
        for key, value in tampered_state.items()
        if key != "state_sha256"
    }
    tampered_state["state_sha256"] = canonical_json_sha256(state_core)
    with pytest.raises(MicroLiveExecutionError, match="do not match semantic"):
        _StrictMicroLiveExecutor.restore(
            authorization=verified,
            transport=FakeTransport(),
            raw_state=_json_bytes(tampered_state),
        )

    provider_tampered_state = executor.export_state()
    provider_evaluated_event = next(
        event
        for event in provider_tampered_state["events"]
        if event["event_type"] == "SIGNAL_EVALUATED"
    )
    provider_evaluated_event["payload"]["raw_provider_feature_evidence_jsonl"][
        "raw_polymarket_chainlink_prices.jsonl"
    ] = '{"available_at_ts":1,"price":1,"ts":1}\n'
    provider_audit_core = {
        key: value
        for key, value in provider_evaluated_event["payload"].items()
        if key != "decision_audit_sha256"
    }
    provider_evaluated_event["payload"]["decision_audit_sha256"] = (
        canonical_json_sha256(provider_audit_core)
    )
    previous = "GENESIS"
    for event in provider_tampered_state["events"]:
        event["previous_event_sha256"] = previous
        event_core = {
            key: value for key, value in event.items() if key != "event_sha256"
        }
        event["event_sha256"] = canonical_json_sha256(event_core)
        previous = event["event_sha256"]
    provider_state_core = {
        key: value
        for key, value in provider_tampered_state.items()
        if key != "state_sha256"
    }
    provider_tampered_state["state_sha256"] = canonical_json_sha256(
        provider_state_core
    )
    with pytest.raises(
        MicroLiveExecutionError,
        match="stored provider feature evidence",
    ):
        _StrictMicroLiveExecutor.restore(
            authorization=verified,
            transport=FakeTransport(),
            raw_state=_json_bytes(provider_tampered_state),
        )

    parsed = _StrictMicroLiveExecutor(verified, transport=FakeTransport())
    with pytest.raises(MicroLiveExecutionError, match="raw bytes are invalid"):
        parsed.submit_signal(
            raw_signal_payload=payload["signal_payload"],
            raw_feature_row=raw_feature,
            provider_feature_evidence=BASE_PROVIDER_FEATURE_EVIDENCE,
            now_ts_ms=payload["now_ts_ms"],
            operator_heartbeat_ts_ms=payload["operator_heartbeat_ts_ms"],
            market_identity_evidence=payload["market_identity_evidence"],
        )

    duplicate = _StrictMicroLiveExecutor(verified, transport=FakeTransport())
    duplicate_signal = raw_signal[:-1] + b',"market_id":"0x' + b"2" * 64 + b'"}'
    with pytest.raises(MicroLiveExecutionError, match="strict JSON"):
        duplicate.submit_signal(
            raw_signal_payload=duplicate_signal,
            raw_feature_row=raw_feature,
            provider_feature_evidence=BASE_PROVIDER_FEATURE_EVIDENCE,
            now_ts_ms=payload["now_ts_ms"],
            operator_heartbeat_ts_ms=payload["operator_heartbeat_ts_ms"],
            market_identity_evidence=payload["market_identity_evidence"],
        )

    overflow = _StrictMicroLiveExecutor(verified, transport=FakeTransport())
    overflow_feature = raw_feature[:-1] + b',"ambiguous":1e400}'
    with pytest.raises(MicroLiveExecutionError, match="strict JSON"):
        overflow.submit_signal(
            raw_signal_payload=raw_signal,
            raw_feature_row=overflow_feature,
            provider_feature_evidence=BASE_PROVIDER_FEATURE_EVIDENCE,
            now_ts_ms=payload["now_ts_ms"],
            operator_heartbeat_ts_ms=payload["operator_heartbeat_ts_ms"],
            market_identity_evidence=payload["market_identity_evidence"],
        )


def test_provider_feature_evidence_reconstructs_exact_deterministic_row() -> None:
    first = verify_provider_feature_evidence(
        raw_evidence=BASE_PROVIDER_FEATURE_EVIDENCE,
        signal=BASE_SIGNAL_PAYLOAD,
        feature_row=BASE_FEATURE_ROW,
    )
    second = verify_provider_feature_evidence(
        raw_evidence=BASE_PROVIDER_FEATURE_EVIDENCE,
        signal=BASE_SIGNAL_PAYLOAD,
        feature_row=BASE_FEATURE_ROW,
    )
    assert first == second
    assert first.reconstructed_feature_row_sha256 == canonical_json_sha256(
        BASE_FEATURE_ROW
    )
    assert first.file_sha256 == {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in BASE_PROVIDER_FEATURE_EVIDENCE.items()
    }
    assert len(build_provider_bound_feature_rows(BASE_PROVIDER_FEATURE_EVIDENCE)) == 3


def test_provider_feature_evidence_schema_and_raw_types_fail_closed() -> None:
    missing = dict(BASE_PROVIDER_FEATURE_EVIDENCE)
    missing.pop("raw_polymarket_chainlink_prices.jsonl")
    extra = {**BASE_PROVIDER_FEATURE_EVIDENCE, "unexpected.jsonl": b""}
    semantic_not_raw = dict(BASE_PROVIDER_FEATURE_EVIDENCE)
    semantic_not_raw["raw_polymarket_chainlink_prices.jsonl"] = ""
    for evidence in (missing, extra, semantic_not_raw):
        with pytest.raises(ProviderFeatureEvidenceError):
            build_provider_bound_feature_rows(evidence)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ambiguous_raw",
    (
        b'{"x":1,"x":2}\n',
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
        b'{"x":1e400}\n',
        b"\xff\n",
    ),
)
def test_provider_feature_evidence_rejects_ambiguous_json(
    ambiguous_raw: bytes,
) -> None:
    evidence = dict(BASE_PROVIDER_FEATURE_EVIDENCE)
    evidence["raw_polymarket_chainlink_prices.jsonl"] = ambiguous_raw
    with pytest.raises(ProviderFeatureEvidenceError):
        build_provider_bound_feature_rows(evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("resolved_outcome", "UP"),
        ("realized_pnl", 1.0),
        ("training_label", 1),
        ("wallet_signing_allowed", True),
        ("outcomes_accessed", True),
        ("settlement_accessed", True),
        ("pnl_accessed", True),
    ),
)
def test_provider_feature_evidence_forbids_results_and_safety_unlocks(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ProviderFeatureEvidenceError):
        build_provider_bound_feature_rows(
            _provider_evidence_with_market_patch(**{field: value})
        )


def test_provider_feature_evidence_binds_feature_and_market_identity() -> None:
    feature_drift = copy.deepcopy(BASE_FEATURE_ROW)
    feature_drift["features"]["up_ask"] += 0.01
    with pytest.raises(ProviderFeatureEvidenceError, match="does not match submitted"):
        verify_provider_feature_evidence(
            raw_evidence=BASE_PROVIDER_FEATURE_EVIDENCE,
            signal=BASE_SIGNAL_PAYLOAD,
            feature_row=feature_drift,
        )
    wrong_slug = copy.deepcopy(BASE_SIGNAL_PAYLOAD)
    wrong_slug["slug"] = "btc-updown-15m-1789949700"
    with pytest.raises(ProviderFeatureEvidenceError):
        verify_provider_feature_evidence(
            raw_evidence=BASE_PROVIDER_FEATURE_EVIDENCE,
            signal=wrong_slug,
            feature_row=BASE_FEATURE_ROW,
        )
    wrong_token = copy.deepcopy(BASE_SIGNAL_PAYLOAD)
    wrong_token["up_token_id"] = "99999"
    with pytest.raises(ProviderFeatureEvidenceError, match="identity"):
        verify_provider_feature_evidence(
            raw_evidence=BASE_PROVIDER_FEATURE_EVIDENCE,
            signal=wrong_token,
            feature_row=BASE_FEATURE_ROW,
        )


def test_provider_feature_evidence_requires_a_decision_time_causal_prefix() -> None:
    decision_ts = int(BASE_SIGNAL_PAYLOAD["decision_ts_ms"])
    for filename in PROVIDER_FEATURE_FILENAMES[1:]:
        for line in BASE_PROVIDER_FEATURE_EVIDENCE[filename].decode().splitlines():
            row = json.loads(line)
            assert int(row["available_at_ts"]) <= decision_ts

    full_round_status = _provider_evidence_with_market_patch(
        trade_full_round_coverage_complete=True
    )
    with pytest.raises(ProviderFeatureEvidenceError, match="terminal coverage"):
        verify_provider_feature_evidence(
            raw_evidence=full_round_status,
            signal=BASE_SIGNAL_PAYLOAD,
            feature_row=BASE_FEATURE_ROW,
        )

    future_market_metadata = _provider_evidence_with_market_patch(
        trade_stream_ended_at_ts=decision_ts + 1
    )
    with pytest.raises(ProviderFeatureEvidenceError, match="post-decision"):
        verify_provider_feature_evidence(
            raw_evidence=future_market_metadata,
            signal=BASE_SIGNAL_PAYLOAD,
            feature_row=BASE_FEATURE_ROW,
        )

    future_book = dict(BASE_PROVIDER_FEATURE_EVIDENCE)
    book_rows = [
        json.loads(line)
        for line in future_book["raw_polymarket_orderbooks.jsonl"].decode().splitlines()
    ]
    post_decision = copy.deepcopy(book_rows[-1])
    post_decision["ts"] = decision_ts + 1
    post_decision["available_at_ts"] = decision_ts + 1
    book_rows.append(post_decision)
    future_book["raw_polymarket_orderbooks.jsonl"] = b"".join(
        _json_bytes(row) + b"\n" for row in book_rows
    )
    with pytest.raises(ProviderFeatureEvidenceError, match="post-decision"):
        verify_provider_feature_evidence(
            raw_evidence=future_book,
            signal=BASE_SIGNAL_PAYLOAD,
            feature_row=BASE_FEATURE_ROW,
        )

    missing_availability = dict(BASE_PROVIDER_FEATURE_EVIDENCE)
    chainlink_row = {"price": 100_000.0, "source_ts": decision_ts}
    missing_availability["raw_polymarket_chainlink_prices.jsonl"] = (
        _json_bytes(chainlink_row) + b"\n"
    )
    with pytest.raises(ProviderFeatureEvidenceError, match="timestamp is invalid"):
        verify_provider_feature_evidence(
            raw_evidence=missing_availability,
            signal=BASE_SIGNAL_PAYLOAD,
            feature_row=BASE_FEATURE_ROW,
        )


def test_conflicting_duplicate_engages_kill_switch_and_cancels_open_order(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    signal = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    executor.submit_signal(**signal)
    conflict = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256,
        down_token_id="54321",
    )
    with pytest.raises(MicroLiveExecutionError, match="conflicting duplicate"):
        executor.submit_signal(**conflict)
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["kill_switch_reason"] == "conflicting_duplicate_intent"
    assert len(transport.cancel_calls) == 1
    assert snapshot["open_order_count"] == 0


def test_exchange_order_identity_reuse_fails_closed_before_acknowledgement(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport(fixed_exchange_order_id="exchange-shared")
    executor = MicroLiveExecutor(verified, transport=transport)
    first = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    assert first["status"] == "ORDER_ACKNOWLEDGED"

    with pytest.raises(MicroLiveExecutionError, match="submission became unknown"):
        executor.submit_signal(
            **_signal(
                candidate_bundle_sha256=verified.candidate_bundle_sha256,
                market_id=f"0x{222:064x}",
                up_token_id="62220",
                down_token_id="62221",
            )
        )
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["kill_switch_reason"] == "order_submission_unknown"
    assert len(transport.submit_calls) == 2
    assert len(transport.cancel_calls) == 1
    assert sum(
        event["event_type"] == "ORDER_ACKNOWLEDGED" for event in executor.events
    ) == 1


def test_allowlist_no_trade_candidate_and_token_contract_block_without_transport(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    failed_closed = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256
    )
    failed_closed["feature_row"]["max_input_ts"] -= 10_000
    failed_closed = _bind_signal_to_runtime(failed_closed, verified.runtime)
    with pytest.raises(MicroLiveExecutionError, match="not bound to provider bytes"):
        executor.submit_signal(**failed_closed)
    executor = MicroLiveExecutor(verified, transport=transport)
    with pytest.raises(MicroLiveExecutionError, match="identity or safety"):
        executor.submit_signal(
            **_signal(
                candidate_bundle_sha256=verified.candidate_bundle_sha256,
                market_family="ETH-15M",
            )
        )
    with pytest.raises(MicroLiveExecutionError, match="identity or safety"):
        executor.submit_signal(
            **_signal(candidate_bundle_sha256="0" * 64)
        )
    duplicate_tokens = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256,
        up_token_id="12345",
        down_token_id="12345",
    )
    with pytest.raises(MicroLiveExecutionError, match="identity or safety"):
        executor.submit_signal(**duplicate_tokens)
    assert transport.submit_calls == []
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True


def test_signal_envelope_tampering_and_outcome_fields_fail_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)

    mismatched = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    mismatched["signal_payload"]["selected_action"] = "BUY_DOWN_HOLD"
    with pytest.raises(MicroLiveExecutionError, match="zero-threshold decision"):
        executor.submit_signal(**mismatched)

    internally_coherent_but_fabricated = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256,
        action_values={
            "NO_TRADE": 0.0,
            "BUY_UP_HOLD": -0.25,
            "BUY_DOWN_HOLD": 0.25,
        },
    )
    internally_coherent_but_fabricated["signal_payload"][
        "selected_action"
    ] = "BUY_DOWN_HOLD"
    with pytest.raises(MicroLiveExecutionError, match="does not match frozen runtime"):
        executor.submit_signal(**internally_coherent_but_fabricated)

    outcome_bearing = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256
    )
    outcome_bearing["signal_payload"]["outcome"] = "UP"
    with pytest.raises(MicroLiveExecutionError, match="schema is not exact"):
        executor.submit_signal(**outcome_bearing)

    outcome_opened = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256,
        outcomes_accessed=True,
    )
    with pytest.raises(MicroLiveExecutionError, match="identity or safety"):
        executor.submit_signal(**outcome_opened)

    outcome_feature = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256
    )
    outcome_feature["feature_row"]["settlement_price"] = 1.0
    with pytest.raises(MicroLiveExecutionError, match="forbidden field"):
        executor.submit_signal(**outcome_feature)

    off_schedule = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256,
        decision_ts_ms=AUTHORIZED_AT_TS_MS + 300_001,
        observed_at_ts_ms=AUTHORIZED_AT_TS_MS + 300_001,
    )
    with pytest.raises(MicroLiveExecutionError, match="frozen schedule"):
        executor.submit_signal(**off_schedule)

    assert transport.submit_calls == []
    assert transport.cancel_calls == []
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True


def test_verified_capability_cannot_be_derived_with_changed_limits(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    derived = replace(verified, maximum_notional_usd=Decimal("999"))
    assert authorization_capability_is_verified(verified) is True
    assert authorization_capability_is_verified(derived) is False
    with pytest.raises(MicroLiveExecutionError, match="capability is unverified"):
        MicroLiveExecutor(derived, transport=FakeTransport())


def test_verified_capability_shallow_copy_is_not_registered(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    copied = copy.copy(verified)
    assert copied is not verified
    assert authorization_capability_is_verified(verified) is True
    assert authorization_capability_is_verified(copied) is False
    with pytest.raises(MicroLiveExecutionError, match="capability is unverified"):
        MicroLiveExecutor(copied, transport=FakeTransport())


def test_verified_capability_in_place_tampering_invalidates_integrity(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    assert authorization_capability_is_verified(verified) is True
    object.__setattr__(verified, "maximum_realized_loss_usd", Decimal("999"))
    assert authorization_capability_is_verified(verified) is False
    with pytest.raises(MicroLiveExecutionError, match="capability is unverified"):
        MicroLiveExecutor(verified, transport=FakeTransport())


def test_post_construction_capability_tampering_kills_and_cancels_open_order(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    object.__setattr__(verified, "maximum_notional_usd", Decimal("999"))

    with pytest.raises(
        MicroLiveExecutionError,
        match="capability changed after executor construction",
    ):
        executor.enforce_runtime_safety(
            now_ts_ms=NOW_TS_MS + 1,
            operator_heartbeat_ts_ms=NOW_TS_MS,
        )

    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["kill_switch_reason"] == "authorization_capability_integrity_failed"
    assert snapshot["maximum_realized_loss_usd"] == "1.00"
    assert len(transport.submit_calls) == 1
    assert len(transport.cancel_calls) == 1
    assert transport.cancel_calls[0]["authorization_id"] == verified.authorization_id
    late_fill = _record_fill(
        executor,
        client_order_id=transport.submit_calls[0]["client_order_id"],
        fill_id="post-tamper-reconciled-fill",
        now_ts_ms=NOW_TS_MS + 2,
        quantity="1",
        price="0.39",
        fee_usd="0.0002",
        transport_event_sha256="f" * 64,
        event_overrides={"executed_at_ts_ms": NOW_TS_MS},
    )
    assert late_fill["status"] == "FILL_RECORDED"
    assert late_fill["snapshot"]["kill_switch_active"] is True
    assert late_fill["snapshot"]["open_order_count"] == 0


def test_post_construction_capability_tampering_blocks_first_submission(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    object.__setattr__(verified, "maximum_open_orders", 10)

    with pytest.raises(
        MicroLiveExecutionError,
        match="capability changed after executor construction",
    ):
        executor.submit_signal(
            **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
        )

    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["kill_switch_reason"] == "authorization_capability_integrity_failed"
    assert transport.submit_calls == []
    assert transport.cancel_calls == []


def test_post_construction_runtime_tampering_uses_bound_clone_to_cancel(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    object.__setattr__(
        verified.runtime,
        "maximum_source_age_ms",
        verified.runtime.maximum_source_age_ms + 1,
    )

    with pytest.raises(
        MicroLiveExecutionError,
        match="capability changed after executor construction",
    ):
        executor.enforce_runtime_safety(
            now_ts_ms=NOW_TS_MS + 1,
            operator_heartbeat_ts_ms=NOW_TS_MS,
        )

    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["kill_switch_reason"] == "authorization_capability_integrity_failed"
    assert len(transport.submit_calls) == 1
    assert len(transport.cancel_calls) == 1


def test_second_frozen_decision_is_audited_and_blocked_without_kill(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    first = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    assert executor.submit_signal(**first)["status"] == "ORDER_ACKNOWLEDGED"

    second = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    second_ts = AUTHORIZED_AT_TS_MS + 600_000
    second["signal_payload"]["decision_ts_ms"] = second_ts
    second["signal_payload"]["observed_at_ts_ms"] = second_ts
    second["signal_payload"]["market_identity"][
        "clob_revalidated_at_ts_ms"
    ] = second_ts
    second["feature_row"] = copy.deepcopy(BASE_PROVIDER_FEATURE_ROWS[second_ts])
    second["now_ts_ms"] = second_ts + 1_000
    second["operator_heartbeat_ts_ms"] = second_ts + 950
    second = _bind_signal_to_runtime(second, verified.runtime)
    blocked = executor.submit_signal(**second)
    assert blocked["reason"] == "one_trade_maximum_per_market"
    assert len(transport.submit_calls) == 1
    assert executor.reconciliation_snapshot()["kill_switch_active"] is False
    audited = [
        event["payload"]
        for event in executor.events
        if event["event_type"] == "SIGNAL_EVALUATED"
    ]
    assert audited[-1]["disposition"] == "BLOCKED_NO_TRADE"
    assert audited[-1]["reason"] == "one_trade_maximum_per_market"


def test_market_identity_token_binding_and_live_revalidation_fail_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)

    mismatched_transport = FakeTransport()
    mismatched_executor = MicroLiveExecutor(verified, transport=mismatched_transport)
    mismatched = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    mismatched["signal_payload"]["market_identity"]["down_token_id"] = "99999"
    with pytest.raises(MicroLiveExecutionError, match="identity binding"):
        mismatched_executor.submit_signal(**mismatched)
    assert mismatched_transport.submit_calls == []
    assert mismatched_executor.reconciliation_snapshot()["kill_switch_active"] is True

    stale_transport = FakeTransport()
    stale_executor = MicroLiveExecutor(verified, transport=stale_transport)
    stale = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    stale["signal_payload"]["market_identity"]["clob_revalidated_at_ts_ms"] = (
        NOW_TS_MS - 5_001
    )
    with pytest.raises(MicroLiveExecutionError, match="market identity is stale"):
        stale_executor.submit_signal(**stale)
    assert stale_transport.submit_calls == []
    assert stale_executor.reconciliation_snapshot()["kill_switch_reason"] == (
        "market_identity_stale"
    )


def test_market_identity_raw_provider_bytes_are_mandatory_and_verified(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)

    missing = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    missing.pop("market_identity_evidence")
    missing_executor = MicroLiveExecutor(verified, transport=FakeTransport())
    with pytest.raises(MicroLiveExecutionError, match="evidence schema"):
        missing_executor.submit_signal(**missing)
    assert missing_executor.reconciliation_snapshot()["kill_switch_active"] is True

    byte_drift = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    byte_drift["market_identity_evidence"]["raw_gamma_payload"] += b"\n"
    drift_executor = MicroLiveExecutor(verified, transport=FakeTransport())
    with pytest.raises(MicroLiveExecutionError, match="raw-byte SHA-256 mismatch"):
        drift_executor.submit_signal(**byte_drift)
    assert drift_executor.reconciliation_snapshot()["kill_switch_active"] is True

    semantic_drift = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    clob = json.loads(
        semantic_drift["market_identity_evidence"][
            "raw_clob_revalidation_payload"
        ]
    )
    clob["tokens"][1]["token_id"] = "99999"
    clob_raw = json.dumps(clob, separators=(",", ":"), sort_keys=True).encode()
    semantic_drift["market_identity_evidence"][
        "raw_clob_revalidation_payload"
    ] = clob_raw
    semantic_drift["signal_payload"]["market_identity"][
        "clob_revalidation_payload_sha256"
    ] = hashlib.sha256(clob_raw).hexdigest()
    semantic_executor = MicroLiveExecutor(verified, transport=FakeTransport())
    with pytest.raises(MicroLiveExecutionError, match="CLOB condition, slug, or token"):
        semantic_executor.submit_signal(**semantic_drift)
    assert semantic_executor.reconciliation_snapshot()["kill_switch_active"] is True


def test_market_identity_provider_json_ambiguity_fails_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)

    duplicate = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    gamma = json.loads(duplicate["market_identity_evidence"]["raw_gamma_payload"])
    row_json = json.dumps(gamma[0], separators=(",", ":"), sort_keys=True)
    duplicate_raw = (
        '[{"conditionId":"0x'
        + "0" * 64
        + '",'
        + row_json[1:]
        + "]"
    ).encode()
    duplicate["market_identity_evidence"]["raw_gamma_payload"] = duplicate_raw
    duplicate["signal_payload"]["market_identity"]["raw_gamma_payload_sha256"] = (
        hashlib.sha256(duplicate_raw).hexdigest()
    )
    duplicate_transport = FakeTransport()
    duplicate_executor = MicroLiveExecutor(verified, transport=duplicate_transport)
    with pytest.raises(MicroLiveExecutionError, match="strict JSON"):
        duplicate_executor.submit_signal(**duplicate)
    assert duplicate_transport.submit_calls == []
    assert duplicate_executor.reconciliation_snapshot()["kill_switch_active"] is True

    for number in (b"NaN", b"1e400"):
        nonfinite = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
        clob_raw = nonfinite["market_identity_evidence"][
            "raw_clob_revalidation_payload"
        ]
        assert clob_raw.endswith(b"}")
        nonfinite_raw = clob_raw[:-1] + b',"diagnostic":' + number + b"}"
        nonfinite["market_identity_evidence"][
            "raw_clob_revalidation_payload"
        ] = nonfinite_raw
        nonfinite["signal_payload"]["market_identity"][
            "clob_revalidation_payload_sha256"
        ] = hashlib.sha256(nonfinite_raw).hexdigest()
        nonfinite_transport = FakeTransport()
        nonfinite_executor = MicroLiveExecutor(verified, transport=nonfinite_transport)
        with pytest.raises(MicroLiveExecutionError, match="strict JSON"):
            nonfinite_executor.submit_signal(**nonfinite)
        assert nonfinite_transport.submit_calls == []
        assert nonfinite_executor.reconciliation_snapshot()["kill_switch_active"] is True


def test_open_order_and_authorization_lifetime_notional_caps(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    for index in (1, 2):
        result = executor.submit_signal(
            **_signal(
                candidate_bundle_sha256=verified.candidate_bundle_sha256,
                market_id=f"0x{index:064x}",
                up_token_id=str(10_000 + index * 2),
                down_token_id=str(10_001 + index * 2),
            )
        )
        assert result["status"] == "ORDER_ACKNOWLEDGED"
    blocked = executor.submit_signal(
        **_signal(
            candidate_bundle_sha256=verified.candidate_bundle_sha256,
            market_id=f"0x{3:064x}",
            up_token_id="10006",
            down_token_id="10007",
        )
    )
    assert blocked["reason"] == "maximum_open_orders_reached"
    assert len(transport.submit_calls) == 2

    lifetime_transport = FakeTransport()
    lifetime = MicroLiveExecutor(verified, transport=lifetime_transport)
    selected_side = (
        "UP"
        if BASE_SIGNAL_PAYLOAD["selected_action"] == "BUY_UP_HOLD"
        else "DOWN"
    )
    unit_notional = Decimal(BASE_SIGNAL_PAYLOAD["executable_asks"][selected_side])
    allowed_count = int(verified.maximum_notional_usd // unit_notional)
    for index in range(1, allowed_count + 1):
        result = lifetime.submit_signal(
            **_signal(
                candidate_bundle_sha256=verified.candidate_bundle_sha256,
                market_id=f"0x{index:064x}",
                up_token_id=str(20_000 + index * 2),
                down_token_id=str(20_001 + index * 2),
            )
        )
        assert result["status"] == "ORDER_ACKNOWLEDGED"
        _record_order_closed(
            lifetime,
            client_order_id=result["client_order_id"],
            status="CANCELED",
            now_ts_ms=NOW_TS_MS,
            transport_event_sha256=f"{index:064x}",
        )
    capped = lifetime.submit_signal(
        **_signal(
            candidate_bundle_sha256=verified.candidate_bundle_sha256,
            market_id=f"0x{allowed_count + 1:064x}",
            up_token_id=str(20_000 + (allowed_count + 1) * 2),
            down_token_id=str(20_001 + (allowed_count + 1) * 2),
        )
    )
    assert capped["reason"] == "authorization_notional_cap_exceeded"
    assert len(lifetime_transport.submit_calls) == allowed_count


def test_stale_heartbeat_kills_and_cancels_existing_order(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    base = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    executor.submit_signal(**base)
    with pytest.raises(MicroLiveExecutionError, match="heartbeat is stale"):
        executor.submit_signal(
            **_signal(
                candidate_bundle_sha256=verified.candidate_bundle_sha256,
                operator_heartbeat_ts_ms=NOW_TS_MS - 6_000,
            )
        )
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True
    assert len(transport.cancel_calls) == 1


def test_runtime_watchdog_checks_heartbeat_without_waiting_for_a_signal(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    base = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    executor.submit_signal(**base)

    healthy = executor.enforce_runtime_safety(
        now_ts_ms=NOW_TS_MS + 1_000,
        operator_heartbeat_ts_ms=NOW_TS_MS + 950,
    )
    assert healthy == {
        "status": "RUNTIME_SAFETY_OK",
        "checked_at_ts_ms": NOW_TS_MS + 1_000,
        "operator_heartbeat_ts_ms": NOW_TS_MS + 950,
        "transport_called": False,
    }
    assert executor.reconciliation_snapshot()["kill_switch_active"] is False

    killed = executor.enforce_runtime_safety(
        now_ts_ms=NOW_TS_MS + 7_000,
        operator_heartbeat_ts_ms=NOW_TS_MS + 1_000,
    )
    assert killed["status"] == "KILL_SWITCH_ENGAGED"
    assert killed["reason"] == "operator_heartbeat_stale"
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True
    assert len(transport.cancel_calls) == 1


def test_runtime_watchdog_invalid_clock_persists_kill(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    with pytest.raises(MicroLiveExecutionError, match="watchdog timestamp is invalid"):
        executor.enforce_runtime_safety(
            now_ts_ms=0,
            operator_heartbeat_ts_ms=NOW_TS_MS,
        )
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["kill_switch_reason"] == "runtime_watchdog_clock_invalid"


def test_signal_event_clock_regression_persists_kill_and_rejection_audit(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    with pytest.raises(MicroLiveExecutionError, match="event clock regressed"):
        executor.submit_signal(
            **_signal(
                candidate_bundle_sha256=verified.candidate_bundle_sha256,
                market_id=f"0x{333:064x}",
                up_token_id="53330",
                down_token_id="53331",
                now_ts_ms=NOW_TS_MS - 1,
            )
        )
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["kill_switch_reason"] == "event_clock_regression"
    assert len(transport.cancel_calls) == 1
    assert any(event["event_type"] == "SIGNAL_REJECTED" for event in executor.events)


def test_fill_cash_position_settlement_and_restart_reconcile(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    fill = _record_fill(
        executor,
        client_order_id=order["client_order_id"],
        fill_id="fill-001",
        now_ts_ms=NOW_TS_MS,
        quantity="1",
        price="0.39",
        fee_usd="0.0002",
        transport_event_sha256="f" * 64,
    )
    assert fill["snapshot"]["cash_usd"] == "9.6098"
    selected_side = str(BASE_SIGNAL_PAYLOAD["selected_action"]).split("_")[1]
    assert fill["snapshot"]["positions"][selected_side] == "1"
    assert _record_fill(
        executor,
        client_order_id=order["client_order_id"],
        fill_id="fill-001",
        now_ts_ms=NOW_TS_MS,
        quantity="1",
        price="0.39",
        fee_usd="0.0002",
        transport_event_sha256="f" * 64,
    )["status"] == "IDEMPOTENT_FILL_REPLAY"
    settled = _record_settlement(
        executor,
        client_order_id=order["client_order_id"],
        settlement_id="settlement-001",
        now_ts_ms=SETTLEMENT_NOW_TS_MS,
        payout_per_token="1",
        official_settlement_sha256="1" * 64,
    )
    assert settled["snapshot"]["cash_usd"] == "10.6098"
    assert settled["snapshot"]["realized_pnl_usd"] == "0.6098"
    assert settled["snapshot"]["positions"][selected_side] == "0"
    state = executor.export_state()
    restored = MicroLiveExecutor.restore(
        authorization=verified,
        transport=transport,
        raw_state=_json_bytes(state),
    )
    assert restored.export_state() == state
    assert restored.reconciliation_snapshot() == executor.reconciliation_snapshot()


def test_restart_requires_strict_raw_state_bytes_and_blocks_event_injection(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    raw_state = executor.export_state_bytes()
    assert raw_state == executor.export_state_bytes()
    assert json.loads(raw_state) == executor.export_state()

    with pytest.raises(MicroLiveExecutionError, match="raw bytes are invalid"):
        MicroLiveExecutor.restore(
            authorization=verified,
            transport=transport,
            raw_state=executor.export_state(),
        )

    state_sha256 = executor.export_state()["state_sha256"]
    duplicate_key_state = (
        raw_state[:-1]
        + f',"state_sha256":"{state_sha256}"}}'.encode()
    )
    with pytest.raises(MicroLiveExecutionError, match="strict JSON"):
        MicroLiveExecutor.restore(
            authorization=verified,
            transport=transport,
            raw_state=duplicate_key_state,
        )

    numeric_overflow_state = raw_state.replace(
        b'"events":[]',
        b'"events":[{"event_ts_ms":1e400}]',
    )
    assert numeric_overflow_state != raw_state
    with pytest.raises(MicroLiveExecutionError, match="strict JSON"):
        MicroLiveExecutor.restore(
            authorization=verified,
            transport=transport,
            raw_state=numeric_overflow_state,
        )

    with pytest.raises(TypeError, match="events"):
        MicroLiveExecutor(
            verified,
            transport=transport,
            events=executor.events,
        )
    with pytest.raises(MicroLiveExecutionError, match="strict raw-state restore"):
        executor._initialize(
            authorization=verified,
            transport=transport,
            events=({"event_type": "FORGED"},),
        )

    restored = MicroLiveExecutor.restore(
        authorization=verified,
        transport=transport,
        raw_state=raw_state,
    )
    assert restored.export_state_bytes() == raw_state


def test_loss_budget_reservation_prevents_realized_loss_cap_overshoot(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    assert verified.maximum_realized_loss_usd == Decimal("1.00")
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    orders = []
    for index in range(1, 3):
        order = executor.submit_signal(
            **_signal(
                candidate_bundle_sha256=verified.candidate_bundle_sha256,
                market_id=f"0x{index + 100:064x}",
                up_token_id=str(30_000 + index * 2),
                down_token_id=str(30_001 + index * 2),
            )
        )
        _record_fill(
            executor,
            client_order_id=order["client_order_id"],
            fill_id=f"loss-fill-{index}",
            now_ts_ms=NOW_TS_MS,
            quantity="1",
            price="0.39",
            fee_usd="0.0002",
            transport_event_sha256=f"{index + 10:064x}",
        )
        orders.append(order)
    blocked = executor.submit_signal(
        **_signal(
            candidate_bundle_sha256=verified.candidate_bundle_sha256,
            market_id=f"0x{999:064x}",
            up_token_id="41000",
            down_token_id="41001",
        )
    )
    assert blocked["reason"] == "maximum_loss_reservation_exceeded"
    for index, order in enumerate(orders, start=1):
        _record_settlement(
            executor,
            client_order_id=order["client_order_id"],
            settlement_id=f"loss-settlement-{index}",
            now_ts_ms=SETTLEMENT_NOW_TS_MS,
            payout_per_token="0",
            official_settlement_sha256=f"{index + 20:064x}",
        )
    snapshot = executor.reconciliation_snapshot()
    assert Decimal(snapshot["realized_pnl_usd"]) == Decimal("-0.7804")
    assert snapshot["maximum_realized_loss_usd"] == "1.00"
    assert snapshot["loss_budget_consumed_usd"] == "0.7804"
    assert snapshot["kill_switch_active"] is False
    assert len(transport.submit_calls) == 2


def test_exact_human_loss_limit_persistently_kills_at_boundary(
    authorized_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "future_evidence"
    shutil.copytree(authorized_fixture["evidence_root"], evidence_root)
    authorization = _authorization(
        authorized_fixture["root"],
        evidence_root,
        copy.deepcopy(authorized_fixture["authorization"]["required_evidence"]),
        maximum_realized_loss_usd="0.4302",
    )
    verified = verify_micro_live_authorization(
        _json_bytes(authorization),
        repository_root=authorized_fixture["root"],
        evidence_root=evidence_root,
        now_ts_ms=NOW_TS_MS,
    )
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    prepared = next(
        event["payload"]
        for event in executor.events
        if event["event_type"] == "ORDER_PREPARED"
    )
    assert prepared["maximum_loss_usd"] == "0.4302"
    _record_fill(
        executor,
        client_order_id=order["client_order_id"],
        fill_id="boundary-loss-fill",
        now_ts_ms=NOW_TS_MS,
        quantity="1",
        price="0.43",
        fee_usd="0.0002",
        transport_event_sha256="d" * 64,
    )
    settled = _record_settlement(
        executor,
        client_order_id=order["client_order_id"],
        settlement_id="boundary-loss-settlement",
        now_ts_ms=SETTLEMENT_NOW_TS_MS,
        payout_per_token="0",
        official_settlement_sha256="e" * 64,
    )
    assert settled["snapshot"]["realized_pnl_usd"] == "-0.4302"
    assert settled["snapshot"]["loss_budget_consumed_usd"] == "0.4302"
    assert settled["snapshot"]["kill_switch_active"] is True
    assert settled["snapshot"]["kill_switch_reason"] == (
        "maximum_realized_loss_reached"
    )


def test_fill_fee_above_frozen_cost_contract_persistently_kills(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    with pytest.raises(MicroLiveExecutionError, match="frozen execution contract"):
        _record_fill(
            executor,
            client_order_id=order["client_order_id"],
            fill_id="fee-drift-fill",
            now_ts_ms=NOW_TS_MS,
            quantity="1",
            price="0.39",
            fee_usd="0.0003",
            transport_event_sha256="f" * 64,
        )
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["fill_count"] == 0
    assert snapshot["kill_switch_reason"] == "fill_fee_above_frozen_contract"
    assert len(transport.cancel_calls) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("client_order_id", "wrong-client-order"),
        ("exchange_order_id", "wrong-exchange-order"),
        ("market_id", "0x" + "9" * 64),
        ("token_id", "999999"),
    ),
)
def test_fill_transport_identity_mismatch_fails_closed(
    authorized_fixture: dict[str, Any],
    field: str,
    value: str,
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    with pytest.raises(MicroLiveExecutionError, match="fill transport identity"):
        _record_fill(
            executor,
            client_order_id=order["client_order_id"],
            fill_id="identity-mismatch-fill",
            now_ts_ms=NOW_TS_MS,
            quantity="1",
            price="0.39",
            fee_usd="0.0002",
            transport_event_sha256="0" * 64,
            event_overrides={field: value},
        )
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["fill_count"] == 0
    assert snapshot["kill_switch_reason"] == "fill_reconciliation_failed"
    assert len(transport.cancel_calls) == 1


def test_fill_transport_duplicate_json_key_fails_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    prepared, acknowledgement = _order_identity(executor, order["client_order_id"])
    event = {
        "event_type": "FILL",
        "client_order_id": order["client_order_id"],
        "exchange_order_id": acknowledgement["exchange_order_id"],
        "fill_id": "duplicate-key-fill",
        "market_id": prepared["market_id"],
        "token_id": prepared["token_id"],
        "quantity": "1",
        "price": "0.39",
        "fee_usd": "0.0002",
        "executed_at_ts_ms": NOW_TS_MS,
    }
    event_json = json.dumps(event, separators=(",", ":"), sort_keys=True)
    duplicate_raw = ('{"market_id":"0x' + "0" * 64 + '",' + event_json[1:]).encode()
    with pytest.raises(MicroLiveExecutionError, match="strict JSON"):
        executor.record_fill(
            client_order_id=order["client_order_id"],
            fill_id="duplicate-key-fill",
            now_ts_ms=NOW_TS_MS,
            quantity="1",
            price="0.39",
            fee_usd="0.0002",
            raw_transport_event=duplicate_raw,
        )
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["fill_count"] == 0
    assert snapshot["kill_switch_reason"] == "fill_reconciliation_failed"
    assert len(transport.cancel_calls) == 1


def test_late_fill_before_close_effective_time_reconciles_but_after_close_kills(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    close_effective_ts_ms = NOW_TS_MS + 50
    _record_order_closed(
        executor,
        client_order_id=order["client_order_id"],
        status="CANCELED",
        now_ts_ms=NOW_TS_MS + 100,
        transport_event_sha256="0" * 64,
        event_overrides={"effective_at_ts_ms": close_effective_ts_ms},
    )
    reconciled = _record_fill(
        executor,
        client_order_id=order["client_order_id"],
        fill_id="late-observed-pre-close-fill",
        now_ts_ms=NOW_TS_MS + 200,
        quantity="1",
        price="0.39",
        fee_usd="0.0002",
        transport_event_sha256="0" * 64,
        event_overrides={"executed_at_ts_ms": close_effective_ts_ms - 1},
    )
    assert reconciled["status"] == "FILL_RECORDED"
    assert reconciled["snapshot"]["fill_count"] == 1
    assert reconciled["snapshot"]["open_order_count"] == 0

    rejected_transport = FakeTransport()
    rejected = MicroLiveExecutor(verified, transport=rejected_transport)
    rejected_order = rejected.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    _record_order_closed(
        rejected,
        client_order_id=rejected_order["client_order_id"],
        status="CANCELED",
        now_ts_ms=NOW_TS_MS + 100,
        transport_event_sha256="0" * 64,
        event_overrides={"effective_at_ts_ms": close_effective_ts_ms},
    )
    with pytest.raises(MicroLiveExecutionError, match="executed after order close"):
        _record_fill(
            rejected,
            client_order_id=rejected_order["client_order_id"],
            fill_id="post-close-fill",
            now_ts_ms=NOW_TS_MS + 200,
            quantity="1",
            price="0.39",
            fee_usd="0.0002",
            transport_event_sha256="0" * 64,
            event_overrides={"executed_at_ts_ms": close_effective_ts_ms + 1},
        )
    assert rejected.reconciliation_snapshot()["kill_switch_active"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("client_order_id", "wrong-client-order"),
        ("exchange_order_id", "wrong-exchange-order"),
        ("market_id", "0x" + "6" * 64),
        ("token_id", "999999"),
    ),
)
def test_order_close_transport_identity_mismatch_fails_closed(
    authorized_fixture: dict[str, Any],
    field: str,
    value: str,
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    with pytest.raises(MicroLiveExecutionError, match="close transport identity"):
        _record_order_closed(
            executor,
            client_order_id=order["client_order_id"],
            status="CANCELED",
            now_ts_ms=NOW_TS_MS + 100,
            transport_event_sha256="0" * 64,
            event_overrides={field: value},
        )
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["kill_switch_reason"] == "order_close_reconciliation_failed"
    assert snapshot["open_order_count"] == 0


def test_official_settlement_before_market_end_fails_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    _record_fill(
        executor,
        client_order_id=order["client_order_id"],
        fill_id="premature-settlement-fill",
        now_ts_ms=NOW_TS_MS,
        quantity="1",
        price="0.39",
        fee_usd="0.0002",
        transport_event_sha256="0" * 64,
    )
    with pytest.raises(MicroLiveExecutionError, match="settlement identity"):
        _record_settlement(
            executor,
            client_order_id=order["client_order_id"],
            settlement_id="premature-settlement",
            now_ts_ms=NOW_TS_MS + 1,
            payout_per_token="1",
            official_settlement_sha256="0" * 64,
        )
    assert executor.reconciliation_snapshot()["settlement_count"] == 0
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("market_id", "0x" + "8" * 64),
        ("slug", "btc-updown-15m-1789949700"),
        ("winning_token_id", "999999"),
    ),
)
def test_official_settlement_identity_mismatch_fails_closed(
    authorized_fixture: dict[str, Any],
    field: str,
    value: str,
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    _record_fill(
        executor,
        client_order_id=order["client_order_id"],
        fill_id="settlement-identity-fill",
        now_ts_ms=NOW_TS_MS,
        quantity="1",
        price="0.39",
        fee_usd="0.0002",
        transport_event_sha256="0" * 64,
    )
    with pytest.raises(MicroLiveExecutionError, match="settlement identity"):
        _record_settlement(
            executor,
            client_order_id=order["client_order_id"],
            settlement_id="identity-mismatch-settlement",
            now_ts_ms=SETTLEMENT_NOW_TS_MS,
            payout_per_token="1",
            official_settlement_sha256="0" * 64,
            event_overrides={field: value},
        )
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["settlement_count"] == 0
    assert snapshot["kill_switch_reason"] == "settlement_reconciliation_failed"


def test_rehashed_lifecycle_raw_event_identity_tamper_fails_restore(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    executor = MicroLiveExecutor(verified, transport=FakeTransport())
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    _record_fill(
        executor,
        client_order_id=order["client_order_id"],
        fill_id="tamper-fill",
        now_ts_ms=NOW_TS_MS,
        quantity="1",
        price="0.39",
        fee_usd="0.0002",
        transport_event_sha256="0" * 64,
    )
    state = executor.export_state()
    fill_event = next(
        event for event in state["events"] if event["event_type"] == "FILL_RECORDED"
    )
    raw = json.loads(fill_event["payload"]["raw_transport_event_json"])
    raw["market_id"] = "0x" + "7" * 64
    raw_json = json.dumps(raw, separators=(",", ":"), sort_keys=True)
    fill_event["payload"]["raw_transport_event_json"] = raw_json
    fill_event["payload"]["transport_event_sha256"] = hashlib.sha256(
        raw_json.encode()
    ).hexdigest()
    previous = "GENESIS"
    for event in state["events"]:
        event["previous_event_sha256"] = previous
        core = {key: value for key, value in event.items() if key != "event_sha256"}
        event["event_sha256"] = canonical_json_sha256(core)
        previous = event["event_sha256"]
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(MicroLiveExecutionError, match="fill payload values"):
        MicroLiveExecutor.restore(
            authorization=verified,
            transport=FakeTransport(),
            raw_state=_json_bytes(state),
        )


def test_rehashed_stored_raw_event_duplicate_key_fails_restore(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    executor = MicroLiveExecutor(verified, transport=FakeTransport())
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    _record_fill(
        executor,
        client_order_id=order["client_order_id"],
        fill_id="stored-duplicate-fill",
        now_ts_ms=NOW_TS_MS,
        quantity="1",
        price="0.39",
        fee_usd="0.0002",
        transport_event_sha256="0" * 64,
    )
    state = executor.export_state()
    fill_event = next(
        event for event in state["events"] if event["event_type"] == "FILL_RECORDED"
    )
    raw_json = fill_event["payload"]["raw_transport_event_json"]
    duplicate_json = '{"market_id":"0x' + "0" * 64 + '",' + raw_json[1:]
    fill_event["payload"]["raw_transport_event_json"] = duplicate_json
    fill_event["payload"]["transport_event_sha256"] = hashlib.sha256(
        duplicate_json.encode()
    ).hexdigest()
    previous = "GENESIS"
    for event in state["events"]:
        event["previous_event_sha256"] = previous
        core = {key: value for key, value in event.items() if key != "event_sha256"}
        event["event_sha256"] = canonical_json_sha256(core)
        previous = event["event_sha256"]
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(MicroLiveExecutionError, match="strict JSON"):
        MicroLiveExecutor.restore(
            authorization=verified,
            transport=FakeTransport(),
            raw_state=_json_bytes(state),
        )


def test_conflicting_fill_and_partial_open_settlement_engage_kill_switch(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    _record_fill(
        executor,
        client_order_id=order["client_order_id"],
        fill_id="fill-partial",
        now_ts_ms=NOW_TS_MS,
        quantity="0.5",
        price="0.39",
        fee_usd="0.0001",
        transport_event_sha256="a" * 64,
    )
    with pytest.raises(MicroLiveExecutionError, match="conflicting duplicate fill"):
        _record_fill(
            executor,
            client_order_id=order["client_order_id"],
            fill_id="fill-partial",
            now_ts_ms=NOW_TS_MS,
            quantity="0.5",
            price="0.38",
            fee_usd="0.0001",
            transport_event_sha256="a" * 64,
        )
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True
    assert executor.reconciliation_snapshot()["open_order_count"] == 0
    assert len(transport.cancel_calls) == 1

    second_transport = FakeTransport()
    second = MicroLiveExecutor(verified, transport=second_transport)
    second_order = second.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    _record_fill(
        second,
        client_order_id=second_order["client_order_id"],
        fill_id="fill-partial-2",
        now_ts_ms=NOW_TS_MS,
        quantity="0.5",
        price="0.39",
        fee_usd="0.0001",
        transport_event_sha256="b" * 64,
    )
    with pytest.raises(MicroLiveExecutionError, match="open order cannot settle"):
        _record_settlement(
            second,
            client_order_id=second_order["client_order_id"],
            settlement_id="settlement-too-early",
            now_ts_ms=SETTLEMENT_NOW_TS_MS,
            payout_per_token="1",
            official_settlement_sha256="c" * 64,
        )
    assert second.reconciliation_snapshot()["kill_switch_active"] is True
    assert len(second_transport.cancel_calls) == 1


def test_unknown_submission_engages_kill_switch_without_retry(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport(fail_submit=True)
    executor = MicroLiveExecutor(verified, transport=transport)
    with pytest.raises(MicroLiveExecutionError, match="submission became unknown"):
        executor.submit_signal(
            **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
        )
    assert len(transport.submit_calls) == 1
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True


def test_transport_response_raw_bytes_are_preserved_and_hash_bound(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    acknowledgement = next(
        event["payload"]
        for event in executor.events
        if event["event_type"] == "ORDER_ACKNOWLEDGED"
    )
    raw_acknowledgement = acknowledgement["raw_transport_event_json"].encode()
    assert acknowledgement["transport_event_sha256"] == hashlib.sha256(
        raw_acknowledgement
    ).hexdigest()
    decoded = json.loads(raw_acknowledgement)
    assert all(
        decoded[key] == acknowledgement[key]
        for key in (
            "client_order_id",
            "exchange_order_id",
            "status",
            "market_id",
            "token_id",
            "accepted_quantity",
            "limit_price",
        )
    )

    executor.engage_kill_switch(reason="raw-cancel-evidence", now_ts_ms=NOW_TS_MS + 1)
    cancellation = next(
        event["payload"]
        for event in executor.events
        if event["event_type"] == "ORDER_CANCELED"
    )
    raw_cancellation = cancellation["raw_transport_event_json"].encode()
    assert cancellation["transport_event_sha256"] == hashlib.sha256(
        raw_cancellation
    ).hexdigest()
    assert json.loads(raw_cancellation)["status"] == "CANCELED"


def test_raw_rejected_submission_is_closed_and_restart_reconciles(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport(submit_status="REJECTED")
    executor = MicroLiveExecutor(verified, transport=transport)
    result = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    assert result["status"] == "ORDER_REJECTED"
    rejection = next(
        event["payload"]
        for event in executor.events
        if event["event_type"] == "ORDER_REJECTED"
    )
    raw_rejection = rejection["raw_transport_event_json"].encode()
    assert rejection["transport_event_sha256"] == hashlib.sha256(
        raw_rejection
    ).hexdigest()
    assert json.loads(raw_rejection)["status"] == "REJECTED"
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["open_order_count"] == 0
    assert snapshot["kill_switch_active"] is False
    restored = MicroLiveExecutor.restore(
        authorization=verified,
        transport=transport,
        raw_state=executor.export_state_bytes(),
    )
    assert restored.reconciliation_snapshot() == snapshot


def test_parsed_submit_response_fails_closed_as_unknown(
    authorized_fixture: dict[str, Any],
) -> None:
    class ParsedSubmitTransport(FakeTransport):
        def submit_order(self, request: dict[str, Any]) -> Any:
            return json.loads(super().submit_order(request))

    verified = _verified(authorized_fixture)
    transport = ParsedSubmitTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    with pytest.raises(MicroLiveExecutionError, match="submission became unknown"):
        executor.submit_signal(
            **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
        )
    assert len(transport.submit_calls) == 1
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["open_order_count"] == 0
    assert any(
        event["event_type"] == "ORDER_SUBMISSION_UNKNOWN"
        for event in executor.events
    )


@pytest.mark.parametrize("ambiguity", ("duplicate_key", "numeric_overflow"))
def test_ambiguous_raw_submit_response_fails_closed(
    authorized_fixture: dict[str, Any],
    ambiguity: str,
) -> None:
    class AmbiguousSubmitTransport(FakeTransport):
        def submit_order(self, request: dict[str, Any]) -> bytes:
            raw = super().submit_order(request)
            if ambiguity == "duplicate_key":
                return raw[:-1] + b',"status":"ACCEPTED"}'
            old = f'"limit_price":"{request["limit_price"]}"'.encode()
            assert old in raw
            return raw.replace(old, b'"limit_price":1e400')

    verified = _verified(authorized_fixture)
    transport = AmbiguousSubmitTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    with pytest.raises(MicroLiveExecutionError, match="submission became unknown"):
        executor.submit_signal(
            **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
        )
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True
    assert not any(
        event["event_type"] == "ORDER_ACKNOWLEDGED" for event in executor.events
    )


def test_parsed_cancel_response_remains_unknown_and_killed(
    authorized_fixture: dict[str, Any],
) -> None:
    class ParsedCancelTransport(FakeTransport):
        def cancel_order(self, request: dict[str, Any]) -> Any:
            return json.loads(super().cancel_order(request))

    verified = _verified(authorized_fixture)
    transport = ParsedCancelTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    killed = executor.engage_kill_switch(
        reason="parsed-cancel-response",
        now_ts_ms=NOW_TS_MS + 1,
    )
    assert killed["unknown_cancel_client_order_ids"] == [order["client_order_id"]]
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["open_order_count"] == 1


def test_parsed_lookup_response_cannot_reconcile_unknown_submission(
    authorized_fixture: dict[str, Any],
) -> None:
    class ParsedLookupTransport(FakeTransport):
        def lookup_order(self, request: dict[str, Any]) -> Any:
            return json.loads(super().lookup_order(request))

    verified = _verified(authorized_fixture)
    transport = ParsedLookupTransport(fail_submit=True)
    executor = MicroLiveExecutor(verified, transport=transport)
    with pytest.raises(MicroLiveExecutionError, match="submission became unknown"):
        executor.submit_signal(
            **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
        )
    client_order_id = transport.submit_calls[0]["client_order_id"]
    with pytest.raises(MicroLiveExecutionError, match="reconciliation failed closed"):
        executor.reconcile_unknown_submission(
            client_order_id=client_order_id,
            now_ts_ms=NOW_TS_MS + 1,
        )
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True
    assert len(transport.lookup_calls) == 1


def test_unknown_submission_read_only_reconciliation_never_resubmits_or_unlocks(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport(fail_submit=True)
    executor = MicroLiveExecutor(verified, transport=transport)
    with pytest.raises(MicroLiveExecutionError, match="submission became unknown"):
        executor.submit_signal(
            **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
        )
    client_order_id = transport.submit_calls[0]["client_order_id"]
    reconciled = executor.reconcile_unknown_submission(
        client_order_id=client_order_id,
        now_ts_ms=NOW_TS_MS + 1,
    )
    assert reconciled["status"] == "ORDER_SUBMISSION_RECONCILED_ACCEPTED"
    assert reconciled["kill_switch_active"] is True
    assert len(transport.submit_calls) == 1
    assert len(transport.lookup_calls) == 1
    assert len(transport.cancel_calls) == 1
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["kill_switch_reason"] == "order_submission_unknown"
    assert snapshot["open_order_count"] == 0
    restored = MicroLiveExecutor.restore(
        authorization=verified,
        transport=transport,
        raw_state=executor.export_state_bytes(),
    )
    assert restored.reconciliation_snapshot() == snapshot

    failing_transport = FakeTransport(fail_submit=True, fail_lookup=True)
    failing = MicroLiveExecutor(verified, transport=failing_transport)
    with pytest.raises(MicroLiveExecutionError, match="submission became unknown"):
        failing.submit_signal(
            **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
        )
    failing_client_order_id = failing_transport.submit_calls[0]["client_order_id"]
    with pytest.raises(MicroLiveExecutionError, match="reconciliation failed closed"):
        failing.reconcile_unknown_submission(
            client_order_id=failing_client_order_id,
            now_ts_ms=NOW_TS_MS + 1,
        )
    assert len(failing_transport.submit_calls) == 1
    assert len(failing_transport.lookup_calls) == 1
    assert failing.reconciliation_snapshot()["kill_switch_active"] is True
    assert any(
        event["event_type"] == "ORDER_SUBMISSION_RECONCILIATION_FAILED"
        for event in failing.events
    )


@pytest.mark.parametrize("closed_status", ("CANCELED", "EXPIRED"))
def test_unknown_cancel_read_only_reconciliation_closes_without_unlock_or_write(
    authorized_fixture: dict[str, Any],
    closed_status: str,
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport(
        fail_cancel=True,
        cancel_lookup_status=closed_status,
    )
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    killed = executor.engage_kill_switch(
        reason="synthetic_cancel_unknown",
        now_ts_ms=NOW_TS_MS + 1,
    )
    assert killed["unknown_cancel_client_order_ids"] == [order["client_order_id"]]
    assert executor.reconciliation_snapshot()["open_order_count"] == 1
    reconciled = executor.reconcile_unknown_cancellation(
        client_order_id=order["client_order_id"],
        now_ts_ms=NOW_TS_MS + 2,
    )
    assert reconciled == {
        "status": f"ORDER_CANCEL_RECONCILED_{closed_status}",
        "client_order_id": order["client_order_id"],
        "kill_switch_active": True,
        "order_closed": True,
        "lookup_called": True,
        "write_transport_called": False,
    }
    assert len(transport.submit_calls) == 1
    assert len(transport.cancel_calls) == 1
    assert len(transport.lookup_calls) == 1
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["open_order_count"] == 0
    restored = MicroLiveExecutor.restore(
        authorization=verified,
        transport=transport,
        raw_state=executor.export_state_bytes(),
    )
    assert restored.reconciliation_snapshot() == snapshot


@pytest.mark.parametrize("observed_status", ("OPEN", "FILLED"))
def test_unknown_cancel_unresolved_lookup_stays_killed_until_explicit_retry(
    authorized_fixture: dict[str, Any],
    observed_status: str,
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport(
        fail_cancel=True,
        cancel_lookup_status=observed_status,
    )
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    executor.engage_kill_switch(
        reason="synthetic_cancel_unknown",
        now_ts_ms=NOW_TS_MS + 1,
    )
    unresolved = executor.reconcile_unknown_cancellation(
        client_order_id=order["client_order_id"],
        now_ts_ms=NOW_TS_MS + 2,
    )
    assert unresolved["status"] == f"ORDER_CANCEL_RECONCILIATION_{observed_status}"
    assert unresolved["order_closed"] is False
    assert unresolved["write_transport_called"] is False
    assert len(transport.cancel_calls) == 1
    assert executor.reconciliation_snapshot()["open_order_count"] == 1
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True
    assert any(
        event["event_type"] == "ORDER_CANCEL_RECONCILIATION_FAILED"
        for event in executor.events
    )

    transport.fail_cancel = False
    retried = executor.engage_kill_switch(
        reason="explicit_cancel_retry",
        now_ts_ms=NOW_TS_MS + 3,
    )
    assert retried["canceled_client_order_ids"] == [order["client_order_id"]]
    assert len(transport.cancel_calls) == 2
    assert executor.reconciliation_snapshot()["open_order_count"] == 0
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True


def test_unknown_cancel_lookup_identity_drift_fails_closed_and_remains_open(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport(
        fail_cancel=True,
        cancel_lookup_overrides={"market_id": "0x" + "5" * 64},
    )
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    executor.engage_kill_switch(
        reason="synthetic_cancel_unknown",
        now_ts_ms=NOW_TS_MS + 1,
    )
    with pytest.raises(MicroLiveExecutionError, match="failed closed"):
        executor.reconcile_unknown_cancellation(
            client_order_id=order["client_order_id"],
            now_ts_ms=NOW_TS_MS + 2,
        )
    assert len(transport.lookup_calls) == 1
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["open_order_count"] == 1
    assert any(
        event["event_type"] == "ORDER_CANCEL_RECONCILIATION_FAILED"
        for event in executor.events
    )


def test_all_lifecycle_reconciliation_ambiguities_persistently_kill(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)

    unknown_fill = MicroLiveExecutor(verified, transport=FakeTransport())
    with pytest.raises(MicroLiveExecutionError, match="acknowledged order"):
        _record_fill(
            unknown_fill,
            client_order_id="missing-order",
            fill_id="missing-fill",
            now_ts_ms=NOW_TS_MS,
            quantity="1",
            price="0.39",
            fee_usd="0.0002",
            transport_event_sha256="a" * 64,
        )
    assert unknown_fill.reconciliation_snapshot()["kill_switch_reason"] == (
        "fill_reconciliation_failed"
    )

    close_transport = FakeTransport()
    bad_close = MicroLiveExecutor(verified, transport=close_transport)
    close_order = bad_close.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    with pytest.raises(MicroLiveExecutionError, match="status is invalid"):
        _record_order_closed(
            bad_close,
            client_order_id=close_order["client_order_id"],
            status="FILLED",
            now_ts_ms=NOW_TS_MS,
            transport_event_sha256="b" * 64,
        )
    assert bad_close.reconciliation_snapshot()["kill_switch_reason"] == (
        "order_close_reconciliation_failed"
    )
    assert len(close_transport.cancel_calls) == 1

    settlement_transport = FakeTransport()
    bad_settlement = MicroLiveExecutor(verified, transport=settlement_transport)
    settlement_order = bad_settlement.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    with pytest.raises(MicroLiveExecutionError, match="unfilled order"):
        _record_settlement(
            bad_settlement,
            client_order_id=settlement_order["client_order_id"],
            settlement_id="unfilled-settlement",
            now_ts_ms=SETTLEMENT_NOW_TS_MS,
            payout_per_token="1",
            official_settlement_sha256="c" * 64,
        )
    assert bad_settlement.reconciliation_snapshot()["kill_switch_reason"] == (
        "settlement_reconciliation_failed"
    )
    assert len(settlement_transport.cancel_calls) == 1


def test_lifecycle_timestamp_regression_still_persists_kill(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    with pytest.raises(MicroLiveExecutionError, match="timestamp regressed"):
        _record_fill(
            executor,
            client_order_id=order["client_order_id"],
            fill_id="regressed-fill",
            now_ts_ms=NOW_TS_MS - 1,
            quantity="1",
            price="0.39",
            fee_usd="0.0002",
            transport_event_sha256="d" * 64,
        )
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["kill_switch_reason"] == "fill_reconciliation_failed"
    assert snapshot["fill_count"] == 0
    assert len(transport.cancel_calls) == 1


def test_rehashed_tampered_state_still_fails_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    executor = MicroLiveExecutor(verified, transport=FakeTransport())
    executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    state = executor.export_state()
    prepared = next(
        event for event in state["events"] if event["event_type"] == "ORDER_PREPARED"
    )
    prepared["payload"]["selected_action"] = "BUY_DOWN_HOLD"
    previous = "GENESIS"
    for event in state["events"]:
        event["previous_event_sha256"] = previous
        core = {key: value for key, value in event.items() if key != "event_sha256"}
        event["event_sha256"] = canonical_json_sha256(core)
        previous = event["event_sha256"]
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(MicroLiveExecutionError, match="prepared order identity"):
        MicroLiveExecutor.restore(
            authorization=verified,
            transport=FakeTransport(),
            raw_state=_json_bytes(state),
        )


def test_rehashed_duplicate_exchange_order_identity_fails_closed_on_restore(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    executor = MicroLiveExecutor(verified, transport=FakeTransport())
    executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    executor.submit_signal(
        **_signal(
            candidate_bundle_sha256=verified.candidate_bundle_sha256,
            market_id=f"0x{444:064x}",
            up_token_id="64440",
            down_token_id="64441",
        )
    )
    state = executor.export_state()
    acknowledgements = [
        event
        for event in state["events"]
        if event["event_type"] == "ORDER_ACKNOWLEDGED"
    ]
    assert len(acknowledgements) == 2
    acknowledgements[1]["payload"]["exchange_order_id"] = acknowledgements[0][
        "payload"
    ]["exchange_order_id"]
    raw_response = json.loads(
        acknowledgements[1]["payload"]["raw_transport_event_json"]
    )
    raw_response["exchange_order_id"] = acknowledgements[1]["payload"][
        "exchange_order_id"
    ]
    raw_response_bytes = _json_bytes(raw_response)
    acknowledgements[1]["payload"]["raw_transport_event_json"] = (
        raw_response_bytes.decode("utf-8")
    )
    acknowledgements[1]["payload"]["transport_event_sha256"] = hashlib.sha256(
        raw_response_bytes
    ).hexdigest()
    previous = "GENESIS"
    for event in state["events"]:
        event["previous_event_sha256"] = previous
        core = {key: value for key, value in event.items() if key != "event_sha256"}
        event["event_sha256"] = canonical_json_sha256(core)
        previous = event["event_sha256"]
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(MicroLiveExecutionError, match="acknowledgement is duplicated"):
        MicroLiveExecutor.restore(
            authorization=verified,
            transport=FakeTransport(),
            raw_state=_json_bytes(state),
        )


def test_rehashed_order_without_execution_intent_audit_fails_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    executor = MicroLiveExecutor(verified, transport=FakeTransport())
    executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    state = executor.export_state()
    state["events"] = [
        event
        for event in state["events"]
        if event["event_type"] != "SIGNAL_EVALUATED"
    ]
    previous = "GENESIS"
    for sequence, event in enumerate(state["events"], start=1):
        event["sequence"] = sequence
        event["previous_event_sha256"] = previous
        core = {key: value for key, value in event.items() if key != "event_sha256"}
        event["event_sha256"] = canonical_json_sha256(core)
        previous = event["event_sha256"]
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(MicroLiveExecutionError, match="prepared order identity"):
        MicroLiveExecutor.restore(
            authorization=verified,
            transport=FakeTransport(),
            raw_state=_json_bytes(state),
        )


def test_rehashed_event_timestamp_regression_still_fails_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    executor = MicroLiveExecutor(verified, transport=FakeTransport())
    executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    state = executor.export_state()
    state["events"][1]["event_ts_ms"] = state["events"][0]["event_ts_ms"] - 1
    previous = "GENESIS"
    for event in state["events"]:
        event["previous_event_sha256"] = previous
        core = {key: value for key, value in event.items() if key != "event_sha256"}
        event["event_sha256"] = canonical_json_sha256(core)
        previous = event["event_sha256"]
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(MicroLiveExecutionError, match="event chain"):
        MicroLiveExecutor.restore(
            authorization=verified,
            transport=FakeTransport(),
            raw_state=_json_bytes(state),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("explicit_human_approval_recorded", False, "state is not explicit"),
        ("micro_live_authorized", False, "state is not explicit"),
        ("automatic_launch_allowed", True, "state is not explicit"),
        ("requested_initial_capital_fraction", "0.02", "limits or validity"),
        ("maximum_realized_loss_usd", "10.01", "limits or validity"),
    ),
)
def test_authorization_tampering_fails_closed(
    authorized_fixture: dict[str, Any],
    field: str,
    value: Any,
    message: str,
) -> None:
    changed = copy.deepcopy(authorized_fixture["authorization"])
    changed[field] = value
    with pytest.raises(MicroLiveAuthorizationError, match=message):
        verify_micro_live_authorization(
            _json_bytes(changed),
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
        )


def test_expired_authorization_and_evidence_sha_drift_fail_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    authorization = authorized_fixture["authorization"]
    with pytest.raises(MicroLiveAuthorizationError, match="validity window"):
        verify_micro_live_authorization(
            _json_bytes(authorization),
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=authorization["expires_at_ts_ms"],
        )
    changed = copy.deepcopy(authorization)
    changed["required_evidence"]["fresh_evaluation_manifest"]["sha256"] = "0" * 64
    with pytest.raises(MicroLiveAuthorizationError, match="path or SHA-256 mismatch"):
        verify_micro_live_authorization(
            _json_bytes(changed),
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
        )


@pytest.mark.parametrize(
    "ambiguity",
    ("duplicate_key", "nonfinite_constant", "numeric_overflow"),
)
def test_authorization_evidence_ambiguous_json_fails_closed(
    authorized_fixture: dict[str, Any],
    tmp_path: Path,
    ambiguity: str,
) -> None:
    evidence_root = tmp_path / "ambiguous_authorization_evidence"
    shutil.copytree(authorized_fixture["evidence_root"], evidence_root)
    authorization = copy.deepcopy(authorized_fixture["authorization"])
    descriptor = authorization["human_approval"]["github_comment_payload"]
    path = evidence_root / descriptor["path"]
    original = path.read_text(encoding="utf-8").lstrip()
    if ambiguity == "duplicate_key":
        ambiguous = '{"id":0,' + original[1:]
    elif ambiguity == "nonfinite_constant":
        payload = json.loads(original)
        needle = f'"id": {payload["id"]}'
        assert needle in original
        ambiguous = original.replace(needle, '"id": NaN', 1)
    else:
        payload = json.loads(original)
        needle = f'"id": {payload["id"]}'
        assert needle in original
        ambiguous = original.replace(needle, '"id": 1e400', 1)
    path.write_text(ambiguous, encoding="utf-8")
    descriptor["sha256"] = sha256_file(path)

    with pytest.raises(MicroLiveAuthorizationError, match="JSON evidence is invalid"):
        verify_micro_live_authorization(
            _json_bytes(authorization),
            repository_root=authorized_fixture["root"],
            evidence_root=evidence_root,
            now_ts_ms=NOW_TS_MS,
        )


def test_human_approval_owner_and_timestamp_are_exact(
    authorized_fixture: dict[str, Any],
) -> None:
    changed = copy.deepcopy(authorized_fixture["authorization"])
    changed["human_approval"]["github_login"] = "untrusted-user"
    with pytest.raises(MicroLiveAuthorizationError, match="provenance"):
        verify_micro_live_authorization(
            _json_bytes(changed),
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
        )

    approval_descriptor = authorized_fixture["authorization"]["human_approval"][
        "github_comment_payload"
    ]
    github = _json(authorized_fixture["evidence_root"] / approval_descriptor["path"])
    assert "capital_base_usd=1000" in github["body"]
    assert "maximum_notional_usd=10.00" in github["body"]
    assert "maximum_realized_loss_usd=1.00" in github["body"]
    assert "maximum_open_orders=2" in github["body"]

    changed = copy.deepcopy(authorized_fixture["authorization"])
    changed["created_at"] = "2026-09-21T00:00:01Z"
    with pytest.raises(MicroLiveAuthorizationError, match="limits or validity"):
        verify_micro_live_authorization(
            _json_bytes(changed),
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
        )
