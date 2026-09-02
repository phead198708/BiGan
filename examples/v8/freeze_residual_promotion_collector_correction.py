"""Freeze the zero-attempt Chainlink collector correction for promotion v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.challenge_development_lane import sha256_file  # noqa: E402
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY  # noqa: E402
from bigan.v8.polymarket.residual_promotion_v1 import (  # noqa: E402
    LINEAGE_ID,
    _descriptor,
    _verified_json,
    _write_frozen_json,
)

HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
CONFIG = ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID
ORIGINAL_AUTHORIZATION = CONFIG / "manual_collection_authorization.json"
ORIGINAL_COLLECTOR_PROTOCOL = CONFIG / "prospective_collector_protocol.json"
STATISTICAL_PROTOCOL = CONFIG / "prospective_statistical_protocol.json"
CORRECTION = CONFIG / "collector_pre_attempt_engineering_correction.json"
CORRECTED_COLLECTOR_PROTOCOL = CONFIG / "prospective_collector_protocol_v2.json"
CORRECTED_AUTHORIZATION = CONFIG / "manual_collection_authorization_v2.json"
SOURCE_OPERATOR_COMMIT = "0595b168512c43f45966957dde8b36a23723cbce"
SOURCE_OPERATOR_SHA256 = (
    "cce3240aec777fcde05f3d3aa609bf882ec34850c01d70ca0e220a76586edcc8"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--prior-service-root", type=Path, required=True)
    parser.add_argument("--created-at")
    args = parser.parse_args()
    if not HEX_GIT_SHA.fullmatch(args.source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    service_root = args.prior_service_root.expanduser().resolve()
    start_record = service_root / "collection_start_record.json"
    ledger = service_root / "outcome_blind_attempts.jsonl"
    if not start_record.is_file():
        raise ValueError("prior collection start record is missing")
    if ledger.exists() and ledger.read_text(encoding="utf-8").strip():
        raise ValueError("collector correction is allowed only before attempt 1")
    start = json.loads(start_record.read_text(encoding="utf-8"))
    if not (
        start.get("fresh_collection_started") is True
        and start.get("fresh_outcomes_opened") is False
        and start.get("capital_at_risk") is False
        and start.get("polymarket_write_allowed") is False
        and start.get("wallet_signing_allowed") is False
    ):
        raise ValueError("prior start record is not outcome-blind and zero-capital")

    original_authorization = _verified_json(ORIGINAL_AUTHORIZATION)
    original_collector = _verified_json(ORIGINAL_COLLECTOR_PROTOCOL)
    _verified_json(STATISTICAL_PROTOCOL)
    stamp = args.created_at or datetime.now(UTC).isoformat()
    bindings = {
        "collector_cli": _descriptor(
            ROOT / "examples/v8/run_residual_promotion_v1_collector.py", ROOT
        ),
        "collection_ledger": _descriptor(
            ROOT / "src/bigan/v8/polymarket/residual_promotion_collection.py", ROOT
        ),
        "pending_capture": _descriptor(
            ROOT / "src/bigan/v8/polymarket/recorder/async_settlement.py", ROOT
        ),
        "chainlink_rtds": _descriptor(
            ROOT / "src/bigan/v8/polymarket/recorder/chainlink_rtds.py", ROOT
        ),
        "live_round_finalizer": _descriptor(
            ROOT / "src/bigan/v8/polymarket/live/operator.py", ROOT
        ),
    }
    if bindings["live_round_finalizer"]["sha256"] != SOURCE_OPERATOR_SHA256:
        raise ValueError("restored Chainlink finalizer is not source-byte-identical")
    correction = {
        "schema_version": (
            "bigan-btc-15m-residual-promotion-v1-collector-"
            "pre-attempt-engineering-correction-v1"
        ),
        "lineage_id": LINEAGE_ID,
        "created_at": stamp,
        "corrected_source_commit": args.source_commit,
        "defect": (
            "collector_did_not_inject_existing_read_only_chainlink_rtds_source_"
            "while_frozen_quality_gate_requires_chainlink_capture"
        ),
        "detected_before_first_attempt": True,
        "prior_attempts_consumed": 0,
        "prior_quality_valid_market_count": 0,
        "prior_start_record_sha256": sha256_file(start_record),
        "original_authorization": _descriptor(ORIGINAL_AUTHORIZATION, ROOT),
        "original_collector_protocol": _descriptor(
            ORIGINAL_COLLECTOR_PROTOCOL, ROOT
        ),
        "statistical_protocol_unchanged": _descriptor(
            STATISTICAL_PROTOCOL, ROOT
        ),
        "candidate_bundle_unchanged": original_authorization["candidate_bundle"],
        "corrected_implementation_bindings": bindings,
        "source_provenance": {
            "source_ref": "refs/heads/codex/v8-challenge-model-254-259",
            "source_commit": SOURCE_OPERATOR_COMMIT,
            "source_path": "src/bigan/v8/polymarket/live/operator.py",
            "source_content_sha256": SOURCE_OPERATOR_SHA256,
            "destination_path": "src/bigan/v8/polymarket/live/operator.py",
            "destination_sha256": bindings["live_round_finalizer"]["sha256"],
        },
        "original_authorization_invalidated_for_execution": True,
        "candidate_slot_consumed": False,
        "model_prediction_bytes_changed": False,
        "threshold_gate_cost_baseline_or_population_changed": False,
        "fresh_outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(CORRECTION, correction)
    correction_descriptor = _descriptor(CORRECTION, ROOT)

    corrected_collector = {
        **original_collector,
        "schema_version": (
            "bigan-btc-15m-residual-promotion-v1-collector-protocol-v2"
        ),
        "created_at": stamp,
        "supersedes": _descriptor(ORIGINAL_COLLECTOR_PROTOCOL, ROOT),
        "collector_engineering_correction": correction_descriptor,
        "collector_implementation": bindings["collector_cli"],
        "implementation_bindings": bindings,
        "chainlink_rtds_background_collector_required": True,
        "chainlink_rtds_injected_into_capture": True,
        "chainlink_missing_or_stale_behavior": "quality_invalid_fail_closed",
        "prior_attempts_consumed": 0,
        "fresh_outcomes_accessed": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(CORRECTED_COLLECTOR_PROTOCOL, corrected_collector)

    corrected_authorization = {
        **original_authorization,
        "schema_version": (
            "bigan-btc-15m-residual-promotion-v1-"
            "manual-collection-authorization-v2"
        ),
        "created_at": stamp,
        "supersedes": _descriptor(ORIGINAL_AUTHORIZATION, ROOT),
        "collector_engineering_correction": correction_descriptor,
        "collector_protocol": _descriptor(CORRECTED_COLLECTOR_PROTOCOL, ROOT),
        "prior_collector_started_without_attempt": True,
        "prior_attempts_consumed": 0,
        "prior_fresh_outcomes_opened": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(CORRECTED_AUTHORIZATION, corrected_authorization)
    report = {
        "correction": _descriptor(CORRECTION, ROOT),
        "collector_protocol": _descriptor(CORRECTED_COLLECTOR_PROTOCOL, ROOT),
        "authorization": _descriptor(CORRECTED_AUTHORIZATION, ROOT),
        "implementation_graph_sha256": _canonical_sha256(bindings),
        "prior_attempts_consumed": 0,
        "fresh_outcomes_accessed": False,
        "safety": dict(SAFETY),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
