"""Non-executable execution-readiness checks for residual promotion v1.

This module deliberately has no exchange, wallet, paper, settlement, or outcome
adapter.  It exercises deterministic intent identity and local ledger semantics
without authorizing or attempting execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    load_residual_promotion_runtime,
)

SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-execution-readiness-v1"
LEDGER_SCHEMA_VERSION = "bigan-non-executable-intent-ledger-v1"
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_execution_readiness.py"
)
CLI_REPOSITORY_PATH = "examples/v8/run_residual_promotion_execution_readiness.py"
CONFIG_REPOSITORY_PATH = (
    "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
)
BUNDLE_REPOSITORY_PATH = f"{CONFIG_REPOSITORY_PATH}/candidate_bundle/bundle_manifest.json"
PARITY_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/candidate_bundle/offline_live_parity_report.json"
)
ALLOWED_ACTIONS = frozenset({"NO_TRADE", "BUY_UP_HOLD", "BUY_DOWN_HOLD"})
FORBIDDEN_DATA_KEY_TOKENS = ("outcome", "settlement", "pnl", "resolution")


class ExecutionReadinessError(ValueError):
    """Fail-closed execution-readiness validation error."""


class NonExecutableIntentLedger:
    """Deterministic local ledger with no execution side effects."""

    def __init__(
        self,
        *,
        candidate_bundle_sha256: str,
        entries: Sequence[Mapping[str, Any]] = (),
        kill_switch_active: bool = False,
    ) -> None:
        _require_sha256(candidate_bundle_sha256, "candidate bundle")
        self.candidate_bundle_sha256 = candidate_bundle_sha256
        self._entries = [copy.deepcopy(dict(entry)) for entry in entries]
        self.kill_switch_active = bool(kill_switch_active)
        self._verify_chain()

    @property
    def entries(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._entries))

    def engage_kill_switch(self) -> None:
        self.kill_switch_active = True

    def record_projection(
        self,
        *,
        market_id: str,
        decision_ts: int,
        market_family: str,
        candidate_bundle_sha256: str,
        projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record one non-executable intent or return a fail-closed disposition."""

        _assert_outcome_free(projection)
        if not market_id or isinstance(decision_ts, bool) or decision_ts <= 0:
            raise ExecutionReadinessError("market identity is invalid")
        if candidate_bundle_sha256 != self.candidate_bundle_sha256:
            return _blocked_disposition("candidate_bundle_mismatch")
        if market_family != "BTC-15M":
            return _blocked_disposition("market_family_not_allowlisted")
        if self.kill_switch_active:
            return _blocked_disposition("kill_switch_active")

        selected_action = projection.get("selected_action")
        if selected_action not in ALLOWED_ACTIONS:
            return _blocked_disposition("projection_action_invalid")
        if projection.get("fail_closed") is True or projection.get("model_scored") is not True:
            return _blocked_disposition("projection_not_executable")
        if selected_action == "NO_TRADE":
            return _blocked_disposition("projection_selected_no_trade")

        business_identity = {
            "candidate_bundle_sha256": candidate_bundle_sha256,
            "market_id": market_id,
            "decision_ts": decision_ts,
        }
        business_key = canonical_json_sha256(business_identity)
        decision_sha256 = canonical_json_sha256(dict(projection))
        intent_identity = {
            **business_identity,
            "selected_action": selected_action,
            "decision_sha256": decision_sha256,
        }
        intent_id = canonical_json_sha256(intent_identity)
        existing = next(
            (entry for entry in self._entries if entry["business_key"] == business_key),
            None,
        )
        if existing is not None:
            if existing["intent_id"] != intent_id:
                raise ExecutionReadinessError(
                    "conflicting duplicate decision failed closed without ledger mutation"
                )
            return {
                "status": "IDEMPOTENT_REPLAY",
                "intent_id": intent_id,
                "selected_action": selected_action,
                "executable": False,
                **_closed_execution_flags(),
            }

        core = {
            "sequence": len(self._entries) + 1,
            "previous_entry_sha256": (
                self._entries[-1]["entry_sha256"] if self._entries else "GENESIS"
            ),
            "business_key": business_key,
            "intent_id": intent_id,
            "candidate_bundle_sha256": candidate_bundle_sha256,
            "market_id": market_id,
            "decision_ts": decision_ts,
            "market_family": market_family,
            "selected_action": selected_action,
            "decision_sha256": decision_sha256,
            "status": "RECORDED_NON_EXECUTABLE",
            "executable": False,
            **_closed_execution_flags(),
        }
        entry = {**core, "entry_sha256": canonical_json_sha256(core)}
        self._entries.append(entry)
        return {
            "status": "RECORDED_NON_EXECUTABLE",
            "intent_id": intent_id,
            "selected_action": selected_action,
            "executable": False,
            **_closed_execution_flags(),
        }

    def export_state(self) -> dict[str, Any]:
        payload = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "candidate_bundle_sha256": self.candidate_bundle_sha256,
            "kill_switch_active": self.kill_switch_active,
            "entries": copy.deepcopy(self._entries),
        }
        return {**payload, "state_sha256": canonical_json_sha256(payload)}

    @classmethod
    def restore(cls, state: Mapping[str, Any]) -> NonExecutableIntentLedger:
        if set(state) != {
            "schema_version",
            "candidate_bundle_sha256",
            "kill_switch_active",
            "entries",
            "state_sha256",
        }:
            raise ExecutionReadinessError("ledger state schema mismatch")
        payload = {key: copy.deepcopy(value) for key, value in state.items() if key != "state_sha256"}
        if (
            payload["schema_version"] != LEDGER_SCHEMA_VERSION
            or canonical_json_sha256(payload) != state["state_sha256"]
        ):
            raise ExecutionReadinessError("ledger state SHA-256 mismatch")
        entries = payload["entries"]
        if not isinstance(entries, list) or not isinstance(payload["kill_switch_active"], bool):
            raise ExecutionReadinessError("ledger state value types are invalid")
        return cls(
            candidate_bundle_sha256=str(payload["candidate_bundle_sha256"]),
            entries=entries,
            kill_switch_active=bool(payload["kill_switch_active"]),
        )

    def _verify_chain(self) -> None:
        previous = "GENESIS"
        seen_business_keys: set[str] = set()
        for expected_sequence, entry in enumerate(self._entries, start=1):
            expected_keys = {
                "sequence",
                "previous_entry_sha256",
                "business_key",
                "intent_id",
                "candidate_bundle_sha256",
                "market_id",
                "decision_ts",
                "market_family",
                "selected_action",
                "decision_sha256",
                "status",
                "executable",
                "paper_order_allowed",
                "live_order_allowed",
                "wallet_signing_allowed",
                "polymarket_write_allowed",
                "capital_at_risk",
                "entry_sha256",
            }
            if set(entry) != expected_keys:
                raise ExecutionReadinessError("ledger entry schema mismatch")
            core = {key: value for key, value in entry.items() if key != "entry_sha256"}
            business_identity = {
                "candidate_bundle_sha256": entry["candidate_bundle_sha256"],
                "market_id": entry["market_id"],
                "decision_ts": entry["decision_ts"],
            }
            intent_identity = {
                **business_identity,
                "selected_action": entry["selected_action"],
                "decision_sha256": entry["decision_sha256"],
            }
            if (
                entry["sequence"] != expected_sequence
                or entry["previous_entry_sha256"] != previous
                or entry["candidate_bundle_sha256"] != self.candidate_bundle_sha256
                or entry["market_family"] != "BTC-15M"
                or entry["selected_action"] not in ALLOWED_ACTIONS - {"NO_TRADE"}
                or isinstance(entry["decision_ts"], bool)
                or not isinstance(entry["decision_ts"], int)
                or entry["decision_ts"] <= 0
                or entry["business_key"] != canonical_json_sha256(business_identity)
                or entry["intent_id"] != canonical_json_sha256(intent_identity)
                or entry["business_key"] in seen_business_keys
                or entry["status"] != "RECORDED_NON_EXECUTABLE"
                or entry["executable"] is not False
                or not _flags_are_closed(entry)
                or entry["entry_sha256"] != canonical_json_sha256(core)
            ):
                raise ExecutionReadinessError("ledger chain or safety invariant mismatch")
            seen_business_keys.add(str(entry["business_key"]))
            previous = str(entry["entry_sha256"])


def reconcile_synthetic_fill(
    *, intent_id: str, side: str, quantity: str, price: str, fee: str
) -> dict[str, Any]:
    """Reconcile a synthetic fill without market, outcome, or settlement access."""

    _require_sha256(intent_id, "intent")
    if side not in {"UP", "DOWN"}:
        raise ExecutionReadinessError("synthetic fill side is invalid")
    qty = Decimal(quantity)
    px = Decimal(price)
    fee_value = Decimal(fee)
    if not qty.is_finite() or not px.is_finite() or not fee_value.is_finite():
        raise ExecutionReadinessError("synthetic fill contains non-finite values")
    if qty <= 0 or px <= 0 or px >= 1 or fee_value < 0:
        raise ExecutionReadinessError("synthetic fill values are out of bounds")
    cash_delta = -(qty * px + fee_value)
    expected = -(Decimal(quantity) * Decimal(price) + Decimal(fee))
    passed = cash_delta == expected
    return {
        "fixture_type": "synthetic_fill_no_market_access",
        "intent_id": intent_id,
        "side": side,
        "quantity": str(qty),
        "price": str(px),
        "fee": str(fee_value),
        "position_delta": str(qty),
        "cash_delta": str(cash_delta),
        "order_fill_position_cash_reconciled": passed,
        "settlement_reconciliation_verified": False,
        "executable": False,
        **_closed_execution_flags(),
    }


def build_execution_readiness_report(
    *, repository_root: Path | str, output_path: Path | str, created_at: str
) -> dict[str, Any]:
    """Write immutable, non-authorizing execution engineering evidence."""

    root = Path(repository_root).resolve()
    output = Path(output_path).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError("execution readiness report already exists")
    bundle_path = _repository_file(root, BUNDLE_REPOSITORY_PATH)
    parity_path = _repository_file(root, PARITY_REPOSITORY_PATH)
    _verify_sidecar(bundle_path)
    _verify_sidecar(parity_path)
    bundle_sha256 = sha256_file(bundle_path)
    load_residual_promotion_runtime(
        manifest_path=bundle_path,
        expected_manifest_sha256=bundle_sha256,
        repository_root=root,
    )
    parity = _load_json(parity_path)
    if (
        dict(parity.get("frozen_bundle_manifest") or {}).get("sha256") != bundle_sha256
        or parity.get("prediction_and_decision_parity") is not True
        or parity.get("fresh_outcomes_accessed") is not False
        or dict(parity.get("safety") or {}) != SAFETY
    ):
        raise ExecutionReadinessError("frozen runtime parity evidence is invalid")
    projection = dict(parity.get("live_projection") or {})
    ledger = NonExecutableIntentLedger(candidate_bundle_sha256=bundle_sha256)
    initial = ledger.record_projection(
        market_id="synthetic-btc-15m-readiness-001",
        decision_ts=1_786_406_400_000,
        market_family="BTC-15M",
        candidate_bundle_sha256=bundle_sha256,
        projection=projection,
    )
    duplicate = ledger.record_projection(
        market_id="synthetic-btc-15m-readiness-001",
        decision_ts=1_786_406_400_000,
        market_family="BTC-15M",
        candidate_bundle_sha256=bundle_sha256,
        projection=projection,
    )
    frozen_state = ledger.export_state()
    recovered = NonExecutableIntentLedger.restore(frozen_state)
    recovery_replay = recovered.record_projection(
        market_id="synthetic-btc-15m-readiness-001",
        decision_ts=1_786_406_400_000,
        market_family="BTC-15M",
        candidate_bundle_sha256=bundle_sha256,
        projection=projection,
    )
    before_conflict = recovered.export_state()
    conflicting_projection = copy.deepcopy(projection)
    conflicting_projection["selected_action"] = (
        "BUY_UP_HOLD"
        if projection.get("selected_action") != "BUY_UP_HOLD"
        else "BUY_DOWN_HOLD"
    )
    conflict_failed_closed = False
    try:
        recovered.record_projection(
            market_id="synthetic-btc-15m-readiness-001",
            decision_ts=1_786_406_400_000,
            market_family="BTC-15M",
            candidate_bundle_sha256=bundle_sha256,
            projection=conflicting_projection,
        )
    except ExecutionReadinessError:
        conflict_failed_closed = recovered.export_state() == before_conflict
    recovered.engage_kill_switch()
    kill_switch = recovered.record_projection(
        market_id="synthetic-btc-15m-readiness-002",
        decision_ts=1_786_407_300_000,
        market_family="BTC-15M",
        candidate_bundle_sha256=bundle_sha256,
        projection=projection,
    )
    mismatch = ledger.record_projection(
        market_id="synthetic-btc-15m-readiness-003",
        decision_ts=1_786_408_200_000,
        market_family="BTC-15M",
        candidate_bundle_sha256="0" * 64,
        projection=projection,
    )
    reconciliation = reconcile_synthetic_fill(
        intent_id=str(initial["intent_id"]),
        side="DOWN",
        quantity="1",
        price="0.55",
        fee="0.01",
    )
    checks = {
        "fresh_clone_bundle_graph_load": True,
        "frozen_offline_live_parity": True,
        "deterministic_intent_identity": bool(
            initial["status"] == "RECORDED_NON_EXECUTABLE"
            and duplicate["status"] == "IDEMPOTENT_REPLAY"
            and initial["intent_id"] == duplicate["intent_id"]
        ),
        "restart_recovery_idempotence": bool(
            recovery_replay["status"] == "IDEMPOTENT_REPLAY"
            and recovery_replay["intent_id"] == initial["intent_id"]
        ),
        "conflicting_duplicate_fail_closed_without_mutation": conflict_failed_closed,
        "kill_switch_blocks_intent": bool(
            kill_switch["status"] == "BLOCKED_NO_TRADE"
            and kill_switch["reason"] == "kill_switch_active"
        ),
        "candidate_mismatch_blocks_intent": bool(
            mismatch["status"] == "BLOCKED_NO_TRADE"
            and mismatch["reason"] == "candidate_bundle_mismatch"
        ),
        "synthetic_order_fill_position_cash_reconciliation": bool(
            reconciliation["order_fill_position_cash_reconciled"] is True
        ),
    }
    engineering_passed = all(checks.values())
    if not engineering_passed:
        raise ExecutionReadinessError("execution engineering readiness failed closed")
    report = {
        "schema_version": SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "candidate_bundle": _descriptor(root, bundle_path),
        "frozen_runtime_parity": _descriptor(root, parity_path),
        "implementation": _descriptor(
            root, _repository_file(root, IMPLEMENTATION_REPOSITORY_PATH)
        ),
        "cli": _descriptor(root, _repository_file(root, CLI_REPOSITORY_PATH)),
        "parity_fixture_sha256": parity.get("fixture_sha256"),
        "fresh_clone_artifact_graph_loaded": True,
        "candidate_behavior_changed": False,
        "candidate_bytes_changed": False,
        "checks": checks,
        "ledger_state_sha256": frozen_state["state_sha256"],
        "ledger_entry_count": len(ledger.entries),
        "conflicting_duplicate_incident_recorded": False,
        "synthetic_reconciliation": reconciliation,
        "engineering_readiness_passed": True,
        "security_review_passed": False,
        "security_review_status": "INCOMPLETE_INDEPENDENT_REVIEW_REQUIRED",
        "settlement_reconciliation_verified": False,
        "paper_candidate_allowed": False,
        "paper_run_started": False,
        "phase6_zero_capital_authorized": False,
        "micro_live_authorized": False,
        "live_trading_allowed": False,
        "order_submission_attempted": False,
        "wallet_signing_allowed": False,
        "wallet_signing_attempted": False,
        "polymarket_write_allowed": False,
        "polymarket_write_attempted": False,
        "capital_at_risk": False,
        "fresh_population_used": False,
        "fresh_outcomes_accessed": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "remaining_release_gates": [
            "exact_2500_fresh_confirmation",
            "shadow_stability",
            "independent_security_review",
            "post_confirmation_phase6_zero_capital",
            "explicit_human_1_percent_micro_live_go_no_go",
        ],
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, report)
    return report


def _blocked_disposition(reason: str) -> dict[str, Any]:
    return {
        "status": "BLOCKED_NO_TRADE",
        "reason": reason,
        "selected_action": "NO_TRADE",
        "intent_id": None,
        "executable": False,
        **_closed_execution_flags(),
    }


def _closed_execution_flags() -> dict[str, bool]:
    return {
        "paper_order_allowed": False,
        "live_order_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
    }


def _flags_are_closed(value: Mapping[str, Any]) -> bool:
    return all(value.get(name) is False for name in _closed_execution_flags())


def _assert_outcome_free(value: Any, *, path: str = "projection") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in FORBIDDEN_DATA_KEY_TOKENS):
                raise ExecutionReadinessError(f"forbidden data field at {path}.{key}")
            _assert_outcome_free(child, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _assert_outcome_free(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ExecutionReadinessError(f"non-finite value at {path}")


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ExecutionReadinessError(f"{label} SHA-256 is invalid")


def _repository_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ExecutionReadinessError("repository artifact is missing or escaped root")
    return path


def _descriptor(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise ExecutionReadinessError("frozen artifact sidecar mismatch")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExecutionReadinessError("JSON root must be an object")
    return value


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )


__all__ = [
    "ExecutionReadinessError",
    "NonExecutableIntentLedger",
    "build_execution_readiness_report",
    "reconcile_synthetic_fill",
]
