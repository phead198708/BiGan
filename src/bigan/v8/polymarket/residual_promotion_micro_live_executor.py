"""Capability-gated BTC-15M micro-live execution core.

There is intentionally no Polymarket client, network session, wallet, key, or
credential implementation here.  A transport capability must be injected, and
it is never called unless the separately verified future authorization passes.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import hmac
import json
import math
import os
import platform
import queue
import re
import stat
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import wraps
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import scipy
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus.builder import (
    _normalize_book_snapshots,
    _normalize_candles,
    _normalize_chainlink_prices,
    _normalize_markets,
    _normalize_trades,
)
from bigan.v8.polymarket.corpus.contracts import PolymarketCorpusBuildConfig
from bigan.v8.polymarket.corpus.features import build_polymarket_corpus_feature_rows
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_micro_live_authorization import (
    VerifiedMicroLiveAuthorization,
    authorization_capability_is_verified,
    verify_micro_live_authorization,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    ResidualPromotionRuntime,
)

STATE_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-micro-live-state-v16"
SIGNAL_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-execution-signal-v2"
JOURNAL_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-durable-journal-v1"
JOURNAL_NAMESPACE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-journal-namespace-lease-receipt-v5"
)
JOURNAL_HIGH_WATER_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-journal-authority-high-water-v1"
)
JOURNAL_KILL_RECEIPT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-journal-authority-kill-v1"
)
EXECUTION_INVOCATION_FENCE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-invocation-fence-v1"
)
EXECUTION_INVOCATION_ACCEPTANCE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-outbox-acceptance-v3"
)
EXECUTION_OUTBOX_COMMAND_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-outbox-command-v1"
)
EXECUTION_OUTBOX_RECOVERY_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-outbox-recovery-v1"
)
EXECUTION_DISPATCH_RECEIPT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-dispatch-receipt-v1"
)
EXECUTION_DISPATCH_COMPLETION_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-dispatch-completion-v1"
)
EXECUTION_DISPATCH_FENCE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-dispatch-fence-v1"
)
EXECUTION_BINDING_ATTESTATION_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-binding-attestation-v8"
)
EXECUTION_OPERATION_RECEIPT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-operation-receipt-v4"
)
EXECUTION_CURSOR_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-signed-fill-cursor-v1"
)
TRUSTED_TIME_RECEIPT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-trusted-time-receipt-v1"
)
SETTLEMENT_RECEIPT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-signed-settlement-receipt-v2"
)
SUBMISSION_RECOVERY_OPERATION = "recover_order_submission"
SUBMISSION_RECOVERY_SEMANTICS = (
    "venue_idempotency_lookup_only_no_submit_sign_wallet_or_write"
)
EXECUTION_TRANSPORT_OPERATION_INVENTORY_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-transport-operations-v1"
)
REQUIRED_EXECUTION_TRANSPORT_OPERATIONS = (
    "attest_execution_binding",
    "bind_execution_dispatch_authority",
    "read_trusted_time",
    "submit_order",
    SUBMISSION_RECOVERY_OPERATION,
    "cancel_order",
    "lookup_order",
    "read_order_fill_cursor",
    "fence_order_invocation",
)
REQUIRED_EXECUTION_TRANSPORT_OPERATIONS_SHA256 = canonical_json_sha256(
    {
        "schema_version": EXECUTION_TRANSPORT_OPERATION_INVENTORY_SCHEMA_VERSION,
        "required_operations": list(REQUIRED_EXECUTION_TRANSPORT_OPERATIONS),
    }
)
CANCELLATION_OPERATION = "cancel_order"
CANCELLATION_SEMANTICS = (
    "authenticated_cancel_of_acknowledged_open_order_with_unknown_fail_closed"
)
TERMINAL_CURSOR_OPERATION = "read_order_fill_cursor"
TERMINAL_CURSOR_SEMANTICS = (
    "authoritative_monotonic_fill_cursor_and_terminal_order_state"
)
EMERGENCY_KILL_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-emergency-kill-v1"
)
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_micro_live_executor.py"
)
PRODUCTION_PYTHON_IMPLEMENTATION = "CPython"
PRODUCTION_PYTHON_VERSION = "3.12.4"
PRODUCTION_NUMPY_VERSION = "2.4.6"
PRODUCTION_SCIPY_VERSION = "1.17.1"
PRODUCTION_XGBOOST_VERSION = "3.2.0"
GENESIS = "GENESIS"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SLUG = re.compile(r"^btc-updown-15m-[1-9][0-9]*$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[1-9][0-9]*$")
_FORBIDDEN_FEATURE_KEY_TOKENS = ("outcome", "settlement", "resolution", "pnl")
_RAW_STATE_RESTORE_TOKEN = object()
FROZEN_EXECUTION_FEE_PER_UNIT_USD = Decimal("0.0002")
MAX_RAW_JSON_BYTES = 1_048_576
MAX_TRANSPORT_RESPONSE_BYTES = 1_048_576
MAX_EXECUTION_TRANSPORT_EVENT_BYTES = 65_536
MAX_PROVIDER_STREAM_BYTES = 16_777_216
MAX_PROVIDER_AGGREGATE_BYTES = 33_554_432
MAX_JSON_DEPTH = 32
MAX_PROVIDER_ROWS_PER_STREAM = 100_000
MAX_EVENT_COUNT = 4_096
EVENT_RECOVERY_RESERVE = 128
MAXIMUM_FILL_DELIVERY_EVENTS_PER_ORDER = 64
MAX_RESTORED_STATE_BYTES = 67_108_864
MAX_TRANSPORT_CALL_DURATION_MS = 1_000
MAX_AUTHORITY_CALL_DURATION_MS = 1_000
MAX_EXECUTION_DISPATCH_DURATION_MS = 1_000
VENUE_IDEMPOTENCY_KEY_FIELD = "client_order_id"
VENUE_IDEMPOTENCY_SCOPE = "exchange_account"
VENUE_IDEMPOTENCY_SEMANTICS = "venue_enforced_exactly_once_v1"
_JOURNAL_HEADER_MAX_BYTES = 4_096
_JOURNAL_BINDING_MAX_BYTES = 4_096
_EMERGENCY_KILL_MAX_BYTES = 16_384
# A strict JSON byte can require up to six encoded bytes when stored inside the
# WAL's raw-JSON string.  The fixed envelope covers all non-raw event fields.
MAX_SERIALIZED_RECOVERY_EVENT_BYTES = (
    6 * MAX_EXECUTION_TRANSPORT_EVENT_BYTES + 32_768
)
PROVIDER_FEATURE_FILENAMES = (
    "raw_polymarket_markets.jsonl",
    "raw_polymarket_orderbooks.jsonl",
    "raw_polymarket_trades.jsonl",
    "raw_binance_btcusdt_klines.jsonl",
    "raw_polymarket_chainlink_prices.jsonl",
)
_REQUIRED_NONEMPTY_PROVIDER_FILES = frozenset(
    {
        "raw_polymarket_markets.jsonl",
        "raw_polymarket_orderbooks.jsonl",
        "raw_binance_btcusdt_klines.jsonl",
    }
)
_FORBIDDEN_PROVIDER_RESULT_KEYS = frozenset(
    {
        "official_settlement",
        "payout",
        "payout_down",
        "payout_per_token",
        "payout_up",
        "raw_resolution_text",
        "resolution_status",
        "resolved_outcome",
        "settlement_price",
        "winner",
        "winning_outcome",
        "winning_token_id",
    }
)
_FORBIDDEN_PROVIDER_RESULT_KEY_TOKENS = ("label", "pnl", "profit")
_LOCKED_FALSE_PROVIDER_KEYS = frozenset(
    {
        "broker_exchange_write_enabled",
        "capital_at_risk",
        "live_exchange_write_enabled",
        "live_trading_allowed",
        "polymarket_write_allowed",
        "polymarket_write_enabled",
        "wallet_signing_allowed",
        "wallet_signing_enabled",
    }
)
_DYNAMIC_PROVIDER_FILENAMES = frozenset(PROVIDER_FEATURE_FILENAMES[1:])
_PROVIDER_DECISION_TIME_FIELDS = frozenset(
    {
        "available_at_ts",
        "close_time",
        "collection_end_ts",
        "orderbook_latest_covered_decision_ts",
        "orderbook_observed_collection_end_ts",
        "provider_published_at_ts",
        "received_at_ts",
        "source_ts",
        "trade_api_collection_ts",
        "trade_api_newest_ts",
        "trade_source_receive_time",
        "trade_stream_ended_at_ts",
        "ts",
    }
)
_PREMARKET_TERMINAL_STATUS_FIELDS = frozenset(
    {
        "orderbook_full_decision_window_coverage_passed",
        "trade_full_round_coverage_complete",
    }
)
_EVENT_TYPES = {
    "SIGNAL_REJECTED",
    "SIGNAL_EVALUATED",
    "ORDER_PREPARED",
    "ORDER_CANCEL_PREPARED",
    "ORDER_ACKNOWLEDGED",
    "ORDER_REJECTED",
    "ORDER_SUBMISSION_UNKNOWN",
    "ORDER_SUBMISSION_RECONCILED",
    "ORDER_SUBMISSION_RECONCILIATION_FAILED",
    "ORDER_CANCEL_RECONCILED",
    "ORDER_CANCEL_RECONCILIATION_FAILED",
    "FILL_RECORDED",
    "ORDER_FILLED",
    "ORDER_CANCELED",
    "ORDER_EXPIRED",
    "ORDER_CANCEL_UNKNOWN",
    "SETTLEMENT_RECORDED",
    "KILL_SWITCH_ENGAGED",
}
_RECOVERY_EVENT_TYPES = {
    "SIGNAL_REJECTED",
    "ORDER_ACKNOWLEDGED",
    "ORDER_REJECTED",
    "ORDER_SUBMISSION_UNKNOWN",
    "ORDER_SUBMISSION_RECONCILED",
    "ORDER_SUBMISSION_RECONCILIATION_FAILED",
    "ORDER_CANCEL_PREPARED",
    "ORDER_FILLED",
    "ORDER_CANCELED",
    "ORDER_EXPIRED",
    "ORDER_CANCEL_UNKNOWN",
    "ORDER_CANCEL_RECONCILED",
    "ORDER_CANCEL_RECONCILIATION_FAILED",
    "FILL_RECORDED",
    "SETTLEMENT_RECORDED",
    "KILL_SWITCH_ENGAGED",
}
_SIGNED_EXECUTION_OPERATIONS = frozenset(
    {"submit_order", "cancel_order", "lookup_order", "fence_order_invocation"}
)


class MicroLiveExecutionError(RuntimeError):
    """Raised whenever execution or reconciliation cannot remain deterministic."""


class SubmissionRecoveryOutcomeNotFoundError(MicroLiveExecutionError):
    """Raised only when a lookup-only recovery proves no venue outcome exists."""


class ProviderFeatureEvidenceError(ValueError):
    """Raised when exact provider inputs cannot prove one causal feature row."""


@dataclass(frozen=True, slots=True)
class DurableJournalSnapshot:
    """One fsync-committed exact executor state and its monotonic generation."""

    generation: int
    state_sha256: str
    raw_state: bytes


@dataclass(frozen=True, slots=True)
class EmergencyKillSnapshot:
    """Irreversible kill record persisted outside the main WAL lease."""

    authorization_id: str
    risk_domain_id: str
    reason: str
    event_ts_ms: int
    payload_sha256: str


class DurableRiskDomainLeaseBackend(Protocol):
    """Externally authenticated monotonic authority outside every process.

    The service must sign the exact claim receipt with the public key pinned in
    the authorization.  Backend object identity and caller-reported properties
    are deliberately outside the trust boundary.
    """

    def claim_risk_domain(
        self,
        *,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        authorization_id: str,
        risk_domain_id: str,
        journal_namespace_id: str,
    ) -> bytes:
        """Atomically bind one risk domain and return exact receipt bytes."""

    def advance_risk_domain_high_water(
        self,
        *,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        authorization_id: str,
        risk_domain_id: str,
        journal_namespace_id: str,
        journal_epoch: str,
        expected_generation: int,
        next_generation: int,
        next_state_sha256: str,
    ) -> bytes:
        """CAS one server-maintained generation/state high-water mark."""

    def persist_risk_domain_kill(
        self,
        *,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        authorization_id: str,
        risk_domain_id: str,
        journal_namespace_id: str,
        journal_epoch: str,
        reason: str,
        event_ts_ms: int,
        payload_sha256: str,
    ) -> bytes:
        """Irreversibly kill and atomically fence every ACTIVE invocation."""

    def register_execution_invocation(
        self,
        *,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        authorization_id: str,
        risk_domain_id: str,
        journal_namespace_id: str,
        journal_epoch: str,
        transport_invocation_id: str,
        operation: str,
    ) -> bytes:
        """Register one risk-increasing invocation unless already killed."""

    def commit_execution_outbox_command(
        self,
        *,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        authorization_id: str,
        risk_domain_id: str,
        journal_namespace_id: str,
        journal_epoch: str,
        transport_invocation_id: str,
        operation: str,
        fence_receipt_sha256: str,
        raw_outbox_command: bytes,
    ) -> bytes:
        """Atomically accept one exact command for dispatch; replay exact bytes."""

    def recover_execution_outbox_command(
        self,
        *,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        authorization_id: str,
        risk_domain_id: str,
        journal_namespace_id: str,
        journal_epoch: str,
        transport_invocation_id: str,
    ) -> bytes:
        """Return a signed exact outbox/acceptance recovery record."""

    def begin_execution_dispatch(
        self,
        *,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        authorization_id: str,
        risk_domain_id: str,
        journal_namespace_id: str,
        journal_epoch: str,
        transport_invocation_id: str,
        outbox_command_sha256: str,
        outbox_acceptance_receipt_sha256: str,
        venue_idempotency_key: str,
        venue_idempotency_scope: str,
        dispatch_deadline_ts_ms: int,
        authorization_expires_at_ts_ms: int,
    ) -> bytes:
        """Consume one grant before its deadline; issued grants do not auto-fence."""

    def complete_execution_dispatch(
        self,
        *,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        authorization_id: str,
        risk_domain_id: str,
        journal_namespace_id: str,
        journal_epoch: str,
        transport_invocation_id: str,
        dispatch_receipt_sha256: str,
        raw_outcome: bytes,
    ) -> bytes:
        """Persist one exact venue outcome for the consumed dispatch."""

    def recover_execution_dispatch(
        self,
        *,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        authorization_id: str,
        risk_domain_id: str,
        journal_namespace_id: str,
        journal_epoch: str,
        transport_invocation_id: str,
        outbox_command_sha256: str,
        raw_outcome: bytes,
    ) -> bytes:
        """Complete from the durable grant after an idempotent venue lookup."""

    def fence_execution_dispatch(
        self,
        *,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        authorization_id: str,
        risk_domain_id: str,
        journal_namespace_id: str,
        journal_epoch: str,
        transport_invocation_id: str,
        outbox_command_sha256: str,
    ) -> bytes:
        """Fence a not-started dispatch or report its terminal/current state."""


RISK_DOMAIN_RECEIPT_SIGNATURE_ALGORITHM = "RSASSA-PKCS1-v1_5-SHA256"
_RSA_SHA256_DIGEST_INFO_PREFIX = bytes.fromhex(
    "3031300d060960864801650304020105000420"
)


class DurableJournalTransaction(Protocol):
    """Exclusive single-writer transaction held across a risk-bearing action."""

    @property
    def generation(self) -> int:
        """Return the latest generation held by this transaction."""

    def commit(self, *, expected_generation: int, raw_state: bytes) -> int:
        """Fsync one exact next-generation state with compare-and-swap semantics."""


class MicroLiveStateJournal(Protocol):
    """Durable journal capability required by every executor instance."""

    @property
    def durable_single_writer(self) -> bool:
        """Return whether fsync plus cross-process exclusion is provided."""

    @property
    def authenticated_risk_domain_authority_binding_sha256(self) -> str | None:
        """Return the server-signed authority binding established by claim."""

    def bind_risk_domain(
        self,
        *,
        authorization_id: str,
        risk_domain_id: str,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        public_key_modulus_hex: str,
        public_key_exponent: int,
    ) -> None:
        """Bind one authorization/risk domain to this unique journal namespace."""

    def persist_emergency_kill(
        self,
        *,
        authorization_id: str,
        risk_domain_id: str,
        reason: str,
        event_ts_ms: int,
    ) -> EmergencyKillSnapshot:
        """Fsync an irreversible kill without waiting for the main WAL lease."""

    def emergency_kill_snapshot(self) -> EmergencyKillSnapshot | None:
        """Read and verify the independently persisted kill record, if present."""

    def register_execution_invocation(
        self,
        *,
        transport_invocation_id: str,
        operation: str,
    ) -> bytes:
        """Return a signed authority fence for one risk-increasing invocation."""

    def commit_execution_outbox_command(
        self,
        raw_fence_receipt: bytes,
        raw_outbox_command: bytes,
    ) -> bytes:
        """Atomically make one exact outbox command dispatchable or return FENCED."""

    def recover_execution_outbox_request(
        self,
        *,
        transport_invocation_id: str,
    ) -> bytes:
        """Recover and verify the exact accepted submit request without a grant."""

    def begin_execution_dispatch(
        self,
        raw_outbox_acceptance_receipt: bytes,
        *,
        venue_idempotency_key: str,
        venue_idempotency_scope: str,
        dispatch_deadline_ts_ms: int,
        authorization_expires_at_ts_ms: int,
    ) -> bytes:
        """Consume one command before its deadline without timing out its holder."""

    def complete_execution_dispatch(
        self,
        raw_dispatch_receipt: bytes,
        raw_outcome: bytes,
    ) -> bytes:
        """Persist the exact result of one consumed venue dispatch."""

    def recover_execution_dispatch(
        self,
        *,
        transport_invocation_id: str,
        outbox_command_sha256: str,
        raw_outcome: bytes,
    ) -> bytes:
        """Complete an in-progress dispatch after lookup-only recovery."""

    def fence_execution_dispatch(
        self,
        *,
        transport_invocation_id: str,
        outbox_command_sha256: str,
    ) -> bytes:
        """Fence a command that has not started dispatching."""

    def initialize(self, raw_state: bytes) -> DurableJournalSnapshot:
        """Create the generation-zero state, rejecting an existing journal."""

    @contextmanager
    def transaction(
        self,
        *,
        expected_generation: int,
    ) -> Iterator[DurableJournalTransaction]:
        """Hold an exclusive cross-process transaction at the expected generation."""

    def snapshot(self) -> DurableJournalSnapshot:
        """Read and independently verify the latest committed state."""


class _AtomicFileJournalTransaction:
    def __init__(
        self,
        journal: AtomicFileMicroLiveStateJournal,
        snapshot: DurableJournalSnapshot,
    ) -> None:
        self._journal = journal
        self._generation = snapshot.generation

    @property
    def generation(self) -> int:
        return self._generation

    def commit(self, *, expected_generation: int, raw_state: bytes) -> int:
        if expected_generation != self._generation:
            raise MicroLiveExecutionError(
                "micro-live journal transaction generation is stale"
            )
        next_generation = expected_generation + 1
        self._journal._commit_locked(  # noqa: SLF001 - journal transaction boundary
            expected_generation=expected_generation,
            next_generation=next_generation,
            raw_state=raw_state,
        )
        self._generation = next_generation
        return next_generation


class AtomicFileMicroLiveStateJournal:
    """Fsync/CAS journal with a process lock and atomic single-file replacement.

    The lock is held for the complete validate -> reserve -> persist -> external
    side effect -> persist transaction.  The state file contains a bounded JSON
    header followed by the exact strict-JSON executor state bytes.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        risk_domain_lease: DurableRiskDomainLeaseBackend,
    ) -> None:
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise MicroLiveExecutionError("micro-live journal root is not a directory")
        if risk_domain_lease is None:
            raise MicroLiveExecutionError(
                "external risk-domain lease service is missing"
            )
        self.root = root
        self.risk_domain_lease = risk_domain_lease
        self.state_path = root / "micro_live_state.wal"
        self.pending_state_path = root / "micro_live_state.pending"
        self.lock_path = root / "micro_live_state.lock"
        self.emergency_kill_path = root / "micro_live_emergency_kill.json"
        self.emergency_kill_pending_path = (
            root / "micro_live_emergency_kill.pending"
        )
        self.emergency_kill_lock_path = root / "micro_live_emergency_kill.lock"
        self.risk_domain_receipt_path = root / "micro_live_risk_domain_receipt.json"
        self._thread_lock = threading.RLock()
        self._emergency_thread_lock = threading.RLock()
        self._authorization_id: str | None = None
        self._risk_domain_id: str | None = None
        self._authenticated_authority_binding_sha256: str | None = None
        self._journal_namespace_id: str | None = None
        self._journal_epoch: str | None = None
        self._authority_high_water_generation: int | None = None
        self._authority_high_water_state_sha256: str | None = None
        self._authority_killed = False
        self._authority_kill_reason: str | None = None
        self._authority_kill_event_ts_ms: int | None = None
        self._authority_kill_payload_sha256: str | None = None
        self._authority_kill_uncertain = False
        self._lease_identity: dict[str, Any] | None = None

    @property
    def durable_single_writer(self) -> bool:
        return True

    @property
    def authenticated_risk_domain_authority_binding_sha256(self) -> str | None:
        return self._authenticated_authority_binding_sha256

    def bind_risk_domain(
        self,
        *,
        authorization_id: str,
        risk_domain_id: str,
        lease_id: str,
        service_identity_sha256: str,
        tenant_id: str,
        key_identity_sha256: str,
        public_key_modulus_hex: str,
        public_key_exponent: int,
    ) -> None:
        if not isinstance(authorization_id, str) or not authorization_id:
            raise MicroLiveExecutionError("journal authorization identity is invalid")
        _require_sha256(risk_domain_id, "journal risk domain")
        root_stat = self.root.stat()
        journal_namespace_id = canonical_json_sha256(
            {
                "resolved_journal_root": str(self.root),
                "root_device": root_stat.st_dev,
                "root_inode": root_stat.st_ino,
            }
        )
        try:
            raw_receipt = self._bounded_authority_call(
                "claim_risk_domain",
                lease_id=lease_id,
                service_identity_sha256=service_identity_sha256,
                tenant_id=tenant_id,
                key_identity_sha256=key_identity_sha256,
                authorization_id=authorization_id,
                risk_domain_id=risk_domain_id,
                journal_namespace_id=journal_namespace_id,
            )
        except Exception as exc:
            raise MicroLiveExecutionError(
                "external risk-domain lease claim failed closed"
            ) from exc
        receipt, receipt_json, _ = _raw_json_object(
            raw_receipt,
            "risk-domain lease receipt",
        )
        canonical_receipt = receipt_json.encode("utf-8")
        claim_status = receipt.get("claim_status")
        journal_epoch = receipt.get("journal_epoch")
        high_water_generation = receipt.get("high_water_generation")
        high_water_state_sha256 = receipt.get("high_water_state_sha256")
        killed = receipt.get("killed")
        kill_reason = receipt.get("kill_reason")
        kill_event_ts_ms = receipt.get("kill_event_ts_ms")
        kill_payload_sha256 = receipt.get("kill_payload_sha256")
        expected_receipt_core = {
            "schema_version": JOURNAL_NAMESPACE_SCHEMA_VERSION,
            "lease_id": lease_id,
            "service_identity_sha256": service_identity_sha256,
            "tenant_id": tenant_id,
            "key_identity_sha256": key_identity_sha256,
            "authorization_id": authorization_id,
            "risk_domain_id": risk_domain_id,
            "journal_namespace_id": journal_namespace_id,
            "claim_status": claim_status,
            "journal_epoch": journal_epoch,
            "high_water_generation": high_water_generation,
            "high_water_state_sha256": high_water_state_sha256,
            "killed": killed,
            "kill_reason": kill_reason,
            "kill_event_ts_ms": kill_event_ts_ms,
            "kill_payload_sha256": kill_payload_sha256,
        }
        if not (
            claim_status in {"FIRST_CLAIM", "EXISTING_CLAIM"}
            and _is_sha256(journal_epoch)
            and isinstance(high_water_generation, int)
            and not isinstance(high_water_generation, bool)
            and high_water_generation >= -1
            and (
                (
                    high_water_generation == -1
                    and high_water_state_sha256 is None
                )
                or (
                    high_water_generation >= 0
                    and _is_sha256(high_water_state_sha256)
                )
            )
            and isinstance(killed, bool)
            and (
                (
                    killed is False
                    and kill_reason is None
                    and kill_event_ts_ms is None
                    and kill_payload_sha256 is None
                )
                or (
                    killed is True
                    and isinstance(kill_reason, str)
                    and bool(kill_reason)
                    and isinstance(kill_event_ts_ms, int)
                    and not isinstance(kill_event_ts_ms, bool)
                    and kill_event_ts_ms > 0
                    and _is_sha256(kill_payload_sha256)
                )
            )
            and _verify_signed_risk_domain_receipt(
                receipt,
                expected_core=expected_receipt_core,
                public_key_modulus_hex=public_key_modulus_hex,
                public_key_exponent=public_key_exponent,
            )
        ):
            raise MicroLiveExecutionError(
                "external risk-domain lease signed receipt is invalid"
            )
        if self.risk_domain_receipt_path.exists():
            existing, _, _ = _raw_json_object(
                _read_bounded_stable_regular_file(
                    self.risk_domain_receipt_path,
                    maximum_bytes=_JOURNAL_BINDING_MAX_BYTES,
                    label="journal risk-domain receipt",
                ),
                "stored journal risk-domain receipt",
            )
            if not (
                claim_status == "EXISTING_CLAIM"
                and existing.get("authorization_id") == authorization_id
                and existing.get("risk_domain_id") == risk_domain_id
                and existing.get("journal_namespace_id") == journal_namespace_id
                and existing.get("journal_epoch") == journal_epoch
            ):
                raise MicroLiveExecutionError(
                    "journal risk-domain receipt is mismatched"
                )
        else:
            if claim_status == "FIRST_CLAIM" and (
                self.state_path.exists() or self.pending_state_path.exists()
            ):
                raise MicroLiveExecutionError(
                    "server-attested first claim has pre-existing local state"
                )
            self._atomic_write(
                path=self.risk_domain_receipt_path,
                raw_payload=canonical_receipt,
            )
        self._authorization_id = authorization_id
        self._risk_domain_id = risk_domain_id
        self._journal_namespace_id = journal_namespace_id
        self._journal_epoch = str(journal_epoch)
        self._authority_high_water_generation = int(high_water_generation)
        self._authority_high_water_state_sha256 = (
            str(high_water_state_sha256)
            if high_water_state_sha256 is not None
            else None
        )
        self._authority_killed = bool(killed)
        self._authority_kill_reason = str(kill_reason) if killed else None
        self._authority_kill_event_ts_ms = int(kill_event_ts_ms) if killed else None
        self._authority_kill_payload_sha256 = (
            str(kill_payload_sha256) if killed else None
        )
        self._lease_identity = {
            "lease_id": lease_id,
            "service_identity_sha256": service_identity_sha256,
            "tenant_id": tenant_id,
            "key_identity_sha256": key_identity_sha256,
            "public_key_modulus_hex": public_key_modulus_hex,
            "public_key_exponent": public_key_exponent,
            "claim_status": claim_status,
        }
        self._authenticated_authority_binding_sha256 = canonical_json_sha256(
            {
                "lease_id": lease_id,
                "service_identity_sha256": service_identity_sha256,
                "tenant_id": tenant_id,
                "key_identity_sha256": key_identity_sha256,
                "signature_algorithm": RISK_DOMAIN_RECEIPT_SIGNATURE_ALGORITHM,
                "public_key_modulus_hex": public_key_modulus_hex,
                "public_key_exponent": public_key_exponent,
            }
        )
        with self._exclusive_lock():
            self._recover_pending_transition_locked()
        with self._emergency_lock():
            self._recover_pending_kill_locked()

    def persist_emergency_kill(
        self,
        *,
        authorization_id: str,
        risk_domain_id: str,
        reason: str,
        event_ts_ms: int,
    ) -> EmergencyKillSnapshot:
        self._require_bound_identity(authorization_id, risk_domain_id)
        if not isinstance(reason, str) or not reason:
            raise MicroLiveExecutionError("emergency kill reason is invalid")
        _require_positive_timestamp(event_ts_ms, "emergency kill")
        payload = {
            "schema_version": EMERGENCY_KILL_SCHEMA_VERSION,
            "authorization_id": authorization_id,
            "risk_domain_id": risk_domain_id,
            "reason": reason,
            "event_ts_ms": event_ts_ms,
        }
        payload_sha256 = canonical_json_sha256(payload)
        raw_payload = self._emergency_kill_bytes(
            payload=payload,
            payload_sha256=payload_sha256,
        )
        with self._emergency_lock():
            if self._authority_killed:
                self._synchronize_authority_kill_locked()
                return self._read_emergency_kill_locked()
            if self.emergency_kill_pending_path.exists():
                pending = self._read_emergency_kill_path_locked(
                    self.emergency_kill_pending_path,
                    "pending emergency kill record",
                )
                if pending.payload_sha256 != payload_sha256:
                    raise MicroLiveExecutionError(
                        "a different emergency kill is already pending"
                    )
            else:
                self._atomic_write(
                    path=self.emergency_kill_pending_path,
                    raw_payload=raw_payload,
                )
            self._persist_authority_kill(
                reason=reason,
                event_ts_ms=event_ts_ms,
                payload_sha256=payload_sha256,
            )
            self._promote_pending_kill_locked()
            return self._read_emergency_kill_locked()

    def register_execution_invocation(
        self,
        *,
        transport_invocation_id: str,
        operation: str,
    ) -> bytes:
        self._require_bound()
        _require_sha256(transport_invocation_id, "execution invocation")
        if operation != "submit_order":
            raise MicroLiveExecutionError(
                "only risk-increasing submit invocations require a fence"
            )
        identity = self._required_lease_identity()
        with self._emergency_lock():
            self._recover_pending_kill_locked()
            if self._authority_killed:
                raise MicroLiveExecutionError(
                    "execution invocation is fenced by external kill authority"
                )
            try:
                raw_receipt = self._bounded_authority_call(
                    "register_execution_invocation",
                    lease_id=str(identity["lease_id"]),
                    service_identity_sha256=str(
                        identity["service_identity_sha256"]
                    ),
                    tenant_id=str(identity["tenant_id"]),
                    key_identity_sha256=str(identity["key_identity_sha256"]),
                    authorization_id=str(self._authorization_id),
                    risk_domain_id=str(self._risk_domain_id),
                    journal_namespace_id=str(self._journal_namespace_id),
                    journal_epoch=str(self._journal_epoch),
                    transport_invocation_id=transport_invocation_id,
                    operation=operation,
                )
            except Exception as exc:
                raise MicroLiveExecutionError(
                    "execution invocation authority registration failed closed"
                ) from exc
        receipt, receipt_json, _ = _raw_json_object(
            raw_receipt,
            "execution invocation fence receipt",
        )
        expected_core = self._execution_invocation_receipt_core(
            schema_version=EXECUTION_INVOCATION_FENCE_SCHEMA_VERSION,
            transport_invocation_id=transport_invocation_id,
            operation=operation,
            status="ACTIVE",
            fence_receipt_sha256=None,
            outbox_command_sha256=None,
        )
        if not _verify_signed_risk_domain_receipt(
            receipt,
            expected_core=expected_core,
            public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
            public_key_exponent=int(identity["public_key_exponent"]),
        ):
            raise MicroLiveExecutionError(
                "execution invocation fence receipt is invalid"
            )
        return receipt_json.encode("utf-8")

    def commit_execution_outbox_command(
        self,
        raw_fence_receipt: bytes,
        raw_outbox_command: bytes,
    ) -> bytes:
        self._require_bound()
        identity = self._required_lease_identity()
        outbox_command, outbox_command_json, outbox_command_sha256 = (
            _raw_json_object(
                raw_outbox_command,
                "execution durable outbox command",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
        )
        raw_command_json = outbox_command.get("raw_command_json")
        if not isinstance(raw_command_json, str) or not raw_command_json:
            raise MicroLiveExecutionError(
                "execution durable outbox command payload is absent"
            )
        command, canonical_command_json, command_sha256 = _raw_json_object(
            raw_command_json.encode("utf-8"),
            "execution durable outbox command payload",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        fence, _, fence_receipt_sha256 = _raw_json_object(
            raw_fence_receipt,
            "execution invocation fence receipt",
        )
        transport_invocation_id = fence.get("transport_invocation_id")
        operation = fence.get("operation")
        expected_fence_core = self._execution_invocation_receipt_core(
            schema_version=EXECUTION_INVOCATION_FENCE_SCHEMA_VERSION,
            transport_invocation_id=transport_invocation_id,
            operation=operation,
            status="ACTIVE",
            fence_receipt_sha256=None,
            outbox_command_sha256=None,
        )
        if not (
            _is_sha256(transport_invocation_id)
            and operation == "submit_order"
            and outbox_command
            == {
                "schema_version": EXECUTION_OUTBOX_COMMAND_SCHEMA_VERSION,
                "transport_invocation_id": transport_invocation_id,
                "operation": operation,
                "command_sha256": command_sha256,
                "raw_command_json": canonical_command_json,
            }
            and command.get("transport_invocation_id")
            == transport_invocation_id
            and _verify_signed_risk_domain_receipt(
                fence,
                expected_core=expected_fence_core,
                public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
                public_key_exponent=int(identity["public_key_exponent"]),
            )
        ):
            raise MicroLiveExecutionError(
                "execution invocation fence or outbox command is invalid"
            )
        try:
            raw_commit = self._bounded_authority_call(
                "commit_execution_outbox_command",
                lease_id=str(identity["lease_id"]),
                service_identity_sha256=str(identity["service_identity_sha256"]),
                tenant_id=str(identity["tenant_id"]),
                key_identity_sha256=str(identity["key_identity_sha256"]),
                authorization_id=str(self._authorization_id),
                risk_domain_id=str(self._risk_domain_id),
                journal_namespace_id=str(self._journal_namespace_id),
                journal_epoch=str(self._journal_epoch),
                transport_invocation_id=str(transport_invocation_id),
                operation=str(operation),
                fence_receipt_sha256=fence_receipt_sha256,
                raw_outbox_command=outbox_command_json.encode("utf-8"),
            )
        except Exception as exc:
            raise MicroLiveExecutionError(
                "execution durable outbox acceptance failed closed"
            ) from exc
        commit, commit_json, _ = _raw_json_object(
            raw_commit,
            "execution outbox acceptance receipt",
        )
        status = commit.get("status")
        expected_commit_core = self._execution_invocation_receipt_core(
            schema_version=EXECUTION_INVOCATION_ACCEPTANCE_SCHEMA_VERSION,
            transport_invocation_id=transport_invocation_id,
            operation=operation,
            status=status,
            fence_receipt_sha256=fence_receipt_sha256,
            outbox_command_sha256=(
                outbox_command_sha256 if status == "DISPATCHABLE" else None
            ),
        )
        if not (
            status in {"DISPATCHABLE", "FENCED"}
            and _verify_signed_risk_domain_receipt(
                commit,
                expected_core=expected_commit_core,
                public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
                public_key_exponent=int(identity["public_key_exponent"]),
            )
        ):
            raise MicroLiveExecutionError(
                "execution outbox acceptance receipt is invalid"
            )
        return commit_json.encode("utf-8")

    def recover_execution_outbox_request(
        self,
        *,
        transport_invocation_id: str,
    ) -> bytes:
        """Recover one signed exact submit request without a dispatch grant."""

        self._require_bound()
        _require_sha256(transport_invocation_id, "execution outbox recovery")
        identity = self._required_lease_identity()
        try:
            raw_recovery = self._bounded_authority_call(
                "recover_execution_outbox_command",
                lease_id=str(identity["lease_id"]),
                service_identity_sha256=str(identity["service_identity_sha256"]),
                tenant_id=str(identity["tenant_id"]),
                key_identity_sha256=str(identity["key_identity_sha256"]),
                authorization_id=str(self._authorization_id),
                risk_domain_id=str(self._risk_domain_id),
                journal_namespace_id=str(self._journal_namespace_id),
                journal_epoch=str(self._journal_epoch),
                transport_invocation_id=transport_invocation_id,
            )
        except Exception as exc:
            raise MicroLiveExecutionError(
                "execution outbox recovery failed closed"
            ) from exc
        recovery, _, _ = _raw_json_object(
            raw_recovery,
            "execution outbox recovery record",
            maximum_bytes=MAX_SERIALIZED_RECOVERY_EVENT_BYTES,
        )
        raw_outbox_json = recovery.get("raw_outbox_command_json")
        raw_acceptance_json = recovery.get(
            "raw_outbox_acceptance_receipt_json"
        )
        if not (
            isinstance(raw_outbox_json, str)
            and raw_outbox_json
            and isinstance(raw_acceptance_json, str)
            and raw_acceptance_json
        ):
            raise MicroLiveExecutionError(
                "execution outbox recovery material is absent"
            )
        outbox, canonical_outbox_json, outbox_sha256 = _raw_json_object(
            raw_outbox_json.encode("utf-8"),
            "recovered execution outbox command",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        acceptance, canonical_acceptance_json, acceptance_sha256 = (
            _raw_json_object(
                raw_acceptance_json.encode("utf-8"),
                "recovered execution outbox acceptance",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
        )
        expected_recovery_core = {
            **self._authority_receipt_identity_core(),
            "schema_version": EXECUTION_OUTBOX_RECOVERY_SCHEMA_VERSION,
            "transport_invocation_id": transport_invocation_id,
            "operation": "submit_order",
            "outbox_command_sha256": outbox_sha256,
            "raw_outbox_command_json": canonical_outbox_json,
            "outbox_acceptance_receipt_sha256": acceptance_sha256,
            "raw_outbox_acceptance_receipt_json": canonical_acceptance_json,
        }
        expected_acceptance_core = self._execution_invocation_receipt_core(
            schema_version=EXECUTION_INVOCATION_ACCEPTANCE_SCHEMA_VERSION,
            transport_invocation_id=transport_invocation_id,
            operation="submit_order",
            status="DISPATCHABLE",
            fence_receipt_sha256=acceptance.get("fence_receipt_sha256"),
            outbox_command_sha256=outbox_sha256,
        )
        raw_command_json = outbox.get("raw_command_json")
        if not (
            _verify_signed_risk_domain_receipt(
                recovery,
                expected_core=expected_recovery_core,
                public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
                public_key_exponent=int(identity["public_key_exponent"]),
            )
            and outbox.get("schema_version")
            == EXECUTION_OUTBOX_COMMAND_SCHEMA_VERSION
            and outbox.get("transport_invocation_id")
            == transport_invocation_id
            and outbox.get("operation") == "submit_order"
            and isinstance(raw_command_json, str)
            and raw_command_json
            and hashlib.sha256(raw_command_json.encode("utf-8")).hexdigest()
            == outbox.get("command_sha256")
            and acceptance_sha256
            == recovery.get("outbox_acceptance_receipt_sha256")
            and acceptance.get("outbox_command_sha256") == outbox_sha256
            and _verify_signed_risk_domain_receipt(
                acceptance,
                expected_core=expected_acceptance_core,
                public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
                public_key_exponent=int(identity["public_key_exponent"]),
            )
        ):
            raise MicroLiveExecutionError(
                "execution outbox recovery record is invalid"
            )
        command, _, _ = _raw_json_object(
            raw_command_json.encode("utf-8"),
            "recovered execution submit command",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        authentication = dict(command.get("execution_authentication") or {})
        authentication.update(
            {
                "execution_outbox_command_json": canonical_outbox_json,
                "execution_outbox_command_sha256": outbox_sha256,
                "raw_execution_outbox_acceptance_receipt_json": (
                    canonical_acceptance_json
                ),
                "execution_outbox_acceptance_receipt_sha256": (
                    acceptance_sha256
                ),
            }
        )
        command["execution_authentication"] = authentication
        verify_dispatchable_outbox_request(
            command,
            authorization_id=str(self._authorization_id),
            risk_domain_id=str(self._risk_domain_id),
            risk_domain_authority_binding_sha256=str(
                self._authenticated_authority_binding_sha256
            ),
            lease_id=str(identity["lease_id"]),
            service_identity_sha256=str(identity["service_identity_sha256"]),
            tenant_id=str(identity["tenant_id"]),
            key_identity_sha256=str(identity["key_identity_sha256"]),
            public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
            public_key_exponent=int(identity["public_key_exponent"]),
        )
        return _canonical_json_bytes(command)

    def begin_execution_dispatch(
        self,
        raw_outbox_acceptance_receipt: bytes,
        *,
        venue_idempotency_key: str,
        venue_idempotency_scope: str,
        dispatch_deadline_ts_ms: int,
        authorization_expires_at_ts_ms: int,
    ) -> bytes:
        self._require_bound()
        identity = self._required_lease_identity()
        acceptance, _, acceptance_receipt_sha256 = _raw_json_object(
            raw_outbox_acceptance_receipt,
            "execution outbox acceptance receipt",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        transport_invocation_id = acceptance.get("transport_invocation_id")
        outbox_command_sha256 = acceptance.get("outbox_command_sha256")
        expected_acceptance = self._execution_invocation_receipt_core(
            schema_version=EXECUTION_INVOCATION_ACCEPTANCE_SCHEMA_VERSION,
            transport_invocation_id=transport_invocation_id,
            operation="submit_order",
            status="DISPATCHABLE",
            fence_receipt_sha256=acceptance.get("fence_receipt_sha256"),
            outbox_command_sha256=outbox_command_sha256,
        )
        if not (
            _is_sha256(transport_invocation_id)
            and _is_sha256(outbox_command_sha256)
            and venue_idempotency_scope == VENUE_IDEMPOTENCY_SCOPE
            and isinstance(venue_idempotency_key, str)
            and bool(venue_idempotency_key)
            and isinstance(dispatch_deadline_ts_ms, int)
            and not isinstance(dispatch_deadline_ts_ms, bool)
            and isinstance(authorization_expires_at_ts_ms, int)
            and not isinstance(authorization_expires_at_ts_ms, bool)
            and _verify_signed_risk_domain_receipt(
                acceptance,
                expected_core=expected_acceptance,
                public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
                public_key_exponent=int(identity["public_key_exponent"]),
            )
        ):
            raise MicroLiveExecutionError(
                "execution dispatch outbox acceptance is invalid"
            )
        try:
            raw_receipt = self._bounded_authority_call(
                "begin_execution_dispatch",
                lease_id=str(identity["lease_id"]),
                service_identity_sha256=str(identity["service_identity_sha256"]),
                tenant_id=str(identity["tenant_id"]),
                key_identity_sha256=str(identity["key_identity_sha256"]),
                authorization_id=str(self._authorization_id),
                risk_domain_id=str(self._risk_domain_id),
                journal_namespace_id=str(self._journal_namespace_id),
                journal_epoch=str(self._journal_epoch),
                transport_invocation_id=str(transport_invocation_id),
                outbox_command_sha256=str(outbox_command_sha256),
                outbox_acceptance_receipt_sha256=acceptance_receipt_sha256,
                venue_idempotency_key=venue_idempotency_key,
                venue_idempotency_scope=venue_idempotency_scope,
                dispatch_deadline_ts_ms=dispatch_deadline_ts_ms,
                authorization_expires_at_ts_ms=authorization_expires_at_ts_ms,
            )
        except Exception as exc:
            raise MicroLiveExecutionError(
                "execution dispatch consumption failed closed"
            ) from exc
        receipt, receipt_json, _ = _raw_json_object(
            raw_receipt,
            "execution dispatch receipt",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        status = receipt.get("status")
        raw_outcome_json = receipt.get("raw_outcome_json")
        outcome_sha256 = receipt.get("outcome_sha256")
        expected_core = self._execution_dispatch_receipt_core(
            schema_version=EXECUTION_DISPATCH_RECEIPT_SCHEMA_VERSION,
            transport_invocation_id=transport_invocation_id,
            status=status,
            outbox_command_sha256=outbox_command_sha256,
            outbox_acceptance_receipt_sha256=acceptance_receipt_sha256,
            venue_idempotency_key=venue_idempotency_key,
            venue_idempotency_scope=venue_idempotency_scope,
            dispatch_deadline_ts_ms=dispatch_deadline_ts_ms,
            authorization_expires_at_ts_ms=authorization_expires_at_ts_ms,
            dispatch_receipt_sha256=None,
            raw_outcome_json=raw_outcome_json,
            outcome_sha256=outcome_sha256,
        )
        if not (
            status
            in {
                "DISPATCHING",
                "IN_PROGRESS",
                "DISPATCHED",
                "FENCED",
                "EXPIRED",
            }
            and (status == "DISPATCHED")
            == (
                isinstance(raw_outcome_json, str)
                and bool(raw_outcome_json)
                and _is_sha256(outcome_sha256)
                and hashlib.sha256(raw_outcome_json.encode("utf-8")).hexdigest()
                == outcome_sha256
            )
            and _verify_signed_risk_domain_receipt(
                receipt,
                expected_core=expected_core,
                public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
                public_key_exponent=int(identity["public_key_exponent"]),
            )
        ):
            raise MicroLiveExecutionError("execution dispatch receipt is invalid")
        return receipt_json.encode("utf-8")

    def complete_execution_dispatch(
        self,
        raw_dispatch_receipt: bytes,
        raw_outcome: bytes,
    ) -> bytes:
        self._require_bound()
        identity = self._required_lease_identity()
        dispatch, dispatch_json, dispatch_receipt_sha256 = _raw_json_object(
            raw_dispatch_receipt,
            "execution dispatch receipt",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        _, outcome_json, outcome_sha256 = _raw_json_object(
            raw_outcome,
            "execution dispatch outcome",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        if dispatch.get("status") != "DISPATCHING":
            raise MicroLiveExecutionError(
                "only a newly consumed dispatch can record an outcome"
            )
        expected_dispatch = self._execution_dispatch_receipt_core(
            schema_version=EXECUTION_DISPATCH_RECEIPT_SCHEMA_VERSION,
            transport_invocation_id=dispatch.get("transport_invocation_id"),
            status="DISPATCHING",
            outbox_command_sha256=dispatch.get("outbox_command_sha256"),
            outbox_acceptance_receipt_sha256=dispatch.get(
                "outbox_acceptance_receipt_sha256"
            ),
            venue_idempotency_key=dispatch.get("venue_idempotency_key"),
            venue_idempotency_scope=dispatch.get("venue_idempotency_scope"),
            dispatch_deadline_ts_ms=dispatch.get("dispatch_deadline_ts_ms"),
            authorization_expires_at_ts_ms=dispatch.get(
                "authorization_expires_at_ts_ms"
            ),
            dispatch_receipt_sha256=None,
            raw_outcome_json=None,
            outcome_sha256=None,
        )
        if not _verify_signed_risk_domain_receipt(
            dispatch,
            expected_core=expected_dispatch,
            public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
            public_key_exponent=int(identity["public_key_exponent"]),
        ):
            raise MicroLiveExecutionError("execution dispatch receipt is invalid")
        try:
            raw_completion = self._bounded_authority_call(
                "complete_execution_dispatch",
                lease_id=str(identity["lease_id"]),
                service_identity_sha256=str(identity["service_identity_sha256"]),
                tenant_id=str(identity["tenant_id"]),
                key_identity_sha256=str(identity["key_identity_sha256"]),
                authorization_id=str(self._authorization_id),
                risk_domain_id=str(self._risk_domain_id),
                journal_namespace_id=str(self._journal_namespace_id),
                journal_epoch=str(self._journal_epoch),
                transport_invocation_id=str(
                    dispatch["transport_invocation_id"]
                ),
                dispatch_receipt_sha256=dispatch_receipt_sha256,
                raw_outcome=outcome_json.encode("utf-8"),
            )
        except Exception as exc:
            raise MicroLiveExecutionError(
                "execution dispatch completion failed closed"
            ) from exc
        completion, completion_json, _ = _raw_json_object(
            raw_completion,
            "execution dispatch completion receipt",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        expected_completion = self._execution_dispatch_receipt_core(
            schema_version=EXECUTION_DISPATCH_COMPLETION_SCHEMA_VERSION,
            transport_invocation_id=dispatch["transport_invocation_id"],
            status="DISPATCHED",
            outbox_command_sha256=dispatch["outbox_command_sha256"],
            outbox_acceptance_receipt_sha256=dispatch[
                "outbox_acceptance_receipt_sha256"
            ],
            venue_idempotency_key=dispatch["venue_idempotency_key"],
            venue_idempotency_scope=dispatch["venue_idempotency_scope"],
            dispatch_deadline_ts_ms=dispatch["dispatch_deadline_ts_ms"],
            authorization_expires_at_ts_ms=dispatch[
                "authorization_expires_at_ts_ms"
            ],
            dispatch_receipt_sha256=dispatch_receipt_sha256,
            raw_outcome_json=outcome_json,
            outcome_sha256=outcome_sha256,
        )
        if not _verify_signed_risk_domain_receipt(
            completion,
            expected_core=expected_completion,
            public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
            public_key_exponent=int(identity["public_key_exponent"]),
        ):
            raise MicroLiveExecutionError(
                "execution dispatch completion receipt is invalid"
            )
        return completion_json.encode("utf-8")

    def recover_execution_dispatch(
        self,
        *,
        transport_invocation_id: str,
        outbox_command_sha256: str,
        raw_outcome: bytes,
    ) -> bytes:
        """Complete from authority state after a lookup-only venue recovery.

        The raw DISPATCHING receipt remains inside the durable authority.  A
        replacement gateway supplies only the exact result returned by the
        venue's idempotency lookup, so this recovery capability cannot become
        a second bearer grant for submission.
        """

        self._require_bound()
        _require_sha256(transport_invocation_id, "execution dispatch")
        _require_sha256(outbox_command_sha256, "execution outbox command")
        _, outcome_json, outcome_sha256 = _raw_json_object(
            raw_outcome,
            "recovered execution dispatch outcome",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        identity = self._required_lease_identity()
        try:
            raw_receipt = self._bounded_authority_call(
                "recover_execution_dispatch",
                lease_id=str(identity["lease_id"]),
                service_identity_sha256=str(identity["service_identity_sha256"]),
                tenant_id=str(identity["tenant_id"]),
                key_identity_sha256=str(identity["key_identity_sha256"]),
                authorization_id=str(self._authorization_id),
                risk_domain_id=str(self._risk_domain_id),
                journal_namespace_id=str(self._journal_namespace_id),
                journal_epoch=str(self._journal_epoch),
                transport_invocation_id=transport_invocation_id,
                outbox_command_sha256=outbox_command_sha256,
                raw_outcome=outcome_json.encode("utf-8"),
            )
        except Exception as exc:
            raise MicroLiveExecutionError(
                "execution dispatch recovery failed closed"
            ) from exc
        receipt, receipt_json, _ = _raw_json_object(
            raw_receipt,
            "recovered execution dispatch receipt",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        expected_core = self._execution_dispatch_receipt_core(
            schema_version=EXECUTION_DISPATCH_COMPLETION_SCHEMA_VERSION,
            transport_invocation_id=transport_invocation_id,
            status="DISPATCHED",
            outbox_command_sha256=outbox_command_sha256,
            outbox_acceptance_receipt_sha256=receipt.get(
                "outbox_acceptance_receipt_sha256"
            ),
            venue_idempotency_key=receipt.get("venue_idempotency_key"),
            venue_idempotency_scope=receipt.get("venue_idempotency_scope"),
            dispatch_deadline_ts_ms=receipt.get("dispatch_deadline_ts_ms"),
            authorization_expires_at_ts_ms=receipt.get(
                "authorization_expires_at_ts_ms"
            ),
            dispatch_receipt_sha256=receipt.get("dispatch_receipt_sha256"),
            raw_outcome_json=outcome_json,
            outcome_sha256=outcome_sha256,
        )
        if not (
            receipt.get("status") == "DISPATCHED"
            and _is_sha256(receipt.get("outbox_acceptance_receipt_sha256"))
            and _is_sha256(receipt.get("dispatch_receipt_sha256"))
            and receipt.get("venue_idempotency_scope") == VENUE_IDEMPOTENCY_SCOPE
            and isinstance(receipt.get("venue_idempotency_key"), str)
            and bool(receipt.get("venue_idempotency_key"))
            and isinstance(receipt.get("dispatch_deadline_ts_ms"), int)
            and not isinstance(receipt.get("dispatch_deadline_ts_ms"), bool)
            and isinstance(receipt.get("authorization_expires_at_ts_ms"), int)
            and not isinstance(
                receipt.get("authorization_expires_at_ts_ms"), bool
            )
            and _verify_signed_risk_domain_receipt(
                receipt,
                expected_core=expected_core,
                public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
                public_key_exponent=int(identity["public_key_exponent"]),
            )
        ):
            raise MicroLiveExecutionError(
                "recovered execution dispatch completion is invalid"
            )
        return receipt_json.encode("utf-8")

    def fence_execution_dispatch(
        self,
        *,
        transport_invocation_id: str,
        outbox_command_sha256: str,
    ) -> bytes:
        self._require_bound()
        _require_sha256(transport_invocation_id, "execution dispatch")
        _require_sha256(outbox_command_sha256, "execution outbox command")
        identity = self._required_lease_identity()
        try:
            raw_receipt = self._bounded_authority_call(
                "fence_execution_dispatch",
                lease_id=str(identity["lease_id"]),
                service_identity_sha256=str(identity["service_identity_sha256"]),
                tenant_id=str(identity["tenant_id"]),
                key_identity_sha256=str(identity["key_identity_sha256"]),
                authorization_id=str(self._authorization_id),
                risk_domain_id=str(self._risk_domain_id),
                journal_namespace_id=str(self._journal_namespace_id),
                journal_epoch=str(self._journal_epoch),
                transport_invocation_id=transport_invocation_id,
                outbox_command_sha256=outbox_command_sha256,
            )
        except Exception as exc:
            raise MicroLiveExecutionError(
                "execution dispatch fence failed closed"
            ) from exc
        receipt, receipt_json, _ = _raw_json_object(
            raw_receipt,
            "execution dispatch fence receipt",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        expected_core = {
            **self._authority_receipt_identity_core(),
            "schema_version": EXECUTION_DISPATCH_FENCE_SCHEMA_VERSION,
            "transport_invocation_id": transport_invocation_id,
            "outbox_command_sha256": outbox_command_sha256,
            "status": receipt.get("status"),
        }
        if not (
            receipt.get("status") in {"FENCED", "IN_PROGRESS", "DISPATCHED"}
            and _verify_signed_risk_domain_receipt(
                receipt,
                expected_core=expected_core,
                public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
                public_key_exponent=int(identity["public_key_exponent"]),
            )
        ):
            raise MicroLiveExecutionError("execution dispatch fence is invalid")
        return receipt_json.encode("utf-8")

    def emergency_kill_snapshot(self) -> EmergencyKillSnapshot | None:
        with self._emergency_lock():
            self._recover_pending_kill_locked()
            if not self._authority_killed:
                if self.emergency_kill_path.exists():
                    raise MicroLiveExecutionError(
                        "local emergency kill lacks external authority backing"
                    )
                return None
            return self._read_emergency_kill_locked()

    def initialize(self, raw_state: bytes) -> DurableJournalSnapshot:
        self._require_bound()
        _require_bounded_state_bytes(raw_state)
        with self._exclusive_lock():
            if self.pending_state_path.exists():
                raise MicroLiveExecutionError(
                    "pending journal initialization requires authority rebind recovery"
                )
            if self.state_path.exists():
                raise MicroLiveExecutionError(
                    "micro-live journal already exists; strict restore is required"
                )
            generation = _state_generation(raw_state)
            if not (
                generation == 0
                and self._lease_identity is not None
                and self._lease_identity.get("claim_status")
                in {"FIRST_CLAIM", "EXISTING_CLAIM"}
                and self._authority_high_water_generation == -1
                and self._authority_high_water_state_sha256 is None
                and not self.pending_state_path.exists()
                and not self.state_path.exists()
            ):
                raise MicroLiveExecutionError(
                    "generation zero requires an uninitialized exact authority namespace"
                )
            self._write_pending_locked(generation=generation, raw_state=raw_state)
            self._advance_authority_high_water(
                expected_generation=-1,
                next_generation=0,
                next_state_sha256=hashlib.sha256(raw_state).hexdigest(),
            )
            self._promote_pending_locked()
            return self._read_locked()

    @contextmanager
    def transaction(
        self,
        *,
        expected_generation: int,
    ) -> Iterator[DurableJournalTransaction]:
        self._require_bound()
        with self._exclusive_lock():
            if self.pending_state_path.exists():
                raise MicroLiveExecutionError(
                    "pending journal transition requires authority rebind recovery"
                )
            snapshot = self._read_locked()
            self._require_authority_high_water_matches(snapshot)
            if snapshot.generation != expected_generation:
                raise MicroLiveExecutionError(
                    "micro-live journal high-water generation is stale"
                )
            transaction = _AtomicFileJournalTransaction(self, snapshot)
            yield transaction

    def snapshot(self) -> DurableJournalSnapshot:
        self._require_bound()
        with self._exclusive_lock():
            if self.pending_state_path.exists():
                raise MicroLiveExecutionError(
                    "pending journal transition requires authority rebind recovery"
                )
            snapshot = self._read_locked()
            self._require_authority_high_water_matches(snapshot)
            return snapshot

    def _require_bound(self) -> None:
        if self._authorization_id is None or self._risk_domain_id is None:
            raise MicroLiveExecutionError(
                "micro-live journal risk domain is not bound"
            )

    def _require_bound_identity(
        self,
        authorization_id: str,
        risk_domain_id: str,
    ) -> None:
        self._require_bound()
        if not (
            authorization_id == self._authorization_id
            and risk_domain_id == self._risk_domain_id
        ):
            raise MicroLiveExecutionError(
                "micro-live journal risk-domain identity is mismatched"
            )

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        with self._thread_lock:
            descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    @contextmanager
    def _emergency_lock(self) -> Iterator[None]:
        with self._emergency_thread_lock:
            descriptor = os.open(
                self.emergency_kill_lock_path,
                os.O_RDWR | os.O_CREAT,
                0o600,
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _commit_locked(
        self,
        *,
        expected_generation: int,
        next_generation: int,
        raw_state: bytes,
    ) -> None:
        current = self._read_locked()
        if not (
            current.generation == expected_generation
            and next_generation == expected_generation + 1
            and _state_generation(raw_state) == next_generation
        ):
            raise MicroLiveExecutionError("micro-live journal CAS failed closed")
        self._require_authority_high_water_matches(current)
        self._write_pending_locked(generation=next_generation, raw_state=raw_state)
        self._advance_authority_high_water(
            expected_generation=expected_generation,
            next_generation=next_generation,
            next_state_sha256=hashlib.sha256(raw_state).hexdigest(),
        )
        self._promote_pending_locked()

    def _require_authority_high_water_matches(
        self,
        snapshot: DurableJournalSnapshot,
    ) -> None:
        if not (
            snapshot.generation == self._authority_high_water_generation
            and snapshot.state_sha256 == self._authority_high_water_state_sha256
        ):
            raise MicroLiveExecutionError(
                "local WAL does not match the external authority high-water"
            )

    def _advance_authority_high_water(
        self,
        *,
        expected_generation: int,
        next_generation: int,
        next_state_sha256: str,
    ) -> None:
        self._require_bound()
        identity = self._lease_identity
        if not (
            identity is not None
            and self._journal_namespace_id is not None
            and self._journal_epoch is not None
            and self._authorization_id is not None
            and self._risk_domain_id is not None
            and expected_generation == self._authority_high_water_generation
            and next_generation == expected_generation + 1
            and _is_sha256(next_state_sha256)
        ):
            raise MicroLiveExecutionError(
                "authority high-water transition precondition failed"
            )
        try:
            raw_receipt = self._bounded_authority_call(
                "advance_risk_domain_high_water",
                lease_id=str(identity["lease_id"]),
                service_identity_sha256=str(
                    identity["service_identity_sha256"]
                ),
                tenant_id=str(identity["tenant_id"]),
                key_identity_sha256=str(identity["key_identity_sha256"]),
                authorization_id=self._authorization_id,
                risk_domain_id=self._risk_domain_id,
                journal_namespace_id=self._journal_namespace_id,
                journal_epoch=self._journal_epoch,
                expected_generation=expected_generation,
                next_generation=next_generation,
                next_state_sha256=next_state_sha256,
            )
        except Exception as exc:
            raise MicroLiveExecutionError(
                "external authority high-water advance failed closed"
            ) from exc
        receipt, _, _ = _raw_json_object(
            raw_receipt,
            "risk-domain authority high-water receipt",
        )
        expected_core = {
            "schema_version": JOURNAL_HIGH_WATER_SCHEMA_VERSION,
            "lease_id": identity["lease_id"],
            "service_identity_sha256": identity["service_identity_sha256"],
            "tenant_id": identity["tenant_id"],
            "key_identity_sha256": identity["key_identity_sha256"],
            "authorization_id": self._authorization_id,
            "risk_domain_id": self._risk_domain_id,
            "journal_namespace_id": self._journal_namespace_id,
            "journal_epoch": self._journal_epoch,
            "previous_generation": expected_generation,
            "high_water_generation": next_generation,
            "high_water_state_sha256": next_state_sha256,
        }
        if not _verify_signed_risk_domain_receipt(
            receipt,
            expected_core=expected_core,
            public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
            public_key_exponent=int(identity["public_key_exponent"]),
        ):
            raise MicroLiveExecutionError(
                "external authority high-water receipt is invalid"
            )
        self._authority_high_water_generation = next_generation
        self._authority_high_water_state_sha256 = next_state_sha256

    def _bounded_authority_call(
        self,
        authority_operation: str,
        **kwargs: Any,
    ) -> bytes:
        method = getattr(self.risk_domain_lease, authority_operation, None)
        if not callable(method):
            raise MicroLiveExecutionError(
                f"external authority operation {authority_operation} is absent"
            )
        completed = threading.Event()
        results: queue.SimpleQueue[tuple[bool, Any]] = queue.SimpleQueue()

        def invoke() -> None:
            try:
                results.put((True, method(**copy.deepcopy(kwargs))))
            except BaseException as exc:
                results.put((False, exc))
            finally:
                completed.set()

        threading.Thread(
            target=invoke,
            name=f"micro-live-authority-{authority_operation}",
            daemon=True,
        ).start()
        if not completed.wait(MAX_AUTHORITY_CALL_DURATION_MS / 1_000):
            raise MicroLiveExecutionError(
                "external authority operation "
                f"{authority_operation} exceeded its deadline"
            )
        succeeded, value = results.get_nowait()
        if not succeeded:
            raise value
        if not isinstance(value, bytes):
            raise MicroLiveExecutionError(
                "external authority operation "
                f"{authority_operation} returned non-bytes"
            )
        return value

    def _persist_authority_kill(
        self,
        *,
        reason: str,
        event_ts_ms: int,
        payload_sha256: str,
    ) -> None:
        identity = self._required_lease_identity()
        self._authority_kill_uncertain = True
        try:
            raw_receipt = self._bounded_authority_call(
                "persist_risk_domain_kill",
                lease_id=str(identity["lease_id"]),
                service_identity_sha256=str(identity["service_identity_sha256"]),
                tenant_id=str(identity["tenant_id"]),
                key_identity_sha256=str(identity["key_identity_sha256"]),
                authorization_id=self._authorization_id,
                risk_domain_id=self._risk_domain_id,
                journal_namespace_id=self._journal_namespace_id,
                journal_epoch=self._journal_epoch,
                reason=reason,
                event_ts_ms=event_ts_ms,
                payload_sha256=payload_sha256,
            )
        except Exception as exc:
            raise MicroLiveExecutionError(
                "external authority emergency kill failed closed"
            ) from exc
        receipt, _, _ = _raw_json_object(raw_receipt, "risk-domain kill receipt")
        expected_core = {
            "schema_version": JOURNAL_KILL_RECEIPT_SCHEMA_VERSION,
            "lease_id": identity["lease_id"],
            "service_identity_sha256": identity["service_identity_sha256"],
            "tenant_id": identity["tenant_id"],
            "key_identity_sha256": identity["key_identity_sha256"],
            "authorization_id": self._authorization_id,
            "risk_domain_id": self._risk_domain_id,
            "journal_namespace_id": self._journal_namespace_id,
            "journal_epoch": self._journal_epoch,
            "reason": reason,
            "event_ts_ms": event_ts_ms,
            "payload_sha256": payload_sha256,
            "killed": True,
        }
        if not _verify_signed_risk_domain_receipt(
            receipt,
            expected_core=expected_core,
            public_key_modulus_hex=str(identity["public_key_modulus_hex"]),
            public_key_exponent=int(identity["public_key_exponent"]),
        ):
            raise MicroLiveExecutionError("external authority kill receipt is invalid")
        self._authority_killed = True
        self._authority_kill_reason = reason
        self._authority_kill_event_ts_ms = event_ts_ms
        self._authority_kill_payload_sha256 = payload_sha256
        self._authority_kill_uncertain = False

    def _required_lease_identity(self) -> dict[str, Any]:
        identity = self._lease_identity
        if not (
            identity is not None
            and self._journal_namespace_id is not None
            and self._journal_epoch is not None
            and self._authorization_id is not None
            and self._risk_domain_id is not None
        ):
            raise MicroLiveExecutionError(
                "external authority identity precondition failed"
            )
        return identity

    def _execution_invocation_receipt_core(
        self,
        *,
        schema_version: str,
        transport_invocation_id: Any,
        operation: Any,
        status: Any,
        fence_receipt_sha256: Any,
        outbox_command_sha256: Any,
    ) -> dict[str, Any]:
        identity = self._required_lease_identity()
        core = {
            "schema_version": schema_version,
            "lease_id": identity["lease_id"],
            "service_identity_sha256": identity["service_identity_sha256"],
            "tenant_id": identity["tenant_id"],
            "key_identity_sha256": identity["key_identity_sha256"],
            "authorization_id": self._authorization_id,
            "risk_domain_id": self._risk_domain_id,
            "journal_namespace_id": self._journal_namespace_id,
            "journal_epoch": self._journal_epoch,
            "transport_invocation_id": transport_invocation_id,
            "operation": operation,
            "status": status,
        }
        if schema_version == EXECUTION_INVOCATION_ACCEPTANCE_SCHEMA_VERSION:
            core["fence_receipt_sha256"] = fence_receipt_sha256
            core["outbox_command_sha256"] = outbox_command_sha256
        return core

    def _authority_receipt_identity_core(self) -> dict[str, Any]:
        identity = self._required_lease_identity()
        return {
            "lease_id": identity["lease_id"],
            "service_identity_sha256": identity["service_identity_sha256"],
            "tenant_id": identity["tenant_id"],
            "key_identity_sha256": identity["key_identity_sha256"],
            "authorization_id": self._authorization_id,
            "risk_domain_id": self._risk_domain_id,
            "journal_namespace_id": self._journal_namespace_id,
            "journal_epoch": self._journal_epoch,
        }

    def _execution_dispatch_receipt_core(
        self,
        *,
        schema_version: str,
        transport_invocation_id: Any,
        status: Any,
        outbox_command_sha256: Any,
        outbox_acceptance_receipt_sha256: Any,
        venue_idempotency_key: Any,
        venue_idempotency_scope: Any,
        dispatch_deadline_ts_ms: Any,
        authorization_expires_at_ts_ms: Any,
        dispatch_receipt_sha256: Any,
        raw_outcome_json: Any,
        outcome_sha256: Any,
    ) -> dict[str, Any]:
        return {
            **self._authority_receipt_identity_core(),
            "schema_version": schema_version,
            "transport_invocation_id": transport_invocation_id,
            "operation": "submit_order",
            "status": status,
            "outbox_command_sha256": outbox_command_sha256,
            "outbox_acceptance_receipt_sha256": (
                outbox_acceptance_receipt_sha256
            ),
            "venue_idempotency_key": venue_idempotency_key,
            "venue_idempotency_scope": venue_idempotency_scope,
            "dispatch_deadline_ts_ms": dispatch_deadline_ts_ms,
            "authorization_expires_at_ts_ms": authorization_expires_at_ts_ms,
            "dispatch_receipt_sha256": dispatch_receipt_sha256,
            "raw_outcome_json": raw_outcome_json,
            "outcome_sha256": outcome_sha256,
        }

    def _recover_pending_kill_locked(self) -> None:
        if self.emergency_kill_pending_path.exists():
            pending = self._read_emergency_kill_path_locked(
                self.emergency_kill_pending_path,
                "pending emergency kill record",
            )
            if self._authority_killed:
                if pending.payload_sha256 != self._authority_kill_payload_sha256:
                    raise MicroLiveExecutionError(
                        "pending emergency kill conflicts with external authority"
                    )
            else:
                self._persist_authority_kill(
                    reason=pending.reason,
                    event_ts_ms=pending.event_ts_ms,
                    payload_sha256=pending.payload_sha256,
                )
            self._promote_pending_kill_locked()
        self._synchronize_authority_kill_locked()

    def _promote_pending_kill_locked(self) -> None:
        pending = self._read_emergency_kill_path_locked(
            self.emergency_kill_pending_path,
            "pending emergency kill record",
        )
        if not (
            self._authority_killed
            and pending.payload_sha256 == self._authority_kill_payload_sha256
        ):
            raise MicroLiveExecutionError(
                "pending emergency kill lacks matching external authority"
            )
        if self.emergency_kill_path.exists():
            existing = self._read_emergency_kill_locked()
            if existing != pending:
                raise MicroLiveExecutionError(
                    "pending emergency kill conflicts with committed local kill"
                )
            self.emergency_kill_pending_path.unlink()
        else:
            os.replace(
                self.emergency_kill_pending_path,
                self.emergency_kill_path,
            )
        directory_descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    def _synchronize_authority_kill_locked(self) -> None:
        if self._authority_kill_uncertain:
            raise MicroLiveExecutionError(
                "external authority kill result is uncertain; restart is required"
            )
        if not self._authority_killed:
            return
        payload = {
            "schema_version": EMERGENCY_KILL_SCHEMA_VERSION,
            "authorization_id": self._authorization_id,
            "risk_domain_id": self._risk_domain_id,
            "reason": self._authority_kill_reason,
            "event_ts_ms": self._authority_kill_event_ts_ms,
        }
        if canonical_json_sha256(payload) != self._authority_kill_payload_sha256:
            raise MicroLiveExecutionError("external authority kill payload is invalid")
        if not self.emergency_kill_path.exists():
            self._atomic_write(
                path=self.emergency_kill_path,
                raw_payload=self._emergency_kill_bytes(
                    payload=payload,
                    payload_sha256=str(self._authority_kill_payload_sha256),
                ),
            )
        snapshot = self._read_emergency_kill_locked()
        if snapshot.payload_sha256 != self._authority_kill_payload_sha256:
            raise MicroLiveExecutionError(
                "local emergency kill conflicts with external authority"
            )

    def _recover_pending_transition_locked(self) -> None:
        local = self._read_locked() if self.state_path.exists() else None
        pending = (
            self._read_snapshot_path(self.pending_state_path, "pending journal state")
            if self.pending_state_path.exists()
            else None
        )
        if pending is not None:
            expected_generation = local.generation if local is not None else -1
            expected_state_sha256 = local.state_sha256 if local is not None else None
            if pending.generation != expected_generation + 1:
                raise MicroLiveExecutionError("pending journal generation is invalid")
            remote_matches_pending = (
                self._authority_high_water_generation == pending.generation
                and self._authority_high_water_state_sha256 == pending.state_sha256
            )
            remote_matches_local = (
                self._authority_high_water_generation == expected_generation
                and self._authority_high_water_state_sha256 == expected_state_sha256
            )
            if remote_matches_local:
                self._advance_authority_high_water(
                    expected_generation=expected_generation,
                    next_generation=pending.generation,
                    next_state_sha256=pending.state_sha256,
                )
            elif not remote_matches_pending:
                raise MicroLiveExecutionError(
                    "pending journal does not reconcile with external authority"
                )
            self._promote_pending_locked()
            local = self._read_locked()
        if local is None:
            if not (
                self._authority_high_water_generation == -1
                and self._authority_high_water_state_sha256 is None
            ):
                raise MicroLiveExecutionError(
                    "external authority high-water has no recoverable local state"
                )
            return
        self._require_authority_high_water_matches(local)

    def _journal_envelope(self, *, generation: int, raw_state: bytes) -> bytes:
        _require_bounded_state_bytes(raw_state)
        header = json.dumps(
            {
                "generation": generation,
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "state_sha256": hashlib.sha256(raw_state).hexdigest(),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(header) > _JOURNAL_HEADER_MAX_BYTES or b"\n" in header:
            raise MicroLiveExecutionError("micro-live journal header is invalid")
        return header + b"\n" + raw_state

    def _write_pending_locked(self, *, generation: int, raw_state: bytes) -> None:
        self._atomic_write(
            path=self.pending_state_path,
            raw_payload=self._journal_envelope(
                generation=generation,
                raw_state=raw_state,
            ),
        )

    def _promote_pending_locked(self) -> None:
        pending = self._read_snapshot_path(
            self.pending_state_path,
            "pending journal state",
        )
        self._require_authority_high_water_matches(pending)
        os.replace(self.pending_state_path, self.state_path)
        directory_descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    def _write_locked(self, *, generation: int, raw_state: bytes) -> None:
        self._atomic_write(
            path=self.state_path,
            raw_payload=self._journal_envelope(
                generation=generation,
                raw_state=raw_state,
            ),
        )

    def _atomic_write(self, *, path: Path, raw_payload: bytes) -> None:
        temporary_path = path.parent / (
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(raw_payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary_path, path)
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_path.exists():
                temporary_path.unlink()

    def _emergency_kill_bytes(
        self,
        *,
        payload: Mapping[str, Any],
        payload_sha256: str,
    ) -> bytes:
        return json.dumps(
            {**dict(payload), "payload_sha256": payload_sha256},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def _read_emergency_kill_locked(self) -> EmergencyKillSnapshot:
        return self._read_emergency_kill_path_locked(
            self.emergency_kill_path,
            "emergency kill record",
        )

    def _read_emergency_kill_path_locked(
        self,
        path: Path,
        label: str,
    ) -> EmergencyKillSnapshot:
        raw_payload = _read_bounded_stable_regular_file(
            path,
            maximum_bytes=_EMERGENCY_KILL_MAX_BYTES,
            label=label,
        )
        value, _, _ = _raw_json_object(
            raw_payload,
            label,
        )
        if set(value) != {
            "schema_version",
            "authorization_id",
            "risk_domain_id",
            "reason",
            "event_ts_ms",
            "payload_sha256",
        }:
            raise MicroLiveExecutionError("emergency kill record schema is invalid")
        payload = {
            key: value[key]
            for key in (
                "schema_version",
                "authorization_id",
                "risk_domain_id",
                "reason",
                "event_ts_ms",
            )
        }
        if not (
            value["schema_version"] == EMERGENCY_KILL_SCHEMA_VERSION
            and value["authorization_id"] == self._authorization_id
            and value["risk_domain_id"] == self._risk_domain_id
            and isinstance(value["reason"], str)
            and bool(value["reason"])
            and isinstance(value["event_ts_ms"], int)
            and not isinstance(value["event_ts_ms"], bool)
            and value["event_ts_ms"] > 0
            and _is_sha256(value["payload_sha256"])
            and value["payload_sha256"] == canonical_json_sha256(payload)
        ):
            raise MicroLiveExecutionError("emergency kill record identity is invalid")
        return EmergencyKillSnapshot(
            authorization_id=str(value["authorization_id"]),
            risk_domain_id=str(value["risk_domain_id"]),
            reason=str(value["reason"]),
            event_ts_ms=int(value["event_ts_ms"]),
            payload_sha256=str(value["payload_sha256"]),
        )

    def _read_locked(self) -> DurableJournalSnapshot:
        return self._read_snapshot_path(self.state_path, "micro-live journal state")

    def _read_snapshot_path(
        self,
        path: Path,
        label: str,
    ) -> DurableJournalSnapshot:
        raw = _read_bounded_stable_regular_file(
            path,
            maximum_bytes=(
                MAX_RESTORED_STATE_BYTES + _JOURNAL_HEADER_MAX_BYTES + 1
            ),
            label=label,
        )
        header_raw, separator, raw_state = raw.partition(b"\n")
        if not separator or not header_raw or len(header_raw) > _JOURNAL_HEADER_MAX_BYTES:
            raise MicroLiveExecutionError("micro-live journal envelope is invalid")
        try:
            header = json.loads(
                header_raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            )
            _validate_finite_json_tree(header)
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise MicroLiveExecutionError("micro-live journal header is not strict JSON") from exc
        if not (
            isinstance(header, dict)
            and set(header) == {"generation", "schema_version", "state_sha256"}
            and header.get("schema_version") == JOURNAL_SCHEMA_VERSION
            and isinstance(header.get("generation"), int)
            and not isinstance(header.get("generation"), bool)
            and int(header["generation"]) >= 0
            and _SHA256.fullmatch(str(header.get("state_sha256") or "")) is not None
        ):
            raise MicroLiveExecutionError("micro-live journal header schema is invalid")
        _require_bounded_state_bytes(raw_state)
        generation = int(header["generation"])
        state_sha256 = hashlib.sha256(raw_state).hexdigest()
        if not (
            state_sha256 == header["state_sha256"]
            and _state_generation(raw_state) == generation
        ):
            raise MicroLiveExecutionError("micro-live journal state hash is invalid")
        return DurableJournalSnapshot(
            generation=generation,
            state_sha256=state_sha256,
            raw_state=raw_state,
        )


@dataclass(frozen=True, slots=True)
class VerifiedProviderFeatureEvidence:
    """Exact raw streams and the feature row deterministically rebuilt from them."""

    file_sha256_items: tuple[tuple[str, str], ...]
    raw_jsonl_items: tuple[tuple[str, str], ...]
    reconstructed_feature_row_sha256: str
    evidence_graph_sha256: str

    @property
    def file_sha256(self) -> dict[str, str]:
        return dict(self.file_sha256_items)

    @property
    def raw_jsonl(self) -> dict[str, str]:
        return dict(self.raw_jsonl_items)


class MicroLiveOrderTransport(Protocol):
    """Minimal injected capability returning exact exchange response bytes.

    The adapter must not parse or normalize a response before handing it to
    the executor.  This keeps duplicate-key, non-finite-number, and exact-byte
    evidence checks inside the fail-closed trust boundary.
    """

    def attest_execution_binding(self, request: Mapping[str, Any]) -> bytes:
        """Return signed deployment identity bytes pinned by authorization."""

    def bind_execution_dispatch_authority(
        self,
        begin: Callable[..., bytes],
        recover: Callable[..., bytes],
        complete: Callable[[bytes, bytes], bytes],
        fence: Callable[..., bytes],
        *,
        authorization_id: str,
        risk_domain_id: str,
        risk_domain_authority_binding_sha256: str,
        authorization_expires_at_ts_ms: int,
    ) -> None:
        """Bind one-shot dispatch and lookup-only crash recovery authority."""

    def read_trusted_time(self, request: Mapping[str, Any]) -> bytes:
        """Return a signed completion timestamp from the pinned clock."""

    def submit_order(self, request: Mapping[str, Any]) -> bytes:
        """Atomically consume the outbox grant before any venue side effect.

        The signed DISPATCHABLE proof is not itself a bearer grant.  A gateway
        must verify it and atomically consume it through the bound authority,
        yielding one DISPATCHING receipt, before its first network, signer,
        wallet, or exchange side effect.  Duplicate workers must return the
        stored outcome without another venue call.  The dispatch deadline
        limits initial consumption only: once DISPATCHING is issued, timeout
        alone cannot prove the holder is fenced.
        """

    def recover_order_submission(self, request: Mapping[str, Any]) -> bytes:
        """Lookup only by venue idempotency key and complete durable dispatch.

        This operation must never submit, sign, or otherwise create a venue
        side effect.  It receives the exact authority-recovered original
        request, validates the venue lookup result against that request, and
        invokes the bound recovery callback with the exact outcome bytes.
        """

    def cancel_order(self, request: Mapping[str, Any]) -> bytes:
        """Cancel one acknowledged order and return raw JSON bytes."""

    def lookup_order(self, request: Mapping[str, Any]) -> bytes:
        """Read one order by exact identity and return raw JSON bytes."""

    def read_order_fill_cursor(self, request: Mapping[str, Any]) -> bytes:
        """Return authoritative cumulative fills and terminal exchange state."""

    def fence_order_invocation(self, request: Mapping[str, Any]) -> bytes:
        """Prove a timed-out submit can no longer create a later side effect.

        Returning ``side_effects_fenced=true`` is an adapter-level durable
        guarantee, not a point-in-time lookup.  The executor obtains it before
        the only authoritative post-timeout lookup.
        """


def verify_dispatchable_outbox_request(
    request: Mapping[str, Any],
    *,
    authorization_id: str,
    risk_domain_id: str,
    risk_domain_authority_binding_sha256: str,
    lease_id: str,
    service_identity_sha256: str,
    tenant_id: str,
    key_identity_sha256: str,
    public_key_modulus_hex: str,
    public_key_exponent: int,
) -> dict[str, str]:
    """Verify the exact pre-venue durable outbox proof, or fail closed.

    This function is the reviewed execution-gateway boundary.  Concrete venue
    adapters must call it immediately before their first external side effect.
    It strips only the proof envelope, reconstructs the authority-accepted
    command byte for byte, and rejects identity drift or post-acceptance edits.
    """

    outbound_request = copy.deepcopy(dict(request))
    authentication = dict(outbound_request.get("execution_authentication") or {})
    raw_fence_json = authentication.get(
        "execution_invocation_fence_receipt_json"
    )
    raw_outbox_command_json = authentication.get("execution_outbox_command_json")
    raw_acceptance_json = authentication.get(
        "raw_execution_outbox_acceptance_receipt_json"
    )
    if not all(
        isinstance(value, str) and value
        for value in (
            raw_fence_json,
            raw_outbox_command_json,
            raw_acceptance_json,
        )
    ):
        raise MicroLiveExecutionError(
            "venue dispatch lacks a complete durable outbox proof"
        )
    if not (
        authentication.get("authorization_id") == authorization_id
        and authentication.get("risk_domain_id") == risk_domain_id
        and authentication.get("risk_domain_authority_binding_sha256")
        == risk_domain_authority_binding_sha256
        and outbound_request.get("transport_invocation_id") is not None
        and authentication.get("venue_idempotency_key_field")
        == VENUE_IDEMPOTENCY_KEY_FIELD
        and authentication.get("venue_idempotency_key")
        == outbound_request.get(VENUE_IDEMPOTENCY_KEY_FIELD)
        and authentication.get("venue_idempotency_scope")
        == VENUE_IDEMPOTENCY_SCOPE
        and authentication.get("venue_idempotency_semantics")
        == VENUE_IDEMPOTENCY_SEMANTICS
        and isinstance(authentication.get("dispatch_deadline_ts_ms"), int)
        and not isinstance(authentication.get("dispatch_deadline_ts_ms"), bool)
        and isinstance(authentication.get("authorization_expires_at_ts_ms"), int)
        and not isinstance(
            authentication.get("authorization_expires_at_ts_ms"), bool
        )
        and authentication["dispatch_deadline_ts_ms"]
        <= authentication["authorization_expires_at_ts_ms"]
        and authentication["dispatch_deadline_ts_ms"]
        == min(
            int(outbound_request.get("submitted_at_ts_ms", 0))
            + MAX_EXECUTION_DISPATCH_DURATION_MS,
            authentication["authorization_expires_at_ts_ms"],
        )
    ):
        raise MicroLiveExecutionError("venue dispatch outbox identity is invalid")

    fence, canonical_fence_json, fence_sha256 = _raw_json_object(
        raw_fence_json.encode("utf-8"),
        "venue dispatch invocation fence",
        maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
    )
    expected_fence = {
        "schema_version": EXECUTION_INVOCATION_FENCE_SCHEMA_VERSION,
        "lease_id": lease_id,
        "service_identity_sha256": service_identity_sha256,
        "tenant_id": tenant_id,
        "key_identity_sha256": key_identity_sha256,
        "authorization_id": authorization_id,
        "risk_domain_id": risk_domain_id,
        "journal_namespace_id": fence.get("journal_namespace_id"),
        "journal_epoch": fence.get("journal_epoch"),
        "transport_invocation_id": outbound_request["transport_invocation_id"],
        "operation": "submit_order",
        "status": "ACTIVE",
    }
    if not (
        _is_sha256(expected_fence["journal_namespace_id"])
        and _is_sha256(expected_fence["journal_epoch"])
        and canonical_fence_json == raw_fence_json
        and fence_sha256
        == authentication.get("execution_invocation_fence_receipt_sha256")
        and _verify_signed_risk_domain_receipt(
            fence,
            expected_core=expected_fence,
            public_key_modulus_hex=public_key_modulus_hex,
            public_key_exponent=public_key_exponent,
        )
    ):
        raise MicroLiveExecutionError("venue dispatch invocation fence is invalid")

    outbox_command, canonical_outbox_json, outbox_command_sha256 = (
        _raw_json_object(
            raw_outbox_command_json.encode("utf-8"),
            "venue dispatch outbox command",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
    )
    command_authentication = dict(outbound_request["execution_authentication"])
    for key in (
        "execution_outbox_command_json",
        "execution_outbox_command_sha256",
        "raw_execution_outbox_acceptance_receipt_json",
        "execution_outbox_acceptance_receipt_sha256",
    ):
        command_authentication.pop(key, None)
    outbound_request["execution_authentication"] = command_authentication
    raw_command = _canonical_json_bytes(outbound_request)
    expected_outbox_command = {
        "schema_version": EXECUTION_OUTBOX_COMMAND_SCHEMA_VERSION,
        "transport_invocation_id": request["transport_invocation_id"],
        "operation": "submit_order",
        "command_sha256": hashlib.sha256(raw_command).hexdigest(),
        "raw_command_json": raw_command.decode("utf-8"),
    }
    if not (
        outbox_command == expected_outbox_command
        and canonical_outbox_json == raw_outbox_command_json
        and outbox_command_sha256
        == authentication.get("execution_outbox_command_sha256")
    ):
        raise MicroLiveExecutionError("venue dispatch outbox command is invalid")

    acceptance, canonical_acceptance_json, acceptance_sha256 = _raw_json_object(
        raw_acceptance_json.encode("utf-8"),
        "venue dispatch outbox acceptance",
        maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
    )
    expected_acceptance = {
        **expected_fence,
        "schema_version": EXECUTION_INVOCATION_ACCEPTANCE_SCHEMA_VERSION,
        "status": "DISPATCHABLE",
        "fence_receipt_sha256": fence_sha256,
        "outbox_command_sha256": outbox_command_sha256,
    }
    if not (
        canonical_acceptance_json == raw_acceptance_json
        and acceptance_sha256
        == authentication.get("execution_outbox_acceptance_receipt_sha256")
        and _verify_signed_risk_domain_receipt(
            acceptance,
            expected_core=expected_acceptance,
            public_key_modulus_hex=public_key_modulus_hex,
            public_key_exponent=public_key_exponent,
        )
    ):
        raise MicroLiveExecutionError("venue dispatch outbox acceptance is invalid")
    return {
        "fence_receipt_sha256": fence_sha256,
        "outbox_command_sha256": outbox_command_sha256,
        "outbox_acceptance_receipt_json": canonical_acceptance_json,
        "outbox_acceptance_receipt_sha256": acceptance_sha256,
        "status": "DISPATCHABLE",
    }


def verify_recovered_submission_outcome(
    request: Mapping[str, Any],
    raw_outcome: bytes,
) -> bytes:
    """Validate a lookup-only venue result against the exact submit command.

    Concrete gateways must invoke this before the authority recovery callback.
    The helper performs no network, signing, wallet, or exchange operation.
    """

    outcome, outcome_json, _ = _raw_json_object(
        raw_outcome,
        "lookup-only recovered submission outcome",
        maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
    )
    expected_keys = {
        "client_order_id",
        "exchange_order_id",
        "status",
        "market_id",
        "token_id",
        "accepted_quantity",
        "limit_price",
    }
    if not (
        set(outcome) == expected_keys
        and outcome.get("client_order_id") == request.get("client_order_id")
        and outcome.get("market_id") == request.get("market_id")
        and outcome.get("token_id") == request.get("token_id")
        and outcome.get("accepted_quantity") == request.get("quantity")
        and outcome.get("limit_price") == request.get("limit_price")
        and outcome.get("status") in {"ACCEPTED", "REJECTED"}
        and isinstance(outcome.get("exchange_order_id"), str)
        and bool(outcome.get("exchange_order_id"))
    ):
        raise MicroLiveExecutionError(
            "lookup-only recovered submission outcome is mismatched"
        )
    return outcome_json.encode("utf-8")


def _risk_domain_authority_binding_sha256(
    authorization: VerifiedMicroLiveAuthorization,
) -> str:
    return canonical_json_sha256(
        {
            "lease_id": authorization.risk_domain_lease_id,
            "service_identity_sha256": (
                authorization.risk_domain_lease_service_identity_sha256
            ),
            "tenant_id": authorization.risk_domain_lease_tenant_id,
            "key_identity_sha256": (
                authorization.risk_domain_lease_key_identity_sha256
            ),
            "signature_algorithm": RISK_DOMAIN_RECEIPT_SIGNATURE_ALGORITHM,
            "public_key_modulus_hex": (
                authorization.risk_domain_lease_public_key_modulus_hex
            ),
            "public_key_exponent": (
                authorization.risk_domain_lease_public_key_exponent
            ),
        }
    )


def _execution_service_binding_sha256(
    authorization: VerifiedMicroLiveAuthorization,
) -> str:
    return canonical_json_sha256(
        {
            "service_identity_sha256": authorization.execution_service_identity_sha256,
            "adapter_implementation_sha256": (
                authorization.execution_adapter_implementation_sha256
            ),
            "configuration_sha256": authorization.execution_configuration_sha256,
            "exchange_endpoint_sha256": (
                authorization.execution_exchange_endpoint_sha256
            ),
            "exchange_account_sha256": (
                authorization.execution_exchange_account_sha256
            ),
            "signer_identity_sha256": (
                authorization.execution_signer_identity_sha256
            ),
            "cursor_key_identity_sha256": (
                authorization.execution_cursor_key_identity_sha256
            ),
            "clock_identity_sha256": authorization.execution_clock_identity_sha256,
            "settlement_authority_identity_sha256": (
                authorization.execution_settlement_authority_identity_sha256
            ),
            "signature_algorithm": RISK_DOMAIN_RECEIPT_SIGNATURE_ALGORITHM,
            "public_key_modulus_hex": authorization.execution_public_key_modulus_hex,
            "public_key_exponent": authorization.execution_public_key_exponent,
            "maximum_clock_skew_ms": (
                authorization.execution_maximum_clock_skew_ms
            ),
            "maximum_call_duration_ms": (
                authorization.execution_maximum_call_duration_ms
            ),
            "deployment_runtime_lock_sha256": (
                authorization.deployment_runtime_lock_sha256
            ),
            "deployment_requirements_lock_sha256": (
                authorization.deployment_requirements_lock_sha256
            ),
            "deployment_image_manifest_digest": (
                authorization.deployment_image_manifest_digest
            ),
        }
    )


def _expected_final_fill_watermark(cursor: Mapping[str, Any]) -> str:
    """Bind terminal finality to the full signed cursor content except signatures."""

    return canonical_json_sha256(
        {
            key: copy.deepcopy(value)
            for key, value in cursor.items()
            if key
            not in {
                "final_fill_watermark",
                "cursor_payload_sha256",
                "signature_algorithm",
                "signature_hex",
            }
        }
    )


def _trusted_time_receipt_core(
    authorization: Any,
    *,
    request_started_at_ts_ms: int,
    operation: str,
    response_sha256: str,
    request_completed_at_ts_ms: Any,
) -> dict[str, Any]:
    nonce_sha256 = canonical_json_sha256(
        {
            "authorization_id": authorization.authorization_id,
            "execution_service_binding_sha256": (
                authorization.execution_service_binding_sha256
            ),
            "operation": operation,
            "request_started_at_ts_ms": request_started_at_ts_ms,
            "response_sha256": response_sha256,
        }
    )
    return {
        "schema_version": TRUSTED_TIME_RECEIPT_SCHEMA_VERSION,
        "authorization_id": authorization.authorization_id,
        "execution_service_binding_sha256": (
            authorization.execution_service_binding_sha256
        ),
        "clock_identity_sha256": authorization.execution_clock_identity_sha256,
        "operation": operation,
        "request_started_at_ts_ms": request_started_at_ts_ms,
        "response_sha256": response_sha256,
        "nonce_sha256": nonce_sha256,
        "request_completed_at_ts_ms": request_completed_at_ts_ms,
    }


def _trusted_time_receipt_is_valid(
    receipt: Mapping[str, Any],
    authorization: Any,
    *,
    request_started_at_ts_ms: int,
    operation: str,
    response_sha256: str,
    request_completed_at_ts_ms: Any,
) -> bool:
    maximum_elapsed_ms = (
        2 * authorization.execution_maximum_call_duration_ms
        + authorization.execution_maximum_clock_skew_ms
    )
    expected_core = _trusted_time_receipt_core(
        authorization,
        request_started_at_ts_ms=request_started_at_ts_ms,
        operation=operation,
        response_sha256=response_sha256,
        request_completed_at_ts_ms=request_completed_at_ts_ms,
    )
    return bool(
        isinstance(request_completed_at_ts_ms, int)
        and not isinstance(request_completed_at_ts_ms, bool)
        and request_started_at_ts_ms
        <= request_completed_at_ts_ms
        <= request_started_at_ts_ms + maximum_elapsed_ms
        and _verify_signed_risk_domain_receipt(
            receipt,
            expected_core=expected_core,
            public_key_modulus_hex=(
                authorization.execution_public_key_modulus_hex
            ),
            public_key_exponent=authorization.execution_public_key_exponent,
        )
    )


def _verify_signed_risk_domain_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_core: Mapping[str, Any],
    public_key_modulus_hex: str,
    public_key_exponent: int,
) -> bool:
    """Verify an exact RSA PKCS#1 v1.5 SHA-256 authority receipt.

    This deliberately validates raw claim content against authorization-pinned
    key material.  No backend property, process-local object registry, cwd, or
    journal-path identity can substitute for the external signature.
    """

    signature_hex = receipt.get("signature_hex")
    if not (
        set(receipt) == {*expected_core, "signature_algorithm", "signature_hex"}
        and all(receipt.get(key) == value for key, value in expected_core.items())
        and receipt.get("signature_algorithm")
        == RISK_DOMAIN_RECEIPT_SIGNATURE_ALGORITHM
        and isinstance(public_key_modulus_hex, str)
        and re.fullmatch(r"[0-9a-f]{512}", public_key_modulus_hex) is not None
        and public_key_exponent == 65_537
        and isinstance(signature_hex, str)
        and re.fullmatch(r"[0-9a-f]{512}", signature_hex) is not None
    ):
        return False
    try:
        signed_bytes = json.dumps(
            dict(expected_core),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest_info = _RSA_SHA256_DIGEST_INFO_PREFIX + hashlib.sha256(
            signed_bytes
        ).digest()
        encoded_size = len(public_key_modulus_hex) // 2
        padding_size = encoded_size - len(digest_info) - 3
        if padding_size < 8:
            return False
        expected_encoded = (
            b"\x00\x01" + b"\xff" * padding_size + b"\x00" + digest_info
        )
        recovered = pow(
            int(signature_hex, 16),
            public_key_exponent,
            int(public_key_modulus_hex, 16),
        ).to_bytes(encoded_size, "big")
    except (OverflowError, TypeError, ValueError):
        return False
    return hmac.compare_digest(recovered, expected_encoded)


def _runtime_integrity_sha256(runtime: ResidualPromotionRuntime) -> str:
    residual_model_sha256 = hashlib.sha256(
        bytes(runtime.residual_booster.save_raw(raw_format="ubj"))
    ).hexdigest()
    logit_model_sha256 = hashlib.sha256(
        bytes(runtime.logit_booster.save_raw(raw_format="ubj"))
    ).hexdigest()
    if not (
        residual_model_sha256 == runtime.residual_model_sha256
        and logit_model_sha256 == runtime.logit_model_sha256
    ):
        raise MicroLiveExecutionError("bound runtime model bytes are mismatched")
    return canonical_json_sha256(
        {
            "candidate_id": runtime.candidate_id,
            "lineage_id": runtime.lineage_id,
            "manifest_sha256": runtime.manifest_sha256,
            "residual_model_sha256": runtime.residual_model_sha256,
            "logit_model_sha256": runtime.logit_model_sha256,
            "adapter_sha256": runtime.adapter_sha256,
            "maximum_decision_lag_ms": runtime.maximum_decision_lag_ms,
            "maximum_source_age_ms": runtime.maximum_source_age_ms,
            "coefficients": list(runtime.coefficients),
            "loaded_residual_model_sha256": residual_model_sha256,
            "loaded_logit_model_sha256": logit_model_sha256,
        }
    )


def _require_production_runtime_matrix() -> None:
    """Reject dependency drift before binding any external authority."""

    if not (
        platform.python_implementation() == PRODUCTION_PYTHON_IMPLEMENTATION
        and platform.python_version() == PRODUCTION_PYTHON_VERSION
        and np.__version__ == PRODUCTION_NUMPY_VERSION
        and scipy.__version__ == PRODUCTION_SCIPY_VERSION
        and xgb.__version__ == PRODUCTION_XGBOOST_VERSION
    ):
        raise MicroLiveExecutionError(
            "micro-live production runtime matrix is mismatched"
        )


@dataclass(frozen=True, slots=True)
class _BoundAuthorization:
    """Immutable operational snapshot of one verified capability."""

    authorization_id: str
    authorization_payload_sha256: str
    candidate_bundle_sha256: str
    risk_domain_lease_id: str
    risk_domain_lease_service_identity_sha256: str
    risk_domain_lease_tenant_id: str
    risk_domain_lease_key_identity_sha256: str
    risk_domain_lease_public_key_modulus_hex: str
    risk_domain_lease_public_key_exponent: int
    risk_domain_authority_binding_sha256: str
    execution_service_identity_sha256: str
    execution_adapter_implementation_sha256: str
    execution_configuration_sha256: str
    execution_exchange_endpoint_sha256: str
    execution_exchange_account_sha256: str
    execution_signer_identity_sha256: str
    execution_cursor_key_identity_sha256: str
    execution_clock_identity_sha256: str
    execution_settlement_authority_identity_sha256: str
    execution_public_key_modulus_hex: str
    execution_public_key_exponent: int
    execution_maximum_clock_skew_ms: int
    execution_maximum_call_duration_ms: int
    deployment_runtime_lock_sha256: str
    deployment_requirements_lock_sha256: str
    deployment_image_manifest_digest: str
    execution_service_binding_sha256: str
    capital_base_usd: Decimal
    maximum_notional_usd: Decimal
    maximum_realized_loss_usd: Decimal
    maximum_open_orders: int
    authorized_at_ts_ms: int
    expires_at_ts_ms: int
    maximum_signal_age_ms: int
    maximum_operator_heartbeat_age_ms: int
    market_allowlist: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    runtime: ResidualPromotionRuntime
    runtime_integrity_sha256: str

    @classmethod
    def from_verified(
        cls,
        authorization: VerifiedMicroLiveAuthorization,
    ) -> _BoundAuthorization:
        runtime = copy.deepcopy(authorization.runtime)
        return cls(
            authorization_id=authorization.authorization_id,
            authorization_payload_sha256=authorization.authorization_payload_sha256,
            candidate_bundle_sha256=authorization.candidate_bundle_sha256,
            risk_domain_lease_id=authorization.risk_domain_lease_id,
            risk_domain_lease_service_identity_sha256=(
                authorization.risk_domain_lease_service_identity_sha256
            ),
            risk_domain_lease_tenant_id=authorization.risk_domain_lease_tenant_id,
            risk_domain_lease_key_identity_sha256=(
                authorization.risk_domain_lease_key_identity_sha256
            ),
            risk_domain_lease_public_key_modulus_hex=(
                authorization.risk_domain_lease_public_key_modulus_hex
            ),
            risk_domain_lease_public_key_exponent=(
                authorization.risk_domain_lease_public_key_exponent
            ),
            risk_domain_authority_binding_sha256=(
                _risk_domain_authority_binding_sha256(authorization)
            ),
            execution_service_identity_sha256=(
                authorization.execution_service_identity_sha256
            ),
            execution_adapter_implementation_sha256=(
                authorization.execution_adapter_implementation_sha256
            ),
            execution_configuration_sha256=(
                authorization.execution_configuration_sha256
            ),
            execution_exchange_endpoint_sha256=(
                authorization.execution_exchange_endpoint_sha256
            ),
            execution_exchange_account_sha256=(
                authorization.execution_exchange_account_sha256
            ),
            execution_signer_identity_sha256=(
                authorization.execution_signer_identity_sha256
            ),
            execution_cursor_key_identity_sha256=(
                authorization.execution_cursor_key_identity_sha256
            ),
            execution_clock_identity_sha256=(
                authorization.execution_clock_identity_sha256
            ),
            execution_settlement_authority_identity_sha256=(
                authorization.execution_settlement_authority_identity_sha256
            ),
            execution_public_key_modulus_hex=(
                authorization.execution_public_key_modulus_hex
            ),
            execution_public_key_exponent=authorization.execution_public_key_exponent,
            execution_maximum_clock_skew_ms=(
                authorization.execution_maximum_clock_skew_ms
            ),
            execution_maximum_call_duration_ms=(
                authorization.execution_maximum_call_duration_ms
            ),
            deployment_runtime_lock_sha256=(
                authorization.deployment_runtime_lock_sha256
            ),
            deployment_requirements_lock_sha256=(
                authorization.deployment_requirements_lock_sha256
            ),
            deployment_image_manifest_digest=(
                authorization.deployment_image_manifest_digest
            ),
            execution_service_binding_sha256=(
                _execution_service_binding_sha256(authorization)
            ),
            capital_base_usd=authorization.capital_base_usd,
            maximum_notional_usd=authorization.maximum_notional_usd,
            maximum_realized_loss_usd=authorization.maximum_realized_loss_usd,
            maximum_open_orders=authorization.maximum_open_orders,
            authorized_at_ts_ms=authorization.authorized_at_ts_ms,
            expires_at_ts_ms=authorization.expires_at_ts_ms,
            maximum_signal_age_ms=authorization.maximum_signal_age_ms,
            maximum_operator_heartbeat_age_ms=(
                authorization.maximum_operator_heartbeat_age_ms
            ),
            market_allowlist=authorization.market_allowlist,
            allowed_actions=authorization.allowed_actions,
            runtime=runtime,
            runtime_integrity_sha256=_runtime_integrity_sha256(runtime),
        )

    def matches_verified(self, authorization: VerifiedMicroLiveAuthorization) -> bool:
        return (
            self.authorization_id == authorization.authorization_id
            and self.authorization_payload_sha256
            == authorization.authorization_payload_sha256
            and self.candidate_bundle_sha256 == authorization.candidate_bundle_sha256
            and self.risk_domain_lease_id == authorization.risk_domain_lease_id
            and self.risk_domain_authority_binding_sha256
            == _risk_domain_authority_binding_sha256(authorization)
            and self.execution_service_binding_sha256
            == _execution_service_binding_sha256(authorization)
            and self.deployment_runtime_lock_sha256
            == authorization.deployment_runtime_lock_sha256
            and self.deployment_requirements_lock_sha256
            == authorization.deployment_requirements_lock_sha256
            and self.deployment_image_manifest_digest
            == authorization.deployment_image_manifest_digest
            and self.capital_base_usd == authorization.capital_base_usd
            and self.maximum_notional_usd == authorization.maximum_notional_usd
            and self.maximum_realized_loss_usd
            == authorization.maximum_realized_loss_usd
            and self.maximum_open_orders == authorization.maximum_open_orders
            and self.authorized_at_ts_ms == authorization.authorized_at_ts_ms
            and self.expires_at_ts_ms == authorization.expires_at_ts_ms
            and self.maximum_signal_age_ms == authorization.maximum_signal_age_ms
            and self.maximum_operator_heartbeat_age_ms
            == authorization.maximum_operator_heartbeat_age_ms
            and self.market_allowlist == authorization.market_allowlist
            and self.allowed_actions == authorization.allowed_actions
            and self.runtime_integrity_sha256
            == _runtime_integrity_sha256(authorization.runtime)
            == _runtime_integrity_sha256(self.runtime)
        )


def create_micro_live_executor(
    *,
    raw_authorization: bytes,
    repository_root: Path | str,
    evidence_root: Path | str,
    now_ts_ms: int,
    transport: MicroLiveOrderTransport,
    journal_root: Path | str,
    risk_domain_lease: DurableRiskDomainLeaseBackend,
) -> MicroLiveExecutor:
    """Create an executor with the deployment-owned concrete WAL implementation."""

    verified = verify_micro_live_authorization(
        raw_authorization,
        repository_root=repository_root,
        evidence_root=evidence_root,
        now_ts_ms=now_ts_ms,
    )
    return MicroLiveExecutor._from_verified_authorization(
        verified,
        transport=transport,
        journal=AtomicFileMicroLiveStateJournal(
            journal_root,
            risk_domain_lease=risk_domain_lease,
        ),
    )


def _durable_entry(method: Any) -> Any:
    """Serialize one complete risk-bearing entry across threads and processes."""

    @wraps(method)
    def wrapped(self: MicroLiveExecutor, *args: Any, **kwargs: Any) -> Any:
        with self._durable_transaction():
            return method(self, *args, **kwargs)

    return wrapped


def _require_startup_execution_capabilities(
    *,
    transport: MicroLiveOrderTransport,
    journal: MicroLiveStateJournal,
) -> None:
    """Reject an incomplete execution/unwind surface before journal activation."""

    missing_operations = [
        operation
        for operation in REQUIRED_EXECUTION_TRANSPORT_OPERATIONS
        if not callable(getattr(transport, operation, None))
    ]
    if missing_operations:
        raise MicroLiveExecutionError(
            "authenticated execution gateway lacks required startup operations: "
            + ",".join(missing_operations)
        )
    authority = getattr(journal, "risk_domain_lease", None)
    if not callable(
        getattr(authority, "recover_execution_outbox_command", None)
    ):
        raise MicroLiveExecutionError(
            "external risk-domain authority lacks startup outbox recovery"
        )


class MicroLiveExecutor:
    """Append-only execution state machine behind a verified authorization."""

    def __init__(
        self,
        authorization: VerifiedMicroLiveAuthorization,
        *,
        transport: MicroLiveOrderTransport,
        journal: MicroLiveStateJournal,
    ) -> None:
        self._initialize(
            authorization=authorization,
            transport=transport,
            journal=journal,
            events=(),
            generation=0,
            initialize_journal=True,
        )

    def _initialize(
        self,
        *,
        authorization: VerifiedMicroLiveAuthorization,
        transport: MicroLiveOrderTransport,
        journal: MicroLiveStateJournal,
        events: Sequence[Mapping[str, Any]],
        generation: int,
        initialize_journal: bool,
        restore_token: object | None = None,
    ) -> None:
        """Initialize only fresh state or state decoded by ``restore``."""

        if events and restore_token is not _RAW_STATE_RESTORE_TOKEN:
            raise MicroLiveExecutionError(
                "nonempty micro-live events require strict raw-state restore"
            )
        if not authorization_capability_is_verified(authorization):
            raise MicroLiveExecutionError("micro-live authorization capability is unverified")
        if transport is None:
            raise MicroLiveExecutionError("micro-live transport capability is missing")
        if journal.__class__ is not AtomicFileMicroLiveStateJournal:
            raise MicroLiveExecutionError(
                "micro-live journal must be the deployment-owned concrete implementation"
            )
        _require_startup_execution_capabilities(
            transport=transport,
            journal=journal,
        )
        _require_production_runtime_matrix()
        if not (
            authorization.runtime.lineage_id == LINEAGE_ID
            and authorization.runtime.candidate_id == CANDIDATE_ID
            and authorization.runtime.manifest_sha256
            == authorization.candidate_bundle_sha256
        ):
            raise MicroLiveExecutionError("micro-live runtime capability is mismatched")
        if _new_order_lifecycle_capacity() > EVENT_RECOVERY_RESERVE:
            raise MicroLiveExecutionError(
                "micro-live recovery event reserve cannot cover one accepted order"
            )
        self.authorization = authorization
        self._authorization = _BoundAuthorization.from_verified(authorization)
        maximum_call_duration_ms = (
            self._authorization.execution_maximum_call_duration_ms
        )
        if not (1 <= maximum_call_duration_ms <= MAX_TRANSPORT_CALL_DURATION_MS):
            raise MicroLiveExecutionError(
                "authorized execution service deadline is unsafe"
            )
        self._transport = transport
        self._transport_object_id = id(transport)
        self._maximum_transport_call_duration_ms = maximum_call_duration_ms
        self._risk_domain_id = canonical_json_sha256(
            {
                "authorization_id": self._authorization.authorization_id,
                "candidate_bundle_sha256": (
                    self._authorization.candidate_bundle_sha256
                ),
                "risk_domain_lease_id": self._authorization.risk_domain_lease_id,
                "risk_domain_authority_binding_sha256": (
                    self._authorization.risk_domain_authority_binding_sha256
                ),
                "execution_service_binding_sha256": (
                    self._authorization.execution_service_binding_sha256
                ),
                "lineage_id": LINEAGE_ID,
            }
        )
        self._journal = journal
        # The signed gateway recovery claim is verified before the journal is
        # bound or initialized.  A legacy gateway cannot activate execution
        # state and only then reveal that crash recovery is unavailable.
        self._verify_execution_binding_attestation()
        self._journal.bind_risk_domain(
            authorization_id=self._authorization.authorization_id,
            risk_domain_id=self._risk_domain_id,
            lease_id=self._authorization.risk_domain_lease_id,
            service_identity_sha256=(
                self._authorization.risk_domain_lease_service_identity_sha256
            ),
            tenant_id=self._authorization.risk_domain_lease_tenant_id,
            key_identity_sha256=(
                self._authorization.risk_domain_lease_key_identity_sha256
            ),
            public_key_modulus_hex=(
                self._authorization.risk_domain_lease_public_key_modulus_hex
            ),
            public_key_exponent=(
                self._authorization.risk_domain_lease_public_key_exponent
            ),
        )
        if (
            self._journal.authenticated_risk_domain_authority_binding_sha256
            != self._authorization.risk_domain_authority_binding_sha256
        ):
            raise MicroLiveExecutionError(
                "micro-live journal authority binding is authorization-mismatched"
            )
        self._bind_transport_dispatch_authority()
        self._generation = generation
        self._transaction_depth = 0
        self._active_journal_transaction: DurableJournalTransaction | None = None
        self._executor_thread_lock = threading.RLock()
        self._events = [copy.deepcopy(dict(event)) for event in events]
        _require_event_count(len(self._events))
        self._verify_event_chain()
        view = self._reconcile_view()
        if _realized_loss_limit_reached(view, self._authorization) and not view[
            "kill_switch_active"
        ]:
            raise MicroLiveExecutionError(
                "micro-live state crossed realized-loss limit without kill switch"
            )
        if (
            view["loss_budget_consumed_usd"]
            > self._authorization.maximum_realized_loss_usd
            and not view["kill_switch_active"]
        ):
            raise MicroLiveExecutionError(
                "micro-live state exceeded loss budget without kill switch"
            )
        if initialize_journal:
            self._journal.initialize(self.export_state_bytes())

    @classmethod
    def _from_verified_authorization(
        cls,
        authorization: VerifiedMicroLiveAuthorization,
        *,
        transport: MicroLiveOrderTransport,
        journal: MicroLiveStateJournal,
    ) -> MicroLiveExecutor:
        return cls(authorization, transport=transport, journal=journal)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        with self._executor_thread_lock:
            return tuple(copy.deepcopy(self._events))

    @property
    def transport(self) -> MicroLiveOrderTransport:
        """Return the construction-bound adapter without permitting replacement."""

        return self._transport

    @contextmanager
    def _durable_transaction(self) -> Iterator[None]:
        """Hold one reentrant executor lock plus journal process lease/CAS view."""

        with self._executor_thread_lock:
            if self._transaction_depth:
                self._transaction_depth += 1
                try:
                    yield
                finally:
                    self._transaction_depth -= 1
                return
            with self._journal.transaction(
                expected_generation=self._generation,
            ) as transaction:
                if transaction.generation != self._generation:
                    raise MicroLiveExecutionError(
                        "micro-live journal transaction opened at stale generation"
                    )
                self._transaction_depth = 1
                self._active_journal_transaction = transaction
                try:
                    self._synchronize_emergency_kill()
                    yield
                finally:
                    self._active_journal_transaction = None
                    self._transaction_depth = 0

    def _persist_emergency_kill(
        self,
        *,
        reason: str,
        event_ts_ms: int,
    ) -> EmergencyKillSnapshot:
        return self._journal.persist_emergency_kill(
            authorization_id=self._authorization.authorization_id,
            risk_domain_id=self._risk_domain_id,
            reason=reason,
            event_ts_ms=event_ts_ms,
        )

    def _synchronize_emergency_kill(self) -> None:
        snapshot = self._journal.emergency_kill_snapshot()
        if snapshot is None:
            return
        view = self._reconcile_view()
        if view["kill_switch_active"]:
            return
        event_ts_ms = max(
            snapshot.event_ts_ms,
            self._safe_persistence_timestamp(),
        )
        self._append_event(
            "KILL_SWITCH_ENGAGED",
            {
                "reason": snapshot.reason,
                "engaged_at_ts_ms": event_ts_ms,
            },
            event_ts_ms=event_ts_ms,
        )

    def _bounded_transport_call(
        self,
        *,
        operation: str,
        request: Mapping[str, Any],
    ) -> bytes:
        if id(self._transport) != self._transport_object_id:
            raise MicroLiveExecutionError(
                "authenticated execution transport changed after construction"
            )
        outbound_request = copy.deepcopy(dict(request))
        raw_invocation_fence: bytes | None = None
        invocation_fence_sha256: str | None = None
        if operation == "submit_order":
            transport_invocation_id = outbound_request.get(
                "transport_invocation_id"
            )
            raw_invocation_fence = self._journal.register_execution_invocation(
                transport_invocation_id=str(transport_invocation_id),
                operation=operation,
            )
            invocation_fence_sha256 = hashlib.sha256(
                raw_invocation_fence
            ).hexdigest()
        if operation in _SIGNED_EXECUTION_OPERATIONS:
            request_nonce_sha256 = hashlib.sha256(os.urandom(32)).hexdigest()
            request_core_sha256 = canonical_json_sha256(outbound_request)
            outbound_request["execution_authentication"] = {
                "authorization_id": self._authorization.authorization_id,
                "execution_service_binding_sha256": (
                    self._authorization.execution_service_binding_sha256
                ),
                "exchange_endpoint_sha256": (
                    self._authorization.execution_exchange_endpoint_sha256
                ),
                "exchange_account_sha256": (
                    self._authorization.execution_exchange_account_sha256
                ),
                "signer_identity_sha256": (
                    self._authorization.execution_signer_identity_sha256
                ),
                "operation": operation,
                "request_nonce_sha256": request_nonce_sha256,
                "request_core_sha256": request_core_sha256,
                "risk_domain_id": self._risk_domain_id,
                "risk_domain_authority_binding_sha256": (
                    self._authorization.risk_domain_authority_binding_sha256
                ),
                "execution_invocation_fence_receipt_json": (
                    raw_invocation_fence.decode("utf-8")
                    if raw_invocation_fence is not None
                    else None
                ),
                "execution_invocation_fence_receipt_sha256": (
                    invocation_fence_sha256
                ),
            }
        if operation == "submit_order":
            outbound_request["execution_authentication"].update(
                {
                    "dispatch_deadline_ts_ms": min(
                        int(outbound_request["submitted_at_ts_ms"])
                        + MAX_EXECUTION_DISPATCH_DURATION_MS,
                        self._authorization.expires_at_ts_ms,
                    ),
                    "authorization_expires_at_ts_ms": (
                        self._authorization.expires_at_ts_ms
                    ),
                    "venue_idempotency_key_field": VENUE_IDEMPOTENCY_KEY_FIELD,
                    "venue_idempotency_key": outbound_request[
                        VENUE_IDEMPOTENCY_KEY_FIELD
                    ],
                    "venue_idempotency_scope": VENUE_IDEMPOTENCY_SCOPE,
                    "venue_idempotency_semantics": VENUE_IDEMPOTENCY_SEMANTICS,
                }
            )
            raw_command = _canonical_json_bytes(outbound_request)
            raw_outbox_command = _canonical_json_bytes(
                {
                    "schema_version": EXECUTION_OUTBOX_COMMAND_SCHEMA_VERSION,
                    "transport_invocation_id": outbound_request.get(
                        "transport_invocation_id"
                    ),
                    "operation": operation,
                    "command_sha256": hashlib.sha256(raw_command).hexdigest(),
                    "raw_command_json": raw_command.decode("utf-8"),
                }
            )
            if raw_invocation_fence is None:
                raise MicroLiveExecutionError(
                    "submit_order outbox command lacks an invocation fence"
                )
            try:
                raw_outbox_acceptance = (
                    self._journal.commit_execution_outbox_command(
                        raw_invocation_fence,
                        raw_outbox_command,
                    )
                )
            except MicroLiveExecutionError:
                # The first response can be lost after the authority has made
                # the command durable.  Exact replay is the only safe recovery:
                # the authority returns the same DISPATCHABLE receipt or keeps
                # the command fenced, and rejects any byte drift.
                raw_outbox_acceptance = (
                    self._journal.commit_execution_outbox_command(
                        raw_invocation_fence,
                        raw_outbox_command,
                    )
                )
            outbox_acceptance = json.loads(raw_outbox_acceptance)
            if outbox_acceptance.get("status") != "DISPATCHABLE":
                raise MicroLiveExecutionError(
                    "submit_order outbox command was fenced before dispatch"
                )
            outbound_request["execution_authentication"].update(
                {
                    "execution_outbox_command_json": (
                        raw_outbox_command.decode("utf-8")
                    ),
                    "execution_outbox_command_sha256": hashlib.sha256(
                        raw_outbox_command
                    ).hexdigest(),
                    "raw_execution_outbox_acceptance_receipt_json": (
                        raw_outbox_acceptance.decode("utf-8")
                    ),
                    "execution_outbox_acceptance_receipt_sha256": (
                        hashlib.sha256(raw_outbox_acceptance).hexdigest()
                    ),
                }
            )
        method = getattr(self._transport, operation, None)
        if not callable(method):
            raise MicroLiveExecutionError(
                f"{operation} is absent from the authenticated execution service"
            )
        completed = threading.Event()
        result_queue: queue.SimpleQueue[tuple[bool, Any]] = queue.SimpleQueue()

        def invoke() -> None:
            try:
                result_queue.put((True, method(copy.deepcopy(outbound_request))))
            except BaseException as exc:  # propagate crash injection and transport faults
                result_queue.put((False, exc))
            finally:
                completed.set()

        worker = threading.Thread(
            target=invoke,
            name=f"micro-live-{operation}",
            daemon=True,
        )
        worker.start()
        if not completed.wait(self._maximum_transport_call_duration_ms / 1_000):
            raise MicroLiveExecutionError(
                f"{operation} exceeded the mandatory transport deadline"
            )
        succeeded, value = result_queue.get_nowait()
        if not succeeded:
            raise value
        if not isinstance(value, bytes):
            raise MicroLiveExecutionError(
                f"{operation} did not return exact raw bytes"
            )
        if operation not in _SIGNED_EXECUTION_OPERATIONS:
            return value
        return self._verify_execution_operation_receipt(
            operation=operation,
            outbound_request=outbound_request,
            raw_receipt=value,
        )

    def _bounded_submission_recovery_call(
        self,
        *,
        prepared: Mapping[str, Any],
    ) -> bytes:
        """Run the public gateway's lookup-only recovery on durable bytes."""

        if id(self._transport) != self._transport_object_id:
            raise MicroLiveExecutionError(
                "authenticated execution transport changed after construction"
            )
        raw_request = self._journal.recover_execution_outbox_request(
            transport_invocation_id=str(prepared["transport_invocation_id"]),
        )
        recovered_request, _, _ = _raw_json_object(
            raw_request,
            "recovered execution outbox request",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        expected_request = {
            key: prepared[key]
            for key in (
                "authorization_id",
                "authorization_payload_sha256",
                "candidate_bundle_sha256",
                "business_key",
                "market_id",
                "slug",
                "market_family",
                "decision_ts_ms",
                "selected_action",
                "token_id",
                "token_side",
                "limit_price",
                "quantity",
                "notional_usd",
                "maximum_fee_usd",
                "maximum_loss_usd",
                "signal_payload_sha256",
                "raw_signal_payload_sha256",
                "market_identity_sha256",
                "market_identity",
                "raw_feature_row_sha256",
                "provider_feature_evidence_graph_sha256",
                "provider_feature_file_sha256",
                "intent_id",
                "client_order_id",
                "transport_invocation_id",
                "submitted_at_ts_ms",
            )
        }
        request_without_authentication = copy.deepcopy(recovered_request)
        authentication = request_without_authentication.pop(
            "execution_authentication",
            None,
        )
        if not (
            request_without_authentication == expected_request
            and isinstance(authentication, Mapping)
            and authentication.get("operation") == "submit_order"
            and authentication.get("authorization_id")
            == self._authorization.authorization_id
            and authentication.get("risk_domain_id") == self._risk_domain_id
        ):
            raise MicroLiveExecutionError(
                "recovered execution outbox request is preparation-mismatched"
            )
        method = getattr(self._transport, "recover_order_submission", None)
        if not callable(method):
            raise MicroLiveExecutionError(
                "authenticated execution gateway lacks lookup-only recovery"
            )
        completed = threading.Event()
        result_queue: queue.SimpleQueue[tuple[bool, Any]] = queue.SimpleQueue()

        def invoke() -> None:
            try:
                result_queue.put(
                    (True, method(copy.deepcopy(recovered_request)))
                )
            except BaseException as exc:
                result_queue.put((False, exc))
            finally:
                completed.set()

        worker = threading.Thread(
            target=invoke,
            name="micro-live-recover-order-submission",
            daemon=True,
        )
        worker.start()
        if not completed.wait(self._maximum_transport_call_duration_ms / 1_000):
            raise MicroLiveExecutionError(
                "recover_order_submission exceeded the mandatory transport deadline"
            )
        succeeded, value = result_queue.get_nowait()
        if not succeeded:
            raise value
        if not isinstance(value, bytes):
            raise MicroLiveExecutionError(
                "recover_order_submission did not return exact raw bytes"
            )
        return self._verify_execution_operation_receipt(
            operation="submit_order",
            outbound_request=recovered_request,
            raw_receipt=value,
        )

    def _bind_transport_dispatch_authority(self) -> None:
        method = getattr(self._transport, "bind_execution_dispatch_authority", None)
        if not callable(method):
            raise MicroLiveExecutionError(
                "authenticated execution gateway lacks one-shot dispatch authority"
            )
        method(
            self._journal.begin_execution_dispatch,
            self._journal.recover_execution_dispatch,
            self._journal.complete_execution_dispatch,
            self._journal.fence_execution_dispatch,
            authorization_id=self._authorization.authorization_id,
            risk_domain_id=self._risk_domain_id,
            risk_domain_authority_binding_sha256=(
                self._authorization.risk_domain_authority_binding_sha256
            ),
            authorization_expires_at_ts_ms=self._authorization.expires_at_ts_ms,
        )

    def _verify_execution_operation_receipt(
        self,
        *,
        operation: str,
        outbound_request: Mapping[str, Any],
        raw_receipt: bytes,
    ) -> bytes:
        receipt, _, _ = _raw_json_object(
            raw_receipt,
            f"{operation} authenticated receipt",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        authentication = dict(outbound_request.get("execution_authentication") or {})
        raw_response_json = receipt.get("raw_response_json")
        if not isinstance(raw_response_json, str) or not raw_response_json:
            raise MicroLiveExecutionError(
                f"{operation} authenticated response payload is absent"
            )
        raw_response = raw_response_json.encode("utf-8")
        if len(raw_response) > MAX_EXECUTION_TRANSPORT_EVENT_BYTES:
            raise MicroLiveExecutionError(
                f"{operation} authenticated response payload is oversized"
            )
        outbox_acceptance_receipt_json: str | None = None
        outbox_acceptance_receipt_sha256: str | None = None
        outbox_command_sha256: str | None = None
        dispatch_terminal_receipt_json: str | None = None
        dispatch_terminal_receipt_sha256: str | None = None
        fence_status = "NOT_APPLICABLE"
        if operation == "submit_order":
            raw_fence_json = authentication.get(
                "execution_invocation_fence_receipt_json"
            )
            raw_outbox_command_json = authentication.get(
                "execution_outbox_command_json"
            )
            raw_acceptance_json = receipt.get(
                "raw_execution_outbox_acceptance_receipt_json"
            )
            if not (
                isinstance(raw_fence_json, str)
                and raw_fence_json
                and isinstance(raw_outbox_command_json, str)
                and raw_outbox_command_json
                and isinstance(raw_acceptance_json, str)
                and raw_acceptance_json
            ):
                raise MicroLiveExecutionError(
                    "submit_order durable outbox evidence is absent"
                )
            fence, canonical_fence_json, fence_sha256 = _raw_json_object(
                raw_fence_json.encode("utf-8"),
                "submit_order execution invocation fence",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            expected_fence_core = {
                "schema_version": EXECUTION_INVOCATION_FENCE_SCHEMA_VERSION,
                "lease_id": self._authorization.risk_domain_lease_id,
                "service_identity_sha256": (
                    self._authorization.risk_domain_lease_service_identity_sha256
                ),
                "tenant_id": self._authorization.risk_domain_lease_tenant_id,
                "key_identity_sha256": (
                    self._authorization.risk_domain_lease_key_identity_sha256
                ),
                "authorization_id": self._authorization.authorization_id,
                "risk_domain_id": self._risk_domain_id,
                "journal_namespace_id": fence.get("journal_namespace_id"),
                "journal_epoch": fence.get("journal_epoch"),
                "transport_invocation_id": (
                    outbound_request.get("transport_invocation_id")
                ),
                "operation": "submit_order",
                "status": "ACTIVE",
            }
            if not (
                _is_sha256(expected_fence_core["journal_namespace_id"])
                and _is_sha256(expected_fence_core["journal_epoch"])
                and canonical_fence_json == raw_fence_json
                and fence_sha256
                == authentication.get(
                    "execution_invocation_fence_receipt_sha256"
                )
                and _verify_signed_risk_domain_receipt(
                    fence,
                    expected_core=expected_fence_core,
                    public_key_modulus_hex=(
                        self._authorization.risk_domain_lease_public_key_modulus_hex
                    ),
                    public_key_exponent=(
                        self._authorization.risk_domain_lease_public_key_exponent
                    ),
                )
            ):
                raise MicroLiveExecutionError(
                    "submit_order execution invocation fence is invalid"
                )
            outbox_command, canonical_outbox_command_json, outbox_command_sha256 = (
                _raw_json_object(
                    raw_outbox_command_json.encode("utf-8"),
                    "submit_order durable outbox command",
                    maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
                )
            )
            command_without_outbox_proof = copy.deepcopy(dict(outbound_request))
            command_authentication = dict(
                command_without_outbox_proof["execution_authentication"]
            )
            for key in (
                "execution_outbox_command_json",
                "execution_outbox_command_sha256",
                "raw_execution_outbox_acceptance_receipt_json",
                "execution_outbox_acceptance_receipt_sha256",
            ):
                command_authentication.pop(key, None)
            command_without_outbox_proof["execution_authentication"] = (
                command_authentication
            )
            raw_command = _canonical_json_bytes(command_without_outbox_proof)
            expected_outbox_command = {
                "schema_version": EXECUTION_OUTBOX_COMMAND_SCHEMA_VERSION,
                "transport_invocation_id": outbound_request.get(
                    "transport_invocation_id"
                ),
                "operation": "submit_order",
                "command_sha256": hashlib.sha256(raw_command).hexdigest(),
                "raw_command_json": raw_command.decode("utf-8"),
            }
            if not (
                outbox_command == expected_outbox_command
                and canonical_outbox_command_json == raw_outbox_command_json
                and outbox_command_sha256
                == authentication.get("execution_outbox_command_sha256")
            ):
                raise MicroLiveExecutionError(
                    "submit_order durable outbox command is invalid"
                )
            (
                acceptance,
                outbox_acceptance_receipt_json,
                outbox_acceptance_receipt_sha256,
            ) = _raw_json_object(
                raw_acceptance_json.encode("utf-8"),
                "submit_order durable outbox acceptance receipt",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            expected_acceptance_core = {
                **expected_fence_core,
                "schema_version": EXECUTION_INVOCATION_ACCEPTANCE_SCHEMA_VERSION,
                "status": "DISPATCHABLE",
                "fence_receipt_sha256": fence_sha256,
                "outbox_command_sha256": outbox_command_sha256,
            }
            if not (
                outbox_acceptance_receipt_sha256
                == authentication.get(
                    "execution_outbox_acceptance_receipt_sha256"
                )
                and outbox_acceptance_receipt_json
                == authentication.get(
                    "raw_execution_outbox_acceptance_receipt_json"
                )
                and _verify_signed_risk_domain_receipt(
                    acceptance,
                    expected_core=expected_acceptance_core,
                    public_key_modulus_hex=(
                        self._authorization.risk_domain_lease_public_key_modulus_hex
                    ),
                    public_key_exponent=(
                        self._authorization.risk_domain_lease_public_key_exponent
                    ),
                )
            ):
                raise MicroLiveExecutionError(
                    "submit_order durable outbox acceptance receipt is invalid"
                )
            raw_dispatch_terminal_json = receipt.get(
                "raw_execution_dispatch_terminal_receipt_json"
            )
            if not (
                isinstance(raw_dispatch_terminal_json, str)
                and raw_dispatch_terminal_json
            ):
                raise MicroLiveExecutionError(
                    "submit_order dispatch terminal evidence is absent"
                )
            (
                dispatch_terminal,
                dispatch_terminal_receipt_json,
                dispatch_terminal_receipt_sha256,
            ) = _raw_json_object(
                raw_dispatch_terminal_json.encode("utf-8"),
                "submit_order dispatch terminal receipt",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            dispatch_schema_version = dispatch_terminal.get("schema_version")
            original_dispatch_receipt_sha256 = dispatch_terminal.get(
                "dispatch_receipt_sha256"
            )
            if not (
                dispatch_terminal_receipt_sha256
                == receipt.get("execution_dispatch_terminal_receipt_sha256")
                and dispatch_schema_version
                in {
                    EXECUTION_DISPATCH_RECEIPT_SCHEMA_VERSION,
                    EXECUTION_DISPATCH_COMPLETION_SCHEMA_VERSION,
                }
                and (
                    (
                        dispatch_schema_version
                        == EXECUTION_DISPATCH_RECEIPT_SCHEMA_VERSION
                        and original_dispatch_receipt_sha256 is None
                    )
                    or (
                        dispatch_schema_version
                        == EXECUTION_DISPATCH_COMPLETION_SCHEMA_VERSION
                        and _is_sha256(original_dispatch_receipt_sha256)
                    )
                )
                and _verify_signed_risk_domain_receipt(
                    dispatch_terminal,
                    expected_core={
                        **{
                            key: expected_fence_core[key]
                            for key in (
                                "lease_id",
                                "service_identity_sha256",
                                "tenant_id",
                                "key_identity_sha256",
                                "authorization_id",
                                "risk_domain_id",
                                "journal_namespace_id",
                                "journal_epoch",
                            )
                        },
                        "schema_version": dispatch_schema_version,
                        "transport_invocation_id": outbound_request.get(
                            "transport_invocation_id"
                        ),
                        "operation": "submit_order",
                        "status": "DISPATCHED",
                        "outbox_command_sha256": outbox_command_sha256,
                        "outbox_acceptance_receipt_sha256": (
                            outbox_acceptance_receipt_sha256
                        ),
                        "venue_idempotency_key": outbound_request.get(
                            VENUE_IDEMPOTENCY_KEY_FIELD
                        ),
                        "venue_idempotency_scope": VENUE_IDEMPOTENCY_SCOPE,
                        "dispatch_deadline_ts_ms": authentication.get(
                            "dispatch_deadline_ts_ms"
                        ),
                        "authorization_expires_at_ts_ms": authentication.get(
                            "authorization_expires_at_ts_ms"
                        ),
                        "dispatch_receipt_sha256": (
                            original_dispatch_receipt_sha256
                        ),
                        "raw_outcome_json": raw_response_json,
                        "outcome_sha256": hashlib.sha256(raw_response).hexdigest(),
                    },
                    public_key_modulus_hex=(
                        self._authorization.risk_domain_lease_public_key_modulus_hex
                    ),
                    public_key_exponent=(
                        self._authorization.risk_domain_lease_public_key_exponent
                    ),
                )
            ):
                raise MicroLiveExecutionError(
                    "submit_order dispatch terminal receipt is invalid"
                )
            fence_status = "DISPATCHED"
        expected_core = {
            "schema_version": EXECUTION_OPERATION_RECEIPT_SCHEMA_VERSION,
            "authorization_id": self._authorization.authorization_id,
            "execution_service_binding_sha256": (
                self._authorization.execution_service_binding_sha256
            ),
            "exchange_endpoint_sha256": (
                self._authorization.execution_exchange_endpoint_sha256
            ),
            "exchange_account_sha256": (
                self._authorization.execution_exchange_account_sha256
            ),
            "signer_identity_sha256": (
                self._authorization.execution_signer_identity_sha256
            ),
            "operation": operation,
            "request_nonce_sha256": authentication.get("request_nonce_sha256"),
            "request_sha256": canonical_json_sha256(dict(outbound_request)),
            "response_sha256": hashlib.sha256(raw_response).hexdigest(),
            "raw_response_json": raw_response_json,
            "execution_invocation_fence_receipt_sha256": (
                authentication.get(
                    "execution_invocation_fence_receipt_sha256"
                )
            ),
            "execution_invocation_fence_status": (
                fence_status
            ),
            "execution_outbox_command_sha256": outbox_command_sha256,
            "raw_execution_outbox_acceptance_receipt_json": (
                outbox_acceptance_receipt_json
            ),
            "execution_outbox_acceptance_receipt_sha256": (
                outbox_acceptance_receipt_sha256
            ),
            "raw_execution_dispatch_terminal_receipt_json": (
                dispatch_terminal_receipt_json
            ),
            "execution_dispatch_terminal_receipt_sha256": (
                dispatch_terminal_receipt_sha256
            ),
        }
        if not _verify_signed_risk_domain_receipt(
            receipt,
            expected_core=expected_core,
            public_key_modulus_hex=(
                self._authorization.execution_public_key_modulus_hex
            ),
            public_key_exponent=self._authorization.execution_public_key_exponent,
        ):
            raise MicroLiveExecutionError(
                f"{operation} authenticated receipt is invalid"
            )
        return raw_response

    def _verify_execution_binding_attestation(self) -> None:
        challenge_sha256 = canonical_json_sha256(
            {
                "authorization_id": self._authorization.authorization_id,
                "authorization_payload_sha256": (
                    self._authorization.authorization_payload_sha256
                ),
                "execution_service_binding_sha256": (
                    self._authorization.execution_service_binding_sha256
                ),
                "fresh_nonce_sha256": hashlib.sha256(os.urandom(32)).hexdigest(),
            }
        )
        request = {
            "authorization_id": self._authorization.authorization_id,
            "challenge_sha256": challenge_sha256,
            "risk_domain_id": self._risk_domain_id,
            "risk_domain_authority_binding_sha256": (
                self._authorization.risk_domain_authority_binding_sha256
            ),
            "authorization_expires_at_ts_ms": self._authorization.expires_at_ts_ms,
            "execution_fence_protocol_schema_version": (
                EXECUTION_INVOCATION_FENCE_SCHEMA_VERSION
            ),
            "execution_acceptance_protocol_schema_version": (
                EXECUTION_INVOCATION_ACCEPTANCE_SCHEMA_VERSION
            ),
            "execution_dispatch_protocol_schema_version": (
                EXECUTION_DISPATCH_RECEIPT_SCHEMA_VERSION
            ),
            "execution_outbox_recovery_protocol_schema_version": (
                EXECUTION_OUTBOX_RECOVERY_SCHEMA_VERSION
            ),
            "submission_recovery_operation": SUBMISSION_RECOVERY_OPERATION,
            "submission_recovery_semantics": SUBMISSION_RECOVERY_SEMANTICS,
            "submission_recovery_lookup_only_enforced": True,
            "execution_transport_operation_inventory_schema_version": (
                EXECUTION_TRANSPORT_OPERATION_INVENTORY_SCHEMA_VERSION
            ),
            "required_execution_transport_operations": list(
                REQUIRED_EXECUTION_TRANSPORT_OPERATIONS
            ),
            "required_execution_transport_operations_sha256": (
                REQUIRED_EXECUTION_TRANSPORT_OPERATIONS_SHA256
            ),
            "cancellation_operation": CANCELLATION_OPERATION,
            "cancellation_semantics": CANCELLATION_SEMANTICS,
            "terminal_cursor_operation": TERMINAL_CURSOR_OPERATION,
            "terminal_cursor_semantics": TERMINAL_CURSOR_SEMANTICS,
            "venue_idempotency_key_field": VENUE_IDEMPOTENCY_KEY_FIELD,
            "venue_idempotency_scope": VENUE_IDEMPOTENCY_SCOPE,
            "venue_idempotency_semantics": VENUE_IDEMPOTENCY_SEMANTICS,
            "venue_idempotency_enforced": True,
            "deployment_runtime_lock_sha256": (
                self._authorization.deployment_runtime_lock_sha256
            ),
            "deployment_requirements_lock_sha256": (
                self._authorization.deployment_requirements_lock_sha256
            ),
            "deployment_image_manifest_digest": (
                self._authorization.deployment_image_manifest_digest
            ),
        }
        raw_attestation = self._bounded_transport_call(
            operation="attest_execution_binding",
            request=request,
        )
        attestation, _, _ = _raw_json_object(
            raw_attestation,
            "execution service binding attestation",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        expected_core = {
            "schema_version": EXECUTION_BINDING_ATTESTATION_SCHEMA_VERSION,
            "authorization_id": self._authorization.authorization_id,
            "challenge_sha256": challenge_sha256,
            "execution_service_binding_sha256": (
                self._authorization.execution_service_binding_sha256
            ),
            "service_identity_sha256": (
                self._authorization.execution_service_identity_sha256
            ),
            "adapter_implementation_sha256": (
                self._authorization.execution_adapter_implementation_sha256
            ),
            "configuration_sha256": (
                self._authorization.execution_configuration_sha256
            ),
            "exchange_endpoint_sha256": (
                self._authorization.execution_exchange_endpoint_sha256
            ),
            "exchange_account_sha256": (
                self._authorization.execution_exchange_account_sha256
            ),
            "signer_identity_sha256": (
                self._authorization.execution_signer_identity_sha256
            ),
            "cursor_key_identity_sha256": (
                self._authorization.execution_cursor_key_identity_sha256
            ),
            "clock_identity_sha256": (
                self._authorization.execution_clock_identity_sha256
            ),
            "settlement_authority_identity_sha256": (
                self._authorization.execution_settlement_authority_identity_sha256
            ),
            "risk_domain_id": self._risk_domain_id,
            "risk_domain_authority_binding_sha256": (
                self._authorization.risk_domain_authority_binding_sha256
            ),
            "authorization_expires_at_ts_ms": self._authorization.expires_at_ts_ms,
            "execution_fence_protocol_schema_version": (
                EXECUTION_INVOCATION_FENCE_SCHEMA_VERSION
            ),
            "execution_acceptance_protocol_schema_version": (
                EXECUTION_INVOCATION_ACCEPTANCE_SCHEMA_VERSION
            ),
            "execution_dispatch_protocol_schema_version": (
                EXECUTION_DISPATCH_RECEIPT_SCHEMA_VERSION
            ),
            "execution_outbox_recovery_protocol_schema_version": (
                EXECUTION_OUTBOX_RECOVERY_SCHEMA_VERSION
            ),
            "submission_recovery_operation": SUBMISSION_RECOVERY_OPERATION,
            "submission_recovery_semantics": SUBMISSION_RECOVERY_SEMANTICS,
            "submission_recovery_lookup_only_enforced": True,
            "execution_transport_operation_inventory_schema_version": (
                EXECUTION_TRANSPORT_OPERATION_INVENTORY_SCHEMA_VERSION
            ),
            "required_execution_transport_operations": list(
                REQUIRED_EXECUTION_TRANSPORT_OPERATIONS
            ),
            "required_execution_transport_operations_sha256": (
                REQUIRED_EXECUTION_TRANSPORT_OPERATIONS_SHA256
            ),
            "cancellation_operation": CANCELLATION_OPERATION,
            "cancellation_semantics": CANCELLATION_SEMANTICS,
            "terminal_cursor_operation": TERMINAL_CURSOR_OPERATION,
            "terminal_cursor_semantics": TERMINAL_CURSOR_SEMANTICS,
            "venue_idempotency_key_field": VENUE_IDEMPOTENCY_KEY_FIELD,
            "venue_idempotency_scope": VENUE_IDEMPOTENCY_SCOPE,
            "venue_idempotency_semantics": VENUE_IDEMPOTENCY_SEMANTICS,
            "venue_idempotency_enforced": True,
            "deployment_runtime_lock_sha256": (
                self._authorization.deployment_runtime_lock_sha256
            ),
            "deployment_requirements_lock_sha256": (
                self._authorization.deployment_requirements_lock_sha256
            ),
            "deployment_image_manifest_digest": (
                self._authorization.deployment_image_manifest_digest
            ),
        }
        if not _verify_signed_risk_domain_receipt(
            attestation,
            expected_core=expected_core,
            public_key_modulus_hex=(
                self._authorization.execution_public_key_modulus_hex
            ),
            public_key_exponent=self._authorization.execution_public_key_exponent,
        ):
            raise MicroLiveExecutionError(
                "execution service binding attestation is invalid"
            )

    def _verified_trusted_completion(
        self,
        *,
        request_started_at_ts_ms: int,
        operation: str,
        response_sha256: str,
    ) -> tuple[int, str, str]:
        expected_without_completion = _trusted_time_receipt_core(
            self._authorization,
            request_started_at_ts_ms=request_started_at_ts_ms,
            operation=operation,
            response_sha256=response_sha256,
            request_completed_at_ts_ms=None,
        )
        request = {
            key: value
            for key, value in expected_without_completion.items()
            if key not in {"schema_version", "request_completed_at_ts_ms"}
        }
        raw_receipt = self._bounded_transport_call(
            operation="read_trusted_time",
            request=request,
        )
        receipt, raw_receipt_json, receipt_sha256 = _raw_json_object(
            raw_receipt,
            "trusted execution completion time",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        completed_at_ts_ms = receipt.get("request_completed_at_ts_ms")
        if not _trusted_time_receipt_is_valid(
            receipt,
            self._authorization,
            request_started_at_ts_ms=request_started_at_ts_ms,
            operation=operation,
            response_sha256=response_sha256,
            request_completed_at_ts_ms=completed_at_ts_ms,
        ):
            raise MicroLiveExecutionError(
                "trusted execution completion time receipt is invalid"
            )
        return int(completed_at_ts_ms), raw_receipt_json, receipt_sha256

    def _require_authorization_integrity(
        self,
        *,
        now_ts_ms: int,
        reconciliation_only: bool = False,
    ) -> None:
        if authorization_capability_is_verified(self.authorization):
            try:
                if (
                    self._authorization.matches_verified(self.authorization)
                    and id(self._transport) == self._transport_object_id
                    and self._journal.authenticated_risk_domain_authority_binding_sha256
                    == self._authorization.risk_domain_authority_binding_sha256
                ):
                    return
            except Exception:
                pass
        now_is_valid = (
            isinstance(now_ts_ms, int)
            and not isinstance(now_ts_ms, bool)
            and now_ts_ms > 0
        )
        event_ts_ms = (
            now_ts_ms
            if now_is_valid
            else (
                int(self._events[-1]["event_ts_ms"])
                if self._events
                else self._authorization.authorized_at_ts_ms
            )
        )
        try:
            self.engage_kill_switch(
                reason="authorization_capability_integrity_failed",
                now_ts_ms=event_ts_ms,
            )
        except Exception as exc:
            raise MicroLiveExecutionError(
                "micro-live authorization capability changed and kill persistence failed"
            ) from exc
        if reconciliation_only:
            return
        raise MicroLiveExecutionError(
            "micro-live authorization capability changed after executor construction"
        )

    def _safe_persistence_timestamp(self) -> int:
        return (
            int(self._events[-1]["event_ts_ms"])
            if self._events
            else self._authorization.authorized_at_ts_ms
        )

    def _require_risk_entry_clock(
        self,
        now_ts_ms: Any,
        *,
        operation: str,
        signal_rejection: bool = False,
    ) -> int:
        valid = (
            isinstance(now_ts_ms, int)
            and not isinstance(now_ts_ms, bool)
            and now_ts_ms > 0
        )
        regressed = bool(
            valid
            and self._events
            and int(now_ts_ms) < int(self._events[-1]["event_ts_ms"])
        )
        if valid and not regressed:
            return int(now_ts_ms)
        fallback_ts_ms = self._safe_persistence_timestamp()
        failure = "clock_regression" if regressed else "clock_invalid"
        kill_reason = f"{operation}_{failure}"
        self._persist_emergency_kill(
            reason=kill_reason,
            event_ts_ms=fallback_ts_ms,
        )
        if signal_rejection:
            with suppress(MicroLiveExecutionError):
                self._append_event(
                    "SIGNAL_REJECTED",
                    {
                        "authorization_id": self._authorization.authorization_id,
                        "candidate_bundle_sha256": (
                            self._authorization.candidate_bundle_sha256
                        ),
                        "reason": "signal_validation_or_clock_failed",
                        "error_type": "InvalidTrustedClock",
                    },
                    event_ts_ms=fallback_ts_ms,
                )
            # The independent kill record is authoritative.  Rejection audit
            # is best effort when only lifecycle recovery capacity remains.
        self.engage_kill_switch(
            reason=kill_reason,
            now_ts_ms=fallback_ts_ms,
        )
        label = operation.replace("_", " ")
        message = "timestamp regressed" if regressed else "timestamp is invalid"
        raise MicroLiveExecutionError(f"{label} {message}")

    def enforce_runtime_safety(
        self,
        *,
        now_ts_ms: int,
        operator_heartbeat_ts_ms: int,
    ) -> dict[str, Any]:
        """Enforce time/loss safety even when no new model signal arrives.

        The future operator must call this watchdog independently of signal
        production.  A stale heartbeat or expired authorization therefore
        cancels acknowledged open orders instead of waiting for another market
        decision to exercise the signal-submission checks.
        """

        valid_now = (
            isinstance(now_ts_ms, int)
            and not isinstance(now_ts_ms, bool)
            and now_ts_ms > 0
        )
        emergency_ts_ms = (
            int(now_ts_ms)
            if valid_now
            else self._authorization.authorized_at_ts_ms
        )
        emergency_reason: str | None = None
        if not valid_now:
            emergency_reason = "runtime_watchdog_clock_invalid"
        elif now_ts_ms < self._authorization.authorized_at_ts_ms:
            emergency_reason = "runtime_clock_before_authorization"
        elif now_ts_ms >= self._authorization.expires_at_ts_ms:
            emergency_reason = "authorization_expired"
        elif (
            isinstance(operator_heartbeat_ts_ms, bool)
            or not isinstance(operator_heartbeat_ts_ms, int)
            or operator_heartbeat_ts_ms <= 0
            or operator_heartbeat_ts_ms > now_ts_ms
            or now_ts_ms - operator_heartbeat_ts_ms
            > self._authorization.maximum_operator_heartbeat_age_ms
        ):
            emergency_reason = "operator_heartbeat_stale"
        if emergency_reason is not None:
            self._persist_emergency_kill(
                reason=emergency_reason,
                event_ts_ms=emergency_ts_ms,
            )
        with self._durable_transaction():
            return self._enforce_runtime_safety_locked(
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )

    def _enforce_runtime_safety_locked(
        self,
        *,
        now_ts_ms: int,
        operator_heartbeat_ts_ms: int,
    ) -> dict[str, Any]:

        self._require_authorization_integrity(now_ts_ms=now_ts_ms)
        now_ts_ms = self._require_risk_entry_clock(
            now_ts_ms,
            operation="runtime_watchdog",
        )

        view = self._reconcile_view()
        if view["kill_switch_active"]:
            if any(order["is_open"] for order in view["orders"].values()):
                return self.engage_kill_switch(
                    reason=str(view["kill_switch_reason"]),
                    now_ts_ms=now_ts_ms,
                )
            return {
                "status": "KILL_SWITCH_ALREADY_ACTIVE",
                "reason": view["kill_switch_reason"],
                "transport_called": False,
            }
        reason: str | None = None
        if now_ts_ms < self._authorization.authorized_at_ts_ms:
            reason = "runtime_clock_before_authorization"
        elif now_ts_ms >= self._authorization.expires_at_ts_ms:
            reason = "authorization_expired"
        elif (
            isinstance(operator_heartbeat_ts_ms, bool)
            or not isinstance(operator_heartbeat_ts_ms, int)
            or operator_heartbeat_ts_ms <= 0
            or operator_heartbeat_ts_ms > now_ts_ms
            or now_ts_ms - operator_heartbeat_ts_ms
            > self._authorization.maximum_operator_heartbeat_age_ms
        ):
            reason = "operator_heartbeat_stale"
        elif _realized_loss_limit_reached(view, self._authorization):
            reason = "maximum_realized_loss_reached"
        elif (
            view["loss_budget_consumed_usd"]
            > self._authorization.maximum_realized_loss_usd
        ):
            reason = "maximum_loss_budget_exceeded"

        if reason is not None:
            return self.engage_kill_switch(reason=reason, now_ts_ms=now_ts_ms)
        return {
            "status": "RUNTIME_SAFETY_OK",
            "checked_at_ts_ms": now_ts_ms,
            "operator_heartbeat_ts_ms": operator_heartbeat_ts_ms,
            "transport_called": False,
        }

    @_durable_entry
    def submit_signal(
        self,
        *,
        raw_signal_payload: bytes,
        raw_feature_row: bytes,
        provider_feature_evidence: Mapping[str, bytes],
        now_ts_ms: int,
        operator_heartbeat_ts_ms: int,
        market_identity_evidence: Mapping[str, bytes] | None = None,
    ) -> dict[str, Any]:
        """Strictly decode one signal/input pair before any executable decision."""

        self._require_authorization_integrity(now_ts_ms=now_ts_ms)
        now_ts_ms = self._require_risk_entry_clock(
            now_ts_ms,
            operation="signal_submission",
            signal_rejection=True,
        )
        try:
            signal_payload, raw_signal_json, raw_signal_sha256 = _raw_json_object(
                raw_signal_payload,
                "candidate signal payload",
            )
            feature_row, raw_feature_json, raw_feature_sha256 = _raw_json_object(
                raw_feature_row,
                "candidate feature row",
            )
            signal = _validated_candidate_signal(
                signal_payload,
                expected_candidate_bundle_sha256=(
                    self._authorization.candidate_bundle_sha256
                ),
            )
            _validate_market_identity_evidence(
                signal=signal,
                evidence=market_identity_evidence,
            )
            features = _validated_runtime_binding(
                signal=signal,
                feature_row=feature_row,
                runtime=self._authorization.runtime,
            )
            try:
                verified_feature_evidence = verify_provider_feature_evidence(
                    raw_evidence=provider_feature_evidence,
                    signal=signal,
                    feature_row=features,
                )
            except ProviderFeatureEvidenceError as exc:
                raise MicroLiveExecutionError(
                    "candidate feature row is not bound to provider bytes"
                ) from exc
            decision_ts_ms = int(signal["decision_ts_ms"])
            self._validate_clock(
                now_ts_ms,
                operator_heartbeat_ts_ms,
                decision_ts_ms,
                int(signal["observed_at_ts_ms"]),
                int(dict(signal["market_identity"])["clob_revalidated_at_ts_ms"]),
            )
        except MicroLiveExecutionError as exc:
            rejection_ts_ms = max(
                now_ts_ms,
                int(self._events[-1]["event_ts_ms"])
                if self._events
                else now_ts_ms,
            )
            self._persist_emergency_kill(
                reason="signal_validation_or_clock_failed",
                event_ts_ms=rejection_ts_ms,
            )
            with suppress(MicroLiveExecutionError):
                self._append_event(
                    "SIGNAL_REJECTED",
                    {
                        "authorization_id": self._authorization.authorization_id,
                        "candidate_bundle_sha256": (
                            self._authorization.candidate_bundle_sha256
                        ),
                        "reason": "signal_validation_or_clock_failed",
                        "error_type": exc.__class__.__name__,
                    },
                    event_ts_ms=rejection_ts_ms,
                )
            self.engage_kill_switch(
                reason="signal_validation_or_clock_failed",
                now_ts_ms=now_ts_ms,
            )
            raise
        view = self._reconcile_view()
        if (
            len(self._events)
            + 2
            + _required_lifecycle_event_capacity(view)
            + _new_order_lifecycle_capacity()
            > MAX_EVENT_COUNT
        ):
            self.engage_kill_switch(
                reason="journal_routine_capacity_exhausted",
                now_ts_ms=now_ts_ms,
            )
            return _blocked("journal_routine_capacity_exhausted")
        if _realized_loss_limit_reached(view, self._authorization) and not view[
            "kill_switch_active"
        ]:
            self.engage_kill_switch(
                reason="maximum_realized_loss_reached",
                now_ts_ms=now_ts_ms,
            )
            view = self._reconcile_view()
        if view["kill_switch_active"]:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                raw_signal_json=raw_signal_json,
                raw_signal_sha256=raw_signal_sha256,
                raw_feature_json=raw_feature_json,
                raw_feature_sha256=raw_feature_sha256,
                provider_feature_evidence=verified_feature_evidence,
                disposition="BLOCKED_NO_TRADE",
                reason="kill_switch_active",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("kill_switch_active")
        market_id = str(signal["market_id"])
        slug = str(signal["slug"])
        market_family = str(signal["market_family"])
        candidate_bundle_sha256 = str(signal["candidate_bundle_sha256"])
        selected_action = str(signal["selected_action"])
        if market_family not in self._authorization.market_allowlist:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                raw_signal_json=raw_signal_json,
                raw_signal_sha256=raw_signal_sha256,
                raw_feature_json=raw_feature_json,
                raw_feature_sha256=raw_feature_sha256,
                provider_feature_evidence=verified_feature_evidence,
                disposition="BLOCKED_NO_TRADE",
                reason="market_not_allowlisted",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("market_not_allowlisted")
        if signal["fail_closed"] is True or signal["model_scored"] is not True:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                raw_signal_json=raw_signal_json,
                raw_signal_sha256=raw_signal_sha256,
                raw_feature_json=raw_feature_json,
                raw_feature_sha256=raw_feature_sha256,
                provider_feature_evidence=verified_feature_evidence,
                disposition="BLOCKED_NO_TRADE",
                reason="signal_failed_closed",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("signal_failed_closed")
        if selected_action == "NO_TRADE":
            self._audit_signal_decision(
                signal=signal,
                features=features,
                raw_signal_json=raw_signal_json,
                raw_signal_sha256=raw_signal_sha256,
                raw_feature_json=raw_feature_json,
                raw_feature_sha256=raw_feature_sha256,
                provider_feature_evidence=verified_feature_evidence,
                disposition="BLOCKED_NO_TRADE",
                reason="signal_selected_no_trade",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("signal_selected_no_trade")
        if selected_action not in self._authorization.allowed_actions:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                raw_signal_json=raw_signal_json,
                raw_signal_sha256=raw_signal_sha256,
                raw_feature_json=raw_feature_json,
                raw_feature_sha256=raw_feature_sha256,
                provider_feature_evidence=verified_feature_evidence,
                disposition="BLOCKED_NO_TRADE",
                reason="action_not_allowlisted",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("action_not_allowlisted")
        token_side = "UP" if selected_action == "BUY_UP_HOLD" else "DOWN"
        token_id = str(signal[f"{token_side.lower()}_token_id"])
        price = _positive_decimal(
            dict(signal["executable_asks"])[token_side],
            "limit price",
        )
        qty = Decimal("1")
        notional = price * qty
        maximum_fee = FROZEN_EXECUTION_FEE_PER_UNIT_USD * qty
        maximum_loss = notional + maximum_fee
        signal_payload_sha256 = canonical_json_sha256(signal)
        business_key = canonical_json_sha256(
            {
                "authorization_id": self._authorization.authorization_id,
                "candidate_bundle_sha256": candidate_bundle_sha256,
                "market_id": market_id,
            }
        )
        intent_core = {
            "authorization_id": self._authorization.authorization_id,
            "candidate_bundle_sha256": candidate_bundle_sha256,
            "business_key": business_key,
            "market_id": market_id,
            "slug": slug,
            "market_family": market_family,
            "decision_ts_ms": decision_ts_ms,
            "selected_action": selected_action,
            "token_id": token_id,
            "token_side": token_side,
            "limit_price": str(price),
            "quantity": str(qty),
            "notional_usd": str(notional),
            "maximum_fee_usd": str(maximum_fee),
            "maximum_loss_usd": str(maximum_loss),
            "signal_payload_sha256": signal_payload_sha256,
            "signal_payload": signal,
            "raw_signal_payload_sha256": raw_signal_sha256,
            "raw_signal_payload_json": raw_signal_json,
            "market_identity_sha256": canonical_json_sha256(
                dict(signal["market_identity"])
            ),
            "market_identity": dict(signal["market_identity"]),
            "feature_row_sha256": canonical_json_sha256(features),
            "feature_row": features,
            "raw_feature_row_sha256": raw_feature_sha256,
            "raw_feature_row_json": raw_feature_json,
            "provider_feature_evidence_graph_sha256": (
                verified_feature_evidence.evidence_graph_sha256
            ),
            "provider_feature_file_sha256": verified_feature_evidence.file_sha256,
            "provider_reconstructed_feature_row_sha256": (
                verified_feature_evidence.reconstructed_feature_row_sha256
            ),
            "raw_provider_feature_evidence_jsonl": (
                verified_feature_evidence.raw_jsonl
            ),
        }
        intent_id = canonical_json_sha256(intent_core)
        client_order_id = intent_id
        transport_invocation_id = canonical_json_sha256(
            {
                "authorization_id": self._authorization.authorization_id,
                "client_order_id": client_order_id,
                "operation": "submit_order",
                "invocation_number": 1,
            }
        )
        existing = view["orders_by_business_key"].get(business_key)
        if existing is not None:
            if existing["intent_id"] != intent_id:
                if existing["prepared"]["decision_ts_ms"] != decision_ts_ms:
                    self._audit_signal_decision(
                        signal=signal,
                        features=features,
                        raw_signal_json=raw_signal_json,
                        raw_signal_sha256=raw_signal_sha256,
                        raw_feature_json=raw_feature_json,
                        raw_feature_sha256=raw_feature_sha256,
                        provider_feature_evidence=verified_feature_evidence,
                        disposition="BLOCKED_NO_TRADE",
                        reason="one_trade_maximum_per_market",
                        now_ts_ms=now_ts_ms,
                        operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
                    )
                    return _blocked("one_trade_maximum_per_market")
                self._audit_signal_decision(
                    signal=signal,
                    features=features,
                    raw_signal_json=raw_signal_json,
                    raw_signal_sha256=raw_signal_sha256,
                    raw_feature_json=raw_feature_json,
                    raw_feature_sha256=raw_feature_sha256,
                    provider_feature_evidence=verified_feature_evidence,
                    disposition="REJECTED_CONFLICT",
                    reason="conflicting_duplicate_intent",
                    now_ts_ms=now_ts_ms,
                    operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
                )
                self.engage_kill_switch(
                    reason="conflicting_duplicate_intent",
                    now_ts_ms=now_ts_ms,
                )
                raise MicroLiveExecutionError("conflicting duplicate intent failed closed")
            if existing.get("acknowledgement") is not None:
                self._audit_signal_decision(
                    signal=signal,
                    features=features,
                    raw_signal_json=raw_signal_json,
                    raw_signal_sha256=raw_signal_sha256,
                    raw_feature_json=raw_feature_json,
                    raw_feature_sha256=raw_feature_sha256,
                    provider_feature_evidence=verified_feature_evidence,
                    disposition="IDEMPOTENT_REPLAY",
                    reason="existing_identical_intent",
                    now_ts_ms=now_ts_ms,
                    operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
                )
                return {
                    "status": "IDEMPOTENT_REPLAY",
                    "client_order_id": client_order_id,
                    "exchange_order_id": existing["acknowledgement"]["exchange_order_id"],
                    "transport_called": False,
                }
            self._audit_signal_decision(
                signal=signal,
                features=features,
                raw_signal_json=raw_signal_json,
                raw_signal_sha256=raw_signal_sha256,
                raw_feature_json=raw_feature_json,
                raw_feature_sha256=raw_feature_sha256,
                provider_feature_evidence=verified_feature_evidence,
                disposition="BLOCKED_NO_TRADE",
                reason="existing_order_requires_reconciliation",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("existing_order_requires_reconciliation", client_order_id)
        if view["submitted_notional_usd"] + notional > self._authorization.maximum_notional_usd:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                raw_signal_json=raw_signal_json,
                raw_signal_sha256=raw_signal_sha256,
                raw_feature_json=raw_feature_json,
                raw_feature_sha256=raw_feature_sha256,
                provider_feature_evidence=verified_feature_evidence,
                disposition="BLOCKED_NO_TRADE",
                reason="authorization_notional_cap_exceeded",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("authorization_notional_cap_exceeded")
        if view["open_order_count"] >= self._authorization.maximum_open_orders:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                raw_signal_json=raw_signal_json,
                raw_signal_sha256=raw_signal_sha256,
                raw_feature_json=raw_feature_json,
                raw_feature_sha256=raw_feature_sha256,
                provider_feature_evidence=verified_feature_evidence,
                disposition="BLOCKED_NO_TRADE",
                reason="maximum_open_orders_reached",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("maximum_open_orders_reached")
        if (
            view["loss_budget_consumed_usd"] + maximum_loss
            > self._authorization.maximum_realized_loss_usd
        ):
            self._audit_signal_decision(
                signal=signal,
                features=features,
                raw_signal_json=raw_signal_json,
                raw_signal_sha256=raw_signal_sha256,
                raw_feature_json=raw_feature_json,
                raw_feature_sha256=raw_feature_sha256,
                provider_feature_evidence=verified_feature_evidence,
                disposition="BLOCKED_NO_TRADE",
                reason="maximum_loss_reservation_exceeded",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("maximum_loss_reservation_exceeded")

        prepared = {
            **intent_core,
            "intent_id": intent_id,
            "client_order_id": client_order_id,
            "transport_invocation_id": transport_invocation_id,
            "submitted_at_ts_ms": now_ts_ms,
            "operator_heartbeat_ts_ms": operator_heartbeat_ts_ms,
            "authorization_payload_sha256": (
                self._authorization.authorization_payload_sha256
            ),
        }
        audit_payload = self._signal_decision_audit_payload(
            signal=signal,
            features=features,
            raw_signal_json=raw_signal_json,
            raw_signal_sha256=raw_signal_sha256,
            raw_feature_json=raw_feature_json,
            raw_feature_sha256=raw_feature_sha256,
            provider_feature_evidence=verified_feature_evidence,
            disposition="EXECUTION_INTENT",
            reason=None,
            operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
        )
        try:
            self._require_execution_intent_preparation_capacity(
                audit_payload=audit_payload,
                prepared_payload=prepared,
                event_ts_ms=now_ts_ms,
            )
        except MicroLiveExecutionError:
            self.engage_kill_switch(
                reason="journal_byte_capacity_exhausted",
                now_ts_ms=now_ts_ms,
            )
            return _blocked("journal_byte_capacity_exhausted")
        self._append_event(
            "SIGNAL_EVALUATED",
            audit_payload,
            event_ts_ms=now_ts_ms,
        )
        self._append_event("ORDER_PREPARED", prepared, event_ts_ms=now_ts_ms)
        transport_request = {
            key: prepared[key]
            for key in (
                "authorization_id",
                "authorization_payload_sha256",
                "candidate_bundle_sha256",
                "business_key",
                "market_id",
                "slug",
                "market_family",
                "decision_ts_ms",
                "selected_action",
                "token_id",
                "token_side",
                "limit_price",
                "quantity",
                "notional_usd",
                "maximum_fee_usd",
                "maximum_loss_usd",
                "signal_payload_sha256",
                "raw_signal_payload_sha256",
                "market_identity_sha256",
                "market_identity",
                "raw_feature_row_sha256",
                "provider_feature_evidence_graph_sha256",
                "provider_feature_file_sha256",
                "intent_id",
                "client_order_id",
                "transport_invocation_id",
                "submitted_at_ts_ms",
            )
        }
        try:
            response, raw_response_json, response_sha256 = _raw_json_object(
                self._bounded_transport_call(
                    operation="submit_order",
                    request=transport_request,
                ),
                "order submission response",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            disposition = self._validate_submission_response(transport_request, response)
            if (
                disposition["status"] == "ACCEPTED"
                and disposition["exchange_order_id"]
                in view["orders_by_exchange_id"]
            ):
                raise MicroLiveExecutionError(
                    "exchange order identity was reused across client orders"
                )
        except Exception as exc:
            self._append_event(
                "ORDER_SUBMISSION_UNKNOWN",
                {
                    "client_order_id": client_order_id,
                    "error_type": exc.__class__.__name__,
                },
                event_ts_ms=now_ts_ms,
            )
            self.engage_kill_switch(
                reason="order_submission_unknown",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("order submission became unknown; kill switch engaged") from exc
        try:
            disposition_event_type = (
                "ORDER_REJECTED"
                if disposition["status"] == "REJECTED"
                else "ORDER_ACKNOWLEDGED"
            )
            self._append_event(
                disposition_event_type,
                {
                    **disposition,
                    "transport_event_sha256": response_sha256,
                    "raw_transport_event_json": raw_response_json,
                },
                event_ts_ms=now_ts_ms,
            )
        except Exception as exc:
            self._append_event(
                "ORDER_SUBMISSION_UNKNOWN",
                {
                    "client_order_id": client_order_id,
                    "error_type": exc.__class__.__name__,
                },
                event_ts_ms=now_ts_ms,
            )
            self.engage_kill_switch(
                reason="order_submission_disposition_persistence_failed",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError(
                "order submission disposition could not be persisted; "
                "submission remains unknown"
            ) from exc
        if disposition["status"] == "REJECTED":
            return {
                "status": "ORDER_REJECTED",
                "client_order_id": client_order_id,
                "transport_called": True,
            }
        self._reconcile_view()
        return {
            "status": "ORDER_ACKNOWLEDGED",
            "client_order_id": client_order_id,
            "exchange_order_id": disposition["exchange_order_id"],
            "transport_called": True,
        }

    @_durable_entry
    def reconcile_unknown_submission(
        self,
        *,
        client_order_id: str,
        now_ts_ms: int,
    ) -> dict[str, Any]:
        """Resolve one unknown submission through lookup-only durable recovery.

        Reconciliation never clears the persistent kill switch and never
        resubmits an order.  It first asks the gateway to recover the exact
        durable outbox request and perform a venue-idempotency lookup only.  A
        successful recovery terminalizes dispatch before the signed fence
        check; otherwise the legacy fence-then-lookup path remains fail-closed.
        """

        self._require_authorization_integrity(
            now_ts_ms=now_ts_ms,
            reconciliation_only=True,
        )
        now_ts_ms = self._require_risk_entry_clock(
            now_ts_ms,
            operation="submission_reconciliation",
        )
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if not (
            order is not None
            and order.get("submission_unknown") is True
            and order.get("acknowledgement") is None
            and order.get("closed_status") is None
            and view["kill_switch_active"] is True
        ):
            self.engage_kill_switch(
                reason="submission_reconciliation_precondition_failed",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError(
                "submission reconciliation requires one killed unknown order"
            )
        prepared = dict(order["prepared"])
        lookup_request = {
            "authorization_id": self._authorization.authorization_id,
            "client_order_id": client_order_id,
            "business_key": prepared["business_key"],
            "market_id": prepared["market_id"],
            "token_id": prepared["token_id"],
        }
        raw_response_json: str | None = None
        response_sha256: str | None = None
        raw_fence_response_json: str | None = None
        fence_response_sha256: str | None = None
        fence_request = {
            "authorization_id": self._authorization.authorization_id,
            "client_order_id": client_order_id,
            "business_key": prepared["business_key"],
            "market_id": prepared["market_id"],
            "token_id": prepared["token_id"],
            "transport_invocation_id": prepared["transport_invocation_id"],
        }
        try:
            recovered_response: bytes | None = None
            try:
                recovered_response = self._bounded_submission_recovery_call(
                    prepared=prepared,
                )
            except SubmissionRecoveryOutcomeNotFoundError:
                # No outcome is normal for a command that never reached the
                # venue.  The following durable fence decides whether an
                # ordinary lookup is safe; IN_PROGRESS remains unresolved.
                recovered_response = None
            fence_response, raw_fence_response_json, fence_response_sha256 = (
                _raw_json_object(
                    self._bounded_transport_call(
                        operation="fence_order_invocation",
                        request=fence_request,
                    ),
                    "submission invocation fence response",
                    maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
                )
            )
            self._validate_submission_fence_response(
                prepared=prepared,
                response=fence_response,
            )
            raw_lookup_response = (
                recovered_response
                if recovered_response is not None
                else self._bounded_transport_call(
                    operation="lookup_order",
                    request=lookup_request,
                )
            )
            response, raw_response_json, response_sha256 = _raw_json_object(
                raw_lookup_response,
                "submission lookup response",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            disposition = self._validate_submission_response(prepared, response)
            exchange_order_id = disposition["exchange_order_id"]
            if (
                disposition["status"] == "ACCEPTED"
                and exchange_order_id in view["orders_by_exchange_id"]
            ):
                raise MicroLiveExecutionError(
                    "reconciled exchange order identity was reused"
                )
        except Exception as exc:
            self._append_event(
                "ORDER_SUBMISSION_RECONCILIATION_FAILED",
                {
                    "client_order_id": client_order_id,
                    "lookup_request_sha256": canonical_json_sha256(lookup_request),
                    "lookup_response_sha256": response_sha256,
                    "raw_lookup_response_json": raw_response_json,
                    "fence_request_sha256": canonical_json_sha256(fence_request),
                    "fence_response_sha256": fence_response_sha256,
                    "raw_fence_response_json": raw_fence_response_json,
                    "error_type": exc.__class__.__name__,
                },
                event_ts_ms=max(
                    now_ts_ms,
                    int(self._events[-1]["event_ts_ms"]),
                ),
            )
            self.engage_kill_switch(
                reason="submission_reconciliation_failed",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError(
                "unknown submission reconciliation failed closed"
            ) from exc
        self._append_event(
            "ORDER_SUBMISSION_RECONCILED",
            {
                **disposition,
                "lookup_response_sha256": response_sha256,
                "raw_lookup_response_json": raw_response_json,
                "transport_invocation_id": prepared["transport_invocation_id"],
                "fence_response_sha256": fence_response_sha256,
                "raw_fence_response_json": raw_fence_response_json,
            },
            event_ts_ms=now_ts_ms,
        )
        cancel_result = self.engage_kill_switch(
            reason="submission_reconciled_after_unknown",
            now_ts_ms=now_ts_ms,
        )
        self._reconcile_view()
        return {
            "status": f"ORDER_SUBMISSION_RECONCILED_{disposition['status']}",
            "client_order_id": client_order_id,
            "exchange_order_id": disposition["exchange_order_id"],
            "kill_switch_active": True,
            "cancel_result": cancel_result,
        }

    @_durable_entry
    def reconcile_unknown_cancellation(
        self,
        *,
        client_order_id: str,
        now_ts_ms: int,
    ) -> dict[str, Any]:
        """Resolve one ambiguous cancel through the authoritative fill cursor.

        This method never submits or cancels an order and never clears the
        persistent kill switch.  It first ingests every missing cumulative fill
        and closes only on cursor-proven FILLED, CANCELED, or EXPIRED finality.
        OPEN or invalid results remain audited and unresolved.
        """

        self._require_authorization_integrity(
            now_ts_ms=now_ts_ms,
            reconciliation_only=True,
        )
        now_ts_ms = self._require_risk_entry_clock(
            now_ts_ms,
            operation="cancel_reconciliation",
        )
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if not (
            order is not None
            and order.get("acknowledgement") is not None
            and order.get("cancel_unknown") is True
            and order.get("closed_status") is None
            and order.get("settlement") is None
            and view["kill_switch_active"] is True
        ):
            self.engage_kill_switch(
                reason="cancel_reconciliation_precondition_failed",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError(
                "cancel reconciliation requires one killed cancel-unknown order"
            )
        try:
            result = self._reconcile_authoritative_fill_cursor_locked(
                client_order_id=client_order_id,
                now_ts_ms=now_ts_ms,
            )
        except Exception as exc:
            self._append_event(
                "ORDER_CANCEL_RECONCILIATION_FAILED",
                {
                    "client_order_id": client_order_id,
                    "lookup_request_sha256": canonical_json_sha256(
                        {
                            "authorization_id": self._authorization.authorization_id,
                            "client_order_id": client_order_id,
                            "business_key": order["prepared"]["business_key"],
                            "exchange_order_id": order["acknowledgement"][
                                "exchange_order_id"
                            ],
                            "market_id": order["prepared"]["market_id"],
                            "token_id": order["prepared"]["token_id"],
                        }
                    ),
                    "authenticated_cursor_request_sha256": canonical_json_sha256(
                        {
                            "authorization_id": self._authorization.authorization_id,
                            "execution_service_binding_sha256": (
                                self._authorization.execution_service_binding_sha256
                            ),
                            "client_order_id": client_order_id,
                            "business_key": order["prepared"]["business_key"],
                            "exchange_order_id": order["acknowledgement"][
                                "exchange_order_id"
                            ],
                            "market_id": order["prepared"]["market_id"],
                            "token_id": order["prepared"]["token_id"],
                            "request_started_at_ts_ms": now_ts_ms,
                        }
                    ),
                    "request_started_at_ts_ms": now_ts_ms,
                    "request_completed_at_ts_ms": None,
                    "trusted_time_receipt_sha256": None,
                    "raw_trusted_time_receipt_json": None,
                    "error_type": exc.__class__.__name__,
                    "observed_status": None,
                    "lookup_response_sha256": None,
                    "raw_lookup_response_json": None,
                },
                event_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError(
                "unknown cancellation reconciliation failed closed"
            ) from exc
        if result["status"] == "ORDER_FILL_CURSOR_OPEN":
            self._append_event(
                "ORDER_CANCEL_RECONCILIATION_FAILED",
                {
                    "client_order_id": client_order_id,
                    "lookup_request_sha256": result["cursor_request_sha256"],
                    "authenticated_cursor_request_sha256": result[
                        "authenticated_cursor_request_sha256"
                    ],
                    "request_started_at_ts_ms": result[
                        "request_started_at_ts_ms"
                    ],
                    "request_completed_at_ts_ms": result[
                        "request_completed_at_ts_ms"
                    ],
                    "trusted_time_receipt_sha256": result[
                        "trusted_time_receipt_sha256"
                    ],
                    "raw_trusted_time_receipt_json": result[
                        "raw_trusted_time_receipt_json"
                    ],
                    "error_type": None,
                    "observed_status": "OPEN",
                    "lookup_response_sha256": result[
                        "cursor_response_sha256"
                    ],
                    "raw_lookup_response_json": result[
                        "raw_cursor_response_json"
                    ],
                },
                event_ts_ms=result["request_completed_at_ts_ms"],
            )
        self._reconcile_view()
        reconciled_status = {
            "ORDER_FILLED": "FILLED",
            "ORDER_CANCELED": "CANCELED",
            "ORDER_EXPIRED": "EXPIRED",
            "ORDER_FILL_CURSOR_OPEN": "OPEN",
        }.get(str(result["status"]), str(result["status"]))
        return {
            "status": (
                "ORDER_CANCEL_RECONCILIATION_OPEN"
                if reconciled_status == "OPEN"
                else f"ORDER_CANCEL_RECONCILED_{reconciled_status}"
            ),
            "client_order_id": client_order_id,
            "kill_switch_active": True,
            "order_closed": result["status"] != "ORDER_FILL_CURSOR_OPEN",
            "lookup_called": True,
            "write_transport_called": False,
        }

    def _audit_signal_decision(
        self,
        *,
        signal: Mapping[str, Any],
        features: Mapping[str, Any],
        raw_signal_json: str,
        raw_signal_sha256: str,
        raw_feature_json: str,
        raw_feature_sha256: str,
        provider_feature_evidence: VerifiedProviderFeatureEvidence,
        disposition: str,
        reason: str | None,
        now_ts_ms: int,
        operator_heartbeat_ts_ms: int,
    ) -> None:
        """Append the complete causal signal/input/decision audit row."""

        payload = self._signal_decision_audit_payload(
            signal=signal,
            features=features,
            raw_signal_json=raw_signal_json,
            raw_signal_sha256=raw_signal_sha256,
            raw_feature_json=raw_feature_json,
            raw_feature_sha256=raw_feature_sha256,
            provider_feature_evidence=provider_feature_evidence,
            disposition=disposition,
            reason=reason,
            operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
        )
        self._append_event(
            "SIGNAL_EVALUATED",
            payload,
            event_ts_ms=now_ts_ms,
        )

    def _signal_decision_audit_payload(
        self,
        *,
        signal: Mapping[str, Any],
        features: Mapping[str, Any],
        raw_signal_json: str,
        raw_signal_sha256: str,
        raw_feature_json: str,
        raw_feature_sha256: str,
        provider_feature_evidence: VerifiedProviderFeatureEvidence,
        disposition: str,
        reason: str | None,
        operator_heartbeat_ts_ms: int,
    ) -> dict[str, Any]:
        signal_copy = copy.deepcopy(dict(signal))
        feature_copy = copy.deepcopy(dict(features))
        core = {
            "authorization_id": self._authorization.authorization_id,
            "candidate_bundle_sha256": self._authorization.candidate_bundle_sha256,
            "market_id": signal_copy["market_id"],
            "decision_ts_ms": signal_copy["decision_ts_ms"],
            "operator_heartbeat_ts_ms": operator_heartbeat_ts_ms,
            "signal_payload_sha256": canonical_json_sha256(signal_copy),
            "signal_payload": signal_copy,
            "raw_signal_payload_sha256": raw_signal_sha256,
            "raw_signal_payload_json": raw_signal_json,
            "market_identity_sha256": canonical_json_sha256(
                dict(signal_copy["market_identity"])
            ),
            "market_identity": copy.deepcopy(dict(signal_copy["market_identity"])),
            "feature_row_sha256": canonical_json_sha256(feature_copy),
            "feature_row": feature_copy,
            "raw_feature_row_sha256": raw_feature_sha256,
            "raw_feature_row_json": raw_feature_json,
            "provider_feature_evidence_graph_sha256": (
                provider_feature_evidence.evidence_graph_sha256
            ),
            "provider_feature_file_sha256": provider_feature_evidence.file_sha256,
            "provider_reconstructed_feature_row_sha256": (
                provider_feature_evidence.reconstructed_feature_row_sha256
            ),
            "raw_provider_feature_evidence_jsonl": (
                provider_feature_evidence.raw_jsonl
            ),
            "disposition": disposition,
            "reason": reason,
        }
        return {**core, "decision_audit_sha256": canonical_json_sha256(core)}

    @_durable_entry
    def record_fill(
        self,
        *,
        client_order_id: str,
        fill_id: str,
        now_ts_ms: int,
        quantity: str,
        price: str,
        fee_usd: str,
        fill_event_sequence: int,
        cumulative_filled_quantity: str,
        cumulative_fill_count: int,
        raw_transport_event: bytes,
    ) -> dict[str, Any]:
        """Record one fill; every trusted-time reconciliation error kills."""

        self._require_authorization_integrity(
            now_ts_ms=now_ts_ms,
            reconciliation_only=True,
        )
        now_ts_ms = self._require_risk_entry_clock(
            now_ts_ms,
            operation="fill_observation",
        )
        try:
            return self._record_fill(
                client_order_id=client_order_id,
                fill_id=fill_id,
                now_ts_ms=now_ts_ms,
                quantity=quantity,
                price=price,
                fee_usd=fee_usd,
                fill_event_sequence=fill_event_sequence,
                cumulative_filled_quantity=cumulative_filled_quantity,
                cumulative_fill_count=cumulative_fill_count,
                raw_transport_event=raw_transport_event,
            )
        except MicroLiveExecutionError:
            self.engage_kill_switch(
                reason="fill_reconciliation_failed",
                now_ts_ms=now_ts_ms,
            )
            raise

    def _record_fill(
        self,
        *,
        client_order_id: str,
        fill_id: str,
        now_ts_ms: int,
        quantity: str,
        price: str,
        fee_usd: str,
        fill_event_sequence: int,
        cumulative_filled_quantity: str,
        cumulative_fill_count: int,
        raw_transport_event: bytes,
    ) -> dict[str, Any]:
        """Apply one externally observed fill to the append-only ledger."""

        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if order is None or order.get("acknowledgement") is None:
            raise MicroLiveExecutionError("fill has no acknowledged order")
        if order.get("settlement") is not None:
            self.engage_kill_switch(reason="fill_after_terminal_state", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("fill arrived after settlement")
        if not isinstance(fill_id, str) or not fill_id:
            raise MicroLiveExecutionError("fill identity is invalid")
        prepared = dict(order["prepared"])
        acknowledgement = dict(order["acknowledgement"])
        transport_event, raw_json, transport_event_sha256 = _raw_json_object(
            raw_transport_event,
            "fill transport event",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        expected_transport_keys = {
            "event_type",
            "client_order_id",
            "exchange_order_id",
            "fill_id",
            "market_id",
            "token_id",
            "quantity",
            "price",
            "fee_usd",
            "executed_at_ts_ms",
            "fill_event_sequence",
            "cumulative_filled_quantity",
            "cumulative_fill_count",
        }
        executed_at_ts_ms = transport_event.get("executed_at_ts_ms")
        if not (
            set(transport_event) == expected_transport_keys
            and transport_event.get("event_type") == "FILL"
            and transport_event.get("client_order_id") == client_order_id
            and transport_event.get("exchange_order_id")
            == acknowledgement["exchange_order_id"]
            and transport_event.get("fill_id") == fill_id
            and transport_event.get("market_id") == prepared["market_id"]
            and transport_event.get("token_id") == prepared["token_id"]
            and transport_event.get("quantity") == quantity
            and transport_event.get("price") == price
            and transport_event.get("fee_usd") == fee_usd
            and transport_event.get("fill_event_sequence") == fill_event_sequence
            and transport_event.get("cumulative_filled_quantity")
            == cumulative_filled_quantity
            and transport_event.get("cumulative_fill_count")
            == cumulative_fill_count
            and isinstance(executed_at_ts_ms, int)
            and not isinstance(executed_at_ts_ms, bool)
            and prepared["submitted_at_ts_ms"] <= executed_at_ts_ms <= now_ts_ms
        ):
            raise MicroLiveExecutionError("fill transport identity is mismatched")
        if order.get("close_event") is not None:
            self.engage_kill_switch(reason="fill_after_terminal_state", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("fill arrived after authoritative order close")
        expected_fill_event_sequence = len(order["fills"]) + 1
        previous_cumulative_fill_count = (
            int(order["fills"][-1]["cumulative_fill_count"])
            if order["fills"]
            else 0
        )
        existing_filled_quantity = sum(
            (Decimal(fill["quantity"]) for fill in order["fills"]),
            Decimal("0"),
        )
        existing_fill = view["fills"].get(fill_id)
        payload = {
            "client_order_id": client_order_id,
            "exchange_order_id": acknowledgement["exchange_order_id"],
            "fill_id": fill_id,
            "market_id": prepared["market_id"],
            "token_id": prepared["token_id"],
            "quantity": str(_positive_decimal(quantity, "fill quantity")),
            "price": str(_positive_decimal(price, "fill price")),
            "fee_usd": str(_nonnegative_decimal(fee_usd, "fill fee")),
            "executed_at_ts_ms": executed_at_ts_ms,
            "fill_event_sequence": fill_event_sequence,
            "cumulative_filled_quantity": str(
                _positive_decimal(
                    cumulative_filled_quantity,
                    "cumulative filled quantity",
                )
            ),
            "cumulative_fill_count": cumulative_fill_count,
            "transport_event_sha256": transport_event_sha256,
            "raw_transport_event_json": raw_json,
        }
        if existing_fill is not None:
            semantic_keys = {
                "client_order_id",
                "exchange_order_id",
                "fill_id",
                "market_id",
                "token_id",
                "quantity",
                "price",
                "fee_usd",
                "executed_at_ts_ms",
                "fill_event_sequence",
                "cumulative_filled_quantity",
                "cumulative_fill_count",
            }
            if any(
                existing_fill.get(key) != payload.get(key)
                for key in semantic_keys
            ):
                self.engage_kill_switch(
                    reason="conflicting_duplicate_fill",
                    now_ts_ms=now_ts_ms,
                )
                raise MicroLiveExecutionError("conflicting duplicate fill failed closed")
            return {"status": "IDEMPOTENT_FILL_REPLAY", "fill_id": fill_id}
        if not (
            isinstance(fill_event_sequence, int)
            and not isinstance(fill_event_sequence, bool)
            and fill_event_sequence == expected_fill_event_sequence
            and isinstance(cumulative_fill_count, int)
            and not isinstance(cumulative_fill_count, bool)
            and cumulative_fill_count > previous_cumulative_fill_count
            and cumulative_fill_count >= fill_event_sequence
        ):
            raise MicroLiveExecutionError(
                "authoritative cumulative fill sequence is invalid"
            )
        if len(order["fills"]) >= MAXIMUM_FILL_DELIVERY_EVENTS_PER_ORDER:
            self.engage_kill_switch(
                reason="maximum_fill_events_per_order_exceeded",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError(
                "fill event count exceeds the bounded lifecycle contract"
            )
        fill_qty = Decimal(payload["quantity"])
        fill_price = Decimal(payload["price"])
        fee = Decimal(payload["fee_usd"])
        if Decimal(payload["cumulative_filled_quantity"]) != (
            existing_filled_quantity + fill_qty
        ):
            raise MicroLiveExecutionError(
                "authoritative cumulative filled quantity is mismatched"
            )
        if fee > fill_qty * FROZEN_EXECUTION_FEE_PER_UNIT_USD:
            self.engage_kill_switch(
                reason="fill_fee_above_frozen_contract",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("fill fee exceeds frozen execution contract")
        if fill_price > Decimal(order["prepared"]["limit_price"]):
            self.engage_kill_switch(reason="fill_price_above_limit", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("buy fill price exceeds authorized limit")
        if fill_qty > order["remaining_quantity"]:
            self.engage_kill_switch(reason="fill_quantity_exceeded", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("fill quantity exceeds remaining order")
        if fill_qty * fill_price + fee > view["cash_usd"]:
            self.engage_kill_switch(reason="fill_cash_cap_exceeded", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("fill would make authorized cash negative")
        self._append_event("FILL_RECORDED", payload, event_ts_ms=now_ts_ms)
        snapshot = self.reconciliation_snapshot()
        return {"status": "FILL_RECORDED", "fill_id": fill_id, "snapshot": snapshot}

    @_durable_entry
    def reconcile_authoritative_fill_cursor(
        self,
        *,
        client_order_id: str,
        now_ts_ms: int,
    ) -> dict[str, Any]:
        """Ingest server-owned cumulative fills before accepting finality."""

        self._require_authorization_integrity(
            now_ts_ms=now_ts_ms,
            reconciliation_only=True,
        )
        now_ts_ms = self._require_risk_entry_clock(
            now_ts_ms,
            operation="authoritative_fill_cursor_reconciliation",
        )
        try:
            return self._reconcile_authoritative_fill_cursor_locked(
                client_order_id=client_order_id,
                now_ts_ms=now_ts_ms,
            )
        except MicroLiveExecutionError:
            self.engage_kill_switch(
                reason="authoritative_fill_cursor_reconciliation_failed",
                now_ts_ms=now_ts_ms,
            )
            raise

    def _reconcile_authoritative_fill_cursor_locked(
        self,
        *,
        client_order_id: str,
        now_ts_ms: int,
    ) -> dict[str, Any]:
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if not (
            order is not None
            and order.get("acknowledgement") is not None
            and order.get("settlement") is None
        ):
            raise MicroLiveExecutionError(
                "authoritative fill cursor requires one acknowledged order"
            )
        prepared = dict(order["prepared"])
        acknowledgement = dict(order["acknowledgement"])
        request = {
            "authorization_id": self._authorization.authorization_id,
            "execution_service_binding_sha256": (
                self._authorization.execution_service_binding_sha256
            ),
            "client_order_id": client_order_id,
            "business_key": prepared["business_key"],
            "exchange_order_id": acknowledgement["exchange_order_id"],
            "market_id": prepared["market_id"],
            "token_id": prepared["token_id"],
            "request_started_at_ts_ms": now_ts_ms,
        }
        request_identity = {
            key: value
            for key, value in request.items()
            if key
            not in {
                "execution_service_binding_sha256",
                "request_started_at_ts_ms",
            }
        }
        raw_cursor = self._bounded_transport_call(
            operation="read_order_fill_cursor",
            request=request,
        )
        cursor, raw_cursor_json, cursor_sha256 = _raw_json_object(
            raw_cursor,
            "authoritative order fill cursor",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        expected_keys = {
            "schema_version",
            "authorization_id",
            "execution_service_binding_sha256",
            "request_started_at_ts_ms",
            "event_type",
            "client_order_id",
            "exchange_order_id",
            "market_id",
            "token_id",
            "status",
            "observed_at_ts_ms",
            "effective_at_ts_ms",
            "cumulative_filled_quantity",
            "final_fill_event_sequence",
            "final_fill_count",
            "final_fill_watermark",
            "fill_delivery_complete",
            "fill_events",
            "cursor_payload_sha256",
            "signature_algorithm",
            "signature_hex",
        }
        status = cursor.get("status")
        observed_at_ts_ms = cursor.get("observed_at_ts_ms")
        effective_at_ts_ms = cursor.get("effective_at_ts_ms")
        fill_events = cursor.get("fill_events")
        unsigned_cursor = {
            key: value
            for key, value in cursor.items()
            if key
            not in {
                "cursor_payload_sha256",
                "signature_algorithm",
                "signature_hex",
            }
        }
        cursor_payload_sha256 = canonical_json_sha256(unsigned_cursor)
        signed_cursor_core = {
            **unsigned_cursor,
            "cursor_payload_sha256": cursor_payload_sha256,
        }
        if not (
            set(cursor) == expected_keys
            and cursor.get("schema_version") == EXECUTION_CURSOR_SCHEMA_VERSION
            and cursor.get("authorization_id")
            == self._authorization.authorization_id
            and cursor.get("execution_service_binding_sha256")
            == self._authorization.execution_service_binding_sha256
            and cursor.get("request_started_at_ts_ms") == now_ts_ms
            and cursor.get("cursor_payload_sha256") == cursor_payload_sha256
            and _verify_signed_risk_domain_receipt(
                cursor,
                expected_core=signed_cursor_core,
                public_key_modulus_hex=(
                    self._authorization.execution_public_key_modulus_hex
                ),
                public_key_exponent=(
                    self._authorization.execution_public_key_exponent
                ),
            )
            and cursor.get("event_type") == "ORDER_FILL_CURSOR"
            and cursor.get("client_order_id") == client_order_id
            and cursor.get("exchange_order_id")
            == acknowledgement["exchange_order_id"]
            and cursor.get("market_id") == prepared["market_id"]
            and cursor.get("token_id") == prepared["token_id"]
            and status in {"OPEN", "FILLED", "CANCELED", "EXPIRED"}
            and isinstance(observed_at_ts_ms, int)
            and not isinstance(observed_at_ts_ms, bool)
            and isinstance(fill_events, list)
            and len(fill_events) <= MAXIMUM_FILL_DELIVERY_EVENTS_PER_ORDER
        ):
            raise MicroLiveExecutionError(
                "authoritative order fill cursor identity is invalid"
            )
        (
            request_completed_at_ts_ms,
            raw_trusted_time_receipt_json,
            trusted_time_receipt_sha256,
        ) = self._verified_trusted_completion(
            request_started_at_ts_ms=now_ts_ms,
            operation="read_order_fill_cursor",
            response_sha256=cursor_sha256,
        )
        maximum_clock_skew_ms = self._authorization.execution_maximum_clock_skew_ms
        if not (
            prepared["submitted_at_ts_ms"] <= observed_at_ts_ms
            and now_ts_ms - maximum_clock_skew_ms
            <= observed_at_ts_ms
            <= request_completed_at_ts_ms + maximum_clock_skew_ms
        ):
            raise MicroLiveExecutionError(
                "authoritative fill cursor timestamp is outside trusted completion bounds"
            )
        for fill_event in fill_events:
            if not isinstance(fill_event, Mapping):
                raise MicroLiveExecutionError(
                    "authoritative fill cursor event is invalid"
                )
            fill = dict(fill_event)
            self._record_fill(
                client_order_id=client_order_id,
                fill_id=str(fill.get("fill_id") or ""),
                now_ts_ms=request_completed_at_ts_ms,
                quantity=str(fill.get("quantity") or ""),
                price=str(fill.get("price") or ""),
                fee_usd=str(fill.get("fee_usd") or ""),
                fill_event_sequence=fill.get("fill_event_sequence"),
                cumulative_filled_quantity=str(
                    fill.get("cumulative_filled_quantity") or ""
                ),
                cumulative_fill_count=fill.get("cumulative_fill_count"),
                raw_transport_event=_canonical_json_bytes(fill),
            )
        refreshed = self._reconcile_view()["orders"][client_order_id]
        cumulative_filled_quantity = str(refreshed["filled_quantity"])
        final_fill_event_sequence = len(refreshed["fills"])
        final_fill_count = (
            int(refreshed["fills"][-1]["cumulative_fill_count"])
            if refreshed["fills"]
            else 0
        )
        if not (
            cursor.get("cumulative_filled_quantity")
            == cumulative_filled_quantity
            and cursor.get("final_fill_event_sequence")
            == final_fill_event_sequence
            and cursor.get("final_fill_count") == final_fill_count
        ):
            raise MicroLiveExecutionError(
                "authoritative fill cursor does not reconcile with ingested fills"
            )
        if status == "OPEN":
            if not (
                effective_at_ts_ms is None
                and cursor.get("fill_delivery_complete") is False
                and cursor.get("final_fill_watermark") is None
            ):
                raise MicroLiveExecutionError(
                    "open authoritative fill cursor claims terminal finality"
                )
            return {
                "status": "ORDER_FILL_CURSOR_OPEN",
                "client_order_id": client_order_id,
                "order_closed": False,
                "cursor_request_sha256": canonical_json_sha256(request_identity),
                "authenticated_cursor_request_sha256": canonical_json_sha256(
                    request
                ),
                "cursor_response_sha256": cursor_sha256,
                "raw_cursor_response_json": raw_cursor_json,
                "request_started_at_ts_ms": now_ts_ms,
                "request_completed_at_ts_ms": request_completed_at_ts_ms,
                "trusted_time_receipt_sha256": trusted_time_receipt_sha256,
                "raw_trusted_time_receipt_json": raw_trusted_time_receipt_json,
            }
        if not (
            isinstance(effective_at_ts_ms, int)
            and not isinstance(effective_at_ts_ms, bool)
            and prepared["submitted_at_ts_ms"]
            <= effective_at_ts_ms
            <= observed_at_ts_ms
            and cursor.get("fill_delivery_complete") is True
            and _is_sha256(cursor.get("final_fill_watermark"))
            and cursor.get("final_fill_watermark")
            == _expected_final_fill_watermark(cursor)
            and (
                status != "FILLED"
                or Decimal(cumulative_filled_quantity)
                == Decimal(prepared["quantity"])
            )
        ):
            raise MicroLiveExecutionError(
                "authoritative terminal fill cursor is incomplete"
            )
        return self._record_order_closed(
            client_order_id=client_order_id,
            status=str(status),
            now_ts_ms=request_completed_at_ts_ms,
            request_started_at_ts_ms=now_ts_ms,
            request_completed_at_ts_ms=request_completed_at_ts_ms,
            trusted_time_receipt_sha256=trusted_time_receipt_sha256,
            raw_trusted_time_receipt_json=raw_trusted_time_receipt_json,
            raw_transport_event=raw_cursor,
        )

    def _record_order_closed(
        self,
        *,
        client_order_id: str,
        status: str,
        now_ts_ms: int,
        request_started_at_ts_ms: int,
        request_completed_at_ts_ms: int,
        trusted_time_receipt_sha256: str,
        raw_trusted_time_receipt_json: str,
        raw_transport_event: bytes,
    ) -> dict[str, Any]:
        if status not in {"FILLED", "CANCELED", "EXPIRED"}:
            raise MicroLiveExecutionError("order close status is invalid")
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if order is None or order.get("acknowledgement") is None:
            raise MicroLiveExecutionError("order close has no acknowledged order")
        if order.get("settlement") is not None:
            self.engage_kill_switch(reason="order_close_after_settlement", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("order close arrived after settlement")
        prepared = dict(order["prepared"])
        acknowledgement = dict(order["acknowledgement"])
        transport_event, raw_json, transport_event_sha256 = _raw_json_object(
            raw_transport_event,
            "order close transport event",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        expected_transport_keys = {
            "schema_version",
            "authorization_id",
            "execution_service_binding_sha256",
            "request_started_at_ts_ms",
            "event_type",
            "client_order_id",
            "exchange_order_id",
            "market_id",
            "token_id",
            "status",
            "observed_at_ts_ms",
            "effective_at_ts_ms",
            "cumulative_filled_quantity",
            "final_fill_event_sequence",
            "final_fill_count",
            "final_fill_watermark",
            "fill_delivery_complete",
            "fill_events",
            "cursor_payload_sha256",
            "signature_algorithm",
            "signature_hex",
        }
        effective_at_ts_ms = transport_event.get("effective_at_ts_ms")
        observed_at_ts_ms = transport_event.get("observed_at_ts_ms")
        cumulative_filled_quantity = str(
            sum(
                (Decimal(fill["quantity"]) for fill in order["fills"]),
                Decimal("0"),
            )
        )
        final_fill_event_sequence = len(order["fills"])
        final_fill_count = (
            int(order["fills"][-1]["cumulative_fill_count"])
            if order["fills"]
            else 0
        )
        final_fill_watermark = transport_event.get("final_fill_watermark")
        trusted_time_receipt = _stored_raw_json_object(
            raw_trusted_time_receipt_json,
            trusted_time_receipt_sha256,
            "stored trusted completion time receipt",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        if not (
            set(transport_event) == expected_transport_keys
            and transport_event.get("schema_version")
            == EXECUTION_CURSOR_SCHEMA_VERSION
            and transport_event.get("authorization_id")
            == self._authorization.authorization_id
            and transport_event.get("execution_service_binding_sha256")
            == self._authorization.execution_service_binding_sha256
            and transport_event.get("request_started_at_ts_ms")
            == request_started_at_ts_ms
            and transport_event.get("event_type") == "ORDER_FILL_CURSOR"
            and transport_event.get("client_order_id") == client_order_id
            and transport_event.get("exchange_order_id")
            == acknowledgement["exchange_order_id"]
            and transport_event.get("market_id") == prepared["market_id"]
            and transport_event.get("token_id") == prepared["token_id"]
            and transport_event.get("status") == status
            and transport_event.get("cumulative_filled_quantity")
            == cumulative_filled_quantity
            and transport_event.get("final_fill_event_sequence")
            == final_fill_event_sequence
            and transport_event.get("final_fill_count") == final_fill_count
            and _is_sha256(final_fill_watermark)
            and final_fill_watermark
            == _expected_final_fill_watermark(transport_event)
            and transport_event.get("fill_delivery_complete") is True
            and isinstance(transport_event.get("fill_events"), list)
            and len(transport_event["fill_events"])
            <= MAXIMUM_FILL_DELIVERY_EVENTS_PER_ORDER
            and transport_event["fill_events"]
            == [
                json.loads(fill["raw_transport_event_json"])
                for fill in order["fills"]
            ]
            and isinstance(effective_at_ts_ms, int)
            and not isinstance(effective_at_ts_ms, bool)
            and isinstance(observed_at_ts_ms, int)
            and not isinstance(observed_at_ts_ms, bool)
            and prepared["submitted_at_ts_ms"]
            <= effective_at_ts_ms
            <= observed_at_ts_ms
            <= request_completed_at_ts_ms
            + self._authorization.execution_maximum_clock_skew_ms
            and request_started_at_ts_ms <= request_completed_at_ts_ms <= now_ts_ms
            and _trusted_time_receipt_is_valid(
                trusted_time_receipt,
                self._authorization,
                request_started_at_ts_ms=request_started_at_ts_ms,
                operation="read_order_fill_cursor",
                response_sha256=transport_event_sha256,
                request_completed_at_ts_ms=request_completed_at_ts_ms,
            )
            and (
                status != "FILLED"
                or Decimal(cumulative_filled_quantity)
                == Decimal(prepared["quantity"])
            )
        ):
            raise MicroLiveExecutionError("order close transport identity is mismatched")
        payload = {
            "client_order_id": client_order_id,
            "exchange_order_id": acknowledgement["exchange_order_id"],
            "market_id": prepared["market_id"],
            "token_id": prepared["token_id"],
            "effective_at_ts_ms": effective_at_ts_ms,
            "cumulative_filled_quantity": cumulative_filled_quantity,
            "final_fill_event_sequence": final_fill_event_sequence,
            "final_fill_count": final_fill_count,
            "final_fill_watermark": final_fill_watermark,
            "fill_delivery_complete": True,
            "request_started_at_ts_ms": request_started_at_ts_ms,
            "request_completed_at_ts_ms": request_completed_at_ts_ms,
            "trusted_time_receipt_sha256": trusted_time_receipt_sha256,
            "raw_trusted_time_receipt_json": raw_trusted_time_receipt_json,
            "transport_event_sha256": transport_event_sha256,
            "raw_transport_event_json": raw_json,
        }
        existing = order.get("close_event")
        if existing is not None:
            if existing != payload or order.get("closed_status") != status:
                self.engage_kill_switch(
                    reason="conflicting_order_close",
                    now_ts_ms=now_ts_ms,
                )
                raise MicroLiveExecutionError("conflicting order close status")
            return {"status": "IDEMPOTENT_ORDER_CLOSE", "client_order_id": client_order_id}
        event_type = {
            "FILLED": "ORDER_FILLED",
            "CANCELED": "ORDER_CANCELED",
            "EXPIRED": "ORDER_EXPIRED",
        }[status]
        self._append_event(
            event_type,
            payload,
            event_ts_ms=now_ts_ms,
        )
        return {"status": event_type, "client_order_id": client_order_id}

    @_durable_entry
    def record_settlement(
        self,
        *,
        client_order_id: str,
        settlement_id: str,
        now_ts_ms: int,
        payout_per_token: str,
        raw_official_settlement_event: bytes,
    ) -> dict[str, Any]:
        """Record official settlement; any reconciliation ambiguity kills."""

        self._require_authorization_integrity(
            now_ts_ms=now_ts_ms,
            reconciliation_only=True,
        )
        now_ts_ms = self._require_risk_entry_clock(
            now_ts_ms,
            operation="settlement_observation",
        )
        try:
            return self._record_settlement(
                client_order_id=client_order_id,
                settlement_id=settlement_id,
                now_ts_ms=now_ts_ms,
                payout_per_token=payout_per_token,
                raw_official_settlement_event=raw_official_settlement_event,
            )
        except MicroLiveExecutionError:
            self.engage_kill_switch(
                reason="settlement_reconciliation_failed",
                now_ts_ms=now_ts_ms,
            )
            raise

    def _record_settlement(
        self,
        *,
        client_order_id: str,
        settlement_id: str,
        now_ts_ms: int,
        payout_per_token: str,
        raw_official_settlement_event: bytes,
    ) -> dict[str, Any]:
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if order is None or order.get("acknowledgement") is None:
            raise MicroLiveExecutionError("settlement has no acknowledged order")
        payout = _nonnegative_decimal(payout_per_token, "settlement payout")
        if payout not in {Decimal("0"), Decimal("1")}:
            raise MicroLiveExecutionError("settlement payout must be official binary")
        if not isinstance(settlement_id, str) or not settlement_id:
            raise MicroLiveExecutionError("settlement identity is invalid")
        if order["filled_quantity"] <= 0:
            raise MicroLiveExecutionError("unfilled order cannot settle")
        close_event = order.get("close_event")
        filled_quantity = sum(
            (Decimal(fill["quantity"]) for fill in order["fills"]),
            Decimal("0"),
        )
        final_fill_count = (
            int(order["fills"][-1]["cumulative_fill_count"])
            if order["fills"]
            else 0
        )
        if not (
            isinstance(close_event, Mapping)
            and close_event.get("fill_delivery_complete") is True
            and close_event.get("cumulative_filled_quantity")
            == str(filled_quantity)
            and close_event.get("final_fill_event_sequence")
            == len(order["fills"])
            and close_event.get("final_fill_count") == final_fill_count
        ):
            raise MicroLiveExecutionError(
                "settlement requires authoritative final fill delivery watermark"
            )
        prepared = dict(order["prepared"])
        signal = dict(prepared["signal_payload"])
        official, raw_json, official_settlement_sha256 = _raw_json_object(
            raw_official_settlement_event,
            "official settlement event",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        expected_official_keys = {
            "schema_version",
            "authorization_id",
            "execution_service_binding_sha256",
            "settlement_authority_identity_sha256",
            "event_type",
            "settlement_id",
            "market_id",
            "slug",
            "winning_token_id",
            "payout_per_token",
            "finalized_at_ts_ms",
            "observed_at_ts_ms",
            "finality_status",
            "confirmation_depth",
            "provider_url",
            "provider_parameters",
            "provider_retrieved_at_ts_ms",
            "raw_provider_request_json",
            "raw_provider_request_sha256",
            "raw_provider_response_json",
            "raw_provider_response_sha256",
            "finality_metadata",
            "provider_provenance_sha256",
            "signature_algorithm",
            "signature_hex",
        }
        winning_token_id = official.get("winning_token_id")
        expected_winning_token_id = (
            prepared["token_id"]
            if payout == Decimal("1")
            else (
                signal["down_token_id"]
                if prepared["token_id"] == signal["up_token_id"]
                else signal["up_token_id"]
            )
        )
        finalized_at_ts_ms = official.get("finalized_at_ts_ms")
        observed_at_ts_ms = official.get("observed_at_ts_ms")
        raw_provider_request_json = official.get("raw_provider_request_json")
        raw_provider_response_json = official.get("raw_provider_response_json")
        provider_parameters = official.get("provider_parameters")
        finality_metadata = official.get("finality_metadata")
        raw_provider_request: dict[str, Any] = {}
        raw_provider_response: dict[str, Any] = {}
        raw_provider_request_sha256: str | None = None
        raw_provider_response_sha256: str | None = None
        if isinstance(raw_provider_request_json, str):
            try:
                (
                    raw_provider_request,
                    _,
                    raw_provider_request_sha256,
                ) = _raw_json_object(
                    raw_provider_request_json.encode("utf-8"),
                    "official settlement provider request",
                    maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
                )
            except MicroLiveExecutionError:
                raw_provider_request = {}
        if isinstance(raw_provider_response_json, str):
            try:
                (
                    raw_provider_response,
                    _,
                    raw_provider_response_sha256,
                ) = _raw_json_object(
                    raw_provider_response_json.encode("utf-8"),
                    "official settlement provider response",
                    maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
                )
            except MicroLiveExecutionError:
                raw_provider_response = {}
        provider_provenance = {
            "provider_parameters": provider_parameters,
            "provider_retrieved_at_ts_ms": official.get(
                "provider_retrieved_at_ts_ms"
            ),
            "provider_url": official.get("provider_url"),
            "raw_provider_request_sha256": official.get(
                "raw_provider_request_sha256"
            ),
            "raw_provider_response_sha256": official.get(
                "raw_provider_response_sha256"
            ),
            "finality_metadata": finality_metadata,
        }
        official_core = {
            key: official.get(key)
            for key in expected_official_keys
            if key not in {"signature_algorithm", "signature_hex"}
        }
        signed_official = _verify_signed_risk_domain_receipt(
            official,
            expected_core=official_core,
            public_key_modulus_hex=(
                self._authorization.execution_public_key_modulus_hex
            ),
            public_key_exponent=self._authorization.execution_public_key_exponent,
        )
        (
            request_completed_at_ts_ms,
            raw_trusted_time_receipt_json,
            trusted_time_receipt_sha256,
        ) = self._verified_trusted_completion(
            request_started_at_ts_ms=now_ts_ms,
            operation="official_settlement",
            response_sha256=official_settlement_sha256,
        )
        if not (
            set(official) == expected_official_keys
            and signed_official
            and official.get("schema_version") == SETTLEMENT_RECEIPT_SCHEMA_VERSION
            and official.get("authorization_id")
            == self._authorization.authorization_id
            and official.get("execution_service_binding_sha256")
            == self._authorization.execution_service_binding_sha256
            and official.get("settlement_authority_identity_sha256")
            == self._authorization.execution_settlement_authority_identity_sha256
            and official.get("event_type") == "OFFICIAL_SETTLEMENT"
            and official.get("settlement_id") == settlement_id
            and official.get("market_id") == prepared["market_id"]
            and official.get("slug") == prepared["slug"]
            and winning_token_id == expected_winning_token_id
            and official.get("payout_per_token") == payout_per_token
            and isinstance(finalized_at_ts_ms, int)
            and not isinstance(finalized_at_ts_ms, bool)
            and finalized_at_ts_ms >= int(
                dict(signal["market_identity"])["market_end_ts_ms"]
            )
            and isinstance(observed_at_ts_ms, int)
            and not isinstance(observed_at_ts_ms, bool)
            and finalized_at_ts_ms <= observed_at_ts_ms
            and observed_at_ts_ms
            <= request_completed_at_ts_ms
            + self._authorization.execution_maximum_clock_skew_ms
            and official.get("finality_status") == "FINAL"
            and isinstance(official.get("confirmation_depth"), int)
            and not isinstance(official.get("confirmation_depth"), bool)
            and official["confirmation_depth"] >= 1
            and isinstance(official.get("provider_url"), str)
            and official["provider_url"].startswith("https://")
            and isinstance(provider_parameters, Mapping)
            and bool(provider_parameters)
            and isinstance(official.get("provider_retrieved_at_ts_ms"), int)
            and not isinstance(
                official.get("provider_retrieved_at_ts_ms"), bool
            )
            and finalized_at_ts_ms
            <= official["provider_retrieved_at_ts_ms"]
            <= observed_at_ts_ms
            and isinstance(raw_provider_request_json, str)
            and bool(raw_provider_request_json)
            and _is_sha256(official.get("raw_provider_request_sha256"))
            and raw_provider_request_sha256
            == official["raw_provider_request_sha256"]
            and set(raw_provider_request) == {"method", "parameters", "url"}
            and raw_provider_request["method"] == "GET"
            and raw_provider_request["url"] == official["provider_url"]
            and raw_provider_request["parameters"] == provider_parameters
            and isinstance(raw_provider_response_json, str)
            and bool(raw_provider_response_json)
            and _is_sha256(official.get("raw_provider_response_sha256"))
            and raw_provider_response_sha256
            == official["raw_provider_response_sha256"]
            and raw_provider_response.get("condition_id") == prepared["market_id"]
            and raw_provider_response.get("confirmation_depth")
            == official["confirmation_depth"]
            and raw_provider_response.get("finality_status")
            == official["finality_status"]
            and raw_provider_response.get("settlement_id") == settlement_id
            and raw_provider_response.get("winning_token_id") == winning_token_id
            and isinstance(finality_metadata, Mapping)
            and set(finality_metadata)
            == {
                "confirmation_depth",
                "finality_policy",
                "source_block_hash",
                "source_block_number",
            }
            and finality_metadata.get("confirmation_depth")
            == official["confirmation_depth"]
            and isinstance(finality_metadata.get("finality_policy"), str)
            and bool(finality_metadata["finality_policy"])
            and _CONDITION_ID.fullmatch(
                str(finality_metadata.get("source_block_hash"))
            )
            is not None
            and isinstance(finality_metadata.get("source_block_number"), int)
            and not isinstance(finality_metadata.get("source_block_number"), bool)
            and finality_metadata["source_block_number"] >= 0
            and _is_sha256(official.get("provider_provenance_sha256"))
            and official["provider_provenance_sha256"]
            == canonical_json_sha256(provider_provenance)
        ):
            raise MicroLiveExecutionError("official settlement identity is mismatched")
        payload = {
            "client_order_id": client_order_id,
            "settlement_id": settlement_id,
            "market_id": prepared["market_id"],
            "slug": prepared["slug"],
            "token_id": prepared["token_id"],
            "winning_token_id": winning_token_id,
            "payout_per_token": str(payout),
            "finalized_at_ts_ms": finalized_at_ts_ms,
            "observed_at_ts_ms": observed_at_ts_ms,
            "settlement_authority_identity_sha256": (
                self._authorization.execution_settlement_authority_identity_sha256
            ),
            "finality_status": official["finality_status"],
            "confirmation_depth": official["confirmation_depth"],
            "provider_url": official["provider_url"],
            "provider_parameters": copy.deepcopy(dict(provider_parameters)),
            "provider_retrieved_at_ts_ms": official[
                "provider_retrieved_at_ts_ms"
            ],
            "raw_provider_request_json": raw_provider_request_json,
            "raw_provider_request_sha256": official[
                "raw_provider_request_sha256"
            ],
            "raw_provider_response_json": raw_provider_response_json,
            "raw_provider_response_sha256": official[
                "raw_provider_response_sha256"
            ],
            "finality_metadata": copy.deepcopy(dict(finality_metadata)),
            "provider_provenance_sha256": official[
                "provider_provenance_sha256"
            ],
            "official_settlement_sha256": official_settlement_sha256,
            "raw_official_settlement_json": raw_json,
            "request_started_at_ts_ms": now_ts_ms,
            "request_completed_at_ts_ms": request_completed_at_ts_ms,
            "trusted_time_receipt_sha256": trusted_time_receipt_sha256,
            "raw_trusted_time_receipt_json": raw_trusted_time_receipt_json,
        }
        existing = order.get("settlement")
        settlement_elsewhere = view["settlements"].get(settlement_id)
        if settlement_elsewhere is not None and settlement_elsewhere != payload:
            self.engage_kill_switch(
                reason="settlement_identity_reused",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("settlement identity is reused across orders")
        if existing is not None:
            if existing != payload:
                self.engage_kill_switch(
                    reason="conflicting_duplicate_settlement",
                    now_ts_ms=now_ts_ms,
                )
                raise MicroLiveExecutionError("conflicting duplicate settlement")
            return {"status": "IDEMPOTENT_SETTLEMENT_REPLAY", "settlement_id": settlement_id}
        if order["is_open"]:
            self.engage_kill_switch(
                reason="settlement_while_order_open",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("open order cannot settle before cancellation")
        self._append_event(
            "SETTLEMENT_RECORDED",
            payload,
            event_ts_ms=request_completed_at_ts_ms,
        )
        settled_view = self._reconcile_view()
        if _realized_loss_limit_reached(settled_view, self._authorization):
            self.engage_kill_switch(
                reason="maximum_realized_loss_reached",
                now_ts_ms=now_ts_ms,
            )
        return {
            "status": "SETTLEMENT_RECORDED",
            "settlement_id": settlement_id,
            "snapshot": self.reconciliation_snapshot(),
        }

    def engage_kill_switch(self, *, reason: str, now_ts_ms: int) -> dict[str, Any]:
        """Irreversibly stop submissions and best-effort cancel open orders."""

        if not isinstance(reason, str) or not reason:
            raise MicroLiveExecutionError("kill-switch reason is invalid")
        latest_ts_ms = self._safe_persistence_timestamp()
        clock_invalid = (
            isinstance(now_ts_ms, bool)
            or not isinstance(now_ts_ms, int)
            or now_ts_ms <= 0
        )
        clock_regressed = bool(not clock_invalid and now_ts_ms < latest_ts_ms)
        emergency_reason = (
            "kill_switch_clock_regression"
            if clock_regressed
            else "kill_switch_clock_invalid" if clock_invalid else reason
        )
        emergency_ts_ms = (
            latest_ts_ms
            if clock_invalid or clock_regressed
            else max(now_ts_ms, latest_ts_ms)
        )
        self._persist_emergency_kill(
            reason=emergency_reason,
            event_ts_ms=emergency_ts_ms,
        )
        with self._durable_transaction():
            return self._engage_kill_switch_locked(
                reason=reason,
                now_ts_ms=now_ts_ms,
            )

    def _engage_kill_switch_locked(
        self,
        *,
        reason: str,
        now_ts_ms: int,
    ) -> dict[str, Any]:
        clock_invalid = (
            isinstance(now_ts_ms, bool)
            or not isinstance(now_ts_ms, int)
            or now_ts_ms <= 0
        )
        latest_ts_ms = self._safe_persistence_timestamp()
        clock_regressed = bool(not clock_invalid and now_ts_ms < latest_ts_ms)
        clock_failure = clock_invalid or clock_regressed
        event_ts_ms = latest_ts_ms if clock_failure else max(now_ts_ms, latest_ts_ms)
        effective_reason = (
            "kill_switch_clock_regression"
            if clock_regressed
            else "kill_switch_clock_invalid" if clock_invalid else reason
        )
        view = self._reconcile_view()
        if not view["kill_switch_active"]:
            self._append_event(
                "KILL_SWITCH_ENGAGED",
                {"reason": effective_reason, "engaged_at_ts_ms": event_ts_ms},
                event_ts_ms=event_ts_ms,
            )
            view = self._reconcile_view()
        canceled: list[str] = []
        unknown: list[str] = []
        for order in view["orders"].values():
            iteration_ts_ms = max(
                event_ts_ms,
                self._safe_persistence_timestamp(),
            )
            if not order["is_open"]:
                continue
            if order.get("cancel_unknown") is True and not (
                reason == "explicit_cancel_retry"
                and order.get("cancel_retry_authorized") is True
            ):
                unknown.append(str(order["prepared"]["client_order_id"]))
                continue
            existing_cancel = order.get("cancel_prepared")
            if existing_cancel is None:
                cancel_core = {
                    "authorization_id": self._authorization.authorization_id,
                    "client_order_id": order["prepared"]["client_order_id"],
                    "exchange_order_id": order["acknowledgement"]["exchange_order_id"],
                    "market_id": order["prepared"]["market_id"],
                    "token_id": order["prepared"]["token_id"],
                    "reason": effective_reason,
                    "requested_at_ts_ms": iteration_ts_ms,
                }
                cancel_intent_id = canonical_json_sha256(cancel_core)
                self._append_event(
                    "ORDER_CANCEL_PREPARED",
                    {**cancel_core, "cancel_intent_id": cancel_intent_id},
                    event_ts_ms=iteration_ts_ms,
                )
            else:
                cancel_intent_id = str(existing_cancel["cancel_intent_id"])
                cancel_core = {
                    key: value
                    for key, value in dict(existing_cancel).items()
                    if key != "cancel_intent_id"
                }
            request = {
                **cancel_core,
                "cancel_intent_id": cancel_intent_id,
            }
            try:
                response, _, _ = _raw_json_object(
                    self._bounded_transport_call(
                        operation="cancel_order",
                        request=request,
                    ),
                    "order cancellation response",
                    maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
                )
                response_keys = {
                    "client_order_id",
                    "exchange_order_id",
                    "status",
                }
                if set(response) != response_keys or not (
                    response.get("client_order_id") == request["client_order_id"]
                    and response.get("exchange_order_id") == request["exchange_order_id"]
                    and response.get("status") == "CANCEL_REQUESTED"
                ):
                    raise MicroLiveExecutionError("cancel response contract mismatch")
                close_result = self._reconcile_authoritative_fill_cursor_locked(
                    client_order_id=str(request["client_order_id"]),
                    now_ts_ms=max(
                        iteration_ts_ms,
                        self._safe_persistence_timestamp(),
                    ),
                )
                if close_result["status"] not in {
                    "ORDER_FILLED",
                    "ORDER_CANCELED",
                    "ORDER_EXPIRED",
                    "IDEMPOTENT_ORDER_CLOSE",
                }:
                    raise MicroLiveExecutionError(
                        "cancel did not produce an authoritative terminal fill cursor"
                    )
                canceled.append(str(request["client_order_id"]))
            except Exception as exc:
                self._append_event(
                    "ORDER_CANCEL_UNKNOWN",
                    {
                        "client_order_id": request["client_order_id"],
                        "error_type": exc.__class__.__name__,
                    },
                    event_ts_ms=max(
                        iteration_ts_ms,
                        self._safe_persistence_timestamp(),
                    ),
                )
                unknown.append(str(request["client_order_id"]))
        result = {
            "status": "KILL_SWITCH_ENGAGED",
            "reason": effective_reason,
            "canceled_client_order_ids": canceled,
            "unknown_cancel_client_order_ids": unknown,
        }
        if clock_failure:
            raise MicroLiveExecutionError("kill-switch trusted clock is invalid")
        return result

    def reconciliation_snapshot(self) -> dict[str, Any]:
        """Return a deterministic cash/order/position reconciliation snapshot."""

        view = self._reconcile_view()
        return {
            "authorization_id": self._authorization.authorization_id,
            "event_count": len(self._events),
            "kill_switch_active": view["kill_switch_active"],
            "kill_switch_reason": view["kill_switch_reason"],
            "cash_usd": str(view["cash_usd"]),
            "realized_pnl_usd": str(view["realized_pnl_usd"]),
            "maximum_realized_loss_usd": str(
                self._authorization.maximum_realized_loss_usd
            ),
            "unsettled_maximum_loss_usd": str(
                view["unsettled_maximum_loss_usd"]
            ),
            "loss_budget_consumed_usd": str(view["loss_budget_consumed_usd"]),
            "positions": {key: str(value) for key, value in sorted(view["positions"].items())},
            "submitted_notional_usd": str(view["submitted_notional_usd"]),
            "open_order_count": view["open_order_count"],
            "order_count": len(view["orders"]),
            "fill_count": len(view["fills"]),
            "settlement_count": sum(
                order.get("settlement") is not None for order in view["orders"].values()
            ),
            "reconciled": True,
            "state_sha256": canonical_json_sha256(self.export_state(include_state_sha=False)),
        }

    def export_state(self, *, include_state_sha: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "journal_generation": self._generation,
            "risk_domain_id": self._risk_domain_id,
            "authorization_id": self._authorization.authorization_id,
            "authorization_payload_sha256": self._authorization.authorization_payload_sha256,
            "candidate_bundle_sha256": self._authorization.candidate_bundle_sha256,
            "risk_domain_lease_id": self._authorization.risk_domain_lease_id,
            "risk_domain_authority_binding_sha256": (
                self._authorization.risk_domain_authority_binding_sha256
            ),
            "execution_service_binding_sha256": (
                self._authorization.execution_service_binding_sha256
            ),
            "events": copy.deepcopy(self._events),
        }
        return (
            {**payload, "state_sha256": canonical_json_sha256(payload)}
            if include_state_sha
            else payload
        )

    def export_state_bytes(self) -> bytes:
        """Return deterministic strict JSON bytes for durable persistence."""

        return json.dumps(
            self.export_state(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def restore(
        cls,
        *,
        authorization: VerifiedMicroLiveAuthorization,
        transport: MicroLiveOrderTransport,
        journal: MicroLiveStateJournal,
        raw_state: bytes,
    ) -> MicroLiveExecutor:
        if not authorization_capability_is_verified(authorization):
            raise MicroLiveExecutionError(
                "micro-live restore authorization capability is unverified"
            )
        if journal.__class__ is not AtomicFileMicroLiveStateJournal:
            raise MicroLiveExecutionError(
                "micro-live journal must be the deployment-owned concrete implementation"
            )
        _require_startup_execution_capabilities(
            transport=transport,
            journal=journal,
        )
        authority_binding_sha256 = _risk_domain_authority_binding_sha256(authorization)
        execution_service_binding_sha256 = _execution_service_binding_sha256(
            authorization
        )
        restore_risk_domain_id = canonical_json_sha256(
            {
                "authorization_id": authorization.authorization_id,
                "candidate_bundle_sha256": authorization.candidate_bundle_sha256,
                "risk_domain_lease_id": authorization.risk_domain_lease_id,
                "risk_domain_authority_binding_sha256": authority_binding_sha256,
                "execution_service_binding_sha256": execution_service_binding_sha256,
                "lineage_id": LINEAGE_ID,
            }
        )
        state, _, _ = _raw_json_object(
            raw_state,
            "micro-live state",
            maximum_bytes=MAX_RESTORED_STATE_BYTES,
        )
        if set(state) != {
            "schema_version",
            "journal_generation",
            "risk_domain_id",
            "authorization_id",
            "authorization_payload_sha256",
            "candidate_bundle_sha256",
            "risk_domain_lease_id",
            "risk_domain_authority_binding_sha256",
            "execution_service_binding_sha256",
            "events",
            "state_sha256",
        }:
            raise MicroLiveExecutionError("micro-live state schema is invalid")
        payload = {key: copy.deepcopy(value) for key, value in state.items() if key != "state_sha256"}
        events = payload.get("events")
        generation = payload.get("journal_generation")
        if not (
            payload.get("schema_version") == STATE_SCHEMA_VERSION
            and isinstance(generation, int)
            and not isinstance(generation, bool)
            and generation >= 0
            and payload.get("risk_domain_id") == restore_risk_domain_id
            and payload.get("authorization_id") == authorization.authorization_id
            and payload.get("authorization_payload_sha256")
            == authorization.authorization_payload_sha256
            and payload.get("candidate_bundle_sha256") == authorization.candidate_bundle_sha256
            and payload.get("risk_domain_lease_id")
            == authorization.risk_domain_lease_id
            and payload.get("risk_domain_authority_binding_sha256")
            == authority_binding_sha256
            and payload.get("execution_service_binding_sha256")
            == execution_service_binding_sha256
            and isinstance(events, list)
            and len(events) <= MAX_EVENT_COUNT
            and state.get("state_sha256") == canonical_json_sha256(payload)
        ):
            raise MicroLiveExecutionError("micro-live state identity or SHA-256 mismatch")
        restored = cls.__new__(cls)
        restored._initialize(
            authorization=authorization,
            transport=transport,
            journal=journal,
            events=events,
            generation=int(generation),
            initialize_journal=False,
            restore_token=_RAW_STATE_RESTORE_TOKEN,
        )
        snapshot = journal.snapshot()
        if not (
            journal.durable_single_writer is True
            and snapshot.generation == generation
            and snapshot.raw_state == raw_state
            and snapshot.state_sha256 == hashlib.sha256(raw_state).hexdigest()
        ):
            raise MicroLiveExecutionError(
                "micro-live restore state is not the journal high-water snapshot"
            )
        restored._recover_incomplete_external_side_effects()
        return restored

    def _recover_incomplete_external_side_effects(self) -> None:
        """Complete safety invariants for every fsync-committed crash prefix."""

        with self._durable_transaction():
            view = self._reconcile_view()
            event_ts_ms = (
                int(self._events[-1]["event_ts_ms"])
                if self._events
                else self._authorization.authorized_at_ts_ms
            )
            for order in view["orders"].values():
                if (
                    order["acknowledgement"] is None
                    and order["closed_status"] is None
                    and order["submission_unknown"] is not True
                ):
                    self._append_event(
                        "ORDER_SUBMISSION_UNKNOWN",
                        {
                            "client_order_id": order["prepared"]["client_order_id"],
                            "error_type": "CrashRecoveryIncompleteSubmission",
                        },
                        event_ts_ms=event_ts_ms,
                    )
            view = self._reconcile_view()
            if any(
                order["submission_unknown"] is True
                for order in view["orders"].values()
            ) and not view["kill_switch_active"]:
                self.engage_kill_switch(
                    reason="crash_recovery_incomplete_submission",
                    now_ts_ms=event_ts_ms,
                )
            view = self._reconcile_view()
            if view["kill_switch_active"]:
                for order in view["orders"].values():
                    if not order["is_open"]:
                        continue
                    if order.get("cancel_prepared") is None:
                        cancel_core = {
                            "authorization_id": self._authorization.authorization_id,
                            "client_order_id": order["prepared"]["client_order_id"],
                            "exchange_order_id": order["acknowledgement"][
                                "exchange_order_id"
                            ],
                            "market_id": order["prepared"]["market_id"],
                            "token_id": order["prepared"]["token_id"],
                            "reason": "crash_recovery_killed_open_order",
                            "requested_at_ts_ms": event_ts_ms,
                        }
                        self._append_event(
                            "ORDER_CANCEL_PREPARED",
                            {
                                **cancel_core,
                                "cancel_intent_id": canonical_json_sha256(
                                    cancel_core
                                ),
                            },
                            event_ts_ms=event_ts_ms,
                        )
                    refreshed_order = self._reconcile_view()["orders"][
                        order["prepared"]["client_order_id"]
                    ]
                    if refreshed_order["cancel_unknown"] is not True:
                        self._append_event(
                            "ORDER_CANCEL_UNKNOWN",
                            {
                                "client_order_id": order["prepared"][
                                    "client_order_id"
                                ],
                                "error_type": (
                                    "CrashRecoveryKilledOpenOrderRequiresLookup"
                                ),
                            },
                            event_ts_ms=event_ts_ms,
                        )
            else:
                for order in view["orders"].values():
                    if not (
                        order.get("cancel_prepared") is not None
                        and order["closed_status"] is None
                        and order["cancel_unknown"] is not True
                    ):
                        continue
                    self._append_event(
                        "ORDER_CANCEL_UNKNOWN",
                        {
                            "client_order_id": order["prepared"]["client_order_id"],
                            "error_type": "CrashRecoveryIncompleteCancellation",
                        },
                        event_ts_ms=event_ts_ms,
                    )

    def _validate_clock(
        self,
        now_ts_ms: int,
        operator_heartbeat_ts_ms: int,
        decision_ts_ms: int,
        signal_observed_at_ts_ms: int,
        identity_revalidated_at_ts_ms: int,
    ) -> None:
        values = (
            now_ts_ms,
            operator_heartbeat_ts_ms,
            decision_ts_ms,
            signal_observed_at_ts_ms,
            identity_revalidated_at_ts_ms,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise MicroLiveExecutionError("execution timestamp is invalid")
        if self._events and now_ts_ms < int(self._events[-1]["event_ts_ms"]):
            self.engage_kill_switch(
                reason="event_clock_regression",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("micro-live event clock regressed")
        if now_ts_ms >= self._authorization.expires_at_ts_ms:
            self.engage_kill_switch(reason="authorization_expired", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live authorization expired")
        if not decision_ts_ms <= now_ts_ms or (
            now_ts_ms - decision_ts_ms > self._authorization.maximum_signal_age_ms
        ):
            self.engage_kill_switch(reason="signal_stale", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live signal is stale")
        if not decision_ts_ms <= signal_observed_at_ts_ms <= now_ts_ms or (
            now_ts_ms - signal_observed_at_ts_ms
            > self._authorization.maximum_signal_age_ms
        ):
            self.engage_kill_switch(reason="signal_observation_stale", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live signal observation is stale")
        if not operator_heartbeat_ts_ms <= now_ts_ms or (
            now_ts_ms - operator_heartbeat_ts_ms
            > self._authorization.maximum_operator_heartbeat_age_ms
        ):
            self.engage_kill_switch(reason="operator_heartbeat_stale", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live operator heartbeat is stale")
        if not identity_revalidated_at_ts_ms <= signal_observed_at_ts_ms or (
            now_ts_ms - identity_revalidated_at_ts_ms
            > self._authorization.maximum_signal_age_ms
        ):
            self.engage_kill_switch(reason="market_identity_stale", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live market identity is stale")

    def _validate_submission_response(
        self,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_keys = {
            "client_order_id",
            "exchange_order_id",
            "status",
            "market_id",
            "token_id",
            "accepted_quantity",
            "limit_price",
        }
        if set(response) != expected_keys:
            raise MicroLiveExecutionError("order response schema mismatch")
        if not (
            response.get("client_order_id") == request["client_order_id"]
            and response.get("market_id") == request["market_id"]
            and response.get("token_id") == request["token_id"]
            and response.get("accepted_quantity") == request["quantity"]
            and response.get("limit_price") == request["limit_price"]
            and response.get("status") in {"ACCEPTED", "REJECTED"}
            and isinstance(response.get("exchange_order_id"), str)
            and response["exchange_order_id"]
        ):
            raise MicroLiveExecutionError("order response identity mismatch")
        return dict(response)

    def _validate_submission_fence_response(
        self,
        *,
        prepared: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        expected_keys = {
            "authorization_id",
            "client_order_id",
            "transport_invocation_id",
            "side_effects_fenced",
        }
        if not (
            set(response) == expected_keys
            and response.get("authorization_id")
            == self._authorization.authorization_id
            and response.get("client_order_id") == prepared["client_order_id"]
            and response.get("transport_invocation_id")
            == prepared["transport_invocation_id"]
            and response.get("side_effects_fenced") is True
        ):
            raise MicroLiveExecutionError(
                "submission invocation is not durably fenced"
            )

    def _append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        event_ts_ms: int,
    ) -> None:
        transaction = self._active_journal_transaction
        if transaction is None or self._transaction_depth <= 0:
            raise MicroLiveExecutionError(
                "micro-live event append requires a durable journal transaction"
            )
        if event_type not in _EVENT_TYPES:
            raise MicroLiveExecutionError("micro-live event type is invalid")
        next_event_count = len(self._events) + 1
        _require_event_count(next_event_count)
        if (
            event_type not in _RECOVERY_EVENT_TYPES
            and next_event_count > _routine_event_limit()
        ):
            raise MicroLiveExecutionError(
                "micro-live routine event capacity is exhausted"
            )
        if (
            event_type == "ORDER_PREPARED"
            and self._events
            and self._events[-1]["event_type"] == "SIGNAL_EVALUATED"
            and self._events[-1]["payload"].get("disposition")
            == "EXECUTION_INTENT"
        ):
            pending_audit = self._events.pop()
            try:
                view_before = self._reconcile_view()
            finally:
                self._events.append(pending_audit)
        else:
            view_before = self._reconcile_view()
        if event_type in _RECOVERY_EVENT_TYPES:
            required_before = _required_lifecycle_event_capacity(view_before)
            progress = event_type not in {
                "SIGNAL_REJECTED",
                "ORDER_SUBMISSION_RECONCILIATION_FAILED",
                "ORDER_CANCEL_RECONCILIATION_FAILED",
            }
            required_after_upper_bound = max(
                required_before - (1 if progress else 0),
                0,
            )
            if next_event_count + required_after_upper_bound > MAX_EVENT_COUNT:
                raise MicroLiveExecutionError(
                    "micro-live lifecycle reserve would be exhausted"
                )
        _require_positive_timestamp(event_ts_ms, "event")
        if self._events and event_ts_ms < self._events[-1]["event_ts_ms"]:
            raise MicroLiveExecutionError("micro-live event timestamp regressed")
        core = {
            "sequence": len(self._events) + 1,
            "event_ts_ms": event_ts_ms,
            "previous_event_sha256": (
                self._events[-1]["event_sha256"] if self._events else GENESIS
            ),
            "event_type": event_type,
            "payload": copy.deepcopy(dict(payload)),
        }
        event = {**core, "event_sha256": canonical_json_sha256(core)}
        previous_generation = self._generation
        self._events.append(event)
        self._generation = previous_generation + 1
        try:
            raw_state = self.export_state_bytes()
            required_recovery_events = (
                _required_lifecycle_event_capacity(view_before)
                if event_type == "SIGNAL_EVALUATED"
                and payload.get("disposition") == "EXECUTION_INTENT"
                else _required_lifecycle_event_capacity(self._reconcile_view())
            )
            required_recovery_bytes = (
                required_recovery_events * MAX_SERIALIZED_RECOVERY_EVENT_BYTES
            )
            if len(raw_state) + required_recovery_bytes > MAX_RESTORED_STATE_BYTES:
                raise MicroLiveExecutionError(
                    "micro-live WAL byte reserve cannot cover all remaining "
                    "side-effect recovery events"
                )
            committed_generation = transaction.commit(
                expected_generation=previous_generation,
                raw_state=raw_state,
            )
            if committed_generation != self._generation:
                raise MicroLiveExecutionError(
                    "micro-live journal committed an unexpected generation"
                )
        except Exception:
            self._generation = previous_generation
            self._events.pop()
            raise

    def _require_execution_intent_preparation_capacity(
        self,
        *,
        audit_payload: Mapping[str, Any],
        prepared_payload: Mapping[str, Any],
        event_ts_ms: int,
    ) -> None:
        """Prove WAL count and worst-case byte reserve before committing audit."""

        if len(self._events) + 2 > _routine_event_limit():
            raise MicroLiveExecutionError(
                "micro-live routine event capacity cannot prepare an order"
            )
        prospective_events = copy.deepcopy(self._events)
        for event_type, payload in (
            ("SIGNAL_EVALUATED", audit_payload),
            ("ORDER_PREPARED", prepared_payload),
        ):
            core = {
                "sequence": len(prospective_events) + 1,
                "event_ts_ms": event_ts_ms,
                "previous_event_sha256": (
                    prospective_events[-1]["event_sha256"]
                    if prospective_events
                    else GENESIS
                ),
                "event_type": event_type,
                "payload": copy.deepcopy(dict(payload)),
            }
            prospective_events.append(
                {**core, "event_sha256": canonical_json_sha256(core)}
            )
        original_events = self._events
        original_generation = self._generation
        self._events = prospective_events
        self._generation = original_generation + 2
        try:
            prospective_view = self._reconcile_view()
            raw_state = self.export_state_bytes()
        finally:
            self._events = original_events
            self._generation = original_generation
        required_recovery_bytes = (
            _required_lifecycle_event_capacity(prospective_view)
            * MAX_SERIALIZED_RECOVERY_EVENT_BYTES
        )
        if len(raw_state) + required_recovery_bytes > MAX_RESTORED_STATE_BYTES:
            raise MicroLiveExecutionError(
                "micro-live WAL byte reserve cannot prepare a side effect"
            )

    def _verify_event_chain(self) -> None:
        previous = GENESIS
        previous_ts_ms = 0
        for expected_sequence, event in enumerate(self._events, start=1):
            if set(event) != {
                "sequence",
                "event_ts_ms",
                "previous_event_sha256",
                "event_type",
                "payload",
                "event_sha256",
            }:
                raise MicroLiveExecutionError("micro-live event schema is invalid")
            core = {key: value for key, value in event.items() if key != "event_sha256"}
            if not (
                event["sequence"] == expected_sequence
                and isinstance(event["event_ts_ms"], int)
                and not isinstance(event["event_ts_ms"], bool)
                and event["event_ts_ms"] > 0
                and event["event_ts_ms"] >= previous_ts_ms
                and event["previous_event_sha256"] == previous
                and event["event_type"] in _EVENT_TYPES
                and isinstance(event["payload"], Mapping)
                and event["event_sha256"] == canonical_json_sha256(core)
            ):
                raise MicroLiveExecutionError("micro-live event chain is invalid")
            self._validate_event_payload(
                str(event["event_type"]),
                event["payload"],
                event_ts_ms=int(event["event_ts_ms"]),
            )
            previous = str(event["event_sha256"])
            previous_ts_ms = int(event["event_ts_ms"])

    def _validate_event_payload(
        self,
        event_type: str,
        payload_value: Mapping[str, Any],
        *,
        event_ts_ms: int,
    ) -> None:
        payload = dict(payload_value)
        if event_type == "SIGNAL_REJECTED":
            if set(payload) != {
                "authorization_id",
                "candidate_bundle_sha256",
                "reason",
                "error_type",
            } or not (
                payload.get("authorization_id") == self._authorization.authorization_id
                and payload.get("candidate_bundle_sha256")
                == self._authorization.candidate_bundle_sha256
                and payload.get("reason") == "signal_validation_or_clock_failed"
                and isinstance(payload.get("error_type"), str)
                and payload["error_type"]
            ):
                raise MicroLiveExecutionError("rejected signal audit payload is invalid")
            return
        if event_type == "SIGNAL_EVALUATED":
            expected = {
                "authorization_id",
                "candidate_bundle_sha256",
                "market_id",
                "decision_ts_ms",
                "operator_heartbeat_ts_ms",
                "signal_payload_sha256",
                "signal_payload",
                "raw_signal_payload_sha256",
                "raw_signal_payload_json",
                "market_identity_sha256",
                "market_identity",
                "feature_row_sha256",
                "feature_row",
                "raw_feature_row_sha256",
                "raw_feature_row_json",
                "provider_feature_evidence_graph_sha256",
                "provider_feature_file_sha256",
                "provider_reconstructed_feature_row_sha256",
                "raw_provider_feature_evidence_jsonl",
                "disposition",
                "reason",
                "decision_audit_sha256",
            }
            if set(payload) != expected:
                raise MicroLiveExecutionError("evaluated signal audit schema is invalid")
            signal, feature_row, _ = _validated_stored_signal_and_feature(
                payload=payload,
                expected_candidate_bundle_sha256=(
                    self._authorization.candidate_bundle_sha256
                ),
                runtime=self._authorization.runtime,
            )
            disposition = payload.get("disposition")
            reason = payload.get("reason")
            audit_core = {
                key: value
                for key, value in payload.items()
                if key != "decision_audit_sha256"
            }
            if not (
                payload.get("authorization_id") == self._authorization.authorization_id
                and payload.get("candidate_bundle_sha256")
                == self._authorization.candidate_bundle_sha256
                and payload.get("market_id") == signal["market_id"]
                and payload.get("decision_ts_ms") == signal["decision_ts_ms"]
                and isinstance(payload.get("operator_heartbeat_ts_ms"), int)
                and not isinstance(payload.get("operator_heartbeat_ts_ms"), bool)
                and signal["decision_ts_ms"] <= event_ts_ms
                and event_ts_ms - signal["decision_ts_ms"]
                <= self._authorization.maximum_signal_age_ms
                and payload["operator_heartbeat_ts_ms"] <= event_ts_ms
                and event_ts_ms - payload["operator_heartbeat_ts_ms"]
                <= self._authorization.maximum_operator_heartbeat_age_ms
                and dict(signal["market_identity"])["clob_revalidated_at_ts_ms"]
                <= signal["observed_at_ts_ms"]
                and event_ts_ms
                - dict(signal["market_identity"])["clob_revalidated_at_ts_ms"]
                <= self._authorization.maximum_signal_age_ms
                and payload.get("signal_payload_sha256")
                == canonical_json_sha256(signal)
                and payload.get("market_identity") == signal["market_identity"]
                and payload.get("market_identity_sha256")
                == canonical_json_sha256(dict(signal["market_identity"]))
                and payload.get("feature_row_sha256")
                == canonical_json_sha256(feature_row)
                and disposition
                in {
                    "EXECUTION_INTENT",
                    "BLOCKED_NO_TRADE",
                    "IDEMPOTENT_REPLAY",
                    "REJECTED_CONFLICT",
                }
                and (
                    (disposition == "EXECUTION_INTENT" and reason is None)
                    or (
                        disposition != "EXECUTION_INTENT"
                        and isinstance(reason, str)
                        and reason
                    )
                )
                and payload.get("decision_audit_sha256")
                == canonical_json_sha256(audit_core)
            ):
                raise MicroLiveExecutionError("evaluated signal audit payload is invalid")
            return
        if event_type == "ORDER_PREPARED":
            expected = {
                "authorization_id",
                "authorization_payload_sha256",
                "candidate_bundle_sha256",
                "business_key",
                "market_id",
                "slug",
                "market_family",
                "decision_ts_ms",
                "submitted_at_ts_ms",
                "operator_heartbeat_ts_ms",
                "selected_action",
                "token_id",
                "token_side",
                "limit_price",
                "quantity",
                "notional_usd",
                "maximum_fee_usd",
                "maximum_loss_usd",
                "signal_payload_sha256",
                "signal_payload",
                "raw_signal_payload_sha256",
                "raw_signal_payload_json",
                "market_identity_sha256",
                "market_identity",
                "feature_row_sha256",
                "feature_row",
                "raw_feature_row_sha256",
                "raw_feature_row_json",
                "provider_feature_evidence_graph_sha256",
                "provider_feature_file_sha256",
                "provider_reconstructed_feature_row_sha256",
                "raw_provider_feature_evidence_jsonl",
                "intent_id",
                "client_order_id",
                "transport_invocation_id",
            }
            if set(payload) != expected:
                raise MicroLiveExecutionError("prepared order payload schema is invalid")
            price = _positive_decimal(payload.get("limit_price"), "stored limit price")
            quantity = _positive_decimal(payload.get("quantity"), "stored quantity")
            maximum_fee = _nonnegative_decimal(
                payload.get("maximum_fee_usd"), "stored maximum fee"
            )
            maximum_loss = _positive_decimal(
                payload.get("maximum_loss_usd"), "stored maximum loss"
            )
            if (
                price >= 1
                or quantity != Decimal("1")
                or payload.get("notional_usd") != str(price * quantity)
                or maximum_fee != quantity * FROZEN_EXECUTION_FEE_PER_UNIT_USD
                or maximum_loss != price * quantity + maximum_fee
            ):
                raise MicroLiveExecutionError("prepared order economics are invalid")
            signal, feature_row, _ = _validated_stored_signal_and_feature(
                payload=payload,
                expected_candidate_bundle_sha256=self._authorization.candidate_bundle_sha256,
                runtime=self._authorization.runtime,
            )
            market_id = payload.get("market_id")
            business_key = canonical_json_sha256(
                {
                    "authorization_id": self._authorization.authorization_id,
                    "candidate_bundle_sha256": self._authorization.candidate_bundle_sha256,
                    "market_id": market_id,
                }
            )
            intent_core = {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "authorization_payload_sha256",
                    "submitted_at_ts_ms",
                    "operator_heartbeat_ts_ms",
                    "intent_id",
                    "client_order_id",
                    "transport_invocation_id",
                }
            }
            intent_id = canonical_json_sha256(intent_core)
            transport_invocation_id = canonical_json_sha256(
                {
                    "authorization_id": self._authorization.authorization_id,
                    "client_order_id": intent_id,
                    "operation": "submit_order",
                    "invocation_number": 1,
                }
            )
            if not (
                payload.get("authorization_id") == self._authorization.authorization_id
                and payload.get("authorization_payload_sha256")
                == self._authorization.authorization_payload_sha256
                and payload.get("candidate_bundle_sha256")
                == self._authorization.candidate_bundle_sha256
                and isinstance(market_id, str)
                and market_id
                and isinstance(payload.get("slug"), str)
                and _SLUG.fullmatch(payload["slug"]) is not None
                and payload.get("market_family") in self._authorization.market_allowlist
                and isinstance(payload.get("decision_ts_ms"), int)
                and not isinstance(payload.get("decision_ts_ms"), bool)
                and payload["decision_ts_ms"] > 0
                and payload.get("selected_action") in self._authorization.allowed_actions
                and payload.get("selected_action") == signal["selected_action"]
                and payload.get("market_id") == signal["market_id"]
                and payload.get("slug") == signal["slug"]
                and payload.get("market_family") == signal["market_family"]
                and payload.get("decision_ts_ms") == signal["decision_ts_ms"]
                and payload.get("submitted_at_ts_ms") == event_ts_ms
                and isinstance(payload.get("operator_heartbeat_ts_ms"), int)
                and not isinstance(payload.get("operator_heartbeat_ts_ms"), bool)
                and self._authorization.authorized_at_ts_ms
                <= event_ts_ms
                < self._authorization.expires_at_ts_ms
                and signal["decision_ts_ms"] <= event_ts_ms
                and event_ts_ms - signal["decision_ts_ms"]
                <= self._authorization.maximum_signal_age_ms
                and payload["operator_heartbeat_ts_ms"] <= event_ts_ms
                and event_ts_ms - payload["operator_heartbeat_ts_ms"]
                <= self._authorization.maximum_operator_heartbeat_age_ms
                and isinstance(payload.get("token_id"), str)
                and _TOKEN_ID.fullmatch(payload["token_id"]) is not None
                and payload.get("token_side")
                == ("UP" if payload.get("selected_action") == "BUY_UP_HOLD" else "DOWN")
                and payload.get("token_id")
                == signal[f"{str(payload.get('token_side')).lower()}_token_id"]
                and payload.get("limit_price")
                == dict(signal["executable_asks"])[payload["token_side"]]
                and payload.get("signal_payload_sha256") == canonical_json_sha256(signal)
                and payload.get("market_identity") == signal["market_identity"]
                and payload.get("market_identity_sha256")
                == canonical_json_sha256(dict(signal["market_identity"]))
                and payload.get("feature_row_sha256")
                == canonical_json_sha256(feature_row)
                and payload.get("business_key") == business_key
                and payload.get("intent_id") == intent_id
                and payload.get("client_order_id") == intent_id
                and payload.get("transport_invocation_id")
                == transport_invocation_id
            ):
                raise MicroLiveExecutionError("prepared order identity is invalid")
            return
        if event_type == "ORDER_CANCEL_PREPARED":
            expected = {
                "authorization_id",
                "client_order_id",
                "exchange_order_id",
                "market_id",
                "token_id",
                "reason",
                "requested_at_ts_ms",
                "cancel_intent_id",
            }
            if set(payload) != expected:
                raise MicroLiveExecutionError("prepared cancellation schema is invalid")
            core = {
                key: value for key, value in payload.items() if key != "cancel_intent_id"
            }
            if not (
                payload.get("authorization_id") == self._authorization.authorization_id
                and _is_sha256(payload.get("client_order_id"))
                and isinstance(payload.get("exchange_order_id"), str)
                and payload["exchange_order_id"]
                and isinstance(payload.get("market_id"), str)
                and _CONDITION_ID.fullmatch(payload["market_id"]) is not None
                and isinstance(payload.get("token_id"), str)
                and _TOKEN_ID.fullmatch(payload["token_id"]) is not None
                and isinstance(payload.get("reason"), str)
                and payload["reason"]
                and payload.get("requested_at_ts_ms") == event_ts_ms
                and payload.get("cancel_intent_id") == canonical_json_sha256(core)
            ):
                raise MicroLiveExecutionError("prepared cancellation payload is invalid")
            return
        if event_type in {"ORDER_ACKNOWLEDGED", "ORDER_REJECTED"}:
            response_keys = {
                "client_order_id",
                "exchange_order_id",
                "status",
                "market_id",
                "token_id",
                "accepted_quantity",
                "limit_price",
            }
            if set(payload) != {
                *response_keys,
                "transport_event_sha256",
                "raw_transport_event_json",
            }:
                raise MicroLiveExecutionError("order disposition payload is invalid")
            response = _stored_raw_json_object(
                payload.get("raw_transport_event_json"),
                payload.get("transport_event_sha256"),
                "stored order disposition transport response",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            accepted_quantity = _positive_decimal(
                payload.get("accepted_quantity"), "stored accepted quantity"
            )
            limit_price = _positive_decimal(
                payload.get("limit_price"), "stored disposition limit price"
            )
            if not (
                payload.get("status")
                == ("ACCEPTED" if event_type == "ORDER_ACKNOWLEDGED" else "REJECTED")
                and _is_sha256(payload.get("client_order_id"))
                and isinstance(payload.get("exchange_order_id"), str)
                and payload["exchange_order_id"]
                and isinstance(payload.get("market_id"), str)
                and _CONDITION_ID.fullmatch(payload["market_id"]) is not None
                and isinstance(payload.get("token_id"), str)
                and _TOKEN_ID.fullmatch(payload["token_id"]) is not None
                and accepted_quantity == Decimal("1")
                and limit_price < 1
                and set(response) == response_keys
                and all(response.get(key) == payload.get(key) for key in response_keys)
            ):
                raise MicroLiveExecutionError("order disposition payload is invalid")
            return
        if event_type == "ORDER_SUBMISSION_RECONCILED":
            response_keys = {
                "client_order_id",
                "exchange_order_id",
                "status",
                "market_id",
                "token_id",
                "accepted_quantity",
                "limit_price",
            }
            if set(payload) != {
                *response_keys,
                "lookup_response_sha256",
                "raw_lookup_response_json",
                "transport_invocation_id",
                "fence_response_sha256",
                "raw_fence_response_json",
            }:
                raise MicroLiveExecutionError(
                    "submission reconciliation payload is invalid"
                )
            response = _stored_raw_json_object(
                payload.get("raw_lookup_response_json"),
                payload.get("lookup_response_sha256"),
                "stored submission lookup response",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            fence_response = _stored_raw_json_object(
                payload.get("raw_fence_response_json"),
                payload.get("fence_response_sha256"),
                "stored submission invocation fence response",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            accepted_quantity = _positive_decimal(
                response.get("accepted_quantity"),
                "reconciled accepted quantity",
            )
            limit_price = _positive_decimal(
                response.get("limit_price"),
                "reconciled limit price",
            )
            if not (
                response.get("status") in {"ACCEPTED", "REJECTED"}
                and _is_sha256(response.get("client_order_id"))
                and isinstance(response.get("exchange_order_id"), str)
                and response["exchange_order_id"]
                and isinstance(response.get("market_id"), str)
                and _CONDITION_ID.fullmatch(response["market_id"]) is not None
                and isinstance(response.get("token_id"), str)
                and _TOKEN_ID.fullmatch(response["token_id"]) is not None
                and accepted_quantity == Decimal("1")
                and limit_price < 1
                and set(response) == response_keys
                and all(response.get(key) == payload.get(key) for key in response_keys)
                and payload.get("transport_invocation_id")
                == canonical_json_sha256(
                    {
                        "authorization_id": self._authorization.authorization_id,
                        "client_order_id": response["client_order_id"],
                        "operation": "submit_order",
                        "invocation_number": 1,
                    }
                )
                and set(fence_response)
                == {
                    "authorization_id",
                    "client_order_id",
                    "transport_invocation_id",
                    "side_effects_fenced",
                }
                and fence_response.get("authorization_id")
                == self._authorization.authorization_id
                and fence_response.get("client_order_id")
                == response["client_order_id"]
                and fence_response.get("transport_invocation_id")
                == payload.get("transport_invocation_id")
                and fence_response.get("side_effects_fenced") is True
            ):
                raise MicroLiveExecutionError(
                    "submission reconciliation payload is invalid"
                )
            return
        if event_type == "ORDER_SUBMISSION_RECONCILIATION_FAILED":
            if set(payload) != {
                "client_order_id",
                "lookup_request_sha256",
                "lookup_response_sha256",
                "raw_lookup_response_json",
                "fence_request_sha256",
                "fence_response_sha256",
                "raw_fence_response_json",
                "error_type",
            } or not (
                _is_sha256(payload.get("client_order_id"))
                and _is_sha256(payload.get("lookup_request_sha256"))
                and (
                    payload.get("lookup_response_sha256") is None
                    or _is_sha256(payload.get("lookup_response_sha256"))
                )
                and (
                    payload.get("raw_lookup_response_json") is None
                    or isinstance(payload.get("raw_lookup_response_json"), str)
                )
                and (
                    payload.get("fence_request_sha256") is None
                    or _is_sha256(payload.get("fence_request_sha256"))
                )
                and (
                    payload.get("fence_response_sha256") is None
                    or _is_sha256(payload.get("fence_response_sha256"))
                )
                and (
                    payload.get("raw_fence_response_json") is None
                    or isinstance(payload.get("raw_fence_response_json"), str)
                )
                and isinstance(payload.get("error_type"), str)
                and payload["error_type"]
            ):
                raise MicroLiveExecutionError(
                    "failed submission reconciliation audit is invalid"
                )
            lookup_pair = (
                payload.get("lookup_response_sha256"),
                payload.get("raw_lookup_response_json"),
            )
            fence_pair = (
                payload.get("fence_response_sha256"),
                payload.get("raw_fence_response_json"),
            )
            if (lookup_pair[0] is None) != (lookup_pair[1] is None) or (
                fence_pair[0] is None
            ) != (fence_pair[1] is None):
                raise MicroLiveExecutionError(
                    "failed submission reconciliation raw evidence is incomplete"
                )
            if lookup_pair[0] is not None:
                _stored_raw_json_object(
                    lookup_pair[1],
                    lookup_pair[0],
                    "stored failed submission lookup response",
                    maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
                )
            if fence_pair[0] is not None:
                if payload.get("fence_request_sha256") is None:
                    raise MicroLiveExecutionError(
                        "failed submission fence request is absent"
                    )
                _stored_raw_json_object(
                    fence_pair[1],
                    fence_pair[0],
                    "stored failed submission fence response",
                    maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
                )
            return
        if event_type == "ORDER_CANCEL_RECONCILED":
            response_keys = {
                "client_order_id",
                "exchange_order_id",
                "status",
                "market_id",
                "token_id",
                "accepted_quantity",
                "limit_price",
                "observed_at_ts_ms",
                "effective_at_ts_ms",
                "cumulative_filled_quantity",
                "final_fill_event_sequence",
                "final_fill_count",
                "final_fill_watermark",
                "fill_delivery_complete",
            }
            if set(payload) != {
                *response_keys,
                "lookup_response_sha256",
                "raw_lookup_response_json",
            }:
                raise MicroLiveExecutionError(
                    "cancel reconciliation payload is invalid"
                )
            response = _stored_raw_json_object(
                payload.get("raw_lookup_response_json"),
                payload.get("lookup_response_sha256"),
                "stored cancel lookup response",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            if not (
                set(response) == response_keys
                and all(response.get(key) == payload.get(key) for key in response_keys)
                and payload.get("status") in {"CANCELED", "EXPIRED"}
                and isinstance(payload.get("effective_at_ts_ms"), int)
                and not isinstance(payload.get("effective_at_ts_ms"), bool)
                and isinstance(payload.get("observed_at_ts_ms"), int)
                and not isinstance(payload.get("observed_at_ts_ms"), bool)
                and 0
                < payload["effective_at_ts_ms"]
                <= payload["observed_at_ts_ms"]
                <= event_ts_ms
            ):
                raise MicroLiveExecutionError(
                    "cancel reconciliation payload is invalid"
                )
            return
        if event_type == "ORDER_CANCEL_RECONCILIATION_FAILED":
            if set(payload) != {
                "client_order_id",
                "lookup_request_sha256",
                "authenticated_cursor_request_sha256",
                "request_started_at_ts_ms",
                "request_completed_at_ts_ms",
                "trusted_time_receipt_sha256",
                "raw_trusted_time_receipt_json",
                "error_type",
                "observed_status",
                "lookup_response_sha256",
                "raw_lookup_response_json",
            } or not (
                _is_sha256(payload.get("client_order_id"))
                and _is_sha256(payload.get("lookup_request_sha256"))
                and _is_sha256(
                    payload.get("authenticated_cursor_request_sha256")
                )
                and isinstance(payload.get("request_started_at_ts_ms"), int)
                and not isinstance(payload.get("request_started_at_ts_ms"), bool)
                and 0 < payload["request_started_at_ts_ms"] <= event_ts_ms
            ):
                raise MicroLiveExecutionError(
                    "failed cancel reconciliation audit is invalid"
                )
            error_type = payload.get("error_type")
            observed_status = payload.get("observed_status")
            if isinstance(error_type, str) and error_type:
                if any(
                    payload.get(key) is not None
                    for key in (
                        "observed_status",
                        "lookup_response_sha256",
                        "raw_lookup_response_json",
                        "request_completed_at_ts_ms",
                        "trusted_time_receipt_sha256",
                        "raw_trusted_time_receipt_json",
                    )
                ):
                    raise MicroLiveExecutionError(
                        "failed cancel reconciliation audit is invalid"
                    )
                return
            if error_type is not None or observed_status not in {"OPEN", "FILLED"}:
                raise MicroLiveExecutionError(
                    "failed cancel reconciliation audit is invalid"
                )
            response = _stored_raw_json_object(
                payload.get("raw_lookup_response_json"),
                payload.get("lookup_response_sha256"),
                "stored unresolved cancel lookup response",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            trusted_time_receipt = _stored_raw_json_object(
                payload.get("raw_trusted_time_receipt_json"),
                payload.get("trusted_time_receipt_sha256"),
                "stored trusted completion time receipt",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            request_completed_at_ts_ms = payload.get("request_completed_at_ts_ms")
            unsigned_cursor = {
                key: value
                for key, value in response.items()
                if key
                not in {
                    "cursor_payload_sha256",
                    "signature_algorithm",
                    "signature_hex",
                }
            }
            cursor_payload_sha256 = canonical_json_sha256(unsigned_cursor)
            if not (
                isinstance(request_completed_at_ts_ms, int)
                and not isinstance(request_completed_at_ts_ms, bool)
                and payload["request_started_at_ts_ms"]
                <= request_completed_at_ts_ms
                == event_ts_ms
                and response.get("schema_version")
                == EXECUTION_CURSOR_SCHEMA_VERSION
                and response.get("authorization_id")
                == self._authorization.authorization_id
                and response.get("execution_service_binding_sha256")
                == self._authorization.execution_service_binding_sha256
                and response.get("request_started_at_ts_ms")
                == payload["request_started_at_ts_ms"]
                and response.get("cursor_payload_sha256")
                == cursor_payload_sha256
                and _verify_signed_risk_domain_receipt(
                    response,
                    expected_core={
                        **unsigned_cursor,
                        "cursor_payload_sha256": cursor_payload_sha256,
                    },
                    public_key_modulus_hex=(
                        self._authorization.execution_public_key_modulus_hex
                    ),
                    public_key_exponent=(
                        self._authorization.execution_public_key_exponent
                    ),
                )
                and _trusted_time_receipt_is_valid(
                    trusted_time_receipt,
                    self._authorization,
                    request_started_at_ts_ms=payload["request_started_at_ts_ms"],
                    operation="read_order_fill_cursor",
                    response_sha256=payload["lookup_response_sha256"],
                    request_completed_at_ts_ms=request_completed_at_ts_ms,
                )
                and response.get("status") == observed_status
                and response.get("effective_at_ts_ms") is None
                and response.get("observed_at_ts_ms")
                <= event_ts_ms
                + self._authorization.execution_maximum_clock_skew_ms
            ):
                raise MicroLiveExecutionError(
                    "failed cancel reconciliation audit is invalid"
                )
            return
        if event_type in {"ORDER_SUBMISSION_UNKNOWN", "ORDER_CANCEL_UNKNOWN"}:
            if set(payload) != {"client_order_id", "error_type"} or not (
                isinstance(payload.get("client_order_id"), str)
                and payload["client_order_id"]
                and isinstance(payload.get("error_type"), str)
                and payload["error_type"]
            ):
                raise MicroLiveExecutionError("unknown transport event payload is invalid")
            return
        if event_type == "FILL_RECORDED":
            if set(payload) != {
                "client_order_id",
                "exchange_order_id",
                "fill_id",
                "market_id",
                "token_id",
                "quantity",
                "price",
                "fee_usd",
                "executed_at_ts_ms",
                "fill_event_sequence",
                "cumulative_filled_quantity",
                "cumulative_fill_count",
                "transport_event_sha256",
                "raw_transport_event_json",
            }:
                raise MicroLiveExecutionError("fill payload schema is invalid")
            fill_quantity = _positive_decimal(
                payload.get("quantity"), "stored fill quantity"
            )
            fill_price = _positive_decimal(payload.get("price"), "stored fill price")
            fill_fee = _nonnegative_decimal(payload.get("fee_usd"), "stored fill fee")
            cumulative_filled_quantity = _positive_decimal(
                payload.get("cumulative_filled_quantity"),
                "stored cumulative filled quantity",
            )
            transport_event = _stored_raw_json_object(
                payload.get("raw_transport_event_json"),
                payload.get("transport_event_sha256"),
                "stored fill transport event",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            if (
                fill_price >= 1
                or fill_fee > fill_quantity * FROZEN_EXECUTION_FEE_PER_UNIT_USD
                or set(transport_event)
                != {
                    "event_type",
                    "client_order_id",
                    "exchange_order_id",
                    "fill_id",
                    "market_id",
                    "token_id",
                    "quantity",
                    "price",
                    "fee_usd",
                    "executed_at_ts_ms",
                    "fill_event_sequence",
                    "cumulative_filled_quantity",
                    "cumulative_fill_count",
                }
                or transport_event.get("event_type") != "FILL"
                or any(
                    transport_event.get(key) != payload.get(key)
                    for key in (
                        "client_order_id",
                        "exchange_order_id",
                        "fill_id",
                        "market_id",
                        "token_id",
                        "quantity",
                        "price",
                        "fee_usd",
                        "executed_at_ts_ms",
                        "fill_event_sequence",
                        "cumulative_filled_quantity",
                        "cumulative_fill_count",
                    )
                )
                or cumulative_filled_quantity < fill_quantity
                or not isinstance(payload.get("fill_event_sequence"), int)
                or isinstance(payload.get("fill_event_sequence"), bool)
                or payload["fill_event_sequence"] <= 0
                or not isinstance(payload.get("cumulative_fill_count"), int)
                or isinstance(payload.get("cumulative_fill_count"), bool)
                or payload["cumulative_fill_count"]
                < payload["fill_event_sequence"]
                or not isinstance(payload.get("executed_at_ts_ms"), int)
                or isinstance(payload.get("executed_at_ts_ms"), bool)
                or payload["executed_at_ts_ms"] > event_ts_ms
            ):
                raise MicroLiveExecutionError("fill payload values are invalid")
            return
        if event_type in {"ORDER_FILLED", "ORDER_CANCELED", "ORDER_EXPIRED"}:
            if set(payload) != {
                "client_order_id",
                "exchange_order_id",
                "market_id",
                "token_id",
                "effective_at_ts_ms",
                "cumulative_filled_quantity",
                "final_fill_event_sequence",
                "final_fill_count",
                "final_fill_watermark",
                "fill_delivery_complete",
                "request_started_at_ts_ms",
                "request_completed_at_ts_ms",
                "trusted_time_receipt_sha256",
                "raw_trusted_time_receipt_json",
                "transport_event_sha256",
                "raw_transport_event_json",
            } or not (
                isinstance(payload.get("client_order_id"), str)
                and payload["client_order_id"]
                and isinstance(payload.get("exchange_order_id"), str)
                and payload["exchange_order_id"]
                and isinstance(payload.get("market_id"), str)
                and _CONDITION_ID.fullmatch(payload["market_id"]) is not None
                and isinstance(payload.get("token_id"), str)
                and _TOKEN_ID.fullmatch(payload["token_id"]) is not None
                and isinstance(payload.get("effective_at_ts_ms"), int)
                and not isinstance(payload.get("effective_at_ts_ms"), bool)
                and 0 < payload["effective_at_ts_ms"] <= event_ts_ms
                and isinstance(payload.get("cumulative_filled_quantity"), str)
                and isinstance(payload.get("final_fill_event_sequence"), int)
                and not isinstance(payload.get("final_fill_event_sequence"), bool)
                and payload["final_fill_event_sequence"] >= 0
                and isinstance(payload.get("final_fill_count"), int)
                and not isinstance(payload.get("final_fill_count"), bool)
                and payload["final_fill_count"]
                >= payload["final_fill_event_sequence"]
                and _is_sha256(payload.get("final_fill_watermark"))
                and payload.get("fill_delivery_complete") is True
                and isinstance(payload.get("request_started_at_ts_ms"), int)
                and not isinstance(payload.get("request_started_at_ts_ms"), bool)
                and isinstance(payload.get("request_completed_at_ts_ms"), int)
                and not isinstance(payload.get("request_completed_at_ts_ms"), bool)
                and 0
                < payload["request_started_at_ts_ms"]
                <= payload["request_completed_at_ts_ms"]
                == event_ts_ms
                and _is_sha256(payload.get("transport_event_sha256"))
                and _is_sha256(payload.get("trusted_time_receipt_sha256"))
            ):
                raise MicroLiveExecutionError("order close payload is invalid")
            transport_event = _stored_raw_json_object(
                payload.get("raw_transport_event_json"),
                payload.get("transport_event_sha256"),
                "stored order close transport event",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            trusted_time_receipt = _stored_raw_json_object(
                payload.get("raw_trusted_time_receipt_json"),
                payload.get("trusted_time_receipt_sha256"),
                "stored trusted completion time receipt",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            external_keys = {
                "schema_version",
                "authorization_id",
                "execution_service_binding_sha256",
                "request_started_at_ts_ms",
                "event_type",
                "client_order_id",
                "exchange_order_id",
                "market_id",
                "token_id",
                "status",
                "observed_at_ts_ms",
                "effective_at_ts_ms",
                "cumulative_filled_quantity",
                "final_fill_event_sequence",
                "final_fill_count",
                "final_fill_watermark",
                "fill_delivery_complete",
                "fill_events",
                "cursor_payload_sha256",
                "signature_algorithm",
                "signature_hex",
            }
            expected_status = {
                "ORDER_FILLED": "FILLED",
                "ORDER_CANCELED": "CANCELED",
                "ORDER_EXPIRED": "EXPIRED",
            }[event_type]
            unsigned_cursor = {
                key: value
                for key, value in transport_event.items()
                if key
                not in {
                    "cursor_payload_sha256",
                    "signature_algorithm",
                    "signature_hex",
                }
            }
            cursor_payload_sha256 = canonical_json_sha256(unsigned_cursor)
            valid_transport = (
                set(transport_event) == external_keys
                and transport_event.get("schema_version")
                == EXECUTION_CURSOR_SCHEMA_VERSION
                and transport_event.get("authorization_id")
                == self._authorization.authorization_id
                and transport_event.get("execution_service_binding_sha256")
                == self._authorization.execution_service_binding_sha256
                and transport_event.get("request_started_at_ts_ms")
                == payload.get("request_started_at_ts_ms")
                and transport_event.get("cursor_payload_sha256")
                == cursor_payload_sha256
                and transport_event.get("final_fill_watermark")
                == _expected_final_fill_watermark(transport_event)
                and _verify_signed_risk_domain_receipt(
                    transport_event,
                    expected_core={
                        **unsigned_cursor,
                        "cursor_payload_sha256": cursor_payload_sha256,
                    },
                    public_key_modulus_hex=(
                        self._authorization.execution_public_key_modulus_hex
                    ),
                    public_key_exponent=(
                        self._authorization.execution_public_key_exponent
                    ),
                )
                and transport_event.get("event_type") == "ORDER_FILL_CURSOR"
                and transport_event.get("status") == expected_status
                and isinstance(transport_event.get("observed_at_ts_ms"), int)
                and transport_event["observed_at_ts_ms"]
                <= event_ts_ms
                + self._authorization.execution_maximum_clock_skew_ms
                and _trusted_time_receipt_is_valid(
                    trusted_time_receipt,
                    self._authorization,
                    request_started_at_ts_ms=payload["request_started_at_ts_ms"],
                    operation="read_order_fill_cursor",
                    response_sha256=payload["transport_event_sha256"],
                    request_completed_at_ts_ms=payload[
                        "request_completed_at_ts_ms"
                    ],
                )
                and isinstance(transport_event.get("fill_events"), list)
                and len(transport_event["fill_events"])
                <= MAXIMUM_FILL_DELIVERY_EVENTS_PER_ORDER
                and all(
                    transport_event.get(key) == payload.get(key)
                    for key in (
                        "client_order_id",
                        "exchange_order_id",
                        "market_id",
                        "token_id",
                        "effective_at_ts_ms",
                        "cumulative_filled_quantity",
                        "final_fill_event_sequence",
                        "final_fill_count",
                        "final_fill_watermark",
                        "fill_delivery_complete",
                    )
                )
            )
            if not valid_transport:
                raise MicroLiveExecutionError("order close transport identity is invalid")
            return
        if event_type == "SETTLEMENT_RECORDED":
            payout = _nonnegative_decimal(
                payload.get("payout_per_token"), "stored settlement payout"
            )
            if set(payload) != {
                "client_order_id",
                "settlement_id",
                "market_id",
                "slug",
                "token_id",
                "winning_token_id",
                "payout_per_token",
                "finalized_at_ts_ms",
                "observed_at_ts_ms",
                "settlement_authority_identity_sha256",
                "finality_status",
                "confirmation_depth",
                "provider_url",
                "provider_parameters",
                "provider_retrieved_at_ts_ms",
                "raw_provider_request_json",
                "raw_provider_request_sha256",
                "raw_provider_response_json",
                "raw_provider_response_sha256",
                "finality_metadata",
                "provider_provenance_sha256",
                "official_settlement_sha256",
                "raw_official_settlement_json",
                "request_started_at_ts_ms",
                "request_completed_at_ts_ms",
                "trusted_time_receipt_sha256",
                "raw_trusted_time_receipt_json",
            } or not (
                isinstance(payload.get("settlement_id"), str)
                and payload["settlement_id"]
                and isinstance(payload.get("market_id"), str)
                and _CONDITION_ID.fullmatch(payload["market_id"]) is not None
                and isinstance(payload.get("slug"), str)
                and _SLUG.fullmatch(payload["slug"]) is not None
                and isinstance(payload.get("token_id"), str)
                and _TOKEN_ID.fullmatch(payload["token_id"]) is not None
                and isinstance(payload.get("winning_token_id"), str)
                and _TOKEN_ID.fullmatch(payload["winning_token_id"]) is not None
                and payout in {Decimal("0"), Decimal("1")}
                and isinstance(payload.get("finalized_at_ts_ms"), int)
                and not isinstance(payload.get("finalized_at_ts_ms"), bool)
                and 0 < payload["finalized_at_ts_ms"] <= event_ts_ms
                and isinstance(payload.get("observed_at_ts_ms"), int)
                and not isinstance(payload.get("observed_at_ts_ms"), bool)
                and payload["finalized_at_ts_ms"]
                <= payload["observed_at_ts_ms"]
                <= event_ts_ms + self._authorization.execution_maximum_clock_skew_ms
                and payload.get("settlement_authority_identity_sha256")
                == self._authorization.execution_settlement_authority_identity_sha256
                and payload.get("finality_status") == "FINAL"
                and isinstance(payload.get("confirmation_depth"), int)
                and not isinstance(payload.get("confirmation_depth"), bool)
                and payload["confirmation_depth"] >= 1
                and isinstance(payload.get("provider_url"), str)
                and payload["provider_url"].startswith("https://")
                and isinstance(payload.get("provider_parameters"), Mapping)
                and bool(payload["provider_parameters"])
                and isinstance(
                    payload.get("provider_retrieved_at_ts_ms"), int
                )
                and not isinstance(
                    payload.get("provider_retrieved_at_ts_ms"), bool
                )
                and payload["finalized_at_ts_ms"]
                <= payload["provider_retrieved_at_ts_ms"]
                <= payload["observed_at_ts_ms"]
                and _is_sha256(payload.get("raw_provider_request_sha256"))
                and _is_sha256(payload.get("raw_provider_response_sha256"))
                and isinstance(payload.get("finality_metadata"), Mapping)
                and payload["finality_metadata"].get("confirmation_depth")
                == payload["confirmation_depth"]
                and _is_sha256(payload.get("provider_provenance_sha256"))
                and _is_sha256(payload.get("official_settlement_sha256"))
                and isinstance(payload.get("request_started_at_ts_ms"), int)
                and not isinstance(payload.get("request_started_at_ts_ms"), bool)
                and isinstance(payload.get("request_completed_at_ts_ms"), int)
                and not isinstance(payload.get("request_completed_at_ts_ms"), bool)
                and payload["request_completed_at_ts_ms"] == event_ts_ms
                and _is_sha256(payload.get("trusted_time_receipt_sha256"))
            ):
                raise MicroLiveExecutionError("settlement payload is invalid")
            official = _stored_raw_json_object(
                payload.get("raw_official_settlement_json"),
                payload.get("official_settlement_sha256"),
                "stored official settlement event",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            if set(official) != {
                "schema_version",
                "authorization_id",
                "execution_service_binding_sha256",
                "settlement_authority_identity_sha256",
                "event_type",
                "settlement_id",
                "market_id",
                "slug",
                "winning_token_id",
                "payout_per_token",
                "finalized_at_ts_ms",
                "observed_at_ts_ms",
                "finality_status",
                "confirmation_depth",
                "provider_url",
                "provider_parameters",
                "provider_retrieved_at_ts_ms",
                "raw_provider_request_json",
                "raw_provider_request_sha256",
                "raw_provider_response_json",
                "raw_provider_response_sha256",
                "finality_metadata",
                "provider_provenance_sha256",
                "signature_algorithm",
                "signature_hex",
            } or not (
                official.get("schema_version") == SETTLEMENT_RECEIPT_SCHEMA_VERSION
                and official.get("authorization_id")
                == self._authorization.authorization_id
                and official.get("execution_service_binding_sha256")
                == self._authorization.execution_service_binding_sha256
                and official.get("settlement_authority_identity_sha256")
                == self._authorization.execution_settlement_authority_identity_sha256
                and official.get("event_type") == "OFFICIAL_SETTLEMENT"
                and all(
                    official.get(key) == payload.get(key)
                    for key in (
                        "settlement_id",
                        "market_id",
                        "slug",
                        "winning_token_id",
                        "payout_per_token",
                        "finalized_at_ts_ms",
                        "observed_at_ts_ms",
                        "finality_status",
                        "confirmation_depth",
                        "provider_url",
                        "provider_parameters",
                        "provider_retrieved_at_ts_ms",
                        "raw_provider_request_json",
                        "raw_provider_request_sha256",
                        "raw_provider_response_json",
                        "raw_provider_response_sha256",
                        "finality_metadata",
                        "provider_provenance_sha256",
                    )
                )
                and _verify_signed_risk_domain_receipt(
                    official,
                    expected_core={
                        key: value
                        for key, value in official.items()
                        if key not in {"signature_algorithm", "signature_hex"}
                    },
                    public_key_modulus_hex=(
                        self._authorization.execution_public_key_modulus_hex
                    ),
                    public_key_exponent=(
                        self._authorization.execution_public_key_exponent
                    ),
                )
            ):
                raise MicroLiveExecutionError("settlement transport identity is invalid")
            raw_provider_request = _stored_raw_json_object(
                payload.get("raw_provider_request_json"),
                payload.get("raw_provider_request_sha256"),
                "stored settlement provider request",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            raw_provider_response = _stored_raw_json_object(
                payload.get("raw_provider_response_json"),
                payload.get("raw_provider_response_sha256"),
                "stored settlement provider response",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            provider_provenance = {
                "provider_parameters": payload["provider_parameters"],
                "provider_retrieved_at_ts_ms": payload[
                    "provider_retrieved_at_ts_ms"
                ],
                "provider_url": payload["provider_url"],
                "raw_provider_request_sha256": payload[
                    "raw_provider_request_sha256"
                ],
                "raw_provider_response_sha256": payload[
                    "raw_provider_response_sha256"
                ],
                "finality_metadata": payload["finality_metadata"],
            }
            if not (
                raw_provider_request
                and raw_provider_response
                and set(raw_provider_request) == {"method", "parameters", "url"}
                and raw_provider_request["method"] == "GET"
                and raw_provider_request["url"] == payload["provider_url"]
                and raw_provider_request["parameters"]
                == payload["provider_parameters"]
                and raw_provider_response.get("condition_id")
                == payload["market_id"]
                and raw_provider_response.get("confirmation_depth")
                == payload["confirmation_depth"]
                and raw_provider_response.get("finality_status")
                == payload["finality_status"]
                and raw_provider_response.get("settlement_id")
                == payload["settlement_id"]
                and raw_provider_response.get("winning_token_id")
                == payload["winning_token_id"]
                and set(payload["finality_metadata"])
                == {
                    "confirmation_depth",
                    "finality_policy",
                    "source_block_hash",
                    "source_block_number",
                }
                and payload["finality_metadata"]["confirmation_depth"]
                == payload["confirmation_depth"]
                and isinstance(
                    payload["finality_metadata"]["finality_policy"], str
                )
                and bool(payload["finality_metadata"]["finality_policy"])
                and _CONDITION_ID.fullmatch(
                    str(payload["finality_metadata"]["source_block_hash"])
                )
                is not None
                and isinstance(
                    payload["finality_metadata"]["source_block_number"], int
                )
                and not isinstance(
                    payload["finality_metadata"]["source_block_number"], bool
                )
                and payload["finality_metadata"]["source_block_number"] >= 0
                and payload["provider_provenance_sha256"]
                == canonical_json_sha256(provider_provenance)
            ):
                raise MicroLiveExecutionError(
                    "settlement provider provenance is invalid"
                )
            trusted_time_receipt = _stored_raw_json_object(
                payload.get("raw_trusted_time_receipt_json"),
                payload.get("trusted_time_receipt_sha256"),
                "stored settlement trusted time receipt",
                maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
            )
            if not _trusted_time_receipt_is_valid(
                trusted_time_receipt,
                self._authorization,
                request_started_at_ts_ms=payload["request_started_at_ts_ms"],
                operation="official_settlement",
                response_sha256=payload["official_settlement_sha256"],
                request_completed_at_ts_ms=payload["request_completed_at_ts_ms"],
            ):
                raise MicroLiveExecutionError(
                    "settlement trusted completion identity is invalid"
                )
            return
        if event_type == "KILL_SWITCH_ENGAGED":
            if set(payload) != {"reason", "engaged_at_ts_ms"} or not (
                isinstance(payload.get("reason"), str)
                and payload["reason"]
                and isinstance(payload.get("engaged_at_ts_ms"), int)
                and not isinstance(payload.get("engaged_at_ts_ms"), bool)
                and payload["engaged_at_ts_ms"] > 0
            ):
                raise MicroLiveExecutionError("kill-switch payload is invalid")
            return
        raise MicroLiveExecutionError("unhandled micro-live event payload")

    def _reconcile_view(self) -> dict[str, Any]:
        self._verify_event_chain()
        orders: dict[str, dict[str, Any]] = {}
        business_keys: dict[str, dict[str, Any]] = {}
        exchange_order_ids: dict[str, dict[str, Any]] = {}
        fills: dict[str, dict[str, Any]] = {}
        settlements: dict[str, dict[str, Any]] = {}
        kill_switch_active = False
        kill_switch_reason: str | None = None
        pending_execution_audit: dict[str, Any] | None = None
        for event in self._events:
            event_type = event["event_type"]
            payload = dict(event["payload"])
            client_order_id = payload.get("client_order_id")
            if pending_execution_audit is not None and event_type != "ORDER_PREPARED":
                raise MicroLiveExecutionError(
                    "execution-intent audit is not followed by order preparation"
                )
            if event_type == "SIGNAL_EVALUATED":
                if payload.get("disposition") == "EXECUTION_INTENT":
                    pending_execution_audit = payload
                continue
            if event_type == "SIGNAL_REJECTED":
                continue
            if event_type == "ORDER_PREPARED":
                if (
                    pending_execution_audit is None
                    or not isinstance(client_order_id, str)
                    or client_order_id in orders
                    or payload.get("intent_id") != client_order_id
                    or payload.get("authorization_id") != self._authorization.authorization_id
                    or payload.get("candidate_bundle_sha256")
                    != self._authorization.candidate_bundle_sha256
                ):
                    raise MicroLiveExecutionError("prepared order identity is invalid")
                if not (
                    pending_execution_audit.get("market_id") == payload.get("market_id")
                    and pending_execution_audit.get("decision_ts_ms")
                    == payload.get("decision_ts_ms")
                    and pending_execution_audit.get("signal_payload_sha256")
                    == payload.get("signal_payload_sha256")
                    and pending_execution_audit.get("raw_signal_payload_sha256")
                    == payload.get("raw_signal_payload_sha256")
                    and pending_execution_audit.get("market_identity_sha256")
                    == payload.get("market_identity_sha256")
                    and pending_execution_audit.get("feature_row_sha256")
                    == payload.get("feature_row_sha256")
                    and pending_execution_audit.get("raw_feature_row_sha256")
                    == payload.get("raw_feature_row_sha256")
                    and pending_execution_audit.get(
                        "provider_feature_evidence_graph_sha256"
                    )
                    == payload.get("provider_feature_evidence_graph_sha256")
                    and pending_execution_audit.get("provider_feature_file_sha256")
                    == payload.get("provider_feature_file_sha256")
                ):
                    raise MicroLiveExecutionError(
                        "order preparation does not reconcile with execution-intent audit"
                    )
                pending_execution_audit = None
                business_key = payload.get("business_key")
                if not isinstance(business_key, str) or business_key in business_keys:
                    raise MicroLiveExecutionError("prepared order business identity is duplicated")
                order = {
                    "intent_id": client_order_id,
                    "prepared": payload,
                    "acknowledgement": None,
                    "submission_unknown": False,
                    "cancel_prepared": None,
                    "cancel_unknown": False,
                    "cancel_retry_authorized": False,
                    "closed_status": None,
                    "close_event": None,
                    "fills": [],
                    "settlement": None,
                }
                orders[client_order_id] = order
                business_keys[business_key] = order
            elif event_type in {
                "ORDER_ACKNOWLEDGED",
                "ORDER_REJECTED",
                "ORDER_SUBMISSION_UNKNOWN",
                "ORDER_SUBMISSION_RECONCILED",
                "ORDER_SUBMISSION_RECONCILIATION_FAILED",
                "ORDER_CANCEL_PREPARED",
                "ORDER_CANCEL_RECONCILED",
                "ORDER_CANCEL_RECONCILIATION_FAILED",
                "ORDER_FILLED",
                "ORDER_CANCELED",
                "ORDER_EXPIRED",
                "ORDER_CANCEL_UNKNOWN",
            }:
                order = orders.get(str(client_order_id))
                if order is None:
                    raise MicroLiveExecutionError("order lifecycle event lacks preparation")
                if event_type == "ORDER_CANCEL_PREPARED":
                    prepared = order["prepared"]
                    acknowledgement = order["acknowledgement"]
                    if (
                        acknowledgement is None
                        or order["cancel_prepared"] is not None
                        or order["closed_status"] is not None
                        or payload.get("client_order_id") != prepared["client_order_id"]
                        or payload.get("exchange_order_id")
                        != acknowledgement["exchange_order_id"]
                        or payload.get("market_id") != prepared["market_id"]
                        or payload.get("token_id") != prepared["token_id"]
                    ):
                        raise MicroLiveExecutionError(
                            "prepared cancellation lifecycle is invalid"
                        )
                    order["cancel_prepared"] = payload
                elif event_type == "ORDER_ACKNOWLEDGED":
                    prepared = order["prepared"]
                    exchange_order_id = payload.get("exchange_order_id")
                    if (
                        order["acknowledgement"] is not None
                        or order["closed_status"] is not None
                        or order["submission_unknown"]
                        or not isinstance(exchange_order_id, str)
                        or exchange_order_id in exchange_order_ids
                        or payload.get("market_id") != prepared["market_id"]
                        or payload.get("token_id") != prepared["token_id"]
                        or payload.get("accepted_quantity") != prepared["quantity"]
                        or payload.get("limit_price") != prepared["limit_price"]
                    ):
                        raise MicroLiveExecutionError("order acknowledgement is duplicated")
                    order["acknowledgement"] = payload
                    exchange_order_ids[exchange_order_id] = order
                elif event_type == "ORDER_REJECTED":
                    if (
                        order["acknowledgement"] is not None
                        or order["closed_status"] is not None
                        or order["submission_unknown"]
                    ):
                        raise MicroLiveExecutionError("order rejection lifecycle is invalid")
                    order["closed_status"] = "REJECTED"
                elif event_type == "ORDER_SUBMISSION_UNKNOWN":
                    if order["acknowledgement"] is not None or order["closed_status"] is not None:
                        raise MicroLiveExecutionError("unknown submission lifecycle is invalid")
                    order["submission_unknown"] = True
                elif event_type == "ORDER_SUBMISSION_RECONCILED":
                    exchange_order_id = payload.get("exchange_order_id")
                    prepared = order["prepared"]
                    if (
                        order["submission_unknown"] is not True
                        or order["acknowledgement"] is not None
                        or order["closed_status"] is not None
                        or payload.get("market_id") != prepared["market_id"]
                        or payload.get("token_id") != prepared["token_id"]
                        or payload.get("accepted_quantity") != prepared["quantity"]
                        or payload.get("limit_price") != prepared["limit_price"]
                    ):
                        raise MicroLiveExecutionError(
                            "submission reconciliation lifecycle is invalid"
                        )
                    order["submission_unknown"] = False
                    if payload.get("status") == "ACCEPTED":
                        if (
                            not isinstance(exchange_order_id, str)
                            or exchange_order_id in exchange_order_ids
                        ):
                            raise MicroLiveExecutionError(
                                "reconciled exchange order identity is duplicated"
                            )
                        acknowledgement = {
                            key: value
                            for key, value in payload.items()
                            if key
                            not in {
                                "lookup_response_sha256",
                                "raw_lookup_response_json",
                                "transport_invocation_id",
                                "fence_response_sha256",
                                "raw_fence_response_json",
                            }
                        }
                        order["acknowledgement"] = acknowledgement
                        exchange_order_ids[exchange_order_id] = order
                    elif payload.get("status") == "REJECTED":
                        order["closed_status"] = "REJECTED"
                    else:
                        raise MicroLiveExecutionError(
                            "submission reconciliation status is invalid"
                        )
                elif event_type == "ORDER_SUBMISSION_RECONCILIATION_FAILED":
                    prepared = order["prepared"]
                    expected_lookup_request = {
                        "authorization_id": self._authorization.authorization_id,
                        "client_order_id": client_order_id,
                        "business_key": prepared["business_key"],
                        "market_id": prepared["market_id"],
                        "token_id": prepared["token_id"],
                    }
                    if (
                        order["submission_unknown"] is not True
                        or payload.get("lookup_request_sha256")
                        != canonical_json_sha256(expected_lookup_request)
                    ):
                        raise MicroLiveExecutionError(
                            "failed submission reconciliation lifecycle is invalid"
                        )
                elif event_type == "ORDER_CANCEL_RECONCILED":
                    prepared = order["prepared"]
                    acknowledgement = order["acknowledgement"]
                    if (
                        acknowledgement is None
                        or order["cancel_prepared"] is None
                        or order["cancel_unknown"] is not True
                        or order["closed_status"] is not None
                        or order["settlement"] is not None
                        or payload.get("exchange_order_id")
                        != acknowledgement["exchange_order_id"]
                        or payload.get("market_id") != prepared["market_id"]
                        or payload.get("token_id") != prepared["token_id"]
                        or payload.get("accepted_quantity") != prepared["quantity"]
                        or payload.get("limit_price") != prepared["limit_price"]
                        or payload.get("status") not in {"CANCELED", "EXPIRED"}
                        or not _close_event_matches_fill_ledger(order, payload)
                        or int(payload["effective_at_ts_ms"])
                        < int(prepared["submitted_at_ts_ms"])
                    ):
                        raise MicroLiveExecutionError(
                            "cancel reconciliation lifecycle is invalid"
                        )
                    order["cancel_unknown"] = False
                    order["closed_status"] = payload["status"]
                    order["close_event"] = payload
                elif event_type == "ORDER_CANCEL_RECONCILIATION_FAILED":
                    prepared = order["prepared"]
                    acknowledgement = order["acknowledgement"]
                    if acknowledgement is None or order["cancel_prepared"] is None:
                        raise MicroLiveExecutionError(
                            "failed cancel reconciliation lifecycle is invalid"
                        )
                    expected_lookup_request = {
                        "authorization_id": self._authorization.authorization_id,
                        "client_order_id": client_order_id,
                        "business_key": prepared["business_key"],
                        "exchange_order_id": acknowledgement["exchange_order_id"],
                        "market_id": prepared["market_id"],
                        "token_id": prepared["token_id"],
                    }
                    request_started_at_ts_ms = payload.get(
                        "request_started_at_ts_ms"
                    )
                    expected_authenticated_request = {
                        "authorization_id": self._authorization.authorization_id,
                        "execution_service_binding_sha256": (
                            self._authorization.execution_service_binding_sha256
                        ),
                        "client_order_id": client_order_id,
                        "business_key": prepared["business_key"],
                        "exchange_order_id": acknowledgement["exchange_order_id"],
                        "market_id": prepared["market_id"],
                        "token_id": prepared["token_id"],
                        "request_started_at_ts_ms": request_started_at_ts_ms,
                    }
                    if (
                        order["cancel_unknown"] is not True
                        or order["closed_status"] is not None
                        or payload.get("lookup_request_sha256")
                        != canonical_json_sha256(expected_lookup_request)
                        or payload.get("authenticated_cursor_request_sha256")
                        != canonical_json_sha256(expected_authenticated_request)
                    ):
                        raise MicroLiveExecutionError(
                            "failed cancel reconciliation lifecycle is invalid"
                        )
                    if payload.get("observed_status") is not None:
                        response = _stored_raw_json_object(
                            payload["raw_lookup_response_json"],
                            payload["lookup_response_sha256"],
                            "stored unresolved cancel lookup response",
                            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
                        )
                        if response.get("event_type") == "ORDER_FILL_CURSOR":
                            if not (
                                response.get("status") == "OPEN"
                                and response.get("client_order_id")
                                == client_order_id
                                and response.get("exchange_order_id")
                                == acknowledgement["exchange_order_id"]
                                and response.get("market_id")
                                == prepared["market_id"]
                                and response.get("token_id")
                                == prepared["token_id"]
                                and response.get("fill_delivery_complete") is False
                            ):
                                raise MicroLiveExecutionError(
                                    "unresolved fill cursor identity is invalid"
                                )
                        else:
                            raise MicroLiveExecutionError(
                                "unresolved cancellation evidence is not an "
                                "authoritative fill cursor"
                            )
                        order["cancel_retry_authorized"] = True
                elif event_type in {
                    "ORDER_FILLED",
                    "ORDER_CANCELED",
                    "ORDER_EXPIRED",
                }:
                    if (
                        order["acknowledgement"] is None
                        or order["closed_status"] is not None
                        or order["settlement"] is not None
                        or payload.get("exchange_order_id")
                        != order["acknowledgement"]["exchange_order_id"]
                        or payload.get("market_id")
                        != order["prepared"]["market_id"]
                        or payload.get("token_id")
                        != order["prepared"]["token_id"]
                        or int(payload["effective_at_ts_ms"])
                        < int(order["prepared"]["submitted_at_ts_ms"])
                        or not _close_event_matches_fill_ledger(order, payload)
                        or (
                            event_type == "ORDER_FILLED"
                            and Decimal(payload["cumulative_filled_quantity"])
                            != Decimal(order["prepared"]["quantity"])
                        )
                    ):
                        raise MicroLiveExecutionError("order close lifecycle is invalid")
                    order["cancel_unknown"] = False
                    order["cancel_retry_authorized"] = False
                    order["closed_status"] = {
                        "ORDER_FILLED": "FILLED",
                        "ORDER_CANCELED": "CANCELED",
                        "ORDER_EXPIRED": "EXPIRED",
                    }[event_type]
                    order["close_event"] = payload
                else:
                    if (
                        order["acknowledgement"] is None
                        or order["cancel_prepared"] is None
                        or order["closed_status"] is not None
                        or order["settlement"] is not None
                    ):
                        raise MicroLiveExecutionError("unknown cancellation lifecycle is invalid")
                    order["cancel_unknown"] = True
                    order["cancel_retry_authorized"] = False
            elif event_type == "FILL_RECORDED":
                order = orders.get(str(client_order_id))
                fill_id = payload.get("fill_id")
                close_event = None if order is None else order.get("close_event")
                existing_filled_quantity = (
                    Decimal("0")
                    if order is None
                    else sum(
                        (
                            Decimal(fill["quantity"])
                            for fill in order["fills"]
                        ),
                        Decimal("0"),
                    )
                )
                previous_cumulative_fill_count = (
                    0
                    if order is None or not order["fills"]
                    else int(order["fills"][-1]["cumulative_fill_count"])
                )
                if (
                    order is None
                    or order["acknowledgement"] is None
                    or order["settlement"] is not None
                    or not isinstance(fill_id, str)
                    or fill_id in fills
                    or payload.get("exchange_order_id")
                    != order["acknowledgement"]["exchange_order_id"]
                    or payload.get("market_id") != order["prepared"]["market_id"]
                    or payload.get("token_id") != order["prepared"]["token_id"]
                    or int(payload["executed_at_ts_ms"])
                    < int(order["prepared"]["submitted_at_ts_ms"])
                    or close_event is not None
                    or payload.get("fill_event_sequence")
                    != len(order["fills"]) + 1
                    or int(payload["cumulative_fill_count"])
                    <= previous_cumulative_fill_count
                    or Decimal(payload["cumulative_filled_quantity"])
                    != existing_filled_quantity + Decimal(payload["quantity"])
                ):
                    raise MicroLiveExecutionError("fill event identity is invalid")
                fills[fill_id] = payload
                order["fills"].append(payload)
            elif event_type == "SETTLEMENT_RECORDED":
                order = orders.get(str(client_order_id))
                prepared = None if order is None else dict(order["prepared"])
                signal = (
                    None
                    if prepared is None
                    else dict(prepared["signal_payload"])
                )
                if (
                    order is None
                    or order["acknowledgement"] is None
                    or order["settlement"] is not None
                    or not order["fills"]
                    or order["close_event"] is None
                    or not _close_event_matches_fill_ledger(
                        order,
                        order["close_event"],
                    )
                    or payload.get("market_id") != prepared["market_id"]
                    or payload.get("slug") != prepared["slug"]
                    or payload.get("token_id") != prepared["token_id"]
                    or int(payload["finalized_at_ts_ms"])
                    < int(dict(signal["market_identity"])["market_end_ts_ms"])
                    or payload.get("winning_token_id")
                    != (
                        prepared["token_id"]
                        if Decimal(payload["payout_per_token"]) == Decimal("1")
                        else (
                            signal["down_token_id"]
                            if prepared["token_id"] == signal["up_token_id"]
                            else signal["up_token_id"]
                        )
                    )
                ):
                    raise MicroLiveExecutionError("settlement event identity is invalid")
                settlement_id = payload.get("settlement_id")
                if not isinstance(settlement_id, str) or settlement_id in settlements:
                    raise MicroLiveExecutionError("settlement identity is duplicated")
                settlements[settlement_id] = payload
                order["settlement"] = payload
            elif event_type == "KILL_SWITCH_ENGAGED":
                if kill_switch_active:
                    raise MicroLiveExecutionError("kill switch event is duplicated")
                kill_switch_active = True
                kill_switch_reason = str(payload.get("reason") or "")

        if pending_execution_audit is not None:
            raise MicroLiveExecutionError(
                "execution-intent audit is missing order preparation"
            )

        cash = self._authorization.maximum_notional_usd
        realized_pnl = Decimal("0")
        unsettled_maximum_loss = Decimal("0")
        positions = {"UP": Decimal("0"), "DOWN": Decimal("0")}
        submitted_notional = Decimal("0")
        open_order_count = 0
        for order in orders.values():
            prepared = order["prepared"]
            requested_qty = Decimal(prepared["quantity"])
            submitted_notional += Decimal(prepared["notional_usd"])
            filled_qty = Decimal("0")
            filled_cost = Decimal("0")
            for fill in order["fills"]:
                qty = Decimal(fill["quantity"])
                price = Decimal(fill["price"])
                fee = Decimal(fill["fee_usd"])
                filled_qty += qty
                filled_cost += qty * price + fee
                cash -= qty * price + fee
                side = "UP" if prepared["selected_action"] == "BUY_UP_HOLD" else "DOWN"
                positions[side] += qty
            if filled_qty > requested_qty or cash < 0:
                raise MicroLiveExecutionError("fill ledger exceeds quantity or cash")
            order["filled_quantity"] = filled_qty
            order["remaining_quantity"] = requested_qty - filled_qty
            order["is_open"] = bool(
                order["acknowledgement"] is not None
                and order["closed_status"] is None
                and order["settlement"] is None
                and order["remaining_quantity"] > 0
            )
            if order["is_open"]:
                open_order_count += 1
            settlement = order.get("settlement")
            if settlement is None:
                if order["submission_unknown"] or (
                    order["acknowledgement"] is None
                    and order["closed_status"] is None
                ):
                    unsettled_maximum_loss += Decimal(
                        prepared["maximum_loss_usd"]
                    )
                elif order["closed_status"] != "REJECTED":
                    unsettled_maximum_loss += filled_cost
                    if order["is_open"]:
                        remaining = order["remaining_quantity"]
                        unsettled_maximum_loss += remaining * (
                            Decimal(prepared["limit_price"])
                            + FROZEN_EXECUTION_FEE_PER_UNIT_USD
                        )
            if settlement is not None:
                if filled_qty < requested_qty and order["closed_status"] is None:
                    raise MicroLiveExecutionError(
                        "partially filled order settled before terminal close"
                    )
                payout = Decimal(settlement["payout_per_token"])
                cash += filled_qty * payout
                realized_cost = sum(
                    Decimal(fill["quantity"]) * Decimal(fill["price"])
                    + Decimal(fill["fee_usd"])
                    for fill in order["fills"]
                )
                realized_pnl += filled_qty * payout - realized_cost
                side = "UP" if prepared["selected_action"] == "BUY_UP_HOLD" else "DOWN"
                positions[side] -= filled_qty
        if submitted_notional > self._authorization.maximum_notional_usd:
            raise MicroLiveExecutionError("submitted notional exceeds authorization")
        if open_order_count > self._authorization.maximum_open_orders:
            raise MicroLiveExecutionError("open order count exceeds authorization")
        loss_budget_consumed = max(-realized_pnl, Decimal("0")) + (
            unsettled_maximum_loss
        )
        return {
            "orders": orders,
            "orders_by_business_key": business_keys,
            "orders_by_exchange_id": exchange_order_ids,
            "fills": fills,
            "settlements": settlements,
            "cash_usd": cash,
            "realized_pnl_usd": realized_pnl,
            "unsettled_maximum_loss_usd": unsettled_maximum_loss,
            "loss_budget_consumed_usd": loss_budget_consumed,
            "positions": positions,
            "submitted_notional_usd": submitted_notional,
            "open_order_count": open_order_count,
            "kill_switch_active": kill_switch_active,
            "kill_switch_reason": kill_switch_reason,
        }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _close_event_matches_fill_ledger(
    order: Mapping[str, Any],
    close_event: Mapping[str, Any],
) -> bool:
    fills = order.get("fills")
    if not isinstance(fills, list):
        return False
    cumulative_filled_quantity = str(
        sum(
            (Decimal(fill["quantity"]) for fill in fills),
            Decimal("0"),
        )
    )
    final_fill_count = (
        int(fills[-1]["cumulative_fill_count"]) if fills else 0
    )
    try:
        transport_event = _stored_raw_json_object(
            close_event.get("raw_transport_event_json"),
            close_event.get("transport_event_sha256"),
            "stored fill cursor close event",
            maximum_bytes=MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        expected_fill_events = [
            json.loads(fill["raw_transport_event_json"]) for fill in fills
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        close_event.get("fill_delivery_complete") is True
        and close_event.get("cumulative_filled_quantity")
        == cumulative_filled_quantity
        and close_event.get("final_fill_event_sequence") == len(fills)
        and close_event.get("final_fill_count") == final_fill_count
        and _is_sha256(close_event.get("final_fill_watermark"))
        and transport_event.get("event_type") == "ORDER_FILL_CURSOR"
        and transport_event.get("fill_events") == expected_fill_events
    )


def _blocked(reason: str, client_order_id: str | None = None) -> dict[str, Any]:
    return {
        "status": "BLOCKED_NO_TRADE",
        "reason": reason,
        "client_order_id": client_order_id,
        "transport_called": False,
    }


def _realized_loss_limit_reached(
    view: Mapping[str, Any],
    authorization: VerifiedMicroLiveAuthorization | _BoundAuthorization,
) -> bool:
    realized = view.get("realized_pnl_usd")
    if not isinstance(realized, Decimal):
        raise MicroLiveExecutionError("realized PnL reconciliation is invalid")
    return realized <= -authorization.maximum_realized_loss_usd


def _new_order_lifecycle_capacity() -> int:
    """Worst-case durable events remaining after one new preparation.

    This covers UNKNOWN + kill + fenced reconciliation, bounded partial fills,
    cancellation intent/result, and settlement.  The count is intentionally
    conservative so a boundary order can always complete its lifecycle.
    """

    return MAXIMUM_FILL_DELIVERY_EVENTS_PER_ORDER + 8


def _required_lifecycle_event_capacity(view: Mapping[str, Any]) -> int:
    """Return still-reserved event slots for every nonterminal order."""

    orders = view.get("orders")
    if not isinstance(orders, Mapping):
        raise MicroLiveExecutionError("lifecycle reserve view is invalid")
    required = 0
    unresolved_exists = False
    for raw_order in orders.values():
        if not isinstance(raw_order, Mapping):
            raise MicroLiveExecutionError("lifecycle reserve order is invalid")
        order = raw_order
        fills = order.get("fills")
        if (
            not isinstance(fills, list)
            or len(fills) > MAXIMUM_FILL_DELIVERY_EVENTS_PER_ORDER
        ):
            raise MicroLiveExecutionError("bounded fill lifecycle is invalid")
        if order.get("settlement") is not None or order.get("closed_status") == "REJECTED":
            continue
        unresolved_exists = True
        acknowledgement = order.get("acknowledgement")
        closed_status = order.get("closed_status")
        if acknowledgement is None:
            # Submission disposition/fence plus the complete accepted path.
            required += _new_order_lifecycle_capacity()
            continue
        if closed_status is not None:
            if fills:
                required += 1  # official settlement
            continue
        remaining_fill_events = (
            MAXIMUM_FILL_DELIVERY_EVENTS_PER_ORDER - len(fills)
        )
        required += remaining_fill_events
        if order.get("cancel_prepared") is None:
            required += 1  # cancel/close intent
        required += 2  # terminal close and settlement
        if order.get("cancel_unknown") is True:
            required += 1  # conservative cancel reconciliation
    if unresolved_exists and view.get("kill_switch_active") is not True:
        required += 1
    return required


def _validated_candidate_signal(
    value: Any,
    *,
    expected_candidate_bundle_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MicroLiveExecutionError("candidate signal payload is not an object")
    signal = copy.deepcopy(dict(value))
    expected_keys = {
        "schema_version",
        "lineage_id",
        "candidate_id",
        "candidate_bundle_sha256",
        "market_id",
        "slug",
        "market_family",
        "decision_ts_ms",
        "observed_at_ts_ms",
        "action_values",
        "executable_asks",
        "up_token_id",
        "down_token_id",
        "market_identity",
        "selected_action",
        "model_scored",
        "fail_closed",
        "fail_closed_reasons",
        "decision_influenced_collection",
        "outcomes_accessed",
        "settlement_accessed",
        "pnl_accessed",
        "safety",
    }
    if set(signal) != expected_keys:
        raise MicroLiveExecutionError("candidate signal schema is not exact")
    if not (
        signal.get("schema_version") == SIGNAL_SCHEMA_VERSION
        and signal.get("lineage_id") == LINEAGE_ID
        and signal.get("candidate_id") == CANDIDATE_ID
        and signal.get("candidate_bundle_sha256")
        == expected_candidate_bundle_sha256
        and isinstance(signal.get("market_id"), str)
        and _CONDITION_ID.fullmatch(signal["market_id"]) is not None
        and isinstance(signal.get("slug"), str)
        and _SLUG.fullmatch(signal["slug"]) is not None
        and signal.get("market_family") == "BTC-15M"
        and isinstance(signal.get("up_token_id"), str)
        and _TOKEN_ID.fullmatch(signal["up_token_id"]) is not None
        and isinstance(signal.get("down_token_id"), str)
        and _TOKEN_ID.fullmatch(signal["down_token_id"]) is not None
        and signal["up_token_id"] != signal["down_token_id"]
        and isinstance(signal.get("decision_ts_ms"), int)
        and not isinstance(signal.get("decision_ts_ms"), bool)
        and signal["decision_ts_ms"] > 0
        and isinstance(signal.get("observed_at_ts_ms"), int)
        and not isinstance(signal.get("observed_at_ts_ms"), bool)
        and signal["observed_at_ts_ms"] >= signal["decision_ts_ms"]
        and signal.get("decision_influenced_collection") is False
        and signal.get("outcomes_accessed") is False
        and signal.get("settlement_accessed") is False
        and signal.get("pnl_accessed") is False
        and dict(signal.get("safety") or {}) == SAFETY
    ):
        raise MicroLiveExecutionError("candidate signal identity or safety is invalid")
    market_start_ts_ms = int(str(signal["slug"]).rsplit("-", maxsplit=1)[1]) * 1_000
    if signal["decision_ts_ms"] not in {
        market_start_ts_ms + 300_000,
        market_start_ts_ms + 600_000,
    }:
        raise MicroLiveExecutionError("candidate signal decision is off the frozen schedule")
    _validated_market_identity(signal)
    action_values = signal.get("action_values")
    executable_asks = signal.get("executable_asks")
    fail_reasons = signal.get("fail_closed_reasons")
    if not isinstance(action_values, Mapping) or set(action_values) != {
        "NO_TRADE",
        "BUY_UP_HOLD",
        "BUY_DOWN_HOLD",
    }:
        raise MicroLiveExecutionError("candidate signal action values are invalid")
    if not isinstance(executable_asks, Mapping) or set(executable_asks) != {"UP", "DOWN"}:
        raise MicroLiveExecutionError("candidate signal executable asks are invalid")
    asks = {
        side: _positive_decimal(executable_asks[side], f"{side} executable ask")
        for side in ("UP", "DOWN")
    }
    if any(value >= 1 for value in asks.values()):
        raise MicroLiveExecutionError("candidate signal executable ask is outside binary range")
    if not isinstance(fail_reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in fail_reasons
    ):
        raise MicroLiveExecutionError("candidate signal fail-closed reasons are invalid")
    model_scored = signal.get("model_scored")
    failed_closed = signal.get("fail_closed")
    if model_scored is True and failed_closed is False:
        values = dict(action_values)
        if (
            isinstance(values.get("NO_TRADE"), bool)
            or values.get("NO_TRADE") != 0.0
            or any(
            isinstance(values.get(action), bool)
            or not isinstance(values.get(action), (int, float))
            or not math.isfinite(float(values[action]))
            for action in ("BUY_UP_HOLD", "BUY_DOWN_HOLD")
            )
        ):
            raise MicroLiveExecutionError("candidate signal scored values are invalid")
        up = float(values["BUY_UP_HOLD"])
        down = float(values["BUY_DOWN_HOLD"])
        expected_action = (
            ("BUY_UP_HOLD" if up >= down else "BUY_DOWN_HOLD")
            if max(up, down) > 0.0
            else "NO_TRADE"
        )
        if signal.get("selected_action") != expected_action or fail_reasons:
            raise MicroLiveExecutionError("candidate signal zero-threshold decision mismatches")
    elif model_scored is False and failed_closed is True:
        if not (
            dict(action_values)
            == {"NO_TRADE": 0.0, "BUY_UP_HOLD": None, "BUY_DOWN_HOLD": None}
            and signal.get("selected_action") == "NO_TRADE"
            and fail_reasons
        ):
            raise MicroLiveExecutionError("candidate signal fail-closed state is invalid")
    else:
        raise MicroLiveExecutionError("candidate signal scoring state is invalid")
    return signal


def _validated_market_identity(signal: Mapping[str, Any]) -> None:
    identity_value = signal.get("market_identity")
    if not isinstance(identity_value, Mapping):
        raise MicroLiveExecutionError("candidate market identity is absent")
    identity = dict(identity_value)
    expected_keys = {
        "source_type",
        "condition_id",
        "slug",
        "market_family",
        "market_start_ts_ms",
        "market_end_ts_ms",
        "up_token_id",
        "down_token_id",
        "gamma_fetched_at_ts_ms",
        "clob_revalidated_at_ts_ms",
        "raw_gamma_payload_sha256",
        "clob_revalidation_payload_sha256",
        "clob_revalidation_passed",
        "outcomes_accessed",
        "settlement_accessed",
        "pnl_accessed",
    }
    if set(identity) != expected_keys:
        raise MicroLiveExecutionError("candidate market identity schema is not exact")
    start = int(str(signal["slug"]).rsplit("-", maxsplit=1)[1]) * 1_000
    fetched_at = identity.get("gamma_fetched_at_ts_ms")
    revalidated_at = identity.get("clob_revalidated_at_ts_ms")
    if not (
        identity.get("source_type") == "gamma_primary_plus_live_clob_revalidation"
        and identity.get("condition_id") == signal["market_id"]
        and identity.get("slug") == signal["slug"]
        and identity.get("market_family") == "btc_updown_15m"
        and identity.get("market_start_ts_ms") == start
        and identity.get("market_end_ts_ms") == start + 900_000
        and identity.get("up_token_id") == signal["up_token_id"]
        and identity.get("down_token_id") == signal["down_token_id"]
        and isinstance(fetched_at, int)
        and not isinstance(fetched_at, bool)
        and fetched_at > 0
        and fetched_at <= signal["observed_at_ts_ms"]
        and isinstance(revalidated_at, int)
        and not isinstance(revalidated_at, bool)
        and fetched_at <= revalidated_at <= signal["observed_at_ts_ms"]
        and _is_sha256(identity.get("raw_gamma_payload_sha256"))
        and _is_sha256(identity.get("clob_revalidation_payload_sha256"))
        and identity.get("clob_revalidation_passed") is True
        and identity.get("outcomes_accessed") is False
        and identity.get("settlement_accessed") is False
        and identity.get("pnl_accessed") is False
    ):
        raise MicroLiveExecutionError("candidate market identity binding is invalid")


def _validate_market_identity_evidence(
    *,
    signal: Mapping[str, Any],
    evidence: Any,
) -> None:
    """Verify the exact provider bytes behind the signal identity hashes."""

    if not isinstance(evidence, Mapping) or set(evidence) != {
        "raw_gamma_payload",
        "raw_clob_revalidation_payload",
    }:
        raise MicroLiveExecutionError("market identity evidence schema is not exact")
    gamma_raw = evidence.get("raw_gamma_payload")
    clob_raw = evidence.get("raw_clob_revalidation_payload")
    if (
        not isinstance(gamma_raw, bytes)
        or not gamma_raw
        or len(gamma_raw) > MAX_RAW_JSON_BYTES
    ):
        raise MicroLiveExecutionError("raw Gamma market identity bytes are invalid")
    if (
        not isinstance(clob_raw, bytes)
        or not clob_raw
        or len(clob_raw) > MAX_RAW_JSON_BYTES
    ):
        raise MicroLiveExecutionError("raw CLOB revalidation bytes are invalid")
    identity = dict(signal["market_identity"])
    if not (
        hashlib.sha256(gamma_raw).hexdigest()
        == identity["raw_gamma_payload_sha256"]
        and hashlib.sha256(clob_raw).hexdigest()
        == identity["clob_revalidation_payload_sha256"]
    ):
        raise MicroLiveExecutionError("market identity raw-byte SHA-256 mismatch")
    gamma = _decode_provider_json(gamma_raw, "Gamma market identity")
    clob = _decode_provider_json(clob_raw, "CLOB market identity")
    slug = str(signal["slug"])
    market_id = str(signal["market_id"])
    expected_tokens = {
        "UP": str(signal["up_token_id"]),
        "DOWN": str(signal["down_token_id"]),
    }
    gamma_rows = _gamma_market_rows(gamma)
    exact_gamma = [
        row
        for row in gamma_rows
        if str(row.get("slug") or row.get("market_slug") or "") == slug
    ]
    if len(exact_gamma) != 1:
        raise MicroLiveExecutionError("Gamma identity does not resolve exact slug once")
    gamma_row = exact_gamma[0]
    gamma_condition = str(
        gamma_row.get("conditionId") or gamma_row.get("condition_id") or ""
    )
    gamma_tokens = _gamma_up_down_tokens(gamma_row)
    if gamma_condition != market_id or gamma_tokens != expected_tokens:
        raise MicroLiveExecutionError("Gamma condition or token identity mismatches")
    if not isinstance(clob, Mapping):
        raise MicroLiveExecutionError("CLOB market identity payload is not an object")
    clob_row = dict(clob)
    clob_condition = str(
        clob_row.get("condition_id")
        or clob_row.get("conditionId")
        or clob_row.get("market")
        or ""
    )
    clob_slug = str(clob_row.get("market_slug") or clob_row.get("slug") or "")
    clob_tokens = _clob_up_down_tokens(clob_row.get("tokens"))
    if not (
        clob_condition == market_id
        and clob_slug == slug
        and clob_tokens == expected_tokens
    ):
        raise MicroLiveExecutionError("CLOB condition, slug, or token identity mismatches")


def _decode_provider_json(
    raw: bytes,
    label: str,
    *,
    maximum_bytes: int = MAX_RAW_JSON_BYTES,
) -> Any:
    if not isinstance(raw, bytes) or not raw or len(raw) > maximum_bytes:
        raise MicroLiveExecutionError(f"{label} raw byte limit is invalid or exceeded")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
        _validate_finite_json_tree(value)
        return value
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise MicroLiveExecutionError(f"{label} raw bytes are not strict JSON") from exc


def _raw_json_object(
    raw: Any,
    label: str,
    *,
    maximum_bytes: int = MAX_RAW_JSON_BYTES,
) -> tuple[dict[str, Any], str, str]:
    if not isinstance(raw, bytes) or not raw or len(raw) > maximum_bytes:
        raise MicroLiveExecutionError(f"{label} raw bytes are invalid")
    value = _decode_provider_json(raw, label, maximum_bytes=maximum_bytes)
    if not isinstance(value, Mapping):
        raise MicroLiveExecutionError(f"{label} is not a JSON object")
    return dict(value), raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def _stored_raw_json_object(
    raw_json: Any,
    expected_sha256: Any,
    label: str,
    *,
    maximum_bytes: int = MAX_RAW_JSON_BYTES,
) -> dict[str, Any]:
    if not isinstance(raw_json, str) or not raw_json:
        raise MicroLiveExecutionError(f"{label} raw JSON is invalid")
    raw = raw_json.encode("utf-8")
    if not _is_sha256(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise MicroLiveExecutionError(f"{label} SHA-256 mismatch")
    value = _decode_provider_json(raw, label, maximum_bytes=maximum_bytes)
    if not isinstance(value, Mapping):
        raise MicroLiveExecutionError(f"{label} is not a JSON object")
    return dict(value)


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _validate_finite_json_tree(value: Any, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ValueError("decoded JSON exceeds maximum nesting depth")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("decoded JSON contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_json_tree(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_finite_json_tree(item, depth=depth + 1)


def _gamma_market_rows(value: Any) -> list[dict[str, Any]]:
    rows: Any
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        wrapper = dict(value)
        if isinstance(wrapper.get("data"), list):
            rows = wrapper["data"]
        elif isinstance(wrapper.get("markets"), list):
            rows = wrapper["markets"]
        else:
            rows = [wrapper]
    else:
        raise MicroLiveExecutionError("Gamma market identity payload shape is invalid")
    if not all(isinstance(row, Mapping) for row in rows):
        raise MicroLiveExecutionError("Gamma market identity row is not an object")
    return [dict(row) for row in rows]


def _json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(
                value,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            )
            _validate_finite_json_tree(decoded)
        except (ValueError, json.JSONDecodeError):
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _gamma_up_down_tokens(row: Mapping[str, Any]) -> dict[str, str] | None:
    outcomes = _json_array(row.get("outcomes"))
    token_ids = _json_array(row.get("clobTokenIds"))
    if len(outcomes) != len(token_ids):
        return None
    return _up_down_tokens(zip(outcomes, token_ids, strict=True))


def _clob_up_down_tokens(value: Any) -> dict[str, str] | None:
    if not isinstance(value, list):
        return None
    pairs: list[tuple[Any, Any]] = []
    for token in value:
        if not isinstance(token, Mapping):
            return None
        row = dict(token)
        outcome = row.get("outcome") or row.get("name") or row.get("label")
        token_id = row.get("token_id") or row.get("tokenId") or row.get("asset_id")
        pairs.append((outcome, token_id))
    return _up_down_tokens(pairs)


def _up_down_tokens(pairs: Any) -> dict[str, str] | None:
    tokens: dict[str, str] = {}
    for outcome_value, token_value in pairs:
        outcome = str(outcome_value or "").strip().upper()
        token_id = str(token_value or "")
        if outcome not in {"UP", "DOWN"} or _TOKEN_ID.fullmatch(token_id) is None:
            return None
        if outcome in tokens:
            return None
        tokens[outcome] = token_id
    return tokens if set(tokens) == {"UP", "DOWN"} else None


def _validated_runtime_binding(
    *,
    signal: Mapping[str, Any],
    feature_row: Any,
    runtime: ResidualPromotionRuntime,
) -> dict[str, Any]:
    if not isinstance(feature_row, Mapping):
        raise MicroLiveExecutionError("candidate feature row is not an object")
    features = copy.deepcopy(dict(feature_row))
    _reject_forbidden_feature_keys(features)
    if not (
        features.get("market_id") == signal["market_id"]
        and features.get("market_family") == "btc_updown_15m"
        and features.get("horizon_ms") == 900_000
        and features.get("decision_ts") == signal["decision_ts_ms"]
    ):
        raise MicroLiveExecutionError("candidate feature row identity is invalid")
    raw = features.get("features")
    if not isinstance(raw, Mapping):
        raise MicroLiveExecutionError("candidate raw feature mapping is absent")
    try:
        raw_asks = {
            "UP": _numeric_decimal(raw.get("up_ask"), "UP feature ask"),
            "DOWN": _numeric_decimal(raw.get("down_ask"), "DOWN feature ask"),
        }
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MicroLiveExecutionError("candidate raw executable asks are invalid") from exc
    signal_asks = {
        side: _positive_decimal(
            dict(signal["executable_asks"])[side],
            f"{side} signal ask",
        )
        for side in ("UP", "DOWN")
    }
    if raw_asks != signal_asks:
        raise MicroLiveExecutionError("candidate signal ask does not match feature row")
    result = runtime.score_feature_row(
        features,
        observed_at_ts=int(signal["observed_at_ts_ms"]),
    )
    supplied_projection = {
        "action_values": signal["action_values"],
        "selected_action": signal["selected_action"],
        "model_scored": signal["model_scored"],
        "fail_closed": signal["fail_closed"],
        "fail_closed_reasons": signal["fail_closed_reasons"],
    }
    expected_projection = {
        name: result[name]
        for name in (
            "action_values",
            "selected_action",
            "model_scored",
            "fail_closed",
            "fail_closed_reasons",
        )
    }
    if not (
        canonical_json_sha256(supplied_projection)
        == canonical_json_sha256(expected_projection)
        and result.get("lineage_id") == LINEAGE_ID
        and result.get("candidate_id") == CANDIDATE_ID
        and result.get("market_id") == signal["market_id"]
        and result.get("decision_ts") == signal["decision_ts_ms"]
        and result.get("observed_at_ts") == signal["observed_at_ts_ms"]
        and result.get("manifest_sha256") == signal["candidate_bundle_sha256"]
        and result.get("outcomes_accessed") is False
        and result.get("settlement_accessed") is False
        and result.get("pnl_accessed") is False
    ):
        raise MicroLiveExecutionError("candidate signal does not match frozen runtime")
    return features


def _validated_stored_signal_and_feature(
    *,
    payload: Mapping[str, Any],
    expected_candidate_bundle_sha256: str,
    runtime: ResidualPromotionRuntime,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    VerifiedProviderFeatureEvidence,
]:
    """Re-decode exact stored input bytes before replaying semantic bindings."""

    raw_signal = _stored_raw_json_object(
        payload.get("raw_signal_payload_json"),
        payload.get("raw_signal_payload_sha256"),
        "stored candidate signal payload",
    )
    raw_feature = _stored_raw_json_object(
        payload.get("raw_feature_row_json"),
        payload.get("raw_feature_row_sha256"),
        "stored candidate feature row",
    )
    signal = _validated_candidate_signal(
        raw_signal,
        expected_candidate_bundle_sha256=expected_candidate_bundle_sha256,
    )
    features = _validated_runtime_binding(
        signal=signal,
        feature_row=raw_feature,
        runtime=runtime,
    )
    stored_signal = payload.get("signal_payload")
    stored_feature = payload.get("feature_row")
    if not (
        isinstance(stored_signal, Mapping)
        and isinstance(stored_feature, Mapping)
        and canonical_json_sha256(signal)
        == canonical_json_sha256(dict(stored_signal))
        and canonical_json_sha256(features)
        == canonical_json_sha256(dict(stored_feature))
    ):
        raise MicroLiveExecutionError(
            "stored raw candidate inputs do not match semantic audit copies"
        )
    raw_provider_value = payload.get("raw_provider_feature_evidence_jsonl")
    if not (
        isinstance(raw_provider_value, Mapping)
        and set(raw_provider_value) == set(PROVIDER_FEATURE_FILENAMES)
        and all(isinstance(value, str) for value in raw_provider_value.values())
    ):
        raise MicroLiveExecutionError(
            "stored raw provider feature evidence schema is invalid"
        )
    raw_provider = {
        name: str(raw_provider_value[name]).encode("utf-8")
        for name in PROVIDER_FEATURE_FILENAMES
    }
    try:
        verified_provider = verify_provider_feature_evidence(
            raw_evidence=raw_provider,
            signal=signal,
            feature_row=features,
        )
    except ProviderFeatureEvidenceError as exc:
        raise MicroLiveExecutionError(
            "stored provider feature evidence does not reconstruct input"
        ) from exc
    if not (
        payload.get("provider_feature_evidence_graph_sha256")
        == verified_provider.evidence_graph_sha256
        and payload.get("provider_feature_file_sha256")
        == verified_provider.file_sha256
        and payload.get("provider_reconstructed_feature_row_sha256")
        == verified_provider.reconstructed_feature_row_sha256
    ):
        raise MicroLiveExecutionError(
            "stored provider feature evidence hashes do not reconcile"
        )
    return signal, features, verified_provider


def build_provider_bound_feature_rows(
    raw_evidence: Mapping[str, bytes],
) -> tuple[dict[str, Any], ...]:
    """Rebuild causal feature rows from exactly five strict raw provider streams."""

    rows, _, _ = _decode_provider_feature_evidence(raw_evidence)
    return _reconstruct_provider_feature_rows(rows)


def verify_provider_feature_evidence(
    *,
    raw_evidence: Mapping[str, bytes],
    signal: Mapping[str, Any],
    feature_row: Mapping[str, Any],
) -> VerifiedProviderFeatureEvidence:
    """Bind one submitted feature row to exact causal provider bytes."""

    rows, raw_jsonl, file_sha256 = _decode_provider_feature_evidence(raw_evidence)
    _assert_decision_time_provider_prefix(rows=rows, signal=signal)
    feature_rows = _reconstruct_provider_feature_rows(rows)
    market_id = signal.get("market_id")
    decision_ts = signal.get("decision_ts_ms")
    matches = [
        row
        for row in feature_rows
        if row.get("market_id") == market_id
        and row.get("decision_ts") == decision_ts
    ]
    if len(matches) != 1:
        raise ProviderFeatureEvidenceError(
            "provider evidence does not reconstruct exactly one decision feature row"
        )
    reconstructed = matches[0]
    submitted = dict(feature_row)
    if canonical_json_sha256(reconstructed) != canonical_json_sha256(submitted):
        raise ProviderFeatureEvidenceError(
            "provider-reconstructed feature row does not match submitted feature bytes"
        )
    market_rows = rows[PROVIDER_FEATURE_FILENAMES[0]]
    if len(market_rows) != 1:
        raise ProviderFeatureEvidenceError(
            "provider evidence must contain exactly one BTC-15M market"
        )
    market = market_rows[0]
    slug = str(signal.get("slug") or "")
    try:
        start_ts = int(slug.rsplit("-", maxsplit=1)[1]) * 1_000
    except (IndexError, ValueError) as exc:
        raise ProviderFeatureEvidenceError("provider signal slug is invalid") from exc
    if not (
        market.get("market_id") == market_id
        and market.get("condition_id") == market_id
        and market.get("slug") == slug
        and market.get("market_family") == "btc_updown_15m"
        and market.get("market_start_ts") == start_ts
        and market.get("market_end_ts") == start_ts + 900_000
        and market.get("up_token_id") == signal.get("up_token_id")
        and market.get("down_token_id") == signal.get("down_token_id")
    ):
        raise ProviderFeatureEvidenceError(
            "provider market identity does not match executable signal"
        )
    reconstructed_sha256 = canonical_json_sha256(reconstructed)
    graph = {
        "market_id": market_id,
        "decision_ts_ms": decision_ts,
        "file_sha256": file_sha256,
        "reconstructed_feature_row_sha256": reconstructed_sha256,
    }
    return VerifiedProviderFeatureEvidence(
        file_sha256_items=tuple(sorted(file_sha256.items())),
        raw_jsonl_items=tuple(
            (name, raw_jsonl[name]) for name in PROVIDER_FEATURE_FILENAMES
        ),
        reconstructed_feature_row_sha256=reconstructed_sha256,
        evidence_graph_sha256=canonical_json_sha256(graph),
    )


def _assert_decision_time_provider_prefix(
    *,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    signal: Mapping[str, Any],
) -> None:
    decision_ts = signal.get("decision_ts_ms")
    if (
        isinstance(decision_ts, bool)
        or not isinstance(decision_ts, int)
        or decision_ts <= 0
    ):
        raise ProviderFeatureEvidenceError(
            "provider evidence decision timestamp is invalid"
        )
    market_rows = rows[PROVIDER_FEATURE_FILENAMES[0]]
    if len(market_rows) != 1:
        raise ProviderFeatureEvidenceError(
            "provider evidence must contain exactly one BTC-15M market"
        )
    market = market_rows[0]
    market_end_ts = market.get("market_end_ts")
    if (
        isinstance(market_end_ts, bool)
        or not isinstance(market_end_ts, int)
        or market_end_ts <= decision_ts
    ):
        raise ProviderFeatureEvidenceError(
            "provider evidence market is not open at decision time"
        )
    for field in ("trade_api_collection_ts", "trade_stream_ended_at_ts"):
        _assert_provider_timestamp_not_after_decision(
            market.get(field),
            field=field,
            decision_ts=decision_ts,
            required=False,
        )
    if market.get("trade_full_round_coverage_complete") is True:
        raise ProviderFeatureEvidenceError(
            "provider evidence contains pre-close terminal coverage status"
        )

    for filename in _DYNAMIC_PROVIDER_FILENAMES:
        for index, row in enumerate(rows[filename]):
            _assert_provider_timestamp_not_after_decision(
                row.get("available_at_ts"),
                field=f"{filename}[{index}].available_at_ts",
                decision_ts=decision_ts,
                required=True,
            )
            for field in _PROVIDER_DECISION_TIME_FIELDS - {"available_at_ts"}:
                if field in row:
                    _assert_provider_timestamp_not_after_decision(
                        row.get(field),
                        field=f"{filename}[{index}].{field}",
                        decision_ts=decision_ts,
                        required=False,
                    )
            if any(
                row.get(field) is True
                for field in _PREMARKET_TERMINAL_STATUS_FIELDS
            ):
                raise ProviderFeatureEvidenceError(
                    "provider evidence contains pre-close terminal coverage status"
                )


def _assert_provider_timestamp_not_after_decision(
    value: Any,
    *,
    field: str,
    decision_ts: int,
    required: bool,
) -> None:
    if value is None and not required:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderFeatureEvidenceError(
            f"provider evidence timestamp is invalid: {field}"
        )
    if value > decision_ts:
        raise ProviderFeatureEvidenceError(
            f"provider evidence contains post-decision data: {field}"
        )


def _decode_provider_feature_evidence(
    raw_evidence: Mapping[str, bytes],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], dict[str, str]]:
    if not isinstance(raw_evidence, Mapping) or set(raw_evidence) != set(
        PROVIDER_FEATURE_FILENAMES
    ):
        raise ProviderFeatureEvidenceError(
            "provider feature evidence schema is not exact"
        )
    _require_provider_byte_limits(raw_evidence)
    rows: dict[str, list[dict[str, Any]]] = {}
    raw_jsonl: dict[str, str] = {}
    file_sha256: dict[str, str] = {}
    for name in PROVIDER_FEATURE_FILENAMES:
        raw = raw_evidence.get(name)
        if not isinstance(raw, bytes):
            raise ProviderFeatureEvidenceError(
                f"provider feature evidence is not raw bytes: {name}"
            )
        if len(raw) > MAX_PROVIDER_STREAM_BYTES:
            raise ProviderFeatureEvidenceError(
                f"provider feature evidence exceeds stream byte limit: {name}"
            )
        decoded_rows, text = _strict_provider_jsonl(raw, name)
        if name in _REQUIRED_NONEMPTY_PROVIDER_FILES and not decoded_rows:
            raise ProviderFeatureEvidenceError(
                f"required provider feature evidence is empty: {name}"
            )
        for row in decoded_rows:
            _assert_outcome_blind_provider_row(row, path=name)
        rows[name] = decoded_rows
        raw_jsonl[name] = text
        file_sha256[name] = hashlib.sha256(raw).hexdigest()
    return rows, raw_jsonl, file_sha256


def _reconstruct_provider_feature_rows(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    config = PolymarketCorpusBuildConfig(
        input_dir=Path(__file__).resolve().parent,
        output_dir=Path(__file__).resolve().parent,
        market_families=("btc_updown_15m",),
        sample_interval_seconds={"btc_updown_15m": 300},
        min_time_to_close_seconds=0,
        include_trade_labels=True,
        include_settlement_labels=False,
        overwrite_existing=False,
    )
    try:
        markets = _normalize_markets(
            [dict(row) for row in rows[PROVIDER_FEATURE_FILENAMES[0]]],
            config,
        )
        if len(markets) != 1:
            raise ProviderFeatureEvidenceError(
                "provider feature evidence market population is not exactly one"
            )
        books = _normalize_book_snapshots(
            [dict(row) for row in rows[PROVIDER_FEATURE_FILENAMES[1]]],
            markets,
        )
        trades = _normalize_trades(
            [dict(row) for row in rows[PROVIDER_FEATURE_FILENAMES[2]]],
            markets,
        )
        candles = _normalize_candles(
            [dict(row) for row in rows[PROVIDER_FEATURE_FILENAMES[3]]]
        )
        chainlink = _normalize_chainlink_prices(
            [dict(row) for row in rows[PROVIDER_FEATURE_FILENAMES[4]]]
        )
        feature_rows = build_polymarket_corpus_feature_rows(
            markets=markets,
            book_snapshots=books,
            trades=trades,
            btc_candles=candles,
            chainlink_prices=chainlink,
            config=config,
        )
    except ProviderFeatureEvidenceError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderFeatureEvidenceError(
            "provider feature evidence cannot reconstruct causal features"
        ) from exc
    return tuple(row.to_dict() for row in feature_rows)


def _strict_provider_jsonl(
    raw: bytes,
    label: str,
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(raw, bytes) or len(raw) > MAX_PROVIDER_STREAM_BYTES:
        raise ProviderFeatureEvidenceError(
            f"provider feature evidence exceeds stream byte limit: {label}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProviderFeatureEvidenceError(
            f"provider feature evidence is not UTF-8: {label}"
        ) from exc
    if not raw:
        return [], text
    lines = text.splitlines()
    _require_provider_row_count(len(lines), label=label)
    if not lines or any(not line.strip() for line in lines):
        raise ProviderFeatureEvidenceError(
            f"provider feature evidence contains an empty JSONL row: {label}"
        )
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            )
            _validate_finite_json_tree(value)
        except (ValueError, json.JSONDecodeError, RecursionError) as exc:
            raise ProviderFeatureEvidenceError(
                f"provider feature evidence is not strict JSONL: {label}"
            ) from exc
        if not isinstance(value, dict):
            raise ProviderFeatureEvidenceError(
                f"provider feature evidence row is not an object: {label}"
            )
        rows.append(value)
    return rows, text


def _assert_outcome_blind_provider_row(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProviderFeatureEvidenceError(
                    "provider feature evidence key is not a string"
                )
            lowered = key.lower()
            if lowered in _FORBIDDEN_PROVIDER_RESULT_KEYS or any(
                token in lowered
                for token in _FORBIDDEN_PROVIDER_RESULT_KEY_TOKENS
            ):
                raise ProviderFeatureEvidenceError(
                    "provider feature evidence contains result-bearing field: "
                    f"{path}.{key}"
                )
            if lowered in _LOCKED_FALSE_PROVIDER_KEYS and child is not False:
                raise ProviderFeatureEvidenceError(
                    "provider feature evidence safety field is not false: "
                    f"{path}.{key}"
                )
            if (
                lowered
                in {"outcomes_accessed", "settlement_accessed", "pnl_accessed"}
                and child is not False
            ):
                raise ProviderFeatureEvidenceError(
                    "provider feature evidence accessed a forbidden stream: "
                    f"{path}.{key}"
                )
            _assert_outcome_blind_provider_row(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_outcome_blind_provider_row(child, path=f"{path}[{index}]")


def _reject_forbidden_feature_keys(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise MicroLiveExecutionError("candidate feature key is not a string")
            lowered = key.lower()
            if any(token in lowered for token in _FORBIDDEN_FEATURE_KEY_TOKENS):
                raise MicroLiveExecutionError(
                    f"candidate feature row contains forbidden field: {path}{key}"
                )
            _reject_forbidden_feature_keys(nested, f"{path}{key}.")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_feature_keys(nested, f"{path}{index}.")


def _numeric_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise MicroLiveExecutionError(f"{label} is invalid")
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed <= 0 or parsed >= 1:
        raise MicroLiveExecutionError(f"{label} is outside the binary range")
    return parsed


def _positive_decimal(value: Any, label: str) -> Decimal:
    parsed = _decimal(value, label)
    if parsed <= 0:
        raise MicroLiveExecutionError(f"{label} must be positive")
    return parsed


def _nonnegative_decimal(value: Any, label: str) -> Decimal:
    parsed = _decimal(value, label)
    if parsed < 0:
        raise MicroLiveExecutionError(f"{label} must be nonnegative")
    return parsed


def _decimal(value: Any, label: str) -> Decimal:
    if not isinstance(value, str):
        raise MicroLiveExecutionError(f"{label} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise MicroLiveExecutionError(f"{label} is invalid") from exc
    if not parsed.is_finite() or str(parsed) != value:
        raise MicroLiveExecutionError(f"{label} is not canonical")
    return parsed


def _require_positive_timestamp(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MicroLiveExecutionError(f"{label} timestamp is invalid")
    return value


def _require_bounded_state_bytes(raw_state: Any) -> bytes:
    if (
        not isinstance(raw_state, bytes)
        or not raw_state
        or len(raw_state) > MAX_RESTORED_STATE_BYTES
    ):
        raise MicroLiveExecutionError("micro-live state exceeds byte limit")
    return raw_state


def _require_event_count(event_count: Any) -> int:
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 0
        or event_count > MAX_EVENT_COUNT
    ):
        raise MicroLiveExecutionError("micro-live event count exceeds limit")
    return event_count


def _read_bounded_stable_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    """Read one immutable-size regular file without following symlinks.

    Size is checked from the open descriptor before allocation, at most
    ``maximum_bytes + 1`` bytes are ever retained, and descriptor metadata is
    compared before/after the read so replacement or mutation fails closed.
    """

    if not (
        isinstance(maximum_bytes, int)
        and not isinstance(maximum_bytes, bool)
        and maximum_bytes >= 0
    ):
        raise MicroLiveExecutionError(f"{label} byte limit is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    else:
        try:
            path_before_open = os.lstat(path)
        except OSError as exc:
            raise MicroLiveExecutionError(f"{label} is unreadable") from exc
        if stat.S_ISLNK(path_before_open.st_mode):
            raise MicroLiveExecutionError(f"{label} symbolic link is forbidden")
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MicroLiveExecutionError(f"{label} is unreadable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MicroLiveExecutionError(f"{label} is not a regular file")
        if before.st_size < 0 or before.st_size > maximum_bytes:
            raise MicroLiveExecutionError(f"{label} exceeds byte limit")
        chunks: list[bytes] = []
        retained = 0
        while retained <= maximum_bytes:
            chunk = os.read(descriptor, min(1_048_576, maximum_bytes + 1 - retained))
            if not chunk:
                break
            chunks.append(chunk)
            retained += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        stable_fields_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_fields_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise MicroLiveExecutionError(
                f"{label} path changed during bounded read"
            ) from exc
        path_identity_after = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if not (
            len(raw) <= maximum_bytes
            and len(raw) == before.st_size
            and stable_fields_after == stable_fields_before
            and path_identity_after == stable_fields_before
        ):
            raise MicroLiveExecutionError(f"{label} changed during bounded read")
        return raw
    except OSError as exc:
        raise MicroLiveExecutionError(f"{label} bounded read failed") from exc
    finally:
        os.close(descriptor)


def _routine_event_limit() -> int:
    if not (
        isinstance(EVENT_RECOVERY_RESERVE, int)
        and not isinstance(EVENT_RECOVERY_RESERVE, bool)
        and 1 <= EVENT_RECOVERY_RESERVE < MAX_EVENT_COUNT
    ):
        raise MicroLiveExecutionError(
            "micro-live recovery event reserve is invalid"
        )
    return MAX_EVENT_COUNT - EVENT_RECOVERY_RESERVE


def _require_provider_byte_limits(raw_evidence: Any) -> None:
    if not isinstance(raw_evidence, Mapping) or any(
        not isinstance(raw_evidence.get(name), bytes)
        for name in PROVIDER_FEATURE_FILENAMES
    ):
        raise ProviderFeatureEvidenceError(
            "provider feature evidence contains a non-byte stream"
        )
    if any(
        len(raw_evidence[name]) > MAX_PROVIDER_STREAM_BYTES
        for name in PROVIDER_FEATURE_FILENAMES
    ):
        raise ProviderFeatureEvidenceError(
            "provider feature evidence exceeds stream byte limit"
        )
    if (
        sum(len(raw_evidence[name]) for name in PROVIDER_FEATURE_FILENAMES)
        > MAX_PROVIDER_AGGREGATE_BYTES
    ):
        raise ProviderFeatureEvidenceError(
            "provider feature evidence exceeds aggregate byte limit"
        )


def _require_provider_row_count(row_count: Any, *, label: str) -> int:
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 0
        or row_count > MAX_PROVIDER_ROWS_PER_STREAM
    ):
        raise ProviderFeatureEvidenceError(
            f"provider feature evidence exceeds row limit: {label}"
        )
    return row_count


def _state_generation(raw_state: bytes) -> int:
    _require_bounded_state_bytes(raw_state)
    value, _, _ = _raw_json_object(
        raw_state,
        "micro-live journal state",
        maximum_bytes=MAX_RESTORED_STATE_BYTES,
    )
    generation = value.get("journal_generation")
    if (
        value.get("schema_version") != STATE_SCHEMA_VERSION
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise MicroLiveExecutionError("micro-live journal state generation is invalid")
    return generation


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise MicroLiveExecutionError(f"{label} SHA-256 is invalid")
    return str(value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


__all__ = [
    "IMPLEMENTATION_REPOSITORY_PATH",
    "MicroLiveExecutionError",
    "MicroLiveExecutor",
    "MicroLiveOrderTransport",
    "PRODUCTION_NUMPY_VERSION",
    "PRODUCTION_PYTHON_IMPLEMENTATION",
    "PRODUCTION_PYTHON_VERSION",
    "PRODUCTION_SCIPY_VERSION",
    "PRODUCTION_XGBOOST_VERSION",
    "PROVIDER_FEATURE_FILENAMES",
    "ProviderFeatureEvidenceError",
    "SIGNAL_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "VerifiedProviderFeatureEvidence",
    "build_provider_bound_feature_rows",
    "create_micro_live_executor",
    "verify_dispatchable_outbox_request",
    "verify_provider_feature_evidence",
]
