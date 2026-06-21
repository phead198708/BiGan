"""Paper-only execution harness for v8."""

from bigan.v8.paper.contracts import (
    DEFAULT_PAPER_CREATED_AT,
    PAPER_ARTIFACT_FILENAMES,
    PAPER_TRADING_HARNESS_PHASE,
    PaperDegradationConfig,
    PaperFill,
    PaperHarnessConfig,
    PaperLedgerEntry,
    PaperOrder,
    PaperPositionSnapshot,
    PaperRunReport,
    PaperSide,
    PaperTradingError,
    canonical_payload_sha256,
    stream_sha256,
)
from bigan.v8.paper.engine import (
    PaperHarnessResult,
    paper_fills_to_live_observations,
    run_paper_trading_harness,
)
from bigan.v8.paper.ledger import PaperLedger
from bigan.v8.paper.replay import synthetic_phase4_decisions

__all__ = [
    "DEFAULT_PAPER_CREATED_AT",
    "PAPER_ARTIFACT_FILENAMES",
    "PAPER_TRADING_HARNESS_PHASE",
    "PaperDegradationConfig",
    "PaperFill",
    "PaperHarnessConfig",
    "PaperHarnessResult",
    "PaperLedger",
    "PaperLedgerEntry",
    "PaperOrder",
    "PaperPositionSnapshot",
    "PaperRunReport",
    "PaperSide",
    "PaperTradingError",
    "canonical_payload_sha256",
    "paper_fills_to_live_observations",
    "run_paper_trading_harness",
    "stream_sha256",
    "synthetic_phase4_decisions",
]
