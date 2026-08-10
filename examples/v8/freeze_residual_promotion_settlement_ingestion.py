"""Freeze one-shot official-settlement ingestion without opening outcomes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.challenge_development_lane import sha256_file  # noqa: E402
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY  # noqa: E402
from bigan.v8.polymarket.residual_promotion_evaluation import (  # noqa: E402
    validate_evaluation_execution_contract,
)
from bigan.v8.polymarket.residual_promotion_release_readiness import (  # noqa: E402
    freeze_release_readiness_contract,
)
from bigan.v8.polymarket.residual_promotion_settlement import (  # noqa: E402
    DEFAULT_MAX_WORKERS,
    PROVIDER_ATTEMPTS,
    SETTLEMENT_CONTRACT_SCHEMA_VERSION,
    validate_settlement_ingestion_contract,
)
from bigan.v8.polymarket.residual_promotion_v1 import (  # noqa: E402
    CANDIDATE_ID,
    LINEAGE_ID,
    TARGET_MARKETS,
)

CONFIG = ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID
EVALUATOR = ROOT / "src/bigan/v8/polymarket/residual_promotion_evaluation.py"
SETTLEMENT = ROOT / "src/bigan/v8/polymarket/residual_promotion_settlement.py"
SETTLEMENT_CLI = ROOT / "examples/v8/run_residual_promotion_settlement.py"
OLD_EXECUTION = CONFIG / "promotion_evaluation_execution_contract_v3.json"
NEW_EXECUTION = CONFIG / "promotion_evaluation_execution_contract_v4.json"
SETTLEMENT_CONTRACT = CONFIG / "promotion_settlement_ingestion_contract.json"
OLD_AUTHORIZATION = CONFIG / "promotion_outcome_evaluation_authorization_template_v3.json"
NEW_AUTHORIZATION = CONFIG / "promotion_outcome_evaluation_authorization_template_v4.json"
RELEASE_V4 = (
    CONFIG / "micro_live_preapproval_contract_v4.json",
    CONFIG / "micro_live_preapproval_preflight_report_v4.json",
    CONFIG / "micro_live_authorization_template_v4.json",
)


def main() -> int:
    outputs = (NEW_EXECUTION, SETTLEMENT_CONTRACT, NEW_AUTHORIZATION, *RELEASE_V4)
    if any(path.exists() for path in outputs):
        raise FileExistsError("settlement hardening output already exists")
    old_execution = _verified_json(OLD_EXECUTION)
    old_authorization = _verified_json(OLD_AUTHORIZATION)
    created_at = "2026-08-10T17:05:54+00:00"

    execution = {
        **old_execution,
        "created_at": created_at,
        "contract_revision": "official_settlement_ingestion_v4",
        "implementation": _descriptor(EVALUATOR),
        "settlement_ingestion_implementation": _descriptor(SETTLEMENT),
        "supersedes_execution_contract": _descriptor(OLD_EXECUTION),
    }
    _write_frozen_json(NEW_EXECUTION, execution)
    validate_evaluation_execution_contract(execution, repository_root=ROOT)

    settlement_contract = {
        "schema_version": SETTLEMENT_CONTRACT_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "target_market_count": TARGET_MARKETS,
        "provider_attempts": PROVIDER_ATTEMPTS,
        "max_workers": DEFAULT_MAX_WORKERS,
        "official_settlement_only": True,
        "inferred_settlement_allowed": False,
        "unresolved_market_allowed": False,
        "exact_population_order_required": True,
        "outcome_access_claim_before_provider_call": True,
        "attempt_consumed_on_any_provider_call": True,
        "partial_failure_terminalizes_lineage": True,
        "rerun_allowed": False,
        "automatic_evaluation_or_promotion": False,
        "implementation": _descriptor(SETTLEMENT),
        "cli": _descriptor(SETTLEMENT_CLI),
        "evaluation_execution_contract": _descriptor(NEW_EXECUTION),
        "finalization_implementation": _descriptor(
            ROOT / "src/bigan/v8/polymarket/residual_promotion_finalization.py"
        ),
        "provider_implementation": _descriptor(
            ROOT / "src/bigan/v8/polymarket/recorder/public_provider.py"
        ),
        "resolution_normalization_implementation": _descriptor(
            ROOT / "src/bigan/v8/polymarket/recorder/resolution.py"
        ),
        "recorder_config_implementation": _descriptor(
            ROOT / "src/bigan/v8/polymarket/recorder/async_settlement.py"
        ),
        "fresh_outcomes_accessed_when_frozen": False,
        "settlement_accessed_when_frozen": False,
        "pnl_accessed_when_frozen": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(SETTLEMENT_CONTRACT, settlement_contract)
    validate_settlement_ingestion_contract(settlement_contract, repository_root=ROOT)

    authorization = {
        **old_authorization,
        "authorization_template_revision": "official_settlement_ingestion_v4",
        "execution_contract": _descriptor(NEW_EXECUTION),
        "settlement_ingestion_contract": _descriptor(SETTLEMENT_CONTRACT),
        "supersedes_authorization_template": _descriptor(OLD_AUTHORIZATION),
        "settlement_provider_attempts": PROVIDER_ATTEMPTS,
        "settlement_max_workers": DEFAULT_MAX_WORKERS,
        "fresh_outcome_access_authorized": False,
        "official_settlement_ingestion_authorized": False,
        "outcome_access_claim_authorized": False,
        "evaluation_exactly_once_authorized": False,
        "authorization_record_executable": False,
        "template_is_executable": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(NEW_AUTHORIZATION, authorization)

    release = freeze_release_readiness_contract(
        repository_root=ROOT,
        created_at=created_at,
    )
    print(
        json.dumps(
            {
                "evaluation_execution_contract_v4": _descriptor(NEW_EXECUTION),
                "settlement_ingestion_contract": _descriptor(SETTLEMENT_CONTRACT),
                "outcome_authorization_template_v4": _descriptor(NEW_AUTHORIZATION),
                "release_readiness_v4": release,
                "fresh_outcomes_accessed": False,
                "settlement_accessed": False,
                "pnl_accessed": False,
                "safety": dict(SAFETY),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _verified_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not path.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path)
    ):
        raise ValueError(f"frozen artifact sidecar mismatch: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen JSON root must be an object")
    return value


def _descriptor(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def _write_frozen_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError(f"frozen artifact already exists: {path.name}")
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
