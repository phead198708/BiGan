"""Freeze an additive operational-rollback preapproval reconciliation.

This does not rewrite the frozen v5 preflight and never grants Phase 6 or live
authorization.  It only proves that the subsequently frozen operational
rollback report satisfies the already-frozen v5 contract.
"""

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
from bigan.v8.polymarket.residual_promotion_release_readiness import (  # noqa: E402
    IMPLEMENTATION_REPOSITORY_PATH,
    assess_micro_live_preapproval,
    validate_release_readiness_contract,
)
from bigan.v8.polymarket.residual_promotion_v1 import (  # noqa: E402
    CANDIDATE_ID,
    LINEAGE_ID,
)

CONFIG = ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID
CONTRACT = CONFIG / "micro_live_preapproval_contract_v5.json"
HISTORICAL_PREFLIGHT = CONFIG / "micro_live_preapproval_preflight_report_v5.json"
OPERATIONAL_ROLLBACK = CONFIG / "operational_rollback_drill_report.json"
OUTPUT = CONFIG / "micro_live_preapproval_operational_reconciliation_v1.json"
GENERATOR = Path(__file__).resolve()
SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-operational-preapproval-reconciliation-v1"
)
CREATED_AT = "2026-08-10T21:10:40Z"
EXPECTED_V5_CHECKS = {
    "fresh_confirmation": False,
    "functional_rollback": True,
    "operational_rollback": False,
    "phase6_zero_capital_pipeline": False,
    "runtime_parity": True,
    "security_review": False,
    "shadow_stability_and_monitoring": False,
}
EXPECTED_RECONCILED_CHECKS = {
    **EXPECTED_V5_CHECKS,
    "operational_rollback": True,
}
REMAINING_CHECKS = [
    "fresh_confirmation",
    "phase6_zero_capital_pipeline",
    "security_review",
    "shadow_stability_and_monitoring",
]


def main() -> int:
    if OUTPUT.exists() or OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").exists():
        raise FileExistsError("operational preapproval reconciliation already exists")

    contract = _verified_json(CONTRACT)
    historical_preflight = _verified_json(HISTORICAL_PREFLIGHT)
    operational_rollback = _verified_json(OPERATIONAL_ROLLBACK)
    validate_release_readiness_contract(
        contract,
        repository_root=ROOT,
        expected_implementation_sha256=sha256_file(ROOT / IMPLEMENTATION_REPOSITORY_PATH),
    )
    if historical_preflight.get("technical_checks") != EXPECTED_V5_CHECKS:
        raise ValueError("frozen v5 preflight checks drifted")

    assessment = assess_micro_live_preapproval(
        contract=contract,
        evidence={"operational_rollback": operational_rollback},
        created_at=CREATED_AT,
    )
    if assessment.get("technical_checks") != EXPECTED_RECONCILED_CHECKS:
        raise ValueError("operational rollback evidence did not reconcile")
    if assessment.get("failed_or_missing_checks") != REMAINING_CHECKS:
        raise ValueError("remaining release gates changed unexpectedly")
    if assessment.get("ready_to_request_micro_live_approval") is not False:
        raise ValueError("preapproval must remain fail closed")

    report = {
        "schema_version": SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": CREATED_AT,
        "generator": _descriptor(GENERATOR),
        "preapproval_contract": _descriptor(CONTRACT),
        "historical_v5_preflight_preserved": _descriptor(HISTORICAL_PREFLIGHT),
        "operational_rollback_evidence": _descriptor(OPERATIONAL_ROLLBACK),
        "completed_since_v5": ["operational_rollback"],
        "remaining_release_checks": REMAINING_CHECKS,
        "assessment": assessment,
        "historical_artifacts_modified": False,
        "candidate_behavior_changed": False,
        "fresh_population_accessed": False,
        "fresh_outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "paper_run_started": False,
        "phase6_zero_capital_authorized": False,
        "security_review_passed": False,
        "micro_live_authorized": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(OUTPUT, report)
    print(json.dumps(_descriptor(OUTPUT), indent=2, sort_keys=True))
    return 0


def _verified_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not path.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path)
    ):
        raise ValueError(f"frozen artifact sidecar mismatch: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen JSON root must be an object")
    return payload


def _descriptor(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def _write_frozen_json(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
