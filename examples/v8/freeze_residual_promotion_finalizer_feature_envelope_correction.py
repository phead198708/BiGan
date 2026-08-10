"""Freeze the outcome-blind finalizer feature-envelope correction."""

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
from bigan.v8.polymarket.residual_promotion_v1 import (  # noqa: E402
    CANDIDATE_ID,
    LINEAGE_ID,
)

CONFIG = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "BTC-15M-cost-aware-market-residual-promotion-v1"
)
FINALIZER = ROOT / "src/bigan/v8/polymarket/residual_promotion_finalization.py"
EVALUATOR = ROOT / "src/bigan/v8/polymarket/residual_promotion_evaluation.py"
OLD_CONTRACT = CONFIG / "promotion_evaluation_execution_contract_v2.json"
OLD_TEMPLATE = CONFIG / "promotion_outcome_evaluation_authorization_template_v2.json"
OLD_PREFLIGHT = CONFIG / "micro_live_preapproval_contract_v2.json"
CORRECTION = CONFIG / "finalization_feature_envelope_correction.json"
NEW_CONTRACT = CONFIG / "promotion_evaluation_execution_contract_v3.json"
NEW_TEMPLATE = CONFIG / "promotion_outcome_evaluation_authorization_template_v3.json"
RELEASE_V3 = (
    CONFIG / "micro_live_preapproval_contract_v3.json",
    CONFIG / "micro_live_preapproval_preflight_report_v3.json",
    CONFIG / "micro_live_authorization_template_v3.json",
)


def main() -> int:
    for path in (CORRECTION, NEW_CONTRACT, NEW_TEMPLATE, *RELEASE_V3):
        if path.exists():
            raise FileExistsError(f"correction artifact already exists: {path.name}")
    for path in (OLD_CONTRACT, OLD_TEMPLATE, OLD_PREFLIGHT):
        _verified_json(path)

    created_at = "2026-08-10T16:40:00+00:00"
    correction = {
        "schema_version": (
            "bigan-btc-15m-residual-promotion-finalization-"
            "feature-envelope-correction-v1"
        ),
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "outcome_blind_observation": {
            "attempt_index": 2,
            "attempt_hash": (
                "05bf716040e913d2a566334b78cabd6f50c39d82f89e26018802da47960e31e0"
            ),
            "capture_report_sha256": (
                "c56443908df6781d058cf6d7f020dde7829120d22810fb6f41cd056859b223d5"
            ),
            "quality_valid": True,
            "feature_row_schema": {
                "outer_market_id": True,
                "outer_decision_ts": True,
                "execution_feature_envelope": "features",
                "required_execution_fields": [
                    "up_ask",
                    "up_bid",
                    "up_liquidity_depth",
                    "down_ask",
                    "down_bid",
                    "down_liquidity_depth",
                ],
            },
            "outcomes_accessed": False,
            "settlement_accessed": False,
            "pnl_accessed": False,
        },
        "defect": (
            "finalizer_read_execution_fields_from_feature_row_outer_object_"
            "instead_of_frozen_features_envelope"
        ),
        "correction": {
            "collector_quality_eligibility_changed": False,
            "population_selection_changed": False,
            "model_prediction_behavior_changed": False,
            "execution_values_changed": False,
            "statistical_gate_changed": False,
            "finalizer_reads_frozen_features_envelope": True,
            "required_execution_fields_unchanged": True,
            "missing_or_nonnumeric_execution_fields_fail_closed": True,
        },
        "supersedes_for_future_execution_only": {
            "evaluation_execution_contract": _descriptor(OLD_CONTRACT),
            "outcome_evaluation_authorization_template": _descriptor(OLD_TEMPLATE),
            "micro_live_preapproval_contract": _descriptor(OLD_PREFLIGHT),
        },
        "prior_artifacts_rewritten": False,
        "fresh_outcomes_accessed": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(CORRECTION, correction)

    execution = _verified_json(OLD_CONTRACT)
    execution.update(
        {
            "created_at": created_at,
            "contract_revision": "feature_envelope_reconciliation_v3",
            "implementation": _descriptor(EVALUATOR),
            "finalization_implementation": _descriptor(FINALIZER),
            "finalization_feature_envelope_correction": _descriptor(CORRECTION),
            "supersedes_execution_contract": _descriptor(OLD_CONTRACT),
        }
    )
    _write_frozen_json(NEW_CONTRACT, execution)
    validate_evaluation_execution_contract(execution, repository_root=ROOT)

    authorization = _verified_json(OLD_TEMPLATE)
    authorization.update(
        {
            "authorization_template_revision": (
                "feature_envelope_reconciliation_v3"
            ),
            "execution_contract": _descriptor(NEW_CONTRACT),
            "finalization_feature_envelope_correction": _descriptor(CORRECTION),
            "supersedes_authorization_template": _descriptor(OLD_TEMPLATE),
            "evaluation_exactly_once_authorized": False,
            "fresh_outcome_access_authorized": False,
            "official_settlement_ingestion_authorized": False,
            "template_is_executable": False,
            "safety": dict(SAFETY),
        }
    )
    _write_frozen_json(NEW_TEMPLATE, authorization)

    release = freeze_release_readiness_contract(
        repository_root=ROOT,
        created_at=created_at,
    )
    print(
        json.dumps(
            {
                "correction": _descriptor(CORRECTION),
                "execution_contract_v3": _descriptor(NEW_CONTRACT),
                "authorization_template_v3": _descriptor(NEW_TEMPLATE),
                "release_readiness_v3": release,
                "fresh_outcomes_accessed": False,
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
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
