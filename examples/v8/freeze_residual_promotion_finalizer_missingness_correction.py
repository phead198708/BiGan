"""Freeze the outcome-blind native-missingness finalizer correction."""

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
OLD_CONTRACT = CONFIG / "promotion_evaluation_execution_contract.json"
OLD_TEMPLATE = CONFIG / "promotion_outcome_evaluation_authorization_template.json"
OLD_PREFLIGHT = CONFIG / "micro_live_preapproval_contract.json"
CORRECTION = CONFIG / "finalization_native_missingness_correction.json"
NEW_CONTRACT = CONFIG / "promotion_evaluation_execution_contract_v2.json"
NEW_TEMPLATE = CONFIG / "promotion_outcome_evaluation_authorization_template_v2.json"


def main() -> int:
    for path in (CORRECTION, NEW_CONTRACT, NEW_TEMPLATE):
        if path.exists():
            raise FileExistsError(f"correction artifact already exists: {path.name}")
    for path in (OLD_CONTRACT, OLD_TEMPLATE, OLD_PREFLIGHT):
        _verified_json(path)
    created_at = "2026-08-10T16:10:00+00:00"
    correction = {
        "schema_version": (
            "bigan-btc-15m-residual-promotion-finalization-"
            "native-missingness-correction-v1"
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
            "missing_feature_count": 12,
            "missing_feature_counts": {
                "opposite_recent_trade_volume": 4,
                "selected_minus_opposite_recent_trade_volume": 4,
                "selected_recent_trade_volume": 4,
            },
            "missing_values_encoded_as_zero": False,
            "outcomes_accessed": False,
            "settlement_accessed": False,
            "pnl_accessed": False,
        },
        "defect": (
            "collector_quality_contract_allows_native_missing_values_but_"
            "finalizer_required_missing_feature_count_equal_zero"
        ),
        "correction": {
            "collector_quality_eligibility_changed": False,
            "population_selection_changed": False,
            "model_prediction_behavior_changed": False,
            "statistical_gate_changed": False,
            "native_missing_values_remain_nan": True,
            "missing_values_encoded_as_zero": False,
            "finalizer_requires_nonnegative_integer_missing_counts": True,
            "finalizer_requires_missing_count_sum_reconciliation": True,
            "finalizer_requires_all_existing_quality_observations_true": True,
        },
        "supersedes_for_future_execution_only": {
            "evaluation_execution_contract": _descriptor(OLD_CONTRACT),
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
            "contract_revision": "native_missingness_reconciliation_v2",
            "finalization_implementation": _descriptor(FINALIZER),
            "finalization_correction": _descriptor(CORRECTION),
            "supersedes_execution_contract": _descriptor(OLD_CONTRACT),
        }
    )
    _write_frozen_json(NEW_CONTRACT, execution)
    validate_evaluation_execution_contract(execution, repository_root=ROOT)

    authorization = _verified_json(OLD_TEMPLATE)
    authorization.update(
        {
            "authorization_template_revision": (
                "native_missingness_reconciliation_v2"
            ),
            "execution_contract": _descriptor(NEW_CONTRACT),
            "finalization_correction": _descriptor(CORRECTION),
            "supersedes_authorization_template": _descriptor(OLD_TEMPLATE),
            "evaluation_exactly_once_authorized": False,
            "fresh_outcome_access_authorized": False,
            "official_settlement_ingestion_authorized": False,
            "service_root_id": (
                "BTC-15M-cost-aware-market-residual-promotion-v1-"
                "coverage-corrected-v3"
            ),
            "template_is_executable": False,
            "safety": dict(SAFETY),
        }
    )
    _write_frozen_json(NEW_TEMPLATE, authorization)
    print(
        json.dumps(
            {
                "correction": _descriptor(CORRECTION),
                "execution_contract_v2": _descriptor(NEW_CONTRACT),
                "authorization_template_v2": _descriptor(NEW_TEMPLATE),
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
