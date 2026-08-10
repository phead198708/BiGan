"""Freeze a separate post-confirmation Phase 6 zero-capital authorization lane."""

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
    PHASE6_AUTHORIZATION_SCHEMA_VERSION,
    PHASE6_AUTHORIZATION_TEMPLATE_SCHEMA_VERSION,
    freeze_release_readiness_contract,
)
from bigan.v8.polymarket.residual_promotion_v1 import (  # noqa: E402
    CANDIDATE_ID,
    LINEAGE_ID,
)

CONFIG = ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID
CANDIDATE_BUNDLE = CONFIG / "candidate_bundle/bundle_manifest.json"
COLLECTION_AUTHORIZATION = CONFIG / "manual_collection_authorization_v3.json"
PHASE6_TEMPLATE = CONFIG / "phase6_zero_capital_authorization_template.json"
RELEASE_V5 = (
    CONFIG / "micro_live_preapproval_contract_v5.json",
    CONFIG / "micro_live_preapproval_preflight_report_v5.json",
    CONFIG / "micro_live_authorization_template_v5.json",
)


def main() -> int:
    for path in (PHASE6_TEMPLATE, *RELEASE_V5):
        if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
            raise FileExistsError(f"Phase 6 correction artifact exists: {path.name}")
    _verified_json(CANDIDATE_BUNDLE)
    collection_authorization = _verified_json(COLLECTION_AUTHORIZATION)
    if collection_authorization.get("authorization_scope") != (
        "zero_capital_read_only_outcome_blind_capture_only"
    ):
        raise ValueError("collection authorization scope changed unexpectedly")
    created_at = "2026-08-10T17:30:00+00:00"
    template = {
        "schema_version": PHASE6_AUTHORIZATION_TEMPLATE_SCHEMA_VERSION,
        "authorized_record_schema_version": PHASE6_AUTHORIZATION_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "authorization_scope": "post_confirmation_phase6_zero_capital_only",
        "candidate_bundle_sha256": sha256_file(CANDIDATE_BUNDLE),
        "collection_authorization": _descriptor(COLLECTION_AUTHORIZATION),
        "collection_authorization_reused": False,
        "fresh_confirmation_required_before_authorization": True,
        "fresh_evaluation_manifest_payload_sha256": None,
        "phase6_zero_capital_authorized": False,
        "requested_capital_fraction": 0.0,
        "rollout_step_index": 0,
        "explicit_human_zero_capital_approval_recorded": False,
        "authorization_record_executable": False,
        "template_is_executable": False,
        "micro_live_authorized": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(PHASE6_TEMPLATE, template)
    release = freeze_release_readiness_contract(
        repository_root=ROOT,
        created_at=created_at,
    )
    print(
        json.dumps(
            {
                "phase6_zero_capital_authorization_template": _descriptor(PHASE6_TEMPLATE),
                "release_readiness_v5": release,
                "collection_authorization_reused_for_phase6": False,
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
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
