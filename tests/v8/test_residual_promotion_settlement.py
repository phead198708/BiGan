"""Fail-closed official settlement ingestion tests for residual promotion v1."""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_evaluation import (
    AUTHORIZATION_SCHEMA_VERSION,
    _validate_evaluation_authorization,
)
from bigan.v8.polymarket.residual_promotion_settlement import (
    DEFAULT_MAX_WORKERS,
    PROVIDER_ATTEMPTS,
    SETTLEMENT_CONTRACT_SCHEMA_VERSION,
    _fetch_exact_settlements,
    ingest_authorized_official_settlements,
    validate_settlement_ingestion_contract,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    TARGET_MARKETS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT / "examples/v8/polymarket_configs" / "BTC-15M-cost-aware-market-residual-promotion-v1"
)
CONTRACT = CONFIG / "promotion_settlement_ingestion_contract.json"
AUTHORIZATION_TEMPLATE = CONFIG / "promotion_outcome_evaluation_authorization_template_v4.json"


def _write_frozen_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")


def _market(index: int) -> dict:
    market_id = f"market-{index:04d}"
    return {
        "market_id": market_id,
        "condition_id": f"condition-{index:04d}",
        "reference_price_source": "chainlink-btc-usd",
    }


def _resolution(market: dict, *, outcome: str = "UP") -> dict:
    return {
        "market_id": market["market_id"],
        "reference_price_source": market["reference_price_source"],
        "resolution_status": "normal",
        "resolved_outcome": outcome,
        "payout_up": 1.0 if outcome == "UP" else 0.0,
        "payout_down": 0.0 if outcome == "UP" else 1.0,
        "resolution_source_type": "polymarket_clob_market_tokens",
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _contexts(count: int) -> list[dict]:
    return [
        {
            "market_id": f"market-{index:04d}",
            "market": _market(index),
            "recorder_config": object(),
        }
        for index in range(count)
    ]


def test_frozen_settlement_contract_and_authorization_template_reconcile() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema_version"] == SETTLEMENT_CONTRACT_SCHEMA_VERSION
    assert CONTRACT.with_suffix(".json.sha256").read_text().strip() == sha256_file(CONTRACT)
    validate_settlement_ingestion_contract(contract, repository_root=REPO_ROOT)

    template = json.loads(AUTHORIZATION_TEMPLATE.read_text(encoding="utf-8"))
    assert AUTHORIZATION_TEMPLATE.with_suffix(".json.sha256").read_text().strip() == sha256_file(
        AUTHORIZATION_TEMPLATE
    )
    assert template["settlement_ingestion_contract"] == {
        "path": CONTRACT.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(CONTRACT),
    }
    assert template["candidate_id"] == CANDIDATE_ID
    assert template["authorized_record_schema_version"] == AUTHORIZATION_SCHEMA_VERSION
    assert template["fresh_outcome_access_authorized"] is False
    assert template["official_settlement_ingestion_authorized"] is False
    assert template["outcome_access_claim_authorized"] is False
    assert template["evaluation_exactly_once_authorized"] is False
    assert template["authorization_record_executable"] is False
    assert template["template_is_executable"] is False
    assert template["safety"] == SAFETY
    with pytest.raises(ValueError, match="invalid"):
        _validate_evaluation_authorization(
            template,
            execution_contract=template["execution_contract"],
            population_manifest_sha256="0" * 64,
        )


def test_contract_child_sha_drift_fails_closed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    changed = copy.deepcopy(contract)
    changed["provider_implementation"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptor SHA-256 mismatch"):
        validate_settlement_ingestion_contract(changed, repository_root=REPO_ROOT)


def test_exact_settlement_fetch_is_single_attempt_and_order_independent() -> None:
    lock = threading.Lock()
    calls: list[str] = []

    class Provider:
        def resolution_rows(self, markets: list[dict], _config: object) -> list[dict]:
            with lock:
                calls.append(markets[0]["market_id"])
            return [_resolution(markets[0])]

    contexts = list(reversed(_contexts(40)))
    rows, failures = _fetch_exact_settlements(
        contexts=contexts,
        provider_factory=Provider,
        settlement_finalized_at="2030-01-01T00:00:00+00:00",
        max_workers=8,
    )
    assert failures == []
    assert len(calls) == len(set(calls)) == 40
    assert set(rows) == {context["market_id"] for context in contexts}
    assert all(row["settlement_source"] == "official_polymarket" for row in rows.values())
    assert all(row["pnl_accessed"] is False for row in rows.values())


def test_unresolved_or_duplicate_provider_row_fails_closed() -> None:
    class MissingProvider:
        def resolution_rows(self, _markets: list[dict], _config: object) -> list[dict]:
            return []

    rows, failures = _fetch_exact_settlements(
        contexts=_contexts(1),
        provider_factory=MissingProvider,
        settlement_finalized_at="2030-01-01T00:00:00+00:00",
        max_workers=1,
    )
    assert rows == {}
    assert failures[0]["market_id"] == "market-0000"
    assert "exactly one resolution" in failures[0]["reason_codes"][0]


def test_authorized_ingestion_claims_before_provider_and_cannot_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repo"
    service_root = tmp_path / "service"
    freeze = service_root / "freeze"
    output = service_root / "settlement"
    repository.mkdir()
    freeze.mkdir(parents=True)
    resume = service_root / "collection_resume_record_v3.json"
    resume.write_text("{}\n", encoding="utf-8")
    population_sha = "a" * 64

    execution_file = repository / "execution.json"
    settlement_file = repository / "settlement.json"
    authorization_file = repository / "authorization.json"
    execution = {"fixture": "execution"}
    _write_frozen_json(execution_file, execution)
    execution_descriptor = {
        "path": "execution.json",
        "sha256": sha256_file(execution_file),
    }
    settlement = {"evaluation_execution_contract": execution_descriptor}
    _write_frozen_json(settlement_file, settlement)
    settlement_descriptor = {
        "path": "settlement.json",
        "sha256": sha256_file(settlement_file),
    }
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "fresh_outcome_access_authorized": True,
        "official_settlement_ingestion_authorized": True,
        "outcome_access_claim_authorized": True,
        "evaluation_exactly_once_authorized": True,
        "interim_evaluation_authorized": False,
        "rerun_authorized": False,
        "population_manifest_sha256": population_sha,
        "execution_contract": execution_descriptor,
        "settlement_ingestion_contract": settlement_descriptor,
        "service_root_id": "fixture-service",
        "collection_start_record_sha256": sha256_file(resume),
        "settlement_provider_attempts": PROVIDER_ATTEMPTS,
        "settlement_max_workers": DEFAULT_MAX_WORKERS,
        "paper_live_wallet_write_or_capital_authorized": False,
        "authorization_record_executable": True,
        "template_is_executable": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(authorization_file, authorization)

    monkeypatch.setattr(
        "bigan.v8.polymarket.residual_promotion_settlement.validate_evaluation_execution_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.residual_promotion_settlement.validate_settlement_ingestion_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.residual_promotion_settlement.validate_frozen_population",
        lambda *_args, **_kwargs: {"validation_passed": True},
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.residual_promotion_settlement._load_exact_contexts",
        lambda **_kwargs: _contexts(TARGET_MARKETS),
    )

    lock = threading.Lock()
    calls = 0

    class Provider:
        def resolution_rows(self, markets: list[dict], _config: object) -> list[dict]:
            nonlocal calls
            assert (output / "promotion_outcome_access_claim.json").is_file()
            with lock:
                calls += 1
            return [_resolution(markets[0])]

    result = ingest_authorized_official_settlements(
        repository_root=repository,
        service_root=service_root,
        freeze_dir=freeze,
        expected_population_manifest_sha256=population_sha,
        execution_contract_path=execution_file,
        expected_execution_contract_sha256=sha256_file(execution_file),
        settlement_contract_path=settlement_file,
        expected_settlement_contract_sha256=sha256_file(settlement_file),
        authorization_path=authorization_file,
        expected_authorization_sha256=sha256_file(authorization_file),
        output_dir=output,
        provider_factory=Provider,
        created_at="2030-01-01T00:00:00+00:00",
    )
    assert calls == TARGET_MARKETS
    assert result["evaluation_allowed"] is True
    assert result["rerun_allowed"] is False
    assert result["pnl_accessed"] is False
    assert not Path(result["manifest"]["path"]).is_absolute()
    rows = (output / "official_settlement_rows.jsonl").read_text().splitlines()
    assert len(rows) == TARGET_MARKETS
    assert [json.loads(row)["market_id"] for row in rows] == [
        f"market-{index:04d}" for index in range(TARGET_MARKETS)
    ]
    with pytest.raises(FileExistsError, match="rerun forbidden"):
        ingest_authorized_official_settlements(
            repository_root=repository,
            service_root=service_root,
            freeze_dir=freeze,
            expected_population_manifest_sha256=population_sha,
            execution_contract_path=execution_file,
            expected_execution_contract_sha256=sha256_file(execution_file),
            settlement_contract_path=settlement_file,
            expected_settlement_contract_sha256=sha256_file(settlement_file),
            authorization_path=authorization_file,
            expected_authorization_sha256=sha256_file(authorization_file),
            output_dir=output,
            provider_factory=Provider,
        )
