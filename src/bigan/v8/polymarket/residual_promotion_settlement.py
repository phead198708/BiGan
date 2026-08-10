"""Single-use official settlement ingestion for residual promotion v1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.recorder import PolymarketPublicHTTPRealCorpusProvider
from bigan.v8.polymarket.recorder.async_settlement import _config_from_manifest
from bigan.v8.polymarket.recorder.resolution import (
    normalize_resolution_for_settlement,
)
from bigan.v8.polymarket.residual_promotion_evaluation import (
    _validate_evaluation_authorization,
    validate_evaluation_execution_contract,
)
from bigan.v8.polymarket.residual_promotion_finalization import (
    validate_frozen_population,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    TARGET_MARKETS,
)

SETTLEMENT_CONTRACT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-settlement-ingestion-contract-v1"
)
OUTCOME_ACCESS_CLAIM_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-outcome-access-claim-v1"
SETTLEMENT_ROW_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-official-settlement-row-v1"
SETTLEMENT_REPORT_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-settlement-ingestion-report-v1"
SETTLEMENT_MANIFEST_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-settlement-ingestion-manifest-v1"
)
IMPLEMENTATION_REPOSITORY_PATH = "src/bigan/v8/polymarket/residual_promotion_settlement.py"
CLI_REPOSITORY_PATH = "examples/v8/run_residual_promotion_settlement.py"
CONFIG_REPOSITORY_PATH = (
    "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
)
DEFAULT_MAX_WORKERS = 16
PROVIDER_ATTEMPTS = 1


def ingest_authorized_official_settlements(
    *,
    repository_root: Path | str,
    service_root: Path | str,
    freeze_dir: Path | str,
    expected_population_manifest_sha256: str,
    execution_contract_path: Path | str,
    expected_execution_contract_sha256: str,
    settlement_contract_path: Path | str,
    expected_settlement_contract_sha256: str,
    authorization_path: Path | str,
    expected_authorization_sha256: str,
    output_dir: Path | str,
    provider_factory: Callable[[], Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Open outcomes once after exact authorization and ingest official settlements."""

    repo = Path(repository_root).resolve()
    root = Path(service_root).resolve()
    freeze = Path(freeze_dir).resolve()
    output = Path(output_dir).resolve()
    if not freeze.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("settlement paths must remain inside the collection service root")
    if output.exists():
        raise FileExistsError("settlement ingestion output already exists; rerun forbidden")

    execution_file = _repo_file(execution_contract_path, repo)
    settlement_file = _repo_file(settlement_contract_path, repo)
    authorization_file = _repo_file(authorization_path, repo)
    for path, expected, label in (
        (execution_file, expected_execution_contract_sha256, "execution contract"),
        (settlement_file, expected_settlement_contract_sha256, "settlement contract"),
        (authorization_file, expected_authorization_sha256, "authorization"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
    execution = _verified_json(execution_file)
    validate_evaluation_execution_contract(execution, repository_root=repo)
    settlement_contract = _verified_json(settlement_file)
    validate_settlement_ingestion_contract(
        settlement_contract,
        repository_root=repo,
    )
    if _pair(settlement_contract.get("evaluation_execution_contract")) != _pair(
        _descriptor(execution_file, repo)
    ):
        raise ValueError("settlement and evaluation contract binding mismatch")

    freeze_validation = validate_frozen_population(
        freeze_dir=freeze,
        service_root=root,
        repository_root=repo,
        expected_manifest_sha256=expected_population_manifest_sha256,
    )
    if freeze_validation.get("validation_passed") is not True:
        raise ValueError("exact population validation did not pass")
    authorization = _verified_json(authorization_file)
    _validate_evaluation_authorization(
        authorization,
        execution_contract=_descriptor(execution_file, repo),
        population_manifest_sha256=expected_population_manifest_sha256,
    )
    _validate_settlement_authorization(
        authorization,
        settlement_contract=_descriptor(settlement_file, repo),
        service_root=root,
    )
    contexts = _load_exact_contexts(freeze=freeze, service_root=root)
    if len(contexts) != TARGET_MARKETS:
        raise ValueError("settlement context population is not exactly 2500")

    stamp = created_at or datetime.now(UTC).isoformat()
    output.mkdir(parents=True, exist_ok=False)
    claim = _service_descriptor(
        _write_frozen_json(
            output / "promotion_outcome_access_claim.json",
            {
                "schema_version": OUTCOME_ACCESS_CLAIM_SCHEMA_VERSION,
                "lineage_id": LINEAGE_ID,
                "candidate_id": CANDIDATE_ID,
                "created_at": stamp,
                "population_manifest_sha256": expected_population_manifest_sha256,
                "ordered_market_ids_sha256": canonical_json_sha256(
                    [row["market_id"] for row in contexts]
                ),
                "evaluation_execution_contract": _descriptor(execution_file, repo),
                "settlement_ingestion_contract": _descriptor(settlement_file, repo),
                "outcome_evaluation_authorization": _descriptor(authorization_file, repo),
                "requested_market_count": TARGET_MARKETS,
                "provider_attempts": PROVIDER_ATTEMPTS,
                "attempt_and_alpha_consumed": True,
                "partial_incremental_or_reordered_opening": False,
                "rerun_allowed": False,
                "outcomes_accessed": True,
                "settlement_accessed": True,
                "pnl_accessed": False,
                "automatic_evaluation_or_promotion": False,
                "wallet_signing_allowed": False,
                "polymarket_write_allowed": False,
                "capital_at_risk": False,
                "safety": dict(SAFETY),
            },
        ),
        service_root=root,
    )
    factory = provider_factory or (
        lambda: PolymarketPublicHTTPRealCorpusProvider(
            max_markets=1,
            timeout_seconds=20.0,
            http_timeout_seconds=10.0,
            use_rest_orderbooks=False,
        )
    )
    settlements, failures = _fetch_exact_settlements(
        contexts=contexts,
        provider_factory=factory,
        settlement_finalized_at=stamp,
        max_workers=DEFAULT_MAX_WORKERS,
    )
    ordered_rows = [
        settlements[context["market_id"]]
        for context in contexts
        if context["market_id"] in settlements
    ]
    rows_descriptor = _service_descriptor(
        _write_frozen_jsonl(
            output / "official_settlement_rows.jsonl",
            ordered_rows,
        ),
        service_root=root,
    )
    complete = len(ordered_rows) == TARGET_MARKETS and not failures
    report = _service_descriptor(
        _write_frozen_json(
            output / "settlement_ingestion_report.json",
            {
                "schema_version": SETTLEMENT_REPORT_SCHEMA_VERSION,
                "lineage_id": LINEAGE_ID,
                "candidate_id": CANDIDATE_ID,
                "created_at": stamp,
                "outcome_access_claim": claim,
                "requested_market_count": TARGET_MARKETS,
                "settled_market_count": len(ordered_rows),
                "unresolved_market_count": len(failures),
                "unresolved_markets": failures,
                "outcome_distribution": dict(
                    sorted(Counter(row["resolved_outcome"] for row in ordered_rows).items())
                ),
                "official_settlement_only": True,
                "population_changed": False,
                "source_capture_mutated": False,
                "provider_attempts": PROVIDER_ATTEMPTS,
                "no_rerun": True,
                "evaluation_allowed": complete,
                "lineage_terminalized": not complete,
                "outcomes_accessed": True,
                "settlement_accessed": True,
                "pnl_accessed": False,
                "automatic_evaluation_or_promotion": False,
                "wallet_signing_allowed": False,
                "polymarket_write_allowed": False,
                "capital_at_risk": False,
                "safety": dict(SAFETY),
            },
        ),
        service_root=root,
    )
    manifest = _service_descriptor(
        _write_frozen_json(
            output / "settlement_ingestion_manifest.json",
            {
                "schema_version": SETTLEMENT_MANIFEST_SCHEMA_VERSION,
                "lineage_id": LINEAGE_ID,
                "candidate_id": CANDIDATE_ID,
                "created_at": stamp,
                "population_manifest_sha256": expected_population_manifest_sha256,
                "settlement_ingestion_contract": _descriptor(settlement_file, repo),
                "outcome_access_claim": claim,
                "official_settlement_rows": rows_descriptor,
                "settlement_ingestion_report": report,
                "settlement_population_complete": complete,
                "evaluation_allowed": complete,
                "rerun_allowed": False,
                "automatic_evaluation_or_promotion": False,
                "wallet_signing_allowed": False,
                "polymarket_write_allowed": False,
                "capital_at_risk": False,
                "safety": dict(SAFETY),
            },
        ),
        service_root=root,
    )
    if not complete:
        raise ValueError(
            "official settlement population is incomplete; attempt consumed and lineage terminalized"
        )
    return {
        "manifest": manifest,
        "official_settlement_rows": rows_descriptor,
        "settlement_ingestion_report": report,
        "evaluation_allowed": True,
        "rerun_allowed": False,
        "pnl_accessed": False,
        "automatic_evaluation_or_promotion": False,
        "safety": dict(SAFETY),
    }


def validate_settlement_ingestion_contract(
    contract: Mapping[str, Any], *, repository_root: Path | str
) -> None:
    """Verify every frozen settlement-ingestion implementation binding."""

    root = Path(repository_root).resolve()
    if not (
        contract.get("schema_version") == SETTLEMENT_CONTRACT_SCHEMA_VERSION
        and contract.get("lineage_id") == LINEAGE_ID
        and contract.get("candidate_id") == CANDIDATE_ID
        and contract.get("target_market_count") == TARGET_MARKETS
        and contract.get("provider_attempts") == PROVIDER_ATTEMPTS
        and contract.get("max_workers") == DEFAULT_MAX_WORKERS
        and contract.get("official_settlement_only") is True
        and contract.get("inferred_settlement_allowed") is False
        and contract.get("unresolved_market_allowed") is False
        and contract.get("exact_population_order_required") is True
        and contract.get("outcome_access_claim_before_provider_call") is True
        and contract.get("attempt_consumed_on_any_provider_call") is True
        and contract.get("partial_failure_terminalizes_lineage") is True
        and contract.get("rerun_allowed") is False
        and contract.get("automatic_evaluation_or_promotion") is False
        and contract.get("fresh_outcomes_accessed_when_frozen") is False
        and contract.get("settlement_accessed_when_frozen") is False
        and contract.get("pnl_accessed_when_frozen") is False
        and dict(contract.get("safety") or {}) == SAFETY
    ):
        raise ValueError("settlement ingestion contract is invalid")
    for name in (
        "implementation",
        "cli",
        "evaluation_execution_contract",
        "finalization_implementation",
        "provider_implementation",
        "resolution_normalization_implementation",
        "recorder_config_implementation",
    ):
        _verify_descriptor(dict(contract.get(name) or {}), repository_root=root)
    if dict(contract["implementation"])["path"] != IMPLEMENTATION_REPOSITORY_PATH:
        raise ValueError("settlement implementation path mismatch")
    if dict(contract["cli"])["path"] != CLI_REPOSITORY_PATH:
        raise ValueError("settlement CLI path mismatch")


def _validate_settlement_authorization(
    authorization: Mapping[str, Any],
    *,
    settlement_contract: Mapping[str, Any],
    service_root: Path,
) -> None:
    resume = service_root / "collection_resume_record_v3.json"
    if not resume.is_file():
        raise ValueError("coverage-corrected collection resume record is missing")
    if not (
        _pair(authorization.get("settlement_ingestion_contract")) == _pair(settlement_contract)
        and authorization.get("outcome_access_claim_authorized") is True
        and authorization.get("authorization_record_executable") is True
        and authorization.get("settlement_provider_attempts") == PROVIDER_ATTEMPTS
        and authorization.get("settlement_max_workers") == DEFAULT_MAX_WORKERS
        and authorization.get("collection_start_record_sha256") == sha256_file(resume)
        and authorization.get("template_is_executable") is not True
        and dict(authorization.get("safety") or {}) == SAFETY
    ):
        raise ValueError("official settlement authorization is invalid")


def _load_exact_contexts(*, freeze: Path, service_root: Path) -> list[dict[str, Any]]:
    manifest = _verified_json(freeze / "exact_population_manifest.json")
    artifacts = dict(manifest.get("artifacts") or {})
    population = _load_jsonl(freeze / dict(artifacts["population_rows"])["path"])
    captures = _load_jsonl(freeze / dict(artifacts["raw_capture_index"])["path"])
    if len(population) != TARGET_MARKETS or len(captures) != TARGET_MARKETS:
        raise ValueError("frozen settlement context count mismatch")
    contexts = []
    for expected_position, (row, capture) in enumerate(
        zip(population, captures, strict=True), start=1
    ):
        market_id = str(row["market_id"])
        if not (
            int(row["population_position"]) == expected_position
            and int(capture["population_position"]) == expected_position
            and str(capture["market_id"]) == market_id
        ):
            raise ValueError("settlement contexts are not in exact frozen order")
        raw_markets = [
            item
            for item in list(capture["files"])
            if str(item.get("path") or "").endswith("/raw/raw_polymarket_markets.jsonl")
        ]
        if len(raw_markets) != 1:
            raise ValueError("frozen raw market artifact did not reconcile")
        raw_descriptor = raw_markets[0]
        market_path = (service_root / str(raw_descriptor["path"])).resolve()
        if (
            not market_path.is_relative_to(service_root)
            or sha256_file(market_path) != raw_descriptor["sha256"]
        ):
            raise ValueError("frozen raw market artifact SHA-256 mismatch")
        markets = _load_jsonl(market_path)
        if len(markets) != 1 or str(markets[0].get("market_id")) != market_id:
            raise ValueError("frozen raw market identity mismatch")
        manifest_path = (service_root / str(capture["capture_manifest_path"])).resolve()
        if (
            not manifest_path.is_relative_to(service_root)
            or sha256_file(manifest_path) != capture["capture_manifest_sha256"]
        ):
            raise ValueError("frozen capture manifest SHA-256 mismatch")
        contexts.append(
            {
                "market_id": market_id,
                "market": markets[0],
                "recorder_config": _config_from_manifest(_load_json(manifest_path)),
            }
        )
    return contexts


def _fetch_exact_settlements(
    *,
    contexts: Sequence[Mapping[str, Any]],
    provider_factory: Callable[[], Any],
    settlement_finalized_at: str,
    max_workers: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    settlements: dict[str, dict[str, Any]] = {}
    failures: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_settlement,
                context,
                provider_factory,
                settlement_finalized_at,
            ): str(context["market_id"])
            for context in contexts
        }
        for future in as_completed(futures):
            market_id = futures[future]
            try:
                settlements[market_id] = future.result()
            except Exception as exc:  # noqa: BLE001
                failures[market_id] = [f"{type(exc).__name__}:{exc}"]
    return settlements, [
        {"market_id": market_id, "reason_codes": failures[market_id]}
        for market_id in sorted(failures)
    ]


def _fetch_settlement(
    context: Mapping[str, Any],
    provider_factory: Callable[[], Any],
    settlement_finalized_at: str,
) -> dict[str, Any]:
    provider = provider_factory()
    market = dict(context["market"])
    rows = provider.resolution_rows([market], context["recorder_config"])
    candidates = [row for row in rows if str(row.get("market_id")) == context["market_id"]]
    if len(candidates) != 1:
        raise ValueError("official provider did not return exactly one resolution")
    normalized, reasons = normalize_resolution_for_settlement(
        market=market,
        resolution=dict(candidates[0]),
    )
    if normalized is None:
        raise ValueError("official resolution rejected:" + ",".join(reasons))
    payout_up = float(normalized["payout_up"])
    payout_down = float(normalized["payout_down"])
    if payout_up not in {0.0, 1.0} or payout_down not in {0.0, 1.0}:
        raise ValueError("non-binary official payout is unresolved for promotion")
    raw_sha = canonical_json_sha256(normalized)
    condition_id = str(market.get("condition_id") or context["market_id"])
    return {
        "schema_version": SETTLEMENT_ROW_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "market_id": str(context["market_id"]),
        "settlement_source": "official_polymarket",
        "official_resolution_reference": f"polymarket:{condition_id}:{raw_sha}",
        "settlement_finalized_at": settlement_finalized_at,
        "official_final": True,
        "inferred": False,
        "unresolved": False,
        "resolved_outcome": str(normalized["resolved_outcome"]),
        "payout_up": payout_up,
        "payout_down": payout_down,
        "resolution_source_type": normalized["resolution_source_type"],
        "resolved_outcome_source": normalized["resolved_outcome_source"],
        "raw_resolution_sha256": raw_sha,
        "official_read_only": True,
        "source_capture_mutated": False,
        "outcomes_accessed": True,
        "settlement_accessed": True,
        "pnl_accessed": False,
        "promotion_evidence_eligible": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }


def _verified_json(path: Path) -> dict[str, Any]:
    _verify_sidecar(path)
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError("frozen JSON root must be an object")
    return value


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not path.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path)
    ):
        raise ValueError(f"frozen artifact sidecar mismatch: {path.name}")


def _verify_descriptor(descriptor: Mapping[str, Any], *, repository_root: Path) -> None:
    path = _repo_file(str(descriptor.get("path") or ""), repository_root)
    if sha256_file(path) != descriptor.get("sha256"):
        raise ValueError("repository artifact descriptor SHA-256 mismatch")


def _repo_file(path: Path | str, root: Path) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("repository artifact path is not portable")
    return resolved


def _descriptor(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def _pair(value: Any) -> tuple[Any, Any]:
    descriptor = dict(value or {})
    return descriptor.get("path"), descriptor.get("sha256")


def _service_descriptor(descriptor: Mapping[str, str], *, service_root: Path) -> dict[str, str]:
    path = Path(descriptor["path"]).resolve()
    if not path.is_relative_to(service_root):
        raise ValueError("settlement output escaped collection service root")
    return {
        "path": path.relative_to(service_root).as_posix(),
        "sha256": descriptor["sha256"],
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    raw = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    return _write_frozen_bytes(path, raw)


def _write_frozen_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    raw = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    return _write_frozen_bytes(path, raw)


def _write_frozen_bytes(path: Path, raw: bytes) -> dict[str, str]:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError(f"single-use settlement artifact already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return {"path": path.as_posix(), "sha256": digest}


__all__ = [
    "ingest_authorized_official_settlements",
    "validate_settlement_ingestion_contract",
]
