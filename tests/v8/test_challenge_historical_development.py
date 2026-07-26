from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_historical_development import (
    ZERO_SHA256,
    ChallengeHistoricalDevelopmentError,
    HistoricalDevelopmentEvaluationConfig,
    evaluate_historical_development_candidate,
    run_historical_development_evaluation,
    validate_attempt_001_closure,
    validate_historical_development_ledger_root,
    validate_historical_development_registry,
    validate_historical_development_success_standard,
    validate_iteration_preregistration,
)
from examples.v8 import run_challenge_historical_development as runner

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples" / "v8" / "polymarket_configs"
EVIDENCE_LEDGER_SHA256 = (
    "98f43a1a9526d9b21342c6047aaf6ca34f78e0149a0c31adbd58fd7e98a11bf3"
)
SAFETY = {
    "capital_at_risk": False,
    "collection_start_allowed": False,
    "handoff_allowed": False,
    "live_allowed": False,
    "paper_allowed": False,
    "promotion_allowed": False,
    "wallet_allowed": False,
    "write_allowed": False,
}


def _json(name: str) -> dict:
    payload = json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar(name: str) -> str:
    return (CONFIG_DIR / name).read_text(encoding="ascii").strip()


def _market_ids() -> list[str]:
    return (
        CONFIG_DIR / "challenge_historical_development_exact_195_market_ids.txt"
    ).read_text(encoding="utf-8").splitlines()


def _passing_rows() -> list[dict]:
    rows = []
    for index, market_id in enumerate(_market_ids()):
        side = "UP" if index % 2 == 0 else "DOWN"
        rows.append(
            {
                "market_id": market_id,
                "candidate_action": f"BUY_{side}_SELL_BEFORE_CLOSE",
                "candidate_side": side,
                "candidate_after_cost_pnl": 0.1,
                "baseline_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
                "baseline_side": "DOWN",
                "baseline_after_cost_pnl": 0.0,
                "candidate_minus_baseline_pnl": 0.1,
            }
        )
    return rows


def _preregistration(
    *,
    comparison_sha256: str,
    implementation_base_commit: str,
) -> dict:
    return {
        "schema_version": (
            "bigan-v8-challenge-historical-development-preregistration-v1"
        ),
        "iteration_number": 1,
        "candidate_id": "synthetic-pipeline-canary",
        "development_corpus_registry_sha256": _sidecar(
            "challenge_historical_development_data_registry.sha256"
        ),
        "success_standard_sha256": _sidecar(
            "challenge_historical_development_success_standard.sha256"
        ),
        "ledger_root_sha256": _sidecar(
            "challenge_historical_development_iteration_ledger.sha256"
        ),
        "previous_iteration_entry_sha256": ZERO_SHA256,
        "changed_components": ["sizing"],
        "change_description": "Synthetic positive constant PnL rows for pipeline validation.",
        "mechanistic_rationale": "Exercise every frozen gate without opening real labels.",
        "expected_mechanism": "All deterministic synthetic confidence bounds are positive.",
        "input_artifact_sha256s": {
            "synthetic_comparison_rows": comparison_sha256,
        },
        "implementation_commit": implementation_base_commit,
        "implementation_commit_role": "prechange_base_commit",
        "candidate_count": 1,
        "grid_search": False,
        "result_selected_parameter_search": False,
        "evaluation_started": False,
        "outcome_aware_replay_started": False,
        "promotion_evidence_eligible": False,
        "safety": SAFETY,
    }


def test_frozen_artifacts_validate_and_sidecars_match() -> None:
    closure = _json("challenge_attempt_001_closure.json")
    registry = _json("challenge_historical_development_data_registry.json")
    standard = _json("challenge_historical_development_success_standard.json")
    ledger = _json("challenge_historical_development_iteration_ledger.json")

    for stem in (
        "challenge_attempt_001_closure",
        "challenge_historical_development_data_registry",
        "challenge_historical_development_success_standard",
        "challenge_historical_development_iteration_ledger",
        "challenge_historical_development_exact_195_market_ids",
    ):
        suffix = ".txt" if stem.endswith("market_ids") else ".json"
        assert _sha256(CONFIG_DIR / f"{stem}{suffix}") == _sidecar(f"{stem}.sha256")

    validate_attempt_001_closure(
        closure,
        expected_collection_plan_sha256=closure["frozen_lineage"][
            "collection_plan_sha256"
        ],
        expected_supersession_governance_sha256=closure["frozen_lineage"][
            "supersession_governance_sha256"
        ],
    )
    validate_historical_development_registry(
        registry,
        expected_attempt_closure_sha256=_sidecar(
            "challenge_attempt_001_closure.sha256"
        ),
        expected_evidence_ledger_sha256=EVIDENCE_LEDGER_SHA256,
    )
    validate_historical_development_success_standard(
        standard,
        expected_registry_sha256=_sidecar(
            "challenge_historical_development_data_registry.sha256"
        ),
    )
    validate_historical_development_ledger_root(
        ledger,
        expected_attempt_closure_sha256=_sidecar(
            "challenge_attempt_001_closure.sha256"
        ),
        expected_registry_sha256=_sidecar(
            "challenge_historical_development_data_registry.sha256"
        ),
        expected_success_standard_sha256=_sidecar(
            "challenge_historical_development_success_standard.sha256"
        ),
    )
    assert len(_market_ids()) == 195
    assert len(set(_market_ids())) == 195


@pytest.mark.parametrize(
    ("artifact_name", "mutate", "validator"),
    [
        (
            "challenge_attempt_001_closure.json",
            lambda value: value["raw_collection_audit"].update(
                {"quality_valid_capture_count": 1}
            ),
            lambda value: validate_attempt_001_closure(
                value,
                expected_collection_plan_sha256=value["frozen_lineage"][
                    "collection_plan_sha256"
                ],
                expected_supersession_governance_sha256=value["frozen_lineage"][
                    "supersession_governance_sha256"
                ],
            ),
        ),
        (
            "challenge_historical_development_data_registry.json",
            lambda value: value["development_policy"].update(
                {"historical_results_can_unlock_promotion": True}
            ),
            lambda value: validate_historical_development_registry(
                value,
                expected_attempt_closure_sha256=_sidecar(
                    "challenge_attempt_001_closure.sha256"
                ),
                expected_evidence_ledger_sha256=EVIDENCE_LEDGER_SHA256,
            ),
        ),
        (
            "challenge_historical_development_success_standard.json",
            lambda value: value["full_window_paired_gate"].update(
                {
                    "candidate_minus_baseline_after_cost_pnl_bootstrap_lcb_minimum_exclusive": -1.0
                }
            ),
            lambda value: validate_historical_development_success_standard(
                value,
                expected_registry_sha256=_sidecar(
                    "challenge_historical_development_data_registry.sha256"
                ),
            ),
        ),
        (
            "challenge_historical_development_iteration_ledger.json",
            lambda value: value.update({"maximum_development_iterations": 6}),
            lambda value: validate_historical_development_ledger_root(
                value,
                expected_attempt_closure_sha256=_sidecar(
                    "challenge_attempt_001_closure.sha256"
                ),
                expected_registry_sha256=_sidecar(
                    "challenge_historical_development_data_registry.sha256"
                ),
                expected_success_standard_sha256=_sidecar(
                    "challenge_historical_development_success_standard.sha256"
                ),
            ),
        ),
    ],
)
def test_semantic_tampering_fails_closed(
    artifact_name: str,
    mutate,
    validator,
) -> None:
    payload = copy.deepcopy(_json(artifact_name))
    mutate(payload)

    with pytest.raises(ChallengeHistoricalDevelopmentError):
        validator(payload)


def test_synthetic_candidate_passes_every_historical_gate() -> None:
    report = evaluate_historical_development_candidate(
        _passing_rows(),
        success_standard=_json(
            "challenge_historical_development_success_standard.json"
        ),
        candidate_id="synthetic-pipeline-canary",
        iteration_number=1,
    )

    assert all(report["checks"].values())
    assert report["all_historical_success_criteria_passed"] is True
    assert report["attempt_002_preregistration_allowed"] is True
    assert report["promotion_evidence_eligible"] is False
    assert report["safety"] == SAFETY
    assert report["concentration_diagnostics"]["hard_gate"] is False


def test_market_order_and_no_trade_accounting_fail_closed() -> None:
    standard = _json("challenge_historical_development_success_standard.json")
    reordered = _passing_rows()
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(
        ChallengeHistoricalDevelopmentError,
        match="frozen exact-195 corpus",
    ):
        evaluate_historical_development_candidate(
            reordered,
            success_standard=standard,
            candidate_id="reordered",
            iteration_number=1,
        )

    invalid_no_trade = _passing_rows()
    invalid_no_trade[0].update(
        {
            "candidate_action": "NO_TRADE",
            "candidate_side": "NONE",
            "candidate_after_cost_pnl": 0.01,
            "candidate_minus_baseline_pnl": 0.01,
        }
    )
    with pytest.raises(ChallengeHistoricalDevelopmentError, match="NO_TRADE"):
        evaluate_historical_development_candidate(
            invalid_no_trade,
            success_standard=standard,
            candidate_id="invalid-no-trade",
            iteration_number=1,
        )


def test_preregistration_binds_candidate_commit_and_one_iteration() -> None:
    preregistration = _preregistration(
        comparison_sha256="a" * 64,
        implementation_base_commit="b" * 40,
    )
    validate_iteration_preregistration(
        preregistration,
        expected_iteration_number=1,
        expected_candidate_id="synthetic-pipeline-canary",
        expected_registry_sha256=_sidecar(
            "challenge_historical_development_data_registry.sha256"
        ),
        expected_success_standard_sha256=_sidecar(
            "challenge_historical_development_success_standard.sha256"
        ),
        expected_ledger_root_sha256=_sidecar(
            "challenge_historical_development_iteration_ledger.sha256"
        ),
        expected_previous_entry_sha256=ZERO_SHA256,
        expected_implementation_base_commit="b" * 40,
    )

    preregistration["grid_search"] = True
    with pytest.raises(ChallengeHistoricalDevelopmentError, match="no_grid"):
        validate_iteration_preregistration(
            preregistration,
            expected_iteration_number=1,
            expected_candidate_id="synthetic-pipeline-canary",
            expected_registry_sha256=_sidecar(
                "challenge_historical_development_data_registry.sha256"
            ),
            expected_success_standard_sha256=_sidecar(
                "challenge_historical_development_success_standard.sha256"
            ),
            expected_ledger_root_sha256=_sidecar(
                "challenge_historical_development_iteration_ledger.sha256"
            ),
            expected_previous_entry_sha256=ZERO_SHA256,
            expected_implementation_base_commit="b" * 40,
        )


def test_synthetic_pipeline_dry_run_emits_hash_chained_entry(
    tmp_path: Path,
) -> None:
    comparison_path = tmp_path / "synthetic-comparison.jsonl"
    comparison_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in _passing_rows()
        ),
        encoding="utf-8",
    )
    implementation_base_commit = "b" * 40
    preregistration_commit = "c" * 40
    implementation_commit = "d" * 40
    preregistration_path = tmp_path / "synthetic-preregistration.json"
    preregistration_path.write_text(
        json.dumps(
            _preregistration(
                comparison_sha256=_sha256(comparison_path),
                implementation_base_commit=implementation_base_commit,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_historical_development_evaluation(
        HistoricalDevelopmentEvaluationConfig(
            run_id="synthetic-historical-development-dry-run",
            output_dir=tmp_path / "runs",
            iteration_number=1,
            candidate_id="synthetic-pipeline-canary",
            comparison_rows_path=comparison_path,
            expected_comparison_rows_sha256=_sha256(comparison_path),
            preregistration_path=preregistration_path,
            expected_preregistration_sha256=_sha256(preregistration_path),
            success_standard_path=(
                CONFIG_DIR
                / "challenge_historical_development_success_standard.json"
            ),
            expected_success_standard_sha256=_sidecar(
                "challenge_historical_development_success_standard.sha256"
            ),
            registry_path=(
                CONFIG_DIR
                / "challenge_historical_development_data_registry.json"
            ),
            expected_registry_sha256=_sidecar(
                "challenge_historical_development_data_registry.sha256"
            ),
            ledger_root_path=(
                CONFIG_DIR
                / "challenge_historical_development_iteration_ledger.json"
            ),
            expected_ledger_root_sha256=_sidecar(
                "challenge_historical_development_iteration_ledger.sha256"
            ),
            attempt_closure_path=(
                CONFIG_DIR / "challenge_attempt_001_closure.json"
            ),
            expected_attempt_closure_sha256=_sidecar(
                "challenge_attempt_001_closure.sha256"
            ),
            implementation_base_commit=implementation_base_commit,
            preregistration_commit=preregistration_commit,
            implementation_commit=implementation_commit,
            evaluated_at="synthetic-no-real-labels",
        )
    )

    assert result["report"]["all_historical_success_criteria_passed"] is True
    assert result["report"]["promotion_evidence_eligible"] is False
    assert result["iteration_entry"]["previous_entry_sha256"] == ZERO_SHA256
    assert (
        result["iteration_entry"]["entry_sha256"]
        == result["iteration_entry_sha256"]
    )
    assert result["iteration_entry_file_sha256"] == _sha256(
        result["iteration_entry_path"]
    )
    assert result["manifest_path"].is_file()


def test_cli_requires_preregistration_commit_before_candidate_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test User")
    base_path = repository / "base.txt"
    base_path.write_text("base\n", encoding="utf-8")
    git("add", "base.txt")
    git("commit", "-m", "base")
    base_commit = git("rev-parse", "HEAD")

    preregistration_path = repository / "iteration-001.json"
    preregistration_path.write_text(
        json.dumps(
            {
                "implementation_commit": base_commit,
                "implementation_commit_role": "prechange_base_commit",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preregistration_path.with_suffix(".sha256").write_text(
        _sha256(preregistration_path) + "\n",
        encoding="ascii",
    )
    git("add", "iteration-001.json", "iteration-001.sha256")
    git("commit", "-m", "preregister iteration 1")
    preregistration_commit = git("rev-parse", "HEAD")

    candidate_path = repository / "candidate.py"
    candidate_path.write_text("PRICE_FLOOR = 0.30\n", encoding="utf-8")
    git("add", "candidate.py")
    git("commit", "-m", "implement candidate")
    implementation_commit = git("rev-parse", "HEAD")

    monkeypatch.setattr(runner, "REPO_ROOT", repository)
    state = runner._require_clean_preregistered_state(preregistration_path)

    assert state["implementation_base_commit"] == base_commit
    assert state["preregistration_commit"] == preregistration_commit
    assert state["implementation_commit"] == implementation_commit


def test_iteration_001_failure_is_immutably_recorded() -> None:
    stems = (
        "challenge_historical_development_iteration_001_entry",
        "challenge_historical_development_iteration_001_result",
        "challenge_historical_development_iteration_001_target_free_manifest",
        "challenge_historical_development_iteration_001_manifest",
    )
    for stem in stems:
        path = CONFIG_DIR / f"{stem}.json"
        assert _sha256(path) == _sidecar(f"{stem}.sha256")

    entry = _json("challenge_historical_development_iteration_001_entry.json")
    result = _json("challenge_historical_development_iteration_001_result.json")
    semantic_payload = {
        key: value for key, value in entry.items() if key != "entry_sha256"
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(
            semantic_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert entry["entry_sha256"] == (
        "76b1ea851c40c3b66aa47e762cfcfdeaada0b6372232a1f4c4365cb6cbebff95"
    )
    assert semantic_sha256 == entry["entry_sha256"]
    assert entry["consumes_development_iteration_slot"] is True
    assert entry["consumes_fresh_promotion_alpha"] is False
    assert entry["result"]["sha256"] == _sidecar(
        "challenge_historical_development_iteration_001_result.sha256"
    )
    assert result["all_historical_success_criteria_passed"] is False
    assert result["attempt_002_preregistration_allowed"] is False
    assert result["checks"][
        "full_window_paired_bootstrap_97_5_lcb_positive"
    ] is False
    assert result["bootstrap"]["paired_delta_97_5_lcb"] == pytest.approx(
        -0.15145125
    )


def test_iteration_002_failure_is_immutably_recorded() -> None:
    stems = (
        "challenge_historical_development_iteration_002_entry",
        "challenge_historical_development_iteration_002_result",
        "challenge_historical_development_iteration_002_target_free_manifest",
        "challenge_historical_development_iteration_002_manifest",
    )
    for stem in stems:
        path = CONFIG_DIR / f"{stem}.json"
        assert _sha256(path) == _sidecar(f"{stem}.sha256")

    entry = _json("challenge_historical_development_iteration_002_entry.json")
    result = _json("challenge_historical_development_iteration_002_result.json")
    semantic_payload = {
        key: value for key, value in entry.items() if key != "entry_sha256"
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(
            semantic_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert entry["entry_sha256"] == (
        "b608b9e7b884ca636610f18bd89ef780ac2a4c3fecff65b45b7f836dd9367740"
    )
    assert semantic_sha256 == entry["entry_sha256"]
    assert entry["previous_entry_sha256"] == (
        "76b1ea851c40c3b66aa47e762cfcfdeaada0b6372232a1f4c4365cb6cbebff95"
    )
    assert entry["consumes_development_iteration_slot"] is True
    assert entry["result"]["sha256"] == _sidecar(
        "challenge_historical_development_iteration_002_result.sha256"
    )
    assert result["all_historical_success_criteria_passed"] is False
    assert result["metrics"]["candidate_total_after_cost_pnl"] == pytest.approx(
        -0.9187
    )
    assert result["bootstrap"]["paired_delta_97_5_lcb"] == pytest.approx(
        -0.6944075
    )


def test_iteration_003_success_is_immutably_recorded() -> None:
    stems = (
        "challenge_historical_development_iteration_003_entry",
        "challenge_historical_development_iteration_003_result",
        "challenge_historical_development_iteration_003_target_free_manifest",
        "challenge_historical_development_iteration_003_manifest",
    )
    for stem in stems:
        path = CONFIG_DIR / f"{stem}.json"
        assert _sha256(path) == _sidecar(f"{stem}.sha256")

    entry = _json("challenge_historical_development_iteration_003_entry.json")
    result = _json("challenge_historical_development_iteration_003_result.json")
    semantic_payload = {
        key: value for key, value in entry.items() if key != "entry_sha256"
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(
            semantic_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert entry["entry_sha256"] == (
        "abe1eb3b6e03530c15cb51326801e783454a0be4d5885edd7dcca8dab22779ae"
    )
    assert semantic_sha256 == entry["entry_sha256"]
    assert entry["previous_entry_sha256"] == (
        "b608b9e7b884ca636610f18bd89ef780ac2a4c3fecff65b45b7f836dd9367740"
    )
    assert entry["consumes_development_iteration_slot"] is True
    assert entry["consumes_fresh_promotion_alpha"] is False
    assert entry["result"]["sha256"] == _sidecar(
        "challenge_historical_development_iteration_003_result.sha256"
    )
    assert result["all_historical_success_criteria_passed"] is True
    assert result["attempt_002_preregistration_allowed"] is True
    assert result["checks"][
        "full_window_paired_bootstrap_97_5_lcb_positive"
    ] is True
    assert result["checks"][
        "candidate_absolute_bootstrap_97_5_lcb_positive"
    ] is True
    assert result["metrics"]["candidate_total_after_cost_pnl"] == pytest.approx(
        1.18375
    )
    assert result["bootstrap"]["paired_delta_97_5_lcb"] == pytest.approx(
        0.6115975
    )
    assert result["bootstrap"]["candidate_absolute_97_5_lcb"] == pytest.approx(
        0.23475
    )
    assert result["promotion_evidence_eligible"] is False
    assert result["safety"] == SAFETY
