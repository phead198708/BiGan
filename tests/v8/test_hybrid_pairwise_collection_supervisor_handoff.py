"""Tests for exclusive collection supervisor handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.training.hybrid_pairwise_collection_supervisor_handoff import (
    HybridCollectionSupervisorHandoffConfig,
    ProcessIdentity,
    perform_exclusive_collection_supervisor_handoff,
)


def test_apply_terminates_only_superseded_and_preserves_protected_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    alive = {100, 200, 300, 301, 302, 400}
    commands = {
        100: "python /tmp/issue177_collection_supervisor.py",
        200: "python /tmp/issue178_collection_handoff_supervisor.py",
        300: "python examples/v8/run_polymarket_async_round_collector.py",
        301: "python /tmp/issue176_round_comment_monitor.py",
        302: "python /tmp/issue176_orderbook_failure_watchdog.py",
        400: "python /tmp/issue183_post_terminal_freeze_supervisor.py",
    }
    terminated: list[int] = []
    _patch_processes(monkeypatch, alive, commands)

    def terminate(pid: int, *, wait_seconds: float) -> None:
        assert wait_seconds == 2.0
        terminated.append(pid)
        alive.remove(pid)

    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_collection_supervisor_handoff._terminate_pid",
        terminate,
    )

    result = _run(tmp_path, fixture, apply=True)
    report = result["report"]

    assert terminated == [100]
    assert report["handoff_applied"] is True
    assert report["superseded_supervisor_alive_after"] is False
    assert report["successor_supervisor_alive_after"] is True
    assert report["all_protected_processes_alive_after"] is True
    assert alive == {200, 300, 301, 302, 400}
    assert result["claim_path"].is_file()
    _assert_safety(report)


def test_repeated_claim_fails_closed_before_second_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    alive = {100, 200, 300, 301, 302, 400}
    commands = _commands()
    _patch_processes(monkeypatch, alive, commands)
    termination_count = 0

    def terminate(pid: int, *, wait_seconds: float) -> None:
        nonlocal termination_count
        del wait_seconds
        termination_count += 1
        alive.remove(pid)

    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_collection_supervisor_handoff._terminate_pid",
        terminate,
    )
    _run(tmp_path, fixture, apply=True, run_id="first")
    alive.add(100)

    with pytest.raises(
        ValueError,
        match="exclusive handoff claim already exists",
    ):
        _run(tmp_path, fixture, apply=True, run_id="second")
    assert termination_count == 1


def test_superseded_pid_cannot_be_a_protected_collector(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    collector = fixture["protected"][0]

    with pytest.raises(
        ValueError,
        match="superseded supervisor PID cannot be a protected PID",
    ):
        HybridCollectionSupervisorHandoffConfig(
            run_id="invalid",
            output_dir=tmp_path,
            claim_path=tmp_path / "claim.json",
            batch_progress_path=fixture["batch_path"],
            expected_batch_id="batch-1",
            observed_at_ts=1,
            superseded_supervisor=collector,
            successor_supervisor=fixture["successor"],
            protected_processes=fixture["protected"],
        )


def test_process_command_identity_mismatch_fails_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    alive = {100, 200, 300, 301, 302, 400}
    commands = _commands()
    commands[100] = "python unrelated.py"
    _patch_processes(monkeypatch, alive, commands)

    with pytest.raises(
        RuntimeError,
        match="superseded_supervisor PID command identity mismatch",
    ):
        _run(tmp_path, fixture, apply=True)
    assert not fixture["claim_path"].exists()


def test_forbidden_batch_outcome_field_fails_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    payload = _load_json(fixture["batch_path"])
    payload["settlement_pnl"] = 0.2
    _write_json(fixture["batch_path"], payload)
    _patch_processes(
        monkeypatch,
        {100, 200, 300, 301, 302, 400},
        _commands(),
    )

    with pytest.raises(
        ValueError,
        match="batch progress contains forbidden outcome fields",
    ):
        _run(tmp_path, fixture, apply=True)
    assert not fixture["claim_path"].exists()


def test_dry_run_validates_without_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    alive = {100, 200, 300, 301, 302, 400}
    _patch_processes(monkeypatch, alive, _commands())
    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_collection_supervisor_handoff._terminate_pid",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not terminate")
        ),
    )

    result = _run(tmp_path, fixture, apply=False)

    assert result["report"]["status"] == "dry_run_exclusive_handoff_validated"
    assert result["report"]["termination_attempted"] is False
    assert result["report"]["superseded_supervisor_alive_after"] is True
    assert alive == {100, 200, 300, 301, 302, 400}


def _fixture(tmp_path: Path) -> dict[str, Any]:
    scripts = {}
    for name in ("old", "successor", "downstream"):
        path = tmp_path / f"{name}.py"
        path.write_text(f"# {name}\n", encoding="utf-8")
        scripts[name] = path
    batch_path = tmp_path / "batch-progress.json"
    _write_json(
        batch_path,
        {
            "batch_id": "batch-1",
            "paper_only": True,
            "capital_at_risk": False,
            "capture_count": 2,
            "captures": [{"run_id": "a"}, {"run_id": "b"}],
            "error_count": 0,
            "errors": [],
        },
    )
    superseded = ProcessIdentity(
        role="superseded_supervisor",
        pid=100,
        required_command_substring="issue177_collection_supervisor.py",
        script_path=scripts["old"],
        expected_script_sha256=_sha256(scripts["old"]),
    )
    successor = ProcessIdentity(
        role="successor_supervisor",
        pid=200,
        required_command_substring=(
            "issue178_collection_handoff_supervisor.py"
        ),
        script_path=scripts["successor"],
        expected_script_sha256=_sha256(scripts["successor"]),
    )
    protected = (
        ProcessIdentity(
            role="active_collector",
            pid=300,
            required_command_substring=(
                "run_polymarket_async_round_collector.py"
            ),
        ),
        ProcessIdentity(
            role="round_comment_monitor",
            pid=301,
            required_command_substring="issue176_round_comment_monitor.py",
        ),
        ProcessIdentity(
            role="orderbook_watchdog",
            pid=302,
            required_command_substring=(
                "issue176_orderbook_failure_watchdog.py"
            ),
        ),
        ProcessIdentity(
            role="post_terminal_freeze_supervisor",
            pid=400,
            required_command_substring=(
                "issue183_post_terminal_freeze_supervisor.py"
            ),
            script_path=scripts["downstream"],
            expected_script_sha256=_sha256(scripts["downstream"]),
        ),
    )
    return {
        "batch_path": batch_path,
        "claim_path": tmp_path / "handoff-claim.json",
        "superseded": superseded,
        "successor": successor,
        "protected": protected,
    }


def _run(
    tmp_path: Path,
    fixture: dict[str, Any],
    *,
    apply: bool,
    run_id: str = "handoff",
) -> dict[str, Any]:
    return perform_exclusive_collection_supervisor_handoff(
        HybridCollectionSupervisorHandoffConfig(
            run_id=run_id,
            output_dir=tmp_path / "runs",
            claim_path=fixture["claim_path"],
            batch_progress_path=fixture["batch_path"],
            expected_batch_id="batch-1",
            observed_at_ts=123,
            superseded_supervisor=fixture["superseded"],
            successor_supervisor=fixture["successor"],
            protected_processes=fixture["protected"],
            apply_termination=apply,
            termination_wait_seconds=2.0,
        )
    )


def _patch_processes(
    monkeypatch: pytest.MonkeyPatch,
    alive: set[int],
    commands: dict[int, str],
) -> None:
    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_collection_supervisor_handoff._pid_alive",
        lambda pid: pid in alive,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "hybrid_pairwise_collection_supervisor_handoff._process_snapshot",
        lambda pid: {
            "pid": pid,
            "ppid": 1,
            "elapsed": "00:01",
            "stat": "S",
            "command": commands.get(pid, ""),
            "alive": pid in alive,
        },
    )


def _commands() -> dict[int, str]:
    return {
        100: "python /tmp/issue177_collection_supervisor.py",
        200: "python /tmp/issue178_collection_handoff_supervisor.py",
        300: "python examples/v8/run_polymarket_async_round_collector.py",
        301: "python /tmp/issue176_round_comment_monitor.py",
        302: "python /tmp/issue176_orderbook_failure_watchdog.py",
        400: "python /tmp/issue183_post_terminal_freeze_supervisor.py",
    }


def _assert_safety(payload: dict[str, Any]) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["broker_exchange_write_enabled"] is False
    assert payload["live_exchange_write_enabled"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
    assert payload["source_model_candidate_eligible"] is False
    assert payload["freeze_ready"] is False
    assert payload["promotion_evidence_eligible"] is False
    assert payload["v8_execution_handoff_allowed"] is False
    assert payload["#134_resume_allowed"] is False
    assert payload["#146_start_allowed"] is False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
