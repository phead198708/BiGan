import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "cleanup_paper_run_artifacts.py"

spec = importlib.util.spec_from_file_location("cleanup_paper_run_artifacts", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_cleanup_plan_keeps_training_material_and_dry_runs(tmp_path: Path) -> None:
    live_root, scorer_log_dir, paper_log_dir = _write_run_tree(tmp_path, summary=True)

    plan = module.build_cleanup_plan(
        live_root=live_root,
        scorer_log_dir=scorer_log_dir,
        paper_log_dir=paper_log_dir,
    )

    assert plan.dry_run is True
    assert _candidate_reasons(plan)["ephemeral low-latency queue"]
    assert _candidate_reasons(plan)["ephemeral feature state snapshot"]
    assert _candidate_reasons(plan)["scorer diagnostic log"]
    assert str(live_root / "warehouse") in plan.preserved_paths
    assert str(live_root / "raw") in plan.preserved_paths
    assert (live_root / "low-latency" / "raw-btc15m.jsonl").exists()
    assert (scorer_log_dir / "scorer-20260619.log").exists()


def test_cleanup_execute_removes_ephemeral_paths_and_preserves_warehouse(tmp_path: Path) -> None:
    live_root, scorer_log_dir, paper_log_dir = _write_run_tree(tmp_path, summary=True)

    plan = module.build_cleanup_plan(
        live_root=live_root,
        scorer_log_dir=scorer_log_dir,
        paper_log_dir=paper_log_dir,
        execute=True,
    )

    assert plan.dry_run is False
    assert plan.freed_bytes > 0
    assert not (live_root / "low-latency" / "raw-btc15m.jsonl").exists()
    assert not (live_root / "low-latency" / "features-state.json").exists()
    assert not (scorer_log_dir / "scorer-20260619.log").exists()
    assert not (paper_log_dir / "low-confidence-scalp" / "screen.stdout.log").exists()
    assert (live_root / "warehouse" / "predictions" / "part.parquet").exists()
    assert (live_root / "raw" / "ws_market" / "2026-06-19.ndjson.gz").exists()
    assert (paper_log_dir / "phase4-20260619-summary.json").exists()
    assert (paper_log_dir / "phase4-20260619.jsonl").exists()


def test_prediction_warehouse_profile_removes_raw_source(tmp_path: Path) -> None:
    live_root, scorer_log_dir, paper_log_dir = _write_run_tree(tmp_path, summary=True)

    plan = module.build_cleanup_plan(
        live_root=live_root,
        scorer_log_dir=scorer_log_dir,
        paper_log_dir=paper_log_dir,
        profile="prediction-warehouse",
        execute=True,
    )

    assert not (live_root / "raw").exists()
    assert (live_root / "warehouse" / "predictions" / "part.parquet").exists()
    assert any("prediction-warehouse" in candidate.reason for candidate in plan.delete_candidates)


def test_cleanup_refuses_incomplete_run_by_default(tmp_path: Path) -> None:
    live_root, scorer_log_dir, paper_log_dir = _write_run_tree(tmp_path, summary=False)

    try:
        module.build_cleanup_plan(
            live_root=live_root,
            scorer_log_dir=scorer_log_dir,
            paper_log_dir=paper_log_dir,
        )
    except RuntimeError as exc:
        assert "no phase4 summary found" in str(exc)
    else:
        raise AssertionError("expected incomplete run refusal")


def test_cleanup_refuses_active_live_root(tmp_path: Path) -> None:
    live_root, scorer_log_dir, paper_log_dir = _write_run_tree(tmp_path, summary=True)
    lock = live_root / ".run_champion_live.lock"
    lock.mkdir()
    (lock / "pid").write_text(str(os.getpid()), encoding="utf-8")

    try:
        module.build_cleanup_plan(
            live_root=live_root,
            scorer_log_dir=scorer_log_dir,
            paper_log_dir=paper_log_dir,
        )
    except RuntimeError as exc:
        assert "still appears active" in str(exc)
    else:
        raise AssertionError("expected active live root refusal")


def test_cli_writes_manifest_on_execute(tmp_path: Path, capsys) -> None:
    live_root, scorer_log_dir, paper_log_dir = _write_run_tree(tmp_path, summary=True)
    manifest_path = tmp_path / "cleanup.json"

    status = module.main(
        [
            "--live-root",
            str(live_root),
            "--scorer-log-dir",
            str(scorer_log_dir),
            "--paper-log-dir",
            str(paper_log_dir),
            "--execute",
            "--manifest-path",
            str(manifest_path),
        ]
    )

    assert status == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is False
    assert payload["freed_bytes"] > 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["paper_log_dir"] == str(paper_log_dir)


def _write_run_tree(tmp_path: Path, *, summary: bool) -> tuple[Path, Path, Path]:
    live_root = tmp_path / "data" / "live" / "xgboost-v7-lowconf-20260619T000000Z"
    scorer_log_dir = tmp_path / "data" / "logs" / "champion-live-v7-lowconf-20260619T000000Z"
    paper_log_dir = (
        tmp_path
        / "data"
        / "logs"
        / "xgboost-v7-paper-shadow-20260619T000000Z-lowconf-scalp-30round"
    )
    _write(live_root / "warehouse" / "predictions" / "part.parquet", "warehouse")
    _write(live_root / "warehouse" / "features_15m_v1" / "part.parquet", "features")
    _write(live_root / "raw" / "ws_market" / "2026-06-19.ndjson.gz", "raw")
    _write(live_root / "rollup" / "ws_market" / "2026-06-19.parquet", "rollup")
    _write(live_root / "low-latency" / "raw-btc15m.jsonl", "raw-queue")
    _write(live_root / "low-latency" / "features-state.json", "{}")
    _write(live_root / "low-latency" / "features.cursor", "10")
    _write(live_root / "signals.jsonl", "{}\n")
    _write(scorer_log_dir / "screen.stdout.log", "stdout")
    _write(scorer_log_dir / "scorer-20260619.log", "scorer")
    _write(scorer_log_dir / "capture-20260619.log", "capture")
    _write(paper_log_dir / "phase4-20260619.jsonl", "{}\n")
    if summary:
        _write(paper_log_dir / "phase4-20260619-summary.json", "{}")
    _write(paper_log_dir / "phase4-20260619-signals-by-round" / "round.csv", "round")
    _write(paper_log_dir / "low-confidence-scalp" / "screen.stdout.log", "paper stdout")
    return live_root, scorer_log_dir, paper_log_dir


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _candidate_reasons(plan) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    for candidate in plan.delete_candidates:
        reasons.setdefault(candidate.reason, []).append(candidate.path)
    return reasons
