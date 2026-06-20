#!/usr/bin/env python3
"""Clean bulky paper-run artifacts after a run has finished.

The default profile keeps the compact training/replay material and removes
ephemeral queues, feature state snapshots, and debug logs. It is intentionally
dry-run by default; pass ``--execute`` to delete files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROFILES = ("training", "prediction-warehouse")


@dataclass(frozen=True, slots=True)
class CleanupCandidate:
    path: str
    kind: str
    size_bytes: int
    reason: str


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    generated_at: str
    profile: str
    dry_run: bool
    live_root: str | None
    scorer_log_dir: str | None
    paper_log_dir: str | None
    summary_paths: list[str]
    preserved_paths: list[str]
    delete_candidates: list[CleanupCandidate]
    deleted_paths: list[str]
    freed_bytes: int
    notes: list[str]


def build_cleanup_plan(
    *,
    live_root: Path | None = None,
    scorer_log_dir: Path | None = None,
    paper_log_dir: Path | None = None,
    profile: str = "training",
    execute: bool = False,
    allow_incomplete: bool = False,
    keep_signal_queues: bool = False,
) -> CleanupPlan:
    """Return the cleanup plan, optionally deleting candidate paths."""

    if profile not in PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(PROFILES)}")
    roots = [path for path in (live_root, scorer_log_dir, paper_log_dir) if path is not None]
    if not roots:
        raise ValueError("at least one of live_root, scorer_log_dir, or paper_log_dir is required")
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)

    notes: list[str] = []
    if live_root is not None:
        _raise_if_live_root_active(live_root)

    summary_paths = _summary_paths(paper_log_dir)
    if not summary_paths and not allow_incomplete:
        raise RuntimeError(
            "no phase4 summary found; refusing to clean an incomplete run "
            "(pass --allow-incomplete to override)"
        )
    if not summary_paths:
        notes.append("no phase4 summary found; cleanup allowed by allow_incomplete")

    candidates = _dedupe_candidates(
        [
            *_live_root_candidates(
                live_root,
                profile=profile,
                keep_signal_queues=keep_signal_queues,
            ),
            *_scorer_log_candidates(scorer_log_dir),
            *_paper_log_candidates(paper_log_dir),
        ]
    )
    preserved_paths = _preserved_paths(
        live_root=live_root,
        scorer_log_dir=scorer_log_dir,
        paper_log_dir=paper_log_dir,
        profile=profile,
    )

    deleted_paths: list[str] = []
    freed_bytes = 0
    if execute:
        for candidate in candidates:
            target = Path(candidate.path)
            if not target.exists():
                continue
            _delete_path(target)
            deleted_paths.append(candidate.path)
            freed_bytes += candidate.size_bytes
        _remove_empty_dirs([live_root, scorer_log_dir, paper_log_dir])

    return CleanupPlan(
        generated_at=datetime.now(UTC).isoformat(),
        profile=profile,
        dry_run=not execute,
        live_root=str(live_root) if live_root is not None else None,
        scorer_log_dir=str(scorer_log_dir) if scorer_log_dir is not None else None,
        paper_log_dir=str(paper_log_dir) if paper_log_dir is not None else None,
        summary_paths=[str(path) for path in summary_paths],
        preserved_paths=preserved_paths,
        delete_candidates=candidates,
        deleted_paths=deleted_paths,
        freed_bytes=freed_bytes,
        notes=notes,
    )


def resolve_run_paths(run_id: str, *, data_root: Path = Path("data")) -> tuple[Path | None, Path | None, Path | None]:
    live_matches = sorted((data_root / "live").glob(f"*{run_id}*"))
    log_matches = sorted((data_root / "logs").glob(f"*{run_id}*"))
    live_root = _single_or_none(live_matches, "live root", run_id)
    scorer_log_dir = _single_or_none(
        [path for path in log_matches if path.name.startswith("champion-live")],
        "scorer log dir",
        run_id,
    )
    paper_log_dir = _single_or_none(
        [path for path in log_matches if "paper-shadow" in path.name],
        "paper log dir",
        run_id,
    )
    return live_root, scorer_log_dir, paper_log_dir


def write_manifest(plan: CleanupPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_plan_json(plan), encoding="utf-8")


def _live_root_candidates(
    live_root: Path | None,
    *,
    profile: str,
    keep_signal_queues: bool,
) -> list[CleanupCandidate]:
    if live_root is None:
        return []
    candidates: list[CleanupCandidate] = []
    for pattern, reason in (
        ("low-latency/*.jsonl", "ephemeral low-latency queue"),
        ("low-latency/*state*.json", "ephemeral feature state snapshot"),
        ("low-latency/*.cursor", "ephemeral queue cursor"),
        ("low-latency/*.tmp", "temporary low-latency state"),
    ):
        candidates.extend(_candidate_glob(live_root, pattern, reason))
    if not keep_signal_queues:
        candidates.extend(_candidate_glob(live_root, "signals*.jsonl", "executor-ready queue snapshot"))
    lock_dir = live_root / ".run_champion_live.lock"
    if lock_dir.exists():
        candidates.append(_candidate(lock_dir, "stale live-root lock"))
    if profile == "prediction-warehouse":
        for dirname in ("raw", "rollup"):
            path = live_root / dirname
            if path.exists():
                candidates.append(_candidate(path, f"not needed by prediction-warehouse profile: {dirname}"))
    return candidates


def _scorer_log_candidates(scorer_log_dir: Path | None) -> list[CleanupCandidate]:
    if scorer_log_dir is None:
        return []
    candidates: list[CleanupCandidate] = []
    for pattern, reason in (
        ("screen.stdout.log", "scorer debug stdout"),
        ("capture-*.log", "capture debug log"),
        ("scorer-*.log", "scorer diagnostic log"),
        ("*.tmp", "temporary scorer artifact"),
    ):
        candidates.extend(_candidate_glob(scorer_log_dir, pattern, reason))
    return candidates


def _paper_log_candidates(paper_log_dir: Path | None) -> list[CleanupCandidate]:
    if paper_log_dir is None:
        return []
    candidates: list[CleanupCandidate] = []
    for path in paper_log_dir.glob("**/screen.stdout.log"):
        candidates.append(_candidate(path, "paper executor debug stdout"))
    for path in paper_log_dir.glob("**/*.tmp"):
        candidates.append(_candidate(path, "temporary paper artifact"))
    return candidates


def _candidate_glob(root: Path, pattern: str, reason: str) -> list[CleanupCandidate]:
    return [_candidate(path, reason) for path in root.glob(pattern)]


def _candidate(path: Path, reason: str) -> CleanupCandidate:
    return CleanupCandidate(
        path=str(path),
        kind="dir" if path.is_dir() else "file",
        size_bytes=_path_size(path),
        reason=reason,
    )


def _dedupe_candidates(candidates: list[CleanupCandidate]) -> list[CleanupCandidate]:
    by_path: dict[str, CleanupCandidate] = {}
    for candidate in candidates:
        by_path.setdefault(candidate.path, candidate)
    return sorted(by_path.values(), key=lambda item: item.path)


def _preserved_paths(
    *,
    live_root: Path | None,
    scorer_log_dir: Path | None,
    paper_log_dir: Path | None,
    profile: str,
) -> list[str]:
    paths: list[Path] = []
    if live_root is not None:
        for dirname in ("warehouse",):
            path = live_root / dirname
            if path.exists():
                paths.append(path)
        if profile == "training":
            for dirname in ("raw", "rollup"):
                path = live_root / dirname
                if path.exists():
                    paths.append(path)
    if paper_log_dir is not None:
        paths.extend(_summary_paths(paper_log_dir))
        paths.extend(sorted(paper_log_dir.glob("phase4-*.jsonl")))
        paths.extend(sorted(paper_log_dir.glob("phase4-*-signals-by-round")))
    if scorer_log_dir is not None and not any(scorer_log_dir.iterdir()):
        paths.append(scorer_log_dir)
    return sorted({str(path) for path in paths})


def _summary_paths(paper_log_dir: Path | None) -> list[Path]:
    if paper_log_dir is None or not paper_log_dir.exists():
        return []
    return sorted(paper_log_dir.glob("phase4-*-summary.json"))


def _raise_if_live_root_active(live_root: Path) -> None:
    pid_file = live_root / ".run_champion_live.lock" / "pid"
    if not pid_file.exists():
        return
    raw = pid_file.read_text(encoding="utf-8").strip()
    if not raw.isdigit():
        return
    pid = int(raw)
    if _pid_is_alive(pid):
        raise RuntimeError(f"live root still appears active: {live_root} pid={pid}")


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file() or child.is_symlink():
            total += int(child.stat().st_size)
    return total


def _delete_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _remove_empty_dirs(roots: list[Path | None]) -> None:
    for root in roots:
        if root is None or not root.exists():
            continue
        for path in sorted([item for item in root.rglob("*") if item.is_dir()], reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass


def _single_or_none(paths: list[Path], label: str, run_id: str) -> Path | None:
    if len(paths) > 1:
        joined = ", ".join(str(path) for path in paths)
        raise RuntimeError(f"ambiguous {label} for run id {run_id}: {joined}")
    return paths[0] if paths else None


def _plan_json(plan: CleanupPlan) -> str:
    payload = asdict(plan)
    payload["delete_candidates"] = [asdict(candidate) for candidate in plan.delete_candidates]
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Resolve data/live and data/logs directories containing this run id.")
    parser.add_argument("--data-root", default="data", help="Data root used with --run-id. Default: data")
    parser.add_argument("--live-root")
    parser.add_argument("--scorer-log-dir")
    parser.add_argument("--paper-log-dir")
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="training",
        help=(
            "training keeps warehouse plus compressed raw/rollup source data; "
            "prediction-warehouse keeps only warehouse material."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Delete candidates. Default is dry-run.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow cleanup without a phase4 summary. Use only for abandoned runs.",
    )
    parser.add_argument(
        "--keep-signal-queues",
        action="store_true",
        help="Keep live-root signals*.jsonl queue snapshots.",
    )
    parser.add_argument("--manifest-path", help="Write the cleanup manifest to this path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    live_root = Path(args.live_root) if args.live_root else None
    scorer_log_dir = Path(args.scorer_log_dir) if args.scorer_log_dir else None
    paper_log_dir = Path(args.paper_log_dir) if args.paper_log_dir else None
    if args.run_id:
        resolved_live_root, resolved_scorer_log_dir, resolved_paper_log_dir = resolve_run_paths(
            args.run_id,
            data_root=Path(args.data_root),
        )
        live_root = live_root or resolved_live_root
        scorer_log_dir = scorer_log_dir or resolved_scorer_log_dir
        paper_log_dir = paper_log_dir or resolved_paper_log_dir

    plan = build_cleanup_plan(
        live_root=live_root,
        scorer_log_dir=scorer_log_dir,
        paper_log_dir=paper_log_dir,
        profile=args.profile,
        execute=args.execute,
        allow_incomplete=args.allow_incomplete,
        keep_signal_queues=args.keep_signal_queues,
    )
    if args.manifest_path:
        write_manifest(plan, Path(args.manifest_path))
    elif args.execute and paper_log_dir is not None and paper_log_dir.exists():
        write_manifest(plan, paper_log_dir / "run-artifact-cleanup-manifest.json")

    sys.stdout.write(_plan_json(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
