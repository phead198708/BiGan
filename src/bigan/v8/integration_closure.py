"""Deterministic, fail-closed verification for issue #264 integration closures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "bigan-issue-264-integration-closure-v1"
HEX_40_OR_64 = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
SAFETY_FLAGS = (
    "source_model_candidate_eligible",
    "freeze_ready",
    "promotion_evidence_eligible",
    "paper_candidate_allowed",
    "v8_execution_handoff_allowed",
    "live_trading_allowed",
    "wallet_signing_allowed",
    "polymarket_write_allowed",
    "capital_at_risk",
)


class IntegrationClosureError(ValueError):
    """Raised whenever an integration closure cannot be proven exactly."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=False,
        capture_output=True,
        text=text,
    )
    if result.returncode != 0:
        stderr = result.stderr if text else result.stderr.decode(errors="replace")
        raise IntegrationClosureError(
            f"git {' '.join(args)} failed closed: {stderr.strip()}"
        )
    return result.stdout


def _safe_repo_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise IntegrationClosureError(f"{field} must be a non-empty repository path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise IntegrationClosureError(f"unsafe or non-canonical {field}: {value!r}")
    return value


def _git_object_bytes(root: Path, commit: str, path: str) -> bytes:
    return _git(root, "show", f"{commit}:{path}", text=False)  # type: ignore[return-value]


def _git_blob_oid(root: Path, commit: str, path: str) -> str:
    return str(_git(root, "rev-parse", f"{commit}:{path}")).strip()


def _git_mode(root: Path, commit: str, path: str) -> str:
    output = str(_git(root, "ls-tree", commit, "--", path)).strip()
    if not output:
        raise IntegrationClosureError(f"missing source tree entry: {commit}:{path}")
    return output.split(maxsplit=1)[0]


def _working_blob_oid(root: Path, path: str) -> str:
    return str(_git(root, "hash-object", "--", path)).strip()


def _working_mode(root: Path, path: str) -> str:
    output = str(_git(root, "ls-files", "--stage", "--", path)).strip()
    if not output:
        raise IntegrationClosureError(f"destination is not tracked: {path}")
    return output.split(maxsplit=1)[0]


def _changed_paths(root: Path, base_commit: str) -> tuple[str, ...]:
    deleted = str(
        _git(root, "diff", "--name-only", "--diff-filter=D", base_commit, "HEAD")
    ).splitlines()
    if deleted:
        raise IntegrationClosureError(
            "integration closure may not delete paths: " + ", ".join(sorted(deleted))
        )
    changed = str(
        _git(
            root,
            "diff",
            "--name-only",
            "--diff-filter=ACMRT",
            base_commit,
            "HEAD",
        )
    ).splitlines()
    return tuple(sorted(path for path in changed if path))


def _import_reason(path: str) -> str:
    if path == "pyproject.toml":
        return "runtime_dependency_declaration"
    if path == ".github/workflows/v8-phase0.yml":
        return "integration_ci_hard_gate"
    if path == "src/bigan/v8/integration_closure.py":
        return "deterministic_integration_closure_verifier"
    if path == "tests/v8/test_issue264_integration_closure.py":
        return "integration_closure_fail_closed_regression_test"
    if path.startswith("src/"):
        return "runtime_dependency_closure"
    if path.startswith("tests/"):
        return "behavior_and_governance_regression_coverage"
    if path.endswith(".sha256"):
        return "frozen_artifact_hash_sidecar"
    if "/polymarket_artifacts/" in path:
        return "content_addressed_runtime_or_evidence_artifact"
    if "/polymarket_configs/" in path:
        return "frozen_protocol_or_evidence_artifact"
    if path.startswith("examples/v8/"):
        return "deterministic_operator_or_evaluator_entrypoint"
    return "dependency_closed_issue_264_integration"


def _walk_candidate_implementation_descriptors(
    value: object,
    *,
    pointer: tuple[str, ...] = (),
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        descriptor = value.get("candidate_implementation")
        if isinstance(descriptor, Mapping):
            yield "/" + "/".join((*pointer, "candidate_implementation")), descriptor
        for key, child in value.items():
            yield from _walk_candidate_implementation_descriptors(
                child,
                pointer=(*pointer, str(key)),
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_candidate_implementation_descriptors(
                child,
                pointer=(*pointer, str(index)),
            )


def _sidecar_target(root: Path, path: str) -> str | None:
    if not path.endswith(".sha256"):
        return None
    stem = path[: -len(".sha256")]
    direct = root / stem
    if direct.is_file():
        return stem
    json_target = root / f"{stem}.json"
    if json_target.is_file():
        return f"{stem}.json"
    raise IntegrationClosureError(f"cannot resolve sidecar target: {path}")


def _sidecar_path_for_target(
    entry_by_path: Mapping[str, Mapping[str, Any]], target: str
) -> str | None:
    candidates = [f"{target}.sha256"]
    if PurePosixPath(target).suffix:
        candidates.append(str(PurePosixPath(target).with_suffix(".sha256")))
    return next((path for path in candidates if path in entry_by_path), None)


def _artifact_bindings(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entry_by_path = {str(entry["destination_path"]): entry for entry in entries}
    bindings: list[dict[str, Any]] = []
    for path, entry in sorted(entry_by_path.items()):
        name = PurePosixPath(path).name
        if not path.endswith(".json"):
            continue
        if "terminal_review" in name:
            kind = "frozen_terminal"
        elif "reconciliation" in name:
            kind = "frozen_reconciliation"
        elif "frozen_artifact_binding_audit" in name:
            kind = "candidate_implementation_binding_audit"
        else:
            continue
        sidecar_path = _sidecar_path_for_target(entry_by_path, path)
        sidecar = entry_by_path.get(sidecar_path) if sidecar_path is not None else None
        if sidecar is None:
            raise IntegrationClosureError(f"missing frozen sidecar entry for {path}")
        bindings.append(
            {
                "artifact_path": path,
                "artifact_sha256": entry["destination_sha256"],
                "kind": kind,
                "sidecar_path": sidecar_path,
                "sidecar_sha256": sidecar["destination_sha256"],
            }
        )
    return bindings


def _candidate_bindings(root: Path, entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entry_by_path = {str(entry["destination_path"]): entry for entry in entries}
    bindings: list[dict[str, Any]] = []
    for protocol_path, protocol_entry in sorted(entry_by_path.items()):
        if not protocol_path.endswith("_protocol.json"):
            continue
        protocol = json.loads((root / protocol_path).read_text())
        for pointer, descriptor in _walk_candidate_implementation_descriptors(protocol):
            implementation_path = _safe_repo_path(
                descriptor.get("path"), field="candidate implementation path"
            )
            implementation_sha256 = descriptor.get("sha256")
            if not isinstance(implementation_sha256, str) or not HEX_64.fullmatch(
                implementation_sha256
            ):
                raise IntegrationClosureError(
                    f"invalid candidate implementation hash in {protocol_path}{pointer}"
                )
            implementation_entry = entry_by_path.get(implementation_path)
            if implementation_entry is None:
                raise IntegrationClosureError(
                    f"candidate implementation absent from closure: {implementation_path}"
                )
            if implementation_entry["destination_sha256"] != implementation_sha256:
                raise IntegrationClosureError(
                    f"candidate implementation descriptor mismatch: {protocol_path}{pointer}"
                )
            bindings.append(
                {
                    "candidate_implementation_path": implementation_path,
                    "candidate_implementation_sha256": implementation_sha256,
                    "descriptor_json_pointer": pointer,
                    "protocol_path": protocol_path,
                    "protocol_sha256": protocol_entry["destination_sha256"],
                }
            )
    return bindings


def build_integration_closure_manifest(
    *,
    root: Path,
    closure_id: str,
    destination_pr: int,
    destination_ref: str,
    base_ref: str,
    base_commit: str,
    self_paths: Sequence[str],
    source_catalog: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic manifest from an already committed integration tree."""

    root = root.resolve()
    excluded = {_safe_repo_path(path, field="self path") for path in self_paths}
    if len(excluded) != 2:
        raise IntegrationClosureError("manifest and sidecar must be the only self paths")
    sources: list[dict[str, Any]] = []
    for raw in source_catalog:
        source = {
            "source_pr": int(raw["source_pr"]),
            "source_ref": str(raw["source_ref"]),
            "source_commit": str(raw["source_commit"]),
        }
        if not HEX_40_OR_64.fullmatch(source["source_commit"]):
            raise IntegrationClosureError("invalid source commit")
        _git(root, "cat-file", "-e", f"{source['source_commit']}^{{commit}}")
        sources.append(source)

    entries: list[dict[str, Any]] = []
    for path in _changed_paths(root, base_commit):
        if path in excluded:
            continue
        destination = root / path
        if not destination.is_file():
            raise IntegrationClosureError(f"missing destination file: {path}")
        destination_bytes = destination.read_bytes()
        matched_source: dict[str, Any] | None = None
        source_bytes: bytes | None = None
        for source in sources:
            try:
                candidate = _git_object_bytes(root, source["source_commit"], path)
            except IntegrationClosureError:
                continue
            if candidate == destination_bytes:
                matched_source = source
                source_bytes = candidate
                break
        if matched_source is None or source_bytes is None:
            raise IntegrationClosureError(
                f"no declared source snapshot supplies exact destination bytes: {path}"
            )
        entry: dict[str, Any] = {
            **matched_source,
            "source_path": path,
            "source_git_blob_oid": _git_blob_oid(
                root, matched_source["source_commit"], path
            ),
            "source_git_mode": _git_mode(root, matched_source["source_commit"], path),
            "source_content_sha256": _sha256(source_bytes),
            "destination_path": path,
            "destination_git_blob_oid": _working_blob_oid(root, path),
            "destination_git_mode": _working_mode(root, path),
            "destination_sha256": _sha256(destination_bytes),
            "import_reason": _import_reason(path),
        }
        target = _sidecar_target(root, path)
        if target is not None:
            entry["sidecar_for"] = target
        entries.append(entry)

    entries.sort(key=lambda item: item["destination_path"])
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "closure_id": closure_id,
        "repository": "phead198708/BiGan",
        "destination_pr": destination_pr,
        "destination_ref": destination_ref,
        "base": {"ref": base_ref, "commit": base_commit},
        "generation_head": str(_git(root, "rev-parse", "HEAD")).strip(),
        "source_catalog": sources,
        "self_paths": sorted(excluded),
        "entries": entries,
        "frozen_evidence_bindings": _artifact_bindings(entries),
        "candidate_implementation_bindings": _candidate_bindings(root, entries),
        "safety": dict.fromkeys(SAFETY_FLAGS, False),
        "closure_policy": {
            "byte_drift_allowed": False,
            "deletions_allowed": False,
            "duplicate_destinations_allowed": False,
            "extra_changed_paths_allowed": False,
            "historical_evidence_rewrite_allowed": False,
        },
    }
    return payload


def write_manifest(payload: Mapping[str, Any], output_path: Path) -> str:
    """Write canonical manifest bytes and the raw-byte SHA-256 sidecar."""

    raw = _canonical_json_bytes(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    digest = _sha256(raw)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def _validate_structure(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise IntegrationClosureError("unsupported integration closure schema")
    base = payload.get("base")
    if not isinstance(base, Mapping) or not HEX_40_OR_64.fullmatch(
        str(base.get("commit", ""))
    ):
        raise IntegrationClosureError("invalid closure base commit")
    safety = payload.get("safety")
    if not isinstance(safety, Mapping) or any(safety.get(flag) is not False for flag in SAFETY_FLAGS):
        raise IntegrationClosureError("all integration safety flags must remain false")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise IntegrationClosureError("integration closure entries must be non-empty")
    destinations = [
        _safe_repo_path(entry.get("destination_path"), field="destination path")
        for entry in entries
        if isinstance(entry, Mapping)
    ]
    if len(destinations) != len(entries):
        raise IntegrationClosureError("every entry must be an object")
    if destinations != sorted(destinations):
        raise IntegrationClosureError("integration closure entries are not sorted")
    if len(destinations) != len(set(destinations)):
        raise IntegrationClosureError("duplicate destination path in closure")
    source_keys = [
        (
            entry.get("source_commit"),
            entry.get("source_path"),
            entry.get("destination_path"),
        )
        for entry in entries
    ]
    if len(source_keys) != len(set(source_keys)):
        raise IntegrationClosureError("duplicate source/destination entry in closure")


def verify_integration_closure_payload(
    payload: Mapping[str, Any],
    *,
    root: Path,
    verify_diff_inventory: bool = True,
) -> dict[str, Any]:
    """Verify source, destination, sidecar, evidence, and implementation bindings."""

    root = root.resolve()
    _validate_structure(payload)
    entries = payload["entries"]
    entry_by_path = {entry["destination_path"]: entry for entry in entries}

    for entry in entries:
        source_path = _safe_repo_path(entry["source_path"], field="source path")
        destination_path = _safe_repo_path(
            entry["destination_path"], field="destination path"
        )
        source_commit = str(entry["source_commit"])
        source_bytes = _git_object_bytes(root, source_commit, source_path)
        destination = root / destination_path
        if not destination.is_file():
            raise IntegrationClosureError(f"missing destination: {destination_path}")
        destination_bytes = destination.read_bytes()
        checks = {
            "source_git_blob_oid": _git_blob_oid(root, source_commit, source_path),
            "source_git_mode": _git_mode(root, source_commit, source_path),
            "source_content_sha256": _sha256(source_bytes),
            "destination_git_blob_oid": _working_blob_oid(root, destination_path),
            "destination_git_mode": _working_mode(root, destination_path),
            "destination_sha256": _sha256(destination_bytes),
        }
        for field, actual in checks.items():
            if entry.get(field) != actual:
                raise IntegrationClosureError(
                    f"{field} drift for {destination_path}: expected {entry.get(field)}, got {actual}"
                )
        if source_bytes != destination_bytes:
            raise IntegrationClosureError(
                f"source/destination byte drift for {destination_path}"
            )

    for path, entry in entry_by_path.items():
        target = _sidecar_target(root, path)
        if target is None:
            continue
        if entry.get("sidecar_for") != target:
            raise IntegrationClosureError(f"sidecar target mismatch: {path}")
        target_file = root / target
        if not target_file.is_file():
            raise IntegrationClosureError(f"sidecar target missing: {target}")
        expected = _sha256(target_file.read_bytes())
        actual = (root / path).read_text().strip()
        if actual != expected:
            raise IntegrationClosureError(f"sidecar content mismatch: {path}")

    expected_artifacts = _artifact_bindings(entries)
    if payload.get("frozen_evidence_bindings") != expected_artifacts:
        raise IntegrationClosureError("frozen terminal/reconciliation binding drift")
    expected_candidates = _candidate_bindings(root, entries)
    if payload.get("candidate_implementation_bindings") != expected_candidates:
        raise IntegrationClosureError("candidate implementation binding drift")

    if verify_diff_inventory:
        base_commit = str(payload["base"]["commit"])
        expected_paths = set(entry_by_path)
        self_paths_raw = payload.get("self_paths")
        if not isinstance(self_paths_raw, list):
            raise IntegrationClosureError("self_paths must be a list")
        self_paths = {
            _safe_repo_path(path, field="self path") for path in self_paths_raw
        }
        actual_paths = set(_changed_paths(root, base_commit))
        missing = sorted((expected_paths | self_paths) - actual_paths)
        extra = sorted(actual_paths - (expected_paths | self_paths))
        if missing or extra:
            raise IntegrationClosureError(
                f"closure inventory mismatch; missing={missing}, extra={extra}"
            )

    return {
        "closure_id": payload["closure_id"],
        "entry_count": len(entries),
        "frozen_evidence_binding_count": len(expected_artifacts),
        "candidate_implementation_binding_count": len(expected_candidates),
        "verification_passed": True,
    }


def verify_integration_closure(manifest_path: Path, *, root: Path) -> dict[str, Any]:
    """Load and verify a closure manifest plus its own SHA-256 sidecar."""

    manifest_path = manifest_path.resolve()
    raw = manifest_path.read_bytes()
    sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar_path.is_file():
        raise IntegrationClosureError(f"missing manifest sidecar: {sidecar_path}")
    expected = sidecar_path.read_text().strip()
    actual = _sha256(raw)
    if expected != actual:
        raise IntegrationClosureError(
            f"manifest SHA-256 mismatch: expected {expected}, got {actual}"
        )
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise IntegrationClosureError("manifest root must be an object")
    return verify_integration_closure_payload(payload, root=root)


def _parse_source(value: str) -> dict[str, Any]:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("source must be PR,REF,COMMIT")
    try:
        source_pr = int(parts[0])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source PR must be an integer") from exc
    return {
        "source_pr": source_pr,
        "source_ref": parts[1],
        "source_commit": parts[2],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--root", type=Path, default=Path.cwd())
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, default=Path.cwd())
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--closure-id", required=True)
    build.add_argument("--destination-pr", type=int, required=True)
    build.add_argument("--destination-ref", required=True)
    build.add_argument("--base-ref", required=True)
    build.add_argument("--base-commit", required=True)
    build.add_argument("--self-path", action="append", required=True)
    build.add_argument("--source", action="append", type=_parse_source, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        report = verify_integration_closure(args.manifest, root=args.root)
        print(json.dumps(report, sort_keys=True))
        return 0
    payload = build_integration_closure_manifest(
        root=args.root,
        closure_id=args.closure_id,
        destination_pr=args.destination_pr,
        destination_ref=args.destination_ref,
        base_ref=args.base_ref,
        base_commit=args.base_commit,
        self_paths=args.self_path,
        source_catalog=args.source,
    )
    digest = write_manifest(payload, args.output)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
