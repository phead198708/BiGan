"""Read-only verification of the executing package, independent of cwd or Git.

Build metadata is not a signature. Trust the builder/distribution channel; this
detects accidental wrong revisions and locally altered/incomplete installations,
not an adversary able to replace both verifier and build metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PROVENANCE_FILE = "_build_provenance.json"


class BuildProvenanceError(ValueError):
    """Fixed diagnostic only; no paths, file contents or Git output."""


def _verify_package(package: Path) -> dict[str, str]:
    try:
        metadata = package / PROVENANCE_FILE
        if metadata.is_symlink() or metadata.stat().st_size > 4_000_000:
            raise BuildProvenanceError("INVALID_BUILD_PROVENANCE")
        encoded = metadata.read_bytes()
        payload = json.loads(encoded)
        if (not isinstance(payload, dict) or set(payload) != {"schema_version", "kind", "source_commit", "files"}
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or payload["kind"] != "wheel"):
            raise BuildProvenanceError("INVALID_BUILD_PROVENANCE")
        commit = payload["source_commit"]
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise BuildProvenanceError("UNVERIFIED_BUILD_SOURCE")
        expected = payload["files"]
        if not isinstance(expected, dict) or not 1 <= len(expected) <= 20000:
            raise BuildProvenanceError("INVALID_BUILD_PROVENANCE")
        actual = {}
        for path in package.rglob("*"):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise BuildProvenanceError("ALTERED_BUILD_CONTENTS")
            if path.is_file() and path != metadata:
                actual[path.relative_to(package).as_posix()] = path
        if set(actual) != set(expected):
            raise BuildProvenanceError("ALTERED_BUILD_CONTENTS")
        for name, path in actual.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
                raise BuildProvenanceError("ALTERED_BUILD_CONTENTS")
        return {"kind": "wheel", "source_commit": commit, "manifest_sha256": hashlib.sha256(encoded).hexdigest()}
    except BuildProvenanceError:
        raise
    except (OSError, ValueError, TypeError, RecursionError):
        raise BuildProvenanceError("BUILD_PROVENANCE_UNAVAILABLE") from None


def runtime_provenance() -> dict[str, str]:
    """Verify bytes beside THIS imported module. Never infer revision from cwd."""
    return _verify_package(Path(__file__).resolve().parent)


def require_source_commit(expected: str) -> dict[str, str]:
    provenance = runtime_provenance()
    if expected != provenance["source_commit"]:
        raise BuildProvenanceError("SOURCE_COMMIT_BUILD_MISMATCH")
    return provenance


if __name__ == "__main__":
    try:
        print(json.dumps(runtime_provenance(), sort_keys=True))
    except BuildProvenanceError as exc:
        print(str(exc))
        raise SystemExit(2) from None
