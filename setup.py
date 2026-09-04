"""Seal regular wheels with Git-verified source provenance (never editable trees)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

PROVENANCE_FILE = "_build_provenance.json"


def source_commit(root: Path, packaged: dict[str, bytes]) -> str | None:
    """Only attest bytes that match HEAD, including the build definition.

    Unrelated notes/tests/docs may be dirty. Missing Git, source archives, dirty
    build inputs, stale build output or untracked packaged code yield UNVERIFIED
    wheels: usable for development/mock, never accepted by live preflight.
    No user/environment SHA override is accepted.
    """
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, timeout=10)

    try:
        if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
            return None
        commit = git("rev-parse", "HEAD").decode().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            return None
        tree = {}
        for row in git("ls-tree", "-rz", commit, "--", "src/bigan", "setup.py", "pyproject.toml").split(b"\0"):
            if row:
                metadata, name = row.split(b"\t", 1)
                tree[name.decode()] = metadata.split()[2].decode()
        inputs = {f"src/bigan/{name}": value for name, value in packaged.items()}
        inputs.update({name: (root / name).read_bytes() for name in ("setup.py", "pyproject.toml")})
        required = {name for name in tree if name.endswith(".py") or name.startswith(
            "src/bigan/paper_trading/dashboard/static/"
        )}
        if not required.issubset(inputs):
            return None
        for name, contents in inputs.items():
            blob = b"blob " + str(len(contents)).encode() + b"\0" + contents
            if tree.get(name) != hashlib.sha1(blob).hexdigest():
                return None
        # Re-check HEAD so a concurrent checkout cannot silently change identity.
        return commit if git("rev-parse", "HEAD").decode().strip() == commit else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class ProvenanceBuildPy(build_py):
    def run(self) -> None:
        if self.editable_mode:
            # Do not stamp mutable source directories or change the checkout.
            return
        super().run()
        package = Path(self.build_lib) / "bigan"
        contents = {
            path.relative_to(package).as_posix(): path.read_bytes()
            for path in package.rglob("*")
            if path.is_file() and path.name != PROVENANCE_FILE
            and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
        }
        commit = source_commit(Path(__file__).resolve().parent, contents)
        payload = {"schema_version": 1, "kind": "wheel", "source_commit": commit,
                   "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())}}
        (package / PROVENANCE_FILE).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = super().get_outputs(include_bytecode)
        if not self.editable_mode:
            outputs.append(str(Path(self.build_lib) / "bigan" / PROVENANCE_FILE))
        return outputs


setup(cmdclass={"build_py": ProvenanceBuildPy})
