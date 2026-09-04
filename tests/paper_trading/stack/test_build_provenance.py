from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from bigan.build_provenance import PROVENANCE_FILE, BuildProvenanceError, _verify_package
from tests.paper_trading.stack.conftest import COMMIT


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "installed" / "bigan"
    root.mkdir(parents=True)
    source = b"VERSION = 'test'\n"
    (root / "__init__.py").write_bytes(source)
    (root / PROVENANCE_FILE).write_text(json.dumps({
        "schema_version": 1, "kind": "wheel", "source_commit": COMMIT,
        "files": {"__init__.py": hashlib.sha256(source).hexdigest()},
    }))
    return root


def test_sealed_build_needs_no_git_cwd_env_or_processes(package, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "f" * 40)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("runtime called Git"))
    proof = _verify_package(package)
    assert proof["source_commit"] == COMMIT and proof["kind"] == "wheel"
    assert str(package) not in json.dumps(proof)


@pytest.mark.parametrize("damage", ["missing", "modified", "injected", "deleted", "dirty", "symlink", "traversal"])
def test_unknown_or_modified_package_fails_closed(package, damage):
    if damage == "missing":
        (package / PROVENANCE_FILE).unlink()
    elif damage == "modified":
        (package / "__init__.py").write_text("VERSION = 'different'\n")
    elif damage == "injected":
        (package / "extra.py").write_text("NEW = True\n")
    elif damage == "deleted":
        (package / "__init__.py").unlink()
    elif damage == "symlink":
        (package / "extra.py").symlink_to(package / "__init__.py")
    else:
        manifest = json.loads((package / PROVENANCE_FILE).read_text())
        if damage == "dirty":
            manifest["source_commit"] = None
        else:
            manifest["files"]["../../private"] = "0" * 64
        (package / PROVENANCE_FILE).write_text(json.dumps(manifest))
    with pytest.raises(BuildProvenanceError) as exc:
        _verify_package(package)
    assert str(package) not in str(exc.value)
