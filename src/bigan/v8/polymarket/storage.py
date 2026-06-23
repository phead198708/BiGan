"""Storage locations for v8 Polymarket trainable corpora."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

V8_TRAINING_CORPUS_ROOT = Path("/Volumes/PHILIPS/v8")
V8_POLYMARKET_TRAINING_CORPUS_DIR = V8_TRAINING_CORPUS_ROOT / "polymarket"


def require_v8_training_corpus_root() -> Path:
    """Return the mounted direct-training corpus root or fail clearly."""

    root = V8_TRAINING_CORPUS_ROOT.expanduser()
    if not root.parent.exists():
        raise FileNotFoundError(
            f"external volume parent is missing: {root.parent}; "
            "mount the training data volume before exporting corpora"
        )
    root.mkdir(parents=True, exist_ok=True)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"training corpus root is not a directory: {root}")
    return root.resolve()


def export_trainable_corpus(
    *,
    corpus_dir: Path | str,
    corpus_id: str,
    provenance: dict[str, Any],
    destination_root: Path | str | None = None,
    overwrite_existing: bool = False,
) -> Path:
    """Copy a validated direct-training corpus into the external corpus root."""

    if not corpus_id.strip():
        raise ValueError("corpus_id is required")
    source = Path(corpus_dir).expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"corpus_dir is not a directory: {source}")
    root = (
        require_v8_training_corpus_root()
        if destination_root is None
        else Path(destination_root).expanduser().resolve()
    )
    target = root / "polymarket" / corpus_id
    if target.exists():
        if not overwrite_existing:
            raise FileExistsError(
                f"training corpus already exists: {target}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    provenance_path = target / "training_corpus_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return target


def round_corpus_id_from_corpus_dir(corpus_dir: Path | str) -> str:
    """Return the single accepted Polymarket round slug for a corpus directory."""

    source = Path(corpus_dir).expanduser().resolve()
    metadata_path = source / "polymarket_market_metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing market metadata: {metadata_path}")
    slugs = {
        str(row.get("slug") or "").strip()
        for row in _read_jsonl(metadata_path)
        if str(row.get("slug") or "").strip()
    }
    if not slugs:
        raise ValueError(f"no round slug found in market metadata: {metadata_path}")
    if len(slugs) != 1:
        raise ValueError(
            "round-scoped training corpus export requires exactly one round slug; "
            f"found {len(slugs)}"
        )
    return next(iter(slugs))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows
