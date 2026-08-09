from __future__ import annotations

import hashlib
from pathlib import Path

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_collection_boundary_r2 import _write_new_frozen_json
from bigan.v8.polymarket.raw_capture_archive import (
    _split_payload,
    _write_deterministic_tar_gz,
    restore_recovered_capture_archive,
    verify_recovered_capture_archive,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_split_archive_deep_verifies_and_restores_exact_bytes(tmp_path) -> None:
    source_a = tmp_path / "captures/round-001/provider_raw/raw.jsonl"
    source_b = tmp_path / "captures/round-001/pending_round_capture_manifest.json"
    source_a.parent.mkdir(parents=True)
    source_a.write_bytes(b'{"value":1}\n' * 20)
    source_b.write_bytes(b'{"pending_resolution":true}\n')
    entries = [
        {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
            "role": role,
        }
        for path, role in (
            (source_a, "frozen_raw_stream"),
            (source_b, "pending_round_capture_manifest"),
        )
    ]
    index = tmp_path / "index.jsonl"
    index.write_text("{}\n", encoding="utf-8")
    source_index = {
        "path": index.relative_to(tmp_path).as_posix(),
        "sha256": _sha(index),
    }
    identity = {"source_index": source_index, "entries": entries}
    bundle_hash = canonical_json_sha256(identity)
    bundle = tmp_path / "artifacts" / bundle_hash
    bundle.mkdir(parents=True)
    payload = tmp_path / "payload.tar.gz"
    _write_deterministic_tar_gz(
        payload,
        entries=entries,
        repository_root=tmp_path,
    )
    chunks, payload_sha = _split_payload(
        payload,
        output_dir=bundle,
        chunk_bytes=97,
    )
    for chunk in chunks:
        chunk["path"] = (bundle / chunk.pop("name")).relative_to(tmp_path).as_posix()
    manifest = {
        "schema_version": "test",
        "bundle_hash": bundle_hash,
        "source_index": source_index,
        "entries": entries,
        "entry_count": 2,
        "raw_stream_count": 1,
        "pending_manifest_count": 1,
        "uncompressed_bytes": sum(row["size_bytes"] for row in entries),
        "archive": {
            "payload_sha256": payload_sha,
            "compressed_bytes": sum(row["size_bytes"] for row in chunks),
            "chunks": chunks,
        },
    }
    _write_new_frozen_json(bundle / "bundle_manifest.json", manifest)

    verified = verify_recovered_capture_archive(
        bundle_dir=bundle,
        repository_root=tmp_path,
        deep=True,
    )
    assert verified["deep_verified_entry_count"] == 2

    source_a.unlink()
    source_b.unlink()
    restored = restore_recovered_capture_archive(
        bundle_dir=bundle,
        repository_root=tmp_path,
    )
    assert restored["restored_entry_count"] == 2
    assert _sha(source_a) == entries[0]["sha256"]
    assert _sha(source_b) == entries[1]["sha256"]
