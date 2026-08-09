"""Content-addressed recovery bundle for ignored BTC 15m raw captures."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_collection_boundary_r2 import _write_new_frozen_json
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

DEFAULT_INDEX = REPO_ROOT / (
    "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2/"
    "confirmatory_collection_freeze_001/raw_evidence_manifest_index.jsonl"
)
DEFAULT_ARCHIVE_PARENT = REPO_ROOT / (
    "examples/v8/polymarket_artifacts/raw_archives/sha256"
)
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024


def inventory_recovered_capture_archive(
    *,
    index_path: Path | str = DEFAULT_INDEX,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Hash every recovered raw stream and bind every pending manifest."""

    root = Path(repository_root).resolve()
    index = Path(index_path).resolve()
    if not index.is_relative_to(root):
        raise ValueError("raw evidence index escaped repository")
    records = _load_jsonl(index)
    if len(records) != 889:
        raise ValueError("raw evidence index must contain exactly 889 records")
    entries: dict[str, dict[str, Any]] = {}
    for record in records:
        raw_streams = dict(record["raw_streams"])
        if len(raw_streams) != 5:
            raise ValueError("each raw evidence record must bind five streams")
        raw_paths = []
        for name, descriptor in raw_streams.items():
            path = (root / str(descriptor["path"])).resolve()
            _assert_repository_file(path, root)
            actual = sha256_file(path)
            if actual != descriptor["sha256"]:
                raise ValueError(f"recovered raw stream SHA mismatch: {path}")
            relative = path.relative_to(root).as_posix()
            if relative in entries:
                raise ValueError("raw recovery archive contains a duplicate path")
            entries[relative] = {
                "path": relative,
                "sha256": actual,
                "size_bytes": path.stat().st_size,
                "role": "frozen_raw_stream",
                "stream_name": name,
            }
            raw_paths.append(path)
        manifest_path = raw_paths[0].parent.parent / "pending_round_capture_manifest.json"
        _assert_repository_file(manifest_path, root)
        manifest = _load_json(manifest_path)
        for name, descriptor in raw_streams.items():
            manifest_hash = (
                manifest.get("provider_chainlink_raw_artifact_sha256")
                if name == "raw_polymarket_chainlink_prices.jsonl"
                else dict(manifest["provider_raw_artifact_hashes"]).get(name)
            )
            if manifest_hash != descriptor["sha256"]:
                raise ValueError("pending manifest raw-stream binding mismatch")
        relative_manifest = manifest_path.relative_to(root).as_posix()
        if relative_manifest in entries:
            raise ValueError("raw recovery archive contains a duplicate manifest")
        entries[relative_manifest] = {
            "path": relative_manifest,
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "role": "pending_round_capture_manifest",
            "run_id": manifest["run_id"],
        }
    ordered = [entries[name] for name in sorted(entries)]
    raw_count = sum(row["role"] == "frozen_raw_stream" for row in ordered)
    manifest_count = sum(
        row["role"] == "pending_round_capture_manifest" for row in ordered
    )
    if raw_count != 4445 or manifest_count != 889:
        raise ValueError("recovered raw archive population mismatch")
    identity = {
        "source_index": {
            "path": index.relative_to(root).as_posix(),
            "sha256": sha256_file(index),
        },
        "entries": ordered,
    }
    return {
        "bundle_hash": canonical_json_sha256(identity),
        "source_index": identity["source_index"],
        "entry_count": len(ordered),
        "raw_stream_count": raw_count,
        "pending_manifest_count": manifest_count,
        "uncompressed_bytes": sum(int(row["size_bytes"]) for row in ordered),
        "entries": ordered,
    }


def build_recovered_capture_archive(
    *,
    index_path: Path | str = DEFAULT_INDEX,
    archive_parent: Path | str = DEFAULT_ARCHIVE_PARENT,
    repository_root: Path | str = REPO_ROOT,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    """Build a deterministic, split tar.gz bundle from recovered byte-identical files."""

    if chunk_bytes <= 0 or chunk_bytes >= 100 * 1024 * 1024:
        raise ValueError("archive chunks must be positive and below GitHub's 100 MiB limit")
    root = Path(repository_root).resolve()
    parent = Path(archive_parent).resolve()
    if not parent.is_relative_to(root):
        raise ValueError("raw archive output escaped repository")
    inventory = inventory_recovered_capture_archive(
        index_path=index_path,
        repository_root=root,
    )
    output = parent / inventory["bundle_hash"]
    if output.exists():
        raise FileExistsError(f"raw recovery bundle already exists: {output}")
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="raw-archive-", dir=parent) as temporary:
        temporary_path = Path(temporary)
        payload = temporary_path / "btc15m_moe_raw_capture_archive.tar.gz"
        _write_deterministic_tar_gz(
            payload,
            entries=inventory["entries"],
            repository_root=root,
        )
        chunks, payload_sha256 = _split_payload(
            payload,
            output_dir=temporary_path,
            chunk_bytes=chunk_bytes,
        )
        manifest = {
            "schema_version": "bigan-btc-15m-raw-capture-recovery-bundle-v1",
            "lineage_id": "BTC-15M-MoE-confirmatory-v2",
            "role": "byte_identical_recovered_outcome_blind_capture_archive",
            "bundle_hash": inventory["bundle_hash"],
            "source_index": inventory["source_index"],
            "entry_count": inventory["entry_count"],
            "raw_stream_count": inventory["raw_stream_count"],
            "pending_manifest_count": inventory["pending_manifest_count"],
            "uncompressed_bytes": inventory["uncompressed_bytes"],
            "entries": inventory["entries"],
            "archive": {
                "format": "deterministic_tar_gzip",
                "gzip_compresslevel": 9,
                "gzip_mtime": 0,
                "tar_member_mtime": 0,
                "tar_member_uid_gid": 0,
                "payload_sha256": payload_sha256,
                "compressed_bytes": sum(row["size_bytes"] for row in chunks),
                "chunk_size_limit_bytes": chunk_bytes,
                "chunks": chunks,
            },
            "recovery_complete": True,
            "raw_files_hash_verified": 4445,
            "raw_files_hash_mismatched": 0,
            "pending_manifests_hash_bound": 889,
            "source_capture_mutated": False,
            "promotion_evidence_eligible": False,
        }
        output.mkdir()
        for chunk in chunks:
            source = temporary_path / chunk["name"]
            destination = output / chunk["name"]
            source.replace(destination)
            chunk["path"] = destination.relative_to(root).as_posix()
            chunk.pop("name")
        manifest_artifact = _write_new_frozen_json(
            output / "bundle_manifest.json",
            manifest,
        )
    verification = verify_recovered_capture_archive(
        bundle_dir=output,
        repository_root=root,
        deep=True,
    )
    return {
        "bundle_dir": output,
        "bundle_hash": inventory["bundle_hash"],
        "bundle_manifest_sha256": manifest_artifact["sha256"],
        **verification,
    }


def verify_recovered_capture_archive(
    *,
    bundle_dir: Path | str,
    repository_root: Path | str = REPO_ROOT,
    deep: bool = True,
) -> dict[str, Any]:
    """Verify manifest, split payload, and optionally every tar member byte."""

    root = Path(repository_root).resolve()
    bundle = Path(bundle_dir).resolve()
    if not bundle.is_relative_to(root):
        raise ValueError("raw archive bundle escaped repository")
    manifest_path = bundle / "bundle_manifest.json"
    sidecar = bundle / "bundle_manifest.sha256"
    if not manifest_path.is_file() or not sidecar.is_file():
        raise ValueError("raw archive bundle manifest is not frozen")
    if sha256_file(manifest_path) != sidecar.read_text(encoding="utf-8").strip():
        raise ValueError("raw archive bundle manifest sidecar mismatch")
    manifest = _load_json(manifest_path)
    identity = {
        "source_index": manifest["source_index"],
        "entries": manifest["entries"],
    }
    if canonical_json_sha256(identity) != manifest["bundle_hash"]:
        raise ValueError("raw archive content-address identity mismatch")
    chunk_paths = []
    payload_hash = hashlib.sha256()
    total = 0
    for descriptor in manifest["archive"]["chunks"]:
        path = (root / descriptor["path"]).resolve()
        _assert_repository_file(path, root)
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError("raw archive chunk SHA mismatch")
        if path.stat().st_size != descriptor["size_bytes"]:
            raise ValueError("raw archive chunk size mismatch")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                payload_hash.update(block)
                total += len(block)
        chunk_paths.append(path)
    if payload_hash.hexdigest() != manifest["archive"]["payload_sha256"]:
        raise ValueError("raw archive combined payload SHA mismatch")
    if total != manifest["archive"]["compressed_bytes"]:
        raise ValueError("raw archive combined payload size mismatch")
    deep_count = 0
    if deep:
        deep_count = _verify_tar_members(
            chunk_paths,
            entries=manifest["entries"],
        )
    return {
        "verification_passed": True,
        "deep_verification_performed": deep,
        "deep_verified_entry_count": deep_count,
        "entry_count": manifest["entry_count"],
        "raw_stream_count": manifest["raw_stream_count"],
        "pending_manifest_count": manifest["pending_manifest_count"],
        "compressed_bytes": total,
        "uncompressed_bytes": manifest["uncompressed_bytes"],
    }


def restore_recovered_capture_archive(
    *,
    bundle_dir: Path | str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Restore missing files without overwriting any existing byte mismatch."""

    root = Path(repository_root).resolve()
    bundle = Path(bundle_dir).resolve()
    verification = verify_recovered_capture_archive(
        bundle_dir=bundle,
        repository_root=root,
        deep=False,
    )
    manifest = _load_json(bundle / "bundle_manifest.json")
    chunk_paths = [
        (root / row["path"]).resolve() for row in manifest["archive"]["chunks"]
    ]
    expected = {row["path"]: row for row in manifest["entries"]}
    restored = 0
    already_present = 0
    with _tar_reader(chunk_paths) as archive:
        for member in archive:
            if member.name not in expected or not member.isfile():
                raise ValueError("raw archive contains an unexpected member")
            descriptor = expected[member.name]
            destination = (root / member.name).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("raw archive member escaped repository")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("raw archive member has no payload")
            if destination.exists():
                if sha256_file(destination) != descriptor["sha256"]:
                    raise ValueError("existing raw archive destination differs; fail closed")
                _drain(source)
                already_present += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with destination.open("xb") as handle:
                for block in iter(
                    lambda stream=source: stream.read(1024 * 1024), b""
                ):
                    handle.write(block)
                    digest.update(block)
            if digest.hexdigest() != descriptor["sha256"]:
                raise ValueError("restored raw archive member SHA mismatch")
            restored += 1
    return {
        **verification,
        "restored_entry_count": restored,
        "already_present_entry_count": already_present,
        "restore_fail_closed": True,
    }


def _write_deterministic_tar_gz(
    path: Path,
    *,
    entries: Sequence[Mapping[str, Any]],
    repository_root: Path,
) -> None:
    with (
        path.open("xb") as compressed,
        gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=compressed,
            mtime=0,
        ) as gzip_file,
        tarfile.open(
            fileobj=gzip_file,
            mode="w|",
            format=tarfile.PAX_FORMAT,
        ) as archive,
    ):
        for descriptor in entries:
            source = (repository_root / descriptor["path"]).resolve()
            _assert_repository_file(source, repository_root)
            info = tarfile.TarInfo(name=descriptor["path"])
            info.size = int(descriptor["size_bytes"])
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with source.open("rb") as handle:
                archive.addfile(info, fileobj=handle)


def _split_payload(
    payload: Path,
    *,
    output_dir: Path,
    chunk_bytes: int,
) -> tuple[list[dict[str, Any]], str]:
    payload_hash = hashlib.sha256()
    chunks = []
    with payload.open("rb") as source:
        chunk_index = 0
        while True:
            data = source.read(chunk_bytes)
            if not data:
                break
            payload_hash.update(data)
            name = f"btc15m_moe_raw_capture_archive.tar.gz.part-{chunk_index:03d}"
            path = output_dir / name
            path.write_bytes(data)
            chunks.append(
                {
                    "name": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                    "ordinal": chunk_index,
                }
            )
            chunk_index += 1
    return chunks, payload_hash.hexdigest()


def _verify_tar_members(
    chunk_paths: Sequence[Path],
    *,
    entries: Sequence[Mapping[str, Any]],
) -> int:
    expected = list(entries)
    count = 0
    with _tar_reader(chunk_paths) as archive:
        for member, descriptor in zip(archive, expected, strict=False):
            if member.name != descriptor["path"] or not member.isfile():
                raise ValueError("raw archive tar member identity mismatch")
            if member.size != descriptor["size_bytes"]:
                raise ValueError("raw archive tar member size mismatch")
            source = archive.extractfile(member)
            if source is None or _sha256_stream(source) != descriptor["sha256"]:
                raise ValueError("raw archive tar member SHA mismatch")
            count += 1
    if count != len(expected):
        raise ValueError("raw archive tar member count mismatch")
    return count


class _ChunkReader(io.RawIOBase):
    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = iter(paths)
        self._current: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        view = memoryview(buffer)
        while True:
            if self._current is None:
                try:
                    self._current = next(self._paths).open("rb")
                except StopIteration:
                    return 0
            count = self._current.readinto(view)
            if count:
                return int(count)
            self._current.close()
            self._current = None

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None
        super().close()


class _TarReader:
    def __init__(self, chunk_paths: Sequence[Path]) -> None:
        self.raw = _ChunkReader(chunk_paths)
        self.buffered = io.BufferedReader(self.raw)
        self.gzip = gzip.GzipFile(fileobj=self.buffered, mode="rb")
        self.archive = tarfile.open(fileobj=self.gzip, mode="r|")  # noqa: SIM115

    def __enter__(self) -> tarfile.TarFile:
        return self.archive

    def __exit__(self, *_: object) -> None:
        self.archive.close()
        self.gzip.close()
        self.buffered.close()


def _tar_reader(chunk_paths: Sequence[Path]) -> _TarReader:
    return _TarReader(chunk_paths)


def _sha256_stream(source: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _drain(source: BinaryIO) -> None:
    for _ in iter(lambda: source.read(1024 * 1024), b""):
        pass


def _assert_repository_file(path: Path, repository_root: Path) -> None:
    if not path.is_relative_to(repository_root) or not path.is_file():
        raise ValueError(f"required raw archive file unavailable: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = [
    "build_recovered_capture_archive",
    "inventory_recovered_capture_archive",
    "restore_recovered_capture_archive",
    "verify_recovered_capture_archive",
]
