"""Per-round GitHub progress comments for v8 paper operator runs."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.paper.contracts import json_ready
from bigan.v8.paper.feed import ReadOnlyFeedEvent

PAPER_ROUND_PROGRESS_COMMENT_PHASE = "paper_round_progress_comment"
PAPER_ROUND_PROGRESS_COMMENT_SCHEMA_VERSION = (
    "bigan-v8-paper-round-progress-comment-v1"
)
DEFAULT_ROUND_PROGRESS_COMMENT_CREATED_AT = "2026-06-22T06:30:00Z"

GitHubRoundProgressPostMode = Literal["dry_run", "gh_command", "direct_comment"]

ROUND_PROGRESS_PAYLOAD_STREAM_FILENAME = "github_round_progress_payloads.jsonl"


class PaperRoundProgressCommentError(RuntimeError):
    """Raised when per-round comment evidence cannot be produced safely."""


@dataclass(frozen=True, slots=True)
class GitHubRoundProgressCommentConfig:
    """Configuration for deterministic per-round GitHub progress comments."""

    repo_full_name: str
    issue_number: int
    output_dir: Path | str
    run_id: str
    feed_mode: str
    provider_name: str
    provider_endpoint: str
    instrument_id: str
    post_mode: GitHubRoundProgressPostMode = "dry_run"
    created_at: str = DEFAULT_ROUND_PROGRESS_COMMENT_CREATED_AT
    comment_interval_rounds: int = 1
    max_comments: int = 0
    overwrite_existing: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False

    def __post_init__(self) -> None:
        if not self.repo_full_name.strip() or "/" not in self.repo_full_name:
            raise ValueError("repo_full_name must be owner/repo")
        if self.issue_number <= 0:
            raise ValueError("issue_number must be positive")
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.post_mode not in ("dry_run", "gh_command", "direct_comment"):
            raise ValueError("post_mode must be dry_run, gh_command, or direct_comment")
        if self.comment_interval_rounds <= 0:
            raise ValueError("comment_interval_rounds must be positive")
        if self.max_comments < 0:
            raise ValueError("max_comments must be non-negative")
        if not self.feed_mode.strip():
            raise ValueError("feed_mode is required")
        if not self.provider_name.strip():
            raise ValueError("provider_name is required")
        if not self.provider_endpoint.strip():
            raise ValueError("provider_endpoint is required")
        if not self.instrument_id.strip():
            raise ValueError("instrument_id is required")
        if not self.created_at:
            raise ValueError("created_at is required")
        if self.broker_exchange_write_enabled:
            raise PaperRoundProgressCommentError("broker/exchange writes are forbidden")
        if self.live_exchange_write_enabled:
            raise PaperRoundProgressCommentError("live exchange writes are forbidden")
        if self.paper_only is not True:
            raise PaperRoundProgressCommentError("progress comments must be paper-only")
        if self.capital_at_risk is not False:
            raise PaperRoundProgressCommentError(
                "progress comments cannot put capital at risk"
            )

    @property
    def output_path(self) -> Path:
        return self.output_dir.expanduser().resolve()

    @property
    def payload_stream_path(self) -> Path:
        return self.output_path / ROUND_PROGRESS_PAYLOAD_STREAM_FILENAME

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class GitHubRoundProgressCommentPayload:
    """One deterministic per-round GitHub comment payload."""

    schema_version: str
    phase: str
    run_id: str
    round_index: int
    issue_number: int
    repo_full_name: str
    feed_mode: str
    real_live_data: bool
    deterministic_replay: bool
    provider_name: str
    provider_endpoint_or_endpoint_type: str
    instrument_id: str
    event_ts: int
    received_ts: int
    source: str
    feed_sequence: int
    bid_price: float
    ask_price: float
    mid_price: float
    spread_bps: float
    volume: float
    trade_count: int
    read_only: bool
    write_capable: bool
    paper_only: bool
    capital_at_risk: bool
    broker_exchange_write_enabled: bool
    live_exchange_write_enabled: bool
    capital_deployment_allowed: bool
    live_deployment_allowed: bool
    broker_exchange_write_allowed: bool
    comment_title: str
    comment_body: str
    post_mode: GitHubRoundProgressPostMode
    gh_command: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GitHubRoundProgressCommentResult:
    """Generated artifacts for one per-round progress comment."""

    payload: GitHubRoundProgressCommentPayload
    output_dir: Path
    artifact_paths: dict[str, Path]
    delivery_receipt: dict[str, Any] | None = None


class GitHubRoundProgressCommentWriter:
    """Stateful writer that throttles and emits per-round comment evidence."""

    def __init__(self, config: GitHubRoundProgressCommentConfig) -> None:
        self.config = config
        self.comment_count = 0
        self.last_commented_round: int | None = None
        output_path = config.output_path
        if output_path.exists():
            if not config.overwrite_existing:
                raise FileExistsError(
                    f"round progress comment output_dir already exists: {output_path}; "
                    "set overwrite_existing=True to replace it"
                )
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)

    def maybe_comment_round(
        self,
        *,
        round_index: int,
        event: ReadOnlyFeedEvent,
    ) -> GitHubRoundProgressCommentResult | None:
        """Emit comment evidence for a round when interval and max gates allow it."""

        if round_index <= 0:
            raise ValueError("round_index must be positive")
        if round_index % self.config.comment_interval_rounds != 0:
            return None
        if self.config.max_comments and self.comment_count >= self.config.max_comments:
            return None
        result = deliver_github_round_progress_comment(
            event=event,
            round_index=round_index,
            config=self.config,
        )
        _append_jsonl(self.config.payload_stream_path, result.payload.to_dict())
        self.comment_count += 1
        self.last_commented_round = round_index
        return result

    def summary(self) -> dict[str, Any]:
        """Return deterministic manifest-ready progress-comment evidence."""

        output_path = self.config.output_path
        artifact_paths = _artifact_paths(output_path)
        return {
            "round_progress_comments_enabled": True,
            "round_progress_comment_dir": str(output_path),
            "round_progress_comment_post_mode": self.config.post_mode,
            "round_progress_comment_interval_rounds": (
                self.config.comment_interval_rounds
            ),
            "max_round_progress_comments": self.config.max_comments,
            "round_progress_comment_count": self.comment_count,
            "last_commented_round": self.last_commented_round,
            "round_progress_payload_stream_path": str(
                self.config.payload_stream_path
            )
            if self.config.payload_stream_path.exists()
            else None,
            "round_progress_payload_stream_sha256": _optional_sha256_file(
                self.config.payload_stream_path
            ),
            "round_progress_comment_artifact_hashes": {
                str(path.relative_to(output_path)): _sha256_file(path)
                for path in artifact_paths
            },
        }


def disabled_round_progress_comment_summary(
    *,
    output_dir: Path | str,
    post_mode: GitHubRoundProgressPostMode,
    comment_interval_rounds: int,
    max_comments: int,
) -> dict[str, Any]:
    """Return manifest-ready evidence when per-round comments are disabled."""

    return {
        "round_progress_comments_enabled": False,
        "round_progress_comment_dir": str(Path(output_dir).expanduser().resolve()),
        "round_progress_comment_post_mode": post_mode,
        "round_progress_comment_interval_rounds": comment_interval_rounds,
        "max_round_progress_comments": max_comments,
        "round_progress_comment_count": 0,
        "last_commented_round": None,
        "round_progress_payload_stream_path": None,
        "round_progress_payload_stream_sha256": None,
        "round_progress_comment_artifact_hashes": {},
    }


def build_github_round_progress_comment_payload(
    *,
    event: ReadOnlyFeedEvent,
    round_index: int,
    config: GitHubRoundProgressCommentConfig,
) -> GitHubRoundProgressCommentPayload:
    """Build a deterministic per-round progress comment payload."""

    if round_index <= 0:
        raise ValueError("round_index must be positive")
    read_only = event.read_only is True
    write_capable = event.write_capable is True
    paper_only = event.paper_only is True and config.paper_only is True
    capital_at_risk = event.capital_at_risk is True or config.capital_at_risk is True
    if not read_only or write_capable:
        raise PaperRoundProgressCommentError("round progress event must be read-only")
    if not paper_only or capital_at_risk:
        raise PaperRoundProgressCommentError(
            "round progress event must be paper-only with no capital at risk"
        )
    title = f"v8 Paper Round Progress: {config.run_id} round {round_index}"
    body = _comment_body(
        title=title,
        event=event,
        round_index=round_index,
        config=config,
    )
    command = None
    if config.post_mode in {"gh_command", "direct_comment"}:
        command = _gh_command(config, _comment_body_path(config, round_index))
    return GitHubRoundProgressCommentPayload(
        schema_version=PAPER_ROUND_PROGRESS_COMMENT_SCHEMA_VERSION,
        phase=PAPER_ROUND_PROGRESS_COMMENT_PHASE,
        run_id=config.run_id,
        round_index=round_index,
        issue_number=config.issue_number,
        repo_full_name=config.repo_full_name,
        feed_mode=config.feed_mode,
        real_live_data=config.feed_mode == "live-readonly",
        deterministic_replay=config.feed_mode == "deterministic-replay",
        provider_name=config.provider_name,
        provider_endpoint_or_endpoint_type=config.provider_endpoint,
        instrument_id=config.instrument_id,
        event_ts=event.event_ts,
        received_ts=event.received_ts,
        source=event.source,
        feed_sequence=event.feed_sequence,
        bid_price=event.bid_price,
        ask_price=event.ask_price,
        mid_price=event.mid_price,
        spread_bps=event.spread_bps,
        volume=event.volume,
        trade_count=event.trade_count,
        read_only=read_only,
        write_capable=write_capable,
        paper_only=paper_only,
        capital_at_risk=capital_at_risk,
        broker_exchange_write_enabled=config.broker_exchange_write_enabled,
        live_exchange_write_enabled=config.live_exchange_write_enabled,
        capital_deployment_allowed=False,
        live_deployment_allowed=False,
        broker_exchange_write_allowed=False,
        comment_title=title,
        comment_body=body,
        post_mode=config.post_mode,
        gh_command=command,
        created_at=config.created_at,
    )


def deliver_github_round_progress_comment(
    *,
    event: ReadOnlyFeedEvent,
    round_index: int,
    config: GitHubRoundProgressCommentConfig,
) -> GitHubRoundProgressCommentResult:
    """Write per-round comment artifacts and optionally post the comment."""

    output_path = config.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    payload = build_github_round_progress_comment_payload(
        event=event,
        round_index=round_index,
        config=config,
    )
    artifact_paths = {
        "payload": output_path
        / f"round_{round_index:06d}_github_progress_payload.json",
        "comment_body": output_path
        / f"round_{round_index:06d}_github_progress_comment.md",
    }
    _write_json(artifact_paths["payload"], payload.to_dict())
    _write_text(artifact_paths["comment_body"], payload.comment_body)

    if config.post_mode in {"gh_command", "direct_comment"}:
        artifact_paths["gh_command"] = output_path / (
            f"round_{round_index:06d}_github_progress_gh_command.sh"
        )
        _write_text(
            artifact_paths["gh_command"],
            (payload.gh_command or _gh_command(config, artifact_paths["comment_body"]))
            + "\n",
        )

    receipt = None
    if config.post_mode == "direct_comment":
        receipt = _post_direct_comment(
            config=config,
            comment_body_path=artifact_paths["comment_body"],
        )
        artifact_paths["delivery_receipt"] = output_path / (
            f"round_{round_index:06d}_github_progress_delivery_receipt.json"
        )
        _write_json(artifact_paths["delivery_receipt"], receipt)

    return GitHubRoundProgressCommentResult(
        payload=payload,
        output_dir=output_path,
        artifact_paths=artifact_paths,
        delivery_receipt=receipt,
    )


def _comment_body(
    *,
    title: str,
    event: ReadOnlyFeedEvent,
    round_index: int,
    config: GitHubRoundProgressCommentConfig,
) -> str:
    lines = [
        f"## {title}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| run_id | `{config.run_id}` |",
        f"| round_index | `{round_index}` |",
        f"| feed_mode | `{config.feed_mode}` |",
        f"| real_live_data | `{str(config.feed_mode == 'live-readonly').lower()}` |",
        f"| provider_name | `{config.provider_name}` |",
        f"| instrument_id | `{config.instrument_id}` |",
        f"| event_ts | `{event.event_ts}` |",
        f"| received_ts | `{event.received_ts}` |",
        f"| feed_sequence | `{event.feed_sequence}` |",
        f"| source | `{event.source}` |",
        f"| mid_price | `{event.mid_price}` |",
        f"| spread_bps | `{event.spread_bps}` |",
        "",
        "**Safety Flags**",
        "",
        f"- read_only: `{str(event.read_only).lower()}` expected `true`",
        f"- write_capable: `{str(event.write_capable).lower()}` expected `false`",
        f"- paper_only: `{str(event.paper_only).lower()}` expected `true`",
        f"- capital_at_risk: `{str(event.capital_at_risk).lower()}` expected `false`",
        "- capital_deployment_allowed: `false` expected `false`",
        "- live_deployment_allowed: `false` expected `false`",
        "- broker_exchange_write_allowed: `false` expected `false`",
        "",
        (
            "Progress-only heartbeat evidence. This is paper observation "
            "telemetry, not a promotion or profitability claim."
        ),
        "",
        f"_created_at: `{config.created_at}`_",
        "",
    ]
    return "\n".join(lines)


def _artifact_paths(output_path: Path) -> list[Path]:
    if not output_path.exists():
        return []
    return sorted(path for path in output_path.iterdir() if path.is_file())


def _comment_body_path(
    config: GitHubRoundProgressCommentConfig,
    round_index: int,
) -> Path:
    return (
        config.output_path / f"round_{round_index:06d}_github_progress_comment.md"
    )


def _gh_command(config: GitHubRoundProgressCommentConfig, body_file: Path) -> str:
    return " ".join(
        [
            "gh",
            "issue",
            "comment",
            str(config.issue_number),
            "--repo",
            shlex.quote(config.repo_full_name),
            "--body-file",
            shlex.quote(str(body_file)),
        ]
    )


def _post_direct_comment(
    *,
    config: GitHubRoundProgressCommentConfig,
    comment_body_path: Path,
) -> dict[str, Any]:
    command = [
        "gh",
        "issue",
        "comment",
        str(config.issue_number),
        "--repo",
        config.repo_full_name,
        "--body-file",
        str(comment_body_path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "post_mode": config.post_mode,
        "repo_full_name": config.repo_full_name,
        "issue_number": config.issue_number,
        "comment_url": completed.stdout.strip(),
        "created_at": config.created_at,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                json_ready(row),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _optional_sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return _sha256_file(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
