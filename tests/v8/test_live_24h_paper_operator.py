"""Live read-only feed integration tests for the 24h paper operator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.paper import (
    DeterministicReplayFeed,
    FeedHealthSnapshot,
    LiveReadOnlyFeedConfig,
    PaperOperatorCLIError,
    PaperOperatorRunConfig,
    PublicTickerLiveReadOnlyFeed,
    ReadOnlyFeedEvent,
    build_live_feed_metadata,
    compute_feed_health,
    run_24h_paper_operator,
    synthetic_readonly_feed_events,
)


def test_live_readonly_operator_short_mocked_run_completes(
    tmp_path: Path,
) -> None:
    result = run_24h_paper_operator(
        config=_config(tmp_path, run_id="live-healthy"),
        feed=_mock_live_feed(row_count=6),
    )
    manifest = result.manifest
    comment_body = result.comment_result.artifact_paths["comment_body"].read_text(
        encoding="utf-8"
    )
    observability = _read_json(
        result.observability_dir / "paper_observability_report.json"
    )

    assert manifest["status"] == "completed_continue_paper"
    assert manifest["feed_mode"] == "live-readonly"
    assert manifest["real_live_data"] is True
    assert manifest["deterministic_replay"] is False
    assert manifest["provider_name"] == "mock_live_provider"
    assert manifest["instrument_id"] == "BTCUSDT"
    assert manifest["live_feed_metadata_sha256"] is not None
    assert manifest["live_feed_health_sha256"] is not None
    assert manifest["phase6_deployment_status"] == "approved_for_staged_live"
    assert manifest["paper_only"] is True
    assert manifest["capital_at_risk"] is False
    assert (result.paper_run_dir / "live_feed_metadata.json").exists()
    assert (result.paper_run_dir / "live_feed_health_report.json").exists()
    assert observability["feed_metrics"]["feed_mode"] == "live-readonly"
    assert observability["feed_metrics"]["provider_name"] == "mock_live_provider"
    assert "| feed_mode | `live-readonly` |" in comment_body
    assert "| provider_name | `mock_live_provider` |" in comment_body
    assert "| instrument_id | `BTCUSDT` |" in comment_body


def test_live_readonly_feed_gap_blocks_phase6_fail_closed(tmp_path: Path) -> None:
    events = list(synthetic_readonly_feed_events(row_count=5, source="mock_live"))
    events[2] = replace(
        events[2],
        event_ts=events[1].event_ts + 180_000,
        received_ts=events[1].event_ts + 180_250,
    )
    result = run_24h_paper_operator(
        config=_config(tmp_path, run_id="live-gap"),
        feed=_mock_live_feed(events=tuple(events)),
    )
    manifest = result.manifest

    assert manifest["status"] == "completed_blocked_fail_closed"
    assert manifest["feed_health_status"] == "failed"
    assert manifest["phase6_deployment_status"] == "blocked_fail_closed"
    assert manifest["operator_recommendation"] == "blocked_fail_closed"
    assert "feed_gap_breach" in manifest["reason_codes"]


def test_live_readonly_out_of_order_blocks_phase6_fail_closed(
    tmp_path: Path,
) -> None:
    events = list(synthetic_readonly_feed_events(row_count=5, source="mock_live"))
    events[2] = replace(
        events[2],
        event_ts=events[1].event_ts - 1_000,
        received_ts=events[1].received_ts + 250,
    )

    result = run_24h_paper_operator(
        config=_config(tmp_path, run_id="live-out-of-order"),
        feed=_mock_live_feed(events=tuple(events)),
    )

    assert result.manifest["status"] == "completed_blocked_fail_closed"
    assert "feed_out_of_order_breach" in result.manifest["reason_codes"]
    assert result.manifest["phase6_deployment_status"] == "blocked_fail_closed"


def test_live_readonly_stale_provider_health_blocks_fail_closed(
    tmp_path: Path,
) -> None:
    feed = _mock_live_feed(
        row_count=5,
        health_override={"stale_event_count": 2},
    )
    result = run_24h_paper_operator(
        config=_config(tmp_path, run_id="live-stale"),
        feed=feed,
    )
    live_health = _read_json(result.paper_run_dir / "live_feed_health_report.json")

    assert result.manifest["status"] == "completed_blocked_fail_closed"
    assert "stale_event_breach" in result.manifest["reason_codes"]
    assert live_health["stale_event_count"] == 2
    assert live_health["acceptance"]["stale_event_breach"] is True


def test_live_readonly_provider_disconnect_counts_are_recorded(
    tmp_path: Path,
) -> None:
    result = run_24h_paper_operator(
        config=_config(tmp_path, run_id="live-disconnect"),
        feed=_mock_live_feed(
            row_count=5,
            health_override={
                "provider_disconnect_count": 1,
                "provider_reconnect_count": 1,
            },
        ),
    )

    assert result.manifest["provider_disconnect_count"] == 1
    assert result.manifest["provider_reconnect_count"] == 1
    assert result.manifest["provider_error_count"] == 0
    assert result.manifest["status"] == "completed_continue_paper"


def test_live_readonly_write_capable_adapter_rejected_before_run(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, run_id="live-unsafe")

    with pytest.raises(PaperOperatorCLIError, match="write-capable live feed"):
        run_24h_paper_operator(config=config, feed=_UnsafeMockLiveFeed())

    manifest = _read_json(config.manifest_path)
    assert manifest["status"] == "failed_fail_closed"
    assert manifest["feed_mode"] == "live-readonly"
    assert manifest["capital_deployment_allowed"] is False


def test_live_readonly_mode_refuses_deterministic_replay_fallback(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, run_id="live-no-fallback")

    with pytest.raises(PaperOperatorCLIError, match="refusing deterministic replay"):
        run_24h_paper_operator(
            config=config,
            feed=DeterministicReplayFeed(
                events=synthetic_readonly_feed_events(row_count=3)
            ),
        )

    manifest = _read_json(config.manifest_path)
    assert manifest["status"] == "failed_fail_closed"
    assert manifest["reason_codes"] == ["paper_run_failed"]
    assert manifest["feed_mode"] == "live-readonly"
    assert manifest["real_live_data"] is True
    assert manifest["deterministic_replay"] is False


def test_live_readonly_invalid_provider_payload_has_specific_manifest_reason(
    tmp_path: Path,
) -> None:
    clock = _MutableClock(1_000.0)

    def missing_ask(_url: str, _timeout: float) -> dict[str, object]:
        return {"bidPrice": "100.0", "closeTime": int(clock() * 1000)}

    config = _config(tmp_path, run_id="live-invalid-payload")
    feed = PublicTickerLiveReadOnlyFeed(
        config=LiveReadOnlyFeedConfig(
            provider_name="mock_live_provider",
            provider_endpoint="mock://readonly",
            instrument_id="BTCUSDT",
            poll_interval_seconds=60.0,
            request_timeout_seconds=1.0,
            max_event_count=1,
        ),
        request_json=missing_ask,
        clock=clock,
        sleep=clock.advance,
    )

    with pytest.raises(PaperOperatorCLIError, match="missing_ask_price"):
        run_24h_paper_operator(config=config, feed=feed)

    manifest = _read_json(config.manifest_path)
    assert manifest["status"] == "failed_fail_closed"
    assert manifest["reason_codes"] == [
        "paper_run_failed",
        "invalid_provider_payload",
        "missing_ask_price",
    ]
    assert manifest["real_live_data"] is True
    assert manifest["deterministic_replay"] is False
    assert manifest["capital_deployment_allowed"] is False


def test_live_readonly_stop_path_writes_metadata(tmp_path: Path) -> None:
    result = run_24h_paper_operator(
        config=_config(tmp_path, run_id="live-stop", stop_after_events=3),
        feed=_mock_live_feed(row_count=6),
    )
    metadata = _read_json(result.paper_run_dir / "live_feed_metadata.json")

    assert result.manifest["status"] == "operator_stopped"
    assert result.manifest["stop_reason"] == "operator_stop"
    assert result.manifest["live_feed_metadata_sha256"] is not None
    assert metadata["feed_mode"] == "live-readonly"
    assert metadata["paper_only"] is True
    assert metadata["capital_at_risk"] is False


def test_live_readonly_operator_outputs_are_deterministic(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        run_id="live-deterministic",
        post_mode="gh_command",
        overwrite_existing=True,
    )
    first = run_24h_paper_operator(config=config, feed=_mock_live_feed(row_count=5))
    first_hashes = {
        "manifest": _sha256_file(first.manifest_path),
        "metadata": _sha256_file(first.paper_run_dir / "live_feed_metadata.json"),
        "comment": _sha256_file(first.comment_result.artifact_paths["comment_body"]),
    }
    second = run_24h_paper_operator(config=config, feed=_mock_live_feed(row_count=5))

    assert first_hashes == {
        "manifest": _sha256_file(second.manifest_path),
        "metadata": _sha256_file(second.paper_run_dir / "live_feed_metadata.json"),
        "comment": _sha256_file(second.comment_result.artifact_paths["comment_body"]),
    }


def _config(
    output_dir: Path,
    *,
    run_id: str,
    post_mode: str = "dry_run",
    overwrite_existing: bool = False,
    stop_after_events: int | None = None,
) -> PaperOperatorRunConfig:
    return PaperOperatorRunConfig(
        run_id=run_id,
        output_dir=output_dir,
        repo_full_name="phead198708/BiGan",
        issue_number=129,
        post_mode=post_mode,  # type: ignore[arg-type]
        duration_seconds=300,
        feed_event_interval_seconds=60,
        heartbeat_interval_seconds=30,
        summary_interval_seconds=120,
        feed_mode="live-readonly",
        provider_name="mock_live_provider",
        provider_endpoint="mock://readonly",
        instrument_id="BTCUSDT",
        overwrite_existing=overwrite_existing,
        stop_after_events=stop_after_events,
    )


def _mock_live_feed(
    *,
    row_count: int | None = None,
    events: tuple[ReadOnlyFeedEvent, ...] | None = None,
    health_override: dict[str, Any] | None = None,
) -> _MockLiveFeed:
    resolved_events = events or synthetic_readonly_feed_events(
        row_count=row_count or 5,
        source="mock_live_provider",
        instrument_id="BTCUSDT",
    )
    return _MockLiveFeed(
        events=tuple(resolved_events),
        health_override=health_override or {},
    )


class _MockLiveFeed:
    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    feed_mode = "live-readonly"

    def __init__(
        self,
        *,
        events: tuple[ReadOnlyFeedEvent, ...],
        health_override: dict[str, Any],
    ) -> None:
        self._events = events
        self._health_override = health_override
        self._closed = False

    def iter_events(self):
        if self._closed:
            return
        yield from self._events

    def health_snapshot(self) -> FeedHealthSnapshot:
        health = compute_feed_health(
            self._events,
            max_allowed_gap_ms=120_000,
            max_event_lag_ms=10_000,
        )
        if not self._health_override:
            return health
        return replace(health, **self._health_override)

    def metadata_snapshot(self, *, ended_at: str):
        config = LiveReadOnlyFeedConfig(
            provider_name="mock_live_provider",
            provider_endpoint="mock://readonly",
            instrument_id="BTCUSDT",
            poll_interval_seconds=60.0,
            request_timeout_seconds=1.0,
            expected_wall_clock_duration_seconds=300,
        )
        return build_live_feed_metadata(
            config=config,
            started_at_wall_clock="2026-06-22T03:00:00Z",
            ended_at_wall_clock=ended_at,
            wall_clock_duration_seconds=300,
        )

    def close(self) -> None:
        self._closed = True


class _UnsafeMockLiveFeed(_MockLiveFeed):
    write_capable = True

    def __init__(self) -> None:
        super().__init__(
            events=synthetic_readonly_feed_events(row_count=3),
            health_override={},
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _MutableClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds
