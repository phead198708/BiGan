"""Async settlement tests for round-scoped Polymarket corpus collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import bigan.v8.polymarket.recorder.async_settlement as async_settlement_module
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.recorder import (
    PolymarketRealCorpusRecorderConfig,
    capture_polymarket_pending_round,
    finalize_polymarket_pending_round,
)
from bigan.v8.polymarket.recorder.btc_reference import mock_btc_feature_candle_rows
from bigan.v8.polymarket.recorder.chainlink_rtds import (
    CHAINLINK_RTDS_CORPUS_FILENAME,
    CHAINLINK_RTDS_CORPUS_MANIFEST_FILENAME,
    CHAINLINK_RTDS_RAW_FILENAME,
)
from bigan.v8.polymarket.recorder.market_discovery import discover_mock_market_rows
from bigan.v8.polymarket.recorder.orderbook_state import (
    mock_orderbook_rows,
    mock_trade_rows,
)


def test_pending_capture_does_not_call_resolution_or_export(tmp_path: Path) -> None:
    provider = AsyncSettlementFakeProvider(resolved=False)
    config = PolymarketRealCorpusRecorderConfig(
        run_id="pending-round",
        output_dir=tmp_path,
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    result = capture_polymarket_pending_round(config, public_provider=provider)

    assert provider.resolution_calls == 0
    assert result.report["capture_status"] == "pending_resolution"
    assert result.report["pending_resolution"] is True
    assert result.report["resolution_provider_called"] is False
    assert result.report["raw_polymarket_market_count"] == 1
    assert result.report["raw_orderbook_row_count"] > 0
    assert result.report["provider_raw_orderbook_snapshot_count"] >= result.report[
        "training_sampled_orderbook_row_count"
    ]
    assert result.report["provider_raw_artifacts_preserved"] is True
    assert result.report["raw_resolution_count"] == 0
    assert result.report["training_eligible"] is False
    assert result.report["exported_training_corpus_dir"] is None
    assert (result.raw_dir / "raw_polymarket_resolutions.jsonl").read_text() == ""
    provider_raw_books = result.artifact_paths["provider_raw_polymarket_orderbooks"]
    assert provider_raw_books.exists()
    assert len(_read_jsonl(provider_raw_books)) == result.report[
        "provider_raw_orderbook_snapshot_count"
    ]
    assert result.manifest["provider_raw_artifact_hashes"][
        "raw_polymarket_orderbooks.jsonl"
    ]
    assert result.manifest["training_raw_is_validated_sampled_view"] is True


def test_pending_round_persists_causal_chainlink_evidence_through_export(
    tmp_path: Path,
) -> None:
    provider = AsyncSettlementFakeProvider(resolved=True)
    config = PolymarketRealCorpusRecorderConfig(
        run_id="pending-chainlink-round",
        output_dir=tmp_path,
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )
    market = provider.market_rows(config)[0]
    market_start_ts = int(market["market_start_ts"])
    market_end_ts = int(market["market_end_ts"])
    collector = AsyncSettlementFakeChainlinkCollector(
        rows=[
            _chainlink_row(market_start_ts - 1_000, 65_000.0),
            _chainlink_row(market_start_ts + 60_000, 65_025.0),
            _chainlink_row(market_end_ts + 1_000, 99_999.0),
        ]
    )

    capture = capture_polymarket_pending_round(
        config,
        public_provider=provider,
        chainlink_rtds_collector=collector,
    )

    assert capture.report["raw_chainlink_price_row_count"] == 2
    assert capture.report["chainlink_timestamp_causality_violation_count"] == 0
    assert capture.manifest["chainlink_raw_artifact_row_count"] == 2
    assert capture.manifest["chainlink_raw_artifact_sha256"]
    raw_rows = _read_jsonl(capture.raw_dir / CHAINLINK_RTDS_RAW_FILENAME)
    assert [row["price"] for row in raw_rows] == [65_000.0, 65_025.0]

    finalized = finalize_polymarket_pending_round(
        capture.run_dir,
        public_provider=provider,
        destination_root=tmp_path / "training_root",
        overwrite_existing=True,
    )

    assert finalized.report["finalization_status"] == "exported"
    assert finalized.report["raw_chainlink_price_row_count"] == 2
    assert finalized.report["chainlink_corpus_evidence"]["attached"] is True
    rounds_index = _read_jsonl(finalized.artifact_paths["rounds_index"])
    training_raw_dir = finalized.run_dir / rounds_index[0]["training_raw_dir"]
    assert len(_read_jsonl(training_raw_dir / CHAINLINK_RTDS_RAW_FILENAME)) == 2
    training_manifest = _read_json(training_raw_dir / "round_training_manifest.json")
    assert training_manifest["raw_chainlink_price_row_count"] == 2
    assert training_manifest["supplemental_decision_time_artifact_hashes"][
        CHAINLINK_RTDS_RAW_FILENAME
    ]
    assert finalized.exported_training_corpus_dir is not None
    assert len(
        _read_jsonl(
            finalized.exported_training_corpus_dir / CHAINLINK_RTDS_CORPUS_FILENAME
        )
    ) == 2
    evidence_manifest = _read_json(
        finalized.exported_training_corpus_dir
        / CHAINLINK_RTDS_CORPUS_MANIFEST_FILENAME
    )
    assert evidence_manifest["decision_time_only"] is True
    assert evidence_manifest["timestamp_causality_violation_count"] == 0


def test_pending_round_rejects_noncausal_chainlink_rows_without_blocking_core_capture(
    tmp_path: Path,
) -> None:
    provider = AsyncSettlementFakeProvider(resolved=False)
    config = PolymarketRealCorpusRecorderConfig(
        run_id="pending-invalid-chainlink",
        output_dir=tmp_path,
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )
    market = provider.market_rows(config)[0]
    source_ts = int(market["market_start_ts"])
    invalid = _chainlink_row(source_ts, 65_000.0)
    invalid["available_at_ts"] = source_ts - 1

    capture = capture_polymarket_pending_round(
        config,
        public_provider=provider,
        chainlink_rtds_collector=AsyncSettlementFakeChainlinkCollector(
            rows=[invalid]
        ),
    )

    assert capture.report["capture_status"] == "pending_resolution"
    assert capture.report["raw_chainlink_price_row_count"] == 0
    assert "chainlink_rtds_timestamp_causality_violation" in capture.report[
        "chainlink_capture_reason_codes"
    ]


def test_pending_finalization_completes_payout_only_resolution_from_chainlink(
    tmp_path: Path,
) -> None:
    provider = AsyncSettlementPayoutOnlyProvider(resolved=True)
    config = PolymarketRealCorpusRecorderConfig(
        run_id="pending-payout-only-resolution",
        output_dir=tmp_path,
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )
    market = provider.market_rows(config)[0]
    start_ts = int(market["market_start_ts"])
    end_ts = int(market["market_end_ts"])
    capture = capture_polymarket_pending_round(
        config,
        public_provider=provider,
        chainlink_rtds_collector=AsyncSettlementFakeChainlinkCollector(
            rows=[
                _chainlink_row(start_ts - 1_000, 65_000.0),
                _chainlink_row(end_ts - 1_000, 65_025.0),
            ]
        ),
    )

    finalized = finalize_polymarket_pending_round(
        capture.run_dir,
        public_provider=provider,
        destination_root=tmp_path / "training_root",
        overwrite_existing=True,
    )

    assert finalized.report["finalization_status"] == "exported"
    resolution = _read_jsonl(
        finalized.raw_dir / "raw_polymarket_resolutions.jsonl"
    )[0]
    assert resolution["reference_price_start"] == 65_000.0
    assert resolution["reference_price_end"] == 65_025.0
    assert resolution["reference_price_pair_completed_from_chainlink_rtds"] is True
    assert resolution["resolved_outcome"] == "UP"


def test_pending_capture_explains_orderbook_rejection_without_raw_book_dump(
    tmp_path: Path,
) -> None:
    provider = AsyncSettlementMissingBookProvider(resolved=False)
    config = PolymarketRealCorpusRecorderConfig(
        run_id="pending-missing-book",
        output_dir=tmp_path,
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    result = capture_polymarket_pending_round(config, public_provider=provider)

    assert result.report["capture_status"] == "blocked_fail_closed"
    assert result.report["reject_reason_counts"] == {"missing_complete_up_down_orderbook": 1}
    rejected = _read_jsonl(result.artifact_paths["pending_round_rejected_rows"])
    detail = rejected[0]["reason_details"]["orderbook_completeness"]
    assert detail["raw_book_rows_persisted"] is False
    assert detail["valid_book_rows_by_outcome"]["UP"] > 0
    assert detail["valid_book_rows_by_outcome"]["DOWN"] == 0
    assert detail["explanation"] == "No valid DOWN orderbook rows were available for this market."
    assert "raw_rows" not in detail
    assert len(
        _read_jsonl(result.artifact_paths["provider_raw_polymarket_orderbooks"])
    ) > 0


def test_pending_finalization_waits_for_resolution_before_export(tmp_path: Path) -> None:
    provider = AsyncSettlementFakeProvider(resolved=False)
    config = PolymarketRealCorpusRecorderConfig(
        run_id="pending-round",
        output_dir=tmp_path,
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )
    capture = capture_polymarket_pending_round(config, public_provider=provider)

    unresolved = finalize_polymarket_pending_round(
        capture.run_dir,
        public_provider=provider,
        destination_root=tmp_path / "training_root",
    )

    assert provider.resolution_calls == 1
    assert unresolved.report["finalization_status"] == "pending_resolution"
    assert unresolved.report["pending_resolution"] is True
    assert unresolved.report["training_eligible"] is False
    assert unresolved.exported_training_corpus_dir is None
    assert unresolved.report["reject_reason_counts"] == {"missing_resolution": 1}

    provider.resolved = True
    finalized = finalize_polymarket_pending_round(
        capture.run_dir,
        public_provider=provider,
        destination_root=tmp_path / "training_root",
        overwrite_existing=True,
    )

    assert provider.resolution_calls == 2
    assert finalized.report["finalization_status"] == "exported"
    assert finalized.report["pending_resolution"] is False
    assert finalized.report["training_eligible"] is True
    assert finalized.report["raw_resolution_count"] == 1
    assert finalized.report["provider_raw_resolution_row_count"] == 1
    assert finalized.report["provider_raw_artifacts_preserved"] is True
    assert len(
        _read_jsonl(
            finalized.artifact_paths["provider_raw_polymarket_resolutions"]
        )
    ) == 1
    assert finalized.exported_training_corpus_dir is not None
    assert finalized.exported_training_corpus_dir.name.startswith("btc-updown-5m-")
    assert (finalized.exported_training_corpus_dir / "polymarket_label_rows.jsonl").exists()
    provenance = finalized.exported_training_corpus_dir / "training_corpus_provenance.json"
    assert '"round_scoped_export": true' in provenance.read_text(encoding="utf-8")
    assert finalized.report["round_artifact_export_mode"] == "round_finalization_lifecycle"
    assert finalized.report["round_artifacts_written"] == 1
    assert finalized.report["training_raw_round_count"] == 1
    assert finalized.report["paper_audit_round_count"] == 1
    rounds_index = _read_jsonl(finalized.artifact_paths["rounds_index"])
    assert len(rounds_index) == 1
    round_dir = finalized.run_dir / rounds_index[0]["round_dir"]
    training_raw_dir = finalized.run_dir / rounds_index[0]["training_raw_dir"]
    assert (round_dir / "round_summary.json").exists()
    assert (round_dir / "paper_audit" / "paper_audit_manifest.json").exists()
    assert (training_raw_dir / "round_training_manifest.json").exists()
    assert _read_json(round_dir / "round_summary.json")["training_eligibility_policy"] == (
        "min_one_complete_book_sample"
    )
    assert _read_json(training_raw_dir / "round_training_manifest.json")[
        "training_eligibility_policy"
    ] == "min_one_complete_book_sample"

    provider.resolved = False
    finalized_from_existing_raw = finalize_polymarket_pending_round(
        capture.run_dir,
        public_provider=provider,
        destination_root=tmp_path / "training_root",
        overwrite_existing=True,
    )

    assert provider.resolution_calls == 3
    assert finalized_from_existing_raw.report["finalization_status"] == "exported"
    assert finalized_from_existing_raw.report["reject_reason_counts"] == {}
    assert finalized_from_existing_raw.report["raw_resolution_count"] == 1
    assert finalized_from_existing_raw.exported_training_corpus_dir is not None
    assert finalized_from_existing_raw.report["round_artifacts_newly_finalized"] == 0


def test_pending_finalization_preserves_unknown_50_50_resolution(
    tmp_path: Path,
) -> None:
    provider = AsyncSettlementFakeProvider(
        resolved=True,
        resolution_status="unknown_50_50",
    )
    config = PolymarketRealCorpusRecorderConfig(
        run_id="pending-unknown-50-50",
        output_dir=tmp_path,
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )
    capture = capture_polymarket_pending_round(config, public_provider=provider)

    finalized = finalize_polymarket_pending_round(
        capture.run_dir,
        public_provider=provider,
        destination_root=tmp_path / "training_root",
        overwrite_existing=True,
    )

    assert finalized.report["finalization_status"] == "exported"
    assert finalized.report["training_eligible"] is True
    rounds_index = _read_jsonl(finalized.artifact_paths["rounds_index"])
    assert len(rounds_index) == 1
    round_dir = finalized.run_dir / rounds_index[0]["round_dir"]
    training_raw_dir = finalized.run_dir / rounds_index[0]["training_raw_dir"]
    paper_audit_dir = finalized.run_dir / rounds_index[0]["paper_audit_dir"]

    round_summary = _read_json(round_dir / "round_summary.json")
    training_manifest = _read_json(training_raw_dir / "round_training_manifest.json")
    training_resolution = _read_jsonl(
        training_raw_dir / "raw_polymarket_resolutions.jsonl"
    )[0]
    paper_settlement = _read_jsonl(
        paper_audit_dir / "polymarket_settlement_events.jsonl"
    )[0]
    paper_audit_manifest = _read_json(paper_audit_dir / "paper_audit_manifest.json")

    for payload in (round_summary, training_resolution, paper_settlement):
        assert payload["resolution_status"] == "unknown_50_50"
        assert payload["resolved_outcome"] == "UNKNOWN_50_50"
        assert payload["payout_up"] == 0.5
        assert payload["payout_down"] == 0.5
        assert payload["reference_price_start"] == 65000.0
    for payload in (round_summary, training_manifest, paper_audit_manifest):
        assert payload["round_finalization_only"] is True
        assert payload["model_signal_used"] is False
        assert payload["paper_decision_used"] is False
        assert payload["paper_audit_only"] is True


def test_pending_finalization_round_artifacts_survive_crash_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AsyncSettlementTwoRoundProvider(resolved=True)
    config = PolymarketRealCorpusRecorderConfig(
        run_id="pending-two-rounds",
        output_dir=tmp_path,
        market_families=("btc_updown_5m", "btc_updown_15m"),
        mock_public_data=False,
    )
    capture = capture_polymarket_pending_round(config, public_provider=provider)
    real_writer = async_settlement_module.write_polymarket_round_lifecycle_indexes
    flush_count = 0

    def crash_after_first_flush(**kwargs: Any) -> None:
        nonlocal flush_count
        flush_count += 1
        real_writer(**kwargs)
        if flush_count == 1:
            raise RuntimeError("simulated crash after first round artifact flush")

    monkeypatch.setattr(
        async_settlement_module,
        "write_polymarket_round_lifecycle_indexes",
        crash_after_first_flush,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        finalize_polymarket_pending_round(
            capture.run_dir,
            public_provider=provider,
            destination_root=tmp_path / "training_root",
        )

    rounds_index_path = capture.run_dir / "rounds_index.jsonl"
    persisted_rows = _read_jsonl(rounds_index_path)
    assert len(persisted_rows) == 1
    assert (capture.run_dir / persisted_rows[0]["round_dir"] / "round_summary.json").exists()
    assert (
        capture.run_dir
        / persisted_rows[0]["training_raw_dir"]
        / "round_training_manifest.json"
    ).exists()

    monkeypatch.setattr(
        async_settlement_module,
        "write_polymarket_round_lifecycle_indexes",
        real_writer,
    )
    provider.resolved = False
    resumed = finalize_polymarket_pending_round(
        capture.run_dir,
        public_provider=provider,
        destination_root=tmp_path / "training_root",
        overwrite_existing=True,
    )

    assert resumed.report["finalization_status"] == "blocked_fail_closed"
    assert resumed.report["phase2_corpus_built"] is True
    assert "exactly one round slug" in resumed.report["phase2_error"]
    assert resumed.report["round_artifacts_written"] == 2
    assert resumed.report["round_artifacts_newly_finalized"] == 1
    assert len(_read_jsonl(rounds_index_path)) == 2
    assert len(_read_jsonl(capture.run_dir / "training_raw_index.jsonl")) == 2


class AsyncSettlementFakeProvider:
    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def __init__(
        self,
        *,
        resolved: bool,
        resolution_status: str = "normal",
    ) -> None:
        self.resolved = resolved
        self.resolution_status = resolution_status
        self.resolution_calls = 0

    def market_rows(
        self,
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return [_as_real_public_market_row(row) for row in discover_mock_market_rows(config)[:1]]

    def orderbook_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return mock_orderbook_rows(markets, config)

    def trade_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del config
        return mock_trade_rows(markets)

    def btc_feature_candle_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return mock_btc_feature_candle_rows(markets, config)

    def resolution_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del config
        self.resolution_calls += 1
        if not self.resolved:
            return []
        return [
            {
                "market_id": market["market_id"],
                "reference_price_start": 65000.0,
                "reference_price_end": 65025.0,
                "reference_price_source": market["reference_price_source"],
                "resolution_status": self.resolution_status,
                "raw_resolution_text": "Resolved from official test reference.",
                "paper_only": True,
                "capital_at_risk": False,
                "broker_exchange_write_enabled": False,
                "live_exchange_write_enabled": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
            for market in markets
        ]


class AsyncSettlementMissingBookProvider(AsyncSettlementFakeProvider):
    def orderbook_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return [
            row
            for row in mock_orderbook_rows(markets, config)
            if row.get("outcome") == "UP"
        ]


class AsyncSettlementPayoutOnlyProvider(AsyncSettlementFakeProvider):
    def resolution_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del config
        self.resolution_calls += 1
        if not self.resolved:
            return []
        return [
            {
                "market_id": market["market_id"],
                "reference_price_source": market["reference_price_source"],
                "resolution_status": "normal",
                "resolved_outcome": "UP",
                "payout_up": 1.0,
                "payout_down": 0.0,
                "raw_resolution_text": "Official payout is available before references.",
                "paper_only": True,
                "capital_at_risk": False,
                "broker_exchange_write_enabled": False,
                "live_exchange_write_enabled": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
            for market in markets
        ]


class AsyncSettlementTwoRoundProvider(AsyncSettlementFakeProvider):
    def market_rows(
        self,
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return [_as_real_public_market_row(row) for row in discover_mock_market_rows(config)[:2]]


class AsyncSettlementFakeChainlinkCollector:
    def __init__(self, *, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]

    def collection_report(self) -> dict[str, Any]:
        return {
            "report_type": "polymarket_chainlink_rtds_collection",
            "source_type": "polymarket_rtds_chainlink",
            "raw_price_row_count": len(self._rows),
            "read_only": True,
            "paper_only": True,
            "capital_at_risk": False,
        }


def _as_real_public_market_row(row: dict[str, Any]) -> dict[str, Any]:
    market = dict(row)
    market["raw_market_sha256"] = canonical_json_sha256(
        {
            "market_id": market["market_id"],
            "family": market["market_family"],
            "source": "async_settlement_fake_provider",
        }
    )
    market["raw_public_payload"] = {"mock_public_data": False}
    return market


def _chainlink_row(source_ts: int, price: float) -> dict[str, Any]:
    return {
        "source_type": "polymarket_rtds_chainlink",
        "source_ts": source_ts,
        "available_at_ts": source_ts,
        "price": price,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
