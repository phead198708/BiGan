from __future__ import annotations

import json
import os

import pytest

from bigan.paper_trading.stack.preflight import PreflightError
from bigan.paper_trading.stack.report import ReportWriter, require_finite


async def test_zero_fill_can_pass_and_report_is_safe(observation):
    report = observation.report
    report.data["polls"].update(attempted=2, successful=2)
    report.finish()
    assert report.data["result"] == "PASS"
    writer = ReportWriter(observation.check.report_dir, output=observation.check.config.output_dir)
    try:
        writer.write(report)
    finally:
        writer.close()
    directory = observation.check.report_dir
    encoded = (directory / "soak_report.json").read_text()
    payload = json.loads(encoded)
    assert payload["schema_version"] == 1
    require_finite(payload)
    assert "# Paper stack soak — PASS" in (directory / "soak_summary.md").read_text()
    assert str(directory.parent) not in encoded
    assert "traceback" not in encoded and "payload" not in encoded
    assert {p.name for p in directory.iterdir()} == {"soak_report.json", "soak_summary.md"}
    with pytest.raises(PreflightError):
        ReportWriter(directory, output=observation.check.config.output_dir)


async def test_result_precedence(observation):
    report = observation.report
    report.issue("NO_MARKET", informational=True)
    assert report.data["result"] == "PASS"
    report.issue("TRANSIENT_RECONNECT")
    assert report.data["result"] == "WARN"
    report.issue("OPERATOR_FORCED_TERMINATION", hard=True)
    report.issue("TRANSIENT_RECONNECT")
    assert report.data["result"] == "FAIL"


async def test_atomic_write_failure_no_partial_artifact(observation, monkeypatch):
    writer = ReportWriter(observation.check.report_dir, output=observation.check.config.output_dir)
    monkeypatch.setattr(os, "replace", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("private path")))
    try:
        with pytest.raises(OSError):
            writer.write(observation.report)
        assert not (observation.check.report_dir / "soak_report.json").exists()
        assert not list(observation.check.report_dir.glob("*.tmp"))
    finally:
        writer.close()


async def test_no_report_writer_inside_output(observation):
    with pytest.raises(PreflightError):
        ReportWriter(observation.check.config.output_dir / "report", output=observation.check.config.output_dir)


async def test_bounded_diagnostics_and_no_raw_text(observation):
    for n in range(1000):
        observation.report.issue(f"ISSUE_{n}")
    assert len(observation.report.data["warnings"]) == 64
    with pytest.raises(ValueError):
        observation.report.issue("Traceback /private/secret")
