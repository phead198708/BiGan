from __future__ import annotations

import json
import os

import pytest

from bigan.paper_trading.stack.preflight import PreflightError
from bigan.paper_trading.stack.report import (
    COMPLETE_FILE,
    INCOMPLETE_FILE,
    ReportWriter,
    load_completed_report,
    require_finite,
)


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
    assert {p.name for p in directory.iterdir()} == {"soak_report.json", "soak_summary.md", COMPLETE_FILE}
    assert load_completed_report(directory)["result"] == "PASS"
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
    assert (observation.check.report_dir / INCOMPLETE_FILE).exists()


@pytest.mark.parametrize("failure_at", [1, 2, 3, 4])
async def test_each_publication_failure_cannot_leave_a_valid_pass(observation, monkeypatch, failure_at):
    report = observation.report
    report.data["polls"].update(attempted=1, successful=1)
    report.finish()
    directory = observation.check.report_dir
    writer = ReportWriter(directory, output=observation.check.config.output_dir)
    original = os.replace
    count = 0

    def fail_one(*args, **kwargs):
        nonlocal count
        count += 1
        if count == failure_at:
            raise OSError("private-path disk failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_one)
    with pytest.raises(OSError):
        writer.write(report)
    writer.close()
    assert report.failed
    assert (directory / INCOMPLETE_FILE).exists()
    assert not (directory / "soak_report.json").exists()
    assert not (directory / COMPLETE_FILE).exists()
    with pytest.raises(ValueError, match="REPORT_NOT_COMPLETE"):
        load_completed_report(directory)


async def test_json_candidate_is_invalid_before_completion(observation, monkeypatch):
    directory = observation.check.report_dir
    writer = ReportWriter(directory, output=observation.check.config.output_dir)
    original = writer._replace
    seen = []

    def inspect(source, destination):
        original(source, destination)
        if destination in {"soak_report.json", "soak_summary.md", INCOMPLETE_FILE}:
            with pytest.raises(ValueError, match="REPORT_NOT_COMPLETE"):
                load_completed_report(directory)
            seen.append(destination)

    monkeypatch.setattr(writer, "_replace", inspect)
    try:
        writer.write(observation.report)
    finally:
        writer.close()
    assert len(seen) == 3
    assert load_completed_report(directory)["result"] == "PASS"


@pytest.mark.parametrize("damage", ["json", "markdown", "marker", "incomplete"])
async def test_consumers_reject_changed_or_incomplete_artifacts(observation, damage):
    directory = observation.check.report_dir
    writer = ReportWriter(directory, output=observation.check.config.output_dir)
    writer.write(observation.report)
    writer.close()
    if damage == "incomplete":
        (directory / INCOMPLETE_FILE).touch()
    elif damage == "marker":
        (directory / COMPLETE_FILE).unlink()
    else:
        name = "soak_report.json" if damage == "json" else "soak_summary.md"
        with (directory / name).open("a") as stream:
            stream.write(" ")
    with pytest.raises(ValueError, match="REPORT_NOT_COMPLETE"):
        load_completed_report(directory)


async def test_no_report_writer_inside_output(observation):
    with pytest.raises(PreflightError):
        ReportWriter(observation.check.config.output_dir / "report", output=observation.check.config.output_dir)


async def test_bounded_diagnostics_and_no_raw_text(observation):
    for n in range(1000):
        observation.report.issue(f"ISSUE_{n}")
    assert len(observation.report.data["warnings"]) == 64
    with pytest.raises(ValueError):
        observation.report.issue("Traceback /private/secret")
