"""Runbook coverage tests for issue #46."""

from __future__ import annotations

from pathlib import Path


def test_model_rollback_runbook_contains_required_operational_sections() -> None:
    text = Path("docs/runbooks/model_rollback.md").read_text(encoding="utf-8")

    required_sections = [
        "## Trigger Conditions",
        "## Manual Rollback Steps",
        "## Automatic Rollback Preconditions",
        "## Post-Rollback Verification",
        "## Notification And Audit",
        "## Database Schema Compatibility",
        "## Drill Procedure",
    ]
    for section in required_sections:
        assert section in text

    required_phrases = [
        "Health check fails",
        "p95 serving latency",
        "Prediction error rate",
        "Prediction distribution drift",
        "Data missing rate",
        "rollback_deployment",
        "current_online_models",
        "feature_schema.json",
        "model_registry",
        "model_deployments",
    ]
    for phrase in required_phrases:
        assert phrase in text
