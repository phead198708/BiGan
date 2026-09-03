"""Schema and hard paper-only boundary tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from bigan.paper_trading.contracts import PaperDecisionEvent, PaperRunManifest
from tests.paper_trading.helpers import manifest, paper_decision


def test_contracts_are_json_native_and_round_trip_exactly() -> None:
    decision = paper_decision(1)
    payload = decision.to_dict()

    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert PaperDecisionEvent.from_dict(payload) == decision
    assert PaperRunManifest.from_dict(manifest().to_dict()) == manifest()
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["broker_exchange_write_enabled"] is False
    assert payload["live_exchange_write_enabled"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("paper_only", False),
        ("capital_at_risk", True),
        ("broker_exchange_write_enabled", True),
        ("live_exchange_write_enabled", True),
        ("polymarket_write_enabled", True),
        ("wallet_signing_enabled", True),
    ],
)
def test_safety_boundary_cannot_be_relaxed(field: str, unsafe: bool) -> None:
    with pytest.raises(ValueError, match="safety boundary"):
        replace(manifest(), **{field: unsafe})


def test_non_finite_contract_number_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        replace(paper_decision(1).decision, cash_after=float("nan"))


def test_schema_field_set_is_strict() -> None:
    payload = paper_decision(1).to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        PaperDecisionEvent.from_dict(payload)
