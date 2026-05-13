"""Tests for the locked v1 feature dictionary."""

from __future__ import annotations

from pathlib import Path

import pytest

from bigan.features.registry import (
    FEATURE_SET_ID,
    FEATURE_VERSION,
    FEATURE_VERSION_STATUS,
    FEATURES,
    feature_names,
    features_by_group,
    get_feature,
)


def test_feature_version_is_locked() -> None:
    assert FEATURE_SET_ID == "bigan-mvp-v1"
    assert FEATURE_VERSION == "bigan-mvp-v1.0.0"
    assert FEATURE_VERSION_STATUS == "locked"


def test_feature_names_are_unique_and_stable_count() -> None:
    names = feature_names()
    assert len(names) == len(set(names))
    assert len(names) == 56
    assert names == tuple(spec.name for spec in FEATURES)


def test_each_required_group_is_present() -> None:
    grouped = features_by_group()
    assert {group: len(specs) for group, specs in grouped.items()} == {
        "order_book": 17,
        "trade_flow": 15,
        "price_return": 12,
        "volatility_regime": 12,
    }


@pytest.mark.parametrize("spec", FEATURES)
def test_every_feature_has_formula_window_unit_and_source(spec) -> None:  # type: ignore[no-untyped-def]
    assert spec.name
    assert spec.formula
    assert spec.window
    assert spec.unit
    assert spec.source_tables
    assert spec.dtype == "float64"
    assert spec.null_policy


def test_get_feature_returns_spec_or_raises() -> None:
    assert get_feature("ob_mid_price").formula == "(ob_bid_price + ob_ask_price) / 2."
    with pytest.raises(KeyError):
        get_feature("not_a_feature")


def test_feature_dictionary_document_references_locked_version() -> None:
    doc = Path("docs/features/feature_dictionary_v1.md").read_text(encoding="utf-8")
    assert f"Feature set id: `{FEATURE_SET_ID}`" in doc
    assert f"Feature version: `{FEATURE_VERSION}`" in doc
    for spec in FEATURES:
        assert f"`{spec.name}`" in doc
