"""Probe-construction known-answer tests."""

from __future__ import annotations

from datetime import date

import pytest

from gaming_oracle.leakage.probes import (
    Probe,
    ProbeKind,
    make_fake_date_pairs,
    make_entity_mask_pairs,
    _shift_years,
)


def test_fake_date_shifts_only_the_stated_date(base_probes):
    pairs = make_fake_date_pairs(base_probes, shift_years=3)
    assert len(pairs) == len(base_probes)
    for pr, orig in zip(pairs, base_probes):
        assert pr.kind is ProbeKind.FAKE_DATE
        # context and forward return unchanged; only the date moves
        assert pr.counterfactual.context == orig.context
        assert pr.counterfactual.forward_return == orig.forward_return
        assert pr.counterfactual.asof.year == orig.asof.year - 3
        assert pr.factual.asof == orig.asof


def test_fake_date_rejects_nonpositive_shift(base_probes):
    with pytest.raises(ValueError):
        make_fake_date_pairs(base_probes, shift_years=0)


def test_shift_years_handles_leap_day():
    assert _shift_years(date(2020, 2, 29), 1) == date(2019, 2, 28)


def test_entity_mask_removes_name_and_ticker(base_probes):
    pairs = make_entity_mask_pairs(base_probes, entity="Apple", ticker="AAPL")
    assert len(pairs) == len(base_probes)
    for pr in pairs:
        assert pr.kind is ProbeKind.ENTITY_MASK
        assert "Apple" not in pr.counterfactual.context
        assert "AAPL" not in pr.counterfactual.context
        assert "Company X" in pr.counterfactual.context
        # factual arm keeps the real name
        assert "Apple" in pr.factual.context


def test_entity_mask_skips_probes_without_the_name():
    probes = [
        Probe("q0", date(2020, 1, 1), "Neutral note. mom=0.1", 0.01),
    ]
    pairs = make_entity_mask_pairs(probes, entity="Apple", ticker="AAPL")
    # nothing to mask -> no pair (a no-op pair would bias the score to 0)
    assert pairs == []


def test_entity_mask_rejects_empty_entity(base_probes):
    with pytest.raises(ValueError):
        make_entity_mask_pairs(base_probes, entity="")
