"""Embargo / ingestion-lag auditor known-answer tests."""

from __future__ import annotations

from datetime import date

from gaming_oracle.leakage.embargo import (
    FeatureTimeline,
    FeatureUse,
    audit_embargo,
)


def test_clean_timeline_has_no_violations():
    tl = FeatureTimeline()
    # value available the day before the decision, no restatement
    tl.add(FeatureUse("pe_ratio", date(2020, 6, 10), available_at=date(2020, 6, 9),
                      value_asof=15.0, value_current=15.0))
    assert audit_embargo(tl) == []


def test_ingestion_lag_flagged():
    tl = FeatureTimeline()
    # earnings dated the decision day but only ingestible two days later
    tl.add(FeatureUse("eps", date(2020, 6, 10), available_at=date(2020, 6, 12)))
    v = audit_embargo(tl)
    assert len(v) == 1
    assert v[0].kind == "ingestion_lag"
    assert v[0].lag_days == 2


def test_restatement_flagged():
    tl = FeatureTimeline()
    tl.add(FeatureUse("revenue", date(2020, 6, 10), available_at=date(2020, 6, 1),
                      value_asof=100.0, value_current=112.0))
    v = audit_embargo(tl)
    assert len(v) == 1
    assert v[0].kind == "restatement"
    assert v[0].detail["delta"] == 12.0


def test_embargo_zone_flagged_when_inside_purge_window():
    tl = FeatureTimeline()
    # value knowable by decision date, but inside a 3-day embargo purge zone
    tl.add(FeatureUse("factor", date(2020, 6, 10), available_at=date(2020, 6, 9)))
    v = audit_embargo(tl, embargo_days=3)
    assert len(v) == 1
    assert v[0].kind == "embargo_zone"


def test_restatement_tolerance_suppresses_tiny_revisions():
    tl = FeatureTimeline()
    tl.add(FeatureUse("x", date(2020, 6, 10), available_at=date(2020, 6, 1),
                      value_asof=100.0, value_current=100.0005))
    assert audit_embargo(tl, restatement_tol=1e-3) == []


def test_both_lag_and_restatement_on_one_use():
    tl = FeatureTimeline()
    tl.add(FeatureUse("y", date(2020, 6, 10), available_at=date(2020, 6, 15),
                      value_asof=1.0, value_current=2.0))
    kinds = sorted(v.kind for v in audit_embargo(tl))
    assert kinds == ["ingestion_lag", "restatement"]
