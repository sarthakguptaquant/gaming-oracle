"""Tests for the Cycle-7 tolerance map (the honest replacement for the false
'count tests are dead weight' claim).
"""

from __future__ import annotations

import numpy as np
import pytest

from gaming_oracle import gates as g
from gaming_oracle.data import load_data
from gaming_oracle.decomposition import representative_window
from gaming_oracle.tolerance_map import DEFAULT_TOL_GRID, tolerance_map

ALPHA = 0.01


def _window(asset):
    ds = load_data(asset, method="rolling_garch")
    rr, tv = ds.returns, ds.true_var
    ws, we = representative_window(rr, tv, panel=g.PANEL, alpha=ALPHA)
    return rr, tv, ws, we


def test_conformal_binds_at_small_tol_regulatory_at_large():
    """The core finding: conformal binds at the tight default-ish tolerances and a
    regulatory gate takes over once the tolerance widens past the regulatory band.
    """
    rr, tv, ws, we = _window("AAPL")
    tm = tolerance_map(rr, tv, asset="AAPL", window_start=ws, window_end=we, alpha=ALPHA)
    # conformal binds somewhere on the small-tol end
    assert any(r.conformal_binds for r in tm.rows)
    # and stops binding (regulatory takes over) at a finite crossover
    assert tm.crossover_tol is not None
    assert tm.regulatory_binding_gate in {"basel_traffic_light", "kupiec_pof",
                                          "christoffersen"}
    # the default tol=0.005 sits in the conformal-binding (artifact) zone
    row005 = next(r for r in tm.rows if abs(r.tol - 0.005) < 1e-9)
    assert row005.conformal_binds


def test_crossover_is_monotone_story():
    """Once conformal stops binding as tol increases, it does not start binding
    again (widening a tolerance only relaxes that gate)."""
    rr, tv, ws, we = _window("AAPL")
    tm = tolerance_map(rr, tv, asset="AAPL", window_start=ws, window_end=we, alpha=ALPHA)
    binds = [r.conformal_binds for r in tm.rows]
    # find first False after a True; everything after must remain False
    if tm.crossover_tol is not None:
        seen_false = False
        for b in binds:
            if not b:
                seen_false = True
            if seen_false:
                assert not b


def test_full_mul_monotone_in_tol():
    """A wider conformal tolerance is a looser constraint, so the full-panel MUL is
    non-decreasing in tol (up to solver noise) on definable rows."""
    rr, tv, ws, we = _window("BTC-USD")
    tm = tolerance_map(rr, tv, asset="BTC-USD", window_start=ws, window_end=we,
                       alpha=ALPHA)
    defn = [r.mul_full_bps for r in tm.rows if r.mul_full_bps > 1e-9]
    for a, b in zip(defn, defn[1:]):
        assert b >= a - 1e-6


def test_default_tol_is_in_artifact_zone_all_assets():
    """On every asset the hand-set default tol=0.005 is in the conformal-binding
    (artifact) zone, i.e. tighter than the regulatory band — the whole point."""
    for asset in ["AAPL", "SPY", "BTC-USD"]:
        rr, tv, ws, we = _window(asset)
        tm = tolerance_map(rr, tv, asset=asset, window_start=ws, window_end=we,
                           alpha=ALPHA)
        if tm.crossover_tol is not None:
            assert tm.crossover_tol > 0.005 - 1e-9
