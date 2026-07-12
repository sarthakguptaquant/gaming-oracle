"""Tests for the Cycle-7 causal rolling-out-of-sample GARCH reference (look-ahead
fix) and the regulatory-panel / gate-rename relabelings.
"""

from __future__ import annotations

import numpy as np

from gaming_oracle import gates as g
from gaming_oracle.data import (load_data, reference_var, rolling_garch_vol)


def test_rolling_garch_is_causal():
    """The rolling reference at index t must not depend on returns AFTER t.

    Perturbing a future return must leave the rolling sigma at earlier indices
    unchanged; the full-sample garch (look-ahead) would change them.
    """
    rng = np.random.default_rng(0)
    r = rng.standard_normal(900) * 0.01
    sig = rolling_garch_vol(r, min_train=500, refit_every=25)
    r2 = r.copy()
    r2[800] += 0.5  # a huge shock far in the future
    sig2 = rolling_garch_vol(r2, min_train=500, refit_every=25)
    # everything strictly before the perturbation index is untouched
    assert np.allclose(sig[:800], sig2[:800])


def test_rolling_garch_matches_ewma_in_warmup():
    """During the warm-up (before min_train) the rolling reference is the causal
    EWMA, so there is no look-ahead in the seed either."""
    rng = np.random.default_rng(1)
    r = rng.standard_normal(700) * 0.01
    from gaming_oracle.data import ewma_vol
    sig = rolling_garch_vol(r, min_train=500, refit_every=25)
    ew = ewma_vol(r)
    assert np.allclose(sig[:500], ew[:500])


def test_reference_var_rolling_garch_runs_and_is_positive():
    r = np.random.default_rng(2).standard_normal(800) * 0.01
    v = reference_var(r, alpha=0.01, method="rolling_garch")
    assert v.shape == r.shape
    assert np.all(v >= 0.0)


def test_load_data_rolling_garch():
    ds = load_data("AAPL", method="rolling_garch")
    assert ds.method == "rolling_garch"
    assert ds.true_var.shape == ds.returns.shape
    assert np.all(ds.true_var >= 0.0)


def test_regulatory_panel_excludes_conformal():
    names = {gate.__name__ for gate in g.PANEL_REGULATORY}
    assert names == {"kupiec_pof", "christoffersen", "basel_traffic_light"}
    assert "conformal_coverage" not in names
    # the augmented PANEL is the regulatory three plus conformal
    assert g.conformal_coverage in g.PANEL
    assert all(x in g.PANEL for x in g.PANEL_REGULATORY)


def test_gate_rename_alias_preserved():
    """The gate is now oneside_uc_score; du_escanciano is a kept alias and the
    detail label reflects the honest name."""
    assert g.oneside_uc_score is g.du_escanciano
    r = np.random.default_rng(3).standard_normal(250) * 0.01
    out = g.oneside_uc_score(np.abs(r) + 0.02, r)
    assert out.detail["test"] == "oneside_uc_score"
