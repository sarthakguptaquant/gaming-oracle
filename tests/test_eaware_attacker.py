"""Tests for the Cycle-7 e-process-aware attacker (the honest defense residual).

These assert the things the OLD Cycle-4 'the attack is voided' claim hid:
  - a budget-bounded attacker keeps the e-process GREEN while still hiding risk,
  - stating ES INDEPENDENTLY of VaR neutralizes the e-process (the 'cannot dodge
    ES' claim was true only by construction),
  - the residual is NOT a hardcoded-cap artifact (max per-period e_t << 1e6).
"""

from __future__ import annotations

import numpy as np
import pytest

from gaming_oracle import gates as g
from gaming_oracle.data import load_data
from gaming_oracle.eaware_attacker import max_residual_under_panel, solve_eaware
from gaming_oracle.ebacktest import gaussian_es_var_ratio

ALPHA = 0.01


def _win(asset, w=250):
    ds = load_data(asset, method="rolling_garch")
    return ds.returns[-w:], ds.true_var[-w:]


def test_eaware_keeps_eprocess_green_but_hides_risk():
    """A budget-bounded e-aware attacker stays below the e-process threshold yet
    hides a strictly positive amount of risk."""
    rr, tv = _win("AAPL")
    res = solve_eaware(tv, rr, budget=0.10, es_independent=True, alpha=ALPHA)
    assert res.eprocess_passed
    assert res.eprocess_max < res.eprocess_threshold
    assert res.mul_bps > 0.0


def test_independent_es_neutralizes_eprocess_more_than_dependent():
    """Stating ES independently (over-stating ES while under-stating VaR) drives the
    e-process LOWER than tying ES to VaR by the Gaussian multiple. That is the
    'cannot dodge ES is true only by construction' result."""
    rr, tv = _win("AAPL")
    indep = solve_eaware(tv, rr, budget=0.20, es_independent=True, alpha=ALPHA)
    dep = solve_eaware(tv, rr, budget=0.20, es_independent=False, alpha=ALPHA)
    # the independent-ES attacker pushes the e-process strictly below the
    # tied-ES attacker (it can widen the e-statistic denominator)
    assert indep.eprocess_max <= dep.eprocess_max + 1e-9
    # and meaningfully so: the dependent one is materially higher on this window
    assert dep.eprocess_max > indep.eprocess_max


def test_independent_es_over_states_es_while_under_stating_var():
    """The independent-ES move is: under-reserve VaR (the lie) but OVER-state ES to
    dodge the ES e-backtest. Confirm both directions hold."""
    rr, tv = _win("AAPL")
    res = solve_eaware(tv, rr, budget=0.20, es_independent=True, alpha=ALPHA)
    ratio = gaussian_es_var_ratio(ALPHA)
    honest_es = ratio * tv
    # VaR genuinely under-reserved
    assert np.all(res.stated_var <= tv + 1e-12)
    assert res.stated_var.mean() < tv.mean()
    # ES over-stated relative to the honest ES (the dodge)
    assert np.all(res.stated_es >= honest_es - 1e-9)
    assert res.stated_es.mean() > honest_es.mean()


def test_residual_is_not_a_cap_artifact():
    """The disclosed 1e6 collapsed-gap cap must not drive the result: the e-aware
    attacker never collapses the ES-VaR gap, so its max per-period e_t is tiny vs
    the cap."""
    for asset in ["AAPL", "SPY", "BTC-USD"]:
        rr, tv = _win(asset)
        res = solve_eaware(tv, rr, budget=0.20, es_independent=True, alpha=ALPHA)
        assert res.max_per_period_e < 1e6 / 100.0  # orders of magnitude below the cap


def test_full_panel_residual_is_positive_and_green():
    """The realistic defense residual: the most aggressive e-aware lie that keeps
    EVERY gate (regulatory + conformal + e-backtest) green still hides > 0 bps."""
    rr, tv = _win("BTC-USD")
    res = max_residual_under_panel(tv, rr, es_independent=True, alpha=ALPHA)
    assert res.panel_passed_all
    assert res.mul_bps > 0.0


def test_eaware_var_never_exceeds_honest():
    """Genuine under-reservation on VaR (stated <= true) everywhere, the same
    discipline the rest of the repo enforces."""
    rr, tv = _win("SPY")
    res = solve_eaware(tv, rr, budget=0.15, es_independent=True, alpha=ALPHA)
    assert np.all(res.stated_var <= tv + 1e-12)
    assert np.all(res.stated_var >= 0.0)
