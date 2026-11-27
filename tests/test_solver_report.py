"""Solver + report tests, including the validation contract from the spec."""

import numpy as np
import pytest

from gaming_oracle import gates as g
from gaming_oracle.report import price
from gaming_oracle.solver import solve

ALPHA = 0.01
LEVEL = 0.95


def _synthetic():
    rng = np.random.default_rng(7)
    n = 1000
    sigma = 0.01
    returns = rng.normal(0, sigma, n)
    from scipy.stats import norm
    z = -norm.ppf(ALPHA)
    true_var = np.full(n, z * sigma)
    return true_var, returns


def test_solver_produces_feasible_lie():
    true_var, returns = _synthetic()
    res = solve(true_var, returns, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    # validation contract:
    # 1. recompute every gate from scratch -> all pass
    out = g.run_panel(res.stated_var, returns, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    assert out["passed_all"], {k: v.detail for k, v in out["gates"].items()}
    # 2. stated_var <= true_var everywhere (genuinely under-reserving)
    assert np.all(res.stated_var <= true_var + 1e-12)
    # 3. the lie is strictly positive (we exhibited something)
    assert res.mean_understatement > 0


def test_pricing_reconciles():
    true_var, returns = _synthetic()
    res = solve(true_var, returns, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    p = price(res.stated_var, res.true_var, returns, notional=1e9)
    # re-derive bps independently
    delta = float(np.mean(true_var - res.stated_var))
    assert abs(p.mul_bps - delta * 1e4) < 1e-6
    assert abs(p.mul_usd - delta * 1e9) < 1e-3
    # extra exceptions per year non-negative (stated breaches >= true breaches)
    assert p.extra_exceptions_per_year >= -1e-9


def test_adding_du_escanciano_does_not_increase_lie():
    """The 5-gate panel is a superset of constraints, so its exhibited lie must
    be <= the 4-gate lie (killer demo: detection only shrinks the lie)."""
    true_var, returns = _synthetic()
    r4 = solve(true_var, returns, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    r5 = solve(true_var, returns, panel=g.PANEL_WITH_DE, alpha=ALPHA, level=LEVEL)
    assert r5.mean_understatement <= r4.mean_understatement + 1e-9
