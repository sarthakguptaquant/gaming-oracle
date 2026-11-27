"""Known-answer self-tests for each gate.

A well-calibrated series PASSES; an overconfident (understated-VaR) series FAILS.
We construct synthetic returns with a known generating process so the answers
are deterministic given the seed.
"""

import numpy as np
import pytest

from gaming_oracle import gates as g

ALPHA = 0.01
LEVEL = 0.95
N = 1000


def _calibrated_series(n=N, alpha=ALPHA, seed=0):
    """Returns ~ N(0, sigma^2); honest VaR = z_alpha * sigma (covers at nominal)."""
    rng = np.random.default_rng(seed)
    sigma = 0.01
    returns = rng.normal(0, sigma, n)
    z = -1.0 * __import__("scipy.stats", fromlist=["norm"]).norm.ppf(alpha)
    var = np.full(n, z * sigma)  # positive loss magnitude
    return var, returns


def _overconfident_series(n=N, alpha=ALPHA, seed=0, shrink=0.55):
    """Same returns, but the stated VaR is badly understated -> too many breaches."""
    var, returns = _calibrated_series(n, alpha, seed)
    return var * (1 - shrink), returns


# --- Kupiec ---------------------------------------------------------------

def test_kupiec_calibrated_passes():
    var, returns = _calibrated_series()
    res = g.kupiec_pof(var, returns, alpha=ALPHA, level=LEVEL)
    assert res.passed, res.detail


def test_kupiec_overconfident_fails():
    var, returns = _overconfident_series()
    res = g.kupiec_pof(var, returns, alpha=ALPHA, level=LEVEL)
    assert not res.passed, res.detail


# --- Christoffersen -------------------------------------------------------

def test_christoffersen_calibrated_passes():
    var, returns = _calibrated_series()
    res = g.christoffersen(var, returns, alpha=ALPHA, level=LEVEL)
    assert res.passed, res.detail


def test_christoffersen_overconfident_fails():
    # Heavy understatement blows conditional coverage (too many breaches).
    var, returns = _overconfident_series(shrink=0.6)
    res = g.christoffersen(var, returns, alpha=ALPHA, level=LEVEL)
    assert not res.passed, res.detail


def test_christoffersen_clustering_fails_independence():
    """Inject clustered exceptions -> independence test should reject."""
    n = 500
    var = np.full(n, 0.05)
    returns = np.zeros(n)
    # cluster of consecutive big losses early -> 11 transitions dominate
    for t in range(10, 25):
        returns[t] = -0.10  # loss 0.10 > var 0.05 => exception, consecutive
    res = g.christoffersen(var, returns, alpha=ALPHA, level=LEVEL)
    assert res.detail["lr_ind"] > 0
    assert not res.detail["pass_ind"], res.detail


# --- Basel traffic-light --------------------------------------------------

def test_basel_green_passes():
    # exactly 4 exceptions in last 250 days -> green
    n = 250
    var = np.full(n, 0.05)
    returns = np.zeros(n)
    for t in [10, 50, 120, 200]:
        returns[t] = -0.10
    res = g.basel_traffic_light(var, returns)
    assert res.detail["zone"] == "green"
    assert res.passed


def test_basel_amber_red_fails():
    n = 250
    var = np.full(n, 0.05)
    returns = np.zeros(n)
    for t in range(0, 60, 5):  # 12 exceptions -> red
        returns[t] = -0.10
    res = g.basel_traffic_light(var, returns)
    assert res.detail["zone"] == "red"
    assert not res.passed


# --- Conformal coverage ---------------------------------------------------

def test_conformal_calibrated_passes():
    var, returns = _calibrated_series()
    res = g.conformal_coverage(var, returns, alpha=ALPHA, level=LEVEL, tol=0.01)
    assert res.passed, res.detail


def test_conformal_overconfident_fails():
    var, returns = _overconfident_series(shrink=0.6)
    res = g.conformal_coverage(var, returns, alpha=ALPHA, level=LEVEL, tol=0.005)
    assert not res.passed, res.detail


# --- Du-Escanciano (killer-demo gate) -------------------------------------

def test_du_escanciano_calibrated_passes():
    var, returns = _calibrated_series()
    res = g.du_escanciano(var, returns, alpha=ALPHA, level=LEVEL)
    assert res.passed, res.detail


def _series_with_exceptions(x, n=N):
    """Deterministic series with exactly ``x`` evenly-spread exceptions."""
    var = np.full(n, 0.05)
    returns = np.zeros(n)
    idx = np.linspace(0, n - 1, x, dtype=int) if x > 0 else []
    for t in idx:
        returns[t] = -0.10
    return var, returns


def test_du_escanciano_higher_power_than_kupiec():
    """Du-Escanciano is ONE-SIDED against under-coverage, so it rejects an
    over-breaching (understated-VaR) series at a strictly lower exception count
    than the TWO-SIDED Kupiec test. Deterministic at N=1000, alpha=0.01: the
    one-sided test's upper rejection threshold sits below Kupiec's.
    """
    def upper_threshold(fn):
        # expected count is alpha*N = 10; find smallest x>=11 that rejects
        for x in range(11, 60):
            var, returns = _series_with_exceptions(x)
            if not fn(var, returns, alpha=ALPHA, level=LEVEL).passed:
                return x
        return None

    de_thr = upper_threshold(g.du_escanciano)
    k_thr = upper_threshold(g.kupiec_pof)
    assert de_thr is not None and k_thr is not None
    # one-sided test detects under-coverage with fewer exceptions
    assert de_thr < k_thr, (de_thr, k_thr)
    # and there is a concrete regime where Kupiec passes but DE rejects
    var, returns = _series_with_exceptions(de_thr)
    assert g.kupiec_pof(var, returns, alpha=ALPHA, level=LEVEL).passed
    assert not g.du_escanciano(var, returns, alpha=ALPHA, level=LEVEL).passed


# --- panel ----------------------------------------------------------------

def test_run_panel_calibrated_all_pass():
    var, returns = _calibrated_series()
    out = g.run_panel(var, returns, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    assert out["passed_all"], {k: v.detail for k, v in out["gates"].items()}


def test_run_panel_overconfident_not_all_pass():
    var, returns = _overconfident_series(shrink=0.6)
    out = g.run_panel(var, returns, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    assert not out["passed_all"]
