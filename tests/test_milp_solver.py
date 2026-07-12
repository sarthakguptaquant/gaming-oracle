"""Tests for the non-uniform MILP solver (Cycle 3).

Validation contract (spec): the produced stated_var must (1) pass every gate
recomputed from scratch, (2) be <= true_var everywhere (genuine under-reserving),
(3) the lie must be positive. Plus MILP-specific properties:
  * the MILP lie >= the uniform-shrink lie (per-point >= scalar shrink),
  * the count gates are enforced exactly (optimum is at a gate's count bound),
  * the count-feasibility intervals match the gate pass-conditions,
  * empty / no-gates ceiling logic is consistent.
"""

import numpy as np
import pytest

from gaming_oracle import gates as g
from gaming_oracle.report import price
from gaming_oracle.solver import solve
from gaming_oracle.milp_solver import (
    solve_milp,
    _kupiec_count_interval,
    _conformal_count_interval,
    _du_escanciano_count_hi,
    _stated_from_exceptions,
)

ALPHA = 0.01
LEVEL = 0.95


def _definable_window(n=250, sigma=0.01, n_exc=3, seed=11):
    """A 250-day calibrated window the honest reference PASSES (definable lie).

    Constant-vol normal returns with a constant honest VaR at z*sigma, then we
    plant exactly ``n_exc`` isolated exceptions (losses just past the honest VaR)
    on spread-out days. ``n_exc`` near alpha*n keeps Kupiec/conformal green, the
    spacing keeps Christoffersen/Basel green; so the honest reference passes and a
    lie is definable. This avoids the seed lottery of raw normals where coverage
    is too tight (0 exceptions) or Basel breaks (clustered tails).
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, sigma, n)
    from scipy.stats import norm
    z = -norm.ppf(ALPHA)
    true_var = np.full(n, z * sigma)
    # damp any natural exceptions so we control the exact exception set
    nat = (-returns > true_var)
    returns[nat] = -0.5 * true_var[nat]
    # plant n_exc isolated exceptions, evenly spaced, just past the honest VaR
    if n_exc > 0:
        idx = np.linspace(0, n - 1, n_exc + 2).astype(int)[1:-1]
        returns[idx] = -(true_var[idx] + 0.0005)
    return true_var, returns


# backward-compatible alias (definable synthetic window)
def _calibrated_normal(n=250, sigma=0.01, seed=11):
    return _definable_window(n=n, sigma=sigma, seed=seed)


# ---------------------------------------------------------------------------
# validation contract
# ---------------------------------------------------------------------------

def test_milp_validation_contract():
    tv, rr = _calibrated_normal()
    res = solve_milp(tv, rr, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    # 1. recompute every gate from scratch -> all pass
    out = g.run_panel(res.stated_var, rr, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    assert out["passed_all"], {k: v.detail for k, v in out["gates"].items()}
    # 2. stated_var <= true_var everywhere
    assert np.all(res.stated_var <= tv + 1e-12)
    # 3. stated_var >= 0 everywhere (a VaR magnitude)
    assert np.all(res.stated_var >= -1e-12)
    # 4. the lie is strictly positive
    assert res.mean_understatement > 0


def test_milp_pricing_reconciles():
    tv, rr = _calibrated_normal(seed=3)
    res = solve_milp(tv, rr, panel=g.PANEL, alpha=ALPHA)
    p = price(res.stated_var, tv, rr, notional=1e9)
    delta = float(np.mean(tv - res.stated_var))
    assert abs(p.mul_bps - delta * 1e4) < 1e-6
    assert abs(p.mul_usd - delta * 1e9) < 1e-3


# ---------------------------------------------------------------------------
# the headline property: non-uniform >= uniform
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_milp_lie_at_least_uniform_lie(seed):
    """Per-point optimization dominates the scalar shrink: MILP MUL >= uniform MUL.

    The uniform shrink is a feasible point in the MILP's space (a particular
    exception vector), so the optimum can never be smaller. On the full panel the
    MILP may use the labeled repair; even then it should not fall below uniform on
    these calibrated windows. We assert with a tiny tolerance for repair noise.
    """
    tv, rr = _calibrated_normal(seed=seed)
    ru = solve(tv, rr, panel=g.PANEL, alpha=ALPHA)
    rm = solve_milp(tv, rr, panel=g.PANEL, alpha=ALPHA)
    pu = price(ru.stated_var, tv, rr)
    pm = price(rm.stated_var, tv, rr)
    assert pm.mul_bps >= pu.mul_bps - 1e-6


def test_milp_optimal_flag_means_full_panel_feasible():
    """When optimal=True the solution must pass the full panel (true maximum)."""
    tv, rr = _calibrated_normal(seed=7)
    rm = solve_milp(tv, rr, panel=g.PANEL, alpha=ALPHA)
    if rm.optimal:
        out = g.run_panel(rm.stated_var, rr, panel=g.PANEL, alpha=ALPHA)
        assert out["passed_all"]
        assert rm.method == "milp"


# ---------------------------------------------------------------------------
# count-feasibility intervals match the gate pass conditions
# ---------------------------------------------------------------------------

def test_kupiec_interval_matches_gate():
    n = 250
    lo, hi = _kupiec_count_interval(n, ALPHA, LEVEL)
    # every count inside [lo,hi] passes Kupiec; lo-1 and hi+1 fail
    sigma = 0.01
    from scipy.stats import norm
    z = -norm.ppf(ALPHA)
    tv = np.full(n, z * sigma)
    for x in (lo, hi):
        # build a stated series with exactly x exceptions on the worst days
        returns = np.full(n, 0.0)
        returns[:x] = -(tv[0] + 0.001)   # x guaranteed exceptions
        stated = tv.copy()
        gr = g.kupiec_pof(stated, returns, alpha=ALPHA, level=LEVEL)
        assert gr.passed, f"count {x} should pass Kupiec"
    if hi + 1 <= n:
        returns = np.full(n, 0.0)
        returns[: hi + 1] = -(tv[0] + 0.001)
        gr = g.kupiec_pof(tv, returns, alpha=ALPHA, level=LEVEL)
        assert not gr.passed, f"count {hi+1} should fail Kupiec"


def test_conformal_interval_matches_gate():
    n = 250
    lo, hi = _conformal_count_interval(n, ALPHA, tol=0.005)
    from scipy.stats import norm
    z = -norm.ppf(ALPHA)
    tv = np.full(n, z * 0.01)
    for x in range(lo, hi + 1):
        returns = np.full(n, 0.0)
        returns[:x] = -(tv[0] + 0.001)
        gr = g.conformal_coverage(tv, returns, alpha=ALPHA)
        assert gr.passed, f"count {x} should pass conformal"
    # hi+1 fails
    x = hi + 1
    returns = np.full(n, 0.0)
    returns[:x] = -(tv[0] + 0.001)
    gr = g.conformal_coverage(tv, returns, alpha=ALPHA)
    assert not gr.passed


def test_du_escanciano_hi_matches_gate():
    n = 250
    hi = _du_escanciano_count_hi(n, ALPHA, LEVEL)
    from scipy.stats import norm
    z = -norm.ppf(ALPHA)
    tv = np.full(n, z * 0.01)
    returns = np.full(n, 0.0)
    returns[:hi] = -(tv[0] + 0.001)
    assert g.du_escanciano(tv, returns, alpha=ALPHA).passed
    returns2 = np.full(n, 0.0)
    returns2[: hi + 1] = -(tv[0] + 0.001)
    assert not g.du_escanciano(tv, returns2, alpha=ALPHA).passed


# ---------------------------------------------------------------------------
# stated-from-exceptions reconstruction
# ---------------------------------------------------------------------------

def test_stated_from_exceptions_consistency():
    """Reconstructed stated VaR must induce exactly the chosen exception vector."""
    tv, rr = _calibrated_normal(seed=2)
    loss = -rr
    # a feasible exception vector: only days with loss>0 can be exceptions
    e = (loss > np.median(loss[loss > 0])).astype(int)
    e[loss <= 0] = 0
    stated = _stated_from_exceptions(e, loss, tv)
    realized = g.exceptions(stated, rr)
    assert np.array_equal(realized, e)
    assert np.all(stated <= tv + 1e-12)


def test_no_gates_drives_stated_to_zero():
    """With no count gates and no forced exceptions logic, the unconstrained lie
    pushes every eligible day to an exception (stated -> 0)."""
    tv, rr = _calibrated_normal(seed=8)
    res = solve_milp(tv, rr, panel=[], alpha=ALPHA)
    # empty panel: no constraints, every loss>0 day becomes an exception (V=0)
    loss = -rr
    expected_exc = int((loss > 0).sum())
    assert res.exception_count == expected_exc
