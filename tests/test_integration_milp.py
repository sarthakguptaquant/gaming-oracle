"""Real-data integration test for the non-uniform MILP solver (Cycle 3).

On the representative window of each cached asset, verifies the validation
contract for the MILP solution and the headline property that the non-uniform
(MILP) lie is at least as large as the uniform-shrink lie. Skips per-asset if the
npz cache is absent (fresh checkout, no network).
"""

import os

import numpy as np
import pytest

from gaming_oracle import gates as g
from gaming_oracle.data import CACHE_DIR, load_data
from gaming_oracle.decomposition import representative_window
from gaming_oracle.solver import solve
from gaming_oracle.milp_solver import solve_milp
from gaming_oracle.milp_decomposition import leave_one_out_milp
from gaming_oracle.report import price

ALPHA = 0.01
ASSETS = ["AAPL", "SPY", "BTC-USD"]


def _available(asset):
    return os.path.exists(os.path.join(CACHE_DIR, f"{asset}_returns.npz"))


@pytest.mark.parametrize("asset", ASSETS)
def test_milp_validation_contract_real(asset):
    if not _available(asset):
        pytest.skip(f"{asset} cache missing")
    ds = load_data(asset, method="garch", use_cache=True)
    ws, we = representative_window(ds.returns, ds.true_var, window=250, step=25,
                                   panel=g.PANEL, alpha=ALPHA)
    rr, tv = ds.returns[ws:we], ds.true_var[ws:we]
    res = solve_milp(tv, rr, panel=g.PANEL, alpha=ALPHA)
    # validation contract: re-verify every gate from scratch
    out = g.run_panel(res.stated_var, rr, panel=g.PANEL, alpha=ALPHA)
    assert out["passed_all"], {k: v.detail for k, v in out["gates"].items()}
    # genuinely under-reserving and a real magnitude
    assert np.all(res.stated_var <= tv + 1e-12)
    assert np.all(res.stated_var >= -1e-12)
    assert res.mean_understatement > 0
    # if labeled optimal, it must be a true full-panel maximum (no repair)
    if res.optimal:
        assert res.method == "milp"


@pytest.mark.parametrize("asset", ASSETS)
def test_non_uniform_hides_at_least_as_much_as_uniform(asset):
    if not _available(asset):
        pytest.skip(f"{asset} cache missing")
    ds = load_data(asset, method="garch", use_cache=True)
    ws, we = representative_window(ds.returns, ds.true_var, window=250, step=25,
                                   panel=g.PANEL, alpha=ALPHA)
    rr, tv = ds.returns[ws:we], ds.true_var[ws:we]
    ru = solve(tv, rr, panel=g.PANEL, alpha=ALPHA)
    rm = solve_milp(tv, rr, panel=g.PANEL, alpha=ALPHA)
    pu = price(ru.stated_var, tv, rr)
    pm = price(rm.stated_var, tv, rr)
    # per-point optimum >= scalar shrink (the uniform lie is one feasible point)
    assert pm.mul_bps >= pu.mul_bps - 1e-6
    # both genuinely under-reserve
    assert np.all(rm.stated_var <= tv + 1e-12)


@pytest.mark.parametrize("asset", ASSETS)
def test_milp_decomposition_real_partition(asset):
    if not _available(asset):
        pytest.skip(f"{asset} cache missing")
    ds = load_data(asset, method="garch", use_cache=True)
    ws, we = representative_window(ds.returns, ds.true_var, window=250, step=25,
                                   panel=g.PANEL, alpha=ALPHA)
    decomp = leave_one_out_milp(ds.returns, ds.true_var, asset=asset,
                                window_start=ws, window_end=we, panel=g.PANEL,
                                alpha=ALPHA)
    binders = set(decomp.binding_gates())
    dead = set(decomp.dead_weight_gates())
    # binding + dead-weight partition the panel
    assert binders.isdisjoint(dead)
    assert binders | dead == set(decomp.panel_names)
    # conformal_coverage binds on every asset's representative window (the
    # Cycle-2 finding: it is the dominant gate; this asserts it never drops out)
    assert "conformal_coverage" in binders
    # contributions non-negative and ranked descending; table renders
    bps = [c.contribution_bps for c in decomp.contributions]
    assert bps == sorted(bps, reverse=True)
    assert all(b >= -1e-9 for b in bps)
    assert isinstance(decomp.ranking_table(), str)


def _brute_force_max(panel, rr, tv, top_k=14):
    """Brute-force the true max-understatement lie over exception sets, by
    searching the highest-gain eligible days. Used to cross-check that the SPY
    Christoffersen binding is real against the global optimum, not a repair
    artifact. Bounded to top_k candidates so it stays fast on this fixed window.
    """
    import itertools
    from gaming_oracle.milp_solver import _stated_from_exceptions, _gate_count_bounds
    n = len(rr)
    loss = -rr
    gain = np.maximum(loss, 0.0)
    x_lo, x_hi, _ = _gate_count_bounds(panel, n, ALPHA, 0.95, 0.005)
    eligible = np.where(loss > 0)[0]
    order = eligible[np.argsort(-gain[eligible])]
    cand = list(order[:top_k])
    best = -1.0
    for x in range(x_lo, x_hi + 1):
        for combo in itertools.combinations(cand, x):
            e = np.zeros(n, int)
            for t in combo:
                e[t] = 1
            st = _stated_from_exceptions(e, loss, tv)
            if g.run_panel(st, rr, panel=panel, alpha=ALPHA)["passed_all"]:
                m = float(np.mean(tv - st)) * 1e4
                if m > best:
                    best = m
    return best


def test_spy_christoffersen_genuinely_binds_vs_brute_force():
    """On SPY's representative window the non-uniform optimum clusters its two
    worst-loss days (adjacent), so Christoffersen BINDS: the best full-panel lie
    is strictly smaller than the best lie without Christoffersen. Cross-checked
    against a brute-force global optimum so the binding is not a repair artifact.
    """
    if not _available("SPY"):
        pytest.skip("SPY cache missing")
    ds = load_data("SPY", method="garch", use_cache=True)
    ws, we = representative_window(ds.returns, ds.true_var, window=250, step=25,
                                   panel=g.PANEL, alpha=ALPHA)
    rr, tv = ds.returns[ws:we], ds.true_var[ws:we]
    full = _brute_force_max(g.PANEL, rr, tv)
    no_chris = _brute_force_max([q for q in g.PANEL if q.__name__ != "christoffersen"],
                                rr, tv)
    # dropping Christoffersen strictly grows the achievable lie -> it binds
    assert no_chris > full + 1e-6
    # and the MILP solver's repaired full-panel lie matches the brute-force max
    rm = solve_milp(tv, rr, panel=g.PANEL, alpha=ALPHA)
    p = price(rm.stated_var, tv, rr)
    assert abs(p.mul_bps - full) < 1e-2
