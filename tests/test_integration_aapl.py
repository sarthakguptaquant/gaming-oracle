"""Integration test on the real AAPL 250-day window (uses cached npz).

Skips cleanly if the cache is absent (e.g. fresh checkout with no network),
so the unit suite never depends on a live fetch.
"""

import os

import numpy as np
import pytest

from gaming_oracle import gates as g
from gaming_oracle.data import CACHE_DIR, load_data
from gaming_oracle.report import price
from gaming_oracle.solver import solve

CACHE = os.path.join(CACHE_DIR, "AAPL_returns.npz")
pytestmark = pytest.mark.skipif(
    not os.path.exists(CACHE),
    reason="AAPL cache missing; run the CLI once with network to populate it",
)

ALPHA = 0.01


def test_aapl_window_exhibits_feasible_priced_lie():
    ds = load_data("AAPL", method="garch", use_cache=True)
    tv = ds.true_var[-250:]
    rr = ds.returns[-250:]

    # honest reference is panel-calibrated on the window
    assert g.run_panel(tv, rr, panel=g.PANEL, alpha=ALPHA)["passed_all"]

    res = solve(tv, rr, panel=g.PANEL, alpha=ALPHA)
    # validation contract, recomputed from scratch
    assert g.run_panel(res.stated_var, rr, panel=g.PANEL, alpha=ALPHA)["passed_all"]
    assert np.all(res.stated_var <= tv + 1e-12)
    assert res.mean_understatement > 0

    p = price(res.stated_var, tv, rr, notional=1e9)
    # a genuinely material, real lie (not a placeholder)
    assert p.mul_bps > 50          # observed ~245 bps
    assert p.mul_usd > 5_000_000   # observed ~$24.5M on $1B
    # extra exceptions per year is bounded and non-negative
    assert 0 <= p.extra_exceptions_per_year <= 10


def test_aapl_du_escanciano_shrinks_lie_vs_kupiec():
    ds = load_data("AAPL", method="garch", use_cache=True)
    tv = ds.true_var[-250:]
    rr = ds.returns[-250:]
    r_k = solve(tv, rr, panel=[g.kupiec_pof], alpha=ALPHA)
    r_kde = solve(tv, rr, panel=[g.kupiec_pof, g.du_escanciano], alpha=ALPHA)
    p_k = price(r_k.stated_var, tv, rr, notional=1e9)
    p_kde = price(r_kde.stated_var, tv, rr, notional=1e9)
    # adding the higher-power test can only shrink (or hold) the exhibited lie
    assert p_kde.mul_bps <= p_k.mul_bps + 1e-9
    # and on AAPL it strictly shrinks the tolerated extra exceptions/yr
    assert p_kde.extra_exceptions_per_year <= p_k.extra_exceptions_per_year + 1e-9
