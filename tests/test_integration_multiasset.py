"""Multi-asset Cycle-2 integration test on real AAPL / SPY / BTC-USD windows.

Uses cached npz; skips cleanly per-asset if a cache is absent (fresh checkout
with no network). Validates the rolling distribution + leave-one-out
decomposition end to end on real data, and re-verifies the validation contract
(every solved window's stated series passes the panel independently).
"""

import os

import numpy as np
import pytest

from gaming_oracle import gates as g
from gaming_oracle.data import CACHE_DIR, load_data
from gaming_oracle.decomposition import leave_one_out, representative_window
from gaming_oracle.rolling import rolling_mul
from gaming_oracle.solver import solve

ALPHA = 0.01
ASSETS = ["AAPL", "SPY", "BTC-USD"]


def _cache(asset):
    return os.path.join(CACHE_DIR, f"{asset}_returns.npz")


def _available(asset):
    return os.path.exists(_cache(asset))


@pytest.mark.parametrize("asset", ASSETS)
def test_rolling_distribution_runs_and_is_consistent(asset):
    if not _available(asset):
        pytest.skip(f"{asset} cache missing")
    ds = load_data(asset, method="garch", use_cache=True)
    dist = rolling_mul(ds.returns, ds.true_var, asset=asset, window=250,
                       step=50, panel=g.PANEL, alpha=ALPHA)
    assert dist.n_windows > 0
    assert 0.0 <= dist.frac_honest_passes <= 1.0
    # every definable window's solved stated series passes the panel from scratch
    for w in dist.windows:
        if not w.honest_passes:
            continue
        rr = ds.returns[w.start:w.end]
        tv = ds.true_var[w.start:w.end]
        res = solve(tv, rr, panel=g.PANEL, alpha=ALPHA)
        out = g.run_panel(res.stated_var, rr, panel=g.PANEL, alpha=ALPHA)
        assert out["passed_all"]
        assert np.all(res.stated_var <= tv + 1e-12)
    if dist.n_definable:
        assert dist.mul_bps_min <= dist.mul_bps_median <= dist.mul_bps_max


@pytest.mark.parametrize("asset", ASSETS)
def test_decomposition_on_representative_window(asset):
    if not _available(asset):
        pytest.skip(f"{asset} cache missing")
    ds = load_data(asset, method="garch", use_cache=True)
    dist = rolling_mul(ds.returns, ds.true_var, asset=asset, window=250,
                       step=50, panel=g.PANEL, alpha=ALPHA)
    if dist.n_definable == 0:
        pytest.skip(f"{asset}: no definable window, decomposition not meaningful")
    ws, we = representative_window(ds.returns, ds.true_var, window=250, step=50,
                                   panel=g.PANEL, alpha=ALPHA)
    decomp = leave_one_out(ds.returns, ds.true_var, asset=asset,
                           window_start=ws, window_end=we, panel=g.PANEL,
                           alpha=ALPHA)
    # contributions ranked descending, non-negative, bounded by the ceiling span
    bps = [c.contribution_bps for c in decomp.contributions]
    assert bps == sorted(bps, reverse=True)
    assert all(b >= -1e-9 for b in bps)
    assert decomp.mul_full_bps <= decomp.mul_nopanel_bps + 1e-6
    # the table renders without error
    assert isinstance(decomp.ranking_table(), str)
