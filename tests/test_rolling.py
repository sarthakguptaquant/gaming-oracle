"""Rolling-window MUL distribution tests (Cycle 2).

Includes a DETERMINISM test (same inputs -> identical distribution) and the
key honesty check: stats are computed over definable windows only and
frac_honest_passes reports the conditioning.
"""

import numpy as np
import pytest

from gaming_oracle import gates as g
from gaming_oracle.rolling import RollingDistribution, rolling_mul

ALPHA = 0.01


def _calibrated_normal(n=1200, sigma=0.01, seed=11):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, sigma, n)
    from scipy.stats import norm
    z = -norm.ppf(ALPHA)
    true_var = np.full(n, z * sigma)
    return true_var, returns


def test_rolling_basic_shape_and_window_coverage():
    tv, rr = _calibrated_normal(n=1000, seed=1)
    dist = rolling_mul(rr, tv, asset="SYNTH", window=250, step=100, panel=g.PANEL,
                       alpha=ALPHA)
    assert isinstance(dist, RollingDistribution)
    # windows cover the series; the last window must end at n
    assert dist.windows[-1].end == 1000
    assert dist.windows[0].start == 0
    assert dist.n_windows == len(dist.windows)
    # frac in [0,1]
    assert 0.0 <= dist.frac_honest_passes <= 1.0
    assert dist.n_definable <= dist.n_windows


def test_rolling_stats_only_over_definable_windows():
    tv, rr = _calibrated_normal(n=1000, seed=2)
    dist = rolling_mul(rr, tv, asset="SYNTH", window=250, step=50, panel=g.PANEL,
                       alpha=ALPHA)
    # recompute median over definable windows independently
    definable = [w.mul_bps for w in dist.windows if w.honest_passes]
    assert len(definable) == dist.n_definable
    if definable:
        assert dist.mul_bps_median == pytest.approx(float(np.median(definable)), abs=1e-9)
        assert dist.mul_bps_min == pytest.approx(min(definable), abs=1e-9)
        assert dist.mul_bps_max == pytest.approx(max(definable), abs=1e-9)
        assert dist.mul_bps_iqr == pytest.approx(dist.mul_bps_q3 - dist.mul_bps_q1, abs=1e-9)
        # ordering sanity
        assert dist.mul_bps_min <= dist.mul_bps_median <= dist.mul_bps_max


def test_rolling_determinism():
    tv, rr = _calibrated_normal(n=900, seed=3)
    d1 = rolling_mul(rr, tv, window=250, step=50, panel=g.PANEL, alpha=ALPHA)
    d2 = rolling_mul(rr, tv, window=250, step=50, panel=g.PANEL, alpha=ALPHA)
    assert d1.n_definable == d2.n_definable
    assert d1.frac_honest_passes == d2.frac_honest_passes
    assert d1.mul_bps_median == d2.mul_bps_median
    assert [w.mul_bps for w in d1.windows] == [w.mul_bps for w in d2.windows]


def test_rolling_raises_when_series_shorter_than_window():
    tv, rr = _calibrated_normal(n=100)
    with pytest.raises(ValueError):
        rolling_mul(rr, tv, window=250, step=25, panel=g.PANEL, alpha=ALPHA)


def test_undefinable_everywhere_reports_zero_distribution():
    """If the honest reference never passes (deliberately under-vol'd ref so it
    over-breaches), the distribution is reported as zero with frac=0, not hidden.
    """
    rng = np.random.default_rng(7)
    n = 600
    returns = rng.normal(0, 0.02, n)        # realized vol 2%
    # honest ref deliberately too small -> chronic breaches -> never passes
    true_var = np.full(n, 0.005)
    dist = rolling_mul(
        returns, true_var, window=250, step=50, panel=g.PANEL, alpha=ALPHA)
    assert dist.n_definable == 0
    assert dist.frac_honest_passes == 0.0
    assert dist.mul_bps_median == 0.0
    assert "UNDEFINABLE" in dist.summary()
