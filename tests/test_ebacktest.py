"""Known-answer + property tests for the Wang-Wang-Ziegel e-process ES backtest
gate (the DEFENSE, gap #7).

We verify, against deterministic synthetic data and the real cached assets:
  * the per-period ES e-statistic has E[e] <= 1 under a calibrated null,
  * the e-process is a non-negative supermartingale under the null (anytime-valid
    size control: it rarely crosses 1/alpha when the forecast is honest),
  * the gate PASSES a calibrated series and REJECTS an understated one,
  * the gate is MAGNITUDE-sensitive: it catches a lie that keeps the exception
    COUNT in-band (which every count-only gate misses), and
  * adding the e-backtest to the panel SHRINKS the exhibited lie on all 3 assets
    (the attack-vs-defense result) and never grows it.
"""

import numpy as np
import pytest
from scipy import stats

from gaming_oracle import gates as g
from gaming_oracle.data import load_data
from gaming_oracle.ebacktest import (
    es_e_statistic,
    e_process,
    gaussian_es_var_ratio,
)
from gaming_oracle.report import price
from gaming_oracle.solver import solve

ALPHA = 0.01
LEVEL = 0.95
ASSETS = ["AAPL", "SPY", "BTC-USD"]


# ---------------------------------------------------------------------------
# e-statistic and ES/VaR ratio
# ---------------------------------------------------------------------------

def test_es_var_ratio_gaussian():
    # ES_p/VaR_p for alpha=0.01 Gaussian tail is ~1.1457; strictly > 1.
    r = gaussian_es_var_ratio(0.01)
    assert 1.14 < r < 1.15
    assert gaussian_es_var_ratio(0.05) > 1.0


def test_e_statistic_zero_when_no_breach():
    # no breach (loss below VaR) -> e_t = 0
    loss = np.array([0.001, 0.002, -0.01])
    z = np.array([0.05, 0.05, 0.05])
    r = 1.15 * z
    e = es_e_statistic(loss, z, r, ALPHA)
    assert np.allclose(e, 0.0)


def test_e_statistic_grows_with_breach_depth():
    # deeper breach (same z, r) -> strictly larger e
    z = np.array([0.02, 0.02])
    r = 1.15 * z
    loss = np.array([0.03, 0.10])
    e = es_e_statistic(loss, z, r, ALPHA)
    assert e[1] > e[0] > 0.0


def test_e_statistic_expectation_at_most_one_under_null():
    # Under a calibrated Gaussian null, E[e] <= 1 (the defining e-value property).
    rng = np.random.default_rng(7)
    sigma = 0.01
    n = 200_000
    loss = -rng.normal(0, sigma, n)
    z = -stats.norm.ppf(ALPHA) * sigma
    r = (stats.norm.pdf(stats.norm.ppf(ALPHA)) / ALPHA) * sigma  # true ES
    e = es_e_statistic(loss, np.full(n, z), np.full(n, r), ALPHA)
    assert e.mean() <= 1.0 + 0.05  # MC tolerance; honest forecast is not exploitable


# ---------------------------------------------------------------------------
# e-process size control (anytime-valid) and supermartingale non-negativity
# ---------------------------------------------------------------------------

def _calibrated(n=1000, alpha=ALPHA, seed=0):
    rng = np.random.default_rng(seed)
    sigma = 0.01
    returns = rng.normal(0, sigma, n)
    z = -stats.norm.ppf(alpha) * sigma
    var = np.full(n, z)
    return var, returns


def test_eprocess_calibrated_rarely_rejects():
    # Anytime-valid size: over many honest replications the e-process crosses
    # 1/alpha_test on at most ~alpha_test of them. With 200 seeds at level 0.95,
    # well under 20% should reject (typically a couple percent).
    rejects = 0
    R = 200
    for s in range(R):
        var, returns = _calibrated(seed=s)
        res = g.e_backtest(var, returns, alpha=ALPHA, level=LEVEL)
        rejects += int(not res.passed)
    assert rejects / R <= 0.10, rejects


def test_eprocess_nonnegative_and_starts_at_one():
    var, returns = _calibrated(seed=1)
    res = g.e_backtest(var, returns, alpha=ALPHA, level=LEVEL)
    assert res.stat >= 0.0
    # the max e-process is at least the starting capital 1.0
    assert res.detail["max_eprocess"] >= 1.0
    assert res.detail["threshold"] == pytest.approx(1.0 / (1.0 - LEVEL))


# ---------------------------------------------------------------------------
# gate behaviour: calibrated passes, understated rejected
# ---------------------------------------------------------------------------

def test_e_backtest_calibrated_passes():
    var, returns = _calibrated(seed=0)
    res = g.e_backtest(var, returns, alpha=ALPHA, level=LEVEL)
    assert res.passed, res.detail


def test_e_backtest_understated_rejects():
    var, returns = _calibrated(seed=0)
    res = g.e_backtest(var * 0.4, returns, alpha=ALPHA, level=LEVEL)  # heavy shrink
    assert not res.passed, res.detail
    assert res.detail["crossing_index"] >= 0


# ---------------------------------------------------------------------------
# the structural point: magnitude-sensitivity vs the count-only gates
# ---------------------------------------------------------------------------

def test_e_backtest_catches_count_in_band_magnitude_lie():
    """A lie with the SAME, in-band exception COUNT but huge breach MAGNITUDE.

    Build a series with exactly 2 exceptions over 250 days (Basel green, Kupiec
    fine), where the stated VaR on the breach days is collapsed to nearly zero so
    the realized loss dwarfs it. Every count-only gate (Kupiec, Basel) passes; the
    magnitude-based e-backtest must catch it.
    """
    n = 250
    var = np.full(n, 0.05)
    returns = np.zeros(n)
    # two deep breaches; stated VaR there is tiny so the loss is enormous vs it
    for t in (40, 160):
        returns[t] = -0.20
        var[t] = 1e-4
    # count-only gates see only 2 exceptions -> green
    assert g.basel_traffic_light(var, returns).passed
    assert g.kupiec_pof(var, returns, alpha=ALPHA, level=LEVEL).passed
    # the e-backtest reads the magnitude and rejects
    res = g.e_backtest(var, returns, alpha=ALPHA, level=LEVEL)
    assert not res.passed, res.detail


# ---------------------------------------------------------------------------
# panel wiring
# ---------------------------------------------------------------------------

def test_panel_with_ebt_contains_gate():
    assert g.e_backtest in g.PANEL_WITH_EBT
    assert g.PANEL_WITH_EBT[:4] == g.PANEL


def test_run_panel_with_ebt_calibrated_passes():
    var, returns = _calibrated(seed=0)
    out = g.run_panel(var, returns, panel=g.PANEL_WITH_EBT, alpha=ALPHA, level=LEVEL)
    assert out["passed_all"], {k: v.detail for k, v in out["gates"].items()}
    assert "e_backtest" in out["gates"]


# ---------------------------------------------------------------------------
# attack-vs-defense: adding the gate shrinks (never grows) the lie, all 3 assets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("asset", ASSETS)
def test_defense_shrinks_or_holds_lie(asset):
    """On the most-recent 250-day window where the honest reference passes, the
    4-gate uniform lie must (a) be CAUGHT by the e-backtest, and (b) adding the
    e-backtest must not INCREASE the lie (it shrinks or holds it). The lie can
    never grow because PANEL_WITH_EBT is a superset of PANEL (a tighter feasible
    set). This is the attack-vs-defense contract.
    """
    ds = load_data(asset=asset, alpha=ALPHA)
    # find a recent window where the honest reference passes the 4-gate panel
    rr, tv = ds.returns[-250:], ds.true_var[-250:]
    if not g.run_panel(tv, rr, panel=g.PANEL, alpha=ALPHA)["passed_all"]:
        # fall back to a representative definable window
        from gaming_oracle.decomposition import representative_window
        ws, we = representative_window(ds.returns, ds.true_var, panel=g.PANEL, alpha=ALPHA)
        rr, tv = ds.returns[ws:we], ds.true_var[ws:we]
    assert g.run_panel(tv, rr, panel=g.PANEL, alpha=ALPHA)["passed_all"]

    r4 = solve(tv, rr, panel=g.PANEL, alpha=ALPHA)
    p4 = price(r4.stated_var, tv, rr)
    rE = solve(tv, rr, panel=g.PANEL_WITH_EBT, alpha=ALPHA)
    pE = price(rE.stated_var, tv, rr)

    # the count-only 4-gate lie is caught by the magnitude e-backtest
    caught = not g.e_backtest(r4.stated_var, rr, alpha=ALPHA).passed
    assert caught, "the 4-gate lie should be caught by the e-backtest"
    # defended lie never exceeds the undefended lie (superset feasible set)
    assert pE.mul_bps <= p4.mul_bps + 1e-6
    # and the defended series genuinely passes the full defended panel
    assert rE.passed_all
    # under-reserving is still genuine
    assert np.all(rE.stated_var <= tv + 1e-12)
