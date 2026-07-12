"""Tests for the Desk-Backtestability POWER CALCULATOR (gaming_oracle.power).

The load-bearing test is the KNOWN-ANSWER power case: for the Kupiec POF gate the exact
power is a closed-form finite sum over a Binomial(N, p1) pmf, so the Monte-Carlo power
estimate must agree with that analytic value within Monte-Carlo error. The rest pin down
the honest properties the report relies on: correct SIZE at u=0 for the well-calibrated
gates, monotonicity of power in the understatement and in the window length, the ES test
being magnitude-sensitive (higher power than the count gate at the same lie), and the
RED/AMBER/GREEN verdict + minimum-N thresholds.

Math is prior (Kupiec 1995; BCBS 1996; Acerbi-Szekely 2014). These tests validate the
CALCULATOR, not any new statistical result.
"""

import numpy as np
import pytest

from gaming_oracle import power as P


ALPHA = 0.01
LEVEL = 0.95


# ---------------------------------------------------------------------------
# TailModel primitives
# ---------------------------------------------------------------------------

def test_gaussian_true_var_matches_closed_form():
    m = P.TailModel(nu=float("inf"), sigma=0.01)
    # VaR_0.01 = -sigma * Phi^{-1}(0.01) = sigma * 2.3263...
    from scipy import stats
    expect = -0.01 * stats.norm.ppf(0.01)
    assert m.true_var(0.01) == pytest.approx(expect, rel=1e-12)


def test_student_t_standardized_to_unit_variance():
    # A large sample from the t model should have volatility ~= sigma regardless of nu.
    m = P.TailModel(nu=5.0, sigma=0.02)
    rng = np.random.default_rng(0)
    r = m.sample_returns(200_000, rng)
    assert np.std(r) == pytest.approx(0.02, rel=0.03)


def test_student_t_es_deeper_than_var():
    m = P.TailModel(nu=5.0, sigma=0.01)
    assert m.true_es(0.01) > m.true_var(0.01) > 0


def test_true_exception_prob_is_alpha_at_zero_understatement():
    for nu in (float("inf"), 5.0):
        m = P.TailModel(nu=nu, sigma=0.01)
        assert P.true_exception_prob(m, ALPHA, 0.0) == pytest.approx(ALPHA, abs=1e-9)


def test_true_exception_prob_increases_with_understatement():
    m = P.TailModel(nu=float("inf"), sigma=0.01)
    p0 = P.true_exception_prob(m, ALPHA, 0.0)
    p1 = P.true_exception_prob(m, ALPHA, 0.15)
    p2 = P.true_exception_prob(m, ALPHA, 0.30)
    assert p0 < p1 < p2
    assert p0 == pytest.approx(ALPHA, abs=1e-9)


def test_tail_model_rejects_degenerate_nu():
    with pytest.raises(ValueError):
        P.TailModel(nu=1.5)  # variance undefined


# ---------------------------------------------------------------------------
# KNOWN-ANSWER: analytic Kupiec power vs Monte-Carlo
# ---------------------------------------------------------------------------

def test_kupiec_analytic_power_matches_monte_carlo():
    """The load-bearing known-answer case.

    Kupiec's LR depends on data only through X ~ Binomial(N, p1), so its power is an exact
    finite sum. The MC estimate must land within a few standard errors of it.
    """
    m = P.TailModel(nu=float("inf"), sigma=0.01)
    for u in (0.15, 0.25, 0.35):
        ana = P.kupiec_power_analytic(m, n=250, u=u, alpha=ALPHA, level=LEVEL)
        mc = P.mc_power("kupiec_pof", m, n=250, u=u, alpha=ALPHA, level=LEVEL,
                        n_sims=6000, seed=101)
        # tolerance = 4 * binomial standard error (very safe), floored so tiny-power
        # cases still have a usable band
        tol = max(4.0 * mc.stderr, 0.02)
        assert abs(mc.power - ana) <= tol, (
            f"u={u}: analytic={ana:.4f} MC={mc.power:.4f} +-{mc.stderr:.4f}")


def test_kupiec_analytic_power_at_zero_is_the_test_size():
    """Analytic 'power' at u=0 equals the Kupiec test's realized size under H0."""
    m = P.TailModel(nu=float("inf"), sigma=0.01)
    size = P.kupiec_power_analytic(m, n=250, u=0.0, alpha=ALPHA, level=LEVEL)
    # over-sized at N=250/alpha=0.01 (Kupiec's own small-sample chi-square caveat),
    # but bounded well away from 1
    assert 0.02 < size < 0.20


def test_kupiec_power_monotone_in_understatement_analytic():
    m = P.TailModel(nu=float("inf"), sigma=0.01)
    us = [0.0, 0.1, 0.2, 0.3, 0.4]
    powers = [P.kupiec_power_analytic(m, n=250, u=u, alpha=ALPHA, level=LEVEL) for u in us]
    assert all(b >= a - 1e-9 for a, b in zip(powers, powers[1:]))


def test_kupiec_power_monotone_in_window_analytic():
    m = P.TailModel(nu=float("inf"), sigma=0.01)
    ns = [250, 500, 1000, 2000]
    powers = [P.kupiec_power_analytic(m, n=n, u=0.25, alpha=ALPHA, level=LEVEL) for n in ns]
    assert all(b >= a - 1e-9 for a, b in zip(powers, powers[1:]))
    # and it really does climb toward 1 with more data (Kupiec's caveat, made concrete)
    assert powers[-1] > powers[0]


# ---------------------------------------------------------------------------
# SIZE calibration of the well-calibrated gates
# ---------------------------------------------------------------------------

def test_es_z2_size_is_calibrated():
    """The MC critical value must control the ES-Z2 gate's size near 1 - level."""
    m = P.TailModel(nu=float("inf"), sigma=0.01)
    sz = P.mc_power("es_z2_unconditional", m, n=250, u=0.0, alpha=ALPHA, level=LEVEL,
                    n_sims=1200, seed=202)
    assert sz.power == pytest.approx(1 - LEVEL, abs=0.03)


def test_basel_binomial_size_not_above_target():
    """The one-sided exact-binomial gate is conservative: size <= 1 - level (+ MC noise)."""
    m = P.TailModel(nu=float("inf"), sigma=0.01)
    sz = P.mc_power("basel_binomial", m, n=250, u=0.0, alpha=ALPHA, level=LEVEL,
                    n_sims=6000, seed=303)
    assert sz.power <= (1 - LEVEL) + 0.02


# ---------------------------------------------------------------------------
# Magnitude sensitivity: ES test beats the count gate at the same lie
# ---------------------------------------------------------------------------

def test_es_z2_more_powerful_than_kupiec_at_same_lie():
    """Acerbi-Szekely's point: an ES test reads breach magnitude, so at a fixed lie it
    should not be less powerful than the pure count gate. We assert it is at least as
    powerful (with an MC margin)."""
    m = P.TailModel(nu=float("inf"), sigma=0.01)
    k = P.mc_power("kupiec_pof", m, n=250, u=0.20, alpha=ALPHA, level=LEVEL,
                   n_sims=2000, seed=404)
    e = P.mc_power("es_z2_unconditional", m, n=250, u=0.20, alpha=ALPHA, level=LEVEL,
                   n_sims=1000, seed=404)
    assert e.power >= k.power - 0.05


# ---------------------------------------------------------------------------
# Report assembly: verdict, MDU, min-N
# ---------------------------------------------------------------------------

def test_verdict_thresholds():
    assert P.verdict_from_power(0.90) == "green"
    assert P.verdict_from_power(0.80) == "green"
    assert P.verdict_from_power(0.65) == "amber"
    assert P.verdict_from_power(0.50) == "amber"
    assert P.verdict_from_power(0.30) == "red"


def test_min_n_for_power_returns_a_reachable_window():
    m = P.TailModel(nu=float("inf"), sigma=0.01)
    n_star, curve = P.min_n_for_power("basel_binomial", m, u=0.30, alpha=ALPHA,
                                      level=LEVEL, target_power=0.80,
                                      n_grid=[125, 250, 500, 1000], n_sims=1500, seed=5)
    powers = [p for (_, p) in curve]
    # power is non-decreasing in N in expectation; the largest window is the strongest
    assert powers[-1] >= powers[0] - 0.05
    if n_star is not None:
        assert n_star in (125, 250, 500, 1000)


def test_min_detectable_understatement_orders_correctly():
    m = P.TailModel(nu=float("inf"), sigma=0.01)
    # at a SHORT window the min-detectable u should be no smaller than at a LONG window
    u_short, _ = P.min_detectable_understatement("kupiec_pof", m, n=250, alpha=ALPHA,
                                                 level=LEVEL, target_power=0.80,
                                                 n_sims=2000, seed=6)
    u_long, _ = P.min_detectable_understatement("kupiec_pof", m, n=1000, alpha=ALPHA,
                                                level=LEVEL, target_power=0.80,
                                                n_sims=2000, seed=6)
    if u_short is not None and u_long is not None:
        assert u_long <= u_short + 1e-9


def test_backtestability_report_smoke_and_fields():
    rep = P.backtestability_report("kupiec_pof", nu=float("inf"), sigma=0.01, n=250,
                                   ref_understatement=0.25, alpha=ALPHA, level=LEVEL,
                                   target_power=0.80, n_sims=800, seed=0,
                                   n_grid=[250, 500, 1000])
    assert rep.gate == "kupiec_pof"
    assert 0.0 <= rep.power_at_ref <= 1.0
    assert rep.verdict in ("red", "amber", "green")
    assert len(rep.mdu_curve) == 3
    s = rep.summary()
    assert "BACKTESTABILITY" in s
    assert "VERDICT" in s


def test_conformal_report_carries_size_warning():
    """The conformal gate at default tol rejects the honest model often, so its report
    must carry the inflated-size WARNING rather than pretend the false positives are
    detection power."""
    rep = P.backtestability_report("conformal_coverage", nu=float("inf"), sigma=0.01,
                                   n=250, ref_understatement=0.25, alpha=ALPHA,
                                   level=LEVEL, target_power=0.80, n_sims=600, seed=0,
                                   n_grid=[250])
    assert rep.size_at_zero > 0.2  # documented tuning artifact, not detection
    assert "WARNING" in rep.summary()
