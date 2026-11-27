"""Tests for the exact minimum detectable understatement (gaming_oracle.mdu).

The load-bearing checks: the dynamic program conserves probability; its pass flags
are the gates.py flags; the T = 250 values reproduce the numbers of the R1
manuscript (computed there by a separate script); and Monte Carlo agrees with the
exact rejection probability. Math is prior (Kupiec 1995; Christoffersen 1998;
Basel Committee 1996 / MAR99.17).
"""

import numpy as np
import pytest
from scipy import stats

from gaming_oracle import gates as G
from gaming_oracle import mdu as M
from gaming_oracle.feasibility import HitEvaluator


def test_traffic_light_bounds_match_basel_table_at_250():
    assert M.traffic_light_bounds(250) == (5, 10)
    for x in range(13):
        h = np.zeros(250, dtype=bool)
        h[:x] = True
        green = G.basel_traffic_light(*M._synthetic(h)).passed
        assert green == (x < 5)


def test_state_distribution_conserves_mass():
    for T, p, xc in ((120, 0.02, 6), (300, 0.05, 12), (50, 0.3, 4)):
        P, over = M.state_distribution(T, p, xc)
        assert abs(P.sum() + over - 1.0) < 1e-12
        # count marginal equals the binomial below the cap
        marg = P.sum(axis=(0, 1, 3))
        assert np.allclose(marg, stats.binom.pmf(np.arange(xc + 1), T, p), atol=1e-12)


def test_build_seq_has_requested_statistic():
    T = 40
    for f in (0, 1):
        for l in (0, 1):
            for x in range(8):
                for n11 in range(x + 1):
                    s = M.build_seq(T, f, l, x, n11)
                    if s is None:
                        continue
                    assert s[0] == f and s[-1] == l and s.sum() == x
                    assert int(np.sum(s[:-1] & s[1:])) == n11


def test_flags_equal_gates_on_random_sequences():
    T, alpha = 120, 0.025
    tab = M.panel_tables(T, alpha)
    rng = np.random.default_rng(0)
    checked = 0
    while checked < 300:
        h = rng.random(T) < rng.uniform(0.005, 0.06)
        x = int(h.sum())
        if x > tab.xc:
            continue
        n11 = int(np.sum(h[:-1] & h[1:]))
        sv, rr = M._synthetic(h)
        k = G.kupiec_pof(sv, rr, alpha=alpha).passed
        c = G.christoffersen(sv, rr, alpha=alpha).passed
        assert tab.flags[("K", "strict")][int(h[0]), int(h[-1]), x, n11] == k
        assert tab.flags[("KC", "strict")][int(h[0]), int(h[-1]), x, n11] == (k and c)
        checked += 1


def test_reproduces_r1_values_at_250():
    z = stats.norm.isf(0.01)
    r0 = M.reject_probs(0.01, 250, 0.01)
    r10 = M.reject_probs(float(stats.norm.sf(0.9 * z)), 250, 0.01)
    assert abs(r0[("KCB", "strict")] - 0.2024) < 5e-5      # R1: size of P3
    assert abs(r0[("KCB", "lenient")] - 0.0276) < 5e-5     # R1: size of P3L
    assert abs(r10[("KCB", "strict")] - 0.4998) < 5e-5     # R1: pi(0.10)
    p = M.min_detectable_p(250, 0.01, "KCB", "strict", 0.5)
    assert abs(M.mdu_from_p(p, 0.01) - 0.1001) < 5e-5      # R1: exact crossing 0.1001


def test_monte_carlo_agrees_with_exact():
    T, alpha, p, n = 400, 0.025, 0.04, 6000
    rng = np.random.default_rng(np.random.SeedSequence([5424, 99, 1]))
    hits = rng.random((n, T)) < p
    f = HitEvaluator(T, alpha=alpha).evaluate(hits)
    mc = 1 - np.mean(f["kupiec_pof"] & f["christoffersen"])
    ex = M.reject_prob(p, T, alpha, "KC", "strict")
    assert abs(mc - ex) < 4 * np.sqrt(ex * (1 - ex) / n)


def test_mdu_orderings():
    # more data detects smaller understatements; fatter tails hide larger ones
    ps = [M.min_detectable_p(T, 0.01, "K", "strict", 0.8) for T in (250, 500, 1000)]
    assert ps[0] > ps[1] > ps[2] > 0.01
    u = {t: M.mdu_from_p(ps[0], 0.01, t) for t in ("gaussian", 10.0, 5.0, 3.0)}
    assert u["gaussian"] < u[10.0] < u[5.0] < u[3.0]
    # p_from_understatement inverts mdu_from_p
    for t in ("gaussian", 5.0):
        assert abs(M.p_from_understatement(u[t], 0.01, t) - ps[0]) < 1e-10


def test_expected_capital_factor():
    assert M.expected_capital_factor(0.0) == pytest.approx(1.0)
    assert 1.5 < M.expected_multiplier(0.01) < 1.6
    assert M.expected_multiplier(0.5) == pytest.approx(2.0)


def test_ump_rule_has_exact_size_and_dominates_size_adjusted_panels():
    T, a = 250, 0.01
    ump = M.ump_rule(T, a, 0.05)
    assert M.ump_reject_prob(a, T, ump) == pytest.approx(0.05, abs=1e-12)
    for panel, direction in (("K", "two"), ("KC", "upper"), ("KCB", "two")):
        rule = M.size_adjusted_rule(T, a, panel, direction, 0.05)
        assert M.size_adjusted_reject_prob(a, T, a, panel, direction, rule) == pytest.approx(0.05, abs=1e-10)
        # Karlin-Rubin: no test of the exception sequence beats the one-sided binomial test
        for p in (0.015, 0.02, 0.03, 0.04):
            assert M.size_adjusted_reject_prob(p, T, a, panel, direction, rule) <= M.ump_reject_prob(p, T, ump) + 1e-12


def test_desk_rule_against_monte_carlo():
    rng = np.random.default_rng(np.random.SeedSequence([5424, 99]))
    z99, z975 = stats.norm.isf(0.01), stats.norm.isf(0.025)
    for u in (0.2, 0.3):
        e = rng.standard_normal((40000, 250))
        x99 = (e > (1 - u) * z99).sum(axis=1)
        x975 = (e > (1 - u) * z975).sum(axis=1)
        mc = np.mean((x99 > 12) | (x975 > 30))
        ex = M.desk_reject_prob(u)
        assert abs(mc - ex) < 4 * np.sqrt(ex * (1 - ex) / 40000) + 1e-4
    assert M.desk_reject_prob(0.0) < 1e-5
