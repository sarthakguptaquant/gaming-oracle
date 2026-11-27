"""Known-answer tests for the R1 feasibility study (gaming_oracle.feasibility).

Fast: no data download, small N. They check that the vectorized evaluation is the
gates.py evaluation, that Monte Carlo rejection rates match exact binomial sums,
that the exact breakpoint search returns the true uniform limit, and that seeds
reproduce.
"""

import numpy as np
import pytest
from scipy import stats

from gaming_oracle import feasibility as F
from gaming_oracle import gates as G
from gaming_oracle import power as P

PARAMS = (1.45e-05, 0.0998, 0.8554)


def _paths(n=300, law="D1", nu=None, seed=(9, 9, 1), T=250):
    rng = F.rng_for(*seed)
    r, sig, r_full = F.simulate_garch(rng, PARAMS, n, T, law, nu)
    return -r, F.Z_ALPHA * sig, r_full


def test_vectorized_panel_equals_gates_py():
    loss, vstar, _ = _paths(n=200)
    for c in (0.0, 0.1, 0.3):
        var = (1 - c) * vstar
        f = F.evaluate_all(loss, var, F.KAPPA * var)
        F.guard(loss, var, F.KAPPA * var, f, range(200))
        for i in range(0, 200, 7):
            out = G.run_panel(var[i], -loss[i], panel=G.PANEL_WITH_EBT)
            assert out["passed_all"] == bool(f["P4+E"][i])
            out = G.run_panel(var[i], -loss[i], panel=G.PANEL_REGULATORY)
            assert out["passed_all"] == bool(f["P3"][i])


def test_decoupled_es_and_student_t_guard():
    loss, vstar, _ = _paths(n=150, law="D2", nu=4.0, seed=(9, 9, 2))
    var = 0.85 * vstar
    es = 1.4 * F.KAPPA * vstar
    f = F.evaluate_all(loss, var, es)
    F.guard(loss, var, es, f, range(150))


@pytest.mark.parametrize("c", [0.0, 0.10, 0.20])
def test_mc_count_tests_match_exact_binomial(c):
    n = 4000
    rng = np.random.default_rng(F.seed_tuple(9, 9, 1)[-1] + int(c * 100))
    eps = rng.standard_normal((n, 250))
    loss = -eps
    var = np.full(250, (1 - c) * F.Z_ALPHA)
    f = F.evaluate_all(loss, var, ebt=False)
    p1 = P.true_exception_prob(P.TailModel(), F.ALPHA, c)
    assert p1 == pytest.approx(stats.norm.cdf(-(1 - c) * F.Z_ALPHA))
    pmf = stats.binom.pmf(np.arange(251), 250, p1)
    ev = F.evaluator(250)
    exact = {
        "kupiec_pof": P.kupiec_power_analytic(P.TailModel(), n=250, u=c, alpha=F.ALPHA),
        "basel_traffic_light": float(np.sum(pmf[5:])),
        "conformal_coverage": float(np.sum(pmf[~ev.conformal])),
        "oneside_uc_score": float(np.sum(pmf[~ev.score])),
    }
    assert exact["kupiec_pof"] == pytest.approx(float(np.sum(pmf[~ev.kupiec])), abs=1e-12)
    for k, p in exact.items():
        mc = 1 - f[k].mean()
        se = np.sqrt(p * (1 - p) / n)
        assert abs(mc - p) <= 3 * se + 1e-12, (k, c, mc, p)


def test_count_tables_known_values():
    ev = F.evaluator(250)
    # Kupiec non-rejection band at T=250, alpha=0.01 is x in [1, 6]
    assert not ev.kupiec[0] and ev.kupiec[1] and ev.kupiec[6] and not ev.kupiec[7]
    assert ev.kupiec_dir[0] and ev.kupiec_dir[2] and not ev.kupiec_dir[7]
    # Basel green is x <= 4; conformal band at tol 0.005 is x in [2, 3]
    assert ev.basel[4] and not ev.basel[5]
    assert list(np.nonzero(ev.conformal)[0]) == [2, 3]
    assert list(np.nonzero(ev.conformal_dir)[0]) == [0, 1, 2, 3]


def test_zero_haircut_reproduces_reference():
    loss, vstar, _ = _paths(n=50)
    f0 = F.evaluate_all(loss, vstar)
    fc = F.evaluate_all(loss, (1 - 0.0) * vstar)
    for k in ("P3", "P4+E", "x"):
        assert np.array_equal(f0[k], fc[k])
    # S2 with c=0 is the reference too
    assert np.array_equal(F.count_feedback_var(loss, vstar, 0.0, 3), vstar)


def test_exact_uniform_limit_is_the_breakpoint():
    loss, vstar, r_full = _paths(n=120, seed=(9, 9, 3))
    from gaming_oracle.solver import solve
    done = 0
    for i in range(120):
        s, ok = F.exact_uniform_limit(vstar[i], loss[i], F.PANELS["P3"])
        if not ok or s >= 1.0:
            continue
        # at s* the panel passes; just above the breakpoint it fails
        assert G.run_panel((1 - s) * vstar[i], -loss[i], panel=G.PANEL_REGULATORY)["passed_all"]
        assert not G.run_panel((1 - s - 1e-9) * vstar[i], -loss[i],
                               panel=G.PANEL_REGULATORY)["passed_all"]
        if done < 10:
            b = solve(vstar[i], -loss[i], panel=G.PANEL_REGULATORY, refine=False).shrink
            assert abs(b - s) < 1e-9
        done += 1
    assert done > 20


def test_count_feedback_matches_loop():
    loss, vstar, _ = _paths(n=30, seed=(9, 9, 4))
    var = F.count_feedback_var(loss, vstar, 0.3, 2)
    for i in range(30):
        cnt = 0
        for t in range(250):
            v = (0.7 if cnt < 2 else 1.0) * vstar[i, t]
            assert var[i, t] == pytest.approx(v)
            cnt += int(loss[i, t] > v)


def test_ewma_batch_matches_data_ewma():
    from gaming_oracle.data import ewma_vol
    _, _, r_full = _paths(n=5)
    ew = F.ewma_vol_batch(r_full)
    for i in range(5):
        np.testing.assert_allclose(ew[i], ewma_vol(r_full[i]), rtol=1e-12)


def test_same_seed_same_output():
    a = _paths(n=20, seed=(2, 1, 2), law="D2", nu=5.0)[0]
    b = _paths(n=20, seed=(2, 1, 2), law="D2", nu=5.0)[0]
    assert np.array_equal(a, b)
    c1, _, _ = F.haircut_curves(*_paths(n=40)[:2], cgrid=np.array([0.0, 0.1]), guard_pairs=5)
    c2, _, _ = F.haircut_curves(*_paths(n=40)[:2], cgrid=np.array([0.0, 0.1]), guard_pairs=5)
    for k in c1:
        assert np.array_equal(c1[k], c2[k])


def test_limit_from_curve():
    cg = np.array([0.0, 0.1, 0.2, 0.3])
    assert F.limit_from_curve(cg, np.array([0.2, 0.3, 0.6, 0.4]), 0.5) == 0.1
    assert F.limit_from_curve(cg, np.array([0.6, 0.7, 0.8, 0.9]), 0.5) is None
    assert F.limit_from_curve(cg, np.array([0.1, 0.1, 0.1, 0.1]), 0.5) == 0.3
