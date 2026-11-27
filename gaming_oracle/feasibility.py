"""Feasibility study for the R1 revision (cycle 9).

Question. The per-day Maximum Undetectable Lie (MUL) series is built with
knowledge of the realized losses of the window, so no forecaster could report
it. This module measures (i) how often those ex post series are rejected when
they meet fresh losses (experiment E1), (ii) the ex ante detection limit of a
feasible, non-anticipating constant haircut V_t = (1 - c) V*_t (E2), (iii) the
e-backtest against feasible haircuts, with ES tied to VaR and with ES stated
separately, at several horizons (E3), (iv) the sampling distribution of the ex
post uniform limit s* (E4), and (v) a count-feedback haircut that switches back
to the reference after k exceptions (E5).

Design choices are fixed in ``paper/revision-R1/PLAN.md`` (Section 3). Nothing
here re-derives a test: every pass/fail flag is the flag of the corresponding
function in ``gates.py``. Speed comes from two exact devices, not from
approximations:

* Every count test in ``gates.py`` depends on the stated series only through
  the hit sequence, and in fact only through a small sufficient statistic
  (the count x, the last-250-day count, or the Markov transition counts for
  Christoffersen). ``HitEvaluator`` evaluates the real ``gates.py`` function
  once per distinct statistic, on a synthetic series with that statistic, and
  looks the answer up for every path.
* The e-process of ``ebacktest.e_process`` is run for all paths at once with
  the same arithmetic (``eprocess_batch``).

``guard`` re-checks a random sample of (path, setting) pairs against the
``gates.py`` functions called directly and raises on any disagreement.

Math is prior: Kupiec (1995), Christoffersen (1998), BCBS (1996), Wang, Wang
and Ziegel (e-backtesting), Bollerslev (1986) GARCH(1,1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from . import gates as G
from .data import _garch_filter, _garch_fit_params, load_data
from .decomposition import representative_window
from .ebacktest import MIXTURE_GRID, es_e_statistic, gaussian_es_var_ratio

# ---------------------------------------------------------------------------
# constants (identical to the paper)
# ---------------------------------------------------------------------------

ALPHA = 0.01
LEVEL = 0.95
T_WINDOW = 250
NOTIONAL = 1_000_000_000.0
BURN = 500
Z_ALPHA = float(-stats.norm.ppf(ALPHA))
KAPPA = gaussian_es_var_ratio(ALPHA)
MASTER_SEED = 5424

ASSETS = ("AAPL", "SPY", "BTC-USD")
ASSET_INDEX = {"AAPL": 0, "SPY": 1, "BTC-USD": 2}
POOLED = 9
LAW_INDEX = {"D1": 1, "D2": 2}

#: representative windows of the current paper (four-test panel, rolling GARCH)
EXPECTED_WINDOWS = {"AAPL": (2225, 2475), "SPY": (250, 500), "BTC-USD": (1475, 1725)}

#: numbers printed in the current paper, used only as reproduction checks
PAPER_BPS = {
    "S3_P4": {"AAPL": 17.35, "SPY": 5.28, "BTC-USD": 45.33},       # Table 2 uniform
    "S4_P4": {"AAPL": 243.19, "SPY": 125.31, "BTC-USD": 581.23},   # Table 2 greedy
    "S5_P4": {"AAPL": 313.38, "SPY": 166.35, "BTC-USD": 759.15},   # Table 2 MILP
    "S6_P4E": {"AAPL": 241.20, "SPY": 123.92, "BTC-USD": 577.38},  # Table 5 row 2
    "S7_b20": {"AAPL": 71.0, "SPY": 37.9, "BTC-USD": 168.4},       # Table 4 col 1
    "S7_swept": {"AAPL": 14.2, "SPY": 3.8, "BTC-USD": 42.1},       # Table 4 col 3
}

COUNT_TESTS = ("kupiec_pof", "christoffersen", "basel_traffic_light",
               "conformal_coverage", "oneside_uc_score")
PANELS = {
    "P3": ("kupiec_pof", "christoffersen", "basel_traffic_light"),
    "P4": ("kupiec_pof", "christoffersen", "basel_traffic_light",
           "conformal_coverage"),
    "P5": ("kupiec_pof", "christoffersen", "basel_traffic_light",
           "conformal_coverage", "oneside_uc_score"),
    "P3+E": ("kupiec_pof", "christoffersen", "basel_traffic_light", "e_backtest"),
    "P4+E": ("kupiec_pof", "christoffersen", "basel_traffic_light",
             "conformal_coverage", "e_backtest"),
}
#: understatement-direction variants: a rejection caused by too FEW exceptions
#: is not counted (the ``ignore_too_few`` convention).
DIR_PANELS = {
    "P3dir": ("kupiec_dir", "christoffersen_dir", "basel_traffic_light"),
    "P4dir": ("kupiec_dir", "christoffersen_dir", "basel_traffic_light",
              "conformal_dir"),
}
GATE_FUNCS = {
    "kupiec_pof": G.kupiec_pof,
    "christoffersen": G.christoffersen,
    "basel_traffic_light": G.basel_traffic_light,
    "conformal_coverage": G.conformal_coverage,
    "oneside_uc_score": G.oneside_uc_score,
    "e_backtest": G.e_backtest,
}
TAUS = (0.05, 0.10, 0.25, 0.50, 0.80)
DELTAS = (0.05, 0.25, 0.50)
C_GRID = np.round(np.arange(0.0, 0.5001, 0.01), 2)


def seed_tuple(experiment: int, asset: int, law: int) -> list[int]:
    return [MASTER_SEED, int(experiment), int(asset), int(law)]


def rng_for(experiment: int, asset: int, law: int) -> np.random.Generator:
    """numpy Generator seeded by SeedSequence([5424, e, a, d])."""
    return np.random.default_rng(np.random.SeedSequence(seed_tuple(experiment, asset, law)))


# ---------------------------------------------------------------------------
# exact, cached evaluation of the count tests through gates.py
# ---------------------------------------------------------------------------

def _synthetic(hits_row: np.ndarray):
    """(stated_var, returns) whose hit sequence is ``hits_row``."""
    h = np.asarray(hits_row, dtype=bool)
    return np.ones(h.size), np.where(h, -2.0, 0.0)


class HitEvaluator:
    """Look-up evaluation of the ``gates.py`` count tests for one length n.

    Each table entry is produced by calling the real gate on a synthetic series
    with the required sufficient statistic, so the flags are the gates.py flags.
    """

    def __init__(self, n: int, alpha: float = ALPHA, level: float = LEVEL):
        self.n, self.alpha, self.level = int(n), alpha, level
        xs = np.arange(n + 1)
        kup, kdir, conf, cdir, score = [], [], [], [], []
        for x in xs:
            h = np.zeros(n, dtype=bool)
            h[:x] = True
            sv, rr = _synthetic(h)
            k = G.kupiec_pof(sv, rr, alpha=alpha, level=level)
            kup.append(k.passed)
            kdir.append(k.passed or x <= alpha * n)
            c = G.conformal_coverage(sv, rr, alpha=alpha, level=level)
            conf.append(c.passed)
            cdir.append(c.passed or c.detail["gap"] > 0)   # too few -> ignored
            score.append(G.oneside_uc_score(sv, rr, alpha=alpha, level=level).passed)
        self.kupiec = np.array(kup)
        self.kupiec_dir = np.array(kdir)
        self.conformal = np.array(conf)
        self.conformal_dir = np.array(cdir)
        self.score = np.array(score)
        w = min(n, 250)
        bas = []
        for x in range(w + 1):
            h = np.zeros(n, dtype=bool)
            h[n - w:n - w + x] = True
            bas.append(G.basel_traffic_light(*_synthetic(h), alpha=alpha,
                                             level=level).passed)
        self.basel = np.array(bas)
        self._chris: dict[tuple, tuple[bool, bool]] = {}

    # Christoffersen depends on (x, n01, n10, n11) only (n00 = n-1-n01-n10-n11)
    def _chris_lookup(self, keys: np.ndarray, hits: np.ndarray):
        uniq, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        p_all = np.empty(uniq.shape[0], dtype=bool)
        p_dir = np.empty(uniq.shape[0], dtype=bool)
        for j, key in enumerate(map(tuple, uniq)):
            if key not in self._chris:
                r = G.christoffersen(*_synthetic(hits[first[j]]), alpha=self.alpha,
                                     level=self.level)
                x = key[0]
                d = r.detail
                # direction-aware: keep an independence rejection; drop a
                # conditional-coverage rejection driven by too few exceptions
                rej_dir = (not d["pass_ind"]) or ((not d["pass_cc"]) and x > self.alpha * self.n)
                self._chris[key] = (bool(r.passed), not rej_dir)
            p_all[j], p_dir[j] = self._chris[key]
        return p_all[inv], p_dir[inv]

    def evaluate(self, hits: np.ndarray) -> dict:
        """Pass flags (True = passed) for every count test, one per row of hits."""
        hits = np.atleast_2d(np.asarray(hits, dtype=bool))
        n = hits.shape[1]
        assert n == self.n
        x = hits.sum(axis=1)
        prev, cur = hits[:, :-1], hits[:, 1:]
        n01 = np.sum(~prev & cur, axis=1)
        n10 = np.sum(prev & ~cur, axis=1)
        n11 = np.sum(prev & cur, axis=1)
        keys = np.stack([x, n01, n10, n11], axis=1)
        ch, ch_dir = self._chris_lookup(keys, hits)
        x250 = hits[:, -min(n, 250):].sum(axis=1)
        return {
            "x": x,
            "kupiec_pof": self.kupiec[x],
            "christoffersen": ch,
            "basel_traffic_light": self.basel[x250],
            "conformal_coverage": self.conformal[x],
            "oneside_uc_score": self.score[x],
            "kupiec_dir": self.kupiec_dir[x],
            "christoffersen_dir": ch_dir,
            "conformal_dir": self.conformal_dir[x],
        }

    def passes_row(self, hits_row: np.ndarray, panel: tuple) -> bool:
        f = self.evaluate(hits_row[None, :])
        return bool(all(f[g][0] for g in panel))


_EVALUATORS: dict[int, HitEvaluator] = {}


def evaluator(n: int) -> HitEvaluator:
    if n not in _EVALUATORS:
        _EVALUATORS[n] = HitEvaluator(n)
    return _EVALUATORS[n]


# ---------------------------------------------------------------------------
# e-process for many paths (same arithmetic as ebacktest.e_process)
# ---------------------------------------------------------------------------

def eprocess_batch(loss, var, es, alpha: float = ALPHA, level: float = LEVEL,
                   grid=MIXTURE_GRID):
    """Mixture e-process for each row. Returns (passed, max_M, first_cross)."""
    e = es_e_statistic(loss, var, es, alpha)
    e = np.atleast_2d(e)
    n_paths, n = e.shape
    lams = np.asarray(grid, dtype=float)
    w = np.full(lams.size, 1.0 / lams.size)
    thr = 1.0 / (1.0 - level)
    comp = np.ones((n_paths, lams.size))
    max_m = np.ones(n_paths)
    cross = np.full(n_paths, -1, dtype=np.int64)
    one_minus = 1.0 - lams
    for t in range(n):
        comp = np.maximum(comp * (one_minus + lams * e[:, t:t + 1]), 0.0)
        m = comp @ w
        np.maximum(max_m, m, out=max_m)
        new = (cross < 0) & (m >= thr)
        cross[new] = t
    return max_m < thr, max_m, cross


def evaluate_all(loss, var, es=None, *, ebt: bool = True) -> dict:
    """Every test and panel for each row of (loss, var, es). True = passed."""
    loss = np.atleast_2d(loss)
    var = np.broadcast_to(var, loss.shape)
    hits = loss > var
    f = evaluator(loss.shape[1]).evaluate(hits)
    if ebt:
        if es is None:
            es = KAPPA * var
        es = np.broadcast_to(es, loss.shape)
        p, mx, cr = eprocess_batch(loss, var, es)
        f["e_backtest"], f["e_max"], f["e_cross"] = p, mx, cr
    for name, members in {**PANELS, **DIR_PANELS}.items():
        if "e_backtest" in members and not ebt:
            continue
        f[name] = np.logical_and.reduce([f[m] for m in members])
    return f


def guard(loss, var, es, flags: dict, rows, tests=None) -> int:
    """Compare stored flags with gates.py called directly on the given rows.

    Raises AssertionError on the first disagreement. Returns rows checked.
    """
    loss = np.atleast_2d(loss)
    var = np.broadcast_to(var, loss.shape)
    if es is not None:
        es = np.broadcast_to(es, loss.shape)
    if tests is None:
        tests = [t for t in GATE_FUNCS if t in flags]
    for i in rows:
        rr = -loss[i]
        for t in tests:
            fn = GATE_FUNCS[t]
            if t == "e_backtest":
                ok = fn(var[i], rr, alpha=ALPHA, level=LEVEL,
                        stated_es=None if es is None else es[i]).passed
            else:
                ok = fn(var[i], rr, alpha=ALPHA, level=LEVEL).passed
            if bool(ok) != bool(flags[t][i]):
                raise AssertionError(f"guard mismatch: test {t} row {i}: gates.py={ok} "
                                     f"vectorized={flags[t][i]}")
    return len(rows)


# ---------------------------------------------------------------------------
# data-generating processes
# ---------------------------------------------------------------------------

def std_t(rng, nu: float, size) -> np.ndarray:
    """Student-t draws scaled to unit variance."""
    return rng.standard_t(nu, size=size) * np.sqrt((nu - 2.0) / nu)


def draw_innov(rng, law: str, nu: float | None, size) -> np.ndarray:
    if law == "D1":
        return rng.standard_normal(size)
    if law == "D2":
        return std_t(rng, nu, size)
    raise ValueError(law)


def fit_std_t_nu(z: np.ndarray, floor: float = 2.5, cap: float = 200.0) -> float:
    """ML degrees of freedom of a unit-variance Student-t fitted to z."""
    z = np.asarray(z, dtype=float)

    def nll(nu):
        s = np.sqrt(nu / (nu - 2.0))
        return -float(np.sum(stats.t.logpdf(z * s, nu) + np.log(s)))

    res = optimize.minimize_scalar(nll, bounds=(floor, cap), method="bounded",
                                   options={"xatol": 1e-6})
    return float(max(res.x, floor))


def simulate_garch(rng, params, n_paths: int, n: int, law: str, nu=None,
                   burn: int = BURN, eps: np.ndarray | None = None):
    """Zero-mean GARCH(1,1) paths started at the unconditional variance.

    Returns (r_eval, sigma_eval, r_full) with r_full including the burn-in.
    """
    omega, a, b = params
    total = burn + n
    if eps is None:
        eps = draw_innov(rng, law, nu, (n_paths, total))
    r = np.empty((n_paths, total))
    sig = np.empty((n_paths, total))
    s2 = np.full(n_paths, omega / (1.0 - a - b))
    for t in range(total):
        sig[:, t] = np.sqrt(s2)
        r[:, t] = sig[:, t] * eps[:, t]
        s2 = omega + a * r[:, t] ** 2 + b * s2
    return r[:, burn:], sig[:, burn:], r


def ewma_vol_batch(r_full: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Row-wise copy of data.ewma_vol (same seeding rule)."""
    n_paths, n = r_full.shape
    seed = max(20, int(0.05 * n))
    s = np.var(r_full[:, :seed], axis=1)
    out = np.empty_like(r_full)
    for t in range(n):
        out[:, t] = s
        s = lam * s + (1 - lam) * r_full[:, t] ** 2
    return np.sqrt(out)


# ---------------------------------------------------------------------------
# asset set-up
# ---------------------------------------------------------------------------

@dataclass
class AssetSetup:
    name: str
    index: int
    ws: int
    we: int
    rr: np.ndarray            # realized returns in the representative window
    tv: np.ndarray            # reference VaR V* in the window
    mean_var: float           # mean V* in the window (return units)
    params: tuple             # GARCH(1,1) fit (primary)
    fit_sample: str
    nu: float                 # Student-t df fitted to standardized residuals
    params_alt: tuple | None = None   # SPY sensitivity fit r[:250]
    nu_alt: float | None = None
    kurt_window: float = float("nan")
    n_returns: int = 0
    extra: dict = field(default_factory=dict)


def setup_asset(name: str, log=print) -> AssetSetup:
    ds = load_data(asset=name, alpha=ALPHA, method="rolling_garch")
    r, tv_full = ds.returns, ds.true_var
    ws, we = representative_window(r, tv_full, window=T_WINDOW, step=25,
                                   panel=G.PANEL, alpha=ALPHA)
    if (ws, we) != EXPECTED_WINDOWS[name]:
        raise AssertionError(f"{name}: representative window {(ws, we)} != "
                             f"{EXPECTED_WINDOWS[name]}")
    fit_end = 500 if name == "SPY" else ws
    params = _garch_fit_params(r[:fit_end])
    zres = r[:fit_end] / _garch_filter(r[:fit_end], *params)
    nu = fit_std_t_nu(zres)
    params_alt = nu_alt = None
    if name == "SPY":
        params_alt = _garch_fit_params(r[:250])
        nu_alt = fit_std_t_nu(r[:250] / _garch_filter(r[:250], *params_alt))
    rr, tv = r[ws:we], tv_full[ws:we]
    zw = rr / (tv / Z_ALPHA)
    st = AssetSetup(name=name, index=ASSET_INDEX[name], ws=ws, we=we, rr=rr, tv=tv,
                    mean_var=float(np.mean(tv)), params=params,
                    fit_sample=f"r[:{fit_end}]", nu=nu, params_alt=params_alt,
                    nu_alt=nu_alt, kurt_window=float(stats.kurtosis(zw, fisher=False)),
                    n_returns=int(r.size))
    log(f"  {name}: n={r.size} window [{ws}:{we}] mean V*={st.mean_var * 1e4:.2f} bps; "
        f"GARCH fit {st.fit_sample}: omega={params[0]:.4e} a={params[1]:.4f} "
        f"b={params[2]:.4f}; nu={nu:.3f}"
        + (f"; alt fit r[:250]: omega={params_alt[0]:.4e} a={params_alt[1]:.4f} "
           f"b={params_alt[2]:.4f} nu={nu_alt:.3f}" if params_alt else ""))
    return st


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def _rate(flags_pass: np.ndarray) -> float:
    return float(1.0 - np.mean(flags_pass))


def _se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 0.0) / n))


def limit_from_curve(cgrid, pi, bound: float):
    """Largest grid c with pi(c') <= bound for every grid c' <= c; None if pi(0) > bound."""
    cgrid = np.asarray(cgrid)
    pi = np.asarray(pi)
    if pi[0] > bound:
        return None
    ok = np.cumprod(pi <= bound).astype(bool)
    return float(cgrid[np.nonzero(ok)[0][-1]])


def summarize_series_on_paths(f: dict, n_paths: int) -> dict:
    out = {"n_paths": n_paths}
    for k in list(COUNT_TESTS) + ["e_backtest"] + list(PANELS) + list(DIR_PANELS):
        if k in f:
            out[f"rej_{k}"] = _rate(f[k])
    x = f["x"]
    out["x_mean"] = float(np.mean(x))
    out["x_p5"], out["x_p50"], out["x_p95"] = (float(v) for v in np.percentile(x, [5, 50, 95]))
    if "e_cross" in f:
        cr = f["e_cross"]
        rej = cr >= 0
        out["e_cross_median_day"] = float(np.median(cr[rej] + 1)) if rej.any() else None
    return out


# ---------------------------------------------------------------------------
# E1: ex post series on fresh losses
# ---------------------------------------------------------------------------

def build_expost_series(st: AssetSetup, log=print) -> dict:
    """S0 and the ex post series S3 to S7 on the observed window."""
    from .eaware_attacker import max_residual_under_panel, solve_eaware
    from .milp_solver import solve_milp
    from .solver import solve

    tv, rr = st.tv, st.rr
    S = {}
    S["S0"] = (tv.copy(), KAPPA * tv, "reference")
    r = solve(tv, rr, panel=G.PANEL, alpha=ALPHA, refine=False)
    S["S3_P4"] = (r.stated_var, KAPPA * r.stated_var, f"uniform s*={r.shrink:.6f}")
    r = solve(tv, rr, panel=G.PANEL_REGULATORY, alpha=ALPHA, refine=False)
    S["S3_P3"] = (r.stated_var, KAPPA * r.stated_var, f"uniform s*={r.shrink:.6f}")
    r = solve(tv, rr, panel=G.PANEL, alpha=ALPHA)
    S["S4_P4"] = (r.stated_var, KAPPA * r.stated_var, "greedy")
    r = solve_milp(tv, rr, panel=G.PANEL, alpha=ALPHA)
    S["S5_P4"] = (r.stated_var, KAPPA * r.stated_var, f"MILP optimal={r.optimal}")
    r = solve_milp(tv, rr, panel=G.PANEL_REGULATORY, alpha=ALPHA)
    S["S5_P3"] = (r.stated_var, KAPPA * r.stated_var, f"MILP optimal={r.optimal} {r.method}")
    r = solve(tv, rr, panel=G.PANEL_WITH_EBT, alpha=ALPHA)
    S["S6_P4E"] = (r.stated_var, KAPPA * r.stated_var, "greedy vs P4+E")
    r = solve_eaware(tv, rr, budget=0.20, es_independent=True, alpha=ALPHA)
    S["S7_b20"] = (r.stated_var, r.stated_es, "budget 0.20, ES independent")
    r = max_residual_under_panel(tv, rr, es_independent=True, alpha=ALPHA)
    S["S7_swept"] = (r.stated_var, r.stated_es, f"budget swept, chosen b={r.budget:.2f}")
    out = {}
    for k, (v, es, note) in S.items():
        bps = float(np.mean(tv - v) * 1e4)
        f_obs = evaluate_all(-rr, v, es)
        out[k] = {"var": v, "es": es, "m": v / tv, "m_es": es / tv, "note": note,
                  "bps": bps, "frac_zero": float(np.mean(v <= 1e-12)),
                  "obs_pass": {p: bool(f_obs[p][0]) for p in ("P3", "P4", "P4+E")},
                  "obs_exceptions": int(f_obs["x"][0])}
        log(f"    {k:9s} {bps:8.2f} bps  zero-days {out[k]['frac_zero']:.3f}  "
            f"obs exceptions {out[k]['obs_exceptions']}  obs pass P3/P4/P4+E "
            f"{out[k]['obs_pass']['P3']}/{out[k]['obs_pass']['P4']}/"
            f"{out[k]['obs_pass']['P4+E']}  ({note})")
    # reproduction check against the paper
    checks = {}
    for key, ref in PAPER_BPS.items():
        val = out[key]["bps"]
        dec = 2 if key.startswith(("S3", "S4", "S5", "S6")) else 1
        checks[key] = {"paper": ref[st.name], "code": round(val, dec),
                       "match": bool(round(val, dec) == ref[st.name])}
    out["_checks"] = checks
    bad = {k: v for k, v in checks.items() if not v["match"]}
    if bad:
        raise AssertionError(f"{st.name}: ex post series do not reproduce the paper: {bad}")
    return out


def run_e1(setups, n_paths: int = 10_000, log=print) -> dict:
    log("\n=== E1: ex post series on fresh losses ===")
    res = {}
    for st in setups:
        log(f"  -- {st.name}: building ex post series on the observed window")
        series = build_expost_series(st, log=log)
        res[st.name] = {"series_bps": {k: v["bps"] for k, v in series.items() if k[0] == "S"},
                        "frac_zero": {k: v["frac_zero"] for k, v in series.items() if k[0] == "S"},
                        "checks": series["_checks"], "results": {}}
        for law in ("D1", "D2"):
            seed = seed_tuple(1, st.index, LAW_INDEX[law])
            rng = rng_for(1, st.index, LAW_INDEX[law])
            # fixed order: E1a shocks first, then E1b GARCH innovations
            eps_a = draw_innov(rng, law, st.nu, (n_paths, T_WINDOW))
            loss_a = -(st.tv / Z_ALPHA) * eps_a
            r_b, sig_b, _ = simulate_garch(rng, st.params, n_paths, T_WINDOW, law, st.nu)
            loss_b = -r_b
            vstar_b = Z_ALPHA * sig_b
            for scheme, loss, base in (("E1a", loss_a, st.tv), ("E1b", loss_b, vstar_b)):
                key = f"{scheme}/{law}"
                res[st.name]["results"][key] = {"seed": seed}
                for sname, s in ((k, v) for k, v in series.items() if k[0] == "S"):
                    var = s["m"] * base
                    es = s["m_es"] * base
                    f = evaluate_all(loss, var, es)
                    rows = np.random.default_rng(0).choice(n_paths, 60, replace=False)
                    guard(loss, var, es, f, rows)
                    res[st.name]["results"][key][sname] = summarize_series_on_paths(f, n_paths)
                log(f"    {key}: " + "; ".join(
                    f"{k} P3 {v['rej_P3']:.3f} P4 {v['rej_P4']:.3f} P4+E {v['rej_P4+E']:.3f}"
                    for k, v in res[st.name]["results"][key].items() if k != "seed"))
    return res


# ---------------------------------------------------------------------------
# E2 + E3 (T=250): constant haircut S1 power curves
# ---------------------------------------------------------------------------

E2_TESTS = list(COUNT_TESTS) + ["e_backtest", "kupiec_dir", "christoffersen_dir",
                                "conformal_dir"] + list(PANELS) + list(DIR_PANELS)
G_DECOUPLED = (0.0, 0.2, 0.4)


def haircut_curves(loss, vstar, cgrid=C_GRID, guard_pairs: int = 500, guard_seed: int = 0,
                   decoupled=G_DECOUPLED):
    """pi(c) for every test and panel; ES coupled and ES decoupled (E3)."""
    n_paths = loss.shape[0]
    curves = {k: np.empty(len(cgrid)) for k in E2_TESTS}
    for g in decoupled:
        for k in ("e_backtest", "P3+E", "P4+E"):
            curves[f"{k}|g={g}"] = np.empty(len(cgrid))
    cross_med = np.full(len(cgrid), np.nan)
    stored = []
    for j, c in enumerate(cgrid):
        var = (1.0 - c) * vstar
        f = evaluate_all(loss, var, KAPPA * var)
        for k in E2_TESTS:
            curves[k][j] = _rate(f[k])
        cr = f["e_cross"]
        if (cr >= 0).any():
            cross_med[j] = float(np.median(cr[cr >= 0] + 1))
        fd = {}
        for g in decoupled:
            es = (1.0 + g) * KAPPA * vstar
            p, _, _ = eprocess_batch(loss, var, es)
            curves[f"e_backtest|g={g}"][j] = _rate(p)
            curves[f"P3+E|g={g}"][j] = _rate(f["P3"] & p)
            curves[f"P4+E|g={g}"][j] = _rate(f["P4"] & p)
            fd[g] = p
        stored.append((f, fd))
    # guard: random (path, c) pairs against gates.py
    grng = np.random.default_rng(guard_seed)
    pairs = list(zip(grng.integers(0, n_paths, guard_pairs),
                     grng.integers(0, len(cgrid), guard_pairs)))
    for i, j in pairs:
        c = cgrid[j]
        var = (1.0 - c) * vstar[i]
        f, fd = stored[j]
        sub = {t: np.array([f[t][i]]) for t in GATE_FUNCS}
        guard(loss[i:i + 1], var[None, :], KAPPA * var[None, :], sub, [0])
        g = decoupled[(i + j) % len(decoupled)]
        es = (1.0 + g) * KAPPA * vstar[i]
        ok = G.e_backtest(var, -loss[i], alpha=ALPHA, level=LEVEL, stated_es=es).passed
        if bool(ok) != bool(fd[g][i]):
            raise AssertionError(f"guard mismatch decoupled ES g={g} row {i} c={c}")
    return curves, cross_med, len(pairs)


def curve_limits(cgrid, curves: dict, keys) -> dict:
    out = {}
    for k in keys:
        pi = curves[k]
        lim = {"size": float(pi[0])}
        lim["c_tau"] = {str(t): limit_from_curve(cgrid, pi, t) for t in TAUS}
        lim["c_excess"] = {str(d): limit_from_curve(cgrid, pi - pi[0], d) for d in DELTAS}
        out[k] = lim
    return out


def run_e2(setups, n_pooled: int = 20_000, n_asset: int = 10_000, log=print) -> dict:
    log("\n=== E2/E3 (T=250): feasible constant haircut S1 ===")
    by = {s.name: s for s in setups}
    configs = [("D1/R1 pooled", POOLED, "D1", by["AAPL"].params, None, "R1", n_pooled)]
    for st in setups:
        configs.append((f"D2/R1 {st.name}", st.index, "D2", st.params, st.nu, "R1", n_asset))
    spy = by["SPY"]
    configs.append(("D1/R2 SPY", spy.index, "D1", spy.params, None, "R2", n_asset))
    configs.append(("D1/R2 SPY fit r[:250]", spy.index, "D1", spy.params_alt, None, "R2", n_asset))
    configs.append(("D2/R1 SPY fit r[:250]", spy.index, "D2", spy.params_alt, spy.nu_alt, "R1", n_asset))
    out = {"c_grid": C_GRID.tolist(), "configs": {}}
    for name, a, law, params, nu, ref, n in configs:
        t0 = time.time()
        seed = seed_tuple(2, a, LAW_INDEX[law])
        rng = rng_for(2, a, LAW_INDEX[law])
        r, sig, r_full = simulate_garch(rng, params, n, T_WINDOW, law, nu)
        if ref == "R1":
            vstar = Z_ALPHA * sig
        else:
            vstar = Z_ALPHA * ewma_vol_batch(r_full)[:, BURN:]
        curves, cross_med, ng = haircut_curves(-r, vstar)
        keys = [k for k in curves]
        lim = curve_limits(C_GRID, curves, keys)
        out["configs"][name] = {"seed": seed, "n_paths": n, "law": law, "nu": nu,
                                "params": list(params), "reference": ref,
                                "curves": {k: v.tolist() for k, v in curves.items()},
                                "e_cross_median_day": cross_med.tolist(),
                                "limits": lim, "guard_pairs": ng,
                                "seconds": time.time() - t0}
        p3 = curves["P3"]
        log(f"  {name:24s} seed {seed} N={n}: size P3 {p3[0]:.4f} P4 {curves['P4'][0]:.4f} "
            f"ebt {curves['e_backtest'][0]:.4f}; P3 at c=.05/.10/.20 "
            f"{p3[5]:.3f}/{p3[10]:.3f}/{p3[20]:.3f}; ebt "
            f"{curves['e_backtest'][5]:.3f}/{curves['e_backtest'][10]:.3f}/"
            f"{curves['e_backtest'][20]:.3f}; guard {ng} ok; {time.time() - t0:.1f}s")
    return out


# ---------------------------------------------------------------------------
# E3 horizons: kupiec and e-backtest at T = 250, 500, 1000
# ---------------------------------------------------------------------------

HORIZONS = (250, 500, 1000)


def run_e3_horizons(setups, n_paths: int = 5_000, log=print) -> dict:
    log("\n=== E3: horizons T = 250, 500, 1000 (Kupiec, e-backtest) ===")
    by = {s.name: s for s in setups}
    configs = [("D1/R1 pooled", POOLED, "D1", by["AAPL"].params, None)]
    for st in setups:
        configs.append((f"D2/R1 {st.name}", st.index, "D2", st.params, st.nu))
    out = {"c_grid": C_GRID.tolist(), "horizons": list(HORIZONS), "configs": {}}
    tmax = max(HORIZONS)
    for name, a, law, params, nu in configs:
        t0 = time.time()
        seed = seed_tuple(3, a, LAW_INDEX[law])
        rng = rng_for(3, a, LAW_INDEX[law])
        r, sig, _ = simulate_garch(rng, params, n_paths, tmax, law, nu)
        loss, vstar = -r, Z_ALPHA * sig
        cur = {}
        for T in HORIZONS:
            cur[f"kupiec_pof|T={T}"] = np.empty(len(C_GRID))
            for g in ("coupled",) + G_DECOUPLED:
                cur[f"e_backtest[{g}]|T={T}"] = np.empty(len(C_GRID))
        cross_med = {T: np.full(len(C_GRID), np.nan) for T in HORIZONS}
        grng = np.random.default_rng(1)
        n_guard = 0
        for j, c in enumerate(C_GRID):
            var = (1.0 - c) * vstar
            hits = loss > var
            for T in HORIZONS:
                x = hits[:, :T].sum(axis=1)
                cur[f"kupiec_pof|T={T}"][j] = 1.0 - np.mean(evaluator(T).kupiec[x])
            for g in ("coupled",) + G_DECOUPLED:
                es = KAPPA * var if g == "coupled" else (1.0 + g) * KAPPA * vstar
                _, _, cr = eprocess_batch(loss, var, es)
                for T in HORIZONS:
                    rej = (cr >= 0) & (cr < T)
                    cur[f"e_backtest[{g}]|T={T}"][j] = float(np.mean(rej))
                    if g == "coupled" and rej.any():
                        cross_med[T][j] = float(np.median(cr[rej] + 1))
                # guard: a few rows per (c, variant) at a random horizon
                if j % 10 == 0:
                    for i in grng.integers(0, n_paths, 5):
                        T = int(grng.choice(HORIZONS))
                        ok = G.e_backtest(var[i, :T], -loss[i, :T], alpha=ALPHA,
                                          level=LEVEL, stated_es=es[i, :T]).passed
                        mine = not ((cr[i] >= 0) and (cr[i] < T))
                        if bool(ok) != bool(mine):
                            raise AssertionError(f"horizon guard mismatch {name} c={c} g={g}")
                        k_ok = G.kupiec_pof(var[i, :T], -loss[i, :T], alpha=ALPHA,
                                            level=LEVEL).passed
                        if bool(k_ok) != bool(evaluator(T).kupiec[hits[i, :T].sum()]):
                            raise AssertionError("horizon guard mismatch kupiec")
                        n_guard += 1
        lim = curve_limits(C_GRID, cur, list(cur))
        out["configs"][name] = {"seed": seed, "n_paths": n_paths, "law": law, "nu": nu,
                                "curves": {k: v.tolist() for k, v in cur.items()},
                                "limits": lim, "guard_rows": n_guard,
                                "e_cross_median_day": {str(T): v.tolist()
                                                       for T, v in cross_med.items()},
                                "seconds": time.time() - t0}
        log(f"  {name:16s} seed {seed}: c=0.10 rejection "
            + ", ".join(f"T={T}: kupiec {cur[f'kupiec_pof|T={T}'][10]:.3f} ebt "
                        f"{cur[f'e_backtest[coupled]|T={T}'][10]:.3f} ebt(g=0.4) "
                        f"{cur[f'e_backtest[0.4]|T={T}'][10]:.3f}" for T in HORIZONS)
            + f"; guard {n_guard} ok; {time.time() - t0:.1f}s")
    return out


# ---------------------------------------------------------------------------
# E4: ex post uniform limit as a random variable
# ---------------------------------------------------------------------------

def exact_uniform_limit(var: np.ndarray, loss: np.ndarray, panel: tuple):
    """Largest s such that (1-s')V passes ``panel`` for every s' in [0, s].

    Exceptions under (1-s)V are the days with s > s_t, s_t = 1 - loss_t/V_t, so
    the hit set only changes at the breakpoints s_t. Returns (s_star, definable).
    s_star is 1.0 if the panel still passes with V = 0.
    """
    ev = evaluator(var.size)
    s_t = 1.0 - loss / var
    hits = s_t < 0.0
    if not ev.passes_row(hits, panel):
        return float("nan"), False
    idx = np.nonzero((s_t >= 0.0) & (s_t < 1.0))[0]
    for t in idx[np.argsort(s_t[idx], kind="stable")]:
        hits[t] = True
        if not ev.passes_row(hits, panel):
            return float(s_t[t]), True
    return 1.0, True


def run_e4(setups, n_paths: int = 5_000, n_check: int = 100, log=print) -> dict:
    from .solver import solve

    log("\n=== E4: distribution of the ex post uniform limit s* ===")
    out = {}
    for st in setups:
        out[st.name] = {"realized": {}, "laws": {}}
        for pname, gpanel in (("P3", G.PANEL_REGULATORY), ("P4", G.PANEL)):
            s_obs, ok = exact_uniform_limit(st.tv, -st.rr, PANELS[pname])
            s_bis = solve(st.tv, st.rr, panel=gpanel, alpha=ALPHA, refine=False).shrink
            out[st.name]["realized"][pname] = {"s_star": s_obs, "s_bisection": s_bis,
                                               "bps": s_obs * st.mean_var * 1e4}
        for law in ("D1", "D2"):
            seed = seed_tuple(4, st.index, LAW_INDEX[law])
            rng = rng_for(4, st.index, LAW_INDEX[law])
            r, sig, _ = simulate_garch(rng, st.params, n_paths, T_WINDOW, law, st.nu)
            loss, vstar = -r, Z_ALPHA * sig
            rec = {"seed": seed, "n_paths": n_paths}
            for pname, gpanel in (("P3", G.PANEL_REGULATORY), ("P4", G.PANEL)):
                ss = np.full(n_paths, np.nan)
                for i in range(n_paths):
                    ss[i], _ = exact_uniform_limit(vstar[i], loss[i], PANELS[pname])
                good = ~np.isnan(ss)
                vals = ss[good]
                # cross-check against solver.solve(refine=False) on n_check paths
                mism = 0
                checked = 0
                for i in np.nonzero(good)[0][:n_check]:
                    b = solve(vstar[i], r[i], panel=gpanel, alpha=ALPHA, refine=False).shrink
                    checked += 1
                    if abs(b - ss[i]) > 1e-9:
                        mism += 1
                real = out[st.name]["realized"][pname]["s_star"]
                rec[pname] = {
                    "share_definable": float(good.mean()),
                    "pct": {str(q): float(np.percentile(vals, q)) for q in (5, 25, 50, 75, 95)},
                    "mean": float(vals.mean()),
                    "share_s_eq_1": float(np.mean(vals >= 1.0)),
                    "realized_percentile_rank": float(np.mean(vals <= real) * 100),
                    "bisection_checked": checked, "bisection_mismatch": mism,
                    "_vals": vals,
                }
            out[st.name]["laws"][law] = rec
            log(f"  {st.name} {law} seed {seed}: "
                + "; ".join(f"{p}: definable {rec[p]['share_definable']:.3f}, s* p5/p50/p95 "
                            f"{rec[p]['pct']['5']:.4f}/{rec[p]['pct']['50']:.4f}/"
                            f"{rec[p]['pct']['95']:.4f}, realized "
                            f"{out[st.name]['realized'][p]['s_star']:.4f} at pct "
                            f"{rec[p]['realized_percentile_rank']:.1f}, bisection mismatches "
                            f"{rec[p]['bisection_mismatch']}/{rec[p]['bisection_checked']}"
                            for p in ("P3", "P4")))
    return out


# ---------------------------------------------------------------------------
# E5: count-feedback haircut S2
# ---------------------------------------------------------------------------

def count_feedback_var(loss, vstar, c: float, k: int):
    """S2: (1-c)V* until k exceptions have occurred up to t-1, then V*."""
    n_paths, n = loss.shape
    var = np.empty_like(vstar)
    cnt = np.zeros(n_paths, dtype=np.int64)
    for t in range(n):
        mult = np.where(cnt < k, 1.0 - c, 1.0)
        var[:, t] = mult * vstar[:, t]
        cnt += loss[:, t] > var[:, t]
    return var


E5_C = np.round(np.arange(0.05, 0.5001, 0.05), 2)
E5_K = (2, 3, 4)
E5_PANELS = ("P3", "P4", "P3+E", "P4+E", "e_backtest", "P3dir")


def run_e5(setups, n_paths: int = 5_000, log=print) -> dict:
    log("\n=== E5: count-feedback haircut S2 ===")
    configs = [("D1/R1 pooled", POOLED, "D1", setups[0].params, None)]
    for st in setups:
        configs.append((f"D2/R1 {st.name}", st.index, "D2", st.params, st.nu))
    out = {"c_grid": E5_C.tolist(), "k": list(E5_K), "configs": {}}
    for name, a, law, params, nu in configs:
        seed = seed_tuple(5, a, LAW_INDEX[law])
        rng = rng_for(5, a, LAW_INDEX[law])
        r, sig, _ = simulate_garch(rng, params, n_paths, T_WINDOW, law, nu)
        loss, vstar = -r, Z_ALPHA * sig
        f0 = evaluate_all(loss, vstar, KAPPA * vstar)
        base = {p: _rate(f0[p]) for p in E5_PANELS}
        rec = {"seed": seed, "n_paths": n_paths, "S0": base, "k": {}}
        grng = np.random.default_rng(2)
        for k in E5_K:
            rows = {p: [base[p]] for p in E5_PANELS}
            under = [0.0]
            for c in E5_C:
                var = count_feedback_var(loss, vstar, float(c), k)
                f = evaluate_all(loss, var, KAPPA * var)
                for p in E5_PANELS:
                    rows[p].append(_rate(f[p]))
                under.append(float(np.mean(1.0 - var / vstar)))
                gi = grng.integers(0, n_paths, 20)
                guard(loss, var, KAPPA * var, f, gi)
            cg = np.concatenate([[0.0], E5_C])
            lim = {p: {str(d): limit_from_curve(cg, np.array(rows[p]) - rows[p][0], d)
                       for d in DELTAS} for p in E5_PANELS}
            rec["k"][str(k)] = {"rej": rows, "understatement_frac": under,
                                "c_excess": lim}
        out["configs"][name] = rec
        log(f"  {name:16s} seed {seed}: S0 P3 {base['P3']:.3f}; " + "; ".join(
            f"k={k}: c=0.20 P3 {rec['k'][str(k)]['rej']['P3'][4]:.3f} "
            f"under {rec['k'][str(k)]['understatement_frac'][4]:.4f}, c=0.50 P3 "
            f"{rec['k'][str(k)]['rej']['P3'][10]:.3f} under "
            f"{rec['k'][str(k)]['understatement_frac'][10]:.4f}" for k in E5_K))
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_all(log=print, quick: bool = False) -> dict:
    """Run E1 to E5. ``quick`` shrinks every N for smoke tests."""
    t_all = time.time()
    timings = {}
    sc = 0.02 if quick else 1.0

    def n(v):
        return max(int(v * sc), 50)

    log("== Gaming Oracle cycle 9: feasibility study (R1) ==")
    log(f"alpha={ALPHA} level={LEVEL} T={T_WINDOW} burn-in={BURN} z_alpha={Z_ALPHA:.6f} "
        f"kappa={KAPPA:.6f} notional=${NOTIONAL:,.0f}; seeds SeedSequence([5424, e, a, d])")
    t0 = time.time()
    setups = [setup_asset(a, log=log) for a in ASSETS]
    timings["setup"] = time.time() - t0
    res = {"settings": {"alpha": ALPHA, "level": LEVEL, "T": T_WINDOW, "burn": BURN,
                        "z_alpha": Z_ALPHA, "kappa": KAPPA, "notional": NOTIONAL,
                        "mixture_grid": list(MIXTURE_GRID), "quick": quick},
           "assets": {s.name: {"window": [s.ws, s.we], "mean_var_bps": s.mean_var * 1e4,
                               "params": list(s.params), "fit_sample": s.fit_sample,
                               "nu": s.nu, "params_alt": (list(s.params_alt)
                                                          if s.params_alt else None),
                               "nu_alt": s.nu_alt, "kurtosis_window": s.kurt_window,
                               "n_returns": s.n_returns} for s in setups}}
    for key, fn, kw in (("E1", run_e1, {"n_paths": n(10_000)}),
                        ("E2", run_e2, {"n_pooled": n(20_000), "n_asset": n(10_000)}),
                        ("E3h", run_e3_horizons, {"n_paths": n(5_000)}),
                        ("E4", run_e4, {"n_paths": n(5_000), "n_check": 100 if not quick else 10}),
                        ("E5", run_e5, {"n_paths": n(5_000)})):
        t0 = time.time()
        res[key] = fn(setups, log=log, **kw)
        timings[key] = time.time() - t0
        log(f"  [{key} done in {timings[key]:.1f}s]")
    timings["total"] = time.time() - t_all
    res["timings_s"] = timings
    log(f"\nTotal wall-clock {timings['total']:.1f}s")
    return res
