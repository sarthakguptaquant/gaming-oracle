"""Exact minimum detectable exception probability and understatement (MDU).

For a panel of count-based VaR backtests the rejection probability depends on the
reported series only through the exception (hit) sequence. If exceptions are
independent with common probability p, the law of that sequence depends on p only,
so the rejection probability pi(p) of the panel is a function of
(T, alpha, panel, pass rule, p) and of nothing else. This module computes pi(p)
exactly with a dynamic program over the sufficient statistic of the Kupiec and
Christoffersen tests (first hit, last hit, count, number of one-one transitions),
and inverts it:

* minimum detectable exception probability
      p*_beta = inf{ p > alpha : pi(p') >= beta for every p' >= p },
  the "every larger p" clause handling the non-monotone two-sided Kupiec curve;
* minimum detectable understatement (MDU) for a loss tail with upper quantile q,
      u*_beta = 1 - q(p*_beta) / q(alpha).

Every pass or fail flag of the Kupiec and Christoffersen tests is the flag of the
function in ``gates.py``, called on a synthetic sequence with the required
statistic. The traffic light uses the zone rule of the Basel Committee for any
sample size (Basel Framework MAR99.17: the amber zone begins at the smallest count
whose cumulative binomial probability under 99% coverage is at least 95%, the red
zone at 99.99%); at T = 250 this gives the familiar 5 and 10 and agrees with
``gates.basel_traffic_light``.

Math is prior: Kupiec (1995), Christoffersen (1998), Basel Committee (1996), and
the power of the Kupiec test under VaR under-reporting is studied by Campbell
(2005/2007). Exact finite-sample distributions of the Christoffersen statistics by
dynamic programming are also available in the R package ExactVaRTest. Nothing here
is a new test; the module tabulates the inverse of a power function.

Pass rules
----------
strict  : any rejection counts (two-sided Kupiec and Christoffersen conditional
          coverage; traffic light fails outside the green zone).
lenient : only evidence of too many exceptions counts (Kupiec and the coverage part
          of Christoffersen fail only when x > alpha T; the independence test is kept;
          the traffic light fails only in the red zone).

Panels: "K" Kupiec POF; "B" traffic light alone; "KC" Kupiec + Christoffersen;
"KCB" Kupiec + Christoffersen + traffic light (alpha = 0.01 only).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import optimize, stats

from . import gates as G

LEVEL = 0.95
PANELS = ("K", "B", "KC", "KCB")
RULES = ("strict", "lenient")
#: backtesting-dependent multiplier by exception count over 250 days (Basel
#: Framework MAR32.9 Table 1 / MAR99 Table 2, in force 1 January 2023)
MAR_MULTIPLIER = {0: 1.50, 1: 1.50, 2: 1.50, 3: 1.50, 4: 1.50,
                  5: 1.70, 6: 1.76, 7: 1.83, 8: 1.88, 9: 1.92}
MAR_MULTIPLIER_RED = 2.00


def _synthetic(h: np.ndarray):
    """(stated_var, returns) whose hit sequence is ``h``."""
    h = np.asarray(h, dtype=bool)
    return np.ones(h.size), np.where(h, -2.0, 0.0)


def traffic_light_bounds(T: int, coverage: float = 0.99) -> tuple[int, int]:
    """(first amber count, first red count) by the MAR99.17 cumulative rule."""
    x = np.arange(T + 1)
    cdf = stats.binom.cdf(x, T, 1.0 - coverage)
    amber = int(np.argmax(cdf >= 0.95 - 1e-12))
    red = int(np.argmax(cdf >= 0.9999 - 1e-12))
    return amber, red


def build_seq(T: int, f: int, l: int, x: int, n11: int):
    """A 0/1 sequence of length T with first f, last l, count x and n11 one-one pairs.

    Returns None when no such sequence exists.
    """
    if x == 0:
        return np.zeros(T, dtype=bool) if (f == 0 and l == 0) else None
    runs = x - n11
    if runs < 1:
        return None
    if f == 1 and l == 1 and runs == 1 and x != T:
        return None
    lens = [x - runs + 1] + [1] * (runs - 1)
    seq: list[int] = []
    if f == 0:
        seq.append(0)
    for i, L in enumerate(lens):
        seq += [1] * L
        if i < runs - 1:
            seq.append(0)
    if l == 0:
        seq.append(0)
    if len(seq) > T:
        return None
    pad = T - len(seq)
    if f == 0:
        seq = [0] * pad + seq
    elif l == 0:
        seq = seq + [0] * pad
    elif runs >= 2:
        j = lens[0]
        seq = seq[:j] + [0] * pad + seq[j:]
    elif pad > 0:
        return None
    s = np.array(seq, dtype=bool)
    if s[0] != f or s[-1] != l or s.sum() != x or int(np.sum(s[:-1] & s[1:])) != n11:
        return None
    return s


@dataclass
class PanelTables:
    """Pass flags on the sufficient-statistic grid for one (T, alpha)."""

    T: int
    alpha: float
    xc: int                       # largest count at which any panel can pass
    valid: np.ndarray             # (2, 2, xc+1, xc+1) reachable states
    flags: dict                   # (panel, rule) -> bool array, True = pass
    kupiec_band: tuple[int, int]  # two-sided Kupiec acceptance band
    tl_bounds: tuple[int, int] | None


@lru_cache(maxsize=None)
def panel_tables(T: int, alpha: float) -> PanelTables:
    """Build the pass flags of every panel and rule from the gates.py functions."""
    T = int(T)
    kup = np.zeros(T + 1, dtype=bool)
    for x in range(T + 1):
        h = np.zeros(T, dtype=bool)
        h[:x] = True
        kup[x] = G.kupiec_pof(*_synthetic(h), alpha=alpha, level=LEVEL).passed
    acc = np.nonzero(kup)[0]
    k_lo, k_hi = int(acc.min()), int(acc.max())
    assert np.all(kup[k_lo:k_hi + 1]), "Kupiec acceptance region is not an interval"
    kup_len = kup | (np.arange(T + 1) <= alpha * T)
    # every panel contains the Kupiec test or the traffic light; above k_hi the
    # (strict or lenient) Kupiec test rejects, and B alone is handled separately
    tl = traffic_light_bounds(T) if abs(alpha - 0.01) < 1e-12 else None
    xc = k_hi
    if tl is not None:
        xc = max(xc, tl[1] - 1)
    shape = (2, 2, xc + 1, xc + 1)
    valid = np.zeros(shape, dtype=bool)
    flags = {(p, r): np.zeros(shape, dtype=bool) for p in PANELS for r in RULES}
    for f in (0, 1):
        for l in (0, 1):
            for x in range(xc + 1):
                for n11 in range(x + 1):
                    s = build_seq(T, f, l, x, n11)
                    if s is None:
                        continue
                    valid[f, l, x, n11] = True
                    ch = G.christoffersen(*_synthetic(s), alpha=alpha, level=LEVEL)
                    d = ch.detail
                    ch_len = not ((not d["pass_ind"]) or ((not d["pass_cc"]) and x > alpha * T))
                    k_s, k_l = bool(kup[x]), bool(kup_len[x])
                    for rule, k, c in (("strict", k_s, bool(ch.passed)), ("lenient", k_l, ch_len)):
                        flags[("K", rule)][f, l, x, n11] = k
                        flags[("KC", rule)][f, l, x, n11] = k and c
                        if tl is not None:
                            b = x < tl[0] if rule == "strict" else x < tl[1]
                            flags[("B", rule)][f, l, x, n11] = b
                            flags[("KCB", rule)][f, l, x, n11] = k and c and b
    return PanelTables(T=T, alpha=alpha, xc=xc, valid=valid, flags=flags,
                       kupiec_band=(k_lo, k_hi), tl_bounds=tl)


def state_distribution(T: int, p: float, xc: int):
    """Exact law of (first, last, count, n11) for i.i.d. Bernoulli(p), count <= xc.

    Returns (P, overflow), where overflow is the probability that the count
    exceeds xc.
    """
    P = np.zeros((2, 2, xc + 1, xc + 1))
    P[0, 0, 0, 0] = 1.0 - p
    if xc >= 1:
        P[1, 1, 1, 0] = p
    over = 0.0 if xc >= 1 else p
    for _ in range(T - 1):
        N = np.zeros_like(P)
        N[:, 0] += P.sum(axis=1) * (1.0 - p)
        over += p * P[:, :, xc, :].sum()
        N[:, 1, 1:, :] += p * P[:, 0, :-1, :]
        N[:, 1, 1:, 1:] += p * P[:, 1, :-1, :-1]
        over += p * P[:, 1, :-1, -1].sum()
        P = N
    return P, over


def reject_probs(p: float, T: int, alpha: float) -> dict:
    """Exact rejection probability of every (panel, rule) at exception probability p."""
    tab = panel_tables(T, alpha)
    P, _ = state_distribution(T, p, tab.xc)
    assert P[~tab.valid].sum() < 1e-10
    out = {}
    for key, fl in tab.flags.items():
        if key[0] in ("B", "KCB") and tab.tl_bounds is None:
            continue
        if key[0] == "B":
            # traffic light alone: closed form on the binomial count
            amber, red = tab.tl_bounds
            cut = amber if key[1] == "strict" else red
            out[key] = float(stats.binom.sf(cut - 1, T, p))
            continue
        out[key] = float(1.0 - (P * fl).sum())
    return out


def reject_prob(p: float, T: int, alpha: float, panel: str, rule: str) -> float:
    return reject_probs(p, T, alpha)[(panel, rule)]


def p_grid(alpha: float, p_max: float = 0.25, n_grid: int = 300) -> np.ndarray:
    """Grid on [alpha, p_max], denser near alpha."""
    return alpha + (p_max - alpha) * np.linspace(0.0, 1.0, n_grid + 1) ** 2


def power_curves(T: int, alpha: float, grid: np.ndarray) -> dict:
    """Rejection probability of every (panel, rule) on a grid of p."""
    rows = [reject_probs(float(p), T, alpha) for p in grid]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def _invert(grid, vals, f, beta):
    below = np.nonzero(vals < beta)[0]
    if below.size == 0:
        return float(grid[0])
    i = int(below.max())
    if i == grid.size - 1:
        return None
    return float(optimize.brentq(lambda p: f(p) - beta, grid[i], grid[i + 1], xtol=1e-12))


def min_detectable_p(T: int, alpha: float, panel: str, rule: str, beta: float,
                     grid: np.ndarray | None = None, curve: np.ndarray | None = None
                     ) -> float | None:
    """p*_beta: smallest p > alpha with rejection probability >= beta at every p' >= p.

    The grid locates the last p with power below beta; brentq then finds the
    crossing exactly. Returns None if the panel never reaches beta on the grid.
    """
    if grid is None:
        grid = p_grid(alpha)
    if curve is None:
        curve = power_curves(T, alpha, grid)[(panel, rule)]
    f = lambda p: reject_prob(p, T, alpha, panel, rule)
    return _invert(grid, curve, f, beta)


# ---------------------------------------------------------------------------
# tails: quantile ratios
# ---------------------------------------------------------------------------

def upper_quantile(p, tail: str | float = "gaussian"):
    """Upper p-quantile of the standardized loss: 'gaussian' or Student-t with nu dof.

    The scale cancels in every ratio used here, so no variance scaling is applied.
    """
    if tail == "gaussian":
        return stats.norm.isf(p)
    return stats.t.isf(p, float(tail))


def tail_prob(q, tail: str | float = "gaussian"):
    if tail == "gaussian":
        return stats.norm.sf(q)
    return stats.t.sf(q, float(tail))


def mdu_from_p(p_star: float, alpha: float, tail: str | float = "gaussian") -> float:
    """u* = 1 - q(p*) / q(alpha)."""
    return float(1.0 - upper_quantile(p_star, tail) / upper_quantile(alpha, tail))


def p_from_understatement(u: float, alpha: float, tail: str | float = "gaussian") -> float:
    """Exception probability of a VaR understated by the fraction u of the true quantile."""
    return float(tail_prob((1.0 - u) * upper_quantile(alpha, tail), tail))


# ---------------------------------------------------------------------------
# expected backtesting multiplier (N7)
# ---------------------------------------------------------------------------

def expected_multiplier(p: float, T: int = 250) -> float:
    """E[m(X)] for X ~ Bin(T, p) under the MAR32 zone multipliers (T = 250)."""
    x = np.arange(T + 1)
    pmf = stats.binom.pmf(x, T, p)
    m = np.array([MAR_MULTIPLIER.get(int(k), MAR_MULTIPLIER_RED) for k in x])
    return float(pmf @ m)


def expected_capital_factor(u: float, alpha: float = 0.01, tail="gaussian", T: int = 250) -> float:
    """(1 - u) E[m(X_u)] / E[m(X_0)]: capital of a constant understatement u relative
    to the correct forecast, when the capital measure is understated in the same
    proportion and the multiplier follows the exception count of one window."""
    pu = p_from_understatement(u, alpha, tail)
    return float((1.0 - u) * expected_multiplier(pu, T) / expected_multiplier(alpha, T))


# ---------------------------------------------------------------------------
# round 3: size-adjusted panels, the power envelope and the desk-level rule
# ---------------------------------------------------------------------------
#
# Size adjustment. Each component test of a panel is given a p-value-type
# statistic on the sufficient-statistic grid: the chi-square tail probability of
# the Kupiec LR statistic and of the Christoffersen LR_ind and LR_cc statistics
# (all taken from the gates.py results), and the binomial tail P(X > x) for the
# traffic light (the Basel amber rule at level g is P(X <= x) >= 1 - g). At a
# common nominal level g a panel rejects when the smallest component statistic is
# below g, so the panel statistic is m = min over its components (a min-p or
# Tippett combination). With direction "upper" a component that rejects for too
# few exceptions is ignored (its statistic is set to 1 when x <= alpha T, except
# the independence test). The size-adjusted test rejects when m < c and with
# probability w when m = c, with c and w set so that the size is exactly the
# target; this randomization is the textbook device for equal sizes of discrete
# tests (Lehmann and Romano, 2005, Section 3.4). Nothing here is a new test.
#
# Power envelope. Under i.i.d. Bernoulli(p) exceptions the count x is sufficient
# and the likelihood ratio is increasing in x, so the randomized one-sided
# binomial test is uniformly most powerful against p > alpha (Karlin-Rubin). No
# test that reads only the exception sequence has more power at the same size.

SIZE_PANELS = ("K", "KC", "KCB", "B")
DIRECTIONS = ("two", "upper")


@dataclass
class MinPTables:
    T: int
    alpha: float
    xc: int
    valid: np.ndarray
    m: dict          # (panel, direction) -> panel statistic on the grid (1.0 where invalid)


@lru_cache(maxsize=None)
def minp_tables(T: int, alpha: float, xc: int) -> MinPTables:
    """Panel statistics m on the (first, last, count, n11) grid up to count xc."""
    T = int(T)
    shape = (2, 2, xc + 1, xc + 1)
    valid = np.zeros(shape, dtype=bool)
    comp = {k: np.ones(shape) for k in ("pof", "ind", "cc", "tl")}
    tl_ok = abs(alpha - 0.01) < 1e-12
    for f in (0, 1):
        for l in (0, 1):
            for x in range(xc + 1):
                for n11 in range(x + 1):
                    s = build_seq(T, f, l, x, n11)
                    if s is None:
                        continue
                    valid[f, l, x, n11] = True
                    k = G.kupiec_pof(*_synthetic(s), alpha=alpha, level=LEVEL)
                    d = G.christoffersen(*_synthetic(s), alpha=alpha, level=LEVEL).detail
                    comp["pof"][f, l, x, n11] = float(stats.chi2.sf(k.stat, 1))
                    comp["ind"][f, l, x, n11] = float(stats.chi2.sf(d["lr_ind"], 1))
                    comp["cc"][f, l, x, n11] = float(stats.chi2.sf(d["lr_cc"], 2))
                    if tl_ok:
                        comp["tl"][f, l, x, n11] = float(stats.binom.sf(x, T, alpha))
    xs = np.arange(xc + 1)[None, None, :, None]
    up = xs > alpha * T
    m = {}
    for direction in DIRECTIONS:
        pof = comp["pof"] if direction == "two" else np.where(up, comp["pof"], 1.0)
        cc = comp["cc"] if direction == "two" else np.where(up, comp["cc"], 1.0)
        chris = np.minimum(comp["ind"], cc)
        m[("K", direction)] = pof
        m[("KC", direction)] = np.minimum(pof, chris)
        if tl_ok:
            m[("B", direction)] = comp["tl"]
            m[("KCB", direction)] = np.minimum(np.minimum(pof, chris), comp["tl"])
    for key in m:
        m[key] = np.where(valid, m[key], 1.0)
    return MinPTables(T=T, alpha=alpha, xc=xc, valid=valid, m=m)


def _atoms(mvals: np.ndarray, probs: np.ndarray, tol: float = 1e-12):
    """Group equal statistic values: sorted unique values and their probabilities."""
    order = np.argsort(mvals, kind="stable")
    mv, pv = mvals[order], probs[order]
    keys, mass = [], []
    for v, q in zip(mv, pv):
        if keys and abs(v - keys[-1]) <= tol * max(1.0, abs(v)):
            mass[-1] += q
        else:
            keys.append(float(v))
            mass.append(float(q))
    return np.array(keys), np.array(mass)


def size_adjusted_rule(T: int, alpha: float, panel: str, direction: str,
                       size: float = 0.05, xc: int | None = None) -> dict:
    """Critical value c and randomization weight w giving exact ``size`` at p = alpha.

    States with count above xc are treated as rejections; xc is chosen so that the
    smallest panel statistic among them is below c (checked).
    """
    if xc is None:
        xc = int(stats.binom.isf(1e-12, T, max(alpha, 0.06))) + 5
    tab = minp_tables(T, alpha, xc)
    P, over = state_distribution(T, alpha, xc)
    mv = tab.m[(panel, direction)][tab.valid]
    pv = P[tab.valid]
    keys, mass = _atoms(mv, pv)
    cum = over + np.concatenate([[0.0], np.cumsum(mass)])   # P(m < keys[i]) = cum[i]
    i = int(np.searchsorted(cum[1:], size + 1e-15, side="right"))
    # keys[i] is the boundary atom: P(m < keys[i]) <= size < P(m <= keys[i])
    c = float(keys[i])
    w = float((size - cum[i]) / mass[i])
    # states beyond xc must be rejected for sure: their Kupiec p-value is tiny
    x_hi = xc + 1
    h = np.zeros(T, dtype=bool)
    h[:x_hi] = True
    p_hi = float(stats.chi2.sf(G.kupiec_pof(*_synthetic(h), alpha=alpha).stat, 1))
    assert p_hi < c or panel == "B", "xc too small for the size-adjusted rule"
    return {"c": c, "w": w, "xc": xc, "size": float(cum[i] + w * mass[i]),
            "p_below": float(cum[i]), "p_atom": float(mass[i])}


def size_adjusted_reject_prob(p: float, T: int, alpha: float, panel: str, direction: str,
                              rule: dict) -> float:
    """Exact rejection probability of the size-adjusted panel at exception probability p."""
    tab = minp_tables(T, alpha, rule["xc"])
    P, over = state_distribution(T, p, rule["xc"])
    m = tab.m[(panel, direction)]
    c = rule["c"]
    tol = 1e-12 * max(1.0, abs(c))
    below = tab.valid & (m < c - tol)
    atom = tab.valid & (np.abs(m - c) <= tol)
    return float(over + P[below].sum() + rule["w"] * P[atom].sum())


def ump_rule(T: int, alpha: float, size: float) -> dict:
    """Randomized one-sided binomial test of exact ``size``: reject x > k, and x = k w.p. w."""
    sf = stats.binom.sf(np.arange(T + 1), T, alpha)          # P(X > k)
    k = int(np.argmax(sf <= size + 1e-15))
    w = float((size - sf[k]) / stats.binom.pmf(k, T, alpha))
    return {"k": k, "w": w}


def ump_reject_prob(p: float, T: int, rule: dict) -> float:
    return float(stats.binom.sf(rule["k"], T, p) + rule["w"] * stats.binom.pmf(rule["k"], T, p))


def invert_power(f, alpha: float, beta: float, p_max: float = 0.25, n_grid: int = 300):
    """p*_beta for a generic power function f(p): same rule as ``min_detectable_p``."""
    grid = p_grid(alpha, p_max=p_max, n_grid=n_grid)
    vals = np.array([f(float(p)) for p in grid])
    return _invert(grid, vals, f, beta)


def desk_reject_prob(u: float, T: int = 250, tail: str | float = "gaussian",
                     max99: int = 12, max975: int = 30) -> float:
    """Exact probability that a desk fails the Basel desk-level backtest.

    The desk fails with more than ``max99`` exceptions at the 99th percentile or
    more than ``max975`` at the 97.5th percentile over T days. Both VaRs are cut by
    the same fraction u of their true quantiles; each day falls in one of three
    cells (below both, between, above both), so the two counts are trinomial.
    """
    p99 = p_from_understatement(u, 0.01, tail)
    p975 = p_from_understatement(u, 0.025, tail)
    p_mid = p975 - p99
    # P(pass) = sum_{a <= max99} P(X99 = a) P(X975 - a <= max975 - a | X99 = a)
    tot = 0.0
    for a in range(max99 + 1):
        pa = stats.binom.pmf(a, T, p99)
        tot += pa * stats.binom.cdf(max975 - a, T - a, p_mid / (1.0 - p99))
    return float(1.0 - tot)


def desk_mdu(beta: float, T: int = 250, tail: str | float = "gaussian") -> float:
    """Smallest u with desk rejection probability >= beta (the curve is increasing)."""
    f = lambda u: desk_reject_prob(u, T, tail) - beta
    return float(optimize.brentq(f, 0.0, 0.95, xtol=1e-10))
