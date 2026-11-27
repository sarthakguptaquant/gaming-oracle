"""Adaptive betting processes for the ES e-backtest (GREE, GREL, GREM).

Wang, Wang and Ziegel ("E-backtesting", Management Science 72(6):4952-4973,
DOI 10.1287/mnsc.2023.01659; arXiv:2209.00991, Section 5.1) define three
data-driven betting processes for the e-process M_t = prod_s (1 - l_s + l_s e_s),
each with l_t restricted to [0, gamma], gamma = 1/2:

* GREE  l_t = argmax_l  sum_{s<t} log(1 - l + l e(L_s, r_s, z_s))
        (empirical distribution of past e-values);
* GREL  l_t = argmax_l  sum_{s<t} log(1 - l + l e(L_s, r_t, z_t))
        (past losses evaluated at the CURRENT forecast (r_t, z_t));
* GREM  M_t = (M_t^GREE + M_t^GREL) / 2, the method the authors recommend as the
        default.

All three are predictable (l_t uses information to t-1 and the forecast for t),
so each M_t is a nonnegative supermartingale under the null and Ville's
inequality gives an anytime-valid test that rejects when M_t reaches
1 / (1 - level). The argmax is taken over a fixed grid of l values in [0, 1/2]
(step 0.01 by default); any predictable choice keeps validity, the grid only
affects how close l_t is to the exact maximiser. With no history l_1 = 0.

This module is a batch (many paths at once) implementation used for the power
study; ``eprocess_adaptive`` is a plain single-path loop used to test it.
The math is prior work; nothing here is new.
"""

from __future__ import annotations

import numpy as np

from .ebacktest import es_e_statistic

LAMBDA_GRID = np.round(np.arange(0.0, 0.5001, 0.01), 2)


def _argmax_rows(score: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return grid[np.argmax(score, axis=1)]


def eprocess_adaptive_batch(loss, var, es, alpha: float = 0.01, level: float = 0.95,
                            grid=LAMBDA_GRID):
    """GREE, GREL and GREM e-processes for each row (path).

    ``loss``, ``var`` (stated VaR z_t) and ``es`` (stated ES r_t) have shape
    (n_paths, n). Returns a dict with, for each method, the first crossing index
    of the threshold (-1 if none) and the running maximum of M_t.
    """
    loss = np.atleast_2d(np.asarray(loss, dtype=float))
    var = np.broadcast_to(np.asarray(var, dtype=float), loss.shape)
    es = np.broadcast_to(np.asarray(es, dtype=float), loss.shape)
    n_paths, n = loss.shape
    grid = np.asarray(grid, dtype=float)
    log1m = np.log1p(-grid)                      # log(1 - l) for e = 0
    thr = 1.0 / (1.0 - level)
    e_all = es_e_statistic(loss, var, es, alpha)  # e(L_t, r_t, z_t)

    score_ee = np.zeros((n_paths, grid.size))     # GREE running log-growth per l
    m = {"GREE": np.ones(n_paths), "GREL": np.ones(n_paths)}
    mx = {k: np.ones(n_paths) for k in ("GREE", "GREL", "GREM")}
    cross = {k: np.full(n_paths, -1, dtype=np.int64) for k in ("GREE", "GREL", "GREM")}
    rows = np.arange(n_paths)
    denom_scale = alpha
    for t in range(n):
        # ---- betting fractions for day t (predictable) ----
        lam_ee = _argmax_rows(score_ee, grid) if t > 0 else np.zeros(n_paths)
        if t > 0:
            z_t = var[:, t]
            gap = es[:, t] - z_t
            past = loss[:, :t]
            over = past > z_t[:, None]
            r_idx, c_idx = np.nonzero(over)
            score_el = np.broadcast_to(t * log1m, (n_paths, grid.size)).copy()
            if r_idx.size:
                safe_gap = np.where(gap > 1e-15, gap, np.nan)[r_idx]
                ev = (past[r_idx, c_idx] - z_t[r_idx]) / (denom_scale * safe_gap)
                ev = np.where(np.isnan(ev), 1e6, ev)
                contrib = np.log1p(grid[None, :] * (ev[:, None] - 1.0)) - log1m[None, :]
                np.add.at(score_el, r_idx, contrib)
            lam_el = _argmax_rows(score_el, grid)
        else:
            lam_el = np.zeros(n_paths)
        # ---- update the three e-processes with today's e-value ----
        e_t = e_all[:, t]
        m["GREE"] = np.maximum(m["GREE"] * (1.0 - lam_ee + lam_ee * e_t), 0.0)
        m["GREL"] = np.maximum(m["GREL"] * (1.0 - lam_el + lam_el * e_t), 0.0)
        m_grem = 0.5 * (m["GREE"] + m["GREL"])
        for k, val in (("GREE", m["GREE"]), ("GREL", m["GREL"]), ("GREM", m_grem)):
            np.maximum(mx[k], val, out=mx[k])
            new = (cross[k] < 0) & (val >= thr)
            cross[k][new] = t
        # ---- GREE history ----
        score_ee += np.log1p(grid[None, :] * (e_t[:, None] - 1.0))
    del rows
    return {k: {"cross": cross[k], "max": mx[k]} for k in ("GREE", "GREL", "GREM")}


def eprocess_adaptive(loss, var, es, alpha: float = 0.01, level: float = 0.95,
                      grid=LAMBDA_GRID) -> dict:
    """Single-path reference implementation (plain loops) of GREE, GREL, GREM."""
    loss = np.asarray(loss, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = loss.size
    thr = 1.0 / (1.0 - level)
    e = es_e_statistic(loss, var, es, alpha)
    m_ee = m_el = 1.0
    out = {k: {"cross": -1, "max": 1.0} for k in ("GREE", "GREL", "GREM")}
    for t in range(n):
        if t == 0:
            l_ee = l_el = 0.0
        else:
            s_ee = [np.sum(np.log1p(l * (e[:t] - 1.0))) for l in grid]
            l_ee = float(grid[int(np.argmax(s_ee))])
            e_cur = es_e_statistic(loss[:t], np.full(t, var[t]), np.full(t, es[t]), alpha)
            s_el = [np.sum(np.log1p(l * (e_cur - 1.0))) for l in grid]
            l_el = float(grid[int(np.argmax(s_el))])
        m_ee = max(m_ee * (1.0 - l_ee + l_ee * e[t]), 0.0)
        m_el = max(m_el * (1.0 - l_el + l_el * e[t]), 0.0)
        for k, val in (("GREE", m_ee), ("GREL", m_el), ("GREM", 0.5 * (m_ee + m_el))):
            out[k]["max"] = max(out[k]["max"], val)
            if out[k]["cross"] < 0 and val >= thr:
                out[k]["cross"] = t
    return out
