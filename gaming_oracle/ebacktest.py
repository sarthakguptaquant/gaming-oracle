"""Anytime-valid e-process backtest for Expected Shortfall (the DEFENSE, gap #7).

Every gate in the Cycle 1-3 panel (Kupiec, Christoffersen, Basel, conformal,
Du-Escanciano) depends on the stated risk series ONLY through the binary
exception sequence  e_t = 1[ loss_t > V_t ].  That is exactly why the Oracle can
game them: it pushes stated VaR to ~0 on the days it concedes as exceptions and
to max(loss_t, 0) on the rest, so an unbounded amount of tail MAGNITUDE is hidden
while the exception COUNT stays inside every gate's band. A count-only panel
cannot see how deep each breach is.

The e-backtest of Wang, Wang & Ziegel closes that hole. It is an
Expected-Shortfall backtest, so it is sensitive to breach MAGNITUDE, not just the
breach count, and it is anytime-valid: the test statistic is a non-negative
supermartingale under the null, so Ville's inequality controls the type-I error
at any stopping time.

CITATION (math is prior, never claimed as ours):
    Qiuqi Wang, Ruodu Wang, Johanna Ziegel. "E-backtesting." Management Science
    72(6):4952-4973 (published online 2025-09-23, DOI 10.1287/mnsc.2023.01659;
    print issue June 2026). Preprint arXiv:2209.00991 / SSRN 4206997 (2022). The
    per-period ES e-statistic, the multiplicative e-process, and the
    Ville/supermartingale anytime-validity argument are all theirs.

------------------------------------------------------------------------------
The construction (Wang-Wang-Ziegel)
------------------------------------------------------------------------------
Null  H0: the realized loss is NOT under-forecast, i.e. for each t the stated ES
forecast r_t >= ES_p(L_t | F_{t-1}) and the stated VaR forecast z_t = VaR_p(L_t).
Here p = 1 - alpha is the ES/VaR level (p = 0.99 for a 99% / alpha=0.01 tail).

Per-period ES e-statistic (their eq.; one-sided, monotone decreasing in r):

    e_t = (x_t - z_t)_+ / [ (1 - p) * (r_t - z_t) ]

with x_t the realized loss (= -return_t), z_t the stated VaR, r_t the stated ES,
and (.)_+ = max(., 0). Under H0, E[e_t | F_{t-1}] <= 1. The key property: e_t
grows when a breach is DEEP relative to the stated ES-minus-VaR gap, which is
precisely what the count-gamer manufactures by collapsing z_t toward 0.

E-process by GREE-style betting (their M_t):

    M_t(lambda) = prod_{s<=t} ( 1 - lambda_s + lambda_s * e_s ),   M_0 = 1,

with betting fraction lambda_s in [0, 1] predictable (it may depend only on info
up to s-1). (M_t) is a non-negative supermartingale under H0 with M_0 = 1, so by
Ville's inequality  P( sup_t M_t >= 1/alpha ) <= alpha.

Rejection (anytime-valid): reject H0 the first time M_t >= 1/alpha. The gate
PASSES (does not reject) iff  max_t M_t < 1/alpha.

Betting fraction. Two predictable strategies, both anytime-valid:

  * MIXTURE (default): a method-of-mixtures e-process. Run one multiplicative
    e-process per lambda on a fixed grid lambda in {grid} and AVERAGE them with
    fixed prior weights. A convex average of non-negative supermartingales is a
    non-negative supermartingale, so Ville still applies and the size is exactly
    controlled. The grid + prior are fixed in advance (not data-driven), so the
    construction is honestly predictable, and a single very large e-value on a
    deep breach is caught by the larger-lambda components even if it is one spike
    among many zeros. This is the robust default that defeats the count-gamer's
    "few huge breaches" lie.

  * GREE plug-in (optional): pick lambda_s to approximately maximise expected
    log-growth using only PAST e-values, clipped to [0, lambda_max]. Predictable,
    but it bootstraps slowly when breaches are rare (the running mean of mostly
    zero e-values stays below 1, so it under-bets) — which is why the mixture is
    the default. A constant-lambda variant is available for the known-answer tests.

This module is pure and depends only on numpy. The gate wrapper in ``gates.py``
adapts it to the common Gate(stated_var, returns, alpha, level) signature by
deriving the stated ES from the stated VaR through the consistent Gaussian
ES/VaR ratio (so the SAME uniform/per-point lie that shrinks stated VaR also
shrinks the stated ES, which is the honest way to let the Oracle attack ES too).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ES/VaR ratio for a Gaussian tail: ES_p / VaR_p = phi(z_alpha) / (alpha * z_alpha).
def gaussian_es_var_ratio(alpha: float) -> float:
    """ES-to-VaR multiplier under a Gaussian tail at tail prob ``alpha``.

    VaR multiple = z = -Phi^{-1}(alpha); ES multiple = phi(Phi^{-1}(alpha))/alpha.
    The ratio is >1 (ES is deeper in the tail than VaR). For alpha=0.01 it is
    ~1.1457. A stated ES consistent with a stated VaR is ratio * stated_var.
    """
    q = stats.norm.ppf(alpha)              # negative quantile
    z = -q                                  # positive VaR multiple
    es_mult = stats.norm.pdf(q) / alpha     # positive ES multiple
    return float(es_mult / z)


def es_e_statistic(loss: np.ndarray, stated_var: np.ndarray, stated_es: np.ndarray,
                   alpha: float) -> np.ndarray:
    """Per-period Wang-Wang-Ziegel ES e-statistic e_t.

        e_t = (loss_t - z_t)_+ / [ (1 - p) * (r_t - z_t) ],   p = 1 - alpha.

    z_t = stated_var (VaR forecast), r_t = stated_es (ES forecast). Under H0,
    E[e_t] <= 1. Guards: r_t - z_t must be > 0 (ES strictly deeper than VaR); a
    non-positive gap is treated as a maximally informative breach so the gamer
    cannot dodge the test by collapsing the ES-VaR gap to zero.
    """
    loss = np.asarray(loss, dtype=float)
    z = np.asarray(stated_var, dtype=float)
    r = np.asarray(stated_es, dtype=float)
    p = 1.0 - alpha
    num = np.maximum(loss - z, 0.0)
    gap = r - z
    # where the ES-VaR gap has collapsed (gap<=0) the denominator is degenerate;
    # any actual breach there is infinitely surprising under a coherent ES null,
    # so we cap e_t at a large finite value rather than let the gamer exploit a
    # zero gap. No breach there (num==0) gives e_t = 0 as usual.
    safe_gap = np.where(gap > 1e-15, gap, np.nan)
    e = num / ((1.0 - p) * safe_gap)
    # gap collapsed AND a breach happened -> maximally informative
    collapsed_breach = (gap <= 1e-15) & (num > 0)
    e = np.where(np.isnan(e), 0.0, e)
    e = np.where(collapsed_breach, 1e6, e)
    return e


def _predictable_lambda(e_hist: list[float], lam_max: float = 0.5) -> float:
    """GREE/GRO-style predictable betting fraction from PAST e-values only.

    Approximate the log-optimal Kelly fraction lambda that maximises
    E[log(1 - lambda + lambda * e)] using the empirical mean of the e-values seen
    so far (a plug-in for E[e]). With mbar = mean(past e), the first-order
    interior optimum of log-growth is lambda* ~ (mbar - 1) / (mbar - 1 + var-ish);
    we use the stable closed form lambda* = clip((mbar - 1)/ (mbar), 0, lam_max),
    which is 0 when the past gives no evidence (mbar<=1) and grows toward lam_max
    as evidence of under-forecasting accumulates. Predictable by construction (it
    uses only e_1..e_{s-1}), so anytime-validity holds.
    """
    if not e_hist:
        return 0.0
    mbar = float(np.mean(e_hist))
    if mbar <= 1.0:
        return 0.0
    lam = (mbar - 1.0) / mbar
    return float(min(max(lam, 0.0), lam_max))


#: Fixed lambda grid + uniform prior for the method-of-mixtures e-process. Fixed
#: in advance (data-independent) so the mixture stays predictable / anytime-valid.
MIXTURE_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)


@dataclass
class EProcessResult:
    rejected: bool          # did the e-process cross 1/alpha at any time
    max_eprocess: float     # sup_t M_t
    threshold: float        # 1/alpha
    final_eprocess: float   # M_T
    crossing_index: int     # first t with M_t >= 1/alpha, or -1
    n: int
    method: str = "mixture"


def e_process(loss: np.ndarray, stated_var: np.ndarray, stated_es: np.ndarray,
              alpha: float, level: float = 0.95, lam: float | None = None,
              lam_max: float = 0.5, grid=MIXTURE_GRID) -> EProcessResult:
    """Build the ES e-process and apply the anytime-valid rule.

    Reject H0 (the series IS under-forecasting risk) the first time
    M_t >= 1/alpha_test, where alpha_test = 1 - level is the test size (so a 95%
    test rejects at M_t >= 20).

    Betting strategy:
      * ``lam=None`` (default) -> METHOD OF MIXTURES over the fixed ``grid``, with
        a uniform prior. M_t = mean_k prod_{s<=t}(1 - lam_k + lam_k e_s). A convex
        average of supermartingales is a supermartingale, so Ville holds and the
        size is controlled at ``1 - level``. Robust to rare-but-deep breaches.
      * ``lam`` a float -> a single constant-fraction e-process (known-answer tests).
      * ``lam='gree'`` -> the predictable GREE plug-in from past e-values.
    """
    loss = np.asarray(loss, dtype=float)
    e = es_e_statistic(loss, stated_var, stated_es, alpha)
    n = e.size
    alpha_test = 1.0 - level
    threshold = 1.0 / alpha_test if alpha_test > 0 else np.inf

    if lam == "gree":
        # single predictable plug-in e-process
        m = 1.0
        max_m = 1.0
        crossing = -1
        e_hist: list[float] = []
        for t in range(n):
            lam_t = _predictable_lambda(e_hist, lam_max=lam_max)
            m = max(m * (1.0 - lam_t + lam_t * e[t]), 0.0)
            max_m = max(max_m, m)
            if crossing < 0 and m >= threshold:
                crossing = t
            e_hist.append(float(e[t]))
        return EProcessResult(bool(max_m >= threshold), float(max_m),
                              float(threshold), float(m), int(crossing), int(n),
                              method="gree")

    if lam is not None:
        # single constant-fraction e-process
        lams = np.array([float(lam)])
    else:
        lams = np.asarray(grid, dtype=float)

    k = lams.size
    comp = np.ones(k)                      # per-lambda capital
    weights = np.full(k, 1.0 / k)          # uniform prior over the grid
    m = 1.0
    max_m = 1.0
    crossing = -1
    for t in range(n):
        comp = comp * (1.0 - lams + lams * e[t])
        comp = np.maximum(comp, 0.0)
        m = float(np.dot(weights, comp))   # mixture e-process value at t
        max_m = max(max_m, m)
        if crossing < 0 and m >= threshold:
            crossing = t

    return EProcessResult(
        rejected=bool(max_m >= threshold),
        max_eprocess=float(max_m),
        threshold=float(threshold),
        final_eprocess=float(m),
        crossing_index=int(crossing),
        n=int(n),
        method="mixture" if lam is None else "constant",
    )
