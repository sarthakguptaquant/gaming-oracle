"""Backtest gate panel for the Gaming Oracle.

Every gate is a pure function with the common signature

    Gate(stated_var, returns, alpha=0.01, level=0.95) -> GateResult

where ``stated_var`` is the STATED (reported) VaR series as a POSITIVE loss
magnitude (so an exception is ``loss_t > stated_var_t`` with
``loss_t = -returns_t``), ``returns`` are the realized simple returns, ``alpha``
is the nominal VaR tail probability (0.01 = 99% VaR), and ``level`` is the
test confidence (0.95). ``passed`` is True when the gate does NOT reject the
null that the series is well calibrated.

MATH IS PRIOR. None of these tests are novel:
- Kupiec (1995) POF unconditional-coverage LR test.       [REGULATORY]
- Christoffersen (1998) independence + conditional-coverage LR tests.  [REGULATORY]
- Basel traffic-light zones (BCBS 1996, "Supervisory framework for the use of
  backtesting").                                          [REGULATORY]
  ^ These three are the STANDARD REGULATORY PANEL (``PANEL_REGULATORY``). Every
    headline number leads with this panel.
- ``conformal_coverage`` is a NON-REGULATORY, AUTHOR-ADDED empirical
  interval-coverage check (split-conformal coverage idea, Vovk et al.) with a
  hand-set tolerance. It is NOT co-equal with the regulatory three; with the
  default tol it is tighter than every regulatory band and so binds by
  construction. Reported separately. See its docstring and ``tolerance_map``.
- A one-sided unconditional-coverage LM/score test on the exception (hit)
  sequence is the "killer-demo" gate that SHRINKS the Maximum Undetectable Lie.
  Its function is named ``oneside_uc_score`` (the old name ``du_escanciano`` is a
  deprecated alias; it once carried a fabricated citation, now removed). It is the
  score-test form of the Kupiec POF restriction, in the spirit of the higher-power
  score-based backtests of Du & Escanciano (2017, Management Science
  63(4):940-958) and the VaR/ES power literature (Christoffersen & Pelletier 2004;
  Escanciano & Olmo 2010). It is NOT their full ES tail-risk statistic, which
  needs the predictive CDF; see the gate docstring.

The Gaming Oracle's contribution is the shipped artifact + the priced metric,
not any of this math.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .ebacktest import e_process, gaussian_es_var_ratio


@dataclass
class GateResult:
    """Common return type for every gate."""

    passed: bool
    stat: float
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def exceptions(stated_var: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Exception indicator e_t = 1[ loss_t > stated_var_t ].

    ``stated_var`` is a positive loss magnitude; loss_t = -returns_t.
    """
    stated_var = np.asarray(stated_var, dtype=float)
    returns = np.asarray(returns, dtype=float)
    loss = -returns
    return (loss > stated_var).astype(int)


def _safe_log(x: float) -> float:
    return np.log(x) if x > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. Kupiec POF (unconditional coverage)
# ---------------------------------------------------------------------------

def kupiec_pof(stated_var, returns, alpha: float = 0.01, level: float = 0.95) -> GateResult:
    """Kupiec (1995) proportion-of-failures LR test.

    H0: P(exception) = alpha. LR_uc ~ chi2(1). Pass = not rejected.
    """
    e = exceptions(stated_var, returns)
    n = e.size
    x = int(e.sum())
    pi_hat = x / n if n else 0.0

    # log-likelihood under H0 (pi=alpha) and under the empirical pi_hat
    ll_null = x * _safe_log(alpha) + (n - x) * _safe_log(1 - alpha)
    if x == 0:
        ll_alt = (n - x) * _safe_log(1 - pi_hat) if pi_hat < 1 else 0.0
    elif x == n:
        ll_alt = x * _safe_log(pi_hat)
    else:
        ll_alt = x * _safe_log(pi_hat) + (n - x) * _safe_log(1 - pi_hat)

    lr = -2.0 * (ll_null - ll_alt)
    lr = max(lr, 0.0)
    crit = stats.chi2.ppf(level, df=1)
    pval = 1.0 - stats.chi2.cdf(lr, df=1)
    passed = lr <= crit
    return GateResult(
        passed=bool(passed),
        stat=float(lr),
        detail={
            "n": n,
            "exceptions": x,
            "pi_hat": pi_hat,
            "expected": alpha * n,
            "crit": float(crit),
            "pvalue": float(pval),
            "test": "kupiec_pof",
        },
    )


# ---------------------------------------------------------------------------
# 2. Christoffersen independence + conditional coverage
# ---------------------------------------------------------------------------

def _transition_counts(e: np.ndarray):
    """Return n00, n01, n10, n11 from the exception sequence."""
    prev = e[:-1]
    cur = e[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    return n00, n01, n10, n11


def christoffersen(stated_var, returns, alpha: float = 0.01, level: float = 0.95) -> GateResult:
    """Christoffersen (1998) independence (LR_ind) and conditional coverage (LR_cc).

    LR_cc = LR_uc + LR_ind. Pass = conditional-coverage test not rejected at
    ``level`` AND independence test not rejected at ``level``.
    """
    e = exceptions(stated_var, returns)
    n = e.size
    x = int(e.sum())
    n00, n01, n10, n11 = _transition_counts(e)

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n - 1) > 0 else 0.0

    # log-lik under Markov (alt) and under independence (null for LR_ind)
    ll_markov = (
        n00 * _safe_log(1 - pi01)
        + n01 * _safe_log(pi01)
        + n10 * _safe_log(1 - pi11)
        + n11 * _safe_log(pi11)
    )
    ll_indep = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    lr_ind = -2.0 * (ll_indep - ll_markov)
    lr_ind = max(lr_ind, 0.0)

    # unconditional-coverage piece
    pi_hat = x / n if n else 0.0
    ll_null = x * _safe_log(alpha) + (n - x) * _safe_log(1 - alpha)
    if x == 0:
        ll_alt = (n - x) * _safe_log(1 - pi_hat) if pi_hat < 1 else 0.0
    elif x == n:
        ll_alt = x * _safe_log(pi_hat)
    else:
        ll_alt = x * _safe_log(pi_hat) + (n - x) * _safe_log(1 - pi_hat)
    lr_uc = max(-2.0 * (ll_null - ll_alt), 0.0)

    lr_cc = lr_uc + lr_ind

    crit_ind = stats.chi2.ppf(level, df=1)
    crit_cc = stats.chi2.ppf(level, df=2)
    pass_ind = lr_ind <= crit_ind
    pass_cc = lr_cc <= crit_cc
    passed = bool(pass_ind and pass_cc)
    return GateResult(
        passed=passed,
        stat=float(lr_cc),
        detail={
            "lr_ind": float(lr_ind),
            "lr_uc": float(lr_uc),
            "lr_cc": float(lr_cc),
            "crit_ind": float(crit_ind),
            "crit_cc": float(crit_cc),
            "pass_ind": bool(pass_ind),
            "pass_cc": bool(pass_cc),
            "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
            "pvalue_cc": float(1.0 - stats.chi2.cdf(lr_cc, df=2)),
            "test": "christoffersen",
        },
    )


# ---------------------------------------------------------------------------
# 3. Basel traffic-light
# ---------------------------------------------------------------------------

def basel_traffic_light(stated_var, returns, alpha: float = 0.01, level: float = 0.95) -> GateResult:
    """Basel traffic-light over the most recent 250-day window.

    Green zone <= 4 exceptions, amber 5-9, red >= 10. Pass = green.
    """
    e = exceptions(stated_var, returns)
    window = e[-250:] if e.size > 250 else e
    x = int(window.sum())
    if x <= 4:
        zone = "green"
    elif x <= 9:
        zone = "amber"
    else:
        zone = "red"
    passed = zone == "green"
    return GateResult(
        passed=bool(passed),
        stat=float(x),
        detail={
            "window": int(window.size),
            "exceptions": x,
            "zone": zone,
            "green_max": 4,
            "test": "basel_traffic_light",
        },
    )


# ---------------------------------------------------------------------------
# 4. Conformal / interval-coverage
# ---------------------------------------------------------------------------

def conformal_coverage(stated_var, returns, alpha: float = 0.01, level: float = 0.95,
                       tol: float = 0.005) -> GateResult:
    """NON-REGULATORY, AUTHOR-ADDED empirical interval-coverage gate.

    *** THIS IS NOT A REGULATORY BACKTEST. *** Kupiec, Christoffersen and the
    Basel traffic-light are the gates a real validator runs and a supervisor
    recognizes. This gate is an interval-coverage check the AUTHORS added, with a
    hand-set tolerance ``tol``. It is NOT co-equal with the regulatory three and
    must never be reported as such.

    It passes when the empirical coverage 1 - (exceptions/n) is within ``tol`` of
    the nominal 1 - alpha. The default ``tol=0.005`` on a 99% interval is a
    BAND OF +-1.25 EXCEPTIONS over 250 days, which is structurally TIGHTER than
    every real regulatory gate (Basel green allows up to 4 exceptions; Kupiec's
    non-rejection band is wider still). So with the default tolerance this gate
    BINDS BY CONSTRUCTION whenever it is in the panel: it is the tightest constraint
    not because interval coverage is the load-bearing risk test, but because its
    tolerance was set tighter than the regulatory bands. That is a TUNING CHOICE,
    not a property of the data. The honest, portable finding is therefore the
    TOLERANCE MAP (see ``gaming_oracle.tolerance_map``): which gate is binding is a
    function of ``tol``, and at any regulatorily-plausible tolerance the binding
    gate is one of the regulatory three. The 'split-conformal coverage' framing
    (Vovk et al.) motivates the form but does not make it a regulatory test.
    """
    e = exceptions(stated_var, returns)
    n = e.size
    emp_cov = 1.0 - (e.sum() / n) if n else 0.0
    nominal = 1.0 - alpha
    gap = emp_cov - nominal
    passed = abs(gap) <= tol
    return GateResult(
        passed=bool(passed),
        stat=float(gap),
        detail={
            "empirical_coverage": float(emp_cov),
            "nominal_coverage": float(nominal),
            "tol": tol,
            "gap": float(gap),
            "test": "conformal_coverage",
        },
    )


# ---------------------------------------------------------------------------
# 5. One-sided unconditional-coverage LM/score test  (killer demo)
#    (the score-test form of Kupiec POF; in the spirit of Du & Escanciano 2017)
# ---------------------------------------------------------------------------

def oneside_uc_score(stated_var, returns, alpha: float = 0.01, level: float = 0.95) -> GateResult:
    """One-sided unconditional-coverage LM/score test on the hit sequence.

    Uses the exception (hit) sequence h_t = 1[loss_t > VaR_t]. The standardized
    statistic for unconditional coverage is

        U_uc = sqrt(n) * (mean(h) - alpha) / sqrt(alpha*(1-alpha))

    which is asymptotically N(0,1). One-sided (against UNDER-coverage, i.e. too
    many exceptions): reject when U_uc > z_level. Pass = not rejected.

    HONEST PROVENANCE (and why this function was RENAMED). An earlier draft named
    this ``du_escanciano`` and once carried a fabricated "Du-Escanciano 2019, J.
    Banking & Finance" citation that does not exist. To remove that risk the
    function is now named ``oneside_uc_score`` to reflect exactly what it computes:
    a ONE-SIDED unconditional-coverage score / LM test "in the spirit of" the
    higher-power score-based backtest literature, NOT the full Du-Escanciano (2017)
    Expected-Shortfall statistic. It is the score-test / LM form of the Kupiec
    (1995) POF restriction, made one-sided so it has higher power against the
    understatement direction (which is why adding it shrinks the exhibited lie). It
    is in the spirit of the higher-power score-based backtests of Du & Escanciano
    (2017, Management Science 63(4):940-958) and the VaR/ES backtest-power
    literature (Christoffersen & Pelletier 2004; Escanciano & Olmo 2010), but it is
    NOT their ES tail-risk statistic, which requires the predictive CDF / cumulative
    violations. We use the hit-based one-sided UC form so it runs on the same
    stated_var/returns inputs as the other gates. ``du_escanciano`` is kept below
    ONLY as a deprecated backward-compatibility alias.
    """
    e = exceptions(stated_var, returns).astype(float)
    n = e.size
    h_bar = e.mean() if n else 0.0
    denom = np.sqrt(alpha * (1 - alpha))
    u = np.sqrt(n) * (h_bar - alpha) / denom if denom > 0 else 0.0
    z = stats.norm.ppf(level)  # one-sided critical value
    passed = u <= z
    pval = 1.0 - stats.norm.cdf(u)
    return GateResult(
        passed=bool(passed),
        stat=float(u),
        detail={
            "h_bar": float(h_bar),
            "alpha": alpha,
            "z_crit": float(z),
            "pvalue_onesided": float(pval),
            "n": n,
            "test": "oneside_uc_score",
        },
    )


#: Deprecated backward-compatibility alias. The earlier name ``du_escanciano``
#: overstated provenance (and once carried a fabricated citation); use
#: ``oneside_uc_score``. Kept so old imports and the MILP/CLI keep working.
du_escanciano = oneside_uc_score


# ---------------------------------------------------------------------------
# 6. e-backtest (Wang-Wang-Ziegel) — the DEFENSE gate (gap #7)
# ---------------------------------------------------------------------------

def e_backtest(stated_var, returns, alpha: float = 0.01, level: float = 0.95,
               stated_es=None, lam=None, lam_max: float = 0.5) -> GateResult:
    """Anytime-valid e-process backtest for Expected Shortfall.

    Wang, Wang & Ziegel ("E-backtesting", Management Science 72(6):4952-4973,
    online 2025, DOI 10.1287/mnsc.2023.01659; arXiv:2209.00991, 2022). Unlike
    every other gate in the panel, this one does NOT reduce to the binary
    exception sequence: it reads the MAGNITUDE of each breach via the ES
    e-statistic  e_t = (loss_t - z_t)_+ / [(1-p)(r_t - z_t)]  with z_t = stated
    VaR, r_t = stated ES, p = 1 - alpha. The e-values are accumulated into the
    multiplicative e-process M_t = prod (1 - lam_s + lam_s e_s); H0 ("risk is not
    under-forecast") is rejected the first time M_t >= 1/(1-level). Because (M_t)
    is a non-negative supermartingale under H0, Ville's inequality controls the
    size at ``1 - level`` at any stopping time (anytime-valid).

    To fit the common Gate(stated_var, returns) signature, the stated ES is, by
    default, the stated VaR scaled by the consistent Gaussian ES/VaR ratio: the
    SAME lie that shrinks stated VaR shrinks stated ES proportionally, so the
    Oracle attacks ES exactly as it attacks VaR (no free pass). Pass a ``stated_es``
    array to override. PASS = not rejected (the e-process never crosses 1/alpha_test).
    """
    returns = np.asarray(returns, dtype=float)
    stated_var = np.asarray(stated_var, dtype=float)
    loss = -returns
    if stated_es is None:
        stated_es = gaussian_es_var_ratio(alpha) * stated_var
    res = e_process(loss, stated_var, np.asarray(stated_es, dtype=float),
                    alpha=alpha, level=level, lam=lam, lam_max=lam_max)
    passed = not res.rejected
    return GateResult(
        passed=bool(passed),
        stat=float(res.max_eprocess),
        detail={
            "max_eprocess": float(res.max_eprocess),
            "final_eprocess": float(res.final_eprocess),
            "threshold": float(res.threshold),
            "crossing_index": int(res.crossing_index),
            "rejected": bool(res.rejected),
            "n": int(res.n),
            "es_var_ratio": float(gaussian_es_var_ratio(alpha)),
            "test": "e_backtest",
        },
    )


# ---------------------------------------------------------------------------
# panel assembly
# ---------------------------------------------------------------------------

#: The STANDARD REGULATORY PANEL: the three gates a real risk validator runs and
#: a supervisor recognizes (Kupiec POF, Christoffersen ind/cc, Basel traffic-light).
#: Every headline number leads with THIS panel. ``conformal_coverage`` is NOT here:
#: it is a non-regulatory, author-added interval-coverage gate (see its docstring),
#: reported separately, never as co-equal with the regulatory three.
PANEL_REGULATORY = [kupiec_pof, christoffersen, basel_traffic_light]

#: The author-augmented panel: the regulatory three PLUS the non-regulatory
#: conformal-coverage gate. Retained as ``PANEL`` for backward compatibility, but
#: it is explicitly the regulatory panel plus one author-added gate, not "the panel".
PANEL = PANEL_REGULATORY + [conformal_coverage]

#: The killer-demo panel: the augmented panel PLUS the one-sided UC score test.
PANEL_WITH_DE = PANEL + [oneside_uc_score]

#: The DEFENDED panel (Cycle 4): the augmented panel PLUS the Wang-Wang-Ziegel
#: e-backtest. This is the attack-vs-defense comparison: PANEL (gameable) vs
#: PANEL_WITH_EBT (the e-process gate added).
PANEL_WITH_EBT = PANEL + [e_backtest]


def run_panel(stated_var, returns, panel=None, alpha: float = 0.01,
              level: float = 0.95) -> dict:
    """Run every gate in ``panel`` and report joint pass + per-gate results."""
    if panel is None:
        panel = PANEL
    results = {}
    for gate in panel:
        res = gate(stated_var, returns, alpha=alpha, level=level)
        results[gate.__name__] = res
    passed_all = all(r.passed for r in results.values())
    return {"passed_all": bool(passed_all), "gates": results}
