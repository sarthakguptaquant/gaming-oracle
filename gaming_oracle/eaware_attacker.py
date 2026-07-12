"""e-process-AWARE attacker (Cycle 7): the honest defense-residual.

WHAT WAS WRONG WITH THE OLD CYCLE-4 RESULT
------------------------------------------
Cycle 4 reported that adding the Wang-Wang-Ziegel e-process ES backtest "voids the
attack". But the only attacker it beat was a STRAWMAN: the count-reduction MILP that
sets stated VaR to ~0 on every conceded breach. That gamer drives the per-period ES
e-statistic to enormous values, so of course the e-process catches it. Worse, part
of that ~1e10 e-process value came from a HARDCODED cap ``e_t = 1e6`` applied
whenever the gamer collapses the ES-VaR gap to zero (see ``ebacktest.es_e_statistic``
``collapsed_breach``). A defense that only beats a self-destructing strawman, partly
via a hardcoded cap, is not a real defense result.

WHAT THIS MODULE DOES INSTEAD
-----------------------------
It builds an attacker that is AWARE of the e-process gate and plays against it:

1. MAGNITUDE BUDGET. The attacker does NOT collapse stated VaR to 0 on breach days.
   It under-reserves by at most a fraction ``budget`` of the honest reference at any
   point (stated_var_t >= (1 - budget) * true_var_t). So every breach is shallow
   relative to the honest tail, keeping each per-period e-value small. This is the
   realistic adversary: hide as much as possible while staying within a plausible
   magnitude band, not the suicidal zero-VaR gamer.

2. e-PROCESS AWARENESS. The attacker simulates the e-process forward as it chooses
   the per-point understatement, and pulls back (raises stated VaR / stated ES) on
   exactly the days that would push the running e-process toward the 1/alpha_test
   threshold. It maximizes hidden risk subject to KEEPING the e-process below the
   threshold at every step — i.e. it optimizes against the very gate meant to stop
   it.

3. INDEPENDENT ES. Critically, the attacker states ES SEPARATELY from VaR. In the
   shipped panel the stated ES is a FIXED Gaussian multiple of stated VaR, so "you
   cannot dodge ES" is true by construction. Here the attacker is free to inflate
   the stated ES above that multiple (within the budget) precisely to shrink each
   e-value's denominator gap ``(r_t - z_t)`` from the other side — the honest test
   of whether the e-process catches an INDEPENDENTLY-stated ES.

THE HONEST DEFENSE RESULT is then: how much risk does this e-aware, budget-bounded,
independent-ES attacker STILL hide while keeping the e-process green? That residual
(in bps), NOT "the strawman is voided", is the real defense number.

THE CAP IS DISCLOSED. ``es_e_statistic`` caps a collapsed-gap breach at ``e_t=1e6``.
This attacker never collapses the gap (it inflates ES), so it does not rely on or
trigger that cap; we report the max per-period e-value it produces so the reader can
confirm the residual is not an artifact of the cap.

Math is prior (Wang-Wang-Ziegel 2025 for the e-process; Kupiec/Christoffersen/Basel
for the panel). The contribution is the artifact: an adversary that respects the
e-process and the honest residual it leaves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gates as gates_mod
from .ebacktest import MIXTURE_GRID, es_e_statistic, gaussian_es_var_ratio
from .report import price


@dataclass
class EAwareResult:
    stated_var: np.ndarray
    stated_es: np.ndarray
    true_var: np.ndarray
    returns: np.ndarray
    budget: float                       # max fractional under-reservation
    es_independent: bool                # was ES stated independently of VaR
    mul_bps: float                      # hidden risk that STILL passes (the residual)
    mul_usd: float
    extra_exceptions_per_year: float
    eprocess_max: float                 # sup_t M_t under the chosen series
    eprocess_threshold: float
    eprocess_passed: bool               # did it stay below threshold (green)
    max_per_period_e: float             # largest single e_t (cap-disclosure check)
    panel_passed_all: bool              # full augmented+ebt panel green
    gate_results: dict


def _mixture_eprocess_step(comp: np.ndarray, lams: np.ndarray, weights: np.ndarray,
                           e_t: float):
    """Advance the per-lambda mixture capital by one e-value; return (comp, M)."""
    comp = np.maximum(comp * (1.0 - lams + lams * e_t), 0.0)
    return comp, float(np.dot(weights, comp))


def solve_eaware(
    true_var,
    returns,
    budget: float = 0.10,
    es_independent: bool = True,
    es_inflate_max: float = 0.40,
    alpha: float = 0.01,
    level: float = 0.95,
    grid=MIXTURE_GRID,
    safety: float = 0.97,
    notional: float = 1_000_000_000.0,
    panel=None,
) -> EAwareResult:
    """Construct an e-process-aware, magnitude-budgeted, independent-ES attacker.

    At each day the attacker would like to under-reserve maximally (stated VaR
    pushed down to ``(1 - budget) * true_var``, stated ES pushed down with it). It
    simulates the e-process one step ahead and only takes the deep cut if doing so
    keeps the mixture e-process below ``safety * threshold``; otherwise it backs the
    stated VaR/ES off toward the honest reference just enough to keep the e-process
    green. When ``es_independent`` is True the attacker may additionally INFLATE the
    stated ES (up to ``es_inflate_max`` above the Gaussian-consistent ES) to widen
    the e-statistic denominator and absorb a breach without tripping the gate; this
    is the test of whether the e-process catches an independently-stated ES.

    Returns the residual hidden risk (priced) and the realized e-process maximum.
    """
    true_var = np.asarray(true_var, dtype=float)
    returns = np.asarray(returns, dtype=float)
    loss = -returns
    n = returns.size
    if panel is None:
        panel = gates_mod.PANEL_WITH_EBT

    p = 1.0 - alpha
    alpha_test = 1.0 - level
    threshold = 1.0 / alpha_test if alpha_test > 0 else np.inf
    target = safety * threshold
    es_ratio = gaussian_es_var_ratio(alpha)

    floor_var = (1.0 - budget) * true_var          # deepest allowed stated VaR
    # honest ES consistent with the HONEST var (what a calibrated forecaster states)
    honest_es = es_ratio * true_var

    lams = np.asarray(grid, dtype=float)
    k = lams.size
    comp = np.ones(k)
    weights = np.full(k, 1.0 / k)
    m = 1.0

    stated_var = true_var.copy()
    stated_es = honest_es.copy()
    e_values = np.zeros(n)

    for t in range(n):
        # candidate deepest cut within the magnitude budget
        z_lo = floor_var[t]
        # ES floor under the budget; if ES stated independently it may be raised
        # ABOVE the Gaussian-consistent level (never below the budgeted floor).
        r_lo = es_ratio * z_lo
        r_hi = honest_es[t] * (1.0 + es_inflate_max) if es_independent else es_ratio * z_lo

        # try the most aggressive feasible (z as low as possible). For a given z,
        # the attacker picks the ES that MINIMIZES the resulting e-value while
        # staying a valid ES forecast (r > z, r <= r_hi). e_t decreases in (r - z),
        # so the e-minimizing choice is the LARGEST allowed r = r_hi (independent-ES
        # case) or the consistent r (dependent case).
        def e_after(z, r):
            zz = np.array([z]); rr = np.array([max(r, z * (1 + 1e-6))])
            return float(es_e_statistic(np.array([loss[t]]), zz, rr, alpha)[0])

        chosen_z = true_var[t]
        chosen_r = honest_es[t]
        # binary-search the deepest z in [z_lo, true_var] that keeps M below target,
        # using the e-minimizing ES at each z.
        lo, hi = z_lo, true_var[t]
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            r_mid = min(max(r_hi, mid * (1 + 1e-6)), honest_es[t] * (1.0 + es_inflate_max)) \
                if es_independent else es_ratio * mid
            e_mid = e_after(mid, r_mid)
            comp_try, m_try = _mixture_eprocess_step(comp, lams, weights, e_mid)
            if m_try <= target:
                hi = mid          # this depth is safe, try deeper
            else:
                lo = mid          # too deep, pull back up
        chosen_z = hi
        chosen_r = min(max(r_hi, chosen_z * (1 + 1e-6)),
                       honest_es[t] * (1.0 + es_inflate_max)) if es_independent \
            else es_ratio * chosen_z
        # never exceed honest (genuine under-reservation on VaR); ES may exceed honest
        chosen_z = min(chosen_z, true_var[t])

        e_t = e_after(chosen_z, chosen_r)
        comp, m = _mixture_eprocess_step(comp, lams, weights, e_t)
        stated_var[t] = chosen_z
        stated_es[t] = chosen_r
        e_values[t] = e_t

    # recompute the e-process cleanly from the chosen series (independent check)
    from .ebacktest import e_process
    eres = e_process(loss, stated_var, stated_es, alpha=alpha, level=level, grid=grid)

    # price the residual; run the full augmented+ebt panel with the INDEPENDENT ES
    pr = price(stated_var, true_var, returns, notional=notional,
               panel_names=[g.__name__ for g in panel])
    # full panel pass: every non-ebt gate on stated_var, ebt with the stated_es
    results = {}
    for g in panel:
        if g.__name__ == "e_backtest":
            results[g.__name__] = g(stated_var, returns, alpha=alpha, level=level,
                                    stated_es=stated_es)
        else:
            results[g.__name__] = g(stated_var, returns, alpha=alpha, level=level)
    passed_all = all(r.passed for r in results.values())

    return EAwareResult(
        stated_var=stated_var, stated_es=stated_es, true_var=true_var, returns=returns,
        budget=float(budget), es_independent=bool(es_independent),
        mul_bps=float(pr.mul_bps), mul_usd=float(pr.mul_usd),
        extra_exceptions_per_year=float(pr.extra_exceptions_per_year),
        eprocess_max=float(eres.max_eprocess), eprocess_threshold=float(threshold),
        eprocess_passed=bool(not eres.rejected),
        max_per_period_e=float(np.max(e_values)) if n else 0.0,
        panel_passed_all=bool(passed_all), gate_results=results,
    )


def max_residual_under_panel(
    true_var,
    returns,
    es_independent: bool = True,
    alpha: float = 0.01,
    level: float = 0.95,
    budget_grid=(0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20),
    notional: float = 1_000_000_000.0,
    panel=None,
) -> EAwareResult:
    """Largest e-aware residual that keeps the FULL defended panel green.

    Sweeps the magnitude budget upward and returns the most aggressive
    ``solve_eaware`` result whose ENTIRE augmented+e-backtest panel still passes
    (not just the e-process). This is the realistic, full-panel-green defense
    residual: how much an e-aware, budget-bounded, independent-ES attacker hides
    while every gate, regulatory and e-process, flashes green.
    """
    best = None
    for b in budget_grid:
        res = solve_eaware(true_var, returns, budget=b, es_independent=es_independent,
                           alpha=alpha, level=level, notional=notional, panel=panel)
        if res.panel_passed_all and (best is None or res.mul_bps > best.mul_bps):
            best = res
    if best is None:
        # even the smallest budget fails some gate; return the smallest-budget run
        best = solve_eaware(true_var, returns, budget=budget_grid[0],
                            es_independent=es_independent, alpha=alpha, level=level,
                            notional=notional, panel=panel)
    return best
