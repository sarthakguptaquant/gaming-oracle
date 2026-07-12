"""Non-uniform lie solver (Cycle 3): a MILP over the discrete exception sequence.

Cycle 1/2 used a UNIFORM shrink: stated_var = (1 - s) * true_var everywhere,
one scalar knob. Under that uniform lie the leave-one-out decomposition ranked
conformal-coverage as the only binding gate and Kupiec/Christoffersen as
"dead weight".

NOTE (superseded by Cycle 7): that "conformal is load-bearing / the count tests
are dead weight" ranking was later shown to be a TUNING ARTIFACT of the
author-added conformal gate's hand-set tol=0.005 (tighter than every regulatory
band), not a property of the risk tests. Do NOT quote it as a live finding; the
portable result is the tolerance map (``tolerance_map.py``). It is retained here
only as the historical question this MILP module was built to probe, verbatim:
does a NON-UNIFORM attacker (per-point under-reservation) make Kupiec and/or
Christoffersen BIND, or does conformal still dominate? The Cycle-3 answer below
stands independent of the artifact (non-uniform hides more; Christoffersen can
bind under a clustering attacker).

This module answers it with a per-point optimum, not a scalar shrink.

------------------------------------------------------------------------------
The reduction that makes it a clean MILP
------------------------------------------------------------------------------
Every gate depends on the stated VaR series ONLY through the binary exception
sequence  e_t = 1[ loss_t > V_t ]  (loss_t = -returns_t). So the real decision
object is which days are made exceptions. Given a choice of e, the maximum
understatement is closed-form per day:

  * If we make day t an exception (e_t = 1): push V_t as low as possible. The
    deepest under-reservation is V_t -> 0, giving understatement u1_t = V*_t.
    Feasible as an exception only if loss_t > 0 (a non-negative VaR floor of 0
    can never be breached by a non-positive loss).

  * If day t is NOT an exception (e_t = 0): we need loss_t <= V_t <= V*_t, so the
    deepest under-reservation is V_t = max(loss_t, 0), giving understatement
    u0_t = V*_t - max(loss_t, 0). Feasible only if max(loss_t, 0) <= V*_t, i.e.
    loss_t <= V*_t.

So the marginal gain of flipping day t to an exception is

    g_t = u1_t - u0_t = max(loss_t, 0)  >= 0,

and total understatement = sum_t u0_t + sum_t g_t * e_t. The constant sum_t u0_t
drops out of the argmax, so the objective is LINEAR in e:  maximize sum_t g_t e_t.

Forced values:
  * loss_t <= 0           -> e_t = 0 forced (cannot be an exception).
  * loss_t >  V*_t        -> e_t = 1 forced (the honest series itself breaches;
                             the stated series is <= honest so it breaches too).

------------------------------------------------------------------------------
Gate constraints in terms of e
------------------------------------------------------------------------------
Let x = sum_t e_t (total exceptions over the window of length n), and let
x250 = sum over the last min(n,250) days.

  * Kupiec POF        : LR_uc(x) <= chi2_crit(1)   -> an interval x in [kx_lo, kx_hi]
  * Conformal coverage: |(1 - x/n) - (1-alpha)| <= tol  -> x in [cx_lo, cx_hi]
  * Du-Escanciano     : sqrt(n)*(x/n - alpha)/sqrt(alpha(1-alpha)) <= z -> x <= dx_hi
  * Basel traffic     : x250 <= 4
  * Christoffersen    : independence + conditional coverage on the transition
    counts (n00,n01,n10,n11). This is NONLINEAR in e. We handle it the honest
    way (see below).

The count-and-window gates (Kupiec, conformal, DE, Basel) are EXACT linear
constraints on sums of e, so scipy.optimize.milp solves them to a true optimum.
The word "maximum" is earned for any panel built only from those.

Christoffersen is the one nonlinear gate. We do NOT pretend the MILP enforces it.
We solve the MILP over the linear gates, then VERIFY Christoffersen on the
solution. Two outcomes, both honest:
  (a) Christoffersen passes on the MILP optimum  -> the MILP optimum is feasible
      for the FULL panel, so it is the true maximum for the full panel too. We
      label it "maximum".
  (b) Christoffersen fails -> we fall back to a clearly-labeled local-search
      repair (greedy de-clustering of exceptions) and label the result "exhibit",
      NOT "maximum", because the repair is heuristic.

In Cycle 2 Christoffersen did not bind under the uniform lie (a mechanical fact
about that window, not the superseded "dead weight" ranking above), so (a) is
expected on real data; we check and report which branch fired, never assume it.

Math is prior (Kupiec 1995; Christoffersen 1998; Basel/BCBS 1996; Du & Escanciano
2017). The contribution is the artifact: a per-point optimum and the priced lie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import LinearConstraint, milp

from . import gates as gates_mod


@dataclass
class MilpSolveResult:
    stated_var: np.ndarray
    true_var: np.ndarray
    returns: np.ndarray
    mean_understatement: float          # Delta, in return units
    passed_all: bool
    gate_results: dict
    optimal: bool                       # True only if the MILP optimum is itself
                                        # full-panel feasible (earns "maximum")
    method: str                         # "milp" or "milp+heuristic_repair"
    christoffersen_handled: str         # "exact-feasible" | "repair" | "n/a"
    exception_count: int
    milp_status: str


# ---------------------------------------------------------------------------
# count feasibility intervals for the linear gates
# ---------------------------------------------------------------------------

def _kupiec_count_interval(n: int, alpha: float, level: float) -> tuple[int, int]:
    """Integer counts x in [lo, hi] for which Kupiec POF does NOT reject."""
    crit = stats.chi2.ppf(level, df=1)
    lo, hi = 0, n
    ok = []
    for x in range(n + 1):
        pi = x / n if n else 0.0
        ll_null = x * (np.log(alpha) if alpha > 0 else 0.0) + \
            (n - x) * (np.log(1 - alpha) if alpha < 1 else 0.0)
        if x == 0:
            ll_alt = (n - x) * (np.log(1 - pi) if pi < 1 else 0.0)
        elif x == n:
            ll_alt = x * (np.log(pi) if pi > 0 else 0.0)
        else:
            ll_alt = x * np.log(pi) + (n - x) * np.log(1 - pi)
        lr = max(-2.0 * (ll_null - ll_alt), 0.0)
        if lr <= crit:
            ok.append(x)
    if not ok:
        return 0, 0
    return ok[0], ok[-1]


def _conformal_count_interval(n: int, alpha: float, tol: float) -> tuple[int, int]:
    """Integer counts x with |(1 - x/n) - (1-alpha)| <= tol  <=>  |x/n - alpha| <= tol."""
    lo = int(np.ceil((alpha - tol) * n))
    hi = int(np.floor((alpha + tol) * n))
    lo = max(lo, 0)
    hi = min(hi, n)
    return lo, hi


def _du_escanciano_count_hi(n: int, alpha: float, level: float) -> int:
    """Largest integer x with the one-sided DE LM statistic not rejecting."""
    z = stats.norm.ppf(level)
    denom = np.sqrt(alpha * (1 - alpha))
    # sqrt(n)*(x/n - alpha)/denom <= z  ->  x <= n*alpha + z*denom*sqrt(n)
    hi = int(np.floor(n * alpha + z * denom * np.sqrt(n)))
    return min(max(hi, 0), n)


# ---------------------------------------------------------------------------
# the MILP
# ---------------------------------------------------------------------------

def _gate_count_bounds(panel, n: int, alpha: float, level: float, tol: float):
    """Aggregate per-gate count constraints into [x_lo, x_hi] and a Basel-250 cap.

    Returns (x_lo, x_hi, basel_cap_or_None). Christoffersen contributes no count
    bound (handled post-hoc). Gates not in the panel contribute nothing.
    """
    names = {g.__name__ for g in panel}
    x_lo, x_hi = 0, n
    basel_cap = None
    if "kupiec_pof" in names:
        klo, khi = _kupiec_count_interval(n, alpha, level)
        x_lo, x_hi = max(x_lo, klo), min(x_hi, khi)
    if "conformal_coverage" in names:
        clo, chi = _conformal_count_interval(n, alpha, tol)
        x_lo, x_hi = max(x_lo, clo), min(x_hi, chi)
    if "du_escanciano" in names:
        dhi = _du_escanciano_count_hi(n, alpha, level)
        x_hi = min(x_hi, dhi)
    if "basel_traffic_light" in names:
        basel_cap = 4
    return x_lo, x_hi, basel_cap


def solve_milp(true_var, returns, panel=None, alpha: float = 0.01,
               level: float = 0.95, tol: float = 0.005,
               repair: bool = True) -> MilpSolveResult:
    """Non-uniform MILP solver.

    Maximizes per-point understatement subject to the count/window gate
    constraints exactly, then verifies the nonlinear Christoffersen gate and (if
    needed and ``repair``) applies a labeled heuristic repair.
    """
    true_var = np.asarray(true_var, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if panel is None:
        panel = gates_mod.PANEL
    n = returns.size
    loss = -returns

    # per-day marginal gain of making day t an exception
    gain = np.maximum(loss, 0.0)

    # forced values
    forced_zero = loss <= 0.0           # cannot be an exception
    forced_one = loss > true_var        # honest series already breaches

    x_lo, x_hi, basel_cap = _gate_count_bounds(panel, n, alpha, level, tol)

    # bounds on each e_t in {0,1} with forcing baked into integrality bounds
    lb = np.where(forced_one, 1.0, 0.0)
    ub = np.where(forced_zero, 0.0, 1.0)
    # a day cannot be simultaneously forced_one and forced_zero; if loss>true_var
    # then loss>0, so forced_zero is False there. Safe.

    # objective: maximize sum gain_t e_t  ->  minimize -gain . e
    c = -gain

    constraints = []
    # total-count window: x_lo <= sum e <= x_hi  (only if any count gate present)
    if x_lo > 0 or x_hi < n:
        A_all = np.ones((1, n))
        constraints.append(LinearConstraint(A_all, x_lo, x_hi))
    # Basel: sum over last min(n,250) <= cap
    if basel_cap is not None:
        w = min(n, 250)
        a = np.zeros((1, n))
        a[0, n - w:] = 1.0
        constraints.append(LinearConstraint(a, 0, basel_cap))

    integrality = np.ones(n)
    from scipy.optimize import Bounds
    bounds = Bounds(lb, ub)

    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

    if not res.success or res.x is None:
        # forced exceptions alone may already violate a count cap (honest series
        # itself fails the panel on this window): no lie is definable. Return the
        # honest series as the (zero-lie) feasible point and report status.
        e = forced_one.astype(int)
        stated = _stated_from_exceptions(e, loss, true_var)
        out = gates_mod.run_panel(stated, returns, panel=panel, alpha=alpha, level=level)
        return MilpSolveResult(
            stated_var=stated, true_var=true_var, returns=returns,
            mean_understatement=float(np.mean(true_var - stated)),
            passed_all=bool(out["passed_all"]), gate_results=out["gates"],
            optimal=False, method="milp", christoffersen_handled="n/a",
            exception_count=int(e.sum()),
            milp_status=str(getattr(res, "message", "infeasible")),
        )

    e = (np.asarray(res.x) > 0.5).astype(int)
    stated = _stated_from_exceptions(e, loss, true_var)

    # verify the full panel (recompute from scratch)
    out = gates_mod.run_panel(stated, returns, panel=panel, alpha=alpha, level=level)
    has_chris = any(g.__name__ == "christoffersen" for g in panel)

    if out["passed_all"]:
        # the MILP optimum is feasible for the FULL panel -> true maximum
        chris_state = "exact-feasible" if has_chris else "n/a"
        return MilpSolveResult(
            stated_var=stated, true_var=true_var, returns=returns,
            mean_understatement=float(np.mean(true_var - stated)),
            passed_all=True, gate_results=out["gates"],
            optimal=True, method="milp", christoffersen_handled=chris_state,
            exception_count=int(e.sum()),
            milp_status=str(res.message),
        )

    # otherwise the only thing that can have failed is Christoffersen (the linear
    # gates are enforced exactly). Repair by de-clustering exceptions if allowed.
    if repair and has_chris:
        e_rep, stated_rep, out_rep = _repair_christoffersen(
            e, loss, true_var, returns, panel, alpha, level, x_lo, x_hi, basel_cap)
        return MilpSolveResult(
            stated_var=stated_rep, true_var=true_var, returns=returns,
            mean_understatement=float(np.mean(true_var - stated_rep)),
            passed_all=bool(out_rep["passed_all"]), gate_results=out_rep["gates"],
            optimal=False, method="milp+heuristic_repair",
            christoffersen_handled="repair", exception_count=int(e_rep.sum()),
            milp_status=str(res.message),
        )

    # no repair: report the infeasible-on-Christoffersen MILP point honestly
    return MilpSolveResult(
        stated_var=stated, true_var=true_var, returns=returns,
        mean_understatement=float(np.mean(true_var - stated)),
        passed_all=False, gate_results=out["gates"],
        optimal=False, method="milp", christoffersen_handled="failed-no-repair",
        exception_count=int(e.sum()), milp_status=str(res.message),
    )


def _stated_from_exceptions(e, loss, true_var, eps: float = 1e-9) -> np.ndarray:
    """Deepest under-reserving stated VaR consistent with exception vector e.

    e_t = 1 -> V_t = 0 (must satisfy loss_t > 0; guaranteed by feasibility).
    e_t = 0 -> V_t = max(loss_t, 0), capped at true_var (a non-exception must
               have loss_t <= V_t <= V*_t).
    """
    e = np.asarray(e)
    loss = np.asarray(loss, dtype=float)
    true_var = np.asarray(true_var, dtype=float)
    stated = np.where(
        e == 1,
        0.0,
        np.minimum(np.maximum(loss, 0.0), true_var),
    )
    # guard: never exceed true_var, never negative
    stated = np.clip(stated, 0.0, true_var)
    return stated


def _repair_christoffersen(e, loss, true_var, returns, panel, alpha, level,
                           x_lo, x_hi, basel_cap):
    """Greedy de-clustering repair when the MILP optimum fails Christoffersen.

    Christoffersen's independence test rejects when exceptions CLUSTER (high
    11-transition rate). We keep the exception COUNT inside [x_lo, x_hi] (so the
    coverage gates stay green) but swap clustered exceptions for isolated ones,
    re-checking the full panel after each move. Heuristic: the result is an
    EXHIBIT, not a proven maximum.
    """
    e = np.asarray(e).copy()
    n = e.size

    # which days are eligible to BE an exception (loss>0) and which are forced
    eligible = loss > 0.0
    forced_one = loss > true_var

    def panel_pass(ev):
        st = _stated_from_exceptions(ev, loss, true_var)
        return gates_mod.run_panel(st, returns, panel=panel, alpha=alpha,
                                   level=level)["passed_all"]

    # up to a bounded number of swap passes
    for _ in range(8):
        out = gates_mod.run_panel(_stated_from_exceptions(e, loss, true_var),
                                  returns, panel=panel, alpha=alpha, level=level)
        if out["passed_all"]:
            break
        # find a clustered exception (its neighbor is also an exception) that is
        # not forced, and a non-exception isolated slot (gain-ranked) to move to
        idx_excess = [t for t in range(n) if e[t] == 1 and not forced_one[t] and
                      ((t > 0 and e[t - 1] == 1) or (t < n - 1 and e[t + 1] == 1))]
        # candidate target slots: currently non-exception, eligible, isolated
        targets = [t for t in range(n) if e[t] == 0 and eligible[t] and
                   (t == 0 or e[t - 1] == 0) and (t == n - 1 or e[t + 1] == 0)]
        # rank targets by understatement gain (max(loss,0)) descending
        targets.sort(key=lambda t: loss[t], reverse=True)
        moved = False
        for src in idx_excess:
            for tgt in targets:
                trial = e.copy()
                trial[src] = 0
                trial[tgt] = 1
                if int(trial.sum()) < x_lo or int(trial.sum()) > x_hi:
                    continue
                if basel_cap is not None:
                    w = min(n, 250)
                    if int(trial[n - w:].sum()) > basel_cap:
                        continue
                if panel_pass(trial):
                    e = trial
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break

    stated = _stated_from_exceptions(e, loss, true_var)
    out = gates_mod.run_panel(stated, returns, panel=panel, alpha=alpha, level=level)
    return e, stated, out
