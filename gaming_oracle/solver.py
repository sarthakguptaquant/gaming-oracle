"""Solver: exhibit a Maximum Undetectable Lie against a gate panel.

We search for a STATED VaR series ``V_t <= V*_t`` (under-reserving relative to
the honest reference ``true_var = V*``) that maximizes the mean understatement

    Delta = mean_t( V*_t - V_t )

subject to the realized exception sequence passing EVERY gate in the panel
JOINTLY.

This is a SIMULATION / GREEDY PROJECTED search (spec Cycle-1 MVP). It EXHIBITS a
lie; it does NOT claim global optimality. The word "maximum" is reserved for the
Cycle-3 MILP. We say "the largest lie this search exhibited".

Two stages:
  1. Scalar bisection on a uniform shrink factor s in [0,1]:
     V_t = (1 - s) * V*_t. Larger s => larger lie => more/worse exceptions =>
     eventually a gate flips red. The pass region in s is (approximately)
     downward-closed, so bisection finds the frontier s*. We verify joint pass
     at the chosen s.
  2. Greedy per-point projected refinement: starting from the scalar solution,
     try to push individual non-exception days further down (more understatement)
     while re-checking that ALL gates still pass. This only ever lowers V_t (the
     lie can only grow), and every candidate is validated against the full panel,
     so the output is feasible by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gates as gates_mod


@dataclass
class SolveResult:
    stated_var: np.ndarray
    true_var: np.ndarray
    returns: np.ndarray
    mean_understatement: float       # Delta, in return units
    shrink: float                    # scalar shrink factor s*
    passed_all: bool
    gate_results: dict
    refined: bool


def _passes(stated_var, returns, panel, alpha, level) -> bool:
    out = gates_mod.run_panel(stated_var, returns, panel=panel, alpha=alpha, level=level)
    return out["passed_all"]


def solve(true_var, returns, panel=None, alpha: float = 0.01, level: float = 0.95,
          bisection_iters: int = 60, refine: bool = True,
          refine_passes: int = 2) -> SolveResult:
    """Find a stated_var <= true_var that maximizes mean understatement and
    passes every gate jointly.
    """
    true_var = np.asarray(true_var, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if panel is None:
        panel = gates_mod.PANEL

    # -- stage 1: scalar bisection on uniform shrink -----------------------
    # s=0 is the honest series (passes by construction if the reference is
    # calibrated). Find the largest s in [0,1] that still passes jointly.
    lo, hi = 0.0, 1.0
    # ensure lo passes; if even the honest series fails some gate, fall back to
    # the largest passing s we can find (lo stays 0 -> zero lie).
    if not _passes((1 - lo) * true_var, returns, panel, alpha, level):
        # honest reference does not pass the panel; nothing to exhibit
        s_star = 0.0
    else:
        for _ in range(bisection_iters):
            mid = 0.5 * (lo + hi)
            if _passes((1 - mid) * true_var, returns, panel, alpha, level):
                lo = mid
            else:
                hi = mid
        s_star = lo

    stated = (1 - s_star) * true_var

    # -- stage 2: greedy per-point projected refinement --------------------
    refined = False
    if refine and s_star >= 0.0:
        loss = -returns
        # only days that are NOT exceptions under the current stated series have
        # slack to push down without immediately creating an exception; but even
        # creating one can be ok if a gate still passes, so we attempt all days
        # ordered by current slack (largest gap first gives biggest local gain).
        for _ in range(refine_passes):
            gap = true_var - stated
            order = np.argsort(-gap)  # largest remaining understatement room first
            changed = False
            for t in order:
                if stated[t] <= 1e-12:
                    continue
                # candidate: push this point down to just below the realized loss
                # (turns it into a non-exception-preserving max lie) or toward 0.
                # Try a geometric ladder of deeper cuts at this point.
                base = stated[t]
                for frac in (0.5, 0.25, 0.1, 0.0):
                    cand_val = frac * base
                    if cand_val >= base:
                        continue
                    trial = stated.copy()
                    trial[t] = cand_val
                    if _passes(trial, returns, panel, alpha, level):
                        stated = trial
                        changed = True
                        break
            if changed:
                refined = True
            else:
                break

    # final validation of the full panel on the returned series
    out = gates_mod.run_panel(stated, returns, panel=panel, alpha=alpha, level=level)
    # safety: stated must never exceed true_var
    stated = np.minimum(stated, true_var)
    delta = float(np.mean(true_var - stated))

    return SolveResult(
        stated_var=stated,
        true_var=true_var,
        returns=returns,
        mean_understatement=delta,
        shrink=float(s_star),
        passed_all=bool(out["passed_all"]),
        gate_results=out["gates"],
        refined=refined,
    )
