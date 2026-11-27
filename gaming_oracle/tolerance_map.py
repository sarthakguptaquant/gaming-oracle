"""Tolerance map (Cycle 7): WHICH gate is the binding constraint, as a function
of the conformal-coverage tolerance band.

This module replaces the old, false "the count tests are dead weight" claim with
the honest, portable finding. The Cycle-2/3 decomposition reported that
``conformal_coverage`` is "load-bearing" and Kupiec/Christoffersen are "dead
weight". That was a TUNING ARTIFACT: the conformal gate is a non-regulatory,
author-added check whose default tolerance ``tol=0.005`` was set TIGHTER than every
real regulatory band (Basel green <= 4 exceptions; Kupiec's non-rejection interval
is wider still). So conformal binds by construction, not because interval coverage
is the load-bearing risk test.

The defensible statement is: *which test is load-bearing is itself a tuning choice,
and here is the map.* We sweep the conformal tolerance over a grid and, at each
tolerance, report which gate is binding (the gate whose solo MUL equals the
full-panel MUL, i.e. the tightest constraint). Two readings fall straight out:

  - At a SMALL tolerance, conformal binds (it is tighter than the regulatory band).
  - As the tolerance widens past the regulatory band, conformal stops binding and
    a REGULATORY gate (Basel or Kupiec) takes over as the binding constraint.

The crossover tolerance is the honest headline: below it the "finding" is just the
author's tolerance choice; at or above any regulatorily-plausible tolerance the
binding constraint is a regulatory test. We run it on all three assets.

Math is prior (Kupiec 1995; Christoffersen 1998; Basel/BCBS 1996). The
contribution is the artifact: the map, and the honest reframing it forces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np

from . import gates as gates_mod
from .report import price
from .solver import solve

#: Default conformal-tolerance grid. Spans well below the regulatory bands (0.001)
#: up to a tolerance so wide the conformal gate is effectively inert (0.05).
DEFAULT_TOL_GRID = (0.001, 0.002, 0.003, 0.004, 0.005, 0.0075,
                    0.01, 0.015, 0.02, 0.03, 0.05)


@dataclass
class ToleranceRow:
    tol: float
    mul_full_bps: float                  # MUL under the full augmented panel at this tol
    binding_gate: str                    # the tightest gate (solo MUL == full MUL)
    solo_muls_bps: dict = field(default_factory=dict)   # gate -> solo MUL bps
    conformal_binds: bool = False        # is conformal the (a) binding gate here


@dataclass
class ToleranceMap:
    asset: str
    window_start: int
    window_end: int
    rows: list = field(default_factory=list)   # list[ToleranceRow]
    crossover_tol: float | None = None         # smallest tol where conformal stops binding
    regulatory_binding_gate: str = ""          # the gate that binds once conformal is inert
    notional: float = 1_000_000_000.0

    def summary(self) -> str:
        lines = [
            f"Tolerance map — {self.asset} window "
            f"[{self.window_start}:{self.window_end}]",
            f"  {'conformal tol':>13s} {'full MUL bps':>12s}  binding gate",
        ]
        for r in self.rows:
            mark = "  (conformal)" if r.conformal_binds else "  (REGULATORY)"
            lines.append(f"  {r.tol:13.4f} {r.mul_full_bps:12.2f}  "
                         f"{r.binding_gate}{mark}")
        if self.crossover_tol is not None:
            lines.append(
                f"  => conformal stops binding at tol >= {self.crossover_tol:.4f}; "
                f"above it the binding constraint is the REGULATORY gate "
                f"'{self.regulatory_binding_gate}'. Below it, conformal binds ONLY "
                f"because its tolerance is set tighter than the regulatory band — a "
                f"tuning choice, not a property of the data."
            )
        else:
            lines.append(
                "  => conformal binds across the whole swept grid; at no swept "
                "tolerance does a regulatory gate take over (widen the grid to "
                "locate the crossover)."
            )
        return "\n".join(lines)


def _solo_mul_bps(true_var, returns, gate, alpha, level, notional, refine) -> float:
    res = solve(true_var, returns, panel=[gate], alpha=alpha, level=level, refine=refine)
    p = price(res.stated_var, true_var, returns, notional=notional,
              panel_names=[gate.__name__])
    return float(p.mul_bps)


def tolerance_map(
    returns: np.ndarray,
    true_var: np.ndarray,
    asset: str = "",
    window_start: int = 0,
    window_end: int | None = None,
    tol_grid=DEFAULT_TOL_GRID,
    alpha: float = 0.01,
    level: float = 0.95,
    notional: float = 1_000_000_000.0,
    refine: bool = True,
) -> ToleranceMap:
    """Sweep the conformal tolerance and report the binding gate at each tol.

    At each tolerance the conformal gate is rebuilt with that ``tol``; the panel is
    (regulatory three + conformal). For each gate we compute its SOLO MUL (the lie
    that gate alone permits). The binding gate is the one whose solo MUL equals the
    full-panel MUL (the tightest constraint). We record whether conformal is that
    gate and locate the crossover tolerance where a regulatory gate takes over.
    """
    returns = np.asarray(returns, dtype=float)
    true_var = np.asarray(true_var, dtype=float)
    if window_end is None:
        window_end = returns.size
    rr = returns[window_start:window_end]
    tv = true_var[window_start:window_end]

    reg_gates = list(gates_mod.PANEL_REGULATORY)
    rows: list[ToleranceRow] = []
    crossover = None
    reg_binder = ""

    for tol in tol_grid:
        conf = partial(gates_mod.conformal_coverage, tol=tol)
        conf.__name__ = "conformal_coverage"  # so price/labels read cleanly
        panel = reg_gates + [conf]

        res = solve(tv, rr, panel=panel, alpha=alpha, level=level, refine=refine)
        p = price(res.stated_var, tv, rr, notional=notional,
                  panel_names=[g.__name__ for g in panel])
        mul_full = float(p.mul_bps)

        solo = {}
        for g in panel:
            solo[g.__name__] = _solo_mul_bps(tv, rr, g, alpha, level, notional, refine)
        # binding gate = tightest solo (smallest solo MUL)
        binding = min(solo, key=lambda k: solo[k])
        reg_only = {k: v for k, v in solo.items() if k != "conformal_coverage"}
        reg_tightest = min(reg_only, key=lambda k: reg_only[k])
        # conformal is the binder ONLY if it is STRICTLY tighter than every
        # regulatory gate (a tie means a regulatory gate already binds, so the
        # conformal tolerance is no longer the load-bearing tuning choice).
        conf_binds = bool(solo["conformal_coverage"] < reg_only[reg_tightest] - 1e-6)
        definable = mul_full > 1e-9

        rows.append(ToleranceRow(
            tol=float(tol), mul_full_bps=mul_full,
            binding_gate=binding if definable else f"{binding} (lie undefinable)",
            solo_muls_bps={k: float(v) for k, v in solo.items()},
            conformal_binds=conf_binds,
        ))
        if crossover is None and not conf_binds:
            crossover = float(tol)
            reg_binder = reg_tightest

    return ToleranceMap(
        asset=asset, window_start=int(window_start), window_end=int(window_end),
        rows=rows, crossover_tol=crossover, regulatory_binding_gate=reg_binder,
        notional=notional,
    )
