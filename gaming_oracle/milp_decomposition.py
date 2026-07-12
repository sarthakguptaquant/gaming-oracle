"""Leave-one-out gate-contribution decomposition under the NON-UNIFORM MILP solver.

This re-runs the Cycle-2 gameability decomposition, but with the per-point MILP
solver (``milp_solver.solve_milp``) instead of the uniform-shrink search. Under a
UNIFORM lie the leave-one-out ranking put conformal-coverage as the only binding
gate and Kupiec/Christoffersen as "dead weight".

NOTE (superseded by Cycle 7): that binder/dead-weight ranking is a TUNING ARTIFACT
of the author-added conformal tolerance (tol=0.005, tighter than every regulatory
band), not a property of the risk tests; see ``tolerance_map.py``. Do not quote it
as a live finding. This module still runs the raw leave-one-out (it is the honest
mechanism), and the substantive question it answers survives the artifact: does a
NON-UNIFORM attacker make Kupiec and/or Christoffersen BIND that a uniform attacker
could not expose?

For each gate g in the panel:
  * mul_solo  = MUL with g as the ONLY gate (its standalone tightness)
  * mul_without = MUL with g removed (leave-one-out)
  * contribution = mul_without - mul_full  (how much the lie grows when g drops)

A gate binds when its removal grows the lie (contribution > 0) OR it is the
single tightest gate (solo MUL == full-panel MUL). A gate whose removal changes
nothing AND whose solo MUL is looser than the full panel is dead weight given the
others on this window.

Because the MILP optimizes exactly over the linear (count/window) gates, the
"maximum" language is earned for those; Christoffersen is verified post-hoc and,
where the MILP optimum is full-panel feasible, the full-panel number is also a
true maximum. We carry the optimal/exhibit flag through so the report never
overclaims.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import gates as gates_mod
from .milp_solver import solve_milp
from .report import price


@dataclass
class MilpGateContribution:
    gate_name: str
    mul_without_bps: float
    contribution_bps: float
    contribution_frac: float
    extra_exc_without: float
    mul_solo_bps: float = float("nan")
    binds_alone: bool = False
    binds: bool = False               # contribution > eps OR binds_alone


@dataclass
class MilpDecomposition:
    asset: str
    window_start: int
    window_end: int
    panel_names: list
    mul_full_bps: float
    mul_nopanel_bps: float
    full_optimal: bool                # was the full-panel MUL a true maximum
    full_method: str
    contributions: list = field(default_factory=list)
    notional: float = 1_000_000_000.0

    def binding_gates(self) -> list:
        return [c.gate_name for c in self.contributions if c.binds]

    def dead_weight_gates(self) -> list:
        return [c.gate_name for c in self.contributions if not c.binds]

    def ranking_table(self) -> str:
        opt = "MAXIMUM (MILP optimum is full-panel feasible)" if self.full_optimal \
            else "exhibit (MILP optimum needed heuristic repair)"
        lines = [
            f"Non-uniform (MILP) gameability decomposition — {self.asset} "
            f"window [{self.window_start}:{self.window_end}]",
            f"  full-panel MUL = {self.mul_full_bps:.2f} bps [{opt}]; "
            f"unconstrained ceiling (no gates) = {self.mul_nopanel_bps:.2f} bps",
            f"  {'gate':24s} {'solo MUL':>9s} {'MUL w/o':>9s} "
            f"{'grows lie':>10s} {'share':>7s}  note",
        ]
        for c in self.contributions:
            if c.binds_alone:
                note = "binds alone"
            elif c.binds:
                note = "binds (marginal)"
            else:
                note = "dead weight"
            lines.append(
                f"  {c.gate_name:24s} {c.mul_solo_bps:9.2f} "
                f"{c.mul_without_bps:9.2f} {c.contribution_bps:10.2f} "
                f"{c.contribution_frac:7.1%}  {note}"
            )
        binders = self.binding_gates()
        dead = self.dead_weight_gates()
        lines.append(f"  => BINDING: {binders if binders else 'none'}")
        lines.append(f"  => DEAD WEIGHT (given the others, this window): "
                     f"{dead if dead else 'none'}")
        return "\n".join(lines)


def _mul_bps_milp(true_var, returns, panel, alpha, level, notional) -> tuple[float, float, bool, str]:
    """Solve with the MILP and price; return (mul_bps, extra/yr, optimal, method).

    Empty panel = no gates: stated -> 0 everywhere, the unconstrained ceiling.
    """
    if not panel:
        delta = float(np.mean(true_var))
        e_true = gates_mod.exceptions(true_var, returns)
        e_zero = gates_mod.exceptions(np.zeros_like(true_var), returns)
        extra = float((e_zero.mean() - e_true.mean()) * 252)
        return delta * 1e4, extra, True, "closed-form"
    res = solve_milp(true_var, returns, panel=panel, alpha=alpha, level=level)
    p = price(res.stated_var, true_var, returns, notional=notional,
              panel_names=[g.__name__ for g in panel])
    return float(p.mul_bps), float(p.extra_exceptions_per_year), bool(res.optimal), res.method


def leave_one_out_milp(
    returns: np.ndarray,
    true_var: np.ndarray,
    asset: str = "",
    window_start: int = 0,
    window_end: int | None = None,
    panel=None,
    alpha: float = 0.01,
    level: float = 0.95,
    notional: float = 1_000_000_000.0,
    eps_bps: float = 1e-6,
) -> MilpDecomposition:
    """Non-uniform leave-one-out gate-contribution decomposition on one window."""
    returns = np.asarray(returns, dtype=float)
    true_var = np.asarray(true_var, dtype=float)
    if panel is None:
        panel = gates_mod.PANEL
    if window_end is None:
        window_end = returns.size

    rr = returns[window_start:window_end]
    tv = true_var[window_start:window_end]
    panel_names = [g.__name__ for g in panel]

    mul_full, _, full_opt, full_method = _mul_bps_milp(tv, rr, panel, alpha, level, notional)
    mul_nopanel, _, _, _ = _mul_bps_milp(tv, rr, [], alpha, level, notional)
    span = max(mul_nopanel - mul_full, 1e-12)

    solo = {}
    for g in panel:
        mul_solo, _, _, _ = _mul_bps_milp(tv, rr, [g], alpha, level, notional)
        solo[g.__name__] = float(mul_solo)

    contribs: list[MilpGateContribution] = []
    for g in panel:
        reduced = [x for x in panel if x is not g]
        mul_wo, extra_wo, _, _ = _mul_bps_milp(tv, rr, reduced, alpha, level, notional)
        contrib = max(mul_wo - mul_full, 0.0)
        binds_alone = bool(abs(solo[g.__name__] - mul_full) <= max(eps_bps, 1e-3))
        binds = bool(contrib > eps_bps or binds_alone)
        contribs.append(MilpGateContribution(
            gate_name=g.__name__,
            mul_without_bps=float(mul_wo),
            contribution_bps=float(contrib),
            contribution_frac=float(contrib / span),
            extra_exc_without=float(extra_wo),
            mul_solo_bps=solo[g.__name__],
            binds_alone=binds_alone,
            binds=binds,
        ))

    contribs.sort(key=lambda c: c.contribution_bps, reverse=True)

    return MilpDecomposition(
        asset=asset, window_start=int(window_start), window_end=int(window_end),
        panel_names=panel_names, mul_full_bps=float(mul_full),
        mul_nopanel_bps=float(mul_nopanel), full_optimal=full_opt,
        full_method=full_method, contributions=contribs, notional=notional,
    )
