"""Leave-one-out gate-contribution / gameability decomposition (Cycle 2, gap #8).

*** CYCLE-7 CAVEAT — READ BEFORE QUOTING THIS MODULE'S OUTPUT. ***
This module's "binds alone / marginal backstop / dead weight" labels are computed
against a panel that INCLUDES the non-regulatory, author-added ``conformal_coverage``
gate, whose default ``tol=0.005`` is tighter than every regulatory band. So the
labels are a TUNING ARTIFACT: conformal looks "load-bearing" and the regulatory
count tests look "dead weight" only because conformal was tuned tighter than them.
DO NOT report "the count tests are dead weight." The honest, portable finding is the
TOLERANCE MAP in ``gaming_oracle.tolerance_map``: which gate binds is a function of
the conformal tolerance, and at any regulatorily plausible tolerance a REGULATORY
gate binds. This module is retained only to exhibit the MECHANISM that produces the
artifact.

The Cycle-1 killer demo was thin: it added ONE higher-power test (the one-sided UC
score test) and showed the lie shrinks. This is the more general replacement.

For a representative window, re-run the solver with each gate REMOVED one at a
time and measure how much the exhibited MUL GROWS when that gate is dropped:

    contribution(g) = MUL(panel without g) - MUL(full panel)

Because dropping a gate only RELAXES the feasible set, the solver can never do
worse, so contribution(g) >= 0 (up to solver noise). The gate whose removal
grows the lie the most is the one doing the most detecting on this window; a
gate whose removal changes nothing is DEAD WEIGHT given the others (already
implied by the binding gates). Ranking the gates by contribution gives a
"gameability decomposition": which gates are load-bearing, which are redundant.

This is panel-relative and window-aware: a gate that is dead weight on one
window can be the binding gate on another (e.g. Basel binds on quiet windows,
conformal/independence on clustered ones). We therefore report the window the
decomposition was run on, and the rolling module supplies the distribution
context so no single decomposition reads as universal.

Math is prior (Kupiec 1995; Christoffersen 1998; Basel/BCBS 1996; Du & Escanciano
2017). The contribution is the leave-one-out artifact + the metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import gates as gates_mod
from .report import price
from .solver import solve


@dataclass
class GateContribution:
    gate_name: str
    mul_without_bps: float        # MUL with this gate removed (leave-one-out)
    contribution_bps: float       # mul_without - mul_full (>= 0)
    contribution_frac: float      # contribution / (panel-vs-no-panel span), [0,1]
    extra_exc_without: float
    mul_solo_bps: float = float("nan")   # MUL if THIS were the only gate
    binds_alone: bool = False            # is this gate the single tightest one


@dataclass
class Decomposition:
    asset: str
    window_start: int
    window_end: int
    panel_names: list
    mul_full_bps: float            # MUL under the full panel
    mul_nopanel_bps: float         # MUL with NO gates (the unconstrained ceiling)
    contributions: list = field(default_factory=list)   # ranked desc
    notional: float = 1_000_000_000.0

    def ranking_table(self) -> str:
        lines = [
            f"Gameability decomposition — {self.asset} "
            f"window [{self.window_start}:{self.window_end}]",
            f"  full-panel MUL = {self.mul_full_bps:.2f} bps; "
            f"unconstrained ceiling (no gates) = {self.mul_nopanel_bps:.2f} bps",
            f"  {'gate':24s} {'solo MUL':>9s} {'MUL w/o':>9s} "
            f"{'grows lie':>10s} {'share':>7s}  note",
        ]
        for c in self.contributions:
            note = "binds alone" if c.binds_alone else (
                "marginal backstop" if c.contribution_bps > 1e-6 else "dead weight")
            lines.append(
                f"  {c.gate_name:24s} {c.mul_solo_bps:9.2f} "
                f"{c.mul_without_bps:9.2f} {c.contribution_bps:10.2f} "
                f"{c.contribution_frac:7.1%}  {note}"
            )
        binders = [c for c in self.contributions if c.binds_alone]
        top = self.contributions[0] if self.contributions else None
        if binders:
            b = binders[0]
            lines.append(
                f"  => LOAD-BEARING: {b.gate_name} binds the whole full-panel "
                f"lie alone (solo MUL {b.mul_solo_bps:.2f} == full-panel "
                f"{self.mul_full_bps:.2f}). Its leave-one-out grows the lie only "
                f"{b.contribution_bps:.2f} bps because the next-tightest gate is "
                f"right behind it; that next gate is the marginal backstop. Gates "
                f"marked 'dead weight' are redundant GIVEN the binders on this "
                f"window."
            )
        elif top is not None and top.contribution_bps <= 1e-6:
            lines.append("  => every gate is non-binding on this window "
                         "(no single gate is load-bearing here).")
        return "\n".join(lines)


def _mul_bps(true_var, returns, panel, alpha, level, notional, refine) -> tuple[float, float]:
    """Solve and price; return (mul_bps, extra_exceptions_per_year).

    An empty panel means 'no gates': the unconstrained ceiling is the lie that
    drives stated VaR to zero everywhere, i.e. mean(true_var)*1e4. We compute it
    directly rather than running the solver on a degenerate constraint set.
    """
    if not panel:
        delta = float(np.mean(true_var))   # stated -> 0 everywhere
        # extra exceptions at stated=0: every loss>0 day is an exception
        e_true = gates_mod.exceptions(true_var, returns)
        e_zero = gates_mod.exceptions(np.zeros_like(true_var), returns)
        extra = float((e_zero.mean() - e_true.mean()) * 252)
        return delta * 1e4, extra
    res = solve(true_var, returns, panel=panel, alpha=alpha, level=level, refine=refine)
    p = price(res.stated_var, true_var, returns, notional=notional,
              panel_names=[g.__name__ for g in panel])
    return float(p.mul_bps), float(p.extra_exceptions_per_year)


def leave_one_out(
    returns: np.ndarray,
    true_var: np.ndarray,
    asset: str = "",
    window_start: int = 0,
    window_end: int | None = None,
    panel=None,
    alpha: float = 0.01,
    level: float = 0.95,
    notional: float = 1_000_000_000.0,
    refine: bool = True,
) -> Decomposition:
    """Leave-one-out gate-contribution decomposition on a single window.

    For each gate g in ``panel``, recompute the MUL with g removed; the
    contribution of g is how much the lie grows when g is dropped. Ranked
    descending = most-detecting first.
    """
    returns = np.asarray(returns, dtype=float)
    true_var = np.asarray(true_var, dtype=float)
    if panel is None:
        panel = gates_mod.PANEL
    if window_end is None:
        window_end = returns.size

    rr = returns[window_start:window_end]
    tv = true_var[window_start:window_end]
    panel_names = [g.__name__ for g in panel]

    mul_full, _ = _mul_bps(tv, rr, panel, alpha, level, notional, refine)
    mul_nopanel, _ = _mul_bps(tv, rr, [], alpha, level, notional, refine)
    span = max(mul_nopanel - mul_full, 1e-12)

    # solo MUL per gate: how loose each gate is BY ITSELF. The gate whose solo
    # MUL equals the full-panel MUL is the single tightest one (it binds alone);
    # the rest are slack given it. This complements leave-one-out: LOO measures
    # what a gate adds at the MARGIN (vs the next-tightest backstop), solo
    # measures its standalone tightness.
    solo = {}
    for g in panel:
        mul_solo, _ = _mul_bps(tv, rr, [g], alpha, level, notional, refine)
        solo[g.__name__] = float(mul_solo)

    contribs: list[GateContribution] = []
    for g in panel:
        reduced = [x for x in panel if x is not g]
        mul_wo, extra_wo = _mul_bps(tv, rr, reduced, alpha, level, notional, refine)
        contrib = mul_wo - mul_full
        # numerical floor: relaxing a constraint can't truly reduce the max
        contrib = max(contrib, 0.0)
        contribs.append(GateContribution(
            gate_name=g.__name__,
            mul_without_bps=float(mul_wo),
            contribution_bps=float(contrib),
            contribution_frac=float(contrib / span),
            extra_exc_without=float(extra_wo),
            mul_solo_bps=solo[g.__name__],
            binds_alone=bool(abs(solo[g.__name__] - mul_full) <= 1e-6),
        ))

    contribs.sort(key=lambda c: c.contribution_bps, reverse=True)

    return Decomposition(
        asset=asset, window_start=int(window_start), window_end=int(window_end),
        panel_names=panel_names, mul_full_bps=float(mul_full),
        mul_nopanel_bps=float(mul_nopanel), contributions=contribs,
        notional=notional,
    )


def representative_window(
    returns: np.ndarray,
    true_var: np.ndarray,
    window: int = 250,
    step: int = 25,
    panel=None,
    alpha: float = 0.01,
    level: float = 0.95,
) -> tuple[int, int]:
    """Pick a representative window for the decomposition.

    A decomposition is only meaningful where the lie is definable (the honest
    reference passes the panel). Among definable windows we pick the one whose
    full-panel MUL is closest to the MEDIAN definable MUL, so the decomposition
    describes a typical window rather than the luckiest one. Falls back to the
    most-recent window if none are definable.
    """
    returns = np.asarray(returns, dtype=float)
    true_var = np.asarray(true_var, dtype=float)
    if panel is None:
        panel = gates_mod.PANEL
    n = returns.size
    starts = list(range(0, n - window + 1, step))
    if not starts or starts[-1] != n - window:
        starts.append(n - window)

    definable = []
    for s in starts:
        e = s + window
        rr, tv = returns[s:e], true_var[s:e]
        if gates_mod.run_panel(tv, rr, panel=panel, alpha=alpha, level=level)["passed_all"]:
            res = solve(tv, rr, panel=panel, alpha=alpha, level=level)
            p = price(res.stated_var, tv, rr, panel_names=[g.__name__ for g in panel])
            definable.append((s, e, float(p.mul_bps)))

    if not definable:
        return n - window, n
    muls = np.array([d[2] for d in definable])
    med = float(np.median(muls))
    idx = int(np.argmin(np.abs(muls - med)))
    return definable[idx][0], definable[idx][1]
