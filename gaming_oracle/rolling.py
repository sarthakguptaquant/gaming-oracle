"""Rolling-window MUL distribution (Cycle 2).

Cycle 1's headline lived on ONE 250-day window, which is cherry-pickable. This
module slides a fixed-length window across the full return history, re-runs the
existing solver on each window against the existing panel, and reports the
DISTRIBUTION of the Maximum Undetectable Lie rather than a single number.

Two things make this honest rather than a lucky-window highlight reel:

1. The honest reference does not pass the panel on every window (fat-tail
   clusters push the reference itself into Basel amber / a conformal-coverage
   miss). A lie is only DEFINABLE on a window where the honest series itself
   flashes green, because the construction is "shrink a passing series until a
   gate flips". So we report ``frac_honest_passes`` = the fraction of windows
   where the honest reference passes the panel, and compute the MUL
   distribution ONLY over those definable windows. The number is therefore
   "the lie budget conditional on the panel being green to begin with", and the
   denominator is reported in the open.

2. We report min / median / max / IQR, not the max. One window's max lie is an
   exhibit, not a property of the model.

Everything here is panel-relative and window-aware, exactly like Cycle 1.
Nothing about the solver, gates, or pricing is re-derived; this is a sweep over
the existing components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import gates as gates_mod
from .report import price
from .solver import solve


@dataclass
class WindowResult:
    """One window's outcome in the rolling sweep."""

    start: int                  # index into the full series (inclusive)
    end: int                    # index into the full series (exclusive)
    honest_passes: bool         # did the honest reference pass the panel here
    mul_bps: float              # exhibited lie in bps (0.0 if not definable)
    mul_usd: float
    extra_exceptions_per_year: float
    shrink: float
    passed_all: bool            # did the solved stated series pass the panel
    honest_exceptions: int      # exception count of the honest reference


@dataclass
class RollingDistribution:
    """Distribution of the MUL across the rolling windows for one asset."""

    asset: str
    window: int
    step: int
    n_windows: int
    n_definable: int                  # windows where honest reference passes
    frac_honest_passes: float         # n_definable / n_windows
    # distribution stats, computed over the DEFINABLE windows only:
    mul_bps_min: float
    mul_bps_median: float
    mul_bps_max: float
    mul_bps_iqr: float                # Q3 - Q1
    mul_bps_q1: float
    mul_bps_q3: float
    mul_usd_median: float
    extra_exc_median: float
    windows: list = field(default_factory=list)   # list[WindowResult]
    notional: float = 1_000_000_000.0
    panel_names: list = field(default_factory=list)

    def summary(self) -> str:
        if self.n_definable == 0:
            return (
                f"{self.asset}: 0/{self.n_windows} windows have a panel-passing "
                f"honest reference; the lie is UNDEFINABLE on every window "
                f"(honest GARCH reference itself fails the {self.window}-day "
                f"panel). No MUL distribution to report."
            )
        return (
            f"{self.asset}: over {self.n_windows} rolling {self.window}-day "
            f"windows (step {self.step}), the honest reference passes the panel "
            f"on {self.n_definable} ({self.frac_honest_passes:.0%}). On those "
            f"definable windows the exhibited MUL is "
            f"min={self.mul_bps_min:.1f} / median={self.mul_bps_median:.1f} / "
            f"max={self.mul_bps_max:.1f} bps, IQR={self.mul_bps_iqr:.1f} bps "
            f"(Q1={self.mul_bps_q1:.1f}, Q3={self.mul_bps_q3:.1f}); median "
            f"${self.mul_usd_median:,.0f} on ${self.notional:,.0f}, median "
            f"{self.extra_exc_median:.2f} extra exceptions/yr. "
            f"Panel-relative; the headline is the distribution, not one window."
        )


def rolling_mul(
    returns: np.ndarray,
    true_var: np.ndarray,
    asset: str = "",
    window: int = 250,
    step: int = 25,
    panel=None,
    alpha: float = 0.01,
    level: float = 0.95,
    notional: float = 1_000_000_000.0,
    refine: bool = True,
) -> RollingDistribution:
    """Slide a ``window``-day window in ``step``-day strides over the full
    series, solve the lie on each, and return the MUL distribution.

    The distribution stats are computed over the DEFINABLE windows only (those
    where the honest reference passes the panel); ``frac_honest_passes`` reports
    how common that is, so the conditioning is never hidden.
    """
    returns = np.asarray(returns, dtype=float)
    true_var = np.asarray(true_var, dtype=float)
    if panel is None:
        panel = gates_mod.PANEL
    panel_names = [g.__name__ for g in panel]

    n = returns.size
    if n < window:
        raise ValueError(f"series of length {n} shorter than window {window}")

    starts = list(range(0, n - window + 1, step))
    # always include the final window so the most-recent slice is represented
    last_start = n - window
    if starts[-1] != last_start:
        starts.append(last_start)

    win_results: list[WindowResult] = []
    definable_bps: list[float] = []
    definable_usd: list[float] = []
    definable_extra: list[float] = []

    for s in starts:
        e = s + window
        rr = returns[s:e]
        tv = true_var[s:e]

        honest = gates_mod.run_panel(tv, rr, panel=panel, alpha=alpha, level=level)
        honest_passes = bool(honest["passed_all"])
        honest_exc = int(gates_mod.exceptions(tv, rr).sum())

        if honest_passes:
            res = solve(tv, rr, panel=panel, alpha=alpha, level=level, refine=refine)
            p = price(res.stated_var, tv, rr, notional=notional,
                      panel_names=panel_names)
            mul_bps = float(p.mul_bps)
            mul_usd = float(p.mul_usd)
            extra = float(p.extra_exceptions_per_year)
            passed_all = bool(res.passed_all)
            shrink = float(res.shrink)
            definable_bps.append(mul_bps)
            definable_usd.append(mul_usd)
            definable_extra.append(extra)
        else:
            # lie is not definable on a window the honest series cannot pass
            mul_bps = mul_usd = extra = shrink = 0.0
            passed_all = False

        win_results.append(WindowResult(
            start=int(s), end=int(e), honest_passes=honest_passes,
            mul_bps=mul_bps, mul_usd=mul_usd,
            extra_exceptions_per_year=extra, shrink=shrink,
            passed_all=passed_all, honest_exceptions=honest_exc,
        ))

    n_windows = len(win_results)
    n_definable = len(definable_bps)
    frac = n_definable / n_windows if n_windows else 0.0

    if n_definable:
        arr = np.asarray(definable_bps)
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        dist = RollingDistribution(
            asset=asset, window=window, step=step,
            n_windows=n_windows, n_definable=n_definable, frac_honest_passes=frac,
            mul_bps_min=float(arr.min()),
            mul_bps_median=float(np.median(arr)),
            mul_bps_max=float(arr.max()),
            mul_bps_iqr=float(q3 - q1), mul_bps_q1=q1, mul_bps_q3=q3,
            mul_usd_median=float(np.median(definable_usd)),
            extra_exc_median=float(np.median(definable_extra)),
            windows=win_results, notional=notional, panel_names=panel_names,
        )
    else:
        dist = RollingDistribution(
            asset=asset, window=window, step=step,
            n_windows=n_windows, n_definable=0, frac_honest_passes=0.0,
            mul_bps_min=0.0, mul_bps_median=0.0, mul_bps_max=0.0,
            mul_bps_iqr=0.0, mul_bps_q1=0.0, mul_bps_q3=0.0,
            mul_usd_median=0.0, extra_exc_median=0.0,
            windows=win_results, notional=notional, panel_names=panel_names,
        )
    return dist
