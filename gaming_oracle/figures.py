"""Publication figures for the Gaming Oracle (matplotlib -> PNG).

Four figures, all computed from the REAL pipeline (no hand-placed numbers):

  (a) figures/fig_rolling_mul_distribution.png
      Rolling Maximum-Undetectable-Lie distribution per asset (violin + box),
      over the definable windows (honest reference passes the panel). This is the
      honest "distribution, not one window" headline from Cycle 2.

  (b) figures/fig_gate_contribution_ranking.png
      Gate-contribution / gameability ranking: solo MUL and leave-one-out
      "grows-lie" contribution per gate, on the representative window per asset.
      Ranks which gates are load-bearing vs dead weight (Cycle 2/3).

  (c) figures/fig_attack_vs_defense.png
      Attack-vs-defense lie budget WITH and WITHOUT the Wang-Wang-Ziegel e-process
      gate, per asset, plus the e-process trajectory crossing 1/alpha on the gamed
      series (the defense catching the lie). This is the Cycle-4 result.

  (d) figures/fig_tolerance_map.png
      Full-panel MUL vs the conformal tolerance per asset, with the
      conformal-binding (artifact) zone and the regulatory-binding zone shaded and
      the hand-set default tol=0.005 marked. Shows that which gate is load-bearing
      is a tuning choice. This is the Cycle-7 foundation-repair result.

Every number is produced by the same gates/solver/report the tool ships; the
figures are a view, never a separate computation. Math is prior (Kupiec 1995;
Christoffersen 1998; Basel/BCBS 1996; one-sided UC score test in the spirit of
Du & Escanciano 2017; e-backtest = Wang, Wang & Ziegel, "E-backtesting",
Management Science 72(6):4952-4973, online 2025, arXiv:2209.00991).
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from . import gates as gates_mod  # noqa: E402
from .data import load_data  # noqa: E402
from .decomposition import leave_one_out, representative_window  # noqa: E402
from .ebacktest import es_e_statistic, gaussian_es_var_ratio  # noqa: E402
from .report import price  # noqa: E402
from .rolling import rolling_mul  # noqa: E402
from .solver import solve  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
ASSETS = ["AAPL", "SPY", "BTC-USD"]

# brand-ish palette (cream/ink/quant-blue/terminal-green), readable in print
INK = "#0F0E0B"
BLUE = "#1B3DFF"
GREEN = "#00B870"
AMBER = "#E8A33D"
RED = "#D7263D"
GREY = "#9A968C"
CREAM = "#F4F1E8"


def _ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def _load_all(method="garch", alpha=0.01):
    data = {}
    for a in ASSETS:
        ds = load_data(asset=a, alpha=alpha, method=method)
        data[a] = ds
    return data


# ---------------------------------------------------------------------------
# (a) rolling MUL distribution per asset
# ---------------------------------------------------------------------------

def fig_rolling_distribution(data, window=250, step=25, alpha=0.01,
                             notional=1_000_000_000.0) -> dict:
    _ensure_dir()
    dists = {}
    samples = []
    labels = []
    for a in ASSETS:
        ds = data[a]
        dist = rolling_mul(ds.returns, ds.true_var, asset=a, window=window,
                           step=step, panel=gates_mod.PANEL, alpha=alpha,
                           notional=notional)
        dists[a] = dist
        vals = [w.mul_bps for w in dist.windows if w.honest_passes]
        samples.append(np.asarray(vals, dtype=float))
        labels.append(f"{a}\n({dist.n_definable}/{dist.n_windows} definable)")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    positions = np.arange(1, len(ASSETS) + 1)
    parts = ax.violinplot(samples, positions=positions, showextrema=False,
                          widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor(BLUE)
        pc.set_alpha(0.20)
        pc.set_edgecolor(BLUE)
    bp = ax.boxplot(samples, positions=positions, widths=0.28, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=2),
                    boxprops=dict(facecolor=CREAM, edgecolor=INK),
                    whiskerprops=dict(color=INK), capprops=dict(color=INK),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=GREY,
                                    markeredgecolor="none", alpha=0.6))
    for i, s in enumerate(samples):
        med = float(np.median(s))
        ax.annotate(f"median {med:.0f}", (positions[i], med),
                    textcoords="offset points", xytext=(-24, 0),
                    ha="right", fontsize=9, color=INK, va="center")
    # widen x-limits so the leftmost median label is not clipped at the axis edge
    ax.set_xlim(positions[0] - 0.9, positions[-1] + 0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Maximum Undetectable Lie (bps of risk understatement)",
                  fontsize=10)
    ax.set_title("Rolling MUL distribution per asset, 4-gate panel\n"
                 f"(99% VaR, GARCH reference, {window}-day window, step {step}; "
                 "definable windows only)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_rolling_mul_distribution.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": path, "dists": dists,
            "medians": {a: float(np.median(s)) for a, s in zip(ASSETS, samples)}}


# ---------------------------------------------------------------------------
# (b) gate-contribution / gameability ranking
# ---------------------------------------------------------------------------

def fig_gate_contribution(data, window=250, step=25, alpha=0.01,
                          notional=1_000_000_000.0) -> dict:
    _ensure_dir()
    decomps = {}
    for a in ASSETS:
        ds = data[a]
        ws, we = representative_window(ds.returns, ds.true_var, window=window,
                                       step=step, panel=gates_mod.PANEL, alpha=alpha)
        d = leave_one_out(ds.returns, ds.true_var, asset=a, window_start=ws,
                          window_end=we, panel=gates_mod.PANEL, alpha=alpha,
                          notional=notional)
        decomps[a] = d

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=False)
    for ax, a in zip(axes, ASSETS):
        d = decomps[a]
        # ranked desc by contribution already; show grows-lie (LOO) as the bar,
        # annotate solo MUL. short gate names.
        gnames = [c.gate_name.replace("_", "\n") for c in d.contributions]
        grows = [c.contribution_bps for c in d.contributions]
        colors = []
        for c in d.contributions:
            if c.contribution_bps > 1e-6:
                colors.append(BLUE if c is d.contributions[0] else GREEN)
            else:
                colors.append(GREY)
        ypos = np.arange(len(gnames))[::-1]
        ax.barh(ypos, grows, color=colors, alpha=0.85, edgecolor=INK, linewidth=0.6)
        for y, c in zip(ypos, d.contributions):
            ax.annotate(f"grows {c.contribution_bps:.2f}\n(solo {c.mul_solo_bps:.0f})",
                        (max(c.contribution_bps, 0.0), y),
                        textcoords="offset points", xytext=(6, 0),
                        fontsize=8, va="center", color=INK)
        ax.set_yticks(ypos)
        ax.set_yticklabels(gnames, fontsize=8)
        ax.set_title(f"{a}  [{d.window_start}:{d.window_end}]\n"
                     f"full-panel MUL {d.mul_full_bps:.1f} bps", fontsize=10)
        ax.set_xlabel("lie growth when gate removed (bps)", fontsize=9)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        # headroom for annotations
        xmax = max([c.contribution_bps for c in d.contributions] + [1.0])
        ax.set_xlim(0, xmax * 2.0)
    fig.suptitle("Leave-one-out gate contribution (MECHANISM ONLY — the "
                 "'binder/dead-weight' reading is a TUNING ARTIFACT of the "
                 "author-added\nconformal tolerance; see fig_tolerance_map for the "
                 "honest finding). conformal is non-regulatory.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(FIG_DIR, "fig_gate_contribution_ranking.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": path, "decomps": decomps}


# ---------------------------------------------------------------------------
# (c) attack vs defense lie budget with/without the e-process gate
# ---------------------------------------------------------------------------

def _eprocess_trajectory(loss, stated_var, alpha, level=0.95):
    """Mixture e-process trajectory M_t for the figure (mirrors ebacktest)."""
    from .ebacktest import MIXTURE_GRID
    ratio = gaussian_es_var_ratio(alpha)
    e = es_e_statistic(loss, stated_var, ratio * stated_var, alpha)
    lams = np.asarray(MIXTURE_GRID, dtype=float)
    comp = np.ones(lams.size)
    w = np.full(lams.size, 1.0 / lams.size)
    traj = np.empty(e.size)
    for t in range(e.size):
        comp = np.maximum(comp * (1.0 - lams + lams * e[t]), 0.0)
        traj[t] = float(np.dot(w, comp))
    return traj


def fig_attack_vs_defense(data, window=250, step=25, alpha=0.01,
                          notional=1_000_000_000.0, level=0.95) -> dict:
    _ensure_dir()
    panel4 = gates_mod.PANEL
    panelE = gates_mod.PANEL_WITH_EBT
    results = {}
    for a in ASSETS:
        ds = data[a]
        ws, we = representative_window(ds.returns, ds.true_var, window=window,
                                       step=step, panel=panel4, alpha=alpha)
        rr, tv = ds.returns[ws:we], ds.true_var[ws:we]
        r4 = solve(tv, rr, panel=panel4, alpha=alpha)
        p4 = price(r4.stated_var, tv, rr, notional=notional)
        rE = solve(tv, rr, panel=panelE, alpha=alpha)
        pE = price(rE.stated_var, tv, rr, notional=notional)
        traj = _eprocess_trajectory(-rr, r4.stated_var, alpha, level=level)
        results[a] = {"window": (ws, we), "mul4": p4.mul_bps, "mulE": pE.mul_bps,
                      "shrink": p4.mul_bps - pE.mul_bps, "traj": traj,
                      "threshold": 1.0 / (1.0 - level)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})

    # left: grouped bars, 4-gate vs 4+e-backtest MUL
    x = np.arange(len(ASSETS))
    w = 0.36
    mul4 = [results[a]["mul4"] for a in ASSETS]
    mulE = [results[a]["mulE"] for a in ASSETS]
    b1 = axL.bar(x - w / 2, mul4, w, label="4-gate panel (gameable)",
                 color=RED, alpha=0.85, edgecolor=INK, linewidth=0.6)
    b2 = axL.bar(x + w / 2, mulE, w, label="4-gate + e-backtest (defended)",
                 color=GREEN, alpha=0.9, edgecolor=INK, linewidth=0.6)
    for a, xi in zip(ASSETS, x):
        sh = results[a]["shrink"]
        top = max(results[a]["mul4"], results[a]["mulE"])
        axL.annotate(f"-{sh:.2f} bps\n(-{100*sh/results[a]['mul4']:.1f}%)",
                     (xi, top), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8, color=INK)
    axL.bar_label(b1, fmt="%.1f", padding=2, fontsize=7)
    axL.bar_label(b2, fmt="%.1f", padding=2, fontsize=7)
    axL.set_xticks(x)
    axL.set_xticklabels(ASSETS)
    axL.set_ylabel("per-point exhibited lie (bps)", fontsize=10)
    axL.set_title("Attack vs defense: lie budget\nwith/without the e-process gate",
                  fontsize=11)
    axL.legend(fontsize=8, frameon=False, loc="upper left")
    axL.grid(axis="y", alpha=0.25)
    axL.spines[["top", "right"]].set_visible(False)

    # right: e-process trajectory on the gamed 4-gate series, log-y, crossing 1/alpha
    cols = {"AAPL": BLUE, "SPY": AMBER, "BTC-USD": GREEN}
    thr = results[ASSETS[0]]["threshold"]
    for a in ASSETS:
        tr = results[a]["traj"]
        axR.plot(np.arange(tr.size), tr, color=cols[a], lw=1.8,
                 label=f"{a} (max {tr.max():.0f})")
    axR.axhline(thr, color=RED, ls="--", lw=1.4,
                label=f"reject threshold 1/(1-level) = {thr:.0f}")
    axR.set_yscale("log")
    axR.set_xlabel("trading day within window", fontsize=10)
    axR.set_ylabel("e-process M_t  (log scale)", fontsize=10)
    axR.set_title("e-process catching the 4-gate lie\n"
                  "(Wang-Wang-Ziegel ES e-backtest, anytime-valid)", fontsize=11)
    axR.legend(fontsize=8, frameon=False, loc="lower right")
    axR.grid(alpha=0.25, which="both")
    axR.spines[["top", "right"]].set_visible(False)

    fig.suptitle("The e-process ES backtest barely dents the per-point lie "
                 "but catches every count-only gamed series", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(FIG_DIR, "fig_attack_vs_defense.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": path, "results": results}


# ---------------------------------------------------------------------------
# (d) the Cycle-7 tolerance map — which gate binds vs the conformal tolerance
# ---------------------------------------------------------------------------

def fig_tolerance_map(data, window=250, step=25, alpha=0.01,
                      notional=1_000_000_000.0) -> dict:
    """The honest replacement for the 'dead weight' figure: full-panel MUL vs the
    conformal tolerance, with the conformal-binding zone shaded and the crossover
    to the regulatory binder marked, per asset."""
    from .tolerance_map import tolerance_map
    _ensure_dir()
    maps = {}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=False)
    for ax, a in zip(axes, ASSETS):
        ds = data[a]
        ws, we = representative_window(ds.returns, ds.true_var, window=window,
                                       step=step, panel=gates_mod.PANEL, alpha=alpha)
        tm = tolerance_map(ds.returns, ds.true_var, asset=a, window_start=ws,
                           window_end=we, alpha=alpha, notional=notional)
        maps[a] = tm
        tols = [r.tol for r in tm.rows]
        muls = [r.mul_full_bps for r in tm.rows]
        binds = [r.conformal_binds for r in tm.rows]
        ax.plot(tols, muls, "-o", color=INK, markersize=4, linewidth=1.4)
        # shade the conformal-binding (artifact) region and the regulatory region
        for r in tm.rows:
            c = AMBER if r.conformal_binds else GREEN
            ax.axvspan(r.tol * 0.93, r.tol * 1.07, color=c, alpha=0.10)
        ax.axvline(0.005, color=RED, ls="--", lw=1.0)
        ax.annotate("default\ntol=0.005", (0.005, ax.get_ylim()[0]),
                    fontsize=7, color=RED, ha="center", va="bottom")
        if tm.crossover_tol is not None:
            ax.axvline(tm.crossover_tol, color=BLUE, ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_title(f"{a}  [{ws}:{we}]\n"
                     f"conformal binds <= {tm.crossover_tol or 'all'}; then "
                     f"{tm.regulatory_binding_gate or 'n/a'}", fontsize=9)
        ax.set_xlabel("conformal tolerance (log)", fontsize=9)
        ax.set_ylabel("full-panel MUL (bps)", fontsize=9)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tolerance map — WHICH test is load-bearing is a TUNING CHOICE. "
                 "Amber = author-added conformal binds (tighter than the regulatory\n"
                 "band); green = a REGULATORY gate (Basel) binds. Red = the hand-set "
                 "default tol=0.005, which sits in the artifact zone.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = os.path.join(FIG_DIR, "fig_tolerance_map.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": path, "maps": maps}


def make_all(method="rolling_garch", alpha=0.01, window=250, step=25,
             notional=1_000_000_000.0) -> dict:
    data = _load_all(method=method, alpha=alpha)
    out = {}
    out["a"] = fig_rolling_distribution(data, window=window, step=step, alpha=alpha,
                                        notional=notional)
    out["b"] = fig_gate_contribution(data, window=window, step=step, alpha=alpha,
                                     notional=notional)
    out["c"] = fig_attack_vs_defense(data, window=window, step=step, alpha=alpha,
                                     notional=notional)
    out["d"] = fig_tolerance_map(data, window=window, step=step, alpha=alpha,
                                 notional=notional)
    print("Figures written:")
    for k in ("a", "b", "c", "d"):
        print("  " + out[k]["path"])
    return out


if __name__ == "__main__":
    make_all()
