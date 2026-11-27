"""Round 2, exact computations N1, N2, N6 and N7 (MS 5424 revision).

Run from the repo root:
    uv run python scripts/run_round2_exact.py [out_dir]

N1  exact minimum detectable exception probability p*_beta and the size of each
    panel, for T in {250, 500, 750, 1000}, alpha in {0.01, 0.025},
    beta in {0.5, 0.8}, panels K, B, KC, KCB, strict and lenient rules.
N2  minimum detectable understatement u*_beta = 1 - q(p*)/q(alpha) for Gaussian
    and Student-t tails (nu = 3, 5, 10 and the fitted 4.97, 6.71, 3.16).
N6  checks: (a) T = 250 cells against the R1 values; (b) Monte Carlo agreement at
    T = 1000 and at T = 500, alpha = 0.025; (c) exact two-sided Kupiec power at
    the under-reporting levels of Campbell (2005, Table 3): T = 255, Gaussian.
N7  expected Basel multiplier and expected capital factor of a constant
    understatement (MAR32.9 multipliers, T = 250, alpha = 0.01).
Also writes the exact curves used by Figure 1 and draws the figure.

Monte Carlo seeds: SeedSequence([5424, 30, k]) for check k.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import stats

from gaming_oracle import mdu as M
from gaming_oracle.feasibility import HitEvaluator

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper/revision-R1/round2"
os.makedirs(OUT, exist_ok=True)
FIG = "figures/fig_mdu_curves.pdf"
TS = (250, 500, 750, 1000)
ALPHAS = (0.01, 0.025)
BETAS = (0.5, 0.8)
TAILS = ("gaussian", 3.0, 5.0, 10.0, 4.972333080308459, 6.709064545105986, 3.1648415166169968)
TAIL_NAMES = {3.0: "t3", 5.0: "t5", 10.0: "t10", 4.972333080308459: "t4.97 (AAPL fit)",
              6.709064545105986: "t6.71 (SPY fit)", 3.1648415166169968: "t3.16 (BTC-USD fit)"}
res: dict = {"N1": {}, "N2": {}, "N6": {}, "N7": {}, "curves": {}}
t0 = time.time()


def tname(t):
    return "gaussian" if t == "gaussian" else TAIL_NAMES[t]


# ---------------------------------------------------------------- N1 and N2
for alpha in ALPHAS:
    for T in TS:
        tab = M.panel_tables(T, alpha)
        grid = M.p_grid(alpha, p_max=0.25, n_grid=300)
        cur = M.power_curves(T, alpha, grid)
        size = M.reject_probs(alpha, T, alpha)
        key = f"T={T}|alpha={alpha}"
        res["N1"][key] = {"kupiec_band": tab.kupiec_band, "traffic_light_amber_red": tab.tl_bounds,
                          "xc": tab.xc, "cells": {}}
        for (panel, rule), vals in cur.items():
            cell = {"size": size[(panel, rule)]}
            for b in BETAS:
                ps = M.min_detectable_p(T, alpha, panel, rule, b, grid=grid, curve=vals)
                cell[f"p*_{b}"] = ps
                cell[f"u*_{b}"] = {tname(t): (None if ps is None else M.mdu_from_p(ps, alpha, t))
                                   for t in TAILS}
            res["N1"][key]["cells"][f"{panel}|{rule}"] = cell
        print(f"{key}: done ({time.time() - t0:.1f}s); KC strict size "
              f"{res['N1'][key]['cells']['KC|strict']['size']:.4f}", flush=True)

# printed table, Gaussian
print("\nN1/N2: p* and Gaussian u* (beta 0.5 / 0.8), size")
for key, blk in res["N1"].items():
    for cell, v in blk["cells"].items():
        def f(x):
            return "-" if x is None else f"{x:.4f}"
        print(f"  {key:22s} {cell:12s} size {v['size']:.4f}  p* {f(v['p*_0.5'])} / {f(v['p*_0.8'])}"
              f"  u*G {f(v['u*_0.5']['gaussian'])} / {f(v['u*_0.8']['gaussian'])}"
              f"  u*t5 {f(v['u*_0.5']['t5'])} / {f(v['u*_0.8']['t5'])}"
              f"  u*t3 {f(v['u*_0.5']['t3'])} / {f(v['u*_0.8']['t3'])}")

# ---------------------------------------------------------------- N6a
z = stats.norm.isf(0.01)
r250 = {c: M.reject_probs(float(stats.norm.sf((1 - c) * z)), 250, 0.01) for c in (0.0, 0.05, 0.10, 0.20)}
n6a = {"P3_size": r250[0.0][("KCB", "strict")], "P3_c0.05": r250[0.05][("KCB", "strict")],
       "P3_c0.10": r250[0.10][("KCB", "strict")], "P3_c0.20": r250[0.20][("KCB", "strict")],
       "P3L_size": r250[0.0][("KCB", "lenient")], "Kupiec_size": r250[0.0][("K", "strict")],
       "Kupiec_c0.05": r250[0.05][("K", "strict")],
       "green_size": r250[0.0][("B", "strict")], "red_size": r250[0.0][("B", "lenient")],
       "c*_P3_half": res["N1"]["T=250|alpha=0.01"]["cells"]["KCB|strict"]["u*_0.5"]["gaussian"],
       "c*_P3L_half": res["N1"]["T=250|alpha=0.01"]["cells"]["KCB|lenient"]["u*_0.5"]["gaussian"]}
R1 = {"P3_size": 0.2024, "P3_c0.05": 0.3015, "P3_c0.10": 0.4998, "P3_c0.20": 0.8987,
      "P3L_size": 0.0276, "Kupiec_size": 0.095, "Kupiec_c0.05": 0.089, "green_size": 0.108,
      "red_size": 0.0003, "c*_P3_half": 0.1001}
n6a_ok = {k: abs(n6a[k] - R1[k]) <= 0.5 * 10 ** -len(str(R1[k]).split('.')[1]) + 1e-12 for k in R1}
res["N6"]["a_T250_vs_R1"] = {"values": n6a, "R1": R1, "match": n6a_ok}
print("\nN6a T=250 against R1:", {k: (round(v, 4), n6a_ok.get(k)) for k, v in n6a.items()})


# ---------------------------------------------------------------- N6b Monte Carlo
def mc_check(T, alpha, panel, rule, p, n, k):
    rng = np.random.default_rng(np.random.SeedSequence([5424, 30, k]))
    hits = rng.random((n, T)) < p
    ev = HitEvaluator(T, alpha=alpha)
    f = ev.evaluate(hits)
    x = f["x"]
    if rule == "strict":
        ok = f["kupiec_pof"] & (f["christoffersen"] if "C" in panel else True)
    else:
        ok = f["kupiec_dir"] & (f["christoffersen_dir"] if "C" in panel else True)
    if "B" in panel:
        amber, red = M.traffic_light_bounds(T)
        ok = ok & (x < (amber if rule == "strict" else red))
    mc = float(1 - ok.mean())
    ex = M.reject_prob(p, T, alpha, panel, rule)
    se = float(np.sqrt(ex * (1 - ex) / n))
    return {"T": T, "alpha": alpha, "panel": panel, "rule": rule, "p": p, "n_paths": n,
            "seed": [5424, 30, k], "exact": ex, "monte_carlo": mc, "se": se,
            "z": (mc - ex) / se}


checks = []
k = 0
for (T, alpha, panel, rule) in ((1000, 0.01, "KC", "strict"), (1000, 0.01, "KC", "lenient"),
                                (1000, 0.01, "KCB", "strict"), (500, 0.025, "KC", "strict"),
                                (250, 0.01, "KCB", "strict")):
    for b in BETAS:
        k += 1
        ps = res["N1"][f"T={T}|alpha={alpha}"]["cells"][f"{panel}|{rule}"][f"p*_{b}"]
        checks.append(mc_check(T, alpha, panel, rule, ps, 20000, k))
        c = checks[-1]
        print(f"N6b {panel} {rule} T={T} a={alpha} beta={b}: exact {c['exact']:.4f} "
              f"MC {c['monte_carlo']:.4f} (se {c['se']:.4f}, z {c['z']:+.2f})", flush=True)
res["N6"]["b_monte_carlo"] = checks

# ---------------------------------------------------------------- N6c Campbell
camp = {0.05: 6.30, 0.10: 19.4, 0.15: 43.8, 0.20: 69.0, 0.25: 79.7}
n6c = {}
for T in (255, 250):
    for u, v in camp.items():
        p = float(stats.norm.sf((1 - u) * z))
        n6c[f"T={T}|u={u}"] = {"exact_kupiec_two_sided": M.reject_prob(p, T, 0.01, "K", "strict"),
                               "exact_kupiec_upper_only": M.reject_prob(p, T, 0.01, "K", "lenient"),
                               "campbell_table3_pct": v, "campbell_mc_se_pct":
                               100 * float(np.sqrt(v / 100 * (1 - v / 100) / 1000))}
res["N6"]["c_campbell"] = n6c
res["N6"]["c_kupiec_band_T255"] = M.panel_tables(255, 0.01).kupiec_band
print("\nN6c Campbell (2005) Table 3, Kupiec, Gaussian (exact two-sided / upper only vs Campbell):")
for kk, v in n6c.items():
    print(f"  {kk}: {100 * v['exact_kupiec_two_sided']:.1f} / {100 * v['exact_kupiec_upper_only']:.1f}"
          f" vs {v['campbell_table3_pct']} (MC se {v['campbell_mc_se_pct']:.1f})")

# ---------------------------------------------------------------- N7
n7 = {}
for tail in ("gaussian", 5.0):
    for u in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        p = M.p_from_understatement(u, 0.01, tail)
        n7[f"{tname(tail)}|u={u}"] = {"p": p, "E_multiplier": M.expected_multiplier(p),
                                      "capital_factor": M.expected_capital_factor(u, tail=tail),
                                      "P_red": float(stats.binom.sf(9, 250, p)),
                                      "P_amber_or_red": float(stats.binom.sf(4, 250, p))}
res["N7"] = n7
print("\nN7 expected multiplier and capital factor (MAR32.9 multipliers, T = 250):")
for kk, v in n7.items():
    print(f"  {kk}: p {v['p']:.4f} E[m] {v['E_multiplier']:.4f} factor {v['capital_factor']:.4f} "
          f"P(amber or red) {v['P_amber_or_red']:.3f} P(red) {v['P_red']:.4f}")

# ---------------------------------------------------------------- figure data
ugrid = np.round(np.arange(0.0, 0.4001, 0.005), 3)
curves = {}
for T in TS:
    rows = [M.reject_probs(float(stats.norm.sf((1 - u) * z)), T, 0.01) for u in ugrid]
    for key in (("KCB", "strict"), ("KCB", "lenient"), ("K", "strict"), ("B", "strict")):
        curves[f"T={T}|{key[0]}|{key[1]}"] = [r[key] for r in rows]
res["curves"] = {"u_grid_gaussian": ugrid.tolist(), "pi": curves}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
# Black-and-white safe: every curve has its own dash pattern, marker and width.
styles = {("KCB", "strict"): ("-", "o", 1.5, "Standard panel, strict"),
          ("KCB", "lenient"): ((0, (6, 2)), "s", 1.2, "Standard panel, lenient"),
          ("K", "strict"): ((0, (1, 1.5)), "^", 1.2, "POF test alone, strict"),
          ("B", "strict"): ((0, (6, 2, 1, 2)), "D", 1.2, "Traffic light, green zone only")}
for i, (key, (ls, mk, lw, lab)) in enumerate(styles.items()):
    ax[0].plot(100 * ugrid, curves[f"T=250|{key[0]}|{key[1]}"], ls=ls, color="black", lw=lw,
               marker=mk, markevery=(2 * i, 8), ms=4, markerfacecolor="white", label=lab)
u5 = 100 * res["N1"]["T=250|alpha=0.01"]["cells"]["KCB|strict"]["u*_0.5"]["gaussian"]
u8 = 100 * res["N1"]["T=250|alpha=0.01"]["cells"]["KCB|strict"]["u*_0.8"]["gaussian"]
for b, u in ((0.5, u5), (0.8, u8)):
    ax[0].plot([u], [b], "o", color="black", ms=6, zorder=5)
ax[0].set_title("(a) T = 250 days, 99% VaR")
ax[0].set_xlabel("Understatement of VaR (%)")
ax[0].set_ylabel("Rejection probability")
ax[0].legend(frameon=False, fontsize=7, loc="lower right", handlelength=3.2)
wstyle = {250: ("-", "o"), 500: ((0, (6, 2)), "s"), 750: ((0, (1, 1.5)), "^"),
          1000: ((0, (6, 2, 1, 2)), "D")}
for i, T in enumerate(TS):
    ls, mk = wstyle[T]
    ax[1].plot(100 * ugrid, curves[f"T={T}|KCB|strict"], ls=ls, color="black", lw=1.3,
               marker=mk, markevery=(2 * i, 8), ms=4, markerfacecolor="white", label=f"T = {T:,}")
ax[1].set_title("(b) Strict standard panel by $T$")
ax[1].set_xlabel("Understatement of VaR (%)")
ax[1].legend(frameon=False, fontsize=7, loc="lower right", handlelength=3.2)
for a in ax:
    a.axhline(0.5, color="0.6", lw=0.5, ls=":")
    a.axhline(0.8, color="0.6", lw=0.5, ls=":")
    a.set_xlim(0, 40)
    a.set_ylim(0, 1)
fig.tight_layout()
os.makedirs(os.path.dirname(FIG), exist_ok=True)
fig.savefig(FIG)
print("wrote", FIG)

res["runtime_s"] = time.time() - t0
with open(os.path.join(OUT, "exact.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"wrote {OUT}/exact.json in {res['runtime_s']:.1f}s")
