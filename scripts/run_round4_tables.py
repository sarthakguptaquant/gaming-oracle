"""Round 4 (revision R1, referee-style review pass): tables built from code output.

Run from the repo root, after run_round2_exact.py and run_round3_sim.py (part S2) have
written their JSON files to the out directory:

    PYTHONPATH=. uv run python scripts/run_round4_tables.py [out_dir]

Inputs (out_dir, default paper/revision-R1/round4):
    exact.json                  exact grid (run_round2_exact.py)
    round3_sim_S2.json          clustered and slow-model simulation (run_round3_sim.py ... S2)
Computed here, with fixed seeds:
    count-feedback simulation   gaming_oracle.feasibility.run_e5, seed SeedSequence([5424, 5, 9, 1]),
                                5,000 Gaussian GARCH paths (AAPL parameters), exactly as in the R1 study
    exact constant-understatement comparators, desk of statements checked against the text

Writes LaTeX fragments to paper/tables/ and a log to out_dir/tables.log. All percentages are
printed in percent; formulas in the paper keep fractions.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

from gaming_oracle import mdu as M

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper/revision-R1/round4"
TABDIR = "paper/tables"
os.makedirs(TABDIR, exist_ok=True)
ex = json.load(open(os.path.join(OUT, "exact.json")))
S2 = json.load(open(os.path.join(OUT, "round3_sim_S2.json")))["S2"]
LOG = open(os.path.join(OUT, "tables.log"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


def write_rows(name, rows):
    """Write table rows; the last row has no trailing row separator (the paper adds it)."""
    text = "\n".join(rows).rstrip()
    assert text.endswith("\\\\")
    open(f"{TABDIR}/{name}", "w").write(text[:-2].rstrip() + "\n")


def psize(x):
    return "$<$0.1" if x < 0.001 else pc(x)


def pc(x, d=1):
    return f"{100 * x:.{d}f}"


TS = (250, 500, 750, 1000)
PANEL_NAME = {"K": "POF", "B": "Traffic light", "KC": "POF + Christ.", "KCB": "Standard panel"}

# ------------------------------------------------------------------ Table: exact grid, 99%
# one row per (T, panel); strict block then lenient block
rows = []
for T in TS:
    blk = ex["N1"][f"T={T}|alpha=0.01"]["cells"]
    first = True
    for panel in ("K", "B", "KC", "KCB"):
        cells = []
        for rule in ("strict", "lenient"):
            c = blk[f"{panel}|{rule}"]
            cells += [psize(c["size"]), pc(c["p*_0.5"], 2), pc(c["u*_0.5"]["gaussian"]),
                      pc(c["p*_0.8"], 2), pc(c["u*_0.8"]["gaussian"])]
        rows.append(f"{format(T, ',') if first else ''} & {PANEL_NAME[panel]} & " + " & ".join(cells) + " \\\\")
        first = False
    if T != TS[-1]:
        rows.append("\\addlinespace")
write_rows("tab_mdu99_rows.tex", rows)
log("wrote tab_mdu99_rows.tex (", len(rows), "lines )")

# ------------------------------------------------------------------ Table: 97.5% and tails
rows = []
for panel, rule in (("K", "strict"), ("KC", "strict"), ("KC", "lenient")):
    sizes = ex["N1"]["T=250|alpha=0.025"]["cells"][f"{panel}|{rule}"]["size"]
    cells = []
    for T in TS:
        c = ex["N1"][f"T={T}|alpha=0.025"]["cells"][f"{panel}|{rule}"]
        cells.append(f"{pc(c['u*_0.5']['gaussian'])} / {pc(c['u*_0.8']['gaussian'])}")
    rows.append(f"{PANEL_NAME[panel]} & {rule} ({pc(sizes)}) & " + " & ".join(cells) + " \\\\")
write_rows("tab_tails_a_rows.tex", rows)
rows = []
c250 = ex["N1"]["T=250|alpha=0.01"]["cells"]
for lab, key in (("Gaussian", "gaussian"),                  ("Student-$t$, $\\nu = 6.71$ (SPY fit)", "t6.71 (SPY fit)"),
                 ("Student-$t$, $\\nu = 4.97$ (AAPL fit)", "t4.97 (AAPL fit)"),
                 ("Student-$t$, $\\nu = 5$", "t5"),
                 ("Student-$t$, $\\nu = 3.16$ (BTC-USD fit)", "t3.16 (BTC-USD fit)"),
                 ("Student-$t$, $\\nu = 3$", "t3")):
    s, l = c250["KCB|strict"], c250["KCB|lenient"]
    rows.append(f"{lab} & {pc(s['u*_0.5'][key])} & {pc(s['u*_0.8'][key])} & "
                f"{pc(l['u*_0.5'][key])} & {pc(l['u*_0.8'][key])} \\\\")
write_rows("tab_tails_b_rows.tex", rows)
log("sizes at 97.5%, strict / lenient POF+C, T = 250..1000:")
for T in TS:
    b = ex["N1"][f"T={T}|alpha=0.025"]["cells"]
    log(f"  T={T}: K strict {b['K|strict']['size']:.4f}, KC strict {b['KC|strict']['size']:.4f}, "
        f"KC lenient {b['KC|lenient']['size']:.4f}")

# ------------------------------------------------------------------ statements about Christoffersen at nominal size
log("\nChristoffersen effect at nominal size: u*(POF) - u*(POF + Christoffersen), percentage points")
mx = {}
for alpha in (0.01, 0.025):
    for T in TS:
        b = ex["N1"][f"T={T}|alpha={alpha}"]["cells"]
        for rule in ("strict", "lenient"):
            for tail in ("gaussian", "t3"):
                for beta in ("0.5", "0.8"):
                    d = 100 * (b[f"K|{rule}"][f"u*_{beta}"][tail] - b[f"KC|{rule}"][f"u*_{beta}"][tail])
                    mx.setdefault(tail, []).append((d, alpha, T, rule, beta))
for tail, v in mx.items():
    log(f"  {tail}: max {max(v)[0]:.3f} {max(v)[1:]}, min {min(v)[0]:.3f} {min(v)[1:]}")

# ------------------------------------------------------------------ capital factor and zone
n7 = ex["N7"]
log("\nN7 (exact): u, E[multiplier], capital factor, P(amber or red)")
for k, v in n7.items():
    if k.startswith("gaussian") or k == "t5|u=0.1":
        log(f"  {k}: E[m] {v['E_multiplier']:.4f}, factor {v['capital_factor']:.4f}, "
            f"P(amber/red) {v['P_amber_or_red']:.4f}, P(red) {v['P_red']:.4f}")

# ------------------------------------------------------------------ slow model (EWMA 0.99) table
ALPHA = 0.01
rows = []
log("\nSlow model, EWMA 0.99, no cut: seed, mean understatement, strict rejection of POF and POF + Christoffersen")
for T in (250, 1000):
    ref = ex["N1"][f"T={T}|alpha=0.01"]["cells"]
    ks, kcs = ref["K|strict"]["size"], ref["KC|strict"]["size"]
    first = True
    for a in ("AAPL", "SPY", "BTC-USD"):
        r = S2[f"T={T}|{a}|slow"]
        assert abs(r["c_grid"][0]) < 1e-12
        mu0 = r["mean_under"][0]
        k0, kc0 = r["curves"]["K|strict"][0], r["curves"]["KC|strict"][0]
        rows.append(f"{format(T, ',') if first else ''} & {a} & {pc(-mu0)} & {pc(k0)} & {pc(kc0)} \\\\")
        log(f"  T={T} {a}: seed {r['seed']}, VaR above true quantile by {pc(-mu0)}%, POF {pc(k0)}%, "
            f"POF+C {pc(kc0)}%, gain {pc(kc0 - k0)} points; size POF {pc(ks)}%, POF+C {pc(kcs)}%")
        first = False
    rows.append(f" & Correct forecast, exact & 0.0 & {pc(ks)} & {pc(kcs)} \\\\")
    if T == 250:
        rows.append("\\addlinespace")
write_rows("tab_slow_rows.tex", rows)

# share of days cut in the 80th percentile variant, and lag-1 autocorrelations
for var in ("median", "q80"):
    sh = [S2[f"T=250|{a}|{var}"]["share_cut"] for a in ("AAPL", "SPY", "BTC-USD")]
    log(f"share of days cut, variant {var}, T=250: " + ", ".join(f"{x:.4f}" for x in sh))

# ------------------------------------------------------------------ count feedback
from gaming_oracle import feasibility as F

log("\nCount feedback (run_e5, seed [5424, 5, 9, 1], 5,000 paths, AAPL GARCH parameters, Gaussian)")
setups = [F.setup_asset(a, log=log) for a in F.ASSETS]
e5 = F.run_e5(setups, n_paths=5000, log=log)
cfg = e5["configs"]["D1/R1 pooled"]
log("seed", cfg["seed"], "n_paths", cfg["n_paths"], "S0 rejection (P3 = strict standard panel):", cfg["S0"]["P3"])
rows = []
cmax = []
for k in ("2", "3", "4"):
    rec = cfg["k"][k]
    cg = [0.0] + e5["c_grid"]
    for i, c in enumerate(cg):
        if i == 0:
            continue
        u = rec["understatement_frac"][i]
        const = M.reject_probs(M.p_from_understatement(u, ALPHA), 250, ALPHA)[("KCB", "strict")]
        rej = rec["rej"]["P3"][i]
        rows.append((int(k), c, u, rej, const, rec["rej"]["e_backtest"][i]))
        if rej <= 0.5:
            cmax.append((u, int(k), c))
log("k, c, mean understatement, strict standard panel rejection, exact constant understatement of the same mean, e-backtest rejection")
for r in rows:
    log(f"  k={r[0]} c={r[1]:.2f} under {r[2]:.4f} S2 {r[3]:.3f} const {r[4]:.3f} ebt {r[5]:.3f}")
log("largest mean understatement among settings with strict-panel rejection <= 0.50:", max(cmax))
show = [(2, 0.05), (3, 0.20), (4, 0.10), (4, 0.20), (4, 0.50)]
tex = []
for k, c in show:
    r = [x for x in rows if x[0] == k and abs(x[1] - c) < 1e-9][0]
    tex.append(f"{k} & {pc(c, 0)} & {pc(r[2])} & {pc(r[3])} & {pc(r[4])} & {pc(r[5])} \\\\")
write_rows("tab_countfb_rows.tex", tex)
json.dump({"seed": cfg["seed"], "S0": cfg["S0"], "rows": rows}, open(os.path.join(OUT, "count_feedback.json"), "w"))

# ------------------------------------------------------------------ Campbell (2005) comparison, one convention per row
log("\nCampbell (2005) Table 3 comparison, T = 255, 99%, Gaussian; exact POF rejection (percent)")
sim = {5: 6.3, 10: 19.4, 15: 43.8, 20: 69.0, 25: 79.7}
z = stats.norm.isf(0.01)
for u in sim:
    p = float(stats.norm.sf((1 - u / 100) * z))
    two = M.reject_prob(p, 255, 0.01, "K", "strict")
    up = M.reject_prob(p, 255, 0.01, "K", "lenient")
    se = np.sqrt(sim[u] / 100 * (1 - sim[u] / 100) / 1000) * 100
    log(f"  u={u}%: two-sided {100*two:.1f}, upper only {100*up:.1f}, simulated {sim[u]} (se {se:.2f}); "
        f"diff two {100*two - sim[u]:+.1f} ({(100*two - sim[u])/se:+.1f} se), upper {100*up - sim[u]:+.1f} ({(100*up - sim[u])/se:+.1f} se)")
LOG.close()
