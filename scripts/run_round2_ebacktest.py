"""Round 2, N3: minimum detectable understatement of the ES e-backtest by simulation.

Run from the repo root:
    PYTHONPATH=. uv run python scripts/run_round2_ebacktest.py [n_mix] [n_grem] [out_dir]

Design. With the reference equal to z_alpha times the true conditional volatility
and a constant understatement c, every daily e-value and every exception depends
only on the standardized shock of that day. A GARCH path with a true-volatility
reference therefore gives the same rejection law as i.i.d. shocks with unit
volatility, which is what is simulated (Gaussian shocks, ES tied to VaR by the
Gaussian ratio, stated VaR (1 - c) z_alpha, stated ES kappa (1 - c) z_alpha).
One path of max(T) days per draw; rejection by horizon T means the e-process
crosses 1/0.05 = 20 before day T. Count panels use the first T days of the same
path (common random numbers across c and across tests).

Runs
  A  alpha = 0.01, T in {250, 500, 750, 1000}: mixture e-backtest; Kupiec,
     Kupiec + Christoffersen + traffic light (strict and lenient) with and without
     the e-backtest at T = 250.        seeds SeedSequence([5424, 31, 1])
  B  alpha = 0.025, T = 250: mixture e-backtest; Kupiec + Christoffersen strict and
     lenient, with and without it.     seeds SeedSequence([5424, 31, 2])
  C  alpha = 0.01, T in {250, 500}: GREM adaptive betting (Wang, Wang and Ziegel,
     recommended default).             seeds SeedSequence([5424, 31, 3])
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import stats

from gaming_oracle import mdu as M
from gaming_oracle.ebacktest import gaussian_es_var_ratio
from gaming_oracle.ebetting import eprocess_adaptive_batch
from gaming_oracle.feasibility import HitEvaluator, eprocess_batch

N_MIX = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
N_GREM = int(sys.argv[2]) if len(sys.argv) > 2 else 5_000
OUT = sys.argv[3] if len(sys.argv) > 3 else "paper/revision-R1/round2"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
res: dict = {"n_mix": N_MIX, "n_grem": N_GREM}


def invert(cgrid, pi, beta):
    """Smallest c with pi >= beta at every larger grid c; linear interpolation."""
    pi = np.asarray(pi)
    below = np.nonzero(pi < beta)[0]
    if below.size == 0:
        return 0.0, None
    i = int(below.max())
    if i == pi.size - 1:
        return None, None
    c0, c1, p0, p1 = cgrid[i], cgrid[i + 1], pi[i], pi[i + 1]
    c = float(c0 + (beta - p0) * (c1 - c0) / (p1 - p0))
    slope = (p1 - p0) / (c1 - c0)
    return c, slope


def summarize(name, cgrid, pi, n):
    pi = np.asarray(pi)
    cg = np.asarray(cgrid)
    at = lambda c: float(pi[int(np.argmin(np.abs(cg - c)))])
    row = {"pi0": at(0.0), "pi10": at(0.10), "pi20": at(0.20), "pi30": at(0.30)}
    for b in (0.5, 0.8):
        c, slope = invert(cgrid, pi, b)
        row[f"c*_{b}"] = c
        # one binomial standard error of pi at the crossing, mapped to c by the slope
        row[f"c*_{b}_se"] = None if c is None else float(np.sqrt(b * (1 - b) / n) / slope)
    print(f"  {name:34s} size {row['pi0']:.4f} pi(.10) {row['pi10']:.4f} "
          f"c*0.5 {row['c*_0.5']} c*0.8 {row['c*_0.8']}", flush=True)
    return row


def count_flags(hits, alpha, T):
    """strict / lenient pass flags of K, KC and (alpha = 0.01) KCB on hits (n, T)."""
    ev = HitEvaluator(T, alpha=alpha)
    f = ev.evaluate(hits)
    x = f["x"]
    out = {"K|strict": f["kupiec_pof"], "K|lenient": f["kupiec_dir"],
           "KC|strict": f["kupiec_pof"] & f["christoffersen"],
           "KC|lenient": f["kupiec_dir"] & f["christoffersen_dir"]}
    if abs(alpha - 0.01) < 1e-12:
        amber, red = M.traffic_light_bounds(T)
        out["KCB|strict"] = out["KC|strict"] & (x < amber)
        out["KCB|lenient"] = out["KC|lenient"] & (x < red)
    return out


# ------------------------------------------------------------------ run A
CG = np.round(np.arange(0.0, 0.4501, 0.01), 2)
HOR = (250, 500, 750, 1000)
alpha = 0.01
z, kap = float(stats.norm.isf(alpha)), gaussian_es_var_ratio(alpha)
rng = np.random.default_rng(np.random.SeedSequence([5424, 31, 1]))
loss = rng.standard_normal((N_MIX, max(HOR)))
curA = {}
for c in CG:
    var = (1 - c) * z
    _, _, cr = eprocess_batch(loss, var, kap * var, alpha=alpha)
    for T in HOR:
        rej_e = (cr >= 0) & (cr < T)
        curA.setdefault(f"E_mixture|T={T}", []).append(float(rej_e.mean()))
    fl = count_flags(loss[:, :250] > var, alpha, 250)
    rej_e250 = (cr >= 0) & (cr < 250)
    for k, ok in fl.items():
        curA.setdefault(f"{k}|T=250", []).append(float(1 - ok.mean()))
        curA.setdefault(f"{k}+E|T=250", []).append(float(1 - (ok & ~rej_e250).mean()))
    if int(round(c * 100)) % 5 == 0:
        print(f"A c={c:.2f} {time.time() - t0:.0f}s E250 {curA['E_mixture|T=250'][-1]:.3f} "
              f"P3 {curA['KCB|strict|T=250'][-1]:.3f}", flush=True)
res["A"] = {"alpha": alpha, "c_grid": CG.tolist(), "seed": [5424, 31, 1], "curves": curA,
            "summary": {k: summarize(k, CG, v, N_MIX) for k, v in curA.items()}}

# ------------------------------------------------------------------ run B
alpha = 0.025
z, kap = float(stats.norm.isf(alpha)), gaussian_es_var_ratio(alpha)
rng = np.random.default_rng(np.random.SeedSequence([5424, 31, 2]))
loss = rng.standard_normal((N_MIX, 250))
curB = {}
for c in CG:
    var = (1 - c) * z
    _, _, cr = eprocess_batch(loss, var, kap * var, alpha=alpha)
    rej_e = cr >= 0
    curB.setdefault("E_mixture|T=250", []).append(float(rej_e.mean()))
    for k, ok in count_flags(loss > var, alpha, 250).items():
        curB.setdefault(f"{k}|T=250", []).append(float(1 - ok.mean()))
        curB.setdefault(f"{k}+E|T=250", []).append(float(1 - (ok & ~rej_e).mean()))
print(f"B done {time.time() - t0:.0f}s", flush=True)
res["B"] = {"alpha": alpha, "kappa": kap, "c_grid": CG.tolist(), "seed": [5424, 31, 2],
            "curves": curB, "summary": {k: summarize(k, CG, v, N_MIX) for k, v in curB.items()}}

# ------------------------------------------------------------------ run C (GREM)
CGG = np.round(np.arange(0.0, 0.4001, 0.02), 2)
HG = (250, 500)
alpha = 0.01
z, kap = float(stats.norm.isf(alpha)), gaussian_es_var_ratio(alpha)
rng = np.random.default_rng(np.random.SeedSequence([5424, 31, 3]))
loss = rng.standard_normal((N_GREM, max(HG)))
curC = {}
tc = time.time()
for c in CGG:
    var = (1 - c) * z
    ad = eprocess_adaptive_batch(loss, np.full(loss.shape, var), np.full(loss.shape, kap * var),
                                 alpha=alpha)
    _, _, crm = eprocess_batch(loss, var, kap * var, alpha=alpha)
    x = (loss > var)
    for T in HG:
        cm = ad["GREM"]["cross"]
        curC.setdefault(f"E_GREM|T={T}", []).append(float(np.mean((cm >= 0) & (cm < T))))
        curC.setdefault(f"E_mixture|T={T}", []).append(float(np.mean((crm >= 0) & (crm < T))))
        k = HitEvaluator(T).kupiec[x[:, :T].sum(axis=1)]
        curC.setdefault(f"K|strict|T={T}", []).append(float(1 - k.mean()))
    print(f"C c={c:.2f} {time.time() - tc:.0f}s GREM250 {curC['E_GREM|T=250'][-1]:.3f}", flush=True)
res["C"] = {"alpha": alpha, "c_grid": CGG.tolist(), "seed": [5424, 31, 3], "curves": curC,
            "runtime_s": time.time() - tc,
            "summary": {k: summarize(k, CGG, v, N_GREM) for k, v in curC.items()}}

res["runtime_s"] = time.time() - t0
with open(os.path.join(OUT, "ebacktest.json"), "w") as fh:
    json.dump(res, fh, indent=1)
print(f"wrote {OUT}/ebacktest.json in {res['runtime_s']:.0f}s")
