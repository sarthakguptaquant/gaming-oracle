"""Round 3 (second-round panel), exact computations for MS 5424.

Run from the repo root (after scripts/run_round2_exact.py and run_round2_windows.py):
    uv run python scripts/run_round3_exact.py [out_dir]

R3a  size-adjusted panels: every panel recalibrated to an exact 5% size by a
     common nominal level for its components (min-p statistic, randomized at the
     boundary), and the power envelope (randomized one-sided binomial test, UMP
     under i.i.d. exceptions) at 5% and at each panel's nominal size; MDU at 50%
     and 80% power, Gaussian tail; T = 250, 500 and 1000, 99% coverage.
R3b  Basel desk-level rule (fail with more than 12 exceptions at 99% or more than
     30 at 97.5% over 250 days): size and MDU, Gaussian and fitted Student-t tails.
R3c  expected capital factor on a fine grid of u (Gaussian), to show where it is flat.
R3d  basis-point conversion of the MDU under the fitted Student-t tail of each asset
     (Table 6), from windows.json: u*_t times the mean reference VaR rescaled by
     q_t(alpha) / z_alpha, with q_t the quantile of the unit-variance Student-t.
No random numbers are used.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import stats

from gaming_oracle import mdu as M

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper/revision-R1/round2"
EX = json.load(open(os.path.join(OUT, "exact.json")))["N1"]
WIN = json.load(open(os.path.join(OUT, "windows.json")))
NU = {"AAPL": 4.972333080308459, "SPY": 6.709064545105986, "BTC-USD": 3.1648415166169968}
ALPHA = 0.01
BETAS = (0.5, 0.8)
t0 = time.time()
res: dict = {"R3a": {}, "R3b": {}, "R3c": {}, "R3d": {}}


def g(p):
    return M.mdu_from_p(p, ALPHA, "gaussian") if p is not None else None


# ---------------------------------------------------------------- R3a
NOMINAL = {("K", "two"): "K|strict", ("K", "upper"): "K|lenient", ("KC", "two"): "KC|strict",
           ("KC", "upper"): "KC|lenient", ("KCB", "two"): "KCB|strict",
           ("KCB", "upper"): "KCB|lenient", ("B", "two"): "B|strict"}
for T in (250, 500, 1000):
    blk = {}
    cells = EX[f"T={T}|alpha={ALPHA}"]["cells"]
    ump5 = M.ump_rule(T, ALPHA, 0.05)
    env5 = {b: M.invert_power(lambda p: M.ump_reject_prob(p, T, ump5), ALPHA, b) for b in BETAS}
    blk["envelope_5pct"] = {"rule": ump5, **{f"p*_{b}": env5[b] for b in BETAS},
                            **{f"u*_{b}": g(env5[b]) for b in BETAS}}
    for (panel, direction), nom in NOMINAL.items():
        rule = M.size_adjusted_rule(T, ALPHA, panel, direction, 0.05)
        f = lambda p, r=rule, pa=panel, d=direction: M.size_adjusted_reject_prob(p, T, ALPHA, pa, d, r)
        adj = {b: M.invert_power(f, ALPHA, b) for b in BETAS}
        nsize = cells[nom]["size"]
        umpn = M.ump_rule(T, ALPHA, nsize)
        envn = {b: M.invert_power(lambda p, r=umpn: M.ump_reject_prob(p, T, r), ALPHA, b) for b in BETAS}
        blk[f"{panel}|{direction}"] = {
            "nominal_cell": nom, "nominal_size": nsize,
            **{f"nominal_u*_{b}": cells[nom][f"u*_{b}"]["gaussian"] for b in BETAS},
            "adj_rule": rule, "adj_size_check": f(ALPHA),
            **{f"adj_p*_{b}": adj[b] for b in BETAS}, **{f"adj_u*_{b}": g(adj[b]) for b in BETAS},
            **{f"env_at_nominal_u*_{b}": g(envn[b]) for b in BETAS}}
    res["R3a"][f"T={T}"] = blk
    print(f"R3a T={T}: envelope 5%: u* {g(env5[0.5]):.4f} / {g(env5[0.8]):.4f} (k={ump5['k']}, w={ump5['w']:.4f})")
    for k, v in blk.items():
        if k == "envelope_5pct":
            continue
        print(f"  {k:10s} nominal size {v['nominal_size']:.4f} u* {v['nominal_u*_0.5']:.4f} / {v['nominal_u*_0.8']:.4f}"
              f" | envelope at nominal size {v['env_at_nominal_u*_0.5']:.4f} / {v['env_at_nominal_u*_0.8']:.4f}"
              f" | 5% adjusted (size {v['adj_size_check']:.5f}) u* {v['adj_u*_0.5']:.4f} / {v['adj_u*_0.8']:.4f}",
              flush=True)

# ---------------------------------------------------------------- R3b
for tail in ("gaussian", NU["AAPL"], NU["SPY"], NU["BTC-USD"], 5.0):
    name = "gaussian" if tail == "gaussian" else f"t{tail:.2f}"
    res["R3b"][name] = {"size": M.desk_reject_prob(0.0, 250, tail),
                        **{f"u*_{b}": M.desk_mdu(b, 250, tail) for b in BETAS},
                        "reject_at_u": {str(u): M.desk_reject_prob(u, 250, tail)
                                        for u in (0.1, 0.2, 0.3, 0.4)}}
    v = res["R3b"][name]
    print(f"R3b desk rule {name}: size {v['size']:.2e}; u* {v['u*_0.5']:.4f} / {v['u*_0.8']:.4f}; "
          f"reject at u 0.1..0.4 {[round(x, 4) for x in v['reject_at_u'].values()]}")

# ---------------------------------------------------------------- R3c
grid = np.round(np.arange(0.0, 0.3001, 0.01), 2)
res["R3c"] = {str(u): {"capital_factor": M.expected_capital_factor(float(u)),
                       "E_multiplier": M.expected_multiplier(M.p_from_understatement(float(u), ALPHA))}
              for u in grid}
print("R3c capital factor:", {k: round(v["capital_factor"], 4) for k, v in res["R3c"].items()})

# ---------------------------------------------------------------- R3d
c250 = EX["T=250|alpha=0.01"]["cells"]
z = float(stats.norm.isf(ALPHA))
for a, nu in NU.items():
    q_unit = float(stats.t.isf(ALPHA, nu) * np.sqrt((nu - 2.0) / nu))
    scale = q_unit / z
    mv = np.array(WIN["N4"][a]["mean_vstar_bps_quartiles"])
    row = {"nu": nu, "q_t_unit": q_unit, "scale_vs_gaussian": scale}
    for rule in ("strict", "lenient"):
        for b in BETAS:
            ut = M.mdu_from_p(c250[f"KCB|{rule}"][f"p*_{b}"], ALPHA, nu)
            ug = c250[f"KCB|{rule}"][f"u*_{b}"]["gaussian"]
            row[f"{rule}|{b}"] = {"u*_t": ut, "bps_t_quartiles": (ut * scale * mv).tolist(),
                                  "u*_gauss": ug, "bps_gauss_quartiles": (ug * mv).tolist()}
    res["R3d"][a] = row
    print(f"R3d {a}: nu {nu:.2f} q_t/z {scale:.4f}; strict 0.5 t-bps median "
          f"{row['strict|0.5']['bps_t_quartiles'][2]:.1f} (IQR {row['strict|0.5']['bps_t_quartiles'][1]:.1f}-"
          f"{row['strict|0.5']['bps_t_quartiles'][3]:.1f}); lenient 0.8 t-bps median "
          f"{row['lenient|0.8']['bps_t_quartiles'][2]:.1f} (IQR {row['lenient|0.8']['bps_t_quartiles'][1]:.1f}-"
          f"{row['lenient|0.8']['bps_t_quartiles'][3]:.1f}); u*_t strict .5 {row['strict|0.5']['u*_t']:.4f}"
          f" lenient .8 {row['lenient|0.8']['u*_t']:.4f}")

res["runtime_s"] = time.time() - t0
with open(os.path.join(OUT, "round3_exact.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"wrote {OUT}/round3_exact.json in {res['runtime_s']:.1f}s")
