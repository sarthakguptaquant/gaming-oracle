"""Round 2, N4 and N5: every rolling window, no selected window.

Run from the repo root (after scripts/run_round2_exact.py):
    PYTHONPATH=. uv run python scripts/run_round2_windows.py [n_paths] [out_dir]

Data: AAPL, SPY, BTC-USD daily returns 2015 to 2024 (local cache), causal rolling
GARCH(1,1) reference at 99% (EWMA for the first 500 days); 250-day windows every
25 days plus the most recent 250 days (92 / 92 / 138 windows).

N4  (a) the minimum detectable understatement of the standard panel in basis
        points of notional, u* times the mean reference VaR of each window (all
        windows; no definability condition);
    (b) the share of windows on which a constant haircut c of the reference passes
        the strict and the lenient standard panel, c = 0 to 0.30; also on
        non-overlapping windows (step 250).
N5  the exact ex post series (MILP, strict standard panel) on every window where
    the reference passes that panel, its size in bps and as a fraction of mean
    reference VaR, and its rejection on fresh losses: the observed volatility
    V*_t / z_alpha is kept and the shocks are redrawn (Gaussian, and Student-t
    scaled to unit variance with the degrees of freedom fitted per asset).
    Seeds SeedSequence([5424, 32, asset, law, window]).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import stats

from gaming_oracle import gates as G
from gaming_oracle import mdu as M
from gaming_oracle.data import load_data
from gaming_oracle.feasibility import HitEvaluator, std_t
from gaming_oracle.milp_solver import solve_milp

N_PATHS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper/revision-R1/round2"
T, ALPHA = 250, 0.01
Z = float(stats.norm.isf(ALPHA))
ASSETS = {"AAPL": 0, "SPY": 1, "BTC-USD": 2}
NU = json.load(open("paper/revision-R1/sim_results.json"))
NU = {a: v["nu"] for a, v in NU["assets"].items()}
EX = json.load(open(os.path.join(OUT, "exact.json")))["N1"]["T=250|alpha=0.01"]["cells"]
USTAR = {f"{rule}|{b}": EX[f"KCB|{rule}"][f"u*_{b}"]["gaussian"]
         for rule in ("strict", "lenient") for b in (0.5, 0.8)}
CS = np.round(np.arange(0.0, 0.3001, 0.05), 2)
HE = HitEvaluator(T)
t0 = time.time()
res: dict = {"ustar_gaussian": USTAR, "N4": {}, "N5": {}, "n_paths": N_PATHS}


def starts_for(n, step):
    s = list(range(0, n - T + 1, step))
    if step == 25 and s[-1] != n - T:
        s.append(n - T)
    return s


def pass_flags(hits):
    f = HE.evaluate(hits)
    x = f["x"]
    strict = f["kupiec_pof"] & f["christoffersen"] & f["basel_traffic_light"]
    lenient = f["kupiec_dir"] & f["christoffersen_dir"] & (x <= 9)
    return strict, lenient, x


def q(v):
    v = np.asarray(v, dtype=float)
    return [float(np.percentile(v, k)) for k in (0, 25, 50, 75, 100)]


for a, ai in ASSETS.items():
    d = load_data(asset=a, alpha=ALPHA, method="rolling_garch")
    r_full, tv_full = d.returns, d.true_var
    starts = starts_for(r_full.size, 25)
    mv = np.array([np.mean(tv_full[s:s + T]) * 1e4 for s in starts])
    # ---- N4a
    bps = {k: q(u * mv) for k, u in USTAR.items()}
    # ---- N4b
    rates = {}
    for step in (25, 250):
        ss = starts_for(r_full.size, step)
        L = np.stack([-r_full[s:s + T] for s in ss])
        V = np.stack([tv_full[s:s + T] for s in ss])
        for c in CS:
            st, le, _ = pass_flags(L > (1 - c) * V)
            rates[f"step={step}|c={c}"] = {"n": len(ss), "strict": float(st.mean()),
                                           "lenient": float(le.mean())}
    res["N4"][a] = {"n_windows": len(starts), "mean_vstar_bps_quartiles": q(mv),
                    "mdu_bps_quartiles": bps, "pass_rate": rates}
    print(f"{a}: windows {len(starts)}; mean V* q {np.round(q(mv), 1)}; strict pass "
          + " ".join(f"{c}:{rates[f'step=25|c={c}']['strict']:.2f}" for c in CS)
          + "; lenient " + " ".join(f"{c}:{rates[f'step=25|c={c}']['lenient']:.2f}" for c in CS),
          flush=True)

    # ---- N5
    rows = []
    for wi, s in enumerate(starts):
        rr, tv = r_full[s:s + T], tv_full[s:s + T]
        if not G.run_panel(tv, rr, panel=G.PANEL_REGULATORY, alpha=ALPHA)["passed_all"]:
            continue
        m = solve_milp(tv, rr, panel=G.PANEL_REGULATORY, alpha=ALPHA)
        sv = m.stated_var
        mul = float(np.mean(tv - sv) * 1e4)
        sig = tv / Z
        row = {"window": wi, "start_1idx": s + 1, "mul_bps": mul, "frac_mean_vstar": mul / mv[wi],
               "optimal": bool(m.optimal), "zero_share": float(np.mean(sv <= 1e-12))}
        for law, li in (("gaussian", 1), ("student", 2)):
            rng = np.random.default_rng(np.random.SeedSequence([5424, 32, ai, li, wi]))
            eps = rng.standard_normal((N_PATHS, T)) if law == "gaussian" else std_t(rng, NU[a], (N_PATHS, T))
            loss = sig[None, :] * eps
            st, le, x = pass_flags(loss > sv[None, :])
            st0, le0, x0 = pass_flags(loss > tv[None, :])
            row[law] = {"rej_strict": float(1 - st.mean()), "rej_lenient": float(1 - le.mean()),
                        "mean_exc": float(x.mean()), "p5_exc": float(np.percentile(x, 5)),
                        "ref_rej_strict": float(1 - st0.mean()), "ref_mean_exc": float(x0.mean())}
        rows.append(row)
    g = lambda law, k: [r[law][k] for r in rows]
    res["N5"][a] = {
        "definable": len(rows), "n_windows": len(starts), "not_certified": int(sum(not r["optimal"] for r in rows)),
        "mul_bps_quartiles": q([r["mul_bps"] for r in rows]),
        "frac_quartiles": q([r["frac_mean_vstar"] for r in rows]),
        "zero_share_range": [min(r["zero_share"] for r in rows), max(r["zero_share"] for r in rows)],
        **{f"{law}_min_rej_strict": min(g(law, "rej_strict")) for law in ("gaussian", "student")},
        **{f"{law}_min_rej_lenient": min(g(law, "rej_lenient")) for law in ("gaussian", "student")},
        **{f"{law}_mean_exc_range": [min(g(law, "mean_exc")), max(g(law, "mean_exc"))]
           for law in ("gaussian", "student")},
        **{f"{law}_min_p5_exc": min(g(law, "p5_exc")) for law in ("gaussian", "student")},
        **{f"{law}_ref_rej_strict_mean": float(np.mean(g(law, "ref_rej_strict")))
           for law in ("gaussian", "student")},
        "windows": rows}
    n5 = res["N5"][a]
    print(f"  N5 {a}: definable {n5['definable']}/{n5['n_windows']}, not certified {n5['not_certified']}; "
          f"MUL q {np.round(n5['mul_bps_quartiles'], 1)} frac q {np.round(n5['frac_quartiles'], 3)}; "
          f"min rej strict G {n5['gaussian_min_rej_strict']:.3f} t {n5['student_min_rej_strict']:.3f}; "
          f"lenient G {n5['gaussian_min_rej_lenient']:.3f} t {n5['student_min_rej_lenient']:.3f}; "
          f"mean exc G {np.round(n5['gaussian_mean_exc_range'], 1)} t {np.round(n5['student_mean_exc_range'], 1)}; "
          f"min p5 exc {n5['gaussian_min_p5_exc']} / {n5['student_min_p5_exc']} "
          f"({time.time() - t0:.0f}s)", flush=True)

res["runtime_s"] = time.time() - t0
with open(os.path.join(OUT, "windows.json"), "w") as fh:
    json.dump(res, fh, indent=1)
print(f"wrote {OUT}/windows.json in {res['runtime_s']:.0f}s")
