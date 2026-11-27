"""Round 3 (second-round panel), simulations for MS 5424.

Run from the repo root:
    PYTHONPATH=. uv run python scripts/run_round3_sim.py [n_est] [n_clu] [out_dir] [S1|S2]

S1  Estimated reference (seed SeedSequence([5424, 40, asset, law])). For each
    asset, GARCH(1,1) paths with the parameters of the earlier study
    (paper/revision-R1/sim_results.json) and Gaussian or Student-t shocks (degrees of
    freedom fitted per asset), 1,000 days of history and a 250-day window. The
    validator's reference is estimated on each path, as on real data, with the
    Gaussian quantile: (i) rolling GARCH(1,1), refitted every 25 days on an expanding
    window (data.rolling_garch_vol with min_train = 1,000, the fast likelihood);
    (ii) EWMA, decay 0.94 (data.ewma_vol); (iii) historical simulation, the empirical
    1% quantile of the previous 250 returns (as data.empirical_var). The reported VaR
    is (1 - u) times the estimated reference, u = 0 to 0.50 by 0.01. Output: size and
    MDU at 50% and 80% power of the standard panel, strict and lenient rules.
S2  Clustered understatement (seed SeedSequence([5424, 41, asset, variant, T])).
    GARCH(1,1) paths with Gaussian shocks and the true reference z sigma_t. The
    forecaster cuts VaR by u only on days whose conditional volatility sigma_t
    (known at t - 1) is above the median (variant "median") or the 80th percentile
    (variant "q80") of its previous 250 values. Non-anticipating; exceptions cluster
    in high-volatility spells. Rejection of POF, POF + Christoffersen and the
    standard panel, against the exact power of a constant understatement with the
    same mean understatement. A third variant ("slow") reports (1 - u) z times an
    EWMA volatility with decay 0.99, a slowly updating model whose exceptions
    cluster after volatility rises. T = 250 and 1,000.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from gaming_oracle import data as D
from gaming_oracle import mdu as M
from gaming_oracle.feasibility import HitEvaluator, ewma_vol_batch, simulate_garch

N_EST = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
N_CLU = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
OUT = sys.argv[3] if len(sys.argv) > 3 else "paper/revision-R1/round2"
ALPHA, T, H = 0.01, 250, 1000
Z = float(stats.norm.isf(ALPHA))
SR = json.load(open("paper/revision-R1/sim_results.json"))["assets"]
ASSETS = {"AAPL": 0, "SPY": 1, "BTC-USD": 2}
PARAMS = {a: tuple(SR[a]["params"]) for a in ASSETS}
NU = {a: float(SR[a]["nu"]) for a in ASSETS}
UGRID = np.round(np.arange(0.0, 0.5001, 0.01), 2)


def garch_ref_rows(r_rows: np.ndarray) -> np.ndarray:
    """Rolling GARCH sigma on the window for each row (worker)."""
    out = np.empty((r_rows.shape[0], T))
    for i, r in enumerate(r_rows):
        s = D.rolling_garch_vol(r, min_train=H, refit_every=25, fit=D._garch_fit_params_fast)
        out[i] = s[H:H + T]
    return out


def hs_ref(r_full: np.ndarray, chunk: int = 250) -> np.ndarray:
    """Historical-simulation VaR on window days: -quantile(r[t-250:t], alpha)."""
    out = np.empty((r_full.shape[0], T))
    for i in range(0, r_full.shape[0], chunk):
        w = sliding_window_view(r_full[i:i + chunk, H - 250:H + T - 1], 250, axis=1)
        out[i:i + chunk] = np.abs(-np.quantile(w, ALPHA, axis=-1))
    return out


def panel_pass(he: HitEvaluator, hits: np.ndarray, n: int):
    f = he.evaluate(hits)
    x = f["x"]
    amber, red = M.traffic_light_bounds(n)
    strict = f["kupiec_pof"] & f["christoffersen"] & (x < amber)
    lenient = f["kupiec_dir"] & f["christoffersen_dir"] & (x < red)
    return {"K|strict": f["kupiec_pof"], "KC|strict": f["kupiec_pof"] & f["christoffersen"],
            "KC|lenient": f["kupiec_dir"] & f["christoffersen_dir"],
            "KCB|strict": strict, "KCB|lenient": lenient, "x": x}


def mdu_from_curve(grid, pi, beta, n):
    """Smallest u with pi >= beta at every larger grid point, linear interpolation;
    standard error = binomial s.e. at the crossing / local slope."""
    grid, pi = np.asarray(grid), np.asarray(pi)
    below = np.nonzero(pi < beta)[0]
    if below.size == 0:
        return 0.0, 0.0
    i = int(below.max())
    if i == grid.size - 1:
        return None, None
    u0, u1, p0, p1 = grid[i], grid[i + 1], pi[i], pi[i + 1]
    u = float(u0 + (beta - p0) * (u1 - u0) / (p1 - p0))
    se = float(np.sqrt(beta * (1 - beta) / n) / ((p1 - p0) / (u1 - u0)))
    return u, se


def run_est(pool) -> dict:
    he = HitEvaluator(T)
    out = {}
    for a, ai in ASSETS.items():
        for law, li in (("gaussian", 1), ("student", 2)):
            t0 = time.time()
            seed = [5424, 40, ai, li]
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            r, sig, r_full = simulate_garch(rng, PARAMS[a], N_EST, T, "D1" if law == "gaussian" else "D2",
                                            NU[a], burn=H)
            if law == "gaussian":
                q_true = Z
            else:
                q_true = float(stats.t.isf(ALPHA, NU[a]) * np.sqrt((NU[a] - 2) / NU[a]))
            vstar = q_true * sig
            loss = -r
            chunks = np.array_split(r_full, 64)
            g = np.vstack(list(pool.map(garch_ref_rows, chunks)))
            refs = {"GARCH": Z * g,
                    "EWMA": Z * D.ewma_vol_rows(r_full)[:, H:H + T] if hasattr(D, "ewma_vol_rows")
                    else Z * np.vstack([D.ewma_vol(x)[H:H + T] for x in r_full]),
                    "HS": hs_ref(r_full)}
            for ref, V in refs.items():
                curves = {k: [] for k in ("KCB|strict", "KCB|lenient")}
                for u in UGRID:
                    pp = panel_pass(he, loss > (1 - u) * V, T)
                    for k in curves:
                        curves[k].append(float(1 - pp[k].mean()))
                rec = {"seed": seed, "n_paths": N_EST, "q_true": q_true,
                       "ref_over_true_mean": float(np.mean(V / vstar)),
                       "ref_over_true_median": float(np.median(V / vstar)),
                       "curves": curves}
                for k, c in curves.items():
                    rec[k] = {"size": c[0]}
                    for b in (0.5, 0.8):
                        u, se = mdu_from_curve(UGRID, c, b, N_EST)
                        rec[k][f"u*_{b}"], rec[k][f"se_{b}"] = u, se
                out[f"{a}|{law}|{ref}"] = rec
                print(f"S1 {a:7s} {law:8s} {ref:5s}: ref/true {rec['ref_over_true_mean']:.3f}; "
                      + "; ".join(f"{k} size {rec[k]['size']:.3f} u* {rec[k]['u*_0.5']} / {rec[k]['u*_0.8']}"
                                  for k in curves) + f" ({time.time() - t0:.0f}s)", flush=True)
    return out


def run_clu() -> dict:
    out = {}
    cgrid = np.round(np.arange(0.0, 0.8001, 0.02), 2)
    for n in (250, 1000):
        he = HitEvaluator(n)
        for a, ai in ASSETS.items():
            for var, vi, q in (("median", 1, 50.0), ("q80", 2, 80.0), ("slow", 3, None)):
                t0 = time.time()
                seed = [5424, 41, ai, vi, n]
                rng = np.random.default_rng(np.random.SeedSequence(seed))
                r, sig, r_full = simulate_garch(rng, PARAMS[a], N_CLU, n, "D1", None, burn=250)
                # sigma for the whole path: rebuild from r_full with the same params
                om, al, be = PARAMS[a]
                s2 = np.full(N_CLU, om / (1 - al - be))
                sig_full = np.empty_like(r_full)
                for t in range(r_full.shape[1]):
                    sig_full[:, t] = np.sqrt(s2)
                    s2 = om + al * r_full[:, t] ** 2 + be * s2
                assert np.allclose(sig_full[:, 250:], sig)
                loss, vstar = -r, Z * sig
                if q is not None:
                    past = sliding_window_view(sig_full[:, :-1], 250, axis=1)[:, :n]   # sigma_{t-250..t-1}
                    thr = np.vstack([np.percentile(past[i:i + 500], q, axis=-1)      # row chunks: memory
                                     for i in range(0, N_CLU, 500)])
                    hi = sig > thr
                    base = vstar
                else:
                    hi = np.ones_like(sig, dtype=bool)
                    base = Z * ewma_vol_batch(r_full, lam=0.99)[:, 250:]
                rec = {"seed": seed, "n_paths": N_CLU, "share_cut": float(hi.mean()), "c_grid": cgrid.tolist(),
                       "mean_under": [], "curves": {k: [] for k in ("K|strict", "KC|strict", "KC|lenient",
                                                                    "KCB|strict", "KCB|lenient")},
                       "lag1_hit_corr": []}
                for c in cgrid:
                    V = np.where(hi, (1 - c) * base, vstar)
                    h = loss > V
                    pp = panel_pass(he, h, n)
                    for k in rec["curves"]:
                        rec["curves"][k].append(float(1 - pp[k].mean()))
                    rec["mean_under"].append(float(np.mean(1.0 - V / vstar)))
                    hm = h.astype(float)
                    num = np.mean(hm[:, 1:] * hm[:, :-1]) - np.mean(hm) ** 2
                    rec["lag1_hit_corr"].append(float(num / max(np.var(hm), 1e-18)))
                mu = np.array(rec["mean_under"])
                rec["mdu_mean_under"] = {}
                for k, cv in rec["curves"].items():
                    rec["mdu_mean_under"][k] = {str(b): mdu_from_curve(mu, cv, b, N_CLU)[0] for b in (0.5, 0.8)}
                # constant understatement with the same mean, exact
                cst = {k: [] for k in ("K|strict", "KC|strict", "KCB|strict", "KCB|lenient")}
                for m_ in mu:
                    rp = M.reject_probs(M.p_from_understatement(float(m_), ALPHA), n, ALPHA)
                    for k in cst:
                        cst[k].append(rp[tuple(k.split("|"))])
                rec["constant_same_mean_exact"] = cst
                out[f"T={n}|{a}|{var}"] = rec
                md = rec["mdu_mean_under"]
                print(f"S2 T={n} {a:7s} {var:6s}: share {rec['share_cut']:.3f}; MDU (mean understatement) "
                      + "; ".join(f"{k} {md[k]['0.5']} / {md[k]['0.8']}" for k in md)
                      + f"; size KC strict {rec['curves']['KC|strict'][0]:.3f}; c=0.4: K {rec['curves']['K|strict'][20]:.3f}"
                      f" KC {rec['curves']['KC|strict'][20]:.3f} const-same-mean K {cst['K|strict'][20]:.3f}"
                      f" KC {cst['KC|strict'][20]:.3f}; lag1 corr {rec['lag1_hit_corr'][20]:.4f} ({time.time() - t0:.0f}s)",
                      flush=True)
    return out


if __name__ == "__main__":
    # optional 4th argument: "S1" or "S2" runs one part and writes round3_sim_<part>.json
    PART = sys.argv[4] if len(sys.argv) > 4 else "all"
    t0 = time.time()
    res = {"settings": {"N_EST": N_EST, "N_CLU": N_CLU, "H": H, "T": T, "params": PARAMS, "nu": NU}}
    if PART in ("all", "S1"):
        with ProcessPoolExecutor(max_workers=8) as pool:
            res["S1"] = run_est(pool)
    if PART in ("all", "S2"):
        res["S2"] = run_clu()
    res["runtime_s"] = time.time() - t0
    name = "round3_sim.json" if PART == "all" else f"round3_sim_{PART}.json"
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {OUT}/{name} in {res['runtime_s']:.0f}s")
