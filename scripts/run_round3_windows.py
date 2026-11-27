"""Round 3: other reference forecasts on every rolling window (MS 5424, Table 7).

Run from the repo root:
    PYTHONPATH=. uv run python scripts/run_round3_windows.py [out_dir]

Same assets, windows and panels as scripts/run_round2_windows.py. References at 99%:
  gauss : z_alpha times the causal rolling GARCH(1,1) volatility (the paper's reference);
  t     : the unit-variance Student-t quantile, degrees of freedom of each asset
          (4.97, 6.71, 3.16, as in the paper), times the same volatility;
  fhs   : filtered historical simulation, the volatility times the empirical 99%
          quantile of the previous 500 standardized losses loss_s / sigma_s;
  hs    : historical simulation, the empirical 99% quantile of the previous 250
          losses (data.empirical_var).
Windows whose start has fewer than 500 (fhs) or 250 (hs) previous days are skipped
for that reference. Share of windows on which (1 - u) times the reference passes
the standard panel, u = 0, 0.05, 0.10, 0.20, 0.30. No random numbers.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

from gaming_oracle.data import empirical_var, load_data
from gaming_oracle.feasibility import HitEvaluator

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper/revision-R1/round2"
T, ALPHA = 250, 0.01
Z = float(stats.norm.isf(ALPHA))
NU = {"AAPL": 4.972333080308459, "SPY": 6.709064545105986, "BTC-USD": 3.1648415166169968}
US = (0.0, 0.05, 0.10, 0.20, 0.30)
HE = HitEvaluator(T)
res = {}


def starts_for(n, step=25):
    s = list(range(0, n - T + 1, step))
    if s[-1] != n - T:
        s.append(n - T)
    return s


for a, nu in NU.items():
    d = load_data(asset=a, alpha=ALPHA, method="rolling_garch")
    r, vg = d.returns, d.true_var
    sig = vg / Z
    loss = -r
    q_t = float(stats.t.isf(ALPHA, nu) * np.sqrt((nu - 2.0) / nu))
    zstd = loss / sig
    fhs = np.full(r.size, np.nan)
    for t in range(500, r.size):
        fhs[t] = sig[t] * np.quantile(zstd[t - 500:t], 1 - ALPHA)
    refs = {"gauss": (vg, 0), "t": (q_t * sig, 0), "fhs": (fhs, 500),
            "hs": (empirical_var(r, alpha=ALPHA, window=250), 250)}
    res[a] = {"q_t_unit": q_t}
    for name, (V, first) in refs.items():
        ss = [s for s in starts_for(r.size) if s >= first]
        L = np.stack([loss[s:s + T] for s in ss])
        VV = np.stack([V[s:s + T] for s in ss])
        row = {"n_windows": len(ss), "mean_ref_over_gauss": float(np.mean(V[first:] / vg[first:]))}
        for u in US:
            f = HE.evaluate(L > (1 - u) * VV)
            x = f["x"]
            st = f["kupiec_pof"] & f["christoffersen"] & (x < 5)
            le = f["kupiec_dir"] & f["christoffersen_dir"] & (x < 10)
            row[f"u={u}"] = {"strict": float(st.mean()), "lenient": float(le.mean()),
                             "mean_exceptions": float(x.mean())}
        res[a][name] = row
        print(f"{a:7s} {name:5s} n={len(ss):3d} ref/gauss {row['mean_ref_over_gauss']:.3f} | strict "
              + " ".join(f"{row[f'u={u}']['strict']:.2f}" for u in US) + " | lenient "
              + " ".join(f"{row[f'u={u}']['lenient']:.2f}" for u in US)
              + f" | mean exc at u=0 {row['u=0.0']['mean_exceptions']:.2f}", flush=True)

with open(os.path.join(OUT, "round3_windows.json"), "w") as fh:
    json.dump(res, fh, indent=1)
print(f"wrote {OUT}/round3_windows.json")
