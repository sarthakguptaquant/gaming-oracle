"""Round 2 checks of the ex post bound on every definable window (strict standard panel).

Run from the repo root:
    PYTHONPATH=. uv run python scripts/run_round2_expost_checks.py [out_dir]

For each 250-day rolling window on which the causal reference passes the strict
standard panel (POF, Christoffersen, traffic light): (i) the closed form (concede
the largest positive losses up to the count bound) against the MILP solution,
(ii) the closed form when the cap V_t <= V*_t is dropped, (iii) the identity
MUL = mean V* - mean positive loss left on non-exception days. Deterministic.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from gaming_oracle import gates as G
from gaming_oracle.data import load_data
from gaming_oracle.milp_solver import _gate_count_bounds, solve_milp

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper/revision-R1/round2"
T, ALPHA = 250, 0.01


def topk(tv, rr, k):
    loss = -rr
    pos = np.maximum(loss, 0.0)
    e = loss > tv
    for t in np.argsort(-pos):
        if e.sum() >= k:
            break
        if pos[t] > 0:
            e[t] = True
    v = np.where(e, 0.0, np.minimum(pos, tv))
    left = float(np.sum(np.where(e, 0.0, pos)) / T)
    return float(np.mean(tv - v) * 1e4), left * 1e4


def relaxed(tv, rr, k):
    pos = np.maximum(-rr, 0.0)
    order = np.argsort(-pos)
    e = np.zeros(T, dtype=bool)
    e[order[:k]] = pos[order[:k]] > 0
    return float(np.mean(tv - np.where(e, 0.0, pos)) * 1e4)


x_lo, x_hi, cap = _gate_count_bounds(G.PANEL_REGULATORY, T, ALPHA, 0.95, 0.005)
K = min(x_hi, cap) if cap is not None else x_hi
out = {"k_max": int(K), "assets": {}}
for a in ("AAPL", "SPY", "BTC-USD"):
    d = load_data(asset=a, alpha=ALPHA, method="rolling_garch")
    n = d.returns.size
    starts = list(range(0, n - T + 1, 25))
    if starts[-1] != n - T:
        starts.append(n - T)
    eq, neq, diff_relax, ident = 0, 0, [], []
    for s in starts:
        rr, tv = d.returns[s:s + T], d.true_var[s:s + T]
        if not G.run_panel(tv, rr, panel=G.PANEL_REGULATORY, alpha=ALPHA)["passed_all"]:
            continue
        m = solve_milp(tv, rr, panel=G.PANEL_REGULATORY, alpha=ALPHA)
        b = m.mean_understatement * 1e4
        tk, left = topk(tv, rr, K)
        if m.optimal:
            if abs(tk - b) < 1e-6:
                eq += 1
            else:
                neq += 1
            ident.append(abs(np.mean(tv) * 1e4 - left - b))
        diff_relax.append(relaxed(tv, rr, K) - tk)
    out["assets"][a] = {"certified_windows": eq + neq, "closed_form_equals_milp": eq,
                        "max_identity_error_bps": float(max(ident)),
                        "relaxed_minus_capped_bps_max": float(max(diff_relax)),
                        "relaxed_minus_capped_bps_min": float(min(diff_relax))}
    print(a, out["assets"][a], flush=True)
with open(os.path.join(OUT, "expost_checks.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", os.path.join(OUT, "expost_checks.json"))
