"""Run the R1 feasibility study (cycle 9) and write every output.

    uv run python scripts/run_feasibility.py            # full run
    uv run python scripts/run_feasibility.py --quick    # smoke run, tiny N
    uv run python scripts/run_feasibility.py --tables-only   # rebuild tables from JSON

Outputs (full run):
    paper/revision-R1/sim_results.json   every summary number
    paper/revision-R1/sim_console.log    console log of the run
    paper/revision-R1/sim_tables.md      markdown tables built from the JSON
    figures/fig_power_vs_haircut.{pdf,png}
    figures/fig_expost_uniform_limit.{pdf,png}
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gaming_oracle import feasibility as F  # noqa: E402


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def git_info():
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", "gaming_oracle", "scripts", "tests"))}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def _day(x):
    return "n/a" if x is None else f"{x:.0f}"


def fc(x):
    if x is None:
        return "none"
    return ">=0.50" if x >= 0.5 else f"{x:.2f}"


def fx(lim, pi0, d):
    """Excess-rejection limit; 'n/a' when 1 - pi(0) < delta (cannot be exceeded)."""
    if 1.0 - pi0 < d:
        return "n/a"
    return fc(lim["c_excess"][str(d)])


SERIES_ORDER = ["S0", "S3_P3", "S3_P4", "S4_P4", "S5_P3", "S5_P4", "S6_P4E", "S7_b20", "S7_swept"]
SERIES_LABEL = {
    "S0": "S0 reference", "S3_P3": "S3 ex post uniform (P3)", "S3_P4": "S3 ex post uniform (P4)",
    "S4_P4": "S4 greedy (P4)", "S5_P3": "S5 exact MILP (P3)", "S5_P4": "S5 exact MILP (P4)",
    "S6_P4E": "S6 greedy vs P4+E", "S7_b20": "S7 budgeted b=0.20, ES indep.",
    "S7_swept": "S7 budgeted b swept, ES indep.",
}


def tables_md(res) -> str:
    L = []
    A = res["assets"]
    L.append("### Table S1. Asset set-up\n")
    L.append("| asset | window | mean V* (bps) | GARCH fit sample | omega | alpha_g | beta | nu (D2) | kurtosis of r/sigma in window |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a, v in A.items():
        p = v["params"]
        L.append(f"| {a} | [{v['window'][0]}:{v['window'][1]}] | {v['mean_var_bps']:.2f} | {v['fit_sample']} | "
                 f"{p[0]:.4e} | {p[1]:.4f} | {p[2]:.4f} | {v['nu']:.3f} | {v['kurtosis_window']:.2f} |")
        if v["params_alt"]:
            q = v["params_alt"]
            L.append(f"| {a} (sensitivity) | same | same | r[:250] | {q[0]:.4e} | {q[1]:.4f} | {q[2]:.4f} | {v['nu_alt']:.3f} | same |")
    L.append("")

    # E1
    L.append("### Table S2. E1, ex post series on fresh losses (rejection rates, N per cell in header)\n")
    for a, e in res["E1"].items():
        L.append(f"**{a}.** Reproduction check against the current paper: " + ", ".join(
            f"{k} {c['code']} vs {c['paper']} ({'match' if c['match'] else 'MISMATCH'})"
            for k, c in e["checks"].items()) + ".\n")
        for key, block in e["results"].items():
            n = next(v["n_paths"] for k, v in block.items() if k != "seed")
            L.append(f"{a}, {key}, seed {block['seed']}, N = {n}\n")
            L.append("| series | bps on observed window | share of days V=0 | P3 | P4 | e-backtest | P4+E | P3 dir. | exceptions mean [p5, p95] | median first e-crossing day |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for s in SERIES_ORDER:
                v = block[s]
                L.append(f"| {SERIES_LABEL[s]} | {e['series_bps'][s]:.2f} | {e['frac_zero'][s]:.3f} | "
                         f"{v['rej_P3']:.3f} | {v['rej_P4']:.3f} | {v['rej_e_backtest']:.3f} | "
                         f"{v['rej_P4+E']:.3f} | {v['rej_P3dir']:.3f} | {v['x_mean']:.2f} [{v['x_p5']:.0f}, {v['x_p95']:.0f}] | "
                         f"{_day(v.get('e_cross_median_day'))} |")
            L.append("")

    # E2
    cg = res["E2"]["c_grid"]
    idx = {c: cg.index(c) for c in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)}
    rows = ["P3", "P3dir", "P4", "P5", "kupiec_pof", "christoffersen", "basel_traffic_light",
            "conformal_coverage", "oneside_uc_score", "e_backtest", "P3+E", "P4+E",
            "e_backtest|g=0.0", "e_backtest|g=0.2", "e_backtest|g=0.4", "P3+E|g=0.4"]
    L.append("### Table S3. E2 and E3 (T = 250), constant haircut S1: rejection probability pi(c) and limits\n")
    L.append("c_tau: largest grid c with pi(c') <= tau for all c' <= c ('none' when pi(0) > tau). "
             "Excess limit: same with pi(c') - pi(0) <= delta. Grid step 0.01. "
             "'|g=x': ES stated as (1+x) kappa V*_t while VaR is cut (ES decoupled). "
             "'dir.': too-few-exception rejections not counted.\n")
    for name, cf in res["E2"]["configs"].items():
        n = cf["n_paths"]
        L.append(f"{name}, seed {cf['seed']}, N = {n}, nu = {f3(cf['nu']) if cf['nu'] else 'n/a'}, "
                 f"max binomial s.e. {0.5 / np.sqrt(n):.4f}\n")
        L.append("| test or panel | pi(0) | pi(.05) | pi(.10) | pi(.15) | pi(.20) | pi(.30) | pi(.50) | c_0.10 | c_0.25 | c_0.50 | c_0.80 | excess 0.05 | excess 0.25 | excess 0.50 |")
        L.append("|---|" + "---|" * 14)
        for r in rows:
            pi = cf["curves"][r]
            lim = cf["limits"][r]
            L.append(f"| {r} | " + " | ".join(f"{pi[idx[c]]:.3f}" for c in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5))
                     + " | " + " | ".join(fc(lim["c_tau"][str(t)]) for t in (0.1, 0.25, 0.5, 0.8))
                     + " | " + " | ".join(fx(lim, pi[0], d) for d in (0.05, 0.25, 0.5)) + " |")
        L.append("")

    # bps conversion
    L.append("### Table S4. Feasible haircut limits in basis points and dollars (P3 and P3+E, limit c times mean V* of the representative window; USD 1 billion notional)\n")
    L.append("| config | panel | limit | c | AAPL bps | SPY bps | BTC-USD bps | AAPL USD | SPY USD | BTC-USD USD |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, cf in res["E2"]["configs"].items():
        for panel in ("P3", "P3+E"):
            lim = cf["limits"][panel]
            for lab, c in (("c_0.50", lim["c_tau"]["0.5"]), ("excess 0.05", lim["c_excess"]["0.05"]),
                           ("excess 0.25", lim["c_excess"]["0.25"])):
                if c is None:
                    L.append(f"| {name} | {panel} | {lab} | none | | | | | | |")
                    continue
                bps = [c * A[a]["mean_var_bps"] for a in F.ASSETS]
                usd = [b * 1e-4 * F.NOTIONAL for b in bps]
                L.append(f"| {name} | {panel} | {lab} | {c:.2f} | " + " | ".join(f"{b:.1f}" for b in bps)
                         + " | " + " | ".join(f"{u / 1e6:.1f}m" for u in usd) + " |")
    L.append("")

    # E3 horizons
    L.append("### Table S5. E3, horizons: Kupiec POF and e-backtest at T = 250, 500, 1000\n")
    for name, cf in res["E3h"]["configs"].items():
        L.append(f"{name}, seed {cf['seed']}, N = {cf['n_paths']}\n")
        L.append("| test | T | pi(0) | pi(.05) | pi(.10) | pi(.20) | pi(.30) | c_0.50 | c_0.80 | excess 0.25 | median first crossing day at c=.10 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for T in res["E3h"]["horizons"]:
            for k in [f"kupiec_pof|T={T}", f"e_backtest[coupled]|T={T}", f"e_backtest[0.0]|T={T}",
                      f"e_backtest[0.2]|T={T}", f"e_backtest[0.4]|T={T}"]:
                pi = cf["curves"][k]
                lim = cf["limits"][k]
                med = cf["e_cross_median_day"][str(T)][idx[0.1]] if "coupled" in k else None
                L.append(f"| {k.split('|')[0]} | {T} | " + " | ".join(f"{pi[idx[c]]:.3f}" for c in (0.0, 0.05, 0.1, 0.2, 0.3))
                         + f" | {fc(lim['c_tau']['0.5'])} | {fc(lim['c_tau']['0.8'])} | {fx(lim, pi[0], 0.25)} | "
                         + ("" if med is None or med != med else f"{med:.0f}") + " |")
        L.append("")

    # E4
    L.append("### Table S6. E4, ex post uniform limit s* on simulated windows\n")
    L.append("| asset | law | panel | seed | share definable (S0 passes) | p5 | p25 | p50 | p75 | p95 | share s*=1 | realized s* | realized bps | percentile of realized | bisection cross-check mismatches |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a, e in res["E4"].items():
        for law, rec in e["laws"].items():
            for p in ("P3", "P4"):
                r = rec[p]
                rz = e["realized"][p]
                L.append(f"| {a} | {law} | {p} | {rec['seed']} | {r['share_definable']:.3f} | "
                         + " | ".join(f"{r['pct'][q]:.4f}" for q in ("5", "25", "50", "75", "95"))
                         + f" | {r['share_s_eq_1']:.3f} | {rz['s_star']:.4f} | {rz['bps']:.2f} | {r['realized_percentile_rank']:.1f} | "
                         f"{r['bisection_mismatch']}/{r['bisection_checked']} |")
    L.append("")

    # E5
    L.append("### Table S7. E5, count-feedback haircut S2 (haircut c until k exceptions, then V*)\n")
    L.append("Understatement is the mean over days and paths of (V*_t - V_t)/V*_t. 'S1 at same understatement' "
             "is the E2 P3 curve of the same law interpolated at that value (different seeds).\n")
    e2 = res["E2"]["configs"]
    for name, cf in res["E5"]["configs"].items():
        e2c = e2[name]["curves"]["P3"]
        L.append(f"{name}, seed {cf['seed']}, N = {cf['n_paths']}; S0 rejection P3 {cf['S0']['P3']:.3f}, P3+E {cf['S0']['P3+E']:.3f}\n")
        L.append("| k | c | understatement | P3 | P3 dir. | P3+E | e-backtest | P4 | S1 P3 at same understatement |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        cgr = [0.0] + res["E5"]["c_grid"]
        for k, rk in cf["k"].items():
            for j, c in enumerate(cgr):
                if j == 0:
                    continue
                u = rk["understatement_frac"][j]
                s1 = float(np.interp(u, cg, e2c))
                L.append(f"| {k} | {c:.2f} | {u:.4f} | {rk['rej']['P3'][j]:.3f} | {rk['rej']['P3dir'][j]:.3f} | "
                         f"{rk['rej']['P3+E'][j]:.3f} | {rk['rej']['e_backtest'][j]:.3f} | {rk['rej']['P4'][j]:.3f} | {s1:.3f} |")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(res, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    cg = np.array(res["E2"]["c_grid"])
    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    cf = res["E2"]["configs"]["D1/R1 pooled"]
    ax = axes[0]
    for key, lab, ls, col in (("P3", "P3 (regulatory three)", "-", "#1B3DFF"),
                              ("P4", "P4 (P3 + conformal)", "-", "#9A968C"),
                              ("e_backtest", "e-backtest, ES tied to VaR", "--", "#D7263D"),
                              ("e_backtest|g=0.4", "e-backtest, ES +40%", ":", "#D7263D"),
                              ("P3+E", "P3 + e-backtest", "-.", "#0F0E0B")):
        ax.plot(cg, cf["curves"][key], ls, color=col, label=lab, lw=1.6)
    ax.axhline(cf["curves"]["P3"][0], color="#1B3DFF", lw=0.8, alpha=0.5)
    ax.set_title("(a) Gaussian shocks, T = 250", fontsize=10)
    ax.set_xlabel("constant haircut c")
    ax.set_ylabel("rejection probability")
    ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[1]
    cols = {"AAPL": "#1B3DFF", "SPY": "#00B870", "BTC-USD": "#E8A33D"}
    for a in F.ASSETS:
        c2 = res["E2"]["configs"][f"D2/R1 {a}"]
        ax.plot(cg, c2["curves"]["P3"], "-", color=cols[a], lw=1.6, label=f"{a}: P3")
        ax.plot(cg, c2["curves"]["e_backtest"], "--", color=cols[a], lw=1.4, label=f"{a}: e-backtest")
    ax.set_title("(b) Student-t shocks, T = 250", fontsize=10)
    ax.set_xlabel("constant haircut c")
    ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[2]
    c3 = res["E3h"]["configs"]["D1/R1 pooled"]
    for T, lw in zip(res["E3h"]["horizons"], (1.0, 1.5, 2.0)):
        ax.plot(cg, c3["curves"][f"kupiec_pof|T={T}"], "-", color="#1B3DFF", lw=lw, label=f"Kupiec POF, T={T}")
        ax.plot(cg, c3["curves"][f"e_backtest[coupled]|T={T}"], "--", color="#D7263D", lw=lw,
                label=f"e-backtest, T={T}")
    ax.set_title("(c) Gaussian shocks, longer horizons", fontsize=10)
    ax.set_xlabel("constant haircut c")
    ax.legend(fontsize=7.5, loc="lower right")
    for ax in axes:
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 0.5)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"fig_power_vs_haircut.{ext}")
        fig.savefig(p, dpi=200)
        paths.append(p)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    bins = np.linspace(0, 0.5, 51)
    for ax, a in zip(axes, F.ASSETS):
        e = res["E4"][a]
        for law, col in (("D1", "#1B3DFF"), ("D2", "#E8A33D")):
            vals = e["laws"][law]["P3"]["_vals"]
            ax.hist(np.clip(vals, 0, 0.5), bins=bins, density=True, histtype="step", color=col, lw=1.5,
                    label=f"{'Gaussian' if law == 'D1' else 'Student-t'} shocks")
        ax.axvline(e["realized"]["P3"]["s_star"], color="#0F0E0B", lw=1.2, ls="--", label="observed window")
        ax.set_title(f"{a}, panel P3", fontsize=10)
        ax.set_xlabel("ex post uniform limit s*")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"fig_expost_uniform_limit.{ext}")
        fig.savefig(p, dpi=200)
        paths.append(p)
    plt.close(fig)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny N smoke run; writes to a _quick suffix")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "paper", "revision-R1"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--tables-only", action="store_true",
                    help="rebuild sim_tables.md from an existing sim_results.json (no simulation)")
    args = ap.parse_args(argv)
    suf = "_quick" if args.quick else ""
    if args.tables_only:
        with open(os.path.join(args.out_dir, f"sim_results{suf}.json")) as fh:
            res = json.load(fh)
        with open(os.path.join(args.out_dir, f"sim_tables{suf}.md"), "w") as fh:
            fh.write(tables_md(res))
        return
    os.makedirs(args.out_dir, exist_ok=True)
    log = Tee(os.path.join(args.out_dir, f"sim_console{suf}.log"))
    gi = git_info()
    started = dt.datetime.now().astimezone()
    log(f"started {started.isoformat(timespec='seconds')} | git {gi['commit']} ({gi['branch']}, "
        f"code dirty={gi['dirty']}) | python {platform.python_version()} numpy {np.__version__} | "
        f"{platform.platform()}")
    t0 = time.time()
    res = F.run_all(log=log, quick=args.quick)
    fig_dir = args.fig_dir if not args.quick else os.path.join(args.out_dir, "quick_figs")
    figs = make_figures(res, fig_dir)
    res["meta"] = {"started": started.isoformat(timespec="seconds"), "git": gi,
                   "python": platform.python_version(), "numpy": np.__version__,
                   "command": "uv run python scripts/run_feasibility.py" + (" --quick" if args.quick else ""),
                   "wall_clock_s": time.time() - t0, "figures": figs}
    with open(os.path.join(args.out_dir, f"sim_results{suf}.json"), "w") as fh:
        json.dump(_clean(res), fh, indent=1)
    with open(os.path.join(args.out_dir, f"sim_tables{suf}.md"), "w") as fh:
        fh.write(tables_md(res))
    log(f"figures: {figs}")
    log(f"wall-clock including figures and tables: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
