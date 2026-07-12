"""End-to-end runner for the Gaming Oracle.

Dispatches on --cycle (default 2). Cycle 1 is the single-asset MVP: load the
asset, build the honest reference VaR, run the greedy projected search against
the 4-gate panel, price the lie three ways, and print the headline; it also runs
the killer demo (add a one-sided UC score test as a 5th gate and show the Maximum
Undetectable Lie SHRINKS). Cycle 2 (the default, no --cycle) is multi-asset +
rolling distribution + uniform decomposition; 3 = non-uniform MILP; 4 = defense
(e-process ES backtest); 7 = foundation repair (regulatory-panel headline +
tolerance map).

Usage:
    uv run python -m gaming_oracle.cli --cycle 1 --asset AAPL --alpha 0.01 --notional 1e9
    uv run python -m gaming_oracle.cli --asset AAPL   # no --cycle = Cycle 2
"""

from __future__ import annotations

import argparse

from . import gates as gates_mod
from .data import load_data
from .decomposition import leave_one_out, representative_window
from .milp_decomposition import leave_one_out_milp
from .milp_solver import solve_milp
from .report import headline, price
from .rolling import rolling_mul
from .solver import solve

CYCLE2_ASSETS = ["AAPL", "SPY", "BTC-USD"]


def run(asset: str = "AAPL", alpha: float = 0.01, method: str = "garch",
        notional: float = 1_000_000_000.0, use_cache: bool = True,
        window: int = 250) -> dict:
    print(f"== Gaming Oracle (Cycle 1 MVP) ==  asset={asset} alpha={alpha} "
          f"reference={method} window={window}")
    print("Math is prior (Kupiec 1995; Christoffersen 1998; Basel/BCBS 1996; "
          "Du-Escanciano 2017). Contribution = the artifact + priced metric.")
    print("Sim-search EXHIBITS a lie; it does not claim the global maximum.\n")

    ds = load_data(asset=asset, alpha=alpha, method=method, use_cache=use_cache)
    n_full = ds.returns.size

    # The gate panel (Basel traffic-light especially) is defined over a 250-day
    # window. We evaluate on the most recent ``window`` days, where the honest
    # reference is panel-calibrated. The full-sample reference can under-cover
    # (fat tails) and itself fail the panel; that mismatch is between panel and
    # window, not a property of the lie. window<=0 means use the full sample.
    if window and window > 0 and window < n_full:
        returns = ds.returns[-window:]
        true_var = ds.true_var[-window:]
    else:
        returns = ds.returns
        true_var = ds.true_var
    n = returns.size
    print(f"Loaded {n_full} daily returns; evaluating on most recent {n}-day "
          f"window. Honest reference VaR ({method}) built.")

    e_true = gates_mod.exceptions(true_var, returns)
    print(f"Honest reference exception rate on window: {e_true.mean():.4f} "
          f"(nominal {alpha:.4f}); count {int(e_true.sum())}/{n}")
    honest_panel = gates_mod.run_panel(true_var, returns, panel=gates_mod.PANEL,
                                       alpha=alpha)
    print(f"Honest reference passes 4-gate panel on window: "
          f"{honest_panel['passed_all']}\n")

    # --- 4-gate panel ---
    panel4 = gates_mod.PANEL
    res4 = solve(true_var, returns, panel=panel4, alpha=alpha)
    p4 = price(res4.stated_var, res4.true_var, res4.returns, notional=notional,
               panel_names=[g.__name__ for g in panel4])
    print("--- 4-gate panel (Kupiec, Christoffersen, Basel, Conformal) ---")
    print(f"shrink s*={res4.shrink:.4f}  refined={res4.refined}  "
          f"passed_all={res4.passed_all}")
    for name, gr in res4.gate_results.items():
        print(f"  {name:22s} passed={gr.passed!s:5s} stat={gr.stat:.4f}")
    print(f"MUL: {p4.mul_bps:.2f} bps | ${p4.mul_usd:,.0f} on ${notional:,.0f} "
          f"| {p4.extra_exceptions_per_year:.2f} extra exceptions/yr")
    print("HEADLINE: " + headline(p4, asset) + "\n")

    # --- full 5-gate panel (augmented panel + one-sided UC score test) ---
    panel5 = gates_mod.PANEL_WITH_DE
    res5 = solve(true_var, returns, panel=panel5, alpha=alpha)
    p5 = price(res5.stated_var, res5.true_var, res5.returns, notional=notional,
               panel_names=[g.__name__ for g in panel5])
    print("--- 5-gate panel (augmented + one-sided UC score test 'oneside_uc_score') ---")
    print(f"shrink s*={res5.shrink:.4f}  refined={res5.refined}  "
          f"passed_all={res5.passed_all}")
    for name, gr in res5.gate_results.items():
        print(f"  {name:22s} passed={gr.passed!s:5s} stat={gr.stat:.4f}")
    print(f"MUL: {p5.mul_bps:.2f} bps | ${p5.mul_usd:,.0f} on ${notional:,.0f} "
          f"| {p5.extra_exceptions_per_year:.2f} extra exceptions/yr")
    diff = p4.mul_bps - p5.mul_bps
    if abs(diff) < 1e-6:
        print("Note: on this window the binding gate is conformal/Basel, which "
              "already caps the lie below where the one-sided UC test would bind, so "
              "adding it to the full panel changes nothing. The detection gain "
              "shows in the canonical Kupiec comparison below.\n")
    else:
        print(f"Adding the one-sided UC test to the full panel shrinks the lie by "
              f"{diff:.2f} bps.\n")

    # --- KILLER DEMO: one-sided UC score test vs the two-sided Kupiec LR ---
    # The 5th gate is the ONE-SIDED unconditional-coverage score test, a
    # higher-power alternative to the two-sided Kupiec POF LR against the
    # understatement direction (in the spirit of Du & Escanciano 2017 and the
    # VaR/ES backtest-power literature). The clean demonstration is Kupiec-only
    # vs Kupiec+DE: the one-sided test detects under-coverage Kupiec misses,
    # shrinking the exhibited lie.
    res_k = solve(true_var, returns, panel=[gates_mod.kupiec_pof], alpha=alpha)
    p_k = price(res_k.stated_var, true_var, returns, notional=notional,
                panel_names=["kupiec_pof"])
    res_kde = solve(true_var, returns,
                    panel=[gates_mod.kupiec_pof, gates_mod.oneside_uc_score], alpha=alpha)
    p_kde = price(res_kde.stated_var, true_var, returns, notional=notional,
                  panel_names=["kupiec_pof", "oneside_uc_score"])
    shrink_bps = p_k.mul_bps - p_kde.mul_bps
    pct = (shrink_bps / p_k.mul_bps * 100.0) if p_k.mul_bps else 0.0
    print("--- KILLER DEMO: Kupiec-only vs Kupiec + one-sided UC score test ---")
    print(f"Kupiec-only        : {p_k.mul_bps:.2f} bps | ${p_k.mul_usd:,.0f} | "
          f"{p_k.extra_exceptions_per_year:.2f} extra exceptions/yr")
    print(f"Kupiec + 1-sided UC: {p_kde.mul_bps:.2f} bps | ${p_kde.mul_usd:,.0f} | "
          f"{p_kde.extra_exceptions_per_year:.2f} extra exceptions/yr")
    print(f"=> Adding the one-sided UC score test shrinks the exhibited lie by "
          f"{shrink_bps:.2f} "
          f"bps ({pct:.1f}% smaller), and tolerated extra exceptions/yr drop "
          f"{p_k.extra_exceptions_per_year:.2f} -> "
          f"{p_kde.extra_exceptions_per_year:.2f}.")
    print("This quantifies how much detection the higher-power test buys.")

    return {
        "asset": asset, "n": n, "alpha": alpha, "method": method, "window": window,
        "panel4": p4, "panel5": p5, "res4": res4, "res5": res5,
        "killer_kupiec": p_k, "killer_kupiec_de": p_kde,
        "killer_shrink_bps": shrink_bps,
    }


def run_cycle2(assets=None, alpha: float = 0.01, method: str = "garch",
               notional: float = 1_000_000_000.0, window: int = 250,
               step: int = 25, use_cache: bool = True, panel=None) -> dict:
    """Cycle 2: multi-asset, rolling-window MUL distribution, and the
    leave-one-out gate-contribution decomposition.

    The headline is now the DISTRIBUTION across rolling windows per asset, not
    one cherry-pickable window; and the gameability decomposition replaces the
    thin Cycle-1 killer demo.
    """
    if assets is None:
        assets = CYCLE2_ASSETS
    if panel is None:
        panel = gates_mod.PANEL
    panel_names = [g.__name__ for g in panel]

    print("== Gaming Oracle (Cycle 2) ==")
    print(f"assets={assets} alpha={alpha} reference={method} "
          f"window={window} step={step}")
    print(f"panel={panel_names}")
    print("Math is prior (Kupiec 1995; Christoffersen 1998; Basel/BCBS 1996; "
          "Du-Escanciano 2017). Contribution = artifact + priced metric.")
    print("Every number is PANEL-RELATIVE and WINDOW-AWARE. Sim-search EXHIBITS "
          "a lie; it does not prove the global maximum.\n")

    out = {"assets": {}, "alpha": alpha, "method": method,
           "window": window, "step": step, "panel_names": panel_names}

    for asset in assets:
        ds = load_data(asset=asset, alpha=alpha, method=method, use_cache=use_cache)
        rr, tv = ds.returns, ds.true_var
        print(f"################ {asset} "
              f"(n={rr.size} daily returns) ################")

        # --- 1. ROLLING-WINDOW MUL DISTRIBUTION ---
        dist = rolling_mul(rr, tv, asset=asset, window=window, step=step,
                           panel=panel, alpha=alpha, notional=notional)
        print("[ROLLING MUL DISTRIBUTION]")
        print("  " + dist.summary())

        # --- 2. GATE-CONTRIBUTION DECOMPOSITION on a representative window ---
        if dist.n_definable > 0:
            ws, we = representative_window(rr, tv, window=window, step=step,
                                           panel=panel, alpha=alpha)
            decomp = leave_one_out(rr, tv, asset=asset, window_start=ws,
                                   window_end=we, panel=panel, alpha=alpha,
                                   notional=notional)
            print("[GATE-CONTRIBUTION DECOMPOSITION]")
            print(decomp.ranking_table())
        else:
            decomp = None
            print("[GATE-CONTRIBUTION DECOMPOSITION]")
            print("  skipped: no window has a panel-passing honest reference, "
                  "so the lie is undefinable and a decomposition is meaningless.")
        print()
        out["assets"][asset] = {"rolling": dist, "decomposition": decomp}

    return out


def run_cycle3(assets=None, alpha: float = 0.01, method: str = "garch",
               notional: float = 1_000_000_000.0, window: int = 250,
               step: int = 25, use_cache: bool = True, panel=None) -> dict:
    """Cycle 3: NON-UNIFORM lie solver (MILP over the discrete exception
    sequence) vs the uniform-shrink solver, on a matched representative window
    per asset, plus the gate-contribution decomposition under the MILP.

    Answers the open question: does a non-uniform attacker make Kupiec and/or
    Christoffersen BIND (so the Cycle-2 dead-weight finding does NOT survive), or
    does conformal still dominate? Also reports non-uniform MUL vs uniform MUL
    (does non-uniform hide more risk).
    """
    if assets is None:
        assets = CYCLE2_ASSETS
    if panel is None:
        panel = gates_mod.PANEL
    panel_names = [g.__name__ for g in panel]

    print("== Gaming Oracle (Cycle 3 — NON-UNIFORM MILP solver) ==")
    print(f"assets={assets} alpha={alpha} reference={method} "
          f"window={window} step={step}")
    print(f"panel={panel_names}")
    print("Math is prior (Kupiec 1995; Christoffersen 1998; Basel/BCBS 1996; "
          "Du-Escanciano 2017). Contribution = artifact + priced metric.")
    print("The MILP optimizes EXACTLY over the count/window gates (Kupiec, "
          "conformal, Basel, DE); Christoffersen is verified post-hoc. Where the "
          "MILP optimum is full-panel feasible we say MAXIMUM; where it needs the "
          "labeled de-clustering repair we say exhibit.\n")

    out = {"assets": {}, "alpha": alpha, "method": method,
           "window": window, "step": step, "panel_names": panel_names}

    for asset in assets:
        ds = load_data(asset=asset, alpha=alpha, method=method, use_cache=use_cache)
        rr_full, tv_full = ds.returns, ds.true_var
        ws, we = representative_window(rr_full, tv_full, window=window, step=step,
                                       panel=panel, alpha=alpha)
        rr, tv = rr_full[ws:we], tv_full[ws:we]
        print(f"################ {asset} "
              f"(n={rr_full.size}) representative window [{ws}:{we}] "
              f"################")

        # true uniform (scalar) vs greedy (per-point) vs exact MILP on the SAME window
        ru_scalar = solve(tv, rr, panel=panel, alpha=alpha, refine=False)
        ps = price(ru_scalar.stated_var, tv, rr, notional=notional, panel_names=panel_names)
        ru = solve(tv, rr, panel=panel, alpha=alpha)  # greedy per-point (refine=True default)
        pu = price(ru.stated_var, tv, rr, notional=notional, panel_names=panel_names)
        rm = solve_milp(tv, rr, panel=panel, alpha=alpha)
        pm = price(rm.stated_var, tv, rr, notional=notional, panel_names=panel_names)
        u_exc = int(gates_mod.exceptions(ru.stated_var, rr).sum())
        delta = pm.mul_bps - pu.mul_bps
        pct = (pm.mul_bps / pu.mul_bps - 1.0) * 100.0 if pu.mul_bps else 0.0
        xscalar = (pu.mul_bps / ps.mul_bps) if ps.mul_bps else float("nan")
        print("[UNIFORM (scalar) vs GREEDY (per-point) vs MILP (exact) MUL]")
        print(f"  uniform (scalar) : {ps.mul_bps:8.2f} bps | ${ps.mul_usd:,.0f}")
        print(f"  greedy  (per-pt) : {pu.mul_bps:8.2f} bps | ${pu.mul_usd:,.0f} | "
              f"{pu.extra_exceptions_per_year:.2f} extra exc/yr | exc={u_exc}")
        tag = "MAXIMUM" if rm.optimal else "exhibit(repair)"
        print(f"  MILP    (exact)  : {pm.mul_bps:8.2f} bps | ${pm.mul_usd:,.0f} | "
              f"{pm.extra_exceptions_per_year:.2f} extra exc/yr | exc="
              f"{rm.exception_count} | {tag} ({rm.method}, "
              f"christoffersen={rm.christoffersen_handled})")
        print(f"  => MILP hides {delta:+.2f} bps MORE than greedy ({pct:+.1f}%); "
              f"greedy hides ~{xscalar:.0f}x the scalar shrink")

        # decomposition under the MILP
        decomp = leave_one_out_milp(rr_full, tv_full, asset=asset,
                                    window_start=ws, window_end=we, panel=panel,
                                    alpha=alpha, notional=notional)
        print("[NON-UNIFORM GATE-CONTRIBUTION DECOMPOSITION]")
        print(decomp.ranking_table())
        print()
        out["assets"][asset] = {
            "window": (ws, we), "uniform_scalar": ps, "uniform": pu, "non_uniform": pm,
            "milp_result": rm, "decomposition": decomp,
        }

    # plain-English verdict on the dead-weight question
    print("=" * 78)
    print("VERDICT — does a non-uniform attacker make Kupiec/Christoffersen BIND?")
    for asset, a in out["assets"].items():
        d = a["decomposition"]
        print(f"  {asset:8s} binding={d.binding_gates()}  "
              f"dead-weight={d.dead_weight_gates()}")
    print("NOTE (Cycle 7): the binder/dead-weight labels above are a TUNING "
          "ARTIFACT of the")
    print("author-added conformal tol=0.005 (tighter than every regulatory band), "
          "NOT a")
    print("property of the risk tests. Do not quote 'the count tests are dead "
          "weight'. The")
    print("portable finding is the tolerance map (--cycle 7): which test is "
          "load-bearing is")
    print("a tuning choice. What survives here is non-uniform hides MORE, and "
          "Christoffersen")
    print("can bind under a clustering attacker.")
    return out


def run_cycle4(assets=None, alpha: float = 0.01, method: str = "garch",
               notional: float = 1_000_000_000.0, window: int = 250,
               step: int = 25, use_cache: bool = True) -> dict:
    """Cycle 4 — the DEFENSE (gap #7).

    Add the Wang-Wang-Ziegel anytime-valid e-process ES backtest as a
    5th gate behind the existing Gate interface, and re-run the ATTACK against the
    panel WITH the e-backtest. Report whether the lie SHRINKS versus the original
    4-gate panel, on all three assets, under both the uniform projected solver and
    the non-uniform MILP attack.

    The e-backtest is the only gate that does NOT reduce to the exception COUNT:
    it reads breach MAGNITUDE via the ES e-statistic, which is exactly what the
    count-gamer manufactures. So it is the structural answer to the Cycle-2/3
    finding that every count-gate is gameable.
    """
    if assets is None:
        assets = CYCLE2_ASSETS
    panel4 = gates_mod.PANEL
    panelE = gates_mod.PANEL_WITH_EBT
    names4 = [g.__name__ for g in panel4]
    namesE = [g.__name__ for g in panelE]

    print("== Gaming Oracle (Cycle 4 — DEFENSE: e-process ES backtest) ==")
    print(f"assets={assets} alpha={alpha} reference={method} "
          f"window={window} step={step}")
    print(f"attack panel (4 gates) = {names4}")
    print(f"defended panel (+e-backtest) = {namesE}")
    print("e-backtest math is prior: Wang, Wang & Ziegel, 'E-backtesting', "
          "Management Science 72(6):4952-4973 (online 2025, arXiv:2209.00991/2022). "
          "Per-period ES e-statistic + multiplicative e-process + Ville's "
          "inequality (anytime-valid). We claim only the shipped artifact.\n")

    out = {"assets": {}, "alpha": alpha, "method": method, "window": window,
           "step": step, "panel4": names4, "panelE": namesE}

    for asset in assets:
        ds = load_data(asset=asset, alpha=alpha, method=method, use_cache=use_cache)
        rr_f, tv_f = ds.returns, ds.true_var
        ws, we = representative_window(rr_f, tv_f, window=window, step=step,
                                       panel=panel4, alpha=alpha)
        rr, tv = rr_f[ws:we], tv_f[ws:we]
        print(f"################ {asset} (n={rr_f.size}) representative window "
              f"[{ws}:{we}] ################")

        # honest reference under the e-backtest (sanity: it should NOT reject a
        # calibrated reference)
        honest_ebt = gates_mod.e_backtest(tv, rr, alpha=alpha)
        print(f"  honest reference e-backtest: passed={honest_ebt.passed} "
              f"(max e-process {honest_ebt.stat:.2f} vs threshold "
              f"{honest_ebt.detail['threshold']:.0f})")

        # --- UNIFORM attack vs defense ---
        ru4 = solve(tv, rr, panel=panel4, alpha=alpha)
        pu4 = price(ru4.stated_var, tv, rr, notional=notional, panel_names=names4)
        lie_ebt = gates_mod.e_backtest(ru4.stated_var, rr, alpha=alpha)
        ruE = solve(tv, rr, panel=panelE, alpha=alpha)
        puE = price(ruE.stated_var, tv, rr, notional=notional, panel_names=namesE)
        shrink_u = pu4.mul_bps - puE.mul_bps
        pct_u = (shrink_u / pu4.mul_bps * 100.0) if pu4.mul_bps else 0.0

        print("[UNIFORM attack vs e-backtest defense]")
        print(f"  4-gate lie          : {pu4.mul_bps:8.2f} bps | "
              f"${pu4.mul_usd:,.0f} | {pu4.extra_exceptions_per_year:.2f} extra exc/yr")
        print(f"  4-gate lie under EBT : {'PASSES' if lie_ebt.passed else 'CAUGHT'} "
              f"(e-process max {lie_ebt.stat:.1f} vs {lie_ebt.detail['threshold']:.0f}; "
              f"first crossing day {lie_ebt.detail['crossing_index']})")
        print(f"  4+e-backtest lie     : {puE.mul_bps:8.2f} bps | "
              f"${puE.mul_usd:,.0f} | {puE.extra_exceptions_per_year:.2f} extra exc/yr "
              f"| passed_all={ruE.passed_all}")
        print(f"  => DEFENSE SHRINKS the uniform lie by {shrink_u:.2f} bps "
              f"({pct_u:.1f}%).")

        # --- NON-UNIFORM (MILP) attack vs defense ---
        rm4 = solve_milp(tv, rr, panel=panel4, alpha=alpha)
        pm4 = price(rm4.stated_var, tv, rr, notional=notional, panel_names=names4)
        milp_lie_ebt = gates_mod.e_backtest(rm4.stated_var, rr, alpha=alpha)
        rmE = solve_milp(tv, rr, panel=panelE, alpha=alpha)
        pmE = price(rmE.stated_var, tv, rr, notional=notional, panel_names=namesE)
        # the projected UNIFORM solver is the attacker's only full-panel-feasible
        # fallback once the MILP count-optimum is voided by the e-backtest
        print("[NON-UNIFORM (MILP) attack vs e-backtest defense]")
        print(f"  MILP 4-gate lie      : {pm4.mul_bps:8.2f} bps | exc="
              f"{rm4.exception_count} | {('MAXIMUM' if rm4.optimal else 'exhibit')}")
        print(f"  MILP lie under EBT   : {'PASSES' if milp_lie_ebt.passed else 'CAUGHT'} "
              f"(e-process max {milp_lie_ebt.stat:.2e} vs "
              f"{milp_lie_ebt.detail['threshold']:.0f})")
        print(f"  MILP +e-backtest     : passed_all={rmE.passed_all} "
              f"(optimal={rmE.optimal}) -> the count-reduction MILP CANNOT satisfy "
              f"the magnitude-based e-backtest;")
        print(f"     the non-uniform zero-VaR-on-breach lie is VOIDED by the "
              f"defense, forcing the attacker back to the uniform projected solver "
              f"(lie {puE.mul_bps:.2f} bps).")

        out["assets"][asset] = {
            "window": (ws, we),
            "honest_ebt": honest_ebt,
            "uniform_4gate": pu4, "uniform_defended": puE,
            "uniform_4gate_ebt_caught": not lie_ebt.passed,
            "uniform_4gate_ebt_max": float(lie_ebt.stat),
            "uniform_shrink_bps": float(shrink_u), "uniform_shrink_pct": float(pct_u),
            "milp_4gate": pm4, "milp_4gate_ebt_caught": not milp_lie_ebt.passed,
            "milp_4gate_ebt_max": float(milp_lie_ebt.stat),
            "milp_defended_feasible": bool(rmE.passed_all),
            "solve4": ru4, "solveE": ruE, "milp4": rm4,
        }
        print()

    # verdict
    print("=" * 78)
    print("VERDICT — does adding the Wang-Wang-Ziegel e-backtest SHRINK the lie?")
    for asset, a in out["assets"].items():
        print(f"  {asset:8s} uniform: {a['uniform_4gate'].mul_bps:7.2f} -> "
              f"{a['uniform_defended'].mul_bps:7.2f} bps "
              f"(-{a['uniform_shrink_bps']:.2f}, -{a['uniform_shrink_pct']:.1f}%); "
              f"4-gate lie caught by EBT={a['uniform_4gate_ebt_caught']} "
              f"(max E {a['uniform_4gate_ebt_max']:.1f}); "
              f"MILP lie caught by EBT={a['milp_4gate_ebt_caught']}")
    print("Every count-only 4-gate lie is CAUGHT by the magnitude-sensitive "
          "e-backtest; adding it shrinks the uniform projected lie and voids the "
          "non-uniform MILP 'maximum' entirely. Panel-relative, window-aware.")
    print("NOTE (Cycle-7 honesty): the above only beats the ZERO-VaR-ON-BREACH "
          "STRAWMAN, and the ~1e10 e-process value is partly the disclosed e_t=1e6 "
          "cap. For the HONEST defense result against an e-process-AWARE, "
          "magnitude-budgeted, INDEPENDENT-ES attacker, run `--cycle 7`.")
    return out


def run_cycle7(assets=None, alpha: float = 0.01, method: str = "rolling_garch",
               notional: float = 1_000_000_000.0, window: int = 250,
               step: int = 25, use_cache: bool = True) -> dict:
    """Cycle 7 — the FOUNDATION REPAIR.

    Leads every headline with the STANDARD REGULATORY PANEL (Kupiec +
    Christoffersen + Basel), reports the conformal gate separately as a
    non-regulatory author-added gate, replaces the false "count tests are dead
    weight" claim with the TOLERANCE MAP (which gate binds as a function of the
    conformal tolerance), uses the causal ROLLING out-of-sample GARCH reference
    (no look-ahead), and reports the e-process-AWARE attacker's honest residual
    (how much a budget-bounded, independent-ES, e-process-aware attacker STILL
    hides under the e-backtest gate).
    """
    from .tolerance_map import tolerance_map
    from .eaware_attacker import solve_eaware, max_residual_under_panel

    if assets is None:
        assets = CYCLE2_ASSETS
    reg = gates_mod.PANEL_REGULATORY
    aug = gates_mod.PANEL                      # regulatory + conformal
    reg_names = [g.__name__ for g in reg]

    print("== Gaming Oracle (Cycle 7 — FOUNDATION REPAIR) ==")
    print(f"assets={assets} alpha={alpha} reference={method} window={window} step={step}")
    print("Reference is ROLLING OUT-OF-SAMPLE GARCH(1,1) (params fit only on past "
          "data; no look-ahead).")
    print("Every headline LEADS with the STANDARD REGULATORY PANEL "
          f"{reg_names}. conformal_coverage is a NON-REGULATORY, author-added gate, "
          "reported separately, never co-equal.")
    print("Math is prior (Kupiec 1995; Christoffersen 1998; Basel/BCBS 1996; "
          "Wang-Wang-Ziegel 2025). 'Maximum' only where the MILP proves it.\n")

    out = {"assets": {}, "alpha": alpha, "method": method, "window": window,
           "regulatory_panel": reg_names}

    for asset in assets:
        ds = load_data(asset=asset, alpha=alpha, method=method, use_cache=use_cache)
        rr_f, tv_f = ds.returns, ds.true_var
        ws, we = representative_window(rr_f, tv_f, window=window, step=step,
                                       panel=aug, alpha=alpha)
        rr, tv = rr_f[ws:we], tv_f[ws:we]
        n_exc = int(gates_mod.exceptions(tv, rr).sum())
        print(f"################ {asset} (n={rr_f.size}) representative window "
              f"[{ws}:{we}], honest exceptions {n_exc}/{rr.size} ################")

        # --- 1. HEADLINE: regulatory panel first, conformal as a separate add-on ---
        res_reg = solve(tv, rr, panel=reg, alpha=alpha)
        p_reg = price(res_reg.stated_var, tv, rr, notional=notional, panel_names=reg_names)
        res_aug = solve(tv, rr, panel=aug, alpha=alpha)
        p_aug = price(res_aug.stated_var, tv, rr, notional=notional,
                      panel_names=[g.__name__ for g in aug])
        conf_delta = p_reg.mul_bps - p_aug.mul_bps
        print("[HEADLINE — STANDARD REGULATORY PANEL]")
        print(f"  REGULATORY (Kupiec+Christoffersen+Basel): {p_reg.mul_bps:.2f} bps "
              f"| ${p_reg.mul_usd:,.0f} on ${notional:,.0f} | "
              f"{p_reg.extra_exceptions_per_year:.2f} extra exc/yr")
        print(f"  +conformal (author-added, non-regulatory): {p_aug.mul_bps:.2f} bps "
              f"=> the author-added gate moves the headline by only {conf_delta:.2f} "
              f"bps ({(conf_delta/p_reg.mul_bps*100 if p_reg.mul_bps else 0):.1f}%).")

        # --- 2. TOLERANCE MAP: which gate is load-bearing vs the conformal tol ---
        tm = tolerance_map(rr_f, tv_f, asset=asset, window_start=ws, window_end=we,
                           alpha=alpha, notional=notional)
        print("[TOLERANCE MAP — which test is load-bearing is a TUNING CHOICE]")
        print("  " + tm.summary().replace("\n", "\n  "))

        # --- 3. e-PROCESS-AWARE attacker: the honest defense residual ---
        ea_indep = max_residual_under_panel(tv, rr, es_independent=True, alpha=alpha,
                                            notional=notional)
        ea_eonly = solve_eaware(tv, rr, budget=0.20, es_independent=True, alpha=alpha,
                                notional=notional)
        ea_eonly_dep = solve_eaware(tv, rr, budget=0.20, es_independent=False,
                                    alpha=alpha, notional=notional)
        print("[e-PROCESS-AWARE ATTACKER — honest defense residual]")
        print(f"  vs e-process ALONE (budget 0.20, INDEPENDENT ES): hides "
              f"{ea_eonly.mul_bps:.1f} bps while the e-process stays GREEN "
              f"(max {ea_eonly.eprocess_max:.2f} vs threshold "
              f"{ea_eonly.eprocess_threshold:.0f}; max per-period e_t "
              f"{ea_eonly.max_per_period_e:.1f}, far below the disclosed 1e6 cap).")
        print(f"  same budget but ES TIED to VaR (Gaussian multiple): e-process max "
              f"{ea_eonly_dep.eprocess_max:.2f} — the attacker UNDER-states VaR (the "
              f"headline lie) while OVER-stating ES to widen the e-statistic "
              f"denominator (r_t - z_t), dropping the e-process to "
              f"{ea_eonly.eprocess_max:.2f}. The shipped panel's fixed-multiple ES "
              f"forbids exactly this move, so 'cannot dodge ES' was true only by "
              f"construction.")
        print(f"  vs the FULL defended panel (every gate green): the e-aware "
              f"attacker still hides {ea_indep.mul_bps:.1f} bps "
              f"(budget {ea_indep.budget:.2f}); that residual, NOT 'the attack is "
              f"voided', is the honest defense result.\n")

        out["assets"][asset] = {
            "window": (ws, we), "regulatory": p_reg, "augmented": p_aug,
            "conformal_delta_bps": float(conf_delta), "tolerance_map": tm,
            "eaware_full_panel": ea_indep, "eaware_eprocess_only": ea_eonly,
            "eaware_eprocess_only_dep": ea_eonly_dep,
        }

    print("=" * 78)
    print("VERDICT (Cycle 7, honest + panel-relative):")
    for asset, a in out["assets"].items():
        tm = a["tolerance_map"]
        cx = (f"conformal stops binding at tol>={tm.crossover_tol:.4f}"
              if tm.crossover_tol is not None else "conformal binds across grid")
        print(f"  {asset:8s} regulatory-panel lie {a['regulatory'].mul_bps:7.2f} bps; "
              f"conformal adds {a['conformal_delta_bps']:+.2f}; {cx}; "
              f"e-aware full-panel residual {a['eaware_full_panel'].mul_bps:.1f} bps")
    print("- The headline is the REGULATORY-panel lie; the author-added conformal "
          "gate barely moves it.")
    print("- Which test is load-bearing is a TUNING CHOICE (the tolerance map), not "
          "'the count tests are dead weight'.")
    print("- The e-process does NOT void a budget-bounded, independent-ES attacker; "
          "it leaves a real, disclosed residual.")
    return out


def run_headline(asset: str = "AAPL", alpha: float = 0.01,
                 notional: float = 1_000_000_000.0, window: int = 250,
                 use_cache: bool = True) -> dict:
    """THE HEADLINE (paper Section 6.1).

    Reproduces the single number the paper leads with: the maximum undetectable
    lie against the STANDARD REGULATORY PANEL (Kupiec + Christoffersen + Basel),
    on the MOST-RECENT ``window``-day AAPL window, using the causal rolling
    out-of-sample GARCH reference (no look-ahead). This is deliberately a
    single-window headline path so the stated command and the stated number are
    mutually consistent; the per-asset representative-window table and the
    tolerance map live under ``--cycle 7``.
    """
    reg = gates_mod.PANEL_REGULATORY
    reg_names = [g.__name__ for g in reg]

    ds = load_data(asset=asset, alpha=alpha, method="rolling_garch",
                   use_cache=use_cache)
    rr_full, tv_full = ds.returns, ds.true_var
    n_full = rr_full.size
    # most-recent window: the causal reference is calibrated here and passes the
    # regulatory panel, so a lie is genuinely definable.
    if window and 0 < window < n_full:
        rr, tv = rr_full[-window:], tv_full[-window:]
        ws, we = n_full - window, n_full
    else:
        rr, tv = rr_full, tv_full
        ws, we = 0, n_full

    n_exc = int(gates_mod.exceptions(tv, rr).sum())
    print("== Gaming Oracle (HEADLINE — paper Section 6.1) ==")
    print(f"asset={asset} alpha={alpha} reference=rolling_garch "
          f"window=most-recent {rr.size} days [{ws}:{we}] notional=${notional:,.0f}")
    print("Reference is the CAUSAL ROLLING OUT-OF-SAMPLE GARCH(1,1) (params fit "
          "only on past data; no look-ahead).")
    print(f"Standard regulatory panel = {reg_names}. Math is prior (Kupiec 1995; "
          "Christoffersen 1998; Basel/BCBS 1996).")
    print(f"Honest reference exceptions on window: {n_exc}/{rr.size} "
          f"(nominal {alpha:.4f}); the honest reference passes the regulatory panel, "
          "so a lie is definable.\n")

    res = solve(tv, rr, panel=reg, alpha=alpha)
    p = price(res.stated_var, tv, rr, notional=notional, panel_names=reg_names)
    print("[HEADLINE — STANDARD REGULATORY PANEL]")
    print(f"  shrink s*={res.shrink:.4f}  passed_all={res.passed_all}")
    for name, gr in res.gate_results.items():
        print(f"    {name:22s} passed={gr.passed!s:5s} stat={gr.stat:.4f}")
    print(f"  MUL: {p.mul_bps:.2f} bps | ${p.mul_usd:,.0f} on ${notional:,.0f} "
          f"| {p.extra_exceptions_per_year:.2f} extra exceptions/yr")
    print("  HEADLINE: " + headline(p, asset))

    # sensitivity: what the author-added conformal gate does to the same window
    aug = gates_mod.PANEL
    res_aug = solve(tv, rr, panel=aug, alpha=alpha)
    p_aug = price(res_aug.stated_var, tv, rr, notional=notional,
                  panel_names=[g.__name__ for g in aug])
    delta = p.mul_bps - p_aug.mul_bps
    pct = (delta / p.mul_bps * 100.0) if p.mul_bps else 0.0
    print(f"\n  +conformal (author-added, non-regulatory): {p_aug.mul_bps:.2f} bps "
          f"=> the author-added gate moves the headline by only {delta:.2f} bps "
          f"({pct:.1f}%).")
    return {"asset": asset, "window": (ws, we), "n_exceptions": n_exc,
            "regulatory": p, "augmented": p_aug}


def run_power(gates=None, nu=float("inf"), sigma: float = 0.01, window: int = 250,
              ref_understatement: float = 0.25, alpha: float = 0.01,
              level: float = 0.95, target_power: float = 0.80, n_sims: int = 2000,
              seed: int = 0) -> dict:
    """Cycle 8 — the DESK-BACKTESTABILITY POWER CALCULATOR (the Oracle's PRECURSOR).

    The Oracle asks 'how big a lie passes this gate'. This asks the question BEFORE it:
    'can the gate detect a misestimate of this size AT ALL, on this window?' i.e. the
    statistical POWER of the backtest against a stated VaR understatement. For each gate
    it Monte-Carlos P(reject | understated by u) -> a RED/AMBER/GREEN backtestability
    verdict, a minimum-detectable-understatement curve, and a minimum-N for adequate power.

    Math is prior: Kupiec (1995) POF + its own low-power caveat; BCBS (1996) binomial
    traffic-light; Acerbi-Szekely (2014) ES backtests (magnitude-sensitive). The
    contribution is the shipped calculator wired to the SAME gates.py the Oracle uses.
    """
    from .power import backtestability_report

    if gates is None:
        gates = ["kupiec_pof", "basel_binomial", "es_z2_unconditional"]

    tail = "Gaussian" if nu == float("inf") else f"Student-t(nu={nu:g})"
    print("== Gaming Oracle (Cycle 8 — DESK-BACKTESTABILITY POWER CALCULATOR) ==")
    print(f"gates={gates} true-tail={tail} sigma={sigma:g} alpha={alpha:g} "
          f"level={level:g}")
    print(f"window N={window} reference understatement u={ref_understatement:.0%} "
          f"target power={target_power:.0%} n_sims={n_sims}")
    print("This is the Oracle's PRECURSOR: the Oracle asks 'how big a lie passes'; this "
          "asks 'can the gate detect a misestimate AT ALL'.")
    print("Math is prior (Kupiec 1995 POF + its own low-power caveat; BCBS 1996 binomial; "
          "Acerbi-Szekely 2014 ES). Every number is GATE-, TAIL-, and WINDOW-relative; "
          "no maximum/universal claim.\n")

    out = {"gates": {}, "nu": nu, "sigma": sigma, "alpha": alpha, "level": level,
           "window": window, "ref_understatement": ref_understatement,
           "target_power": target_power, "n_sims": n_sims}

    for gate in gates:
        rep = backtestability_report(
            gate, nu=nu, sigma=sigma, n=window,
            ref_understatement=ref_understatement, alpha=alpha, level=level,
            target_power=target_power, n_sims=n_sims, seed=seed)
        print(rep.summary())
        print()
        out["gates"][gate] = rep

    print("=" * 78)
    print("VERDICT — is a desk running each gate on this window even backtestable "
          f"against a {ref_understatement:.0%} VaR understatement?")
    for gate, rep in out["gates"].items():
        mdu = (f"{rep.mdu_at_ref_n:.0%}" if rep.mdu_at_ref_n is not None else "none")
        nstar = (str(rep.min_n_for_ref_u) if rep.min_n_for_ref_u is not None
                 else f">{rep.n_grid[-1]}")
        print(f"  {gate:22s} power {rep.power_at_ref:.2f} -> {rep.verdict.upper():5s} "
              f"| min-detectable u @N={window}: {mdu} | min-N for "
              f"{target_power:.0%} power @u={ref_understatement:.0%}: {nstar}")
    print("This is Kupiec's (1995) low-power caveat made a first-class, per-gate, per-N "
          "number: a small window can leave a desk effectively un-backtestable against a "
          "real understatement. Reported before the Oracle's 'how big a lie passes'.")
    return out


def run_appendix():
    """APPENDIX / proof-of-concept (Cycle 6): run the SAME gameability /
    gate-contribution decomposition on a NON-VaR pass/fail gate -- a classifier
    reliability panel. Clearly labeled, explicitly NOT a grand claim.
    """
    from .appendix_classifier import (
        synthetic_classifier, run_reliability_panel, inflate_confidence,
        price_conf, conf_headline, leave_one_out_conf)

    print("=" * 78)
    print("APPENDIX (proof-of-concept): the engine audits a SECOND, non-VaR gate")
    print("  Domain: ML classifier RELIABILITY (calibration/coverage), NOT VaR.")
    print("  This is a generalization TEASER. One extra instance is not a theorem.")
    print("=" * 78)

    honest_conf, correct = synthetic_classifier()
    print(f"\nSynthetic calibrated classifier: n={honest_conf.size}, "
          f"mean confidence={honest_conf.mean():.4f}, accuracy={correct.mean():.4f}")
    out = run_reliability_panel(honest_conf, correct)
    print(f"  honest reference passes the reliability panel: {out['passed_all']}")
    for name, res in out["gates"].items():
        print(f"    {name:28s} passed={res.passed!s:5s} stat={res.stat:.4f}")

    res = inflate_confidence(honest_conf, correct)
    assert res.passed_all, "solver returned an infeasible series"
    p = price_conf(res.stated_conf, honest_conf, correct)
    print("\n" + conf_headline(p))
    print(f"  honest high-confidence-error rate = {p.hc_error_rate_honest:.4f}; "
          f"after the lie = {p.hc_error_rate_stated:.4f}")

    print()
    print(leave_one_out_conf(honest_conf, correct).ranking_table())
    print("\nNOTE: the binder/dead-weight labels above are TOLERANCE-DEPENDENT (as on")
    print("the VaR side). overconfidence_count_gate binds because its cap "
          "(max_rate=0.02)")
    print("is the tightest of the four tolerances here, not because it is the "
          "'load-bearing'")
    print("reliability check; loosen it or tighten another gate and the binder "
          "moves.")
    print("What this shows: the Gate / shrink-search / leave-one-out machinery is")
    print("not VaR-specific; it runs unchanged on a calibration audit, finds A "
          "binding")
    print("gate, and prices the concealable distortion.")
    print("What it does NOT show: that EVERY pass/fail audit is gameable this way,")
    print("nor that one reliability check is load-bearing and the rest useless.")
    print("A universal claim would need a stated theorem about monotone distortion")
    print("frontiers, which this repo does not make.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gaming Oracle")
    ap.add_argument("--cycle", type=int, default=2, choices=[1, 2, 3, 4, 7, 8],
                    help="1 = single-asset MVP; 2 = multi-asset + rolling + "
                         "uniform decomposition (default); 3 = non-uniform MILP "
                         "solver + decomposition; 4 = DEFENSE (e-process ES "
                         "backtest) attack-vs-defense; 7 = FOUNDATION REPAIR "
                         "(regulatory-panel headline + tolerance map + rolling "
                         "GARCH + e-aware residual); 8 = DESK-BACKTESTABILITY "
                         "POWER CALCULATOR (the Oracle's precursor: can the gate "
                         "detect a misestimate AT ALL)")
    ap.add_argument("--asset", default="AAPL", help="Cycle-1 single asset")
    ap.add_argument("--assets", nargs="+", default=CYCLE2_ASSETS,
                    help="Cycle-2 asset list")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--method", default="garch",
                    choices=["garch", "rolling_garch", "ewma", "empirical"],
                    help="reference VaR model; 'rolling_garch' is the causal "
                         "out-of-sample reference (no look-ahead), the Cycle-7 "
                         "default")
    ap.add_argument("--notional", type=float, default=1_000_000_000.0)
    ap.add_argument("--window", type=int, default=250,
                    help="evaluation window in days (Basel panel = 250); "
                         "0 = full sample (Cycle 1 only)")
    ap.add_argument("--step", type=int, default=25,
                    help="rolling-window stride in days (Cycle 2)")
    ap.add_argument("--no-cache", action="store_true")
    # --- Cycle 8 (power calculator) flags ---
    ap.add_argument("--gates", nargs="+",
                    default=["kupiec_pof", "basel_binomial", "es_z2_unconditional"],
                    help="Cycle-8 gates to power-profile (from gaming_oracle.power.GATES)")
    ap.add_argument("--nu", type=float, default=float("inf"),
                    help="Cycle-8 true-tail Student-t degrees of freedom "
                         "(inf = Gaussian; use e.g. 5 for a fat tail)")
    ap.add_argument("--sigma", type=float, default=0.01,
                    help="Cycle-8 true-return volatility")
    ap.add_argument("--understatement", type=float, default=0.25,
                    help="Cycle-8 reference VaR understatement u (stated = (1-u)*true)")
    ap.add_argument("--target-power", type=float, default=0.80,
                    help="Cycle-8 target power for MDU / min-N (0.80 = Cohen adequate)")
    ap.add_argument("--n-sims", type=int, default=2000,
                    help="Cycle-8 Monte-Carlo replications per power estimate")
    ap.add_argument("--seed", type=int, default=0, help="Cycle-8 RNG seed")
    ap.add_argument("--appendix", action="store_true",
                    help="run the APPENDIX proof-of-concept: the same "
                         "gameability decomposition on a NON-VaR classifier "
                         "reliability gate (teaser, not a general theorem)")
    ap.add_argument("--headline", action="store_true",
                    help="reproduce the paper's Section 6.1 headline number: the "
                         "regulatory-panel maximum undetectable lie on the "
                         "MOST-RECENT --window-day --asset window under the causal "
                         "rolling-GARCH reference (default AAPL, 250 days => 249.18 bps)")
    args = ap.parse_args(argv)
    if args.headline:
        run_headline(asset=args.asset, alpha=args.alpha, notional=args.notional,
                     window=args.window, use_cache=not args.no_cache)
        return
    if args.appendix:
        run_appendix()
        return
    if args.cycle == 1:
        run(asset=args.asset, alpha=args.alpha, method=args.method,
            notional=args.notional, use_cache=not args.no_cache,
            window=args.window)
    elif args.cycle == 3:
        run_cycle3(assets=args.assets, alpha=args.alpha, method=args.method,
                   notional=args.notional, window=args.window, step=args.step,
                   use_cache=not args.no_cache)
    elif args.cycle == 4:
        run_cycle4(assets=args.assets, alpha=args.alpha, method=args.method,
                   notional=args.notional, window=args.window, step=args.step,
                   use_cache=not args.no_cache)
    elif args.cycle == 7:
        method7 = "rolling_garch" if args.method == "garch" else args.method
        run_cycle7(assets=args.assets, alpha=args.alpha, method=method7,
                   notional=args.notional, window=args.window, step=args.step,
                   use_cache=not args.no_cache)
    elif args.cycle == 8:
        run_power(gates=args.gates, nu=args.nu, sigma=args.sigma, window=args.window,
                  ref_understatement=args.understatement, alpha=args.alpha,
                  target_power=args.target_power, n_sims=args.n_sims, seed=args.seed)
    else:
        run_cycle2(assets=args.assets, alpha=args.alpha, method=args.method,
                   notional=args.notional, window=args.window, step=args.step,
                   use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
