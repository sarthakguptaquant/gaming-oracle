# Gaming Oracle — Cycle 7 (foundation repair: regulatory-panel headlines, tolerance map, causal GARCH, e-aware residual) / Cycle 4 (e-process defense) / Cycle 3 (MILP) / Cycle 2 (rolling distribution + decomposition)

Given the exact panel of backtests a risk validator runs, construct a
risk-understating VaR forecast series that still passes EVERY test jointly, and
price the gap. Headline metric: the **Maximum Undetectable Lie (MUL)**.

> **Cycle 7 hardens the foundation so the work is defensible to an expert.** Four
> corrections, all in [the Cycle-7 section](#cycle-7--foundation-repair) and the
> updated [`PAPER.md`](PAPER.md):
> 1. **Every headline now leads with the STANDARD REGULATORY PANEL** (Kupiec +
>    Christoffersen + Basel). `conformal_coverage` is a **non-regulatory,
>    author-added** gate, reported separately, never co-equal.
> 2. The old "the count tests are dead weight" claim was a **tuning artifact** of
>    the conformal gate's hand-set `tol=0.005` (tighter than every regulatory
>    band). It is replaced by a **tolerance map**: *which test is load-bearing is
>    a tuning choice, and here is the map.*
> 3. The GARCH reference is now **rolling out-of-sample** (parameters fit only on
>    past data) — the look-ahead is removed.
> 4. The e-process "defense voids the attack" result is replaced by the **honest
>    residual** an e-process-aware, magnitude-budgeted, **independent-ES** attacker
>    still hides under the e-backtest gate (the 1e6 cap is disclosed).

> **The write-up is [`PAPER.md`](PAPER.md)** — a practitioner paper / extended
> conference abstract with the prior art, the original contribution, the four
> findings, and the full honest-caveats section. This README is the engineering
> reference; the paper is the honest narrative. All numbers in both are
> reproducible from the cached data (see Reproducibility / Run below).

> **Program context: [`PROGRAM.md`](PROGRAM.md).** The Gaming Oracle is one of four
> tools under a shared thesis — *verification adequacy*, tools that measure when a
> validation number is a lie: the **Power Calculator** (can the gate detect anything,
> `gaming_oracle/power.py`), this **Gaming Oracle** (how big a lie still passes), the
> **Lookahead Auditor** (how much backtest alpha is memorized,
> `gaming_oracle/leakage/`), and **notsofast** (the public runtime guard). They reuse
> the same `Gate(stated, data) -> GateResult` interface where they audit backtest
> gates. See [`PROGRAM.md`](PROGRAM.md) for the wiring and the honest scope limits.

## What this is (and is not)

- **The math is prior.** Kupiec (1995) POF, Christoffersen (1998)
  independence/conditional-coverage, the Basel/BCBS (1996) traffic-light, a
  conformal interval-coverage check, a **one-sided unconditional-coverage
  score/LM test** as the higher-power 5th gate (the score-test form of Kupiec,
  in the spirit of **Du & Escanciano (2017, Management Science 63(4):940-958)**
  and the VaR/ES backtest-power literature; see the citation note below for the
  exact provenance), and **Wang, Wang & Ziegel ("E-backtesting", Management
  Science 72(6):4952-4973, online 2025, DOI 10.1287/mnsc.2023.01659;
  arXiv:2209.00991, 2022)** as the anytime-valid e-process ES backtest that
  defends the panel (Cycle 4). No novel theorem is claimed.
- **The contribution is the shipped artifact** + the named, priced metric, not
  the insight that backtests can be gamed.
- **The lie is panel-relative.** It is undetectable only against the chosen
  panel. The panel is a parameter; this is never a universal "this model lies by
  X bps" claim.
- **Sim-search exhibits a lie.** The Cycle-1 solver is a greedy projected search.
  It reports "the largest lie this search exhibited", never the global maximum.
  The word "maximum" is reserved for the Cycle-3 MILP.

## Layout

- `gaming_oracle/gates.py` — the 3 STANDARD REGULATORY gates (Kupiec POF,
  Christoffersen, Basel traffic-light; `PANEL_REGULATORY`) plus the NON-REGULATORY,
  author-added `conformal_coverage` gate, all behind a common
  `Gate(stated_var, returns) -> GateResult(passed, stat, detail)`; plus the
  one-sided UC score-test gate (`oneside_uc_score`, formerly `du_escanciano`), the
  Wang-Wang-Ziegel e-backtest gate (the defense; `ebacktest.py`), and panel runners.
- `gaming_oracle/tolerance_map.py` — Cycle-7 tolerance map (which gate binds vs the
  conformal tolerance). `gaming_oracle/eaware_attacker.py` — Cycle-7 e-process-aware,
  budgeted, independent-ES attacker (the honest defense residual).
- `gaming_oracle/data.py` — `load_data(asset)` returns real AAPL returns plus an
  honest reference `true_var` (GARCH(1,1) MLE by default; EWMA and rolling
  empirical also available). Returns are cached to `data_cache/`.
- `gaming_oracle/solver.py` — scalar-shrink bisection + greedy per-point
  projected refinement, finding `stated_var <= true_var` that maximizes mean
  understatement subject to passing all gates jointly. Feasible by construction
  (every candidate is validated against the full panel).
- `gaming_oracle/report.py` — prices the MUL three ways: bps, dollars on a
  stated notional, and extra VaR exceptions per year.
- `gaming_oracle/cli.py` — end-to-end runner.
- `tests/` — known-answer self-tests per gate (calibrated passes, overconfident
  fails), solver/report validation-contract tests, and a real-AAPL integration
  test.
- `gaming_oracle/leakage/` — **Lookahead-Leakage Auditor** (sibling module): a
  probe harness that detects TEMPORAL information leakage in backtests of
  LLM / agentic trading strategies (fake-date / entity-mask / embargo probes) and
  estimates how much reported alpha is memorized future knowledge (leakage score
  + alpha-deflation with a paired bootstrap CI). Distinct failure mode from the
  statistical gates above. Runnable now against a synthetic known-leakage model;
  the live-LLM adapter is HELD. See `gaming_oracle/leakage/README.md` and
  `SPEC.md`; run `uv run python -m gaming_oracle.leakage.cli`.

## Run

    uv run python -m gaming_oracle.cli --cycle 1 --asset AAPL      # Cycle-1 flagship (the numbers below)
    uv run python -m gaming_oracle.cli --cycle 1 --asset AAPL --window 0   # full sample
    uv run python -m gaming_oracle.cli --asset AAPL               # no flag = Cycle 2 (rolling distribution)
    uv run python -m pytest -q

## Real numbers (AAPL, 99% VaR, most recent 250-day window, GARCH reference)

The gate panel — Basel traffic-light especially — is defined over a 250-day
window, so the tool evaluates on the most recent 250 days, where the honest
GARCH reference is panel-calibrated (2 exceptions / 250, passes all 4 gates).
On the full 2514-day sample GARCH under-covers AAPL's fat tails (1.51% vs 1.00%
nominal) and itself fails the panel; that is a panel-vs-window mismatch, and the
solver correctly refuses to claim a lie there rather than fabricate one.

On the 250-day window, leading with the **standard regulatory panel** (Cycle-7
discipline; the full-sample-GARCH numbers below are retained for comparability,
the causal rolling-GARCH numbers are in the Cycle-7 section):

- **MUL (STANDARD REGULATORY PANEL — Kupiec + Christoffersen + Basel): 246.80 bps**
  = $24,680,000 on a $1B notional, all three regulatory gates green, on the
  full-sample-GARCH reference. (`--cycle 1 --asset AAPL` prints the 4-gate
  **245.41** headline below plus the Kupiec killer demo. The regulatory-3-gate
  number for the *causal rolling-GARCH* reference is **249.18 bps** and is printed
  directly by `--headline`; see the Cycle-7 section.)
- Adding the **non-regulatory, author-added** `conformal_coverage` gate moves the
  number to 245.41 bps — a **1.40 bps (0.6%) change**. The author-added gate is
  *not* co-equal with the regulatory three and barely affects the headline; the
  Cycle-2/3 claim that it was "load-bearing" was a tuning artifact of its hand-set
  `tol=0.005` (see the [tolerance map](#cycle-7--foundation-repair)).
- Adding the one-sided UC score test (`oneside_uc_score`, formerly mis-named
  `du_escanciano`) to the full panel changes nothing on this window because Basel
  already binds below where it would (an honest null result).

### Killer demo (Kupiec vs Kupiec + one-sided UC score test)

The 5th gate (function name `oneside_uc_score` in the code, renamed from the
former `du_escanciano` alias) is the **one-sided unconditional-coverage score/LM
test** on the exception sequence: the score-test form of Kupiec, made one-sided so
it has higher power against the understatement direction. That is a documented
property of score-based VaR backtests (in the spirit of Du & Escanciano 2017; the
VaR/ES power literature).
The clean demonstration is Kupiec-only vs Kupiec + the one-sided test:

| panel | MUL (bps) | $ on $1B | extra exceptions/yr |
|---|---|---|---|
| Kupiec only | 253.87 | $25,386,751 | 4.03 |
| Kupiec + one-sided UC test | 249.71 | $24,971,142 | 3.02 |

Adding the one-sided test shrinks the exhibited lie by 4.16 bps (1.6%) and, more
tellingly, drops the tolerated extra exceptions/year from 4.03 to 3.02. That is
exactly how much detection the higher-power one-sided test buys on this window.
(This is NOT the full Du-Escanciano 2017 ES tail-risk statistic, which needs the
predictive CDF; it is the hit-based one-sided UC score test that runs on the same
inputs as every other gate. See the citation note in the Cycle-4 section.)

(Numbers are deterministic given the cached AAPL returns through 2024-12-30.)

## Validation contract (re-checked in tests)

- Recompute every gate from scratch on the produced `stated_var`: all pass.
- Re-derive the 3 prices from `stated_var` and `true_var`: they match.
- `stated_var <= true_var` everywhere (genuinely under-reserving, not cheating).
- No hand-tuned figure: every printed number is computed by the tool.

---

# Cycle 2 — multi-asset, rolling-window distribution, gameability decomposition

Cycle 1's headline lived on one 250-day window, which is cherry-pickable. Cycle 2
adds three things, all against the existing component contracts (no rewrite):

- `gaming_oracle/rolling.py` — `rolling_mul(...)` slides a 250-day window in
  fixed strides across the FULL history, solves the lie per window, and reports
  the MUL **distribution** (min / median / max / IQR) plus
  `frac_honest_passes`: the fraction of windows where the honest reference
  itself passes the panel, so a lie is even definable. Stats are computed over
  the definable windows only and the conditioning is reported in the open.
- `gaming_oracle/decomposition.py` — `leave_one_out(...)` re-runs the solver with
  each gate removed one at a time on a representative window and reports how much
  the lie GROWS, plus each gate's standalone (solo) MUL. This is the honest
  replacement for the thin Cycle-1 killer demo: it shows which gates are
  load-bearing and which are dead weight given the others.
- `gaming_oracle/cli.py --cycle 2` — runs all three assets end to end.

## Run

    uv run python -m gaming_oracle.cli --cycle 2 --step 25     # all three assets
    uv run python -m pytest -q

## Real numbers — rolling MUL distribution (99% VaR, GARCH reference, 250-day window, step 25)

The honest GARCH reference does NOT pass the panel on every window (fat-tail
clusters push it into Basel amber or a conformal-coverage miss), so the lie is
only definable on a subset. That subset, and the distribution over it, is the
honest headline — not one window.

| asset | windows | honest passes (definable) | MUL min / median / max (bps) | IQR (bps) | median $ on $1B | median extra exc/yr |
|---|---|---|---|---|---|---|
| AAPL | 92 | 38 (41%) | 225.0 / 241.7 / 276.5 | 15.8 | $24,172,577 | 1.01 |
| SPY | 92 | 25 (27%) | 96.9 / 129.1 / 206.1 | 54.2 | $12,912,633 | 0.00 |
| BTC-USD | 138 | 44 (32%) | 441.7 / 537.3 / 788.6 | 106.4 | $53,725,004 | 0.00 |

BTC's lie budget is the largest (wider tails, looser conformal band in absolute
return terms); SPY's is the smallest. On no asset is the honest reference green
on a majority of windows, which is itself the point: a single green window is not
representative.

## Real numbers — gate-contribution / gameability decomposition (representative window per asset)

Leave-one-out: re-solve with each gate dropped; "grows lie" is how much the MUL
increases when that gate is removed. "solo MUL" is the lie that gate alone would
allow. A gate whose solo MUL equals the full-panel MUL **binds the whole lie
alone**; gates whose removal changes nothing are dead weight given the others.

| asset | window | conformal (solo / grows) | basel (solo / grows) | kupiec (solo / grows) | christoffersen (solo / grows) |
|---|---|---|---|---|---|
| AAPL | [2225:2475] | 241.86 / **3.91** | 245.77 / 0.00 | 250.61 / 0.00 | 251.72 / 0.00 |
| SPY | [275:525] | 129.13 / **1.93** | 131.06 / 0.00 | 134.35 / 0.00 | 135.65 / 0.00 |
| BTC-USD | [1950:2200] | 541.08 / **4.99** | 546.07 / 0.00 | 563.18 / 0.00 | 563.18 / 0.00 |

> **SUPERSEDED BY CYCLE 7 — read this.** The Cycle-2 phrasing below claimed
> "conformal is load-bearing and Kupiec + Christoffersen are dead weight." That
> claim was a **TUNING ARTIFACT** and is RETRACTED as stated. `conformal_coverage`
> is a non-regulatory, author-added gate whose hand-set `tol=0.005` is **tighter
> than every real regulatory band** (Basel green <= 4 exceptions; Kupiec's
> non-rejection interval is wider still), so it binds *by construction*, not
> because interval coverage is the load-bearing risk test. The honest, portable
> finding is the **[tolerance map](#cycle-7--foundation-repair)**: which gate is
> binding is a function of the conformal tolerance, and at any regulatorily
> plausible tolerance (>= ~0.0075) the binding constraint is a **regulatory** gate
> (Basel). The raw decomposition table above is kept only to show the mechanism;
> do not quote "the count tests are dead weight" — that compares a regulatory gate
> against an author-added gate tuned tighter than it.

This is panel-relative and window-aware. A gate that is dead weight on one window
can bind on another; the decomposition reports the window it was run on, and the
rolling distribution above supplies the context so no single decomposition reads
as universal. The killer demo (Cycle 1) still runs via `--cycle 1`.

(All numbers deterministic given cached returns through 2024; two full `--cycle 2`
runs are byte-for-byte identical.)

---

# Cycle 3 — non-uniform lie solver (MILP over the discrete exception sequence)

Cycles 1-2 used a **uniform** shrink: `stated_var = (1 - s) * true_var`
everywhere, one scalar knob. Under that uniform lie the leave-one-out ranked
conformal-coverage as the only binding gate and Kupiec + Christoffersen as
"dead weight". **That ranking was later shown (Cycle 7) to be a tuning artifact of
the author-added conformal `tol=0.005`, not a property of the risk tests — do not
quote it as a live finding; see the [tolerance map](#cycle-7--foundation-repair)
and the SUPERSEDED note in Cycle 2 above.** Cycle 3 was built to probe the
open question behind that ranking, which survives the artifact: does a
**non-uniform** attacker (per-point under-reservation) make Kupiec and/or
Christoffersen BIND that a uniform attacker could not expose — and does non-uniform
hide more risk than uniform?

- `gaming_oracle/milp_solver.py` — `solve_milp(...)`, a MILP over the binary
  exception sequence via `scipy.optimize.milp` (HiGHS). The reduction: every gate
  depends on `stated_var` only through `e_t = 1[loss_t > V_t]`, and given `e` the
  maximum understatement is closed-form per day (exception day -> push `V_t` to 0;
  non-exception day -> `V_t = max(loss_t,0)`). So the objective is **linear in
  `e`** (`maximize sum_t max(loss_t,0)*e_t`) and the count/window gates (Kupiec,
  conformal, oneside_uc_score, Basel-250) are **exact linear constraints** on sums of
  `e`. Those gates are solved to a true optimum, so "maximum" is earned for them.
- **Christoffersen is the one nonlinear gate** (its LR depends on transition
  counts). The MILP does NOT pretend to enforce it. We solve over the linear
  gates, then **verify Christoffersen post-hoc**. If the MILP optimum already
  passes it, the optimum is full-panel feasible and we say **MAXIMUM**. If it
  fails (the count-optimal exceptions clustered), we fall back to a clearly
  labeled greedy **de-clustering repair** and say **exhibit**, never "maximum".
- `gaming_oracle/milp_decomposition.py` — the leave-one-out gate-contribution
  decomposition re-run under the MILP solver, on the same representative window
  per asset as Cycle 2.
- `gaming_oracle/cli.py --cycle 3` — uniform vs non-uniform MUL + the non-uniform
  decomposition + a plain-English verdict, all three assets.

PuLP/CBC was not needed: `scipy.optimize.milp` (HiGHS) solves each 250-variable
window instantly, so the MILP is fully tractable at the window length.

## Run

    uv run python -m gaming_oracle.cli --cycle 3 --step 25     # all three assets
    uv run python -m pytest -q

## Real numbers — non-uniform (MILP) vs uniform MUL (representative window per asset)

| asset | window | uniform MUL (bps) | non-uniform MUL (bps) | non-uniform hides MORE | status |
|---|---|---|---|---|---|
| AAPL | [2225:2475] | 241.86 | **312.01** | +70.15 (+29.0%) | MAXIMUM (full-panel feasible) |
| SPY | [275:525] | 129.13 | **158.48** | +29.36 (+22.7%) | exhibit (heuristic repair; = brute-force max) |
| BTC-USD | [1950:2200] | 541.08 | **674.19** | +133.11 (+24.6%) | MAXIMUM (full-panel feasible) |

**Non-uniform hides more risk than uniform on every asset (+23% to +29%).** The
per-point optimizer spends its lie budget where each day has the most slack
instead of scaling everything proportionally, so the same number of exceptions
buys a larger average understatement.

## Real numbers — non-uniform gate-contribution decomposition

| asset | conformal (solo / grows) | basel (solo / grows) | kupiec (solo / grows) | christoffersen (solo / grows) | binding |
|---|---|---|---|---|---|
| AAPL | 312.01 / **1.17** | 313.18 / 0.00 | 315.46 / 0.00 | 353.59 / 0.00 | conformal |
| SPY | 158.63 / **0.54** | 159.20 / 0.00 | 160.26 / 0.00 | 175.90 / **0.14** | conformal, **christoffersen** |
| BTC-USD | 674.19 / **2.51** | 676.70 / 0.00 | 681.32 / 0.00 | 736.39 / 0.00 | conformal |

## The dead-weight question — answered

> **CYCLE-7 NOTE.** "Dead weight" here is *relative to the author-added conformal
> gate tuned tighter than the regulatory band*, so the framing is misleading (see
> the Cycle-2 superseded note and the tolerance map). The substantive Cycle-3
> result that survives is the **non-uniform vs uniform** comparison (+23-29% more
> hidden) and the genuine, if tiny, **Christoffersen binding on SPY** under a
> clustering attacker. Read the gate-ranking language below through the tolerance
> map, not as "the regulatory count tests are useless."

**Does a non-uniform attacker make Kupiec/Christoffersen bind? Partly, and only
for Christoffersen, only when the worst-loss days are adjacent.**

- **Kupiec POF stays dead weight on all three assets.** It is a function of the
  exception COUNT only, and the count band Kupiec allows is strictly wider than
  conformal's, so conformal always binds first. Non-uniformity does not change
  this — both gates see only the count.
- **Conformal-coverage still dominates** on every asset (it binds alone on AAPL
  and BTC; it is the largest binder on SPY).
- **Christoffersen flips from dead weight to BINDING on SPY** under the
  non-uniform attacker. The two deepest-loss days in SPY's window are *adjacent*,
  so the count-optimal non-uniform lie naturally clusters its exceptions (one
  1->1 transition), which is exactly what the independence test detects. Forcing
  the exceptions to de-cluster costs the attacker **0.14 bps** (158.63 -> 158.48).
  This is verified against a brute-force global optimum, not just the heuristic
  repair: the best full-panel lie (158.48) is strictly below the best lie without
  Christoffersen (158.63), so the binding is real. On AAPL and BTC the
  count-optimal exception days happen to be non-adjacent, so the same lie passes
  Christoffersen for free and it stays dead weight.

**Plain yes/no:** Two distinct things must not be conflated here. (1) The old
*binder/dead-weight ranking* ("conformal load-bearing, count tests dead weight") is
a **tuning artifact** and is not a live finding at all, at any tolerance — read it
through the [tolerance map](#cycle-7--foundation-repair), not as a conclusion.
(2) The *structural, tolerance-independent* question Cycle 3 actually settles is:
does a non-uniform attacker expose a gate a uniform attacker cannot? **Yes for
Christoffersen, narrowly.** A clustering non-uniform lie can make Christoffersen
bind (it did on SPY: the two deepest-loss days are adjacent), which a uniform
attacker could never expose because uniform scaling cannot cluster exceptions. That
binding is genuine but data-dependent (it needs the worst-loss days temporally
clustered) and tiny (sub-bp). Kupiec, by contrast, sees only the exception COUNT,
so non-uniformity cannot make it bind under any tolerance — that is a mechanical
fact about the test, independent of the retracted ranking. The honest Cycle-3
result is the non-uniform-hides-more comparison (+23-29%) plus this Christoffersen
backstop; it is NOT a revival of "the count tests are dead weight."

## Validation contract (re-checked in Cycle-3 tests)

- Recompute every gate from scratch on the MILP `stated_var`: all pass.
- `stated_var <= true_var` everywhere, `stated_var >= 0` everywhere.
- Non-uniform MUL >= uniform MUL (the uniform lie is one feasible point of the
  MILP, so the optimum can never be smaller) — asserted on synthetic and real data.
- The count-feasibility intervals match the gate pass-conditions exactly
  (Kupiec/conformal/DE boundary counts pass; one past the boundary fails).
- The `optimal=True` flag implies full-panel feasibility (a true maximum).
- SPY's Christoffersen binding is cross-checked against a brute-force global
  optimum, so it is not an artifact of the heuristic repair.

(All numbers deterministic given cached returns through 2024.)

---

# Cycle 4 — the DEFENSE: an anytime-valid e-process ES backtest (gap #7)

Cycles 1-3 established the attack and one structural fact: **every gate in the
4-gate panel depends on the stated risk series ONLY through the binary exception
sequence** `e_t = 1[loss_t > V_t]`. That is exactly why the Oracle wins — it
pushes stated VaR to ~0 on the days it concedes as exceptions and to
`max(loss_t,0)` on the rest, hiding unbounded tail MAGNITUDE while the exception
COUNT stays inside every gate's band. A count-only panel cannot see how deep each
breach is.

Cycle 4 adds the structural answer: the **e-backtest of Wang, Wang & Ziegel**, a
magnitude-sensitive, anytime-valid Expected-Shortfall backtest, as a gate behind
the same `Gate(stated_var, returns) -> GateResult` interface.

- `gaming_oracle/ebacktest.py` — the e-process. Per-period ES e-statistic

      e_t = (loss_t - z_t)_+ / [ (1 - p) * (r_t - z_t) ],   p = 1 - alpha,

  with `z_t` = stated VaR, `r_t` = stated ES. Under the null "risk is not
  under-forecast", `E[e_t] <= 1`. The e-values are accumulated into a
  **method-of-mixtures** e-process `M_t = mean_k prod_{s<=t}(1 - lam_k + lam_k e_s)`
  over a fixed lambda grid; H0 is rejected the first time `M_t >= 1/(1-level)`.
  Because `(M_t)` is a non-negative supermartingale under H0, **Ville's
  inequality** controls the size at any stopping time (anytime-valid). The mixture
  is the robust default — a single deep breach is caught even when it is one spike
  among many zero-breach days, which the slow-bootstrapping GREE plug-in misses.
- `gaming_oracle/gates.py` — `e_backtest(...)` adapts it to the common signature
  by deriving the stated ES from the stated VaR via the consistent Gaussian ES/VaR
  ratio (1.1457 at `alpha=0.01`), so the SAME lie that shrinks stated VaR shrinks
  stated ES proportionally: the Oracle attacks ES exactly as it attacks VaR, no
  free pass. `PANEL_WITH_EBT = PANEL + [e_backtest]`.
- `gaming_oracle/cli.py --cycle 4` — re-runs the attack against the defended panel,
  under both the uniform projected solver and the non-uniform MILP attack, all
  three assets.
- `gaming_oracle/figures.py` — the three publication PNGs (see below).

**The math is prior, cited up front and never claimed as ours:** Qiuqi Wang,
Ruodu Wang, Johanna Ziegel, *"E-backtesting"*, Management Science 72(6):4952-4973
(published online 2025-09-23, DOI 10.1287/mnsc.2023.01659; print issue June 2026;
preprint arXiv:2209.00991 / SSRN 4206997, 2022). The per-period ES e-statistic,
the multiplicative e-process, and the Ville/supermartingale anytime-validity
argument are all theirs. Our contribution is the shipped gate + the
attack-vs-defense lie-budget number.

> **Citation note (5th gate provenance).** Earlier drafts of this repo cited the
> 5th "killer-demo" gate as "Du & Escanciano (2019, J. Banking & Finance)". That
> citation was wrong and has been corrected throughout. The gate the code runs is
> a **one-sided unconditional-coverage score/LM test** on the exception sequence
> (the one-sided score form of the Kupiec 1995 POF restriction). It is in the
> *spirit* of the higher-power score-based VaR/ES backtests — Du & Escanciano,
> *"Backtesting Expected Shortfall: Accounting for Tail Risk"*, Management Science
> 63(4):940-958 (2017); Christoffersen & Pelletier, *"Backtesting Value-at-Risk: A
> Duration-Based Approach"*, J. Financial Econometrics 2(1):84-108 (2004);
> Escanciano & Olmo, *"Backtesting Parametric
> Value-at-Risk With Estimation Risk"*, JBES 28(1):36-51 (2010) — but it is NOT
> any of their full statistics, which require the predictive CDF or violation
> durations. The function is named `du_escanciano` only as a stable code
> identifier; the test it computes is the one-sided UC score test documented in
> its docstring.

## Run

    uv run python -m gaming_oracle.cli --cycle 4          # attack vs defense, 3 assets
    uv run python -m gaming_oracle.figures   # write the 4 PNGs (matplotlib is a core dep)
    uv run python -m pytest -q

## Real numbers — attack vs defense (representative window per asset, 99% level, GARCH reference)

> **SUPERSEDED BY CYCLE 7.** The "voids the non-uniform attack entirely" claim
> below only beats a **strawman**: the zero-VaR-on-breach count-reduction MILP,
> which self-destructs against any ES backtest, and whose ~1e10 e-process value is
> partly the **disclosed hardcoded `e_t=1e6` cap** on collapsed-gap breaches. The
> honest defense result is in the [Cycle-7 section](#cycle-7--foundation-repair):
> an e-process-**aware**, magnitude-budgeted, **independent-ES** attacker keeps the
> e-process green and still hides a real residual (14 / 4 / 42 bps full-panel on
> AAPL / SPY / BTC), because it can over-state ES to neutralize the e-statistic — a
> move the shipped fixed-multiple ES forbids by construction. Keep the tables below
> as the strawman baseline; do not quote "voided" as the defense result.

Does adding the Wang-Wang-Ziegel e-backtest SHRINK the lie versus the original
4-gate panel? **Against a zero-VaR-on-breach strawman, yes** (the honest, e-aware
result is in Cycle 7):

### Uniform projected attack (the solver re-validates every candidate against the full panel)

| asset | window | 4-gate lie (bps) | 4-gate lie caught by e-backtest? (e-process max vs 20) | 4 + e-backtest lie (bps) | shrink |
|---|---|---|---|---|---|
| AAPL | [2225:2475] | 241.86 | **CAUGHT** (1772.0) | 239.85 | -2.01 bps (-0.8%) |
| SPY | [275:525] | 129.13 | **CAUGHT** (7916.4) | 128.14 | -0.99 bps (-0.8%) |
| BTC-USD | [1950:2200] | 541.08 | **CAUGHT** (1430.3) | 537.15 | -3.93 bps (-0.7%) |

Every count-only 4-gate lie sends the e-process far above the reject threshold of
20 (the honest reference, by contrast, peaks at 1.0-1.8). Adding the e-backtest
shrinks the uniform projected lie on every asset. The shrink is modest under the
uniform attacker because conformal-coverage already caps the uniform lie tightly
and uniform scaling cannot concentrate magnitude on a few days — but it is real,
strictly positive, and reproducible.

### Non-uniform (MILP) attack — the e-backtest breaks the count-reduction entirely

| asset | MILP 4-gate lie (bps) | caught by e-backtest? (e-process max) | MILP + e-backtest feasible? |
|---|---|---|---|
| AAPL | 312.01 (MAXIMUM) | **CAUGHT** (5.1e+10) | **No** — voided |
| SPY | 158.48 (exhibit) | **CAUGHT** (9.2e+10) | **No** — voided |
| BTC-USD | 674.19 (MAXIMUM) | **CAUGHT** (3.9e+10) | **No** — voided |

This is the sharpest part of the defense. The Cycle-3 MILP earns the word
"maximum" precisely because every gate reduces to a **linear constraint on the
exception count**. The e-backtest does **not** — it is magnitude-based — so the
MILP cannot represent it and the count-optimal zero-VaR-on-breach lie drives the
e-process to ~1e10. The non-uniform "maximum" is therefore **infeasible under the
defended panel**: the attacker is forced back to the uniform projected solver,
whose lie the e-backtest already shrinks. The very structure that made the MILP
attack provably optimal (count-only gates) is the structure the e-backtest
defeats.

**Plain answer:** adding the anytime-valid e-process ES backtest shrinks the
exhibited lie on all three assets and collapses the non-uniform attacker's
provable maximum. It is the first gate in the panel that the count-gamer cannot
satisfy, because it is the first gate that reads breach magnitude rather than
breach count.

## Figures (`figures/*.png`, all from the real pipeline)

- `figures/fig_rolling_mul_distribution.png` — rolling MUL distribution per asset
  (violin + box) over the definable windows. Medians 242 / 129 / 537 bps
  (AAPL / SPY / BTC-USD).
- `figures/fig_tolerance_map.png` — **the Cycle-7 honest finding**: full-panel MUL
  vs the conformal tolerance per asset, with the conformal-binding (artifact) zone
  shaded amber, the regulatory-binding zone green, the hand-set default tol=0.005
  marked. Which gate is load-bearing is a tuning choice.
- `figures/fig_gate_contribution_ranking.png` — the raw leave-one-out mechanism
  (kept to show how the artifact arises). Its "binder / dead-weight" labels are a
  TUNING ARTIFACT of the author-added conformal tolerance; read it through the
  tolerance map, not as "the count tests are dead weight."
- `figures/fig_attack_vs_defense.png` — the strawman (zero-VaR-on-breach) defense
  baseline: 4-gate vs 4+e-backtest lie budget per asset (left) and the e-process
  trajectory on the gamed series (right). The honest e-aware residual is the
  Cycle-7 section, not this figure.

## Validation contract (re-checked in Cycle-4 tests)

- The ES e-statistic satisfies `E[e] <= 1` under a calibrated Gaussian null (MC).
- The e-process starts at 1, stays non-negative, and rarely crosses `1/(1-level)`
  on honest replications (anytime-valid size control, asserted over 200 seeds).
- The gate PASSES a calibrated series and REJECTS an understated one.
- **Magnitude sensitivity:** a lie with an in-band exception COUNT but huge breach
  magnitude passes Kupiec and Basel yet is CAUGHT by the e-backtest.
- **Attack vs defense:** on every asset the 4-gate lie is caught by the e-backtest
  and the defended lie never exceeds the undefended lie (`PANEL_WITH_EBT` is a
  superset of `PANEL`), while still genuinely under-reserving (`stated <= true`).

(All numbers deterministic given cached returns through 2024; two `--cycle 4` runs
are byte-for-byte identical, and the four figure PNGs are reproducible.)

---

# Cycle 7 — foundation repair

Cycle 7 does not add a new attack; it makes the existing work defensible to an
expert by removing four soft spots a cold reviewer can puncture. Run it with:

    uv run python -m gaming_oracle.cli --cycle 7      # all three assets, rolling GARCH

All Cycle-7 numbers use the **causal rolling out-of-sample GARCH** reference
(`--method rolling_garch`, the Cycle-7 default), 99% VaR, on the representative
definable window per asset.

## 1. Headlines lead with the standard regulatory panel

`conformal_coverage` is relabeled **everywhere** (code, README, paper) as a
**non-regulatory, author-added** gate. Every headline now leads with the
**STANDARD REGULATORY PANEL** = Kupiec POF + Christoffersen + Basel traffic-light
(`gaming_oracle.gates.PANEL_REGULATORY`). The conformal gate is reported as a
separate add-on, never co-equal.

| asset | window | **regulatory-panel MUL (bps)** | +conformal (author-added) | conformal moves headline by |
|---|---|---|---|---|
| AAPL | [2225:2475] | **245.46** | 243.19 | +2.27 bps (0.9%) |
| SPY | [250:500] | **132.33** | 125.31 | +7.01 bps (5.3%) |
| BTC-USD | [1475:1725] | **608.96** | 581.23 | +27.73 bps (4.6%) |

The author-added gate moves the headline by at most ~5%. The regulatory-panel lie
is the headline; conformal is a footnote.

## 2. The tolerance map replaces the false "dead weight" finding

The old claim "the count tests are dead weight" was a tuning artifact: the
conformal gate's hand-set `tol=0.005` is **tighter than every regulatory band**, so
it binds by construction. Sweeping the conformal tolerance shows *which test is
load-bearing is a tuning choice*:

| asset | conformal binds for tol <= | crossover tol | binding gate above crossover |
|---|---|---|---|
| AAPL | 0.0050 | **0.0075** | basel_traffic_light (regulatory) |
| SPY | 0.0050 | **0.0075** | basel_traffic_light (regulatory) |
| BTC-USD | 0.0050 | **0.0075** | basel_traffic_light (regulatory) |

On all three assets the default `tol=0.005` sits in the artifact zone; at any
tolerance >= ~0.0075 (still far tighter than Basel's 4-exception band in coverage
terms) a **regulatory** gate (Basel) is the binding constraint. The portable
finding is the map, not "interval coverage is load-bearing." Module:
[`gaming_oracle/tolerance_map.py`](gaming_oracle/tolerance_map.py).

## 3. Causal rolling out-of-sample GARCH (look-ahead removed)

The old `garch` reference fit GARCH(1,1) parameters by MLE on the **entire** series
and applied them as a causal recursion — the *parameters* used future data, so the
reference had look-ahead. `rolling_garch_vol` fits the parameters only on data
strictly before each point (expanding window, refit every 25 days, EWMA warm-up).
The fractions and bps move modestly, not materially: on the AAPL most-recent
250-day window the regulatory-panel MUL is 249.18 bps causal vs 246.80 bps
look-ahead (the look-ahead reference was slightly *tighter*, so removing it makes
the lie marginally larger). The causal 249.18 bps number is printed directly by
`uv run python -m gaming_oracle.cli --headline`. The structural findings (which
gate binds, the e-aware residual) are unchanged in direction. Module:
`gaming_oracle/data.py` (`rolling_garch_vol`, `reference_var(method="rolling_garch")`).

## 4. The honest e-process defense residual (e-aware, budgeted, independent ES)

The old Cycle-4 "voids the attack" only beat the zero-VaR-on-breach strawman. The
Cycle-7 attacker ([`gaming_oracle/eaware_attacker.py`](gaming_oracle/eaware_attacker.py)):

- respects a **magnitude budget** (`stated_var >= (1 - budget) * true_var`; it never
  collapses VaR to 0, so each breach is shallow and each per-period e-value small),
- is **aware of the e-process** (it simulates the mixture e-process forward and
  pulls back exactly on days that would push it toward the `1/alpha_test = 20`
  threshold),
- states **ES independently of VaR** — it under-reserves VaR (the lie) while
  **over-stating ES** to widen the e-statistic denominator `(r_t - z_t)` and
  neutralize the gate.

| asset | vs e-process ALONE (budget 0.20, independent ES) | e-process max (threshold 20) | max per-period e_t (cap 1e6) | full-panel-green residual |
|---|---|---|---|---|
| AAPL | hides **71.0 bps** | **1.00** | 75.7 | **14.2 bps** (budget 0.04) |
| SPY | hides **37.9 bps** | **2.96** | 219.5 | **3.8 bps** (budget 0.02) |
| BTC-USD | hides **168.4 bps** | **1.00** | 50.9 | **42.1 bps** (budget 0.05) |

Two honest results:

- **Independent ES neutralizes the e-process.** With ES tied to VaR by the Gaussian
  multiple (the shipped panel), the budget-0.20 attacker drives the e-process to
  22.3 / 40.0 / 5.5 (AAPL / SPY / BTC) — it would be caught. Stating ES
  *independently* (over-stating it) drops the e-process to 1.00 / 2.96 / 1.00. The
  "cannot dodge ES" claim was true **only by construction** of the fixed multiple.
- **The defense leaves a real residual.** Even when every gate (regulatory +
  conformal + e-backtest) must stay green, the e-aware attacker still hides
  **14 / 4 / 42 bps**. That residual — not "the attack is voided" — is the honest
  defense result. The max per-period `e_t` (76 / 220 / 51) is **four orders of
  magnitude below** the disclosed `e_t=1e6` collapsed-gap cap, so the residual is
  not a cap artifact.

## 5. The `du_escanciano` -> `oneside_uc_score` rename

The 5th gate is now named `oneside_uc_score` to reflect that it is a **one-sided
unconditional-coverage score test "in the spirit of"** the higher-power literature,
**not** the full Du-Escanciano (2017) ES statistic. An earlier draft once carried a
fabricated "Du-Escanciano 2019, J. Banking & Finance" citation; the rename removes
that risk at the identifier level. `du_escanciano` is kept as a deprecated alias.

## 6. One-command reproduce really is one command

`matplotlib` is now a **core dependency** (`pyproject.toml`), so the single
`uv run python -m gaming_oracle.figures` path produces the figures with no
`--extra`. The figure test no longer skips.

---

## Appendix — generalization proof-of-concept (Cycle 6): the engine audits a second, non-VaR gate

**This is a clearly-labeled teaser, not a grand claim.** The whole repo audits
market-risk VaR/ES backtests. This appendix shows the *same* machinery — a
"stated" quantity a producer can distort away from an honest reference, a panel
of pass/fail gates behind the identical `Gate(stated, data) -> GateResult(passed,
stat, detail)` interface, a projected shrink/inflate search for the largest
distortion that still passes every gate jointly, and the leave-one-out
gate-contribution decomposition — transplanted onto a completely different audit:
an **ML classifier reliability gate** (calibration + selective-coverage).

Module: [`gaming_oracle/appendix_classifier.py`](gaming_oracle/appendix_classifier.py).
Tests: [`tests/test_appendix_classifier.py`](tests/test_appendix_classifier.py) (10 tests, incl. the
same known-answer always-pass-contributes-zero invariant the VaR decomposition asserts, plus the
adding-a-gate-cannot-grow-the-lie monotonicity mirror of the VaR killer-demo).
Run: `uv run python -m gaming_oracle.cli --appendix`.

The mapping (VaR backtest → classifier reliability):

| VaR engine | Reliability appendix |
|---|---|
| under-reserving VaR | over-confident classifier |
| stated VaR `V_t <= V*_t` (hides risk) | stated confidence `p_t >= p*_t` (hides error) |
| exception `1[loss>V]` | high-confidence error `1[conf>=thr AND wrong]` |
| panel: Kupiec / Christoffersen / Basel / conformal | panel: ECE / Brier / selective-accuracy / overconfidence-count |
| MUL (bps of risk understated) | MCL: max concealable confidence inflation (pp / extra silent auto-approvals) |

Same direction discipline as the VaR side: the distortion must be self-serving
(inflate confidence, never deflate it), enforced as `stated_conf >= honest_conf`
everywhere, mirroring `stated_var <= true_var`.

**Real numbers (deterministic, `synthetic_classifier` seed 7, n=4000, calibrated Beta(30,1) confidences):**

The honest calibrated classifier (mean confidence 0.9679, accuracy 0.9688)
passes all four reliability gates honestly. Against that panel the search inflated
stated confidence by up to **2.63 percentage points** while staying green =
**8,750 extra silently-wrong auto-approvals per 1,000,000 served** (the
high-confidence-error rate rises from 1.12% to exactly the 2.00% deployment cap).

Leave-one-out decomposition on the same instance:

```
gate                          solo MCL   MCL w/o  grows lie   share  note
overconfidence_count_gate        2.627     2.948      0.321   54.9%  binds alone
ece_gate                         3.086     2.627      0.000    0.0%  dead weight
brier_gate                       3.211     2.627      0.000    0.0%  dead weight
selective_accuracy_gate          2.948     2.627      0.000    0.0%  dead weight
```

On this instance the leave-one-out ranks one gate (`overconfidence_count_gate`,
the traffic-light analogue) as binding alone — its solo MCL equals the full-panel
MCL — and the other three as slack *given* it. **Read that ranking the same way as
the retracted VaR-side one: it is a TUNING ARTIFACT, not a robust "this check is
load-bearing" claim.** `overconfidence_count_gate` binds because its cap
(`max_rate=0.02`) is the tightest of the four tolerances on this instance, exactly
as the VaR-side conformal gate bound because its hand-set `tol=0.005` was the
tightest band. Loosen that cap or tighten another gate and the binder moves. So
this appendix reproduces the VaR *mechanism* — the machinery finds A binding gate
and prices the concealable distortion — not the (retracted) conclusion that one
audit check is load-bearing and the rest are dead weight. A tolerance map, as on
the VaR side, would be needed before any binder ranking here could be quoted.

**What this shows:** the `Gate` / shrink-search / leave-one-out abstraction is not
VaR-specific; it runs unchanged on a calibration audit, finds the binding gate,
and prices the concealable distortion. **What it does NOT show:** that *every*
pass/fail audit is gameable this way. Two priced instances do not prove
universality. A real generalization claim would need a stated theorem about which
audits admit a monotone distortion frontier — which this repo deliberately does
not make. The reliability metrics themselves are textbook and prior (ECE: Naeini
et al. 2015 / Guo et al. 2017; Brier 1950; selective prediction: Geifman &
El-Yaniv 2017); the only contribution here is that the existing decomposition runs
on them.
