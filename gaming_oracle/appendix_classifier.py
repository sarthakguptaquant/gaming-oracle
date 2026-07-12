"""APPENDIX / proof-of-concept: the same gameability decomposition on a SECOND,
non-VaR pass/fail gate (Cycle 6, generalization teaser).

NOT A GRAND CLAIM. The body of this repo audits market-risk VaR/ES backtests.
This appendix shows the SAME machinery -- a "stated" quantity a producer can
distort away from an honest reference, a panel of pass/fail gates behind the
common ``Gate(stated, data) -> GateResult(passed, stat, detail)`` interface, a
projected search for the largest distortion that still passes every gate
jointly, and a leave-one-out gate-contribution decomposition -- transplants onto
a completely different audit: a machine-learning classifier RELIABILITY gate.

This demonstrates the engine "audits any audit" in the weak, honest sense that
ONE additional concrete instance runs end-to-end with the same abstractions. One
extra instance is a TEASER, not a general theorem. Two priced gates do not prove
the construction is universal; they show the interface is not VaR-specific. A
real generalization claim would need a stated theorem about which pass/fail
audits admit a monotone distortion frontier, which we explicitly do NOT make.

The mapping (VaR backtest  ->  classifier reliability):

    under-reserving VaR  ->  OVER-confident classifier
    stated VaR V_t (<= honest V*_t, hides risk)
                         ->  stated confidence p_t (>= honest p*_t, hides error)
    exception 1[loss>V]  ->  high-confidence error 1[p>=thr AND wrong]
    panel of VaR gates   ->  panel of reliability gates (ECE, Brier,
                             selective-accuracy, overconfidence-count)
    MUL (bps of risk)    ->  MCL: Maximum Concealable Confidence inflation
                             (confidence points / extra silent auto-approvals)

ACRONYM HONESTY (mirrors the MUL convention on the VaR side): the underlying
solver is a bisection + greedy projected search, NOT a MILP, so MCL is a
SEARCH-EXHIBITED bound -- "the largest inflation THIS search found that still
passes the panel" -- not a proven global maximum. The headline and pricing say
"inflated by up to" and tag every result "teaser, not a general theorem,
panel-relative". The word "Maximum" in the acronym is the analogue of "Maximum"
in MUL: aspirational naming for the search target, never a claim of optimality.

Same direction discipline as the VaR side: the distortion must be genuinely
self-serving (inflate confidence, never deflate it), so we enforce
``stated_conf >= honest_conf`` everywhere, mirroring ``stated_var <= true_var``.

DECOMPOSITION CAVEAT (mirrors the Cycle-7 VaR correction). The leave-one-out
ranking below finds ``overconfidence_count_gate`` "binds alone" and calls the
other three "dead weight". Read that the SAME way as the VaR-side decomposition:
it is TOLERANCE-DEPENDENT, not a robust "this check is the load-bearing one"
finding. ``overconfidence_count_gate`` binds because its cap (max_rate=0.02) is
the tightest of the four tolerances on this instance, exactly as the VaR-side
conformal gate bound because its hand-set tol=0.005 was the tightest band. Loosen
that cap or tighten another gate and the binder changes. What this appendix
genuinely shows is the MECHANISM -- the Gate / projected-search / leave-one-out
machinery transplants verbatim onto a non-VaR audit, finds A binding gate, and
prices the concealable distortion -- NOT that "one reliability check is
load-bearing and the rest are useless". A tolerance map (as on the VaR side)
would be needed before any binder/dead-weight ranking here could be quoted.

The reliability metrics are textbook and PRIOR:
- ECE: expected calibration error, binned reliability gap (Naeini et al. 2015;
  Guo et al. 2017, "On Calibration of Modern Neural Networks", ICML).
- Brier score (Brier 1950).
- Selective prediction / accuracy-at-coverage (Geifman & El-Yaniv 2017).
The contribution here is ONLY that the existing decomposition runs on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gates import GateResult


# ---------------------------------------------------------------------------
# helpers shared by the reliability gates
# ---------------------------------------------------------------------------

def high_conf_errors(stated_conf, correct, threshold: float = 0.95) -> np.ndarray:
    """Indicator h_t = 1[ stated_conf_t >= threshold AND prediction wrong ].

    This is the classifier analogue of a VaR exception: the model reported high
    confidence yet was wrong. ``correct`` is 1 when the top-1 prediction matched
    the label, 0 otherwise. ``stated_conf`` is the reported top-1 probability.
    """
    stated_conf = np.asarray(stated_conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    wrong = 1.0 - correct
    return ((stated_conf >= threshold) & (wrong > 0)).astype(int)


def _ece(stated_conf, correct, n_bins: int = 10) -> float:
    """Expected calibration error (equal-width bins of [0,1])."""
    stated_conf = np.asarray(stated_conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = stated_conf.size
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (stated_conf >= lo) & (stated_conf <= hi)
        else:
            mask = (stated_conf >= lo) & (stated_conf < hi)
        m = int(mask.sum())
        if m == 0:
            continue
        conf_bin = float(stated_conf[mask].mean())
        acc_bin = float(correct[mask].mean())
        ece += (m / n) * abs(conf_bin - acc_bin)
    return float(ece)


# ---------------------------------------------------------------------------
# the reliability gate panel  (Gate(stated_conf, correct) -> GateResult)
# ---------------------------------------------------------------------------
# Each gate has the SAME shape as the VaR gates: a pure function returning a
# GateResult with passed / stat / detail. We keep (alpha, level) in the signature
# for interface-compatibility even though these gates use tolerances instead.

def ece_gate(stated_conf, correct, alpha: float = 0.01, level: float = 0.95,
             tol: float = 0.03, n_bins: int = 10) -> GateResult:
    """Expected calibration error within tolerance. Pass when ECE <= tol.

    Inflating confidence pushes bin-confidence above bin-accuracy and raises ECE,
    so this gate eventually flips. tol=0.03 is a common deployment bar.
    """
    val = _ece(stated_conf, correct, n_bins=n_bins)
    passed = val <= tol
    return GateResult(passed=bool(passed), stat=float(val),
                      detail={"ece": float(val), "tol": tol, "n_bins": n_bins,
                              "test": "ece_gate"})


def brier_gate(stated_conf, correct, alpha: float = 0.01, level: float = 0.95,
               tol: float = 0.15) -> GateResult:
    """Brier score of the (stated_conf, correct) pairs within tolerance.

    Brier = mean( (stated_conf - correct)^2 ). Over-confident-but-wrong points
    contribute (conf-0)^2 = conf^2, so inflating confidence on errors raises it.
    Pass when Brier <= tol.
    """
    sc = np.asarray(stated_conf, dtype=float)
    c = np.asarray(correct, dtype=float)
    val = float(np.mean((sc - c) ** 2)) if sc.size else 0.0
    passed = val <= tol
    return GateResult(passed=bool(passed), stat=float(val),
                      detail={"brier": float(val), "tol": tol,
                              "test": "brier_gate"})


def selective_accuracy_gate(stated_conf, correct, alpha: float = 0.01,
                            level: float = 0.95, threshold: float = 0.95,
                            min_acc: float = 0.97) -> GateResult:
    """Accuracy on the auto-accepted (stated_conf >= threshold) subset.

    A selective classifier auto-approves predictions above ``threshold``; the
    deployment contract is that the retained set is >= ``min_acc`` accurate.
    Inflating confidence drags more wrong predictions above the threshold and
    lowers retained accuracy, so this gate flips. Pass when retained accuracy
    >= min_acc. If nothing clears the threshold the gate vacuously passes
    (no auto-approvals, no harm), matching the deployment semantics.
    """
    sc = np.asarray(stated_conf, dtype=float)
    c = np.asarray(correct, dtype=float)
    mask = sc >= threshold
    m = int(mask.sum())
    acc = float(c[mask].mean()) if m else 1.0
    passed = acc >= min_acc
    return GateResult(passed=bool(passed), stat=float(acc),
                      detail={"retained_n": m, "retained_acc": float(acc),
                              "threshold": threshold, "min_acc": min_acc,
                              "test": "selective_accuracy_gate"})


def overconfidence_count_gate(stated_conf, correct, alpha: float = 0.01,
                              level: float = 0.95, threshold: float = 0.95,
                              max_rate: float = 0.02) -> GateResult:
    """Traffic-light-style cap on the high-confidence error RATE.

    This is the direct analogue of the Basel traffic-light: count the
    high-confidence-wrong predictions ``h_t = 1[conf>=thr AND wrong]`` and pass
    only when their rate is <= ``max_rate`` of all predictions. It reads the same
    hit sequence the selective-accuracy gate does but caps the absolute rate
    rather than the retained-set accuracy, so the two are correlated but not
    identical (one is conditional on retention, one is unconditional).
    """
    h = high_conf_errors(stated_conf, correct, threshold=threshold)
    n = h.size
    rate = float(h.mean()) if n else 0.0
    passed = rate <= max_rate
    return GateResult(passed=bool(passed), stat=float(rate),
                      detail={"hc_error_count": int(h.sum()), "n": n,
                              "hc_error_rate": float(rate), "max_rate": max_rate,
                              "threshold": threshold,
                              "test": "overconfidence_count_gate"})


#: The reliability panel, mirroring gates.PANEL on the VaR side.
RELIABILITY_PANEL = [ece_gate, brier_gate, selective_accuracy_gate,
                     overconfidence_count_gate]


def run_reliability_panel(stated_conf, correct, panel=None, alpha: float = 0.01,
                          level: float = 0.95) -> dict:
    """Run every reliability gate; report joint pass + per-gate results."""
    if panel is None:
        panel = RELIABILITY_PANEL
    results = {}
    for gate in panel:
        res = gate(stated_conf, correct, alpha=alpha, level=level)
        results[gate.__name__] = res
    passed_all = all(r.passed for r in results.values())
    return {"passed_all": bool(passed_all), "gates": results}


# ---------------------------------------------------------------------------
# the solver-analogue: inflate confidence toward 1, find the largest inflation
# that still passes the whole reliability panel.  Mirrors solver.solve.
# ---------------------------------------------------------------------------

@dataclass
class ConfSolveResult:
    stated_conf: np.ndarray
    honest_conf: np.ndarray
    correct: np.ndarray
    mean_inflation: float        # Delta = mean(stated - honest), confidence pts
    inflate: float               # scalar interpolation factor s* in [0,1]
    passed_all: bool
    gate_results: dict


def _conf_passes(stated_conf, correct, panel, alpha, level) -> bool:
    return run_reliability_panel(stated_conf, correct, panel=panel,
                                 alpha=alpha, level=level)["passed_all"]


def inflate_confidence(honest_conf, correct, panel=None, alpha: float = 0.01,
                       level: float = 0.95, bisection_iters: int = 60,
                       refine: bool = True, refine_passes: int = 2):
    """Find a stated_conf >= honest_conf that maximizes mean inflation and
    passes every reliability gate jointly.

    Structurally identical to solver.solve: scalar bisection on a uniform
    interpolation toward 1.0 (``stated = honest + s*(1 - honest)``), then a
    greedy per-point projected refinement that only ever raises a point and
    re-validates the full panel, so the output is feasible by construction.
    """
    honest_conf = np.asarray(honest_conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if panel is None:
        panel = RELIABILITY_PANEL

    # -- stage 1: scalar bisection on uniform inflation toward 1.0 ----------
    lo, hi = 0.0, 1.0
    if not _conf_passes(honest_conf, correct, panel, alpha, level):
        s_star = 0.0
    else:
        for _ in range(bisection_iters):
            mid = 0.5 * (lo + hi)
            cand = honest_conf + mid * (1.0 - honest_conf)
            if _conf_passes(cand, correct, panel, alpha, level):
                lo = mid
            else:
                hi = mid
        s_star = lo

    stated = honest_conf + s_star * (1.0 - honest_conf)

    # -- stage 2: greedy per-point projected refinement --------------------
    if refine and s_star >= 0.0:
        for _ in range(refine_passes):
            room = 1.0 - stated            # remaining inflation headroom per point
            order = np.argsort(-room)      # biggest headroom first
            changed = False
            for t in order:
                if stated[t] >= 1.0 - 1e-12:
                    continue
                base = stated[t]
                # geometric ladder toward 1.0 (mirror of the VaR cut ladder)
                for frac in (0.5, 0.25, 0.1, 1.0):
                    cand_val = base + frac * (1.0 - base)
                    if cand_val <= base:
                        continue
                    trial = stated.copy()
                    trial[t] = cand_val
                    if _conf_passes(trial, correct, panel, alpha, level):
                        stated = trial
                        changed = True
                        break
            if not changed:
                break

    # safety: stated must never DROP below honest (genuine self-serving inflation)
    stated = np.maximum(stated, honest_conf)
    stated = np.minimum(stated, 1.0)
    delta = float(np.mean(stated - honest_conf))
    out = run_reliability_panel(stated, correct, panel=panel, alpha=alpha, level=level)
    return ConfSolveResult(
        stated_conf=stated, honest_conf=honest_conf, correct=correct,
        mean_inflation=delta, inflate=float(s_star),
        passed_all=bool(out["passed_all"]), gate_results=out["gates"],
    )


# ---------------------------------------------------------------------------
# pricing the Maximum Concealable Confidence inflation (MCL), mirror of report.price
# ---------------------------------------------------------------------------

@dataclass
class ConfPricing:
    mcl_points: float                 # mean inflation in confidence points (0-1)
    mcl_pct_points: float             # same, in percentage points
    extra_silent_errors: float        # extra auto-approved-but-wrong per N served
    served: float
    hc_error_rate_honest: float
    hc_error_rate_stated: float
    mean_inflation: float
    panel_names: list


def price_conf(stated_conf, honest_conf, correct, served: float = 1_000_000.0,
               threshold: float = 0.95, panel=None, panel_names=None) -> ConfPricing:
    """Price the confidence lie three ways (mirror of report.price).

    (a) mean inflation in confidence points and percentage points;
    (b) extra silently-wrong AUTO-APPROVED predictions per ``served`` decisions:
        the rise in the high-confidence-error rate times the served volume;
    (c) the per-gate share comes from the leave-one-out decomposition below.
    """
    stated_conf = np.asarray(stated_conf, dtype=float)
    honest_conf = np.asarray(honest_conf, dtype=float)
    correct = np.asarray(correct, dtype=float)

    delta = float(np.mean(stated_conf - honest_conf))
    h_honest = high_conf_errors(honest_conf, correct, threshold=threshold)
    h_stated = high_conf_errors(stated_conf, correct, threshold=threshold)
    rate_h = float(h_honest.mean()) if h_honest.size else 0.0
    rate_s = float(h_stated.mean()) if h_stated.size else 0.0
    extra = (rate_s - rate_h) * served

    if panel_names is None:
        panel = panel or RELIABILITY_PANEL
        panel_names = [g.__name__ for g in panel]

    return ConfPricing(
        mcl_points=delta, mcl_pct_points=delta * 100.0,
        extra_silent_errors=extra, served=served,
        hc_error_rate_honest=rate_h, hc_error_rate_stated=rate_s,
        mean_inflation=delta, panel_names=list(panel_names),
    )


def conf_headline(p: ConfPricing) -> str:
    return (
        f"APPENDIX (classifier reliability, NOT the main VaR claim): against the "
        f"panel {p.panel_names}, this search inflated stated confidence by up to "
        f"{p.mcl_pct_points:.2f} percentage points = "
        f"{p.extra_silent_errors:,.0f} extra silently-wrong auto-approvals per "
        f"{p.served:,.0f} served, all reliability gates green. One extra instance "
        f"is a teaser, not a general theorem. (Panel-relative.)"
    )


# ---------------------------------------------------------------------------
# leave-one-out gate-contribution decomposition, mirror of decomposition.leave_one_out
# ---------------------------------------------------------------------------

@dataclass
class ConfGateContribution:
    gate_name: str
    mcl_without_pp: float         # MCL (pct points) with this gate removed
    contribution_pp: float        # mcl_without - mcl_full (>= 0)
    contribution_frac: float      # share of the (panel-vs-no-panel) span
    mcl_solo_pp: float = float("nan")
    binds_alone: bool = False


@dataclass
class ConfDecomposition:
    panel_names: list
    mcl_full_pp: float
    mcl_nopanel_pp: float          # unconstrained ceiling: inflate every conf to 1
    contributions: list = field(default_factory=list)

    def ranking_table(self) -> str:
        lines = [
            "Reliability-gate gameability decomposition (APPENDIX, classifier)",
            f"  full-panel MCL = {self.mcl_full_pp:.3f} pp; "
            f"unconstrained ceiling (no gates) = {self.mcl_nopanel_pp:.3f} pp",
            f"  {'gate':28s} {'solo MCL':>9s} {'MCL w/o':>9s} "
            f"{'grows lie':>10s} {'share':>7s}  note",
        ]
        for c in self.contributions:
            note = "binds alone" if c.binds_alone else (
                "marginal backstop" if c.contribution_pp > 1e-6 else "dead weight")
            lines.append(
                f"  {c.gate_name:28s} {c.mcl_solo_pp:9.3f} "
                f"{c.mcl_without_pp:9.3f} {c.contribution_pp:10.3f} "
                f"{c.contribution_frac:7.1%}  {note}"
            )
        return "\n".join(lines)


def _mcl_pp(honest_conf, correct, panel, alpha, level, threshold=0.95) -> float:
    """Solve + price; return MCL in percentage points. Empty panel = ceiling
    (inflate every confidence to 1.0)."""
    if not panel:
        delta = float(np.mean(1.0 - honest_conf))
        return delta * 100.0
    res = inflate_confidence(honest_conf, correct, panel=panel, alpha=alpha,
                             level=level)
    p = price_conf(res.stated_conf, honest_conf, correct, threshold=threshold,
                   panel_names=[g.__name__ for g in panel])
    return float(p.mcl_pct_points)


def leave_one_out_conf(honest_conf, correct, panel=None, alpha: float = 0.01,
                       level: float = 0.95, threshold: float = 0.95) -> ConfDecomposition:
    """Leave-one-out reliability-gate decomposition. Same logic as the VaR
    decomposition: contribution(g) = MCL(panel without g) - MCL(full panel)."""
    honest_conf = np.asarray(honest_conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if panel is None:
        panel = RELIABILITY_PANEL
    panel_names = [g.__name__ for g in panel]

    mcl_full = _mcl_pp(honest_conf, correct, panel, alpha, level, threshold)
    mcl_nopanel = _mcl_pp(honest_conf, correct, [], alpha, level, threshold)
    span = max(mcl_nopanel - mcl_full, 1e-12)

    solo = {g.__name__: _mcl_pp(honest_conf, correct, [g], alpha, level, threshold)
            for g in panel}

    contribs: list[ConfGateContribution] = []
    for g in panel:
        reduced = [x for x in panel if x is not g]
        mcl_wo = _mcl_pp(honest_conf, correct, reduced, alpha, level, threshold)
        contrib = max(mcl_wo - mcl_full, 0.0)
        contribs.append(ConfGateContribution(
            gate_name=g.__name__, mcl_without_pp=float(mcl_wo),
            contribution_pp=float(contrib),
            contribution_frac=float(contrib / span),
            mcl_solo_pp=float(solo[g.__name__]),
            binds_alone=bool(abs(solo[g.__name__] - mcl_full) <= 1e-6),
        ))
    contribs.sort(key=lambda c: c.contribution_pp, reverse=True)
    return ConfDecomposition(panel_names=panel_names, mcl_full_pp=float(mcl_full),
                             mcl_nopanel_pp=float(mcl_nopanel), contributions=contribs)


# ---------------------------------------------------------------------------
# a self-contained synthetic classifier (no external data, deterministic)
# ---------------------------------------------------------------------------

def synthetic_classifier(n: int = 4000, seed: int = 7):
    """A deterministic, well-calibrated strong toy classifier.

    Returns ``(honest_conf, correct)`` where ``honest_conf`` is calibrated: a
    prediction reported at confidence p is correct with probability EXACTLY p (we
    draw ``correct ~ Bernoulli(honest_conf)``). This is the analogue of the
    panel-calibrated honest VaR reference -- it passes the reliability panel, so
    the concealable confidence inflation is well defined.

    The Beta(30, 1) confidence prior makes this a strong model (mean confidence
    ~0.97, accuracy ~0.97). At the 0.95 auto-approval threshold its honest
    high-confidence-error rate is ~1.1%, genuinely under the 2% deployment cap --
    so the honest model passes the overconfidence gate WITHOUT tuning the gate to
    fit it. The gate bar is a sensible deployment cap; the model clears it
    honestly, leaving real headroom for the inflation attack to fill.
    """
    rng = np.random.default_rng(seed)
    honest_conf = rng.beta(30.0, 1.0, size=n)
    honest_conf = np.clip(honest_conf, 1.0 / 2, 0.999)  # 2-class floor at 0.5
    correct = (rng.random(n) < honest_conf).astype(int)
    return honest_conf, correct
