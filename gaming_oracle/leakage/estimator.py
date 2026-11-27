"""Leakage estimators: score, deflation, and paired bootstrap CI.

Given probe pairs and the model's response on each arm, produce:

1. ``leakage_score`` -- a ``[0, 1]`` detectability summary of how much the
   model's DECISION moves when only the temporal channel is falsified. It is a
   normalized mean-absolute paired delta, GATED by a permutation significance
   test so that noise-only movement scores near zero.

2. ``alpha_deflation`` -- the fraction (and, given a reported gross alpha, the
   amount) of factual backtest alpha that DISAPPEARS when the model is forced to
   decide on the counterfactual (leak-removed) arm, with a paired bootstrap CI.

MATH IS PRIOR. The paired-difference effect size is Cohen's d_z; the confidence
interval is the paired (matched-pairs) bootstrap of Efron & Tibshirani (1993);
the significance gate is a two-sided sign-flip permutation test. All standard.
The contribution is applying them to LLM-backtest leakage, not the statistics.

SCOPE. Every number is relative to (a) the specific probe set and (b) the
specific model under test. There is no universal claim. ``deflation`` is
well-defined only when the factual alpha is non-trivial; the degenerate case is
flagged, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .probes import ProbePair, ProbeResponse

# A response signal lives in [-1, 1], so the maximum possible paired delta
# magnitude is 2.0. Used to normalize the score into [0, 1].
_MAX_DELTA = 2.0


@dataclass
class LeakageScore:
    """Result of :func:`leakage_score`.

    Two nulls are reported because "the decision moved" and "the movement helped
    the backtest" are different questions:

    - ``pvalue`` (directional): sign-flip permutation on the paired delta,
      testing H0 that the decision has no CONSISTENT-DIRECTION shift when the
      temporal channel is falsified. A fixed-direction leak (always more bullish
      on real dates) trips this.
    - ``pvalue_aligned`` (alpha-relevant): sign-flip permutation on
      ``delta_i * sign(forward_return_i)``, testing H0 that the movement is
      UNCORRELATED with the forward return. This is the leak that actually
      inflates backtest alpha (the decision moves *toward the outcome*), and it
      is the null the score gates on, because a magnitude leak that flips sign
      with the outcome is invisible to the directional test but is exactly what
      deflates alpha.

    The score is gated on ``significant`` = (either null rejected).
    """

    score: float                 # [0, 1], significance-gated normalized |delta|
    mean_abs_delta: float        # raw mean |s_f - s_cf|, in signal units
    d_z: float                   # Cohen's d_z (paired effect size), may be inf
    pvalue: float                # directional sign-flip permutation p-value
    pvalue_aligned: float        # alpha-aligned sign-flip permutation p-value
    significant: bool            # min(pvalue, pvalue_aligned) < (1 - level)
    n_pairs: int
    detail: dict = field(default_factory=dict)


@dataclass
class DeflationEstimate:
    """Result of :func:`alpha_deflation`.

    ``fraction`` is (factual_alpha - counterfactual_alpha) / factual_alpha, i.e.
    the share of factual alpha attributable to the leaked channel. It can be
    negative (the leak-removal INCREASED measured alpha -- i.e. no deflation,
    within noise) and is only meaningful when ``factual_alpha`` is materially
    non-zero; ``degenerate`` flags the near-zero-denominator case.
    """

    fraction: float                   # deflation fraction (nan if degenerate)
    ci_low: float                     # bootstrap CI on the fraction
    ci_high: float
    factual_alpha: float              # mean per-probe pnl, factual arm
    counterfactual_alpha: float       # mean per-probe pnl, counterfactual arm
    leaked_alpha: float               # factual - counterfactual (signal units)
    reported_gross_alpha: float | None
    deflated_amount: float | None     # reported_gross_alpha * fraction, if given
    degenerate: bool
    n_pairs: int
    detail: dict = field(default_factory=dict)


def _deltas(pairs: list[ProbePair], responses: dict[str, ProbeResponse]) -> np.ndarray:
    """Paired signal deltas s_factual - s_counterfactual, one per pair."""
    d = []
    for pr in pairs:
        rf = responses[pr.factual.probe_id]
        rc = responses[pr.counterfactual.probe_id]
        d.append(rf.signal - rc.signal)
    return np.asarray(d, dtype=float)


def _permutation_pvalue(delta: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided sign-flip permutation p-value for H0: mean(x) == 0.

    Under the null the paired difference is symmetric about 0, so flipping the
    sign of any subset of pairs is exchangeable. We compare |mean(x)| to the
    null distribution of |mean(sign-flipped x)|.
    """
    n = delta.size
    if n == 0:
        return 1.0
    observed = abs(float(delta.mean()))
    if observed == 0.0:
        return 1.0
    # exact enumeration when small enough, else Monte Carlo
    ge = 1  # +1 for the observed configuration itself (conservative)
    total = 1
    if n <= 18:
        for mask in range(1 << n):
            signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
            m = abs(float((delta * signs).mean()))
            if m >= observed - 1e-15:
                ge += 1
            total += 1
    else:
        for _ in range(n_perm):
            signs = rng.choice(np.array([-1.0, 1.0]), size=n)
            m = abs(float((delta * signs).mean()))
            if m >= observed - 1e-15:
                ge += 1
            total += 1
    return ge / total


def leakage_score(
    pairs: list[ProbePair],
    responses: dict[str, ProbeResponse],
    level: float = 0.95,
    n_perm: int = 20000,
    seed: int = 0,
) -> LeakageScore:
    """Detectability of the temporal channel: how much the decision moves.

    Returns a ``[0, 1]`` score = normalized mean-absolute delta, ZEROED unless a
    permutation test rejects insensitivity at ``level``. Two nulls are tested
    (see :class:`LeakageScore`): a DIRECTIONAL sign-flip on the delta, and an
    ALPHA-ALIGNED sign-flip on ``delta * sign(forward_return)``. The score gates
    on either being significant, so a magnitude leak that flips sign with the
    outcome (invisible to the directional test) is still caught.
    """
    if not pairs:
        raise ValueError("no probe pairs")
    rng = np.random.default_rng(seed)
    delta = _deltas(pairs, responses)
    r_sign = np.sign(np.array([pr.forward_return for pr in pairs], dtype=float))
    aligned = delta * r_sign  # positive when the move helps the factual pnl
    n = delta.size

    mean_abs = float(np.mean(np.abs(delta)))
    sd = float(np.std(delta, ddof=1)) if n > 1 else 0.0
    mean = float(delta.mean())
    d_z = float(mean / sd) if sd > 0 else (np.inf if mean != 0 else 0.0)

    # two independent RNG streams so the two permutation tests do not share draws
    pval = _permutation_pvalue(delta, n_perm=n_perm,
                               rng=np.random.default_rng(seed))
    pval_aligned = _permutation_pvalue(aligned, n_perm=n_perm,
                                       rng=np.random.default_rng(seed + 1))
    significant = min(pval, pval_aligned) < (1.0 - level)

    raw_score = min(mean_abs / _MAX_DELTA, 1.0)
    score = raw_score if significant else 0.0

    return LeakageScore(
        score=float(score),
        mean_abs_delta=mean_abs,
        d_z=d_z,
        pvalue=float(pval),
        pvalue_aligned=float(pval_aligned),
        significant=bool(significant),
        n_pairs=n,
        detail={
            "mean_delta": mean,
            "mean_aligned": float(aligned.mean()),
            "sd_delta": sd,
            "level": level,
            "normalizer": _MAX_DELTA,
            "note": "score is significance-gated normalized mean|delta|; "
                    "d_z is Cohen's d_z (prior); pvalue is directional "
                    "sign-flip permutation; pvalue_aligned is the "
                    "delta*sign(return) sign-flip permutation (the "
                    "alpha-relevant channel).",
        },
    )


def alpha_deflation(
    pairs: list[ProbePair],
    responses: dict[str, ProbeResponse],
    reported_gross_alpha: float | None = None,
    level: float = 0.95,
    n_boot: int = 5000,
    seed: int = 0,
    degenerate_tol: float = 1e-9,
) -> DeflationEstimate:
    """Estimate the share of factual alpha attributable to the leaked channel.

    factual pnl_i        = signal_factual_i        * forward_return_i
    counterfactual pnl_i = signal_counterfactual_i * forward_return_i
    deflation fraction   = (mean(pnl_f) - mean(pnl_cf)) / mean(pnl_f)

    The CI is a paired bootstrap over probes (resample pairs with replacement,
    recompute the fraction). When ``|mean(pnl_f)|`` is below ``degenerate_tol``
    the fraction is undefined (near-zero denominator); we set it to nan and flag
    ``degenerate`` rather than emit a meaningless ratio.

    If ``reported_gross_alpha`` is supplied, ``deflated_amount`` scales it by the
    fraction, giving the leaked amount in the report's own alpha units.
    """
    if not pairs:
        raise ValueError("no probe pairs")
    rng = np.random.default_rng(seed)

    r = np.array([pr.forward_return for pr in pairs], dtype=float)
    s_f = np.array([responses[pr.factual.probe_id].signal for pr in pairs], dtype=float)
    s_cf = np.array([responses[pr.counterfactual.probe_id].signal for pr in pairs], dtype=float)

    pnl_f = s_f * r
    pnl_cf = s_cf * r
    fa = float(pnl_f.mean())
    ca = float(pnl_cf.mean())
    leaked = fa - ca

    degenerate = abs(fa) < degenerate_tol
    fraction = float("nan") if degenerate else leaked / fa

    # paired bootstrap
    n = len(pairs)
    boot = np.empty(n_boot, dtype=float)
    valid = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fb = float(pnl_f[idx].mean())
        cb = float(pnl_cf[idx].mean())
        if abs(fb) < degenerate_tol:
            boot[b] = np.nan
        else:
            boot[b] = (fb - cb) / fb
            valid += 1
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    finite = boot[np.isfinite(boot)]
    if finite.size:
        ci_low = float(np.quantile(finite, lo_q))
        ci_high = float(np.quantile(finite, hi_q))
    else:
        ci_low = ci_high = float("nan")

    deflated_amount = None
    if reported_gross_alpha is not None and not degenerate:
        deflated_amount = float(reported_gross_alpha * fraction)

    return DeflationEstimate(
        fraction=float(fraction),
        ci_low=ci_low,
        ci_high=ci_high,
        factual_alpha=fa,
        counterfactual_alpha=ca,
        leaked_alpha=float(leaked),
        reported_gross_alpha=reported_gross_alpha,
        deflated_amount=deflated_amount,
        degenerate=bool(degenerate),
        n_pairs=n,
        detail={
            "level": level,
            "n_boot": n_boot,
            "boot_valid_frac": valid / n_boot if n_boot else 0.0,
            "note": "fraction = leaked/factual alpha; paired bootstrap CI; "
                    "degenerate when factual alpha ~ 0. Prior stats; "
                    "scope-relative to this probe set and model.",
        },
    )
