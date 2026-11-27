"""Desk-Backtestability POWER CALCULATOR (the Oracle's PRECURSOR).

The Gaming Oracle asks *"how big a lie passes this gate?"* — it takes the gate as a
constraint and finds the Maximum Undetectable Lie. This module asks the question that
comes logically BEFORE that one:

    *"Can this gate detect a misestimate of this size AT ALL, on this window?"*

i.e. the STATISTICAL POWER of the backtest against a stated risk understatement. If a
gate has ~15% power against a 25% VaR understatement over 250 days, then a desk running
that gate on that window is barely backtestable: most understatements of that magnitude
sail through not because the desk is honest but because the test cannot see them. That is
a precondition question. Only once a gate HAS power does the Oracle's "how big a lie
passes" number become the tight, binding thing; against a near-powerless gate the Oracle's
lie is large simply because nothing constrains it.

--------------------------------------------------------------------------------
WHAT IT COMPUTES
--------------------------------------------------------------------------------
Given
  * a GATE (Kupiec POF, a Basel one-sided binomial gate, an Acerbi-Szekely-style ES
    unconditional test, or the conformal coverage gate),
  * a TRUE tail model (Student-t tail index ``nu`` + vol ``sigma``; ``nu=inf`` = Gaussian),
  * a window length ``N`` (default 250, the Basel window),
  * a stated risk UNDERSTATEMENT ``u`` (stated VaR = (1 - u) * true VaR),

it Monte-Carlos the probability that the gate REJECTS the misspecified (understated)
model. That rejection probability is the gate's POWER against that misspecification.
Deliverables:

  1. a RED / AMBER / GREEN "is this desk even backtestable" verdict at a reference
     understatement, from the power there;
  2. a MINIMUM-DETECTABLE-UNDERSTATEMENT curve (smallest u reaching a target power, as a
     function of N) -- the honest read of "what can this gate actually catch";
  3. a MINIMUM-N-FOR-ADEQUATE-POWER (smallest window that reaches the target power at a
     reference understatement).

--------------------------------------------------------------------------------
MATH IS PRIOR. Every statistical object here is standard; the contribution is the
shipped calculator wired to the SAME gates.py the Oracle uses (one gate math, two
questions), and the honest verdict framing.

  * Kupiec (1995), "Techniques for Verifying the Accuracy of Risk Measurement Models",
    J. Derivatives 3(2):73-84 -- the POF test AND its own low-power caveat. Kupiec (1995)
    reports that lengthening the sample from one to two years raises the chance of
    detecting systematic VaR under-reporting from ~65% to ~90%: the test is weak at N~250.
    This module makes that caveat a first-class, per-gate, per-N number. [PRIOR]
  * Basel Committee (BCBS 1996/2006), the traffic-light backtest. Our ``basel_binomial``
    gate here is the ONE-SIDED binomial-tail rejection form (reject when the exception
    count is in the upper binomial tail), NOT the 3-zone green/amber/red penalty schedule
    in ``gates.basel_traffic_light``; we make the direction and rule explicit. [PRIOR]
  * Acerbi & Szekely (2014), "Back-testing Expected Shortfall", Risk 27:76-81 -- the ES
    backtests that are more powerful than the VaR exception count because they read breach
    MAGNITUDE. Our ``es_z2_unconditional`` gate is an Acerbi-Szekely-Z2-STYLE unconditional
    ES statistic with a Monte-Carlo (simulated-null) critical value; it is NOT their exact
    published test with their exact thresholds, and is labeled so. [PRIOR, style-of]
  * The MC-power / minimum-detectable-effect / required-sample-size machinery is textbook
    hypothesis-test power analysis (Cohen 1988), applied to backtests. [PRIOR]

NO number here is claimed as a maximum, universal, or novel result. Every figure is
GATE-relative, TAIL-MODEL-relative, and WINDOW-relative, and reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import gates as gates_mod


# ---------------------------------------------------------------------------
# 1. True tail model + the understated stated-VaR series
# ---------------------------------------------------------------------------

@dataclass
class TailModel:
    """The TRUE data-generating tail model for the loss distribution.

    Returns r_t are drawn iid from a location-0 distribution scaled to volatility
    ``sigma``. ``nu`` is the Student-t degrees of freedom (tail index); ``nu=inf`` is
    the Gaussian limit. The t-draws are STANDARDIZED to unit variance before scaling by
    ``sigma`` so that ``sigma`` is the actual return volatility regardless of ``nu``
    (Student-t has variance nu/(nu-2), so we divide by sqrt(nu/(nu-2))). This keeps the
    Gaussian and t comparisons on the same volatility footing.
    """

    nu: float = float("inf")   # tail index (degrees of freedom); inf = Gaussian
    sigma: float = 0.01        # return volatility (per period)

    def __post_init__(self):
        if self.nu <= 2 and np.isfinite(self.nu):
            raise ValueError("Student-t needs nu > 2 for a finite variance (and a "
                             "well-defined VaR/ES scaling). Got nu=%r." % self.nu)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

    # --- distribution primitives (standardized to unit variance) ---
    def _std_ppf(self, p: float) -> float:
        """Quantile of the standardized (unit-variance) return distribution."""
        if np.isinf(self.nu):
            return float(stats.norm.ppf(p))
        scale = np.sqrt(self.nu / (self.nu - 2.0))  # t has var nu/(nu-2)
        return float(stats.t.ppf(p, df=self.nu) / scale)

    def true_var(self, alpha: float) -> float:
        """True one-period VaR as a POSITIVE loss magnitude at tail prob ``alpha``.

        VaR_alpha = -sigma * Q(alpha) where Q is the standardized-return quantile (the
        alpha-quantile of returns is negative, so the loss VaR is positive).
        """
        return float(-self.sigma * self._std_ppf(alpha))

    def true_es(self, alpha: float) -> float:
        """True one-period Expected Shortfall (positive loss magnitude) at ``alpha``.

        ES_alpha = -sigma * E[Z | Z <= q_alpha] for the standardized return Z. Closed
        form for Gaussian (phi(z)/alpha) and Student-t (the standard t-ES formula),
        both then scaled to unit variance and by sigma.
        """
        a = alpha
        if np.isinf(self.nu):
            q = stats.norm.ppf(a)
            es_std = stats.norm.pdf(q) / a           # positive ES multiple, unit-var
            return float(self.sigma * es_std)
        nu = self.nu
        scale = np.sqrt(nu / (nu - 2.0))
        t_q = stats.t.ppf(a, df=nu)                  # negative quantile of raw t
        # ES of the RAW t (unit-scale, df=nu): standard closed form
        pdf = stats.t.pdf(t_q, df=nu)
        es_raw = (pdf / a) * ((nu + t_q ** 2) / (nu - 1.0))   # positive
        es_std = es_raw / scale                      # standardize to unit variance
        return float(self.sigma * es_std)

    def sample_returns(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` iid returns from the true model (mean 0, volatility sigma)."""
        if np.isinf(self.nu):
            return rng.normal(0.0, self.sigma, n)
        scale = np.sqrt(self.nu / (self.nu - 2.0))
        raw = rng.standard_t(self.nu, size=n)
        return (raw / scale) * self.sigma


def true_exception_prob(model: TailModel, alpha: float, u: float) -> float:
    """P(loss > stated VaR) under the true model when stated VaR is understated by ``u``.

    stated VaR = (1 - u) * true VaR. A loss exceeds it when the standardized return is
    below (1 - u) * q_alpha, i.e. p1 = F_std((1 - u) * q_alpha) where q_alpha < 0. At
    u=0 this is exactly ``alpha`` (correct coverage); u>0 pushes p1 above alpha.
    """
    q_alpha = model._std_ppf(alpha)                  # negative
    thresh = (1.0 - u) * q_alpha                      # less negative as u grows
    if np.isinf(model.nu):
        return float(stats.norm.cdf(thresh))
    scale = np.sqrt(model.nu / (model.nu - 2.0))
    return float(stats.t.cdf(thresh * scale, df=model.nu))


# ---------------------------------------------------------------------------
# 2. Extra gates the calculator supports (consistent with gates.py conventions)
# ---------------------------------------------------------------------------
# Kupiec POF, Christoffersen, conformal, and the traffic-light already live in
# gates.py and are reused directly (they carry the common Gate signature). The two
# below add (a) the ONE-SIDED Basel binomial-tail rejection rule and (b) an
# Acerbi-Szekely-Z2-style unconditional ES test, both matching the GateResult return
# type so run_panel / the Oracle plumbing sees them uniformly.

def basel_binomial(stated_var, returns, alpha: float = 0.01, level: float = 0.95):
    """One-sided Basel-style BINOMIAL-TAIL backtest gate.

    Reject (fail) when the exception count X is in the UPPER binomial tail under
    H0: X ~ Binomial(N, alpha), i.e. when P(Binomial(N, alpha) >= X) <= 1 - level.
    This is the one-sided (understatement-direction) exact-binomial test that underlies
    the Basel traffic-light zones, stated as a hypothesis test so its POWER is
    well-defined. It is NOT the 3-zone penalty schedule (that is
    ``gates.basel_traffic_light``); direction and rule are explicit here. Math is prior
    (BCBS 1996). Pass = not rejected.
    """
    e = gates_mod.exceptions(stated_var, returns)
    n = e.size
    x = int(e.sum())
    # upper-tail exceedance probability P(X >= x) under Binomial(n, alpha)
    p_upper = float(stats.binom.sf(x - 1, n, alpha)) if n else 1.0
    reject = p_upper <= (1.0 - level)
    passed = not reject
    return gates_mod.GateResult(
        passed=bool(passed),
        stat=float(x),
        detail={
            "n": n,
            "exceptions": x,
            "expected": alpha * n,
            "p_upper_tail": p_upper,
            "size": 1.0 - level,
            "test": "basel_binomial",
        },
    )


#: Cache of the scale-invariant Z2 null critical value, keyed on
#: (n, alpha, level, n_null, es_var_ratio_rounded). See ``_z2_null_crit``.
_Z2_CRIT_CACHE: dict = {}


def _z2_null_crit(n: int, alpha: float, level: float, n_null: int,
                  es_var_ratio: float, seed: int = 12345) -> float:
    """One-sided upper critical value of the Acerbi-Szekely Z2 null, cached.

    Under H0 the per-period breach term  1{loss>VaR} * loss/ES  has a distribution that
    is SCALE-INVARIANT in the per-period volatility: writing ES = ratio * VaR and
    VaR_t = -sigma_t * q_alpha, a breach contributes (-z_t)/(ratio*(-q_alpha)) with z_t
    standard normal, so sigma_t cancels. Hence the null distribution of Z2 depends only on
    (n, alpha, ratio, n_null), NOT on the stated-VaR series. We therefore simulate the null
    ONCE on a unit series (sigma_t = 1) and cache the critical value. This is exact (not an
    approximation): the cached value is the same number the per-series simulation would
    produce, just computed once.
    """
    key = (int(n), round(float(alpha), 12), round(float(level), 12), int(n_null),
           round(float(es_var_ratio), 9))
    if key in _Z2_CRIT_CACHE:
        return _Z2_CRIT_CACHE[key]
    rng = np.random.default_rng(seed)
    q_alpha = stats.norm.ppf(alpha)
    var_unit = -q_alpha                         # VaR of a unit-sigma normal
    es_unit = es_var_ratio * var_unit
    null_stats = np.empty(n_null)
    for b in range(n_null):
        r_b = rng.normal(0.0, 1.0, n)
        loss_b = -r_b
        ind = (loss_b > var_unit).astype(float)
        terms = ind * loss_b / es_unit
        null_stats[b] = terms.sum() / (n * alpha) - 1.0
    crit = float(np.quantile(null_stats, level))
    _Z2_CRIT_CACHE[key] = crit
    return crit


def es_z2_unconditional(stated_var, returns, alpha: float = 0.01, level: float = 0.95,
                        stated_es=None, n_null: int = 2000, seed: int = 12345):
    """Acerbi-Szekely-Z2-STYLE unconditional Expected-Shortfall backtest gate.

    Statistic (Acerbi & Szekely 2014, their Z2, per-period pooled form):

        Z2 = ( 1/(N*alpha) ) * sum_t  [ 1{loss_t > VaR_t} * loss_t / ES_t ]  - 1

    Under H0 (VaR and ES correctly stated), E[Z2] = 0; systematic UNDER-statement of ES
    makes the realized breach losses large relative to the stated ES, pushing Z2 > 0.
    Unlike the exception COUNT, Z2 reads breach MAGNITUDE, which is why ES tests have
    higher power against magnitude-hiding misspecification (Acerbi-Szekely 2014).

    HONEST PROVENANCE. This is the Z2 STATISTIC of Acerbi-Szekely with a MONTE-CARLO
    (simulated-null) one-sided critical value, NOT their exact published test with their
    exact tabulated thresholds. To fit the common Gate(stated_var, returns) signature the
    stated ES defaults to the stated VaR times the consistent Gaussian ES/VaR ratio (the
    same convention as ``gates.e_backtest``); pass ``stated_es`` to override. The null
    distribution of Z2 is obtained by simulating breach patterns consistent with H0 (a
    Gaussian tail calibrated so stated VaR/ES are correct) at the SAME N, which is why the
    gate carries a fixed ``seed`` for reproducibility. Pass = not rejected (Z2 not in the
    upper tail of its simulated null).
    """
    returns = np.asarray(returns, dtype=float)
    stated_var = np.asarray(stated_var, dtype=float)
    loss = -returns
    n = loss.size
    if stated_es is None:
        from .ebacktest import gaussian_es_var_ratio
        stated_es = gaussian_es_var_ratio(alpha) * stated_var
    stated_es = np.asarray(stated_es, dtype=float)

    def _z2(loss_arr, var_arr, es_arr):
        ind = (loss_arr > var_arr).astype(float)
        # guard es>0
        safe_es = np.where(es_arr > 1e-15, es_arr, np.nan)
        terms = ind * loss_arr / safe_es
        terms = np.where(np.isnan(terms), 0.0, terms)
        return float(terms.sum() / (n * alpha) - 1.0) if n else 0.0

    z2 = _z2(loss, stated_var, stated_es)

    # Null critical value. Because the Z2 null is scale-invariant in the per-period
    # volatility (sigma_t cancels in loss/ES; see _z2_null_crit), the H0-consistent
    # simulation is independent of the stated-VaR series and depends only on
    # (n, alpha, level, n_null, es/var ratio). We simulate it ONCE and cache it, which is
    # exact and makes the power sweep tractable. The ratio is inferred from the supplied
    # stated ES vs stated VaR (constant when the default fixed-multiple ES is used).
    ratio_arr = np.where(stated_var > 1e-15, stated_es / stated_var, np.nan)
    es_var_ratio = float(np.nanmedian(ratio_arr)) if n else 1.0
    crit = _z2_null_crit(n, alpha, level, n_null, es_var_ratio, seed=seed)
    reject = z2 > crit
    passed = not reject
    return gates_mod.GateResult(
        passed=bool(passed),
        stat=float(z2),
        detail={
            "z2": float(z2),
            "crit_upper": crit,
            "n": n,
            "n_null": n_null,
            "es_var_ratio": es_var_ratio,
            "test": "es_z2_unconditional",
        },
    )


#: Registry so the CLI / callers can name a gate as a string.
GATES = {
    "kupiec_pof": gates_mod.kupiec_pof,
    "basel_binomial": basel_binomial,
    "basel_traffic_light": gates_mod.basel_traffic_light,
    "christoffersen": gates_mod.christoffersen,
    "conformal_coverage": gates_mod.conformal_coverage,
    "oneside_uc_score": gates_mod.oneside_uc_score,
    "es_z2_unconditional": es_z2_unconditional,
}


# ---------------------------------------------------------------------------
# 3. Monte-Carlo power of a gate against an understatement
# ---------------------------------------------------------------------------

@dataclass
class PowerResult:
    gate: str
    nu: float
    sigma: float
    alpha: float
    level: float
    n: int
    understatement: float           # u
    power: float                    # MC P(reject | misspecified by u)
    n_sims: int
    stderr: float                   # binomial standard error of the power estimate
    true_exception_prob: float      # p1 = P(loss > stated VaR) under the true model
    nominal_exception_prob: float   # alpha
    seed: int


def mc_power(gate, model: TailModel, n: int = 250, u: float = 0.25,
             alpha: float = 0.01, level: float = 0.95, n_sims: int = 2000,
             seed: int = 0) -> PowerResult:
    """Monte-Carlo the probability that ``gate`` rejects a model understated by ``u``.

    Each replication: draw ``n`` returns from the TRUE model; build the STATED VaR series
    as (1 - u) * true_VaR (a constant per-period magnitude because the true model here is
    iid); run the gate; count a rejection when ``passed`` is False. Power is the rejection
    fraction. At u=0 this returns the gate's realized SIZE (should sit near 1 - level for a
    two-sided test, or the nominal one-sided size), which is reported so the reader can see
    the test is honestly calibrated before trusting its power.
    """
    gate_fn = GATES[gate] if isinstance(gate, str) else gate
    gate_name = gate if isinstance(gate, str) else getattr(gate, "__name__", str(gate))
    rng = np.random.default_rng(seed)

    tv = model.true_var(alpha)
    stated = np.full(n, (1.0 - u) * tv)

    rejects = 0
    for _ in range(n_sims):
        returns = model.sample_returns(n, rng)
        res = gate_fn(stated, returns, alpha=alpha, level=level)
        if not res.passed:
            rejects += 1
    power = rejects / n_sims if n_sims else 0.0
    stderr = float(np.sqrt(max(power * (1.0 - power), 0.0) / n_sims)) if n_sims else 0.0

    return PowerResult(
        gate=gate_name, nu=model.nu, sigma=model.sigma, alpha=alpha, level=level,
        n=n, understatement=u, power=float(power), n_sims=n_sims, stderr=stderr,
        true_exception_prob=true_exception_prob(model, alpha, u),
        nominal_exception_prob=alpha, seed=seed,
    )


# ---------------------------------------------------------------------------
# 4. Analytic Kupiec power (the KNOWN-ANSWER anchor)
# ---------------------------------------------------------------------------

def kupiec_power_analytic(model: TailModel, n: int = 250, u: float = 0.25,
                          alpha: float = 0.01, level: float = 0.95) -> float:
    """EXACT power of the Kupiec POF LR test against an understatement ``u``.

    The Kupiec statistic depends on the data ONLY through the exception count
    X ~ Binomial(n, p1), with p1 = ``true_exception_prob(model, alpha, u)``. The test
    rejects when LR_uc(X; n, alpha) > chi2_{1, level}. So the exact power is

        sum_{x=0}^{n}  P(X = x) * 1{ LR_uc(x) > crit },   X ~ Binomial(n, p1).

    This is closed form (a finite sum over the binomial pmf) with no Monte-Carlo error,
    and is the KNOWN-ANSWER benchmark the MC estimator is validated against. It is also
    the exact quantity behind Kupiec's (1995) own low-power caveat.
    """
    p1 = true_exception_prob(model, alpha, u)
    crit = float(stats.chi2.ppf(level, df=1))
    xs = np.arange(0, n + 1)
    pmf = stats.binom.pmf(xs, n, p1)

    # LR_uc(x) reusing the exact gate arithmetic (vectorized here for speed)
    def _safe_log(v):
        return np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), 0.0)

    x = xs.astype(float)
    pi_hat = x / n
    ll_null = x * _safe_log(np.full_like(x, alpha)) + (n - x) * _safe_log(np.full_like(x, 1 - alpha))
    ll_alt = np.where(
        x == 0,
        (n - x) * _safe_log(1 - pi_hat),
        np.where(x == n, x * _safe_log(pi_hat),
                 x * _safe_log(pi_hat) + (n - x) * _safe_log(1 - pi_hat)),
    )
    lr = np.maximum(-2.0 * (ll_null - ll_alt), 0.0)
    reject = lr > crit
    return float(np.sum(pmf[reject]))


# ---------------------------------------------------------------------------
# 5. Verdict, minimum-detectable-understatement, minimum-N
# ---------------------------------------------------------------------------

#: Backtestability traffic-light thresholds on POWER at the reference understatement.
#: These are REPORTING thresholds (a UX convention for the verdict), not statistical
#: constants: green = the gate has adequate power to catch the reference lie; amber =
#: marginal; red = the desk is effectively un-backtestable at that lie/window. Standard
#: power-analysis practice takes 0.80 as "adequate" (Cohen 1988); we use 0.50 as the
#: amber floor (the gate is better than a coin flip at catching the reference lie).
POWER_GREEN = 0.80
POWER_AMBER = 0.50


def verdict_from_power(power: float) -> str:
    """RED / AMBER / GREEN backtestability verdict from the power at the reference lie."""
    if power >= POWER_GREEN:
        return "green"
    if power >= POWER_AMBER:
        return "amber"
    return "red"


@dataclass
class BacktestabilityReport:
    gate: str
    model: TailModel
    alpha: float
    level: float
    n: int
    ref_understatement: float
    power_at_ref: float
    verdict: str
    size_at_zero: float                       # realized size (power at u=0)
    mdu_target_power: float                   # target power for the MDU curve
    mdu_at_ref_n: float | None                # min detectable understatement at ref N
    min_n_for_ref_u: int | None               # min N reaching target power at ref u
    n_grid: list = field(default_factory=list)
    mdu_curve: list = field(default_factory=list)   # list[(N, mdu or None)]
    power_curve_u: list = field(default_factory=list)  # list[(u, power)] at ref N
    n_sims: int = 2000
    seed: int = 0

    def _size_flag(self) -> str:
        """Honest flag when the gate's realized size is far off its target.

        A gate whose size at u=0 is well ABOVE 1-level rejects the HONEST model too often,
        so its 'power' is partly false positives, not detection. The clearest case is
        ``conformal_coverage`` at the default tol=0.005, whose band is tighter than any
        regulatory gate (documented in gates.py / tolerance_map): it fails a calibrated
        desk a large fraction of the time, so its power number is NOT comparable to the
        regulatory gates and must not be read as detection. The Kupiec LR test is also
        mildly OVER-sized at N=250/alpha=0.01 (its own small-sample chi-square caveat).
        """
        target = 1.0 - self.level
        if self.size_at_zero > 2.5 * target:
            return (f"  [WARNING: size >> target -> this gate rejects the HONEST model "
                    f"often; its 'power' is inflated by false positives, NOT comparable "
                    f"to a size-{target:.2f} gate]")
        if self.size_at_zero > 1.4 * target:
            return (f"  [note: over-sized vs target -> some 'power' is size, not "
                    f"detection; e.g. the Kupiec chi-square small-sample caveat]")
        return ""

    def summary(self) -> str:
        m = self.model
        tail = "Gaussian" if np.isinf(m.nu) else f"Student-t(nu={m.nu:g})"
        lines = [
            f"BACKTESTABILITY — gate '{self.gate}', true tail {tail}, "
            f"sigma={m.sigma:g}, alpha={self.alpha:g}, level={self.level:g}",
            f"  window N={self.n}, reference understatement u={self.ref_understatement:.0%}",
            f"  realized SIZE at u=0: {self.size_at_zero:.3f} "
            f"(target ~{1 - self.level:.3f}){self._size_flag()}",
            f"  POWER at reference lie: {self.power_at_ref:.3f}  ->  "
            f"VERDICT: {self.verdict.upper()}",
        ]
        if self.mdu_at_ref_n is not None:
            lines.append(f"  minimum-detectable understatement at N={self.n} "
                         f"(power>={self.mdu_target_power:.0%}): "
                         f"u* = {self.mdu_at_ref_n:.1%}")
        else:
            lines.append(f"  minimum-detectable understatement at N={self.n} "
                         f"(power>={self.mdu_target_power:.0%}): NONE on the swept "
                         f"u-grid (gate cannot reach target power even at the largest u)")
        if self.min_n_for_ref_u is not None:
            lines.append(f"  minimum N for power>={self.mdu_target_power:.0%} at "
                         f"u={self.ref_understatement:.0%}: N* = {self.min_n_for_ref_u}")
        else:
            lines.append(f"  minimum N for power>={self.mdu_target_power:.0%} at "
                         f"u={self.ref_understatement:.0%}: NOT reached on the swept "
                         f"N-grid (up to N={self.n_grid[-1] if self.n_grid else '?'})")
        lines.append("  MDU curve (min detectable u vs window N):")
        for (nn, mdu) in self.mdu_curve:
            cell = f"{mdu:.1%}" if mdu is not None else "  none"
            lines.append(f"    N={nn:5d}  u*={cell}")
        lines.append("  Every number is GATE-, TAIL-MODEL-, and WINDOW-relative "
                     "(not a universal or maximum claim).")
        return "\n".join(lines)


def min_detectable_understatement(
    gate, model: TailModel, n: int = 250, alpha: float = 0.01, level: float = 0.95,
    target_power: float = 0.80, u_grid=None, n_sims: int = 2000, seed: int = 0,
) -> tuple[float | None, list]:
    """Smallest understatement ``u`` whose power >= ``target_power`` at window ``n``.

    Returns (u_star or None, power_curve) where power_curve is a list of (u, power). The
    curve is MONOTONE in expectation but MC-noisy, so we take the first grid u whose power
    crosses the target and report it; None if no swept u reaches the target.
    """
    if u_grid is None:
        u_grid = np.round(np.arange(0.0, 0.6001, 0.05), 4)
    curve = []
    u_star = None
    for u in u_grid:
        pr = mc_power(gate, model, n=n, u=float(u), alpha=alpha, level=level,
                      n_sims=n_sims, seed=seed)
        curve.append((float(u), pr.power))
        if u_star is None and u > 0 and pr.power >= target_power:
            u_star = float(u)
    return u_star, curve


def min_n_for_power(
    gate, model: TailModel, u: float = 0.25, alpha: float = 0.01, level: float = 0.95,
    target_power: float = 0.80, n_grid=None, n_sims: int = 2000, seed: int = 0,
) -> tuple[int | None, list]:
    """Smallest window ``N`` reaching ``target_power`` at understatement ``u``.

    Returns (n_star or None, curve) with curve a list of (N, power).
    """
    if n_grid is None:
        n_grid = [125, 250, 500, 750, 1000, 1500, 2000]
    curve = []
    n_star = None
    for nn in n_grid:
        pr = mc_power(gate, model, n=int(nn), u=u, alpha=alpha, level=level,
                      n_sims=n_sims, seed=seed)
        curve.append((int(nn), pr.power))
        if n_star is None and pr.power >= target_power:
            n_star = int(nn)
    return n_star, curve


def backtestability_report(
    gate, nu: float = float("inf"), sigma: float = 0.01, n: int = 250,
    ref_understatement: float = 0.25, alpha: float = 0.01, level: float = 0.95,
    target_power: float = 0.80, n_sims: int = 2000, seed: int = 0,
    u_grid=None, n_grid=None,
) -> BacktestabilityReport:
    """Full 'is this desk even backtestable' report for one gate + tail model.

    Assembles: the power at the reference understatement (-> RED/AMBER/GREEN verdict),
    the realized size at u=0, the minimum-detectable-understatement at the reference N,
    the minimum-N reaching the target power at the reference u, and an MDU-vs-N curve.
    """
    gate_name = gate if isinstance(gate, str) else getattr(gate, "__name__", str(gate))
    model = TailModel(nu=nu, sigma=sigma)
    if n_grid is None:
        n_grid = [125, 250, 500, 750, 1000, 1500, 2000]

    # power at the reference lie + verdict
    pr_ref = mc_power(gate, model, n=n, u=ref_understatement, alpha=alpha, level=level,
                      n_sims=n_sims, seed=seed)
    verdict = verdict_from_power(pr_ref.power)

    # realized size (power at u=0)
    pr_size = mc_power(gate, model, n=n, u=0.0, alpha=alpha, level=level,
                       n_sims=n_sims, seed=seed)

    # MDU at the reference N + the full power-vs-u curve
    mdu_ref, power_curve_u = min_detectable_understatement(
        gate, model, n=n, alpha=alpha, level=level, target_power=target_power,
        u_grid=u_grid, n_sims=n_sims, seed=seed)

    # min N for the reference u
    n_star, _ = min_n_for_power(
        gate, model, u=ref_understatement, alpha=alpha, level=level,
        target_power=target_power, n_grid=n_grid, n_sims=n_sims, seed=seed)

    # MDU curve vs N (min detectable understatement at each window length)
    mdu_curve = []
    for nn in n_grid:
        mdu_nn, _ = min_detectable_understatement(
            gate, model, n=int(nn), alpha=alpha, level=level, target_power=target_power,
            u_grid=u_grid, n_sims=n_sims, seed=seed)
        mdu_curve.append((int(nn), mdu_nn))

    return BacktestabilityReport(
        gate=gate_name, model=model, alpha=alpha, level=level, n=n,
        ref_understatement=ref_understatement, power_at_ref=pr_ref.power,
        verdict=verdict, size_at_zero=pr_size.power, mdu_target_power=target_power,
        mdu_at_ref_n=mdu_ref, min_n_for_ref_u=n_star, n_grid=list(n_grid),
        mdu_curve=mdu_curve, power_curve_u=power_curve_u, n_sims=n_sims, seed=seed,
    )
