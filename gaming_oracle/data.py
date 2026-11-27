"""Data loading + honest reference VaR for the Gaming Oracle.

``load_data(asset)`` returns real daily returns plus an HONEST reference VaR
series ``true_var`` (a positive loss magnitude). The reference is a defensible,
conventional risk model, not a strawman:

- ``method="garch"`` (default): a GARCH(1,1)-style conditional volatility with a
  Gaussian VaR quantile. We fit a lightweight recursive GARCH(1,1) by maximum
  likelihood (no external arch dependency) and form VaR_t = z_{alpha} * sigma_t.
- ``method="ewma"``: RiskMetrics EWMA volatility (lambda=0.94), Gaussian VaR.
- ``method="empirical"``: rolling historical-simulation quantile VaR.

All three are standard reference models a validator would accept. The true_var
is intentionally honest (it covers at roughly the nominal rate); the solver's
job is to under-state it as far as the gate panel allows.

Returns are cached as .npz under ``data_cache/`` so the tool and tests are
deterministic and run offline after the first fetch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.signal import lfilter

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")


@dataclass
class Dataset:
    asset: str
    returns: np.ndarray      # realized simple daily returns
    true_var: np.ndarray     # honest reference VaR, positive loss magnitude
    alpha: float             # nominal tail prob (0.01 = 99% VaR)
    method: str
    dates: np.ndarray | None = None


# ---------------------------------------------------------------------------
# fetch / cache
# ---------------------------------------------------------------------------

def _cache_path(asset: str) -> str:
    return os.path.join(CACHE_DIR, f"{asset}_returns.npz")


def fetch_returns(asset: str, start: str = "2015-01-01", end: str = "2024-12-31",
                  use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Fetch daily simple returns for ``asset``. Cache to npz.

    Returns (returns, dates_as_int64_ns).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(asset)
    if use_cache and os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["returns"], d["dates"]

    import yfinance as yf

    df = yf.download(asset, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {asset}")
    close = df["Close"].to_numpy().reshape(-1)
    rets = np.diff(close) / close[:-1]
    dates = df.index.to_numpy()[1:].astype("datetime64[ns]").astype(np.int64)
    np.savez(path, returns=rets, dates=dates)
    return rets, dates


# ---------------------------------------------------------------------------
# reference VaR models
# ---------------------------------------------------------------------------

def ewma_vol(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA conditional volatility (one-step-ahead, causal)."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    var = np.empty(n)
    # seed with sample variance of the first chunk
    seed = max(20, int(0.05 * n))
    s = np.var(r[:seed])
    for t in range(n):
        var[t] = s
        s = lam * s + (1 - lam) * r[t] ** 2
    return np.sqrt(var)


def _garch11_negloglik(params, r):
    omega, a, b = params
    if omega <= 0 or a < 0 or b < 0 or a + b >= 1:
        return 1e12
    n = r.size
    sig2 = np.empty(n)
    sig2[0] = np.var(r)
    for t in range(1, n):
        sig2[t] = omega + a * r[t - 1] ** 2 + b * sig2[t - 1]
    sig2 = np.maximum(sig2, 1e-12)
    ll = -0.5 * np.sum(np.log(2 * np.pi * sig2) + r ** 2 / sig2)
    return -ll


def garch_vol(returns: np.ndarray) -> np.ndarray:
    """One-step-ahead GARCH(1,1) conditional volatility, fit by MLE.

    Causal: sigma_t depends only on info up to t-1.
    """
    r = np.asarray(returns, dtype=float)
    var0 = np.var(r)
    x0 = np.array([var0 * 0.05, 0.05, 0.90])
    res = optimize.minimize(
        _garch11_negloglik, x0, args=(r,), method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10},
    )
    omega, a, b = res.x
    if omega <= 0 or a < 0 or b < 0 or a + b >= 1:
        omega, a, b = var0 * 0.05, 0.05, 0.90
    n = r.size
    sig2 = np.empty(n)
    sig2[0] = var0
    for t in range(1, n):
        sig2[t] = omega + a * r[t - 1] ** 2 + b * sig2[t - 1]
    return np.sqrt(np.maximum(sig2, 1e-12))


def _garch_fit_params(r: np.ndarray):
    """Fit GARCH(1,1) by MLE on ``r``; return (omega, a, b), defaulting on failure."""
    r = np.asarray(r, dtype=float)
    var0 = np.var(r)
    x0 = np.array([var0 * 0.05, 0.05, 0.90])
    res = optimize.minimize(
        _garch11_negloglik, x0, args=(r,), method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10},
    )
    omega, a, b = res.x
    if omega <= 0 or a < 0 or b < 0 or a + b >= 1:
        omega, a, b = var0 * 0.05, 0.05, 0.90
    return float(omega), float(a), float(b)


def _garch11_negloglik_fast(params, r):
    """Same objective as ``_garch11_negloglik``; the recursion runs in scipy.signal.lfilter."""
    omega, a, b = params
    if omega <= 0 or a < 0 or b < 0 or a + b >= 1:
        return 1e12
    n = r.size
    sig2 = np.empty(n)
    sig2[0] = np.var(r)
    if n > 1:
        u = omega + a * r[:-1] ** 2
        sig2[1:], _ = lfilter([1.0], [1.0, -b], u, zi=[b * sig2[0]])
    sig2 = np.maximum(sig2, 1e-12)
    ll = -0.5 * np.sum(np.log(2 * np.pi * sig2) + r ** 2 / sig2)
    return -ll


def _garch_fit_params_fast(r: np.ndarray):
    """``_garch_fit_params`` with the lfilter likelihood (same start, method and tolerances)."""
    r = np.asarray(r, dtype=float)
    var0 = np.var(r)
    x0 = np.array([var0 * 0.05, 0.05, 0.90])
    res = optimize.minimize(
        _garch11_negloglik_fast, x0, args=(r,), method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10},
    )
    omega, a, b = res.x
    if omega <= 0 or a < 0 or b < 0 or a + b >= 1:
        omega, a, b = var0 * 0.05, 0.05, 0.90
    return float(omega), float(a), float(b)


def _garch_filter(r: np.ndarray, omega: float, a: float, b: float,
                  var0: float | None = None) -> np.ndarray:
    """Run the GARCH(1,1) variance recursion with fixed params; return sigma_t.

    sigma_t depends only on info up to t-1 GIVEN the params; whether the WHOLE
    series is causal depends on how the params were obtained (see rolling vs
    full-sample below).
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    if var0 is None:
        var0 = np.var(r)
    sig2 = np.empty(n)
    sig2[0] = var0
    for t in range(1, n):
        sig2[t] = omega + a * r[t - 1] ** 2 + b * sig2[t - 1]
    return np.sqrt(np.maximum(sig2, 1e-12))


def rolling_garch_vol(returns: np.ndarray, min_train: int = 500,
                      refit_every: int = 25, fit=None) -> np.ndarray:
    """ROLLING OUT-OF-SAMPLE GARCH(1,1) one-step-ahead volatility.

    The full-sample ``garch_vol`` fits the GARCH parameters by MLE on the ENTIRE
    series and then runs the recursion forward. Even though the recursion itself
    is causal in the state, the PARAMETERS were estimated using future data, so the
    reference series has look-ahead. This function removes it: at each point the
    parameters are fit only on data strictly BEFORE that point.

    Concretely, parameters are re-estimated on an expanding window every
    ``refit_every`` days using returns[:t] only, then the variance recursion is run
    forward over the whole prefix to produce the one-step-ahead sigma at each day
    of the next block. The first ``min_train`` days are seeded with EWMA (there is
    not yet enough history for a stable GARCH fit), so they carry no look-ahead
    either. This is the causal, genuinely out-of-sample reference.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    sigma = np.empty(n)

    # warm-up block: EWMA (causal) until we have min_train days to fit GARCH
    ew = ewma_vol(r)
    warm = min(min_train, n)
    sigma[:warm] = ew[:warm]
    if warm >= n:
        return sigma

    # expanding-window refits; between refits reuse the last fitted params but keep
    # the recursion seeded from the variance the model implies at the block start.
    t = warm
    params = None
    while t < n:
        # refit on data strictly before t (expanding window), at refit boundaries
        if params is None or (t - warm) % refit_every == 0:
            params = (fit or _garch_fit_params)(r[:t])
        omega, a, b = params
        block_end = min(t + refit_every, n)
        # run the causal recursion over the whole prefix r[:block_end] with these
        # params, but only READ sigma at indices [t:block_end]; sigma[t] uses
        # r[:t] (past) only, so it is one-step-ahead and look-ahead-free.
        sig_prefix = _garch_filter(r[:block_end], omega, a, b)
        sigma[t:block_end] = sig_prefix[t:block_end]
        t = block_end
    return sigma


def empirical_var(returns: np.ndarray, alpha: float = 0.01, window: int = 250) -> np.ndarray:
    """Rolling historical-simulation VaR (positive loss magnitude), causal."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    var = np.empty(n)
    for t in range(n):
        lo = max(0, t - window)
        hist = r[lo:t] if t > lo else r[:1]
        # VaR = -(alpha quantile of returns)
        var[t] = -np.quantile(hist, alpha)
    return np.abs(var)


def reference_var(returns: np.ndarray, alpha: float = 0.01, method: str = "garch") -> np.ndarray:
    """Honest reference VaR as a positive loss magnitude.

    ``method``:
      - ``"garch"``        : full-sample-MLE GARCH(1,1). Parameters fit on the
                             ENTIRE series, so the reference has LOOK-AHEAD. Kept
                             for backward comparability; not the causal reference.
      - ``"rolling_garch"``: ROLLING OUT-OF-SAMPLE GARCH(1,1). Parameters fit only
                             on past data at each point. The honest, causal
                             reference (no look-ahead).
      - ``"ewma"``         : RiskMetrics EWMA (causal).
      - ``"empirical"``    : rolling historical-simulation quantile (causal).
    """
    z = -stats.norm.ppf(alpha)  # positive, e.g. 2.326 for alpha=0.01
    if method == "garch":
        return z * garch_vol(returns)
    if method == "rolling_garch":
        return z * rolling_garch_vol(returns)
    if method == "ewma":
        return z * ewma_vol(returns)
    if method == "empirical":
        return empirical_var(returns, alpha=alpha)
    raise ValueError(f"unknown method {method!r}")


# ---------------------------------------------------------------------------
# public loader
# ---------------------------------------------------------------------------

def load_data(asset: str = "AAPL", alpha: float = 0.01, method: str = "garch",
              start: str = "2015-01-01", end: str = "2024-12-31",
              use_cache: bool = True) -> Dataset:
    """Load returns + an honest reference VaR for ``asset``."""
    rets, dates = fetch_returns(asset, start=start, end=end, use_cache=use_cache)
    tv = reference_var(rets, alpha=alpha, method=method)
    return Dataset(asset=asset, returns=rets, true_var=tv, alpha=alpha,
                   method=method, dates=dates)
