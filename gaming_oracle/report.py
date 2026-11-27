"""Pricing the Maximum Undetectable Lie three ways + headline assembly.

Given the solved ``stated_var`` and the honest ``true_var`` (both positive loss
magnitudes), price the lie as:

(a) bps   -- mean understatement in basis points of return:
              mul_bps = mean_t( true_var_t - stated_var_t ) * 1e4
(b) dollars -- on a stated notional N:
              mul_usd = mean understatement (fractional) * N
(c) extra exceptions per year -- how many MORE VaR breaches per 252 trading
    days the stated series tolerates versus the honest reference:
              extra/yr = (exc_rate_stated - exc_rate_true) * 252

The lie is PANEL-RELATIVE: undetectable only against the specific gate panel
used. Never read as a universal "this model lies by X bps".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gates as gates_mod

TRADING_DAYS = 252


@dataclass
class Pricing:
    mul_bps: float
    mul_usd: float
    extra_exceptions_per_year: float
    notional: float
    exc_rate_true: float
    exc_rate_stated: float
    mean_understatement: float
    panel_names: list


def price(stated_var, true_var, returns, notional: float = 1_000_000_000.0,
          panel=None, panel_names=None) -> Pricing:
    """Price the lie three ways."""
    stated_var = np.asarray(stated_var, dtype=float)
    true_var = np.asarray(true_var, dtype=float)
    returns = np.asarray(returns, dtype=float)

    delta = float(np.mean(true_var - stated_var))   # fractional return units
    mul_bps = delta * 1e4
    mul_usd = delta * notional

    e_true = gates_mod.exceptions(true_var, returns)
    e_stated = gates_mod.exceptions(stated_var, returns)
    rate_true = float(e_true.mean()) if e_true.size else 0.0
    rate_stated = float(e_stated.mean()) if e_stated.size else 0.0
    extra_per_year = (rate_stated - rate_true) * TRADING_DAYS

    if panel_names is None:
        panel = panel or gates_mod.PANEL
        panel_names = [g.__name__ for g in panel]

    return Pricing(
        mul_bps=mul_bps,
        mul_usd=mul_usd,
        extra_exceptions_per_year=extra_per_year,
        notional=notional,
        exc_rate_true=rate_true,
        exc_rate_stated=rate_stated,
        mean_understatement=delta,
        panel_names=list(panel_names),
    )


def headline(pricing: Pricing, asset: str) -> str:
    """The conference-slide sentence. Panel-relative, no universal claim."""
    n = pricing.notional
    return (
        f"On {asset}, against the panel {pricing.panel_names}, this search "
        f"exhibited a lie of up to {pricing.mul_bps:.1f} bps of risk "
        f"understatement = ${pricing.mul_usd:,.0f} on a ${n:,.0f} notional = "
        f"{pricing.extra_exceptions_per_year:.2f} extra VaR exceptions/year, "
        f"all gates green. (Panel-relative; not a universal model claim.)"
    )
