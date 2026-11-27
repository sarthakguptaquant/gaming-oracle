"""Tests for the non-uniform MILP gate-contribution decomposition (Cycle 3)."""

import numpy as np
import pytest

from gaming_oracle import gates as g
from gaming_oracle.gates import GateResult
from gaming_oracle.milp_decomposition import leave_one_out_milp

ALPHA = 0.01
LEVEL = 0.95


def _definable_window(n=250, sigma=0.01, n_exc=3, seed=11):
    """A 250-day calibrated window the honest reference passes (definable lie)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, sigma, n)
    from scipy.stats import norm
    z = -norm.ppf(ALPHA)
    true_var = np.full(n, z * sigma)
    nat = (-returns > true_var)
    returns[nat] = -0.5 * true_var[nat]
    if n_exc > 0:
        idx = np.linspace(0, n - 1, n_exc + 2).astype(int)[1:-1]
        returns[idx] = -(true_var[idx] + 0.0005)
    return true_var, returns


def _calibrated_normal(n=250, sigma=0.01, seed=11):
    return _definable_window(n=n, sigma=sigma, seed=seed)


def always_pass(stated_var, returns, alpha=0.01, level=0.95):
    return GateResult(passed=True, stat=0.0, detail={"test": "always_pass"})


def test_milp_known_answer_always_pass_is_dead_weight():
    """An always-pass gate next to a real gate contributes exactly 0 under MILP."""
    tv, rr = _calibrated_normal()
    panel = [g.kupiec_pof, always_pass]
    decomp = leave_one_out_milp(rr, tv, asset="SYNTH", panel=panel,
                                alpha=ALPHA, level=LEVEL)
    by_name = {c.gate_name: c for c in decomp.contributions}
    assert by_name["always_pass"].contribution_bps == pytest.approx(0.0, abs=1e-9)
    assert not by_name["always_pass"].binds
    assert by_name["always_pass"].mul_without_bps == pytest.approx(
        decomp.mul_full_bps, abs=1e-6)


def test_milp_contributions_non_negative():
    """Dropping any gate can only grow (or hold) the lie."""
    tv, rr = _calibrated_normal(seed=3)
    decomp = leave_one_out_milp(rr, tv, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    for c in decomp.contributions:
        assert c.contribution_bps >= -1e-9
        assert c.mul_without_bps >= decomp.mul_full_bps - 1e-3
        assert c.mul_without_bps <= decomp.mul_nopanel_bps + 1e-6


def test_milp_full_panel_le_ceiling():
    tv, rr = _calibrated_normal(seed=5)
    decomp = leave_one_out_milp(rr, tv, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    assert decomp.mul_full_bps <= decomp.mul_nopanel_bps + 1e-6
    assert decomp.mul_nopanel_bps == pytest.approx(float(np.mean(tv)) * 1e4, rel=1e-9)


def test_milp_binding_and_dead_weight_partition():
    """binding_gates() and dead_weight_gates() partition the panel."""
    tv, rr = _calibrated_normal(seed=4)
    decomp = leave_one_out_milp(rr, tv, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    binders = set(decomp.binding_gates())
    dead = set(decomp.dead_weight_gates())
    assert binders.isdisjoint(dead)
    assert binders | dead == {c.gate_name for c in decomp.contributions}


def test_milp_decomposition_determinism():
    tv, rr = _calibrated_normal(seed=6)
    d1 = leave_one_out_milp(rr, tv, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    d2 = leave_one_out_milp(rr, tv, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    assert d1.mul_full_bps == d2.mul_full_bps
    assert [c.contribution_bps for c in d1.contributions] == \
           [c.contribution_bps for c in d2.contributions]
