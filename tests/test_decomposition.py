"""Leave-one-out gate-contribution decomposition tests (Cycle 2, gap #8).

Includes a KNOWN-ANSWER leave-one-out test: with an always-pass mock gate added
to a real gate, the mock's contribution must be EXACTLY zero (removing a gate
that never binds cannot relax the feasible set) and the real gate must carry the
entire detection share.
"""

import numpy as np
import pytest

from gaming_oracle import gates as g
from gaming_oracle.decomposition import (
    Decomposition,
    leave_one_out,
    representative_window,
)
from gaming_oracle.gates import GateResult

ALPHA = 0.01
LEVEL = 0.95


def _calibrated_normal(n=500, sigma=0.01, seed=11):
    """Constant-vol normal returns + matching constant honest VaR.

    Calibrated so the honest reference passes the panel (definable lie).
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, sigma, n)
    from scipy.stats import norm
    z = -norm.ppf(ALPHA)
    true_var = np.full(n, z * sigma)
    return true_var, returns


def always_pass(stated_var, returns, alpha=0.01, level=0.95):
    """A mock gate that ALWAYS passes (never binds)."""
    return GateResult(passed=True, stat=0.0, detail={"test": "always_pass"})


# ---------------------------------------------------------------------------
# KNOWN-ANSWER leave-one-out test
# ---------------------------------------------------------------------------

def test_known_answer_always_pass_gate_has_zero_contribution():
    """An always-pass gate added next to a real gate contributes exactly 0.

    Removing a gate that never binds cannot enlarge the lie, so its
    contribution_bps must be 0 and the real gate must own ~100% of the share.
    """
    tv, rr = _calibrated_normal()
    panel = [g.kupiec_pof, always_pass]
    decomp = leave_one_out(rr, tv, asset="SYNTH", panel=panel,
                           alpha=ALPHA, level=LEVEL)

    by_name = {c.gate_name: c for c in decomp.contributions}
    assert "always_pass" in by_name and "kupiec_pof" in by_name

    # the always-pass gate is dead weight: removing it changes nothing
    assert by_name["always_pass"].contribution_bps == pytest.approx(0.0, abs=1e-9)
    # MUL with the mock removed == MUL with the full panel
    assert by_name["always_pass"].mul_without_bps == pytest.approx(
        decomp.mul_full_bps, abs=1e-6)
    # the real gate is the one doing all the detecting -> ranked first
    assert decomp.contributions[0].gate_name == "kupiec_pof"
    # and its share is the whole span (the mock owns none of it)
    assert by_name["kupiec_pof"].contribution_frac == pytest.approx(1.0, abs=1e-6)


def test_contributions_are_non_negative_monotone():
    """Dropping any gate can only grow (or hold) the lie: contribution >= 0."""
    tv, rr = _calibrated_normal(seed=3)
    decomp = leave_one_out(rr, tv, asset="SYNTH", panel=g.PANEL,
                           alpha=ALPHA, level=LEVEL)
    for c in decomp.contributions:
        assert c.contribution_bps >= -1e-9
        # each leave-one-out MUL is >= the full-panel MUL
        assert c.mul_without_bps >= decomp.mul_full_bps - 1e-6
        # and <= the unconstrained ceiling
        assert c.mul_without_bps <= decomp.mul_nopanel_bps + 1e-6


def test_full_panel_mul_le_unconstrained_ceiling():
    tv, rr = _calibrated_normal(seed=5)
    decomp = leave_one_out(rr, tv, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    assert decomp.mul_full_bps <= decomp.mul_nopanel_bps + 1e-6
    # ceiling = mean(true_var) in bps (stated -> 0 everywhere)
    assert decomp.mul_nopanel_bps == pytest.approx(float(np.mean(tv)) * 1e4, rel=1e-9)


def test_ranking_is_sorted_descending():
    tv, rr = _calibrated_normal(seed=9)
    decomp = leave_one_out(rr, tv, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    bps = [c.contribution_bps for c in decomp.contributions]
    assert bps == sorted(bps, reverse=True)


def test_representative_window_is_definable():
    """The chosen representative window must have a panel-passing honest ref."""
    tv, rr = _calibrated_normal(n=900, seed=2)
    ws, we = representative_window(rr, tv, window=250, step=50,
                                   panel=g.PANEL, alpha=ALPHA)
    assert we - ws == 250
    out = g.run_panel(tv[ws:we], rr[ws:we], panel=g.PANEL, alpha=ALPHA)
    assert out["passed_all"]


def test_decomposition_determinism():
    """Same inputs -> identical decomposition (no hidden randomness)."""
    tv, rr = _calibrated_normal(seed=4)
    d1 = leave_one_out(rr, tv, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    d2 = leave_one_out(rr, tv, panel=g.PANEL, alpha=ALPHA, level=LEVEL)
    assert d1.mul_full_bps == d2.mul_full_bps
    assert [c.contribution_bps for c in d1.contributions] == \
           [c.contribution_bps for c in d2.contributions]
