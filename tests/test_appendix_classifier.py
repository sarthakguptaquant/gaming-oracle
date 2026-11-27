"""APPENDIX tests (Cycle 6): the gameability decomposition on a SECOND, non-VaR
pass/fail gate -- a classifier reliability panel.

These tests prove the generalization runs end-to-end with the SAME contracts:
- the honest calibrated reference passes the reliability panel (definable lie),
- the inflation solver produces a stated_conf that is genuinely self-serving
  (stated_conf >= honest_conf everywhere, mirror of stated_var <= true_var),
- the produced stated_conf passes ALL gates jointly (independent re-verify),
- the prices re-derive from (stated_conf, honest_conf, correct),
- and a KNOWN-ANSWER leave-one-out test: an always-pass mock gate added next to a
  real gate contributes EXACTLY zero, the real gate owns the whole share. This is
  the identical known-answer invariant the VaR decomposition test asserts.

This is a teaser, not a general theorem: it shows the interface is not
VaR-specific, nothing more.
"""

import numpy as np
import pytest

from gaming_oracle.gates import GateResult
from gaming_oracle.appendix_classifier import (
    synthetic_classifier,
    run_reliability_panel,
    inflate_confidence,
    price_conf,
    leave_one_out_conf,
    ece_gate,
    overconfidence_count_gate,
    selective_accuracy_gate,
    high_conf_errors,
    RELIABILITY_PANEL,
)

ALPHA = 0.01
LEVEL = 0.95


def always_pass(stated_conf, correct, alpha=0.01, level=0.95):
    """A mock reliability gate that ALWAYS passes (never binds)."""
    return GateResult(passed=True, stat=0.0, detail={"test": "always_pass"})


# ---------------------------------------------------------------------------
# the honest reference is definable: it passes the whole panel
# ---------------------------------------------------------------------------

def test_honest_reference_passes_panel():
    """The calibrated synthetic classifier passes every reliability gate, so the
    concealable inflation is well defined (mirror of the panel-calibrated VaR
    reference)."""
    hc, correct = synthetic_classifier()
    out = run_reliability_panel(hc, correct)
    assert out["passed_all"], {k: v.passed for k, v in out["gates"].items()}


# ---------------------------------------------------------------------------
# the solver exhibits a genuine, self-serving, feasible lie
# ---------------------------------------------------------------------------

def test_inflation_is_self_serving_and_feasible():
    hc, correct = synthetic_classifier()
    res = inflate_confidence(hc, correct)

    # genuinely self-serving: confidence only ever inflated, never deflated
    assert np.all(res.stated_conf >= hc - 1e-12)
    assert np.all(res.stated_conf <= 1.0 + 1e-12)
    # a non-trivial lie was found
    assert res.mean_inflation > 0.0
    assert res.passed_all

    # INDEPENDENT re-verification: recompute the panel from scratch on the
    # produced series; it must pass every gate jointly.
    out = run_reliability_panel(res.stated_conf, correct)
    assert out["passed_all"]


def test_prices_rederive_from_series():
    """The 3-way price must re-derive from (stated, honest, correct); no hidden
    state, mirror of the VaR pricing re-derivation check."""
    hc, correct = synthetic_classifier()
    res = inflate_confidence(hc, correct)
    p = price_conf(res.stated_conf, hc, correct, served=1_000_000.0)

    # (a) mean inflation in points re-derived
    expected_delta = float(np.mean(res.stated_conf - hc))
    assert p.mcl_points == pytest.approx(expected_delta, abs=1e-12)
    assert p.mcl_pct_points == pytest.approx(expected_delta * 100.0, abs=1e-9)

    # (b) extra silent errors = rise in high-confidence-error rate * served
    h_h = high_conf_errors(hc, correct)
    h_s = high_conf_errors(res.stated_conf, correct)
    expected_extra = (h_s.mean() - h_h.mean()) * 1_000_000.0
    assert p.extra_silent_errors == pytest.approx(expected_extra, rel=1e-9)
    # inflating confidence can only create high-confidence errors, never remove
    assert p.hc_error_rate_stated >= p.hc_error_rate_honest - 1e-12


# ---------------------------------------------------------------------------
# KNOWN-ANSWER leave-one-out test (identical invariant to the VaR side)
# ---------------------------------------------------------------------------

def test_known_answer_always_pass_gate_has_zero_contribution():
    """An always-pass gate next to a real gate contributes exactly 0; the real
    gate owns the whole detection share. Same known-answer invariant as
    tests/test_decomposition.py on the VaR engine."""
    hc, correct = synthetic_classifier()
    panel = [overconfidence_count_gate, always_pass]
    d = leave_one_out_conf(hc, correct, panel=panel, alpha=ALPHA, level=LEVEL)

    by_name = {c.gate_name: c for c in d.contributions}
    assert "always_pass" in by_name and "overconfidence_count_gate" in by_name

    # the mock is dead weight: removing it changes nothing
    assert by_name["always_pass"].contribution_pp == pytest.approx(0.0, abs=1e-9)
    assert by_name["always_pass"].mcl_without_pp == pytest.approx(
        d.mcl_full_pp, abs=1e-6)
    # the real gate is ranked first and owns the whole span
    assert d.contributions[0].gate_name == "overconfidence_count_gate"
    assert by_name["overconfidence_count_gate"].contribution_frac == pytest.approx(
        1.0, abs=1e-6)


def test_contributions_non_negative_and_bounded():
    """Dropping any reliability gate can only grow (or hold) the lie."""
    hc, correct = synthetic_classifier()
    d = leave_one_out_conf(hc, correct, panel=RELIABILITY_PANEL,
                           alpha=ALPHA, level=LEVEL)
    for c in d.contributions:
        assert c.contribution_pp >= -1e-9
        assert c.mcl_without_pp >= d.mcl_full_pp - 1e-6
        assert c.mcl_without_pp <= d.mcl_nopanel_pp + 1e-6
    # ranked descending
    pps = [c.contribution_pp for c in d.contributions]
    assert pps == sorted(pps, reverse=True)


def test_unconstrained_ceiling_is_inflate_to_one():
    """No-gate ceiling = inflate every confidence to 1.0 = mean(1 - honest)."""
    hc, correct = synthetic_classifier()
    d = leave_one_out_conf(hc, correct, panel=RELIABILITY_PANEL,
                           alpha=ALPHA, level=LEVEL)
    expected_ceiling = float(np.mean(1.0 - hc)) * 100.0
    assert d.mcl_nopanel_pp == pytest.approx(expected_ceiling, rel=1e-9)
    assert d.mcl_full_pp <= d.mcl_nopanel_pp + 1e-6


def test_a_real_gate_binds():
    """On this instance at least one gate is load-bearing (binds alone): its solo
    MCL equals the full-panel MCL. This is the substantive generalization result
    -- the decomposition identifies which audit check is doing the detecting,
    exactly as on the VaR engine."""
    hc, correct = synthetic_classifier()
    d = leave_one_out_conf(hc, correct, panel=RELIABILITY_PANEL,
                           alpha=ALPHA, level=LEVEL)
    binders = [c for c in d.contributions if c.binds_alone]
    assert binders, "expected at least one load-bearing reliability gate"


def test_determinism():
    """Same inputs -> identical decomposition (no hidden randomness)."""
    hc, correct = synthetic_classifier()
    d1 = leave_one_out_conf(hc, correct, panel=RELIABILITY_PANEL)
    d2 = leave_one_out_conf(hc, correct, panel=RELIABILITY_PANEL)
    assert d1.mcl_full_pp == d2.mcl_full_pp
    assert [c.contribution_pp for c in d1.contributions] == \
           [c.contribution_pp for c in d2.contributions]


def test_ece_gate_rejects_overconfidence():
    """Sanity: a grossly over-confident series (all conf -> 0.999) fails the ECE
    gate, so the gate is not a rubber stamp."""
    hc, correct = synthetic_classifier()
    overconf = np.full_like(hc, 0.999)
    assert not ece_gate(overconf, correct).passed


def test_adding_a_gate_does_not_increase_the_lie():
    """Mirror of the VaR killer-demo invariant
    (tests/test_solver_report.py::test_adding_du_escanciano_does_not_increase_lie):
    adding a real gate to the panel can only SHRINK or hold the exhibited MCL,
    because more gates = a superset of constraints = a smaller feasible set. This
    is the monotonicity that makes the leave-one-out contribution non-negative,
    asserted here on the classifier side. The search is exhibited (bisection +
    greedy), not a proven max, so we allow a tiny numerical slack and assert the
    inequality direction, not an exact equality."""
    hc, correct = synthetic_classifier()
    base = [selective_accuracy_gate]
    bigger = [selective_accuracy_gate, overconfidence_count_gate]

    r_base = inflate_confidence(hc, correct, panel=base, alpha=ALPHA, level=LEVEL)
    r_big = inflate_confidence(hc, correct, panel=bigger, alpha=ALPHA, level=LEVEL)

    # both feasible against their own panels
    assert run_reliability_panel(r_base.stated_conf, correct, panel=base)["passed_all"]
    assert run_reliability_panel(r_big.stated_conf, correct, panel=bigger)["passed_all"]

    # adding the overconfidence gate cannot ENLARGE the exhibited lie
    assert r_big.mean_inflation <= r_base.mean_inflation + 1e-9
    # and the larger panel's stated series still passes the smaller panel
    # (its constraints are a subset), confirming the superset relationship
    assert run_reliability_panel(r_big.stated_conf, correct, panel=base)["passed_all"]
