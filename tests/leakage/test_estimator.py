"""Estimator recovery tests: plant a known leak, confirm it is recovered.

The central validation. We build a synthetic model with a KNOWN leakage gain and
check that:
  1. a leak-free model (gain=0) scores ~0 and its deflation CI straddles 0;
  2. a leaking model scores significantly and its deflation is strictly positive
     with a CI above 0;
  3. the estimator is honest about the degenerate (zero factual alpha) case.
"""

from __future__ import annotations

import math

import numpy as np

from gaming_oracle.leakage.estimator import leakage_score, alpha_deflation
from gaming_oracle.leakage.probes import ProbeResponse
from gaming_oracle.leakage.synthetic import SyntheticLeakyModel


def _responses(pairs, model):
    resp = {}
    for pr in pairs:
        for p in (pr.factual, pr.counterfactual):
            resp[p.probe_id] = model.decide(p)
    return resp


# --- leak-free baseline ---------------------------------------------------

def test_leakfree_model_scores_zero_fake_date(fake_date_pairs):
    model = SyntheticLeakyModel(gain=0.0, remembered_years=(2018, 2021))
    resp = _responses(fake_date_pairs, model)
    ls = leakage_score(fake_date_pairs, resp)
    assert ls.score == 0.0
    assert not ls.significant
    assert ls.mean_abs_delta == 0.0


def test_leakfree_model_deflation_straddles_zero(fake_date_pairs):
    model = SyntheticLeakyModel(gain=0.0, remembered_years=(2018, 2021))
    resp = _responses(fake_date_pairs, model)
    de = alpha_deflation(fake_date_pairs, resp)
    # with no leak, factual alpha ~ 0 -> degenerate is expected & honest
    assert de.degenerate or (de.ci_low <= 0.0 <= de.ci_high)


# --- planted leak, fake-date channel --------------------------------------

def test_planted_leak_is_detected_fake_date(fake_date_pairs):
    model = SyntheticLeakyModel(gain=0.6, remembered_years=(2018, 2021))
    resp = _responses(fake_date_pairs, model)
    ls = leakage_score(fake_date_pairs, resp)
    assert ls.significant
    assert ls.score > 0.0
    # every pair's delta is +/- gain, so mean|delta| == gain
    assert math.isclose(ls.mean_abs_delta, 0.6, abs_tol=1e-9)


def test_planted_leak_deflation_recovered_fake_date(fake_date_pairs):
    """With gain>0 and an honest edge of 0, ALL factual alpha is the leak, so
    the deflation fraction must be ~100% with a CI strictly above 0."""
    model = SyntheticLeakyModel(gain=0.6, honest_scale=0.0,
                                remembered_years=(2018, 2021))
    resp = _responses(fake_date_pairs, model)
    de = alpha_deflation(fake_date_pairs, resp, reported_gross_alpha=1.0)
    assert not de.degenerate
    # counterfactual (leak-removed) alpha is ~0 -> fraction ~ 1.0
    assert de.ci_low > 0.0
    assert 0.8 <= de.fraction <= 1.2
    # leaked amount scales the reported alpha by the fraction
    assert de.deflated_amount is not None
    assert math.isclose(de.deflated_amount, de.fraction, rel_tol=1e-9)


def test_leak_signal_equals_gain_times_forward_sign(fake_date_pairs):
    """Mechanistic check: factual pnl per pair = gain*sign(r)*r = gain*|r|,
    so factual alpha = gain * mean(|r|), which the estimator reports."""
    gain = 0.5
    model = SyntheticLeakyModel(gain=gain, honest_scale=0.0,
                                remembered_years=(2018, 2021))
    resp = _responses(fake_date_pairs, model)
    de = alpha_deflation(fake_date_pairs, resp)
    absr = np.mean([abs(pr.forward_return) for pr in fake_date_pairs])
    assert math.isclose(de.factual_alpha, gain * absr, rel_tol=1e-9)
    assert math.isclose(de.counterfactual_alpha, 0.0, abs_tol=1e-12)


# --- planted leak, entity-mask channel ------------------------------------

def test_planted_leak_detected_entity_mask(mask_pairs):
    model = SyntheticLeakyModel(gain=0.5, remembered_years=(2018, 2021))
    resp = _responses(mask_pairs, model)
    ls = leakage_score(mask_pairs, resp)
    de = alpha_deflation(mask_pairs, resp)
    assert ls.significant and ls.score > 0
    assert de.ci_low > 0.0


# --- honest-edge case: partial deflation ----------------------------------

def test_partial_leak_gives_partial_deflation(base_probes):
    """A model with BOTH a genuine momentum edge and a leak should deflate by
    less than 100%: some alpha survives the counterfactual arm."""
    from gaming_oracle.leakage.probes import make_fake_date_pairs

    # rebuild probes where momentum genuinely predicts the forward return, so
    # the honest channel carries real (leak-free) alpha.
    rng = np.random.default_rng(7)
    from gaming_oracle.leakage.probes import Probe
    from datetime import date

    probes = []
    for i in range(120):
        mom = float(rng.normal(0, 0.3))
        # forward return correlated with momentum + noise
        r = 0.5 * mom * 0.02 + float(rng.normal(0, 0.01))
        ctx = (f"note for Apple (AAPL). mom={mom:.3f}. decide.")
        probes.append(Probe(f"m{i:03d}", date(2019, 1 + i % 12, 1 + i % 27),
                            ctx, r))
    pairs = make_fake_date_pairs(probes, shift_years=15)

    model = SyntheticLeakyModel(gain=0.4, honest_scale=1.0,
                                remembered_years=(2018, 2021))
    resp = _responses(pairs, model)
    de = alpha_deflation(pairs, resp)
    # honest alpha survives -> deflation strictly between 0 and 1
    assert not de.degenerate
    assert 0.0 < de.fraction < 1.0
    assert de.counterfactual_alpha > 0.0  # honest edge remains after leak removal
