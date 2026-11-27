"""End-to-end audit + adapter-boundary tests (incl. the HELD live adapter)."""

from __future__ import annotations

import pytest

from gaming_oracle.leakage.adapters import (
    HeldError,
    LiveLLMAdapter,
    ModelAdapter,
)
from gaming_oracle.leakage.audit import run_audit
from gaming_oracle.leakage.probes import ProbeKind
from gaming_oracle.leakage.synthetic import SyntheticLeakyModel


def test_synthetic_model_satisfies_adapter_protocol():
    assert isinstance(SyntheticLeakyModel(), ModelAdapter)


def test_live_adapter_is_held_and_says_what_it_needs():
    adapter = LiveLLMAdapter(api_key="ignored")
    # constructing is fine; calling raises a clear HELD error with the checklist
    assert isinstance(adapter, ModelAdapter)
    with pytest.raises(HeldError) as exc:
        adapter.decide(None)
    msg = str(exc.value)
    assert "HELD" in msg
    for need in ("endpoint", "parser", "budget", "temperature"):
        assert need in msg.lower()


def test_audit_clean_on_leakfree_model(fake_date_pairs):
    model = SyntheticLeakyModel(gain=0.0, remembered_years=(2018, 2021))
    rep = run_audit(fake_date_pairs, model)
    assert rep.verdict == "clean"
    assert rep.kind == ProbeKind.FAKE_DATE.value
    assert "clean" in rep.summary().lower()


def test_audit_leaking_on_leaky_model(fake_date_pairs):
    model = SyntheticLeakyModel(gain=0.6, honest_scale=0.0,
                                remembered_years=(2018, 2021))
    rep = run_audit(fake_date_pairs, model, reported_gross_alpha=100.0)
    assert rep.verdict == "leaking"
    # ~all reported alpha is attributed to the leak
    assert rep.deflation.deflated_amount is not None
    assert rep.deflation.deflated_amount > 50.0
    assert "LEAKING" in rep.summary()


def test_audit_rejects_mixed_kinds(fake_date_pairs, mask_pairs):
    mixed = fake_date_pairs[:5] + mask_pairs[:5]
    with pytest.raises(ValueError):
        run_audit(mixed, SyntheticLeakyModel())


def test_audit_shares_responses_across_arms(fake_date_pairs):
    """Each unique probe is queried once (paired-arm caching)."""
    calls = {"n": 0}
    base = SyntheticLeakyModel(gain=0.3, remembered_years=(2018, 2021))

    class Counting:
        def decide(self, probe):
            calls["n"] += 1
            return base.decide(probe)

    run_audit(fake_date_pairs, Counting())
    # 2 unique probes per pair (factual + counterfactual), all distinct ids
    assert calls["n"] == 2 * len(fake_date_pairs)
