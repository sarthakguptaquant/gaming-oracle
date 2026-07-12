"""A synthetic model whose leakage is KNOWN by construction.

This is the validation ground truth. It is NOT a real LLM and never claims to
be. Its only job is to let the estimator be tested end to end: we inject a known
amount of temporal leakage and confirm the auditor recovers it within its
confidence interval. If the estimator could not recover a leak we planted
ourselves, no number it reports on a real model could be trusted.

MODEL
-----
For each probe the model emits

    signal = clip( honest(context) + gain * memory(probe) , -1, 1 )

where

- ``honest(context)`` reads a PUBLIC feature embedded in the context (a
  momentum-like number in the text) and takes its sign-scaled value. This is the
  genuine, real-time-available edge (which may be ~0).

- ``memory(probe)`` is the LEAK: it returns the sign of the forward return, but
  ONLY when the probe is "recognized" -- i.e. the stated date matches a date the
  model "remembers" (its trained era) AND the entity name is present. This
  mimics memorization keyed on the real calendar date and the real entity name.

Counterfactual probes falsify exactly one of those keys:
- FAKE_DATE moves ``asof`` outside the remembered era -> memory() returns 0.
- ENTITY_MASK removes the name from the context -> memory() returns 0.

So the paired delta isolates ``gain * sign(forward_return)``, and the deflation
the auditor should recover is analytically the share of factual pnl coming from
that ``gain`` term. ``gain = 0`` gives a leak-free model (score ~ 0, deflation
CI straddling 0); larger ``gain`` gives larger, recoverable leakage.
"""

from __future__ import annotations

import re
from datetime import date

import numpy as np

from .probes import Probe, ProbeResponse

# The public momentum feature is written into probe context as e.g. "mom=0.20".
_MOM_RE = re.compile(r"mom=(-?\d+(?:\.\d+)?)")


class SyntheticLeakyModel:
    """Known-leakage stand-in adapter (implements the ModelAdapter protocol).

    Parameters
    ----------
    gain:
        Strength of the memorized-future channel. 0 = honest, no leak.
    honest_scale:
        How strongly the public momentum feature drives the honest signal.
    remembered_years:
        The (inclusive) year range the model "remembers". A probe whose stated
        year falls outside this range gets no memory contribution -- this is how
        a back-dated (fake-date) probe defeats the leak.
    entity_token:
        The entity marker whose presence in context is required to trigger
        memory. Masking it (ENTITY_MASK) removes the leak. Default matches the
        placeholder convention in ``probes.make_entity_mask_pairs`` (real name
        present vs "Company X").
    """

    def __init__(
        self,
        gain: float = 0.5,
        honest_scale: float = 0.0,
        remembered_years: tuple[int, int] = (2000, 2100),
        entity_token: str = "Company X",
    ):
        self.gain = float(gain)
        self.honest_scale = float(honest_scale)
        self.remembered_years = remembered_years
        # memory fires when the entity is NOT masked; we detect masking by the
        # placeholder token appearing in the (masked) counterfactual context.
        self.mask_token = entity_token

    def _honest(self, context: str) -> float:
        m = _MOM_RE.search(context)
        if not m:
            return 0.0
        return self.honest_scale * float(m.group(1))

    def _remembers(self, probe: Probe) -> bool:
        lo, hi = self.remembered_years
        in_era = lo <= probe.asof.year <= hi
        not_masked = self.mask_token not in probe.context
        return in_era and not_masked

    def decide(self, probe: Probe) -> ProbeResponse:
        honest = self._honest(probe.context)
        memory = 0.0
        if self._remembers(probe):
            memory = self.gain * float(np.sign(probe.forward_return))
        signal = float(np.clip(honest + memory, -1.0, 1.0))
        return ProbeResponse(
            probe_id=probe.probe_id,
            signal=signal,
            raw=f"honest={honest:+.3f} memory={memory:+.3f}",
        )
