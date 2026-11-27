"""Shared fixtures for the leakage-auditor tests.

Builds a deterministic probe set: N contexts each carrying a public momentum
feature (``mom=...``) and a known forward return, for a fixed entity. Both the
fake-date and entity-mask probe families are derived from this base set so the
same ground truth drives every test.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from gaming_oracle.leakage.probes import Probe, make_fake_date_pairs, make_entity_mask_pairs

ENTITY = "Apple"
TICKER = "AAPL"
N = 60


def _base_probes(n=N, seed=0):
    """n probes for one entity, real dates in the 2019-2020 era.

    Forward returns are drawn i.i.d.; momentum is written into the context but,
    by default, uncorrelated with the forward return (so an HONEST model that
    only reads momentum has ~0 alpha, and any recovered alpha is the leak).
    """
    rng = np.random.default_rng(seed)
    probes = []
    for i in range(n):
        r = float(rng.normal(0, 0.02))
        mom = float(rng.normal(0, 0.2))
        ctx = (
            f"As-of market note for {ENTITY} ({TICKER}). "
            f"Trailing momentum mom={mom:.3f}. Decide a position."
        )
        probes.append(
            Probe(
                probe_id=f"p{i:03d}",
                asof=date(2019 + (i % 2), 1 + (i % 12), 1 + (i % 27)),
                context=ctx,
                forward_return=r,
                meta={"i": i},
            )
        )
    return probes


@pytest.fixture
def base_probes():
    return _base_probes()


@pytest.fixture
def fake_date_pairs(base_probes):
    # shift the stated date back far enough to leave the "remembered era"
    return make_fake_date_pairs(base_probes, shift_years=15)


@pytest.fixture
def mask_pairs(base_probes):
    return make_entity_mask_pairs(base_probes, entity=ENTITY, ticker=TICKER)
