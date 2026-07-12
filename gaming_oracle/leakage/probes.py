"""Paired probes for temporal-leakage detection.

A *probe* is a single decision context handed to the model under test: an
as-of date, the text/features it may see, and the ground-truth forward return
of the asset over the decision horizon (used only by the auditor to score
alpha, never shown to the model).

A *probe pair* is two probes that are IDENTICAL except for the one channel we
want to test. The auditor compares the model's responses across the pair.

Three probe kinds are shipped:

FAKE_DATE (counterfactual date)
    The factual probe carries the true as-of date; the counterfactual probe
    carries a *shifted* as-of date (e.g. one year earlier) with the SAME market
    context text. A model with no lookahead leakage should decide the same way
    regardless of the stated calendar date -- the decision depends on the
    context, not on "what year it is". A model that memorized the future keys
    off the true date (it "knows" what happens next in 2020) and its decision
    moves when the date is falsified. The response delta is the leakage signal.

ENTITY_MASK (name masking)
    The factual probe names the real entity ("Apple", "AAPL"); the
    counterfactual probe replaces it with a neutral placeholder ("Company X")
    while keeping the numeric context identical. Memorized outcomes are keyed on
    the entity's name; masking it removes that channel. A decision that changes
    only because the name was hidden is leaking name-conditioned future
    knowledge.

EMBARGO (ingestion-lag / point-in-time)
    The factual probe includes a feature whose *value as revised later* differs
    from its *value as it stood at t* (e.g. a restated fundamental, a
    backfilled index membership); the counterfactual probe uses the point-in-time
    value. This is a DATA-side leak, audited structurally in ``embargo.py``; the
    probe form here lets a model-side sensitivity test run alongside the
    structural audit.

These are standard placebo / manipulation-check designs from causal inference,
applied to LLM backtests. Nothing here is novel; see the package docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class ProbeKind(str, Enum):
    """The temporal channel a probe pair isolates."""

    FAKE_DATE = "fake_date"
    ENTITY_MASK = "entity_mask"
    EMBARGO = "embargo"


@dataclass(frozen=True)
class Probe:
    """One decision context shown to the model under test.

    ``asof`` is the date the model is TOLD it is (may be falsified in the
    counterfactual arm). ``context`` is the text/feature blob the model sees.
    ``forward_return`` is the realized return of the asset over the decision
    horizon starting at the TRUE as-of date; it is auditor-only ground truth and
    must never be embedded in ``context``.
    """

    probe_id: str
    asof: date
    context: str
    forward_return: float
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProbePair:
    """A factual probe and its counterfactual twin, differing in one channel."""

    kind: ProbeKind
    factual: Probe
    counterfactual: Probe
    # The true forward return is a property of the underlying date/asset and is
    # shared by both arms (only the *stated* date / name / feature is falsified).
    forward_return: float


@dataclass(frozen=True)
class ProbeResponse:
    """The model's numeric decision for one probe.

    ``signal`` is a directional position in ``[-1, 1]`` (short..long); this is
    the quantity whose across-arm delta is the leakage signal and whose product
    with ``forward_return`` is the realized per-probe pnl.
    """

    probe_id: str
    signal: float
    raw: Optional[str] = None  # optional model text, for audit trails


def _shift_years(d: date, years: int) -> date:
    """Shift a date by whole years, clamping Feb 29 to Feb 28."""
    try:
        return d.replace(year=d.year - years)
    except ValueError:  # Feb 29 -> non-leap year
        return d.replace(year=d.year - years, day=28)


def make_fake_date_pairs(
    probes: list[Probe],
    shift_years: int = 1,
) -> list[ProbePair]:
    """Build FAKE_DATE pairs by back-dating the stated as-of by ``shift_years``.

    The counterfactual keeps the SAME context text and forward return; only the
    stated calendar date changes. ``shift_years`` should be large enough that a
    memorizing model's "what happens next" prior differs between the arms
    (default 1 year), but the context is unchanged so a leak-free model is
    invariant.
    """
    if shift_years <= 0:
        raise ValueError("shift_years must be positive")
    pairs: list[ProbePair] = []
    for p in probes:
        cf = Probe(
            probe_id=f"{p.probe_id}__cf_date",
            asof=_shift_years(p.asof, shift_years),
            context=p.context,
            forward_return=p.forward_return,
            meta={**p.meta, "cf": "fake_date", "shift_years": shift_years},
        )
        pairs.append(
            ProbePair(
                kind=ProbeKind.FAKE_DATE,
                factual=p,
                counterfactual=cf,
                forward_return=p.forward_return,
            )
        )
    return pairs


def make_entity_mask_pairs(
    probes: list[Probe],
    entity: str,
    placeholder: str = "Company X",
    ticker: Optional[str] = None,
    ticker_placeholder: str = "TICKER",
) -> list[ProbePair]:
    """Build ENTITY_MASK pairs by replacing the real name with a placeholder.

    The replacement is a literal, case-sensitive string swap on ``entity`` (and
    ``ticker`` if given) in the context text. If the name does not appear in a
    probe's context, that probe is skipped (masking it would be a no-op and a
    no-op pair carries no signal, which would bias the score toward zero).
    """
    if not entity:
        raise ValueError("entity must be a non-empty string")
    pairs: list[ProbePair] = []
    for p in probes:
        if entity not in p.context and (ticker is None or ticker not in p.context):
            continue
        masked = p.context.replace(entity, placeholder)
        if ticker is not None:
            masked = masked.replace(ticker, ticker_placeholder)
        cf = Probe(
            probe_id=f"{p.probe_id}__cf_mask",
            asof=p.asof,
            context=masked,
            forward_return=p.forward_return,
            meta={**p.meta, "cf": "entity_mask", "entity": entity},
        )
        pairs.append(
            ProbePair(
                kind=ProbeKind.ENTITY_MASK,
                factual=p,
                counterfactual=cf,
                forward_return=p.forward_return,
            )
        )
    return pairs
