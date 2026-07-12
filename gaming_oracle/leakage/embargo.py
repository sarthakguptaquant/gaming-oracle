"""Ingestion-lag / embargo auditor over a feature timeline.

This is the DATA side of leakage, independent of any model. A feature used at
decision date ``t`` is point-in-time-valid only if the value the backtest fed
the model was actually KNOWABLE at ``t``: it must have an ``available_at`` (the
timestamp the value was first published / ingestible) no later than ``t`` minus
the required embargo.

Two classic violations:

- INGESTION LAG: a value dated ``event_date`` but only ``available_at`` some days
  later (earnings released after close, macro prints, revised index membership).
  Using it at ``t`` in ``[event_date, available_at)`` is lookahead.

- RESTATEMENT / BACKFILL: ``value_asof`` (as it stood at t) differs from
  ``value_current`` (as later revised). A backtest that fed ``value_current`` at
  ``t`` used a number that did not yet exist in that form.

An ``embargo_days`` requirement (e.g. "no feature within 1 trading day of the
decision") models a deliberate purge zone; PurgedKFold-style embargoing is the
standard reference (López de Prado, *Advances in Financial ML*, 2018).

This module is pure and deterministic; no external data, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta


@dataclass(frozen=True)
class FeatureUse:
    """One (feature, decision-date) usage the backtest actually made."""

    name: str
    decision_date: date
    available_at: date            # when the value became ingestible
    value_asof: float | None = None      # value as it stood at decision_date
    value_current: float | None = None   # value as later revised/backfilled


@dataclass
class FeatureTimeline:
    """A collection of feature uses to audit for point-in-time validity."""

    uses: list[FeatureUse] = field(default_factory=list)

    def add(self, use: FeatureUse) -> None:
        self.uses.append(use)


@dataclass(frozen=True)
class EmbargoViolation:
    """A single point-in-time violation found by :func:`audit_embargo`."""

    name: str
    decision_date: date
    kind: str                 # "ingestion_lag" | "restatement" | "embargo_zone"
    lag_days: int             # how far the value's availability post-dates the
                              # embargoed decision boundary (0 for restatement)
    detail: dict = field(default_factory=dict)


def audit_embargo(
    timeline: FeatureTimeline,
    embargo_days: int = 0,
    restatement_tol: float = 0.0,
) -> list[EmbargoViolation]:
    """Return every point-in-time violation in ``timeline``.

    A use is a violation when either:

    - ``available_at`` > ``decision_date - embargo_days`` (the value was not
      knowable by the embargoed boundary): ``ingestion_lag`` (or ``embargo_zone``
      if it WAS knowable by decision_date but falls inside the embargo purge
      window), or
    - ``value_asof`` and ``value_current`` are both present and differ by more
      than ``restatement_tol``: ``restatement``.

    ``embargo_days`` is the required purge before the decision (0 = only require
    availability by the decision date itself).
    """
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    boundary_offset = timedelta(days=embargo_days)
    violations: list[EmbargoViolation] = []

    for u in timeline.uses:
        boundary = u.decision_date - boundary_offset
        if u.available_at > boundary:
            # not knowable by the embargoed boundary
            if u.available_at > u.decision_date:
                kind = "ingestion_lag"     # not even knowable by decision date
                lag = (u.available_at - u.decision_date).days
            else:
                kind = "embargo_zone"      # knowable, but inside the purge window
                lag = (u.available_at - boundary).days
            violations.append(
                EmbargoViolation(
                    name=u.name,
                    decision_date=u.decision_date,
                    kind=kind,
                    lag_days=int(lag),
                    detail={
                        "available_at": u.available_at.isoformat(),
                        "boundary": boundary.isoformat(),
                        "embargo_days": embargo_days,
                    },
                )
            )

        if (
            u.value_asof is not None
            and u.value_current is not None
            and abs(u.value_current - u.value_asof) > restatement_tol
        ):
            violations.append(
                EmbargoViolation(
                    name=u.name,
                    decision_date=u.decision_date,
                    kind="restatement",
                    lag_days=0,
                    detail={
                        "value_asof": u.value_asof,
                        "value_current": u.value_current,
                        "delta": u.value_current - u.value_asof,
                    },
                )
            )

    return violations
