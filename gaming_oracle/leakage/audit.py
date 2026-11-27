"""Audit orchestrator: run the probe harness against a model, emit a report.

``run_audit`` is model-agnostic: give it probe pairs, a ``ModelAdapter``, and
optionally a reported gross alpha, and it returns a ``LeakageReport`` combining
the leakage score and the alpha-deflation estimate. It works identically with
the synthetic adapter (real, tested) and the HELD ``LiveLLMAdapter`` (which will
raise a clear ``HeldError`` when called) -- the only difference is which model
is behind the ``decide`` boundary.

VERDICT is a coarse, honest label, NOT a universal claim:
- "clean"     : score ~ 0 and deflation CI straddles / below 0.
- "suspect"   : score significant OR deflation CI strictly above 0.
- "leaking"   : score significant AND deflation CI strictly above 0.
The verdict is relative to THIS probe set and THIS model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .adapters import ModelAdapter
from .estimator import (
    DeflationEstimate,
    LeakageScore,
    alpha_deflation,
    leakage_score,
)
from .probes import ProbePair, ProbeResponse


@dataclass
class LeakageReport:
    """Combined output of an audit run."""

    verdict: str                      # "clean" | "suspect" | "leaking"
    score: LeakageScore
    deflation: DeflationEstimate
    n_pairs: int
    kind: str
    detail: dict = field(default_factory=dict)

    def summary(self) -> str:
        d = self.deflation
        frac = "n/a (degenerate)" if d.degenerate else f"{d.fraction:+.1%}"
        ci = (
            "n/a"
            if d.degenerate
            else f"[{d.ci_low:+.1%}, {d.ci_high:+.1%}]"
        )
        amt = ""
        if d.deflated_amount is not None:
            amt = f"  leaked_amount={d.deflated_amount:+.4g}"
        return (
            f"[{self.kind}] verdict={self.verdict.upper()}  "
            f"score={self.score.score:.3f} (p={self.score.pvalue:.3g})  "
            f"deflation={frac} CI{ci}{amt}  n={self.n_pairs}"
        )


def _run_pairs(
    pairs: list[ProbePair], model: ModelAdapter
) -> dict[str, ProbeResponse]:
    """Query the model on both arms of every pair."""
    responses: dict[str, ProbeResponse] = {}
    for pr in pairs:
        for probe in (pr.factual, pr.counterfactual):
            if probe.probe_id not in responses:
                responses[probe.probe_id] = model.decide(probe)
    return responses


def _verdict(score: LeakageScore, defl: DeflationEstimate) -> str:
    score_hit = score.significant and score.score > 0
    # deflation CI strictly above zero (leak-removal reduced alpha)
    defl_hit = (not defl.degenerate) and defl.ci_low > 0.0
    if score_hit and defl_hit:
        return "leaking"
    if score_hit or defl_hit:
        return "suspect"
    return "clean"


def run_audit(
    pairs: list[ProbePair],
    model: ModelAdapter,
    reported_gross_alpha: float | None = None,
    level: float = 0.95,
    seed: int = 0,
) -> LeakageReport:
    """Run the harness and return a ``LeakageReport``.

    ``pairs`` must share one ``ProbeKind`` (mixing kinds muddies the verdict);
    callers should audit each kind separately and compare.
    """
    if not pairs:
        raise ValueError("no probe pairs")
    kinds = {p.kind for p in pairs}
    if len(kinds) != 1:
        raise ValueError(f"audit one ProbeKind at a time; got {kinds}")
    kind = next(iter(kinds)).value

    responses = _run_pairs(pairs, model)
    score = leakage_score(pairs, responses, level=level, seed=seed)
    defl = alpha_deflation(
        pairs,
        responses,
        reported_gross_alpha=reported_gross_alpha,
        level=level,
        seed=seed,
    )
    verdict = _verdict(score, defl)
    return LeakageReport(
        verdict=verdict,
        score=score,
        deflation=defl,
        n_pairs=len(pairs),
        kind=kind,
        detail={
            "level": level,
            "note": "verdict is scope-relative to this probe set and model; "
                    "not a universal claim.",
        },
    )
