"""Lookahead-Leakage Auditor.

A probe harness for detecting *temporal information leakage* in backtests of
LLM / agentic trading strategies, and estimating how much of a reported
backtest "alpha" could be memorized future knowledge rather than a genuine
edge available in real time.

WHAT THIS IS
------------
A backtest of an LLM strategy asks a model, at each simulated decision date
``t``, to act using only information available up to ``t``. *Temporal leakage*
is any channel by which the model's decision is influenced by knowledge of what
happened *after* ``t`` -- most importantly, facts memorized during pretraining
(the model "already knows" the 2020 crash, an earnings surprise, a merger), and
secondarily un-embargoed features whose values were revised after the fact.

The Auditor does not read the model's mind. It runs *paired probes* that hold
everything constant except the temporal-information channel, measures the
response delta, and turns those deltas into two numbers:

- a **leakage score** in ``[0, 1]`` (an effect-size / detectability summary), and
- an **alpha-deflation estimate**: the fraction (and, given a reported gross
  alpha, the amount) of backtest performance attributable to the leaked
  channel, with a bootstrap confidence interval.

WHAT IS REAL vs HELD (read this honestly)
-----------------------------------------
REAL and runnable now, with zero external LLM and zero paid data:

- ``probes``     -- the paired-probe *design* (fake-date, entity-mask,
                    embargo/ingestion-lag conditions) as pure data structures.
- ``estimator``  -- the statistics that turn probe deltas into a leakage score
                    and a deflation estimate with a paired bootstrap CI. This is
                    the mathematically load-bearing part and is fully tested.
- ``synthetic``  -- a *known-leakage* stand-in "model" whose leakage is injected
                    by construction, so the estimator can be validated end to
                    end (it must recover the injected leakage within CI).
- ``embargo``    -- ingestion-lag / embargo auditing over a feature timeline
                    (pure date logic; catches point-in-time violations).
- ``audit``      -- the orchestrator that runs the harness against any model
                    adapter and emits a ``LeakageReport``.

HELD, and cannot be honestly faked here:

- ``LiveLLMAdapter`` -- the adapter that actually queries a real LLM / agent
  once per probe. Its interface is shipped and the synthetic adapter proves the
  pipeline; the live implementation is a stub that raises with EXACTLY what it
  needs (an API key / local model, a per-call budget, and a decision-extraction
  contract). See ``adapters.LiveLLMAdapter`` and ``HELD.md``.

MATH PROVENANCE (prior, not novel)
----------------------------------
The estimator composes standard tools: a paired-difference effect size
(Cohen's d_z on matched pairs), a paired bootstrap CI (Efron & Tibshirani,
1993), and a permutation/sign test for the null of "no date sensitivity". The
counterfactual-date probe is the classic manipulation-check / placebo design
from causal inference. None of these are novel; the contribution is the shipped
harness that applies them to LLM-backtest leakage, plus the named metrics
(leakage score, alpha deflation). Numbers are scope-relative to the probe set
and the model under test; nothing here is a universal "LLM backtests overstate
alpha by X" claim.
"""

from .probes import (
    Probe,
    ProbeKind,
    ProbePair,
    ProbeResponse,
    make_fake_date_pairs,
    make_entity_mask_pairs,
)
from .estimator import (
    DeflationEstimate,
    LeakageScore,
    leakage_score,
    alpha_deflation,
)
from .embargo import (
    FeatureTimeline,
    EmbargoViolation,
    audit_embargo,
)
from .adapters import ModelAdapter, LiveLLMAdapter, HeldError
from .synthetic import SyntheticLeakyModel
from .audit import LeakageReport, run_audit

__all__ = [
    "Probe",
    "ProbeKind",
    "ProbePair",
    "ProbeResponse",
    "make_fake_date_pairs",
    "make_entity_mask_pairs",
    "DeflationEstimate",
    "LeakageScore",
    "leakage_score",
    "alpha_deflation",
    "FeatureTimeline",
    "EmbargoViolation",
    "audit_embargo",
    "ModelAdapter",
    "LiveLLMAdapter",
    "HeldError",
    "SyntheticLeakyModel",
    "LeakageReport",
    "run_audit",
]
