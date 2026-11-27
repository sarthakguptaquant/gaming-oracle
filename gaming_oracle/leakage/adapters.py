"""Model adapters: the boundary between the harness and the thing being tested.

The harness (``audit.run_audit``) is model-agnostic. It hands each probe to a
``ModelAdapter`` and gets back a ``ProbeResponse`` (a numeric signal in
``[-1, 1]``). Everything upstream and downstream is real and tested with the
synthetic adapter.

``LiveLLMAdapter`` is the ONE piece that is HELD: it is the adapter that would
query a real LLM / agent once per probe. It cannot be honestly faked without an
actual model, so it is a stub that raises :class:`HeldError` describing EXACTLY
what is required to make it real. See ``HELD.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .probes import Probe, ProbeResponse


@runtime_checkable
class ModelAdapter(Protocol):
    """Anything that turns a probe into a numeric decision signal.

    Implementations must be PURE with respect to the probe: the signal may
    depend only on ``probe.context`` and ``probe.asof`` (the information the
    model is allowed to see), never on ``probe.forward_return`` (auditor-only
    ground truth). Violating this makes the audit meaningless.
    """

    def decide(self, probe: Probe) -> ProbeResponse:
        ...


class HeldError(NotImplementedError):
    """Raised by parts that are deliberately HELD (need a live LLM / paid data).

    This is not a bug. It is the honest boundary of the runnable MVP.
    """


class LiveLLMAdapter:
    """HELD. Queries a real LLM / agent once per probe.

    To make this real you must supply, at minimum:

    1. A MODEL ENDPOINT: an Anthropic / OpenAI-style API key, or a local model
       (e.g. an Ollama / vLLM server). The Auditor deliberately ships none, and
       the guardrails on this build forbid network calls.
    2. A DECISION CONTRACT: a prompt template that instructs the model to return
       a single directional signal in [-1, 1] (short..long) for the probe's
       context, plus a parser that extracts that number from the model text
       robustly (refusals, hedges, and formatting drift must map to a defined
       value, not crash).
    3. A PER-CALL BUDGET + RATE POLICY: token/cost cap and backoff, because a
       real audit runs O(pairs x arms) calls and paired arms should ideally
       share a cache-friendly prefix.
    4. DETERMINISM CONTROL: temperature 0 (or fixed seed) so the paired delta
       reflects the probe change, not sampling noise. With sampling on, each arm
       must be averaged over K draws and the estimator's noise floor rises.

    Until those are provided, ``decide`` raises :class:`HeldError` with this
    checklist. The synthetic adapter (``SyntheticLeakyModel``) proves the full
    pipeline in the meantime.
    """

    def __init__(self, *_, **__):
        # Intentionally accept and ignore any config so callers can wire it up
        # and get a clear HELD error at call time rather than construction time.
        self._config = {"args": _, "kwargs": __}

    def decide(self, probe: Probe) -> ProbeResponse:  # noqa: D401
        raise HeldError(
            "LiveLLMAdapter is HELD. To run against a real model, supply: "
            "(1) a model endpoint/API key or local model; "
            "(2) a decision-contract prompt + robust [-1,1] signal parser; "
            "(3) a per-call token/cost budget with backoff; "
            "(4) temperature 0 or K-draw averaging for determinism. "
            "See gaming_oracle/leakage/HELD.md. The synthetic adapter proves "
            "the pipeline without a live model."
        )
