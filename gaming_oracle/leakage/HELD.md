# HELD: the live-LLM path

The Lookahead-Leakage Auditor ships a **complete, tested, runnable** pipeline
against a *synthetic* known-leakage model. Exactly one component is **HELD**: the
adapter that queries a **real** LLM / agent once per probe. It is held, not
faked, because it cannot be made honest without an actual model, and this build
is forbidden from making network calls. This file states precisely what is
needed to lift the hold.

## What is REAL (runnable now, no external model, no paid data, no network)

| Piece | File | Status |
|---|---|---|
| Paired probe design (fake-date, entity-mask, embargo) | `probes.py` | real, tested |
| Leakage-score + alpha-deflation estimator (paired bootstrap, sign-flip permutation, Cohen's d_z) | `estimator.py` | real, tested |
| Known-leakage synthetic model (validation ground truth) | `synthetic.py` | real, tested |
| Ingestion-lag / embargo / restatement auditor | `embargo.py` | real, tested |
| Audit orchestrator + `LeakageReport` | `audit.py` | real, tested |
| Demo runner | `cli.py` | real, runs |

The estimator is validated by *recovering an injected leak*: plant a known
`gain`, confirm the reported deflation fraction lands on the analytic value with
a bootstrap CI that excludes 0 (`tests/leakage/test_estimator.py`). If it could
not recover a leak we planted ourselves, no number on a real model would be
trustworthy.

## What is HELD (`adapters.LiveLLMAdapter`)

Calling `LiveLLMAdapter.decide(probe)` raises `HeldError`. To make it real,
supply all four:

1. **Model endpoint.** An Anthropic / OpenAI-style API key, or a local model
   (Ollama / vLLM). For Claude specifically, see the `claude-api` skill for the
   current model ids and params before wiring anything.
2. **Decision contract.** A prompt template instructing the model to return a
   single directional signal in `[-1, 1]` (short..long) for `probe.context`,
   plus a robust parser: refusals, hedges, and formatting drift must map to a
   defined value (e.g. 0.0 = no position), never crash. The parser is itself
   worth a small test set.
3. **Budget + rate policy.** A per-call token/cost cap and backoff. A real audit
   is `O(n_pairs x 2)` calls; the two arms of a pair share almost all of their
   prompt, so a cache-friendly common prefix cuts cost materially.
4. **Determinism control.** Temperature 0 (or a fixed seed) so the paired delta
   reflects the probe change, not sampling noise. If sampling is on, each arm
   must be averaged over K draws and the estimator's noise floor rises
   accordingly (raise `n_pairs` to compensate).

### Sharp edges the live path must handle (and the synthetic model cannot surface)

- **Refusal asymmetry.** A model may refuse the real-name arm but answer the
  masked arm (or vice versa). A refusal mapped to 0.0 is itself a delta and will
  register as leakage; decide deliberately whether that is signal or artifact.
- **Prompt-order / anchoring effects.** Present arms in randomized order across
  probes so position does not confound the delta.
- **Context that leaks the date implicitly.** The fake-date probe only works if
  the context does not itself contain post-hoc numbers (prices, indices) that
  betray the true era. The probe builder keeps context date-neutral; a live
  corpus must be scrubbed the same way.
- **Ground-truth hygiene.** `probe.forward_return` is auditor-only. Any adapter
  that reads it is invalid by construction (`ModelAdapter` docstring says so).

## One-line status

Real: the harness, the estimator, the validation, the embargo audit. Held: the
single live-model adapter, with the four requirements above. The synthetic
adapter proves the whole pipeline end to end in the meantime.
