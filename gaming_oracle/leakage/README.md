# Lookahead-Leakage Auditor

A probe harness for detecting **temporal information leakage** in backtests of
LLM / agentic trading strategies, and estimating how much of a reported backtest
"alpha" could be **memorized future knowledge** rather than a real-time edge.

This is a sibling to the Gaming Oracle. Gaming Oracle games the *statistical*
backtest panel (Kupiec / Christoffersen / Basel); the Auditor attacks a
different failure mode entirely: an LLM strategy that scores well because the
model already *knows the future* (from pretraining) or was fed features that
were not point-in-time-valid.

## The idea in one paragraph

At each simulated decision date `t`, an honest strategy uses only information
available at `t`. Leakage = the decision is influenced by what happened *after*
`t`. You cannot read the model's mind, so instead you run **paired probes**:
two decision contexts identical except for one temporal channel, and measure how
much the decision moves. If falsifying the date (or masking the entity name)
changes the decision *in a way that helps the backtest pnl*, that movement is
leaked future knowledge, and its contribution to alpha can be priced.

## What it emits

- **Leakage score** `[0, 1]`: significance-gated normalized decision movement.
- **Alpha deflation**: the fraction of factual alpha that disappears when the
  model is forced onto the leak-removed (counterfactual) arm, with a paired
  bootstrap CI; scaled to a reported gross alpha if you give one.
- **Verdict**: `clean | suspect | leaking`, scope-relative to the probe set and
  model (never a universal claim).
- **Embargo violations**: structural point-in-time audit of a feature timeline
  (ingestion lag, restatement/backfill, embargo purge zone).

## Probe kinds

| Kind | Falsifies | Catches |
|---|---|---|
| `FAKE_DATE` | the stated as-of date (back-dated, same context) | date-keyed memorization ("it knows what 2020 does next") |
| `ENTITY_MASK` | the entity name (real -> "Company X") | name-keyed memorization of outcomes |
| `EMBARGO` | as-revised vs point-in-time feature value | data-side lookahead (restated / backfilled features) |

## Run it

```bash
# full pipeline against the synthetic known-leakage model (no LLM, no network)
uv run python -m gaming_oracle.leakage.cli --gain 0.5 --reported-alpha 250

# a clean (leak-free) model
uv run python -m gaming_oracle.leakage.cli --gain 0.0

# see the HELD live-LLM adapter's exact requirements
uv run python -m gaming_oracle.leakage.cli --live

# tests (25, all offline)
uv run pytest tests/leakage/ -q
```

## What is real vs held

Real and tested: the probes, the estimator (paired bootstrap + sign-flip
permutation + Cohen's d_z), the known-leakage synthetic model that validates the
estimator, the embargo auditor, the orchestrator. **Held:** the single adapter
that queries a real LLM per probe — see [`HELD.md`](HELD.md) for the exact four
requirements to lift the hold.

## Honesty notes (same discipline as the parent repo)

- **The math is prior.** Paired bootstrap (Efron & Tibshirani 1993), sign-flip
  permutation test, Cohen's d_z, and the placebo / manipulation-check design
  from causal inference. No novel theorem. The contribution is the shipped
  harness + the two named metrics.
- **Every number is scope-relative** to (a) the probe set and (b) the model
  under test. There is no universal "LLM backtests overstate alpha by X" claim.
- **The synthetic model is a validation stand-in, not an LLM.** It exists so the
  estimator can be shown to recover a *known* leak; it makes no claim about any
  real model's behavior.
- **The estimator gates on the alpha-aligned null**, not just a directional one:
  a magnitude leak whose sign flips with the outcome (invisible to a naive
  "did the decision drift one way" test) is exactly the leak that inflates
  alpha, and it is the one the score is built to catch. See `estimator.py`
  `LeakageScore`.

## Design spec

The validated design spec is `SPEC.md` (estimands, nulls, guarantees, and the
threat model the harness does and does not cover).
