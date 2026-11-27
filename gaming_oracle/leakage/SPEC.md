# Lookahead-Leakage Auditor — design spec

Status: **validated design + runnable MVP.** Everything below marked REAL is
implemented and tested; the one HELD piece is `LiveLLMAdapter` (`HELD.md`).

## 1. Problem

A backtest of an LLM / agentic trading strategy queries a model at each
simulated decision date `t` and expects it to act on information available at
`t`. Two temporal-leakage channels break this:

1. **Memorization** (model side): the model's pretraining includes facts dated
   after `t` (the crash, the earnings surprise, the merger). The "decision"
   partly recalls the outcome.
2. **Non-point-in-time data** (data side): a feature fed at `t` carries a value
   that was revised, restated, or backfilled after `t` (ingestion lag,
   restatement).

The auditor quantifies both and estimates their contribution to reported alpha.

## 2. Estimands (what the numbers mean)

Let a probe pair `i` have factual signal `s^f_i`, counterfactual signal
`s^c_i`, and true forward return `r_i`. Signals lie in `[-1, 1]`.

- **Paired delta** `d_i = s^f_i - s^c_i`.
- **Leakage score** = `min(mean_i |d_i| / 2, 1)`, gated to 0 unless a
  permutation test rejects insensitivity (Section 3). Units: fraction of the
  maximum possible decision swing. REAL.
- **Factual alpha** `A_f = mean_i (s^f_i · r_i)`; **counterfactual alpha**
  `A_c = mean_i (s^c_i · r_i)`.
- **Alpha deflation fraction** `phi = (A_f - A_c) / A_f`: the share of factual
  alpha that vanishes when the leak channel is removed. Undefined (flagged
  `degenerate`) when `|A_f|` < tol. REAL.
- **Deflated amount** = `reported_gross_alpha · phi` (report's own units). REAL.

## 3. Nulls and tests (prior statistics)

- **Directional null** H0: the paired delta has no consistent direction.
  Two-sided **sign-flip permutation** on `d` (exact enumeration for n<=18, Monte
  Carlo otherwise). Catches a fixed-direction leak. REAL.
- **Alpha-aligned null** H0: the delta is uncorrelated with the outcome.
  Sign-flip permutation on `g_i = d_i · sign(r_i)`. This is the load-bearing
  null: a magnitude leak that flips sign with `r_i` has `mean(d) ~ 0` (passes
  the directional test) yet `mean(g) > 0`, and it is exactly the leak that
  inflates alpha. The score gates on **either** null rejecting. REAL, and the
  reason the naive directional-only design (an earlier iteration) was wrong.
- **Deflation CI**: paired (matched-pairs) **bootstrap** over probes; the CI on
  `phi` excludes 0 when the leak materially moves alpha. REAL.
- **Effect size**: Cohen's `d_z = mean(d)/sd(d)`, reported for interpretability.
  REAL.

Provenance: paired bootstrap (Efron & Tibshirani 1993); sign-flip permutation
(standard nonparametric); `d_z` (Cohen 1988); the counterfactual-date/name probe
is the placebo / manipulation-check design from causal inference. No novel
theorem.

## 4. Verdict rule (coarse, scope-relative)

- `leaking`: score significant AND deflation CI strictly above 0.
- `suspect`: exactly one of those holds.
- `clean`: neither. All relative to THIS probe set and model; never universal.

## 5. Validation (why the numbers can be trusted)

The estimator is checked against a synthetic model whose leak is injected by
construction (`synthetic.py`). With honest edge 0 and injected gain `g`:
`A_f = g · mean|r|`, `A_c = 0`, so `phi -> 1`; the tests assert the reported
`phi` CI excludes 0 and lands in `[0.8, 1.2]`, and the mechanistic identity
`A_f = g · mean|r|` holds exactly. With a genuine momentum edge AND a leak, `phi`
must land strictly in `(0, 1)` with `A_c > 0` (real edge survives). A leak-free
model must score 0 with a deflation CI straddling 0. All in
`tests/leakage/test_estimator.py`.

## 6. Embargo audit (data-side, model-free)

`audit_embargo` over a `FeatureTimeline` flags: `ingestion_lag` (value not
knowable by the decision date), `embargo_zone` (knowable but inside a required
purge window, PurgedKFold-style, cf. López de Prado 2018), and `restatement`
(as-of vs current value differ beyond tolerance). Pure date logic, REAL, tested.

## 7. Threat model — what this DOES and DOES NOT cover

Covers:
- Date-keyed and name-keyed memorization detectable by a decision shift on a
  falsified arm.
- Non-point-in-time feature values, structurally.

Does NOT cover (honest limits):
- **Leakage with zero decision shift.** If a model has memorized the future but
  its output is invariant to the date/name falsification (e.g. it re-derives the
  same answer), the paired probe sees no delta and reports clean. The auditor
  detects *exploited, channel-sensitive* leakage, not latent knowledge.
- **Context that betrays the era.** If the probe context itself contains
  post-hoc numbers (prices, index levels), the fake-date arm is not truly
  counterfactual. The probe builder keeps context date-neutral; a real corpus
  must be scrubbed the same way (noted in `HELD.md`).
- **Sampling noise** with a non-deterministic model raises the estimator's noise
  floor; the live path must use temperature 0 or K-draw averaging (`HELD.md`).
- **Refusal asymmetry** in a live model can masquerade as a delta; the live
  adapter must decide how refusals map to a signal (`HELD.md`).

## 8. Real vs held (one table)

| Component | Status |
|---|---|
| probes, estimator, synthetic model, embargo, orchestrator, CLI, 25 tests | REAL |
| `LiveLLMAdapter` (queries a real LLM per probe) | HELD — 4 requirements in `HELD.md` |
