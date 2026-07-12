# Gaming Oracle

Given the exact panel of backtests a risk-model validator runs, this tool
constructs the most risk-understating Value-at-Risk (VaR) forecast series that
still passes *every* test jointly, and prices the gap. The headline metric is the
**Maximum Undetectable Lie (MUL)**: the largest VaR understatement that evades a
fixed validation panel, expressed three ways, in basis points of the honest
reference, in dollars on a stated notional, and in additional exceptions
tolerated per year.

The panel spans the standard regulatory backtests (Kupiec proportion-of-failures,
Christoffersen independence and conditional coverage, and the Basel traffic-light
exception count), an author-added split-conformal coverage gate, a one-sided
coverage score test, and the anytime-valid e-process Expected Shortfall backtest
of Wang, Wang and Ziegel (2025). Two solvers exhibit and then certify the lie: a
greedy projected search that exhibits a feasible lie, and a mixed-integer program
that certifies the exact maximum over the linear (count and window) gates, with
the one nonlinear gate (Christoffersen independence) verified post hoc.

Every reported number is reproducible from the cached returns under `data_cache/`;
no network access or paid data is required.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Reproduce

```bash
# Headline: most-recent 250-day AAPL window at the 99% level, rolling-GARCH reference
uv run python -m gaming_oracle.cli --headline

# Per-asset representative windows, tolerance map, and the e-aware defence residual
uv run python -m gaming_oracle.cli --cycle 7

# Uniform vs greedy vs exact (MILP) comparison
uv run python -m gaming_oracle.cli --cycle 3 --method rolling_garch

# Rolling-window MUL distribution over definable windows
uv run python -m gaming_oracle.cli --cycle 2 --method rolling_garch

# Regenerate the figures into figures/
uv run python -m gaming_oracle.figures
```

## Tests

```bash
uv run pytest
```

## Also in this package

`gaming_oracle/leakage/` is a separate lookahead-leakage auditor for LLM and
agentic trading strategies, using fake-date, entity-mask, and embargo probes to
estimate how much of a reported backtest is an artifact of the model having seen
the future. See `gaming_oracle/leakage/README.md`. Its live-model adapter is
intentionally held; the synthetic validation path runs fully offline.

## Scope

Every backtest used here is prior work, cited in the accompanying paper, and is
never claimed as novel. The contribution is the constructive solver, the named
and priced metric, the tolerance map, and the measured e-aware residual. Every
result is panel-relative and window-conditioned: a lie is "undetectable" only
against the specific panel evaluated, on the specific window reported, and is
never claimed as a property of an asset in general.

## License

MIT. See [LICENSE](LICENSE).
