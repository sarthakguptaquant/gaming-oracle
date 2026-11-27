"""Runnable demo of the Lookahead-Leakage Auditor.

This exercises the FULL real pipeline against the synthetic known-leakage model
(no external LLM, no paid data, no network). It builds a probe set, runs the
fake-date and entity-mask audits at a chosen leak strength, and prints the
leakage score and the alpha-deflation estimate with its bootstrap CI. It also
runs the embargo auditor on a small feature timeline.

    uv run python -m gaming_oracle.leakage.cli               # default demo
    uv run python -m gaming_oracle.leakage.cli --gain 0.0    # a clean model
    uv run python -m gaming_oracle.leakage.cli --gain 0.6 --reported-alpha 250

The live-LLM path is HELD (see HELD.md); pass --live to see the exact HeldError.
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np

from .adapters import HeldError, LiveLLMAdapter
from .audit import run_audit
from .embargo import FeatureTimeline, FeatureUse, audit_embargo
from .probes import Probe, make_entity_mask_pairs, make_fake_date_pairs
from .synthetic import SyntheticLeakyModel


def _build_probes(n: int, honest_edge: float, seed: int) -> list[Probe]:
    """n one-entity probes; momentum predicts the return with strength
    ``honest_edge`` (the genuine, leak-free component)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        mom = float(rng.normal(0, 0.3))
        r = honest_edge * mom * 0.02 + float(rng.normal(0, 0.012))
        ctx = (f"As-of note for Apple (AAPL). Trailing momentum mom={mom:.3f}. "
               f"Decide a position in [-1,1].")
        out.append(Probe(f"p{i:03d}",
                         date(2019 + i % 2, 1 + i % 12, 1 + i % 27), ctx, r))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gain", type=float, default=0.5,
                    help="injected leak strength (0 = honest model)")
    ap.add_argument("--honest-edge", type=float, default=0.6,
                    help="genuine momentum edge (leak-free alpha source)")
    ap.add_argument("--n", type=int, default=120, help="number of probes")
    ap.add_argument("--reported-alpha", type=float, default=None,
                    help="a reported gross alpha to deflate (report units)")
    ap.add_argument("--shift-years", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--live", action="store_true",
                    help="attempt the HELD live-LLM adapter (shows HeldError)")
    args = ap.parse_args(argv)

    print("== Lookahead-Leakage Auditor (synthetic-model demo) ==")
    print("REAL: probes + estimator + embargo + synthetic known-leakage model.")
    print("HELD: the live-LLM adapter (needs an external model; see HELD.md).")
    print("Math is prior (paired bootstrap, sign-flip permutation, Cohen's d_z).")
    print("Numbers are scope-relative to THIS probe set and THIS model.\n")

    probes = _build_probes(args.n, args.honest_edge, args.seed)

    if args.live:
        print("-- live adapter (HELD) --")
        try:
            LiveLLMAdapter().decide(probes[0])
        except HeldError as e:
            print(f"HeldError: {e}\n")

    model = SyntheticLeakyModel(gain=args.gain, honest_scale=1.0,
                                remembered_years=(2018, 2021))

    fd = make_fake_date_pairs(probes, shift_years=args.shift_years)
    em = make_entity_mask_pairs(probes, entity="Apple", ticker="AAPL")

    for name, pairs in (("FAKE-DATE", fd), ("ENTITY-MASK", em)):
        rep = run_audit(pairs, model, reported_gross_alpha=args.reported_alpha,
                        seed=args.seed)
        print(f"-- {name} --")
        print("  " + rep.summary())
        print(f"  score p_dir={rep.score.pvalue:.3g} "
              f"p_aligned={rep.score.pvalue_aligned:.3g} "
              f"mean|delta|={rep.score.mean_abs_delta:.3f}")
        print(f"  factual_alpha={rep.deflation.factual_alpha:+.5f} "
              f"cf_alpha={rep.deflation.counterfactual_alpha:+.5f}\n")

    # embargo demo
    tl = FeatureTimeline()
    tl.add(FeatureUse("eps", date(2020, 6, 10), available_at=date(2020, 6, 12)))
    tl.add(FeatureUse("revenue", date(2020, 6, 10), available_at=date(2020, 6, 1),
                      value_asof=100.0, value_current=112.0))
    tl.add(FeatureUse("pe", date(2020, 6, 10), available_at=date(2020, 6, 8)))
    v = audit_embargo(tl, embargo_days=1)
    print(f"-- EMBARGO AUDIT -- {len(v)} violation(s):")
    for x in v:
        print(f"  {x.name}: {x.kind} (lag={x.lag_days}d)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
