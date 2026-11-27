"""Smoke test for the publication figures: the pipeline runs and writes 4 PNGs,
and the numbers behind figure (c) satisfy the attack-vs-defense contract.

We do not pixel-compare (matplotlib output is platform-sensitive); we assert the
files exist, are non-trivially sized, and that the underlying lie-budget numbers
match the gate/solver pipeline (defended <= undefended on every asset).
"""

import os

import pytest

matplotlib = pytest.importorskip("matplotlib")

from gaming_oracle import figures as figs  # noqa: E402


def test_make_all_writes_four_pngs(tmp_path, monkeypatch):
    # redirect the figure dir to a temp dir so the test is hermetic
    monkeypatch.setattr(figs, "FIG_DIR", str(tmp_path))
    out = figs.make_all()
    # (d) is the Cycle-7 tolerance_map figure; assert it too so a regression that
    # drops it fails red instead of passing green.
    for key in ("a", "b", "c", "d"):
        p = out[key]["path"]
        assert os.path.exists(p), p
        assert os.path.getsize(p) > 10_000, p  # a real chart, not an empty canvas


def test_attack_vs_defense_numbers_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(figs, "FIG_DIR", str(tmp_path))
    data = figs._load_all()
    res = figs.fig_attack_vs_defense(data)["results"]
    for asset, r in res.items():
        # defended lie never exceeds undefended lie
        assert r["mulE"] <= r["mul4"] + 1e-6, asset
        # the gamed 4-gate series drives the e-process above the reject threshold
        assert r["traj"].max() >= r["threshold"], asset
