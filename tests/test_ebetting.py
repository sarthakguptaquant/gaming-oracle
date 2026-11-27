"""Batch GREE/GREL/GREM e-processes agree with the plain single-path loop."""
import numpy as np

from gaming_oracle.ebacktest import gaussian_es_var_ratio
from gaming_oracle.ebetting import eprocess_adaptive, eprocess_adaptive_batch


def test_batch_matches_loop():
    rng = np.random.default_rng(7)
    n_paths, n = 6, 80
    sig = 0.01 * np.exp(0.3 * rng.standard_normal((n_paths, n)))
    loss = sig * rng.standard_t(4, size=(n_paths, n))
    z = 2.3263 * sig * 0.8                       # understated VaR
    r = gaussian_es_var_ratio(0.01) * z
    b = eprocess_adaptive_batch(loss, z, r)
    for i in range(n_paths):
        one = eprocess_adaptive(loss[i], z[i], r[i])
        for k in ("GREE", "GREL", "GREM"):
            assert one[k]["cross"] == int(b[k]["cross"][i])
            assert np.isclose(one[k]["max"], b[k]["max"][i], rtol=1e-9)


def test_no_breach_never_bets():
    loss = np.full((2, 30), -0.01)
    z = np.full((2, 30), 0.02)
    out = eprocess_adaptive_batch(loss, z, 1.2 * z)
    for k in ("GREE", "GREL", "GREM"):
        assert np.all(out[k]["max"] == 1.0)
        assert np.all(out[k]["cross"] == -1)
