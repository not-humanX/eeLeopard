import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eeleopard import GF2m, LCHBasis  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def random_basis(gf: GF2m, rng) -> list[int]:
    """A random GF(2)-basis of GF(2^m) (rejection sampling)."""
    while True:
        b = [int(x) for x in rng.integers(1, gf.q, gf.m)]
        try:
            LCHBasis(gf, b)
            return b
        except ValueError:
            continue


def corrupt(code, cw, v, f, rng, erasure_pool=None):
    """Add v random non-zero errors and f erasures (garbage values) to cw.
    Returns (received, error_positions, erasure_positions)."""
    n = cw.size
    if erasure_pool is None:
        pos = rng.choice(n, v + f, replace=False)
        epos, fpos = pos[:v], pos[v:]
    else:
        fpos = rng.choice(erasure_pool, f, replace=False)
        rest = np.setdiff1d(np.arange(n), fpos)
        epos = rng.choice(rest, v, replace=False)
    rx = cw.copy()
    rx[epos] ^= rng.integers(1, code.gf.q, v).astype(rx.dtype)
    rx[fpos] = rng.integers(0, code.gf.q, f).astype(rx.dtype)
    return rx, np.sort(epos), np.sort(fpos)
