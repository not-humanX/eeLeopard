"""decode(..., errors=False): erasure-only decoding that skips the key-equation
solver and the root search (lambda = 1), O(n log n)."""
import numpy as np
import pytest

import eeleopard
from eeleopard import FFTRSCode, GF2m, ReedSolomon, DecodeFailure, HAVE_NUMBA, set_numba, numba_enabled
from conftest import random_basis

PARAMS = [(3, 1), (4, 2), (5, 3), (8, 3), (8, 5), (8, 7), (10, 6), (12, 8)]


@pytest.fixture(autouse=True)
def restore_switch():
    before = numba_enabled()
    yield
    set_numba(before)


def erase(code, cw, f, rng, where="any"):
    """Overwrite f positions of cw with garbage; return (received, positions)."""
    n, d = code.n, code.d
    pool = {"any": np.arange(n), "parity": np.arange(d), "message": np.arange(d, n)}[where]
    f = min(f, pool.size)
    pos = rng.choice(pool, f, replace=False)
    rx = cw.copy()
    rx[pos] = rng.integers(0, n, f)
    return rx, pos


@pytest.mark.parametrize("m,t", PARAMS)
@pytest.mark.parametrize("std_basis", [True, False])
def test_recovers_every_erasure_count(m, t, std_basis, rng):
    basis = None if std_basis else random_basis(GF2m(m), rng)
    code = FFTRSCode(m, t, basis=basis)
    cw = code.encode(rng.integers(0, code.n, code.k))
    d = code.d
    counts = sorted({0, 1, 2, d // 3, d // 2, d - 1, d})
    for f in counts:
        for where in ("any", "parity", "message"):
            rx, pos = erase(code, cw, f, rng, where)
            res = code.decode(rx, pos, errors=False)
            np.testing.assert_array_equal(res.codeword, cw)
            assert res.num_errors == 0
            assert res.solver == "erasure-only"
            assert set(res.erasure_positions.tolist()) == set(pos.tolist())
            # the erasure values are the corrections
            np.testing.assert_array_equal(res.erasure_values, (rx ^ cw)[res.erasure_positions])


@pytest.mark.parametrize("m,t", PARAMS)
def test_matches_full_decoder(m, t, rng):
    code = FFTRSCode(m, t)
    cw = code.encode(rng.integers(0, code.n, code.k))
    for f in (0, 1, code.d // 2, code.d):
        rx, pos = erase(code, cw, f, rng)
        a = code.decode(rx, pos, errors=False)
        b = code.decode(rx, pos)
        np.testing.assert_array_equal(a.codeword, b.codeword)
        np.testing.assert_array_equal(a.erasure_values, b.erasure_values)


@pytest.mark.parametrize("m,t", PARAMS)
def test_errors_within_distance_are_always_detected(m, t, rng):
    """v errors + f erasures with v + f <= n - k: a wrong erasure-only result would be
    a second codeword within distance v + f < d_min = n - k + 1 of the first, so the
    decoder must raise."""
    code = FFTRSCode(m, t)
    n, d = code.n, code.d
    cw = code.encode(rng.integers(0, n, code.k))
    for trial in range(40):
        v = int(rng.integers(1, d + 1))
        f = int(rng.integers(0, d - v + 1))
        pos = rng.choice(n, v + f, replace=False)
        rx = cw.copy()
        rx[pos[:v]] ^= rng.integers(1, n, v).astype(rx.dtype)
        rx[pos[v:]] = rng.integers(0, n, f)
        with pytest.raises(DecodeFailure):
            code.decode(rx, pos[v:], errors=False)


def test_output_is_always_a_codeword(rng):
    """Far beyond the radius the decoder may fail or alias, but never returns a non-codeword,
    and what it returns agrees with the received word off the erasures."""
    code = FFTRSCode(6, 3)
    n, d = code.n, code.d
    ok = fail = 0
    for trial in range(300):
        rx = rng.integers(0, n, n).astype(code.gf.dtype)
        f = int(rng.integers(0, d + 1))
        er = rng.choice(n, f, replace=False)
        try:
            res = code.decode(rx, er, errors=False)
        except DecodeFailure:
            fail += 1
            continue
        ok += 1
        assert not code.syndrome(res.codeword).any()
        keep = np.setdiff1d(np.arange(n), er)
        np.testing.assert_array_equal(res.codeword[keep], rx[keep])
    assert fail > 0


def test_zero_erasures(rng):
    code = FFTRSCode(8, 4)
    cw = code.encode(rng.integers(0, 256, code.k))
    res = code.decode(cw, errors=False)
    np.testing.assert_array_equal(res.codeword, cw)
    bad = cw.copy()
    bad[17] ^= 9
    with pytest.raises(DecodeFailure):
        code.decode(bad, errors=False)
    assert code.decode(bad).num_errors == 1            # the default mode still corrects it


def test_too_many_erasures(rng):
    code = FFTRSCode(6, 3)
    cw = code.encode(rng.integers(0, 64, code.k))
    with pytest.raises(DecodeFailure):
        code.decode(cw, np.arange(code.d + 1), errors=False)


def test_error_solver_and_root_search_are_skipped(rng, monkeypatch):
    import eeleopard.rs as rs_mod

    def boom(*a, **k):
        raise AssertionError("must not be called in erasure-only mode")

    monkeypatch.setattr(rs_mod, "solve_gke_fast", boom)
    monkeypatch.setattr(rs_mod, "solve_gke_euclid", boom)
    code = FFTRSCode(8, 5)
    monkeypatch.setattr(code.lch, "evaluate_everywhere", boom)
    monkeypatch.setattr(code.lch, "mul", boom)
    cw = code.encode(rng.integers(0, 256, code.k))
    rx, pos = erase(code, cw, code.d, rng)
    np.testing.assert_array_equal(code.decode(rx, pos, errors=False).codeword, cw)
    with pytest.raises(AssertionError):                 # sanity check of the monkeypatch itself
        code.decode(rx, pos)


def test_timings_have_no_error_steps(rng):
    code = FFTRSCode(8, 5)
    cw = code.encode(rng.integers(0, 256, code.k))
    rx, pos = erase(code, cw, 10, rng)
    tm = code.decode(rx, pos, errors=False).timings
    assert tm["step3_roots"] < 1e-3 and tm["step2_key_equation"] < 1e-3


@pytest.mark.parametrize("n,k", [(255, 223), (204, 188), (528, 514), (100, 60), (30, 28)])
def test_reed_solomon_any_n_k(n, k, rng):
    rs = ReedSolomon(n, k)
    data = rng.integers(0, rs.gf.q, k)
    cw = rs.encode(data)
    for f in sorted({0, 1, (n - k) // 2, n - k}):
        pos = rng.choice(n, f, replace=False)
        rx = cw.copy()
        rx[pos] = rng.integers(0, rs.gf.q, f)
        res = rs.decode(rx, pos, errors=False)
        np.testing.assert_array_equal(res.codeword, cw)
        np.testing.assert_array_equal(res.message, data)
    if n - k >= 2:                                      # an error outside the erasures is detected
        pos = rng.choice(n, n - k, replace=False)       # v + f = n - k: within the detection guarantee
        rx = cw.copy()
        rx[pos[0]] ^= 3
        with pytest.raises(DecodeFailure):
            rs.decode(rx, pos[1:], errors=False)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba is not installed")
@pytest.mark.parametrize("m,t", [(8, 5), (12, 8)])
def test_numba_and_numpy_agree(m, t, rng):
    code = FFTRSCode(m, t)
    cw = code.encode(rng.integers(0, code.n, code.k))
    rx, pos = erase(code, cw, code.d, rng)
    set_numba(False)
    a = code.decode(rx, pos, errors=False)
    set_numba(True)
    b = code.decode(rx, pos, errors=False)
    np.testing.assert_array_equal(a.codeword, b.codeword)
    np.testing.assert_array_equal(a.erasure_values, b.erasure_values)
    np.testing.assert_array_equal(a.codeword, cw)
