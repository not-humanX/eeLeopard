"""The optional numba kernels must be bit-identical to the numpy code paths, and
the package must work (with numpy only) when numba is missing or disabled."""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from conftest import corrupt, random_basis
from eeleopard import FFTRSCode, GF2m, LCHBasis, ReedSolomon, HAVE_NUMBA, set_numba, numba_enabled
from eeleopard import keyeq
from eeleopard.lch import degree

ROOT = os.path.join(os.path.dirname(__file__), "..")

needs_numba = pytest.mark.skipif(not HAVE_NUMBA, reason="numba is not installed")


@pytest.fixture(autouse=True)
def restore_switch():
    before = numba_enabled()
    yield
    set_numba(before)


def both(fn):
    """Run fn() with the numpy paths and with the numba kernels; return both results."""
    set_numba(False)
    a = fn()
    assert not numba_enabled()
    set_numba(True)
    assert numba_enabled()
    b = fn()
    return a, b


def same(a, b):
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ------------------------------------------------------------------ lch.py
@needs_numba
@pytest.mark.parametrize("m", [2, 3, 5, 8, 11])
def test_fft_ifft_batched_offsets(m, rng):
    gf = GF2m(m)
    lch = LCHBasis(gf, random_basis(gf, rng))
    for r in range(m + 1):
        size = 1 << r
        A = rng.integers(0, gf.q, (5, size)).astype(gf.dtype)
        offs = rng.integers(0, gf.q, 5) & ~(size - 1)
        for inverse in (False, True):
            run = lambda: lch._transform(A, r, offs, inverse)
            same(*both(run))
        same(*both(lambda: lch.fft(A[0], r, int(offs[0]))))          # scalar offset, 1-D input


@needs_numba
def test_fft_does_not_modify_its_input(rng):
    gf = GF2m(8)
    lch = LCHBasis(gf)
    A = rng.integers(0, 256, 256).astype(gf.dtype)
    keep = A.copy()
    set_numba(True)
    lch.fft(A, 8)
    lch.ifft(A, 8)
    np.testing.assert_array_equal(A, keep)


@needs_numba
@pytest.mark.parametrize("m", [3, 6, 9])
def test_derivative_and_basis_conversions(m, rng):
    gf = GF2m(m)
    lch = LCHBasis(gf, random_basis(gf, rng))
    for L in sorted({1, 2, 3, 7, 8, 9, gf.q // 2 + 1, gf.q} & set(range(1, gf.q + 1))):
        a = rng.integers(0, gf.q, L).astype(gf.dtype)
        a[rng.random(L) < 0.3] = 0                                   # exercise zero coefficients
        same(*both(lambda: lch.derivative(a)))
        same(*both(lambda: lch.to_monomial(a)))
        same(*both(lambda: lch.from_monomial(a)))
        set_numba(True)
        np.testing.assert_array_equal(lch.from_monomial(lch.to_monomial(a)), a)


@needs_numba
def test_mul_via_fft(rng):
    gf = GF2m(8)
    lch = LCHBasis(gf)
    a = rng.integers(0, 256, 40).astype(gf.dtype)
    b = rng.integers(0, 256, 70).astype(gf.dtype)
    same(*both(lambda: lch.mul(a, b)))


# ---------------------------------------------------------------- keyeq.py
@needs_numba
def test_monomial_helpers(rng):
    gf = GF2m(8)
    for _ in range(30):
        a = rng.integers(0, 256, rng.integers(1, 40)).astype(gf.dtype)
        b = rng.integers(0, 256, rng.integers(1, 40)).astype(gf.dtype)
        a[-1] = a[-1] or 1
        b[-1] = b[-1] or 1
        b[rng.random(b.size) < 0.4] = 0
        b[-1] = b[-1] or 1
        same(*both(lambda: keyeq._pmul(gf, a, b)))
        same(*both(lambda: keyeq._pdivmod(gf, a, b)))


@needs_numba
@pytest.mark.parametrize("leaf_log", [0, 1, 3, 6])
def test_solve_gke_fast_matches(leaf_log, rng):
    m, t = 8, 6
    code = FFTRSCode(m, t, leaf_log=leaf_log)
    cw = code.encode(rng.integers(0, code.n, code.k))
    for f in (0, 5, 16):
        v = (code.d - f) // 2
        rx, _, er = corrupt(code, cw, v, f, rng)
        run = lambda: code.decode(rx, er, solver="fast")
        a, b = both(run)
        same(a.codeword, b.codeword)
        same(a.error_positions, b.error_positions)
        same(a.error_values, b.error_values)
        same(a.erasure_values, b.erasure_values)
        same(b.codeword, cw)


# ------------------------------------------------------------- end to end
@needs_numba
@pytest.mark.parametrize("m,t", [(4, 2), (8, 3), (8, 5), (8, 7), (10, 6), (12, 8)])
@pytest.mark.parametrize("solver", ["fast", "euclid"])
@pytest.mark.parametrize("std_basis", [True, False])
def test_decode_identical(m, t, solver, std_basis, rng):
    basis = None if std_basis else random_basis(GF2m(m), rng)
    code = FFTRSCode(m, t, basis=basis)
    cw = both(lambda: code.encode(np.arange(code.k) % code.n))
    same(*cw)
    cw = cw[0]
    for f in (0, code.d // 4, code.d):
        v = (code.d - f) // 2
        rx, _, er = corrupt(code, cw, v, f, rng)
        a, b = both(lambda: code.decode(rx, er, solver=solver))
        same(a.codeword, cw)
        same(b.codeword, cw)
        same(a.error_positions, b.error_positions)
        same(a.error_values, b.error_values)
        same(a.erasure_values, b.erasure_values)


@needs_numba
def test_failures_are_identical(rng):
    """Beyond the radius both paths must fail (or agree) in the same way."""
    from eeleopard import DecodeFailure
    code = FFTRSCode(8, 5)
    cw = code.encode(rng.integers(0, 256, code.k))
    for trial in range(20):
        rx, _, er = corrupt(code, cw, code.d // 2 + 3, 0, rng)

        def run():
            try:
                return code.decode(rx, er).codeword
            except DecodeFailure as e:
                return np.array([len(str(e))])
        a, b = both(run)
        same(a, b)


@needs_numba
def test_reed_solomon_general_n_k(rng):
    rs = ReedSolomon(255, 223)
    data = rng.integers(0, 256, 223)
    cw = rs.encode(data)
    rx = cw.copy()
    rx[rng.choice(255, 16, replace=False)] ^= rng.integers(1, 256, 16).astype(rx.dtype)
    a, b = both(lambda: rs.decode(rx).message)
    same(a, data)
    same(b, data)


# ---------------------------------------------------- fallback without numba
def _run(code, env_extra=None):
    env = dict(os.environ)
    env.pop("EELEOPARD_NUMBA", None)
    env.update(env_extra or {})
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True,
                          text=True, env=env, timeout=300)


ROUNDTRIP = """
    import numpy as np, eeleopard
    code = eeleopard.FFTRSCode(8, 5)
    rng = np.random.default_rng(0)
    cw = code.encode(rng.integers(0, 256, code.k))
    rx = cw.copy(); rx[[5, 40, 99, 200]] ^= 7; rx[[1, 2]] = 0
    for solver in ("fast", "euclid"):
        assert (code.decode(rx, [1, 2], solver=solver).codeword == cw).all()
    eeleopard.warmup()                      # must be a harmless no-op
    print("ok", eeleopard.HAVE_NUMBA, eeleopard.numba_enabled(), eeleopard.set_numba(True))
"""


def test_works_when_numba_is_not_importable():
    r = _run("import sys; sys.modules['numba'] = None\n" + textwrap.dedent(ROUNDTRIP))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["ok", "False", "False", "False"]


def test_env_var_disables_numba():
    r = _run(ROUNDTRIP, {"EELEOPARD_NUMBA": "0"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["ok", "False", "False", "False"]


@needs_numba
def test_numba_is_the_default_when_installed():
    r = _run(ROUNDTRIP)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["ok", "True", "True", "True"]


# ------------------------------------------------------------------ fwht.py
@needs_numba
@pytest.mark.parametrize("size", [1, 2, 4, 64, 1024])
def test_wht(size, rng):
    from eeleopard.fwht import wht
    x = rng.integers(-1000, 1000, size)
    keep = x.copy()
    a, b = both(lambda: wht(x))
    same(a, b)
    same(x, keep)                                   # input is not modified


@needs_numba
@pytest.mark.parametrize("m", [2, 3, 5, 8, 11, 14])
def test_erasure_locator_evaluator(m, rng):
    from eeleopard.fwht import ErasureLocatorEvaluator, erasure_locator_direct
    gf = GF2m(m)
    lch = LCHBasis(gf, random_basis(gf, rng))
    ev = ErasureLocatorEvaluator(lch)               # built with numba on: fwt_log must not depend on the mode
    set_numba(False)
    ev_np = ErasureLocatorEvaluator(lch)
    same(ev.fwt_log, ev_np.fwt_log)
    for f in sorted({0, 1, 2, gf.q // 3, gf.q // 2, gf.q}):
        er = rng.choice(gf.q, f, replace=False)
        a, b = both(lambda: ev(er))
        assert a.dtype == b.dtype
        same(a, b)
        if gf.q <= 256:
            same(b, erasure_locator_direct(gf, lch, er))
        same(ev(list(er)), b) if f else None        # list input works in both modes
