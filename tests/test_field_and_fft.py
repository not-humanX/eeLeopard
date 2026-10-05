"""GF(2^m) arithmetic, the novel basis, Algorithm 1 and the polynomial helpers,
all checked against brute-force definitions."""
import numpy as np
import pytest

from eeleopard import GF2m, LCHBasis, degree
from eeleopard.fwht import ErasureLocatorEvaluator, erasure_locator_direct
from conftest import random_basis


@pytest.mark.parametrize("m", range(2, 17))
def test_field_tables(m, rng):
    gf = GF2m(m)
    a = rng.integers(0, gf.q, 300)
    b = rng.integers(0, gf.q, 300)
    ref = np.array([gf.clmul(int(x), int(y)) for x, y in zip(a, b)])
    assert np.array_equal(gf.mul(a, b), ref)
    nz = b[b != 0]
    assert np.array_equal(gf.mul(gf.div(a[:nz.size], nz), nz), a[:nz.size])
    assert np.all(gf.mul(nz, gf.inv(nz)) == 1)
    assert gf.smul(0, 5 % gf.q) == 0 and gf.sdiv(0, 1) == 0


def test_non_primitive_polynomial_rejected():
    with pytest.raises(ValueError):
        GF2m(4, 0x1F)          # x^4+x^3+x^2+x+1: irreducible, but of order 5
    with pytest.raises(ValueError):
        GF2m(4, 0x15)          # reducible


def _brute_basis_matrix(nb):
    """BV[x, i] = Xbar_i(x) from the product definition, for every field element x."""
    return np.stack([nb.basis_values(x) for x in range(nb.n)])


@pytest.mark.parametrize("m", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("random_b", [False, True])
def test_subspace_polynomials(m, random_b, rng):
    gf = GF2m(m)
    nb = LCHBasis(gf, random_basis(gf, rng) if random_b else None)
    for j in range(m):
        # fbar_j vanishes exactly on V_j, is 1 at b_j, and is GF(2)-linear
        assert not nb.PHI[j][: 1 << j].any()
        assert np.all(nb.PHI[j][1 << j:] != 0)
        assert nb.PHI[j][1 << j] == 1
        i1, i2 = rng.integers(0, nb.n, 50), rng.integers(0, nb.n, 50)
        assert np.array_equal(nb.PHI[j][i1 ^ i2], nb.PHI[j][i1] ^ nb.PHI[j][i2])
        # monomial form agrees with the table
        for i in rng.integers(0, nb.n, 10):
            x, acc = int(nb.elem[i]), 0
            for l, c in enumerate(nb.FBAR_MONO[j]):
                xp = x
                for _ in range(l):
                    xp = gf.smul(xp, xp)
                acc ^= gf.smul(c, xp)
            assert acc == nb.PHI[j][i]


@pytest.mark.parametrize("m", [2, 3, 4, 5, 7])
@pytest.mark.parametrize("random_b", [False, True])
def test_fft_matches_definition(m, random_b, rng):
    gf = GF2m(m)
    nb = LCHBasis(gf, random_basis(gf, rng) if random_b else None)
    BV = _brute_basis_matrix(nb)
    for r in range(m + 1):
        size = 1 << r
        for off in [0] + [int(v) for v in rng.integers(0, nb.n, 3)]:
            coeffs = rng.integers(0, nb.n, size).astype(gf.dtype)
            vals = nb.fft(coeffs, r, off)
            pts = (nb.elem[:size] ^ nb.elem[off]).astype(np.int64)
            expect = np.bitwise_xor.reduce(gf.mul(BV[pts][:, :size], coeffs[None]), axis=1)
            assert np.array_equal(vals, expect)
            assert np.array_equal(nb.ifft(vals, r, off), coeffs)


def test_batched_fft_offsets(rng):
    gf = GF2m(6)
    nb = LCHBasis(gf)
    r = 3
    A = rng.integers(0, 64, (5, 8)).astype(gf.dtype)
    offs = np.array([0, 8, 16, 40, 56])
    batched = nb.fft(A, r, offs)
    for i in range(5):
        assert np.array_equal(batched[i], nb.fft(A[i], r, int(offs[i])))
    assert np.array_equal(nb.ifft(batched, r, offs), A)


@pytest.mark.parametrize("m", [3, 5, 8])
def test_top_coefficient_is_sum_of_values(m, rng):
    """In the normalised basis the Xbar_{2^r - 1} coefficient of the interpolant on a
    coset is the XOR of the values -- the fact behind eq. (8) and the syndrome."""
    gf = GF2m(m)
    nb = LCHBasis(gf)
    for r in range(1, m + 1):
        vals = rng.integers(0, nb.n, 1 << r).astype(gf.dtype)
        off = int(rng.integers(0, nb.n >> r)) << r
        assert nb.ifft(vals, r, off)[-1] == np.bitwise_xor.reduce(vals)


@pytest.mark.parametrize("m", [3, 4, 6, 8])
def test_polynomial_helpers(m, rng):
    gf = GF2m(m)
    nb = LCHBasis(gf, random_basis(gf, rng))
    n = nb.n

    def mono_eval(p, x):
        acc = 0
        for c in p[::-1].tolist():
            acc = gf.smul(acc, x) ^ c
        return acc

    for L in sorted({1, 2, 3, 5, n // 2, n // 2 + 1, n}):
        c = rng.integers(0, n, L).astype(gf.dtype)
        mono = nb.to_monomial(c)
        assert degree(mono) == degree(c)
        assert np.array_equal(nb.from_monomial(mono), c)
        for x in rng.integers(0, n, 3).tolist():
            assert mono_eval(mono, x) == np.bitwise_xor.reduce(gf.mul(c, nb.basis_values(x)[:L]))
        # formal derivative
        d_mono = np.zeros(L, dtype=gf.dtype)
        d_mono[: L - 1][0::2] = mono[1::2][: d_mono[: L - 1][0::2].size]
        dx = nb.derivative(c)
        assert np.array_equal(nb.to_monomial(dx), d_mono[: dx.size])
        # product via FFT
        c2 = rng.integers(0, n, max(1, n - L)).astype(gf.dtype)
        if L + c2.size - 1 <= n:
            pr = nb.to_monomial(nb.mul(c, c2))
            ref = np.zeros(L + c2.size - 1, dtype=gf.dtype)
            for i, co in enumerate(mono.tolist()):
                ref[i:i + c2.size] ^= gf.mulc(nb.to_monomial(c2), co)
            assert np.array_equal(pr, ref)


@pytest.mark.parametrize("m", [3, 5, 8])
def test_evaluation_helpers(m, rng):
    gf = GF2m(m)
    nb = LCHBasis(gf, random_basis(gf, rng))
    BV = _brute_basis_matrix(nb)
    for s in range(m):
        c = rng.integers(0, nb.n, (1 << s) + 1).astype(gf.dtype)      # degree 2^s term included
        allv = np.bitwise_xor.reduce(gf.mul(BV[nb.elem.astype(np.int64)][:, :c.size], c[None]), axis=1)
        assert np.array_equal(nb.evaluate_everywhere(c, s), allv)
        idx = rng.integers(0, nb.n, 9)
        assert np.array_equal(nb.evaluate_at(c, idx, s), allv[idx])
        assert np.array_equal(nb.evaluate_at(np.stack([c, c]), idx, s)[1], allv[idx])
        off = int(rng.integers(0, nb.n >> s)) << s
        assert np.array_equal(nb.evaluate_coset(c, s, off), allv[off:off + (1 << s)])


@pytest.mark.parametrize("m", [2, 4, 7, 10])
def test_fwht_erasure_locator(m, rng):
    gf = GF2m(m)
    nb = LCHBasis(gf, random_basis(gf, rng))
    ev = ErasureLocatorEvaluator(nb)
    for f in sorted({1, 2, 3, nb.n // 4, nb.n - 1, nb.n}):
        E = rng.choice(nb.n, f, replace=False)
        assert np.array_equal(ev(E), erasure_locator_direct(gf, nb, E))
    # all points erased: gamma'(w) = f_m'(w) = 1 everywhere
    assert np.all(ev(np.arange(nb.n)) == 1)


def test_bad_basis_rejected():
    gf = GF2m(4)
    with pytest.raises(ValueError):
        LCHBasis(gf, [1, 2, 3, 4])       # 1 ^ 2 = 3: dependent
    with pytest.raises(ValueError):
        LCHBasis(gf, [1, 2, 4])          # too short
