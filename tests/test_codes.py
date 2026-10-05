"""Encoder, syndrome, key equation and the erasure-and-error decoder."""
import numpy as np
import pytest

from eeleopard import FFTRSCode, ReedSolomon, DecodeFailure, degree
from eeleopard.keyeq import solve_gke_euclid, solve_gke_fast
from eeleopard.reference import classical_decode
from conftest import random_basis, corrupt

CODES = [(2, 0), (2, 1), (3, 1), (3, 2), (4, 0), (4, 2), (4, 3), (5, 3), (6, 4), (8, 4), (8, 5), (8, 7)]


def make(m, t, rng, random_b=False, **kw):
    from eeleopard import GF2m
    basis = random_basis(GF2m(m), rng) if random_b else None
    return FFTRSCode(m, t, basis=basis, **kw)


# ---------------------------------------------------------------- encoder
@pytest.mark.parametrize("m,t", CODES)
@pytest.mark.parametrize("systematic", [True, False])
def test_encoder(m, t, systematic, rng):
    code = make(m, t, rng, random_b=True, systematic=systematic)
    msg = rng.integers(0, code.n, code.k)
    cw = code.encode(msg)
    # a codeword is the evaluation of a polynomial of degree < k on every w_i
    assert degree(code.lch.ifft(cw, m, 0)) < code.k
    assert code.is_codeword(cw)
    assert np.array_equal(code.extract_message(cw), msg)
    if systematic:
        assert np.array_equal(cw[code.d:], msg)             # message blocks c_1..c_{n/2^t-1}
    msg2 = rng.integers(0, code.n, code.k)
    assert np.array_equal(code.encode(msg ^ msg2), cw ^ code.encode(msg2))


@pytest.mark.parametrize("m,t", CODES)
def test_syndrome_is_top_block_of_ifft(m, t, rng):
    code = make(m, t, rng, random_b=True)
    y = rng.integers(0, code.n, code.n)
    full = code.lch.ifft(y, m, 0)
    assert np.array_equal(code.syndrome(y), full[code.k:])   # eq. (11) with Xbar_k


# ------------------------------------------------------ key-equation pieces
@pytest.mark.parametrize("m,t", [(4, 2), (5, 3), (6, 4), (8, 5)])
def test_generalised_syndrome_and_key_equation(m, t, rng):
    """Lemma 3 and the GKE (33) with the true locators, plus the corrected eq. (44)."""
    code = make(m, t, rng, random_b=True)
    gf, lch, n, d = code.gf, code.lch, code.n, code.d
    for _ in range(20):
        f = int(rng.integers(1, d + 1))
        v = int(rng.integers(0, (d - f) // 2 + 1))
        cw = code.encode(rng.integers(0, n, code.k))
        # put some erasures inside V_t (the parity block)
        rx, epos, fpos = corrupt(code, cw, v, f, rng, erasure_pool=np.arange(min(n, 2 * d)))
        s = code.syndrome(rx)
        G = code._gamma(fpos)
        # gamma in Xbar from its values on V_{t+1} (gamma has degree f <= d)
        pts = lch.elem[: 2 * d].astype(np.int64)
        gv = np.ones(2 * d, dtype=gf.dtype)
        for e in fpos.tolist():
            gv = gf.mul(gv, pts ^ int(lch.elem[e]))
        gamma = lch.ifft(gv, t + 1, 0)[: f + 1]             # degree f
        prod = np.zeros(2 * d, dtype=gf.dtype)
        p = lch.mul(s, gamma)
        prod[: p.size] = p
        sg_ref, qg_ref = prod[:d], prod[d:]                 # sbar gamma = qbar^g fbar_t + sbar^g
        # Lemma 3: only 2^t evaluation points are needed
        gam_vt = G[:d].copy()
        gam_vt[fpos[fpos < d]] = 0
        sg = lch.ifft(gf.mul(lch.fft(s, t, 0), gam_vt), t, 0)
        assert np.array_equal(sg, sg_ref)
        # corrected eq. (44): qbar^g(w) = (sbar(w) gamma'(w) + (sbar^g)'(w)) / fbar_t'  on V_t
        inside = fpos[fpos < d]
        if inside.size:
            lhs = lch.evaluate_at(qg_ref, inside, t)
            num = gf.mul(lch.fft(s, t, 0)[inside], G[inside]) ^ lch.fft(lch.derivative(sg), t, 0)[inside]
            assert np.array_equal(lhs, gf.mulc(num, gf.sinv(code.Dt)))
        # eq. (43) outside V_t
        outside = fpos[fpos >= d]
        if outside.size:
            lhs = lch.evaluate_at(qg_ref, outside, t)
            rhs = gf.div(lch.evaluate_at(sg, outside, t), lch.PHI[t][outside])
            assert np.array_equal(lhs, rhs)
        # GKE (33) with the true error locator: deg(sbar^g lambda mod fbar_t) < v + f
        lam_v = np.ones(n, dtype=gf.dtype)
        for e in epos.tolist():
            lam_v = gf.mul(lam_v, lch.elem ^ lch.elem[e])
        lam = lch.ifft(lam_v[: 2 * d], t + 1, 0)[: d]
        z = np.zeros(2 * d, dtype=gf.dtype)
        zz = lch.mul(sg, lam)
        z[: zz.size] = zz
        assert degree(z[:d]) < v + f


@pytest.mark.parametrize("m,t", [(4, 2), (5, 3), (6, 4), (8, 4), (8, 6)])
@pytest.mark.parametrize("leaf_log", [0, 1, 2, 6])
def test_solvers_agree(m, t, leaf_log, rng):
    """Euclid (paper Step 2) and the fast solver return the same locator up to scale."""
    code = make(m, t, rng, random_b=True)
    gf, lch, d = code.gf, code.lch, code.d
    for _ in range(15):
        f = int(rng.integers(0, d + 1))
        v = int(rng.integers(0, (d - f) // 2 + 1))
        cw = code.encode(rng.integers(0, code.n, code.k))
        rx, epos, fpos = corrupt(code, cw, v, f, rng)
        s = code.syndrome(rx)
        G = code._gamma(fpos) if f else np.ones(code.n, gf.dtype)
        gam_vt = G[:d].copy()
        gam_vt[fpos[fpos < d]] = 0
        S = gf.mul(lch.fft(s, t, 0), gam_vt)
        sg = lch.ifft(S, t, 0)
        lam_e, qt_e, z_e = solve_gke_euclid(lch, sg, f, t)
        lam_f, sdeg = solve_gke_fast(lch, S, f, t, leaf_log)
        lam_e = lam_e[: degree(lam_e) + 1]
        lam_f = lam_f[: degree(lam_f) + 1]
        assert degree(lam_e) == degree(lam_f) == v
        assert sdeg == v + f
        scale = gf.sdiv(int(lam_f[-1]), int(lam_e[-1]))
        assert np.array_equal(gf.mulc(lam_e, scale), lam_f)
        # eq. (35)-(36): w1 = lambda sbar^g + qtilde fbar_t
        P = np.zeros(2 * d, dtype=gf.dtype)
        pp = lch.mul(sg, lam_e)
        P[: pp.size] = pp
        qt = np.zeros(d, dtype=gf.dtype)
        qt[: qt_e.size] = qt_e
        assert np.array_equal(P[d:], qt)
        z = np.zeros(d, dtype=gf.dtype)
        z[: z_e.size] = z_e
        assert np.array_equal(P[:d], z)


# ----------------------------------------------------------------- decoding
@pytest.mark.parametrize("m,t", CODES)
@pytest.mark.parametrize("solver", ["fast", "euclid"])
@pytest.mark.parametrize("random_b", [False, True])
def test_decode_within_radius(m, t, solver, random_b, rng):
    code = make(m, t, rng, random_b=random_b, solver=solver, leaf_log=1)
    n, d = code.n, code.d
    for trial in range(40):
        f = int(rng.integers(0, d + 1))
        v = int(rng.integers(0, (d - f) // 2 + 1))
        if trial == 0:
            v, f = d // 2, 0                     # errors only, full radius
        elif trial == 1:
            v, f = 0, d                          # erasures only, all parity budget
        msg = rng.integers(0, n, code.k)
        cw = code.encode(msg)
        pool = np.arange(d) if trial % 3 == 2 and f else None     # erasures in the parity block
        rx, epos, fpos = corrupt(code, cw, v, f, rng, erasure_pool=pool)
        res = code.decode(rx, fpos)
        assert np.array_equal(res.codeword, cw)
        assert np.array_equal(res.message, msg)
        assert np.array_equal(res.error_positions, epos)
        assert np.array_equal(res.error_values, (rx ^ cw)[epos])
        assert np.array_equal(res.erasure_positions, fpos)
        assert np.array_equal(res.erasure_values, (rx ^ cw)[fpos])


@pytest.mark.parametrize("m,t", [(3, 2), (4, 2), (5, 3), (6, 4), (8, 3), (8, 5)])
def test_beyond_radius_matches_classical_decoder(m, t, rng):
    """Beyond 2v + f <= d both bounded-distance decoders must return the same thing:
    the unique codeword within the radius, or a failure."""
    code = make(m, t, rng, random_b=True, leaf_log=1)
    n, d = code.n, code.d
    outcomes = {"fail": 0, "codeword": 0}
    for _ in range(120):
        f = int(rng.integers(0, d + 1))
        v = min(n - f, (d - f) // 2 + int(rng.integers(1, 4)))
        cw = code.encode(rng.integers(0, n, code.k))
        rx, epos, fpos = corrupt(code, cw, v, f, rng)
        got = []
        for dec in (lambda: code.decode(rx, fpos), lambda: classical_decode(code, rx, fpos)):
            try:
                got.append(dec().codeword)
            except DecodeFailure:
                got.append(None)
        a, b = got
        assert (a is None) == (b is None)
        if a is None:
            outcomes["fail"] += 1
            continue
        outcomes["codeword"] += 1
        assert np.array_equal(a, b)
        assert code.is_codeword(a)
        keep = np.setdiff1d(np.arange(n), fpos)
        assert 2 * np.count_nonzero(a[keep] != rx[keep]) + f <= d
    assert outcomes["fail"] > 0


def test_no_errors_and_trivial_cases(rng):
    code = FFTRSCode(8, 4)
    cw = code.encode(rng.integers(0, 256, code.k))
    res = code.decode(cw)
    assert res.num_errors == 0 and np.array_equal(res.codeword, cw)
    # erasures on correct symbols give zero erasure values
    res = code.decode(cw, [0, 5, 100])
    assert np.array_equal(res.codeword, cw) and not res.erasure_values.any()
    # all-zero codeword
    z = np.zeros(code.n, dtype=np.uint16)
    assert code.decode(z).num_errors == 0


def test_input_validation(rng):
    code = FFTRSCode(4, 2)
    cw = code.encode(rng.integers(0, 16, code.k))
    with pytest.raises(ValueError):
        code.decode(cw[:-1])
    with pytest.raises(ValueError):
        code.decode(cw, [16])
    with pytest.raises(ValueError):
        code.encode(np.arange(code.k + 1) % 16)
    with pytest.raises(ValueError):
        code.encode(np.full(code.k, 16))
    with pytest.raises(DecodeFailure):
        code.decode(cw, [0, 1, 2, 3, 4])            # f > n - k
    with pytest.raises(ValueError):
        FFTRSCode(4, 4)


def test_classical_reference_itself(rng):
    code = FFTRSCode(6, 3)
    for _ in range(50):
        f = int(rng.integers(0, 9))
        v = int(rng.integers(0, (8 - f) // 2 + 1))
        cw = code.encode(rng.integers(0, 64, code.k))
        rx, epos, fpos = corrupt(code, cw, v, f, rng)
        assert np.array_equal(classical_decode(code, rx, fpos).codeword, cw)


# ------------------------------------------------------ general (n, k) codes
@pytest.mark.parametrize("n,k", [(255, 223), (204, 188), (255, 239), (15, 11), (7, 3),
                                 (100, 77), (528, 514), (2, 1), (1000, 500)])
def test_reedsolomon_general(n, k, rng):
    rs = ReedSolomon(n, k)
    q = rs.gf.q
    for _ in range(30):
        f = int(rng.integers(0, n - k + 1))
        v = int(rng.integers(0, (n - k - f) // 2 + 1))
        msg = rng.integers(0, q, k)
        cw = rs.encode(msg)
        assert rs.is_codeword(cw)
        assert np.array_equal(cw[n - k:], msg)
        rx, epos, fpos = corrupt(rs, cw, v, f, rng)
        res = rs.decode(rx, fpos)
        assert np.array_equal(res.codeword, cw)
        assert np.array_equal(res.message, msg)
        assert np.array_equal(res.error_positions, epos)
        assert np.array_equal(res.erasure_positions, fpos)


def test_reedsolomon_is_mds(rng):
    """Minimum distance n - k + 1: no non-zero codeword of weight <= n - k."""
    rs = ReedSolomon(12, 5)       # tiny: GF(16), exhaustive over single-symbol messages
    for i in range(5):
        for val in range(1, 16):
            msg = np.zeros(5, dtype=np.int64)
            msg[i] = val
            assert np.count_nonzero(rs.encode(msg)) >= 12 - 5 + 1


def test_reedsolomon_beyond_radius(rng):
    rs = ReedSolomon(60, 40)
    fails = 0
    for _ in range(60):
        cw = rs.encode(rng.integers(0, rs.gf.q, 40))
        rx, epos, _ = corrupt(rs, cw, 12, 0, rng)      # radius is 10
        try:
            res = rs.decode(rx)
            assert rs.is_codeword(res.codeword)
            assert np.count_nonzero(res.codeword != rx) <= 10
        except DecodeFailure:
            fails += 1
    assert fails > 0


# ------------------------------------------------- review follow-ups / API
@pytest.mark.parametrize("m,t", [(3, 1), (4, 2), (5, 3), (6, 2)])
def test_nonsystematic_uses_basis_X_of_eq6(m, t, rng):
    """systematic=False: c_i = u(w_i) with u = sum u_i X_i, X_i = prod_j f_j^{i_j} (unnormalised)."""
    code = make(m, t, rng, random_b=True, systematic=False)
    gf, lch = code.gf, code.lch
    u = rng.integers(0, code.n, code.k)
    cw = code.encode(u)
    p = np.ones(code.n, dtype=gf.dtype)                  # p_i = prod_j f_j(b_j)^{i_j}
    for j in range(m):
        p[1 << j: 2 << j] = gf.mulc(p[: 1 << j], lch.Wd[j])
    for i in rng.integers(0, code.n, 6).tolist():
        x = int(lch.elem[i])
        X = gf.mul(lch.basis_values(x)[: code.k], p[: code.k])     # X_i(x) = p_i Xbar_i(x)
        assert cw[i] == np.bitwise_xor.reduce(gf.mul(u, X))
    assert np.array_equal(code.extract_message(cw), u)
    v = code.d // 4
    rx, epos, fpos = corrupt(code, cw, v, code.d - 2 * v, rng)       # at full radius
    assert np.array_equal(code.decode(rx, fpos).message, u)


def test_cantor_basis_makes_paper_constants_trivial(rng):
    """With a Cantor basis (b_0 = 1, b_i^2 + b_i = b_{i-1}) every f_l(b_l) = 1, so X = Xbar
    and the paper's implicit f_t'(x) = 1 is exact: C = 1 and fbar_t' = 1."""
    b = [1, 214, 152, 146, 86, 200, 88, 230]
    from eeleopard import GF2m
    gf = GF2m(8)
    assert all(gf.smul(b[i], b[i]) ^ b[i] == b[i - 1] for i in range(1, 8))
    for t in range(8):
        code = FFTRSCode(8, t, basis=b)
        assert code.lch.Wd == [1] * 8 and code.C == 1 and code.Dt == 1
    code = FFTRSCode(8, 4, basis=b)
    cw = code.encode(rng.integers(0, 256, code.k))
    rx, epos, fpos = corrupt(code, cw, 3, 10, rng, erasure_pool=np.arange(16))
    assert np.array_equal(code.decode(rx, fpos).codeword, cw)


@pytest.mark.parametrize("m", [3, 6, 9, 12, 16])
def test_product_of_all_f_l_b_l_is_one(m, rng):
    """prod_{l<m} f_l(b_l) = f_m'(x) = 1, hence prod_{l>=t} f_l(b_l) = 1 / f_t'(x): the
    factor by which eq. (8)-style syndromes in basis X differ from rbar div X_k."""
    from eeleopard import GF2m, LCHBasis
    gf = GF2m(m)
    for basis in (None, random_basis(gf, rng)):
        nb = LCHBasis(gf, basis)
        prod = 1
        for w in nb.Wd:
            prod = gf.smul(prod, w)
        assert prod == 1


def test_erasure_argument_forms(rng):
    code = FFTRSCode(5, 3)
    cw = code.encode(rng.integers(0, 32, code.k))
    rx = cw.copy()
    rx[[1, 4, 20]] ^= 7
    ref = code.decode(rx, [1, 4, 20]).codeword
    mask = np.zeros(32, dtype=bool)
    mask[[1, 4, 20]] = True
    for form in ([20, 4, 1], (1, 4, 20), {1, 4, 20}, np.array([4, 20, 1, 1]), mask,
                 np.array([[1, 4], [20, 20]])):
        assert np.array_equal(code.decode(rx, form).codeword, ref)
    for bad, exc in (([1.5], TypeError), (np.ones(31, dtype=bool), ValueError), ([-1], ValueError)):
        with pytest.raises(exc):
            code.decode(rx, bad)
    with pytest.raises(ValueError):
        code.decode(rx, solver="eucild")
    with pytest.raises(ValueError):
        FFTRSCode(5, 3, leaf_log=-1)
    # received words of various integer dtypes / lists; the caller's array is not modified
    for form in (rx.tolist(), rx.astype(np.int64), rx.astype(np.uint8)):
        before = np.array(form, copy=True)
        assert np.array_equal(code.decode(form, [1, 4, 20]).codeword, ref)
        assert np.array_equal(np.asarray(form), before)
    with pytest.raises(ValueError):
        code.decode(np.full(32, 32))
    with pytest.raises(TypeError):
        code.decode(rx.astype(float))


def test_reedsolomon_field_size_rule():
    assert ReedSolomon(255, 223).m == 8 and ReedSolomon(204, 188).m == 8
    assert ReedSolomon(255, 200).m == 9                  # 200 + 64 > 256
    with pytest.raises(ValueError):
        ReedSolomon(255, 200, m=8)
