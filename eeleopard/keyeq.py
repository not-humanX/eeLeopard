"""Solvers for the generalised key equation (GKE), eq. (33):

        zbar(x) = sbar^g(x) * lambda(x) + qtilde(x) * fbar_t(x),
        deg zbar < v + f,   deg lambda = v,   2v + f <= n - k = 2^t = d.

Two interchangeable solvers are provided.  Both return lambda (up to a non-zero
scalar, which cancels in the error/erasure-value formulas) in the Xbar basis.

``solve_gke_euclid``
    Step 2 of the paper literally: divide f_t by sbar^g (quotient qbar_t,
    remainder rbar_t), run the extended Euclidean algorithm on (sbar^g, rbar_t)
    keeping the 2x2 cofactor matrix of eq. (34), and return
    lambda = u_1 - v_1 qbar_t, qtilde = v_1 (eqs. 35-36).  It stops at the first
    remainder of degree < (d + f)/2, the classical erasure-aware stopping rule.
    Polynomial division has no cheap form in Xbar, so this solver converts
    sbar^g to the monomial basis (O(d log^2 d)) and runs a quadratic-time
    Euclid there.  It is the readable reference.

``solve_gke_fast``
    An O(d log^2 d) solver.  The paper delegates this step to the half-GCD of
    Lin, Al-Naffouri & Han (2016b); the variant here is its dual formulation,
    which suits the Xbar basis because it never needs to divide.  The modulus
    fbar_t splits into the linear factors (x - w), w in V_t, so the GKE is the
    rational-interpolation problem

        lambda(w) * S(w) = z(w)  for all w in V_t,  S(w) = sbar(w) gamma(w),

    and the minimal solution is the minimal row of a 2x2 *order basis* in shifted
    reduced form (shift (f, 1)).  The order basis is built by divide and conquer
    over the coset tree V_t = V_{t-1} u (V_{t-1} + b_{t-1}) -- the same tree the
    FFT walks.  Every polynomial is kept in Xbar, cosets are evaluated with
    Algorithm 1, and 2x2 polynomial-matrix products use FFT multiplication.
    Leaves use the quadratic iterative algorithm (Beckermann-Labahn "M-Basis").
    When 2v + f <= d the minimal solution is unique up to a scalar, so both
    solvers return the same locator.
"""

import numpy as np

from . import _accel
from .lch import LCHBasis, degree

DEFAULT_LEAF_LOG = 6          # leaves of 64 points use the iterative algorithm


# ---------------------------------------------------------------------------
# monomial-basis helpers (used only by the Euclidean reference solver)
# ---------------------------------------------------------------------------
def _trim(p: np.ndarray) -> np.ndarray:
    return p[:degree(p) + 1]


def _padd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(max(a.size, b.size), dtype=a.dtype if a.size else b.dtype)
    out[:a.size] ^= a
    out[:b.size] ^= b
    return _trim(out)


def _pmul(gf, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = _trim(a), _trim(b)
    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=gf.dtype)
    if a.size > b.size:
        a, b = b, a
    if _accel.numba_enabled():
        return _accel.pmul(a, b, gf.EXP, gf.LOG)
    out = np.zeros(a.size + b.size - 1, dtype=gf.dtype)
    for i, c in enumerate(a.tolist()):
        if c:
            out[i:i + b.size] ^= gf.mulc(b, c)
    return out


def _pdivmod(gf, a: np.ndarray, b: np.ndarray):
    a = _trim(a).copy()
    b = _trim(b)
    db = b.size - 1
    if db < 0:
        raise ZeroDivisionError("polynomial division by zero")
    da = a.size - 1
    if da < db:
        return np.zeros(0, dtype=gf.dtype), a
    if _accel.numba_enabled():
        q, rem = _accel.pdivmod(a, b, gf.EXP, gf.LOG, gf.order)
        return q, _trim(rem)
    lead_inv = gf.sinv(int(b[-1]))
    q = np.zeros(da - db + 1, dtype=gf.dtype)
    for i in range(da - db, -1, -1):
        c = gf.smul(int(a[i + db]), lead_inv)
        if c:
            q[i] = c
            a[i:i + db + 1] ^= gf.mulc(b, c)
    return q, _trim(a[:db])


# ---------------------------------------------------------------------------
# Step 2 as written in the paper: Euclid on (f_t, sbar^g) with eqs. (34)-(36)
# ---------------------------------------------------------------------------
def solve_gke_euclid(lch: LCHBasis, sg_xbar, f: int, t: int):
    """Step 2 via the extended Euclidean algorithm (eqs. 34-36); see module doc.

    Quadratic in ``d = 2**t``, but short-circuits when there are no errors.

    :param LCHBasis lch: the basis tables of the field.
    :param sg_xbar: the generalised syndrome ``sbar^g`` in Xbar (``2**t``
        coefficients).
    :type sg_xbar: numpy.ndarray
    :param int f: number of erasures.
    :param int t: ``n - k = 2**t``.
    :returns: ``(lambda, qtilde, zbar)``, the error locator, the quotient and
        the evaluator of the GKE (33), all in Xbar.  ``lambda`` is determined up
        to a non-zero scalar and has degree ``v`` when ``2*v + f <= 2**t``.
        ``lambda`` has no valid solution beyond that radius; callers must check
        its degree.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    gf = lch.gf
    d = 1 << t
    thresh = (d + f + 1) // 2           # stop once deg(remainder) < (d + f) / 2
    one = np.ones(1, dtype=gf.dtype)
    sg = _trim(lch.to_monomial(sg_xbar))
    if degree(sg) < thresh:             # no errors: lambda = 1, zbar = sbar^g
        return one, np.zeros(1, dtype=gf.dtype), np.array(sg_xbar, dtype=gf.dtype)
    ft = np.zeros(d + 1, dtype=gf.dtype)            # fbar_t in the monomial basis
    for l, c in enumerate(lch.FBAR_MONO[t]):
        ft[1 << l] = c
    q_t, r_t = _pdivmod(gf, ft, sg)                 # f_t = qbar_t sbar^g + rbar_t
    # (w0, w1)^T = [[u0, v0], [u1, v1]] (sbar^g, rbar_t)^T            -- eq. (34)
    w0, w1 = sg, r_t
    u0, v0 = one.copy(), np.zeros(0, dtype=gf.dtype)
    u1, v1 = np.zeros(0, dtype=gf.dtype), one.copy()
    while degree(w1) >= thresh:
        q, rem = _pdivmod(gf, w0, w1)
        w0, w1 = w1, rem
        u0, u1 = u1, _padd(u0, _pmul(gf, q, u1))
        v0, v1 = v1, _padd(v0, _pmul(gf, q, v1))
    lam = _padd(u1, _pmul(gf, v1, q_t))           # lambda = u1 - v1 qbar_t   (eq. 36)
    qtilde = v1                                     # qtilde = v1
    to_x = lambda p: lch.from_monomial(p) if p.size else np.zeros(1, dtype=gf.dtype)
    return to_x(lam), to_x(qtilde), to_x(w1)


# ---------------------------------------------------------------------------
# fast solver: divide-and-conquer order basis on the coset tree of V_t
# ---------------------------------------------------------------------------
def _order_basis_leaf(lch: LCHBasis, j: int, beta: int, F: np.ndarray, sh):
    """Iterative order basis for the 2^j points of the coset V_j + w_beta.

    F[r, i] is the residual of incoming basis row r at the point w_{beta + i}; we
    look for combinations (a, b) with a F[0] + b F[1] = 0 at every point.  The
    local basis rows are kept as values on the enclosing coset V_{j+1} + w_beta'
    (enough to determine degree <= 2^j), so the residual of a row at the next
    point is read off directly and multiplying a row by (x - w) is a pointwise
    product.  One inverse FFT at the end yields Xbar coefficients."""
    gf = lch.gf
    EXP, LOG, slog, smul = gf.EXP, gf.LOG, gf._log, gf.smul
    H = 1 << j
    bprime = beta & ~H                    # enclosing coset of V_{j+1}
    pos0 = beta & H
    E = lch.elem[bprime + np.arange(2 * H)]
    Bv = np.zeros((2, 2, 2 * H), dtype=gf.dtype)
    Bv[0, 0] = 1
    Bv[1, 1] = 1
    if _accel.numba_enabled():
        sh0, sh1 = _accel.order_basis_leaf(
            Bv, np.ascontiguousarray(F[0]), np.ascontiguousarray(F[1]), E,
            pos0, H, int(sh[0]), int(sh[1]), EXP, LOG, gf.order)
        sh = [sh0, sh1]
    else:
        pts = E[pos0:pos0 + H].tolist()
        # log(x - w_i) on E for every leaf point (precomputed for small leaves)
        LD = LOG[E[None, :] ^ E[pos0:pos0 + H, None]] if H <= 256 else None
        F0, F1 = np.asarray(F[0]).tolist(), np.asarray(F[1]).tolist()
        sh = list(sh)
        for i in range(H):
            (a00, a01), (a10, a11) = Bv[:, :, pos0 + i].tolist()
            r0 = smul(a00, F0[i]) ^ smul(a01, F1[i])
            r1 = smul(a10, F0[i]) ^ smul(a11, F1[i])
            if r0 == 0 and r1 == 0:
                continue
            # pivot: the row of smallest shifted degree among those not yet vanishing
            p = 0 if (r0 != 0 and (r1 == 0 or sh[0] <= sh[1])) else 1
            o = 1 - p
            rp, ro = (r0, r1) if p == 0 else (r1, r0)
            Lp = LOG[Bv[p]]
            if ro:                                   # row_o <- row_o - (ro/rp) row_p
                Bv[o] ^= EXP[Lp + slog[gf.sdiv(ro, rp)]]
            ld = LD[i] if LD is not None else LOG[E ^ pts[i]]
            Bv[p] = EXP[Lp + ld]                     # row_p <- (x - w_i) row_p
            sh[p] += 1
    C = lch.ifft(Bv.reshape(4, 2 * H), j + 1, bprime).reshape(2, 2, 2 * H)
    return C[:, :, :H + 1], sh


def _polymat_mul(lch: LCHBasis, B2: np.ndarray, B1: np.ndarray, j: int) -> np.ndarray:
    """(2x2) x (2x2) product of matrices whose entries have degree <= H = 2^{j-1}.

    Products have degree <= 2^j.  Their residues mod fbar_j come from 2^j-point
    FFTs on V_j; the single remaining Xbar_{2^j} coefficient follows from
    Xbar_H^2 = fbar_{j-1}^2 = theta fbar_j + fbar_{j-1}, theta = f_j(b_j)/f_{j-1}(b_{j-1})^2,
    so it is theta * (sum of products of the Xbar_H coefficients)."""
    gf = lch.gf
    N = 1 << j
    H = N >> 1
    A = np.zeros((8, N), dtype=gf.dtype)
    A[:4, :H + 1] = B2.reshape(4, H + 1)
    A[4:, :H + 1] = B1.reshape(4, H + 1)
    V = lch.fft(A, j, 0)
    V2, V1 = V[:4].reshape(2, 2, N), V[4:].reshape(2, 2, N)
    if _accel.numba_enabled():
        P = _accel.pw_matmul(V2, V1, gf.EXP, gf.LOG)
    else:
        P = np.empty((2, 2, N), dtype=gf.dtype)
        for i in range(2):
            for l in range(2):
                P[i, l] = gf.mul(V2[i, 0], V1[0, l]) ^ gf.mul(V2[i, 1], V1[1, l])
    out = np.empty((2, 2, N + 1), dtype=gf.dtype)
    out[:, :, :N] = lch.ifft(P.reshape(4, N), j, 0).reshape(2, 2, N)
    theta = gf.sdiv(lch.Wd[j], gf.smul(lch.Wd[j - 1], lch.Wd[j - 1]))
    t2, t1 = B2[:, :, H], B1[:, :, H]
    top = gf.mul(t2[:, 0, None], t1[None, 0, :]) ^ gf.mul(t2[:, 1, None], t1[None, 1, :])
    out[:, :, N] = gf.mulc(top, theta)
    return out


def _order_basis(lch: LCHBasis, j: int, beta: int, F: np.ndarray, sh, leaf_log: int):
    if j <= leaf_log:
        return _order_basis_leaf(lch, j, beta, F, sh)
    gf = lch.gf
    h = j - 1
    H = 1 << h
    B1, sh1 = _order_basis(lch, h, beta, F[:, :H], sh, leaf_log)
    beta2 = beta | H                                     # the coset V_{j-1} + b_{j-1} + beta
    V = lch.evaluate_coset(B1.reshape(4, H + 1), h, beta2).reshape(2, 2, H)
    F2 = F[:, H:]
    if _accel.numba_enabled():
        R = _accel.pw_matmul(V, np.ascontiguousarray(F2).reshape(2, 1, H), gf.EXP, gf.LOG)[:, 0]
    else:
        R = np.empty((2, H), dtype=gf.dtype)
        for i in range(2):
            R[i] = gf.mul(V[i, 0], F2[0]) ^ gf.mul(V[i, 1], F2[1])
    B2, sh2 = _order_basis(lch, h, beta2, R, sh1, leaf_log)
    return _polymat_mul(lch, B2, B1, j), sh2


def solve_gke_fast(lch: LCHBasis, S_vals, f: int, t: int, leaf_log: int = DEFAULT_LEAF_LOG):
    """Step 2 in ``O(d log^2 d)``: minimal solution of ``lambda(w) S(w) = z(w)`` on
    ``V_t`` with shift ``(f, 1)`` (order basis, divide and conquer; see module doc).

    :param LCHBasis lch: the basis tables of the field.
    :param S_vals: ``S_vals[i] = sbar^g(w_i) = sbar(w_i) gamma(w_i)`` for
        ``i < 2**t``.
    :type S_vals: numpy.ndarray
    :param int f: number of erasures.
    :param int t: ``n - k = 2**t``.
    :param int leaf_log: cosets of ``2**leaf_log`` points or fewer are solved by
        the iterative quadratic algorithm instead of by recursion.
    :returns: ``(lambda, shifted_degree)``: the error locator in Xbar (length
        ``2**t + 1``, determined up to a non-zero scalar) and its shifted degree.
        Callers must check ``degree(lambda)`` against the decoding radius.
    :rtype: tuple[numpy.ndarray, int]
    """
    gf = lch.gf
    d = 1 << t
    F = np.empty((2, d), dtype=gf.dtype)
    F[0] = S_vals                    # residual of row (1, 0):  1*S + 0
    F[1] = 1                         # residual of row (0, 1):  0*S + 1   (-1 = 1)
    B, sh = _order_basis(lch, t, 0, F, [f, 1], leaf_log)
    i = 0 if sh[0] <= sh[1] else 1
    return B[i, 0], sh[i]
