"""Optional numba kernels for the hot loops of ``lch.py``, ``keyeq.py`` and ``fwht.py``.

numba is **not** required.  If it is missing (or broken, or disabled), the pure
python + numpy code paths in ``lch.py`` / ``keyeq.py`` are used unchanged and
give bit-identical results.  The kernels below implement exactly the same
algorithms; they only remove the numpy temporaries and python-level loops.

.. rubric:: Control

* ``EELEOPARD_NUMBA=0`` in the environment disables numba for the whole process.
* ``eeleopard.set_numba(False / True)`` switches at run time (used by ``bench.py``
  to compare both paths in one process).  ``True`` is ignored when numba is not
  installed.
* ``eeleopard.HAVE_NUMBA`` tells whether numba could be imported and
  ``eeleopard.numba_enabled()`` whether the kernels are currently in use.

Kernels are compiled lazily on first use and cached on disk (``cache=True``), so
the compile cost (a few seconds) is paid once per installation.  ``warmup()``
triggers all of them up front.

All kernels take the field tables as arguments: ``EXP`` (uint16, zero padded)
and ``LOG`` (int32, with the sentinel ``log(0) = 2(q-1)``), so
``EXP[LOG[a] + LOG[b]]`` is a branch-free product that is right for a = 0 or
b = 0 as well.
"""

import os

import numpy as np

#: ``True`` if numba could be imported and was not disabled through the
#: environment variable ``EELEOPARD_NUMBA=0``.
HAVE_NUMBA = False
_ENABLED = False

try:
    if os.environ.get("EELEOPARD_NUMBA", "1").strip().lower() in ("0", "false", "no", "off"):
        raise ImportError("numba disabled through EELEOPARD_NUMBA")
    from numba import njit
    HAVE_NUMBA = True
    _ENABLED = True
except Exception:          # ImportError, or a numba/llvmlite that fails to load
    njit = None


def numba_enabled() -> bool:
    """Whether the numba kernels are currently in use.

    :returns: ``True`` if numba is installed and has not been switched off with
        :func:`set_numba` or ``EELEOPARD_NUMBA=0``.
    :rtype: bool
    """
    return _ENABLED


def set_numba(flag: bool) -> bool:
    """Switch the numba kernels on or off at run time.

    Turning them on has no effect if numba is unavailable (the numpy paths stay
    in use).  Both paths return bit-identical results.

    :param bool flag: ``True`` to use the compiled kernels, ``False`` for the
        pure numpy code.
    :returns: the resulting state, as :func:`numba_enabled` would report it.
    :rtype: bool
    """
    global _ENABLED
    _ENABLED = bool(flag) and HAVE_NUMBA
    return _ENABLED


if HAVE_NUMBA:
    _jit = njit(cache=True, nogil=True)

    # ------------------------------------------------------------------ FFT
    @_jit
    def fft_rows(A, offs, PHI_LOG, EXP, LOG, r, inverse):
        """Algorithm 1 (or its inverse) in place on every row of A (nb, 2^r).

        ``offs`` holds one coset index per row, or a single one for all rows."""
        nb = A.shape[0]
        size = 1 << r
        for b in range(nb):
            off = offs[b] if offs.shape[0] > 1 else offs[0]
            row = A[b]
            if inverse:
                for j in range(r):
                    h = 1 << j
                    for s in range(size >> (j + 1)):
                        tw = PHI_LOG[j, off ^ (s << (j + 1))]
                        base = s << (j + 1)
                        for i in range(base, base + h):
                            row[i + h] ^= row[i]
                            row[i] ^= EXP[LOG[row[i + h]] + tw]
            else:
                for j in range(r - 1, -1, -1):
                    h = 1 << j
                    for s in range(size >> (j + 1)):
                        tw = PHI_LOG[j, off ^ (s << (j + 1))]
                        base = s << (j + 1)
                        for i in range(base, base + h):
                            row[i] ^= EXP[LOG[row[i + h]] + tw]
                            row[i + h] ^= row[i]

    # ----------------------------------------------------------- derivative
    @_jit
    def derivative(A, out, DER_LOG, EXP, LOG, r):
        """out ^= formal derivative of A (length 2^r, both in Xbar); out starts at 0."""
        size = 1 << r
        for j in range(r):
            h = 1 << j
            lg = DER_LOG[j]
            for base in range(0, size, 2 * h):
                for i in range(base, base + h):
                    out[i] ^= EXP[LOG[A[i + h]] + lg]

    # --------------------------------------------------- basis conversions
    @_jit
    def to_monomial(a, FM, EXP, LOG, r):
        """Xbar -> monomial coefficients.  ``a`` has length 2^r; FM[j, l] = FBAR_MONO[j][l]."""
        size = 1 << r
        cur = a.copy()
        out = np.zeros(size, dtype=a.dtype)
        for j in range(r):
            h = 1 << j
            out[:] = 0
            for base in range(0, size, 2 * h):
                for i in range(h):
                    out[base + i] = cur[base + i]
                for l in range(j + 1):
                    c = FM[j, l]
                    if c == 0:
                        continue
                    lc = LOG[c]
                    e = 1 << l
                    for i in range(h):
                        x = cur[base + h + i]
                        if x != 0:
                            out[base + e + i] ^= EXP[LOG[x] + lc]
            cur, out = out, cur
        return cur

    @_jit
    def from_monomial(p, FM, EXP, LOG, r, order):
        """Monomial -> Xbar coefficients (top-down division by the sparse fbar_j)."""
        size = 1 << r
        cur = p.copy()
        hi = np.zeros(max(size, 1), dtype=p.dtype)
        for j in range(r - 1, -1, -1):
            h = 1 << j
            lead = FM[j, j]
            li = order - LOG[lead]                          # log(1 / lead)
            c_lo = 0 if h == 1 else h // 2
            for base in range(0, size, 2 * h):
                # two half-blocks, upper first: quotient coefficient i depends on
                # quotient coefficients >= i + 2^(j-1) only
                for part in range(1 if h == 1 else 2):
                    if h == 1:
                        c0, c1 = 0, 1
                    elif part == 0:
                        c0, c1 = c_lo, h
                    else:
                        c0, c1 = 0, c_lo
                    for c in range(c0, c1):
                        hi[c] = EXP[LOG[cur[base + h + c]] + li]
                    for l in range(j):
                        cl = FM[j, l]
                        if cl == 0:
                            continue
                        lc = LOG[cl]
                        e = 1 << l
                        for c in range(c0, c1):
                            cur[base + c + e] ^= EXP[LOG[hi[c]] + lc]
                for c in range(h):
                    cur[base + h + c] = hi[c]
        return cur

    # ------------------------------------------- keyeq.py: fast solver bits
    @_jit
    def pw_matmul(X, Y, EXP, LOG):
        """Pointwise product of a 2x2 matrix X (2,2,N) with Y (2,c,N) -> (2,c,N)."""
        c = Y.shape[1]
        N = X.shape[2]
        P = np.empty((2, c, N), dtype=X.dtype)
        for i in range(2):
            for l in range(c):
                for k in range(N):
                    P[i, l, k] = (EXP[LOG[X[i, 0, k]] + LOG[Y[0, l, k]]]
                                  ^ EXP[LOG[X[i, 1, k]] + LOG[Y[1, l, k]]])
        return P

    @_jit
    def order_basis_leaf(Bv, F0, F1, E, pos0, H, sh0, sh1, EXP, LOG, order):
        """The iterative order-basis loop of ``keyeq._order_basis_leaf``.

        Bv (2,2,2H) holds the local basis rows as values on the enclosing coset
        (updated in place); E are the 2H elements of that coset.  Returns the new
        shifted degrees."""
        n2 = 2 * H
        for i in range(H):
            a00, a01 = Bv[0, 0, pos0 + i], Bv[0, 1, pos0 + i]
            a10, a11 = Bv[1, 0, pos0 + i], Bv[1, 1, pos0 + i]
            f0, f1 = F0[i], F1[i]
            r0 = EXP[LOG[a00] + LOG[f0]] ^ EXP[LOG[a01] + LOG[f1]]
            r1 = EXP[LOG[a10] + LOG[f0]] ^ EXP[LOG[a11] + LOG[f1]]
            if r0 == 0 and r1 == 0:
                continue
            # pivot: the row of smallest shifted degree among those not yet vanishing
            if r0 != 0 and (r1 == 0 or sh0 <= sh1):
                p, o, rp, ro = 0, 1, r0, r1
            else:
                p, o, rp, ro = 1, 0, r1, r0
            lq = LOG[EXP[LOG[ro] - LOG[rp] + order]] if ro != 0 else 0
            pt = E[pos0 + i]
            for c in range(2):
                for k in range(n2):
                    lp = LOG[Bv[p, c, k]]
                    if ro != 0:                          # row_o <- row_o - (ro/rp) row_p
                        Bv[o, c, k] ^= EXP[lp + lq]
                    Bv[p, c, k] = EXP[lp + LOG[E[k] ^ pt]]   # row_p <- (x - w_i) row_p
            if p == 0:
                sh0 += 1
            else:
                sh1 += 1
        return sh0, sh1

    # ---------------------------------------------- fwht.py: erasure locator
    @_jit
    def wht_inplace(x):
        """Unnormalised Walsh-Hadamard transform of an int64 vector (length 2^m), in place."""
        n = x.shape[0]
        h = 1
        while h < n:
            for base in range(0, n, 2 * h):
                for i in range(base, base + h):
                    a = x[i]
                    b = x[i + h]
                    x[i] = a + b
                    x[i + h] = a - b
            h <<= 1

    @_jit
    def erasure_locator(erasures, fwt_log, EXP, order):
        """gamma(w) off the erasures and gamma'(w) on them, at all n field points
        (eqs. 47-48): two WHTs and a pointwise product mod 2^m - 1, fused in one pass.

        ``fwt_log`` is the precomputed FWHT of the log table.  Python-style ``%``
        (non-negative for a positive modulus) is what the numpy version relies on too."""
        n = fwt_log.shape[0]
        R = np.zeros(n, dtype=np.int64)
        for i in range(erasures.shape[0]):
            R[erasures[i]] = 1
        wht_inplace(R)
        for i in range(n):
            R[i] = (R[i] * fwt_log[i]) % order
        wht_inplace(R)
        out = np.empty(n, dtype=EXP.dtype)
        for i in range(n):
            out[i] = EXP[R[i] % order]
        return out

    # ------------------------------------- keyeq.py: monomial Euclid solver
    @_jit
    def pmul(a, b, EXP, LOG):
        """Schoolbook product of trimmed, non-empty polynomials (len(a) <= len(b) preferred)."""
        out = np.zeros(a.shape[0] + b.shape[0] - 1, dtype=a.dtype)
        for i in range(a.shape[0]):
            c = a[i]
            if c != 0:
                lc = LOG[c]
                for j in range(b.shape[0]):
                    out[i + j] ^= EXP[LOG[b[j]] + lc]
        return out

    @_jit
    def pdivmod(a, b, EXP, LOG, order):
        """Quotient and (untrimmed, length deg b) remainder of a / b; needs deg a >= deg b >= 0."""
        a = a.copy()
        db = b.shape[0] - 1
        da = a.shape[0] - 1
        li = order - LOG[b[db]]
        q = np.zeros(da - db + 1, dtype=a.dtype)
        for i in range(da - db, -1, -1):
            x = a[i + db]
            if x != 0:
                lc = LOG[x] + li
                lc = lc - order if lc >= order else lc
                q[i] = EXP[lc]
                for j in range(db + 1):
                    a[i + j] ^= EXP[LOG[b[j]] + lc]
        return q, a[:db].copy()


def warmup() -> None:
    """Compile (or load from the cache) every kernel by decoding a few small words.

    Not needed for correctness; it only moves the one-off JIT cost out of the
    first real call.  Does nothing when numba is not in use.

    :rtype: None
    """
    if not _ENABLED:
        return
    from .rs import FFTRSCode
    rng = np.random.default_rng(0)
    # t = 7 > default leaf_log, so both the leaf and the divide-and-conquer kernels run
    for m, t in ((8, 7), (8, 5)):
        code = FFTRSCode(m, t)
        cw = code.encode(rng.integers(0, code.n, code.k))
        f = code.d // 4
        v = (code.d - f) // 2
        pos = rng.choice(code.n, v + f, replace=False)
        rx = cw.copy()
        rx[pos[:v]] ^= rng.integers(1, code.n, v).astype(rx.dtype)
        rx[pos[v:]] = 0
        for solver in ("fast", "euclid"):
            code.decode(rx, pos[v:], solver=solver)
