"""The Lin-Chung-Han novel polynomial basis and its FFT (Section 2 of the paper).

.. rubric:: Notation (paper Section 2)

* ``B = (b_0, ..., b_{m-1})`` is a basis of GF(2^m) over GF(2).  Element number
  ``i`` is ``w_i = i_0 b_0 + ... + i_{m-1} b_{m-1}`` where ``i_j`` are the bits of
  ``i`` (eq. 2).  The default basis ``b_j = 2^j`` makes ``w_i`` the field element
  whose integer representation is ``i``.
* ``V_k = span(b_0, ..., b_{k-1})`` and ``f_k(x) = prod_{w in V_k} (x - w)`` is the
  subspace polynomial of degree ``2^k`` (eq. 3).  ``f_k`` is GF(2)-linear.
* ``X_i(x) = prod_j f_j(x)^{i_j}`` (eq. 4) and the *normalised* basis used by
  Algorithm 1 is ``Xbar_i = X_i / p_i`` with ``p_i = prod_j f_j(b_j)^{i_j}``,
  i.e. ``Xbar_i = prod_j fbar_j^{i_j}`` with ``fbar_j = f_j / f_j(b_j)``.

Everything in this package works directly in the normalised basis ``Xbar``.
Algorithm 1 in the paper is the FFT in ``Xbar``; the paper's ``FFT_X`` is that
transform with a diagonal rescaling by ``p_i`` before/after, so working in
``Xbar`` throughout only changes a few constants (see ``rs.py``).

.. rubric:: Useful facts used all over the package

* ``Xbar_{a + b} = Xbar_a * Xbar_b`` whenever ``a & b == 0``.
* ``fbar_j`` vanishes on ``V_j``, equals 1 on ``b_j + V_j`` and is constant on
  every coset of ``V_j``.  Hence ``P mod fbar_s`` is simply the first ``2^s``
  coefficients of ``P`` in ``Xbar``.
* ``fbar_j'(x)`` is the constant ``prod_{l<j} f_l(b_l) / f_j(b_j)`` (linearised
  polynomials have constant derivatives), so formal derivatives in ``Xbar``
  cost O(n log n) (Lin et al. 2016a).
"""

import numpy as np

from . import _accel
from .gf import GF2m


def _next_pow2_exp(length: int) -> int:
    """Smallest r with 2**r >= length (length >= 1)."""
    return max(0, (int(length) - 1).bit_length())


def degree(a) -> int:
    """Degree of a coefficient vector.

    Valid in both the monomial and the Xbar basis, since ``deg Xbar_i = i``.

    :param a: coefficients, lowest degree first.
    :type a: array_like
    :returns: index of the last non-zero coefficient; ``-1`` for the zero
        polynomial.
    :rtype: int
    """
    nz = np.flatnonzero(np.asarray(a))
    return int(nz[-1]) if nz.size else -1


class LCHBasis:
    """Tables for the novel polynomial basis, the FFT/IFFT of Algorithm 1 and the
    polynomial helpers built on it.

    Polynomials are numpy arrays of field elements (``gf.dtype``) holding
    coefficients in the normalised basis ``Xbar`` unless stated otherwise
    (lowest degree first).  Methods never modify their inputs.

    :param GF2m gf: the field.
    :param basis: a GF(2)-basis ``(b_0, ..., b_{m-1})`` of the field as integers.
        Defaults to ``b_j = 2**j``, which makes ``w_i`` the element with integer
        value ``i``.
    :type basis: sequence[int] or None
    :raises ValueError: if ``basis`` does not have ``m`` non-zero, GF(2)-linearly
        independent elements.

    :ivar GF2m gf: the field.
    :ivar int m: extension degree.
    :ivar int n: field size ``2**m`` (the code length of the paper's codes).
    :ivar list[int] basis: the basis ``b_j``.
    :ivar numpy.ndarray elem: ``elem[i] = w_i`` (eq. 2).
    :ivar numpy.ndarray index_of: inverse of ``elem``: the index ``i`` of an element.
    :ivar numpy.ndarray W: ``W[j, l] = f_j(b_l)``, shape ``(m + 1, m)``.
    :ivar list[int] Wd: ``Wd[j] = f_j(b_j)``, all non-zero.
    :ivar numpy.ndarray PHI: ``PHI[j, i] = fbar_j(w_i)``, shape ``(m, n)``.
    :ivar numpy.ndarray PHI_LOG: sentinel logarithms of ``PHI``.
    :ivar list[int] DER: ``DER[j]``, the constant ``fbar_j'(x)``.
    :ivar list[list[int]] FBAR_MONO: ``FBAR_MONO[j][l]`` is the coefficient of
        ``x**(2**l)`` in the linearised polynomial ``fbar_j``.
    """

    def __init__(self, gf: GF2m, basis=None):
        self.gf = gf
        m, n = gf.m, gf.q
        self.m, self.n = m, n
        dt = gf.dtype
        if basis is None:
            basis = [1 << j for j in range(m)]
        b = [int(x) for x in basis]
        if len(b) != m or any(not (0 < x < n) for x in b):
            raise ValueError(f"basis must contain {m} non-zero elements of GF(2^{m})")
        self.basis = b

        # w_i for every index i (eq. 2), and the inverse map.
        elem = np.zeros(n, dtype=np.int64)
        for j in range(m):
            h = 1 << j
            elem[h:2 * h] = elem[:h] ^ b[j]
        if np.unique(elem).size != n:
            raise ValueError("basis elements are not linearly independent over GF(2)")
        self.elem = elem.astype(dt)
        self.index_of = np.empty(n, dtype=np.int64)
        self.index_of[elem] = np.arange(n)

        # W[j][l] = f_j(b_l) via f_{j+1}(y) = f_j(y)^2 + f_j(b_j) f_j(y).
        W = np.zeros((m + 1, m), dtype=np.int64)
        W[0] = b
        for j in range(m):
            cur = W[j].astype(dt)
            W[j + 1] = gf.mul(cur, cur) ^ gf.mulc(cur, int(W[j, j]))
        self.W = W
        self.Wd = [int(W[j, j]) for j in range(m)]          # f_j(b_j) != 0
        assert all(self.Wd) and not W[m].any()

        # PHI[j][i] = fbar_j(w_i), built by linearity; PHI_LOG = its sentinel log.
        PHI = np.zeros((m, n), dtype=dt)
        for j in range(m):
            vals = gf.div(W[j].astype(dt), np.full(m, self.Wd[j], dtype=dt))
            row = np.zeros(n, dtype=dt)
            for l in range(m):
                h = 1 << l
                row[h:2 * h] = row[:h] ^ vals[l]
            PHI[j] = row
        self.PHI = PHI
        self.PHI_LOG = gf.LOG[PHI]

        # DER[j] = fbar_j'(x) (a constant) = prod_{l<j} f_l(b_l) / f_j(b_j).
        der, prod = [], 1
        for j in range(m):
            der.append(gf.sdiv(prod, self.Wd[j]))
            prod = gf.smul(prod, self.Wd[j])
        self.DER = der

        # Monomial coefficients of fbar_j:  fbar_j(x) = sum_l FBAR_MONO[j][l] x^(2^l).
        c = [1]                                   # f_0(x) = x
        mono = []
        for j in range(m):
            mono.append([gf.sdiv(ci, self.Wd[j]) for ci in c])
            new = [0] * (j + 2)
            for l, cl in enumerate(c):
                new[l + 1] ^= gf.smul(cl, cl)
                new[l] ^= gf.smul(self.Wd[j], cl)
            c = new
        self.FBAR_MONO = mono

        # Array forms of the tables above for the optional numba kernels.
        self._fm = np.zeros((m, m + 1), dtype=dt)
        for j, row in enumerate(mono):
            self._fm[j, :len(row)] = row
        self._der_log = np.array([gf.slog(x) for x in self.DER], dtype=np.int32)

        self._aranges = {}

    # ------------------------------------------------------------------ FFT
    def _arange(self, k: int) -> np.ndarray:
        a = self._aranges.get(k)
        if a is None:
            a = self._aranges[k] = np.arange(k, dtype=np.int64)
        return a

    def _transform(self, A, r: int, offset, inverse: bool) -> np.ndarray:
        gf = self.gf
        A = np.array(A, dtype=gf.dtype)            # always a private copy
        size = 1 << r
        if A.shape[-1] != size:
            raise ValueError(f"last axis must have length 2^{r}={size}, got {A.shape[-1]}")
        if r > self.m:
            raise ValueError("transform larger than the field")
        shape = A.shape
        A2 = A.reshape(-1, size)
        nb = A2.shape[0]
        offs = np.asarray(offset, dtype=np.int64).reshape(-1)
        if offs.size not in (1, nb):
            raise ValueError("offset must be a scalar or have one entry per row")
        EXP, LOG = gf.EXP, gf.LOG
        if _accel.numba_enabled():
            _accel.fft_rows(A2, np.ascontiguousarray(offs), self.PHI_LOG, EXP, LOG, r, inverse)
            return A2.reshape(shape)
        levels = range(r) if inverse else range(r - 1, -1, -1)
        for j in levels:
            h = 1 << j
            nblk = size >> (j + 1)
            V = A2.reshape(nb, nblk, 2, h)
            lo = V[:, :, 0, :]
            hi = V[:, :, 1, :]
            # twiddle of block s at level j:  fbar_j(beta + w_s)  (line 4 of Algorithm 1)
            idx = offs[:, None] ^ (self._arange(nblk) << (j + 1))[None, :]
            twl = self.PHI_LOG[j][idx][:, :, None]
            if inverse:
                hi ^= lo
                lo ^= EXP[LOG[hi] + twl]
            else:
                lo ^= EXP[LOG[hi] + twl]
                hi ^= lo
        return A2.reshape(shape)

    def fft(self, A, r: int, offset=0) -> np.ndarray:
        """Algorithm 1, ``FFT_Xbar(P, r, beta)`` with ``beta = w_offset``.

        ``A[..., i]`` are the Xbar-coefficients of a polynomial of degree < 2^r;
        the result holds ``P(w_i + beta)`` for ``i < 2^r``.  Leading axes are
        batched and ``offset`` may give one coset index per batch row.  Cost:
        O(r 2^r) per row.

        :param A: Xbar-coefficients, last axis of length ``2**r``.
        :type A: numpy.ndarray
        :param int r: transform size exponent, ``0 <= r <= m``.
        :param offset: coset index ``i`` (so ``beta = w_i``); a scalar, or one
            entry per row of the flattened leading axes.  Normally a multiple
            of ``2**r``, which selects a coset of ``V_r``.
        :type offset: int or numpy.ndarray
        :returns: evaluations, same shape as ``A``; ``A`` itself is not modified.
        :rtype: numpy.ndarray
        :raises ValueError: if the last axis is not ``2**r`` long, ``r > m``, or
            ``offset`` has the wrong length.
        """
        return self._transform(A, r, offset, inverse=False)

    def ifft(self, A, r: int, offset=0) -> np.ndarray:
        """Inverse of :meth:`fft` (interpolation on the coset ``V_r + w_offset``).

        :param A: values ``P(w_i + beta)`` for ``i < 2**r`` along the last axis.
        :type A: numpy.ndarray
        :param int r: transform size exponent.
        :param offset: coset index, as for :meth:`fft`.
        :type offset: int or numpy.ndarray
        :returns: the Xbar-coefficients, same shape as ``A``.
        :rtype: numpy.ndarray
        :raises ValueError: as for :meth:`fft`.
        """
        return self._transform(A, r, offset, inverse=True)

    # ------------------------------------------------------------ evaluation
    def _add_top(self, vals, top, s: int, offs):
        """Add ``top * Xbar_{2^s}(x) = top * fbar_s(beta)`` (constant on the coset)."""
        gf = self.gf
        fb = self.PHI[s][offs]                       # one value per coset
        return vals ^ gf.mul(top[..., None], fb[..., None])

    def evaluate_everywhere(self, coeffs, s: int) -> np.ndarray:
        """Evaluate polynomial(s) of degree <= 2^s at all ``n`` field points.

        Done as ``n / 2^s`` FFTs of size ``2^s`` on the cosets of ``V_s``, so the
        cost is O(n log 2^s) (Step 3 of the decoder).  The leading coefficient
        ``Xbar_{2^s}`` (a polynomial of degree exactly ``2^s``) is handled
        analytically because ``fbar_s`` is constant on every coset of ``V_s``.

        :param coeffs: Xbar-coefficients, length ``<= 2**s + 1``; one polynomial
            (1-D) or one per row (2-D).
        :type coeffs: numpy.ndarray
        :param int s: size exponent of the cosets, ``0 <= s <= m``.
        :returns: values at ``w_0, ..., w_{n-1}``; shape ``(n,)`` for 1-D input,
            ``(P, n)`` for ``P`` input rows.
        :rtype: numpy.ndarray
        :raises ValueError: if the degree exceeds ``2**s``.
        """
        gf = self.gf
        C = np.atleast_2d(np.asarray(coeffs, dtype=gf.dtype))
        P, L = C.shape
        size = 1 << s
        if L > size:
            if L != size + 1:
                raise ValueError("degree too large for evaluate_everywhere")
        ncos = self.n >> s
        A = np.zeros((P, ncos, size), dtype=gf.dtype)
        A[:, :, :min(L, size)] = C[:, None, :min(L, size)]
        offs = self._arange(ncos) << s
        V = self.fft(A.reshape(P * ncos, size), s, np.tile(offs, P)).reshape(P, ncos, size)
        if L == size + 1:
            V = self._add_top(V, np.repeat(C[:, size:size + 1], ncos, axis=1), s, offs[None, :])
        V = V.reshape(P, self.n)
        return V[0] if np.ndim(coeffs) == 1 else V

    def evaluate_at(self, coeffs, idx, s: int) -> np.ndarray:
        """Evaluate polynomial(s) of degree <= 2^s at selected points.

        Uses FFTs only on the cosets of ``V_s`` that contain a requested point, so
        a few points cost far less than :meth:`evaluate_everywhere`.

        :param coeffs: Xbar-coefficients, length ``<= 2**s + 1``; 1-D (one
            polynomial) or 2-D (one per row).
        :type coeffs: numpy.ndarray
        :param idx: indices ``i`` of the requested points ``w_i``.
        :type idx: array_like of int
        :param int s: size exponent of the cosets.
        :returns: values at ``w_idx``; shape ``(len(idx),)`` for 1-D input,
            ``(P, len(idx))`` for 2-D input.
        :rtype: numpy.ndarray
        :raises ValueError: if the degree exceeds ``2**s``.
        """
        gf = self.gf
        C = np.atleast_2d(np.asarray(coeffs, dtype=gf.dtype))
        P, L = C.shape
        size = 1 << s
        if L > size + 1:
            raise ValueError("degree too large for evaluate_at")
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            out = np.zeros((P, 0), dtype=gf.dtype)
            return out[0] if np.ndim(coeffs) == 1 else out
        cos = idx >> s
        ucos, inv = np.unique(cos, return_inverse=True)
        nc = ucos.size
        A = np.zeros((P, nc, size), dtype=gf.dtype)
        A[:, :, :min(L, size)] = C[:, None, :min(L, size)]
        offs = ucos << s
        V = self.fft(A.reshape(P * nc, size), s, np.tile(offs, P)).reshape(P, nc, size)
        if L == size + 1:
            V = self._add_top(V, np.repeat(C[:, size:size + 1], nc, axis=1), s, offs[None, :])
        out = V[:, inv, idx & (size - 1)]
        return out[0] if np.ndim(coeffs) == 1 else out

    def evaluate_coset(self, coeffs, s: int, offset: int) -> np.ndarray:
        """Values of polynomial(s) of degree <= 2^s on one coset ``V_s + w_offset``.

        :param coeffs: Xbar-coefficients, length ``<= 2**s + 1``; 1-D or 2-D.
        :type coeffs: numpy.ndarray
        :param int s: size exponent of the coset.
        :param int offset: coset index (a multiple of ``2**s``).
        :returns: ``2**s`` values per polynomial, in the order of the coset.
        :rtype: numpy.ndarray
        :raises ValueError: if the degree exceeds ``2**s``.
        """
        gf = self.gf
        C = np.atleast_2d(np.asarray(coeffs, dtype=gf.dtype))
        P, L = C.shape
        size = 1 << s
        A = np.zeros((P, size), dtype=gf.dtype)
        A[:, :min(L, size)] = C[:, :min(L, size)]
        V = self.fft(A, s, offset)
        if L == size + 1:
            V = V ^ gf.mulc(C[:, size:size + 1], int(self.PHI[s][offset]))
        elif L > size + 1:
            raise ValueError("degree too large for evaluate_coset")
        return V[0] if np.ndim(coeffs) == 1 else V

    # ------------------------------------------------------ polynomial tools
    def derivative(self, a) -> np.ndarray:
        """Formal derivative in Xbar.

        Uses ``(Xbar_i)' = sum_{j in bits(i)} DER[j] Xbar_{i-2^j}``, so the cost is
        O(n log n).

        :param a: Xbar-coefficients.
        :type a: array_like
        :returns: Xbar-coefficients of the derivative, same length as ``a``
            (at least 1).
        :rtype: numpy.ndarray
        """
        gf = self.gf
        a = np.asarray(a, dtype=gf.dtype)
        L = a.size
        if L <= 1:
            return np.zeros(max(L, 1), dtype=gf.dtype)
        r = _next_pow2_exp(L)
        A = np.zeros(1 << r, dtype=gf.dtype)
        A[:L] = a
        out = np.zeros_like(A)
        if _accel.numba_enabled():
            _accel.derivative(A, out, self._der_log, gf.EXP, gf.LOG, r)
            return out[:L]
        for j in range(r):
            h = 1 << j
            Av = A.reshape(-1, 2, h)
            Ov = out.reshape(-1, 2, h)
            Ov[:, 0, :] ^= gf.mulc(Av[:, 1, :], self.DER[j])
        return out[:L]

    def mul(self, a, b) -> np.ndarray:
        """Product of two polynomials given in Xbar.

        One forward FFT of both factors, a pointwise product, and one inverse FFT
        of length ``>= deg a + deg b + 1``.

        :param a: Xbar-coefficients of the first factor.
        :type a: array_like
        :param b: Xbar-coefficients of the second factor.
        :type b: array_like
        :returns: ``len(a) + len(b) - 1`` Xbar-coefficients of ``a * b`` (empty if
            either factor is empty).
        :rtype: numpy.ndarray
        :raises ValueError: if the product's length exceeds the field size.
        """
        gf = self.gf
        a = np.asarray(a, dtype=gf.dtype)
        b = np.asarray(b, dtype=gf.dtype)
        if a.size == 0 or b.size == 0:
            return np.zeros(0, dtype=gf.dtype)
        L = a.size + b.size - 1
        r = _next_pow2_exp(L)
        if r > self.m:
            raise ValueError("product degree exceeds the field size")
        A = np.zeros((2, 1 << r), dtype=gf.dtype)
        A[0, :a.size] = a
        A[1, :b.size] = b
        V = self.fft(A, r, 0)
        return self.ifft(gf.mul(V[0], V[1]), r, 0)[:L]

    # --------------------------------------------------- basis conversions
    def to_monomial(self, a) -> np.ndarray:
        """Xbar-coefficients -> monomial coefficients, O(n log^2 n).

        Bottom-up: a block of ``2^{j+1}`` Xbar-coefficients equals
        ``lo + fbar_j * hi`` where lo/hi are its halves, and ``fbar_j`` is sparse.

        :param a: Xbar-coefficients.
        :type a: array_like
        :returns: coefficients of ``1, x, x^2, ...``, same length as ``a``.
        :rtype: numpy.ndarray
        """
        gf = self.gf
        a = np.asarray(a, dtype=gf.dtype)
        L = a.size
        if L == 0:
            return a.copy()
        r = _next_pow2_exp(L)
        cur = np.zeros(1 << r, dtype=gf.dtype)
        cur[:L] = a
        if _accel.numba_enabled():
            return _accel.to_monomial(cur, self._fm, gf.EXP, gf.LOG, r)[:L]
        cur = cur.reshape(-1, 1)
        for j in range(r):
            h = 1 << j
            blocks = cur.reshape(-1, 2, h)
            lo, hi = blocks[:, 0, :], blocks[:, 1, :]
            out = np.zeros((blocks.shape[0], 2 * h), dtype=gf.dtype)
            out[:, :h] = lo
            for l, c in enumerate(self.FBAR_MONO[j]):
                e = 1 << l
                out[:, e:e + h] ^= gf.mulc(hi, c)
            cur = out
        return cur.reshape(-1)[:L]

    def from_monomial(self, p) -> np.ndarray:
        """Monomial coefficients -> Xbar-coefficients, O(n log^2 n).

        Top-down: divide each block by the sparse ``fbar_j``.  Quotient
        coefficient ``i`` only depends on quotient coefficients ``>= i + 2^{j-1}``,
        so each level is two vectorised half-block steps.  Inverse of
        :meth:`to_monomial`.

        :param p: monomial coefficients.
        :type p: array_like
        :returns: Xbar-coefficients, same length as ``p``.
        :rtype: numpy.ndarray
        """
        gf = self.gf
        p = np.asarray(p, dtype=gf.dtype)
        L = p.size
        if L == 0:
            return p.copy()
        r = _next_pow2_exp(L)
        cur = np.zeros(1 << r, dtype=gf.dtype)
        cur[:L] = p
        if _accel.numba_enabled():
            return _accel.from_monomial(cur, self._fm, gf.EXP, gf.LOG, r, gf.order)[:L]
        cur = cur.reshape(1, -1)
        for j in range(r - 1, -1, -1):
            h = 1 << j
            blk = cur.reshape(-1, 2 * h).copy()
            coef = self.FBAR_MONO[j]
            lead_inv = gf.sinv(coef[j])
            hi = np.zeros((blk.shape[0], h), dtype=gf.dtype)
            chunks = [(0, 1)] if h == 1 else [(h // 2, h), (0, h // 2)]
            for c0, c1 in chunks:
                hi[:, c0:c1] = gf.mulc(blk[:, c0 + h:c1 + h], lead_inv)
                for l in range(j):
                    e = 1 << l
                    blk[:, c0 + e:c1 + e] ^= gf.mulc(hi[:, c0:c1], coef[l])
            cur = np.stack([blk[:, :h], hi], axis=1).reshape(-1, h)
        return cur.reshape(-1)[:L]

    # ------------------------------------------------------------ oracles
    def basis_values(self, x: int) -> np.ndarray:
        """``[Xbar_0(x), ..., Xbar_{n-1}(x)]`` by the product definition (tests).

        :param int x: a field element.
        :returns: the ``n`` basis polynomials evaluated at ``x``.
        :rtype: numpy.ndarray
        """
        gf = self.gf
        xi = int(self.index_of[int(x)])
        vals = np.ones(self.n, dtype=gf.dtype)
        for j in range(self.m):
            h = 1 << j
            vals[h:2 * h] = gf.mulc(vals[:h], int(self.PHI[j][xi]))
        return vals
