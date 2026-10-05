"""A textbook syndrome decoder for the same codes, independent of the FFT machinery.

Used by the tests as an oracle and by the benchmark as the "conventional" baseline.

The code of ``FFTRSCode`` is the evaluation code on *all* 2^m field elements.
Its Lagrange weights are ``prod_{j != i} (w_i - w_j) = f_m'(w_i) = 1``, so the
parity checks are plain power sums

    S_j = sum_i y_i w_i^j = 0,   j = 0 .. n-k-1   (with 0^0 = 1).

Decoding: O(n (n-k)) syndromes, Forney-style modified syndromes for the erasures,
Berlekamp-Massey for the error locator, a Chien-type search over all n locators,
Forney's formula for all values, and a final syndrome check so that any returned
word is a codeword within the radius 2v + f <= n - k.
"""

import numpy as np

from .rs import DecodeFailure, DecodeResult


def _horner(gf, coeffs, x):
    """Evaluate monomial-basis polynomial(s) at the points x."""
    out = np.zeros(np.shape(x), dtype=gf.dtype)
    for c in np.asarray(coeffs)[::-1].tolist():
        out = gf.mul(out, x) ^ out.dtype.type(c)
    return out


def power_sums(gf, y, pts, count: int) -> np.ndarray:
    """Power-sum syndromes ``S_j = sum_i y_i * pts_i**j`` for ``j = 0 .. count-1``.

    :param GF2m gf: the field.
    :param y: received symbols.
    :type y: numpy.ndarray
    :param pts: the evaluation point of each symbol, same length as ``y``.
    :type pts: numpy.ndarray
    :param int count: number of syndromes (``n - k`` for these codes).
    :returns: the ``count`` power sums (``0**0`` is taken to be 1).
    :rtype: numpy.ndarray
    """
    S = np.zeros(count, dtype=gf.dtype)
    cur = np.array(y, dtype=gf.dtype)
    for j in range(count):
        S[j] = np.bitwise_xor.reduce(cur)
        cur = gf.mul(cur, pts)
    return S


def berlekamp_massey(gf, S):
    """Shortest LFSR generating a syndrome sequence.

    :param GF2m gf: the field.
    :param S: syndromes (or modified syndromes).
    :type S: array_like
    :returns: ``(C, L)``: the connection polynomial ``C`` (``C[0] = 1``,
        lowest degree first, length ``L + 1``) and the register length ``L``.
    :rtype: tuple[numpy.ndarray, int]
    """
    N = len(S)
    S = np.asarray(S, dtype=gf.dtype)
    C = np.zeros(N + 1, dtype=gf.dtype)
    B = np.zeros(N + 1, dtype=gf.dtype)
    C[0] = B[0] = 1
    L, shift, b = 0, 1, 1
    for r in range(N):
        delta = int(S[r])
        if L:
            delta ^= int(np.bitwise_xor.reduce(gf.mul(C[1:L + 1], S[r - L:r][::-1])))
        if delta == 0:
            shift += 1
            continue
        coef = gf.sdiv(delta, b)
        if 2 * L <= r:
            T = C.copy()
            C[shift:] ^= gf.mulc(B[:N + 1 - shift], coef)
            L, B, b, shift = r + 1 - L, T, delta, 1
        else:
            C[shift:] ^= gf.mulc(B[:N + 1 - shift], coef)
            shift += 1
    return C[:L + 1], L


def classical_decode(code, received, erasures=None) -> DecodeResult:
    """Textbook erasure-and-error decoder for an :class:`~eeleopard.rs.FFTRSCode`.

    Independent of the FFT machinery: power-sum syndromes, Berlekamp-Massey,
    Chien-style search, Forney's formula.  Used as the test oracle and the
    benchmark baseline.  Corrects ``v`` errors and ``f`` erasures with
    ``2*v + f <= n - k`` and takes ``O(n (n-k))``.

    :param FFTRSCode code: the code to decode for (supplies the field, the
        point numbering and ``n``, ``k``).
    :param received: the received word, ``n`` field elements.
    :type received: array_like of int
    :param erasures: erased positions, or ``None``.
    :type erasures: array_like of int or None
    :returns: the result, with ``solver == "classical"``.
    :rtype: DecodeResult
    :raises DecodeFailure: if the word is outside the decoding radius (a returned
        word is always a codeword within it).
    """
    gf = code.gf
    n, d = code.n, code.d
    pts = code.lch.elem
    y = gf.asarray(received).reshape(-1)
    Ef = np.unique(np.asarray([] if erasures is None else erasures, dtype=np.int64).reshape(-1))
    f = int(Ef.size)
    if f > d:
        raise DecodeFailure("too many erasures")
    S = power_sums(gf, y, pts, d)
    empty = np.zeros(0, dtype=np.int64)
    if f == 0 and not S.any():
        return DecodeResult(y.copy(), code.extract_message(y), empty,
                            np.zeros(0, gf.dtype), Ef, np.zeros(0, gf.dtype), "classical")
    # erasure locator Gamma(x) = prod (x - w_e), low -> high coefficients
    Gam = np.ones(1, dtype=gf.dtype)
    for e in Ef.tolist():
        new = np.zeros(Gam.size + 1, dtype=gf.dtype)
        new[1:] ^= Gam
        new[:-1] ^= gf.mulc(Gam, int(pts[e]))
        Gam = new
    # modified syndromes: erasures drop out, errors are weighted by Gamma(w)
    T = np.zeros(d - f, dtype=gf.dtype)
    for i, g in enumerate(Gam.tolist()):
        T ^= gf.mulc(S[i:i + d - f], g)
    Cpoly, L = berlekamp_massey(gf, T)
    if 2 * L + f > d:
        raise DecodeFailure("too many errors")
    lam = np.zeros(L + 1, dtype=gf.dtype)
    lam[:Cpoly.size] = Cpoly
    lam = lam[::-1].copy()                     # lambda(x) = x^L C(1/x)
    roots = empty
    if L:
        roots = np.flatnonzero(_horner(gf, lam, pts) == 0)
        if roots.size != L or np.intersect1d(roots, Ef).size:
            raise DecodeFailure("error locator does not split properly")
    # full locator and evaluator
    Lam = np.zeros(L + f + 1, dtype=gf.dtype)
    for i, c in enumerate(lam.tolist()):
        if c:
            Lam[i:i + Gam.size] ^= gf.mulc(Gam, c)
    V = L + f
    Om = np.zeros(max(V, 1), dtype=gf.dtype)
    for i in range(1, V + 1):
        Om[:i] ^= gf.mulc(S[:i][::-1], int(Lam[i]))
    dLam = np.zeros(max(V, 1), dtype=gf.dtype)
    dLam[0::2] = Lam[1::2][:dLam[0::2].size]
    locs = np.concatenate([roots, Ef]).astype(np.int64)
    x = pts[locs]
    vals = gf.div(_horner(gf, Om, x), _horner(gf, dLam, x))
    corr = y.copy()
    corr[locs] ^= vals
    if power_sums(gf, corr, pts, d).any():
        raise DecodeFailure("correction is not a codeword")
    return DecodeResult(corr, code.extract_message(corr), roots, vals[:L],
                        Ef, vals[L:], "classical")
