"""Evaluating the erasure locator with fast Walsh-Hadamard transforms (Appendix).

For the erasure set E_f with gamma(x) = prod_{w in E_f} (x - w), every point w_l
satisfies

    Log Pi(w_l) = sum_{w in F} R_w Log(w_l + w)          (eq. 47, Log(0) := 0)

with R the indicator of E_f.  The sum is taken over the integers modulo 2^m - 1
and ``w_l + w`` is XOR of the indices, so it is a *logical (dyadic) convolution*
of R with L = (0, Log w_1, ..., Log w_{n-1}) and costs two length-n FWHTs (eq. 48).

Because the term with w = w_l contributes Log(0) = 0, the same convolution gives

    Pi(w_l) = gamma(w_l)   for w_l not in E_f, and
    Pi(w_l) = gamma'(w_l)  for w_l in E_f   (the product over the other erasures),

which is exactly what Steps 1 and 4 of the decoder need.  The missing 1/n
normalisation of the inverse transform is harmless: n = 2^m = 1 (mod 2^m - 1).
"""

import numpy as np

from . import _accel


def wht(x: np.ndarray) -> np.ndarray:
    """Unnormalised Walsh-Hadamard transform of an integer vector.

    Runs as a compiled kernel when numba is in use, in numpy otherwise.

    :param numpy.ndarray x: integer vector of length ``2**m``.
    :returns: the transform (``int64``); ``x`` is not modified.  Applying it
        twice multiplies by the length.
    :rtype: numpy.ndarray
    """
    x = np.array(x, dtype=np.int64)
    if _accel.numba_enabled():
        _accel.wht_inplace(x)
        return x
    n = x.size
    h = 1
    while h < n:
        v = x.reshape(-1, 2, h)
        a = v[:, 0, :].copy()
        b = v[:, 1, :]
        v[:, 0, :] += b
        v[:, 1, :] = a - b
        h <<= 1
    return x


class ErasureLocatorEvaluator:
    """Evaluates the erasure locator ``gamma`` (and ``gamma'`` on the erasures) at
    every field point in ``O(n log n)``.

    Precomputes the transform of the logarithm table once per field and basis,
    so each call needs two Walsh-Hadamard transforms.

    :param LCHBasis lch: basis tables of the field.

    :ivar int order: ``q - 1``.
    :ivar numpy.ndarray fwt_log: Walsh-Hadamard transform of the logarithms of
        the field points, modulo ``q - 1``.
    """

    def __init__(self, lch):
        gf = lch.gf
        self.gf, self.lch = gf, lch
        self.order = gf.order
        logs = gf.LOG[lch.elem].astype(np.int64)       # Log(w_i) for i >= 1
        logs[0] = 0                                     # Log(0) := 0
        self.fwt_log = wht(logs) % self.order           # FWT(L~), precomputed

    def __call__(self, erasures) -> np.ndarray:
        """Evaluate ``gamma`` and ``gamma'`` at all points.

        :param erasures: distinct indices ``i`` of the erased points ``w_i``.
        :type erasures: array_like of int
        :returns: ``out[i] = gamma(w_i)`` where ``w_i`` is not erased and
            ``gamma'(w_i)`` (the product over the other erasures) where it is;
            ``gamma(x) = prod (x - w)`` over the erasures.
        :rtype: numpy.ndarray
        """
        gf, n = self.gf, self.lch.n
        if _accel.numba_enabled():
            return _accel.erasure_locator(np.ascontiguousarray(erasures, dtype=np.int64),
                                          self.fwt_log, gf.EXP, self.order)
        R = np.zeros(n, dtype=np.int64)
        R[np.asarray(erasures, dtype=np.int64)] = 1
        prod = (wht(R) * self.fwt_log) % self.order      # pairwise integer products
        conv = wht(prod) % self.order                    # eq. (48)
        return gf.EXP[conv]


def erasure_locator_direct(gf, lch, erasures) -> np.ndarray:
    """Quadratic-time reference for :class:`ErasureLocatorEvaluator` (tests).

    :param GF2m gf: the field.
    :param LCHBasis lch: basis tables (supplies the point numbering).
    :param erasures: distinct indices of the erased points.
    :type erasures: array_like of int
    :returns: same values as ``ErasureLocatorEvaluator(lch)(erasures)``,
        computed in ``O(n f)``.
    :rtype: numpy.ndarray
    """
    pts = lch.elem
    out = np.ones(lch.n, dtype=gf.dtype)
    er = [int(e) for e in erasures]
    er_set = set(er)
    for e in er:
        diff = pts ^ lch.elem[e]
        mask = diff != 0
        out[mask] = gf.mul(out[mask], diff[mask])
    # entries at erasures skipped their own zero factor -> gamma'(w_e)
    assert len(er_set) == len(er)
    return out
