"""Arithmetic in GF(2^m), vectorised with numpy log/exp table lookups.

Field elements are stored as unsigned integers whose bits are the coefficients
of the polynomial representation (bit i <-> z^i modulo the primitive
polynomial), so field addition is XOR.

Multiplication uses the usual log/exp trick with a *zero sentinel*: log(0) is
set to 2*(q-1) and the exp table is padded with zeros, so ``EXP[LOG[a] + LOG[b]]``
is correct for every a, b (including zero) without any branching.

The only public class is :class:`GF2m`.
"""

import numpy as np

# Primitive polynomials, bit i = coefficient of z^i.  Primitivity is re-verified
# when a field is constructed, so a wrong entry here cannot go unnoticed.
DEFAULT_POLYS = {
    2: 0x7,        # z^2 + z + 1
    3: 0xB,        # z^3 + z + 1
    4: 0x13,       # z^4 + z + 1
    5: 0x25,       # z^5 + z^2 + 1
    6: 0x43,       # z^6 + z + 1
    7: 0x89,       # z^7 + z^3 + 1
    8: 0x11D,      # z^8 + z^4 + z^3 + z^2 + 1
    9: 0x211,      # z^9 + z^4 + 1
    10: 0x409,     # z^10 + z^3 + 1
    11: 0x805,     # z^11 + z^2 + 1
    12: 0x1053,    # z^12 + z^6 + z^4 + z + 1
    13: 0x201B,    # z^13 + z^4 + z^3 + z + 1
    14: 0x4443,    # z^14 + z^10 + z^6 + z + 1
    15: 0x8003,    # z^15 + z + 1
    16: 0x1002D,   # z^16 + z^5 + z^3 + z^2 + 1
}

MIN_M, MAX_M = 2, 16


class GF2m:
    """The finite field GF(2^m) for 2 <= m <= 16.

    Vector operations accept anything numpy can broadcast and return arrays of
    ``self.dtype`` (uint16).  Scalar helpers (``smul``, ``sdiv``, ...) work on
    Python ints and are used inside short sequential loops.

    Construction builds the log/exp tables and verifies that the field
    polynomial is primitive.

    :param int m: extension degree, ``2 <= m <= 16``.
    :param poly: field polynomial as a bit mask (bit i is the coefficient of
        z^i, bit m must be set).  Defaults to a built-in primitive polynomial
        (``0x11D`` for m = 8).
    :type poly: int or None
    :raises ValueError: if ``m`` is out of range, or ``poly`` has the wrong
        degree or is not primitive.

    :ivar int m: extension degree.
    :ivar int q: field size ``2**m``.
    :ivar int order: order of the multiplicative group, ``q - 1``.
    :ivar int poly: the field polynomial.
    :ivar numpy.dtype dtype: element dtype (always ``uint16``).
    :ivar numpy.ndarray LOG: ``int32`` table of length ``q``; ``LOG[0]`` is the
        sentinel ``2*(q-1)``.
    :ivar numpy.ndarray EXP: ``uint16`` table of length ``4*(q-1)+1``, the
        powers of the generator repeated and zero padded so that sums of
        sentinel logarithms index into zeros.
    """

    def __init__(self, m: int, poly: int | None = None):
        m = int(m)
        if not (MIN_M <= m <= MAX_M):
            raise ValueError(f"m must be in [{MIN_M}, {MAX_M}], got {m}")
        self.m = m
        self.q = 1 << m
        self.order = self.q - 1
        self.poly = DEFAULT_POLYS[m] if poly is None else int(poly)
        if self.poly >> m != 1:
            raise ValueError(f"field polynomial 0x{self.poly:x} does not have degree {m}")
        self.dtype = np.dtype(np.uint16)

        o, q = self.order, self.q
        exp = np.zeros(o, dtype=np.int64)
        log = np.full(q, -1, dtype=np.int64)
        x = 1
        for i in range(o):
            if log[x] != -1:
                raise ValueError(f"0x{self.poly:x} is not a primitive polynomial")
            exp[i] = x
            log[x] = i
            x <<= 1
            if x & q:
                x ^= self.poly
        if x != 1:
            raise ValueError(f"0x{self.poly:x} is not a primitive polynomial")

        self.LOG0 = 2 * o                      # log(0) sentinel
        LOG = log.astype(np.int32)
        LOG[0] = self.LOG0
        EXP = np.zeros(4 * o + 1, dtype=self.dtype)
        EXP[:o] = exp
        EXP[o:2 * o] = exp
        self.LOG = LOG
        self.EXP = EXP
        # Python-list copies for fast scalar access.
        self._log = LOG.tolist()
        self._exp = EXP.tolist()

    # ------------------------------------------------------------------ misc
    def __repr__(self) -> str:
        return f"GF2m(m={self.m}, poly=0x{self.poly:x})"

    def asarray(self, a) -> np.ndarray:
        """Convert to a field-element array, validating the range.

        :param a: integers in ``[0, q)`` (scalar, list or array).
        :type a: array_like
        :returns: ``a`` as an array of ``self.dtype`` (no copy if it already is one).
        :rtype: numpy.ndarray
        :raises TypeError: if ``a`` does not have an integer (or bool) dtype.
        :raises ValueError: if any value is outside ``[0, q)``.
        """
        arr = np.asarray(a)
        if arr.dtype.kind not in "iub":
            raise TypeError("field elements must be integers")
        if arr.size and (arr.dtype != self.dtype or self.q < 65536):
            if np.min(arr) < 0 or np.max(arr) >= self.q:
                raise ValueError(f"values must be in [0, {self.q})")
        return arr.astype(self.dtype, copy=False)

    def zeros(self, shape) -> np.ndarray:
        """Zero-filled array of field elements.

        :param shape: array shape.
        :type shape: int or tuple[int, ...]
        :returns: zeros of dtype ``self.dtype``.
        :rtype: numpy.ndarray
        """
        return np.zeros(shape, dtype=self.dtype)

    # ------------------------------------------------------ vector operations
    def mul(self, a, b) -> np.ndarray:
        """Element-wise product (numpy broadcasting applies).

        :param a: field elements.
        :type a: array_like
        :param b: field elements.
        :type b: array_like
        :returns: ``a * b`` in GF(2^m); zero where either factor is zero.
        :rtype: numpy.ndarray
        """
        return self.EXP[self.LOG[a] + self.LOG[b]]

    def mulc(self, a, c: int) -> np.ndarray:
        """Multiply an array by the scalar field element ``c``.

        :param a: field elements.
        :type a: array_like
        :param int c: the constant factor.
        :returns: ``a * c``.
        :rtype: numpy.ndarray
        """
        return self.EXP[self.LOG[a] + self._log[int(c)]]

    def mul_log(self, a, logc) -> np.ndarray:
        """Multiply by constants given through their (sentinel) logarithms.

        Saves the ``LOG`` lookup when the same constants are used repeatedly.

        :param a: field elements.
        :type a: array_like
        :param logc: ``LOG`` values of the constants (``slog`` for a scalar), which
            may be the zero sentinel; broadcast against ``a``.
        :type logc: int or numpy.ndarray
        :returns: ``a * c`` where ``LOG[c] == logc``.
        :rtype: numpy.ndarray
        """
        return self.EXP[self.LOG[a] + logc]

    def div(self, a, b) -> np.ndarray:
        """Element-wise quotient.

        :param a: numerators.
        :type a: array_like
        :param b: denominators, all non-zero.
        :type b: array_like
        :returns: ``a / b``.
        :rtype: numpy.ndarray
        :raises ZeroDivisionError: if any entry of ``b`` is zero.
        """
        b = np.asarray(b)
        if np.any(b == 0):
            raise ZeroDivisionError("division by zero in GF(2^m)")
        return self.EXP[self.LOG[a] - self.LOG[b] + self.order]

    def inv(self, a) -> np.ndarray:
        """Element-wise multiplicative inverse.

        :param a: non-zero field elements.
        :type a: array_like
        :returns: ``1 / a``.
        :rtype: numpy.ndarray
        :raises ZeroDivisionError: if any entry of ``a`` is zero.
        """
        a = np.asarray(a)
        if np.any(a == 0):
            raise ZeroDivisionError("inverse of zero in GF(2^m)")
        return self.EXP[self.order - self.LOG[a]]

    def pow(self, a, e: int) -> np.ndarray:
        """Element-wise power with a non-negative integer exponent.

        :param a: field elements.
        :type a: array_like
        :param int e: exponent, ``e >= 0`` (``0 ** 0`` is taken to be 1).
        :returns: ``a ** e``.
        :rtype: numpy.ndarray
        """
        a = np.asarray(a)
        la = self.LOG[a].astype(np.int64)
        out = self.EXP[(la * int(e)) % self.order]
        if e == 0:
            return np.ones_like(out)
        return np.where(a == 0, 0, out).astype(self.dtype)

    # ------------------------------------------------------ scalar operations
    def smul(self, a: int, b: int) -> int:
        """Scalar product.

        :param int a: field element.
        :param int b: field element.
        :returns: ``a * b``.
        :rtype: int
        """
        return self._exp[self._log[a] + self._log[b]]

    def sdiv(self, a: int, b: int) -> int:
        """Scalar quotient.

        :param int a: numerator.
        :param int b: denominator, non-zero.
        :returns: ``a / b``.
        :rtype: int
        :raises ZeroDivisionError: if ``b`` is zero.
        """
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(2^m)")
        return self._exp[self._log[a] - self._log[b] + self.order]

    def sinv(self, a: int) -> int:
        """Scalar inverse.

        :param int a: non-zero field element.
        :returns: ``1 / a``.
        :rtype: int
        :raises ZeroDivisionError: if ``a`` is zero.
        """
        if a == 0:
            raise ZeroDivisionError("inverse of zero in GF(2^m)")
        return self._exp[self.order - self._log[a]]

    def slog(self, a: int) -> int:
        """Sentinel logarithm of a scalar (log(0) = 2(q-1)).

        :param int a: field element.
        :returns: the discrete logarithm of ``a`` to the base of the generator, or
            ``2*(q-1)`` for ``a == 0``.
        :rtype: int
        """
        return self._log[a]

    # ------------------------------------------------------------ reference
    def clmul(self, a: int, b: int) -> int:
        """Bitwise carry-less multiply modulo the field polynomial (test oracle).

        Independent of the log/exp tables, so it can be used to check them.

        :param int a: field element.
        :param int b: field element.
        :returns: ``a * b``.
        :rtype: int
        """
        r = 0
        while b:
            if b & 1:
                r ^= a
            b >>= 1
            a <<= 1
            if a & self.q:
                a ^= self.poly
        return r
