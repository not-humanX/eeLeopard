"""Reed-Solomon codes encoded and decoded with the LCH FFT.

:class:`FFTRSCode`
    the (n = 2^m, k = n - 2^t) code of the paper: systematic encoder of
    eq. (8) and the erasure-and-error decoder of Sec. 4.
:class:`ReedSolomon`
    any (n, k) code, obtained from the former by shortening and puncturing
    (punctured parity symbols are decoded as erasures).

Decoding returns a :class:`DecodeResult` or raises :class:`DecodeFailure`.

.. rubric:: Working in the normalised basis

The paper states the decoder in the basis X and calls FFT_X, which is Algorithm 1
(an FFT in Xbar) with a diagonal rescaling.  This implementation stays in Xbar,
so ``X_k`` is replaced by ``Xbar_k`` and ``f_t`` by ``fbar_t = Xbar_{2^t}``.  With
that choice:

* the syndrome ``sbar`` (quotient of ``rbar`` by ``Xbar_k``, eq. 11) is exactly the
  XOR of the coset IFFTs of the received blocks (the eq. 8 computation), and
* the key equation keeps its form, but because ``f_m = abar_m + C Xbar_k fbar_t``
  with ``C = f_t(b_t) prod_{l=t}^{m-1} f_l(b_l) = f_t(b_t) / f_t'``, the error
  evaluator is ``q* = C q``.  Error and erasure values therefore carry a factor 1/C.

.. rubric:: Where the paper assumes f_t'(x) = 1

``f_t`` is linearised, so ``f_t'(x)`` is the constant ``prod_{l<t} f_l(b_l)``; and
``prod_{l<m} f_l(b_l) = f_m' = 1``.  f_t' = 1 holds for Cantor bases (all
f_l(b_l) = 1) and for t <= 1 when b_0 = 1, but not in general.  The paper uses
f_t' = 1 twice:

(a) Step 1 obtains sbar "by the encoding scheme of eq. (8)".  In basis X this
    yields ``prod_{l>=t} f_l(b_l) sbar = sbar / f_t'``, which rescales every error
    and erasure value.  (In Xbar, as here, the shortcut is exact and C above
    accounts for the constants.)
(b) Eq. (44) differentiates ``f_t qbar^g = sbar gamma - sbar^g`` and drops f_t'.
    The correct value, used below, is

        qbar^g(w) = (sbar(w) gamma'(w) + (sbar^g)'(w)) / fbar_t'    for w in V_t.

The test-suite exercises erasures inside V_t with random bases to cover this.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .fwht import ErasureLocatorEvaluator
from .gf import GF2m
from .keyeq import solve_gke_euclid, solve_gke_fast, DEFAULT_LEAF_LOG
from .lch import LCHBasis, degree


class DecodeFailure(Exception):
    """The received word is not within the decoding radius (2v + f <= n - k).

    Raised by ``decode`` instead of returning a wrong word: a decode that returns
    always yields a codeword.  Also raised when more than ``n - k`` erasures are
    given, and, in erasure-only mode, when a symbol outside the erasures is wrong.
    """


@dataclass
class DecodeResult:
    """Outcome of a successful decode.

    :ivar numpy.ndarray codeword: the corrected codeword (``uint16``, length ``n``).
    :ivar numpy.ndarray message: the ``k`` message symbols (the tail of
        ``codeword`` for systematic codes).
    :ivar numpy.ndarray error_positions: positions of the corrected errors
        (``int64``).
    :ivar numpy.ndarray error_values: error magnitudes at those positions, i.e.
        ``received ^ codeword`` there.
    :ivar numpy.ndarray erasure_positions: the erasure positions that were given,
        sorted and de-duplicated.
    :ivar numpy.ndarray erasure_values: ``received ^ codeword`` at the erasure
        positions (the "corrections"; the received symbols there are ignored).
    :ivar str solver: ``"fast"`` or ``"euclid"`` (the key-equation solver used),
        or ``"erasure-only"`` for ``errors=False``.
    :ivar dict timings: seconds spent per step, with keys ``step1_syndrome``,
        ``step2_key_equation``, ``step3_roots`` and ``step4_values`` (steps that
        were skipped may be missing).
    """
    codeword: np.ndarray
    message: np.ndarray
    error_positions: np.ndarray
    error_values: np.ndarray
    erasure_positions: np.ndarray
    erasure_values: np.ndarray
    solver: str = ""
    timings: dict = field(default_factory=dict)   # seconds per decoding step

    @property
    def num_errors(self) -> int:
        """Number of corrected errors (``len(error_positions)``)."""
        return int(self.error_positions.size)

    @property
    def num_erasures(self) -> int:
        """Number of erasures (``len(erasure_positions)``)."""
        return int(self.erasure_positions.size)


def _as_positions(positions, n: int) -> np.ndarray:
    """Sorted unique int64 positions from None, a sequence/set/array of integers,
    or a boolean mask of length n."""
    if positions is None:
        return np.zeros(0, dtype=np.int64)
    if isinstance(positions, (set, frozenset)):
        positions = sorted(positions)
    arr = np.asarray(positions)
    if arr.dtype == np.bool_:
        if arr.shape != (n,):
            raise ValueError(f"a boolean erasure mask must have shape ({n},)")
        return np.flatnonzero(arr).astype(np.int64)
    if arr.size and arr.dtype.kind not in "iu":
        raise TypeError("erasure positions must be integers (or a boolean mask)")
    arr = np.unique(arr.astype(np.int64).reshape(-1))
    if arr.size and (arr[0] < 0 or arr[-1] >= n):
        raise ValueError(f"erasure positions must be in [0, {n})")
    return arr


def _pad(a: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=a.dtype)
    out[:min(a.size, length)] = a[:length]
    return out


class FFTRSCode:
    """The (n = 2^m, k = 2^m - 2^t) Reed-Solomon code over GF(2^m) of the paper.

    Codeword ``c_i = u(w_i)`` for a message polynomial u of degree < k, i.e. the
    evaluation code on every field element (ordered by eq. 2).  With
    ``systematic=True`` (default) the parity occupies positions ``0 .. 2^t - 1``
    (the block c_0) and the message the remaining k positions, as in eq. (8);
    otherwise the message symbols are the coefficients u_i of u = sum u_i X_i in
    the paper's (unnormalised) basis X, exactly as in eq. (6).

    The code corrects any ``v`` errors and ``f`` erasures with
    ``2*v + f <= n - k``.

    :param int m: the code lives over GF(2^m); the length is ``n = 2**m``.
    :param int t: the redundancy is ``n - k = 2**t``, ``0 <= t < m``.
    :param basis: GF(2)-basis ``(b_0, ..., b_{m-1})`` that defines the point
        numbering ``w_i``; default ``b_j = 2**j``.
    :type basis: sequence[int] or None
    :param poly: primitive polynomial of the field (default: see
        ``gf.DEFAULT_POLYS``).
    :type poly: int or None
    :param bool systematic: systematic (parity first, eq. 8) or non-systematic
        (eq. 6) encoding.
    :param str solver: default key-equation solver: ``"fast"`` (O(d log^2 d)
        order basis), ``"euclid"`` (the paper's Step 2 via the classical extended
        Euclidean algorithm) or ``"auto"`` (currently ``"fast"``).
    :param int leaf_log: the fast solver switches to its iterative algorithm for
        ``2**leaf_log`` points or fewer.
    :raises ValueError: for ``t`` outside ``[0, m)``, an unknown ``solver``, a
        negative ``leaf_log``, or an invalid field/basis.

    :ivar GF2m gf: the field.
    :ivar LCHBasis lch: the basis tables and FFT.
    :ivar int m: field exponent.
    :ivar int t: redundancy exponent.
    :ivar int n: code length ``2**m``.
    :ivar int d: number of parity symbols ``n - k = 2**t``.
    :ivar int k: number of message symbols ``n - 2**t``.
    :ivar bool systematic: layout flag as above.
    """

    def __init__(self, m: int, t: int, *, basis=None, poly=None, systematic: bool = True,
                 solver: str = "auto", leaf_log: int = DEFAULT_LEAF_LOG):
        self.gf = gf = GF2m(m, poly)
        self.lch = lch = LCHBasis(gf, basis)
        if not (0 <= t < m):
            raise ValueError(f"need 0 <= t < m, got t={t}, m={m}")
        self._check_solver(solver)
        if int(leaf_log) < 0:
            raise ValueError("leaf_log must be >= 0")
        self.m, self.t = m, t
        self.n = 1 << m
        self.d = 1 << t                      # n - k, the number of parity symbols
        self.k = self.n - self.d
        self.systematic = bool(systematic)
        self.solver = solver
        self.leaf_log = int(leaf_log)
        # f_m(x) = abar_m(x) + C * Xbar_k(x) * fbar_t(x), deg abar_m <= k
        C = lch.Wd[t]
        for l in range(t, m):
            C = gf.smul(C, lch.Wd[l])
        self.C = C
        self.Dt = lch.DER[t]                 # fbar_t'(x), a constant
        self._gamma_eval = None
        self._block_offsets = np.arange(self.n >> t, dtype=np.int64) << t
        self._p_log = None                   # log p_i, p_i = prod_j f_j(b_j)^{i_j} (non-systematic mode)

    @staticmethod
    def _check_solver(solver):
        if solver not in ("auto", "fast", "euclid"):
            raise ValueError(f"solver must be 'auto', 'fast' or 'euclid', got {solver!r}")

    def _x_scaling_logs(self) -> np.ndarray:
        """log p_i for i < k, where X_i = p_i Xbar_i (eq. 4 vs. Algorithm 1's basis)."""
        if self._p_log is None:
            gf = self.gf
            p = np.ones(self.n, dtype=gf.dtype)
            for j in range(self.m):
                h = 1 << j
                p[h:2 * h] = gf.mulc(p[:h], self.lch.Wd[j])
            self._p_log = gf.LOG[p[:self.k]]
        return self._p_log

    def __repr__(self) -> str:
        return (f"FFTRSCode(n={self.n}, k={self.k}, GF(2^{self.m}), "
                f"{'systematic' if self.systematic else 'non-systematic'})")

    @property
    def max_errors(self) -> int:
        """Errors correctable without erasures, ``(n - k) // 2``."""
        return self.d // 2

    # ------------------------------------------------------------ encoding
    def encode(self, message) -> np.ndarray:
        """Encode ``k`` symbols.

        Systematic: eq. (8), O(n log(n-k)); the parity goes to positions
        ``0 .. n-k-1`` and the message is copied after it.  Non-systematic: one
        n-point FFT of the message coefficients, O(n log n).

        :param message: ``k`` field elements.
        :type message: array_like of int
        :returns: the codeword (``uint16``, length ``n``).
        :rtype: numpy.ndarray
        :raises ValueError: if ``message`` does not have ``k`` symbols or has
            values outside ``[0, 2**m)``.
        """
        gf, lch = self.gf, self.lch
        msg = gf.asarray(message).reshape(-1)
        if msg.size != self.k:
            raise ValueError(f"message must have {self.k} symbols, got {msg.size}")
        if not self.systematic:
            # u(x) = sum u_i X_i(x) = sum (u_i p_i) Xbar_i(x);  c = FFT(ubar)   (eq. 6)
            coeffs = np.zeros(self.n, dtype=gf.dtype)
            coeffs[:self.k] = gf.mul_log(msg, self._x_scaling_logs())
            return lch.fft(coeffs, self.m, 0)
        t = self.t
        blocks = msg.reshape(-1, self.d)                          # c_1, ..., c_{n/2^t - 1}
        partial = lch.ifft(blocks, t, self._block_offsets[1:])    # IFFT_X(c_i, t, w_{i 2^t})
        c0p = np.bitwise_xor.reduce(partial, axis=0)              # c'_0
        c0 = lch.fft(c0p, t, 0)                                   # c_0 = FFT_X(c'_0, t, w_0)
        return np.concatenate([c0, msg])

    def extract_message(self, codeword) -> np.ndarray:
        """Recover the message from a codeword (no error correction).

        :param codeword: a length-``n`` codeword.
        :type codeword: array_like of int
        :returns: the ``k`` message symbols: the tail of the codeword if systematic,
            else the coefficients ``u_i`` obtained by one inverse FFT.
        :rtype: numpy.ndarray
        """
        c = self.gf.asarray(codeword).reshape(-1)
        if self.systematic:
            return c[self.d:].copy()
        gf = self.gf
        ubar = self.lch.ifft(c, self.m, 0)[:self.k]
        return gf.EXP[gf.LOG[ubar] - self._x_scaling_logs() + gf.order]   # u_i = ubar_i / p_i

    # ------------------------------------------------------------ syndrome
    def syndrome(self, received) -> np.ndarray:
        """The syndrome ``sbar`` (eq. 11).

        The top ``2**t`` Xbar-coefficients of ``rbar = IFFT(y)``, computed as the XOR
        of the ``2**t``-point IFFTs of the ``n / 2**t`` blocks of the word,
        O(n log(n-k)).  It is zero exactly for codewords.

        :param received: a length-``n`` word.
        :type received: array_like of int
        :returns: ``n - k`` field elements.
        :rtype: numpy.ndarray
        """
        y = self.gf.asarray(received).reshape(-1, self.d)
        parts = self.lch.ifft(y, self.t, self._block_offsets)
        return np.bitwise_xor.reduce(parts, axis=0)

    def is_codeword(self, word) -> bool:
        """Whether a word is a codeword (zero syndrome).

        :param word: a length-``n`` word.
        :type word: array_like of int
        :rtype: bool
        """
        return not self.syndrome(word).any()

    # ------------------------------------------------------------ decoding
    def _gamma(self, erasures) -> np.ndarray:
        if self._gamma_eval is None:
            self._gamma_eval = ErasureLocatorEvaluator(self.lch)
        return self._gamma_eval(erasures)

    def _pick_solver(self, solver):
        s = self.solver if solver is None else solver
        self._check_solver(s)
        if s == "auto":
            s = "fast"
        return s

    def decode(self, received, erasures=None, *, solver: str | None = None,
               errors: bool = True) -> DecodeResult:
        """Erasure-and-error decoding (Section 4).

        Corrects any v errors and f erasures with ``2*v + f <= n - k``;
        otherwise raises :class:`DecodeFailure` (a returned word is always a
        codeword within that radius).  Time: O(n log n + (n-k) log^2 (n-k)).

        ``errors=False`` switches to erasure-only decoding: the received word is
        assumed to be wrong only at the ``erasures`` (up to ``n - k`` of them).
        The error locator is then lambda = 1, so the key-equation solver (Step 2)
        and the root search (Step 3) are skipped and the whole decode is
        O(n log n).  If some unflagged symbol is wrong, the degree check
        ``deg(sbar^g) < f`` fails and :class:`DecodeFailure` is raised (errors
        are detected, not corrected).

        :param received: the received word, ``n`` field elements.  The values
            at the erased positions are ignored.
        :type received: array_like of int
        :param erasures: positions known to be unreliable: a sequence, set or
            array of integers in ``[0, n)`` (duplicates are ignored), or a
            boolean mask of length ``n``; ``None`` for none.
        :type erasures: array_like or set or None
        :param solver: key-equation solver for this call, ``"fast"``,
            ``"euclid"`` or ``"auto"``; ``None`` uses the one given to the
            constructor.  Ignored when ``errors=False``.
        :type solver: str or None
        :param bool errors: ``False`` for erasure-only decoding, as above.
        :returns: the corrected codeword and message, the error and erasure
            positions and values, and per-step timings.
        :rtype: DecodeResult
        :raises DecodeFailure: if the word is outside the decoding radius
            (including more than ``n - k`` erasures, or, with
            ``errors=False``, any error outside the erasures).
        :raises ValueError: if ``received`` does not have ``n`` symbols, an
            erasure position is out of range, or ``solver`` is unknown.
        :raises TypeError: if ``erasures`` is not made of integers.
        """
        gf, lch = self.gf, self.lch
        n, d, t = self.n, self.d, self.t
        y = gf.asarray(received).reshape(-1)
        if y.size != n:
            raise ValueError(f"received word must have {n} symbols, got {y.size}")
        Ef = _as_positions(erasures, n)
        f = int(Ef.size)
        if f > d:
            raise DecodeFailure(f"{f} erasures exceed the {d} parity symbols")
        solver = self._pick_solver(solver) if errors else "erasure-only"
        empty = np.zeros(0, dtype=np.int64)
        empty_v = np.zeros(0, dtype=gf.dtype)

        clock = time.perf_counter
        tm = {}
        t0 = clock()

        # ---------------- Step 1: syndrome sbar and generalised syndrome sbar^g
        s = self.syndrome(y)
        if f == 0 and not s.any():
            tm["step1_syndrome"] = clock() - t0
            return self._result(y.copy(), empty, empty_v, Ef, empty_v, solver, tm)
        s_vt = lch.fft(s, t, 0)                                   # sbar on V_t
        if f:
            G = self._gamma(Ef)            # gamma(w) off E_f, gamma'(w) on E_f (Appendix)
            gam_vt = G[:d].copy()
            gam_vt[Ef[Ef < d]] = 0         # gamma vanishes on erasures inside V_t
            S = gf.mul(s_vt, gam_vt)       # sbar^g on V_t (Lemma 3)
            sg = lch.ifft(S, t, 0)         # sbar^g = sbar gamma mod fbar_t
        else:
            G, S, sg = None, s_vt, s
        t1 = clock(); tm["step1_syndrome"] = t1 - t0

        # ---------------- Step 2: error locator from the GKE (33)
        if not errors:
            # no errors: lambda = 1, zbar = sbar^g, qtilde = 0; the GKE degree bound
            # deg zbar < f is then the test that the word really has no other errors
            lam, v = np.ones(1, dtype=gf.dtype), 0
            if degree(sg) >= f:
                raise DecodeFailure("uncorrectable: errors outside the erasures "
                                    "(erasure-only decoding)")
            qt = None
        else:
            if solver == "euclid":
                lam, _, _ = solve_gke_euclid(lch, sg, f, t)
            else:
                lam, _ = solve_gke_fast(lch, S, f, t, self.leaf_log)
            v = degree(lam)
            if v < 0 or 2 * v + f > d:
                raise DecodeFailure("uncorrectable: error locator degree exceeds the decoding radius")
            lam = lam[:v + 1]
            P = _pad(lch.mul(sg, lam), 2 * d)       # sbar^g lambda, degree < d + v <= 2d
            z, qt = P[:d], P[d:]                     # zbar = P mod fbar_t, qtilde = P div fbar_t
            if degree(z) >= v + f:
                raise DecodeFailure("uncorrectable: no key-equation solution of the required degree")
        t2 = clock(); tm["step2_key_equation"] = t2 - t1

        # ---------------- Step 3: error locations = roots of lambda (FFT_X over all cosets)
        if v:
            lam_vals = lch.evaluate_everywhere(lam, t)
            roots = np.flatnonzero(lam_vals == 0)
            if roots.size != v:
                raise DecodeFailure("uncorrectable: error locator does not split over the code locators")
            if f and np.intersect1d(roots, Ef, assume_unique=True).size:
                raise DecodeFailure("uncorrectable: error located on an erasure")
        else:
            roots = empty
        t3 = clock(); tm["step3_roots"] = t3 - t2

        # ---------------- Step 4: error values (39) and erasure values (41)
        corr = y.copy()
        C = self.C
        err_vals = empty_v
        if v:
            dlam = lch.derivative(lam)
            qt_r, dlam_r = lch.evaluate_at(np.stack([qt, _pad(dlam, d)]), roots, t)
            den = gf.mulc(dlam_r, C)                               # C lambda'(w)
            if f:
                den = gf.mul(den, G[roots])                        # ... gamma(w)
            err_vals = gf.div(qt_r, den)                           # eq. (39)
            corr[roots] ^= err_vals
        era_vals = empty_v
        if f:
            if qt is None:                                         # erasure-only: qtilde = 0, lambda = 1
                qt_f, lam_f = np.zeros(f, dtype=gf.dtype), np.ones(f, dtype=gf.dtype)
            else:
                qt_f, lam_f = lch.evaluate_at(np.stack([qt, _pad(lam, d)]), Ef, t)
            qg = np.zeros(f, dtype=gf.dtype)                       # qbar^g(w) on E_f
            inside = Ef < d                                        # fbar_t(w) = 0 there
            if (~inside).any():
                out_idx = Ef[~inside]
                sg_o = lch.evaluate_at(sg, out_idx, t)
                qg[~inside] = gf.div(sg_o, lch.PHI[t][out_idx])    # eq. (43)
            if inside.any():
                in_idx = Ef[inside]
                dsg_vt = lch.fft(lch.derivative(sg), t, 0)        # (sbar^g)' on V_t
                num = gf.mul(s_vt[in_idx], G[in_idx]) ^ dsg_vt[in_idx]
                qg[inside] = gf.mulc(num, gf.sinv(self.Dt))        # eq. (44) with fbar_t' restored
            q_f = qt_f ^ gf.mul(qg, lam_f)                         # q = qtilde - qbar^g lambda
            den = gf.mulc(gf.mul(lam_f, G[Ef]), C)                 # C lambda(w) gamma'(w)
            era_vals = gf.div(q_f, den)                            # eq. (41)
            corr[Ef] ^= era_vals
        tm["step4_values"] = clock() - t3
        return self._result(corr, roots, err_vals, Ef, era_vals, solver, tm)

    def _result(self, cw, err_pos, err_vals, era_pos, era_vals, solver, tm) -> DecodeResult:
        return DecodeResult(codeword=cw, message=self.extract_message(cw),
                            error_positions=np.asarray(err_pos, dtype=np.int64),
                            error_values=np.asarray(err_vals, dtype=self.gf.dtype),
                            erasure_positions=np.asarray(era_pos, dtype=np.int64),
                            erasure_values=np.asarray(era_vals, dtype=self.gf.dtype),
                            solver=solver, timings=tm)


class ReedSolomon:
    """An arbitrary (n, k) Reed-Solomon code, 1 <= k < n, derived from the paper's
    (2^m, 2^m - 2^t) code with 2^t = the next power of two >= n - k:

    * *puncturing*: only the first n - k of the 2^t parity symbols are sent; the
      other 2^t - (n - k) are passed to the decoder as erasures;
    * *shortening*: the last (2^m - 2^t) - k message symbols are fixed to zero and
      not sent.

    The result is MDS with minimum distance n - k + 1: it corrects any v errors
    and f erasures with 2v + f <= n - k.  Codewords are laid out as
    ``[parity (n-k) | message (k)]``.

    Field size: the construction needs ``k + 2^t <= 2^m``.  By default the
    smallest such m is used, which can exceed ceil(log2 n) when n - k is not a
    power of two -- e.g. RS(255, 223) and RS(204, 188) live in GF(2^8), but
    RS(255, 200) (2^t = 64, 200 + 64 > 256) needs GF(2^9).  Check ``.m`` (or pass
    ``m=8`` to get an error instead) before storing symbols in bytes.

    :param int n: code length.
    :param int k: number of message symbols, ``1 <= k < n``.
    :param m: field exponent; defaults to the smallest usable value.
    :type m: int or None
    :param kwargs: passed on to :class:`FFTRSCode` (``basis``, ``poly``,
        ``solver``, ``leaf_log``); ``systematic`` must stay ``True``.
    :raises ValueError: if ``k`` and ``n`` are inconsistent, ``m`` is too small
        for this ``(n, k)``, or ``systematic=False`` is requested.

    :ivar FFTRSCode mother: the underlying ``(2**m, 2**m - 2**t)`` code.
    :ivar GF2m gf: the field.
    :ivar int n: code length.
    :ivar int k: message length.
    :ivar int m: field exponent.
    :ivar int t: ``2**t`` is the number of parity symbols of the mother code.
    :ivar int nparity: ``n - k``.
    :ivar int punctured: number of mother parity symbols that are not sent.
    :ivar int shortened: number of mother message symbols fixed to zero.
    """

    def __init__(self, n: int, k: int, m: int | None = None, **kwargs):
        n, k = int(n), int(k)
        if not (1 <= k < n):
            raise ValueError("need 1 <= k < n")
        p = n - k
        t = (p - 1).bit_length()                       # 2^t >= p
        need = k + (1 << t)                            # 2^m >= k + 2^t
        m_min = max(t + 1, (need - 1).bit_length(), 2)
        if m is None:
            m = m_min
        if m < m_min:
            raise ValueError(
                f"RS(n={n}, k={k}) needs GF(2^m) with m >= {m_min}: the mother code has "
                f"2^t = {1 << t} >= n-k parity symbols and needs k + 2^t = {need} <= 2^m")
        kwargs.setdefault("systematic", True)
        if not kwargs["systematic"]:
            raise ValueError("ReedSolomon wraps the systematic mother code")
        self.mother = FFTRSCode(m, t, **kwargs)
        self.gf = self.mother.gf
        self.n, self.k, self.m, self.t = n, k, m, t
        self.nparity = p
        self.punctured = (1 << t) - p
        self.shortened = self.mother.k - k
        self.positions = np.concatenate([np.arange(p), (1 << t) + np.arange(k)]).astype(np.int64)
        self._inverse = np.full(self.mother.n, -1, dtype=np.int64)
        self._inverse[self.positions] = np.arange(n)
        self._punct_pos = np.arange(p, 1 << t, dtype=np.int64)

    def __repr__(self) -> str:
        return (f"ReedSolomon(n={self.n}, k={self.k}, GF(2^{self.m}); mother "
                f"({self.mother.n},{self.mother.k}), punctured {self.punctured}, "
                f"shortened {self.shortened})")

    @property
    def max_errors(self) -> int:
        """Errors correctable without erasures, ``(n - k) // 2``."""
        return self.nparity // 2

    def encode(self, message) -> np.ndarray:
        """Encode ``k`` symbols into ``n``: ``[parity (n-k) | message (k)]``.

        :param message: ``k`` field elements.
        :type message: array_like of int
        :returns: the codeword (``uint16``, length ``n``).
        :rtype: numpy.ndarray
        :raises ValueError: if ``message`` does not have ``k`` symbols or has
            values outside the field.
        """
        msg = self.gf.asarray(message).reshape(-1)
        if msg.size != self.k:
            raise ValueError(f"message must have {self.k} symbols, got {msg.size}")
        full = np.zeros(self.mother.k, dtype=self.gf.dtype)
        full[:self.k] = msg
        return self.mother.encode(full)[self.positions]

    def _embed(self, received) -> np.ndarray:
        y = self.gf.asarray(received).reshape(-1)
        if y.size != self.n:
            raise ValueError(f"received word must have {self.n} symbols, got {y.size}")
        full = np.zeros(self.mother.n, dtype=self.gf.dtype)
        full[self.positions] = y
        return full

    def is_codeword(self, word) -> bool:
        """Whether a word of length ``n`` is a codeword.

        With punctured parity this decodes the missing symbols as erasures,
        so it costs about one erasure-only decode.

        :param word: a length-``n`` word.
        :type word: array_like of int
        :rtype: bool
        """
        full = self._embed(word)
        if not self.punctured:
            return self.mother.is_codeword(full)
        # the unsent parity symbols are unknown: the word is a codeword iff filling
        # them in (erasure decoding) needs no error correction
        try:
            return self.mother.decode(full, self._punct_pos).num_errors == 0
        except DecodeFailure:
            return False

    def decode(self, received, erasures=None, **kw) -> DecodeResult:
        """Erasure-and-error decoding of a word of length ``n``.

        Corrects ``v`` errors and ``f`` erasures with ``2*v + f <= n - k``.  The
        punctured parity symbols are added to the erasures, shortened positions are
        known zeros, and the result is mapped back to the ``n`` transmitted positions.

        :param received: the received word, ``n`` field elements.
        :type received: array_like of int
        :param erasures: erased positions in ``[0, n)`` (sequence, set, array or
            boolean mask of length ``n``), or ``None``.
        :type erasures: array_like or set or None
        :param kw: passed on to :meth:`FFTRSCode.decode` (``solver``, ``errors``).
        :returns: the result with all positions relative to the length-``n`` word.
        :rtype: DecodeResult
        :raises DecodeFailure: if the word is outside the decoding radius, or a
            correction would fall in a shortened position.
        :raises ValueError: if ``received`` does not have ``n`` symbols or an
            erasure position is out of range.
        """
        full = self._embed(received)
        er = _as_positions(erasures, self.n)
        mother_er = np.concatenate([self.positions[er], self._punct_pos])
        res = self.mother.decode(full, mother_er, **kw)
        errs = self._inverse[res.error_positions]
        if (errs < 0).any():                # a "correction" in a shortened position
            raise DecodeFailure("uncorrectable: error located in a shortened position")
        keep = self._inverse[res.erasure_positions] >= 0
        cw = res.codeword[self.positions]
        order = np.argsort(errs, kind="stable")
        return DecodeResult(codeword=cw, message=cw[self.nparity:].copy(),
                            error_positions=errs[order], error_values=res.error_values[order],
                            erasure_positions=self._inverse[res.erasure_positions[keep]],
                            erasure_values=res.erasure_values[keep], solver=res.solver,
                            timings=res.timings)
