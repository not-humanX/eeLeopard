"""eeleopard -- FFT-based erasure-and-error decoding of Reed-Solomon codes.

A Python/numpy implementation of

    Y. S. Han, C. Chen, S.-J. Lin, B. Bai, "On fast Fourier transform-based
    decoding of Reed-Solomon codes", Int. J. Ad Hoc and Ubiquitous Computing.

built on the Lin-Chung-Han novel polynomial basis and its O(n log n) FFT over
GF(2^m).  numba is optional and speeds up the hot loops when installed.

.. rubric:: Main entry points

:class:`~eeleopard.rs.FFTRSCode`
    the paper's ``(2^m, 2^m - 2^t)`` code: ``encode``, ``decode``.
:class:`~eeleopard.rs.ReedSolomon`
    any ``(n, k)`` code (shortened and punctured).
:class:`~eeleopard.rs.DecodeResult`, :class:`~eeleopard.rs.DecodeFailure`
    what ``decode`` returns and raises.
:func:`~eeleopard._accel.set_numba`
    switch the compiled kernels on or off.

.. rubric:: Example

.. code-block:: python

    from eeleopard import FFTRSCode
    code = FFTRSCode(m=8, t=5)                 # (256, 224) over GF(2^8)
    cw = code.encode(message)                  # parity in cw[:32], message in cw[32:]
    result = code.decode(received, erasures=[3, 17, 90])
    result.message, result.error_positions
"""
from ._accel import HAVE_NUMBA, numba_enabled, set_numba, warmup
from .gf import GF2m
from .lch import LCHBasis, degree
from .fwht import ErasureLocatorEvaluator, wht
from .keyeq import solve_gke_euclid, solve_gke_fast
from .rs import FFTRSCode, ReedSolomon, DecodeResult, DecodeFailure

__all__ = [
    "GF2m", "LCHBasis", "degree", "ErasureLocatorEvaluator", "wht",
    "solve_gke_euclid", "solve_gke_fast",
    "FFTRSCode", "ReedSolomon", "DecodeResult", "DecodeFailure",
    "HAVE_NUMBA", "numba_enabled", "set_numba", "warmup",
]
__version__ = "1.0.0"
