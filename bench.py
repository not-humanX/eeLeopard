"""Benchmark: FFT-based decoding (this paper) vs a textbook syndrome decoder,
with and without the optional numba kernels.

    python bench.py            # GF(2^8), GF(2^12) and GF(2^16) codes up to n-k = 4096
    python bench.py --full     # also n-k = 8192, 16384 and 32768 at n = 65536, incl. the classical run
    python bench.py --numpy    # skip the numba column even if numba is installed
    python bench.py --only 16,15   # a single code: n = 2^16, n-k = 2^15 (repeatable)

Each row decodes received words of the same code:
  fast         -- errors and erasures at the decoding radius (v errors + f erasures,
                  2v + f = n - k, f = (n-k)/4), O(n log n + d log^2 d) key-equation solver
  erasure-only -- n - k erasures and no errors, decoded with errors=False, which skips
                  the key-equation solver and the root search: O(n log n)
  classic      -- the first word again, with power-sum syndromes + Berlekamp-Massey +
                  Chien + Forney (reference.py)
Encode, fast and erasure-only are measured twice, in pure numpy and with numba
(lch.py / keyeq.py kernels), in the same process; "gain" is the numpy time divided by
the numba time and "speed-up" is classic / fast (both for numpy and numba).
Times are the best of a few runs, in seconds; numba's one-off compile time is
excluded (kernels are warmed up first).
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eeleopard  # noqa: E402
from eeleopard import FFTRSCode  # noqa: E402
from eeleopard.reference import classical_decode  # noqa: E402


def best_of(fn, reps):
    best, out = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def sig(x, width=9):
    """Seconds to 3 significant figures."""
    return f"{x:{width}.{max(0, 2 - math.floor(math.log10(x)))}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--numpy", action="store_true", help="pure numpy only (no numba column)")
    ap.add_argument("--only", metavar="M,T", action="append",
                    help="run only the code with n = 2^M, n - k = 2^T (repeatable), e.g. --only 16,15")
    args = ap.parse_args()
    rng = np.random.default_rng(1)

    use_nb = eeleopard.HAVE_NUMBA and not args.numpy
    if use_nb:
        import numba
        t0 = time.perf_counter()
        eeleopard.set_numba(True)
        eeleopard.warmup()
        print(f"numba {numba.__version__}: kernels ready in {time.perf_counter() - t0:.1f} s "
              f"(compile time is excluded from the timings)")
    else:
        why = "disabled by --numpy" if args.numpy else "not installed / disabled"
        print(f"numba {why}: pure numpy only")

    cases = [(8, 4), (8, 5), (8, 7), (12, 5), (12, 8), (16, 6), (16, 8), (16, 10), (16, 12)]
    if args.full:
        cases += [(16, 13), (16, 14), (16, 15)]
    if args.only:
        cases = [tuple(int(x) for x in o.split(",")) for o in args.only]
        args.full = True                       # --only lifts the size limit below
    classic_limit = 1 << 31 if args.full else 1 << 26   # skip classical if n*d is larger

    if use_nb:
        hdr = (f"{'code':>16} {'v':>6} {'f':>6} | {'enc np':>9} {'enc nb':>9} | "
               f"{'fast np':>9} {'fast nb':>9} {'gain':>6} | {'eras np':>9} {'eras nb':>9} {'gain':>6} | "
               f"{'classic':>9} | {'speed np':>8} {'speed nb':>8}")
    else:
        hdr = (f"{'code':>16} {'v':>6} {'f':>6} | {'encode':>9} | {'fast':>9} {'eras-only':>9} {'classic':>9} | "
               f"{'speed-up':>8}")
    print(hdr)
    print("-" * len(hdr))
    for m, t in cases:
        code = FFTRSCode(m, t)
        n, k, d = code.n, code.k, code.d
        reps = 3 if n * d <= (1 << 24) else 1
        msg = rng.integers(0, n, k)
        f = d // 4
        v = (d - f) // 2
        pos = rng.choice(n, v + f, replace=False)
        er = pos[v:]
        pos_e = rng.choice(n, d, replace=False)         # erasure-only word: n - k erasures, no errors

        def measure(numba_on):
            eeleopard.set_numba(numba_on)
            te, cw = best_of(lambda: code.encode(msg), reps)
            rx = cw.copy()
            rx[pos[:v]] ^= rng.integers(1, n, v).astype(rx.dtype)
            rx[pos[v:]] = 0
            tf, res = best_of(lambda: code.decode(rx, er), reps)
            assert np.array_equal(res.codeword, cw)
            rx_e = cw.copy()
            rx_e[pos_e] = 0
            tq, res = best_of(lambda: code.decode(rx_e, pos_e, errors=False), reps)
            assert np.array_equal(res.codeword, cw)
            return te, tf, tq, rx

        te, tf, tq, rx = measure(False)
        if use_nb:
            te2, tf2, tq2, _ = measure(True)
            eeleopard.set_numba(False)
        tc = None
        if n * d <= classic_limit:
            tc, res = best_of(lambda: classical_decode(code, rx, er), 1 if n * d > (1 << 22) else reps)
            assert np.array_equal(res.codeword, code.encode(msg))

        fmt = lambda x: sig(x) if x is not None else f"{'-':>9}"
        gain = lambda a, b: f"{a / b:5.1f}x" if a is not None and b is not None else f"{'-':>6}"
        speed = lambda a, b: f"{a / b:7.1f}x" if a is not None else f"{'-':>8}"
        label = f"({n},{k})"
        if use_nb:
            print(f"{label:>16} {v:6d} {f:6d} | {fmt(te)} {fmt(te2)} | "
                  f"{fmt(tf)} {fmt(tf2)} {gain(tf, tf2)} | {fmt(tq)} {fmt(tq2)} {gain(tq, tq2)} | "
                  f"{fmt(tc)} | {speed(tc, tf)} {speed(tc, tf2)}", flush=True)
        else:
            print(f"{label:>16} {v:6d} {f:6d} | {fmt(te)} | {fmt(tf)} {fmt(tq)} {fmt(tc)} | "
                  f"{speed(tc, tf)}", flush=True)


if __name__ == "__main__":
    main()
