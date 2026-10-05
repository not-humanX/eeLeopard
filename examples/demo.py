"""Quick tour of eeleopard.  Run from the project root:  python examples/demo.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eeleopard import FFTRSCode, ReedSolomon, DecodeFailure  # noqa: E402

rng = np.random.default_rng(2021)

# --- 1. The paper's code family: (n = 2^m, k = n - 2^t) over GF(2^m) -----------
code = FFTRSCode(m=8, t=5)                 # (256, 224) over GF(2^8), n - k = 32
print(code)
msg = rng.integers(0, 256, code.k)
cw = code.encode(msg)                      # systematic: parity in cw[:32], message in cw[32:]

rx = cw.copy()
err_pos = rng.choice(256, 10, replace=False)
rx[err_pos] ^= rng.integers(1, 256, 10).astype(rx.dtype)           # 10 errors
erasures = np.sort(rng.choice(np.setdiff1d(np.arange(256), err_pos), 12, replace=False))
rx[erasures] = 0                                                   # 12 erasures (2*10 + 12 = 32)

res = code.decode(rx, erasures)
assert np.array_equal(res.message, msg)
print(f"  corrected {res.num_errors} errors at {res.error_positions.tolist()}")
print(f"  filled {res.num_erasures} erasures; step timings (ms):",
      {k: round(v * 1e3, 2) for k, v in res.timings.items()})

# one error too many -> DecodeFailure (never a codeword outside the radius)
rx2 = rx.copy()
extra = np.setdiff1d(np.arange(256), np.concatenate([err_pos, erasures]))[0]
rx2[extra] ^= 1
try:
    code.decode(rx2, erasures)
    print("  (beyond the radius but still within reach of some codeword)")
except DecodeFailure as e:
    print("  beyond the radius:", e)

# --- 2. Any (n, k): shortened + punctured, e.g. the classic RS(255, 223) ---------
rs = ReedSolomon(255, 223)
print(rs)
data = rng.integers(0, 256, 223)
word = rs.encode(data)
word[[3, 50, 51, 200]] ^= 0x5A                                     # 4 errors
out = rs.decode(word, erasures=[7, 8, 9])                          # + 3 erasures
assert np.array_equal(out.message, data)
print(f"  RS(255,223): fixed errors at {out.error_positions.tolist()}, erasures {out.erasure_positions.tolist()}")

# a non-power-of-two redundancy works too (unsent parity is decoded as erasures)
rs2 = ReedSolomon(528, 514)                                        # GF(2^10), 14 parity symbols
d2 = rng.integers(0, 1024, 514)
w2 = rs2.encode(d2)
w2[rng.choice(528, 7, replace=False)] ^= 1                         # 7 = (528-514)/2 errors
assert np.array_equal(rs2.decode(w2).message, d2)
print(f"  {rs2}: 7 errors corrected")

# --- 3. The (2^16, 2^15) code from the paper's motivation ------------------------
big = FFTRSCode(m=16, t=15)
m_big = rng.integers(0, 65536, big.k)
c_big = big.encode(m_big)
pos = rng.choice(65536, 16384, replace=False)
r_big = c_big.copy()
r_big[pos] ^= rng.integers(1, 65536, 16384).astype(r_big.dtype)    # 16384 errors = (n-k)/2
res = big.decode(r_big)
assert np.array_equal(res.codeword, c_big)
print(f"  {big}: {res.num_errors} errors corrected in "
      f"{sum(res.timings.values()):.2f} s  ({ {k: round(v, 3) for k, v in res.timings.items()} })")
