# eeleopard: FFT-based erasure-and-error decoding of Reed–Solomon codes

A complete Python/numpy implementation of

> Y. S. Han, C. Chen, S.-J. Lin, B. Bai, **"On fast Fourier transform-based decoding of Reed-Solomon codes"**, *Int. J. Ad Hoc and Ubiquitous Computing* (2021).

# A Note From a Human

This library was written by an LLM. I've looked though the code but haven't vetted it in detail. It appears and tests correct but use at your own risk. Open an issue if you find a problem. PRs are also welcome.

This library only exists because I needed an implementation and could not find one.  If you know of another implementation, especially if it's faster, more mature, or written by humans, let me know by opening an issue and I'll list it in this readme.

The LLM has also added commentary about the paper this is based on. These are not my opinions and I think it's likely there's conventions/subtext/assumptions that the LLM didn't pick up on when it's mentioning "mistakes" it found.

# Overview

The implementation covers the Lin–Chung–Han novel polynomial basis and its $O(n \log n)$ additive FFT over $\mathrm{GF}(2^m)$ (Algorithm 1), the $O(n \log(n-k))$ systematic encoder of eq. (8), and the paper's contribution: the **erasure-and-error decoder** of Section 4. It corrects any $v$ errors and $f$ erasures with $2v + f \le n-k$ in $O\big(n \log n + (n-k)\log^2(n-k)\big)$.

Every step is checked against brute-force definitions and against an independent textbook decoder (291 tests). Implementing it exposed a slip in the paper: in two places it silently assumes $f_t'(x) = 1$. As printed, the decoder is then only correct for special bases. Both places are corrected here; see below.

```
python examples/demo.py        # tour, ends with the (65536, 32768) code: 16384 errors in ~0.13 s (0.9 s without numba)
python -m pytest -q tests      # 291 tests, ~5 s (236 run, 55 skipped, when numba is absent)
python bench.py [--full]       # FFT decoder vs Berlekamp–Massey baseline, numpy vs numba
```

Requires Python ≥ 3.11 and numpy. **numba is optional**: if it is installed (`pip install numba`, or `pip install "eeleopard[numba]"`), the hot loops of `lch.py`, `keyeq.py` and `fwht.py` run as compiled kernels and decoding is roughly 3–7× faster (see [Performance](#performance)); if it is not, the package silently falls back to plain Python + numpy with identical results. pytest is only needed for the tests.

```python
import eeleopard
eeleopard.HAVE_NUMBA        # numba could be imported
eeleopard.numba_enabled()   # kernels are in use right now
eeleopard.set_numba(False)  # switch to the numpy paths (True switches back, if numba is installed)
eeleopard.warmup()          # optional: compile/load every kernel now instead of on first use
```

Setting `EELEOPARD_NUMBA=0` in the environment disables numba for the whole process. Kernels are compiled lazily on first use (a few seconds in total) and cached on disk, so the cost is paid once per installation.

---

## Quick start

```python
import numpy as np
from eeleopard import FFTRSCode, ReedSolomon, DecodeFailure

# The paper's code family: n = 2^m, n - k = 2^t
code = FFTRSCode(m=8, t=5)                  # (256, 224) over GF(2^8)
cw   = code.encode(msg)                     # msg: 224 symbols; parity in cw[:32], message in cw[32:]
res  = code.decode(received, erasures=[3, 17, 90])
res.message, res.codeword, res.error_positions, res.error_values, res.erasure_values
res.timings                                 # seconds spent in Steps 1-4
res  = code.decode(received, erasures=[3, 17, 90], errors=False)   # erasure-only: O(n log n), see §7

# Any (n, k): shortened + punctured, e.g. RS(255, 223), RS(204, 188), RS(528, 514)
rs  = ReedSolomon(255, 223)                 # picks GF(2^8) automatically
out = rs.decode(rs.encode(data) ^ noise, erasures=[...])
```

`decode` raises `DecodeFailure` when the word is outside the decoding radius. **A returned word is always a codeword within the radius**; see [Failure detection](#5-failure-detection).

Symbols are integers in $[0, 2^m)$ and arrays come back as `uint16`. For byte data, use `np.frombuffer(buf, np.uint8)` on the way in and `.astype(np.uint8)` on the way out, but only when the field is $\mathrm{GF}(2^8)$. `ReedSolomon(n, k)` needs $k + 2^{\lceil \log_2(n-k) \rceil} \le 2^m$. RS(255, 223) and RS(204, 188) fit in $\mathrm{GF}(2^8)$, but RS(255, 200) does not and gets $\mathrm{GF}(2^9)$, so check `rs.m`, or pass `m=8` to get an error instead.

`FFTRSCode(m, t, basis=[...])` accepts any $\mathrm{GF}(2)$-basis $b_0, \dots, b_{m-1}$. The default is $b_j = 2^j$, so $\omega_i$ is simply the field element with integer value $i$. With `systematic=False`, the message is the coefficient vector of $u(x) = \sum_i u_i X_i(x)$ in the paper's basis $\mathbb{X}$, exactly as in eq. (6).

---

## Paper → code map

| Paper | What | Code |
|---|---|---|
| eqs. (1)–(4) | subspaces $V_k$, subspace polynomials $f_k$, basis $\mathbb{X}$ / normalised $\bar{\mathbb{X}}$ | `lch.LCHBasis` (tables `W`, `PHI`, `DER`, `FBAR_MONO`) |
| Algorithm 1 | FFT in $\bar{\mathbb{X}}$ on any coset $V_k + \beta$, and its inverse | `LCHBasis.fft`, `LCHBasis.ifft` (batched, per-row coset offsets) |
| eq. (8) | systematic encoder, $O(n \log(n-k))$ | `FFTRSCode.encode` |
| eq. (11), Step 1 | syndrome $\bar s$ = the quotient of $\bar r$ by $X_k$, computed with the eq. (8) machinery | `FFTRSCode.syndrome` |
| Appendix, eqs. (45)–(48) | $\gamma(\omega)$ for all $\omega$ (and $\gamma'$ on the erasures) via two FWHTs | `fwht.ErasureLocatorEvaluator` |
| Lemma 3 | generalised syndrome $\bar s^{g} = \bar s\gamma \bmod f_t$ from $2^t$ points | `FFTRSCode.decode`, Step 1 |
| eq. (33), Step 2 | generalised key equation → $\lambda$ | `keyeq.solve_gke_euclid` (eqs. 34–36 literally), `keyeq.solve_gke_fast` |
| Step 3 | roots of $\lambda$ via FFTs on all cosets of $V_t$, $O(n \log(n-k))$ | `LCHBasis.evaluate_everywhere` |
| Step 4, eqs. (39), (41), (43), (44) | error and erasure values (Forney-like, evaluator $q$) | `FFTRSCode.decode`, Step 4 |
| (Lin et al. 2016a) | formal derivative in $\bar{\mathbb{X}}$ in $O(n \log n)$ | `LCHBasis.derivative` |

Helpers that are not in the paper: `LCHBasis.mul` (FFT polynomial product), `to_monomial` / `from_monomial` ($O(n \log^2 n)$ basis conversion), `evaluate_at` (FFT only on the cosets containing the requested points), and `reference.classical_decode` (textbook decoder used as the oracle and baseline).

---

## Corrections and engineering decisions

### 1. The paper assumes $f_t'(x) = 1$ in two places

$f_t$ is linearised, so its derivative is a constant:

$$f_t'(x) = \prod_{l<t} f_l(b_l).$$

Because $f_m(x) = x^{2^m} - x$, we also have $\prod_{l<m} f_l(b_l) = f_m' = 1$. $f_t' = 1$ holds in special cases: always with a *Cantor basis*, where every $f_l(b_l) = 1$ and hence $\mathbb{X} = \bar{\mathbb{X}}$, and for $t \le 1$ when $b_0 = 1$. It does not hold in general; with the standard basis of $\mathrm{GF}(2^8)$, $f_t' = 6, 35, 112, 37$ for $t = 2, \dots, 5$. The paper relies on $f_t' = 1$ twice.

**(a) Step 1: the shortcut for $\bar s$.** Step 1 says $\bar s$ can be obtained "by the encoding scheme of eq. (8)", i.e. as the sum of the $2^t$-point coset IFFTs of the received blocks. That is exact in the normalised basis $\bar{\mathbb{X}}$. In basis $\mathbb{X}$ it returns $\prod_{l \ge t} f_l(b_l)\cdot\bar s = \bar s / f_t'$. (The other route Step 1 mentions, the top block of the full $n$-point $\mathrm{IFFT}_{\mathbb{X}}$, is exact.) The error locator is unaffected, but the evaluator is scaled, so every error and erasure value from eqs. (39)/(41) comes out divided by $f_t'$.

**(b) Eq. (44).** Erasure values need $\bar q^{g}(\omega)$ at each erased $\omega$. For erasures in $V_t$ (the first $2^t$ positions, which is the *parity block* in the systematic layout), $f_t(\omega) = 0$. The paper then differentiates $f_t \bar q^{g} = \bar s\gamma - \bar s^{g}$ but drops the constant $f_t'$. Written in the normalised basis used here, where the same identity holds with $\bar f_t$, the correct formula is

$$\bar q^{g}(\omega) = \frac{\bar s(\omega)\,\gamma'(\omega) + (\bar s^{g})'(\omega)}{\bar f_t'} \qquad \text{for } \omega \in V_t \cap E_f.$$

This package works in $\bar{\mathbb{X}}$, so (a) doesn't arise. The remaining constants are exact: $C = f_t(b_t)/f_t'$ in the error/erasure values (§2) and $\bar f_t' = \prod_{l<t} f_l(b_l) / f_t(b_t)$ in (b). The table below is measured in $\mathrm{GF}(2^8)$ with 200 random patterns per cell. "Paper as printed" is Step 1 via eq. (8) plus eq. (44), both in basis $\mathbb{X}$.

| basis | $t$ | $f_t'$ | errors only: paper as printed | errors + erasures ($\ge 1$ in parity block): paper as printed | … with only (a) fixed | this package |
|---|---|---|---|---|---|---|
| standard | 0 | 1 | 200/200 | 200/200 | 200/200 | 200/200 |
| standard | 1 | 1 | 200/200 | 200/200 | 200/200 | 200/200 |
| standard | 2 | 6 | **0/200** | **0/200** | **0/200** | 200/200 |
| standard | 3 | 35 | **0/200** | **0/200** | **0/200** | 200/200 |
| standard | 4 | 112 | **0/200** | **0/200** | **0/200** | 200/200 |
| standard | 5 | 37 | **0/200** | **0/200** | **0/200** | 200/200 |
| Cantor | 0–5 | 1 | 200/200 | 200/200 | 200/200 | 200/200 |

The Cantor basis used is (1, 214, 152, 146, 86, 200, 88, 230) for the field polynomial 0x11D. `test_generalised_syndrome_and_key_equation` checks the corrected eq. (44) against $\bar q^{g}$ computed as a polynomial, `test_product_of_all_f_l_b_l_is_one` checks the identity behind (a), and `test_cantor_basis_makes_paper_constants_trivial` checks the Cantor case.

### 2. Working in the normalised basis $\bar{\mathbb{X}}$ throughout

The paper states the decoder in $\mathbb{X}$ and calls $\mathrm{FFT}_{\mathbb{X}}$, which is Algorithm 1 (an FFT in $\bar{\mathbb{X}}$) with a diagonal rescaling by $p_i$. This package never leaves $\bar{\mathbb{X}}$, so $X_k$ becomes $\bar X_k$ and $f_t$ becomes $\bar f_t = \bar X_{2^t}$. Two consequences:

* The syndrome is **exactly** the XOR of the coset IFFTs of the received blocks: in $\bar{\mathbb{X}}$ the top coefficient of an interpolant on a coset is the XOR of the values. The same fact is what makes eq. (8) work.
* The key equation keeps its form. However, $f_m = \hat a_m + C \cdot \bar X_k \cdot \bar f_t$ with

  $$C = f_t(b_t)\prod_{l=t}^{m-1} f_l(b_l) = \frac{f_t(b_t)}{f_t'},$$

  so the evaluator is $q^{*} = C q$, and eqs. (39)/(41) pick up a factor $1/C$. The proof that $\deg \hat a_m \le k$ (used for $\deg \bar z < v + f$) goes through by induction on $m$.

### 3. Step 2: the key-equation solver

The paper hands Step 2 to "the half-GCD algorithm given in Lin et al. (2016b)" without restating it. Two solvers are provided, and they are interchangeable via `solver=`:

* **`"euclid"`** is Step 2 exactly as written. It divides $f_t$ by $\bar s^{g}$ (giving $\bar q_t$ and $\bar r_t$), runs the extended Euclidean algorithm on $(\bar s^{g}, \bar r_t)$ while keeping the cofactor matrix of eq. (34), and returns $\bar\lambda = \bar u_1 - \bar v_1 \bar q_t$ and $\tilde q = \bar v_1$ (eqs. 35–36). It stops at the first remainder of degree $< (d+f)/2$, where $d = n-k$. The paper doesn't state this stopping rule; it is the one that makes the solution unique whenever $2v + f \le d$. Polynomial division has no cheap form in $\bar{\mathbb{X}}$, so this path converts to the monomial basis. It is quadratic in $n-k$.
* **`"fast"`** (the default) runs in $O\big((n-k)\log^2(n-k)\big)$ and matches the paper's complexity claim. It is the *dual* of the half-GCD, chosen because it never divides and so suits $\bar{\mathbb{X}}$. $\bar f_t$ splits into the linear factors $(x - \omega)$, $\omega \in V_t$, so GKE (33) is equivalent to the rational-interpolation problem

  $$\lambda(\omega)\,\bar s^{g}(\omega) = z(\omega) \qquad \text{for all } \omega \in V_t.$$

  Its minimal solution is the minimal row of a $2 \times 2$ *order basis* in shifted-reduced form (shift $(f, 1)$). The order basis is built by divide and conquer over the same coset tree the FFT walks, $V_t = V_{t-1} \cup (V_{t-1} + b_{t-1})$. Cosets are evaluated with Algorithm 1 and $2 \times 2$ matrix products use half-size FFTs plus one analytically known top coefficient. Leaves of 64 points use the iterative Beckermann–Labahn algorithm.

When $2v + f \le n-k$ both solvers return the same $\lambda$ up to a scalar, and `test_solvers_agree` checks this. The fast solver pulls ahead from $n-k \approx 1000$ upward; at $n-k = 32768$ it took 0.8 s against 8.4 s in numpy when both were measured.

### 4. Lemma 2 does not hold as written

The paper remarks (Lemma 2) that $c(\alpha^j) = \sum_i c_i \alpha^{ij}$ vanishes for $j = 1, \dots, n-k$, so a "traditional" Berlekamp–Massey decoder could be used on these codewords. For the $\omega_i$ ordering used here it does not: a random (256, 224) codeword gives $c(\alpha), c(\alpha^2), \dots = 36, 73, 218, \dots$. These codes are extended (length $2^m$) generalised RS codes whose Lagrange weights $\prod_{j \ne i}(\omega_i - \omega_j) = f_m'(\omega_i)$ are all 1, so the correct parity checks are the power sums

$$\sum_i c_i\,\omega_i^{\,j} = 0 \qquad \text{for } 0 \le j < n-k.$$

`reference.py` uses these. The FFT decoder never relies on Lemma 2.

### 5. Failure detection

The paper does not discuss decoding failure. The decoder raises `DecodeFailure` unless all of the following hold:

* $2\deg\lambda + f \le n-k$;
* $\deg \bar z < \deg\lambda + f$;
* $\lambda$ has exactly $\deg\lambda$ distinct roots among the code locators;
* those roots are disjoint from the erasures.

When the checks pass, the correction provably has zero syndrome: the same Lagrange/key-equation argument run in reverse gives $(\bar s_\varepsilon - \bar s)\Lambda$ with degree $< \deg\Lambda$. So the output is always a codeword within the radius. `test_beyond_radius_matches_classical_decoder` confirms that, beyond the radius, the FFT decoder and the textbook decoder agree exactly: either both fail, or both return the same codeword within the radius.

### 6. Arbitrary $(n, k)$

The paper requires $n = 2^m$ and $n-k = 2^t$. `ReedSolomon(n, k)` builds on the $(2^m, 2^m - 2^t)$ mother code with $2^t \ge n-k$ in two ways:

* *Puncturing*: it sends only $n-k$ of the $2^t$ parity symbols. The unsent ones are decoded as erasures, which is exactly where erasure-and-error decoding pays off.
* *Shortening*: it fixes unused message symbols to zero.

The result is MDS with $d = n-k+1$ and corrects $2v + f \le n-k$. An error located in a shortened position is reported as a failure.

### 7. Erasure-only mode: `decode(..., errors=False)`

If the positions of all wrong symbols are known, the key-equation solver (Step 2) and the root search (Step 3) are pure overhead: with no errors the error locator is $\lambda = 1$, so the key equation collapses to

$$\bar z = \bar s^{g}, \qquad \tilde q = 0,$$

and Step 4's erasure values (41) need nothing beyond $\bar s^{g}$ and $\gamma'$. The decode then costs only the syndrome ($O(n\log(n-k))$), the two Walsh–Hadamard transforms for $\gamma$ ($O(n\log n)$), and a few FFTs: $O(n\log n)$ in total, the complexity of the erasure decoder the paper builds on (Lin–Chung–Han, 2016). It corrects up to $f \le n-k$ erasures, where the full decoder corrects $2v + f \le n-k$.

The key-equation degree bound becomes the error test. With $\lambda = 1$ it reads $\deg \bar s^{g} < f$, and `decode` raises `DecodeFailure` when it fails, i.e. when some symbol outside the erasures is wrong. Errors are **detected, not corrected**:

* if $v$ symbols outside the erasure set are wrong and $v + f \le n-k$, the failure is guaranteed. A wrong answer would be a second codeword within distance $v + f$ of the transmitted one, but the code is MDS with minimum distance $n-k+1$;
* whatever is returned is always a codeword that agrees with the received word off the erasures (same argument as in §5).

`ReedSolomon.decode` accepts the same flag; its punctured parity symbols are just additional erasures. The result's `solver` field reads `"erasure-only"`.


---

## Performance

Measured on one core of an Intel Xeon @ 2.1 GHz (Python 3.12.3, numpy 2.4.4, numba 0.67.0) with `python bench.py --full`. Two words are decoded per code. The **fast** column is the full decoder on a word at full radius, with $f = (n-k)/4$ erasures and $v = 3(n-k)/8$ errors. The **erasure-only** column is a second word with $n-k$ erasures and no errors, decoded with `errors=False` (§7). *classic* is the textbook baseline on the first word: power-sum syndromes, Berlekamp–Massey, Chien search and Forney, vectorised numpy (it does not use the FFT code, so numba does not change it). Times are in seconds, best of a few runs; numba's one-off compile time is excluded.

**Pure Python + numpy** (what you get without numba):

| code | $v$ | $f$ | encode | **fast** | erasure-only | classic | speed-up |
|---|---:|---:|---:|---:|---:|---:|---:|
| (256, 240) | 6 | 4 | 0.0000835 | 0.000884 | 0.000334 | 0.000459 | 0.5× |
| (256, 224) | 12 | 8 | 0.0000972 | 0.00115 | 0.000421 | 0.000835 | 0.7× |
| (256, 128) | 48 | 32 | 0.000122 | 0.00256 | 0.000540 | 0.00330 | 1.3× |
| (4096, 4064) | 12 | 8 | 0.000214 | 0.00167 | 0.000765 | 0.00278 | 1.7× |
| (4096, 3840) | 96 | 64 | 0.000281 | 0.00598 | 0.00117 | 0.0218 | 3.6× |
| (65536, 65472) | 24 | 16 | 0.00231 | 0.00998 | 0.00676 | 0.0607 | 6.1× |
| (65536, 65280) | 96 | 64 | 0.00244 | 0.0162 | 0.00797 | 0.233 | 14.5× |
| (65536, 64512) | 384 | 256 | 0.00271 | 0.0400 | 0.00946 | 0.928 | 23× |
| (65536, 61440) | 1536 | 1024 | 0.00319 | 0.0953 | 0.0109 | 3.89 | 41× |
| (65536, 57344) | 3072 | 2048 | 0.00324 | 0.168 | 0.0124 | 9.36 | 56× |
| (65536, 49152) | 6144 | 4096 | 0.00432 | 0.392 | 0.0179 | 19.7 | 50× |
| (65536, 32768) | 12288 | 8192 | 0.00424 | 0.726 | 0.0202 | 45.5 | 63× |

**With numba** (same words, same machine). "vs numpy" is the time above divided by the time in this table, and the last column is *classic* from the table above divided by *fast* here:

| code | $v$ | $f$ | encode | **fast** | vs numpy | erasure-only | vs numpy | speed-up vs classic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (256, 240) | 6 | 4 | 0.0000128 | 0.000276 | 3.2× | 0.000111 | 3.0× | 1.7× |
| (256, 224) | 12 | 8 | 0.0000133 | 0.000288 | 4.0× | 0.000132 | 3.2× | 2.9× |
| (256, 128) | 48 | 32 | 0.0000135 | 0.000421 | 6.1× | 0.000147 | 3.7× | 7.8× |
| (4096, 4064) | 12 | 8 | 0.0000349 | 0.000376 | 4.4× | 0.000219 | 3.5× | 7.4× |
| (4096, 3840) | 96 | 64 | 0.0000419 | 0.000896 | 6.7× | 0.000300 | 3.9× | 24× |
| (65536, 65472) | 24 | 16 | 0.000571 | 0.00303 | 3.3× | 0.00234 | 2.9× | 20× |
| (65536, 65280) | 96 | 64 | 0.000725 | 0.00441 | 3.7× | 0.00278 | 2.9× | 53× |
| (65536, 64512) | 384 | 256 | 0.000822 | 0.00864 | 4.6× | 0.00340 | 2.8× | 108× |
| (65536, 61440) | 1536 | 1024 | 0.000892 | 0.0158 | 6.0× | 0.00408 | 2.7× | 247× |
| (65536, 57344) | 3072 | 2048 | 0.000978 | 0.0258 | 6.5× | 0.00498 | 2.5× | 363× |
| (65536, 49152) | 6144 | 4096 | 0.00122 | 0.0582 | 6.7× | 0.00808 | 2.2× | 339× |
| (65536, 32768) | 12288 | 8192 | 0.00107 | 0.104 | 7.0× | 0.0108 | 1.9× | 438× |

Encoding is $O(n \log(n-k))$ and takes 2–4 ms even at $n = 65536$ in numpy, and 0.6–1.2 ms with numba.

Notes on the numbers:

* numba pays off most where numpy's per-call overhead and temporaries dominated: the leaves of the order-basis solver (a scalar loop over 64 points, run $2^{t-6}$ times) and the many small FFTs. For the (65536, 32768) decode the key-equation step goes from 882 ms to 113 ms (7.8×), the syndrome and erasure locator (step 1) from 15 ms to 5.1 ms, and the error and erasure values (step 4) from 28 ms to 8.8 ms (a separate profiling run, so these do not add up exactly to the table).
* **Erasure-only decoding skips the key-equation solver**, which is almost all of the cost for large $n-k$: at (65536, 32768) it takes 20 ms in numpy against 726 ms for the full decoder (36×), and 11 ms with numba against 104 ms (9.6×). For every $n = 65536$ code in the table it takes 7–20 ms in numpy and 2.3–11 ms with numba, growing slowly with $n-k$, as $O(n \log n)$ predicts. Numba speeds it up 1.9–2.9×. At $n-k = 64$ the two modes are within 1.5× of each other, because the solver is cheap there.
* For tiny $\mathrm{GF}(2^8)$ codes numpy call overhead used to make the textbook decoder faster (0.5× for (256, 240)). With numba the FFT decoder wins on every row of the table, even at $n = 256$.
* What is left after compiling: at (65536, 65472) an erasure-only decode takes about 2.7 ms with numba, of which 1.6 ms is the two length-65536 Walsh–Hadamard transforms of the erasure locator, which are memory-bound rather than compute-bound, and 0.6 ms the syndrome. Everything outside `lch.py`, `keyeq.py` and `fwht.py` (the `gf.py` lookups and the glue in `rs.py`) is still numpy.

---

## Layout

```
eeleopard/
  gf.py         GF(2^m), 2 ≤ m ≤ 16: log/exp tables with a zero sentinel (branch-free multiply)
  lch.py        novel basis, Algorithm 1 FFT/IFFT, derivative, products, conversions, evaluation
  fwht.py       Appendix: erasure locator via fast Walsh–Hadamard transforms
  keyeq.py      GKE solvers: Euclid (eqs. 34–36) and the fast order-basis solver
  _accel.py     optional numba kernels for the hot loops of lch.py, keyeq.py and fwht.py (numpy fallback in place)
  rs.py         FFTRSCode (paper's codes: encoder + Steps 1–4), ReedSolomon (any n, k)
  reference.py  textbook syndrome/BM/Chien/Forney decoder (oracle + baseline)
tests/          291 tests (brute-force FFT checks, random bases, all t, beyond-radius agreement, numba vs numpy, …)
examples/demo.py
bench.py
```

## Tests

`python -m pytest -q tests` covers:

* field arithmetic against carry-less multiplication, for every $m$ from 2 to 16;
* Algorithm 1 against the product definition of $\bar X_i$, for every size, random coset offsets and random bases;
* basis conversions, the derivative and the FFT product against monomial arithmetic;
* the FWHT erasure locator against direct products, including $\gamma'$ on the erasures;
* the encoder (every codeword has $\deg < k$ and zero syndrome, and encoding is linear);
* the syndrome against the top block of the full $n$-point IFFT;
* Lemma 3, eq. (43), the corrected eq. (44), GKE (33) with the true locators, the identity $\prod_l f_l(b_l) = 1$, and the Cantor-basis case;
* agreement of the two solvers, at leaf sizes 1 through 64;
* end-to-end decoding across 12 $(m, t)$ pairs × 2 solvers × standard/random basis, including $v = (n-k)/2$, $f = n-k$ and erasures concentrated in the parity block;
* beyond-radius agreement with the reference decoder;
* general $(n, k)$ codes, including non-power-of-two redundancy and the field-size rule;
* the eq. (6) non-systematic map in basis $\mathbb{X}$, and input handling (erasures as lists, sets, arrays or boolean masks; several received dtypes; inputs never modified);
* erasure-only mode (`tests/test_erasure_only.py`): exact recovery for every erasure count up to $n-k$ (erasures anywhere, in the parity block, or in the message), agreement with the full decoder, guaranteed detection of errors when $v + f \le n-k$, output always a codeword, a check that the key-equation solver and root search are never called, and general $(n, k)$ codes;
* the numba kernels (`tests/test_accel.py`, skipped if numba is absent): every kernel (including the Walsh–Hadamard transform and erasure locator) and every end-to-end decode (both solvers, erasure-only mode, standard/random bases, leaf sizes 1 to 64, failures beyond the radius) is compared bit for bit with the numpy path, and subprocess tests check that the package imports and decodes correctly when numba is missing or `EELEOPARD_NUMBA=0`. The rest of the suite passes in both modes.

## API documentation

The docstrings of the public classes and functions list every parameter, return value and exception using Sphinx field lists (`:param:`, `:type:`, `:returns:`, `:rtype:`, `:raises:`, `:ivar:`), so `sphinx.ext.autodoc` renders them as they are, without `napoleon`. A minimal `conf.py` needs only `extensions = ["sphinx.ext.autodoc"]` and the project directory on `sys.path`, and an `index.rst` with `.. automodule:: eeleopard.rs` (and `:members:`) and so on for each module.

## Limitations

* The field size is capped at $2 \le m \le 16$: elements are `uint16` and the tables grow as $m \cdot 2^m$.
* The paper's codes use every field element as a locator, so $n = 2^m$, and the FFT always runs over $2^m$ points. `ReedSolomon(n, k)` therefore costs about the same as its mother code; for example, RS(528, 514) runs over $\mathrm{GF}(2^{10})$ with $n = 1024$.
* Without numba the package is pure numpy, so constant factors are Python's. With numba the FFT, key-equation and erasure-locator kernels are compiled, but the `gf.py` lookups and the glue in `rs.py` are still numpy, and everything runs on one thread. A C/Rust port would be the way to get line-rate speeds.
* The first call that touches each numba kernel compiles it (a few seconds for all of them, cached afterwards; `eeleopard.warmup()` does it on demand). If the cache directory is not writable numba recompiles in every process.

## License

MIT; see [LICENSE](LICENSE).
