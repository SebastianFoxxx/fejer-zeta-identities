# The Fejér–Dirichlet Lift: Entire Functions and ζ-Factorization Identities

**Author:** Sebastian Fuchs  
**Email:** [sebastian.fuchs@hu-berlin.de](mailto:sebastian.fuchs@hu-berlin.de)  
**Location:** Berlin, Germany  
**Affiliation:** Humboldt-Universität zu Berlin  
**GitHub:** [SebastianFoxxx/fejer-zeta-identities](https://github.com/SebastianFoxxx/fejer-zeta-identities)  
**Paper on arXiv:** [https://arxiv.org/abs/2509.12297](https://arxiv.org/abs/2509.12297)  
**ORCID:** [0009-0009-1237-4804](https://orcid.org/0009-0009-1237-4804)  
**DOI:** [10.5281/zenodo.17122709](https://doi.org/10.5281/zenodo.17122709)  
**Date:** 2025-09-15  
**Version:** 1.0.0  

---
## Abstract

A Fejér–Dirichlet lift is developed that turns divisor information at the integers into entire interpolants with explicit Dirichlet–series factorizations. For absolutely summable weights the lift interpolates $(a*1)(n)$ at each integer $n$ and has Dirichlet series $\zeta(s)A(s)$ on $\Re s>1$. Two applications are emphasized. First, for $q>1$ an entire function $\mathfrak F(\cdot,q)$ is constructed that vanishes at primes and is positive at composite integers; for a tangent-matched variant $\mathfrak F^{\sharp}$ it is proved that, for every $q>1$, the real zeros in $[3,\infty)$ are exactly the primes, each of multiplicity two. The proof is short: a sampling identity identifies the periodic normalizer $S_1$ as the (principal-value) first moment of the $\operatorname{sinc}^2$ kernel, and convexity of the exponential yields an explicit positive lower bound between the integers. Second, a renormalized lift for $a=\mu*\Lambda$ produces an entire interpolant of $\Lambda(n)$ and provides a constructive viewpoint on the appearance of $\zeta'(s)/\zeta(s)$ through the FD-lift spectrum. A Polylog–Zeta factorization for the geometric-weight case links $\zeta(s)$ with ${Li}_s(1/q)$. Prime/composite statements concern integer arguments; off the integers only statements about real zeros are made. Scripts reproducing figures and numerical checks are provided in a public repository with an archival snapshot.

## Introduction and Overview

The Fejér kernel  
$F(z,i) = i + 2\sum_{k=1}^{i-1}(i-k)\cos\left(\frac{2\pi k z}{i}\right) = \left(\frac{\sin(\pi z)}{\sin(\pi z/i)}\right)^2$  
is a finite even trigonometric polynomial whose sine-quotient form has only removable poles. Superpositions of $F(\cdot,i)$ with rapidly decaying weights act as **divisor filters** at integer points: the value $F(n,i)/i^2$ equals $1$ if $i\mid n$ and $0$ otherwise. This yields the **Fejér–Dirichlet lift**  
$T_a(z) = \sum_{i\ge1} a(i)\,\frac{F(z,i)}{i^2}$  
which interpolates $(a*1)(n)$ at integers and admits the Dirichlet–series factorization  
$\sum T_a(n)n^{-s} = \zeta(s)A(s)$ for $\Re s>1$.  
A central specialization chooses geometric weights to form a **prime-indicator** $\mathfrak{F}(z,q)$ ($q>1$), whose integer values vanish at primes and are positive at composites. The raw indicator may admit a **left companion zero** inside $(p-1,p)$; this is removed by a **tangent-matched normalizer**  
$q^{-x}\left(1+(\log q)\,S_1(x)\right),\qquad S_1(x) = \frac{\sin(2\pi x)}{2\pi},$  
producing $\mathfrak{F}^{\sharp}(z,q)$, whose real zeros in $[3,\infty)$ are exactly the primes (each a double zero) for every $q>1$. The proof bounds every Fejér filter from below by its $\operatorname{sinc}^2$ sampling profile, identifies the normalizer as the termwise linearisation of the resulting formal, divergent sampling series (partial fractions of $\csc^2$ and $\cot$), and uses that the exponential lies above its tangent; the parity filter $i=2$ compensates the missing part of the lattice. The threshold $x=3$ cannot be lowered uniformly in $q$: whenever the curvature term $K^\sharp(q,2)$ is nonzero, an additional real zero lies in $(1,2)$ (if it is positive) or in $(2,3)$ (if it is negative).

A **renormalized** variant of the lift yields an entire interpolant of the von Mangoldt function by taking $a=\mu*\Lambda$, and a two-variable version differentiates to recover the classical identity involving $\tau$, $\Lambda$, and $\zeta\,\zeta'$. Polylogarithm–$\zeta$ identities arise naturally from the lift; for $q=-1$ they are interpreted in the Abel sense through the Dirichlet eta function, while for $q<-1$ a weighted alternating $\eta_Q$ appears. Analyticity for $|q|>1$ follows by uniform convergence on compact sets (Weierstrass M-test). Prime/composite claims are statements about values at integer arguments; off the integers only real-zero statements are made.

## Key Contributions

1.  **Fejér–Dirichlet lift (interpolation and factorization):** An explicit entire function $T_a$ (type $\le 2\pi$) is constructed for weights $a$ with mild summability, satisfying $T_a(n) = (a*1)(n)$ and $\sum T_a(n)n^{-s} = \zeta(s)A(s)$ for $\Re s>1$.
2.  **Prime-indicator family and tangent matching:** A geometric-weight construction $\mathfrak{F}(z,q)$ yields prime zeros and composite positivity at integers. A tangent-matched variant $\mathfrak{F}^{\sharp}(z,q)$ with periodic normalizer matches value and slope at integers; for every $q>1$ its **real zeros in $[3,\infty)$ are exactly the primes**, each of multiplicity two, with the explicit lower bound $\mathfrak{F}^{\sharp}(x,q)\ge\frac{2}{5\pi^2}(q-1)q\,q^{-x}\sin^2(\pi x)$ for non-integer $x>3$.
3.  **Quantitative companion-zero analysis (original indicator):** For $\mathfrak{F}$ (without tangent matching), the existence of a left companion zero in $(p-1,p)$ for odd primes $p\ge5$ is shown and its displacement obeys a sharp law $\Delta_p(q)\asymp q^{-p}$, with an explicit asymptotic formula involving a curvature term $K(q,p)$.
4.  **Renormalized lift for $a=\mu*\Lambda$:** An entire interpolant of $\Lambda(n)$ is obtained; a two-variable formulation differentiates to recover the standard identity relating $\tau$, $\Lambda$, and $\zeta\,\zeta'$.
5.  **Polylogarithm–$\zeta$ factorization and alternating regimes:** Polylog-$\zeta$ identities are derived for the lift. At $q=-1$, Dirichlet–series identities hold in the **Abel sense** through the Dirichlet eta function $\eta(s)$; for $q=-Q<-1$, analogues involve a weighted alternating $\eta_Q$.
6.  **Analyticity and explicit bounds:** Entireness for fixed $|q|>1$ is proved by local uniform convergence. The positivity proof for $\mathfrak{F}^{\sharp}$ uses only elementary tools (partial fractions of $\csc^2$ and $\cot$, convexity of the exponential, elementary series bounds) and yields the absolute constant $2/(5\pi^2)$; the curvature terms $K(q,p)$ and $K^\sharp(q,p)$ at primes $p\ge5$ are bounded above and below by positive constants independent of $p$.

> All assertions about primes and composites refer to values at integer arguments, and off the integers only real-zero statements (such as the theorem for $\mathfrak{F}^{\sharp}$ on $[3,\infty)$) are made; complex-plane morphology (e.g., zero sets and domain coloring) is discussed for intuition but is not required for any of the results.

---

## Repository Structure

```
.
├── figures/                  # Generated figures (PDF/JPG)
├── src/
│   ├── real_zero_verifier.py # Numerical cross-check for F♯ on prime windows
│   └── plot_generator.py     # Reproduces all figures from the paper
├── verification_results/     # JSON/CSV outputs from the verifier
├── README.md
├── LICENSE                   # MIT License
└── .gitignore
```
* `src/real_zero_verifier.py` checks integer anchors, window-wise sign conditions and the curvature at primes numerically.
* `src/plot_generator.py` reproduces all illustrative plots.

---

## How to Run the Scripts

All scripts are located in the `src/` directory and should be run from the repository's root directory.

### Reproducibility & Verification

`real_zero_verifier.py` is an independent numerical cross-check for the tangent-matched indicator on the prime windows $(p-1,p)$, $p\ge5$ (optionally also $(2,3)$); the proof in the paper does not depend on it. It uses conservative tail bounds for the series; sign checks are performed on a grid in double precision.

**What the verifier checks:**

* **Integer anchors & tangent matching** at integers $m$.
* **Legacy diagnostics:** existence of an $\alpha\in(0,1/2)$ meeting the window conditions **(L)** and **(M)** of the first version of the paper (no longer needed for the proof; retained for reproducibility).
* **Prime window nonnegativity** on the interval $(p-1,p)$ for odd primes $p \ge 5$.
* **Local curvature at the prime** against a lower bound for $S_q''(p)$ derived from the curvature lemma of the paper.

**Quick Start:**
Minimal run (uses defaults: `q=1.5`, `pmax=500`, `auto-imax` enabled):
```bash
python src/real_zero_verifier.py
```

**Example with custom parameters:**
```bash
# Scan primes up to 251 for q=2.0 with a denser grid and a fixed series cutoff
python src/real_zero_verifier.py --q 2.0 --pmax 251 --grid 800 --no-auto-imax --imax 1200

# Use a tighter tail bound for automatic imax selection
python src/real_zero_verifier.py --q 1.7 --auto-imax --tail-target 1e-15

# Legacy: audit of the (M) condition of the first paper version around q=2.4
python src/real_zero_verifier.py --audit-M-threshold --audit-q-min 2.35 --audit-q-max 2.50
```

**Outputs:**
* A **console summary** for a quick pass/fail overview.
* `verification_results/verifier_report.json`: A detailed JSON report.
* `verification_results/verifier_summary.csv`: A per-prime CSV summary.
* `verification_results/M_threshold_audit.csv`: (If audit is run)

### Reproducing Figures

`plot_generator.py` reproduces the visualizations used in the paper.

**Run:**
```bash
python src/plot_generator.py
```

This will create the `figures/` directory (if it doesn't exist) and save all plots, including:
* `plot_mu_interpolant.pdf` (The "analytic annihilator" $T_\mu$)
* `plot_lambda_interpolant.pdf` (The interpolant for $\Lambda(n)$)
* `plot_operator_effect.pdf` (The effect of the differentiation operator)
* `plot_q_analog_interpolation.pdf` (q-analog curves vs. $\tau(n)$)
* `plot_companion_displacement.pdf` (Decay of companion zero displacement)
* `plot_companion_elimination.pdf` (Comparison of $\mathfrak{F}$ and $\mathfrak{F}^{\sharp}$)
* And all 3D/domain-coloring plots for various `q`.

> **Performance Note:** The 3D and domain-coloring plots are computationally intensive. Expect run times of several minutes. The script uses Numba for JIT compilation, which may add a one-time overhead on the first run.

---

## How to Cite

If you use this work in your research, please cite the paper.

```bibtex
@article{Fuchs2025FejerDirichletLift,
  author  = {Fuchs, Sebastian},
  title   = {The Fej{\'e}r--Dirichlet Lift: Entire Functions and \zeta-Factorization Identities},
  year    = {2025},
  journal = {arXiv preprint arXiv:XXXXXXXX.XXXXX}
}
```

---

## License

The source code in this repository is released under the **MIT License** (see `LICENSE`).
The research paper content is © the author; all rights are reserved.


## Building the paper

**Build from the repository root, not from `paper/`.** The preamble sets
`\graphicspath{{./}{./figures/}}`, and TeX resolves that relative to the *working
directory*. Running `pdflatex` inside `paper/` therefore fails with missing figures and no
useful diagnostic. The build script encodes this:

```bash
python build.py testbuild                 # build into .build/, report the fingerprint
python build.py fingerprint               # store the current fingerprint (.fingerprint.json)
python build.py check                     # rebuild and compare against the stored fingerprint
python build.py release ../my_package     # assemble a flat submission package and verify it
```

`testbuild` never writes into the repository (it uses `-output-directory`).

**The fingerprint** records the page count, every `\label` together with its number, the
number of bibliography entries, undefined references and missing figures. It is the safety
net for structural work: after splitting, moving or reordering the source, `check` must
report *identical*. It is what made the split into `paper/parts/` verifiable.

**`release`** assembles a single self-contained `.tex` (the `\input` parts are inlined
again, so a submission looks exactly as it always did), copies the figures that are actually
referenced, writes a `MANIFEST.json` with an md5 for every source and figure — and then
**builds the package once more inside the package directory** and compares the result. A
package that does not build itself is not a package.

## Source layout

```
paper/fejer_kernel_lift.tex          preamble, front matter, and the \input list
paper/parts/*.tex         one file per section; the appendix and the bibliography separately
figures/                  the figures the paper references (the single canonical copy)
build.py                  the build described above
.fingerprint.json         the reference fingerprint for `build.py check`
```
