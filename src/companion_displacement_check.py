#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Companion-zero checks for the ORIGINAL indicator  F(x,q) = (q-1) q (S_q(x) - q^{-x}),  q > 1 real
("The Fejer-Dirichlet Lift", Appendix E, Section 'Window inclusions and uniqueness ...').

What this script does (multiprecision, mpmath; it is NOT part of any proof):

  1. Thresholds p0(q) of Theorem 'Uniqueness of the companion zero for large p':
     the least odd p >= 5 for which condition (C) holds.
  2. The example q = 2, p = 101 of Section 4 (Remark 'Newton refinement'): the displacement
     Delta_p(q) = p - x_p(q) of the largest zero in (p-1,p), the closed form Delta^(0) = (log q) q^{-p} / K(q,p),
     the relative deviation, the proved bound 2*kappa of Proposition 'Asymptotic displacement law',
     and two Newton steps.

Usage:   python companion_displacement_check.py
Requires: mpmath
"""
import mpmath as mp
from mpmath import mpf


# ----------------------------------------------------------------------------- constants of the paper
def c1(q):
    """Lower bound for K(q,p) (Lemma 'Uniform positivity with explicit bounds')."""
    q = mpf(q); lam = mp.log(q)
    B = q ** 3 / 2 + 8 * q ** 2 / 27 + q / 4 - mpf(2) / 3
    return (mp.pi ** 2 * B - lam ** 2) / (2 * q ** 5)


def L_const(q):
    q = mpf(q); lam = mp.log(q)
    return (2 * mp.pi) ** 3 / (q * (q - 1)) + lam ** 3 * q ** (-4)


def condition_C(q, p):
    q = mpf(q); lam = mp.log(q)
    tA = (2 / mp.pi) * mp.asin(q ** (mpf(3 - p) / 2))
    tB = mp.atan((5 * mp.pi / 2) * lam) / mp.pi
    t0 = q * lam / (q ** (p - 2) - mp.pi ** 2 / 3) if q ** (p - 2) > mp.pi ** 2 / 3 + q * lam else mpf(1)
    tc = min(tA, tB, t0)
    return 2 * c1(q) > 8 * mp.pi ** 3 * tc / (q * (q - 1)) + lam ** 2 * q ** (-p) * (q ** tc - 1)


def p0(q):
    p = 5
    while not condition_C(q, p):
        p += 2
    return p


# ----------------------------------------------------------------------------- the indicator near a prime p
def phi(i, p, eps):
    """phi_i(p+eps); the integer part p is reduced modulo i first (sin(pi z) is never evaluated near its zeros)."""
    s = mp.sin(mp.pi * eps) ** 2
    r = p % i
    den = i * i * (mp.sin(mp.pi * eps / i) ** 2 if r == 0 else mp.sin(mp.pi * (r + eps) / i) ** 2)
    return s / den


def n_terms(q, p, digits=70):
    lam = mp.log(q)
    return int((2 * p * lam + digits * mp.log(10) + 20) / lam) + 5


def H(q, p, eps, N):
    """H(eps) = S_q(p+eps) - q^{-(p+eps)};  F(p+eps,q) = (q-1) q H(eps)."""
    q = mpf(q)
    tot, qi = mpf(0), q ** (-2)
    for i in range(2, N + 1):
        tot += qi * phi(i, p, eps)
        qi /= q
    return tot - q ** (-(p + eps))


def a_b(q, p, N):
    """a = H'(0) = (log q) q^{-p},  b = H''(0)/2 = K(q,p)."""
    q = mpf(q); lam = mp.log(q)
    S2, qi = mpf(0), q ** (-2)
    for i in range(2, N + 1):
        if p % i == 0:
            d2 = -(2 * mp.pi ** 2 / 3) * (1 - mpf(1) / i ** 2)
        else:
            d2 = 2 * mp.pi ** 2 / (i * i * mp.sin(mp.pi * (p % i) / i) ** 2)
        S2 += qi * d2
        qi /= q
    return lam * q ** (-p), (S2 - lam ** 2 * q ** (-p)) / 2


def example(q=2, p=101):
    mp.mp.dps = int(2 * p * float(mp.log10(q)) + 100)
    N = n_terms(mpf(q), p)
    a, b = a_b(q, p, N)
    rho = a / b                                   # Delta^(0)
    kappa = L_const(q) * a / (6 * b ** 2)
    lo, hi = -rho * (1 + 2 * kappa), -rho * (1 - kappa)   # H(lo) > 0 > H(hi) by the Proposition
    assert H(q, p, lo, N) > 0 and H(q, p, hi, N) < 0
    for _ in range(400):
        mid = (lo + hi) / 2
        if H(q, p, mid, N) > 0:
            lo = mid
        else:
            hi = mid
    Delta = -(lo + hi) / 2
    print(f"q = {q}, p = {p}:  K(q,p) = {mp.nstr(b, 12)},  Delta^(0) = {mp.nstr(rho, 15)}")
    print(f"  exact Delta = {mp.nstr(Delta, 18)};  relative deviation Delta/Delta^(0) - 1 = {mp.nstr(Delta / rho - 1, 4)}"
          f"  (proved bound 2*kappa = {mp.nstr(2 * kappa, 4)})")
    x = -rho
    for step in (1, 2):
        x = x - H(q, p, x, N) / mp.diff(lambda e: H(q, p, e, N), x)
        print(f"  Newton step {step}: relative error {mp.nstr(abs((-x) / Delta - 1), 4)}")


if __name__ == "__main__":
    mp.mp.dps = 50
    print("Thresholds p0(q) of Theorem 'Uniqueness of the companion zero for large p':")
    for q in ("1.001", "1.01", "1.1", "1.5", "2", "3", "5", "20", "1000"):
        print(f"  q = {q:>6}:  p0 = {p0(mpf(q))}")
    example(2, 101)
