#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#
#   Numerical Verification for the Paper
#   "The Fejer-Dirichlet Lift: Entire Functions and zeta-Factorization Identities"
#
#   This script implements the numerical verifier for the real-zero structure of F_sharp(·, q) for q > 1.
#   It checks the integer anchors, the existence of a suitable alpha, and the sign pattern on prime windows.
#   It uses conservative lower bounds (partial sums plus a geometric tail) for the Fejér superposition
#   so that all reported window minima are rigorous lower estimates of F_sharp(·, q).
#
#   Author: Sebastian Fuchs | sebastian.fuchs@hu-berlin.de | https://orcid.org/0009-0009-1237-4804
#   Date:   2025-09-14
#   Version: 1.0
#
#   Usage examples:
#   $ python real_zero_verifier.py
#   $ python real_zero_verifier.py --q 1.5 --pmax 101
#   $ python real_zero_verifier.py --q 2.0 --pmax 251 --grid 800 --imax 1200
#   $ python real_zero_verifier.py --q 1.7 --auto-imax --tail-target 1e-15
#   $ python real_zero_verifier.py --audit-M-threshold --audit-q-min 2.35 --audit-q-max 2.50 --audit-q-step 0.005
#
#   Outputs:
#   - JSON summary   : verifier_report.json  (change via --json)
#   - CSV  summary   : verifier_summary.csv  (change via --csv)
#   - Console report : human-readable overview with progress
#   - Optional CSV   : M_threshold_audit.csv (if --audit-M-threshold given)
#
# =============================================================================

from __future__ import annotations
import argparse
import csv
import json
import math
import sys
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# -------------------------- Utilities & numerics ---------------------------

PI = math.pi
TWO_PI = 2.0 * PI


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(n**0.5)
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


def primes_up_to(n: int) -> List[int]:
    """Simple sieve up to n inclusive."""
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:n+1:step] = b"\x00" * ((n - start)//step + 1)
    return [i for i, v in enumerate(sieve) if v]


def frac_part(x: float) -> float:
    """Fractional part in [0,1)."""
    y = math.modf(x)[0]
    return y if y >= 0.0 else y + 1.0


# ------------------------------- Config ------------------------------------

@dataclass
class Config:
    # Core numerics
    q: float = 1.5                  # Default q>1 so script runs without args
    pmax: int = 500                 # Upper bound for primes to check
    imax: int = 1000                # Series cutoff for S(x) (may be overridden if auto_imax=True)
    grid: int = 1000                # Samples per subinterval scan
    tol: float = 1e-12              # Base tolerance used in effective sign tolerance

    # Alpha search
    alpha_min: float = 1e-3
    alpha_max: float = 0.49
    alpha_grid: int = 2000

    # Diagnostics
    anchors_up_to: int = 30

    # Output files
    json_path: str = "../verification_results/verifier_report.json"
    csv_path: str = "../verification_results/verifier_summary.csv"

    # UX
    progress: bool = True           # Enable console progress bar

    # Numerical robustness near p (right subinterval buffer)
    ir_buffer: float = 1e-6         # Minimal IR cut from p (will be adaptively increased)

    # Dynamic imax control
    auto_imax: bool = True          # If True, increase/decrease imax to meet tail_target
    tail_target: float = 1e-14      # Target geometric tail for S(x): sum_{i>imax} q^{-i} <= tail_target

    # Finite-difference safety for S''(p) lower-bound test
    Spp_fd_safety: float = 1e-8     # Subtracted from the paper lower bound when testing numerically

    # (3) (M)-threshold audit options
    audit_M_threshold: bool = False
    audit_q_min: float = 2.35
    audit_q_max: float = 2.50
    audit_q_step: float = 0.005
    audit_csv_path: str = "../verification_results/M_threshold_audit.csv"

    # Special window (2,3) policy:
    #  - "exclude": do not count (2,3) toward pass/fail; print OK (N/A).
    #  - "refine" : re-check (2,3) with tighter tail/grid; print OK (refined) if nonnegative.
    #  - "include": current behaviour (use global imax/grid and show OK/FAIL).
    special_23_policy: str = "exclude"
    special_23_grid: int = 3000
    special_23_tail_target: float = 1e-16

# -------------------------- Fejér kernel machinery ------------------------

def phi_i(x: float, i: int, eps: float = 1e-12) -> float:
    """
    Robust evaluation of phi_i(x) = F(x,i)/i^2 with
        F(x,i) = (sin(pi x) / sin(pi x / i))^2
    and removable poles at x ∈ iZ: the limit equals 1.

    We implement a safe quotient with a small-threshold switch.
    For real x, |phi_i(x)| ≤ 1 holds because |F(x,i)| ≤ i^2.
    """
    t = math.sin(PI * x / i)
    if abs(t) > eps:
        s = math.sin(PI * x)
        val = (s / (i * t)) ** 2
        # Guard against tiny negative due to rounding
        return 0.0 if (val < 0.0 and abs(val) < 1e-16) else val
    # Near removable singularity. Use limiting value 1.
    return 1.0


def _kahan_add(sum_val: float, c: float, term: float) -> Tuple[float, float]:
    """
    One Kahan-compensated addition step:
      y = term - c
      t = sum_val + y
      c = (t - sum_val) - y
      sum_val = t
    Returns (new_sum_val, new_c)
    """
    y = term - c
    t = sum_val + y
    c_new = (t - sum_val) - y
    return t, c_new


def S_value_and_tail(x: float, q: float, imax: int) -> Tuple[float, float]:
    """
    Returns (S_partial(x), tail_upper_bound) with
      S(x) = sum_{i>=2} q^{-i} phi_i(x)
    Tail bound: sum_{i>imax} q^{-i} <= q^{-(imax+1)} / (1 - 1/q).
    Uses Kahan compensated summation to reduce cancellation/roundoff.
    """
    r = 1.0 / q
    val = 0.0
    c = 0.0  # Kahan compensation
    for i in range(2, imax + 1):
        term = (r ** i) * phi_i(x, i)
        y = term - c
        t = val + y
        c = (t - val) - y
        val = t
    tail = (r ** (imax + 1)) / (1.0 - r)
    return val, tail

def Csin_value(x: float, q: float) -> float:
    """
    Calculates the periodic normalizer C_sin(x,q) without the prefactor (q-1)q.
    This is the correct linear form from the paper: q^{-x} * (1 + (log q) * S1(x)).
    """
    logq = math.log(q)
    S1 = math.sin(TWO_PI * x) / TWO_PI
    return (q ** (-x)) * (1.0 + logq * S1)


def Fsharp_bounds(x: float, q: float, imax: int) -> Tuple[float, float]:
    """
    Conservative lower/upper bounds for F_sharp(x,q) using S +/- tail.
    Returns (lower, upper). Note the global prefactor (q-1)q is included.
    """
    S_val, tail = S_value_and_tail(x, q, imax)
    Cval = Csin_value(x, q)
    pref = (q - 1.0) * q
    lower = pref * ((S_val - tail) - Cval)
    upper = pref * ((S_val + tail) - Cval)
    return lower, upper


def Fsharp_value(x: float, q: float, imax: int) -> float:
    """
    Point estimate (not rigorous): use S_val (without +/- tail).
    """
    S_val, _ = S_value_and_tail(x, q, imax)
    Cval = Csin_value(x, q)
    pref = (q - 1.0) * q
    return pref * (S_val - Cval)


# ---------------------------- Alpha search (L/M) --------------------------

def series_sum_q_i2(q: float, imax: int) -> Tuple[float, float]:
    """
    Returns (partial_sum, tail_upper_bound) for sum_{i>=2} q^{-i}/i^2.
    Simple tail bound: for i >= imax+1, 1/i^2 <= 1/(imax+1)^2
    => tail ≤ (1/(imax+1)^2) * sum_{i>=imax+1} q^{-i}.
    """
    r = 1.0 / q
    s = 0.0
    c = 0.0  # Kahan
    for i in range(2, imax + 1):
        term = (r ** i) / (i * i)
        s, c = _kahan_add(s, c, term)
    tail_geom = (r ** (imax + 1)) / (1.0 - r)
    tail = tail_geom / ((imax + 1) ** 2)
    return s, tail


def sigma_lower_analytic(q: float) -> float:
    """
    Analytic lower bound used in the paper for Σ = sum_{i>=2} q^{-i}/i^2:
      Σ ≥ q^{-2}/4 + q^{-3}/9.
    """
    return (q ** -2) / 4.0 + (q ** -3) / 9.0


def alpha_L_from_L_condition(q: float) -> float:
    """
    Alpha suggested by (L): alpha_L(q) = (2/pi) * arccos(q^{-1 + 1/(4π)}).
    Note: This is used for diagnostics and the analytic (M) checker.
    """
    base = q ** (-1.0 + (1.0 / (4.0 * PI)))
    base = min(1.0, max(0.0, base))
    return (2.0 / PI) * math.acos(base)


@dataclass
class AlphaResult:
    alpha: Optional[float]
    L_margin: Optional[float]
    M_margin: Optional[float]
    found: bool
    details: Dict[str, float]
    # (L′) margin and analytic (M) flags/margins
    Lprime_margin: Optional[float] = None
    M_analytic_ok: Optional[bool] = None
    M_analytic_margin: Optional[float] = None
    alpha_L: Optional[float] = None


def find_alpha(cfg: Config) -> AlphaResult:
    """
    Search alpha in (alpha_min, alpha_max) satisfying
      (L) cos^2(pi α / 2) >= q^{-2 + 1/(2π)}
      (M) sin^2(pi α) * Σ_{i≥2} q^{-i}/i^2 >= q^{-(4+α)} q^{1/(2π)}.
    Uses partial sum (lower bound) for the series on the LHS of (M).

    Additionally reports:
      (L′)  cos^2(pi α / 2) - q^{-2 + α},
      Analytic (M) margin using Σ ≥ q^{-2}/4 + q^{-3}/9 and α_L(q).
    """
    q = cfg.q
    imax = cfg.imax
    a_min, a_max, a_grid = cfg.alpha_min, cfg.alpha_max, cfg.alpha_grid

    # Thresholds for (L) and auxiliary quantities
    LHS_L_threshold = q ** (-2.0 + 1.0 / (2.0 * PI))
    partial, _tail_upper = series_sum_q_i2(q, imax)
    sum_qi2_lower = partial  # lower bound for (M)

    best_alpha = None
    best_margin = -1e9
    best_L = None
    best_M = None
    best_Lprime = None

    found = False
    found_res: Optional[AlphaResult] = None

    for j in range(a_grid):
        alpha = a_min + (a_max - a_min) * (j + 0.5) / a_grid
        cos_term = math.cos(PI * alpha / 2.0) ** 2
        L_margin = cos_term - LHS_L_threshold

        # (L′) diagnostic margin
        Lprime_margin = cos_term - (q ** (-2.0 + alpha))

        sin_term = math.sin(PI * alpha) ** 2
        LHS_M_lower = sin_term * sum_qi2_lower
        RHS_M = q ** (-(4.0 + alpha)) * (q ** (1.0 / (2.0 * PI)))
        M_margin = LHS_M_lower - RHS_M

        if L_margin >= 0.0 and M_margin >= 0.0 and not found:
            found = True
            found_res = AlphaResult(
                alpha=alpha,
                L_margin=L_margin,
                M_margin=M_margin,
                found=True,
                details={
                    "LHS_L": cos_term,
                    "RHS_L": LHS_L_threshold,
                    "LHS_M_lower": LHS_M_lower,
                    "RHS_M": RHS_M,
                    "sum_qi2_lower": sum_qi2_lower
                },
                Lprime_margin=Lprime_margin
            )
            # Do not break; we still compute analytic diagnostics below.

        # track best near-feasible score
        score = min(L_margin, M_margin)
        if score > best_margin:
            best_margin = score
            best_alpha = alpha
            best_L = L_margin
            best_M = M_margin
            best_Lprime = Lprime_margin

    # Prepare result (found or best candidate)
    alpha_final = best_alpha if (found_res is None) else found_res.alpha
    result = AlphaResult(
        alpha=alpha_final,
        L_margin=best_L if (found_res is None) else found_res.L_margin,
        M_margin=best_M if (found_res is None) else found_res.M_margin,
        found=(found_res is not None),
        details={
            "LHS_L_threshold": LHS_L_threshold,
            "sum_qi2_lower": sum_qi2_lower
        },
        Lprime_margin=best_Lprime if (found_res is None) else found_res.Lprime_margin
    )

    # Analytic (M) checker for q >= 12/5 using Σ ≥ q^{-2}/4 + q^{-3}/9 and α* = min{α_L(q), 1/2}.
    # This is a strict lower bound independent of imax.
    if q >= 12.0 / 5.0:
        alpha_L = min(alpha_L_from_L_condition(q), 0.5)
        Sigma_lb = sigma_lower_analytic(q)
        sin_sq = math.sin(PI * alpha_L) ** 2
        lhs_M_analytic = sin_sq * Sigma_lb
        rhs_M = q ** (-(4.0 + alpha_L)) * (q ** (1.0 / (2.0 * PI)))
        M_analytic_margin = lhs_M_analytic - rhs_M
        result.M_analytic_ok = (M_analytic_margin >= 0.0)
        result.M_analytic_margin = M_analytic_margin
        result.alpha_L = alpha_L
        # keep original result.alpha as the numeric-search alpha; analytic is additional info.

    return result


# --------------------------- Integer anchor checks ------------------------

def check_integer_anchor(m: int, cfg: Config, h: float = 1e-6) -> Dict[str, float]:
    """
    Verify (diagnostically, non-rigorous for derivatives):
      S(m) = sum_{d|m, d>=2} q^{-d}
      S'(m) ≈ 0 (finite difference)
      C_sin(m,q) = q^{-m}
      C_sin'(m,q) ≈ 0
      F_sharp(m,q) ≥ 0 and equals 0 iff m is prime (up to tiny tol)

    strict tests:
      - |S(m)_approx - S(m)_exact| <= tail_bound_at_m + eps_guard_S
      - If m is prime: |F_sharp(m)| <= tau_anchor
      - If m is composite: pref * ((S(m)_exact - q^{-m}) - tail_bound_at_m) >= -tau_anchor
      - New: C_sin''(m) ≈ 0 (central second difference, diagnostic with ULP-based tolerance)
    """
    q, imax = cfg.q, cfg.imax

    # S and tail at m
    S_m, tail_m = S_value_and_tail(m, q, imax)

    # exact finite sum at integer
    Sm_exact = sum(q ** (-d) for d in range(2, m + 1) if m % d == 0)
    Sm_err = S_m - Sm_exact

    # Derivative S'(m) by central differences (diagnostic)
    S_p, _ = S_value_and_tail(m + h, q, imax)
    S_mi, _ = S_value_and_tail(m - h, q, imax)
    S_prime = (S_p - S_mi) / (2.0 * h)

    # Corrector and its derivatives (diagnostic)
    C_m = Csin_value(m, q)            # should be q^{-m}
    C_p = Csin_value(m + h, q)
    C_mi = Csin_value(m - h, q)
    C_prime = (C_p - C_mi) / (2.0 * h)
    C_pp = (C_p - 2.0 * C_m + C_mi) / (h * h)

    # F_sharp(m,q)
    pref = (q - 1.0) * q
    Fm = pref * (S_m - C_m)

    # --------- strict tests ---------
    # Floating-point guard tailored to this anchor scale
    r = 1.0 / q
    sum_qi_cap = (r * r) / (1.0 - r)   # upper bound on Σ_i>=2 q^{-i}
    scale_F = pref * (sum_qi_cap + q ** (-m))
    eps_guard_S = 50.0 * np.finfo(float).eps * max(1.0, abs(Sm_exact))
    eps_guard_F = 50.0 * np.finfo(float).eps * max(1.0, scale_F)

    # anchor-specific sign tolerance: tau_anchor = tol + pref * tail_m + eps_guard_F
    tau_anchor = cfg.tol + pref * tail_m + eps_guard_F

    # S anchor test
    S_anchor_pass = (abs(Sm_err) <= (tail_m + eps_guard_S))
    S_anchor_margin = (tail_m + eps_guard_S) - abs(Sm_err)

    # C_sin exactness (diagnostic tight test, relative)
    C_rel_err = abs(C_m - (q ** (-m))) / max(1.0, q ** (-m))
    C_anchor_pass = (C_rel_err <= 50.0 * np.finfo(float).eps * 10.0)  # generous ULP-based threshold

    # C_sin'' diagnostic vs exact second derivative: C''(m) = -(log q)^2 * q^{-m}
    Cpp_expected = - (math.log(q) ** 2) * (q ** (-m))
    Cpp_err = abs(C_pp - Cpp_expected)
    # Tolerance scaled by h^2 and by magnitudes to reflect truncation O(h^2)
    Cpp_tol = 100.0 * (h * h) * max(1.0, abs(C_m), abs(Cpp_expected))
    Cpp_anchor_pass = (Cpp_err <= Cpp_tol)

    # F_sharp anchor test: prime vs composite
    if is_prime(m):
        F_anchor_pass = (abs(Fm) <= tau_anchor)
        F_anchor_margin = tau_anchor - abs(Fm)
    else:
        # rigorous lower bound for the exact value using S(m)_exact and tail_m
        F_lower_bound = pref * ((Sm_exact - q ** (-m)) - tail_m)
        F_anchor_pass = (F_lower_bound >= -tau_anchor)
        F_anchor_margin = F_lower_bound + tau_anchor

    return {
        "m": float(m),
        "S(m)_approx": S_m,
        "S(m)_exact": Sm_exact,
        "S(m)_approx_minus_exact": Sm_err,
        "S'(m)_approx": S_prime,
        "C_sin(m)": C_m,
        "C_sin'(m)_approx": C_prime,
        "C_sin''(m)_approx": C_pp,
        "C_sin''(m)_expected": Cpp_expected,
        "C_sin''(m)_abs_error": Cpp_err,
        "C_sin''(m)_tol": Cpp_tol,
        "C_sin''(m)_pass": bool(Cpp_anchor_pass),
        "F_sharp(m)_approx": Fm,
        "tail_bound_at_m": tail_m,
        # strict-test outputs
        "S_anchor_pass": bool(S_anchor_pass),
        "S_anchor_margin": S_anchor_margin,
        "C_anchor_pass": bool(C_anchor_pass),
        "C_rel_err": C_rel_err,
        "F_anchor_pass": bool(F_anchor_pass),
        "F_anchor_margin": F_anchor_margin,
        "tau_anchor": tau_anchor
    }



# --------------------------- Window verification -------------------------

@dataclass
class WindowResult:
    p: int
    alpha: float
    min_IL: float
    argmin_IL: float
    min_IM: float
    argmin_IM: float
    min_IR: float
    argmin_IR: float
    ok_IL: bool
    ok_IM: bool
    ok_IR: bool
    # Analytic "certificates" based on (L)/(L′)/(M) (global, not per-x)
    cert_IL_L_ok: bool
    cert_IL_Lprime_ok: bool
    cert_IM_M_ok: bool
    min_margin_IL_L: float
    min_margin_IM_M: float
    root_count_window: int
    min_margin_IR_quad: float
    alpha_IR_quad_min: Optional[float]


def tail_upper_bound(cfg: Config) -> float:
    """Geometric tail bound: sum_{i>imax} q^{-i}."""
    r = 1.0 / cfg.q
    return (r ** (cfg.imax + 1)) / (1.0 - r)


def sup_S1_on_interval(a: float, b: float) -> float:
    """
    Supremum of S1(x) = sin(2π x)/(2π) on [a,b]. Because of period 1, restrict to fractional parts
    and check endpoints and stationary points x ≡ 1/4, 3/4 (mod 1) that lie in [a,b].
    """
    # Candidate x in [a,b]: endpoints and any k+1/4, k+3/4 in [a,b].
    cand = [a, b]
    # Nearest k for quarter points
    def add_if_in(xc: float):
        if a <= xc <= b:
            cand.append(xc)

    # For an interval length < 1 we need at most two k's; compute k ranges around a..b
    kmin = math.floor(a - 1.0)
    kmax = math.ceil(b + 1.0)
    for k in range(kmin, kmax + 1):
        add_if_in(k + 0.25)
        add_if_in(k + 0.75)

    # Evaluate S1
    vals = [math.sin(TWO_PI * x) / (TWO_PI) for x in cand]
    return max(vals)


def effective_sign_tolerance_interval(a: float, b: float, q: float, imax: int, base_tol: float) -> float:
    """
    Interval-specific effective tolerance:
      tau_eff(a,b) = base_tol + (q-1)q * tail + eps_guard(a,b)

    We bound |F_sharp| by (q-1)q * ( Σ_{i>=2} q^{-i} + sup_{x∈[a,b]} C_sin(x,q) ),
    where sup C_sin on [a,b] is bounded by q^{-a} * q^{sup S1(x)} because
    C_sin(x,q) = q^{-x} * (1 + (log q) * S1(x)) and 1 + (log q) * S1(x) ≤ q^{S1(x)}.
    """
    r = 1.0 / q
    tail = (r ** (imax + 1)) / (1.0 - r)

    # Σ bound
    sum_qi = (r * r) / (1.0 - r)

    # Interval-specific C_sin cap via S1 sup
    S1_sup = sup_S1_on_interval(a, b)
    C_cap = (q ** (-a)) * (q ** S1_sup)  # sup_x q^{-x} ≤ q^{-a}, and C_sin/q^{-x} ≤ q^{S1_sup}

    pref = (q - 1.0) * q
    scale_cap = pref * (sum_qi + C_cap)

    eps_guard = 50.0 * np.finfo(float).eps * max(1.0, scale_cap)
    return base_tol + pref * tail + eps_guard


def scan_interval(a: float, b: float, q: float, imax: int, grid: int, tau_eff: float) -> Tuple[float, float, bool]:
    """
    Scan (a,b) with `grid` points using a conservative lower bound for F_sharp.
    IMPORTANT: We intentionally avoid sampling the *endpoints* to prevent
    artificial negatives right next to removable/limit behavior (e.g., x ~ p^-).

    Returns (min_value, argmin, ok_flag) with ok_flag = (min_value >= -tau_eff).
    """
    if b <= a:
        return float("nan"), float("nan"), True

    # Tiny interior kick so that we never sample the exact boundaries.
    # Using max(spacing, fixed epsilon) protects both tiny and large |a|,|b|.
    kick_a = max(np.spacing(a), 1e-12)
    kick_b = max(np.spacing(b), 1e-12)

    lo = a + kick_a
    hi = b - kick_b
    if hi <= lo:  # extremely tiny interval: nothing meaningful to check
        return float("nan"), float("nan"), True

    xs = np.linspace(lo, hi, grid, endpoint=False, dtype=float)

    min_val = float("inf")
    argmin = xs[0]

    for x in xs:
        lower, _ = Fsharp_bounds(x, q, imax)
        if lower < min_val:
            min_val = lower
            argmin = float(x)

    return min_val, argmin, (min_val >= -tau_eff)


def min_margin_L_IL(a: float, b: float, q: float, imax: int, grid: int, alpha: float) -> float:
    """
    (1) IL L-margin: min_x [ (S_lower(x)) - q^{-2} * cos^2(pi*alpha/2) ]
    where S_lower(x) = sum_{i>=2}^{imax} q^{-i} phi_i(x) - tail.
    """
    if b <= a:
        return float("nan")
    r = 1.0 / q
    tail = (r ** (imax + 1)) / (1.0 - r)
    L_const = (q ** -2.0) * (math.cos(PI * alpha / 2.0) ** 2)

    kick_a = max(np.spacing(a), 1e-12)
    kick_b = max(np.spacing(b), 1e-12)
    lo = a + kick_a
    hi = b - kick_b
    xs = np.linspace(lo, hi, grid, endpoint=False, dtype=float)
    m = float("inf")
    for x in xs:
        S_val, _ = S_value_and_tail(x, q, imax)
        S_lower = S_val - tail
        m = min(m, S_lower - L_const)
    return m


def min_margin_M_IM(a: float, b: float, q: float, imax: int, grid: int, alpha: float) -> float:
    """
    (1) IM M-margin: min_x [ (S_lower(x)) - sin^2(pi*alpha) * Σ_{i>=2} q^{-i}/i^2_lower ]
    Uses partial sum as a lower bound for the series.
    """
    if b <= a:
        return float("nan")
    r = 1.0 / q
    tail = (r ** (imax + 1)) / (1.0 - r)
    series_lower, _ = series_sum_q_i2(q, imax)
    M_const = (math.sin(PI * alpha) ** 2) * series_lower

    kick_a = max(np.spacing(a), 1e-12)
    kick_b = max(np.spacing(b), 1e-12)
    lo = a + kick_a
    hi = b - kick_b
    xs = np.linspace(lo, hi, grid, endpoint=False, dtype=float)
    m = float("inf")
    for x in xs:
        S_val, _ = S_value_and_tail(x, q, imax)
        S_lower = S_val - tail
        m = min(m, S_lower - M_const)
    return m


def root_count_in_window(p: int, alpha: float, cfg: Config) -> int:
    """
    (2) Count sign changes of the *point estimator* F_sharp (not rigorous bound)
        over the open window (p-1, p). A small dead-zone around 0 is used to
        avoid counting numerical wiggles as roots.
    """
    a, b = p - 1.0, float(p)
    # Dense grid to catch even small oscillations
    N = max(5 * cfg.grid, 2000)
    kick_a = max(np.spacing(a), 1e-12)
    kick_b = max(np.spacing(b), 1e-12)
    xs = np.linspace(a + kick_a, b - kick_b, N, endpoint=True, dtype=float)

    vals = np.array([Fsharp_value(x, cfg.q, cfg.imax) for x in xs], dtype=float)
    # Dead-zone threshold tied to effective tolerance magnitude
    # Use interval-level tau with (a,b) for a conservative dead-zone
    tau = effective_sign_tolerance_interval(a, b, cfg.q, cfg.imax, cfg.tol)
    dz = 5.0 * tau  # slightly larger than sign guard to prevent double-counting
    signs = np.sign(np.where(np.abs(vals) <= dz, 0.0, vals))

    # Count sign changes ignoring stretches of zeros
    count = 0
    prev = 0.0
    for s in signs:
        if s == 0.0:
            continue
        if prev == 0.0:
            prev = s
            continue
        if s != prev:
            count += 1
            prev = s
    return count


def verify_prime_window(p: int, alpha: float, cfg: Config,
                        L_ok: bool, Lprime_ok: bool, M_ok: bool) -> WindowResult:
    """
    Verify nonnegativity on IL, IM, IR for window (p-1,p).
    IR is cut slightly short of p to stay away from the exact zero at p.

    IR buffer is made *adaptive* w.r.t. the grid/alpha length; still respect minimal cfg.ir_buffer.
    Additionally returns:
      - per-interval L/M minimum margins (conservative, with tails),
      - root-count in the window (point estimator),
      - quadratic IR bound margin and minimal alpha where it holds.
    """
    IL_a, IL_b = (p - 1.0, p - 1.0 + alpha)
    IM_a, IM_b = (p - 1.0 + alpha, p - alpha)

    # Adaptive right-end buffer: proportional to the subinterval width / grid
    adaptive_buffer = max(5.0 * alpha / max(1, cfg.grid), 1e-8)
    ir_cut = max(cfg.ir_buffer, adaptive_buffer)
    IR_a, IR_b = (p - alpha, p - ir_cut)

    # Interval-specific effective tolerances
    tau_IL = effective_sign_tolerance_interval(IL_a, IL_b, cfg.q, cfg.imax, cfg.tol)
    tau_IM = effective_sign_tolerance_interval(IM_a, IM_b, cfg.q, cfg.imax, cfg.tol)
    tau_IR = effective_sign_tolerance_interval(IR_a, IR_b, cfg.q, cfg.imax, cfg.tol)

    # Conservative scans per subinterval
    min_IL, arg_IL, ok_IL = scan_interval(IL_a, IL_b, cfg.q, cfg.imax, cfg.grid, tau_IL)
    min_IM, arg_IM, ok_IM = scan_interval(IM_a, IM_b, cfg.q, cfg.imax, cfg.grid, tau_IM)
    min_IR, arg_IR, ok_IR = scan_interval(IR_a, IR_b, cfg.q, cfg.imax, cfg.grid, tau_IR)

    # (1) IL/IM margins vs theoretical constants (conservative, subtracting tails)
    min_margin_IL_L = min_margin_L_IL(IL_a, IL_b, cfg.q, cfg.imax, cfg.grid, alpha)
    min_margin_IM_M = min_margin_M_IM(IM_a, IM_b, cfg.q, cfg.imax, cfg.grid, alpha)

    # (2) Root counting in (p-1,p)
    rc = root_count_in_window(p, alpha, cfg)

    # (5) Quadratic IR bound: F_lower(x) >= pref * (1/4 * Spp_lb) * (p-x)^2
    #     We use the *paper* lower bound for S''(p), not the numeric approx.
    Spp_lb = Spp_lower_bound_from_paper(cfg.q) - cfg.Spp_fd_safety
    pref = (cfg.q - 1.0) * cfg.q
    # compute minimal margin on IR: lower_bound(x) - pref * (0.25*Spp_lb)*(p-x)^2
    if IR_b > IR_a and Spp_lb > 0.0:
        xs = np.linspace(IR_a + max(np.spacing(IR_a), 1e-12),
                         IR_b - max(np.spacing(IR_b), 1e-12),
                         cfg.grid, endpoint=False, dtype=float)
        margin = float("inf")
        for x in xs:
            lower, _ = Fsharp_bounds(x, cfg.q, cfg.imax)
            quad = pref * (0.25 * Spp_lb) * ((p - x) * (p - x))
            margin = min(margin, lower - quad)
        min_margin_IR_quad = margin
    else:
        min_margin_IR_quad = float("nan")

    # (5) minimal alpha such that quadratic IR bound holds on (p-α,p) (coarse bisection)
    def quad_ok_for_alpha(alpha_try: float) -> bool:
        a_try, b_try = (p - alpha_try, p - ir_cut)
        if b_try <= a_try:
            return True
        tau_try = effective_sign_tolerance_interval(a_try, b_try, cfg.q, cfg.imax, cfg.tol)
        xs_try = np.linspace(a_try + 1e-12, b_try - 1e-12, max(200, cfg.grid // 2), endpoint=False, dtype=float)
        ok = True
        for x in xs_try:
            lower, _ = Fsharp_bounds(x, cfg.q, cfg.imax)
            quad = pref * (0.25 * Spp_lb) * ((p - x) * (p - x))
            if lower < quad - tau_try:
                ok = False
                break
        return ok

    alpha_IR_quad_min = None
    if math.isfinite(min_margin_IR_quad) and not np.isnan(min_margin_IR_quad):
        lo, hi = 0.0, alpha
        # If it already holds for full alpha, try to shrink with ~10 bisection steps
        if quad_ok_for_alpha(hi):
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if mid <= 1e-6:
                    break
                if quad_ok_for_alpha(mid):
                    hi = mid
                else:
                    lo = mid
            alpha_IR_quad_min = hi  # smallest safe alpha within tolerance

    return WindowResult(
        p=p, alpha=alpha,
        min_IL=min_IL, argmin_IL=arg_IL,
        min_IM=min_IM, argmin_IM=arg_IM,
        min_IR=min_IR, argmin_IR=arg_IR,
        ok_IL=bool(ok_IL), ok_IM=bool(ok_IM), ok_IR=bool(ok_IR),
        cert_IL_L_ok=bool(L_ok),
        cert_IL_Lprime_ok=bool(Lprime_ok),
        cert_IM_M_ok=bool(M_ok),
        min_margin_IL_L=min_margin_IL_L,
        min_margin_IM_M=min_margin_IM_M,
        root_count_window=rc,
        min_margin_IR_quad=min_margin_IR_quad,
        alpha_IR_quad_min=alpha_IR_quad_min
    )


# --------------------------- Local curvature S''(p) -----------------------

def second_derivative_S_at(x0: float, cfg: Config, h: float = 2e-5) -> float:
    """
    Approximate S''(x0) via central differences:
      S''(x0) ≈ (S(x0+h) - 2 S(x0) + S(x0-h)) / h^2
    using partial sums. Diagnostic (non-rigorous).
    """
    S_c, _ = S_value_and_tail(x0, cfg.q, cfg.imax)
    S_p, _ = S_value_and_tail(x0 + h, cfg.q, cfg.imax)
    S_m, _ = S_value_and_tail(x0 - h, cfg.q, cfg.imax)
    return (S_p - 2.0 * S_c + S_m) / (h * h)


def Spp_lower_bound_from_paper(q: float) -> float:
    """
    Derived conservative lower bound at odd primes p>=5 (paper-aligned):
      Use the explicit positive contributions at i=2 and i=3,
      and subtract the worst-case negative term at i=p with p>=5:
        S''(p) >= (π^2/2) q^{-2} + (8π^2/27) q^{-3} - (2π^2/3) q^{-5}.
    """
    return ((PI ** 2) / 2.0) * (q ** -2.0) + (8.0 * (PI ** 2) / 27.0) * (q ** -3.0) - (2.0 * (PI ** 2) / 3.0) * (q ** -5.0)


# ------------------------------ Data models ------------------------------

@dataclass
class IntegerAnchorRecord:
    m: int
    Sm_approx: float
    Sm_exact: float
    Sm_diff: float
    Sprime_approx: float
    Csin_m: float
    Csin_prime_approx: float
    Csin_second_approx: float
    Csin_second_expected: float
    Csin_second_abs_error: float
    Csin_second_tol: float
    Csin_second_pass: bool
    Fsharp_m_approx: float
    tail_bound_at_m: float
    # Strict-test flags/margins
    S_anchor_pass: bool
    S_anchor_margin: float
    C_anchor_pass: bool
    C_rel_err: float
    F_anchor_pass: bool
    F_anchor_margin: float
    tau_anchor: float



@dataclass
class PrimeWindowRecord:
    p: int
    min_IL: float
    argmin_IL: float
    ok_IL: bool
    min_IM: float
    argmin_IM: float
    ok_IM: bool
    min_IR: float
    argmin_IR: float
    ok_IR: bool
    Spp_approx: float  # S''(p) approx
    Spp_pass: bool     # explicit lower-bound test pass/fail
    Spp_margin: float  # Spp_approx - (lower_bound - safety)
    # Analytic certificate flags copied from WindowResult (same for each window)
    cert_IL_L_ok: bool
    cert_IL_Lprime_ok: bool
    cert_IM_M_ok: bool
    min_margin_IL_L: float
    min_margin_IM_M: float
    root_count_window: int
    min_margin_IR_quad: float
    alpha_IR_quad_min: Optional[float]


@dataclass
class Report:
    q: float
    imax: int
    pmax: int
    grid: int
    tol: float
    tau_eff_global: float  # retained for reference; scans use interval-specific taus
    alpha_result: AlphaResult
    prime_windows: List[PrimeWindowRecord]
    integer_anchors: List[IntegerAnchorRecord]    
    found_alpha: bool
    # tail target and whether imax was auto-selected
    tail_target: float
    auto_imax: bool
    special_window_23_min: float
    special_window_23_argmin: float    
    special_window_23_policy: str = "include"
    special_window_23_ok: bool = False
    special_window_23_tau: float = 0.0
    special_window_23_imax: int = 0
    special_window_23_grid: int = 0    
# ------------------------------ Progress bar -----------------------------

def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class ProgressBar:
    """
    Minimal console progress bar with percent and ETA.
    Intended for long prime-window scans.
    """
    def __init__(self, total: int, label: str = "progress", width: int = 32):
        self.total = max(1, total)
        self.label = label
        self.width = max(10, width)
        self.start = time.monotonic()
        self.last_print = self.start

    def update(self, done: int):
        done = min(done, self.total)
        now = time.monotonic()
        # throttle prints to avoid excessive output
        if now - self.last_print < 0.05 and done < self.total:
            return
        self.last_print = now
        frac = done / self.total
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start
        eta = elapsed * (1.0 / frac - 1.0) if frac > 0 else 0.0
        sys.stdout.write(
            f"\r[{self.label}] [{bar}] {100*frac:5.1f}% | "
            f"elapsed { _fmt_hms(elapsed) } | eta { _fmt_hms(eta) }"
        )
        sys.stdout.flush()

    def finish(self):
        self.update(self.total)
        sys.stdout.write("\n")
        sys.stdout.flush()


# ---------------------------- Orchestration ------------------------------

def compute_imax_for_tail(q: float, tail_target: float) -> int:
    """
    Choose the minimal imax such that sum_{i>imax} q^{-i} <= tail_target.
    Geometric tail: r^{imax+1} / (1-r) <= tail_target with r=1/q.
    Solve for imax via logs; ensure imax >= 2.
    """
    if q <= 1.0:
        return 2
    r = 1.0 / q
    if r <= 0.0 or r >= 1.0:
        return 2
    rhs = math.log(max(tail_target * (1.0 - r), np.finfo(float).tiny)) / math.log(r)
    imax = int(math.ceil(rhs - 1.0))
    return max(2, imax)


def analyze(cfg: Config) -> Report:
    if cfg.q <= 1.0:
        raise ValueError("This verifier requires q > 1.")

    # Optionally adjust imax to hit the requested tail_target
    if cfg.auto_imax:
        cfg.imax = compute_imax_for_tail(cfg.q, cfg.tail_target)

    # Keep a global reference tolerance (old behaviour); scans will use interval-specific ones
    tau_eff_global = effective_sign_tolerance_interval(0.0, 1.0, cfg.q, cfg.imax, cfg.tol)

    # 1) Find alpha
    alpha_res = find_alpha(cfg)
    # If not found, still use the best candidate for scanning (diagnostic only)
    alpha = alpha_res.alpha if (alpha_res.alpha is not None) else 0.25

    # 2) Integer anchors (diagnostic + strict)
    anchors: List[IntegerAnchorRecord] = []
    for m in range(2, min(cfg.anchors_up_to, cfg.pmax) + 1):
        chk = check_integer_anchor(m, cfg)
        anchors.append(
            IntegerAnchorRecord(
                m=m,
                Sm_approx=chk["S(m)_approx"],
                Sm_exact=chk["S(m)_exact"],
                Sm_diff=chk["S(m)_approx_minus_exact"],
                Sprime_approx=chk["S'(m)_approx"],
                Csin_m=chk["C_sin(m)"],
                Csin_prime_approx=chk["C_sin'(m)_approx"],
                Csin_second_approx=chk["C_sin''(m)_approx"],
                Csin_second_expected=chk["C_sin''(m)_expected"],
                Csin_second_abs_error=chk["C_sin''(m)_abs_error"],
                Csin_second_tol=chk["C_sin''(m)_tol"],
                Csin_second_pass=chk["C_sin''(m)_pass"],
                Fsharp_m_approx=chk["F_sharp(m)_approx"],
                tail_bound_at_m=chk["tail_bound_at_m"],
                S_anchor_pass=chk["S_anchor_pass"],
                S_anchor_margin=chk["S_anchor_margin"],
                C_anchor_pass=chk["C_anchor_pass"],
                C_rel_err=chk["C_rel_err"],
                F_anchor_pass=chk["F_anchor_pass"],
                F_anchor_margin=chk["F_anchor_margin"],
                tau_anchor=chk["tau_anchor"]
            )
        )


    # 3) Prime windows
    prime_list = [p for p in primes_up_to(cfg.pmax) if p >= 5 and p % 2 == 1]
    window_records: List[PrimeWindowRecord] = []

    bar = ProgressBar(len(prime_list), label=f"prime windows (q={cfg.q})") if cfg.progress else None

    # Analytic "certificates" are global for the chosen alpha: (L), (L′), (M)
    L_ok = (alpha_res.L_margin is not None) and (alpha_res.L_margin >= 0.0)
    Lprime_ok = (alpha_res.Lprime_margin is not None) and (alpha_res.Lprime_margin >= 0.0)
    M_ok = (alpha_res.M_margin is not None) and (alpha_res.M_margin >= 0.0)

    for idx, p in enumerate(prime_list, 1):
        wr = verify_prime_window(p, alpha, cfg, L_ok=L_ok, Lprime_ok=Lprime_ok, M_ok=M_ok)
        # 3b) Local curvature S''(p) and explicit lower-bound check
        Spp = second_derivative_S_at(float(p), cfg)
        lb = Spp_lower_bound_from_paper(cfg.q) - cfg.Spp_fd_safety
        Spp_pass = (Spp >= lb)
        Spp_margin = Spp - lb

        window_records.append(
            PrimeWindowRecord(
                p=wr.p,
                min_IL=wr.min_IL, argmin_IL=wr.argmin_IL, ok_IL=wr.ok_IL,
                min_IM=wr.min_IM, argmin_IM=wr.argmin_IM, ok_IM=wr.ok_IM,
                min_IR=wr.min_IR, argmin_IR=wr.argmin_IR, ok_IR=wr.ok_IR,
                Spp_approx=Spp,
                Spp_pass=Spp_pass,
                Spp_margin=Spp_margin,
                cert_IL_L_ok=wr.cert_IL_L_ok,
                cert_IL_Lprime_ok=wr.cert_IL_Lprime_ok,
                cert_IM_M_ok=wr.cert_IM_M_ok,
                min_margin_IL_L=wr.min_margin_IL_L,
                min_margin_IM_M=wr.min_margin_IM_M,
                root_count_window=wr.root_count_window,
                min_margin_IR_quad=wr.min_margin_IR_quad,
                alpha_IR_quad_min=wr.alpha_IR_quad_min
            )
        )
        if bar:
            bar.update(idx)

    if bar:
        bar.finish()

    # 4) Special window (2,3) with policy
    # Default: use current global imax/grid
    sw_policy = cfg.special_23_policy
    if sw_policy == "exclude":
        # We still compute a quick diagnostic scan (optional) but do not count it.
        sw_imax = cfg.imax
        sw_grid = cfg.grid
        tau_23 = effective_sign_tolerance_interval(2.0, 3.0, cfg.q, sw_imax, cfg.tol)
        min_23, arg_23, _ = scan_interval(2.0 + 1e-9, 3.0 - 1e-9, cfg.q, sw_imax, sw_grid, tau_23)
        sw_ok = True  # rigorously: N/A for theorem scope → report as OK (N/A)
    elif sw_policy == "refine":
        # Locally tighten tail and grid to give a sharper *rigorous* lower bound.
        sw_imax = compute_imax_for_tail(cfg.q, cfg.special_23_tail_target)
        sw_grid = max(cfg.special_23_grid, cfg.grid)
        tau_23 = effective_sign_tolerance_interval(2.0, 3.0, cfg.q, sw_imax, cfg.tol)
        min_23, arg_23, sw_ok = scan_interval(2.0 + 1e-9, 3.0 - 1e-9, cfg.q, sw_imax, sw_grid, tau_23)
    else:  # "include" → previous behaviour
        sw_imax = cfg.imax
        sw_grid = cfg.grid
        tau_23 = effective_sign_tolerance_interval(2.0, 3.0, cfg.q, sw_imax, cfg.tol)
        min_23, arg_23, sw_ok = scan_interval(2.0 + 1e-9, 3.0 - 1e-9, cfg.q, sw_imax, sw_grid, tau_23)

    # Build report object
    return Report(
        q=cfg.q,
        imax=cfg.imax,
        pmax=cfg.pmax,
        grid=cfg.grid,
        tol=cfg.tol,
        tau_eff_global=tau_eff_global,
        alpha_result=alpha_res,
        prime_windows=window_records,
        integer_anchors=anchors,
        found_alpha=bool(alpha_res.found),
        tail_target=cfg.tail_target,
        auto_imax=cfg.auto_imax,
        special_window_23_min=min_23,
        special_window_23_argmin=arg_23,
        special_window_23_policy=sw_policy,
        special_window_23_ok=bool(sw_ok),
        special_window_23_tau=tau_23,
        special_window_23_imax=sw_imax,
        special_window_23_grid=sw_grid
    )

def make_json_safe(obj):
    """
    Recursively convert dataclasses, numpy scalars/arrays, Path, etc. into
    plain JSON-serializable Python types.
    """
    # Dataclass -> dict
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)

    # Dict
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    # List / Tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]

    # Numpy scalars
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)

    # Numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Paths
    if isinstance(obj, Path):
        return str(obj)

    # Everything else (str, int, float, bool, None, etc.)
    return obj

def write_json(report: Report, path: Path) -> None:
    """
    Dump the report to JSON after converting all nested values
    to JSON-serializable plain Python types.
    """
    payload = make_json_safe(report)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(report: Report, path: Path) -> None:
    """
    Compact per-prime summary CSV.
    Columns:
        q, imax, pmax, grid, tol, tau_eff_global, alpha(found?), alpha_value, p,
        min_IL, ok_IL, min_IM, ok_IM, min_IR, ok_IR, Spp, Spp_pass,
        cert_IL_L, cert_IL_L', cert_IM_M, min_margin_IL_L, min_margin_IM_M,
        root_count_window, min_margin_IR_quad, alpha_IR_quad_min
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "q", "imax", "pmax", "grid", "tol", "tau_eff_global",
            "alpha_found", "alpha_value",
            "p",
            "min_IL", "ok_IL",
            "min_IM", "ok_IM",
            "min_IR", "ok_IR",
            "Spp_approx", "Spp_pass",
            "cert_IL_L_ok", "cert_IL_Lprime_ok", "cert_IM_M_ok",
            "min_margin_IL_L", "min_margin_IM_M",
            "root_count_window",
            "min_margin_IR_quad", "alpha_IR_quad_min"
        ])
        a_val = report.alpha_result.alpha if report.alpha_result.alpha is not None else ""
        for rec in report.prime_windows:
            w.writerow([
                report.q, report.imax, report.pmax, report.grid, report.tol, report.tau_eff_global,
                report.found_alpha, a_val,
                rec.p,
                rec.min_IL, rec.ok_IL,
                rec.min_IM, rec.ok_IM,
                rec.min_IR, rec.ok_IR,
                rec.Spp_approx, rec.Spp_pass,
                rec.cert_IL_L_ok, rec.cert_IL_Lprime_ok, rec.cert_IM_M_ok,
                rec.min_margin_IL_L, rec.min_margin_IM_M,
                rec.root_count_window,
                rec.min_margin_IR_quad, rec.alpha_IR_quad_min
            ])

def _pf(ok: bool) -> str:
    """
    Pretty-print a pass/fail flag for console output.
    Uses simple ASCII to be robust on CI consoles.
    """
    return "OK " if ok else "FAIL"

def _min_record_by(recs, attr):
    """
    Return the record with the minimal finite value of a given attribute.
    Skips NaN/inf. Returns None if no finite values exist.
    """
    best = None
    best_val = float("inf")
    for r in recs:
        v = getattr(r, attr, float("nan"))
        if v is None or not math.isfinite(v):
            continue
        if v < best_val:
            best_val = v
            best = r
    return best

def console_summary(report: Report) -> str:
    """
    Human-friendly console summary with a compact 'balance/overview' section.
    This avoids having to open JSON/CSV for a first appraisal.
    """
    lines = []
    lines.append("=" * 72)
    lines.append(f" F_sharp real-zero verifier — q={report.q} | imax={report.imax} | pmax={report.pmax} | grid={report.grid}")
    lines.append("=" * 72)

    # --- Tolerances / Series cutoff ---
    lines.append(f"[tolerance] base tol={report.tol:.1e}  |  global tau_eff≈{report.tau_eff_global:.1e}  "
                 f"(interval-specific taus are used internally)")
    if report.auto_imax:
        lines.append(f"[series cutoff] auto-imax enabled; tail_target={report.tail_target:.1e}, chosen imax={report.imax}")

    # --- Alpha / certificates ---
    ar = report.alpha_result
    if ar.found:
        lines.append(f"[alpha] FOUND alpha={ar.alpha:.6f}  |  margins: (L) {ar.L_margin:.3e}, (M) {ar.M_margin:.3e}")
    else:
        lines.append(f"[alpha] NOT found. Best candidate alpha={ar.alpha:.6f} with near-margins: "
                     f"(L) {ar.L_margin:.3e}, (M) {ar.M_margin:.3e}")
    if ar.Lprime_margin is not None:
        lines.append(f"        (L') margin: {ar.Lprime_margin:+.3e}")
    if ar.M_analytic_ok is not None:
        lines.append(f"        analytic (M) (@ alpha_L={ar.alpha_L:.6f}) => {_pf(ar.M_analytic_ok)} "
                     f"(margin={ar.M_analytic_margin:+.3e})")
    lines.append("        note: using exponential majorant q^{1/(2π)} for C_sin (stricter than Λ_sin=1+log(q)/(2π)).")

    # --- Integer anchors (quick headline) ---
    n_anchors = len(report.integer_anchors)
    if n_anchors > 0:
        S_ok = sum(1 for r in report.integer_anchors if r.S_anchor_pass)
        F_ok = sum(1 for r in report.integer_anchors if r.F_anchor_pass)
        Cpp_ok = sum(1 for r in report.integer_anchors if r.Csin_second_pass)
        # Worst absolute discrepancy on S(m) (diagnostic)
        worst_S = max(report.integer_anchors, key=lambda r: abs(r.Sm_diff))
        lines.append("\n[integer anchors]")
        lines.append(f"  S-anchor: {S_ok}/{n_anchors} {_pf(S_ok==n_anchors)}  |  "
                     f"F-anchor: {F_ok}/{n_anchors} {_pf(F_ok==n_anchors)}  |  "
                     f"C'' vs exact: {Cpp_ok}/{n_anchors} (informational)")
        lines.append(f"  worst |S(m)_approx - S(m)_exact| = {abs(worst_S.Sm_diff):.3e} at m={worst_S.m}")

        # small preview (kept from previous version)
        lines.append("\n  preview (m, Sm_diff, pass_S, pass_F, C''_pass, F_margin):")
        preview = report.integer_anchors[:min(8, n_anchors)]
        for rec in preview:
            lines.append(f"    m={rec.m:2d}  Sm_diff={rec.Sm_diff:+.2e}  pass_S={rec.S_anchor_pass}  "
                         f"pass_F={rec.F_anchor_pass}  C''_pass={rec.Csin_second_pass}  F_margin={rec.F_anchor_margin:+.2e}")

    # --- Prime windows overview (balance) ---
    nW = len(report.prime_windows)
    full_ok = sum(1 for r in report.prime_windows if (r.ok_IL and r.ok_IM and r.ok_IR))
    spp_ok = sum(1 for r in report.prime_windows if r.Spp_pass)
    root_total = sum(int(r.root_count_window) for r in report.prime_windows)

    # Worst minima per subinterval (conservative lower bound)
    worst_IL = _min_record_by(report.prime_windows, "min_IL")
    worst_IM = _min_record_by(report.prime_windows, "min_IM")
    worst_IR = _min_record_by(report.prime_windows, "min_IR")

    # Theoretical margins (conservative vs L/M)
    worst_Lm = _min_record_by(report.prime_windows, "min_margin_IL_L")
    worst_Mm = _min_record_by(report.prime_windows, "min_margin_IM_M")

    # IR quadratic bound (margin)
    worst_IRq = _min_record_by(report.prime_windows, "min_margin_IR_quad")

    lines.append("\n[prime windows — balance]")
    lines.append(f"  Nonnegativity (IL/IM/IR): {full_ok}/{nW} {_pf(full_ok==nW)}")
    lines.append(f"  Curvature bound  S''(p)≥(7π²/54)q⁻³: {spp_ok}/{nW} {_pf(spp_ok==nW)}")
    lines.append(f"  Root-count in (p-1,p): {root_total} (expected: 0) {_pf(root_total==0)}")

    if worst_IL:
        lines.append(f"  Worst min IL ≈ {worst_IL.min_IL:+.3e} at p={worst_IL.p}, x≈{worst_IL.argmin_IL:.6f}  (ok={worst_IL.ok_IL})")
    if worst_IM:
        lines.append(f"  Worst min IM ≈ {worst_IM.min_IM:+.3e} at p={worst_IM.p}, x≈{worst_IM.argmin_IM:.6f}  (ok={worst_IM.ok_IM})")
    if worst_IR:
        lines.append(f"  Worst min IR ≈ {worst_IR.min_IR:+.3e} at p={worst_IR.p}, x≈{worst_IR.argmin_IR:.6f}  (ok={worst_IR.ok_IR})")

    if worst_Lm:
        lines.append(f"  Theoretical IL vs (L): min margin={worst_Lm.min_margin_IL_L:+.3e} at p={worst_Lm.p} "
                     f"{_pf(worst_Lm.min_margin_IL_L >= 0.0)}")
    if worst_Mm:
        lines.append(f"  Theoretical IM vs (M): min margin={worst_Mm.min_margin_IM_M:+.3e} at p={worst_Mm.p} "
                     f"{_pf(worst_Mm.min_margin_IM_M >= 0.0)}")

    if worst_IRq and math.isfinite(worst_IRq.min_margin_IR_quad):
        irq_ok_all = all((r.min_margin_IR_quad is None) or (not math.isfinite(r.min_margin_IR_quad)) or (r.min_margin_IR_quad >= 0.0)
                         for r in report.prime_windows)
        lines.append(f"  IR quadratic bound: worst margin={worst_IRq.min_margin_IR_quad:+.3e} at p={worst_IRq.p}  "
                     f"(alpha_IR_quad_min={worst_IRq.alpha_IR_quad_min}) {_pf(irq_ok_all)}")

    # --- Special window (2,3) ---
    lines.append("\n[(2,3) special window]")
    pol = getattr(report, "special_window_23_policy", "include")
    if pol == "exclude":
        lines.append("  excluded by theorem scope — OK (N/A)")
    else:
        tag = "OK (refined)" if (pol == "refine" and report.special_window_23_ok) else ("OK" if report.special_window_23_ok else "FAIL")
        lines.append(f"  policy={pol} | imax={report.special_window_23_imax} | grid={report.special_window_23_grid}")
        lines.append(f"  conservative min ≈ {report.special_window_23_min:+.3e} at x≈{report.special_window_23_argmin:.6f}  => {tag}")

    # --- Overall verdict (one-liner) ---
    overall_ok = (
        (full_ok == nW) and
        (spp_ok == nW) and
        (root_total == 0) and
        (worst_Lm is not None and worst_Lm.min_margin_IL_L >= 0.0) and
        (worst_Mm is not None and worst_Mm.min_margin_IM_M >= 0.0) and
        (worst_IRq is not None and math.isfinite(worst_IRq.min_margin_IR_quad) and worst_IRq.min_margin_IR_quad >= 0.0)
    )
    lines.append("\n[overall verdict] " + (_pf(overall_ok)) +
                 (" — all window conditions and theoretical margins satisfied."
                  if overall_ok else
                  " — attention: see items above (at least one condition/margin is not satisfied)."))

    # --- If there are issues, print the first offenders (short lists) ---
    if not overall_ok:
        def offenders(predicate, label):
            subset = [r for r in report.prime_windows if predicate(r)]
            return subset[:min(5, len(subset))], len(subset)

        lines.append("\n[first offenders]")
        bad_full, n_bad_full = offenders(lambda r: not (r.ok_IL and r.ok_IM and r.ok_IR), "nonnegativity")
        bad_spp, n_bad_spp = offenders(lambda r: not r.Spp_pass, "curvature")
        bad_root, n_bad_root = offenders(lambda r: r.root_count_window > 0, "roots")
        bad_irq, n_bad_irq = offenders(lambda r: (r.min_margin_IR_quad is not None) and math.isfinite(r.min_margin_IR_quad) and (r.min_margin_IR_quad < 0.0), "IR-quad")

        if n_bad_full:  lines.append(f"  Nonnegativity fails in {n_bad_full} windows; first: {[r.p for r in bad_full]}")
        if n_bad_spp:   lines.append(f"  Curvature bound fails in {n_bad_spp} windows; first: {[r.p for r in bad_spp]}")
        if n_bad_root:  lines.append(f"  Root-count>0 in {n_bad_root} windows; first: {[r.p for r in bad_root]}")
        if n_bad_irq:   lines.append(f"  IR-quad margin<0 in {n_bad_irq} windows; first: {[r.p for r in bad_irq]}")
        if worst_Lm and worst_Lm.min_margin_IL_L < 0.0:
            lines.append(f"  IL vs (L) negative margin at p={worst_Lm.p}: {worst_Lm.min_margin_IL_L:+.3e}")
        if worst_Mm and worst_Mm.min_margin_IM_M < 0.0:
            lines.append(f"  IM vs (M) negative margin at p={worst_Mm.p}: {worst_Mm.min_margin_IM_M:+.3e}")

    lines.append("-" * 72)
    return "\n".join(lines)

# -------------------------- (3) M-threshold audit -------------------------

def audit_M_threshold(cfg: Config) -> None:
    """
    Audit mode for the (M)-condition around the analytic threshold q*=12/5.

    For q in [audit_q_min, audit_q_max] with step audit_q_step:
      - compute alpha_L(q) = min{ α_L(q), 1/2 },
      - compute Σ_lb = q^{-2}/4 + q^{-3}/9 (analytic),
      - compute LHS = sin^2(pi*alpha_L)*Σ_lb and RHS = q^{-(4+alpha_L)} q^{1/(2π)},
      - report margin = LHS - RHS (>=0 means OK).
    Results are written to CSV and printed to console.
    """
    q_values = []
    q = cfg.audit_q_min
    while q <= cfg.audit_q_max + 1e-15:
        q_values.append(q)
        q += cfg.audit_q_step

    rows = []
    for q in q_values:
        alpha_L = min(alpha_L_from_L_condition(q), 0.5)
        Sigma_lb = sigma_lower_analytic(q)
        lhs = (math.sin(PI * alpha_L) ** 2) * Sigma_lb
        rhs = (q ** (-(4.0 + alpha_L))) * (q ** (1.0 / (2.0 * PI)))
        margin = lhs - rhs
        rows.append((q, alpha_L, Sigma_lb, lhs, rhs, margin))

    # Write CSV
    with open(cfg.audit_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q", "alpha_L", "Sigma_lb", "LHS_M_analytic", "RHS_M", "Margin"])
        for row in rows:
            w.writerow(list(row))

    # Console printout
    print("\n[M-threshold audit] Analytic (M) margins around q*=12/5")
    print(f"Range: [{cfg.audit_q_min}, {cfg.audit_q_max}] step={cfg.audit_q_step} -> CSV: {cfg.audit_csv_path}")
    good = sum(1 for r in rows if r[5] >= 0.0)
    print(f"OK entries: {good}/{len(rows)}")
    # Show first & last few for quick glance
    preview = rows[:3] + (["..."] if len(rows) > 6 else []) + rows[-3:]
    for r in preview:
        if r == "...":
            print("  ...")
        else:
            print(f"  q={r[0]:.5f}  alpha_L={r[1]:.6f}  margin={r[5]:+.3e}")


def parse_args(argv: Optional[List[str]] = None) -> Config:
    """
    Build a Config object from CLI arguments, with sensible defaults so that
    running the script without any arguments works out of the box.
    """
    p = argparse.ArgumentParser(
        description="Numerical verifier for the F_sharp real-zero proof (q>1)."
    )
    # All parameters are optional now; Config provides defaults.
    p.add_argument("--q", type=float, default=Config.q, help="Real parameter q>1 (default: 1.5).")
    p.add_argument("--pmax", type=int, default=Config.pmax, help=f"Prime windows up to this prime (default: {Config.pmax}).")
    p.add_argument("--imax", type=int, default=Config.imax, help=f"Series cutoff imax (default: {Config.imax}).")
    p.add_argument("--grid", type=int, default=Config.grid, help=f"Sampling points per subinterval (default: {Config.grid}).")
    p.add_argument("--tol", type=float, default=Config.tol, help=f"Base tolerance for sign checks (default: {Config.tol:g}).")
    p.add_argument("--alpha-min", type=float, default=Config.alpha_min, dest="alpha_min", help="Alpha search lower bound.")
    p.add_argument("--alpha-max", type=float, default=Config.alpha_max, dest="alpha_max", help="Alpha search upper bound (< 0.5).")
    p.add_argument("--alpha-grid", type=int, default=Config.alpha_grid, dest="alpha_grid", help="Number of alpha grid points.")
    p.add_argument("--json", type=str, default=Config.json_path, help="Path to JSON report.")
    p.add_argument("--csv", type=str, default=Config.csv_path, help="Path to CSV summary.")
    p.add_argument("--anchors-up-to", type=int, default=Config.anchors_up_to, dest="anchors_up_to",
                   help=f"Check integer anchors up to this m (default: {Config.anchors_up_to}).")
    p.add_argument("--no-progress", action="store_true", help="Disable console progress bar.")
    p.add_argument("--ir-buffer", type=float, default=Config.ir_buffer, dest="ir_buffer",
                   help=f"Right-end buffer for IR subinterval (minimum, default: {Config.ir_buffer:g}).")

    # CLI toggles for dynamic imax
    p.add_argument("--auto-imax", action="store_true", help="Enable dynamic selection of imax to meet --tail-target.")
    p.add_argument("--no-auto-imax", action="store_true", help="Disable dynamic imax selection (use --imax as-is).")
    p.add_argument("--tail-target", type=float, default=Config.tail_target, dest="tail_target",
                   help="Target geometric tail bound for S(x): sum_{i>imax} q^{-i} <= tail_target (default: 1e-14).")

    # Finite-difference safety for S''(p)
    p.add_argument("--Spp-fd-safety", type=float, default=Config.Spp_fd_safety, dest="Spp_fd_safety",
                   help="Safety margin subtracted from S''(p) paper lower bound (default: 1e-8).")

    # Special window (2,3) policy controls
    p.add_argument("--special-23-policy", choices=["exclude", "refine", "include"],
                   default=Config.special_23_policy, dest="special_23_policy",
                   help='Policy for (2,3): "exclude" (paper N/A), "refine" (tighter local check), or "include".')
    p.add_argument("--special-23-grid", type=int, default=Config.special_23_grid, dest="special_23_grid",
                   help="Grid used for (2,3) when --special-23-policy=refine (default: 3000).")
    p.add_argument("--special-23-tail-target", type=float, default=Config.special_23_tail_target,
                   dest="special_23_tail_target",
                   help="Tail target for (2,3) when --special-23-policy=refine (default: 1e-16).")

    # (3) M-threshold audit options
    p.add_argument("--audit-M-threshold", action="store_true", help="Run the (M)-threshold audit around q=12/5.")
    p.add_argument("--audit-q-min", type=float, default=Config.audit_q_min, dest="audit_q_min", help="Lower q for (M) audit.")
    p.add_argument("--audit-q-max", type=float, default=Config.audit_q_max, dest="audit_q_max", help="Upper q for (M) audit.")
    p.add_argument("--audit-q-step", type=float, default=Config.audit_q_step, dest="audit_q_step", help="Step for (M) audit.")
    p.add_argument("--audit-csv", type=str, default=Config.audit_csv_path, dest="audit_csv_path", help="CSV path for (M) audit.")

    args = p.parse_args(argv)

    # Determine auto_imax (default True), but allow explicit overrides
    auto_imax = Config.auto_imax
    if args.auto_imax:
        auto_imax = True
    if args.no_auto_imax:
        auto_imax = False

    return Config(
        q=args.q,
        pmax=args.pmax,
        imax=args.imax,
        grid=args.grid,
        tol=args.tol,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_grid=args.alpha_grid,
        anchors_up_to=args.anchors_up_to,
        json_path=args.json,
        csv_path=args.csv,
        progress=(not args.no_progress),
        ir_buffer=args.ir_buffer,
        auto_imax=auto_imax,
        tail_target=args.tail_target,
        Spp_fd_safety=args.Spp_fd_safety,
        audit_M_threshold=args.audit_M_threshold,
        audit_q_min=args.audit_q_min,
        audit_q_max=args.audit_q_max,
        audit_q_step=args.audit_q_step,
        audit_csv_path=args.audit_csv_path,
        special_23_policy=args.special_23_policy,
        special_23_grid=args.special_23_grid,
        special_23_tail_target=args.special_23_tail_target
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)

    report = analyze(cfg)

    # Write outputs
    write_json(report, Path(cfg.json_path))
    write_csv(report, Path(cfg.csv_path))

    # Console summary
    print(console_summary(report))

    # Optional (3): M-threshold audit mode
    if cfg.audit_M_threshold:
        audit_M_threshold(cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
