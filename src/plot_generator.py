#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#
#   Plotting Script for the Paper
#   "The Fejer-Dirichlet Lift: Entire Functions and zeta-Factorization Identities"
#
#   This script implements the analytic functions (mathfrak{F}, etc.) and
#   focuses on visualizing their structure in the complex plane and as
#   smooth analogues of classical arithmetic functions.
#
#   Author: Sebastian Fuchs | sebastian.fuchs@hu-berlin.de | https://orcid.org/0009-0009-1237-4804
#   Date:   2025-09-14
#   Version: 1.0
#
# =============================================================================

from math import isqrt
import os
import time
import cmath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from numba import jit
from sympy import sieve, divisor_count
from tqdm import tqdm
from matplotlib.colors import LightSource, PowerNorm, Normalize, LogNorm, ListedColormap
from matplotlib.cm import ScalarMappable

# =============================================================================
# CENTRAL CONFIGURATION
# =============================================================================
class Config:
    # --- Function Parameters ---
    Q_PARAMETER_NEG = -1.0
    Q_PARAMETER_LOW = 2.0
    Q_PARAMETER_HIGH = 1000.0
    Q_ANALOG_Q_VALUES = [1.0, 1.01, 1.1, 1.2, 1.5, 2.0]

    # --- Plotting Configurations ---
    COMPLEX_ZERO_X_RANGE = (0, 100)
    COMPLEX_ZERO_Y_RANGE_LOW_Q = (-15, 15)
    COMPLEX_ZERO_Y_RANGE_HIGH_Q = (-100, 100)
    COMPLEX_GRID_RESOLUTION = (4000, 2000) # (4000, 1200)

    COMPLEX_3D_X_RANGE = (0, 100)
    COMPLEX_3D_Y_RANGE = (-15, 15)
    COMPLEX_3D_X_RANGE_SIDE = (0, 30)
    COMPLEX_3D_Y_RANGE_SIDE = (-2, 2)
    COMPLEX_3D_GRID_RESOLUTION_SHADED = (2000, 2000) # (2000, 2000)
    COMPLEX_3D_GRID_RESOLUTION = (1000, 1000) # (1000, 1000)

    Q_ANALOG_PLOT_X_RANGE = (10, 40)
    
    LAMBDA_PLOT_X_RANGE = (1.5, 30.5)
    LAMBDA_SERIES_LIMIT = 200 # Higher limit for better accuracy of the Lambda interpolant
    LAMBDA_PLOT_FIGSIZE = (20, 10)

    MU_PLOT_X_RANGE = (0.5, 10.5)
    MU_SERIES_LIMIT = 250 # Moebius series converges slowly, needs more terms
    MU_PLOT_FIGSIZE = (20, 8)

    OPTIMIZED_PLOT_Q_VALUE = 1.5
    OPTIMIZED_PLOT_X_RANGE = (1.5, 8.5)
    OPTIMIZED_PLOT_FIGSIZE = (20, 10)

    DOMAIN_PLOT_X_RANGE = (-30, 30)
    DOMAIN_PLOT_Y_RANGE = (-5, 5)
    DOMAIN_PLOT_RESOLUTION = 4000 # 3000, Horizontal resolution, vertical is scaled
    DOMAIN_PLOT_FIGSIZE = (20, 12)

    COMPANION_Q_VALUES = [1.5, 2.0, 2.5, 3.0]
    COMPANION_PRIME_RANGE = (3, 101) # Primes up to 100
    COMPANION_SERIES_LIMIT = 75 # Limit for series in Newton's method
    COMPANION_PLOT_FIGSIZE = (12, 8)

    OPERATOR_PLOT_X_RANGE = (1, 40.5)
    OPERATOR_PLOT_FIGSIZE = (20, 8)

    TRUNCATION_LIMIT = 100  # Default limit for other infinite sum truncations
    
    # --- General Plotting Configuration ---
    PLOT_DPI = 300
    PLOT_FILE_FORMAT = 'pdf'
    PLOT3D_DPI = 200
    PLOT3D_FILE_FORMAT = 'jpg'
    PLOT_COMPLEX_FIGSIZE = (24, 18)
    PLOT_3D_FIGSIZE = (24, 24)
    PLOT_Q_ANALOG_FIGSIZE = (30, 15)

# =============================================================================
# ARITHMETIC HELPER FUNCTIONS (for Lambda Interpolant)
# =============================================================================

_prime_factor_cache = {}
def get_prime_factorization(n):
    """Computes the prime factorization of n as a dictionary."""
    if n in _prime_factor_cache:
        return _prime_factor_cache[n]
    factors = {}
    d = 2
    temp = n
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    _prime_factor_cache[n] = factors
    return factors

def moebius_mu(n):
    """Computes the Moebius function mu(n)."""
    if n == 1:
        return 1
    factors = get_prime_factorization(n)
    for p in factors:
        if factors[p] > 1:
            return 0
    return (-1)**len(factors)

def von_mangoldt_lambda(n):
    """Computes the von Mangoldt function Lambda(n)."""
    if n <= 1:
        return 0
    factors = get_prime_factorization(n)
    if len(factors) == 1:
        p = list(factors.keys())[0]
        return np.log(p)
    return 0

_convolution_cache = {}
def dirichlet_convolution(f, g, n, f_name, g_name):
    """Computes the Dirichlet convolution (f*g)(n) with caching."""
    cache_key = (f_name, g_name, n)
    if cache_key in _convolution_cache:
        return _convolution_cache[cache_key]
    
    sum_val = 0
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            sum_val += f(d) * g(n // d)
            if d * d != n:
                sum_val += f(n // d) * g(d)
    _convolution_cache[cache_key] = sum_val
    return sum_val

def neg_log_divisor_product(n):
    """
    Computes the function g(n) = -log(product of divisors of n),
    which is equivalent to -sum(log(d) for d|n).
    This function arises from differentiating sigma_{-s}(n).
    """
    if n == 1:
        return 0
    
    log_sum = 0
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            log_sum += np.log(d)
            if d * d != n:
                log_sum += np.log(n // d)
    return -log_sum

# =============================================================================
# CORE ANALYTIC FUNCTION IMPLEMENTATIONS
# =============================================================================

@jit(nopython=True, fastmath=True)
def F_complex(z: complex, i: int) -> complex:
    """
    The Fejér kernel term F(z, i), implemented for complex inputs z.
    """
    if i < 1:
        return 0
    if i == 1:
        return 1.0 # F(z,1) = 1
        
    inner_sum_val = complex(i)
    for k in range(1, i):
        term = (i - k) * cmath.cos(2 * cmath.pi * k * z / i)
        inner_sum_val += 2 * term
    return inner_sum_val

def calculate_T_b(x, b_weights, series_limit):
    """
    Calculates the T_b(x) function for b = mu * Lambda.
    This is the entire function that interpolates Lambda(n).
    """
    x = np.asarray(x)
    x_reshaped = x[:, np.newaxis]
    i_values = np.arange(1, series_limit + 1)
    
    weights = np.array([b_weights[i] for i in i_values])
    # Note: Using np.vectorize is simpler than manual broadcasting for this complex function
    fejer_vectorized = np.vectorize(F_complex)
    fejer_vals = fejer_vectorized(x_reshaped, i_values)
    
    series_sum = np.sum(weights * fejer_vals / (i_values**2), axis=1)
    return series_sum.real

@jit(nopython=True, fastmath=True)
def anayltic_sum_part(z: complex, q_inv: float, weight_power: int, TRUNCATION_LIMIT: int) -> complex:
    """
    A generalized, high-performance function for the infinite sum part.
    weight_power: 2 for tau-like functions, 1 for sigma-like.
    """
    summation_part = complex(0.0)

    for i in range(2, TRUNCATION_LIMIT):
        weight = 1.0
        if weight_power == 1:
            weight = i
        elif weight_power == 2:
            weight = i * i

        term = (F_complex(z, i) / weight) * (q_inv ** i)
        summation_part += term
    return summation_part

@jit(nopython=True, fastmath=True)
def mathfrak_F_complex(z: complex, q: float, TRUNCATION_LIMIT: int) -> complex:
    """
    High-performance implementation of the final, corrected analytic function 
    mathfrak{F}(z,q).
    """
    q_inv = 1.0 / q
    summation_part = anayltic_sum_part(z, q_inv, 2, TRUNCATION_LIMIT) # F(z,i)/i^2

    pre_factor = q * (q - 1.0)
    normalized_sum = pre_factor * summation_part
    
    correction_term = pre_factor * (q_inv ** z)
    
    return normalized_sum - correction_term

@jit(nopython=True, fastmath=True)
def mathfrak_F_tau_complex(z: complex, q: float, TRUNCATION_LIMIT: int) -> complex:
    """
    High-performance implementation of the q-analog of the divisor-counting function,
    mathfrak{F}_tau(z,q).
    """
    q_inv = 1.0 / q
    summation_part = anayltic_sum_part(z, q_inv, 2, TRUNCATION_LIMIT) # F(z,i)/i^2
    correction_term = q_inv ** z
    return summation_part - correction_term

@jit(nopython=True, fastmath=True)
def mathfrak_F_sharp_complex(z: complex, q: float, TRUNCATION_LIMIT: int) -> complex:
    """
    High-performance implementation of the optimized analytic indicator
    mathfrak{F}_sin(z,q), which eliminates companion zeros.
    """
    q_inv = 1.0 / q
    
    # The sum part S(z,q) is identical to the original function
    summation_part = anayltic_sum_part(z, q_inv, 2, TRUNCATION_LIMIT)
    pre_factor = q * (q - 1.0)
    S_z_q = pre_factor * summation_part
    
    # The correction term C_sin(z,q) is modified with a periodic normalizer
    log_q = np.log(q)
    S1_z = cmath.sin(2 * cmath.pi * z) / (2 * cmath.pi)
    C_sin_z_q = pre_factor * (q_inv ** z) * (1.0 + log_q * S1_z)
    
    return S_z_q - C_sin_z_q

def calculate_T_mu(x, mu_weights, series_limit):
    """
    Calculates the T_mu(x) function for a(i) = mu(i).
    This is the entire function that interpolates (mu*1)(n), which is 1 at n=1
    and 0 for n > 1.
    """
    x = np.asarray(x)
    x_reshaped = x[:, np.newaxis]
    i_values = np.arange(1, series_limit + 1)
    
    weights = np.array([mu_weights[i] for i in i_values])
    fejer_vectorized = np.vectorize(F_complex)
    fejer_vals = fejer_vectorized(x_reshaped, i_values)
    
    series_sum = np.sum(weights * fejer_vals / (i_values**2), axis=1)
    return series_sum.real


@jit(nopython=True, fastmath=True)
def F_prime_complex(z: complex, i: int) -> complex:
    """
    The first derivative of the Fejer kernel F'(z,i) with respect to z.
    """
    if i < 2:
        return 0
    
    inner_sum_val = complex(0.0)
    for k in range(1, i):
        term = k * (i - k) * cmath.sin(2 * cmath.pi * k * z / i)
        inner_sum_val += term
    return -4 * cmath.pi / i * inner_sum_val

@jit(nopython=True, fastmath=True)
def F_double_prime_int(p: int, i: int) -> float:
    """
    The second derivative of the Fejer kernel F''(p,i) at an integer p.
    This is used for the initial guess in Newton's method.
    """
    if i < 2:
        return 0.0
    
    inner_sum_val = 0.0
    for k in range(1, i):
        term = (k**2) * (i - k) * np.cos(2 * np.pi * k * p / i)
        inner_sum_val += term
    return -8 * (np.pi**2) / (i**2) * inner_sum_val

def calculate_mathfrak_F_prime(z, q, series_limit):
    """
    Calculates the derivative of F(z,q) with respect to z.
    """
    z = np.asarray(z, dtype=np.complex128)
    
    # Sum part derivative
    i_values = np.arange(2, series_limit + 1)
    z_reshaped = z[:, np.newaxis]
    
    F_prime_vec = np.vectorize(F_prime_complex)
    f_prime_vals = F_prime_vec(z_reshaped, i_values)
    q_inv_i = q ** (-i_values)
    
    sum_part_prime = np.sum(f_prime_vals / (i_values**2) * q_inv_i, axis=1)
    S_prime = (q - 1) * q * sum_part_prime
    
    # Correction term derivative
    C_prime = -(q - 1) * q * np.log(q) * (q ** (-z))
    
    return S_prime - C_prime

def calculate_mathfrak_F_sharp_prime(z, q, series_limit):
    """
    Calculates the derivative of the optimized indicator F_sharp(z,q)
    using the updated normalizer (1 + log(q) * S1(z)).
    """
    z = np.asarray(z, dtype=np.complex128)

    # Sum part derivative S'(z,q)
    i_values = np.arange(2, series_limit + 1)
    z_reshaped = z[:, np.newaxis]
    F_prime_vec = np.vectorize(F_prime_complex)
    f_prime_vals = F_prime_vec(z_reshaped, i_values)
    q_inv_i = q ** (-i_values)
    sum_part_prime = np.sum(f_prime_vals / (i_values**2) * q_inv_i, axis=1)
    S_prime = (q - 1) * q * sum_part_prime

    # correction term and its derivative
    log_q = np.log(q)
    S1_z = np.sin(2 * np.pi * z) / (2 * np.pi)
    S1_prime_z = np.cos(2 * np.pi * z)

    #C_sin_z_q = (q - 1) * q * (q ** (-z)) * (1.0 + log_q * S1_z)
    C_sin_prime_z_q = (q - 1) * q * (q ** (-z)) * log_q * (S1_prime_z - 1.0 - log_q * S1_z)

    return S_prime - C_sin_prime_z_q

def find_companion_zero_sin(p, q, config):
    """
    Finds the potential companion zero for F_sharp, specifically for p=3.
    """
    series_limit = config.COMPANION_SERIES_LIMIT
    
    # We use a simple midpoint as the initial guess for the optimized function
    x_n = p - 0.7
    
    # Refine with Newton's method
    for _ in range(5): # More iterations for stability
        f_val = mathfrak_F_sharp_complex(complex(x_n), q, series_limit)
        f_prime_val = calculate_mathfrak_F_sharp_prime(np.array([complex(x_n)]), q, series_limit)[0]
        if abs(f_prime_val) < 1e-9:
             break
        x_n = x_n - (f_val / f_prime_val).real

    return x_n

def find_companion_zero(p, q, config):
    """
    Finds the companion zero x_p(q) for a prime p using the asymptotic
    approximation followed by Newton's method, as described in the paper.
    """
    series_limit = config.COMPANION_SERIES_LIMIT
    
    # 1. Calculate the initial guess based on the asymptotic formula
    phi_double_prime_p = np.array([F_double_prime_int(p, i) / (i**2) for i in range(2, series_limit + 1)])
    i_values = np.arange(2, series_limit + 1)
    q_inv_i = q ** (-i_values)
    
    K_M = 0.5 * (np.sum(q_inv_i * phi_double_prime_p) - (q ** (-p)) * (np.log(q)**2))
    
    if K_M <= 0: # Should not happen for q>1 and large enough M
        return p - 1e-8 # Fallback
        
    delta_0 = (np.log(q) / K_M) * (q ** (-p))
    x0 = p - delta_0
    
    # 2. Refine with 2-3 steps of Newton's method
    x_n = x0
    for _ in range(3): # 3 iterations are sufficient for high accuracy
        f_val = mathfrak_F_complex(complex(x_n), q, series_limit)
        f_prime_val = calculate_mathfrak_F_prime(np.array([complex(x_n)]), q, series_limit)[0]
        if abs(f_prime_val) < 1e-9:
             break # Avoid division by zero
        x_n = x_n - (f_val / f_prime_val).real

    return x_n


# =============================================================================
# PLOTTING SUITE
# =============================================================================

class Plotter:
    """Generates all scientific plots for the paper."""
    
    def __init__(self, config: Config):
        self.config = config
        x_max_range = max(
            config.COMPLEX_ZERO_X_RANGE[1], 
            config.Q_ANALOG_PLOT_X_RANGE[1], 
            config.COMPLEX_3D_X_RANGE[1], 
            config.LAMBDA_PLOT_X_RANGE[1], 
            config.MU_PLOT_X_RANGE[1],
            config.DOMAIN_PLOT_X_RANGE[1],
            config.OPTIMIZED_PLOT_X_RANGE[1]
        )
        self.primes = set(sieve.primerange(1, int(x_max_range) + 1))
        plt.style.use('seaborn-v0_8-whitegrid')
        print("Initializing plotter...")

        # Pre-calculate weights for functions that require them
        self._precompute_weights()
    
    def _precompute_weights(self):
        """Pre-calculates all necessary arithmetic weights."""
        # Pre-calculate b(i) weights for the Lambda interpolant plot
        print("    INFO: Pre-calculating b(i) = (mu*Lambda)(i) weights...")
        start_time = time.time()
        _convolution_cache.clear()
        _prime_factor_cache.clear()
        self.b_weights = {i: dirichlet_convolution(moebius_mu, von_mangoldt_lambda, i, 'mu', 'L') 
                          for i in range(1, self.config.LAMBDA_SERIES_LIMIT + 1)}
        print(f"    SUCCESS: Calculated {self.config.LAMBDA_SERIES_LIMIT} b(i) weights in {time.time() - start_time:.2f} seconds.")

        # Pre-calculate mu(i) weights for the Moebius Annihilator plot
        print("    INFO: Pre-calculating mu(i) weights...")
        start_time = time.time()
        self.mu_weights = {i: moebius_mu(i) for i in range(1, self.config.MU_SERIES_LIMIT + 1)}
        print(f"    SUCCESS: Calculated {self.config.MU_SERIES_LIMIT} mu(i) weights in {time.time() - start_time:.2f} seconds.")

        # Pre-calculate -(tau*Lambda)(n) values for the operator plot
        print("    INFO: Pre-calculating -(tau*Lambda)(n) convolution...")
        start_time = time.time()
        def tau_func(n): # Wrapper for sympy function
            return divisor_count(n)
        
        x_max_operator = int(self.config.OPERATOR_PLOT_X_RANGE[1])
        _convolution_cache.clear()
        self.neg_tau_lambda_conv = {n: -dirichlet_convolution(tau_func, von_mangoldt_lambda, n, 'tau', 'L')
                                    for n in range(1, x_max_operator + 1)}
        print(f"    SUCCESS: Calculated {x_max_operator} convolution values in {time.time() - start_time:.2f} seconds.")


    def generate_lambda_interpolant_plot(self):
        """
        Visualizes how the FD-Lift with the (mu*Lambda) adapter constructs an entire
        function that interpolates the von Mangoldt function Lambda(n).
        """
        print(f"\n--- Generating Lambda(n) Interpolant Plot ---")
        x_min, x_max = self.config.LAMBDA_PLOT_X_RANGE
        start_time = time.time()
        
        x_smooth = np.linspace(x_min, x_max, 2000)
        y_smooth = calculate_T_b(x_smooth, self.b_weights, self.config.LAMBDA_SERIES_LIMIT)
        
        n_discrete = np.arange(2, int(x_max) + 1)
        y_discrete = np.array([von_mangoldt_lambda(n) for n in n_discrete])
        
        computation_time = time.time() - start_time
        print(f"    INFO: Data computation finished in {computation_time:.2f} seconds.")

        fig, ax = plt.subplots(figsize=self.config.LAMBDA_PLOT_FIGSIZE)
        
        markerline, stemlines, baseline = ax.stem(
            n_discrete, y_discrete, linefmt='grey', markerfmt='Dk', basefmt='k-',
            label=r'Von Mangoldt Function $\Lambda(n)$ (Arithmetic Truth)'
        )
        plt.setp(stemlines, 'linewidth', 1.5)
        plt.setp(markerline, 'markersize', 6)
        
        ax.plot(x_smooth, y_smooth, c='crimson', lw=2.5,
                label=r'Smooth Interpolant $\mathcal{T}_{(\mu*\Lambda)}(x)$ (Calculated via $F(x,i)$)')

        ax.set_title(r'Visual Confirmation of the Adapter Method: Interpolating $\Lambda(n)$', fontsize=20, pad=20)
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel(r"Value", fontsize=16)
        ax.legend(fontsize=14)
        ax.grid(True, which='both', linestyle=':', linewidth=0.7, zorder=-10)
        ax.set_xticks(np.arange(2, int(x_max)+1, 1))
        ax.set_ylim(-1.5, 4.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        figures_dir = os.path.join(os.path.dirname(__file__), "../figures")
        filename = os.path.join(figures_dir, f"plot_lambda_interpolant.{self.config.PLOT_FILE_FORMAT}")
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"    SUCCESS: Saved Lambda interpolant plot to '{filename}'.")
        plt.close(fig)
        
    def generate_mu_interpolant_plot(self):
        """
        Visualizes the FD-Lift of the Moebius function, T_mu(z).
        This demonstrates how the framework constructs an "analytic annihilator"
        that interpolates (mu*1)(n), which is 1 at n=1 and 0 for n>1.
        """
        print(f"\n--- Generating Moebius Lift ('Analytic Annihilator') Plot ---")
        x_min, x_max = self.config.MU_PLOT_X_RANGE
        start_time = time.time()
        
        x_smooth = np.linspace(x_min, x_max, 2000)
        y_smooth = calculate_T_mu(x_smooth, self.mu_weights, self.config.MU_SERIES_LIMIT)
        
        computation_time = time.time() - start_time
        print(f"    INFO: Data computation finished in {computation_time:.2f} seconds.")

        fig, ax = plt.subplots(figsize=self.config.MU_PLOT_FIGSIZE)
        
        # --- Plotting ---
        # 1. The smooth, calculated curve from the framework
        ax.plot(x_smooth, y_smooth, c='darkcyan', lw=2.0,
                label=r'Analytic Interpolant $\mathcal{T}_{\mu}(x)$ (Calculated via $F(x,i)$)')

        # 2. The discrete, true arithmetic values (mu*1)
        n_discrete = np.arange(1, int(x_max) + 1)
        y_discrete = np.array([1.0 if n == 1 else 0.0 for n in n_discrete])
        ax.scatter(n_discrete, y_discrete, c='red', s=50, zorder=5, 
                   label=r'Arithmetic Truth: $(\mu*1)(n) = \delta_{1,n}$')

        # --- Labels and Aesthetics ---
        ax.set_title(r'The Moebius Lift as an "Analytic Annihilator"', fontsize=20, pad=20)
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel(r"Value", fontsize=16)
        ax.legend(fontsize=14)
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.75)
        ax.set_xticks(np.arange(1, int(x_max) + 1, 1))
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # --- Save the Figure ---
        figures_dir = os.path.join(os.path.dirname(__file__), "../figures")
        filename = os.path.join(figures_dir, f"plot_mu_interpolant.{self.config.PLOT_FILE_FORMAT}")
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"    SUCCESS: Saved Moebius interpolant plot to '{filename}'.")
        plt.close(fig)

    def generate_differentiation_operator_plot(self):
        """
        Visualizes the effect of the differentiation operator D = d/ds|s=0.
        It plots the two resulting arithmetic functions side-by-side to illustrate
        the transformation.
        """
        print(f"\n--- Generating Differentiation Operator Effect Plot ---")
        x_min, x_max = self.config.OPERATOR_PLOT_X_RANGE
        n_values = np.arange(1, int(x_max))

        # --- Data Generation ---
        # Left panel: g(n) = D[sigma_{-s}(n)]
        g_n = np.array([neg_log_divisor_product(n) for n in n_values])
        
        # Right panel: h(n) = -(tau*Lambda)(n)
        h_n = np.array([self.neg_tau_lambda_conv[n] for n in n_values])

        # --- Plot Setup ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.OPERATOR_PLOT_FIGSIZE, sharey=True)
        fig.suptitle('Visualizing the Effect of the Differentiation Operator $\mathcal{D}$', fontsize=20, y=1.02)

        # --- Left Panel ---
        ax1.stem(n_values, g_n, linefmt='darkslateblue', markerfmt='o', basefmt='grey',
                 label=r'$g(n) = \mathcal{D}[\sigma_{-s}(n)] = -\sum_{d|n} \log d$')
        ax1.set_title('Result on the Arithmetic Side', fontsize=16, pad=15)
        ax1.set_xlabel("n", fontsize=14)
        ax1.set_ylabel("Value", fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, linestyle=':', alpha=0.7)

        # --- Right Panel ---
        ax2.stem(n_values, h_n, linefmt='seagreen', markerfmt='s', basefmt='grey',
                 label=r'$h(n) = -(\tau * \Lambda)(n)$')
        ax2.set_title('Result on the Spectral Side', fontsize=16, pad=15)
        ax2.set_xlabel("n", fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.7)

        # --- Save the Figure ---
        figures_dir = os.path.join(os.path.dirname(__file__), "../figures")
        filename = os.path.join(figures_dir, f"plot_operator_effect.{self.config.PLOT_FILE_FORMAT}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"    SUCCESS: Saved operator effect plot to '{filename}'.")
        plt.close(fig)

    def generate_companion_displacement_plot(self):
        """
        Visualizes the exponential decay of the companion zero displacement Delta_p(q).
        This plot confirms the asymptotic law derived in the paper.
        """
        print(f"\n--- Generating Companion Zero Displacement Plot ---")
        start_time = time.time()
        
        q_values = self.config.COMPANION_Q_VALUES
        p_min, p_max = self.config.COMPANION_PRIME_RANGE
        # Get all odd primes in the specified range
        all_primes = sorted([p for p in self.primes if p >= p_min and p <= p_max and p % 2 != 0])
        
        results = {q: {'primes': [], 'deltas': []} for q in q_values}
        
        print("    INFO: Calculating companion zero locations...")
        for q in q_values:
            for p in all_primes:
                x_p = find_companion_zero(p, q, self.config)
                delta_p = p - x_p
                
                # A more robust check: the zero must be in the correct window (p-1, p)
                if 0 < delta_p < 1:
                    results[q]['primes'].append(p)
                    results[q]['deltas'].append(delta_p)
            print(f"      Computed for q = {q}. Found {len(results[q]['primes'])} valid companion zeros out of {len(all_primes)} primes.")

        computation_time = time.time() - start_time
        print(f"    INFO: Data computation finished in {computation_time:.2f} seconds.")

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=self.config.COMPANION_PLOT_FIGSIZE)
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(q_values)))

        for i, q in enumerate(q_values):
            # Use the filtered lists of primes and deltas that are guaranteed to match in length
            plot_primes = results[q]['primes']
            plot_deltas = results[q]['deltas']
            
            if not plot_primes:  # Skip if no valid data was found for this q
                continue

            # Plot calculated data points
            ax.semilogy(plot_primes, plot_deltas, 'o', markersize=6, color=colors[i], label=f'Calculated $\Delta_p(q)$ for q={q}')
            
            # Plot theoretical slope for comparison
            # We fit the constant C using the first valid data point
            C = plot_deltas[0] * (q ** plot_primes[0])
            p_theory = np.array(plot_primes, dtype=float)
            delta_theory = C * (q ** (-p_theory))
            ax.semilogy(p_theory, delta_theory, '--', color=colors[i], lw=1.5, label=f'Theoretical Slope $\propto q^{{-p}}$')

        # --- Labels and Aesthetics ---
        ax.set_title(r'Exponential Decay of Companion Zero Displacement $\Delta_p(q)$', fontsize=18, pad=20)
        ax.set_xlabel("Prime number $p$", fontsize=14)
        ax.set_ylabel(r"Displacement $\Delta_p(q) = p - x_p(q)$ (log scale)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, which='both', linestyle=':')
        
        # --- Save the Figure ---
        figures_dir = os.path.join(os.path.dirname(__file__), "../figures")
        filename = os.path.join(figures_dir, f"plot_companion_displacement.{self.config.PLOT_FILE_FORMAT}")
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"    SUCCESS: Saved companion displacement plot to '{filename}'.")
        plt.close(fig)

    def generate_companion_zero_elimination_plot(self):
        """
        Generates a two-panel comparison plot showing the original F(x,q)
        and the optimized F_sharp(x,q), with companion zeros calculated numerically.
        """
        q = self.config.OPTIMIZED_PLOT_Q_VALUE
        print(f"\n--- Generating Companion Zero Elimination Plot for q={q} ---")
        x_min, x_max = self.config.OPTIMIZED_PLOT_X_RANGE
        start_time = time.time()
        
        x_smooth = np.linspace(x_min, x_max, 3000)
        
        # Calculate both the original and the optimized function
        y_original = np.vectorize(mathfrak_F_complex)(x_smooth, q, self.config.TRUNCATION_LIMIT).real
        y_optimized = np.vectorize(mathfrak_F_sharp_complex)(x_smooth, q, self.config.TRUNCATION_LIMIT).real
        
        primes = sorted([p for p in self.primes if x_min < p < x_max])
        odd_primes = [p for p in primes if p > 2]

        # --- Numerically find companion zeros ---
        print("    INFO: Numerically calculating companion zero locations...")
        # For the original function F(x,q)
        original_companions = [find_companion_zero(p, q, self.config) for p in odd_primes]
        
        # For the optimized function F_sharp(x,q), only check for p=3
        sin_companion_p3 = find_companion_zero_sin(3, q, self.config)
        sin_companions = []
        if 2 < sin_companion_p3 < 3:
             sin_companions.append(sin_companion_p3)
             print(f"      Found valid companion for F_sharp at x = {sin_companion_p3:.6f}")

        computation_time = time.time() - start_time
        print(f"    INFO: Data and zero calculation finished in {computation_time:.2f} seconds.")

        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.OPTIMIZED_PLOT_FIGSIZE, sharex=True)
        fig.suptitle(f'Visualizing the Elimination of Companion Zeros (q={q})', fontsize=20, y=0.96)

        # --- Top Panel: Original Function ---
        ax1.plot(x_smooth, y_original, c='darkslateblue', lw=2, label=r'Original Indicator $\mathfrak{F}(x,q)$')
        ax1.axhline(0, color='black', lw=0.7, ls='--')
        ax1.set_title('Original Function with Companion Zeros', fontsize=16, pad=10)
        ax1.scatter(primes, np.zeros_like(primes), c='gold', s=100, marker='*', zorder=10, edgecolors='black', linewidth=0.5, label='Prime Zeros')
        ax1.scatter(original_companions, np.zeros(len(original_companions)), c='firebrick', s=30, zorder=9, label='Companion Zeros (Numerically Calculated)')
        ax1.set_ylabel("Value", fontsize=12)
        ax1.grid(True, linestyle=':')
        ax1.legend()
        ax1.set_ylim(-0.03, 0.03)
        
        # --- Bottom Panel: Optimized Function ---
        ax2.plot(x_smooth, y_optimized, c='seagreen', lw=2, label=r'Optimized Indicator $\mathfrak{F}^{\sharp}(x,q)$')
        ax2.axhline(0, color='black', lw=0.7, ls='--')
        ax2.set_title(r'Optimized Function: Companion Zeros for $p \geq 3$ are Eliminated', fontsize=16, pad=10)
        ax2.scatter(primes, np.zeros_like(primes), c='gold', s=100, marker='*', zorder=10, edgecolors='black', linewidth=0.5, label='Prime Zeros (Preserved)')
        if sin_companions:
            ax2.scatter(sin_companions, np.zeros_like(sin_companions), c='firebrick', s=30, zorder=9, label='Companion Zero for p=3 (Remains)')
        ax2.set_xlabel("x", fontsize=12)
        ax2.set_ylabel("Value", fontsize=12)
        ax2.grid(True, linestyle=':')
        ax2.legend()
        ax2.set_ylim(-0.03, 0.03)

        # --- Save the Figure ---
        figures_dir = os.path.join(os.path.dirname(__file__), "../figures")
        filename = os.path.join(figures_dir, f"plot_companion_elimination.{self.config.PLOT_FILE_FORMAT}")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"    SUCCESS: Saved companion zero elimination plot to '{filename}'.")
        plt.close(fig)

    def generate_domain_coloring_plot(self, q_val, x_range, y_range, filename):
        """
        Generates a domain coloring plot for F(z, q) to visualize its
        complex zeros, and poles.
        """
        # Import the colors module explicitly to avoid namespace conflicts
        import matplotlib.colors

        print(f"\n--- Generating Domain Coloring Plot for q={q_val} [{filename}] ---")
        x_min, x_max = x_range
        y_min, y_max = y_range
        x_res = self.config.DOMAIN_PLOT_RESOLUTION
        y_res = int(x_res * (y_max - y_min) / (x_max - x_min))
        
        start_time = time.time()

        # --- Create the complex grid ---
        x = np.linspace(x_min, x_max, x_res)
        y = np.linspace(y_min, y_max, y_res)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        # --- Calculate F(z,q) on the grid ---
        print(f"    INFO: Computing F(z,q) on a {x_res}x{y_res} grid... (this can be slow)")
        F_vectorized = np.vectorize(mathfrak_F_complex)
        W = np.zeros(Z.shape, dtype=np.complex128)
        for i in tqdm(range(y_res), desc="      Progress"):
            W[i, :] = F_vectorized(Z[i, :], q_val, self.config.TRUNCATION_LIMIT)
        print(f"\n    INFO: Grid computation finished in {time.time() - start_time:.2f} seconds.")

        # --- Map complex values to HSV color space ---
        # Hue is determined by the argument (phase)
        hue = (np.angle(W) + np.pi) / (2 * np.pi)
        
        # Saturation is kept constant for vibrancy
        saturation = np.ones_like(hue)
        
        # Value (brightness) is determined by the magnitude
        magnitude = np.abs(W)
        value = (1 - 1 / (1 + magnitude**0.3)) # This inverts the brightness: 0 maps to bright/white, infinity to black
        
        # Stack H, S, V to create an (M, N, 3) HSV image array
        hsv_image = np.stack([hue, saturation, value], axis=-1)
        
        # Convert HSV image to RGB using the explicit matplotlib function call
        rgb_image = matplotlib.colors.hsv_to_rgb(hsv_image)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=self.config.DOMAIN_PLOT_FIGSIZE)
        
        ax.imshow(rgb_image, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto')
        
        # Overlay prime locations for context
        primes_in_range = [p for p in self.primes if x_min < p < x_max]
        ax.scatter(primes_in_range, np.zeros_like(primes_in_range), 
               facecolors='none', edgecolors='red', marker='^', s=30, lw=1.0,
               alpha=0.8, 
               label='Prime Numbers (Real Zeros)')

        # --- Labels and Aesthetics ---
        ax.set_title(fr'Domain Coloring of $\mathfrak{{F}}(z, {q_val})$', fontsize=20, pad=15)
        ax.set_xlabel("Real Part Re(z)", fontsize=14)
        ax.set_ylabel("Imaginary Part Im(z)", fontsize=14)
        ax.legend()
        
        # --- Save the Figure ---
        plt.savefig(filename, dpi=self.config.PLOT3D_DPI, bbox_inches='tight')
        print(f"    SUCCESS: Saved domain coloring plot to '{filename}'.")
        plt.close(fig)

    def generate_3d_magnitude_plot(self, elevation, azimuth, q, x_range, y_range, filename_suffix):
        """
        Generates a 3D surface plot of the magnitude of F(z,q),
        representing the "Zero-Structure".
        """        
        print(f"\n--- Generating 3D Magnitude Plot ('Zero-Structure') for q={q} ---")
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        x_res, y_res = self.config.COMPLEX_3D_GRID_RESOLUTION
        
        start_time = time.time()

        x_real = np.linspace(x_min, x_max, x_res)
        y_imag = np.linspace(y_min, y_max, y_res)
        X, Y = np.meshgrid(x_real, y_imag)
        Z_grid = X + 1j * Y
        W = np.zeros(Z_grid.shape, dtype=np.complex128)

        print("    INFO: Computing function values on the 3D grid...")
        total_rows = Z_grid.shape[0]
        F_vectorized = np.vectorize(mathfrak_F_complex)
        for i in tqdm(range(total_rows), desc="      Progress"):
            W[i, :] = F_vectorized(Z_grid[i, :], q, self.config.TRUNCATION_LIMIT)

        W_magnitude_log = np.log1p(np.abs(W))
        
        computation_time = time.time() - start_time
        print(f"    INFO: Grid computation finished in {computation_time:.2f} seconds.")
        
        fig = plt.figure(figsize=self.config.PLOT_3D_FIGSIZE)
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, W_magnitude_log, cmap=cm.inferno_r, rstride=1, cstride=1, 
                               antialiased=True, edgecolor='black', linewidth=0.05, alpha=0.8)
        
        ax.set_title(r"Zero-Structure: $|\mathfrak{F}(x, %s)|$ on $(\mathbb{R})$" % f"{q:g}", fontsize=20, pad=20)
        ax.set_xlabel(r"Real Part (Re z)", fontsize=14, labelpad=10)
        ax.set_ylabel(r"Imaginary Part (Im z)", fontsize=14, labelpad=10)
        ax.set_zlabel(r"Log Magnitude - $\log(1 + |\mathfrak{F}|)$", fontsize=14, labelpad=10)
        
        ax.view_init(elev=elevation, azim=azimuth)
        
        fig.colorbar(surf, shrink=0.5, aspect=10, label="Log Magnitude")

        figures_dir = os.path.join(os.path.dirname(__file__), "../figures")
        if not os.path.isdir(figures_dir):
            os.makedirs(figures_dir, exist_ok=True)
            print(f"    INFO: Created figures directory at '{figures_dir}'")
            
        filename = os.path.join(figures_dir, f"plot_3d_{filename_suffix}.{self.config.PLOT3D_FILE_FORMAT}")

        plt.savefig(filename, dpi=self.config.PLOT3D_DPI, bbox_inches='tight')
        print(f"    SUCCESS: Saved 3D plot with perspective to '{filename}'.")
        plt.close(fig)

    def generate_3d_magnitude_plot_shaded(
            self,
            elevation,
            azimuth,
            q,
            x_range,
            y_range,
            filename_suffix,
        ):
        """
        3D magnitude plot with lighting-based shading, gradient-alpha (pseudo AO),
        customizable colormap, and extra depth cues (base-plane contours, sparse wireframe).
        """

        # =========================
        # CONFIG (tweak here)
        # =========================
        CFG = {
            # ---- Colormap & Zero-color control ----
            # Name of the base colormap (good choices: 'inferno_r', 'magma_r', 'cividis', 'turbo')
            "cmap_name": "cividis",
            # If True, reverse the colormap (ignored if name already ends with '_r').
            "reverse_cmap": False,
            # Override the very low end color (near zeros). Example to replace yellow:
            # e.g. (0.10, 0.10, 0.10) for dark grey, or (0.0, 0.0, 0.0) for black.
            "low_color_override": None, #(0.08, 0.08, 0.08),   # None to disable
            # Fraction of the colormap to override near the low end (0.01–0.10 sensible)
            "low_clip_frac": 0.05,
            # Optionally override high end color (far from zeros). None to disable.
            "high_color_override": None,
            "high_clip_frac": 0.00,

            # ---- Normalization (tone mapping) ----
            # Choose 'power' (recommended), 'linear', or 'log' (adds epsilon for zero).
            "norm_mode": "power",
            # PowerNorm gamma: smaller => more emphasis/contrast near low values (zeros).
            # Typical range: 0.35–0.9, default 0.65
            "gamma": 0.2,
            # Percentile clipping to keep mid/low structure expressive (1–5% & 95–99% good)
            "vmin_percentile": 1.0,
            "vmax_percentile": 99.0,
            # Epsilon for 'log' mode (ignored otherwise)
            "log_epsilon": 1e-6,

            # ---- Lighting/shading ----
            # If None, reuse camera elevation (clamped). Lower altitude => longer shadows.
            "light_altitude_deg": None,     # default None, e.g. 25–40 to accentuate ridges/valleys
            # Light azimuth offset relative to camera; small negative values often good.
            "light_azimuth_offset_deg": -15,
            # Vertical exaggeration for the shader’s slope estimation (1.0–2.0)
            "vert_exag": 1.2, # default 1.2
            # Blend mode for LightSource: 'overlay' (crisp), 'soft' (gentler), or 'hsv'
            "blend_mode": "overlay",

            # ---- Alpha (pseudo ambient occlusion) ----
            # Opacity for flat regions (0.2–0.7)
            "alpha_flat_min": 0.6,
            # Opacity for steep regions (0.7–1.0)
            "alpha_steep_max": 0.8,
            # Percentiles for slope normalization (spread controls)
            "slope_lo_pct": 5.0,
            "slope_hi_pct": 95.0,

            # ---- Overlays ----
            # Wireframe density: larger => sparser. 0 disables wireframe.
            "wireframe_density": 100, # default 80
            "wireframe_alpha": 0.1, # default 0.18
            "wireframe_lw": 0.01, # default 0.35
            # Base-plane contour lines for depth cue
            "contour_levels": 20, # default 22
            "contour_cmap": "Greys",
            "contour_alpha": 0.55,

            # ---- Edges / output ----
            "edge_alpha": 0.20,
            "edge_lw": 0.01,
            # Rasterize surface for smaller vector outputs; safe for PNGs as well.
            "rasterize_surface": False,

            # ---- Performance (optional) ----
            # Fast preview: compute at reduced grid resolution (e.g. 0.5 = half res)
            "fast_preview_factor": 1.0,  # set to 0.5 for ~4x faster/less memory
        }
        # =========================
        # END CONFIG
        # =========================

        # ----- Helpers -----
        def _make_cmap():
            """Build a ListedColormap with optional low/high overrides."""
            base = cm.get_cmap(CFG["cmap_name"])
            if CFG["reverse_cmap"] and not CFG["cmap_name"].endswith("_r"):
                base = base.reversed()
            N = 256
            colors = base(np.linspace(0, 1, N))
            # Low end override
            if CFG["low_color_override"] is not None and CFG["low_clip_frac"] > 0:
                k = max(1, int(N * float(CFG["low_clip_frac"])))
                colors[:k, :3] = np.array(CFG["low_color_override"][:3])
            # High end override
            if CFG["high_color_override"] is not None and CFG["high_clip_frac"] > 0:
                k = max(1, int(N * float(CFG["high_clip_frac"])))
                colors[-k:, :3] = np.array(CFG["high_color_override"][:3])
            return ListedColormap(colors)

        def _make_norm(arr):
            """Create a normalization (linear/power/log) with percentile clipping."""
            vmin = np.percentile(arr, CFG["vmin_percentile"])
            vmax = np.percentile(arr, CFG["vmax_percentile"])
            if CFG["norm_mode"] == "linear":
                return Normalize(vmin=vmin, vmax=vmax)
            elif CFG["norm_mode"] == "log":
                # Ensure strictly positive values
                eps = float(CFG["log_epsilon"])
                return LogNorm(vmin=max(vmin, eps), vmax=max(vmax, vmin + eps))
            else:
                # 'power'
                return PowerNorm(gamma=float(CFG["gamma"]), vmin=vmin, vmax=vmax)

        print(f"\n--- Generating 3D Shaded Magnitude Plot for q={q} ---")
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Resolution (allow fast preview)
        base_xres, base_yres = self.config.COMPLEX_3D_GRID_RESOLUTION_SHADED
        scale = float(CFG["fast_preview_factor"])
        x_res = max(64, int(base_xres * scale))
        y_res = max(64, int(base_yres * scale))

        # -------------------------
        # Compute complex grid & F
        # -------------------------
        start_time = time.time()
        x_real = np.linspace(x_min, x_max, x_res)
        y_imag = np.linspace(y_min, y_max, y_res)
        X, Y = np.meshgrid(x_real, y_imag)
        Z_grid = X + 1j * Y
        W = np.zeros(Z_grid.shape, dtype=np.complex128)

        print("    INFO: Computing function values on the 3D grid...")
        F_vectorized = np.vectorize(mathfrak_F_complex)
        for i in tqdm(range(Z_grid.shape[0]), desc="      Progress"):
            W[i, :] = F_vectorized(Z_grid[i, :], q, self.config.TRUNCATION_LIMIT)

        data = np.log1p(np.abs(W))  # z-values
        comp_time = time.time() - start_time
        print(f"    INFO: Grid computation finished in {comp_time:.2f} seconds.")

        # -------------------------
        # Colormap, normalization, lighting
        # -------------------------
        cmap = _make_cmap()
        norm = _make_norm(data)

        light_alt = float(np.clip(CFG["light_altitude_deg"] if CFG["light_altitude_deg"] is not None else elevation, 15, 75))
        ls = LightSource(
            azdeg=(azimuth + float(CFG["light_azimuth_offset_deg"])) % 360,
            altdeg=light_alt
        )

        dx = (x_max - x_min) / max(1, (x_res - 1))
        dy = (y_max - y_min) / max(1, (y_res - 1))

        shaded = ls.shade(
            data, cmap=cmap, norm=norm,
            vert_exag=float(CFG["vert_exag"]),
            dx=dx, dy=dy, blend_mode=str(CFG["blend_mode"])
        )

        # -------------------------
        # Pseudo ambient occlusion (gradient-based alpha)
        # -------------------------
        dzdx = np.gradient(data, x_real, axis=1)
        dzdy = np.gradient(data, y_imag, axis=0)
        slope_mag = np.hypot(dzdx, dzdy)
        s_lo = np.percentile(slope_mag, CFG["slope_lo_pct"])
        s_hi = np.percentile(slope_mag, CFG["slope_hi_pct"])
        slope_norm = np.clip((slope_mag - s_lo) / max(1e-12, (s_hi - s_lo)), 0.0, 1.0)

        alpha_flat = float(CFG["alpha_flat_min"])
        alpha_steep = float(CFG["alpha_steep_max"])
        alpha_map = np.clip(alpha_flat + (alpha_steep - alpha_flat) * slope_norm, 0.0, 1.0).astype(np.float32)

        shaded = np.asarray(shaded, dtype=np.float32)
        channels = shaded.shape[-1] if shaded.ndim == 3 else 3
        if channels == 3:
            rgba_vertex = np.dstack([shaded, alpha_map])
        else:
            new_alpha = np.clip(shaded[..., 3] * alpha_map, 0.0, 1.0)
            rgba_vertex = np.dstack([shaded[..., :3], new_alpha])

        # Per-face RGBA (M-1,N-1,4)
        M, N = data.shape
        face_rgba = np.ascontiguousarray(rgba_vertex[:M-1, :N-1, :4], dtype=np.float32)

        # -------------------------
        # Plot
        # -------------------------
        fig = plt.figure(figsize=self.config.PLOT_3D_FIGSIZE)
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(
            X, Y, data,
            facecolors=face_rgba,   # per-face RGBA grid (M-1,N-1,4)
            rstride=5, cstride=5,
            linewidth=float(CFG["edge_lw"]),
            antialiased=True,
            shade=False
        )
        if CFG["rasterize_surface"]:
            surf.set_rasterized(True)

        # Thin edges after creation (safer across mpl versions)
        surf.set_edgecolor((0, 0, 0, float(CFG["edge_alpha"])))
        surf.set_linewidth(float(CFG["edge_lw"]))
        try:
            surf.set_zsort('min')
        except Exception:
            pass

        # Base-plane contours
        zmin = float(np.min(data))
        ax.contour(
            X, Y, data,
            zdir='z', offset=zmin,
            levels=int(CFG["contour_levels"]),
            cmap=str(CFG["contour_cmap"]),
            linewidths=0.6, alpha=float(CFG["contour_alpha"])
        )

        # Sparse wireframe
        den = int(CFG["wireframe_density"])
        if den > 0:
            step = max(1, min(M, N) // den)
            ax.plot_wireframe(
                X[::step, ::step], Y[::step, ::step], data[::step, ::step],
                color=(0, 0, 0, float(CFG["wireframe_alpha"])),
                linewidth=float(CFG["wireframe_lw"]),
                rstride=1, cstride=1
            )

        # Labels/view
        ax.set_title(r"The Zero-Structure (Shaded): "
                    r"$\log(1+|\mathfrak{F}(z," + f"{q}" + r")|)$",
                    fontsize=20, pad=20)
        ax.set_xlabel(r"Real Part (Re z)", fontsize=14, labelpad=10)
        ax.set_ylabel(r"Imaginary Part (Im z)", fontsize=14, labelpad=10)
        ax.set_zlabel(r"Log Magnitude", fontsize=14, labelpad=10)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_zlim(zmin, float(np.max(data)))

        # Colorbar (explicitly bound to ax)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(data)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, pad=0.02)
        cbar.set_label("Log Magnitude")

        # Save
        figures_dir = os.path.join(os.path.dirname(__file__), "../figures")
        os.makedirs(figures_dir, exist_ok=True)
        filename = os.path.join(figures_dir, f"plot_3d_{filename_suffix}.{self.config.PLOT3D_FILE_FORMAT}")
        plt.savefig(filename, dpi=self.config.PLOT3D_DPI, bbox_inches='tight')
        print(f"    SUCCESS: Saved shaded 3D plot to '{filename}'.")
        plt.close(fig)


    def generate_q_analog_plot(self):
        """
        Visualizes how the analytic q-analog F_tau(x,q) for various q-values
        converges towards the classical, discrete divisor-counting function tau(n).
        """
        q_values = self.config.Q_ANALOG_Q_VALUES
        print(f"\n--- Generating q-Analog Interpolation Plot for q-values: {q_values} ---")
        x_min, x_max = self.config.Q_ANALOG_PLOT_X_RANGE
        start_time = time.time()

        # 1. Prepare the continuous x-axis
        x_continuous = np.linspace(x_min, x_max, 4000)

        # 2. Calculate the discrete reference values (for q=1.0)
        integers = np.arange(int(x_min), int(x_max) + 1)
        tau_values = np.array([divisor_count(n) for n in integers])
        y_discrete = tau_values - 2

        # 3. Setup the plot
        fig, ax = plt.subplots(figsize=self.config.PLOT_Q_ANALOG_FIGSIZE)
        
        # Color generator for the curves (from violet to blue/turquoise)
        colors = cm.viridis(np.linspace(0, 0.8, len(q_values)))

        # 4. Loop over all q-values to draw the curves
        for i, q in enumerate(q_values):
            print(f"    INFO: Computing data for q={q}...")
            # Compute the y-values for the current q-value
            y_continuous = np.vectorize(mathfrak_F_tau_complex)(x_continuous, q, self.config.TRUNCATION_LIMIT)
            
            # Plot the curve with a unique color and label
            ax.plot(x_continuous, y_continuous.real, color=colors[i], lw=1.5, label=fr'$\mathfrak{{F}}_\tau(x, q={q})$')
        
        computation_time = time.time() - start_time
        print(f"    INFO: Data computation finished in {computation_time:.2f} seconds.")

        # 5. Plot the discrete points for q=1.0 (as a reference)
        markerline, stemlines, baseline = ax.stem(
            integers, y_discrete,
            linefmt='orangered', markerfmt='D', basefmt='grey',
            label=r'$\tau(n)-2$  (Limit for $q \to 1$)'
        )
        plt.setp(stemlines, 'linewidth', 1.5, 'alpha', 0.7)
        plt.setp(markerline, 'markersize', 7)
        plt.setp(baseline, 'alpha', 0.5)

        # 6. Format the plot
        ax.set_title(r"Convergence of the q-Analog $\mathfrak{F}_\tau(x,q)$ towards $\tau(n)$", fontsize=20, pad=20)
        ax.set_xlabel(r"Real Axis ($x$)", fontsize=16)
        ax.set_ylabel(r"Function Value", fontsize=16)
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.grid(True, which='major', linestyle='--', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=12, title=r"Functions for $q$-values")

        # 7. Save the plot
        figures_dir = os.path.join(os.path.dirname(__file__), "../figures")
        os.makedirs(figures_dir, exist_ok=True)
        filename = os.path.join(figures_dir, f"plot_q_analog_interpolation.{self.config.PLOT_FILE_FORMAT}")

        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"    SUCCESS: Saved q-analog plot to '{filename}'.")
        plt.close(fig)


    def generate_complex_zero_plot(self, q_value, x_range, y_range, filename_suffix):
        """
        Generates a plot of the complex zeros of F(z,q) for a specific q and y_range.
        """
        print(f"\n--- Generating Complex Zero Plot for q={q_value} ---")
        print(f"    Range: x in {x_range}, y in {y_range}")
        print(f"    Grid Resolution: {self.config.COMPLEX_GRID_RESOLUTION[0]} x {self.config.COMPLEX_GRID_RESOLUTION[1]} points")
        
        start_time = time.time()

        x_real = np.linspace(x_range[0], x_range[1], self.config.COMPLEX_GRID_RESOLUTION[0])
        y_imag = np.linspace(y_range[0], y_range[1], self.config.COMPLEX_GRID_RESOLUTION[1])
        X, Y = np.meshgrid(x_real, y_imag)
        Z = X + 1j * Y
        W = np.zeros(Z.shape, dtype=np.complex128)

        # Vectorize the core function for applying to rows
        F_vectorized = np.vectorize(mathfrak_F_complex)
        
        # Compute row by row to provide progress feedback
        print("    INFO: Computing function values on the complex grid...")
        total_rows = Z.shape[0]
        for i in tqdm(range(total_rows), desc="      Progress"):
            W[i, :] = F_vectorized(Z[i, :], q_value, self.config.TRUNCATION_LIMIT)
        
        computation_time = time.time() - start_time
        print(f"    INFO: Grid computation finished in {computation_time:.2f} seconds.")

        fig, ax = plt.subplots(figsize=self.config.PLOT_COMPLEX_FIGSIZE)

        ax.contour(X, Y, W.real, levels=[0], colors='dodgerblue', linewidths=1.0)
        ax.contour(X, Y, W.imag, levels=[0], colors='orangered', linewidths=1.0)

        primes_in_range = sorted([p for p in self.primes if x_range[0] < p < x_range[1]])
        ax.plot(primes_in_range, np.zeros_like(primes_in_range), 'g^', markersize=8, label='Prime Zeros on Real Axis')
        
        ax.set_title(r"Zeros of $\mathfrak{F}(z, " + f"{q_value}" + r")$ in the Complex Plane", fontsize=20, pad=20)
        ax.set_xlabel(r"Real Part (Re z)", fontsize=16)
        ax.set_ylabel(r"Imaginary Part (Im z)", fontsize=16)
        
        ax.set_aspect('auto')
        ax.axhline(0, color='black', lw=0.5, linestyle='--')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)

        legend_elements = [Line2D([0], [0], color='dodgerblue', lw=2, label='Re($\mathfrak{F}$) = 0'),
                           Line2D([0], [0], color='orangered', lw=2, label='Im($\mathfrak{F}$) = 0'),
                           Line2D([0], [0], marker='^', color='g', label='Prime Zeros (z = p)', linestyle='None', markersize=10)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=14)

        figures_dir = os.path.join(os.path.dirname(__file__), "../figures")
        if not os.path.isdir(figures_dir):
            os.makedirs(figures_dir, exist_ok=True)
            print(f"    INFO: Created figures directory at '{figures_dir}'")

        filename = os.path.join(figures_dir, f"plot_complex_zeros_{filename_suffix}.{self.config.PLOT_FILE_FORMAT}")
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"    SUCCESS: Saved complex zero plot to '{filename}'.")
        plt.close(fig)

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    config = Config()

    # --- Create figures directory if it doesn't exist ---
    try:
        figures_dir = os.path.join(os.path.dirname(__file__), "../figures")
        if not os.path.isdir(figures_dir):
            os.makedirs(figures_dir, exist_ok=True)
            print(f"INFO: Created figures directory at '{figures_dir}'")
    except NameError: # Handle case where __file__ is not defined (e.g., in some notebooks)
        figures_dir = "figures"
        if not os.path.isdir(figures_dir):
            os.makedirs(figures_dir, exist_ok=True)
            print(f"INFO: Created figures directory at '{figures_dir}'")

    plotter = Plotter(config)


    plotter.generate_3d_magnitude_plot_shaded(elevation=70, azimuth=175, q=-1.0, x_range=(-15, 15), y_range=(-5, 5), filename_suffix="shaded_neg_q")

    plotter.generate_3d_magnitude_plot(elevation=70, azimuth=170, q=-1.0, x_range=(-30, 30), y_range=(-6, 6), filename_suffix="real_axis_neg_q")
    plotter.generate_3d_magnitude_plot(elevation=70, azimuth=170, q=config.Q_PARAMETER_LOW, x_range=config.COMPLEX_3D_X_RANGE, y_range=config.COMPLEX_3D_Y_RANGE, filename_suffix="real_axis")
    plotter.generate_3d_magnitude_plot(elevation=0, azimuth=-90, q=config.Q_PARAMETER_LOW, x_range=config.COMPLEX_3D_X_RANGE_SIDE, y_range=config.COMPLEX_3D_Y_RANGE_SIDE, filename_suffix="side")

    # --- Generate the plot for the q-analog interpolation ---
    plotter.generate_q_analog_plot()

    # --- Generate the plot for the Lambda(n) interpolant ---
    plotter.generate_lambda_interpolant_plot()

    # --- Generate the plot for the Moebius lift ---
    plotter.generate_mu_interpolant_plot()

    # --- Generate the plot for the differentiation operator effect ---
    plotter.generate_differentiation_operator_plot()
    
    # --- Generate the plot for the companion zero displacement ---
    plotter.generate_companion_displacement_plot()

    # --- Generate the plot for the companion zero elimination ---
    plotter.generate_companion_zero_elimination_plot()

    # --- Generate the domain coloring plot for q=-1 ---
    plotter.generate_domain_coloring_plot(
        q_val=-1.0,
        x_range=config.DOMAIN_PLOT_X_RANGE,
        y_range=config.DOMAIN_PLOT_Y_RANGE,
        filename=os.path.join(figures_dir, f"plot_domain_coloring_q_neg1.{config.PLOT3D_FILE_FORMAT}")
    )

    # --- Generate the existing plots ---
    plotter.generate_complex_zero_plot(
        q_value=config.Q_PARAMETER_NEG,
        x_range=(-50, 50),
        y_range=(-10, 10),
        filename_suffix=f"q{int(config.Q_PARAMETER_NEG)}"
    )
        
    # --- Generate the existing plots ---
    plotter.generate_complex_zero_plot(
        q_value=config.Q_PARAMETER_LOW,
        x_range=config.COMPLEX_ZERO_X_RANGE,
        y_range=config.COMPLEX_ZERO_Y_RANGE_LOW_Q,
        filename_suffix=f"q{int(config.Q_PARAMETER_LOW)}"
    )

    plotter.generate_complex_zero_plot(
        q_value=config.Q_PARAMETER_HIGH,
        x_range=config.COMPLEX_ZERO_X_RANGE,
        y_range=config.COMPLEX_ZERO_Y_RANGE_HIGH_Q,
        filename_suffix=f"q{int(config.Q_PARAMETER_HIGH)}"
    )
       
    print("\nScript finished successfully.")