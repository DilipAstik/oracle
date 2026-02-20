"""
Black-Scholes European option pricing and implied volatility solver.

Module: src/oracle/compute/black_scholes.py
Phase: B5 — IV Computation Service
Spec References:
    - TDR-ORACLE-008 §2.4 (IV computation parameters)
    - Data Contract §3.5 (RIV computation)
    - B5 Design §3 (Black-Scholes implementation)

This module provides:
    - European call/put pricing (Black-Scholes)
    - Analytical vega computation
    - Newton-Raphson IV solver with convergence guarantees
    - Brenner-Subrahmanyam warm-start for ATM options

All computations use float64 arithmetic for reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy.stats import norm

# Enforce strict floating-point error handling for reproducibility (R2 refinement)
np.seterr(all="raise")


# --- Constants ---

IV_STATUS_CONVERGED = "CONVERGED"
IV_STATUS_NOT_CONVERGED = "NOT_CONVERGED"
IV_STATUS_SOLVER_FAILED = "SOLVER_FAILED"
IV_STATUS_EXTREME_VALUE = "EXTREME_VALUE"
IV_STATUS_BELOW_INTRINSIC = "BELOW_INTRINSIC"
IV_STATUS_EXPIRY_DAY = "EXPIRY_DAY"
IV_STATUS_PRICE_TOO_LOW = "PRICE_TOO_LOW"
IV_STATUS_OUT_OF_MONEYNESS = "OUT_OF_MONEYNESS"
IV_STATUS_MISSING_FUTURES = "MISSING_FUTURES"
IV_STATUS_MISSING_RFR = "MISSING_RFR"


@dataclass(frozen=True)
class BSParams:
    """Black-Scholes input parameters.

    All values are in natural units:
        S, K in index points (e.g., 24000.0)
        T in years (e.g., 0.0274 for 10 calendar days)
        r, q as annualised decimals (e.g., 0.065 for 6.5%)
    """
    S: float   # Spot price (underlying index level)
    K: float   # Strike price
    T: float   # Time to expiry in years (calendar days / 365)
    r: float   # Risk-free rate (annualised, continuous compounding)
    q: float   # Dividend yield (annualised, continuous compounding)

    def __post_init__(self) -> None:
        """Validate inputs — all must be finite float64."""
        for field_name in ('S', 'K', 'T', 'r', 'q'):
            val = getattr(self, field_name)
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"BSParams.{field_name} must be finite, got {val}")
        if self.S <= 0:
            raise ValueError(f"BSParams.S must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"BSParams.K must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"BSParams.T must be positive, got {self.T}")


def _d1_d2(params: BSParams, sigma: float) -> Tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes formula.

    d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    """
    S, K, T, r, q = params.S, params.K, params.T, params.r, params.q
    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T

    if sigma_sqrt_T < 1e-15:
        return 0.0, 0.0

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


def bs_call_price(params: BSParams, sigma: float) -> float:
    """Black-Scholes European call price.

    C = S * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)
    """
    S, K, T, r, q = params.S, params.K, params.T, params.r, params.q
    d1, d2 = _d1_d2(params, sigma)
    return (
        S * math.exp(-q * T) * float(norm.cdf(d1))
        - K * math.exp(-r * T) * float(norm.cdf(d2))
    )


def bs_put_price(params: BSParams, sigma: float) -> float:
    """Black-Scholes European put price.

    P = K * exp(-r * T) * N(-d2) - S * exp(-q * T) * N(-d1)
    """
    S, K, T, r, q = params.S, params.K, params.T, params.r, params.q
    d1, d2 = _d1_d2(params, sigma)
    return (
        K * math.exp(-r * T) * float(norm.cdf(-d2))
        - S * math.exp(-q * T) * float(norm.cdf(-d1))
    )


def bs_price(params: BSParams, option_type: str, sigma: float) -> float:
    """Black-Scholes price for call or put.

    Args:
        option_type: "CE" for call, "PE" for put
    """
    if option_type == "CE":
        return bs_call_price(params, sigma)
    elif option_type == "PE":
        return bs_put_price(params, sigma)
    else:
        raise ValueError(f"option_type must be 'CE' or 'PE', got '{option_type}'")


def bs_vega(params: BSParams, sigma: float) -> float:
    """Black-Scholes vega (dC/dsigma = dP/dsigma).

    vega = S * exp(-q * T) * sqrt(T) * n(d1)

    Where n(.) is the standard normal PDF.
    Returns vega in price units per unit of sigma (not per percentage point).
    """
    S, K, T, r, q = params.S, params.K, params.T, params.r, params.q
    d1, _ = _d1_d2(params, sigma)
    return S * math.exp(-q * T) * math.sqrt(T) * float(norm.pdf(d1))


def intrinsic_value(params: BSParams, option_type: str) -> float:
    """Approximate intrinsic value (discounted).

    Call: max(S * exp(-qT) - K * exp(-rT), 0)
    Put:  max(K * exp(-rT) - S * exp(-qT), 0)
    """
    S, K, T, r, q = params.S, params.K, params.T, params.r, params.q
    forward_S = S * math.exp(-q * T)
    pv_K = K * math.exp(-r * T)

    if option_type == "CE":
        return max(forward_S - pv_K, 0.0)
    elif option_type == "PE":
        return max(pv_K - forward_S, 0.0)
    else:
        raise ValueError(f"option_type must be 'CE' or 'PE', got '{option_type}'")


def _brenner_subrahmanyam_guess(params: BSParams, market_price: float) -> float:
    """Brenner-Subrahmanyam initial guess for ATM options.

    sigma_0 ~ P_obs * sqrt(2*pi) / (S * sqrt(T))

    Used when |ln(S/K)| < 0.03 (near-ATM).
    """
    S, T = params.S, params.T
    sqrt_T = math.sqrt(T)
    if S * sqrt_T < 1e-10:
        return 0.20
    guess = market_price * math.sqrt(2.0 * math.pi) / (S * sqrt_T)
    return max(0.01, min(guess, 3.0))


def implied_volatility(
    params: BSParams,
    option_type: str,
    market_price: float,
    tol: float = 1e-8,
    max_iter: int = 50,
    sigma_low: float = 0.005,
    sigma_high: float = 5.0,
    vega_floor: float = 1e-10,
) -> Tuple[Optional[float], str, int]:
    """Solve for implied volatility using Newton-Raphson.

    Args:
        params: Black-Scholes input parameters
        option_type: "CE" or "PE"
        market_price: Observed option price (settlement_price)
        tol: Convergence tolerance in price space
        max_iter: Maximum Newton-Raphson iterations
        sigma_low: Lower bound for IV (0.5% annualised)
        sigma_high: Upper bound for IV (500% annualised)
        vega_floor: Minimum vega to prevent division by zero

    Returns:
        Tuple of (iv, status, iterations):
            iv: Implied volatility (annualised, decimal) or None if failed
            status: One of the IV_STATUS_* constants
            iterations: Number of iterations performed
    """
    if market_price <= 0:
        return None, IV_STATUS_PRICE_TOO_LOW, 0

    # Check if price is below intrinsic value
    iv_approx = intrinsic_value(params, option_type)
    if market_price < iv_approx - 1.0:
        return None, IV_STATUS_BELOW_INTRINSIC, 0

    # Initial guess: Brenner-Subrahmanyam for near-ATM, flat 0.20 otherwise
    log_moneyness = abs(math.log(params.S / params.K))
    if log_moneyness < 0.03:
        sigma = _brenner_subrahmanyam_guess(params, market_price)
    else:
        sigma = 0.20

    # Newton-Raphson iteration
    for i in range(1, max_iter + 1):
        try:
            price = bs_price(params, option_type, sigma)
            v = bs_vega(params, sigma)
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            return None, IV_STATUS_SOLVER_FAILED, i

        if v < vega_floor:
            return None, IV_STATUS_SOLVER_FAILED, i

        price_diff = price - market_price

        if abs(price_diff) < tol:
            if sigma < 0.02 or sigma > 2.0:
                return float(np.float64(sigma)), IV_STATUS_EXTREME_VALUE, i
            return float(np.float64(sigma)), IV_STATUS_CONVERGED, i

        # Newton step
        sigma_new = sigma - price_diff / v

        # Clamp to bounds
        sigma_new = max(sigma_low, min(sigma_new, sigma_high))

        # Check for oscillation (same sigma after clamping)
        if abs(sigma_new - sigma) < 1e-15:
            if abs(price_diff) < 0.01:
                if sigma_new < 0.02 or sigma_new > 2.0:
                    return float(np.float64(sigma_new)), IV_STATUS_EXTREME_VALUE, i
                return float(np.float64(sigma_new)), IV_STATUS_CONVERGED, i
            return None, IV_STATUS_SOLVER_FAILED, i

        sigma = sigma_new

    # Max iterations reached
    return None, IV_STATUS_NOT_CONVERGED, max_iter
