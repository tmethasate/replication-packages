# src/uo2009/dgp/efficient_price.py
"""
Efficient price DGP for Ubukata & Oya (2009), Section 4.1.

Model (continuous time), for each asset y ∈ {1,2}:

    dP*_y(t) = σ_y(t) [ sqrt(1-λ_y^2) dW_y^(A)(t) + λ_y dW_y^(B)(t) ]

    dσ_y^2(t) = κ_y (θ_y - σ_y^2(t)) dt + ω_y σ_y^2(t) dW_y^(B)(t)

Cross-asset correlation is introduced by correlating W_1^(A) and W_2^(A) with
time-varying correlation ρ*(t) = tanh(x(t)), where:

    dx(t) = κ_3 (θ_3 - x(t)) dt + ω_3 x(t) dW(t)

Discretization: Euler–Maruyama on an internal grid of step dt_internal.

Implementation notes:
- We simulate (W_1^B, W_2^B, W, Z_1, Z_2) i.i.d. per step, and set
  W_1^A = Z_1, W_2^A = ρ Z_1 + sqrt(1-ρ^2) Z_2.
- σ^2 is clamped to a small positive floor for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


_SIGMA2_FLOOR = 1e-12


@dataclass(frozen=True)
class EfficientPricePath:
    """Container for simulated latent paths."""
    t: np.ndarray                 # (n+1,)
    p_star: np.ndarray            # (n+1, 2) latent prices P* for assets 1,2
    sigma2: np.ndarray            # (n+1, 2) latent variances σ^2 for assets 1,2
    x: np.ndarray                 # (n+1,)  correlation state
    rho: np.ndarray               # (n+1,)  rho(t)=tanh(x(t))

    # Optional: increments if you want to debug/validate
    meta: Dict[str, Any]


def _as_float(x) -> float:
    return float(np.asarray(x).item())


def simulate_efficient_price(
    params,
    rng: np.random.Generator,
    *,
    p0: Tuple[float, float] = (0.0, 0.0),
    sigma2_0: Optional[Tuple[float, float]] = None,
    x0: Optional[float] = None,
    return_increments: bool = False,
) -> EfficientPricePath:
    """
    Simulate latent efficient prices and volatility/correlation states.

    Parameters
    ----------
    params:
        Expected to have attributes:
        T, dt_internal,
        lambda1, lambda2,
        kappa1, kappa2, theta1, theta2, omega1, omega2,
        kappa3, theta3, omega3
    rng:
        numpy.random.Generator
    p0:
        initial latent prices (asset1, asset2)
    sigma2_0:
        initial latent variances. If None, defaults to (theta1, theta2).
    x0:
        initial correlation state. If None, defaults to theta3.
    return_increments:
        If True, returns Brownian increments in meta for diagnostics.

    Returns
    -------
    EfficientPricePath
    """
    T = _as_float(params.T)
    dt = _as_float(params.dt_internal)
    if dt <= 0:
        raise ValueError("dt_internal must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")

    n_steps = int(np.floor(T / dt))
    # Use grid up to n_steps*dt (<=T). This is usually what you want for stable indexing.
    t = np.arange(n_steps + 1, dtype=float) * dt

    lam1 = _as_float(params.lambda1)
    lam2 = _as_float(params.lambda2)
    if not (0.0 <= lam1 <= 1.0 and 0.0 <= lam2 <= 1.0):
        raise ValueError("lambda1 and lambda2 must be in [0,1].")

    k1, k2 = _as_float(params.kappa1), _as_float(params.kappa2)
    th1, th2 = _as_float(params.theta1), _as_float(params.theta2)
    om1, om2 = _as_float(params.omega1), _as_float(params.omega2)

    k3, th3, om3 = _as_float(params.kappa3), _as_float(params.theta3), _as_float(params.omega3)

    if sigma2_0 is None:
        sigma2_0 = (th1, th2)
    if x0 is None:
        x0 = th3

    p_star = np.empty((n_steps + 1, 2), dtype=float)
    sigma2 = np.empty((n_steps + 1, 2), dtype=float)
    x = np.empty(n_steps + 1, dtype=float)
    rho = np.empty(n_steps + 1, dtype=float)

    p_star[0, :] = np.array(p0, dtype=float)
    sigma2[0, :] = np.maximum(np.array(sigma2_0, dtype=float), _SIGMA2_FLOOR)
    x[0] = float(x0)
    rho[0] = np.tanh(x[0])

    sqrt_dt = np.sqrt(dt)
    sqrt_1_l1 = np.sqrt(max(1.0 - lam1 * lam1, 0.0))
    sqrt_1_l2 = np.sqrt(max(1.0 - lam2 * lam2, 0.0))

    # Optional storage for debugging/validation
    dW_B = np.empty((n_steps, 2), dtype=float) if return_increments else None
    dW_A = np.empty((n_steps, 2), dtype=float) if return_increments else None
    dW_x = np.empty(n_steps, dtype=float) if return_increments else None

    for i in range(n_steps):
        # Use current states at time i
        s2_1, s2_2 = sigma2[i, 0], sigma2[i, 1]
        sig1 = np.sqrt(max(s2_1, _SIGMA2_FLOOR))
        sig2 = np.sqrt(max(s2_2, _SIGMA2_FLOOR))

        # Current correlation for the A-Brownian components
        r = float(np.tanh(x[i]))
        # numerical safety
        r = min(max(r, -0.999999999), 0.999999999)
        rho[i] = r

        # Independent normals
        z1, z2 = rng.standard_normal(2)
        b1, b2 = rng.standard_normal(2)
        zx = rng.standard_normal()

        # Construct correlated A-increments
        dW_A1 = sqrt_dt * z1
        dW_A2 = sqrt_dt * (r * z1 + np.sqrt(max(1.0 - r * r, 0.0)) * z2)

        # B-increments (independent across assets here)
        dW_B1 = sqrt_dt * b1
        dW_B2 = sqrt_dt * b2

        # x-process increment
        dW_xi = sqrt_dt * zx

        # Price increments use σ(t_i) and the same-step Brownian increments
        dP1 = sig1 * (sqrt_1_l1 * dW_A1 + lam1 * dW_B1)
        dP2 = sig2 * (sqrt_1_l2 * dW_A2 + lam2 * dW_B2)

        p_star[i + 1, 0] = p_star[i, 0] + dP1
        p_star[i + 1, 1] = p_star[i, 1] + dP2

        # Volatility updates (Euler–Maruyama) driven by the same B as price
        s2_1_next = s2_1 + k1 * (th1 - s2_1) * dt + om1 * s2_1 * dW_B1
        s2_2_next = s2_2 + k2 * (th2 - s2_2) * dt + om2 * s2_2 * dW_B2

        sigma2[i + 1, 0] = max(s2_1_next, _SIGMA2_FLOOR)
        sigma2[i + 1, 1] = max(s2_2_next, _SIGMA2_FLOOR)

        # Correlation-state update
        x_next = x[i] + k3 * (th3 - x[i]) * dt + om3 * x[i] * dW_xi
        x[i + 1] = x_next

        if return_increments:
            dW_B[i, 0], dW_B[i, 1] = dW_B1, dW_B2
            dW_A[i, 0], dW_A[i, 1] = dW_A1, dW_A2
            dW_x[i] = dW_xi

    # Fill final rho value
    rho[-1] = np.tanh(x[-1])

    meta: Dict[str, Any] = {
        "dt_internal": dt,
        "T_sim": float(t[-1]),
        "sigma2_floor": _SIGMA2_FLOOR,
    }
    if return_increments:
        meta.update({"dW_A": dW_A, "dW_B": dW_B, "dW_x": dW_x})

    return EfficientPricePath(t=t, p_star=p_star, sigma2=sigma2, x=x, rho=rho, meta=meta)


def integrated_variance(path: EfficientPricePath) -> np.ndarray:
    """
    Approximate integrated variance IV_y = ∫_0^T σ_y^2(t) dt by Riemann sum
    on the internal grid.

    Returns
    -------
    iv : (2,) array, for assets (1,2)
    """
    dt = float(path.meta.get("dt_internal", path.t[1] - path.t[0]))
    # Left Riemann sum over [t_i, t_{i+1})
    return dt * np.sum(path.sigma2[:-1, :], axis=0)