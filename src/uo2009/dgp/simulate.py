# src/uo2009/dgp/simulate.py
"""
Simulation glue for Ubukata & Oya (2009), Section 4.1.

This module combines:
1) Efficient (latent) price simulation on an internal grid (Euler–Maruyama),
2) Regular synchronous sampling at a chosen interval Δ (10/15/30 seconds),
3) Bivariate MA(2) microstructure noise calibrated to noise-to-signal ratios,
4) Observed prices: Y = P* + noise.

Section 4.1.1 MA(2) noise (tick time = sampling interval Δ):
    η_t = εη_t - 0.5 εη_{t-1} + 0.4 εη_{t-2} - 0.1 εδ_{t-1} + 0.15 εδ_{t-2}
    δ_t = εδ_t - 0.4 εη_{t-1} + 0.15 εη_{t-2} - 0.1 εδ_{t-1} + 0.3 εδ_{t-2}

Noise-to-signal ratios:
    Var(η) / IV1 = 0.005 , Var(δ) / IV2 = 0.002
where IVy = ∫ σ_y^2(t) dt (approximated on internal grid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .efficient_price import EfficientPricePath, simulate_efficient_price, integrated_variance
from .noise import simulate_bivariate_ma2_with_target_marginals
from .sampling import regular_times


# -----------------------------
# Outputs
# -----------------------------

@dataclass(frozen=True)
class DGPOutput:
    """Standard output for a single simulated day/path."""
    # observation times (seconds)
    t_obs: np.ndarray            # (n_obs,)

    # latent efficient prices at observation times
    p_star_obs: np.ndarray       # (n_obs, 2)

    # observed prices at observation times: y = p_star + noise
    y_obs: np.ndarray            # (n_obs, 2)

    # noise at observation times: (eta, delta)
    noise: np.ndarray            # (n_obs, 2)

    # full latent path on internal grid (useful for debugging/diagnostics)
    latent_path: EfficientPricePath

    # metadata: parameters, calibration, etc.
    meta: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------

def _sample_linear(
    t_internal: np.ndarray,
    x_internal: np.ndarray,
    t_obs: np.ndarray,
) -> np.ndarray:
    """
    Linear interpolation of a (n_internal, d) array onto observation times.

    Parameters
    ----------
    t_internal : (n_internal,)
    x_internal : (n_internal, d)
    t_obs      : (n_obs,)

    Returns
    -------
    x_obs : (n_obs, d)
    """
    if x_internal.ndim != 2:
        raise ValueError("x_internal must be 2D (n_internal, d).")
    d = x_internal.shape[1]
    out = np.empty((len(t_obs), d), dtype=float)
    for j in range(d):
        out[:, j] = np.interp(t_obs, t_internal, x_internal[:, j])
    return out


def _uo2009_ma2_B_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the (B0, B1, B2) matrices for the Section 4.1.1 bivariate MA(2):

        u_t = B0 e_t + B1 e_{t-1} + B2 e_{t-2},
        u_t = (eta_t, delta_t)', e_t = (eps_eta_t, eps_delta_t)'.

    See module docstring for the component equations.
    """
    B0 = np.array([[1.0, 0.0],
                   [0.0, 1.0]], dtype=float)

    B1 = np.array([[-0.5, -0.1],
                   [-0.4, -0.1]], dtype=float)

    B2 = np.array([[0.4, 0.15],
                   [0.15, 0.3]], dtype=float)
    return B0, B1, B2


# -----------------------------
# Main simulator: Section 4.1.1
# -----------------------------

def simulate_section4_1_1(
    eff_params,
    noise_params,
    *,
    delta_seconds: float,
    rng: np.random.Generator,
    p0: Tuple[float, float] = (0.0, 0.0),
    sigma2_0: Optional[Tuple[float, float]] = None,
    x0: Optional[float] = None,
) -> DGPOutput:
    """
    Simulate one path/day under Section 4.1.1:
    - Efficient price with SV + stochastic correlation
    - Regular synchronous sampling at interval Δ
    - Bivariate MA(2) noise calibrated to NSR targets
    - Observed prices = latent + noise

    Parameters
    ----------
    eff_params:
        Efficient price parameters (see dgp/params.py; passed to simulate_efficient_price)
    noise_params:
        Noise parameters; expected to have attributes nsr1, nsr2
    delta_seconds:
        Sampling interval Δ in seconds (e.g., 10, 15, 30)
    rng:
        numpy random generator
    p0, sigma2_0, x0:
        optional initial conditions

    Returns
    -------
    DGPOutput
    """
    delta_seconds = float(delta_seconds)
    if delta_seconds <= 0:
        raise ValueError("delta_seconds must be positive.")

    # 1) Simulate latent efficient prices on internal grid
    latent = simulate_efficient_price(
        eff_params,
        rng,
        p0=p0,
        sigma2_0=sigma2_0,
        x0=x0,
        return_increments=False,
    )

    # 2) Build regular synchronous observation times
    T = float(eff_params.T)
    t_obs = regular_times(T=T, delta=delta_seconds)

    # 3) Sample latent prices onto observation grid
    p_star_obs = _sample_linear(latent.t, latent.p_star, t_obs)

    # 4) Integrated variance from internal grid (used for noise calibration)
    IV = integrated_variance(latent)  # (2,)
    target_var_eta = float(noise_params.nsr1) * float(IV[0])
    target_var_delta = float(noise_params.nsr2) * float(IV[1])

    # 5) Simulate MA(2) noise at observation times, calibrated to target marginal variances
    B0, B1, B2 = _uo2009_ma2_B_matrices()
    n_obs = len(t_obs)
    noise, Sigma_e = simulate_bivariate_ma2_with_target_marginals(
        B0=B0,
        B1=B1,
        B2=B2,
        n_obs=n_obs,
        target_var=(target_var_eta, target_var_delta),
        rng=rng,
        burnin=None,  # default burn-in inside noise.py
    )

    # 6) Observed prices
    y_obs = p_star_obs + noise

    meta: Dict[str, Any] = {
        "delta_seconds": delta_seconds,
        "n_obs": n_obs,
        "IV": IV,
        "nsr": (float(noise_params.nsr1), float(noise_params.nsr2)),
        "target_noise_var": (target_var_eta, target_var_delta),
        "Sigma_e": Sigma_e,
        "B0": B0,
        "B1": B1,
        "B2": B2,
    }

    return DGPOutput(
        t_obs=t_obs,
        p_star_obs=p_star_obs,
        y_obs=y_obs,
        noise=noise,
        latent_path=latent,
        meta=meta,
    )


def simulate_section4_1_1_grid(
    eff_params,
    noise_params,
    *,
    deltas_seconds: Tuple[float, ...] = (10.0, 15.0, 30.0),
    rng: np.random.Generator,
    p0: Tuple[float, float] = (0.0, 0.0),
    sigma2_0: Optional[Tuple[float, float]] = None,
    x0: Optional[float] = None,
) -> Dict[float, DGPOutput]:
    """
    Convenience: simulate multiple sampling intervals (10/15/30 seconds) using the
    same RNG stream (one after another).

    Returns
    -------
    dict mapping delta_seconds -> DGPOutput
    """
    out: Dict[float, DGPOutput] = {}
    for d in deltas_seconds:
        out[float(d)] = simulate_section4_1_1(
            eff_params,
            noise_params,
            delta_seconds=float(d),
            rng=rng,
            p0=p0,
            sigma2_0=sigma2_0,
            x0=x0,
        )
    return out
