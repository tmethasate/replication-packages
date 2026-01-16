# src/uo2009/dgp/noise.py
"""
Microstructure noise generators for Ubukata & Oya (2009), Section 4.1.

In Section 4.1.1 the paper uses *bivariate MA(2)* noise at observation times.
Rather than hard-coding the exact MA(2) coefficients (easy to mistype),
this module implements a general bivariate MA(q):

    u_t = sum_{ell=0}^q B_ell e_{t-ell},     e_t i.i.d. (0, Sigma_e)

where u_t = (eta_t, delta_t)' is the bivariate noise.

Key utilities provided:
- simulate_bivariate_ma(...) : simulate u_t given {B_ell} and Sigma_e
- calibrate_diag_innov_vars_for_target_marginals(...) :
    choose diagonal Sigma_e = diag(s1^2, s2^2) to match target marginal variances
- theoretical_covariances_ma(...) : compute Gamma(h) = Cov(u_t, u_{t+h})

This design lets you plug in the exact MA(2) coefficient matrices from the paper
with no further refactoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MANoiseSpec:
    """Specification for bivariate MA(q) noise."""
    B: List[np.ndarray]         # list of (2,2) matrices: B[0],...,B[q]
    Sigma_e: np.ndarray         # (2,2) innovation covariance


def _validate_B(B: Sequence[np.ndarray]) -> List[np.ndarray]:
    if len(B) == 0:
        raise ValueError("B must contain at least B0.")
    out: List[np.ndarray] = []
    for i, M in enumerate(B):
        M = np.asarray(M, dtype=float)
        if M.shape != (2, 2):
            raise ValueError(f"B[{i}] must be shape (2,2); got {M.shape}.")
        out.append(M)
    return out


def _validate_Sigma_e(Sigma_e: np.ndarray) -> np.ndarray:
    S = np.asarray(Sigma_e, dtype=float)
    if S.shape != (2, 2):
        raise ValueError(f"Sigma_e must be shape (2,2); got {S.shape}.")
    # Symmetrize tiny numerical asymmetry
    S = 0.5 * (S + S.T)
    # Basic PSD check (allow tiny negative eigenvalues from rounding)
    eig = np.linalg.eigvalsh(S)
    if np.min(eig) < -1e-12:
        raise ValueError(f"Sigma_e must be PSD. Min eigenvalue: {np.min(eig)}")
    return S


def calibrate_diag_innov_vars_for_target_marginals(
    B: Sequence[np.ndarray],
    target_var: Tuple[float, float],
    *,
    eps: float = 1e-14,
) -> np.ndarray:
    """
    Choose diagonal innovation covariance Sigma_e = diag(s1^2, s2^2) so that
    the *marginal* variances of u_t match target_var under the assumption
    e_t = (e1_t, e2_t)' has independent components.

    With Sigma_e diagonal:
        Var(u_i) = sum_ell (B_ell[i,0]^2) * s1^2 + sum_ell (B_ell[i,1]^2) * s2^2

    So we solve A @ [s1^2, s2^2]' = target_var, where:
        A[i,j] = sum_ell B_ell[i,j]^2

    Parameters
    ----------
    B:
        sequence of (2,2) MA coefficient matrices
    target_var:
        desired (Var(eta), Var(delta))
    eps:
        floor to avoid negative due to numerical issues

    Returns
    -------
    Sigma_e : (2,2) diagonal matrix
    """
    B_list = _validate_B(B)
    tv = np.array(target_var, dtype=float)
    if np.any(tv < 0):
        raise ValueError("target_var must be nonnegative.")

    A = np.zeros((2, 2), dtype=float)
    for M in B_list:
        A += M**2  # elementwise square then sum across lags

    # Solve for innovation variances
    try:
        s2 = np.linalg.solve(A, tv)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Calibration matrix A is singular; check your MA coefficient matrices."
        ) from e

    # Numerical safety: truncate tiny negatives
    s2 = np.maximum(s2, eps)

    return np.diag(s2)


def theoretical_covariances_ma(
    B: Sequence[np.ndarray],
    Sigma_e: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """
    Compute theoretical autocovariances Gamma(h) = Cov(u_t, u_{t+h})
    for h = 0,...,max_lag for bivariate MA(q).

    For MA(q), for h >= 0:
        Gamma(h) = sum_{k=0}^{q-h} B_{k+h} Sigma_e B_k'
    and Gamma(-h) = Gamma(h)'.

    Returns
    -------
    Gamma : (max_lag+1, 2, 2) array with Gamma[h] = Cov(u_t, u_{t+h})
    """
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0.")
    B_list = _validate_B(B)
    S = _validate_Sigma_e(Sigma_e)

    q = len(B_list) - 1
    Gamma = np.zeros((max_lag + 1, 2, 2), dtype=float)

    for h in range(max_lag + 1):
        if h > q:
            Gamma[h] = 0.0
            continue
        acc = np.zeros((2, 2), dtype=float)
        for k in range(0, q - h + 1):
            acc += B_list[k + h] @ S @ B_list[k].T
        Gamma[h] = acc

    return Gamma


def simulate_bivariate_ma(
    B: Sequence[np.ndarray],
    Sigma_e: np.ndarray,
    n_obs: int,
    rng: np.random.Generator,
    *,
    burnin: Optional[int] = None,
    return_innovations: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Simulate u_t = sum_{ell=0}^q B_ell e_{t-ell} for t=0,...,n_obs-1.

    Parameters
    ----------
    B:
        sequence of (2,2) matrices [B0,...,Bq]
    Sigma_e:
        innovation covariance (2,2)
    n_obs:
        number of noise observations to return
    rng:
        numpy random generator
    burnin:
        number of initial innovations to discard (default: 5*q + 50)
    return_innovations:
        if True, also return the innovations used (after burnin) as (n_obs,2)

    Returns
    -------
    u : (n_obs,2) array
    (u, e) if return_innovations True
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")

    B_list = _validate_B(B)
    S = _validate_Sigma_e(Sigma_e)
    q = len(B_list) - 1

    if burnin is None:
        burnin = 5 * q + 50

    # draw innovations for t = -q-burnin,...,n_obs-1
    n_total = n_obs + burnin + q
    # Cholesky factor (works for PSD; add jitter if needed)
    try:
        L = np.linalg.cholesky(S + 1e-16 * np.eye(2))
    except np.linalg.LinAlgError as e:
        raise ValueError("Sigma_e not numerically PSD for Cholesky.") from e

    z = rng.standard_normal((n_total, 2))
    e_all = z @ L.T  # (n_total,2)

    # compute u for indices corresponding to output region
    u_all = np.zeros((n_total, 2), dtype=float)
    for t in range(n_total):
        acc = np.zeros(2, dtype=float)
        for ell in range(q + 1):
            idx = t - ell
            if idx < 0:
                break
            acc += B_list[ell] @ e_all[idx]
        u_all[t] = acc

    # discard burnin + initial q padding
    start = burnin + q
    u = u_all[start : start + n_obs]
    e = e_all[start : start + n_obs]

    if return_innovations:
        return u, e
    return u


# ---------------------------
# Convenience wrapper: MA(2)
# ---------------------------

def simulate_bivariate_ma2_with_target_marginals(
    B0: np.ndarray,
    B1: np.ndarray,
    B2: np.ndarray,
    n_obs: int,
    target_var: Tuple[float, float],
    rng: np.random.Generator,
    *,
    burnin: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper for MA(2) with diagonal innovation covariance calibrated
    to match target marginal variances of (eta, delta).

    Returns
    -------
    u : (n_obs,2) noise
    Sigma_e : (2,2) diagonal innovations covariance used
    """
    B = [B0, B1, B2]
    Sigma_e = calibrate_diag_innov_vars_for_target_marginals(B, target_var)
    u = simulate_bivariate_ma(B, Sigma_e, n_obs=n_obs, rng=rng, burnin=burnin)
    return u, Sigma_e
