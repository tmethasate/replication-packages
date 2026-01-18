# src/uo2009/estimators/test_statistics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SubsamplingVarianceResult:
    """
    Result for the subsampling variance estimator (Eq. 7 / Eq. 17 in UO2009).
    """
    sigma2_hat: float
    sigma_hat: float
    M: int
    K: int
    N: int
    used_N: int  # K*M (truncation used for blocks)


@dataclass(frozen=True)
class TauTestResult:
    """
    Result for tau-statistics (Eq. 9 / Eq. 16 in UO2009).
    """
    tau: float
    zbar: float
    N: int
    var: SubsamplingVarianceResult


def _validate_1d_array(z: np.ndarray, name: str = "z") -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape(-1)
    if z.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(z)):
        raise ValueError(f"{name} must be finite (no NaN/inf).")
    return z


def subsampling_variance(
    z: np.ndarray,
    M: int,
) -> SubsamplingVarianceResult:
    """
    Subsampling variance estimator in UO (2009), Eq. (7) / Eq. (17):

        K = floor(N/M)
        Z^h := (z_{hM+1}, ..., z_{(h+1)M}),  h=0,...,K-1
        zbar_h = mean(Z^h)
        zbar_blocks = mean_h zbar_h

        sigma2_hat = (M / K) * sum_{h=0}^{K-1} (zbar_h - zbar_blocks)^2

    Notes
    -----
    - Uses *non-overlapping* blocks.
    - Drops the last (N - K*M) observations, as in K = floor(N/M).
    - Requires K >= 2 to form a variance.
    """
    z = _validate_1d_array(z, "z")
    N = int(z.size)

    M = int(M)
    if M <= 0:
        raise ValueError("M must be a positive integer.")
    if M > N:
        raise ValueError(f"M must be <= N. Got M={M}, N={N}.")

    K = N // M
    if K < 2:
        raise ValueError(
            f"Need at least K>=2 blocks for subsampling variance. "
            f"Got N={N}, M={M} => K={K}."
        )

    used_N = K * M
    z_used = z[:used_N]
    blocks = z_used.reshape(K, M)
    block_means = blocks.mean(axis=1)
    mean_of_block_means = block_means.mean()

    sigma2_hat = float((M / K) * np.sum((block_means - mean_of_block_means) ** 2))
    sigma2_hat = max(sigma2_hat, 0.0)  # numerical safety
    sigma_hat = float(np.sqrt(sigma2_hat))

    return SubsamplingVarianceResult(
        sigma2_hat=sigma2_hat,
        sigma_hat=sigma_hat,
        M=M,
        K=K,
        N=N,
        used_N=used_N,
    )


def tau_from_Z(
    Z_ell: np.ndarray,
    *,
    M: int,
) -> TauTestResult:
    """
    Generic tau-statistic based on a sequence {Z_{ell,k}}:

        tau(ell) = sqrt(N_ell) * mean(Z_ell) / sigma_hat_{ell,f}

    This covers UO (2009) Eq. (9) directly. Eq. (16) is the same form, with
    Z_ell built from asset-1 return products.

    Parameters
    ----------
    Z_ell : array-like
        Sequence (Z_{ell,1},...,Z_{ell,N_ell})
    M : int
        Block length M_ell for the subsampling variance estimator.

    Returns
    -------
    TauTestResult
    """
    Z_ell = _validate_1d_array(Z_ell, "Z_ell")
    N = int(Z_ell.size)
    zbar = float(Z_ell.mean())

    var_res = subsampling_variance(Z_ell, M=M)

    if var_res.sigma_hat == 0.0:
        # If variance estimate is 0, tau is undefined/infinite depending on zbar
        tau = np.inf if zbar != 0.0 else 0.0
    else:
        tau = float(np.sqrt(N) * zbar / var_res.sigma_hat)

    return TauTestResult(tau=tau, zbar=zbar, N=N, var=var_res)


def build_Z_from_returns_regular(
    r: np.ndarray,
    lag_steps: int,
) -> np.ndarray:
    """
    Build the UO-style product sequence Z_{ell,k} from *regularly sampled* returns.

    UO defines products r_i r_j for pairs (i,j) such that:
        t_{j-1} - t_i = ell >= 0

    On a regular grid with constant interval Δ, this condition implies:
        j = i + (ell/Δ) + 1

    So if lag_steps = ell/Δ (an integer >= 0), then:
        Z_{ell,k} = r_i * r_{i+lag_steps+1}

    Parameters
    ----------
    r : (n,) returns
    lag_steps : int
        lag_steps = ell/Δ, must be >= 0

    Returns
    -------
    Z_ell : (n - (lag_steps+1),) array
    """
    r = _validate_1d_array(r, "r")
    lag_steps = int(lag_steps)
    if lag_steps < 0:
        raise ValueError("lag_steps must be >= 0.")
    shift = lag_steps + 1
    if r.size <= shift:
        raise ValueError(
            f"Not enough returns to form Z at lag_steps={lag_steps}. "
            f"Need len(r) > {shift}."
        )
    return r[:-shift] * r[shift:]


def tau_eq9_from_returns_regular(
    r: np.ndarray,
    *,
    lag_steps: int,
    M: int,
) -> TauTestResult:
    """
    Convenience wrapper for Eq. (9) when data are regularly sampled:
    - Build Z_{ell,k} from returns r
    - Compute tau via subsampling variance estimator

    Returns TauTestResult.
    """
    Z_ell = build_Z_from_returns_regular(r, lag_steps=lag_steps)
    return tau_from_Z(Z_ell, M=M)


def tau_eq16_asset1_from_returns_regular(
    r1: np.ndarray,
    *,
    lag_steps: int,
    M: int,
) -> TauTestResult:
    """
    Convenience wrapper for Eq. (16) (univariate dependence test for asset 1 noise),
    under regular sampling.

    In implementation, Eq. (16) is identical to Eq. (9) but uses Z_{1,ell,k} built
    from asset-1 returns r1.
    """
    Z1_ell = build_Z_from_returns_regular(r1, lag_steps=lag_steps)
    return tau_from_Z(Z1_ell, M=M)
