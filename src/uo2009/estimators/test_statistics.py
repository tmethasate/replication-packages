# src/uo2009/estimators/test_statistics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple ,Iterable, Dict, Any, Union

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

def compute_Z_lk_sync(
    y1_obs: np.ndarray,
    y2_obs: np.ndarray,
    t_obs: np.ndarray,
    lags: Iterable[Union[int, float]],
    *,
    lags_in: str = "seconds",  # "seconds" or "steps"
    rtol_dt: float = 1e-10,
    atol_dt: float = 1e-12,
    return_meta: bool = True,
) -> Dict[Union[int, float], Dict[str, Any]]:
    """
    Compute Ubukata–Oya Z_{ell,k} objects for *synchronized, equidistant* samples.

    Paper mapping (Eq. (4)):
        Z_{ell,ij} = r1_i * r2_j  for pairs (i,j) s.t. t_{i-1} - s_j = ell.
    Under synchronous sampling: t_i = s_i on a common grid with step Δ.

    If ell > 0 and ell = mΔ, then the constraint implies j = i - 1 - m.
    We can re-index k := i, so Z_{ell,k} = r1_k * r2_{k-1-m} for k = m+2,...,n.

    Parameters
    ----------
    y1_obs, y2_obs : (n+1,)
        Observed log-prices (or prices) sampled on the same grid.
    t_obs : (n+1,)
        Common sampling times (must be equally spaced).
    lags : iterable
        Lags either in seconds (lags_in="seconds") or in steps (lags_in="steps").
        For seconds, each lag must be an integer multiple of Δ.
    lags_in : {"seconds","steps"}
        Interpretation of `lags`.
    return_meta : bool
        If True, returns indices and corresponding times as metadata.

    Returns
    -------
    out : dict
        out[ell] is a dict with:
          - "Z" : array of Z_{ell,k} values
          - "k" : k indices (interval indices for asset 1, i.e., i)
          - "i" : same as k (kept for clarity)
          - "j" : matched interval indices for asset 2
          - "ell_steps" : m
          - "ell_seconds" : m*Δ
          - plus some helpful time stamps if return_meta=True

    Notes
    -----
    Indices here follow the paper's interval indexing:
      - prices are at times t_0,...,t_n
      - interval i is (t_{i-1}, t_i], so i runs 1..n
      - returns r_i correspond to interval i
    """
    y1_obs = np.asarray(y1_obs, dtype=float)
    y2_obs = np.asarray(y2_obs, dtype=float)
    t_obs = np.asarray(t_obs, dtype=float)

    if y1_obs.shape != y2_obs.shape or y1_obs.shape != t_obs.shape:
        raise ValueError("y1_obs, y2_obs, and t_obs must have the same shape (n+1,).")
    if y1_obs.ndim != 1:
        raise ValueError("Inputs must be 1D arrays of shape (n+1,).")
    if np.any(np.diff(t_obs) <= 0):
        raise ValueError("t_obs must be strictly increasing.")

    dt = np.diff(t_obs)
    dt0 = dt[0]
    if not np.allclose(dt, dt0, rtol=rtol_dt, atol=atol_dt):
        raise ValueError("t_obs is not equally spaced (required for Section 4.1.1 sync case).")

    n = len(t_obs) - 1  # number of intervals
    # returns indexed by interval i=1..n; we'll store in arrays r[1..n] with dummy 0 at index 0
    r1 = np.empty(n + 1)
    r2 = np.empty(n + 1)
    r1[0] = np.nan
    r2[0] = np.nan
    r1[1:] = y1_obs[1:] - y1_obs[:-1]
    r2[1:] = y2_obs[1:] - y2_obs[:-1]

    out: Dict[Union[int, float], Dict[str, Any]] = {}

    for ell in lags:
        if lags_in == "steps":
            m = int(ell)
            ell_seconds = m * dt0
        elif lags_in == "seconds":
            # convert seconds -> steps
            m_float = ell / dt0
            m = int(np.rint(m_float))
            if not np.isclose(m_float, m, rtol=0, atol=1e-9):
                raise ValueError(f"lag ell={ell} is not an integer multiple of Δ={dt0}.")
            ell_seconds = float(ell)
        else:
            raise ValueError("lags_in must be 'seconds' or 'steps'.")

        if m < 0:
            # Optional: handle negative lags using the paper's 'other case' idea.
            # For sync grid, ell = -mΔ means asset2 interval is after asset1:
            # j = i + (-m) + 1
            mp = -m
            i = np.arange(1, n - (mp + 1) + 1)  # i <= n-mp-1
            j = i + mp + 1
            Z = r1[i] * r2[j]
        else:
            # ell >= 0: j = i - 1 - m, with i from m+2..n
            i = np.arange(m + 2, n + 1)  # interval indices for asset 1
            j = i - 1 - m
            if len(i) == 0:
                Z = np.array([], dtype=float)
            else:
                Z = r1[i] * r2[j]

        res: Dict[str, Any] = {
            "Z": Z,
            "k": i.copy(),          # treat k := i
            "i": i.copy(),
            "j": j.copy(),
            "ell_steps": m,
            "ell_seconds": ell_seconds,
            "dt": dt0,
        }

        if return_meta and len(i) > 0:
            # interval i is (t_{i-1}, t_i], interval j is (t_{j-1}, t_j]
            res.update(
                {
                    "t_i_start": t_obs[i - 1],
                    "t_i_end": t_obs[i],
                    "t_j_start": t_obs[j - 1],
                    "t_j_end": t_obs[j],
                    "gap_seconds": t_obs[i - 1] - t_obs[j],  # should equal ell_seconds for ell>=0
                }
            )

        out[ell] = res

    return out
