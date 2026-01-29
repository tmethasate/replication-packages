"""
statistics_cross_noise_dep.py

Cross-sectional noise dependence statistics for Ubukata & Oya (2009),
especially the threshold test statistic tau(ell) in Eq. (9).

Core objects:
- Build the Z_{ell,k} sequence from two-asset asynchronous observations
- Compute subsampling variance estimator (Eq. (7))
- Compute tau(ell) (Eq. (9)) :contentReference[oaicite:4]{index=4}
- (Optional) Sequential threshold selection procedure described after Theorem 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np


# ============================================================
# Public result containers
# ============================================================

@dataclass(frozen=True)
class TauResult:
    ell: Union[int, float]
    N_ell: int
    Zbar: float
    sigma2_hat: float
    sigma_hat: float
    tau: float
    # optional diagnostics
    M: int
    K: int


@dataclass(frozen=True)
class ThresholdResult:
    ell_star: Union[int, float]
    cv: float
    direction: Literal["descending_from_L"]
    tested: Tuple[TauResult, ...]  # keep full path for debugging


# ============================================================
# Input validation + small utilities
# ============================================================

def _as_float_1d(x: np.ndarray, name: str) -> np.ndarray:
    """Ensure 1D float array."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={x.shape}.")
    return x


def _check_sorted_increasing(t: np.ndarray, name: str) -> None:
    if t.size >= 2 and np.any(np.diff(t) < 0):
        raise ValueError(f"{name} must be sorted increasing.")


def _default_M_from_c(N: int, c: float) -> int:
    """
    Rule-of-thumb selection mentioned after Theorem 1:
    M_ell ≈ c * N_ell^(1/3)
    """
    if N <= 0:
        return 0
    return int(max(1, np.floor(c * (N ** (1.0 / 3.0)))))


# ============================================================
# Constructing Z_{ell,k}
# ============================================================
def build_Z_sequence(
    t1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    y2: np.ndarray,
    ell: Union[int, float],
    *,
    ell_in: Literal["seconds", "ticks"] = "ticks",
    rtol_dt: float = 1e-10,
    atol_dt: float = 1e-12,
) -> np.ndarray:
    """
    Construct the Ubukata–Oya Z_{ell,k} sequence (threshold-test object) based on Eq. (4).

    We form products of returns on *nonoverlapping* adjacent intervals whose distance is ell.
    Then we order the selected products by increasing asset-1 index i (i.e., increasing interval
    number for asset 1), yielding the 1D sequence {Z_{ell,k}}_{k=1}^{N_ell}.

    Parameters
    ----------
    t1, y1 : arrays
        Observation times and log prices for asset 1, length n1.
    t2, y2 : arrays
        Observation times and log prices for asset 2, length n2.
    ell : int or float
        Lag/distance. If ell_in="ticks", ell must be integer-valued (or very close).
    ell_in : {"ticks","seconds"}
        Units for ell and for time matching.
        - "ticks": exact integer matching (recommended).
        - "seconds": uses isclose matching with (rtol_dt, atol_dt).
    rtol_dt, atol_dt : float
        Tolerances used only when ell_in="seconds".

    Returns
    -------
    Z : np.ndarray
        1D array of selected products, ordered by asset-1 interval index i.
        Length is N_ell (can be 0).

    Notes
    -----
    Let r1_i = y1[i] - y1[i-1] (i=1..n1-1), with start time t1[i-1] and end time t1[i].
    Let r2_j = y2[j] - y2[j-1] (j=1..n2-1), with start time t2[j-1] and end time t2[j].

    Pairing conditions:
      - ell > 0: choose (i,j) s.t. t1[i-1] - t2[j] == ell
      - ell < 0: choose (i,j) s.t. t2[j-1] - t1[i] == -ell
      - ell = 0: adjacent nonoverlap: t1[i-1] == t2[j] OR t2[j-1] == t1[i]
    """
    t1 = _as_float_1d(t1, "t1"); y1 = _as_float_1d(y1, "y1")
    t2 = _as_float_1d(t2, "t2"); y2 = _as_float_1d(y2, "y2")
    _check_sorted_increasing(t1, "t1"); _check_sorted_increasing(t2, "t2")
    if t1.size != y1.size or t2.size != y2.size:
        raise ValueError("Time/price arrays must have matching lengths per asset.")
    if t1.size < 2 or t2.size < 2:
        return np.array([], dtype=float)

    # Returns aligned with intervals:
    # asset 1: i=1..n1-1 corresponds to k=0..n1-2 in arrays below
    r1 = np.diff(y1)
    r1_start = t1[:-1]   # t_{i-1}
    r1_end   = t1[1:]    # t_i

    # asset 2: j=1..n2-1 corresponds to m=0..n2-2 in arrays below
    r2 = np.diff(y2)
    r2_start = t2[:-1]   # s_{j-1}
    r2_end   = t2[1:]    # s_j

    # Interpret ell depending on unit
    if ell_in == "ticks":
        # Require integer-valued ell (or extremely close)
        ell_int = int(np.round(float(ell)))
        if not np.isclose(float(ell), float(ell_int), rtol=0.0, atol=1e-12):
            raise ValueError(f"ell_in='ticks' requires integer ell; got ell={ell}.")
        ell_val = ell_int
    elif ell_in == "seconds":
        ell_val = float(ell)
    else:
        raise ValueError("ell_in must be one of {'ticks','seconds'}.")

    def _index_map(vals: np.ndarray):
        """
        Map a time value to one or more indices.
        For strictly increasing times, each time maps to at most one index.
        """
        mp = {}
        for idx, v in enumerate(vals):
            mp.setdefault(v, []).append(idx)
        return mp

    # For seconds mode, exact dict keys on floats are risky; use searchsorted+isclose
    def _find_indices_sorted(sorted_vals: np.ndarray, target: float):
        """
        Return list of indices in sorted_vals approximately equal to target.
        (Usually 0 or 1 in practice with strictly increasing times.)
        """
        k = int(np.searchsorted(sorted_vals, target))
        out = []
        # check neighbor positions k-1, k
        for cand in (k - 1, k):
            if 0 <= cand < sorted_vals.size and np.isclose(
                sorted_vals[cand], target, rtol=rtol_dt, atol=atol_dt
            ):
                out.append(cand)
        return out

    Z_list: list[float] = []

    if ell_in == "ticks":
        # Use fast dict lookup with exact matching
        map_end = _index_map(r2_end)     # s_j -> m
        map_start = _index_map(r2_start) # s_{j-1} -> m

        if ell_val > 0:
            # t_{i-1} - s_j = ell  =>  s_j = t_{i-1} - ell
            for k in range(r1.size):  # k corresponds to i-1
                target = r1_start[k] - ell_val
                js = map_end.get(target)
                if js:
                    # typically single match
                    for m in js:
                        Z_list.append(float(r1[k] * r2[m]))

        elif ell_val < 0:
            # s_{j-1} - t_i = -ell  =>  s_{j-1} = t_i - ell
            # (since ell is negative, "- ell" adds a positive amount)
            for k in range(r1.size):
                target = r1_end[k] - ell_val
                js = map_start.get(target)
                if js:
                    for m in js:
                        Z_list.append(float(r1[k] * r2[m]))

        else:  # ell == 0
            # adjacent nonoverlap:
            # (A) t_{i-1} == s_j  -> match r1_start with r2_end
            # (B) t_i == s_{j-1}  -> match r1_end with r2_start
            for k in range(r1.size):
                jsA = map_end.get(r1_start[k])
                if jsA:
                    for m in jsA:
                        Z_list.append(float(r1[k] * r2[m]))
                jsB = map_start.get(r1_end[k])
                if jsB:
                    for m in jsB:
                        Z_list.append(float(r1[k] * r2[m]))

    else:
        # seconds: use sorted arrays + searchsorted/isclose
        if ell_val > 0:
            for k in range(r1.size):
                target = float(r1_start[k] - ell_val)
                ms = _find_indices_sorted(r2_end, target)
                for m in ms:
                    Z_list.append(float(r1[k] * r2[m]))

        elif ell_val < 0:
            for k in range(r1.size):
                target = float(r1_end[k] - ell_val)
                ms = _find_indices_sorted(r2_start, target)
                for m in ms:
                    Z_list.append(float(r1[k] * r2[m]))

        else:
            for k in range(r1.size):
                msA = _find_indices_sorted(r2_end, float(r1_start[k]))
                for m in msA:
                    Z_list.append(float(r1[k] * r2[m]))
                msB = _find_indices_sorted(r2_start, float(r1_end[k]))
                for m in msB:
                    Z_list.append(float(r1[k] * r2[m]))

    return np.asarray(Z_list, dtype=float)



# ============================================================
# Subsampling variance estimator: Eq. (7)
# ============================================================

def subsampling_variance_nonoverlapping(
    Z: np.ndarray,
    *,
    M: Optional[int] = None,
    c: Optional[float] = None,
) -> Tuple[float, int, int]:
    """
    Compute sigma_hat^2_{ell,f} via nonoverlapping subsampling blocks (Eq. (7)).

    Given Z of length N:
      - choose M (block length), K = floor(N / M)
      - block means: Zbar_h over h=0,...,K-1
      - sigma2_hat = (M / K) * sum_h (Zbar_h - mean(Zbar_h))^2   (Eq. (7))

    Returns
    -------
    sigma2_hat, M_used, K_used
    """
    Z = _as_float_1d(Z, "Z")
    N = int(Z.size)
    if N == 0:
        return np.nan, 0, 0

    if M is None:
        if c is None:
            raise ValueError("Provide either M or c (for M ≈ c*N^(1/3)).")
        M = _default_M_from_c(N, float(c))

    M = int(M)
    if M <= 0:
        raise ValueError("M must be positive.")
    K = N // M
    if K <= 1:
        # Too few blocks -> variance estimator unstable / undefined in practice
        return np.nan, M, K

    Z_trim = Z[: K * M]
    blocks = Z_trim.reshape(K, M)
    block_means = blocks.mean(axis=1)
    grand_mean = block_means.mean()

    sigma2_hat = (M / K) * np.sum((block_means - grand_mean) ** 2)  # Eq. (7)
    return float(sigma2_hat), M, K


# ============================================================
# 5) tau(ell): Eq. (9)
# ============================================================

def compute_tau_from_Z(
    ell: Union[int, float],
    Z: np.ndarray,
    *,
    M: Optional[int] = None,
    c: Optional[float] = None,
) -> TauResult:
    """
    Compute tau(ell) := sqrt(N_ell) * Zbar / sigma_hat  (Eq. (9)). 
    sigma_hat is sqrt of subsampling variance estimator (Eq. (7)). 
    """
    Z = _as_float_1d(Z, "Z")
    N_ell = int(Z.size)
    Zbar = float(Z.mean()) if N_ell > 0 else np.nan

    sigma2_hat, M_used, K_used = subsampling_variance_nonoverlapping(Z, M=M, c=c)
    sigma_hat = float(np.sqrt(sigma2_hat)) if np.isfinite(sigma2_hat) and sigma2_hat >= 0 else np.nan

    tau = float(np.sqrt(N_ell) * Zbar / sigma_hat) if (N_ell > 0 and np.isfinite(sigma_hat) and sigma_hat > 0) else np.nan

    return TauResult(
        ell=ell,
        N_ell=N_ell,
        Zbar=Zbar,
        sigma2_hat=float(sigma2_hat),
        sigma_hat=sigma_hat,
        tau=tau,
        M=M_used,
        K=K_used,
    )


def compute_tau(
    t1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    y2: np.ndarray,
    ell: Union[int, float],
    *,
    ell_in: Literal["seconds", "ticks"] = "seconds",
    M: Optional[int] = None,
    c: Optional[float] = None,
) -> TauResult:
    """
    Full pipeline: build Z_{ell,k} then compute tau(ell).
    """
    Z = build_Z_sequence(t1, y1, t2, y2, ell, ell_in=ell_in)
    return compute_tau_from_Z(ell, Z, M=M, c=c)


def compute_tau_grid(
    t1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    y2: np.ndarray,
    ells: Iterable[Union[int, float]],
    *,
    ell_in: Literal["seconds", "ticks"] = "seconds",
    M: Optional[int] = None,
    c: Optional[float] = None,
) -> Dict[Union[int, float], TauResult]:
    """Convenience loop over multiple lags."""
    out: Dict[Union[int, float], TauResult] = {}
    for ell in ells:
        out[ell] = compute_tau(t1, y1, t2, y2, ell, ell_in=ell_in, M=M, c=c)
    return out


# ============================================================
# sequential threshold selection (Theorem 1 discussion)
# ============================================================

def select_threshold_via_tau(
    t1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    y2: np.ndarray,
    ells_descending: Iterable[Union[int, float]],
    *,
    ell_in: Literal["seconds", "ticks"] = "seconds",
    cv: float = 1.96,
    M: Optional[int] = None,
    c: Optional[float] = None,
) -> ThresholdResult:
    """
    Implements the sequential testing logic described after Theorem 1:
    start from a large L where dependence should be zero, then move inward
    until |tau(ell)| exceeds the critical value; the first rejection determines ell*.
    """
    tested = []
    ell_star = None

    for ell in ells_descending:
        res = compute_tau(t1, y1, t2, y2, ell, ell_in=ell_in, M=M, c=c)
        tested.append(res)
        if np.isfinite(res.tau) and abs(res.tau) > cv:
            ell_star = ell
            break

    if ell_star is None:
        # If nothing rejected, you can choose to return the smallest tested ell,
        # or np.nan / None. Here: return last tested ell if exists.
        ell_star = tested[-1].ell if tested else np.nan

    return ThresholdResult(
        ell_star=ell_star,
        cv=cv,
        direction="descending_from_L",
        tested=tuple(tested),
    )