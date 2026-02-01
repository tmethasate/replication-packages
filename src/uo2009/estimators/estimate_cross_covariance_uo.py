import numpy as np
from typing import Literal, Optional, Tuple, Union
from uo2009.estimators.statistics_cross_noise_dep import _as_float_1d, _check_sorted_increasing

__all__ = [
    "build_Z_pm_selected",
    "estimate_cross_covariance_uo_eq11",
]

# ============================================================
# Small helpers
# ============================================================

def _interpret_ell(ell: Union[int, float], *, ell_in: Literal["ticks", "seconds"]) -> Union[int, float]:
    if ell_in == "ticks":
        ell_int = int(np.round(float(ell)))
        if not np.isclose(float(ell), float(ell_int), rtol=0.0, atol=1e-12):
            raise ValueError(f"ell_in='ticks' requires integer ell; got ell={ell}.")
        return ell_int
    if ell_in == "seconds":
        return float(ell)
    raise ValueError("ell_in must be one of {'ticks','seconds'}.")


def _build_time_to_interval_index_map(times: np.ndarray) -> dict:
    """
    Map a time value to a list of interval indices.
    Used for exact matching in ticks mode.
    """
    mp: dict = {}
    for idx, v in enumerate(times):
        mp.setdefault(v, []).append(idx)
    return mp


# ============================================================
# Build selected Z^{(±)}_{ell,k} sequence (Eq. 10)
# ============================================================

def build_Z_pm_selected(
    t1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    y2: np.ndarray,
    ell: Union[int, float],
    *,
    m_plus: int,
    m_minus: int,
    ell_in: Literal["ticks", "seconds"] = "ticks",
    eps: float = 1e-12,
    return_pairs: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """
    Build the selected 1D sequence Z^{(±)}_{ell,k} (Eq. 10), already ranked by i.

    We loop i (asset-1 interval index) in increasing order, and append any matching
    (i,j) contributions, so the resulting Z array is ordered by i automatically.

    Currently implemented for ell_in='ticks' (exact matching).
    """
    t1 = _as_float_1d(t1, "t1"); y1 = _as_float_1d(y1, "y1")
    t2 = _as_float_1d(t2, "t2"); y2 = _as_float_1d(y2, "y2")
    _check_sorted_increasing(t1, "t1"); _check_sorted_increasing(t2, "t2")

    if t1.size != y1.size or t2.size != y2.size:
        raise ValueError("Time/price arrays must have matching lengths per asset.")
    if t1.size < 2 or t2.size < 2:
        Z = np.array([], dtype=float)
        return (Z, []) if return_pairs else Z
    if m_plus < 0 or m_minus < 0:
        raise ValueError("m_plus and m_minus must be nonnegative integers.")

    ell_val = _interpret_ell(ell, ell_in=ell_in)

    if ell_in != "ticks":
        raise NotImplementedError("build_Z_pm_selected currently supports ell_in='ticks' only.")

    # Interval endpoints and one-step returns
    # asset 1: interval i corresponds to k = i-1 in these arrays
    t1_start = t1[:-1]   # t_{i-1}
    t1_end   = t1[1:]    # t_i
    # asset 2: interval j corresponds to m = j-1
    t2_start = t2[:-1]   # s_{j-1}
    t2_end   = t2[1:]    # s_j

    # Lookup maps for the indicator equalities
    map_t2_end = _build_time_to_interval_index_map(t2_end)       # s_j -> m
    map_t2_start = _build_time_to_interval_index_map(t2_start)   # s_{j-1} -> m

    Z_list: list[float] = []
    pairs: list[tuple] = []  # (i, j, term-tag)

    # ------------------------------------------------------------
    # Case ell > 0: r1_i^(+) r2_j^(+) 1{ t_{i-1} - s_j = ell }
    # ------------------------------------------------------------
    if ell_val > 0:
        for k in range(t1_start.size):
            sj_target = t1_start[k] - ell_val
            for m in map_t2_end.get(sj_target, []):
                # --- r1_plus: P1(t_i^(+)) - P1(t_{i-1})
                s_j = t2_end[m]
                t_i = t1_end[k]

                lower_time = (s_j + m_plus) + eps
                idx_t1_plus = int(np.searchsorted(t1, lower_time, side="right"))
                idx_after_ti = int(np.searchsorted(t1, t_i + eps, side="right"))
                idx_t1_plus = max(idx_t1_plus, idx_after_ti)
                if idx_t1_plus >= t1.size:
                    continue
                r1_plus = y1[idx_t1_plus] - y1[k]  # y1 at t_{i-1} is y1[k]

                # --- r2_plus: P2(s_j) - P2(s_{j-1}^(+))
                s_jm1 = t2_start[m]
                t_im1 = t1_start[k]

                upper_time = min(s_jm1, (t_im1 - m_plus) - eps)
                idx_s_jm1_plus = int(np.searchsorted(t2, upper_time, side="right") - 1)
                if idx_s_jm1_plus < 0:
                    continue
                r2_plus = y2[m + 1] - y2[idx_s_jm1_plus]  # y2 at s_j is y2[m+1]

                Z_list.append(float(r1_plus * r2_plus))
                if return_pairs:
                    pairs.append((k + 1, m + 1, "ell>0:+"))

    # ------------------------------------------------------------
    # Case ell < 0: r1_i^(-) r2_j^(-) 1{ s_{j-1} - t_i = -ell }
    # ------------------------------------------------------------
    elif ell_val < 0:
        for k in range(t1_start.size):
            sjm1_target = t1_end[k] - ell_val  # since ell<0, subtracting adds
            for m in map_t2_start.get(sjm1_target, []):
                s_jm1 = t2_start[m]
                t_im1 = t1_start[k]

                # --- r1_minus: P1(t_i) - P1(t_{i-1}^(-))
                upper_t1 = min(t_im1, (s_jm1 - m_minus) - eps)
                idx_t_im1_minus = int(np.searchsorted(t1, upper_t1, side="right") - 1)
                if idx_t_im1_minus < 0:
                    continue
                r1_minus = y1[k + 1] - y1[idx_t_im1_minus]

                # --- r2_minus: P2(s_j^(-)) - P2(s_{j-1})
                t_i = t1_end[k]
                s_j = t2_end[m]

                lower_time = (t_i + m_minus) + eps
                idx_after_sj = int(np.searchsorted(t2, s_j + eps, side="right"))
                idx_s_j_minus = int(np.searchsorted(t2, lower_time, side="right"))
                idx_s_j_minus = max(idx_s_j_minus, idx_after_sj)
                if idx_s_j_minus >= t2.size:
                    continue
                r2_minus = y2[idx_s_j_minus] - y2[m]  # y2 at s_{j-1} is y2[m]

                Z_list.append(float(r1_minus * r2_minus))
                if return_pairs:
                    pairs.append((k + 1, m + 1, "ell<0:-"))

    # ------------------------------------------------------------
    # Case ell == 0: sum of two adjacency terms in Eq. (10)
    # ------------------------------------------------------------
    else:
        for k in range(t1_start.size):
            # Term A: r1_plus*r2_plus * 1{ t_{i-1} - s_j = 0 }  => s_j = t_{i-1}
            for m in map_t2_end.get(t1_start[k], []):
                s_j = t2_end[m]
                t_i = t1_end[k]

                lower_time = (s_j + m_plus) + eps
                idx_t1_plus = int(np.searchsorted(t1, lower_time, side="right"))
                idx_after_ti = int(np.searchsorted(t1, t_i + eps, side="right"))
                idx_t1_plus = max(idx_t1_plus, idx_after_ti)
                if idx_t1_plus >= t1.size:
                    continue
                r1_plus = y1[idx_t1_plus] - y1[k]

                s_jm1 = t2_start[m]
                t_im1 = t1_start[k]
                upper_time = min(s_jm1, (t_im1 - m_plus) - eps)
                idx_s_jm1_plus = int(np.searchsorted(t2, upper_time, side="right") - 1)
                if idx_s_jm1_plus < 0:
                    continue
                r2_plus = y2[m + 1] - y2[idx_s_jm1_plus]

                Z_list.append(float(r1_plus * r2_plus))
                if return_pairs:
                    pairs.append((k + 1, m + 1, "ell=0:+"))

            # Term B: r1_minus*r2_minus * 1{ s_{j-1} - t_i = 0 } => s_{j-1} = t_i
            for m in map_t2_start.get(t1_end[k], []):
                s_jm1 = t2_start[m]
                t_im1 = t1_start[k]
                t_i = t1_end[k]

                upper_t1 = min(t_im1, (s_jm1 - m_minus) - eps)
                idx_t_im1_minus = int(np.searchsorted(t1, upper_t1, side="right") - 1)
                if idx_t_im1_minus < 0:
                    continue
                r1_minus = y1[k + 1] - y1[idx_t_im1_minus]

                s_j = t2_end[m]
                lower_time = (t_i + m_minus) + eps
                idx_after_sj = int(np.searchsorted(t2, s_j + eps, side="right"))
                idx_s_j_minus = int(np.searchsorted(t2, lower_time, side="right"))
                idx_s_j_minus = max(idx_s_j_minus, idx_after_sj)
                if idx_s_j_minus >= t2.size:
                    continue
                r2_minus = y2[idx_s_j_minus] - y2[m]

                Z_list.append(float(r1_minus * r2_minus))
                if return_pairs:
                    pairs.append((k + 1, m + 1, "ell=0:-"))

    Z = np.asarray(Z_list, dtype=float)
    return (Z, pairs) if return_pairs else Z



# ============================================================
# Eq. (11): cross-covariance estimator
# ============================================================

def estimate_cross_covariance_uo_eq11(
    t1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    y2: np.ndarray,
    ell: Union[int, float],
    *,
    m_plus: int,
    m_minus: int,
    ell_in: Literal["ticks", "seconds"] = "ticks",
    return_debug: bool = False,
) -> Union[float, Tuple[float, dict]]:
    """
    Eq. (11): gamma_hat(ell) = - mean( Z^{(±)}_{ell,k} ),
    where Z^{(±)}_{ell,k} are the selected values from Eq. (10).
    """
    if return_debug:
        Z, pairs = build_Z_pm_selected(
            t1, y1, t2, y2, ell,
            m_plus=m_plus, m_minus=m_minus,
            ell_in=ell_in,
            return_pairs=True,
        )
    else:
        Z = build_Z_pm_selected(
            t1, y1, t2, y2, ell,
            m_plus=m_plus, m_minus=m_minus,
            ell_in=ell_in,
            return_pairs=False,
        )

    if Z.size == 0:
        gamma_hat = np.nan
    else:
        gamma_hat = float(-Z.mean())

    if not return_debug:
        return gamma_hat

    debug = {
        "ell": ell,
        "ell_in": ell_in,
        "m_plus": int(m_plus),
        "m_minus": int(m_minus),
        "N_pm": int(Z.size),
        "Z_pm": Z,
        "pairs": pairs,
    }
    return gamma_hat, debug