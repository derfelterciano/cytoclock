#!/usr/bin/env python3

import numpy as np
import math
from scipy.stats import norm as scipy_norm
from scipy.stats import false_discovery_control
import logging

_JTK_AMPFACTOR = np.sqrt(2)
_JTK_PIHAT = round(np.pi, 4)


def hlm(z: np.ndarray) -> float:
    """
    Hodges-Lehmann estimator of location.

    This is extremely robust and is resistant to exteme outliers
    and gross outliers. (e.g. a more robust median)

    hlm(z) -> median of all pairwise avgs (z_i + z_j) / 2

    Args:
        z (np.ndarray): input numpy arr data

    Returns:
        float: the median of all pairwase avgs
    """

    finite_vals = z[np.isfinite(z)]

    if len(finite_vals) == 0:
        return 0.0

    zz = np.add.outer(finite_vals, finite_vals)
    zz = zz[np.tril_indices(len(finite_vals))] / 2
    return float(np.median(zz))


def _build_dist_params(n_timepts: int, reps: int = 1) -> dict:
    """
    Build JTk null dist parameters.
    Uses exact Harding algorithm for small n,
    normal approximation when exact is numerically infeasibl

    Args:
        n_timepts (int): the number of timepoints there are
        reps (int, optional): repitions. Defaults to 1.

    Returns:
        dict: params of JTK null dists
    """
    time = [reps] * n_timepts
    num_vals = sum(time)
    max_jtk = (num_vals**2 - sum(t**2 for t in time)) // 2
    max_nlp = math.lgamma(num_vals + 1) - sum(math.lgamma(t + 1) for t in time)
    limit = np.log(np.finfo(float).max)

    if max_nlp > limit - 1:
        # normal approximation
        var = (
            num_vals**2 * (2 * num_vals + 3) - sum(t**2 * (2 * t + 3) for t in time)
        ) // 72

        return {
            "exact": False,
            "max": max_jtk,
            "sdv": np.sqrt(var),
            "exv": max_jtk / 2,
            "cp": None,
        }

    # Harding distribution
    MM = max_jtk // 2
    cf = [1.0] * (MM + 1)
    size = sorted(time)
    k = len(size)

    N = [0] * max(k - 1, 1)
    if k >= 2:
        N[k - 2] = size[k - 1]
        for i in range(k - 3, -1, -1):
            N[i] = size[i + 1] + N[i + 1]

    for i in range(k - 1):
        m, n = size[i], N[i]
        if n < MM:
            P = min(m + n, MM)
            for t in range(n + 1, P + 1):
                for u in range(MM, t - 1, -1):
                    cf[u] = cf[u] - cf[u - t]

        Q = min(m, MM)
        for s in range(1, Q + 1):
            for u in range(s, MM + 1):
                cf[u] = cf[u] + cf[u - s]

    if max_jtk % 2:
        tail = [2 * cf[MM] - cf[MM - 1 - i] for i in range(MM)] + [2 * cf[MM]]
    else:
        tail = [cf[MM] + cf[MM - 1] - cf[MM - 1 - i] for i in range(MM)] + [
            cf[MM] + cf[MM - 1]
        ]

    cf_full = cf + tail
    jtkcf = list(reversed(cf_full))
    ajtkcf = [(jtkcf[i] + jtkcf[i + 1]) / 2 for i in range(len(cf_full) - 1)]

    n_cp = 2 * max_jtk + 1
    cp_array = np.zeros(n_cp)
    for i, v in enumerate(jtkcf):
        if 2 * i < n_cp:
            cp_array[2 * i] = v / jtkcf[0]
    for i, v in enumerate(ajtkcf):
        if 2 * i + 1 < n_cp:
            cp_array[2 * i + 1] = v / jtkcf[0]

    return {"exact": True, "max": max_jtk, "cp": cp_array}


def _s_to_pvalue(S: float, dp: dict) -> float:
    """
    Convert Kendall S to 2-tailed p-value

    Args:
        S (float): Kendall S values
        dp (dict): dist params dict

    Returns:
        float: 2-tailed p-value
    """

    if S == 0:
        return 1.0

    max_jtk = dp["max"]
    jtk = (abs(S) + max_jtk) / 2
    if dp["exact"]:
        idx = int(1 + 2 * jtk)
        return float(2 * dp["cp"][idx] if idx < len(dp["cp"]) else 0.0)
    return float(2 * scipy_norm.cdf(-(jtk - 0.5), loc=-dp["exv"], scale=dp["sdv"]))


def init_ref_cosines(periods: list[int], n_timepoints: int) -> dict:
    """
    Precompute rank-based reference cosines for all (period and phase) combos.
    Only needs to be done once per (periods, n_timepoints) pair

    Args:
        periods (list[int]): desired range of periods to test
        n_timepoints (int): number of timepoints in the timeseries

    Returns:
        dict: holds all the computed ref cosines
    """
    timerange = np.arange(n_timepoints)
    ref = {}

    for pi, period in enumerate(periods):
        time2angle = 2 * _JTK_PIHAT / period
        theta = timerange * time2angle
        n_full_cycles = n_timepoints // period
        full_range = n_full_cycles * period

        cgoosv_cols = []
        signcos_cols = []

        for j in range(1, period + 1):
            delta = (j - 1) * time2angle / 2
            cos_v = np.cos(theta + delta)
            cos_r = np.argsort(np.argsort(cos_v)) + 1
            diff = np.sign(np.subtract.outer(cos_r, cos_r))
            lower = diff[np.tril_indices(n_timepoints, k=-1)]
            cgoosv_cols.append(lower)
            signcos_cols.append(np.sign(cos_v[:full_range]))

        ref[pi] = (np.column_stack(cgoosv_cols), np.column_stack(signcos_cols))

    return ref


def compute_best_tau(
    values: np.ndarray, periods: list[int], ref_cosines: dict, dist_params: dict
) -> tuple[float, float, float, int, float, float]:
    """
    Computes the best tau across all (period, phase) combinations.

    Args:
        values (np.ndarray): time series values
        periods (list[int]): list of possible period to choose from
        ref_cosines (dict): the reference cosines
        dist_params (list[dict]): JTK null dist params

    Returns:
        tuple[float, float, int, float, float]:
        (best_tau, best_raw_pvalue, bonferroni_adj_p, best_period_tp, best_lag_tp, best_amplitude)
    """
    n = len(values)
    z_out = np.sign(np.subtract.outer(values, values))
    foosv = z_out[np.tril_indices(n, k=-1)]

    all_pvals = []
    all_S = []
    all_pi = []
    all_lagi = []

    for pi, period in enumerate(periods):
        cgoosv, _ = ref_cosines[pi]
        dp = dist_params
        S_all = foosv @ cgoosv
        for lagi in range(period):
            S = float(S_all[lagi])
            all_pvals.append(_s_to_pvalue(S, dp))
            all_S.append(S)
            all_pi.append(pi)
            all_lagi.append(lagi)

    padj = np.minimum(np.array(all_pvals) * len(all_pvals), 1.0)
    best_padj = float(padj.min())
    best_mask = padj == best_padj

    # picking best via max amp
    best_amp = -np.inf
    best_tau = 0.0
    best_per = 0
    best_lag_h = 0.0
    best_raw_p = 1.0

    for idx in np.where(best_mask)[0]:
        pi = all_pi[idx]
        lagi = all_lagi[idx]
        period = periods[pi]
        S = all_S[idx]
        dp = dist_params

        s = np.sign(S) if S != 0 else 1.0
        lag = (period + (1 - s) * period / 4 - lagi / 2) % period

        _, signcos = ref_cosines[pi]
        sc = signcos[:, lagi]
        w = values[: len(sc)]
        w = (w - hlm(w)) * _JTK_AMPFACTOR
        tmp = s * w * sc
        amp = hlm(tmp)

        if amp > best_amp:
            best_amp = amp
            best_tau = abs(S) / dp["max"]
            best_per = period
            best_lag_h = lag
            best_raw_p = float(all_pvals[idx])

    return best_tau, best_raw_p, best_padj, best_per, best_lag_h, max(0.0, best_amp)


def _single_perm(seed, values, periods, ref_cosines, dist_params):
    """Single permutation processing"""
    logging.debug("eJTK: applying permutation to values!")
    perm_vals = np.random.default_rng(seed).permutation(values)
    tau, _, _, _, _, _ = compute_best_tau(perm_vals, periods, ref_cosines, dist_params)
    return tau
