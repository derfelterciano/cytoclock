#!/usr/bin/env python3

import numpy as np
from ._ejtk_cycle_tools import hlm


def construct_raw_jtk_curve(
    timepoints: np.ndarray,
    values: np.ndarray,
    period: float,
    lag: float,
    amplitude: float,
) -> tuple[float, list[float]]:
    """
    Reconstructs JTK's RAW rank-based curve — NOT least-squares fitted.
    Uses JTK's own amplitude/phase estimates directly.
    Useful for visualizing what JTK's rank-matching procedure "thinks"
    the rhythm looks like, though it will NOT minimize residuals against
    the actual data (see fit_fixed_period_cosinor for that).
    """
    baseline = hlm(values)
    phase_rad = 2 * np.pi * lag / period
    fitted = baseline + amplitude * np.cos(
        (2 * np.pi * timepoints / period) - phase_rad
    )
    return float(baseline), fitted.tolist()


def jtk_fixed_period_cosinor_regression(
    timepoints: np.ndarray, values: np.ndarray, period: float
) -> dict[str, any]:
    """
    Linear least-squares cosinor fit with period fixed (e.g. from JTK).
    JTK supplies the period; this fits baseline + amplitude + phase
    by multiple linear regression on the cos/sin basis.
    """

    t = np.asarray(timepoints, dtype=float)
    y = np.asarray(values, dtype=float)
    omega = 2 * np.pi / period

    X = np.column_stack([np.ones_like(t), np.cos(omega * t), np.sin(omega * t)])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    baseline, A, B = coeffs

    amplitude = np.hypot(A, B)
    phase = np.arctan2(B, A)
    phase_hours = (phase % (2 * np.pi)) / omega

    y_fit = X @ coeffs
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n_cycles = (t.max() - t.min()) / period

    return {
        "period": period,
        "baseline": baseline,
        "amplitude": amplitude,
        "phase": phase,
        "phase_hours": phase_hours,
        "r_squared": r_squared,
        "n_cycles": n_cycles,
        "y_fit": y_fit.tolist(),
    }
