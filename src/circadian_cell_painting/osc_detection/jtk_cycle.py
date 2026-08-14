#!/usr/bin/env python3

import polars as pl
import numpy as np
from numpy.typing import NDArray
from .base import OscillationBase
from ._ejtk_cycle_tools import (
    _build_dist_params,
    compute_best_tau,
    init_ref_cosines,
)
from ._ejtk_cycle_curve_fits import (
    construct_raw_jtk_curve,
    jtk_fixed_period_cosinor_regression,
)
import logging


class JTK_CYCLE(OscillationBase):
    """
    Implements the Original JTK_CYCLE

    [Original JTk_CYCLE link](https://pmc.ncbi.nlm.nih.gov/articles/PMC3119870/)

    """

    def __init__(
        self,
        periods: list[int],
        interval: float = 1.0,
        time_offset: float = 0.0,
        start_index: int = 0,
    ):
        self.periods = periods
        self.interval = interval
        self.time_offset = time_offset
        self.start_index = start_index

        self._ref_cosines: dict | None = None
        self._dist_params: dict | None = None
        self._n_timepoints: int | None = None

        super().__init__()

    @property
    def params(self):
        return {
            "periods": self.periods,
            "time_interval": self.interval,
            "time_offset": self.time_offset,
            "start_index": self.start_index,
            "period": float(np.mean(self.periods)) * self.interval,
        }

    @property
    def result_schema(self):
        return {
            "p_value": pl.Float64(),  # Bonferroni-adjusted JTK p
            "jtk_corrected_p": pl.Float64(),
            "period": pl.Float64(),  # hours
            "jtk_phase_hours": pl.Float64(),
            "jtk_amplitude": pl.Float64(),
            "tau": pl.Float64(),  # Kendall tau effect size (0-1)
            "jtk_baseline": pl.Float64(),
            "jtk_fitted": pl.List(pl.Float64()),
            "lsq_baseline": pl.Float64(),
            "lsq_amplitude": pl.Float64(),
            "lsq_phase_hours": pl.Float64(),
            "lsq_r_squared": pl.Float64(),
            "lsq_n_cycles": pl.Float64(),
            "lsq_fitted": pl.List(pl.Float64()),
        }

    def _jtk_init(self, n_timepoints: int) -> None:
        """Lazily precomputing reference cosines and distributions"""

        if self._n_timepoints == n_timepoints:
            return

        self._n_timepoints = n_timepoints
        self._ref_cosines = init_ref_cosines(self.periods, n_timepoints)
        self._dist_params = _build_dist_params(n_timepoints)

    def fit(self, timepoints, values):
        logging.info("Initalizing JTK")

        try:
            self._jtk_init(len(timepoints))
            hours = self.time_offset + (timepoints - self.start_index) * self.interval

            # single call — no permutation loop
            tau, raw_p, adj_p, best_per, best_lag, jtk_amplitude = compute_best_tau(
                values=values,
                periods=self.periods,
                ref_cosines=self._ref_cosines,
                dist_params=self._dist_params,
            )

            period_hours = float(best_per) * self.interval
            lag_hours = float(best_lag) * self.interval

            jtk_baseline, jtk_fitted = construct_raw_jtk_curve(
                timepoints=hours,
                values=values,
                period=period_hours,
                lag=lag_hours,
                amplitude=jtk_amplitude,
            )

            lsq = jtk_fixed_period_cosinor_regression(
                timepoints=hours, values=values, period=period_hours
            )

            return {
                "p_value": raw_p,
                "jtk_corrected_p": adj_p,
                "period": period_hours,
                "jtk_phase_hours": lag_hours,
                "jtk_amplitude": jtk_amplitude,
                "tau": tau,
                "jtk_baseline": jtk_baseline,
                "jtk_fitted": jtk_fitted,
                "lsq_baseline": lsq["baseline"],
                "lsq_amplitude": lsq["amplitude"],
                "lsq_phase_hours": lsq["phase_hours"],
                "lsq_r_squared": lsq["r_squared"],
                "lsq_n_cycles": lsq["n_cycles"],
                "lsq_fitted": lsq["y_fit"],
            }
        except Exception as e:
            logging.exception(f"JTK failure: {e}")
            return None
