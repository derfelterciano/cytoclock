#!/usr/bin/env python3

import polars as pl
import numpy as np
from numpy.typing import NDArray
from .base import OscillationBase
from scipy.optimize import curve_fit
from scipy.stats import f


class CosinorDetection(OscillationBase):
    """
    Implements a simple cosinor detection method for circadian data
    https://pmc.ncbi.nlm.nih.gov/articles/PMC3663600/#S4

    Params:
        - period (int): period of time to detect (default: 24 hours)
        - intervals (int): the interval of each timepoint in hours (default: 1 hour)
        - time_offset (int):
            the offset time (in hours) for experiment (default: 0)
        - start_index (int):
            if the timepoints are in 0 or 1 indexing (or another unit) (default: 0)

    """

    def __init__(
        self,
        period: int = 24,
        intervals: int = 1,
        time_offset: int = 0,
        start_index: int = 0,
    ) -> None:
        self.period = period
        self.intervals = intervals
        self.time_offset = time_offset
        self.start_index = start_index
        super().__init__()

    @property
    def params(self) -> dict:
        return {
            "period": self.period,
            "time_interval": self.intervals,
            "time_offset": self.time_offset,
            "start_index": self.start_index,
        }

    def fit(
        self, timepoints: NDArray[np.float64], values: NDArray[np.float64]
    ) -> dict | None:
        try:
            timepoints = (
                self.time_offset + (timepoints - self.start_index) * self.intervals
            )  # adjusts timepoints based on experiment parameters
            popt, _pcov = curve_fit(
                self._cosinor,
                timepoints,
                values,
                p0=[np.mean(values), np.std(values), 0],
            )

            residual_model = values - self._cosinor(timepoints, *popt)
            residual_null = values - np.mean(values)

            ss_model = np.sum(residual_model**2)
            ss_null = np.sum(residual_null**2)

            n = len(values)
            f_stat = ((ss_null - ss_model) / 2) / (ss_model / (n - 3))
            p_value = 1 - f.cdf(f_stat, 2, n - 3)

            phase_rads = popt[2] % (2 * np.pi)
            return {
                "baseline": popt[0],
                "amplitude": abs(popt[1]),
                "phase": phase_rads,
                "phase_hours": (phase_rads / (2 * np.pi)) * self.period,
                "p_value": p_value,
                "f_stat": f_stat,
                "ss_model": ss_model,
                "ss_null": ss_null,
                "fitted": self._cosinor(timepoints, *popt).tolist(),
            }

        except RuntimeError:
            return None

    @property
    def result_schema(self) -> dict[str, pl.DataType]:
        return {
            "baseline": pl.Float64(),
            "amplitude": pl.Float64(),
            "phase": pl.Float64(),
            "phase_hours": pl.Float64(),
            "p_value": pl.Float64(),
            "f_stat": pl.Float64(),
            "ss_model": pl.Float64(),
            "ss_null": pl.Float64(),
            "fitted": pl.List(pl.Float64()),
        }

    def _cosinor(
        self, time: NDArray[np.float64], baseline: float, amplitude: float, phase: float
    ):
        """Cosinor equation"""
        return baseline + (
            amplitude * np.cos(((2 * np.pi * time) / self.period) - phase)
        )


class DampedCosinorDetection(OscillationBase):
    """
    Just like it's sibling (CosinorDetection), it implments
    a Cosinor fit onto a timeseries data but with a damped
    term. This term is to account for fading amplitudes as
    time goes on in circadian data

    Params:
        - period (int): period of time to detect (default: 24 hours)
        - intervals (int): the interval of each timepoint in hours (default: 1 hour)
        - time_offset (int):
            the offset time (in hours) for experiment (default: 0)
        - start_index (int):
            if the timepoints are in 0 or 1 indexing (or another unit) (default: 0)

    """

    def __init__(
        self,
        period: int = 24,
        intervals: int = 1,
        time_offset: int = 0,
        start_index: int = 0,
    ) -> None:
        self.period = period
        self.intervals = intervals
        self.time_offset = time_offset
        self.start_index = start_index
        super().__init__()

    @property
    def params(self) -> dict:
        return {
            "period": self.period,
            "time_interval": self.intervals,
            "time_offset": self.time_offset,
            "start_index": self.start_index,
        }

    @property
    def result_schema(self) -> dict[str, pl.DataType]:
        return {
            "baseline": pl.Float64(),
            "amplitude": pl.Float64(),
            "phase": pl.Float64(),
            "phase_hours": pl.Float64(),
            "p_value": pl.Float64(),
            "f_stat": pl.Float64(),
            "ss_model": pl.Float64(),
            "ss_null": pl.Float64(),
            "damping_lambda": pl.Float64(),
            "tau": pl.Float64(),
            "fitted": pl.List(pl.Float64()),
        }

    def _damped_cosinor(
        self,
        time: NDArray[np.float64],
        baseline: float,
        amplitude: float,
        damping: float,
        phase: float,
    ) -> NDArray[np.float64]:
        """
        Damped cosinor equation (taken from google):

        𝑌(𝑡)=𝑀+𝐴⋅𝑒^(−𝜆𝑡)⋅cos(𝜔𝑡−𝜙)+𝜖
        """
        return baseline + (
            amplitude
            * np.exp(-damping * time)
            * np.cos(((2 * np.pi * time) / self.period) - phase)
        )

    def fit(self, timepoints, values) -> dict | None:
        try:
            timepoints = (
                self.time_offset + (timepoints - self.start_index) * self.intervals
            )  # adjusts timepoints based on experiment parameters

            timepoints = (
                timepoints - timepoints[0]
            )  # we need to anchor the timepoints to 0

            popt, _pcov = curve_fit(
                self._damped_cosinor,
                timepoints,
                values,
                p0=[np.mean(values), np.std(values), 0.01, 0],
                bounds=(
                    [-np.inf, 0, 1e-6, -2 * np.pi],
                    [np.inf, np.inf, 1.0, 2 * np.inf],
                ),
                maxfev=100_000,
            )

            residual_model = values - self._damped_cosinor(timepoints, *popt)
            residual_null = values - np.mean(values)

            ss_model = np.sum(residual_model**2)
            ss_null = np.sum(residual_null**2)

            n = len(values)
            f_stat = ((ss_null - ss_model) / 3) / (ss_model / (n - 4))
            p_value = 1 - f.cdf(f_stat, 3, n - 4)

            phase_rads = popt[3] % (2 * np.pi)
            return {
                "baseline": popt[0],
                "amplitude": abs(popt[1]),
                "phase": phase_rads,
                "phase_hours": (phase_rads / (2 * np.pi)) * self.period,
                "p_value": p_value,
                "f_stat": f_stat,
                "ss_model": ss_model,
                "ss_null": ss_null,
                "damping_lambda": popt[2],
                "tau": 1 / popt[2] if popt[2] != 0 else np.inf,
                "fitted": self._damped_cosinor(timepoints, *popt).tolist(),
            }

        except RuntimeError:
            return None
