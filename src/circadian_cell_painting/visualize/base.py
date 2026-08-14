#!/usr/bin/env python3

from matplotlib.axes import Axes
import polars as pl
from abc import ABC, abstractmethod
from pathlib import Path
from ..utils import OscillationResults
import numpy as np


class VisualizerBase(ABC):
    """
    This will be the base used for all visualization classes
    """

    def __init__(self, results: OscillationResults, tick_interval: float = 12):
        self.results = results
        self.tick_interval = tick_interval

    def plot_timeseries(self, ax: Axes, well: str, feature: str) -> None:
        """
        This plots the raw (but detrended) timeseries data.
        args:
            - ax: matplotlib axes
            - well (str): the well name
            - feature (str): the feature name
        """
        clean_data = (
            self.results.scan_data()
            .filter(
                (pl.col(self.results.well_col) == well)
                & (pl.col(self.results.feature_col) == feature)
            )
            .select([self.results.timepoint_col, self.results.value_col])
            .sort(self.results.timepoint_col)
            .collect()
        )

        hours = clean_data[self.results.timepoint_col].to_numpy()
        hours = self.results.params.get("time_offset", 0) + (
            hours - self.results.params.get("start_index", 0)
        ) * self.results.params.get("time_interval", 1)

        values = clean_data[self.results.value_col].to_numpy()

        ax.plot(
            hours,
            values,
            "-o",
            color="royalblue",
            markersize=4,
            linewidth=5,
            label="detrended data",
        )

        x_min = hours.min()
        x_max = hours.max()

        # period = self.results.params.get("period", 24)
        period_marks = np.arange(
            np.ceil(x_min / self.tick_interval) * self.tick_interval,
            x_max + self.tick_interval,
            self.tick_interval,
        )

        for mark in period_marks:
            ax.axvline(mark, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        ax.set_xticks(period_marks)
        ax.set_xticklabels([f"{int(m)}h" for m in period_marks])

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(f"Detrended {self.results.stat}")

    def get_significant(
        self, alpha: float | None = None, filter_col: str = "p_adjusted"
    ) -> pl.DataFrame:
        """
        Returns all results with p_adjusted values <= alpha.
        This filters out all the non-significant data giving you real
        oscilating results.

        args:
            - alpha (float | None): the alpha value to filter out the p-values
            by
            - filter_col: str: the column to filter p-values from.
            (default: "p_adjusted")

        returns:
            pl.DataFrame: all the results with p values <= alpha.
        """
        threshold = alpha if alpha is not None else self.results.adj_alpha
        return (
            self.results.scan_results()
            .filter(pl.col(filter_col) <= threshold)
            .collect()
        )

    @abstractmethod
    def plot_fit(self, ax: Axes, row: dict) -> None:
        """
        Overlays the model fit on a time series plot.
        Method specific — must be implemented by each visualizer.
        args:
            - ax: matplotlib axes object to plot on
            - row (dict): a single row from the results dataframe
                containing model parameters (amplitude, phase, etc.)
        """
        ...

    @abstractmethod
    def plot_summary(self, ax, significant: pl.DataFrame) -> None:
        """
        Plots a summary of all significant features for a single well.
        Method specific — must be implemented by each visualizer.
        args:
            - ax: matplotlib axes object to plot on
            - significant (pl.DataFrame): filtered results dataframe
        """
        ...

    def view(
        self,
        out_dir: Path,
        alpha: float | None = None,
    ) -> None:
        """
        Generates one PDF per well containing:
            - Page 1: summary of all significant features
            - Page N: one page per significant feature
        args:
            - out_dir (Path): directory to write PDFs to
            - alpha (float | None): significance threshold,
                defaults to results.adj_alpha
        """
        ...
