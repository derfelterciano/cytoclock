#!/usr/bin/env python3
"""
A file of utilitis for this project
"""

from pathlib import Path
import shutil
from time import sleep
from dataclasses import dataclass
import polars as pl
import pickle
from typing import Any
from matplotlib.axes import Axes
import numpy as np


def clean_folder(
    dir: Path, delete_only: bool = False, create_only: bool = False
) -> None:
    if create_only:
        dir.mkdir(parents=True, exist_ok=True)
        return

    if dir.exists():
        shutil.rmtree(dir)
        sleep(0.5)

    if not delete_only:
        dir.mkdir(parents=True, exist_ok=True)


def cosinor_fitted_long(
    results_lf: pl.LazyFrame,
    timepoints: list[float | int],
    timepoint_col: str,
    feature: str | None = None,
    feature_col: str = "feature",
) -> pl.DataFrame:
    """
    Returns the results of ONLY the cosinor analysis with the fitted
    column in long form

    Args:
        results_lf (pl.LazyFrame): A Lazy fram for the results of the cosinor
            analysis
        feature (str | None, optional): A specific feature to select. Defaults to None.

    Returns:
        _type_: Returns a dataframe with the exploded fitted column
    """
    if "fitted" not in results_lf.collect_schema().names():
        raise KeyError(f"This is not a cosinor dataset as 'fitted' is not a column!")

    if feature is not None:
        results_lf = results_lf.filter(pl.col(feature_col) == feature)

    results = results_lf.collect()

    return results.with_columns(
        pl.Series(timepoint_col, [timepoints] * len(results))
    ).explode(["fitted", timepoint_col])


@dataclass
class OscillationResults:
    """
    Bundles the output of detect() into a central class!
    Great for portability of meta information
    *NOTE:* This class does not contain the actual data itself.
    It contains only the meta information and file paths
    """

    # columns metadata
    well_col: str
    feature_col: str
    timepoint_col: str
    value_col: str
    stat: str

    # model params
    detector_params: dict[str, Any]

    # paths
    results_path: Path
    data_path: Path
    platemap: Path | pl.DataFrame | None = None

    # detction metrics
    detector_name: str = "CosinorFitting"
    adj_alpha: float = 0.05
    adj_method: str = "fdr_bh"
    correction_group: list[str] | None = None
    start_time: float | None = None
    end_time: float | None = None

    def scan_data(self):
        return pl.scan_parquet(self.data_path)

    def scan_results(self):
        return pl.scan_parquet(self.results_path)

    def load_result(self):
        return pl.read_parquet(self.results_path)

    def to_pickle(self, path: Path) -> None:
        """
        Saves this object as a pickle file
        """

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @property
    def params(self):
        return self.detector_params

    @classmethod
    def load_pickle(cls, path: Path) -> "OscillationResults":
        with open(path, "rb") as f:
            return pickle.load(f)


def add_period_marks(ax: Axes, x_min: float, x_max: float, interval=12):
    """
    Adding specific period marks to circadian plots

    Args:
        ax (Axes): matplot lib Axes
        x_min (float): the minimum x val
        x_max (float): the maximum x val
        interval (int, optional): interval to put tick marks at. Defaults to 12.
    """
    marks = np.arange(np.ceil(x_min / interval) * interval, x_max, interval)
    for mark in marks:
        ax.axvline(mark, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xticks(marks)
    ax.set_xticklabels([f"{int(m)}h" for m in marks], fontsize=7)


def retrieve_wells(data: pl.DataFrame, well_col: str) -> list[str]:
    return data[well_col].unique().to_list()
