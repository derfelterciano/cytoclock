#!/usr/bin/env python3

import polars as pl
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


class OscillationBase(ABC):
    """
    This is the base class of all oscillation detection.
    Future oscillation methods will ADHERE to this class

    *NOTE:* The basis for any data frame manipulation is polars

    *MUST IMPLEMENT*:
    - `fit()`
    - `result_schema`
    - `params`
    """

    def __init__(self) -> None:
        self._validate_schema()

    @property
    @abstractmethod
    def params(self) -> dict:
        """
        Returns the models parameters
        """
        ...

    @abstractmethod
    def fit(
        self, timepoints: NDArray[np.float64], values: NDArray[np.float64]
    ) -> dict | None:
        """
        Fits the model to a feature's time series data

        args:
            - timepoints (NDArray[np.float64]): a numpy array of timepoints
            - values (NDArray[np.float64]): a numpy array of the time
            series data

        returns:
            dict | None: a dictionary of results or None at all
        """
        ...

    @property
    @abstractmethod
    def result_schema(self) -> dict[str, pl.DataType]:
        """
        Defines output columns and types for building the results dataframe

        ALL SCHEMAS MUST CONTAIN:
            - `p_value: pl.Float64`

        If possible, try to define:
            - `fitted: pl.List(pl.Float64)`: This is for the fitted curve
        """
        ...

    def _validate_schema(self) -> None:
        if "p_value" not in self.result_schema:
            raise NotImplementedError(
                f"{self.__class__.__name__}.results_schema must "
                "include 'p_value: pl.Float64' and 'p_adjusted: pl.Float64'"
            )

    def fit_all(
        self,
        data: pl.DataFrame,
        feature_col: str,
        time_col: str,
        value_col: str,
        stat: str | None = None,
        well: str | None = None,
    ) -> pl.DataFrame:
        """
        Fits the model to ALL features (mainly meant for a single well)

        args:
            - data (pl.DataFrame): experiment data
            - feature_col (str): the feature column in `data`
            - time_col (str): the timepoint column in `data`
            - value_col (str): the value points at each timepoint in `data`
            - stat (str): the title of the metric being tested
            - well (str): the name of the well being tested

        returns:
            (pl.DataFrame): a dataframe of all the results
        """
        features = data[feature_col].unique().to_list()
        time = (
            data.filter(pl.col(feature_col) == features[0])
            .sort(time_col)[time_col]
            .to_numpy()
            .astype(np.float64)
        )

        results = []

        for feat in tqdm(features, total=len(features), desc="Processing features"):
            values = (
                data.filter(pl.col(feature_col) == feat)
                .sort(time_col)[value_col]
                .to_numpy()
                .astype(np.float64)
            )

            if np.all(values == 0):
                fit = None
            else:
                fit = self.fit(timepoints=time, values=values)

            if fit is not None:
                results.append(
                    {"WellName": well, feature_col: feat, "stat": stat, **fit}
                )

        if len(results) == 0:
            return pl.DataFrame()

        well_df = pl.DataFrame(
            results,
            schema={
                "WellName": pl.String,
                feature_col: pl.String,
                "stat": pl.String,
                **self.result_schema,
            },
        )
        return well_df
