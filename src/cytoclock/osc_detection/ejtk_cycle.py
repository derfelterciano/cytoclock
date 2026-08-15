#!/usr/bin/env python3

import polars as pl
import numpy as np
from numpy.typing import NDArray
from .base import OscillationBase
from ._ejtk_cycle_tools import (
    _build_dist_params,
    compute_best_tau,
    init_ref_cosines,
    _single_perm,
)
from ._ejtk_cycle_curve_fits import (
    construct_raw_jtk_curve,
    jtk_fixed_period_cosinor_regression,
)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import logging
from tqdm import tqdm


class eJTK_CYCLE(OscillationBase):
    """
    Implements the *empirical* JTK_CYCLE

    https://pmc.ncbi.nlm.nih.gov/articles/PMC4368642/

    [Original JTk_CYCLE](https://pmc.ncbi.nlm.nih.gov/articles/PMC3119870/)

    """

    def __init__(
        self,
        periods: list[int],
        interval: float = 1.0,
        time_offset: float = 0.0,
        start_index: int = 0,
        n_perms: int = 500,
        n_workers: int = 4,
    ):
        self.periods = periods
        self.interval = interval
        self.time_offset = time_offset
        self.start_index = start_index
        self.n_perms = n_perms
        self.n_workers = n_workers

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
            "n_perms": self.n_perms,
            "period": float(np.mean(self.periods)) * self.interval,
        }

    @property
    def result_schema(self):
        return {
            "p_value": pl.Float64(),  # empirical p-value
            "adj_p_jtk": pl.Float64(),  # Bonferroni JTK p (reference)
            "period": pl.Float64(),  # hours
            "phase_hours": pl.Float64(),  # hours to peak within cycle
            "amplitude": pl.Float64(),  # Hodges-Lehmann amplitude
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

    def _ejtk_fit_single(
        self,
        timepoints: np.ndarray,
        values: np.ndarray,
        periods: list[int],
        ref_cosines: dict,
        dist_params: list[dict],
        interval: float,
        n_perms: int,
        n_workers: int = 1,
        rng_seed: int = 123456789,
    ) -> dict | None:
        """
        running eJTK_CYCLE on a single timeseries data

        1. Compute observed tau from real data
        2. Build empirical null by permuting time labels n_perms times
        3. emp_p = fraction of permutations where |tau_perm| >= |tau_obs|

        Args:
            values (np.ndarray): time series data
            periods (list[int]): list of desired periods to test
            ref_cosines (dict): generated reference cosines
            dist_params (list[dict]): null dist parameters
            interval (float): sampling intervals
            n_perms (int): number of permutations (for empirical)
            n_workers (int): number of threads/workers to use
            rng_seed (int, optional): rng seed. Defaults to 123456789.

        Returns:
            dict | None: _description_
        """

        try:
            rng = np.random.default_rng(seed=rng_seed)

            logging.info("eJTK: Computing the best tau!")
            # observed stats
            obs_tau, raw_p, adj_p, best_per, best_lag, amplitude = compute_best_tau(
                values=values,
                periods=periods,
                ref_cosines=ref_cosines,
                dist_params=dist_params,
            )

            seeds = rng.integers(0, 2**32, size=n_perms, dtype=int).tolist()

            # def _single_perm(seed: int) -> float:
            #     logging.debug("eJTK: applying permutation to values!")

            #     perm_vals = np.random.default_rng(seed).permutation(values)
            #     tau, _, _, _, _ = compute_best_tau(
            #         perm_vals, periods, ref_cosines, dist_params
            #     )
            #     return tau

            # logging.info("eJTK: multithreading permutations")
            # with ThreadPoolExecutor(max_workers=n_workers) as pool:
            #     perm_taus = np.array(list(pool.map(_single_perm, seeds)))

            if n_workers == 1:
                perm_taus = np.array(
                    [
                        _single_perm(seed, values, periods, ref_cosines, dist_params)
                        for seed in seeds
                    ]
                )
            else:
                logging.info("eJTK: multithreading permutations")
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    func = partial(
                        _single_perm,
                        values=values,
                        periods=periods,
                        ref_cosines=ref_cosines,
                        dist_params=dist_params,
                    )
                    perm_taus = np.array(list(pool.map(func, seeds)))

            # empirical p-values
            emp_p = float(np.mean(perm_taus >= obs_tau))
            emp_p = max(emp_p, 1.0 / n_perms)

            period_hours = float(best_per) * interval
            lag_hours = float(best_lag) * interval

            logging.info("eJTK: Building curve fits")
            # JTK-based curve
            jtk_baseline, jtk_fitted = construct_raw_jtk_curve(
                timepoints=timepoints,
                values=values,
                period=period_hours,
                lag=lag_hours,
                amplitude=amplitude,
            )

            # least squares
            lsq = jtk_fixed_period_cosinor_regression(
                timepoints=timepoints, values=values, period=period_hours
            )

            return {
                "p_value": emp_p,
                "adj_p_jtk": adj_p,
                "period": period_hours,
                "phase_hours": lag_hours,
                "amplitude": amplitude,
                "tau": obs_tau,
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
            logging.exception(f"eJTK failure: {e}")
            return None

    def _jtk_init(self, n_timepoints: int) -> None:
        """Lazily precomputing reference cosines and distributions"""

        if self._n_timepoints == n_timepoints:
            return

        self._n_timepoints = n_timepoints
        self._ref_cosines = init_ref_cosines(self.periods, n_timepoints)
        self._dist_params = _build_dist_params(n_timepoints)

    def fit(self, timepoints, values):
        logging.info("Initalizing JTK")
        timepoints = self.time_offset + (timepoints - self.start_index) * self.interval
        self._jtk_init(len(timepoints))

        logging.info("Applying eJTK to a single feature!")
        return self._ejtk_fit_single(
            timepoints=timepoints,
            values=values,
            periods=self.periods,
            ref_cosines=self._ref_cosines,
            dist_params=self._dist_params,
            interval=self.interval,
            n_perms=self.n_perms,
        )

    def fit_all(self, data, feature_col, time_col, value_col, stat=None, well=None):
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

        feature_values = {}
        for feat in features:
            vals = (
                data.filter(pl.col(feature_col) == feat)
                .sort(time_col)[value_col]
                .to_numpy()
                .astype(np.float64)
            )

            if not np.all(vals == 0):
                feature_values[feat] = vals

        results = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {
                pool.submit(self.fit, time, vals): feat
                for feat, vals in feature_values.items()
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing features"
            ):
                feat = futures[future]
                fit = future.result()
                if fit is not None:
                    results.append(
                        {"WellName": well, feature_col: feat, "stat": stat, **fit}
                    )

        # for feat in tqdm(features, total=len(features), desc="Processing features"):
        #     values = (
        #         data.filter(pl.col(feature_col) == feat)
        #         .sort(time_col)[value_col]
        #         .to_numpy()
        #         .astype(np.float64)
        #     )

        #     if np.all(values == 0):
        #         fit = None
        #     else:
        #         fit = self.fit(timepoints=time, values=values)

        #     if fit is not None:
        #         results.append(
        #             {"WellName": well, feature_col: feat, "stat": stat, **fit}
        #         )

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
