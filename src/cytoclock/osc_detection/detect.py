#!/usr/bin/env python3

import polars as pl
from statsmodels.stats.multitest import multipletests
from .base import OscillationBase
from .cosinor import CosinorDetection
from .cosinor import DampedCosinorDetection
from pathlib import Path
from tqdm import tqdm
from ..utils import OscillationResults
from typing import Literal


def detect(
    file_path: Path,
    platemap_path: Path | pl.DataFrame,
    well_column: str,
    feature_col: str,
    timepoint_col: str,
    value_col: str,
    stat: str,
    out_file: Path,
    start_time: float | None = None,
    end_time: float | None = None,
    alpha: float = 0.05,
    adj_method: str = "fdr_bh",
    correction_group: list[str] | None = None,
    detector: OscillationBase | None = None,
) -> OscillationResults:
    """
    Run oscillation detection across all wells and features. This function
    also adjusts p-value based on a population defined `correction_group`

    args:
        - file_path (Path): path to preprocessed parquet file
        - platemap_path (Path | pl.DataFrame): path to platemap CSV or an
          already-loaded platemap DataFrame. Must contain `well_column` and
          all columns specified in `correction_group`
        - well_column (str): column name identifying the well (e.g. 'WellName')
        - feature_col (str): column name identifying the feature
        - timepoint_col (str): column name identifying the timepoint
        - value_col (str): column name of the stat value to fit (e.g. 'mean')
        - stat (str): label for the stat being tested, stored in results
        - start_time (float | None): starts analysis at a given timepoint
          which must be the same time unit as the input data (default: None)
        - end_time (float | None): ends analysis at a given timepoint
          which must be the same time unit as the input data (default: None)
        - out_file (Path): the outfile of results as a .parquet file
        - correction_group (list[str]): platemap columns to group by for
          independent BH correction (e.g. ['cell_line', 'condition'])
        - alpha (float): significance threshold for BH correction (default 0.05)
        - adj_method (str): multiple testing correction method passed to
          statsmodels.stats.multitest.multipletests (default 'fdr_bh')
        - detector (OscillationBase): the oscillation detection function
          to be used for analysis (default: CosinorDetection)

    returns:
        pl.DataFrame: results for all wells with columns from `result_schema`,
        sorted by well and `p_adjusted`

    raises:
        ValueError: if platemap is missing well_column or any correction_group columns
    """

    if isinstance(platemap_path, Path):
        pm = pl.read_csv(platemap_path)
    else:
        pm = platemap_path

    if detector is None:
        detector = CosinorDetection()

    wells = (
        pl.scan_parquet(file_path)
        .select(well_column)
        .unique()
        .collect()[well_column]
        .to_list()
    )

    results = []
    for well in tqdm(wells, total=len(wells), desc="Well progress"):
        well_data = (
            pl.scan_parquet(file_path)
            .filter(pl.col(well_column) == well)
            .select([well_column, feature_col, timepoint_col, value_col])
            .sort([feature_col, timepoint_col])
        )

        if start_time is not None:
            well_data = well_data.filter(pl.col(timepoint_col) >= start_time)
        if end_time is not None:
            well_data = well_data.filter(pl.col(timepoint_col) <= end_time)

        well_data = well_data.collect()

        if well_data.is_empty():
            continue

        result = detector.fit_all(
            data=well_data,
            feature_col=feature_col,
            time_col=timepoint_col,
            value_col=value_col,
            stat=stat,
            well=well,
        )

        if result.is_empty():
            continue

        results.append(result)

    if len(results) == 0:
        raise ValueError("detect(): NO RESULTS FOUND")

    combined_results: pl.DataFrame = (
        pl.concat(results)
        .join(pm, left_on="WellName", right_on=well_column, how="left")
        .rename({"WellName": well_column})
    )

    if correction_group is None:
        _, p_adjusted, _, _ = multipletests(
            combined_results["p_value"].to_numpy(), alpha=alpha, method=adj_method
        )
        corrected_df = combined_results.with_columns(
            pl.Series("p_adjusted", p_adjusted)
        ).sort([well_column, "p_adjusted"], descending=False)
    else:
        corrected = []
        for group_vals, group_df in combined_results.group_by(correction_group):
            _, p_adjusted, _, _ = multipletests(
                group_df["p_value"].to_numpy(), alpha=alpha, method=adj_method
            )

            corrected.append(group_df.with_columns(pl.Series("p_adjusted", p_adjusted)))

        corrected_df: pl.DataFrame = pl.concat(corrected).sort(
            [well_column, "p_adjusted"], descending=False
        )

    corrected_df.write_parquet(out_file)

    return OscillationResults(
        well_col=well_column,
        feature_col=feature_col,
        timepoint_col=timepoint_col,
        value_col=value_col,
        stat=stat,
        results_path=out_file,
        data_path=file_path,
        correction_group=correction_group,
        start_time=start_time,
        detector_name=type(detect).__name__,
        end_time=end_time,
        detector_params=detector.params,
    )


def detect_cosinor_multiperiod(
    file_path: Path,
    platemap_path: Path | pl.DataFrame,
    period_column: str,
    well_column: str,
    feature_col: str,
    timepoint_col: str,
    value_col: str,
    stat: str,
    out_file: Path,
    start_time: float | None = None,
    end_time: float | None = None,
    alpha: float = 0.05,
    adj_method: str = "fdr_bh",
    correction_group: list[str] | None = None,
    damped_method: bool = False,
    intervals: int = 1,
    time_offset: int = 0,
    start_index: int = 0,
) -> OscillationResults:
    if isinstance(platemap_path, Path):
        pm = pl.read_csv(platemap_path)
    else:
        pm = platemap_path

    wells = (
        pl.scan_parquet(file_path)
        .select(well_column)
        .unique()
        .collect()[well_column]
        .to_list()
    )

    results = []
    for well in tqdm(wells, total=len(wells), desc="wells"):
        well_data = (
            pl.scan_parquet(file_path)
            .filter(pl.col(well_column) == well)
            .select([well_column, feature_col, timepoint_col, value_col])
            .sort([feature_col, timepoint_col])
        )

        if start_time is not None:
            well_data = well_data.filter(pl.col(timepoint_col) >= start_time)
        if end_time is not None:
            well_data = well_data.filter(pl.col(timepoint_col) <= end_time)

        well_data = well_data.collect()

        if well_data.is_empty():
            continue

        period_row = pm.filter(pl.col(well_column) == well).select([period_column])
        if period_row.is_empty() or period_row.item() is None:
            print(f"No period found for well {well} — skipping")
            continue
        period = float(period_row.item())
        # print(f"well: {well} | period: {period}")

        if damped_method:
            model = DampedCosinorDetection(
                period=period,
                intervals=intervals,
                time_offset=time_offset,
                start_index=start_index,
            )
        else:
            model = CosinorDetection(
                period=period,
                intervals=intervals,
                time_offset=time_offset,
                start_index=start_index,
            )

        result = model.fit_all(
            data=well_data,
            feature_col=feature_col,
            time_col=timepoint_col,
            value_col=value_col,
            stat=stat,
            well=well,
        )

        if result.is_empty():
            continue

        result = result.with_columns(pl.lit(period).alias("period"))

        results.append(result)

    if len(results) == 0:
        raise ValueError("detect_cosinor_multiperiod(): NO RESULTS FOUND")

    combined_results: pl.DataFrame = (
        pl.concat(results)
        .join(pm, left_on="WellName", right_on=well_column, how="left")
        .rename({"WellName": well_column})
    )

    if correction_group is None:
        _, p_adjusted, _, _ = multipletests(
            combined_results["p_value"].to_numpy(), alpha=alpha, method=adj_method
        )
        corrected_df = combined_results.with_columns(
            pl.Series("p_adjusted", p_adjusted)
        ).sort([well_column, "p_adjusted"], descending=False)
    else:
        corrected = []
        for group_vals, group_df in combined_results.group_by(correction_group):
            _, p_adjusted, _, _ = multipletests(
                group_df["p_value"].to_numpy(), alpha=alpha, method=adj_method
            )

            corrected.append(group_df.with_columns(pl.Series("p_adjusted", p_adjusted)))

        corrected_df: pl.DataFrame = pl.concat(corrected).sort(
            [well_column, "p_adjusted"], descending=False
        )

    corrected_df.write_parquet(out_file)

    return OscillationResults(
        well_col=well_column,
        feature_col=feature_col,
        timepoint_col=timepoint_col,
        value_col=value_col,
        stat=stat,
        results_path=out_file,
        data_path=file_path,
        platemap=pm,
        detector_name="DampedCosinorFitting" if damped_method else "CosinorFitting",
        adj_alpha=alpha,
        adj_method=adj_method,
        correction_group=correction_group,
        start_time=start_time,
        end_time=end_time,
        detector_params=model.params,
    )


# if __name__ == "__main__":
#     data_path = Path(
#         "/home/derfelt/PartchLabFiles/Circadian Rhythms/Live Study/"
#         "preprocessed_data/2026-03-11CelPaint_PhenovNoTreatDex_HARM"
#         "ONY_FIXED.parquet"
#     )
#     pm_path = Path(
#         "/home/derfelt/PartchLabFiles/Circadian"
#         " Rhythms/Live Study/2026-03-16CellPaint_map.xlsx"
#     )

#     pm = pl.read_excel(pm_path, engine="openpyxl")
#     # print(pm)

#     model = CosinorDetection(period=24, intervals=3)

#     result = detect(
#         file_path=data_path,
#         platemap_path=pm,
#         well_column="WellName",
#         feature_col="feature",
#         timepoint_col="Timepoint",
#         value_col="mean",
#         stat="mean",
#         correction_group=["cell line", "condition"],
#         detector=model,
#     )

#     (
#         result.with_columns(
#             pl.col("fitted").list.eval(pl.element().cast(pl.String)).list.join(", ")
#         ).write_csv("test.csv")
#     )

#     print(result)
