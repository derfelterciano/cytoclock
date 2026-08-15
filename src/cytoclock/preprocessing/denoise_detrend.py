#!/usr/bin/env python3
import polars as pl
import logging


def denoise_detrend(
    in_frame: pl.LazyFrame,
    feature_col: str,
    well_col: str | list[str],
    timepoint_col: str,
    window_size: int = 8,
) -> pl.LazyFrame:
    well_cols = well_col if isinstance(well_col, list) else [well_col]
    group_keys = well_cols + [feature_col]
    index_keys = well_cols + [
        timepoint_col
    ]  # an index is considered the timepoint + wellname
    stat_cols = [
        c
        for c in in_frame.collect_schema().names()
        if c not in (set(group_keys) | set(index_keys))
    ]

    # window size must always be odd
    # if its even just add 1
    if window_size % 2 == 0:
        window_size += 1

    logging.info(f"denoise/detrend: window size set to: {window_size} timepoints")

    pass1 = [  # first pass 24-hour rolling avg
        pl.col(stat)
        .rolling_mean(window_size, center=True, min_samples=window_size)
        .over(group_keys)
        .alias(f"{stat}_pass1")
        for stat in stat_cols
    ]

    pass2 = [  # second pass 24-hour rolling avg of the prev. avg
        pl.col(f"{stat}_pass1")
        .rolling_mean(window_size, center=True, min_samples=window_size)
        .over(group_keys)
        .alias(f"{stat}_pass2")
        for stat in stat_cols
    ]

    logging.info("denoise/detrend: completed 2 pass rolling avg")

    # subtracting noise from war signal
    detrend = [pl.col(stat) - pl.col(f"{stat}_pass2") for stat in stat_cols]

    logging.info("denoise/detrend: subtracted raw signal from noise")

    # temp cols
    temp_cols = [f"{stat}_pass2" for stat in stat_cols] + [
        f"{stat}_pass1" for stat in stat_cols
    ]

    return (
        in_frame.sort(well_cols + [feature_col] + [timepoint_col])
        .with_columns(pass1)
        .with_columns(pass2)
        .with_columns(detrend)
        .drop(temp_cols)
        .drop_nulls(subset=[c for c in stat_cols if c != "stddev"])
        .sort(index_keys + [feature_col])
    )
