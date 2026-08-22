#!/usr/bin/env python3

import polars as pl
import logging
from ..preprocessing.preprocess import preprocess_aggregation
from pathlib import Path
from ..utils import clean_folder


def format_sonar(
    input_data: Path,
    out_file: Path,
    dump_path: Path,
    metric: str,
    well_col: str,
    timepoint_col: str,
    additional_metacol: list[str] | None,
) -> None:
    temp_out = Path(f"{out_file.stem}_INTERMEDIATE.parquet")
    meta = additional_metacol or []

    index_cols = [well_col, timepoint_col] + meta

    logging.info("preprocessing (but not detrending) data!")
    preprocess_aggregation(
        input_data=input_data,
        index_keys=index_cols,
        out_file=temp_out,
        dump_path=dump_path,
    )

    try:
        lf = pl.scan_parquet(temp_out)

        out_path = out_file.parent / out_file.stem
        clean_folder(dir=out_path)

        wells = lf.select(well_col).unique().collect()[well_col].to_list()
        timepoints = sorted(
            lf.select(timepoint_col).unique().collect()[timepoint_col].to_list()
        )

        select_data = index_cols + ["feature", metric]
        pivot_index = [well_col, "feature"] + meta

        (
            lf.select(select_data)
            .pivot(
                timepoint_col,
                on_columns=timepoints,
                index=pivot_index,
                values=metric,
            )
            .sink_csv(out_file, separator="\t")
        )

        for well in wells:
            well_wide = (
                lf.filter(pl.col(well_col) == well)
                .select(select_data)
                .pivot(
                    timepoint_col,
                    on_columns=timepoints,
                    index=pivot_index,
                    values=metric,
                )
                .drop(well_col)
            )
            well_wide.sink_csv(out_path / f"{well}_{metric}.txt", separator="\t")

    finally:
        temp_out.unlink()
