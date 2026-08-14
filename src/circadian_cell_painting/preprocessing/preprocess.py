#!/usr/env/bin python3

from .aggregate import batch_summaries, merge_batches
import logging
from .denoise_detrend import denoise_detrend
import polars as pl
from pathlib import Path


def preprocess_aggregation(
    input_data: Path, index_keys: list[str], out_file: Path, dump_path: Path
) -> None:
    paths = batch_summaries(input_data, index_keys, dump_path)
    paths = [str(p) for p in paths]
    lf = merge_batches(batch_files=paths, index_keys=index_keys)

    if out_file.suffix == ".parquet":
        lf.sink_parquet(out_file)
    elif out_file.suffix == ".csv":
        lf.sink_csv(out_file)
    else:
        logging.error("preprocess_aggregation: invalid file extension")
        exit(1)

    return None


def preprocess_to_clean_signal(
    input_data: Path,
    well_cols: str | list[str],
    timepoint_col: str,
    out_file: Path,
    dump_path: Path,
    detrend: bool = True,
    window_size: int = 8,
):
    well_col = well_cols if isinstance(well_cols, list) else [well_cols]
    index_keys = well_col + [timepoint_col]

    paths = batch_summaries(input_data, index_keys, dump_path)
    # paths = Path(
    #     "/home/derfelt/PartchLabFiles/Circadian Rhythms/Live"
    #     " Study/preprocessed_data/.temp_batches"
    # ).iterdir()
    paths = [str(p) for p in paths]

    # lazily merge all the batch files together
    lf = merge_batches(paths, index_keys)

    # clean_lf = lf
    if detrend:
        clean_lf = denoise_detrend(
            lf,
            feature_col="feature",
            well_col=well_col,
            timepoint_col=timepoint_col,
            window_size=window_size,
        )
    else:
        clean_lf = lf

    if out_file.suffix == ".parquet":
        clean_lf.sink_parquet(out_file)
    elif out_file.suffix == ".csv":
        clean_lf.sink_csv(out_file)
    else:
        logging.error("preprocess_to_clean_signal: invalid file extension")
        exit(1)
