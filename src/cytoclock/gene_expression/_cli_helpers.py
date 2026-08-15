from .dct import calculate_DCT
import polars as pl
from pathlib import Path
import logging


def dct_process_file(
    data_path: str,
    platemap_file: str,
    out_dir: str,
    well_col: str,
    timepoint_col: str,
    expression_col: str,
    grouping_cols: list[str],
    sampling_interval: int,
    period_range: tuple[int],
):
    data_df = pl.read_csv(data_path, separator="\t")

    # logging.info(f"{pl.read_excel(platemap_file, engine="openpyxl").schema}")

    platemap = (
        pl.read_excel(platemap_file, engine="openpyxl")
        .select([well_col] + grouping_cols)
        .unique(subset=[well_col])
        .with_columns([pl.col(c).str.strip_chars() for c in grouping_cols])
    )

    logging.info("Merging platemap to data!")
    data_df = data_df.join(platemap, on=well_col)

    logging.info("calculating DCT!")
    calculate_DCT(
        data_df=data_df,
        well_col=well_col,
        timepoint_col=timepoint_col,
        expression_col=expression_col,
        grouping_cols=grouping_cols,
        out_directory=Path(out_dir),
        sampling_interval=sampling_interval,
        period_range=period_range,
    )
