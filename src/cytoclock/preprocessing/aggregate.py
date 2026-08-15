#!/usr/bin/env python3
"""
Process and aggregat all the cell data
"""

import polars as pl
from pathlib import Path
from polars.dataframe.frame import DataFrame
from tqdm import tqdm
import logging
from glob import glob

from ..utils import clean_folder

_QUANTILE_METHOD = "linear"
_BATCH_SIZE = 250


def _process_batch(
    cols: list[str],
    dump_dir: Path,
    data_path: Path,
    curr_iter: int,
    index_keys: list[str],
) -> Path:
    """
    We will process a batch of features and dump it as a parquet file
    """

    out = dump_dir / f"batch{curr_iter}.parquet"

    agg_exp = []

    for col in cols:
        agg_exp.append(
            pl.struct(
                [
                    pl.col(col).mean().alias(f"mean"),
                    pl.col(col).median().alias(f"median"),
                    pl.col(col).std().alias(f"stddev"),
                    pl.col(col)
                    .quantile(0.00, interpolation=_QUANTILE_METHOD)
                    .alias(f"q0"),
                    pl.col(col)
                    .quantile(0.10, interpolation=_QUANTILE_METHOD)
                    .alias(f"q10"),
                    pl.col(col)
                    .quantile(0.25, interpolation=_QUANTILE_METHOD)
                    .alias(f"q25"),
                    pl.col(col)
                    .quantile(0.50, interpolation=_QUANTILE_METHOD)
                    .alias(f"q50"),
                    pl.col(col)
                    .quantile(0.75, interpolation=_QUANTILE_METHOD)
                    .alias(f"q75"),
                    pl.col(col)
                    .quantile(0.90, interpolation=_QUANTILE_METHOD)
                    .alias(f"q90"),
                    pl.col(col)
                    .quantile(1.00, interpolation=_QUANTILE_METHOD)
                    .alias(f"q100"),
                ]
            ).alias(col)
        )

    (
        pl.scan_csv(
            data_path, separator="\t", null_values=["NaN", "nan", "NA", "N/A", ""]
        )
        .select(index_keys + cols)
        .group_by(index_keys)
        .agg(agg_exp)
        .unpivot(index=index_keys, variable_name="feature", value_name="stats")
        .with_columns(pl.col("stats").struct.unnest())
        .drop("stats")
        .sort(index_keys + ["feature"])
        .sink_parquet(out)
    )

    return out


def batch_summaries(
    data_path: Path, index_keys: list[str], dump_outs: Path
) -> list[Path]:
    """
    calculate the collapsed mertics for the data and dump in parquet batches

    args:
        - data_path (Path): input file
        - index_keys (list[str]): the list of columns that signify
        well location, timepoint
        - dump_outs (Path): the file dir to dump batches in

    return:
        list[Path]: a list of the location of all the batch files
    """

    all_cols = pl.scan_csv(data_path, separator="\t").collect_schema().names()
    feats = [c for c in all_cols if c not in index_keys]
    batches = [feats[i : i + _BATCH_SIZE] for i in range(0, len(feats), _BATCH_SIZE)]
    logging.info(f"{len(feats)} features -> {len(batches)} batches of {_BATCH_SIZE}")

    clean_folder(dump_outs)

    batch_files = []
    for i, batch in tqdm(
        enumerate(batches), total=len(batches), desc="Batching progress"
    ):
        file = _process_batch(
            cols=batch,
            dump_dir=dump_outs,
            curr_iter=i,
            index_keys=index_keys,
            data_path=data_path,
        )
        batch_files.append(file)

    return batch_files


def merge_batches(batch_files: list[str], index_keys: list[str]) -> pl.LazyFrame:
    """
    Merge all the batch parquet files together

    args:
        - batch_files (list[str]): the list of file locations of the batches
        - out_file (Path): a file path. files must end in .parquet OR .csv
    """

    files = sorted(batch_files)

    dfs: list[DataFrame] = []
    for f in files:
        df = pl.read_parquet(f).with_columns(
            [
                pl.col("mean").cast(pl.Float64),
                pl.col("median").cast(pl.Float64),
                pl.col("stddev").cast(pl.Float64),
                pl.col("q0").cast(pl.Float64),
                pl.col("q10").cast(pl.Float64),
                pl.col("q25").cast(pl.Float64),
                pl.col("q50").cast(pl.Float64),
                pl.col("q75").cast(pl.Float64),
                pl.col("q90").cast(pl.Float64),
                pl.col("q100").cast(pl.Float64),
            ]
        )
        dfs.append(df)

    final = pl.concat(dfs).sort(index_keys + ["feature"]).lazy()

    return final
