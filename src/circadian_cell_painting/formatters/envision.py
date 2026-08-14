#!/usr/bin/env python3

import polars as pl
from pathlib import Path
from datetime import datetime
import io
import numpy as np
import logging
from tqdm import tqdm

_DATA_HEADERS = ("Plate", "Barcode", "Well")


def parse_envision(path: Path):
    """Returns the data only section of a EnVision output"""
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    header_idx = None

    for i, line in enumerate(lines):
        line = line.strip()
        cols = [c.strip() for c in line.split(",")]
        if (
            _DATA_HEADERS[0] in cols
            and _DATA_HEADERS[1] in cols
            and _DATA_HEADERS[2] in cols
        ):
            header_idx = i
            break

    data_lines = [lines[header_idx]]

    for line in lines[header_idx + 1 :]:
        if not line.strip():
            break
        first_col = line.split(",")[0].strip()
        if not first_col.isdigit():
            break
        data_lines.append(line)

    cleaned = "\n".join(data_lines).replace('=""', "")

    return pl.read_csv(io.StringIO(cleaned), null_values=["", "NA"])


def get_measurement_date(path: Path) -> datetime:
    """Returns the measurement date from the EnVision output"""
    with open(path, "r") as f:
        lines = f.readlines()
    header = [c.strip() for c in lines[1].split(",")]
    values = [c.strip().replace('=""', "") for c in lines[2].split(",")]

    row = dict(zip(header, values))

    return datetime.strptime(row["Measurement date"], "%m/%d/%Y %I:%M:%S %p")


def compute_timepoints(in_dir: Path, sampling_interval: float = 1.0):
    """Computes all the timepoints from all the files."""
    files = sorted(in_dir.iterdir())
    file_dates = [(f, get_measurement_date(f)) for f in files]
    file_dates.sort(key=lambda x: x[1])

    t0 = file_dates[0][1]
    timepoints = [
        (
            f,
            round((dt - t0).total_seconds() / 3600 / sampling_interval)
            * sampling_interval,
        )
        for f, dt in file_dates
    ]
    return timepoints


def format_envision(
    in_dir: Path, out_file: Path, sampling_interval: float = 1.0
) -> None:
    """
    Formats all timepoints exported from EnVision into 1 single dataset as a tsv

    Args:
        in_dir (Path): Directory that contains all timepoint data
        out_file (Path): the final file name (must end in .txt or .tsv)
        sampling_interval (float, optional): The sampling interval of the
        experiment. Defaults to 1.0.

    Returns:
        None:
    """
    dfs = []

    logging.info("Computing timepoints")
    timepoints = compute_timepoints(in_dir, sampling_interval)

    logging.info("Parsing each file")
    for f, time in timepoints:
        df = (
            parse_envision(f)
            .drop(
                [
                    "MeasTime",
                    "CalcResultI_duplicated_0",
                    "Signal_duplicated_0",
                    "MeasTime_duplicated_0",
                    "_duplicated_0",
                    "_duplicated_1",
                    "",
                    "Barcode",
                    "Flashes/Time",
                    "Plate",
                ]
            )
            .with_columns(pl.lit(time).alias("Timepoint"))
        )
        dfs.append(df)

    logging.info("Writing final file!")
    formatted_df = pl.concat(dfs)
    formatted_df.write_csv(out_file, separator="\t")

    logging.info("Calculating gaps! (if any)")

    gaps = []
    # here, sampling interval is the expected interval
    for i in range(1, len(timepoints)):
        prev_t = timepoints[i - 1][1]
        curr_t = timepoints[i][1]
        diff = curr_t - prev_t
        if diff > sampling_interval:
            missing = list(
                np.arange(prev_t + sampling_interval, curr_t, sampling_interval)
            )
            gaps.append(
                {
                    "after_file": timepoints[i - 1][0].stem,
                    "before_file": timepoints[i][0].stem,
                    "after_hour": prev_t,
                    "before_hour": curr_t,
                    "missing_hours": missing,
                    "n_missing": len(missing),
                }
            )

    if gaps:
        logging.info(f"Found {len(gaps)} gap(s):\n")
        for g in gaps:
            logging.info(
                f"  between file {g['after_file']} ({g['after_hour']}h) "
                f"and {g['before_file']} ({g['before_hour']}h) "
                f"— missing timepoints: {g['missing_hours']}"
            )
    else:
        print("No gaps found!")

    return None
