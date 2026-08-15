#!/usr/bin/env python3

import polars as pl
from pathlib import Path
import unicodedata
import logging
from typing import Generator
from ..utils import clean_folder
import shutil

_DROP_COLUMNS = [
    "Row",
    "Column",
    "Field",
    "Object No",
    "X",
    "Y",
    "Bounding Box",
    "Position X [um]",
    "Position Y [um]",
    "Compound",
    "Concentration",
    "Cell Type",
    "Cell Count",
]


def clean_column(name):
    replacements = {
        "µ": "u",
        "μ": "u",
        "²": "2",
        "³": "3",
    }

    for char, replacement in replacements.items():
        name = name.replace(char, replacement)

    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.strip()
    name = name.strip("_")

    return name


def parse_harmony_dataset(root: Path) -> Generator[Path, str, str]:
    """
    Returns ALL the cell data files in a given harmony data set

    Args:
        root (Path): The root of the harmony dataset

    Yields:
        Generator[Path, str, str]: Returns the file path, the dataset name,
        and the cell object type

    **NOTE:** The harmony file structure is:

        raw name/
        ├─ Evaluation/
        │  ├─ PlateResult.txt
        │  ├─ Objects_Population - NoBorderObj.txt (this is analysis file)
        │  ├─ .ExportDone_CellListsToTxt
        ├─ indexfile.txt
    """

    for f in root.glob("*/*"):
        if f.is_file() and not f.name.startswith(".") and f.name != "PlateResults.txt":
            dataset_name = f.relative_to(root).parts[0]
            object_type = f.stem.replace("Objects_Population - ", "")
            yield f, dataset_name, object_type


def format_raw_harmony(
    in_path: Path, out_dir: Path, header_row_index: int = 9
) -> dict[str, dict[str, Path]]:
    """
    formats the raw harmony data into a pute tab delimited file
    Each object type gets its own output file since cell counts
    differ between object types and cannot be mergeed horizontally.

    Args:
        in_path (Path): the root harmomy dataset
        out_dir (Path): outputs each cell object into the given directory
        header_row_index (int, optional): The line index for when the headers
        begin. Defaults to 9.

    Returns:
        dict[str, dict[str, Path]]: returns the dataset's obeject and Path
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, dict[str, Path]] = {}

    for f, dataset_name, object_type in parse_harmony_dataset(in_path):
        logging.info(f"Formatting: {dataset_name} - {object_type}")

        out_path = out_dir / f"{dataset_name}._.{object_type}.formatted.txt"

        if out_path.exists():
            logging.info(f"The following will be overwritten: {out_path}")

        with (
            open(f, "r", encoding="utf-8") as in_file,
            open(out_path, "w", encoding="utf-8") as out_file,
        ):
            for i, line in enumerate(in_file):
                if i < header_row_index:
                    continue
                elif i == header_row_index:
                    cleaned = "\t".join(
                        clean_column(col) for col in line.strip().split("\t")
                    )
                    out_file.write(cleaned + "\n")
                else:
                    out_file.write(line.rstrip("\t\n") + "\n")

        output_paths.setdefault(dataset_name, {})[object_type] = out_path

    return output_paths


def clean_harmony_data(
    in_path: Path,
    out_path: Path,
    timepoint_col: str,
    sampling_interval: float,
    drop_cols: list[str],
):
    """
    With a given formatted raw harmony dataset, clean it so it removes all
    unnecessary columns

    Args:
        in_path (Path): The tab delimited file path
        out_path (Path): The tab delimited output path
        timepoint_col (str): The timepoint column of the data
        sampling_interval (float): The sampling interval of the experiment
        drop_cols (list[str]): Columns to drop
    """
    logging.info(f"Scrubbing: {in_path}")
    lf = (
        pl.scan_csv(
            in_path, separator="\t", null_values=["", "NA", "NaN", "nan", "NULL"]
        )
        .with_columns(
            (
                pl.col("Row").map_elements(
                    lambda x: chr(64 + x), return_dtype=pl.String
                )
                + pl.col("Column").cast(pl.String).str.zfill(2)
            ).alias("WellName")
        )
        .select(["WellName", pl.all().exclude("WellName")])
        .drop(drop_cols, strict=False)
        .with_columns(
            ((pl.col(timepoint_col) - 1) * sampling_interval).cast(pl.Float64)
        )
    )

    lf.sink_csv(out_path, separator="\t")


def process_harmony(
    in_path: Path,
    out_dir: Path,
    timepoint_col: str,
    sampling_interval: float,
    header_row_index: int = 9,
    intermediate_dir: Path | None = None,
    replace_dir: bool = False,
):
    replace_dir = not replace_dir
    clean_folder(out_dir, create_only=replace_dir)

    # handle intermediate files

    keep_intermediate = intermediate_dir is not None
    if intermediate_dir is None:
        intermediate_dir = out_dir / "_intermediate"

    # clean raw tab-delimited files
    logging.info("Formatting raw Harmony data")
    formatted_paths = format_raw_harmony(
        in_path=in_path, out_dir=intermediate_dir, header_row_index=header_row_index
    )

    # clean and format to be analysis - ready
    logging.info("Prepare data for analysis format")
    final_paths = {}

    for dataset_name, object_types in formatted_paths.items():
        final_paths[dataset_name] = {}

        for object_type, formatted_path in object_types.items():
            final_path = (
                out_dir
                / f"{formatted_path.stem.split(".formatted")[0]}.analysis_ready.tsv"
            )

            clean_harmony_data(
                in_path=formatted_path,
                out_path=final_path,
                timepoint_col=timepoint_col,
                sampling_interval=sampling_interval,
                drop_cols=_DROP_COLUMNS,
            )

            final_paths[dataset_name][object_type] = final_path

    # cleanup
    if not keep_intermediate:
        shutil.rmtree(intermediate_dir)
        logging.info(f"Removed intermediate files: {intermediate_dir}")
    else:
        logging.info(f"Kept intermediate files: {intermediate_dir}")

    logging.info(f"pipeline complete - {len(final_paths)} object type(s) processed")
    return final_paths
