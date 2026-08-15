#!/usr/bin/env python3
from pathlib import Path
import typer
from typing import Annotated
import logging
import shutil
import polars as pl
from enum import Enum

from .preprocessing.preprocess import preprocess_to_clean_signal
from .utils import clean_folder
from .gene_expression._cli_helpers import dct_process_file
from .formatters import format_envision, process_harmony

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


app = typer.Typer(
    name="CytoClock, the Circadian Cell Painter Analyzer",
    help="This CLI runs the analysis needed for applying cell painting"
    "techniques onto circadian data",
    no_args_is_help=True,
    add_completion=False,
)

preprocess_data = typer.Typer(
    help="preprocesses cell painting data", no_args_is_help=True
)
dct_data = typer.Typer(
    help="calculates the Discrete Cosinor Transform on RAW data", no_args_is_help=True
)
formatter = typer.Typer(
    help="Formats the raw outputs from equipment into a unified file fromat",
    no_args_is_help=True,
)

app.add_typer(preprocess_data)
app.add_typer(dct_data)
app.add_typer(formatter, name="formatter")

# -- File preprocessing


@preprocess_data.command()
def preprocess(
    input_path: Annotated[
        str, typer.Argument(help="path to data directory (tab delimited file)")
    ],
    output: Annotated[
        str,
        typer.Argument(
            help="output file name. " "(MUST END IN EITHER .parquet or .csv!"
        ),
    ],
    well_columns: Annotated[
        list[str],
        typer.Option(
            "--wellname",
            "-w",
            help="specifies the well columns i.e. WellName, Row, or Col "
            "or Column. You can specify more than 1 column",
        ),
    ],
    timepoint_col: Annotated[
        str,
        typer.Option(
            "--timepoint", "-t", help="specifies the timpoint column of the dataset"
        ),
    ],
    window_size: Annotated[
        int,
        typer.Option(
            "--window",
            help="Specifies the window size (number of timepoints)"
            " of the rolling average",
        ),
    ] = 8,
    formatting: Annotated[
        bool,
        typer.Option(
            "--format",
            "-f",
            help="If set, only calculates cell summaries instead of detrending",
        ),
    ] = False,
):
    in_path = Path(input_path)
    out_path = Path(output)
    dump_dir = out_path.parent / ".temp_batches/"
    clean_folder(dump_dir)  # TODO: uncomment this
    detrend = not formatting

    if in_path.is_file():
        preprocess_to_clean_signal(
            in_path,
            well_columns,
            timepoint_col,
            out_path,
            dump_dir,
            window_size=window_size,
            detrend=detrend,
        )
    else:  # TODO: UNCOMMENT THIS BLOCK
        logging.error("Preprocess: this is something wrong with this input path!")
        shutil.rmtree(dump_dir)
        exit(1)

    # NOTE: We will figure this out later
    # if in_path.is_dir():
    #     for path in in_path.iterdir():
    #         if path.suffix in ('.tsv', '.txt'):
    #           pass
    # elif in_path.is_file():
    #     preprocess_aggregation(in_path, index_keys, out_path, dump_dir)
    # else:
    #     logging.error("Preprocess: this is something wrong with this input path!")
    #     shutil.rmtree(dump_dir)
    #     exit(1)

    shutil.rmtree(dump_dir)  # TODO: uncomment this


# -- DCT calculations


@dct_data.command()
def dct(
    data_path: Annotated[
        str,
        typer.Argument(
            help="The path to your formatted data! " "(must be a tab-delimited format)"
        ),
    ],
    platemap_path: Annotated[
        str, typer.Argument(help="Path to the platemap file. (Must in an excel file)")
    ],
    output_dir: Annotated[
        str,
        typer.Argument(
            help="the path to the output directory. (this will create a new "
            "directory for your results to be stored in)"
        ),
    ],
    well_column: Annotated[
        str,
        typer.Option(
            "--well",
            "-w",
            help="specifies the column that contains the enVision wells",
        ),
    ],
    timepoint_col: Annotated[
        str,
        typer.Option(
            "--timepoint",
            "-t",
            help="Specifies the column that contains the timepoints",
        ),
    ],
    expression_col: Annotated[
        str,
        typer.Option(
            "--expression",
            "-e",
            help="This specifies the main data column. e.g. the column the "
            "specifies the gene expression values.",
        ),
    ],
    grouping_cols: Annotated[
        list[str],
        typer.Option(
            "--groupings",
            "-g",
            help="Specifies the columns that contain meta information like "
            "MoleculeID and such",
        ),
    ],
    sampling_interval: Annotated[
        float,
        typer.Option(
            "-i",
            "--interval",
            help="Specifies the sampling interval of the data being read. "
            "i.e. every 1 hour, 2 hours etc.",
        ),
    ],
    period_min: Annotated[
        int, typer.Option("--period-min", help="The minimum period range in hours")
    ] = 16,
    period_max: Annotated[
        int, typer.Option("--period-max", help="The minimum period range in hours")
    ] = 48,
):

    period_range = (period_min, period_max)
    dct_process_file(
        data_path=data_path,
        platemap_file=platemap_path,
        out_dir=output_dir,
        well_col=well_column,
        timepoint_col=timepoint_col,
        expression_col=expression_col,
        grouping_cols=grouping_cols,
        sampling_interval=sampling_interval,
        period_range=period_range,
    )

    return None


# -- Formatting


@formatter.command()
def envision(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="Path to raw data. The input type depends on the equipment"
            "you're using but most of the time it will just require directory"
            "for the intput"
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Argument(
            help="The output file (with its path). "
            "This must end in .txt or .tsv! i.e. /path/to/file.txt"
        ),
    ],
    sampling_interval: Annotated[
        float,
        typer.Option(
            "--sampling-interval",
            "-s",
            help="The sampling interval of the experiment.",
        ),
    ],
):
    format_envision(
        in_dir=input_path,
        out_file=output_file,
        sampling_interval=sampling_interval,
    )


@formatter.command()
def harmony(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="path to single root Harmony dataset. This will be a directory"
            ", not a specific file."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to where the file will be placed."
            "This is just the directory don't name it a certain file"
        ),
    ],
    timepoint_col: Annotated[
        str,
        typer.Option(
            "--timepoint", "-t", help="Specifies the timepoint column in data."
        ),
    ],
    sampling_interval: Annotated[
        float,
        typer.Option(
            "--sampling-interval",
            "-s",
            help="The sampling interval of the experiment.",
        ),
    ],
    header_row_index: Annotated[
        int,
        typer.Option(
            "--header-index",
            help="Specified the row in the raw file where the headers are located",
        ),
    ] = 9,
    intermediate_dir: Annotated[
        Path | None,
        typer.Option(
            "--intermediate",
            "-i",
            help="Specifies a path to put the intermediate files. Specifing "
            "this option prevents the files from being deleted",
        ),
    ] = None,
    replace_dir: Annotated[
        bool,
        typer.Option(
            "--replace",
            help="This will delete the specified output directory (if it "
            "exists) and then create a new directory.",
        ),
    ] = False,
):
    process_harmony(
        in_path=input_dir,
        out_dir=output_dir,
        timepoint_col=timepoint_col,
        sampling_interval=sampling_interval,
        header_row_index=header_row_index,
        intermediate_dir=intermediate_dir,
        replace_dir=replace_dir,
    )


if __name__ == "__main__":
    app()
