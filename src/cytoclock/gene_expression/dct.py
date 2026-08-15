#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from scipy.fftpack import dct
import polars as pl
from pathlib import Path
from ..utils import clean_folder
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from tqdm import tqdm
from ..utils import add_period_marks
import logging


def dct_period_detection(
    values: npt.NDArray[np.float64], sampling_interval: float, norm: str = "ortho"
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Runs a discrete cosinor. This method was proposed by Devons Mo of the Kimmey lab

    Args:
        values (npt.NDArray[np.float64]): Timeseries measurement (not
        timepoints)
        sampling_interval (float): the amount of reads per hour

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]: returns the periods,
        frequency, and amplitudes
    """
    transformed = dct(values, norm=norm)

    # amplitude
    amplitude = np.abs(transformed)

    N = len(values)
    k = np.arange(N)

    # sampling frequency
    fs = 1 / sampling_interval

    f = fs * k / (2 * N)

    # convert to periods

    with np.errstate(divide="ignore"):
        periods = np.where(f > 0, 1 / f, np.inf)

    return periods, f, amplitude


def calculate_DCT(
    data_df: pl.DataFrame,
    well_col: str,
    timepoint_col: str,
    expression_col: str,
    grouping_cols: list[str],
    out_directory: Path,
    sampling_interval: float = 1.0,
    period_range: tuple[int, int] = (16, 48),
) -> None:
    clean_folder(out_directory)
    logging.info("Cleaned directory!")

    best_periods = []
    well_periods = []

    pdf_out = out_directory / "dct_plots.pdf"
    with PdfPages(pdf_out) as pdf:
        for group_vals, group_df in tqdm(
            data_df.group_by(grouping_cols, maintain_order=True), desc="groups"
        ):

            # group_vals is a tuple when grouping_cols has >1 column,
            # or a single value when grouping_cols has exactly 1 column
            if len(grouping_cols) == 1:
                group_dict = {grouping_cols[0]: group_vals[0]}
            else:
                group_dict = dict(zip(grouping_cols, group_vals))

            group_label = " | ".join(str(v) for v in group_dict.values())

            group_wells = group_df[well_col].unique().to_list()

            detected_periods = []
            all_amplitudes = []
            all_periods_mask = None

            logging.info(f"Calculating wells for: {group_vals}")
            for well in group_wells:
                well_data = (
                    group_df.filter(pl.col(well_col) == well)
                    .sort(timepoint_col)[expression_col]
                    .to_numpy()
                )

                period, freq, amplitude = dct_period_detection(
                    well_data, sampling_interval=sampling_interval
                )
                mask = (period >= period_range[0]) & (period <= period_range[1])

                all_amplitudes.append(amplitude[mask])
                all_periods_mask = period[mask]

                # detected period
                peak_idx = np.argmax(amplitude[mask])
                detected_period = period[mask][peak_idx]
                detected_periods.append(detected_period)

                well_periods.append(
                    {
                        "Well": well,
                        "group_label": group_label,
                        "detected_period": detected_period,
                    }
                )

            all_amplitudes = np.array(all_amplitudes)
            mean_amplitudes = np.mean(all_amplitudes, axis=0)
            std_amplitudes = np.std(all_amplitudes, axis=0)

            mean_period = np.mean(detected_periods)
            std_period = np.std(detected_periods)
            median_period = np.median(detected_periods)

            best_periods.append(
                {
                    **group_dict,
                    "group_label": group_label,
                    "mean_period": mean_period,
                    "median_period": median_period,
                    "std_period": std_period,
                    "n_replicates": len(group_wells),
                    "periods": detected_periods,
                }
            )

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(all_periods_mask, mean_amplitudes, color="blue")
            ax.fill_between(
                all_periods_mask,
                mean_amplitudes - std_amplitudes,
                mean_amplitudes + std_amplitudes,
                alpha=0.2,
                color="blue",
            )
            ax.axvline(
                mean_period,
                color="red",
                linewidth=2,
                label=f"Mean period: {mean_period:.1f} ± {std_period:.1f}h",
            )
            ax.axvline(
                median_period,
                color="orange",
                linewidth=2,
                linestyle="--",
                label=f"median period: {median_period:.1f}h",
                alpha=0.5,
            )
            ax.axvspan(
                mean_period - std_period,
                mean_period + std_period,
                alpha=0.1,
                color="red",
            )
            ax.set_xlim(period_range[0], period_range[1])
            ax.set_xlabel("Period (hours)")
            ax.set_ylabel("Amplitude")
            ax.set_title(
                f"{group_label}\n"
                f"Detected period: {mean_period:.1f} ± {std_period:.1f}h\n"
                f"detected median: {median_period:.1f}h"
            )
            ax.legend(fontsize=7)
            add_period_marks(ax, period_range[0], period_range[1], interval=1)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    logging.info("Figure generated!")

    logging.info("Writing results to csv!")

    # write outputs
    per_well_periods = pl.DataFrame(well_periods)
    per_well_periods.write_csv(out_directory / "per_well_period_results.csv")

    best_periods_df = pl.DataFrame(best_periods)
    (best_periods_df.drop("periods").write_csv(out_directory / "period_results.csv"))
