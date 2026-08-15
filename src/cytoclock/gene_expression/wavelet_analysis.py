#!/usr/bin/env python3

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from pyboat import WAnalyzer
from scipy.signal import savgol_filter
from tqdm import tqdm
from ..utils import add_period_marks


@dataclass
class WaveletResults:
    """
    TODO: Add documentation
    """

    ridge_df: pl.DataFrame
    summary_df: pl.DataFrame
    periods: np.ndarray
    fourier_df: pl.DataFrame

    def to_pickle(self, path: Path) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, path: Path) -> "WaveResults":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


def merge_data_platemap(
    detrended_data: Path,
    plate_map: Path,
    on_well_col: str,
    gene_expression_feature: str,
    feature_col: str = "feature",
):
    """
    Merges detrended gene expression data with platemap metadata.
    Assumes BOTH the detrended data and platemap use the same well_col name.
    Rename beforehand if they differ, e.g.:
        pm = pl.read_excel(...).rename({"384_Well": "Well"})

    args:
        detrended_data (Path): path to the detrended parquet data
        plate_map (Path): path to the platemap file (.xlsx)
        well_col (str): well column name — must match in BOTH files
        gene_expression_feature (str): the specific feature to filter to
        feature_col (str): feature column name in detrended_data (default
        "feature")

    returns:
        pl.DataFrame: merged, feature-filtered dataset with platemap metadata
        joined in
    """
    pm = pl.read_excel(plate_map, engine="openpyxl")

    merged = (
        pl.scan_parquet(detrended_data)
        .filter(pl.col(feature_col) == gene_expression_feature)
        .collect()
        .join(pm, on=on_well_col, how="left")
    )

    return merged


class WaveletAnalyzer:
    """
    Runs wavelet-based circadian analysis (through pyboat) across all wells in a
    dataset, tracking period, amplitude, and phase drift over time.

    Unlike osc_detection models (CosinorDetection, eJTK_CYCLE), this is
    NOT built for high feature-count throughput — intended for gene
    expression pipelines with few features per well.
    """

    def __init__(
        self,
        periods: tuple[int, int],
        sampling_interval: float,
        n_periods: int = 200,
        ridge_thresh: float = 0,
        win_len: int = 13,
        polyorder: int = 3,
        tick_interval: int = 12,
    ):
        self.period_range = periods
        self.sampling_interval = sampling_interval
        self.n_periods = n_periods
        self.ridge_thresh = ridge_thresh
        self.win_len = win_len
        self.polyorder = polyorder
        self.tick_interval = tick_interval
        self.period_axis = np.linspace(periods[0], periods[1], n_periods)

    def analyze(
        self,
        dataset: pl.DataFrame,
        name_col: str,
        well_list: list[str],
        well_col: str,
        stat_val: str,
        timepoint_col: str,
    ) -> WaveletResults:
        """
        Runs wavelet analysis across all wells in the dataset.

        args:
            dataset (pl.DataFrame): detrended signal data (must include name_col)
            name_col (str): column with group label
            well_list (list[str]): wells to process
            well_col (str): column identifying the well
            stat_val (str): column with signal values
            timepoint_col (str): column with timepoints (in hours)

        returns:
            WaveResults
        """

        ridge_results = []
        fourier_results = []
        summary_rows = []

        for well in tqdm(well_list, desc="Wells"):
            well_df = dataset.filter(pl.col(well_col) == well).sort(timepoint_col)
            signal = well_df[stat_val].to_numpy()
            tpoints = well_df[timepoint_col].to_numpy()
            group = str(
                well_df[name_col].unique().item()
            )  # TODO: Change this to take multiple columns

            wan = WAnalyzer(
                periods=self.period_axis, dt=self.sampling_interval, time_unit_label="h"
            )
            wan.compute_spectrum(signal, do_plot=False)

            ridge = wan.get_maxRidge(power_thresh=self.ridge_thresh)
            fourier_power = wan.get_averaged_spectrum()

            # ridge summaries
            ridge["time"] = tpoints
            ridge[well_col] = well
            ridge[name_col] = group
            ridge_results.extend(ridge.to_dict(orient="records"))

            for p, pw in zip(self.period_axis, fourier_power):
                fourier_results.append(
                    {well_col: well, name_col: group, "period": p, "power": pw}
                )

            mid_idx = len(ridge) // 2
            summary_rows.append(
                {
                    well_col: well,
                    name_col: group,
                    "mean_period": ridge["periods"].mean(),
                    "std_period": ridge["periods"].std(),
                    "median_period": ridge["periods"].median(),
                    "mean_amplitude": ridge["amplitude"].mean(),
                    "std_amplitude": ridge["amplitude"].std(),
                    "median_amplitude": ridge["amplitude"].median(),
                    "mid_phase": ridge["phase"].iloc[mid_idx],
                    "peak_fourier_period": self.period_axis[np.argmax(fourier_power)],
                }
            )

        return WaveletResults(
            ridge_df=(
                pl.DataFrame(ridge_results).with_columns(
                    (pl.col("phase") / (2 * np.pi) * pl.col("periods")).alias(
                        "phase_hours"
                    )
                )
            ),
            fourier_df=pl.DataFrame(fourier_results),
            periods=self.period_axis,
            summary_df=pl.DataFrame(summary_rows),
        )

    def visualize_all(
        self,
        wavelet_results: WaveletResults,
        dataset: pl.DataFrame,
        well_list: list[str],
        well_col: str,
        name_col: str,
        stat_val: str,
        timepoint_col: str,
        output_file: Path,
    ) -> None:
        """
        Outputs a pdf file of each well visualized
        """

        with PdfPages(output_file) as pdf:
            for fig in self.visualize(
                wavelet_results=wavelet_results,
                dataset=dataset,
                well_list=well_list,
                well_col=well_col,
                name_col=name_col,
                stat_val=stat_val,
                timepoint_col=timepoint_col,
            ):
                pdf.savefig(fig)
                plt.close(fig)

    def visualize(
        self,
        wavelet_results: WaveletResults,
        dataset: pl.DataFrame,
        well_list: list[str],
        well_col: str,
        name_col: str,
        stat_val: str,
        timepoint_col: str,
    ) -> Generator[plt.Figure, None, None]:
        """
        Yields one 3-panel figure (signal, wavelet spectrum, fourier spectrum)
        per well. Caller is responsible for saving/closing each figure.
        """
        ridge_df = wavelet_results.ridge_df.to_pandas()
        fourier_df = wavelet_results.fourier_df.to_pandas()
        periods = wavelet_results.periods

        for well in tqdm(well_list, desc="well plots"):
            well_signal = (
                dataset.filter(pl.col(well_col) == well)
                .sort(timepoint_col)
                .select([timepoint_col, stat_val])
                .to_pandas()
            )

            tpoints = well_signal[timepoint_col].to_numpy()
            signal = well_signal[stat_val].to_numpy()
            smoothed = savgol_filter(
                signal, window_length=self.win_len, polyorder=self.polyorder
            )

            well_signal[stat_val] = smoothed
            well_ridge = ridge_df[ridge_df[well_col] == well]
            well_fourier = fourier_df[fourier_df[well_col] == well]
            group = well_ridge[name_col].iloc[0]

            fig, axs = plt.subplots(3, 1, figsize=(14, 12))

            # panel 1: detrended signal (smoothed for display only)
            sns.lineplot(
                data=well_signal,
                x=timepoint_col,
                y=stat_val,
                color="royalblue",
                linewidth=1.5,
                ax=axs[0],
            )
            axs[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
            axs[0].set_xlabel("Time (hours)")
            axs[0].set_ylabel("Expression")
            axs[0].set_title("Detrended Signal")
            axs[0].set_xlim(tpoints[0], tpoints[-1])
            add_period_marks(axs[0], tpoints[0], tpoints[-1], self.tick_interval)

            # panel 2: wavelet spectrum
            wan = WAnalyzer(
                periods=periods, dt=(tpoints[1] - tpoints[0]), time_unit_label="h"
            )
            wan.compute_spectrum(signal, do_plot=False)
            extent = [tpoints[0], tpoints[-1], periods[-1], periods[0]]
            axs[1].imshow(
                np.abs(wan.transform),
                aspect="auto",
                extent=extent,
                cmap="viridis",
                origin="upper",
            )
            axs[1].plot(
                well_ridge["time"],
                well_ridge["periods"],
                color="cyan",
                linewidth=2,
                label="ridge",
            )
            axs[1].set_xlabel("Time (hours)")
            axs[1].set_ylabel("Period (hours)")
            axs[1].set_title("Wavelet Power Spectrum")
            axs[1].legend(fontsize=8)
            axs[1].set_xlim(tpoints[0], tpoints[-1])
            axs[1].set_ylim(periods[-1], periods[0])
            add_period_marks(axs[1], tpoints[0], tpoints[-1], self.tick_interval)

            # panel 3: fourier spectrum
            peak_period = well_fourier.loc[well_fourier["power"].idxmax(), "period"]
            axs[2].plot(
                well_fourier["period"],
                well_fourier["power"],
                color="darkred",
                linewidth=1.5,
            )
            axs[2].axvline(
                peak_period,
                color="red",
                linewidth=1.5,
                linestyle="--",
                label=f"Peak: {peak_period:.1f}h",
            )
            axs[2].set_xlabel("Period (hours)")
            axs[2].set_ylabel("Power")
            axs[2].set_title("Fourier Spectrum (time-averaged)")
            axs[2].legend(fontsize=8)
            axs[2].set_xlim(periods[0], periods[-1])

            fig.suptitle(f"Well: {well} - {group}", fontsize=11)
            fig.tight_layout()

            yield fig

    def interactive_ridges(
        self,
        wavelet_res: WaveletResults,
        name_col: str,
        timepoint_col: str,
        metrics: list[str],
        out_file: Path,
    ) -> None:
        """
        Builds an interactive Plotly HTML with mean ± SD ridge metrics
        for all groups, toggleable via legend clicks.
        """

        ridge_df = wavelet_res.ridge_df

        replicate_agg = (
            ridge_df.with_columns(pl.col(name_col).cast(pl.String).fill_null("nan"))
            .group_by([name_col, timepoint_col])
            .agg(
                [
                    *[pl.col(m).mean().alias(f"{m}_mean") for m in metrics],
                    *[pl.col(m).std().alias(f"{m}_std") for m in metrics],
                ]
            )
            .sort([name_col, timepoint_col])
            .to_pandas()
        )

        groups = sorted(replicate_agg[name_col].unique())
        palette = px.colors.qualitative.Plotly
        colormap = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            shared_xaxes=False,
            subplot_titles=[m.capitalize() for m in metrics],
            vertical_spacing=0.05,
        )

        for group in groups:
            color = colormap[group]
            color_rgba = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}"

            gdf = replicate_agg[replicate_agg[name_col] == group]

            for row_idx, metric in enumerate(metrics, start=1):
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"
                show_legend = row_idx == 1

                fig.add_trace(
                    go.Scatter(
                        x=pd.concat(
                            [gdf[timepoint_col], gdf[timepoint_col].iloc[::-1]]
                        ),
                        y=pd.concat(
                            [
                                gdf[mean_col] + gdf[std_col],
                                (gdf[mean_col] - gdf[std_col]).iloc[::-1],
                            ]
                        ),
                        fill="toself",
                        fillcolor=f"{color_rgba}, 0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        showlegend=False,
                        legendgroup=group,
                        name=group,
                        hoverinfo="skip",
                    ),
                    row=row_idx,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=gdf[timepoint_col],
                        y=gdf[mean_col],
                        mode="lines",
                        name=group,
                        legendgroup=group,
                        showlegend=show_legend,
                        line=dict(color=color, width=2),
                        hovertemplate=f"<b>{group}</b><br>Time: %{{x:.1f}}h<br>{metric}: %{{y:.3f}}<extra></extra>",
                    ),
                    row=row_idx,
                    col=1,
                )

        fig.update_layout(
            height=2000,
            title="Wavelet Ridge Analysis — All Groups",
            hovermode="x unified",
            legend=dict(groupclick="toggleitem", x=1.02, y=1),
        )

        fig.update_xaxes(title_text="Time (hours)", row=len(metrics), col=1)
        for row_idx, metric in enumerate(metrics, start=1):
            fig.update_yaxes(title_text=metric.capitalize(), row=row_idx, col=1)

        fig.update_traces(visible="legendonly")
        fig.write_html(
            out_file,
            include_plotlyjs=True,
            post_script="""
            var myPlot = document.getElementsByClassName('plotly-graph-div')[0];
            myPlot.on('plotly_legendclick', function(data) {
                var clickedGroup = data.data[data.curveNumber].legendgroup;
                var indices = [];
                data.data.forEach(function(trace, i) {
                    if (trace.legendgroup === clickedGroup) {
                        indices.push(i);
                    }
                });
                var currentVis = data.data[data.curveNumber].visible;
                var newVis = (currentVis === 'legendonly' || currentVis === false) ? true : 'legendonly';
                Plotly.restyle(myPlot, {'visible': indices.map(function() { return newVis; })}, indices);
                return false;
            });
            """,
        )
        fig.show()
