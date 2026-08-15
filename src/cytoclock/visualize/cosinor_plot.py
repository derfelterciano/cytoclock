#!/usr/bin/env python3

from matplotlib.backends.backend_pdf import PdfPages
from .base import VisualizerBase
from matplotlib import pyplot as plt
from ..utils import OscillationResults
from pathlib import Path
import polars as pl
import numpy as np
from matplotlib.axes import Axes
import seaborn as sns
from tqdm import tqdm


class CosinorVisualize(VisualizerBase):
    """
    Plots the result of the cosinor calculations
    """

    def plot_fit(self, ax: Axes, row: dict) -> None:
        """
        Plots/overlays the fitted curves from the analysis results
        No recalculation needed
        """
        fitted = np.array(row["fitted"][0])
        n = len(fitted)
        hours = (
            self.results.scan_data()
            .filter(
                (pl.col(self.results.well_col) == row[self.results.well_col][0])
                & (pl.col(self.results.feature_col) == row[self.results.feature_col][0])
            )
            .select(self.results.timepoint_col)
            .sort(self.results.timepoint_col)
            .collect()[self.results.timepoint_col]
            .to_numpy()
        )
        hours = self.results.params.get("time_offset", 0) + (
            hours - self.results.params.get("start_index", 0)
        ) * self.results.params.get("time_interval", 1)

        ax.plot(
            hours,
            fitted,
            "-",
            color="crimson",
            linewidth=2,
            label=(
                f"cosinor fit    "
                f"amp={row['amplitude'][0]:.4f}    "
                f"phase={np.degrees(row['phase'][0]):.1f} degrees"
            ),
        )

        return

    def plot_summary(self, ax, significant: pl.DataFrame) -> None:
        """
        Heatmap of amplitude, phase, p_adjusted for all significant feats
        """
        heat_data = (
            significant.select(
                [self.results.feature_col, "amplitude", "phase", "p_adjusted"]
            )
            .to_pandas()
            .set_index(self.results.feature_col)
        )

        sns.heatmap(heat_data, ax=ax, cmap="viridis", annot=True, fmt=".3f")
        ax.set_title(
            f"Significant Features Summary - metric: {self.results.stat} | "
            f"(p-vals <={self.results.adj_alpha} (adjusted)"
        )
        return

    def view(self, out_dir: Path, alpha: float | None = None) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        significants = self.get_significant(alpha)
        wells = significants[self.results.well_col].unique().to_list()

        for well in tqdm(wells, desc="well_processing", total=len(wells)):
            well_sigs = significants.filter(pl.col(self.results.well_col) == well)

            if well_sigs.is_empty():
                continue

            features = well_sigs[self.results.feature_col].to_list()
            out_path = out_dir / f"{well}_{self.results.stat}.pdf"

            with PdfPages(out_path) as pdf:
                # page 1: summary heatmap

                fig, ax = plt.subplots(figsize=(11, max(4, len(features) * 0.3)))
                self.plot_summary(ax, well_sigs)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                for feat in tqdm(
                    features,
                    desc=f"progress of features for {well}",
                    total=len(features),
                ):
                    row = well_sigs.filter(
                        pl.col(self.results.feature_col) == feat
                    ).to_dict(as_series=False)

                    fig, ax = plt.subplots(figsize=(11, 4))

                    # raw time series first
                    self.plot_timeseries(ax, well, feat)

                    # add the fit
                    self.plot_fit(ax, row)

                    ax.set_title(
                        f"{feat}\n"
                        f'p_adj={row["p_adjusted"][0]:.4e}  '
                        f'amp={row["amplitude"][0]:.4f}  '
                        f'phase={np.degrees(row["phase"][0]):.1f}(deg)  '
                        f'baseline={row["baseline"][0]:.4f}'
                    )

                    ax.legend(fontsize=8)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                d = pdf.infodict()
                d["Title"] = f"Cosinor Results — {well} | {self.results.stat}"
                d["Subject"] = "Circadian oscillation via cosinor fitting"
