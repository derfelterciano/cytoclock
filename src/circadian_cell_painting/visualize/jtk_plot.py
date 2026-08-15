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


class JTKVisualize(VisualizerBase):
    """
    Plots the results of the JTK or eJTK results
    """

    def plot_fit(self, ax: Axes, row: dict) -> None:
        """
        Plots/overlays the fitted curves from the analysis results
        No recalculation needed
        """
        jtk_fitted = np.array(row["jtk_fitted"][0])
        lsq_fitted = np.array(row["lsq_fitted"][0])
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
            jtk_fitted,
            "--",
            color="darkorange",
            linewidth=2,
            label="JTK raw estimate",
            zorder=8,
        )

        ax.plot(
            hours,
            lsq_fitted,
            "-",
            color="crimson",
            linewidth=2,
            label=f"least-squares refit (from JKT's period)\n"
            f"baseline={row['lsq_baseline'][0]:.2f}\n"
            f"amp={row['lsq_amplitude'][0]:.2f}\n"
            f"phase={row['lsq_phase_hours'][0]:.1f}h\n"
            f"cycles={row['lsq_n_cycles'][0]:.2f}\n",
            zorder=9,
        )

        return

    def plot_summary(self, ax: Axes, significant):
        """
        Summary plot for JTK results — distribution of detected periods
        among significant features, with R^2 as a secondary check.
        """

        if significant.is_empty():
            ax.text(
                0.5,
                0.5,
                "No significant features found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        periods = significant["period"].to_numpy()

        ax.hist(periods, bins=20, color="royalblue", edgecolor="black", alpha=0.8)
        ax.axvline(
            np.median(periods),
            color="crimson",
            linewidth=2,
            linestyle="--",
            label=f"median: {np.median(periods):.1f}h",
        )
        ax.set_xlabel("Detected period (hours)")
        ax.set_ylabel("Number of significant features")
        ax.set_title(
            f"Period distribution — significant features (n={len(significant)})"
        )
        ax.legend(fontsize=9)

        return

    def view(self, out_dir, alpha=None, filter_col: str = "p_adjusted"):
        out_dir.mkdir(parents=True, exist_ok=True)
        significants = self.get_significant(alpha, filter_col=filter_col)
        wells = significants[self.results.well_col].unique().to_list()

        for well in tqdm(wells, desc="well processing", total=len(wells)):
            well_sigs = significants.filter(pl.col(self.results.well_col) == well)

            if well_sigs.is_empty():
                continue

            features = well_sigs[self.results.feature_col].to_list()
            out_path = out_dir / f"{well}_{self.results.stat}.pdf"

            with PdfPages(out_path) as pdf:

                # summary
                fig, ax = plt.subplots(figsize=(11, max(4, len(features) * 0.3)))
                self.plot_summary(ax=ax, significant=well_sigs)
                plt.tight_layout
                pdf.savefig(fig)
                plt.close(fig)

                for feat in tqdm(
                    features,
                    desc=f"progress of features for: {well}",
                    total=len(features),
                ):
                    row = well_sigs.filter(
                        pl.col(self.results.feature_col) == feat
                    ).to_dict(as_series=False)

                    fig, ax = plt.subplots(figsize=(11, 4))

                    # grab the raw time series
                    self.plot_timeseries(ax=ax, well=well, feature=feat)

                    # add the fits
                    self.plot_fit(ax, row)

                    ax.set_title(
                        f"{feat}\n"
                        f'p_adj={row["p_adjusted"][0]:.4e}  '
                        f'amp={row["jtk_amplitude"][0]:.4f}  '
                        f'phase={row["jtk_phase_hours"][0]:.2f}h  '
                        f'baseline={row["jtk_baseline"][0]:.4f}'
                    )

                    ax.legend(fontsize=8)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                d = pdf.infodict()
                d["Title"] = f"Cosinor Results — {well} | {self.results.stat}"
                d["Subject"] = "Circadian oscillation via cosinor fitting"
