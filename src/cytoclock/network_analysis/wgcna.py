#!/usr/bin/env python3

from __future__ import annotations

import logging
from pathlib import Path
import time

from click import group
import numpy as np
import polars as pl

from ._wgcna_tools import (
    adjacency_matrix,
    compute_eigengene,
    correlation_matrix,
    detect_modules,
    find_redundant_feats,
    merge_similar_modules,
    module_membership,
    pick_soft_threshold as _pick_soft_threshold,
    topological_overlap_matrix,
)

from .dataclasses import AssembledMatrix, WGCNAResults


class WGCNAAnalyzer:
    """
    Runs correlation-network / module detection analysis of high-throughput
    feature data, grouped by condition/treatment - or per well independently if
    if no grouping is given

    Params:
        - signed (bool): signed vs unsigned network (default True — recommended
            for circadian data, see adjacency_matrix docstring)
        - min_module_size (int): minimum features to keep a module (default 30)
        - merge_cut_height (float): eigengene correlation threshold above
            which similar modules are merged (default 0.75)
        - redundancy_threshold (float): |r| threshold for collapsing
            near-duplicate features before running the pipeline (default 0.99)
        - correlation_method (str): "pearson", "spearman", or "bicor" (default "bicor")
        - soft_power (int | None): fixed soft-thresholding power. Must be
            set (via pick_soft_threshold's manual inspection) before fit()/fit_all().
    """

    def __init__(
        self,
        signed: bool = True,
        min_module_size: int = 30,
        merge_cut_height: float = 0.75,
        redundancy_threshold: float = 0.99,
        correlation_method: str = "pearson",
        soft_power: int | None = None,
    ):
        self.signed = signed
        self.min_module_size = min_module_size
        self.merge_cut_height = merge_cut_height
        self.redundancy_threshold = redundancy_threshold
        self.correlation_method = correlation_method
        self.soft_power = soft_power

    @property
    def params(self) -> dict:
        """WGCNA parameters"""
        return {
            "signed": self.signed,
            "min_module_size": self.min_module_size,
            "merge_cut_height": self.merge_cut_height,
            "redundancy_threshold": self.redundancy_threshold,
            "correlation_method": self.correlation_method,
            "soft_power": self.soft_power,
        }

    # -- assembling data

    def _assemble_one_group(
        self,
        dataset: pl.DataFrame,
        wells: list[str],
        well_col: str,
        timepoint_col: str,
        feature_col: str,
        value_col: str,
        group_key: str,
    ) -> AssembledMatrix | None:
        """
        Z-scores each well independently (per feature, across its own
        timepoints), transposes to (features x timepoints), then stacks
        wells side-by-side along the column (observation) axis.
        """

        transposed_frames = []
        col_info_rows = []
        col_offset = 0

        for well in wells:
            well_wide = (
                dataset.filter(pl.col(well_col) == well)
                .pivot(on=feature_col, index=timepoint_col, values=value_col)
                .sort(timepoint_col)
            )
            if well_wide.height == 0:
                continue

            timepoints = well_wide[timepoint_col].to_list()
            feat_cols = [c for c in well_wide.columns if c != timepoint_col]
            if len(feat_cols) == 0:
                continue

            # calulates per-well per-feature Z-scores
            well_wide_z = well_wide.with_columns(
                [
                    pl.when(pl.col(c).std() == 0)
                    .then(None)
                    .otherwise((pl.col(c) - pl.col(c).mean()) / pl.col(c).std())
                    .alias(c)
                    for c in feat_cols
                ]
            )

            n_tp = len(timepoints)
            col_names = [str(i) for i in range(col_offset, col_offset + n_tp)]

            transposed = well_wide_z.select(feat_cols).transpose(
                include_header=True, header_name="feature", column_names=col_names
            )
            transposed_frames.append(transposed)

            col_info_rows.extend([{"well": well, "timepoint": tp} for tp in timepoints])
            col_offset += n_tp

        if not transposed_frames:
            logging.warning(f"Group '{group_key}': no valid wells found, skipping")
            return None

        # outer-join all wells on feature handles any well missing a feature
        stacked = transposed_frames[0]
        for frame in transposed_frames[1:]:
            stacked = stacked.join(frame, on="feature", how="full", coalesce=True)

        col_names_sorted = sorted(
            [c for c in stacked.columns if c != "feature"], key=int
        )
        stacked = stacked.select(["feature"] + col_names_sorted)

        col_info = pl.DataFrame(col_info_rows)
        feature_names = stacked["feature"].to_list()
        matrix = stacked.select(col_names_sorted).to_numpy().astype(np.float64)

        # drop NaNs
        valid_rows = ~np.all(np.isnan(matrix), axis=1)
        row_variance = np.nanvar(matrix, axis=1)
        valid_rows = valid_rows & (row_variance > 0) & np.isfinite(row_variance)

        dropped_low_var = [f for f, ok in zip(feature_names, valid_rows) if not ok]
        matrix = matrix[valid_rows, :]
        feature_names = [f for f, ok in zip(feature_names, valid_rows) if ok]

        matrix = np.nan_to_num(matrix, nan=0.0)

        # -- collapse duplicate features
        dropped_redundant: dict[str, str] = {}
        if len(feature_names) > 1:
            corr = correlation_matrix(matrix=matrix, method=self.correlation_method)
            dropped_redundant = find_redundant_feats(
                corr, feature_names, threshold=self.redundancy_threshold
            )
            keep_mask = np.array([f not in dropped_redundant for f in feature_names])
            matrix = matrix[keep_mask, :]
            feature_names = [f for f, keep in zip(feature_names, keep_mask) if keep]

        return AssembledMatrix(
            matrix=matrix,
            feature_names=feature_names,
            col_info=col_info,
            group_key=group_key,
            dropped_redundant=dropped_redundant,
            dropped_low_var=dropped_low_var,
        )

    def assemble_matrix(
        self,
        dataset: pl.DataFrame,
        well_col: str,
        timepoint_col: str,
        feature_col: str,
        value_col: str,
        platemap: pl.DataFrame | None = None,
        group_col: str | None = None,
    ) -> list[AssembledMatrix]:
        """
        Builds one assembled (z-scored, stacked) matrix per group.
        If a platemap is not provided then each well is independent
        """
        if platemap is not None and group_col is not None:
            merged = dataset.join(
                platemap.select([well_col, group_col]), on=well_col, how="left"
            )
            group_table = (
                merged.select([group_col, well_col])
                .unique()
                .group_by(group_col)
                .agg(pl.col(well_col))
            )
            groups_to_wells = {
                row[group_col]: row[well_col]
                for row in group_table.iter_rows(named=True)
            }
        else:
            logging.info(
                "No platemap/group_col provided - processing each well independently"
            )
            wells = dataset.select(well_col).unique()[well_col].to_list()
            groups_to_wells = {well: [well] for well in wells}

        assembled = []
        for group_key, wells in groups_to_wells.items():
            am = self._assemble_one_group(
                dataset=dataset,
                wells=wells,
                well_col=well_col,
                timepoint_col=timepoint_col,
                feature_col=feature_col,
                value_col=value_col,
                group_key=str(group_key),
            )
            if am is not None:
                assembled.append(am)

        return assembled

    def pick_soft_threshold(
        self, assembled: AssembledMatrix, powers: list[int] | None = None
    ) -> pl.DataFrame:
        """generates a Dataframe to choose a threshold from"""
        corr = correlation_matrix(assembled.matrix, method=self.correlation_method)
        return _pick_soft_threshold(corr, powers=powers, signed=self.signed)

    def analyze(
        self, assembled: AssembledMatrix, edge_threshold: float = 0.1
    ) -> WGCNAResults:
        """
        Runs the full pipeline for ONE assembled group

        Args:
            assembled (AssembledMatrix): output of assemble_matrix()
            edge_threshold (float, optional): minimum TOM weight for an edge to be
                kept in edge_df — a dense feature x feature graph is
                unreadable/enormous even for a few hundred features, so
                only edges above this weight are exported. Defaults to 0.1.

        Returns:
            WGCNAResults: An object to store the results of WGCNA analysis
        """

        if self.soft_power is None:
            raise ValueError(
                "soft_power is not set. Call pick_soft_threshold() first, "
                "inspect the scale-free table/plot, and set analyzer.soft_power"
            )

        matrix = assembled.matrix
        feature_names = assembled.feature_names

        logging.info(f"[{assembled.group_key}] computing correlation matrix")
        corr = correlation_matrix(matrix=matrix, method=self.correlation_method)

        logging.info(
            f"[{assembled.group_key}] building adjacency matrix"
            f"(power={self.soft_power})"
        )
        adj = adjacency_matrix(corr, power=self.soft_power, signed=self.signed)

        logging.info(f"[{assembled.group_key}] computing TOM")
        tom = topological_overlap_matrix(adj)

        logging.info(f"[{assembled.group_key}] detecting modules")
        module_assignments = detect_modules(
            tom, feature_names, min_module_size=self.min_module_size
        )
        module_assignments = merge_similar_modules(
            module_assignments,
            matrix,
            feature_names,
            merge_cut_height=self.merge_cut_height,
        )

        logging.info(f"[{assembled.group_key}] computing eigengenes + membership")
        membership_df = module_membership(matrix, feature_names, module_assignments)
        membership_df = membership_df.with_columns(
            pl.lit(assembled.group_key).alias("group_key")
        )

        # -- edge list
        logging.info(
            f"[{assembled.group_key}] building edge list (thresh={edge_threshold})"
        )
        n = len(feature_names)
        upper_i, upper_j = np.triu_indices(n, k=1)
        weights = tom[upper_i, upper_j]

        keep = weights > edge_threshold
        edge_df = pl.DataFrame(
            {
                "feature_a": [feature_names[i] for i in upper_i[keep]],
                "feature_b": [feature_names[j] for j in upper_j[keep]],
                "weight": weights[keep],
                "group_key": assembled.group_key,
            }
        )

        # -- eigengenes
        module_ids = sorted(set(v for v in module_assignments.values() if v != 0))
        feat_to_idx = {f: i for i, f in enumerate(feature_names)}
        col_info_records = assembled.col_info.to_dicts()

        eigengene_rows = []
        for mod_id in module_ids:
            member_feats = [f for f, m in module_assignments.items() if m == mod_id]
            idx = [feat_to_idx[f] for f in member_feats if f in feat_to_idx]
            if len(idx) < 2:
                continue
            eigengene, _, _ = compute_eigengene(matrix[idx, :])

            for col_idx, score in enumerate(eigengene):
                info = col_info_records[col_idx]
                eigengene_rows.append(
                    {
                        "group_key": assembled.group_key,
                        "well": info["well"],
                        "timepoint": info["timepoint"],
                        "feature": f"Module_{mod_id}",
                        "value": score,
                    }
                )

        eigengene_df = (
            pl.DataFrame(eigengene_rows)
            if eigengene_rows
            else pl.DataFrame(
                schema={
                    "group_key": pl.String,
                    "well": pl.String,
                    "timepoint": pl.Float64,
                    "feature": pl.String,
                    "value": pl.Float64,
                }
            )
        )

        return WGCNAResults(
            module_df=membership_df,
            eigengene_df=eigengene_df,
            edge_df=edge_df,
            params=self.params,
        )

    def analyze_all(
        self,
        dataset: pl.DataFrame,
        well_col: str,
        timepoint_col: str,
        feature_col: str,
        value_col: str,
        platemap: pl.DataFrame | None = None,
        group_col: str | None = None,
        edge_threshold: float = 0.1,
    ) -> WGCNAResults:
        """
        Full pipeline across ALL groups (or all wells, if ungrouped).
        Requires self.soft_power already set.

        Args:
            dataset (pl.DataFrame): long format — well_col, timepoint_col,
                feature_col, value_col
            well_col (str): Column for Well Names
            timepoint_col (str): Column for timepoints
            feature_col (str): Column for features
            value_col (str): Column for value metrics such as median, mean q0 etc.
            platemap (pl.DataFrame | None, optional): well metadata. Must contain
                group_col and well_col if grouping is desired. Defaults to None.
            group_col (str | None, optional): platemap column defining
                replicate groups. Defaults to None.
            edge_threshold (float, optional): min TOM weight kept in edge_df.
                Defaults to 0.1.

        Returns:
            WGCNAResults: _description_
        """

        assembled_list = self.assemble_matrix(
            dataset=dataset,
            well_col=well_col,
            timepoint_col=timepoint_col,
            feature_col=feature_col,
            value_col=value_col,
            platemap=platemap,
            group_col=group_col,
        )

        module_dfs = []
        eigengene_dfs = []
        edge_dfs = []

        for assembled in assembled_list:
            result = self.analyze(assembled, edge_threshold=edge_threshold)
            module_dfs.append(result.module_df)
            eigengene_dfs.append(result.eigengene_df)
            edge_dfs.append(result.edge_df)

        return WGCNAResults(
            module_df=pl.concat(module_dfs, how="diagonal_relaxed"),
            eigengene_df=pl.concat(eigengene_dfs, how="diagonal_relaxed"),
            edge_df=pl.concat(edge_dfs, how="diagonal_relaxed"),
            params=self.params,
        )
