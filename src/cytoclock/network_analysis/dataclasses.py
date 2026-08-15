#!/usr/bin/env python3


import numpy as np
import polars as pl
from dataclasses import dataclass, field
from pathlib import Path
import pickle


@dataclass
class AssembledMatrix:
    """
    Bundles one group's assembled data which is the wide, z-scored, stacked
    matrix ready for correlation, adjacency, and TOM. Additionally handles
    bookkeeping needed to trace columns back to (well, timepoint) and report
    which feats were dropped and why

    args:
        matrix (np.ndarray): (n_features, n_observations) — features as
            rows, timepoints (stacked across replicate wells) as columns,
            already z-scored per well
        feature_names (list[str]): names aligned with matrix's rows
        col_info (pd.DataFrame): columns = [well, timepoint] — aligned
            with matrix's COLUMNS (one row per observation/timepoint)
        group_key (str): the group (or well, if ungrouped) this matrix
            represents
        dropped_redundant (dict[str, str]): {dropped_feature: kept_representative}
        dropped_low_var (list[str]): features dropped for zero/near-zero
            variance before redundancy checking
    """

    matrix: np.ndarray
    feature_names: list[str]
    col_info: pl.DataFrame
    group_key: str
    dropped_redundant: dict[str, str] = field(default_factory=dict)
    dropped_low_var: list[str] = field(default_factory=list)


@dataclass
class WGCNAResults:
    """
    Bundles the output of WGCNAAnalyzer.fit()/fit_all()

    module_df    : one row per feature — its module assignment, kME, group
    eigengene_df : long format (group_key, well, timepoint, feature=module
                   label, value=eigengene) — ready to feed directly into
                   detect() alongside CosinorDetection/JTK_CYCLE/eJTK_CYCLE
    params       : the WGCNAAnalyzer parameters used for this run
    """

    module_df: pl.DataFrame
    eigengene_df: pl.DataFrame
    edge_df: pl.DataFrame
    params: dict

    def to_pickle(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, path: Path) -> "WGCNAResults":
        with open(path, "rb") as f:
            return pickle.load(f)
