#!/usr/bin/env python3
"""
Low level routines and helpers for WGCNA
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

# -- Correlation / Redundancy


def _biweight_standardize(matrix: np.ndarray, c: float = 9.0) -> np.ndarray:
    """
    Robust standardization for biweight midcorrelation (bicor).
    Replaces mean/std centering with median/MAD centering, then down-weights
    outliers via Tukey's biweight function before unit-normalizing each
    column — so a plain dot product between columns gives the correlation.
    """
    med = np.median(matrix, axis=1, keepdims=True)  # per feature (row)
    mad = np.median(np.abs(matrix - med), axis=1, keepdims=True)
    mad = np.where(mad == 0, np.finfo(float).eps, mad)

    u = (matrix - med) / (c * mad)
    weights = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)

    standardized = (matrix - med) * weights
    norms = np.sqrt(np.sum(standardized**2, axis=1, keepdims=True))  # per row
    norms = np.where(norms == 0, np.finfo(float).eps, norms)

    return standardized / norms


def correlation_matrix(matrix: np.ndarray, method: str = "pearson") -> np.ndarray:
    """
    Calculate the correlation matrix for a given matrix of (Features x Timepoints)

    Args:
        matrix (np.ndarray): Input matrix
        method (str, optional): selects the correlation method.
        Defaults to "pearson".

    Returns:
        np.ndarray: nxn Correlation matrix
    """
    if method == "pearson":
        corr = np.corrcoef(matrix)
    elif method == "spearman":
        ranks = rankdata(matrix, axis=1)
        corr = np.corrcoef(ranks)
    else:
        z = _biweight_standardize(matrix)
        corr = z @ z.T

    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def find_redundant_feats(
    corr: np.ndarray, feature_names: list[str], threshold: float = 0.99
) -> dict[str, str]:
    """
    Identifies near-duplicate features (|r| > threshold) and maps each
    redundant feature to the representative feature it should collapse to.
    Representative = the first feature (by column order) in each redundant group.

    args:
        corr (np.ndarray): feature x feature correlation matrix
        feature_names (list[str]): names aligned with corr's rows/columns
        threshold (float): |r| above which two features are considered
            near-duplicates (default 0.99)

    returns:
        dict[str, str]: {redundant_feature: representative_feature}
        (features not in this dict are unique / kept as-is)
    """
    n = len(feature_names)
    redundant_map: dict[str, str] = {}
    assigned = np.zeros(n, dtype=bool)

    for i in range(n):
        if assigned[i]:
            continue

        # find all j that are near-duplicates of i and not yet claimed
        dup_mask = (np.abs(corr[i]) > threshold) & (~assigned)
        dup_mask[i] = False
        dup_idx = np.where(dup_mask)[0]

        if len(dup_idx) > 0:
            representative = feature_names[i]
            for j in dup_idx:
                redundant_map[feature_names[j]] = representative
                assigned[j] = True
            assigned[i] = True
    return redundant_map


# -- Soft power


def scale_free_fit_index(
    adjacency: np.ndarray, n_bins: int = 20
) -> tuple[float, float]:
    """
    Computes the scale-free topology fit R^2 for 1 adjacency matrix
    which is the diagnostic used to justify a soft-power choice.

    Args:
        adjacency (np.ndarray): adjacency matrix
        n_bins (int, optional): number of bins for histogram. Defaults to 20.

    Returns:
        tuple[float, float]: [signed r^2, mean k]
    """

    k = adjacency.sum(axis=1)
    mean_k = float(np.mean(k))

    if np.all(k <= 0) or np.std(k) == 0:
        return 0.0, mean_k

    # bin the degree distribution, then check if log(p(k)) vs log(k) is linear
    # — that linear relationship IS the definition of "scale-free"
    hist, bin_edges = np.histogram(k, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    valid = (hist > 0) & (bin_centers > 0)
    if valid.sum() > 3:
        return 0.0, mean_k

    log_p = np.log10(hist[valid] / hist[valid].sum())
    log_k = np.log10(bin_centers[valid])

    slope, intercept = np.polyfit(log_k, log_p, 1)
    fitted = slope * log_k + intercept
    ss_res = np.sum((log_p - fitted) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # sign(slope) -> a real scale-free network has a negative slope
    # (few hubs, many low-degree features) - positive slope means
    # power is doing something backwards
    signed_r2 = float(np.sign(slope) * r_squared)

    return signed_r2, mean_k


def pick_soft_threshold(
    corr: np.ndarray, powers: list[int] | None = None, signed: bool = True
) -> pl.DataFrame:
    """
    Scans candidate soft-thresholing powers, returns the scale-free fit
    diagnostic table for the user to plot and manually choose beta.

    Args:
        corr (np.ndarray): correlation matrix
        powers (list[int] | None, optional): List of numerical powers. Defaults to None.
        signed (bool, optional): is the correlation is signed. Defaults to True.


    Returns:
        pl.DataFrame: _description_
    """

    if powers is None:
        powers = list(range(1, 21))

    rows = []
    for power in powers:
        adj = adjacency_matrix(corr=corr, power=power, signed=signed)
        np.fill_diagonal(adj, 0.0)
        r2, mean_k = scale_free_fit_index(adj)
        rows.append({"power": power, "signed_r2": r2, "mean_connectivity": mean_k})

    return pl.DataFrame(rows)


# -- Adjacency / topological overlay matrix


def adjacency_matrix(corr: np.ndarray, power: int, signed: bool = True) -> np.ndarray:
    """
    Converts a correlation matrix into a weighted adj. matrix

    Args:
        corr (np.ndarray): Correlation matrix
        power (int): raises the matrix to a certain power
        signed (bool):
                signed=True:   adjacency = ((1 + corr) / 2) ^ power
                   maps corr from [-1, 1] to [0, 1] first, so a perfect
                   negative correlation (r=-1) gets adjacency ~0, and a
                   perfect positive correlation (r=1) gets adjacency ~1.
                   Antiphase features (peak at CT0 vs CT12) end up weakly
                   connected — appropriate for circadian data, since
                   peak-vs-trough timing is a real biological distinction,
                   not just sign noise.

                signed=False:  adjacency = |corr| ^ power
                    treats r=-0.9 and r=+0.9 as EQUALLY strongly connected.
                    Antiphase features would get merged into the same module,
                    which is usually not what you want for rhythm data.

    Returns:
        np.ndarray: weighted adjacency matrix
    """

    if signed:
        adj = ((1 + corr) / 2) ** power
    else:
        adj = np.abs(corr) ** power

    np.fill_diagonal(adj, 0.0)
    return adj


def topological_overlap_matrix(adjacency: np.ndarray) -> np.ndarray:
    """
    Converts an adjacency matrix into a Topological Overlap Matrix (TOM).

    TOM_ij = (sum_k(a_ik * a_kj) + a_ij) / (min(k_i, k_j) + 1 - a_ij)

    Intuition:
      - sum_k(a_ik * a_kj)  = how much i and j share the SAME neighbors
                              (high if both are strongly connected to
                               the same third features)
      - a_ij                = their own direct connection strength
      - min(k_i, k_j)        = normalizes by whichever feature has fewer
                               total connections, so TOM stays bounded
                               regardless of how "popular" a hub feature is

    Diagonal is set to 1 by convention — a feature has perfect overlap
    with itself.

    Args:
        adjacency (np.ndarray): Adjacency matrix

    Returns:
        np.ndarray: Topological overlap matrix
    """
    n = adjacency.shape[0]
    a = adjacency.copy()
    np.fill_diagonal(a, 0.0)

    k = a.sum(axis=1)  # each feats connectivity / degree
    l_matrix = a @ a  # matmul gives us sum_k(a_ik * a_kj) for ALL i,j pairs at once

    k_i = k[:, None]
    k_j = k[None, :]
    min_k = np.minimum(k_i, k_j)

    denom = min_k + 1 - a
    denom[denom == 0] = np.finfo(float).eps  # avoid divide by 0

    tom = (l_matrix + a) / denom
    np.fill_diagonal(tom, 1.0)

    tom = np.clip(tom, 0.0, 1.0)
    return tom


# -- Module detection


def detect_modules(
    tom: np.ndarray,
    feature_names: list[str],
    min_module_size: int = 30,
    cut_height: float | None = None,
    linkage_method: str = "average",
) -> dict[str, int]:
    """
    Hierarchical clustering + simplified dynamic tree cut on TOM
    dissimilarity (1 - TOM). Clusters smaller than min_module_size are
    reassigned to module 0 (the "unassigned/grey" module, per WGCNA
    convention — not every feature belongs to a meaningful module).

    Args:
        tom (np.ndarray): topological overlap matrix (feature x feature)
        feature_names (list[str]): names aligned with tom's rows/cols
        min_module_size (int, optional): minimum features to keep a module. Defaults to 30.
        cut_height (float | None, optional): dendrogram cut height, 0-1 dissimilarity
            scale. If None, searches for the height giving the most valid
            modules meeting min_module_size.. Defaults to None.
        linkage_method (str, optional): scipy linkage method.
            Defaults to 'average'.

    Returns:
        dict[str, str]: {feature_name: module_id}, 0 = unassigned
    """

    dissimilarity = 1 - tom
    np.fill_diagonal(dissimilarity, 0.0)
    dissimilarity = (dissimilarity + dissimilarity.T) / 2  # enforce exact symmetry

    condensed = squareform(dissimilarity, checks=False)
    Z = linkage(condensed, method=linkage_method)

    if cut_height is None:
        # look for candidate heights which gives the most modules
        # that is still above the min module size threshold
        candidate_heights = np.linspace(0.1, 0.99, 35)
        best_height = candidate_heights[-1]
        best_n_valid_modules = -1

        for h in candidate_heights:
            labels = fcluster(Z, t=h, criterion="distance")
            _, counts = np.unique(labels, return_counts=True)
            n_valid = int(np.sum(counts >= min_module_size))
            if n_valid > best_n_valid_modules:
                best_n_valid_modules = n_valid
                best_height = h

        cut_height = best_height

    labels = fcluster(Z, t=cut_height, criterion="distance")

    # lets combine all the small clusters to module 0
    unique_labels, counts = np.unique(labels, return_counts=True)
    small_labels = set(unique_labels[counts < min_module_size])

    # relabel surviving modules to consecutive ints starting at 1
    surviving_labels = sorted(l for l in unique_labels if l not in small_labels)
    relabel_map = {old: new for new, old in enumerate(surviving_labels, start=1)}

    module_assignments = {}
    for feat, lbl in zip(feature_names, labels):
        module_assignments[feat] = relabel_map.get(lbl, 0)

    return module_assignments


def merge_similar_modules(
    module_assignments: dict[str, int],
    matrix: np.ndarray,
    feature_names: list[str],
    merge_cut_height: float = 0.75,
) -> dict[str, int]:
    """
    Merges modules whose eigengenes correlate above merge_cut_height,
    since near-identical eigengenes usually indicate the cut height split
    one real module into two pieces, not two distinct biological modules.

    Args:
        module_assignments (dict[str, int]): the cluster labels
        matrix (np.ndarray): data matrix
        feature_names (list[str]): feature list
        merge_cut_height (float, optional): the threshold for correlations to be
            above. Defaults to 0.75.

    Returns:
        dict[str, int]: {feature: new label id}
    """
    module_ids = sorted(set(v for v in module_assignments.values() if v != 0))
    if len(module_ids) < 2:
        return module_ids

    feat_to_idx = {f: i for i, f in enumerate(feature_names)}

    # compute eigengene per current module
    eigengenes = {}
    for mod_id in module_ids:
        member_feats = [f for f, m in module_assignments.items() if m == mod_id]
        idx = [feat_to_idx[f] for f in member_feats if f in feat_to_idx]
        if len(idx) < 2:
            continue
        eigengene, _, _ = compute_eigengene(
            matrix[idx, :]
        )  # features as rows oriantation
        eigengenes[mod_id] = eigengene

    if len(eigengenes) < 2:
        return module_assignments

    # cluster eigengenes to find duplicates
    mod_ids_with_eigen = list(eigengenes.keys())
    eig_matrix = np.column_stack([eigengenes[m] for m in mod_ids_with_eigen])
    eig_corr = np.corrcoef(eig_matrix, rowvar=False)
    eig_corr = np.nan_to_num(eig_corr, nan=0.0)

    dissimilarity = 1 - np.abs(eig_corr)
    np.fill_diagonal(dissimilarity, 0.0)
    dissimilarity = (dissimilarity + dissimilarity.T) / 2

    if dissimilarity.shape[0] < 2:
        return module_assignments

    condensed = squareform(dissimilarity, checks=False)
    Z = linkage(condensed, method="average")
    merge_labels = fcluster(Z, t=(1 - merge_cut_height), criterion="distance")

    merge_map = {
        mod_ids_with_eigen[i]: int(merge_labels[i])
        for i in range(len(mod_ids_with_eigen))
    }

    updated = {
        feat: merge_map.get(mod_id, mod_id)
        for feat, mod_id in module_assignments.items()
    }

    return updated


def compute_eigengene(
    module_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """

    Args:
        module_matrix (np.ndarray): (n_module_features, n_timepoints),
            features as rows, already z-scored

    Returns:
        tuple[float, float, float]:
            eigengene (np.ndarray): (n_timepoints,) — one score per timepoint
            loadings (np.ndarray): (n_module_features,) — feature weights
            variance_explained (float): fraction of variance PC1 captures
    """

    # center each feature (row) - should be already ~0 mean from z-scoring
    centered = module_matrix - module_matrix.mean(axis=1, keepdims=True)

    # SVD on feat x timepoints: U holds feat loadings
    # VT holds per timepoint componenet scores
    u, s, vt = np.linalg.svd(centered, full_matrices=False)

    pc1_loadings = u[:, 0]
    pc1_scores = vt[0, :] * s[0]

    variance_explained = float((s[0] ** 2) / np.sum(s**2)) if np.sum(s**2) > 0 else 0.0

    # sign_convention: eigenegen should positively correlate with
    # average of member features
    avg_features = centered.mean(axis=0)

    if np.corrcoef(pc1_scores, avg_features)[0, 1] < 0:
        pc1_scores = -1 * pc1_scores
        pc1_loadings = -1 * pc1_loadings

    return pc1_scores, pc1_loadings, variance_explained


def module_membership(
    matrix: np.ndarray, feature_names: list[str], module_assignments: dict[str, int]
) -> pl.DataFrame:
    """
    Computes module membership (kME) — correlation of each feature to its
    own module's eigengene. High kME identifies "hub" features.

    Args:
        matrix (np.ndarray): (n_features, n_timepoints), features as rows
        feature_names (list[str]): names aligned with matrix's rows
        module_assignments (dict[str, int]): {feature: module_id}

    Returns:
        pl.DataFrame: columns = [feature, module_id, kME, variance_explained]
    """

    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    module_ids = sorted(set(v for v in module_assignments.values() if v != 0))

    rows = []
    for mod_id in module_ids:
        member_feats = [f for f, m in module_assignments.items() if m == mod_id]
        idx = [feat_to_idx[f] for f in member_feats if f in feat_to_idx]
        if len(idx) < 2:
            continue

        eigengene, _, var_exp = compute_eigengene(matrix[idx, :])  # grab module's feats

        for feat, row_idx in zip(member_feats, idx):
            kme = np.corrcoef(matrix[row_idx, :], eigengene)[0, 1]  # correlate feat
            rows.append(
                {
                    "feature": feat,
                    "module_id": mod_id,
                    "kME": kme,
                    "variance_explained": var_exp,
                }
            )

    for feat, mod_id in module_assignments.items():
        if mod_id == 0:
            rows.append(
                {
                    "feature": feat,
                    "module_id": 0,
                    "kME": np.nan,
                    "variance_explained": np.nan,
                }
            )

    return pl.DataFrame(rows)
