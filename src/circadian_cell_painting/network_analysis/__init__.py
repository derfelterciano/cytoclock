#!/usr/bin/env python3

"""
Here's the full architecture, organized by file and responsibility:

## `network_analysis/_wgcna_tools.py` — low-level numerical functions (stateless, testable independently)

**Correlation / redundancy**
- `correlation_matrix(matrix) -> np.ndarray` — feature × feature Pearson correlation
- `find_redundant_features(corr, feature_names, threshold) -> dict[str, str]` — maps near-duplicate features to a representative to collapse

**Soft-power selection (manual diagnostic step)**
- `scale_free_fit_index(adjacency, n_bins) -> (r2, mean_k)` — computes the scale-free topology fit for one adjacency matrix
- `pick_soft_threshold(corr, powers, signed) -> pd.DataFrame` — scans candidate powers, returns fit table for the user to plot and inspect

**Adjacency / TOM**
- `adjacency_matrix(corr, power, signed) -> np.ndarray` — correlation → weighted adjacency
- `topological_overlap_matrix(adjacency) -> np.ndarray` — adjacency → TOM

**Module detection**
- `detect_modules(tom, feature_names, min_module_size, cut_height) -> dict[str, int]` — hierarchical clustering + tree cut → `{feature: module_id}` (0 = unassigned)
- `merge_similar_modules(module_assignments, matrix, feature_names, merge_cut_height) -> dict[str, int]` — merges modules with highly correlated eigengenes

**Eigengenes / membership**
- `compute_eigengene(module_matrix) -> (eigengene, loadings, variance_explained)` — PC1 of a module's features, sign-corrected
- `module_membership(matrix, feature_names, module_assignments) -> pd.DataFrame` — kME per feature

## `network_analysis/wgcna.py` — orchestration (stateful config, data assembly, pipeline)

**Data classes**
- `AssembledMatrix` — one group's ready-to-analyze matrix + bookkeeping (dropped features, row→well/timepoint mapping)
- `WGCNAResults` — final output: `module_df` (feature→module→kME) + `eigengene_df` (long format, feeds into `detect()`)

**`WGCNAAnalyzer` class**
| Method | Responsibility |
|---|---|
| `__init__` | stores config: `signed`, `min_module_size`, `merge_cut_height`, `redundancy_threshold`, `soft_power` |
| `assemble_matrix(...)` | long data → list of `AssembledMatrix`, one per group (or per well if no platemap) |
| `_assemble_one_group(...)` | private: pivot → per-well z-score → stack replicates → drop low-var → collapse redundant |
| `pick_soft_threshold(assembled)` | wraps `_wgcna_tools.pick_soft_threshold` for one group's matrix — manual, look-at-plot step |
| `fit(assembled)` | runs the full pipeline on ONE `AssembledMatrix` → one `WGCNAResults` |
| `fit_all(dataset, ...)` | calls `assemble_matrix` then `fit` for every group, concatenates into one combined `WGCNAResults` |

## Call sequence (what you'd actually run)

```
1. analyzer = WGCNAAnalyzer(signed=True, min_module_size=30)

2. assembled = analyzer.assemble_matrix(dataset, ..., platemap=pm, group_col="group")
   → one AssembledMatrix per condition (or per well, if platemap=None)

3. diag = analyzer.pick_soft_threshold(assembled[0])
   → plot diag['power'] vs diag['signed_r2'], manually choose a power

4. analyzer.soft_power = <chosen power>

5. results = analyzer.fit_all(dataset, ..., platemap=pm, group_col="group")
   → results.module_df       (feature, module_id, kME, group_key)
   → results.eigengene_df    (group_key, well, timepoint, feature=Module_N, value)

6. results.eigengene_df → feed into detect() with CosinorDetection/JTK_CYCLE
   (not yet wired up — that's the next step)
```

Everything in step 2-5 is what we tested against synthetic data — the only piece not yet built is the glue code connecting `eigengene_df` to `detect()`'s expected input shape.
"""

from .wgcna import WGCNAAnalyzer
from .dataclasses import WGCNAResults

__all__ = ["WGCNAAnalyzer", "WGCNAResults"]
