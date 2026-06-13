"""GraphClusterFusionEngine — graph-clustering stage-2 fusion engine.

Phase 4 plan 03 Task 2 (ENG-03, ENG-07). Implements ``FusionEngine`` Protocol
by owning the ENTIRE stage-2 pipeline per D-A-06:

  sparse connectivity -> edge weights -> support-outlier removal ->
  graph clustering -> cluster ids -> small-cluster removal

The engine returns a ``FusionResult``; it does NOT write files.
The orchestrator (``pipeline/run.py``) applies labels.

Heavy deps (scipy, igraph, leidenalg, networkx) are already deferred
inside the helper module function bodies (connectivity / clustering).
Only pchandler-dependent operations come through ``Detections3D`` objects
that are passed in already constructed — no new module-level heavy imports.

Closest analog: ``src/tls2dseg/graph_clustering.py`` lines 295-598
(the stage-2 glue + filter_outlier_detections3d_edges_and_nodes).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from tls2dseg.detections_3d import Detections3D
    from tls2dseg.types import FusionInput, FusionResult

logger = logging.getLogger("tls2dseg.engines.fusion.graph")


@dataclasses.dataclass
class GraphClusterFusionEngine:
    """Graph-clustering fusion engine — owns all of stage-2 (D-A-06).

    Parameters
    ----------
    sparse_connectivity_method : str
        "knn" or "radius".
    sparse_connectivity_threshold : float | int
        knn_ps (int) or radius (float) depending on method.
    supporters_iou_threshold : float
        IoU threshold for counting supporters.
    remove_outliers_by_support : bool
        Whether to remove over-segmented detections by support count.
    outlier_detection_method : str
        Method for detect_upper_tail_outliers ("iqr", "mad", "percentile",
        "negative_binomial").
    outlier_detection_threshold : float
        ``alpha`` parameter for the outlier detection method.
    graph_clustering_method : str
        "leiden", "hcs", or "pcc".
    min_supporters : int
        Minimum number of supporters for PCC merges.
    leiden_resolution : float
        Resolution parameter for Leiden partition.
    small_cluster_removal_threshold : int
        Clusters with fewer than this count are zeroed out.
    merge_inst_of_same_class_only : bool
        If True, only merge detections with the same class id.
    """

    # connectivity
    sparse_connectivity_method: str = "knn"
    sparse_connectivity_threshold: float = 2.0
    supporters_iou_threshold: float = 0.15
    # outlier removal
    remove_outliers_by_support: bool = True
    outlier_detection_method: str = "negative_binomial"
    outlier_detection_threshold: float = 0.05
    # clustering
    graph_clustering_method: Literal["leiden", "hcs", "pcc"] = "leiden"
    min_supporters: int = 1
    leiden_resolution: float = 1.0
    small_cluster_removal_threshold: int = 2
    # semantic gate
    merge_inst_of_same_class_only: bool = False

    def fuse(self, fusion_input: FusionInput) -> FusionResult:
        """Run cross-scan fusion on the complete collection of 3D detections.

        Owns the entire stage-2 pipeline per D-A-06:
          1. Merge per-scan Detections3D into a single collection
          2. Sparse connectivity via KD-tree KNN/radius
          3. Edge weight computation (IoU + supporter counts)
          4. Support-outlier removal (detect_upper_tail_outliers +
             filter_outlier_detections3d_edges_and_nodes)
          5. Graph clustering (leiden/hcs/pcc)
          6. Small-cluster removal

        Parameters
        ----------
        fusion_input :
            Frozen aggregate with per-scan Detections3D list + scan ids.

        Returns
        -------
        FusionResult
            ``cluster_ids``: (N,) int32 per detection (0 = removed small cluster).
            ``kept_mask``: (N,) bool; True where detection was not removed.
        """
        from tls2dseg.detections_3d import merge_detections3d
        from tls2dseg.engines.fusion.clustering import (
            detect_upper_tail_outliers,
            graph_clustering,
        )
        from tls2dseg.engines.fusion.connectivity import get_initial_sparse_connectivity
        from tls2dseg.engines.fusion.edge_weights import count_significant_overlaps, get_edge_weights
        from tls2dseg.types import FusionResult

        # 1. Merge per-scan Detections3D into a single flat collection
        detections_list = list(fusion_input.detections_list)
        n_scans = len(detections_list)
        d3d_collection = merge_detections3d(detections_list)
        n_total = int(d3d_collection.pcd_ids.shape[0])

        # Track which original detections survive (for kept_mask construction)
        # We use an index array that gets remapped by filter_outlier_detections3d_edges_and_nodes
        survived_original_indices = np.arange(n_total, dtype=np.int64)

        # 2. Sparse connectivity
        pairs = get_initial_sparse_connectivity(
            centroids=d3d_collection.centroids,
            class_ids=d3d_collection.classes,
            n_scans=float(n_scans),
            method=self.sparse_connectivity_method,
            knn_ps=int(self.sparse_connectivity_threshold),
            radius=float(self.sparse_connectivity_threshold),
            semantic_gate=self.merge_inst_of_same_class_only,
        )

        # 3. Edge weights (IoU + supporter counts)
        edges_iou, edges_supp = get_edge_weights(
            detections3d=d3d_collection,
            pairs=pairs,
            iou_threshold=self.supporters_iou_threshold,
            mode="both",
        )

        # 4. Support-outlier removal
        if self.remove_outliers_by_support and edges_iou is not None:
            n_d3d = d3d_collection.pcd_ids.shape[0]
            counts = count_significant_overlaps(
                pairs=pairs,
                bbox_overlap=edges_iou,
                iou_threshold=self.supporters_iou_threshold,
                N=int(n_d3d),
            )
            outliers, _cutoff = detect_upper_tail_outliers(
                data=counts,
                method=self.outlier_detection_method,
                alpha=self.outlier_detection_threshold,
            )
            d3d_collection, pairs, edges_supp = _filter_outlier_detections3d_edges_and_nodes(
                d3d_collection=d3d_collection,
                pairs=pairs,
                edge_weights=edges_supp,
                outliers=outliers,
            )
            # Update survived index tracking
            survived_original_indices = _filter_survived_indices(survived_original_indices, outliers, n_total)
            n_d3d = int(d3d_collection.pcd_ids.shape[0])
        else:
            n_d3d = int(d3d_collection.pcd_ids.shape[0])

        # 5. Graph clustering
        cluster_ids = graph_clustering(
            num_nodes=n_d3d,
            pairs=pairs,
            edge_weights=edges_supp if edges_supp is not None else np.ones(pairs.shape[0], dtype=np.float32),
            method=self.graph_clustering_method,
            min_supporters=self.min_supporters,
            leiden_resolution=self.leiden_resolution,
        )

        # 6. Small-cluster removal (zero out under-threshold clusters)
        cluster_ids = _small_cluster_removal(cluster_ids, self.small_cluster_removal_threshold)

        # Build full-length (N_total,) arrays with 0 for removed detections
        full_cluster_ids = np.zeros(n_total, dtype=np.int32)
        full_cluster_ids[survived_original_indices] = cluster_ids.astype(np.int32)

        kept_mask = full_cluster_ids != 0

        logger.info(
            "Fusion complete: %d input detections -> %d survived -> %d clusters",
            n_total,
            n_d3d,
            len(np.unique(cluster_ids[cluster_ids > 0])),
        )

        return FusionResult(cluster_ids=full_cluster_ids, kept_mask=kept_mask)


def _filter_outlier_detections3d_edges_and_nodes(
    d3d_collection: Detections3D,
    pairs: np.ndarray,
    edge_weights: np.ndarray,
    outliers: np.ndarray,
) -> tuple[Detections3D, np.ndarray, np.ndarray]:
    """Remove outlier detections and any edges touching them.

    Engine-internal helper (moved from graph_clustering.py lines 295-341).
    Signature mirrors the original; re-implemented here to avoid importing
    the original graph_clustering module which still imports from the old paths.
    """
    from tls2dseg.detections_3d import filter_detections3d

    d3d: Detections3D = d3d_collection
    N_old = int(d3d.pcd_ids.shape[0])
    outliers_arr = np.asarray(outliers, dtype=int)

    # Keep mask
    keep_mask = np.ones(N_old, dtype=bool)
    keep_mask[outliers_arr] = False

    # Old-to-new index map
    old_to_new = np.full(N_old, -1, dtype=int)
    old_to_new[keep_mask] = np.arange(keep_mask.sum(), dtype=int)

    # Remap pairs
    remapped_pairs = old_to_new[pairs]
    keep_edge = np.all(remapped_pairs >= 0, axis=1)
    pairs = remapped_pairs[keep_edge]
    edge_weights = edge_weights[keep_edge]

    d3d = filter_detections3d(d3d, mask=keep_mask)
    return d3d, pairs, edge_weights


def _filter_survived_indices(
    survived: np.ndarray,
    outliers: np.ndarray,
    n_original: int,
) -> np.ndarray:
    """Return the subset of `survived` after removing `outliers` (current indices)."""
    outliers_set = set(outliers.tolist())
    current_keep = np.array([i for i in range(len(survived)) if i not in outliers_set], dtype=np.int64)
    return survived[current_keep]


def _small_cluster_removal(cluster_ids: np.ndarray, threshold: int) -> np.ndarray:
    """Zero out clusters with fewer than `threshold` detections."""
    unique_ids, count_ids = np.unique(cluster_ids, return_counts=True)
    small_clusters = unique_ids[count_ids < threshold]
    result = cluster_ids.copy()
    mask = np.isin(cluster_ids, small_clusters)
    result[mask] = 0
    return result
