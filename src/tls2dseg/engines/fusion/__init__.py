"""Fusion sub-package public surface — re-exports key helpers.

Phase 4 plan 03 Task 1 (ENG-07). Provides the fusion engine and helpers
under ``tls2dseg.engines.fusion.*``.

Sub-modules:
- ``bboxes_iou``: AABB and OBB IoU helpers (moved verbatim per D-D-03)
- ``connectivity``: sparse graph connectivity (KD-tree KNN/radius)
- ``edge_weights``: edge weight computation (IoU + supporter counts)
- ``clustering``: graph clustering algorithms (PCC/HCS/Leiden + outlier detection)
- ``graph``: GraphClusterFusionEngine (owns full stage-2 pipeline per D-A-06)
"""

from __future__ import annotations

import logging

from tls2dseg.engines.fusion.bboxes_iou import compute_aabb_iou_vectorized
from tls2dseg.engines.fusion.clustering import (
    UnionFind,
    detect_upper_tail_outliers,
    graph_clustering,
    hcs_labels,
    pcc_strict_nondecreasing,
)
from tls2dseg.engines.fusion.connectivity import get_initial_sparse_connectivity, sparse_connectivity_pairs2csr_matrix
from tls2dseg.engines.fusion.edge_weights import count_significant_overlaps, get_edge_weights

logger = logging.getLogger("tls2dseg.engines.fusion")

__all__ = [
    "UnionFind",
    "compute_aabb_iou_vectorized",
    "count_significant_overlaps",
    "detect_upper_tail_outliers",
    "get_edge_weights",
    "get_initial_sparse_connectivity",
    "graph_clustering",
    "hcs_labels",
    "pcc_strict_nondecreasing",
    "sparse_connectivity_pairs2csr_matrix",
]
