"""Sparse connectivity helpers — extracted verbatim from graph_clustering.py lines 15-135.

Phase 4 plan 03 Task 1 (ENG-07). Provides initial sparse graph connectivity
for the fusion pipeline:

* ``get_initial_sparse_connectivity`` — KD-tree KNN/radius edge discovery
* ``sparse_connectivity_pairs2csr_matrix`` — pairs → symmetric CSR matrix

Import-cycle rule (RESEARCH Pitfall 2): this module imports ONLY numpy,
scipy (sparse + spatial) and stdlib. It MUST NOT import from engines/__init__.py
or preprocessing/.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

logger = logging.getLogger("tls2dseg.engines.fusion.connectivity")


def get_initial_sparse_connectivity(
    centroids: np.ndarray,  # (N,3) float32/float64
    class_ids: np.ndarray | None = None,  # (N,) int  - needed if semantic_gate=True
    n_scans: float = 1,  # float - needed for knn-threshold
    *,
    method: str = "knn",  # "knn"  or  "radius"
    knn_ps: int = 2,  # 1-3, used only if method=="knn"
    radius: float = 0.20,  # metres, used only if method=="radius"
    semantic_gate: bool = False,  # require identical class_ids?
) -> np.ndarray:
    """
    Fast KD-tree based neighbour discovery -> boolean CSR adjacency.

    Parameters
    ----------
    centroids : (N,3) array
        XYZ of bounding-box centres.
    class_ids : (N,) array or None
        Semantic labels; required iff semantic_gate is True.
    n_scans  : float
        Number of scans * features used to generate masks. Needed to
        compute knn_threshold = knn * n_scans.
    method    : "knn"  |  "radius"
        Neighbour criterion.
    knn_ps       : int
        Nuber of neighbors per scan to search for. Multiplier in knn_threshold = knn_ps * n_scans (1 ≤ knn ≤ 3).
    radius    : float
        Ball-query radius (same unit as centroids) if method=="radius".
    semantic_gate : bool
        If True, keep an edge only when class_ids[i]==class_ids[j].

    Returns
    -------
    pairs : (M,2) int array
        Edge list [i,j] with i<j for which adj[i,j]=True.
    """

    if method not in {"knn", "radius"}:
        raise ValueError("method must be 'knn' or 'radius'")
    if method == "knn" and n_scans is None:
        raise ValueError("scan_ids required for knn method")
    if semantic_gate and class_ids is None:
        raise ValueError("class_ids required when semantic_gate=True")

    # ----------  KD-tree query ----------
    tree = cKDTree(centroids)

    # build neighbour lists ---------------------------------------------------
    if method == "radius":
        neighbour_lists = tree.query_ball_tree(tree, r=radius)
    else:  # "knn"
        k_total = int(knn_ps * n_scans + 1)  # +1 to include self
        _dists, idxs = tree.query(centroids, k=k_total, workers=-1)
        neighbour_lists = [row[1:] for row in idxs]  # drop self

    # ----------  assemble edges ----------
    # The early guard above (line 58) ensures class_ids is not None when semantic_gate is True;
    # this assert pins that for mypy so class_ids[i] is safe to index.
    assert not semantic_gate or class_ids is not None
    rows, cols = [], []
    for i, nbrs in enumerate(neighbour_lists):
        for j in nbrs:
            if j <= i:
                continue  # keep i<j only once
            if semantic_gate and class_ids[i] != class_ids[j]:  # type: ignore[index]
                continue
            rows.append(i)
            cols.append(j)

    pairs = np.column_stack((rows, cols))  # i<j pairs
    return pairs


def sparse_connectivity_pairs2csr_matrix(
    pairs: np.ndarray | list, edge_weights: np.ndarray | None = None
) -> csr_matrix:
    """
    Convert a list/array of node-pairs (i, j) into a symmetric CSR adjacency matrix,
    optionally using precomputed edge weights.

    Parameters
    ----------
    pairs : array-like of shape (M, 2)
        Each row is a pair [i, j] indicating an undirected edge between nodes i and j.
    edge_weights : array-like of shape (M,), optional
        Precomputed weights for each pair. If None, all edges are set to True (boolean adjacency).

    Returns
    -------
    adj : scipy.sparse.csr_matrix
        Symmetric adjacency matrix of shape (N, N), where
        N = max node index in pairs + 1.
        If `edge_weights` is None, `adj` is boolean. Otherwise, numeric dtype of edge_weights.
    """
    # Convert pairs to numpy array
    pairs = np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("`pairs` must be of shape (M, 2)")
    M = pairs.shape[0]

    # Determine data for adjacency entries
    if edge_weights is None:
        data = np.ones(M * 2, dtype=bool)
    else:
        edge_weights = np.asarray(edge_weights)
        if edge_weights.ndim != 1 or edge_weights.shape[0] != M:
            raise ValueError("`edge_weights` must be 1D of length equal to number of pairs")
        # Duplicate weights for symmetric entries
        data = np.concatenate([edge_weights, edge_weights])

    # Build row and column index arrays for symmetric adjacency
    row = np.concatenate([pairs[:, 0], pairs[:, 1]])
    col = np.concatenate([pairs[:, 1], pairs[:, 0]])

    # Infer N from the maximum node index
    N = int(pairs.max()) + 1 if M > 0 else 0

    # Construct CSR matrix
    adj = coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
    return adj
