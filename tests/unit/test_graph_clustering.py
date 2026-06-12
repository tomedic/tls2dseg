"""Unit tests for graph clustering helpers — TEST-07.

Phase 4 plan 03 Task 1 (TEST-07). Locks the contract of
``tls2dseg.engines.fusion.connectivity``,
``tls2dseg.engines.fusion.clustering``, and
``tls2dseg.engines.fusion.edge_weights``.

Tier assignments:
- connectivity / CSR tests: tier_b_light — scipy.spatial.cKDTree and
  scipy.sparse are NOT available in the tier_a nox session (``--no-deps``
  installs only numpy/pydantic/typer/pytest). These tests are marked
  tier_b_light with a scipy skip guard.  [Rule 1 auto-fix: tier_a marker
  would cause ModuleNotFoundError at run time]
- Invalid-method validation test: tier_a safe — ValueError is raised
  BEFORE scipy is imported (validation moved to top of function).
- clustering / outlier / PCC tests: tier_a (stdlib + numpy only).
- HCS graph clustering test: tier_b_light (networkx not in tier_a venv).
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level availability flags for conditional skip
# ---------------------------------------------------------------------------

try:
    import scipy.sparse
    import scipy.spatial

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# connectivity — get_initial_sparse_connectivity
# ---------------------------------------------------------------------------


@pytest.mark.tier_b_light
@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy not available")
def test_get_initial_sparse_connectivity_knn_i_lt_j_invariant() -> None:
    """5-point line, knn=1 per scan → all edges have i < j.

    Locks: TEST-07 get_initial_sparse_connectivity i<j invariant.
    tier_b_light: needs scipy.spatial.cKDTree (not in tier_a --no-deps venv).
    """
    from tls2dseg.engines.fusion.connectivity import get_initial_sparse_connectivity

    centroids = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    pairs = get_initial_sparse_connectivity(centroids, n_scans=1, method="knn", knn_ps=1)
    assert pairs.ndim == 2, "Expected 2D edge list"
    assert pairs.shape[1] == 2, "Expected shape (M, 2)"
    assert np.all(pairs[:, 0] < pairs[:, 1]), f"i<j invariant violated: {pairs[pairs[:, 0] >= pairs[:, 1]]}"


@pytest.mark.tier_b_light
@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy not available")
def test_get_initial_sparse_connectivity_knn_expected_pairs() -> None:
    """3-point 1D line with knn_ps=2 → edges (0,1) and (1,2) present.

    With knn_ps=2, k_total = int(2*1 + 1) = 3 (all neighbors returned).
    Node 0 neighbors: {1, 2}; node 1 neighbors: {0, 2}; node 2 neighbors: {0, 1}.
    After i<j dedup: {(0,1), (0,2), (1,2)}.
    At minimum both (0,1) and (1,2) must be present.

    tier_b_light: needs scipy.spatial.cKDTree (not in tier_a --no-deps venv).
    """
    from tls2dseg.engines.fusion.connectivity import get_initial_sparse_connectivity

    centroids = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    pairs = get_initial_sparse_connectivity(centroids, n_scans=1, method="knn", knn_ps=2)
    pair_set = {(int(r[0]), int(r[1])) for r in pairs}
    assert (0, 1) in pair_set, f"Expected edge (0,1), got {pair_set}"
    assert (1, 2) in pair_set, f"Expected edge (1,2), got {pair_set}"


@pytest.mark.tier_a
def test_get_initial_sparse_connectivity_invalid_method() -> None:
    """Invalid method raises ValueError BEFORE scipy is imported.

    The ValueError is raised at the top of the function (input validation),
    before ``from scipy.spatial import cKDTree``, so this test is tier_a safe.

    Locks: input validation contract.
    """
    from tls2dseg.engines.fusion.connectivity import get_initial_sparse_connectivity

    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="method must be"):
        get_initial_sparse_connectivity(centroids, method="invalid")


# ---------------------------------------------------------------------------
# connectivity — sparse_connectivity_pairs2csr_matrix
# ---------------------------------------------------------------------------


@pytest.mark.tier_b_light
@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy not available")
def test_sparse_connectivity_pairs2csr_matrix_shape_and_symmetry() -> None:
    """pairs → symmetric CSR of expected shape (N, N).

    Locks: TEST-07 sparse_connectivity_pairs2csr_matrix shape+symmetry contract.
    tier_b_light: needs scipy.sparse (not in tier_a --no-deps venv).
    """
    from tls2dseg.engines.fusion.connectivity import sparse_connectivity_pairs2csr_matrix

    pairs = np.array([[0, 1], [1, 2], [0, 2]])
    mat = sparse_connectivity_pairs2csr_matrix(pairs)

    # Shape should be (3, 3) — max index is 2, so N = 3
    assert mat.shape == (3, 3), f"Expected (3, 3), got {mat.shape}"

    # Symmetric: mat[i,j] == mat[j,i]
    diff = mat - mat.T
    assert diff.nnz == 0, "CSR matrix is not symmetric"


@pytest.mark.tier_b_light
@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy not available")
def test_sparse_connectivity_pairs2csr_matrix_with_weights() -> None:
    """Edge weights are preserved and matrix is symmetric.

    Locks: weighted variant contract.
    tier_b_light: needs scipy.sparse (not in tier_a --no-deps venv).
    """
    from tls2dseg.engines.fusion.connectivity import sparse_connectivity_pairs2csr_matrix

    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([3.0, 5.0])
    mat = sparse_connectivity_pairs2csr_matrix(pairs, edge_weights=weights)

    assert mat[0, 1] == pytest.approx(3.0)
    assert mat[1, 0] == pytest.approx(3.0)
    assert mat[1, 2] == pytest.approx(5.0)
    assert mat[2, 1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# clustering — detect_upper_tail_outliers (from graph_clustering)
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_detect_upper_tail_outliers_iqr_detects_large_value() -> None:
    """IQR method detects a clear upper-tail outlier in a small array.

    Locks: detect_upper_tail_outliers basic contract (method='iqr').
    """
    from tls2dseg.engines.fusion.clustering import detect_upper_tail_outliers

    data = np.array([1, 1, 1, 1, 2, 2, 2, 2, 100], dtype=np.float64)
    outliers, cutoff = detect_upper_tail_outliers(data, method="iqr")

    assert len(outliers) >= 1, "Expected at least one outlier for value 100"
    assert 8 in outliers, f"Index 8 (value 100) should be flagged as outlier, got {outliers}"
    assert cutoff > 0, "Cutoff must be positive"


@pytest.mark.tier_a
def test_detect_upper_tail_outliers_uniform_no_outliers() -> None:
    """Uniform array → no IQR outliers (IQR = 0, cutoff = Q3).

    Locks: edge case — no outliers detected in flat data.
    """
    from tls2dseg.engines.fusion.clustering import detect_upper_tail_outliers

    data = np.array([2, 2, 2, 2, 2, 2, 2], dtype=np.float64)
    outliers, _cutoff = detect_upper_tail_outliers(data, method="iqr")
    assert len(outliers) == 0, f"Expected no outliers in uniform array, got {outliers}"


# ---------------------------------------------------------------------------
# clustering — graph_clustering (pcc and hcs methods on tiny synthetic graphs)
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_graph_clustering_pcc_two_components() -> None:
    """PCC on 4-node graph with two disconnected components → 2 clusters.

    Nodes 0-1 connected with high supporters; nodes 2-3 connected with high
    supporters; no edge between components → expect 2 cluster IDs.

    Locks: TEST-07 graph_clustering method='pcc' cluster-count contract.
    """
    from tls2dseg.engines.fusion.clustering import graph_clustering

    # 4 nodes, 2 edges (two disjoint pairs)
    num_nodes = 4
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int32)
    edge_weights = np.array([10, 10], dtype=np.int32)  # high supporters

    labels = graph_clustering(num_nodes, pairs, edge_weights, method="pcc")

    assert labels.shape == (4,), f"Expected shape (4,), got {labels.shape}"
    n_clusters = len(np.unique(labels))
    assert n_clusters == 2, f"Expected 2 clusters, got {n_clusters}: {labels}"

    # Nodes in the same component must share the same label
    assert labels[0] == labels[1], "Nodes 0 and 1 should be in same cluster"
    assert labels[2] == labels[3], "Nodes 2 and 3 should be in same cluster"
    assert labels[0] != labels[2], "Nodes 0 and 2 should be in different clusters"


@pytest.mark.tier_b_light
def test_graph_clustering_hcs_two_components() -> None:
    """HCS on 4-node graph with two disconnected components → 2 clusters.

    Marked tier_b_light because HCS uses networkx which requires a full dep
    install (not available in tier_a --no-deps venv).

    Locks: TEST-07 graph_clustering method='hcs' cluster-count contract.
    """
    pytest.importorskip("networkx")
    from tls2dseg.engines.fusion.clustering import graph_clustering

    num_nodes = 4
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int32)
    edge_weights = np.array([5.0, 5.0], dtype=np.float32)

    labels = graph_clustering(num_nodes, pairs, edge_weights, method="hcs")

    assert labels.shape == (4,), f"Expected shape (4,), got {labels.shape}"
    n_clusters = len(np.unique(labels))
    assert n_clusters == 2, f"Expected 2 clusters, got {n_clusters}: {labels}"

    assert labels[0] == labels[1], "Nodes 0 and 1 should be in same cluster"
    assert labels[2] == labels[3], "Nodes 2 and 3 should be in same cluster"
    assert labels[0] != labels[2], "Components should have different cluster ids"


@pytest.mark.tier_a
def test_graph_clustering_pcc_all_connected_one_cluster() -> None:
    """PCC on fully-connected 3-node clique with high supporters → 1 cluster.

    Locks: degenerate case — single cluster when all edges have max weight.
    """
    from tls2dseg.engines.fusion.clustering import graph_clustering

    num_nodes = 3
    pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)
    edge_weights = np.array([10, 10, 10], dtype=np.int32)

    labels = graph_clustering(num_nodes, pairs, edge_weights, method="pcc")

    assert labels.shape == (3,)
    # All nodes should have the same cluster label
    assert len(np.unique(labels)) == 1, (
        f"Expected 1 cluster for fully connected clique, got {len(np.unique(labels))}: {labels}"
    )
