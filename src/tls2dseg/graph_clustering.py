import numpy as np
from typing import Tuple, Optional, Union, List, Literal
from scipy.spatial import cKDTree  # fast radius search :contentReference[oaicite:2]{index=2}
from scipy.sparse import coo_matrix, csr_matrix
from collections import Counter
import igraph as ig
import leidenalg as la
import trimesh
from scipy.stats import nbinom

# Internal dependencies:
from tls2dseg.detections_3d import Detections3D, filter_detections3d
from tls2dseg.bboxes_iou import *
# from tls2dseg.visualization import *


def get_initial_sparse_connectivity(
        centroids: np.ndarray,  # (N,3) float32/float64
        class_ids: Optional[np.ndarray] = None,  # (N,) int  – needed if semantic_gate=True
        n_scans: float = 1,  # float – needed for knn-threshold
        *,
        method: str = "knn",  # "knn"  or  "radius"
        knn_ps: int = 2,  # 1‒3, used only if method=="knn"
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

    N = centroids.shape[0]
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
        dists, idxs = tree.query(centroids, k=k_total, workers=-1)
        neighbour_lists = [row[1:] for row in idxs]  # drop self

    # ----------  assemble edges ----------
    rows, cols = [], []
    for i, nbrs in enumerate(neighbour_lists):
        for j in nbrs:
            if j <= i:
                continue  # keep i<j only once
            if semantic_gate and class_ids[i] != class_ids[j]:
                continue
            rows.append(i)
            cols.append(j)

    pairs = np.column_stack((rows, cols))  # i<j pairs
    return pairs


def sparse_connectivity_pairs2csr_matrix(
        pairs: Union[np.ndarray, list],
        edge_weights: Optional[np.ndarray] = None
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
    if M > 0:
        N = int(pairs.max()) + 1
    else:
        N = 0

    # Construct CSR matrix
    adj = coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
    return adj


def compute_supporter_counts(pairs: np.ndarray, bbox_overlap: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    M = pairs.shape[0]
    neigh = {i: set() for i in np.unique(pairs)}
    for (i, j), iou in zip(pairs, bbox_overlap):
        if iou >= iou_threshold:
            neigh[i].add(j)
            neigh[j].add(i)
    supporter_counts = np.zeros(M, dtype=np.int16)
    for idx, (i, j) in enumerate(pairs):
        supporter_counts[idx] = len(neigh[i].intersection(neigh[j]))
    return supporter_counts


def get_edge_weights(
        detections3d: Detections3D,
        pairs: np.ndarray,
        iou_threshold: float = 0.15,
        mode: Literal['iou', 'supporters', 'both'] = 'both',
        boolean_engine: str = "scad",
        obb_workers: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute edge weights for sparse pairs: IoU and/or supporter counts.

    Parameters
    ----------
    detections3d : Detections3D object
    pairs : (M,2) int array
    iou_threshold : float, supporters IoU threshold
    mode : 'iou','supporters','both'
    boolean_engine : trimesh boolean engine
    obb_workers : #workers for OBB IoU parallel

    Returns
    -------
    bbox_overlap : (M,) or None
    supporter_counts : (M,) or None
    """

    bboxes = detections3d.bboxes
    bboxes_type = detections3d.bboxes_type
    bbox_overlap = None
    supporter_counts = None

    # Compute 3D IoU (either for AABB or for OBB)
    # TODO: if statement left inside, because maybe I implement edge weights not based on 3D IoU in future
    # AABB
    if bboxes_type == 'aabb':
        aabb = bboxes
        if mode in ('iou', 'both', 'supporters'):
            bbox_overlap = compute_aabb_iou_vectorized(aabb, pairs)
    # OBB
    elif bboxes_type == 'obb':
        if mode in ('iou', 'both', 'supporters'):
            bbox_overlap = compute_obb_iou_parallel(centers=bboxes[:, :3], extents=bboxes[:, 3:6],
                                                    quats=bboxes[:, 6:10], pairs=pairs, engine=boolean_engine,
                                                    max_workers=obb_workers)
    else:
        raise ValueError("bboxes must have 6 or 10 columns")

    if mode in ('supporters', 'both'):
        supporter_counts = compute_supporter_counts(pairs, bbox_overlap, iou_threshold)

    if mode == 'iou':
        return bbox_overlap, None
    elif mode == 'supporters':
        return None, supporter_counts
    elif mode == 'both':
        return bbox_overlap, supporter_counts
    else:
        raise ValueError(f"mode must be 'iou', 'supporters' or 'both', got {mode} instead")


def count_significant_overlaps(pairs: np.ndarray, bbox_overlap: np.ndarray, iou_threshold: float,
                               N: int) -> np.ndarray:
    """
    Count, for each of N detections (its 3D bbox), with how many other detections (3D bboxes) it has a significant
     overlap with (overlaps > iou_threshold).

    Parameters
    ----------
    pairs : (M,2) int array of detection index pairs
    bbox_overlap : (M,) float IoUs for those pairs
    iou_threshold : float
    N : int total number of detections

    Returns
    -------
    counts : (N,) int array
        counts[k] = # of detections j where IoU(k,j) > threshold
    """
    counts = np.zeros(N, dtype=int)
    for (i, j), iou in zip(pairs, bbox_overlap):
        if iou > iou_threshold:
            counts[i] += 1
            counts[j] += 1
    return counts


def detect_upper_tail_outliers(
    data: np.ndarray,
    method: str = "negative_binomial",
    *,
    iqr_factor: float = 1.5,
    mad_factor: float = 3.0,
    percentile: float = 95.0,
         alpha: float = 0.05,
) -> tuple[np.ndarray, float]:
    """
    Detect upper‐tail outliers in a 1D positive integer array.

    Parameters
    ----------
    data : (N,) array
        Any (developed for the per‐detection counts)
    method : {"iqr","mad","percentile","negative_binomial"}
    iqr_factor : float
        multiplier for IQR fence: cutoff = Q3 + iqr_factor*(Q3-Q1)
    mad_factor : float
        multiplier for MAD fence: cutoff = median + mad_factor*MAD
    percentile : float
        for "percentile" method, cutoff = percentile‐th quantile of data
    alpha : float
        for "negative_binomial" method, cutoff = nbinom.ppf(1 - alpha, r, p)

    Returns
    -------
    outlier_indices : 1D int array
        indices in `data` that exceed the cutoff.
    cutoff : float
        the numeric threshold used.
    """
    if method == "iqr":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        cutoff = q3 + iqr_factor * iqr
    elif method == "mad":
        med = np.median(data)
        mad = np.median(np.abs(data - med))
        cutoff = med + mad_factor * mad
    elif method == "percentile":
        cutoff = np.percentile(data, percentile)
    elif method == 'negative_binomial':
        # Detect outliers based on negative binomial distribution
        mean, var = data.mean(), data.var(ddof=1)
        # Var = mean + mean^2 / r  ->  r = mean^2 / (var - mean)
        r = mean ** 2 / max(var - mean, 1e-6)
        p = r / (r + mean)
        cutoff = nbinom.ppf(1 - alpha, r, p)
    else:
        raise ValueError(f"Unknown method {method!r}")

    outliers = np.nonzero(data > cutoff)[0]
    outliers.astype(np.uint16)
    return outliers, cutoff


def filter_outlier_detections3d_edges_and_nodes(
    d3d_collection: Detections3D,
    pairs: np.ndarray,
    edge_weights: np.ndarray,
    outliers: np.ndarray
) -> Tuple[Detections3D, np.ndarray, np.ndarray]:
    """
    Remove outlier detections and any edges touching them.

    Parameters
    ----------
    d3d_collection : Detections3D
        Your per 3d detection summary values.
    pairs : (M, 2) int array
        Candidate edges as (i, j) indices into d3d_memory_bank.
    edge_weights : (M,) array
        Corresponding edge weights.
    outliers : (K,) int array
        Indices of rows in d3d_memory_bank to drop.

    Returns
    -------
    d3d_collection : Detections3D
        d3d_collection with outlier rows removed.
    filtered_pairs : (M', 2) int array
        pairs with any row in `outliers` removed.
    filtered_weights : (M',) array or None
        edge_weights sliced to match filtered_pairs.
    """
    # Filter out the detections themselves
    keep_mask = ~np.isin(np.arange(d3d_collection.pcd_ids.shape[0]), outliers)
    d3d_collection = filter_detections3d(d3d_collection, mask=keep_mask)

    # 2) Build a map from old indices to new indices (or -1 if dropped)
    new_index = np.full(d3d_collection.pcd_ids.shape[0], -1, dtype=int)
    new_index[keep_mask] = np.nonzero(keep_mask)[0]

    # 3) Remap pairs to the new indexing
    remapped = new_index[pairs]  # shape (M,2), values in [-1...]
    # 4) Keep only edges where both endpoints survived
    keep_edge = np.all(remapped >= 0, axis=1)
    pairs = remapped[keep_edge]

    # 5) Slice edge_weights if provided
    edge_weights = edge_weights[keep_edge]

    return d3d_collection, pairs, edge_weights


# ---------- utilities ----------
class UnionFind:
    def __init__(self, n):
        self.par  = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)
    def find(self, x):
        while self.par[x] != x:
            self.par[x] = self.par[self.par[x]]
            x = self.par[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return ra
        if self.rank[ra] < self.rank[rb]:
            self.par[ra] = rb
            return rb
        elif self.rank[rb] < self.rank[ra]:
            self.par[rb] = ra
            return ra
        else:
            self.par[rb] = ra
            self.rank[ra] += 1
            return ra
# ---------------------------------------

def pcc_strict_nondecreasing(
    num_nodes: int,
    pairs: np.ndarray,            # (M,2)
    supporters: np.ndarray,       # (M,)
    min_supporters: int = 2,
    quantiles: List[int] = (99, 95, 90, 80, 70, 60, 50),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCC with multiplicity-preserving supporter counters.
    The supporter count of any surviving edge never goes down.
    """
    uf = UnionFind(num_nodes)

    # --- 1. Initial Counter for each node -------------------------------
    Support = [Counter() for _ in range(num_nodes)]
    for (i, j), sup in zip(pairs, supporters):
        Support[i][j] += 1
        Support[j][i] += 1

    # Sorted edge order
    order = np.argsort(-supporters)
    pairs_sorted, supp_sorted = pairs[order], supporters[order]

    thresholds = [
        max(int(np.percentile(supporters, q)), min_supporters)
        for q in quantiles
    ]
    thresholds.append(min_supporters)

    current = 0            # pointer into sorted edge list
    active_edges = pairs_sorted        # edges still considered
    active_sup   = supp_sorted.copy()  # their current weights

    for thr in thresholds:
        # ---- 2. merge all edges with supp >= thr ----------------------
        while current < len(active_sup) and active_sup[current] >= thr:
            u, v = active_edges[current]
            ru, rv = uf.find(u), uf.find(v)
            if ru != rv:
                # union; root_new is representative
                root_new = uf.union(ru, rv)
                root_old = rv if root_new == ru else ru
                # merge Counters: new counts = sum (keeps multiplicity)
                Support[root_new] += Support[root_old]
                Support[root_old].clear()
            current += 1

        # ---- 3. re-score surviving edges for NEXT round -------------
        if thr == thresholds[-1]:
            break  # last iteration

        root_of = np.fromiter((uf.find(i) for i in range(num_nodes)), dtype=np.int32)

        # Keep only inter-cluster edges
        keep = root_of[active_edges[:, 0]] != root_of[active_edges[:, 1]]
        active_edges = active_edges[keep]
        active_sup   = active_sup[keep]

        # Recompute supporter counts (non-decreasing)
        new_sup = np.empty_like(active_sup)
        for idx, (u, v) in enumerate(active_edges):
            ru, rv = root_of[u], root_of[v]
            cu, cv = Support[ru], Support[rv]
            common = set(cu.keys()).intersection(cv.keys())
            s = sum(min(cu[k], cv[k]) for k in common)
            new_sup[idx] = s
        active_sup = new_sup

        # Sort edges for next threshold loop
        order2 = np.argsort(-active_sup)
        active_edges, active_sup = active_edges[order2], active_sup[order2]
        current = 0

    # ---- 4. final labels ---------------------------------------------
    roots = np.fromiter((uf.find(i) for i in range(num_nodes)), dtype=np.int32)
    uniq, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32), active_sup


def graph_clustering(
    num_nodes: int,
    pairs: np.ndarray,            # (M,2) int32
    edge_weights: np.ndarray,     # (M,)  float  or int
    method: Literal["leiden", "hcs", "pcc"] = "leiden",
    *,
    # PCC-specific
    min_supporters: int = 2,
    quantiles: List[int] = (90, 80, 70, 60, 50, 40, 30, 20, 10),
    # Leiden parameters
    leiden_resolution: float = 1.0,
) -> np.ndarray:
    """
    Parameters
    ----------
    num_nodes : total detections (rows in d3d_memory_bank)
    pairs      : edges (i,j)
    edge_weights : same length; for PCC must hold "supporter counts"
    method     : 'leiden' | 'hcs' | 'pcc'
    min_supporters: minimal number of supporting masks (detections 3d) needed for a mask merge
    quantiles: thresholds for "pcc"
    leiden_resolution: hyper-parameter steering resulting cluster sizes for Leiden
    Returns
    -------
    labels : (num_nodes,) int32 cluster id per detection ( -1 for isolated if HCS/PCC )
    """
    if method == "leiden":
        import igraph as ig, leidenalg as la
        # build igraph
        g = ig.Graph(n=num_nodes, edges=pairs.tolist(), edge_attrs={"weight": edge_weights})
        part = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=leiden_resolution,
        )
        labels = np.array(part.membership, dtype=np.int32)

    elif method == "hcs":
        # Simple Python implementation based on recursive min-cut with NetworkX
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_weighted_edges_from([(int(i), int(j), float(w)) for (i, j), w in zip(pairs, edge_weights)])

        label = -np.ones(num_nodes, dtype=np.int32)
        current_label = 0

        def recurse(subg: nx.Graph):
            nonlocal current_label
            if len(subg) == 0:
                return
            # min-cut size
            mc_value = nx.algorithms.connectivity.stoer_wagner(subg)[0]
            if mc_value > len(subg) / 2:
                # highly connected → assign label
                for node in subg.nodes:
                    label[node] = current_label
                current_label += 1
            else:
                # split at min-cut
                A, B = nx.algorithms.connectivity.stoer_wagner(subg)[1]
                recurse(subg.subgraph(A).copy())
                recurse(subg.subgraph(B).copy())

        recurse(G)
        labels = label.astype(np.int32)

    elif method == "pcc":
        labels = pcc_strict_nondecreasing(num_nodes, pairs, edge_weights, min_supporters, quantiles)
    else:
        raise ValueError(f"Unknown method {method}")

    return labels