from scipy.spatial.transform import Rotation as R
from pchandler.geometry import PointCloudData
from typing import Tuple, Optional, Union, List, Literal
import numpy as np
from scipy.spatial import cKDTree  # fast radius search :contentReference[oaicite:2]{index=2}
from scipy.sparse import coo_matrix, csr_matrix
from collections import Counter
import igraph as ig
import leidenalg as la
import trimesh
from concurrent.futures import ProcessPoolExecutor


def get_detections3d(pcd: PointCloudData, pcd_id: float, d3d_parameters: dict) -> Tuple[np.ndarray, list]:
    """
    Extract a collection of detections 3d instances as a numpy array of relevant features / metadata accompanied
     by a list of feature names explaining columns of the numpy array

    Args:
        pcd: PointCloudData object
        pcd_id: integer tag identifying this point cloud within a project
        d3d_parameters: dictionary with parameters steering the process

    Returns:
        features: np.ndarray, shape (N_instances, 13) or (N_instances, 17), depending on bounding_box_type
        feature_names: list[str] of length 13 or 17

    Notes:
        Features: 1) pcd id; 2) class id; 3) confidence score; 4) nr of points; 5) centroid; 6) bounding box
    """

    # Unpack (hyper-) parameters steering the process
    bounding_box_type = d3d_parameters["bounding_box_type"]
    centroid_type = d3d_parameters["centroid_type"]
    preprocess = d3d_parameters["preprocess"]

    # Create views to relevant point cloud data
    instances_pcd = pcd.scalar_fields['instances']
    classes_pcd = pcd.scalar_fields['classes']
    confidences_pcd = pcd.scalar_fields['confidences']
    pts_pcd = pcd.xyz  # (P,3)

    # Get unique identifiers of detected 3d objects (d3d = detection 3d) and the number of them
    unique_d3d = np.unique(instances_pcd)
    N_d3d = len(unique_d3d)

    # pre-allocate
    pcd_id_d3d = np.full((N_d3d, 1), pcd_id, dtype=np.uint8)
    classes_d3d = np.zeros((N_d3d, 1), dtype=np.uint8)
    confidence_d3d = np.zeros((N_d3d, 1), dtype=np.float16)  # TODO: consider x100 and replace with uint8
    pts_count_d3d = np.zeros((N_d3d, 1), dtype=np.uint32)
    centroids_d3d = np.zeros((N_d3d, 3), dtype=np.float32)

    if bounding_box_type == "aabb":
        # Axis aligned bounding box [min_x,min_y,min_z, max_x,max_y,max_z]
        bbox_d3d = np.zeros((N_d3d, 6), dtype=np.float32)
    elif bounding_box_type == "obb":
        # Oriented bounding box: 3x centroid, 3x axis extent, 4x quaternions
        bbox_d3d = np.zeros((N_d3d, 10), dtype=np.float32)
    else:
        raise ValueError(f"bounding_box_type must be 'aabb' or 'obb', got {bounding_box_type} instead.")

    for i, uid in enumerate(unique_d3d):
        mask = (instances_pcd == uid)
        pts_i = pts_pcd[mask]

        # class & confidence (assumed uniform per-instance)
        classes_d3d[i, 0] = classes_pcd[mask][0]
        confidence_d3d[i, 0] = confidences_pcd[mask][0]

        # number of points
        pts_count_d3d[i, 0] = mask.sum()

        # centroid
        if centroid_type == 'mean':
            centroids_d3d[i] = c = pts_i.mean(axis=0)
        elif centroid_type == 'median':
            centroids_d3d[i] = c = np.median(pts_i, axis=0)
        elif centroid_type == 'bbox_c':
            pass
        else:
            raise ValueError(f"centroid_type must be 'mean', 'median' or 'bbox_c', got {centroid_type} instead")

        # axis-aligned bounding box

        if bounding_box_type == "aabb":
            # Axis aligned bounding box [min_x,min_y,min_z, max_x,max_y,max_z]
            mn = pts_i.min(axis=0)
            mx = pts_i.max(axis=0)
            bbox_d3d[i] = np.hstack((mn, mx))
            if centroid_type == 'bbox_c':
                centroids_d3d = (mx + mn) / 2
        elif bounding_box_type == "obb":
            # oriented bounding box OBB via PCA
            # compute covariance & eigen‐decomposition
            cov = np.cov(pts_i.T, bias=True)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # sort by descending variance
            order = np.argsort(eigvals)[::-1]
            axes = eigvecs[:, order]  # defining PCA-frame (ordered PCA eigenvectors)
            pts_c = pts_i - c  # get centered points
            pts_pca = pts_c @ axes  # get points in PCA frame
            mn_p = pts_pca.min(axis=0)
            mx_p = pts_pca.max(axis=0)
            obb_extent = mx_p - mn_p  # get extent in PCA frame
            ctr_p = (mn_p + mx_p) / 2  # get center in PCA frame
            obb_center = c + axes @ ctr_p  # get center in pcd frame
            if centroid_type == 'bbox_c':
                centroids_d3d = obb_center

            # Get quaternion from 3×3 matrix (axes stored column-wise)
            rotation = R.from_matrix(axes)
            quaternion = rotation.as_quat()  # [x, y, z, w]

            # Store values in bbox_3d3
            bbox_d3d[i, :3] = obb_center
            bbox_d3d[i, 3:6] = obb_extent
            bbox_d3d[i, 6:] = quaternion

    # build feature_names in the same stacking order:
    feature_names = []
    feature_names += ['pcd_id', 'class', 'confidence', 'n_pts']
    feature_names += [f'c_{ax}' for ax in ('x', 'y', 'z')]
    if bounding_box_type == "aabb":
        feature_names += [f'aabb_min_{ax}' for ax in ('x', 'y', 'z')]
        feature_names += [f'aabb_max_{ax}' for ax in ('x', 'y', 'z')]
    elif bounding_box_type == "obb":
        feature_names += [f'obb_c_{ax}' for ax in ('x', 'y', 'z')]
        feature_names += [f'obb_ext_{ax}' for ax in ('x', 'y', 'z')]
        feature_names += [f'obb_quat_{ax}' for ax in ('x', 'y', 'z', 'w')]

    # horizontally stack into (N,13) if aabb, or (N,17) if obb
    features = np.hstack([
        pcd_id_d3d,
        classes_d3d,
        confidence_d3d,
        pts_count_d3d,
        centroids_d3d,
        bbox_d3d
    ])

    return features, feature_names


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


def compute_obb_iou_naive(
        centers: np.ndarray,
        extents: np.ndarray,
        quats: np.ndarray,
        pairs: np.ndarray,
        engine: str = "scad"
) -> np.ndarray:
    """
    Compute 3D IoU for oriented bounding boxes (OBBs) using trimesh boolean intersection.

    Parameters
    ----------
    centers : (N,3) array
        OBB centers.
    extents : (N,3) array
        Full box dimensions along each local axis.
    quats : (N,4) array
        Quaternions in [x, y, z, w] order.
    pairs : (M,2) int array
        Each row [i,j] indicates a pair of boxes.
    engine : str
        Boolean engine for trimesh (e.g. 'scad', 'blender', 'cork').

    Returns
    -------
    ious : (M,) float array
        IoU for each OBB pair.
    """
    M = pairs.shape[0]
    ious = np.zeros(M, dtype=np.float32)

    for idx, (i, j) in enumerate(pairs):
        # Build box meshes
        box1 = trimesh.creation.box(extents=extents[i])
        box2 = trimesh.creation.box(extents=extents[j])

        # Apply transforms (rotation + translation)
        R1 = R.from_quat(quats[i, :]).as_matrix()
        T1 = np.eye(4)
        T1[:3, :3] = R1
        T1[:3, 3] = centers[i]
        box1.apply_transform(T1)

        R2 = R.from_quat(quats[j, :]).as_matrix()
        T2 = np.eye(4)
        T2[:3, :3] = R2
        T2[:3, 3] = centers[j]
        box2.apply_transform(T2)

        # Compute volumes
        vol1 = box1.volume
        vol2 = box2.volume

        # Boolean intersection
        try:
            inter = trimesh.boolean.intersection([box1, box2], engine=engine)
            inter_vol = inter.volume if inter is not None else 0.0
        except BaseException:
            inter_vol = 0.0

        union_vol = vol1 + vol2 - inter_vol
        ious[idx] = inter_vol / union_vol if union_vol > 0 else 0.0

    return ious


def _compute_iou_from_packed(
        args: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]
) -> float:
    ext_i, trans_i, ext_j, trans_j, engine = args
    box1 = trimesh.creation.box(extents=ext_i, transform=trans_i)
    box2 = trimesh.creation.box(extents=ext_j, transform=trans_j)
    v1, v2 = box1.volume, box2.volume
    try:
        inter = trimesh.boolean.intersection([box1, box2], engine=engine)
        iv = inter.volume if inter else 0.0
    except Exception:
        iv = 0.0
    union = v1 + v2 - iv
    return iv / union if union > 0 else 0.0


def compute_obb_iou_parallel(
        centers: np.ndarray,
        extents: np.ndarray,
        quats: np.ndarray,
        pairs: np.ndarray,
        engine: str = "scad",
        max_workers: Optional[int] = None
) -> np.ndarray:
    """
    Compute 3D IoU for OBBs by pre-packing only the per-pair extents/transforms.

    Parameters
    ----------
    centers : (N,3)  OBB centers
    extents : (N,3)  OBB dimensions
    quats   : (N,4)  [x,y,z,w] quaternions
    pairs   : (M,2)  integer index pairs
    engine  : bool engine for trimesh
    max_workers : # of processes

    Returns
    -------
    ious : (M,) float IoU per pair
    """
    # 1) Precompute all transforms
    r = R.from_quat(quat=quats)
    rots = r.as_matrix()  # (N,3,3)
    N = centers.shape[0]
    transforms = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    transforms[:, :3, :3] = rots
    transforms[:, :3, 3] = centers

    # 2) Pack per-pair arguments
    pack_args: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []
    for (i, j) in pairs:
        pack_args.append((extents[i], transforms[i],
                          extents[j], transforms[j],
                          engine))

    # 3) Parallel map
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        ious = list(exe.map(_compute_iou_from_packed, pack_args))

    return np.array(ious, dtype=np.float32)


def compute_aabb_iou_vectorized(
        aabb: np.ndarray,
        pairs: np.ndarray
) -> np.ndarray:
    """
    Vectorized IoU for axis-aligned bounding boxes.

    Parameters
    ----------
    aabb : (N,6) array [minx,miny,minz,maxx,maxy,maxz]
    pairs : (M,2) int array of index pairs

    Returns
    -------
    ious : (M,) float32 IoU per pair
    """
    # gather aabb corners
    mins = aabb[:, :3]
    maxs = aabb[:, 3:]
    # sort them according to pairs
    mins_i = mins[pairs[:, 0]]
    mins_j = mins[pairs[:, 1]]
    maxs_i = maxs[pairs[:, 0]]
    maxs_j = maxs[pairs[:, 1]]
    # intersection aabb and corresponding volume
    inter_min = np.maximum(mins_i, mins_j)
    inter_max = np.minimum(maxs_i, maxs_j)
    inter_dim = np.clip(inter_max - inter_min, 0, None)
    inter_vol = inter_dim[:, 0] * inter_dim[:, 1] * inter_dim[:, 2]
    # volumes (per each box in a pair and intersection)
    vol_i = np.prod(maxs_i - mins_i, axis=1)
    vol_j = np.prod(maxs_j - mins_j, axis=1)
    union_vol = vol_i + vol_j - inter_vol
    # 3D IoU
    return np.where(union_vol > 0, inter_vol / union_vol, 0.0).astype(np.float32)


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


def compute_edge_weights(
        detections3d: np.ndarray,
        pairs: np.ndarray,
        iou_threshold: float = 0.3,
        mode: Literal['iou', 'supporters', 'both'] = 'both',
        boolean_engine: str = "scad",
        obb_workers: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute edge weights for sparse pairs: IoU and/or supporter counts.

    Parameters
    ----------
    detections3d : (N,D) array, with bboxes in cols 7:
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
    bboxes = detections3d[:, 7:]
    bbox_overlap = None
    supporter_counts = None

    # Compute 3D IoU (either for AABB or for OBB)
    # TODO: if statement left inside, because maybe I implement edge weights not based on 3D IoU in future
    # AABB
    if bboxes.shape[1] == 6:
        aabb = bboxes
        if mode in ('iou', 'both', 'supporters'):
            bbox_overlap = compute_aabb_iou_vectorized(aabb, pairs)
    # OBB
    elif bboxes.shape[1] == 10:
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


def cluster_objects(adj: csr_matrix,
                    min_cluster_size: int = 3,
                    method: str = "leiden") -> np.ndarray:
    """
    Cluster adjacency → new object IDs
    Returns cluster label per row (-1 if filtered).
    """
    if method == "leiden":
        g = ig.Graph.Adjacency((adj > 0).tolist(), mode="UNDIRECTED")
        g.es["weight"] = adj.data
        part = la.find_partition(g, la.ModularityVertexPartition, weights="weight")
        labels = np.array(part.membership, dtype=np.int32)
    elif method == "louvain":
        import networkx as nx
        G = nx.from_scipy_sparse_array(adj, edge_attribute="weight")
        labels_dict = nx.algorithms.community.louvain_communities(
            G, weight="weight", resolution=1.0)
        labels = -np.ones(adj.shape[0], dtype=np.int32)
        for cid, comm in enumerate(labels_dict):
            labels[np.fromiter(comm, dtype=int)] = cid
    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        # need dense distance – OK for <50 k nodes
        D = adj.max() - adj.toarray()
        np.fill_diagonal(D, 0)
        model = AgglomerativeClustering(
            affinity="precomputed", linkage="average", distance_threshold=0.5,
            n_clusters=None)
        labels = model.fit_predict(D)
    else:
        raise ValueError(f"Unknown clustering method {method!r}")

    # prune tiny clusters
    counts = Counter(labels)
    valid_lbls = {lbl for lbl, c in counts.items() if c >= min_cluster_size and lbl != -1}
    labels = np.where(np.isin(labels, list(valid_lbls)), labels, -1)

    return labels


# -------------------------------------------------
# 3. Convenience wrapper that returns the mapping
# -------------------------------------------------
def fuse_instances(memory_bank: np.ndarray,
                   radius: float,
                   k_max: int | None = None,
                   semantic_gate: bool = True,
                   min_cluster_size: int = 3,
                   method: str = "leiden"):
    """
    Parameters
    ----------
    memory_bank : ndarray
        (N, D) stacked feature table, must contain uid (col 0), class (2), centroid xyz (5:8)
    radius : float
        Centroid radius (same units as xyz) for edge proposals.
    """
    adj = build_adjacency(memory_bank, radius, k_max, semantic_gate)
    labels = cluster_objects(adj, min_cluster_size, method)
    uid2new = dict(zip(memory_bank[:, 0].astype(np.uint32), labels.astype(np.int32)))
    return uid2new, labels
