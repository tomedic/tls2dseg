"""Point cloud filtering and clustering helpers.

Phase 4 plan 05 Task 1 (ENG-07, D-D-01). Moved from ``tls2dseg.pc_preprocessing``
(filtering/clustering helpers only — ``save_segmented_pcd*`` stay in
``pipeline/run.py`` for Phase 5 per D-D-01).  Also includes
``statistics_generalizable.py`` functions used by outlier removal.

Heavy imports (pchandler, hdbscan, sklearn, scipy) are at module level because
every function here requires them — this module is tier_b_light only.
DO NOT import this from tier_a code.

DAG invariant (Pitfall 2 in RESEARCH.md §9): imports tls2dseg.types and
numpy/scipy/sklearn/pchandler only — never tls2dseg.engines.

Closest analog: ``tls2dseg.pc_preprocessing`` + ``tls2dseg.statistics_generalizable``.
"""

from __future__ import annotations

import logging
import math

import hdbscan
import numpy as np
from pchandler.geometry import PointCloudData
from pchandler.geometry.filters import BoxFilter, RangeFilter, VoxelDownsample
from scipy.stats import chi2
from sklearn.cluster import DBSCAN
from sklearn.covariance import MinCovDet
from sklearn.neighbors import NearestNeighbors

from tls2dseg.preprocessing.roi import roi_mask_xy_rectaware

logger = logging.getLogger("tls2dseg.preprocessing.cleanup")


# ---------------------------------------------------------------------------
# Bounding-box geometry helpers (from pc_preprocessing.py)
# ---------------------------------------------------------------------------


def get_all_bbox_corners_from_min_max_corners(minimum_corner: np.ndarray, maximum_corner: np.ndarray) -> np.ndarray:
    """Get all eight (8) corners of an AABB from 2 corners (min_xyz, max_xyz)."""
    all_bbox_corners = np.stack(
        np.meshgrid(*zip(minimum_corner, maximum_corner, strict=False), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    return all_bbox_corners


def get_min_max_corners_from_all_bbox_corners(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get min_xyz and max_xyz from all eight (8) corners of an AABB."""
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)
    return min_corner, max_corner


# ---------------------------------------------------------------------------
# Clustering (from pc_preprocessing.py)
# ---------------------------------------------------------------------------


def run_dbscan_hdbscan(data: np.ndarray, clusterer_definition: dict) -> np.ndarray:
    """Run DBSCAN or HDBSCAN; return per-point label array."""
    algorithm_type = clusterer_definition["type"]
    min_samples = int(clusterer_definition["min_samples"]) if clusterer_definition["min_samples"] else None
    cluster_selection_epsilon = clusterer_definition["epsilon_hdbscan"]
    if algorithm_type == "dbscan":
        epsilon = clusterer_definition["epsilon"]
        clusterer = DBSCAN(eps=epsilon, min_samples=min_samples)
    elif algorithm_type == "hdbscan":
        min_cluster_size = int(clusterer_definition["min_cluster_size"])
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            allow_single_cluster=True,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
    else:
        raise ValueError("Incorrect clusterer definition: clusterer_type invalid!")

    labels = clusterer.fit_predict(data[:, :3])
    return labels


def main_cluster_extraction(data: np.ndarray, clusterer_definition: dict) -> np.ndarray:
    """Run DBSCAN or HDBSCAN; return boolean mask for the largest cluster."""
    labels = run_dbscan_hdbscan(data, clusterer_definition)
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    mask = labels == largest_cluster_label
    return mask


def unsupervised_pcd_instance_segmentation(
    pcd: PointCloudData, pcp_parameters: dict, d3d_parameters: dict
) -> PointCloudData:
    """Unsupervised instance segmentation via DBSCAN/HDBSCAN on xyz."""
    xyz = pcd.xyz
    expected_point_spacing = pcp_parameters["output_resolution"] * np.sqrt(3) * 1.1
    clusterer_type = "dbscan"
    min_cluster_size = int(d3d_parameters["min_d3d_pcd_point_count"])
    min_samples = 8 if clusterer_type == "dbscan" else None
    clusterer_definition = {
        "type": "hdbscan",
        "epsilon": expected_point_spacing,
        "min_samples": min_samples,
        "min_cluster_size": min_cluster_size,
        "epsilon_hdbscan": 0.0,
    }
    labels = run_dbscan_hdbscan(data=xyz, clusterer_definition=clusterer_definition)
    labels += 1
    if labels.shape[0] == pcd.scalar_fields["instances"].data.shape[0]:
        pcd.scalar_fields["instances"] = np.squeeze(labels).astype(np.uint32)
    else:
        raise ValueError("Labels after DBSCAN/HDBSCAN clustering have wrong size/length")
    return pcd


# ---------------------------------------------------------------------------
# Statistical outlier removal (from pc_preprocessing.py +
# statistics_generalizable.py)
# ---------------------------------------------------------------------------


def robust_fit_multivariate_normal(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust Gaussian fit using Minimum Covariance Determinant."""
    mcd = MinCovDet().fit(data)
    mean = mcd.location_
    cov = mcd.covariance_
    return mean, cov


def get_mahalanobis_distance(data: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance for each data point."""
    diff = data - mean
    cov = cov + 1e-6 * np.eye(cov.shape[0])
    inv_cov = np.linalg.inv(cov)
    left_term = np.dot(diff, inv_cov)
    mh_dist = np.sqrt(np.sum(left_term * diff, axis=1))
    return mh_dist


def multivariate_normal_outlier_removal(data: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """Boolean outlier mask using Mahalanobis distance + chi2 threshold."""
    mean, cov = robust_fit_multivariate_normal(data)
    mh_dist = get_mahalanobis_distance(data, mean, cov)
    df = data.shape[1]
    chi2_val = chi2.ppf(q=quantile, df=df)
    is_outlier = mh_dist > np.sqrt(chi2_val)
    return is_outlier


def statistical_outlier_removal(data: np.ndarray, k: int = 10, std_ratio: float = 2.0) -> np.ndarray:
    """Statistical outlier removal; returns keep-mask (True = keep)."""
    from scipy import stats

    point_cloud = data[:, :3]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(point_cloud)
    distances, _ = nbrs.kneighbors(point_cloud)
    avg_distances = np.mean(distances[:, 1:], axis=1)
    mean_dist = np.median(avg_distances)
    std_dist = stats.median_abs_deviation(avg_distances) * 1.4826
    threshold = mean_dist + std_ratio * std_dist
    mask = avg_distances <= threshold
    return mask


def apply_robust_sor_filter(pcd: PointCloudData, k_neighbors: float | int, std_ratio: float | int) -> None:
    """Apply statistical outlier removal in-place to pcd."""
    pts = pcd.xyz
    if k_neighbors < 1:
        k_neighbors = math.ceil(pts.shape[0] * k_neighbors)
    k = int(k_neighbors)
    if pts.shape[0] > k:
        mask = statistical_outlier_removal(pts, k=k, std_ratio=float(std_ratio))
        pcd.reduce(mask)


# ---------------------------------------------------------------------------
# ROI + range filtering (from pc_preprocessing.py)
# ---------------------------------------------------------------------------


def filter_pcd_roi_range(pcd: PointCloudData, pcp_parameters: dict) -> None:
    """Remove points outside RoI and range limits in-place."""
    from pchandler.geometry.transforms import toggle_socs2prcs

    if pcp_parameters["range_limits"] is not None:
        range_limits = pcp_parameters["range_limits"]
        range_min, range_max = range_limits[0], range_limits[1]
        range_filter = RangeFilter(low=range_min, high=range_max)
        keep_mask = range_filter.mask(pcd)
        pcd.reduce(keep_mask)

    if pcp_parameters["roi_limits"] is not None:
        roi_limits = pcp_parameters["roi_limits"]
        roi_ndim = np.asarray(roi_limits).ndim
        if roi_ndim == 1:
            minimum_corner = np.asarray(roi_limits[:3])
            maximum_corner = np.asarray(roi_limits[3:])
            all_8_corners_PRCS = get_all_bbox_corners_from_min_max_corners(minimum_corner, maximum_corner)
            roi_pcd = PointCloudData(xyz=all_8_corners_PRCS)
            roi_pcd.transform(transformation_matrix=np.linalg.inv(pcd.tmat_socs2prcs))
            all_8_corners_SOCS = roi_pcd.xyz
            minimum_corner, maximum_corner = get_min_max_corners_from_all_bbox_corners(all_8_corners_SOCS)
            minimum_corner = tuple(minimum_corner.tolist())
            maximum_corner = tuple(maximum_corner.tolist())
            roi_filter = BoxFilter(minimum_corner=minimum_corner, maximum_corner=maximum_corner)
            keep_mask = roi_filter.mask(pcd)
            pcd.reduce(keep_mask)
        elif roi_ndim == 2 or roi_ndim == 3:
            roi_limits = np.asarray(roi_limits)
            if roi_ndim == 2:
                n_pts = pcd.xyz.shape[0]
                idx = np.linspace(0, n_pts - 1, num=min(1000, n_pts), dtype=int)
                pcd_small = pcd.sample(idx)
                pcd_small = toggle_socs2prcs(pcd_small)
                mean_z = np.median(pcd_small.xyz[:, 2])
                n_roi_vertices = roi_limits.shape[0]
                roi_limits = np.hstack((roi_limits, np.ones((n_roi_vertices, 1)) * mean_z))
            roi_pcd = PointCloudData(xyz=roi_limits)
            roi_pcd.transform(transformation_matrix=np.linalg.inv(pcd.tmat_socs2prcs))
            roi_limits_socs = roi_pcd.xyz
            roi_limits_socs = roi_limits_socs[:, :2]
            pcd_xy = pcd.xyz[:, :2]
            keep_mask = roi_mask_xy_rectaware(xy=pcd_xy, roi_xy=roi_limits_socs)
            pcd.reduce(keep_mask)
        else:
            logger.warning("Invalid 'roi_limits' in pcp_parameters.")


# ---------------------------------------------------------------------------
# Subsampling and instance management (from pc_preprocessing.py)
# ---------------------------------------------------------------------------


def subsample_pcd_to_output_resolution(pcd: PointCloudData, pcp_parameters: dict) -> PointCloudData:
    """Voxel-downsample pcd to the configured output resolution."""
    output_resolution = pcp_parameters["output_resolution"]
    if output_resolution is None:
        return pcd
    voxel_downsampler = VoxelDownsample(voxel_size=output_resolution, weigthing_method="nearest")
    pcd = voxel_downsampler.sample(pcd)
    return pcd


def remove_unclassified_points(pcd: PointCloudData, task_parameters: dict) -> PointCloudData:
    """Remove points with class==0 or instance==0 (background/unclassified)."""
    task = task_parameters["task"]
    if task == "object_detection":
        mask = np.logical_and(pcd.scalar_fields["classes"].data != 0, pcd.scalar_fields["instances"].data != 0)
        pcd.reduce(mask)
    else:
        raise ValueError(f"Unsupported task_parameter 'task', provided: {task}")
    return pcd


def remove_small_instances(pcd: PointCloudData, min_pts: float | int) -> PointCloudData:
    """Remove instances with fewer than min_pts points."""
    instances = pcd.scalar_fields["instances"].data
    _, inv_idx, counts = np.unique(instances, return_inverse=True, return_counts=True)
    mask = counts[inv_idx] > min_pts
    pcd.reduce(mask)
    return pcd


def color_pcd_instances_by_random(pcd: PointCloudData) -> None:
    """Assign a random RGB color per unique instance label in-place."""
    instances = pcd.scalar_fields["instances"].data
    unique_ids = np.unique(instances)
    colors_for_ids = np.random.randint(low=0, high=256, size=(unique_ids.shape[0], 3), dtype=np.uint8)
    id_to_index = {uid: idx for idx, uid in enumerate(unique_ids)}
    colored = np.empty((instances.shape[0], 3), dtype=np.uint8)
    for i, inst in enumerate(instances):
        colored[i] = colors_for_ids[id_to_index[inst]]
    PointCloudData.set_color(pcd, colored)
