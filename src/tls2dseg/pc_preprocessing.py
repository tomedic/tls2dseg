import json
import logging
import math
from pathlib import Path

import hdbscan
import numpy as np
from pchandler.data_io import save_ply
from pchandler.geometry import PointCloudData
from pchandler.geometry.filters import BoxFilter, RangeFilter, VoxelDownsample
from pchandler.geometry.transforms import toggle_socs2prcs
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from tls2dseg.roi_filter import roi_mask_xy_rectaware

logger = logging.getLogger("tls2dseg.pc_preprocessing")

# TODO: Separate functions operating on Nx3 np.ndarrays and on PointCloudData (above and below in the file)


def get_all_bbox_corners_from_min_max_corners(minimum_corner: np.ndarray, maximum_corner: np.ndarray) -> np.ndarray:
    # Gets all eight (8) corners of an axis-aligned bounding box (aabb) from 2 corners (min_xyz max_xyz)
    all_bbox_corners = np.stack(
        np.meshgrid(*zip(minimum_corner, maximum_corner, strict=False), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    return all_bbox_corners


def get_min_max_corners_from_all_bbox_corners(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Get min_xyz and max_xyz from all eight (8) corners of an axis-aligned bounding box (aabb)
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)
    return min_corner, max_corner


def run_dbscan_hdbscan(data: np.ndarray, clusterer_definition: dict) -> np.ndarray:
    # Run DBSCAN or HDBSCAN
    algorithm_type = clusterer_definition["type"]
    min_samples = int(clusterer_definition["min_samples"]) if clusterer_definition["min_samples"] else None
    cluster_selection_epsilon = clusterer_definition["epsilon_hdbscan"]
    if algorithm_type == "dbscan":
        epsilon = clusterer_definition["epsilon"]
        clusterer = DBSCAN(
            eps=epsilon, min_samples=min_samples
        )  # Adjust eps and min_samples based on your data_example
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
    # Run DBSCAN or HDBSCAN to cluster the data
    labels = run_dbscan_hdbscan(data, clusterer_definition)

    # Identify the largest cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    # Return Boolean Mask related to points of the largest cluster
    mask = labels == largest_cluster_label
    return mask


def unsupervised_pcd_instance_segmentation(
    pcd: PointCloudData, pcp_parameters: dict, d3d_parameters: dict
) -> PointCloudData:
    # Get data for clustering
    xyz = pcd.xyz

    # Set DBSCAN or HDBSCAN
    expected_point_spacing = pcp_parameters["output_resolution"] * np.sqrt(3) * 1.1
    clusterer_type = "dbscan"  # 'dbscan', 'hdbscan'
    min_cluster_size = int(d3d_parameters["min_d3d_pcd_point_count"])
    min_samples = 8 if clusterer_type == "dbscan" else None
    clusterer_definition = {
        "type": "hdbscan",
        "epsilon": expected_point_spacing,
        "min_samples": min_samples,
        "min_cluster_size": min_cluster_size,
        "epsilon_hdbscan": 0.0,
    }
    # Run HDBSCAN
    labels = run_dbscan_hdbscan(data=xyz, clusterer_definition=clusterer_definition)
    # By default, noisy labels are -1, valid clusters are 0,1,...,n_clusters -> push to positive integers
    labels += 1
    # Substitute instance labels with new ones
    if labels.shape[0] == pcd.scalar_fields["instances"].data.shape[0]:
        pcd.scalar_fields["instances"] = np.squeeze(labels).astype(np.uint32)
    else:
        raise ValueError("Labels after DBSCAN/HDBSCAN clustering have wrong size/length")
    return pcd


def statistical_outlier_removal(data: np.ndarray, k: int = 10, std_ratio: float = 2.0) -> np.ndarray:
    """
    Perform statistical outlier removal on a point cloud.
    Parameters:
    - point_cloud: numpy array of shape (n_points, 3), the input point cloud.
    - k: int, the number of neighbors to consider for each point.
    - std_ratio: float, the threshold for determining outliers based on standard deviation.
    Returns:
    - filtered_point_cloud: numpy array of the filtered point cloud.
    - outliers: numpy array of the points that were removed.
    """
    point_cloud = data[:, :3]
    # Compute the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(point_cloud)
    distances, _ = nbrs.kneighbors(point_cloud)

    # Exclude the distance to the point itself (first column)
    avg_distances = np.mean(distances[:, 1:], axis=1)
    mean_dist = np.median(avg_distances)
    std_dist = stats.median_abs_deviation(avg_distances) * 1.4826

    # Define a threshold to detect outliers
    threshold = mean_dist + std_ratio * std_dist

    # Filter points
    mask = avg_distances <= threshold
    # data_filtered = data[mask]
    # data_outliers = data[~mask]

    return mask


def filter_pcd_roi_range(pcd: PointCloudData, pcp_parameters: dict) -> None:
    # Remove points outside RoI and range limits (focus on relevant, speed up computations)

    # Filter ranges:
    if pcp_parameters["range_limits"] is not None:
        range_limits = pcp_parameters["range_limits"]
        range_min, range_max = range_limits[0], range_limits[1]  # min and max range
        range_filter = RangeFilter(low=range_min, high=range_max)
        keep_mask = range_filter.mask(pcd)
        pcd.reduce(keep_mask)

    # Filter ROI:
    if pcp_parameters["roi_limits"] is not None:
        roi_limits = pcp_parameters["roi_limits"]
        roi_ndim = np.asarray(roi_limits).ndim
        if roi_ndim == 1:
            # Transform roi_limits from PRCS to SOCS
            #   - take minimum and maximum corner
            minimum_corner = np.asarray(roi_limits[:3])
            maximum_corner = np.asarray(roi_limits[3:])
            #   - get all 8 corners of an axis aligned bounding box
            all_8_corners_PRCS = get_all_bbox_corners_from_min_max_corners(minimum_corner, maximum_corner)
            #   - numpy to PointCloudData object
            roi_pcd = PointCloudData(xyz=all_8_corners_PRCS)
            #   - apply transformation from PRCS_2_SOCS (inverse of SOCS_2_PRCS stored in pcd.transformation_matrix)
            roi_pcd.transform(transformation_matrix=np.linalg.inv(pcd.tmat_socs2prcs))
            #   - pointCloudData object to numpy
            all_8_corners_SOCS = roi_pcd.xyz
            #   - get new min and max corners in SOCS from all 8 corners
            minimum_corner, maximum_corner = get_min_max_corners_from_all_bbox_corners(all_8_corners_SOCS)
            #   - numpy to tuple
            minimum_corner = tuple(minimum_corner.tolist())
            maximum_corner = tuple(maximum_corner.tolist())
            # -reduce point cloud for roi:
            roi_filter = BoxFilter(minimum_corner=minimum_corner, maximum_corner=maximum_corner)
            keep_mask = roi_filter.mask(pcd)
            pcd.reduce(keep_mask)

        elif roi_ndim == 2 or roi_ndim == 3:
            # Transform ROI definition from PRCS to SOCS
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
            #   - apply transformation from PRCS_2_SOCS (inverse of SOCS_2_PRCS stored in pcd.transformation_matrix)
            roi_pcd.transform(transformation_matrix=np.linalg.inv(pcd.tmat_socs2prcs))
            #   - pointCloudData object to numpy
            roi_limits_socs = roi_pcd.xyz

            # Get ROI boolean mask (using 2d polygon definition, height ignored in current implementation)
            roi_limits_socs = roi_limits_socs[:, :2]
            pcd_xy = pcd.xyz[:, :2]
            keep_mask = roi_mask_xy_rectaware(xy=pcd_xy, roi_xy=roi_limits_socs)
            pcd.reduce(keep_mask)
        else:
            logger.warning("Invalid 'roi_limits' in pcp_parameters.")

    return None


def save_segmented_pcd_ij(
    pcd_path_pathlib: Path,
    pcd: PointCloudData,
    inference_models_parameters: dict,
    class_id_map: dict,
    image_j: tuple,
):
    # Save segmented point cloud and related transformation parameters
    logger.info("Saving Single-station point clouds")
    output_dir_socs_pcds = inference_models_parameters["output_dir_segmented_pcds"]
    feature_name = "_" + image_j[0]
    cs_name = "_prcs" if pcd.xyz_is_prcs else "_socs"
    output_pcd_name = pcd_path_pathlib.stem + feature_name + cs_name + "_seg.ply"  # _seg for segmented
    output_pcd_path = output_dir_socs_pcds / Path(output_pcd_name)
    save_ply(output_pcd_path, pcd, retain_colors=True, retain_normals=True, scalar_fields=None)

    # Save related transformation matrix (for local-to-global conversion)
    transformation_matrix_output_path = output_dir_socs_pcds / Path(pcd_path_pathlib.stem + "_socs2prcs_T.txt")
    np.savetxt(transformation_matrix_output_path, pcd.tmat_socs2prcs, fmt="%.8f", delimiter=" ")

    # Save class name - to - class id map in ascii
    class_id_map["background"] = 0
    inverted_map = {v: k for k, v in class_id_map.items()}
    inverted_map_path = output_dir_socs_pcds / Path("class_names_id_map.txt")
    with open(str(inverted_map_path), "w", encoding="ascii") as f:
        json.dump(inverted_map, f, ensure_ascii=True)

    return None


def save_segmented_pcd(data_dir: Path, out_dir: Path, pcd: PointCloudData, class_id_map: dict):
    # Save segmented point cloud and related transformation parameters
    logger.info("Saving point cloud")

    output_pcd_name = data_dir.name + "_segmented.ply"  # _seg for segmented
    output_pcd_path = out_dir / Path(output_pcd_name)

    save_ply(output_pcd_path, pcd, retain_colors=True, retain_normals=True, scalar_fields=None)

    # Save class name - to - class id map in ascii
    class_id_map["background"] = 0
    inverted_map = {v: k for k, v in class_id_map.items()}
    inverted_map_path = out_dir / Path("class_names_id_map.txt")
    with open(str(inverted_map_path), "w", encoding="ascii") as f:
        json.dump(inverted_map, f, ensure_ascii=True)
    return None


def subsample_pcd_to_output_resolution(pcd: PointCloudData, pcp_parameters: dict) -> PointCloudData:
    output_resolution = pcp_parameters["output_resolution"]
    if output_resolution is None:
        return pcd
    else:
        voxel_downsampler = VoxelDownsample(voxel_size=output_resolution, weigthing_method="nearest")
        pcd = voxel_downsampler.sample(pcd)
        return pcd


def remove_unclassified_points(pcd: PointCloudData, task_parameters: dict) -> PointCloudData:
    task = task_parameters["task"]
    if task == "object_detection":
        mask = np.logical_and(pcd.scalar_fields["classes"].data != 0, pcd.scalar_fields["instances"].data != 0)
        pcd.reduce(mask)
    else:
        raise ValueError(f"Unsupported task_parameter 'task', provided: {task}")
    return pcd


def remove_small_instances(pcd: PointCloudData, min_pts: float | int) -> PointCloudData:
    instances = pcd.scalar_fields["instances"].data
    # Per instance point counts (+ inverse index for mapping back)
    _, inv_idx, counts = np.unique(instances, return_inverse=True, return_counts=True)
    # Mask returning points of all instances that are bigger than a threshold
    mask = counts[inv_idx] > min_pts
    pcd.reduce(mask)
    return pcd


def color_pcd_instances_by_random(pcd: PointCloudData) -> None:
    """
    Assigns a random RGB color to point cloud points, one color per each unique instance in pcd.instances
    Parameters
    ----------
    pcd : PointCloudData
    """
    # Extract the per-point instance IDs
    instances = pcd.scalar_fields["instances"].data
    # Find all unique
    unique_ids = np.unique(instances)
    # Generate one random color per unique ID, RGB triplet of uint8
    colors_for_ids = np.random.randint(low=0, high=256, size=(unique_ids.shape[0], 3), dtype=np.uint8)

    # Build a lookup: instance ID -> row in colors_for_ids
    id_to_index = {uid: idx for idx, uid in enumerate(unique_ids)}

    # Map each point's instance to its color
    colored = np.empty((instances.shape[0], 3), dtype=np.uint8)
    for i, inst in enumerate(instances):
        colored[i] = colors_for_ids[id_to_index[inst]]

    # Apply to the point cloud
    PointCloudData.set_color(pcd, colored)
    # In-place point cloud change
    return None


def apply_robust_sor_filter(pcd: PointCloudData, k_neighbors: float | int, std_ratio: float | int) -> None:
    # Get point cloud points
    pts = pcd.xyz
    # Set knn (if provided as percentage of point cloud points)
    if k_neighbors < 1:
        k_neighbors = math.ceil(pts.shape[0] * k_neighbors)

    # statistical_outlier_removal requires k: int; k_neighbors above is float|int but
    # math.ceil() / a user passing an int gets us to int here.
    k = int(k_neighbors)
    if pts.shape[0] > k:
        mask = statistical_outlier_removal(pts, k=k, std_ratio=float(std_ratio))
        pcd.reduce(mask)

    return None
