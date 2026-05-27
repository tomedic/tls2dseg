import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

import logging

# Silence all tokenizers messages below ERROR
logging.getLogger("tokenizers").setLevel(logging.ERROR)

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pchandler.geometry import PointCloudData
from scipy.spatial.transform import Rotation as R

from src.tls2dseg.statistics_generalizable import multivariate_normal_outlier_removal
from tls2dseg.pc_preprocessing import apply_robust_sor_filter, main_cluster_extraction

# from tls2dseg.visualization import *


@dataclass
class Detections3D:
    pcd_ids: np.ndarray  # (N,)
    instances: np.ndarray  # (N,)
    classes: np.ndarray  # (N,)
    confidences: np.ndarray  # (N,)
    point_counts: np.ndarray  # (N,)
    centroids: np.ndarray  # (N,3)
    bboxes: np.ndarray  # (N,6) or (N,10)
    bboxes_type: Literal["aabb", "obb", "both"]
    centroid_type: Literal["mean", "median", "bbox_c"]
    preprocessing_applied: bool


def clean_pcd_instances_and_get_detections3d(
    pcd: PointCloudData, pcd_id: float, d3d_parameters: dict, pcp_parameters: dict
) -> (Detections3D, PointCloudData):
    """
    Extract a collection of detections 3d instances as a numpy array of relevant features / metadata accompanied
     by a list of feature names explaining columns of the numpy array

    Args:
        pcd: PointCloudData object
        pcd_id: integer tag identifying this point cloud within a project
        d3d_parameters: dictionary with parameters steering the process
        pcp_parameters: dictionary with parameters steering the process

    Returns:
        features: np.ndarray, shape (N_instances, 13) or (N_instances, 17), depending on bounding_box_type
        feature_names: list[str] of length 13 or 17

    Notes:
        Features: 1) pcd id; 2) class id; 3) confidence score; 4) nr of points; 5) centroid; 6) bounding box
    """

    # Minimum point number threshold for data processing and 3d detection estimation:
    min_npts = min([d3d_parameters["min_d3d_pcd_point_count"], 4])

    # Unpack (hyper-) parameters steering the process
    bounding_box_type = d3d_parameters["bounding_box_type"]
    centroid_type = d3d_parameters["centroid_type"]
    preprocess = d3d_parameters["preprocess"]

    # Create views to relevant point cloud data
    instances_pcd = pcd.scalar_fields["instances"]
    classes_pcd = pcd.scalar_fields["classes"]
    if pcp_parameters["keep_confidences"]:
        confidences_pcd = pcd.scalar_fields["confidence"]

    # Get unique identifiers of detected 3d objects (d3d = detection 3d) and the number of them
    unique_d3d = np.unique(instances_pcd).astype(np.uint32)
    N_d3d = len(unique_d3d)

    # pre-allocate empty list for cleaned instance point clouds
    pcds_clean = []

    # pre-allocate detections 3d features
    pcd_id_d3d = np.full(N_d3d, pcd_id, dtype=np.uint8)
    classes_d3d = np.zeros(N_d3d, dtype=np.uint8)
    confidence_d3d = np.ones(N_d3d, dtype=np.float16)
    pts_count_d3d = np.zeros(N_d3d, dtype=np.uint32)
    centroids_d3d = np.zeros((N_d3d, 3), dtype=np.float64)

    if bounding_box_type == "aabb":
        # Axis aligned bounding box [min_x,min_y,min_z, max_x,max_y,max_z]
        bbox_d3d = np.zeros((N_d3d, 6), dtype=np.float64)
    elif bounding_box_type == "obb":
        # Oriented bounding box: 3x centroid, 3x axis extent, 4x quaternions
        bbox_d3d = np.zeros((N_d3d, 10), dtype=np.float64)
    else:
        raise ValueError(
            f"bounding_box_type must be 'aabb' or 'obb', got {bounding_box_type} instead."
        )

    for i, uid in enumerate(unique_d3d):
        mask = instances_pcd == uid
        pcd_i = pcd.sample(mask)
        pts_i, npts_i = pcd_i.xyz.astype(dtype=float), pcd_i.xyz.shape[0]
        # class & confidence (assumed uniform per-instance)
        classes_d3d[i] = classes_pcd.data[mask][0]
        if pcp_parameters["keep_confidences"]:
            confidence_d3d[i] = confidences_pcd.data[mask][0]

        if preprocess and npts_i > min_npts:
            # Preprocess detection 3d instance point clouds

            # 1 - remove remaining outliers by "robustified SOR" filter (scaled MAD instead of std)
            apply_robust_sor_filter(pcd_i, k_neighbors=0.05, std_ratio=2)
            pts_i, npts_i = pcd_i.xyz.astype(dtype=float), pcd_i.xyz.shape[0]

            # 2 - retain only dominant point cluster
            # Expected point spacing (needed for DBSCAN, or HDBSCAN if epsilon_hdbscan != 0.0):
            #   - assumption - uniform density (oversimplification!), 10% noise on point position,
            #     searching for 26 neighbors (in the case of 3d voxel that would be faces, edges and corners of a voxel)
            # TODO: param. tuning (+ expected point spacing using: scan res, max dist, max. AOI 60° + noise, sine() )
            # TODO: alternatively to try: connected components, density peak clustering, ...
            if npts_i > min_npts:
                expected_point_spacing = pcp_parameters["output_resolution"] * np.sqrt(3) * 1.1
                clusterer_type = "hdbscan"  # 'dbscan', 'hdbscan'
                min_cluster_size = int(math.ceil(0.25 * npts_i))
                if clusterer_type == "hdbscan":
                    min_samples = int(math.ceil(0.005 * npts_i))
                elif clusterer_type == "dbscan":
                    min_samples = 5

                clusterer_definition = {
                    "type": "hdbscan",
                    "epsilon": expected_point_spacing,
                    "min_samples": min_samples,
                    "min_cluster_size": min_cluster_size,
                    "epsilon_hdbscan": 0.0,
                }

                mask = main_cluster_extraction(pcd_i.xyz, clusterer_definition)
                pcd_i.reduce(mask)
                pts_i, npts_i = pcd_i.xyz.astype(dtype=float), pcd_i.xyz.shape[0]

            # 4 - visual inspection of instance point cloud pre-processing
            # if uid == 150:
            #     pts_temp = pts_i + 100
            #     pts_i = pts_i[mask]
            #     plot_2_point_clouds(pts_temp, pts_i)

        # number of points
        pts_count_d3d[i] = npts_i

        # centroid
        if centroid_type == "mean":
            centroids_d3d[i] = c = pts_i.mean(axis=0)
        elif centroid_type == "median":
            centroids_d3d[i] = c = np.median(pts_i, axis=0)
        elif centroid_type == "bbox_c":
            pass
        else:
            raise ValueError(
                f"centroid_type must be 'mean', 'median' or 'bbox_c', got {centroid_type} instead"
            )

        if npts_i > min_npts:
            # axis-aligned bounding box
            if bounding_box_type == "aabb":
                # Axis aligned bounding box [min_x,min_y,min_z, max_x,max_y,max_z]
                mn = pts_i.min(axis=0)
                mx = pts_i.max(axis=0)
                bbox_d3d[i] = np.hstack((mn, mx))
                if centroid_type == "bbox_c":
                    centroids_d3d = (mx + mn) / 2
            elif bounding_box_type == "obb":
                # oriented bounding box OBB via PCA
                c = np.mean(pts_i, axis=0)  # get center of pts_i
                # compute covariance & eigen‐decomposition
                cov = np.cov(pts_i, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov)
                # sort by descending variance
                order = np.argsort(eigvals)[::-1]
                axes = eigvecs[:, order]  # defining PCA-frame (ordered PCA eigenvectors)
                # Assuring orthonormal:
                if np.linalg.det(axes) < 0:  # flip 3rd axis to make RH
                    axes[:, 2] *= -1

                pts_c = pts_i - c  # get centered points
                pts_pca = pts_c @ axes  # get points in PCA frame
                mn_p = pts_pca.min(axis=0)
                mx_p = pts_pca.max(axis=0)
                obb_extent = mx_p - mn_p  # get extent in PCA frame
                ctr_p = (mn_p + mx_p) / 2  # get center in PCA frame
                obb_center = c + axes @ ctr_p  # get center in pcd frame
                if centroid_type == "bbox_c":
                    centroids_d3d[i] = obb_center

                # Get quaternion from 3×3 matrix (axes stored column-wise)
                rotation = R.from_matrix(axes)
                quaternion = rotation.as_quat()  # [x, y, z, w]

                # Store values in bbox_3d3
                bbox_d3d[i, :3] = obb_center
                bbox_d3d[i, 3:6] = obb_extent
                bbox_d3d[i, 6:] = quaternion

                # store (optionally cleaned) instance point cloud in a list
                pcds_clean.append(pcd_i)

    # Remove "bad" detections from the collection (if point count too small and bboxes not estimated)
    keep_mask = ~np.all(bbox_d3d == 0, axis=1)  # Boolean mask of rows where all elements are 0
    pcd_id_d3d = pcd_id_d3d[keep_mask]
    unique_d3d = unique_d3d[keep_mask]
    classes_d3d = classes_d3d[keep_mask]
    confidence_d3d = confidence_d3d[keep_mask]
    pts_count_d3d = pts_count_d3d[keep_mask]
    centroids_d3d = centroids_d3d[keep_mask]
    bbox_d3d = bbox_d3d[keep_mask]

    # Create Detections3D object
    detections_3d = Detections3D(
        pcd_ids=pcd_id_d3d,
        instances=unique_d3d,
        classes=classes_d3d,
        confidences=confidence_d3d,
        point_counts=pts_count_d3d,
        centroids=centroids_d3d,
        bboxes=bbox_d3d,
        bboxes_type=bounding_box_type,
        centroid_type=centroid_type,
        preprocessing_applied=preprocess,
    )

    # Update the point cloud with clean instances (only valid ones)
    pcd = PointCloudData.merge_pcd(pcds_clean)
    if "merge_id" in pcd.scalar_fields.keys():
        pcd.scalar_fields.remove_field("merge_id")

    return detections_3d, pcd


def merge_detections3d(detections_list: list[Detections3D]) -> Detections3D:
    if len(detections_list) == 0:
        raise ValueError("detections_list must contain at least one Detections3D object")

    # Validate bbox_type consistency
    # TODO: Implement check if all values consistent across the list of Detections3D
    bbox_type = detections_list[0].bboxes_type
    centroid_type = detections_list[0].centroid_type
    preprocessing_applied = detections_list[0].preprocessing_applied

    # Stack each attribute vertically
    pcd_ids = np.concatenate([det.pcd_ids for det in detections_list])
    instances = np.concatenate([det.instances for det in detections_list])
    classes = np.concatenate([det.classes for det in detections_list])
    confidences = np.concatenate([det.confidences for det in detections_list])
    point_counts = np.concatenate([det.point_counts for det in detections_list])
    centroids = np.vstack([det.centroids for det in detections_list])
    bboxes = np.vstack([det.bboxes for det in detections_list])

    # Create Detections3D object
    detections_3d = Detections3D(
        pcd_ids=pcd_ids,
        instances=instances,
        classes=classes,
        confidences=confidences,
        point_counts=point_counts,
        centroids=centroids,
        bboxes=bboxes,
        bboxes_type=bbox_type,
        centroid_type=centroid_type,
        preprocessing_applied=preprocessing_applied,
    )
    return detections_3d


def filter_detections3d(detections: Detections3D, mask: np.ndarray) -> Detections3D:
    """
    Return a new Detections3D instance containing only the entries
    where `mask` is True.

    Args:
        detections: Detections3D to filter.
        mask: Boolean array of shape (N,) selecting which instances to keep.

    Returns:
        A filtered Detections3D with the same metadata fields but only
        entries corresponding to True in mask.

    Raises:
        ValueError: If mask is not boolean or its length does not match
                    the number of detections.
    """
    if mask.dtype != bool:
        raise ValueError(f"Expected boolean mask, got dtype {mask.dtype}")
    n = detections.pcd_ids.shape[0]
    if mask.shape[0] != n:
        raise ValueError(f"Mask length {mask.shape[0]} does not match number of detections {n}")

    return Detections3D(
        pcd_ids=detections.pcd_ids[mask],
        instances=detections.instances[mask],
        classes=detections.classes[mask],
        confidences=detections.confidences[mask],
        point_counts=detections.point_counts[mask],
        centroids=detections.centroids[mask, :],
        bboxes=detections.bboxes[mask, :],
        bboxes_type=detections.bboxes_type,
        centroid_type=detections.centroid_type,
        preprocessing_applied=detections.preprocessing_applied,
    )


def d3d_outlier_removal(
    d3d: Detections3D, per_class_separation: bool = False, confidence_interval: float = 0.99
) -> tuple[Detections3D, NDArray, NDArray]:

    # Number of detections in d3d:
    n_d3d = d3d.instances.size
    # Extract features used for detecting outliers:
    if d3d.bboxes_type == "aabb":
        features = np.zeros((n_d3d, 5), dtype=np.float32)
        features[:, 0] = d3d.centroids[:, 2]
        features[:, 1:4] = np.log1p(d3d.bboxes[:, 3:] - d3d.bboxes[:, :3])
        features[:, 4] = np.log1p(d3d.point_counts)
    elif d3d.bboxes_type == "obb":
        features = np.zeros((n_d3d, 11), dtype=np.float32)
        features[:, 0] = d3d.centroids[:, 2]
        features[:, 1:4] = np.log1p(d3d.bboxes[:, 3:6])
        # Compute Euler angles and encode in sin(2theta) cos(2theta)
        rot = R.from_quat(d3d.bboxes[:, 6:])
        eulers = rot.as_euler("xyz", degrees=False)
        s1, c1 = np.sin(2 * eulers[:, 0]), np.cos(2 * eulers[:, 0])
        s2, c2 = np.sin(2 * eulers[:, 1]), np.cos(2 * eulers[:, 1])
        s3, c3 = np.sin(2 * eulers[:, 2]), np.cos(2 * eulers[:, 2])
        features[:, 4:10] = np.column_stack((s1, c1, s2, c2, s3, c3))
        features[:, 10] = np.log1p(d3d.point_counts).squeeze()
    else:
        raise ValueError("Unsupported bounding-box type")

    # Get semantic classes (assumed 1 multivariate normal distribution of features per class)
    class_ids = d3d.classes
    # Collapse different classes (optional):
    if per_class_separation is False:
        class_ids = np.ones_like(class_ids)

    # Get unique classes:
    unique_cls = np.unique(class_ids)

    # Compute statistics and outliers per class:
    is_outlier = np.zeros_like(class_ids, dtype=np.bool_)
    nd = features.shape[1]  # get number of dimensions
    for cls in unique_cls:
        class_mask = class_ids == cls
        nr_samples = np.sum(class_mask)
        if nr_samples > nd * 10:
            features_i = features[class_mask, :]
            is_outlier_i = multivariate_normal_outlier_removal(features_i, confidence_interval)
            is_outlier[class_mask] = is_outlier_i
        else:
            print(
                f"Skipped statistical outlier removal for class {cls}: only {nr_samples} samples (<{nd * 10})."
            )

    # Get which instance in which point cloud is an outlier:
    pcd_or = d3d.pcd_ids[is_outlier]
    inst_or = d3d.instances[is_outlier]

    # Filter out 3d detections
    keep_mask = ~is_outlier
    d3d = filter_detections3d(d3d, keep_mask)

    # Return filtered Detections3D and arrays signalizing outlier instances in outlier point clouds:
    return d3d, pcd_or, inst_or


def squeeze_detections3d(d3d: Detections3D) -> None:
    d3d.pcd_ids = np.squeeze(d3d.pcd_ids)
    d3d.instances = np.squeeze(d3d.instances)
    d3d.classes = np.squeeze(d3d.classes)
    d3d.confidences = np.squeeze(d3d.confidences)
    d3d.point_counts = np.squeeze(d3d.point_counts)
    return None
