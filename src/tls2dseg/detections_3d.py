import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)

import logging

# Silence all tokenizers messages below ERROR
logging.getLogger("tokenizers").setLevel(logging.ERROR)

import numpy as np
from scipy.spatial.transform import Rotation as R
from pchandler.geometry import PointCloudData
from typing import Tuple, Optional, Union, List, Literal
from tls2dseg.pc_preprocessing import main_cluster_extraction, statistical_outlier_removal
import math
from dataclasses import dataclass


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
    bboxes_type: Literal['aabb', 'obb', 'both']
    centroid_type: Literal['mean', 'median', 'bbox_c']
    preprocessing_applied: bool


def get_detections3d(pcd: PointCloudData, pcd_id: float, d3d_parameters: dict, pcp_parameters: dict) -> Detections3D:
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

    # Unpack (hyper-) parameters steering the process
    bounding_box_type = d3d_parameters["bounding_box_type"]
    centroid_type = d3d_parameters["centroid_type"]
    preprocess = d3d_parameters["preprocess"]

    # Create views to relevant point cloud data
    instances_pcd = pcd.scalar_fields['instances']
    classes_pcd = pcd.scalar_fields['classes']
    if pcp_parameters["keep_confidences"]:
        confidences_pcd = pcd.scalar_fields['confidence']
    pts_pcd = pcd.xyz  # (P,3)

    # Get unique identifiers of detected 3d objects (d3d = detection 3d) and the number of them
    unique_d3d = np.unique(instances_pcd).astype(np.uint32)
    N_d3d = len(unique_d3d)

    # pre-allocate
    pcd_id_d3d = np.full((N_d3d, 1), pcd_id, dtype=np.uint8)
    classes_d3d = np.zeros((N_d3d, 1), dtype=np.uint8)
    confidence_d3d = np.ones((N_d3d, 1), dtype=np.float16)  # TODO: consider x100 and replace with uint8
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
        classes_d3d[i, 0] = classes_pcd.data[mask][0]
        if pcp_parameters["keep_confidences"]:
            confidence_d3d[i, 0] = confidences_pcd.data[mask][0]

        if preprocess and pts_i.shape[0] > 3:
            # Preprocess detection 3d instance point clouds
            # 1 - retain only dominant point cluster
            # Expected point spacing (needed for DBSCAN, or HDBSCAN if epsilon_hdbscan != 0.0):
            #   - assumption - uniform density (oversimplification!), 10% noise on point position,
            #     searching for 26 neighbors (in the case of 3d voxel that would be faces, edges and corners of a voxel)
            # TODO: param. tuning (+ expected point spacing using: scan res, max dist, max. AOI 60° + noise, sine() )
            # TODO: alternatively to try: connected components, density peak clustering, ...

            expected_point_spacing = pcp_parameters["output_resolution"] * np.sqrt(3) * 1.1
            min_samples = int(math.ceil(0.005 * pts_i.shape[0]))
            min_cluster_size = int(math.ceil(0.25 * pts_i.shape[0]))
            clusterer_definition = {'type': 'hdbscan', 'epsilon': expected_point_spacing, 'min_samples': min_samples,
                                    'min_cluster_size': min_cluster_size,
                                    'epsilon_hdbscan': 0.0}

            mask = main_cluster_extraction(pts_i, clusterer_definition)
            # TODO: REMOVE THIS VISUAL INSPECTION AND TUNING TOOL
            # if uid == 150:
            #     pts_temp = pts_i + 100
            #     pts_i = pts_i[mask]
            #     plot_2_point_clouds(pts_temp, pts_i)

            # 2 - remove remaining outliers by "robustified SOR" filter (scaled MAD instead of std)
            k_neighbors = int(math.ceil(pts_i.shape[0] * 0.05))
            sor_parameters = {'k': k_neighbors, 'std_ratio': 3}  # Check meaning of parameters online
            if pts_i.shape[0] > sor_parameters['k']:
                mask = statistical_outlier_removal(pts_i, k=sor_parameters['k'],
                                                   std_ratio=sor_parameters['std_ratio'])
            pts_i = pts_i[mask]

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
            c = np.mean(pts_i, axis=0)  # get center of pts_i
            # compute covariance & eigen‐decomposition
            cov = np.cov(pts_i.T, bias=True)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # sort by descending variance
            order = np.argsort(eigvals)[::-1]
            axes = eigvecs[:, order]  # defining PCA-frame (ordered PCA eigenvectors)
            # Assuring orhonormal:
            u = axes[:, 0]  # first principal axis
            v = axes[:, 1]  # second principal axis
            w = np.cross(u, v)  # guaranteed right-handed
            w /= np.linalg.norm(w)  # re-normalize (just in case)
            axes = np.column_stack([u, v, w])

            pts_c = pts_i - c  # get centered points
            pts_pca = pts_c @ axes  # get points in PCA frame
            mn_p = pts_pca.min(axis=0)
            mx_p = pts_pca.max(axis=0)
            obb_extent = mx_p - mn_p  # get extent in PCA frame
            ctr_p = (mn_p + mx_p) / 2  # get center in PCA frame
            obb_center = c + axes @ ctr_p  # get center in pcd frame
            if centroid_type == 'bbox_c':
                centroids_d3d[i] = obb_center

            # Get quaternion from 3×3 matrix (axes stored column-wise)
            rotation = R.from_matrix(axes)
            quaternion = rotation.as_quat()  # [x, y, z, w]

            # Store values in bbox_3d3
            bbox_d3d[i, :3] = obb_center
            bbox_d3d[i, 3:6] = obb_extent
            bbox_d3d[i, 6:] = quaternion

    # Create Detections3D object
    detections_3d = Detections3D(pcd_ids=pcd_id_d3d, instances=np.expand_dims(unique_d3d, axis=1), classes=classes_d3d,
                                 confidences=confidence_d3d, point_counts=pts_count_d3d, centroids=centroids_d3d,
                                 bboxes=bbox_d3d, bboxes_type=bounding_box_type, centroid_type=centroid_type,
                                 preprocessing_applied=preprocess)

    return detections_3d


def merge_detections3d(detections_list: List[Detections3D]) -> Detections3D:
    if len(detections_list) == 0:
        raise ValueError("detections_list must contain at least one Detections3D object")

    # Validate bbox_type consistency
    # TODO: Implement check if all values consistent across the list of Detections3D
    bbox_type = detections_list[0].bboxes_type
    centroid_type = detections_list[0].centroid_type
    preprocessing_applied = detections_list[0].preprocessing_applied

    # Stack each attribute vertically
    pcd_ids = np.vstack([det.pcd_ids for det in detections_list])
    instances = np.vstack([det.instances for det in detections_list])
    classes = np.vstack([det.classes for det in detections_list])
    confidences = np.vstack([det.confidences for det in detections_list])
    point_counts = np.vstack([det.point_counts for det in detections_list])
    centroids = np.vstack([det.centroids for det in detections_list])
    bboxes = np.vstack([det.bboxes for det in detections_list])

    # Create Detections3D object
    detections_3d = Detections3D(pcd_ids=pcd_ids, instances=instances, classes=classes,
                                 confidences=confidences, point_counts=point_counts, centroids=centroids,
                                 bboxes=bboxes, bboxes_type=bbox_type, centroid_type=centroid_type,
                                 preprocessing_applied=preprocessing_applied)
    return detections_3d


def filter_detections3d(
        detections: Detections3D,
        mask: np.ndarray
) -> Detections3D:
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
        centroids=detections.centroids[mask],
        bboxes=detections.bboxes[mask],
        bboxes_type=detections.bboxes_type,
        centroid_type=detections.centroid_type,
        preprocessing_applied=detections.preprocessing_applied
    )
