import numpy as np
from typing import Tuple
from pchandler.geometry import PointCloudData
from pchandler.geometry.filters import RangeFilter, BoxFilter, VoxelDownsample, PointCloudFilter
from pchandler.data_io import save_ply
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R


def get_all_bbox_corners_from_min_max_corners(minimum_corner: np.ndarray, maximum_corner: np.ndarray) -> np.ndarray:
    # Gets all eight (8) corners of an axis-aligned bounding box (aabb) from 2 corners (min_xyz max_xyz)
    all_bbox_corners = np.stack(np.meshgrid(*zip(minimum_corner, maximum_corner), indexing='ij'), axis=-1) \
        .reshape(-1, 3)
    return all_bbox_corners

def get_min_max_corners_from_all_bbox_corners(corners: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    # Get min_xyz and max_xyz from all eight (8) corners of an axis-aligned bounding box (aabb)
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)
    return min_corner, max_corner


def filter_pcd_roi_range(pcd: PointCloudData, pcp_parameters: dict) -> None:
    # Remove points outside RoI and range limits (focus on relevant, speed up computations)

    # Filter ranges:
    if pcp_parameters['range_limits'] is not None:
        range_limits = pcp_parameters['range_limits']
        range_min, range_max = range_limits[0], range_limits[1]  # min and max range
        range_filter = RangeFilter(low=range_min, high=range_max)
        mask = range_filter.mask(pcd)
        pcd.reduce(mask)

    # Filter ROI:
    if pcp_parameters['roi_limits'] is not None:
        roi_limits = pcp_parameters['roi_limits']
        # Transform roi_limits from PRCS to SOCS
        #   - take minimum and maximum corner
        minimum_corner = np.asarray(roi_limits[:3])
        maximum_corner = np.asarray(roi_limits[3:])
        #   - get all 8 corners of an axis aligned bounding box
        all_8_corners_PRCS = get_all_bbox_corners_from_min_max_corners(minimum_corner, maximum_corner)
        #   - numpy to PointCloudData object
        roi_pcd = PointCloudData(xyz=all_8_corners_PRCS)
        #   - apply transformation from PRCS_2_SOCS (inverse of SOCS_2_PRCS stored in pcd.transformation_matrix)
        roi_pcd.transform(transformation_matrix=np.linalg.inv(pcd.transformation_matrix))
        #   - pointCloudData object to numpy
        all_8_corners_SOCS = roi_pcd.xyz
        #   - get new min and max corners in SOCS from all 8 corners
        minimum_corner, maximum_corner = get_min_max_corners_from_all_bbox_corners(all_8_corners_SOCS)
        #   - numpy to tuple
        minimum_corner = tuple(minimum_corner.tolist())
        maximum_corner = tuple(maximum_corner.tolist())
        # -reduce point cloud for roi:
        roi_filter = BoxFilter(minimum_corner=minimum_corner, maximum_corner=maximum_corner)
        mask = roi_filter.mask(pcd)
        pcd.reduce(mask)

    return None


def save_segmented_pcds_in_socs(pcd_path_pathlib: Path, pcd: PointCloudData, inference_models_parameters: dict,
                                class_id_map: dict, image_j: tuple):
    # Save segmented point cloud and related transformation parameters
    print("Saving Single-station point clouds")
    output_dir_socs_pcds = inference_models_parameters['output_dir_socs_pcds']
    feature_name = "_" + image_j[0]
    output_pcd_name = pcd_path_pathlib.stem + feature_name + "_seg.ply"  # _seg for segmented
    output_pcd_path = output_dir_socs_pcds / Path(output_pcd_name)
    save_ply(output_pcd_path, pcd, retain_colors=True, retain_normals=True, scalar_fields=None)

    # Save related transformation matrix (for local-to-global conversion)
    transformation_matrix_output_path = output_dir_socs_pcds / Path(pcd_path_pathlib.stem + '_T.txt')
    np.savetxt(transformation_matrix_output_path, pcd.transformation_matrix, fmt="%.8f", delimiter=" ")

    # Save class name - to - class id map in ascii
    class_id_map["background"] = 0
    inverted_map = {v: k for k, v in class_id_map.items()}
    inverted_map_path = output_dir_socs_pcds / Path('class_names_id_map.txt')
    with open(str(inverted_map_path), 'w', encoding='ascii') as f:
        json.dump(inverted_map, f, ensure_ascii=True)

    return None


def subsample_pcd_to_output_resolution(pcd: PointCloudData, pcp_parameters: dict) -> PointCloudData:
    output_resolution = pcp_parameters["output_resolution"]
    if output_resolution is None:
        return pcd
    else:
        voxel_downsampler = VoxelDownsample(voxel_size=output_resolution, weigthing_method='nearest')
        pcd = voxel_downsampler.sample(pcd)
        return pcd


def remove_unclassified_points(pcd: PointCloudData, task_parameters: dict) -> PointCloudData:
    task = task_parameters["task"]
    if task == "object_detection":
            mask = pcd.scalar_fields["classes"] != 0
            pcd.reduce(mask)
    else:
        raise ValueError(f"Unsupported task_parameter 'task', provided: {task}")

    return pcd


def get_detections3d(pcd: PointCloudData, pcd_id: float, d3d_parameters: dict) -> Tuple[np.ndarray, list]:
    """
    Extract per-instance features from a PointCloudData object.

    Args:
        pcd: PointCloudData with
             - pcd.scalar_fields['instances']  ->  (P,) int array
             - pcd.scalar_fields['classes']    ->  (P,) int array
             - pcd.scalar_fields['confidences']->  (P,) float array
             - pcd.xyz                         ->  (P,3) float array
        pcd_id: integer tag identifying this point cloud
        how: 'mean' or 'median' for centroid

    Returns:
        features: np.ndarray, shape (N_instances, 28)
        feature_names: list[str] of length 28
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
    classes_d3d = np.zeros((N_d3d, 1), dtype=np.uint16)
    confidence_d3d = np.zeros((N_d3d, 1), dtype=np.float16)  #TODO: consider x100 and replace with uint8
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
        else:
            raise ValueError(f"centroid_type must be 'mean' or 'median', got {centroid_type} instead")

        # axis-aligned bounding box

        if bounding_box_type == "aabb":
            # Axis aligned bounding box [min_x,min_y,min_z, max_x,max_y,max_z]
            mn = pts_i.min(axis=0)
            mx = pts_i.max(axis=0)
            bbox_d3d[i] = np.hstack((mn, mx))
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








