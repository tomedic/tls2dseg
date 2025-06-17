import numpy as np
from typing import Tuple
from pchandler.geometry import PointCloudData
from pchandler.geometry.filters import RangeFilter, BoxFilter, VoxelDownsample, PointCloudFilter
from pchandler.data_io import save_ply
from pathlib import Path
import json



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








