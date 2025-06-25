from pathlib import Path
from pchandler.geometry import PointCloudData
from pchandler.geometry.transforms import lazy_global_shift_change
import numpy as np
from tls2dseg.pc_preprocessing import subsample_pcd_to_output_resolution
from tls2dseg.detections_3d import Detections3D


def make_output_folders(output_dir_pathlib: Path, image_generation_parameters: dict,
                        inference_models_parameters: dict) -> None:
    # Set intermediate results directory (parent)
    output_dir_intermediate = output_dir_pathlib / "intermediate"  # Create output_dir for intermediate results

    # Set image generation result directory (children)
    image_generation_output_dir = output_dir_intermediate / "images"
    # Store it in image_generation_parameters for later processing
    image_generation_parameters["output_dir_images"] = image_generation_output_dir
    # Make directory if not existing
    image_generation_output_dir.mkdir(parents=True, exist_ok=True)

    # Repeat for: object detection results (.png), SAM2 results (.png) SAM2 masks (.json),
    #             per-station point clouds (.ply)
    object_detection_output_dir = output_dir_intermediate / Path("object_detection")
    sam2_output_dir = output_dir_intermediate / Path("sam2")
    output_dir_masks_json = output_dir_intermediate / Path("masks_json")
    output_dir_socs_pcds = output_dir_intermediate / Path("socs_point_clouds")

    # Store them in inference_models_parameters for later processing
    inference_models_parameters['output_dir_masks_json'] = output_dir_masks_json
    inference_models_parameters['output_dir_od'] = object_detection_output_dir
    inference_models_parameters['output_dir_sam2'] = sam2_output_dir
    inference_models_parameters['output_dir_socs_pcds'] = output_dir_socs_pcds

    # Make directories (if not existing)
    object_detection_output_dir.mkdir(parents=True, exist_ok=True)
    sam2_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_masks_json.mkdir(parents=True, exist_ok=True)
    output_dir_socs_pcds.mkdir(parents=True, exist_ok=True)

    return None


def assure_common_global_shift(pcd_i: PointCloudData, common_global_shift: np.ndarray,
                               point_cloud_id) -> PointCloudData:
    # Assure common global shift for further operations!
    if pcd_i.global_coordinate_shift is None:
        pcd_i_global_shift = np.zeros((3,), dtype=np.float_)
    else:
        pcd_i_global_shift = pcd_i.global_coordinate_shift

    if np.any(common_global_shift != pcd_i_global_shift) and point_cloud_id == 1:
        common_global_shift = pcd_i_global_shift

    if np.any(common_global_shift > 0.0):
        # Heavy (but certainly working) global shift change:
        # translate(pcd, translation=-common_global_shift)

        # Light/lazy (but questionable) global shift change:
        pcd_i = lazy_global_shift_change(pcd_i, common_global_shift)

    return pcd_i, common_global_shift


def get_segmented_and_merged_point_cloud(pcd_ij_collection: list, d3d_collection: Detections3D,
                                         clusters_ids: np.ndarray, pcp_parameters: dict) -> PointCloudData:
    # Unpack necessary values
    pcd_ids = d3d_collection.pcd_ids
    instances = d3d_collection.instances

    # Leave for future work
    # classes = d3d_collection.classes
    # TODO: implement classes weighted majority voting based on number of points and confidence for getting class
    #  label per instance (in case they are allowed to have different class labels), optionally - resolve
    #  only instances now, classes in a point cloud form by majority voting

    for ij, pcd_ij in enumerate(pcd_ij_collection, start=1):

        # Direct mapping between old and new instance labels of Detections3D (1-to-1 correspondence):
        clusters_ids = np.expand_dims(clusters_ids, axis=1)
        old_instances_ij = instances[pcd_ids == ij]
        new_instances_ij = clusters_ids[pcd_ids == ij]
        # View to per point in PointCloudData instance label:
        pcd_instances_ij_old = pcd_ij.scalar_fields["instances"].data

        # Mapping old instance labels between Detections3D and PointCloudData:
        inst_value_to_index = {val: idx for idx, val in enumerate(old_instances_ij)}
        inst_map_d3d_to_pcd = np.fromiter((inst_value_to_index[val] for val in pcd_instances_ij_old), dtype=int)

        # Create and assign new instance labels for PointCloudData:
        pcd_instances_ij_new = new_instances_ij[inst_map_d3d_to_pcd]
        pcd_ij.scalar_fields["instances"] = np.squeeze(pcd_instances_ij_new)

        if ij == 1:
            pcd_all = pcd_ij.copy()
        else:
            pcd_all = PointCloudData.merge_pcd([pcd_all, pcd_ij])
            # Subsample point cloud to desired output resolution (once images generated):
            pcd_all = subsample_pcd_to_output_resolution(pcd_all, pcp_parameters)

        if "merge_id" in pcd_all.scalar_fields.keys():
            pcd_all.scalar_fields.remove_field("merge_id")

    return pcd_all
