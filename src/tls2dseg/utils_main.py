from pathlib import Path
from pchandler.geometry import PointCloudData
from pchandler.geometry.transforms import lazy_global_shift_change
import numpy as np
from numpy.typing import NDArray
from tls2dseg.pc_preprocessing import subsample_pcd_to_output_resolution
from tls2dseg.detections_3d import Detections3D
from tls2dseg.pcd_collection import SegPCDCollection
import pickle


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
    output_dir_segmented_pcds = output_dir_intermediate / Path("segmented_point_clouds")
    stage1_output_dir = output_dir_intermediate / Path("stage_1_results")

    # Store them in inference_models_parameters for later processing
    inference_models_parameters['output_dir_masks_json'] = output_dir_masks_json
    inference_models_parameters['output_dir_od'] = object_detection_output_dir
    inference_models_parameters['output_dir_sam2'] = sam2_output_dir
    inference_models_parameters['output_dir_segmented_pcds'] = output_dir_segmented_pcds
    inference_models_parameters['stage1_output_dir'] = stage1_output_dir

    # Make directories (if not existing)
    object_detection_output_dir.mkdir(parents=True, exist_ok=True)
    sam2_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_masks_json.mkdir(parents=True, exist_ok=True)
    output_dir_segmented_pcds.mkdir(parents=True, exist_ok=True)
    stage1_output_dir.mkdir(parents=True, exist_ok=True)

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


def get_segmented_and_merged_point_cloud(pcd_collection: SegPCDCollection, d3d: Detections3D,
                                         clusters_ids: NDArray, pcp_parameters: dict) -> PointCloudData:
    # Unpack necessary values
    pcd_ids = d3d.pcd_ids
    instances = d3d.instances
    # Find all point clouds with valid 3d detection objects
    unique_seg_pcds = np.unique(pcd_ids)

    # Leave for future work
    # classes = d3d_collection.classes
    # TODO: implement classes weighted majority voting based on number of points and confidence for getting class
    #  label per instance (in case they are allowed to have different class labels), optionally - resolve
    #  only instances now, classes in a point cloud form by majority voting

    pcd_all = None
    clusters_ids = np.expand_dims(clusters_ids, axis=1)

    for seg_pcd_ij_id in unique_seg_pcds:
        # Direct mapping between old and new instance labels of Detections3D (1-to-1 correspondence):
        old_instances_ij = instances[pcd_ids == seg_pcd_ij_id]
        new_instances_ij = clusters_ids[pcd_ids == seg_pcd_ij_id]
        # View to per point in PointCloudData instance label:
        seg_pcd_ij_inst_old = pcd_collection.seg_pcds[seg_pcd_ij_id].scalar_fields["instances"].data

        # Remove pcd points that are not related to any d3d instances:
        keep_mask = np.in1d(seg_pcd_ij_inst_old, old_instances_ij)
        if np.any(~keep_mask):
            pcd_collection.seg_pcds[seg_pcd_ij_id].reduce(keep_mask)
        # Refresh view to per point in PointCloudData instance label:
        seg_pcd_ij_inst_old = pcd_collection.seg_pcds[seg_pcd_ij_id].scalar_fields["instances"].data

        # Mapping old instance labels between Detections3D and PointCloudData:
        inst_value_to_index = {val: idx for idx, val in enumerate(old_instances_ij)}
        inst_map_d3d_to_pcd = np.fromiter((inst_value_to_index[val] for val in seg_pcd_ij_inst_old), dtype=int)

        # Create and assign new instance labels for PointCloudData:
        seg_pcd_i_inst_new = new_instances_ij[inst_map_d3d_to_pcd]
        seg_pcd_i_inst_new = seg_pcd_i_inst_new.astype(np.uint32)
        pcd_collection.seg_pcds[seg_pcd_ij_id].scalar_fields["instances"] = np.squeeze(seg_pcd_i_inst_new)

    pcd_all = PointCloudData.merge_pcd(pcd_collection.seg_pcds)

    # Subsample point cloud to desired output resolution (once images generated):
    pcd_all = subsample_pcd_to_output_resolution(pcd_all, pcp_parameters)

    if "merge_id" in pcd_all.scalar_fields.keys():
        pcd_all.scalar_fields.remove_field("merge_id")

    return pcd_all


def small_cluster_removal(clusters_ids, d3d_parameters) -> np.ndarray:
    threshold = d3d_parameters["small_cluster_removal_threshold"]
    # Find unique values and their counts
    unique_ids, count_ids = np.unique(clusters_ids, return_counts=True)
    # Identify too small clusters
    small_clusters = unique_ids[count_ids <= threshold]
    # Build mask of positions to zero out
    mask = np.isin(clusters_ids, small_clusters)
    # Zero out small clusters and return
    cluster_ids_new = clusters_ids.copy()
    cluster_ids_new[mask] = 0
    return cluster_ids_new


def id_from_path(p: Path) -> int:
    # assumes names like d3d_ij_123.pkl → 123
    return int(p.stem.split("_")[-1])


def load_previously_saved_inference_results_if_any(pcd_i_id, pcd_collection, pcd_map, d3d_collection,
                                                   d3d_map) -> (SegPCDCollection, list, bool):
    # Initial load flag:
    load_flag = False

    # IDs expected for this point cloud (block of n_features):
    nF = pcd_collection.n_features
    start_id = (pcd_i_id - 1) * nF + 1
    end_id = pcd_i_id * nF
    expected_ids = list(range(start_id, end_id + 1))

    # Check if all results already computed:
    have_all = all((i in d3d_map) and (i in pcd_map) for i in expected_ids)

    # If True - load corresponding segmented PointCloudData and Detections3D
    if have_all:
        # Load cached results
        for id_ij in expected_ids:
            try:
                # Load PointCloudData
                with open(pcd_map[id_ij], "rb") as f:
                    pcd_collection.seg_pcds[id_ij-1] = pickle.load(f)
                # Load Detections3D
                with open(d3d_map[id_ij], "rb") as f:
                    d3d_loaded = pickle.load(f)
                d3d_collection.append(d3d_loaded)
                # Point cloud loaded
                load_flag = True
            except:
                print(f"Failed loading previously computed results of {pcd_i_id}th pcd,"
                      f" running inference again.")

    return pcd_collection, d3d_collection, load_flag
