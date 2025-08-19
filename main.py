# Point Cloud Segmentation Using 2D point cloud representation and DL foundational models (e.g. GroundedSAM)
# Author: Tomislav Medic & ChatGPT, 22.04.2025
import gc
# TODO: THIS IS A QUICK-FIX -> remove and do everything properly for pip installable project!
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load libraries:
import numpy as np
import torch
import json
import warnings
import pickle

# Silence all warnings:
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import logging

logging.getLogger("pchandler").setLevel(logging.ERROR)
np.seterr(invalid='ignore')
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Normals, and colors are not retained during `voxel_downsample`!"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The key `labels` is will"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The given NumPy array is not writable"
)

# NICHOLAS:
from pathlib import Path
from pchandler.geometry import PointCloudData
from pchandler.data_io import load_e57, save_ply
from pchandler.geometry.transforms import toggle_socs2prcs

# TOMISLAV:
from src.tls2dseg.pc2img_utils import *
from src.tls2dseg.grounded_sam2 import *
from src.tls2dseg.utils_main import *
from src.tls2dseg.sam_everything import *
from src.tls2dseg.pc_preprocessing import *
from src.tls2dseg.parameters_check import *
from src.tls2dseg.detections_3d import *
from src.tls2dseg.graph_clustering import *
from src.tls2dseg.detections_2d import *
from src.tls2dseg.pcd_collection import *

# I/0 parameters:
# pcd_path = "./data/wheat_heads_small.e57"  # Set path to point cloud
# output_dir = "./results"  # Set path for storing the results
# save_intermediate_results = True  # Save intensity images, gDINO and SAM2 outputs

# Task & I/0 parameters:
task_parameters = {'input_path': "./data/wheat_heads/",  # Set path to input point clouds
                   'checkpoint': True,  # if checkpoint True, start from already generated 3d3
                   'file_format': "e57",
                   'output_path': "./results",  # Set path for storing the results
                   'save_intermediate_results': True,  # Save intensity images, gDINO and SAM2 outputs
                   'task': "object_detection",  # Task choice
                   'results_aggregation_strategy': "object_memory_bank",  # OUT type choice: memory bank vs. voxel-grid
                   'n_workers': 12,  # Number of workers for parallel processing, default choices: 1 or N_cpu_cores -1
                   'save_d2d': False,
                   }

# PointCloud processing parameters
# BAFU settings:
# pcp_parameters = {'output_resolution': 0.5,  # Subsample point cloud
#                   'range_limits': None,  # All points further then will be discarded
#                   'roi_limits': None,  # only region of interest (3D bounding box) is to be analyzed
#                   'keep_confidences': False  # keep confidence
#                   }

# Wheat-heads settings:
pcp_parameters = {'output_resolution': 0.003,  # Subsample point cloud
                  'range_limits': [0., 5.],  # All points further then will be discarded
                  'roi_limits': [-12.5, -0.8, 491.7, 0.3, 7.5, 493.7],
                  # only region of interest (3D bounding box) is to be analyzed
                  'keep_confidences': False,  # keep confidence
                  'assign_random_color_per_instance': True,
                  'flip_upsidedown_scans': 180,  # falsy -> no-flip, number -> flip about x-axis in deg
                  }

# Image generation parameters:

# image width - width in pixels (height adjusted according to aspect ratio); - accepts:
#               1) int - number of pixels,
#               2) str "scan_resolution" - matches scan resolution,
#               3) str e.g."0.2-scan_resolution" - a fraction, here 1/5, of scan resolution

# scan_resolution - point spacing in spherical coordinates (angular encoder increments during measurements) - accepts:
#               1) str "auto" - compute from point cloud data,
#               2) str of formant "X.Xmm@Ym", e.g. "1.6mm@10m", computes angles from there
#               3) float - angular increment in degrees

# rotate_pcd   - rotate point cloud around z-axis to change horizontal borders of spherical image
#                (e.g. if border cuts object/region of interest); - accepts:
#               1) str "auto" - compute from point cloud data to avoid cutting any object (if possible)
#               2) float - angular increment in degrees
#               3) bool False - if no rotation required

# rasterization_method - how to rasterize the point cloud / do interpolation (see pc2img library for details); accepts
#                        str "raw" (no interpolation), "nanconv", "bary_delaunay", "bary_knn"

# features - list of point cloud features used to generate images - default "intensity"

# TODO: extend to all hyperparameters (incl. ones hidden in pc2img_run() function)
image_generation_parameters = {'image_width': "scan_resolution",  # Scan-resolution
                               'scan_resolution': "auto",
                               'rotate_pcd': "auto",
                               'rasterization_method': 'nanconv',
                               'features': ["intensity", "range"],
                               }

# Prompt object detection / segmentation
# TODO: VERY important: text prompts need to be lowercase + end with a dot
# text_prompt = "house.window.bicycle.door.grass.pipe"
# text_prompt = "house.window.wall.door.roof.brick.ground.fence.person"
# text_prompt = "wall.ceiling.plants.plant pot.leaf.leaves.desk.chair.bag.table.keyboard.floor.window.monitor"
# text_prompt = "pine. pine tree"
# text_prompt = "rock.stone.boulder.cliff.tree.pine"
text_prompt = "wheat.wheat head.wheat ear.wheat spike.wheat stalk.leaf.stem"

# Inference model parameters:
inference_models_parameters = {'with_slice_inference': True,
                               'bbox_model_id': 'IDEA-Research/grounding-dino-base',
                               'box_threshold': 0.12,  # 0.35
                               'text_threshold': 0.12,  # 0.25
                               'sam2-model-config': 'configs/sam2.1/sam2.1_hiera_l.yaml',
                               'sam2-checkpoint': '/scratch/projects/sam2/checkpoints/sam2.1_hiera_large.pt',
                               'large_object_removal_threshold': 0.20,
                               'partial_detection_edge_touching_threshold': 5,
                               'sam_box_prompt_batch_size': 32}

# Additional parameters for slice inference (necessary only if inference with SAHI)
slice_inference_parameters = {'slice_width_height': (200, 200),
                              'overlap_width_height': (100, 100),
                              'iou_threshold': 0.80,
                              'overlap_filter_strategy': 'nms',
                              'empty_slice_removal_threshold': 0.95}

# Parameters defining how to get 3d objects from 2d detections + SAM2 masks
d3d_parameters = {'bounding_box_type': 'obb',
                  'centroid_type': 'bbox_c',
                  'preprocess': True,
                  'min_d3d_pcd_point_count': 25,
                  'merge_inst_of_same_class_only': False,
                  'sparse_connectivity_method': 'knn',  # Literal['knn','radius']
                  'sparse_connectivity_threshold': 1,  # 'knn' -> k of nn per scan; 'radius' -> nn radius [m]
                  'supporters_iou_threshold': 0.15,  # necessary IoU between 3d bbox for signif. overlap
                  'remove_outliers_by_support': True,  # remove Detections3D if too large support (under-segmented)
                  'outlier_detection_method': 'negative_binomial',  # "iqr","mad","percentile","negative_binomial"
                  'outlier_detection_threshold': 0.01,  # different for each method, see fun. description
                  'graph_clustering_method': 'hcs',  # 'leiden' | 'hcs' | 'pcc'
                  'min_supporters': 2,  # min. number of supporters necessary for a valid cluster
                  'leiden_resolution': 1,  # hyp.-p. for 'leiden' (<1 - fewer larger clusters, >1 vice versa)
                  'small_cluster_removal_threshold': 2,  # How many times a cluster has to appear to be accepted
                  }


def main():
    # 0. Initial Set-up
    # __________________________________________________________________________________________________________________

    # Check input parameters (Type and Value checks)
    check_all_parameters(task_parameters, pcp_parameters, image_generation_parameters,
                         inference_models_parameters, slice_inference_parameters, text_prompt)

    # Set output path
    output_dir = task_parameters['output_path']
    output_dir_pathlib = Path(task_parameters['output_path'])  # Create pathlib path

    # Create folders and set flags to True for intermediate results (if necessary)
    save_intermediate_results = task_parameters['save_intermediate_results']
    inference_models_parameters['dump_json_results'] = True if save_intermediate_results else False
    if save_intermediate_results:
        make_output_folders(output_dir_pathlib, image_generation_parameters, inference_models_parameters)

    # Check if some point clouds already processed (step 1/2):
    stage1_odir_partial = output_dir_pathlib / "intermediate" / "stage_1_partial"
    stage1_odir_partial.mkdir(parents=True, exist_ok=True)
    d3d_partial_paths = list(stage1_odir_partial.glob(f"d3d_ij_*.pkl"))
    pcd_partial_paths = list(stage1_odir_partial.glob(f"pcd_ij_*.pkl"))
    d3d_map = {id_from_path(p): p for p in d3d_partial_paths}
    pcd_map = {id_from_path(p): p for p in pcd_partial_paths}

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Set inference hardware
    inference_models_parameters['device'] = device  # Add to model parameters dictionary

    # Set other parameters
    slice_inference_parameters["thread_workers"] = task_parameters['n_workers']

    # Set the environment settings (this is necessary for SAM)
    # use bfloat16 where ok, otherwise float32
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize Grounded SAM2 (Grounded DINO + SAM2)
    gdino_model, gdino_processor = initialize_gdino(inference_models_parameters)
    sam2_predictor = initialize_sam2(inference_models_parameters)

    # Update hyper-parameters
    inference_models_parameters['large_object_removal_threshold'] = inference_models_parameters[
        'large_object_removal_threshold']

    # Get dictionary mapping "semantic classes" to unique IDs and vice versa (+ save in results)
    keys = text_prompt.split('.')  # → ['house','window','bicycle','door','grass','leaf']
    class_id_map = {k: i + 1 for i, k in enumerate(keys)}  # → {'house':1, 'window':2, ..., 'leaf':6}
    class_id_map["background"] = 0
    inverted_id_map = {v: k for k, v in class_id_map.items()}
    inverted_map_path = output_dir / Path('class_names_id_map.txt')
    with open(str(inverted_map_path), 'w', encoding='ascii') as f:
        json.dump(inverted_id_map, f, ensure_ascii=True)

    # 1. Point Cloud Processing
    # ______________________________________________________________________________________________________________

    # Find all point cloud files of defined "file_format" within data folder
    data_folder_path = Path(task_parameters["input_path"]).resolve()
    file_format = task_parameters["file_format"]
    pcd_file_paths = list(data_folder_path.glob(f"*.{file_format}"))
    n_scans = len(pcd_file_paths)

    # Generate SegPCDCollection object:
    features = image_generation_parameters['features']
    n_features = len(features)
    # Jump over segmentation if already fully solved
    checkpoint = task_parameters['checkpoint']

    have_processed_all = False
    if len(pcd_map) == n_scans * n_features and len(d3d_map) == n_scans * n_features:
        have_processed_all = True

    # Start the inference loop if no wish to start from checkpoint or not all pcds already processed:
    if not checkpoint or not have_processed_all:

        # Set-up results structure
        how_aggregate_results = task_parameters["results_aggregation_strategy"]
        pcd_collection = SegPCDCollection(raw_pcd_paths=pcd_file_paths, features=features,
                                          class_id_map=class_id_map)
        if how_aggregate_results == "object_memory_bank":
            d3d_collection = []
            if task_parameters['save_d2d']:
                d2d_collection = []
        else:
            raise ValueError("Chosen results_aggregation_strategy is currently not supported!")

        # Set common global shift for all point clouds (precaution, should not be necessary for small projects)
        common_global_shift = np.zeros((3,), dtype=np.float_)

        for pcd_i_id, pcd_path_i in enumerate(pcd_file_paths, start=1):

            # Check if some point clouds already processed (step 2/2) - pcd_i
            pcd_collection, d3d_collection, load_flag = load_previously_saved_inference_results_if_any(pcd_i_id,
                                                        pcd_collection, pcd_map, d3d_collection, d3d_map)

            # Skip the rest of the inference loop if load_flag is True:
            if load_flag:
                continue

            # Load data
            pcd: PointCloudData = load_e57(pcd_path_i, stay_prcs=False, save_prcs_info=True)  # Load point cloud

            # Filter point cloud for ranges and RoI (region of interest)
            filter_pcd_roi_range(pcd, pcp_parameters)

            # Optional: Detect if point cloud upside-down, if yes - flip for theta degrees:
            test_upsidedown, alpha_deg = False, 0.0
            if pcp_parameters["flip_upsidedown_scans"]:
                alpha_deg = pcp_parameters["flip_upsidedown_scans"]
                test_upsidedown = check_was_scanner_upsidedown(pcd)
                if test_upsidedown:
                    rotate_pcd_around_x(pcd, alpha_deg=alpha_deg)

            # Rotate point cloud around z (if necessary), return rotation angle theta in degrees
            theta_deg = resolve_rotate_pcd_parameter(pcd, image_generation_parameters)

            # Compute image dimensions (w and h) in pixels and scan resolution (azimuth and elevation) in radians
            image_width, image_height, d_azim_deg, d_elev_deg = compute_image_dimensions(pcd,
                                                                                         image_generation_parameters)
            print(f"Image height x width: {image_height} x {image_width}")

            # Resolve necessary image resolution
            reduction_coefficient = resolve_necessary_image_resolution(pcd, pcp_parameters, d_azim_deg)

            # Generate images of point cloud i
            print("Generating desired image(s)")
            images_of_pcd_i = pc2img_run(pcd, pcd_path_i, image_generation_parameters, image_width, image_height)

            # Reducing image resolution (if necessary),
            #   + getting rid of NaN values & casting image to float32 (if not already float32)
            if reduction_coefficient < 1:
                print("Reducing image resolution")
                images_of_pcd_i = reduce_image_resolution(images_of_pcd_i, reduction_coefficient,
                                                          image_generation_parameters, pcd_path_i)

            # TODO: Initializing and testing SAM2 everything -> nicely incorporate in the code (how:
            #  semantic segmentation on intensity + instance segmentation SAM2everything on range, combine)
            # Get new image width and height
            # image_height, image_width = images_of_pcd_i[0][1].shape
            # image_test = images_of_pcd_i[1][1]
            # sam2_everything = initialize_sam2_everyting(inference_models_parameters)
            # masks = run_sam2_everything(image_test, sam2_everything, inference_models_parameters)

            # Subsample point cloud to desired output resolution (once images generated):
            pcd = subsample_pcd_to_output_resolution(pcd, pcp_parameters)
            # Assure common global shift for further operations
            pcd, common_global_shift = assure_common_global_shift(pcd, common_global_shift, pcd_i_id)
            pcd_collection.global_shift = common_global_shift

            # 2. Inference: Instance + semantic segmentation -----------------------------------------------------------
            # Set segmented point cloud (pcd_ij) counter:
            pcd_ij_id = pcd_i_id * 2 - 2

            # Run Grounded SAM2 inference (for all images of a point cloud pcd_i)
            for j, image_j in enumerate(images_of_pcd_i):
                pcd_ij_id += 1
                image_j_numpy = image_j[1]

                # TODO: modify text description -> sparse masks!
                # results [dict]: 'masks' with M x w x h (M = mask number, w = width, h = height),
                #                  'input_boxes' with input boinding boxes,
                #                  'confidences' with confidence scores,
                #                  'class_names', class ids, ...

                if inference_models_parameters["with_slice_inference"] is True:
                    print("Grounded SAM2 - Inference on image slices")
                    results = run_grounded_sam2_with_sahi(image=image_j_numpy, text_prompt=text_prompt,
                                                          gdino_model=gdino_model,
                                                          gdino_processor=gdino_processor,
                                                          sam2_predictor=sam2_predictor,
                                                          inference_models_parameters=inference_models_parameters,
                                                          slice_inference_parameters=slice_inference_parameters)
                else:
                    print("Grounded SAM2 - Inference on a whole image")
                    results = run_grounded_sam2(image=image_j_numpy, text_prompt=text_prompt, gdino_model=gdino_model,
                                                gdino_processor=gdino_processor, sam2_predictor=sam2_predictor,
                                                inference_models_parameters=inference_models_parameters)

                # Get per-mask depths:
                get_per_mask_depth_parallel(results, images_of_pcd_i, n_jobs=task_parameters['n_workers'])

                # Filter outlier masks (optionally per class):
                results = d2d_outlier_removal(results, task_parameters, per_class_separation=False,
                                              confidence_interval=0.99)[0]

                # Save object detection (gdino) and segmentation (SAM2) results as .jpeg images and corresponding data in .json:
                if save_intermediate_results:
                    print("Saving intermediate results")
                    save_gsam2_results(image=images_of_pcd_i[j], results=results,
                                       inference_models_parameters=inference_models_parameters)

                # From individual per-object bool masks get:
                #   - 1 instance mask (each instance having one int ID),
                #   - 1 semantic mask (each class having one int ID),
                #   - class_id_map which maps semantic classes provided in text_prompt to semantic class IDs
                print("Getting unified instance and semantic mask from individual masks")
                # instance_mask, semantic_mask, class_id_map = get_instance_and_semantic_mask(results, text_prompt)
                instance_mask, semantic_mask, confidence_mask, class_id_map = \
                    get_instance_and_semantic_mask_with_confidence(results, text_prompt,
                                                                   image_hw=image_j_numpy.shape[:2])

                # Add the generated masks to ImageStack related to the point cloud pcd
                print("Lifting 2d masks to 3d")
                # Create a point cloud copy for further data processing:
                pcd_ij = pcd.copy()
                project_masks2pcd_as_scalarfields(pcd_ij, instance_mask, semantic_mask)
                if pcp_parameters["keep_confidences"]:
                    project_a_mask_2_pcd_as_scalarfield(pcd_ij, mask=confidence_mask, mask_name="confidence")

                del instance_mask, semantic_mask, confidence_mask
                gc.collect()

                # Remove background class (if task = object detection)
                pcd_ij = remove_unclassified_points(pcd_ij, task_parameters)

                # Remove too small object detections
                pcd_ij = remove_small_instances(pcd_ij, d3d_parameters)

                # Transform point cloud to global (project-related) coordinate system
                #   first correct for rotation theta_deg used for more efficient spherical image generation
                if theta_deg != 0.0:
                    rotate_pcd_around_z(pcd_ij, theta_deg=-theta_deg)

                # second: correct for rotation alpha_deg for flipping upside-down scans
                if test_upsidedown:
                    rotate_pcd_around_x(pcd_ij, alpha_deg=-alpha_deg)

                #   then toggle to PRCS
                pcd_ij = toggle_socs2prcs(pcd_ij)

                # Apply global robust SOR filter to kick-out spurious points / floaters
                apply_robust_sor_filter(pcd_ij, k_neighbors=50, std_ratio=2)

                # Preprocess (clean) per-instance point clouds and extract related Detections3D data
                # TODO: Possible additions/modifications to get_detections3d (check OneNote notes)
                d3d_i, pcd_ij = clean_pcd_instances_and_get_detections3d(pcd_ij, pcd_ij_id-1,
                                                                         d3d_parameters, pcp_parameters)


                # Save individual station point clouds (currently aligned in PRCS, if toggle_socs2prcs works)
                if save_intermediate_results:
                    save_segmented_pcd_ij(pcd_path_i, pcd_ij, inference_models_parameters, class_id_map, image_j)

                # Create path to pickled point cloud, d2d and d3d objects
                odir_pcd_ij = stage1_odir_partial / Path(f"pcd_ij_{pcd_ij_id}.pkl")
                odir_d3d_ij = stage1_odir_partial / Path(f"d3d_ij_{pcd_ij_id}.pkl")
                odir_d2d_ij = stage1_odir_partial / Path(f"d2d_ij_{pcd_ij_id}.pkl")


                # Store segmented point cloud
                # TODO: RE-IMPLEMENT storing path odir_pcd_ij
                pcd_collection.seg_pcds[pcd_ij_id-1] = pcd_ij
                inst_count = np.unique(pcd_ij.scalar_fields['instances']).size
                pcd_collection.pcd_n_instances[pcd_ij_id-1] = inst_count

                with open(odir_pcd_ij, 'wb') as f:
                    pickle.dump(pcd_ij, f)

                # Store 2d detections (bounding boxes, masks, confidences, class_ids,...)
                if task_parameters['save_d2d']:
                    # TODO: RE-IMPLEMENT storing path odir_d2d_ij
                    d2d_collection.append(results)

                    with open(odir_d2d_ij, 'wb') as f:
                        pickle.dump(results, f)

                # Store 3d detections
                # TODO: RE-IMPLEMENT storing path odir_pcd_ij odir_d3d_ij
                d3d_collection.append(d3d_i)
                with open(odir_d3d_ij, 'wb') as f:
                    pickle.dump(d3d_i, f)

                del pcd_ij, d3d_i, image_j, image_j_numpy, results
                gc.collect()

            del pcd, images_of_pcd_i
            gc.collect()


        # All point clouds looped through
        # --------------------------------------------------------------------------------------------------------------

        # Pickle and save Stage 1 results
        if save_intermediate_results is True:
            # Set paths
            stage1_output_dir = inference_models_parameters['stage1_output_dir']
            odir_pcd_collection = stage1_output_dir / Path("pcd_collection.pkl")
            if task_parameters['save_d2d']:
                odir_d2d_collection = stage1_output_dir / Path("d2d_collection.pkl")
            odir_d3d_collection = stage1_output_dir / Path("d3d_collection.pkl")
            # Save data
            with open(odir_pcd_collection, 'wb') as f:
                pickle.dump(pcd_collection, f)
            if task_parameters['save_d2d']:
                with open(odir_d2d_collection, 'wb') as f:
                    pickle.dump(d2d_collection, f)
            with open(odir_d3d_collection, 'wb') as f:
                pickle.dump(d3d_collection, f)

    else:
        # Load Stage 1 results (if Grounded SAM2 was already applied on the data)
        # Set paths
        stage1_output_dir = inference_models_parameters['stage1_output_dir']
        odir_pcd_collection = stage1_output_dir / Path("pcd_collection.pkl")
        if task_parameters['save_d2d']:
            odir_d2d_collection = stage1_output_dir / Path("d2d_collection.pkl")
        odir_d3d_collection = stage1_output_dir / Path("d3d_collection.pkl")

        # Load data
        with open(odir_pcd_collection, 'rb') as f:
            pcd_collection = pickle.load(f)
        if task_parameters['save_d2d']:
            with open(odir_d2d_collection, 'rb') as f:
                d2d_collection = pickle.load(f)
        with open(odir_d3d_collection, 'rb') as f:
            d3d_collection = pickle.load(f)

    # SECOND PART OF THE CODE: -----------------------------------------------------------------------------------------

    # Merge all Detections3D objects into 1 large object
    d3d_collection = merge_detections3d(d3d_collection)
    # squeeze_detections3d(d3d_collection)

    # Apply statistical outlier removal using multivariate-normal distribution and mahalanobis distance
    d3d_collection, pcd_or, inst_or = d3d_outlier_removal(d3d_collection, per_class_separation=False,
                                                          confidence_interval=0.99)
    pcd_collection.filter_out_instances(pcd_or, inst_or)

    # Get initial sparse connectivity (relevant/sparse nodes for the graph)
    semantic_gate = d3d_parameters["merge_inst_of_same_class_only"]
    spcon_method = d3d_parameters["sparse_connectivity_method"]
    spcon_threshold = d3d_parameters["sparse_connectivity_threshold"]
    pairs = get_initial_sparse_connectivity(centroids=d3d_collection.centroids, class_ids=d3d_collection.classes,
                                            n_scans=n_scans, method=spcon_method, knn_ps=spcon_threshold,
                                            radius=spcon_threshold, semantic_gate=semantic_gate)

    # get edges for graph
    iou_threshold = d3d_parameters["supporters_iou_threshold"]
    edges_iou, edges_supp = get_edge_weights(detections3d=d3d_collection, pairs=pairs,
                                             iou_threshold=iou_threshold, mode='both')

    # Remove Detections3D that overlap with too many other Detections3D (likely under-segmented)
    remove_outliers_by_support = d3d_parameters["remove_outliers_by_support"]
    or_method = d3d_parameters["outlier_detection_method"]
    or_threshold = d3d_parameters["outlier_detection_threshold"]
    if remove_outliers_by_support:
        n_d3d = d3d_collection.pcd_ids.shape[0]
        counts = count_significant_overlaps(pairs=pairs, bbox_overlap=edges_iou, iou_threshold=iou_threshold, N=n_d3d)
        outliers, cutoff = detect_upper_tail_outliers(data=counts, method=or_method, alpha=or_threshold)
        # TODO: so far not clear if works correctly - 0 outliers in the 1st test data
        d3d_collection, pairs, edges_supp = filter_outlier_detections3d_edges_and_nodes(d3d_collection=d3d_collection,
                                                                                        pairs=pairs,
                                                                                        edge_weights=edges_supp,
                                                                                        outliers=outliers)
        n_d3d = d3d_collection.pcd_ids.shape[0]

    # Find corresponding Detections3D instances using graph clustering
    #   return: cluster_ids / new instance labels with values from 1,2,..., N
    clustering_method = d3d_parameters['graph_clustering_method']
    min_supporters = d3d_parameters['min_supporters']
    leiden_resolution = d3d_parameters['leiden_resolution']
    clusters_ids = graph_clustering(num_nodes=n_d3d, pairs=pairs, edge_weights=edges_supp,
                                    method=clustering_method, min_supporters=min_supporters,
                                    leiden_resolution=leiden_resolution)

    # Kicking-out clusters with too small support (ID appearing less than a threshold times)
    #   -> setting their cluster IDs to 0
    clusters_ids = small_cluster_removal(clusters_ids, d3d_parameters)

    # Assign new instance labels to point clouds and merge them together
    pcd_result = get_segmented_and_merged_point_cloud(pcd_collection, d3d_collection, clusters_ids, pcp_parameters)

    # TODO: Apply filters after merge (visually inspect for which ones) -> HDBSCAN, multi-variate statistics,...

    # Replace pcd RGB colour by random colors for each instance
    if pcp_parameters['assign_random_color_per_instance'] is True:
        color_pcd_instances_by_random(pcd_result)

    # Save point cloud with final results
    save_segmented_pcd(data_folder_path, output_dir_pathlib, pcd_result, class_id_map)
    test = 1
    # TODO: CLEAN MEMORY

if __name__ == "__main__":
    main()
