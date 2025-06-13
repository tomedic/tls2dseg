# Point Cloud Segmentation Using 2D point cloud representation and DL foundational models (e.g. GroundedSAM)
# Author: Tomislav Medic & ChatGPT, 22.04.2025

# TODO: THIS IS A QUICK-FIX -> remove and do everything properly for pip installable project!
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load libraries:

# NICHOLAS:
from pathlib import Path
from pchandler.geometry import PointCloudData
from pchandler.data_io import load_e57, save_ply

# TOMISLAV:
import torch
from src.tls2dseg.pc2img_utils import *
from src.tls2dseg.grounded_sam2 import *
from src.tls2dseg.utils import *
from src.tls2dseg.sam_everything import *
from src.tls2dseg.pc_preprocessing import *
from src.tls2dseg.parameters_check import *
import json

# I/0 parameters:
# pcd_path = "./data/wheat_heads_small.e57"  # Set path to point cloud
# output_dir = "./results"  # Set path for storing the results
# save_intermediate_results = True  # Save intensity images, gDINO and SAM2 outputs

# Task & I/0 parameters:
task_parameters = {'input_path': "./data/bafu/",  # Set path to input point clouds
                   'file_format': "e57",
                   'output_path': "./results",   # Set path for storing the results
                   'save_intermediate_results': True,   # Save intensity images, gDINO and SAM2 outputs
                   'task': "object_detection",  # Task choice
                   'results_aggregation_strategy': "object_memory_bank"  # OUT type choice: memory bank vs. voxel-grid
                   }


# PointCloud processing parameters
pcp_parameters = {'output_resolution': 0.5,  # Subsample point cloud
                  'range_limits': None,  # All points further then will be discarded
                  'roi_limits': None  # only region of interest (3D bounding box) is to be analyzed
                  }
# pcp_parameters = {'output_resolution': 0.003,  # Subsample point cloud
#                   'range_limits': [0., 5.],  # All points further then will be discarded
#                   'roi_limits': [-12.5, -0.8, 491.7, 0.3, 7.5, 493.7]  # only region of interest (3D bounding box) is to be analyzed
#                   }

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
#text_prompt = "house.window.bicycle.door.grass.pipe"
#text_prompt = "house.window.wall.door.roof.brick.ground.fence.person"
#text_prompt = "wall.ceiling.plants.plant pot.leaf.leaves.desk.chair.bag.table.keyboard.floor.window.monitor"
#text_prompt = "pine. pine tree"
text_prompt = "rock.stone.boulder.cliff.tree.pine"
#text_prompt = "wheat.wheat head.wheat ear.wheat spike.wheat spikelet.wheat grain.wheat fruit"

# Inference model parameters:
inference_models_parameters = {'with_slice_inference': False,
                               'bbox_model_id': 'IDEA-Research/grounding-dino-base',
                               'box_threshold': 0.10,  # 0.35
                               'text_threshold': 0.10,  # 0.25
                               'sam2-model-config': 'configs/sam2.1/sam2.1_hiera_l.yaml',
                               'sam2-checkpoint': '/scratch/projects/sam2/checkpoints/sam2.1_hiera_large.pt',
                               'sam_box_prompt_batch_size': 16}

# Additional parameters for slice inference (necessary only if inference with SAHI)
slice_inference_parameters = {'slice_width_height': (400, 400),
                              'overlap_width_height': (100, 100),
                              'iou_threshold': 0.80,
                              'overlap_filter_strategy': 'nms',
                              'large_object_removal_threshold': 0.10,
                              'thread_workers': 16}

# Parameters defining how to get 3d objects from 2d detections + SAM2 masks
d3d_parameters = {'bounding_box_type': 'obb',
                  'centroid_type': 'median',
                  'preprocess': True}


def main():

    # 0. Initial Set-up
    # __________________________________________________________________________________________________________________

    # Check input parameters (Type and Value checks)
    check_all_parameters(task_parameters, pcp_parameters, image_generation_parameters,
                         inference_models_parameters, slice_inference_parameters, text_prompt)

    # Set output path
    output_dir = task_parameters['output_path']
    output_dir_pathlib = Path(task_parameters['output_path'])   # Create pathlib path

    # Create folders and set flags to True for intermediate results (if necessary)
    save_intermediate_results = task_parameters['save_intermediate_results']
    inference_models_parameters['dump_json_results'] = True if save_intermediate_results else False
    if save_intermediate_results:
        make_output_folders(output_dir_pathlib, image_generation_parameters, inference_models_parameters)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Set inference hardware
    inference_models_parameters['device'] = device  # Add to model parameters dictionary

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

    # Get dictionary mapping "semantic classes" to unique IDs and vice versa (+ save in results)
    keys = text_prompt.split('.')  # → ['house','window','bicycle','door','grass','leaf']
    class_id_map = {k: i + 1 for i, k in enumerate(keys)}  # → {'house':1, 'window':2, ..., 'leaf':6}
    class_id_map["background"] = 0
    inverted_id_map = {v: k for k, v in class_id_map.items()}
    inverted_map_path = output_dir / Path('class_names_id_map.txt')
    with open(str(inverted_map_path), 'w', encoding='ascii') as f:
        json.dump(inverted_id_map, f, ensure_ascii=True)

    # 1. Point Cloud Processing
    # __________________________________________________________________________________________________________________

    # Find all point cloud files of defined "file_format" within data folder
    data_folder_path = Path(task_parameters["input_path"]).resolve()
    file_format = task_parameters["file_format"]
    pcd_file_paths = list(data_folder_path.glob(f"*.{file_format}"))

    # Set-up results structure
    how_aggregate_results = task_parameters["results_aggregation_strategy"]
    point_cloud_id = 0
    if how_aggregate_results == "object_memory_bank":
        d3d_memory_bank = []
    else:
        raise ValueError("Chosen results_aggregation_strategy is currently not supported!")

    for pcd_path_i in pcd_file_paths:
        point_cloud_id += 1

        # Load data
        pcd: PointCloudData = load_e57(pcd_path_i, stay_global=False)  # Load point cloud

        # Filter point cloud for ranges and RoI (region of interest)
        filter_pcd_roi_range(pcd, pcp_parameters)

        # Rotate point cloud around z (if necessary), return rotation angle theta in degrees
        theta_deg = resolve_rotate_pcd_parameter(pcd, image_generation_parameters)

        # Compute image dimensions (width and height) in pixels and scan resolution (azimuth and elevation) in radians
        image_width, image_height, d_azim_deg, d_elev_deg = compute_image_dimensions(pcd, image_generation_parameters)
        print(f"Image height x width: {image_height} x {image_width}")

        # Resolve necessary image resolution
        reduction_coefficient = resolve_necessary_image_resolution(pcd, pcp_parameters, d_azim_deg)

        # Generate images of point cloud i
        print("Generating desired image(s)")
        images_pcd_i = pc2img_run(pcd, pcd_path_i, image_generation_parameters, image_width, image_height)

        # Reducing image resolution (if necessary),
        #   + getting rid of NaN values & casting image to float32 (if not already float32)
        if reduction_coefficient < 1:
            print("Reducing image resolution")
            images_pcd_i = reduce_image_resolution(images_pcd_i, reduction_coefficient, image_generation_parameters,
                                                   pcd_path_i)

            # Get new image width and height
            image_height, image_width = images_pcd_i[0][1].shape

        # TODO: Initializing and testing SAM2 everything -> nicely incorporate in the code (how:
        #  semantic segmentation on intensity + instance segmentation SAM2everything on range, combine)
        # image_test = images_pcd_i[1][1]
        # sam2_everything = initialize_sam2_everyting(inference_models_parameters)
        # masks = run_sam2_everything(image_test, sam2_everything, inference_models_parameters)
        # testis = 1

        # Run Grounded SAM2 inference (for all images of a point cloud pcd_i)
        for i, image_j in enumerate(images_pcd_i):
            image_j_numpy = image_j[1]
            # Transform 1 channel (float) ndarray into 3channel (8bit) - "grayscale" to "rgb"
            # image_j_numpy = convert_to_image(image_j_numpy, "max", normalize=True, colormap='gray')

            # results [dict]: 'masks' with M x w x h (M = mask number, w = width, h = height),
            #                  'input_boxes' with input boinding boxes,
            #                  'confidences' with confidence scores,
            #                  'class_names', class ids, ...

            if inference_models_parameters["with_slice_inference"] is True:
                print("Grounded SAM2 - Inference on image slices")
                results = run_grounded_sam2_with_sahi(image=image_j_numpy, text_prompt=text_prompt, gdino_model=gdino_model,
                                            gdino_processor=gdino_processor, sam2_predictor=sam2_predictor,
                                            inference_models_parameters=inference_models_parameters,
                                            slice_inference_parameters=slice_inference_parameters)
            else:
                print("Grounded SAM2 - Inference on a whole image")
                results = run_grounded_sam2(image=image_j_numpy, text_prompt=text_prompt, gdino_model=gdino_model,
                                            gdino_processor=gdino_processor, sam2_predictor=sam2_predictor,
                                            inference_models_parameters=inference_models_parameters)

            images_pcd_i[i] = (images_pcd_i[i][0], image_j_numpy, images_pcd_i[i][2])

            # Save object detection (gdino) and segmentation (SAM2) results as .jpeg images and corresponding data in .json:
            if save_intermediate_results:
                print("Saving intermediate results")
                save_gsam2_results(image=images_pcd_i[i], results=results, inference_models_parameters=inference_models_parameters)

            # From individual per-object bool masks get:
            #   - 1 instance mask (each instance having one int ID),
            #   - 1 semantic mask (each class having one int ID),
            #   - class_id_map which maps semantic classes provided in text_prompt to semantic class IDs
            print("Getting unified instance and semantic mask from individual masks")
            # instance_mask, semantic_mask, class_id_map = get_instance_and_semantic_mask(results, text_prompt)
            image_hw = image_j_numpy.shape[:2]
            instance_mask, semantic_mask, confidence_mask, class_id_map =\
                get_instance_and_semantic_mask_with_confidence(results, text_prompt, image_hw)

            # Add the generated masks to ImageStack related to the point cloud pcd
            print("Lifting 2d masks to 3d")
            project_masks2pcd_as_scalarfields(pcd, instance_mask, semantic_mask)
            project_a_mask_2_pcd_as_scalarfield(pcd, mask=confidence_mask, mask_name="confidence")

            # Subsample point cloud to desired output resolution:
            # TODO: current method keeps points nearest to the voxel center, if I want better results (averaging
            #  instead of subsampling, I need to implement majority voting for instance and semantic masks)
            pcd = subsample_pcd_to_output_resolution(pcd, pcp_parameters)

            # Remove background class (if task = object detection)
            pcd = remove_unclassified_points(pcd, task_parameters)

            if save_intermediate_results:
                save_segmented_pcds_in_socs(pcd_path_i, pcd, inference_models_parameters,
                                            class_id_map, image_j)

            # Extract per-instance metadata:
            # TODO: Transform point cloud to global!
            d3d_i = get_detections3d(pcd, point_cloud_id, d3d_parameters)
            d3d_memory_bank.append(d3d_i)

            # TODO: Create unique identifier later once all pcd_id and
            #  instance_id field for all point clouds are populated
            # TODO: Possible additions/modifications to get_instance_metadata (check OneNote notes)


            # Once all point clouds processed, clean RAM
            # TODO: CLEAN RAM





    test = 1
    banana = 2







if __name__ == "__main__":
    main()

