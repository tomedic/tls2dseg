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
task_parameters = {'input_path': "./data/wheat_heads_small.e57",  # Set path to input point cloud
                   'output_path': "./results",   # Set path for storing the results
                   'save_intermediate_results': True,   # Save intensity images, gDINO and SAM2 outputs
                   'task': "object_detection",  # Task choice
                   'results_aggregation_strategy': "object_memory_bank"  # OUT type choice: memory bank vs. voxel-grid
                   }


# PointCloud processing parameters
pcp_parameters = {'output_resolution': 0.003,  # Subsample point cloud
                  'range_limits': [0., 5.],  # All points further then will be discarded
                  'roi_limits': [-12.5, -0.8, 491.7, 0.3, 7.5, 493.7]  # only region of interest (3D bounding box) is to be analyzed
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
#text_prompt = "house.window.bicycle.door.grass.pipe"
#text_prompt = "house.window.wall.door.roof.brick.ground.fence.person"
#text_prompt = "wall.ceiling.plants.plant pot.leaf.leaves.desk.chair.bag.table.keyboard.floor.window.monitor"
text_prompt = "wheat.wheat head.wheat ear.wheat spike.wheat spikelet.wheat grain.wheat fruit"

# Inference model parameters:
inference_models_parameters = {'with_slice_inference': True,
                               'bbox_model_id': 'IDEA-Research/grounding-dino-base',
                               'box_threshold': 0.30,  # 0.35
                               'text_threshold': 0.15,  # 0.25
                               'sam2-model-config': 'configs/sam2.1/sam2.1_hiera_l.yaml',
                               'sam2-checkpoint': '/scratch/projects/sam2/checkpoints/sam2.1_hiera_large.pt'}

# Additional parameters for slice inference (necessary only if inference with SAHI)
slice_inference_parameters = {'slice_width_height': (960, 960),
                              'overlap_ratio_in_width_height': (0.0, 0.0),
                              'iou_threshold': 0.8,
                              'overlap_filter_strategy': 'nms'}

def main():

    # 0. Initial set-up
    # __________________________________________________________________________________________________________________

    # Check input parameters (Type and Value checks)
    check_all_parameters(task_parameters, pcp_parameters, image_generation_parameters,
                         inference_models_parameters, slice_inference_parameters, text_prompt)

    # Load data and auto update input parameters
    pcd_path_pathlib = Path(task_parameters['input_path'])  # Create pathlib path
    pcd: PointCloudData = load_e57(pcd_path_pathlib, stay_global=False)  # Load point cloud

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

    # 1. Point Cloud Processing
    # __________________________________________________________________________________________________________________

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
    images_pcd_i = pc2img_run(pcd, pcd_path_pathlib, image_generation_parameters, image_width, image_height)

    # Reducing image resolution (if necessary)
    if reduction_coefficient < 1:
        print("Reducing image resolution")
        images_pcd_i = reduce_image_resolution(images_pcd_i, reduction_coefficient, image_generation_parameters,
                                               pcd_path_pathlib)
        # Get new image width and height
        image_height, image_width = images_pcd_i[0][1].shape

    # Initialize SAM2 everything
    # TODO: I HAVE MESSED UP THE OUTPUT OF PC2IMG_RUN -> GIVES RAW NUMPY NOT RGB IMAGE ANYMORE!
    testis = 1
    image_test = images_pcd_i[0][1]
    image_test = convert_to_image(image_test, "max", normalize=True, colormap='gray')  # gray
    testis = 1
    sam2_everything = initialize_sam2_everyting(inference_models_parameters)
    masks = run_sam2_everything(image_test, sam2_everything, inference_models_parameters)
    testis = 1

    # Initialize Grounded SAM2 (Grounded DINO + SAM2)
    gdino_model, gdino_processor = initialize_gdino(inference_models_parameters)
    sam2_predictor = initialize_sam2(inference_models_parameters)



    # Run Grounded SAM2 inference (for all images of a point cloud pcd_i)
    for image_j in images_pcd_i:
        image_j_numpy = image_j[1]  # Get data (8bit np.ndarray) from "image object"
        # RESULTS CONTAIN: 'masks' with M x w x h (M = mask number, w = width, h = height),
        #                  'input_boxes' with input boinding boxes,
        #                  'confidences' with confidence scores,
        #                  'class_names', class ids, ...

        if inference_models_parameters["with_slice_inference"] is True:
            print("Grounded Dino - Inference on image slices")
            results = run_grounded_sam2_with_sahi(image=image_j_numpy, text_prompt=text_prompt, gdino_model=gdino_model,
                                        gdino_processor=gdino_processor, sam2_predictor=sam2_predictor,
                                        inference_models_parameters=inference_models_parameters,
                                        slice_inference_parameters=slice_inference_parameters)
        else:
            print("Grounded Dino - Inference on a whole image")
            results = run_grounded_sam2(image=image_j_numpy, text_prompt=text_prompt, gdino_model=gdino_model,
                                        gdino_processor=gdino_processor, sam2_predictor=sam2_predictor,
                                        inference_models_parameters=inference_models_parameters)

        # Save object detection (gdino) and segmentation (SAM2) results as .jpeg images and corresponding data in .json:
        if save_intermediate_results:
            print("Saving intermediate results")
            save_gsam2_results(image=image_j, results=results, inference_models_parameters=inference_models_parameters)

        # From individual per-object bool masks get:
        #   - 1 instance mask (each instance having one int ID),
        #   - 1 semantic mask (each class having one ing ID),
        #   - class_id_map which maps semantic classes provided in text_prompt to semantic class IDs
        print("Getting unified instance and semantic mask from individual masks")
        instance_mask, semantic_mask, class_id_map = get_instance_and_semantic_mask(results, text_prompt)

        # Add the generated masks to ImageStack related to the point cloud pcd
        print("Lifting 2d masks to 3d")
        project_masks2pcd_as_scalarfields(pcd, instance_mask, semantic_mask)

        # Save segmented point cloud and related transformation parameters
        print("Saving results")
        output_pcd_name = pcd_path_pathlib.stem + "_seg.ply"  # _seg for segmented
        output_pcd_path = output_dir / Path(output_pcd_name)
        save_ply(output_pcd_path, pcd, retain_colors=True, retain_normals=True, scalar_fields=None)
        
        # Save related transformation matrix (for local-to-global conversion)
        transformation_matrix_output_path = output_dir / Path(pcd_path_pathlib.stem + '_T.txt')
        np.savetxt(transformation_matrix_output_path, pcd.transformation_matrix, fmt="%.8f", delimiter=" ")

        # Save class name - to - class id map in ascii
        class_id_map["background"] = 0
        inverted_map = {v: k for k, v in class_id_map.items()}
        inverted_map_path = output_dir / Path('class_names_id_map.txt')
        with open(str(inverted_map_path), 'w', encoding='ascii') as f:
            json.dump(inverted_map, f, ensure_ascii=True)



    test = 1
    banana = 2







if __name__ == "__main__":
    main()

