# Point Cloud Segmentation Using 2D point cloud representation and DL foundational models (e.g. GroundedSAM)
# Author: Tomislav Medic & ChatGPT, 22.04.2025

# TODO: THIS IS A QUICK-FIX -> remove and do everything properly for pip installable project!
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load libraries:

# NICHOLAS:
from pathlib import Path
from pchandler.geometry import PointCloudData
from pchandler.data_io import load_e57

# TOMISLAV:
import torch
from src.tls2dseg.pc2img_utils import pc2img_run
from src.tls2dseg.grounded_sam2 import (initialize_gdino, initialize_sam2, run_grounded_sam2, save_gsam2_results,
                                        make_output_folders)


# I/0 parameters:
pcd_path = "./data/test_small.e57"  # Set path to point cloud
output_dir = "./results"  # Set path for storing the results
save_intermediate_results = True  # Save intensity images, gDINO and SAM2 outputs

# Image generation parameters:
# TODO: make a dictionary with all hyperparameters (incl. ones hidden in pc2img_run() function)
rotate_pcd = False  # Should I rotate point cloud before generating an image?
theta = 0  # Rotate point cloud around Z-axis for theta degrees
image_width = 2000  # Image width in pixels (height adjusted according to aspect ratio)
rasterization_method: str = 'nanconv'
features: list = ["intensity"]

# Prompt object detection / segmentation
# TODO: VERY important: text prompts need to be lowercase + end with a dot
text_prompt = "house.window.bicycle.wall.grass.leaf"

# Inference model parameters:
inference_models_parameters = {'bbox_model_id': 'IDEA-Research/grounding-dino-base',
                               'box_threshold': 0.35,
                               'text_threshold': 0.25,
                               'sam2-model-config': 'configs/sam2.1/sam2.1_hiera_l.yaml',
                               'sam2-checkpoint': '/scratch/projects/sam2/checkpoints/sam2.1_hiera_large.pt',
                               'device': 'cpu',
                               'dump_json_results': True}

def main():

    # Load data and auto update input parameters
    pcd_path_pathlib = Path(pcd_path)  # Create pathlib path
    pcd: PointCloudData = load_e57(pcd_path_pathlib, stay_global=False)  # Load point cloud
    image_height = int(image_width // pcd.fov.ratio())  # set image height w.r.t. image_width
    output_dir_pathlib = Path(output_dir)   # Create pathlib path
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Set inference hardware
    inference_models_parameters['device'] = device  # Add to model parameters dictionary


    # Create folders for intermediate results (if necessary)
    if save_intermediate_results:
        make_output_folders(output_dir_pathlib, inference_models_parameters)

    # Generate images of point cloud i
    images_pcd_i = pc2img_run(pcd, pcd_path_pathlib, inference_models_parameters['output_dir_intermediate'],
                              save_intermediate_results, rotate_pcd,
                              theta, image_height, image_width, rasterization_method, features)

    # Initialize Grounded SAM2 (Grounded DINO + SAM2)
    gdino_model, gdino_processor = initialize_gdino(inference_models_parameters)
    sam2_predictor = initialize_sam2(inference_models_parameters)

    # Set the environment settings
    # use bfloat16 where ok, otherwise float32
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Run Grounded SAM2 inference (for all images of a point cloud pcd_i)
    for image_j in images_pcd_i:
        image_j_numpy = image_j[1]  # Get data (8bit np.ndarray) from "image object"
        results = run_grounded_sam2(image=image_j_numpy, text_prompt=text_prompt, gdino_model=gdino_model,
                                    gdino_processor=gdino_processor, sam2_predictor=sam2_predictor,
                                    inference_models_parameters=inference_models_parameters)

        # Save object detection (gdino) and segmentation (SAM2) results as .jpeg images and corresponding data in .json:
        if save_intermediate_results:
            save_gsam2_results(image=image_j, results=results, inference_models_parameters=inference_models_parameters)




    test = 1
    banana = 2







if __name__ == "__main__":
    main()

