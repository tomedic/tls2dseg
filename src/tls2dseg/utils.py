from pathlib import Path


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

