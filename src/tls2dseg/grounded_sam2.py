# Imports
import gc
import json
import os
import random
from functools import partial
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from supervision.draw.color import ColorPalette
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from tls2dseg.grounded_sam2_utils import (
    check_if_img_slice_complete,
    check_if_img_slice_empty,
    convert_masks_to_sparse_masks,
    mask_to_rle,
    post_process_gdino_results,
    return_empty_detections,
    run_sam2_bbox_prompt_inference_in_batches,
)
from tls2dseg.pc2img_utils import img_1to3_channels_encoding
from tls2dseg.sparse_masks_inference_slicer import SparseMasksInferenceSlicer
from tls2dseg.supervision_utils import CUSTOM_COLOR_MAP

sam_lock = Lock()


def initialize_gdino(
    inference_models_parameters: dict,
) -> tuple[AutoModelForZeroShotObjectDetection, AutoProcessor]:
    # build grounding dino (IDEA huggingface workflow) - set up the model and data processing pipeline
    model_id = inference_models_parameters["bbox_model_id"]
    device = inference_models_parameters["device"]
    gdino_processor = AutoProcessor.from_pretrained(model_id)  # Set correct data preprocessing pipeline
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)  # Load model in CPU/GPU
    return gdino_model, gdino_processor


def initialize_sam2(inference_models_parameters: dict) -> SAM2ImagePredictor:
    # build SAM2 image predictor (Meta GitHub workflow)
    sam2_checkpoint = inference_models_parameters["sam2-checkpoint"]
    model_cfg = inference_models_parameters["sam2-model-config"]
    device = inference_models_parameters["device"]
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    return sam2_predictor


def callback(
    image_slice: np.ndarray,
    text_prompt: str,
    gdino_processor: AutoProcessor,
    gdino_model: AutoModelForZeroShotObjectDetection,
    sam2_predictor: SAM2ImagePredictor,
    inference_models_parameters: dict,
    slice_inference_parameters: dict,
) -> sv.Detections:
    """
    Do inference on a slice - supporting function for run_grounded_sam2_with_sahi()

    Parameters
    ----------
    image_slice
    text_prompt
    gdino_processor
    gdino_model
    sam2_predictor
    inference_models_parameters
    slice_inference_parameters

    Returns
    -------
    Detections (supervision library object with detected bounding boxes and sparse masks
     - scipy csr_matrix stored as detections.data["sparse_masks"])
    """

    # Get Parameters
    # ------------------------------------------------------------------------------------------------------------------
    slice_height, slice_width = image_slice.shape[:2]

    # Check for early termination
    # ------------------------------------------------------------------------------------------------------------------

    # Check if image_slice is predominantly empty space
    if check_if_img_slice_empty(image_slice, slice_inference_parameters) is False:
        return return_empty_detections()

    # Check if slice height and width as big as expected (if not skip this slice)
    # TODO: Warning: this assures no code crashes, but does not process the image edges! (consider better solution)
    if check_if_img_slice_complete(image_slice, slice_inference_parameters) is False:
        return return_empty_detections()

    # Prepare data
    # __________________________________________________________________________________________________________________
    # 1) if NaNs -> replace, 2) if not 0-1 -> normalize to 0-1, 3) replicate channels (H,W,) to (H,W,3)
    # Optional: change dtype to 'uint8' or 'float32'; broadcast channels instead of repeating them (memory save)
    image_slice = img_1to3_channels_encoding(
        image_slice, normalize="0-1", output_dtype="float32", replace_nan_with="max", broadcast=True
    )

    # Run Grounded DINO
    # __________________________________________________________________________________________________________________
    #   - Preprocess data: normalize and rescale images, tokenize text, transform into tensor
    device = inference_models_parameters["device"]
    inputs = gdino_processor(images=image_slice, text=text_prompt, return_tensors="pt", do_rescale=False).to(device)
    #   - Run inference
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    # Postprocess Grounded DINO results
    input_boxes, _, class_ids, confidences, empty_flag = post_process_gdino_results(
        gdino_processor,
        outputs,
        inputs,
        text_prompt,
        inference_models_parameters,
        slice_height,
        slice_width,
    )

    # Cleanup GPU/CPU memory
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    # Skip SAM2 inference if no detections remaining
    if empty_flag:
        return return_empty_detections()

    # 4. Run SAM2 (and store sparse masks)
    # __________________________________________________________________________________________________________________

    # Batch detected bounding boxes to avoid memory explosion when running inference with SAM!
    sam_box_prompt_batch_size = inference_models_parameters["sam_box_prompt_batch_size"]
    # Set output variables
    masks: list = []

    with sam_lock:
        # Set image
        sam2_predictor.set_image(image_slice)
        # Run batched inference
        masks = run_sam2_bbox_prompt_inference_in_batches(sam2_predictor, input_boxes, sam_box_prompt_batch_size, masks)

    masks = convert_masks_to_sparse_masks(masks)

    # Clear GPU/CPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Create an instance of supervision.detections object
    # __________________________________________________________________________________________________________________
    if not len(masks) == input_boxes.shape[0] == confidences.shape[0] == class_ids.shape[0]:
        raise ValueError(
            "Something went wrong while running Grounded SAM2 with SAHI: "
            "Not all sv.detection attributes have the same length!",
            "(attributes: input_boxes, confidences, class_ids, sparse_masks)",
        )

    detections = sv.Detections(xyxy=input_boxes, confidence=confidences, class_id=class_ids)
    detections.data["sparse_masks"] = masks

    return detections


def run_grounded_sam2(
    image: Path | np.ndarray,
    text_prompt: str,
    gdino_model: AutoModelForZeroShotObjectDetection,
    gdino_processor: AutoProcessor,
    sam2_predictor: SAM2ImagePredictor,
    inference_models_parameters: dict,
) -> dict:

    # Set inference hardware
    device = inference_models_parameters["device"]

    # Load images from the disk (if not using ones already in RAM)
    if isinstance(image, Path):
        image_pil = Image.open(image)  # Load image as PIL Image
        image = np.array(image_pil)  # Convert to 3 channel np.ndarray
    elif isinstance(image, np.ndarray):
        pass  # Do nothing
    else:
        raise ValueError("Image passed to run_grounded_sam2() has to be np.ndarray or Path object")

    # Modify image
    # 1) if NaNs -> replace, 2) if not 0-255 -> normalize to 0-255, 3) replicate channels (H,W,) to (H,W,3)
    # Optional: change dtype to 'uint8' or 'float32'; broadcast channels instead of repeating them (memory save)
    image = img_1to3_channels_encoding(
        image, normalize="0-1", output_dtype="float32", replace_nan_with="max", broadcast=True
    )

    image_height, image_width = image.shape[:2]
    # Run Grounded DINO
    #   - Preprocess data: normalize and rescale images, tokenize text, transform into tensor
    inputs = gdino_processor(images=image, text=text_prompt, return_tensors="pt", do_rescale=False).to(device)
    #   - Run inference
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    # Postprocess Grounded DINO results
    input_boxes, class_names, class_ids, confidences, _ = post_process_gdino_results(
        gdino_processor,
        outputs,
        inputs,
        text_prompt,
        inference_models_parameters,
        image_height,
        image_width,
    )

    # Cleanup GPU/CPU memory
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Run SAM2 (and store sparse masks)
    # __________________________________________________________________________________________________________________
    # Set input for SAM2
    sam2_predictor.set_image(image)

    # Batch detected bounding boxes to avoid memory explosion when running inference with SAM!
    sam_box_prompt_batch_size = inference_models_parameters["sam_box_prompt_batch_size"]
    masks: list = []
    # Run batched inference
    masks = run_sam2_bbox_prompt_inference_in_batches(sam2_predictor, input_boxes, sam_box_prompt_batch_size, masks)
    masks = convert_masks_to_sparse_masks(masks)

    # Clear GPU/CPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Prepare results dictionary
    # __________________________________________________________________________________________________________________
    # Create mask labels (class name + confidence scores)
    confidences = confidences.astype(float).tolist()

    labels = [
        f"{class_name} {confidence:.2f}" for class_name, confidence in zip(class_names, confidences, strict=False)
    ]

    # Store results in a dictionary
    #   sparse_masks - list of sparse bool matrices with 1 mask per matrice (N_boxes,_)
    #   input_boxes - np.ndarray of input boxes (N_boxes x 4)
    #   confidences - list of confidences per box (N_boxes,_)
    #   class_names - list of strings with class names per box (N_boxes,_)
    #   class_ids - np.ndarray of class_ids per box (N_boxes,_)
    #   mask_labels - list of strings with labels, name + confidence (N_boxes,_)
    results = {
        "masks": masks,
        "input_boxes": input_boxes,
        "confidences": confidences,
        "class_names": class_names,
        "class_ids": class_ids,
        "mask_labels": labels,
    }

    return results


def run_grounded_sam2_with_sahi(
    image: Path | np.ndarray,
    text_prompt: str,
    gdino_model: AutoModelForZeroShotObjectDetection,
    gdino_processor: AutoProcessor,
    sam2_predictor: SAM2ImagePredictor,
    inference_models_parameters: dict,
    slice_inference_parameters: dict,
) -> dict:

    # Load images from the disk (if not using ones already in RAM)
    if isinstance(image, Path):
        image_pil = Image.open(image)  # Load image as PIL Image
        image = np.array(image_pil)  # Convert to 3 channel np.ndarray
    elif isinstance(image, np.ndarray):
        pass  # Do nothing
    else:
        raise ValueError("Image passed to run_grounded_sam2() has to be np.ndarray or Path object")

    # Unpack variables defining image slicing process
    slice_wh = slice_inference_parameters["slice_width_height"]
    overlap_width_height = slice_inference_parameters["overlap_width_height"]
    iou_threshold = slice_inference_parameters["iou_threshold"]
    filter_strategy = slice_inference_parameters["overlap_filter_strategy"]
    thread_workers = slice_inference_parameters["thread_workers"]
    if filter_strategy.lower() == "nms":
        filter_strategy = sv.OverlapFilter.NON_MAX_SUPPRESSION
    else:
        raise ValueError("Unsupported filter strategy provided - currently only NMS!")

    # Partially initialize the function - populate all inputs in advance besides "image", which is populated
    #   iteratively within sv.InferenceSlicer with image slices
    callback_fn = partial(
        callback,
        text_prompt=text_prompt,
        gdino_processor=gdino_processor,
        gdino_model=gdino_model,
        sam2_predictor=sam2_predictor,
        inference_models_parameters=inference_models_parameters,
        slice_inference_parameters=slice_inference_parameters,
    )

    # Create a slicer object
    slicer = SparseMasksInferenceSlicer(
        callback=callback_fn,
        slice_wh=slice_wh,
        overlap_wh=overlap_width_height,
        overlap_ratio_wh=None,
        iou_threshold=iou_threshold,
        overlap_filter=filter_strategy,
        thread_workers=thread_workers,
    )

    # Run slicer (do detection on different slices)
    print("Running per slice inference")
    detections = slicer(image)
    print("Inference completed")

    # Get class_ids relative to the original text_prompt and corresponding to class_names
    keys = text_prompt.split(".")  # → ['house','window','bicycle','door','grass','leaf']
    class_id_map = {k: i + 1 for i, k in enumerate(keys)}  # → {'house':1, 'window':2, ..., 'leaf':6}
    inverted_map = {v: k for k, v in class_id_map.items()}

    # Set main output variables
    class_names = [inverted_map[id] for id in detections.class_id]
    confidences = detections.confidence.tolist()
    class_ids = detections.class_id
    input_boxes = detections.xyxy
    masks = detections.data.get("sparse_masks", [])

    # Create mask labels (class name + confidence scores)
    labels = [
        f"{class_name} {confidence:.2f}" for class_name, confidence in zip(class_names, confidences, strict=False)
    ]

    # Store results in a dictionary
    #   masks - list of sparse matrices (ndarrays of (M,2) with indices of mask location, M = mask pixel nr.)
    #   input_boxes - np.ndarray of input boxes (N_boxes x 4)
    #   confidences - list of confidences per box (N_boxes,_)
    #   class_names - list of strings with class names per box (N_boxes,_)
    #   class_ids - np.ndarray of class_ids per box (N_boxes,_)
    #   mask_labels - list of strings with labels, name + confidence (N_boxes,_)

    results = {
        "masks": masks,
        "input_boxes": input_boxes,
        "confidences": confidences,
        "class_names": class_names,
        "class_ids": class_ids,
        "mask_labels": labels,
    }

    return results


def save_gsam2_results(image: tuple[str, np.ndarray, Path], results: dict, inference_models_parameters) -> None:
    # 1. Create JPEG files
    # --------------------

    # Load values from the dictionary with results
    input_boxes = results["input_boxes"]
    masks = results["masks"]
    class_ids = results["class_ids"]
    labels = results["mask_labels"]
    scores = results["confidences"]
    class_names = results["class_names"]

    # Get values from image tuple:
    image_data = image[1]
    image_path = image[2]

    # Transform image to 3 channel image:
    image_data = img_1to3_channels_encoding(
        image_data, normalize="0-255", output_dtype="uint8", replace_nan_with="max", broadcast=True
    )

    # Select only a few masks in the case of many:
    subsampled_masks_flag = False
    n_masks = len(masks)
    if n_masks > 16:
        subsampled_masks_flag = True
        indices = random.sample(range(n_masks), k=16)
        masks = [masks[i] for i in indices]
        class_names = [class_names[i] for i in indices]
        labels = [labels[i] for i in indices]
        input_boxes = input_boxes[indices, :]
        class_ids = class_ids[indices]
        scores = [scores[i] for i in indices]

    # Transform a list of sparse masks into a numpy array
    image_height, image_width = image_data.shape[:2]
    masks_numpy = np.zeros((len(masks), image_height, image_width), dtype=np.bool_)
    for i, mask_i in enumerate(masks):
        masks_numpy[i, mask_i[:, 0], mask_i[:, 1]] = True

    # Create detection objects for "supervision useful API"
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks_numpy,  # (n, h, w)
        class_id=class_ids,
    )

    # Save .jpg images of detected objects (bounding box, semantic label, score)
    #   - note: if you want to use default color map, you can set color=ColorPalette.DEFAULT

    #   - "supervision" library commands
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=image_data.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    #   - set output directory and file names for object detection
    output_dir_od = inference_models_parameters["output_dir_od"]
    if subsampled_masks_flag:
        output_jpg_od = f"{image_path.stem}_od_RANDOM_SUBSAMPLE_16.jpg"
    else:
        output_jpg_od = f"{image_path.stem}_od.jpg"
    cv2.imwrite(os.path.join(output_dir_od, output_jpg_od), annotated_frame.astype(np.dtype("uint8")))

    # Save .jpg images of SAM masks (mask, semantic label, score)
    #   - "supervision" library commands

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame.astype(np.dtype("uint8")), detections=detections)
    #   - set output directory and file names for object detection
    output_dir_sam2 = inference_models_parameters["output_dir_sam2"]
    if subsampled_masks_flag:
        output_jpg_sam2 = f"{image_path.stem}_sam2_RANDOM_SUBSAMPLE_16.jpg"
    else:
        output_jpg_sam2 = f"{image_path.stem}_sam2.jpg"
    cv2.imwrite(os.path.join(output_dir_sam2, output_jpg_sam2), annotated_frame.astype(np.dtype("uint8")))

    # 2. Create JSON file
    # -------------------

    if inference_models_parameters["dump_json_results"]:
        # convert mask into rle format
        mask_rles = [mask_to_rle(mask) for mask in masks]
        # convert bounding boxes and scores (confidences) to lists
        input_boxes = input_boxes.tolist()
        if isinstance(scores, np.ndarray):
            scores = scores.astype(float).tolist()

        # save the results in standard format
        results_supervision = {
            "image_path": image_path.as_posix(),
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores, strict=False)
            ],
            "box_format": "xyxy",
            "img_width": image_data.shape[1],
            "img_height": image_data.shape[0],
        }

        output_dir_masks_json = inference_models_parameters["output_dir_masks_json"]
        output_json = f"{image_path.stem}_gsam2_results.json"

        with open(os.path.join(output_dir_masks_json, output_json), "w") as f:
            json.dump(results_supervision, f, indent=4)

    return None
