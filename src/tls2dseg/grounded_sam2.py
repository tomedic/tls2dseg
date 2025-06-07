# Imports
import gc
import json
import os
from functools import partial
from pathlib import Path
from typing import List
from typing import Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from supervision.draw.color import ColorPalette
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from scipy.sparse import csr_matrix
import random
from itertools import compress

from src.tls2dseg.supervision_utils import CUSTOM_COLOR_MAP
from src.tls2dseg.pc2img_utils import img_1to3_channels_encoding

def initialize_gdino(inference_models_parameters: dict) -> Tuple[AutoModelForZeroShotObjectDetection, AutoProcessor]:
    # build grounding dino (IDEA huggingface workflow) - set up the model and data processing pipeline
    model_id = inference_models_parameters['bbox_model_id']
    device = inference_models_parameters['device']
    gdino_processor = AutoProcessor.from_pretrained(model_id)  # Set correct data preprocessing pipeline
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)  # Load model in CPU/GPU
    return gdino_model, gdino_processor


def initialize_sam2(inference_models_parameters: dict) -> SAM2ImagePredictor:
    # build SAM2 image predictor (Meta GitHub workflow)
    sam2_checkpoint = inference_models_parameters['sam2-checkpoint']
    model_cfg = inference_models_parameters['sam2-model-config']
    device = inference_models_parameters['device']
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    return sam2_predictor


def resolve_class_names(class_names: List[str], valid_keys: List[str]) -> List[str]:
    """
    Resolves a list of class names to valid keys.
    If class_name is directly in valid_keys, keep it, else, search for the first valid_key that is a substring of
     class_name.

    Args:
        class_names: List of predicted or raw class names.
        valid_keys: List of valid class labels.

    Returns:
        List of resolved class names.
    """
    resolved = []
    for name in class_names:
        if name in valid_keys:
            resolved.append(name)
        else:
            match = next((key for key in valid_keys if key in name), None)
            resolved.append(match)
    return resolved


def callback(image_slice: np.ndarray, text_prompt: str, device: str, gdino_processor: AutoProcessor,
             gdino_model: AutoModelForZeroShotObjectDetection, inference_models_parameters: dict, slice_inference_parameters: dict) -> sv.Detections:
    '''
    Do inference on a slice - supporting function for run_grounded_sam2_with_sahi()

    Parameters
    ----------
    image_slice
    text_prompt
    device
    gdino_processor
    gdino_model
    inference_models_parameters
    slice_inference_parameters

    Returns
    -------
    Detections (supervision library object with detected bounding boxes)
    '''

    # 1) if NaNs -> replace, 2) if not 0-255 -> normalize to 0-255, 3) replicate channels (H,W,) to (H,W,3)
    # Optional: change dtype to 'uint8' or 'float32'; broadcast channels instead of repeating them (memory save)
    image_slice = img_1to3_channels_encoding(image_slice, normalize='0-1', output_dtype='float32',
                                             replace_nan_with='max', broadcast=True)

    slice_height, slice_width = image_slice.shape[:2]
    slice_height_expected, slice_width_expected = slice_inference_parameters["slice_width_height"]
    if slice_height != slice_height_expected and slice_width != slice_width_expected:
        return sv.Detections(xyxy=np.empty((0, 4)), confidence=np.array([]), class_id=np.array([]))

    # Run Grounded DINO
    #   - Preprocess data: normalize and rescale images, tokenize text, transform into tensor
    inputs = gdino_processor(images=image_slice, text=text_prompt, return_tensors="pt", do_rescale=False).to(device)
    #   - Run inference
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    # Postprocess Grounded DINO results
    # - Likely post-processing steps (need to check it) -> Filters out bounding boxes and text predictions with low
    #   confidence scores, resizes the predictions to original size
    gdino_results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=inference_models_parameters['box_threshold'],
        text_threshold=inference_models_parameters['text_threshold'],
        target_sizes=[(slice_height, slice_width)]
    )

    # Prepare results for supervision library Detections object:
    input_boxes = gdino_results[0]["boxes"].cpu().numpy()  # get the bounding box prompts for SAM2
    confidences = gdino_results[0]["scores"].cpu().numpy()  # tensor -> ndarray
    class_names = gdino_results[0]["labels"]  # get class names

    # Get class_ids corresponding defined relative to the original text_prompt and corresponding to class_names
    keys = text_prompt.split('.')  # → ['house','window','bicycle','door','grass','leaf']
    class_names = resolve_class_names(class_names, keys)  # if class name corresponds to 2 valid classes - pick 1

    # Removing eventual detections that are not related to any of the valid categories
    none_indices = [i for i, name in enumerate(class_names) if name is None]
    class_names = [elem for i, elem in enumerate(class_names) if i not in none_indices]
    mask = np.ones(confidences.shape[0], dtype=bool)
    mask[none_indices] = False
    input_boxes = input_boxes[mask]
    confidences = confidences[mask]

    # Create dictionary mapping class names to class IDs
    id_map = {k: i + 1 for i, k in enumerate(keys)}  # → {'house':1, 'window':2, ..., 'leaf':6}
    class_ids = np.array([id_map[q] for q in class_names])  # a list of corresponding class IDs

    # 5. Cleanup GPU/CPU memory
    del inputs, outputs, gdino_results
    torch.cuda.empty_cache()
    gc.collect()

    return sv.Detections(xyxy=input_boxes, confidence=confidences, class_id=class_ids)


def run_grounded_sam2(image: Path | np.ndarray, text_prompt: str,
                      gdino_model: AutoModelForZeroShotObjectDetection,
                      gdino_processor: AutoProcessor,
                      sam2_predictor: SAM2ImagePredictor,
                      inference_models_parameters: dict) -> dict:

    # Set inference hardware
    device = inference_models_parameters['device']

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
    image = img_1to3_channels_encoding(image, normalize='0-1', output_dtype='float32', replace_nan_with='max',
                                       broadcast=True)

    image_height, image_width = image.shape[:2]
    # Run Grounded DINO
    #   - Preprocess data: normalize and rescale images, tokenize text, transform into tensor
    inputs = gdino_processor(images=image, text=text_prompt, return_tensors="pt", do_rescale=False).to(device)
    #   - Run inference
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    # Postprocess Grounded DINO results
    # - Likely post-processing steps (need to check it) -> Filters out bounding boxes and text predictions with low
    #   confidence scores, resizes the predictions to original size
    gdino_results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=inference_models_parameters['box_threshold'],
        text_threshold=inference_models_parameters['text_threshold'],
        target_sizes=[(image_height, image_width)]
    )

    # Create main outputs (to be saved in a dictionary 'results')
    input_boxes = gdino_results[0]["boxes"].cpu().numpy()  # get the bounding box prompts for SAM2
    confidences = gdino_results[0]["scores"].cpu().numpy().tolist()  # tensor -> ndarray
    class_names = gdino_results[0]["labels"]  # get class names
    # Get class_ids corresponding defined relative to the original text_prompt and corresponding to class_names
    keys = text_prompt.split('.')  # → ['house','window','bicycle','door','grass','leaf']
    class_names = resolve_class_names(class_names, keys)  # if class name corresponds to 2 valid classes - pick 1
    id_map = {k: i + 1 for i, k in enumerate(keys)}  # → {'house':1, 'window':2, ..., 'leaf':6}
    class_ids = np.array([id_map[q] for q in class_names])  # a list of corresponding class IDs

    # Create mask labels (class name + confidence scores)
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    # Cleanup GPU/CPU memory
    del inputs, outputs, gdino_results
    torch.cuda.empty_cache()
    gc.collect()

    # Set input for SAM2
    sam2_predictor.set_image(image)  # Assure np.ndarray with RGB channels

    # Run SAM2
    masks_np, _, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Squeeze out unnecessary dimensions and convert into a list of sparse matrices
    masks = []
    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)  # convert the shape to (n, H, W)
    # Store individual masks j of batch i as sparse booleans
    for mask_j in masks_np:
        masks.append(csr_matrix(mask_j.astype(bool), dtype=bool))

    # Store results in a dictionary
    #   masks - list of sparse bool matrices with 1 mask per matrice (N_boxes,_)
    #   input_boxes - np.ndarray of input boxes (N_boxes x 4)
    #   confidences - list of confidences per box (N_boxes,_)
    #   class_names - list of strings with class names per box (N_boxes,_)
    #   class_ids - np.ndarray of class_ids per box (N_boxes,_)
    #   mask_labels - list of strings with labels, name + confidence (N_boxes,_)
    results = {"masks": masks, "input_boxes": input_boxes, "confidences": confidences, "class_names": class_names,
               "class_ids": class_ids, "mask_labels": labels}

    return results


def run_grounded_sam2_with_sahi(image: Path | np.ndarray, text_prompt: str,
                                gdino_model: AutoModelForZeroShotObjectDetection,
                                gdino_processor: AutoProcessor,
                                sam2_predictor: SAM2ImagePredictor,
                                inference_models_parameters: dict,
                                slice_inference_parameters: dict) -> dict:

    # Set inference hardware
    device = inference_models_parameters['device']
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
    if filter_strategy.lower() == 'nms':
        filter_strategy = sv.OverlapFilter.NON_MAX_SUPPRESSION
    else:
        raise ValueError("Unsupported filter strategy provided - currently only NMS!")

    # Partially initialize the function - populate all inputs in advance besides "image", which is populated
    #   iteratively within sv.InferenceSlicer with image slices
    callback_fn = partial(callback, text_prompt=text_prompt, device=device, gdino_processor=gdino_processor,
                          gdino_model=gdino_model, inference_models_parameters=inference_models_parameters,
                          slice_inference_parameters=slice_inference_parameters)

    # Create a slicer object
    slicer = sv.InferenceSlicer(
        callback=callback_fn,
        slice_wh=slice_wh,
        overlap_wh=overlap_width_height,
        overlap_ratio_wh=None,
        iou_threshold=iou_threshold,
        overlap_filter=filter_strategy,
        thread_workers=thread_workers
    )

    # Run slicer (do detection on different slices)
    print("Running Grounding DINO with SAHI")
    detections = slicer(image)

    # Get class_ids relative to the original text_prompt and corresponding to class_names
    keys = text_prompt.split('.')  # → ['house','window','bicycle','door','grass','leaf']
    class_id_map = {k: i + 1 for i, k in enumerate(keys)}  # → {'house':1, 'window':2, ..., 'leaf':6}
    inverted_map = {v: k for k, v in class_id_map.items()}

    # Set main output variables
    class_names = [inverted_map[id] for id in detections.class_id]
    confidences = detections.confidence.tolist()
    class_ids = detections.class_id
    input_boxes = detections.xyxy

    # Remove too-large object detections (when approaching SAHI slice-size/area, likely to be erroneous)
    lor_threshold = slice_inference_parameters["large_object_removal_threshold"]  # lor = large object removal
    lor_max_area = lor_threshold * slice_wh[0] * slice_wh[1]  # percentage of SAHI slice area
    input_boxes_areas = (input_boxes[:, 2] - input_boxes[:, 0]) * (input_boxes[:, 3] - input_boxes[:, 1])
    keep_mask = input_boxes_areas < lor_max_area

    # Update main output variables
    class_names = list(compress(class_names, keep_mask))
    confidences = list(compress(confidences, keep_mask))
    class_ids = class_ids[keep_mask]
    input_boxes = input_boxes[keep_mask]



    # TODO: MOVE SAM2 predictions within SAHI pipeline! (or give it own SAHI with different parameters)
    # Set input for SAM2

    print("Running SAM2 - no SAHI")
    image = img_1to3_channels_encoding(image, normalize='0-1', output_dtype='float32',
                                       replace_nan_with='max', broadcast=True)

    sam2_predictor.set_image(image)

    # Batch detected bounding boxes to avoid memory explosion when running inference with SAM!
    # When storing masks separate them and store as a list of individual masks as sparse arrays!
    max_boxes_per_batch = 16  # conservative default
    masks = []

    for batch_i in range(0, len(input_boxes), max_boxes_per_batch):
        batch_boxes = input_boxes[batch_i:batch_i + max_boxes_per_batch]

        # Run SAM2
        masks_i, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=batch_boxes,
            multimask_output=False,
        )

        # Squeeze out unnecessary dimensions
        if masks_i.ndim == 4:
            masks_i = masks_i.squeeze(1)  # convert the shape to (n, H, W)
            # Store individual masks j of batch i as sparse booleans
            for mask_j in masks_i:
                masks.append(csr_matrix(mask_j.astype(bool), dtype=bool))

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()


    # Create mask labels (class name + confidence scores)
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    # Store results in a dictionary
    #   masks - list of sparse matrices
    #   input_boxes - np.ndarray of input boxes (N_boxes x 4)
    #   confidences - list of confidences per box (N_boxes,_)
    #   class_names - list of strings with class names per box (N_boxes,_)
    #   class_ids - np.ndarray of class_ids per box (N_boxes,_)
    #   mask_labels - list of strings with labels, name + confidence (N_boxes,_)

    results = {"masks": masks, "input_boxes": input_boxes, "confidences": confidences, "class_names": class_names,
               "class_ids": class_ids, "mask_labels": labels}

    return results


def mask_to_rle(mask):
    # Convert binary masks into RLE (Run-Length Encoding) - common for e.g. COCO-style datasets
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def rle_to_mask(rle):
    # Convert RLE (Run-Length Encoding) into a binary mask
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    mask = mask_util.decode(rle)
    # Remove channel dimension if needed (shape: H x W x 1 -> H x W)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    return mask


def save_gsam2_results(image: tuple[str, np.ndarray, Path], results: dict,
                       inference_models_parameters) -> None:
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
    image_data = img_1to3_channels_encoding(image_data, normalize='0-255', output_dtype='uint8',
                                       replace_nan_with='max', broadcast=True)

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
    masks = np.stack([sparse.toarray() for sparse in masks])

    # Create detection objects for "supervision useful API"
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids)

    # Save .jpg images of detected objects (bounding box, semantic label, score)
    #   - note: if you want to use default color map, you can set color=ColorPalette.DEFAULT

    #   - "supervision" library commands
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=image_data.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    #   - set output directory and file names for object detection
    output_dir_od = inference_models_parameters['output_dir_od']
    if subsampled_masks_flag:
        output_jpg_od = f"{image_path.stem}_od_RANDOM_SUBSAMPLE_16.jpg"
    else:
        output_jpg_od = f"{image_path.stem}_od.jpg"
    cv2.imwrite(os.path.join(output_dir_od, output_jpg_od), annotated_frame.astype(np.dtype('uint8')))

    # Save .jpg images of SAM masks (mask, semantic label, score)
    #   - "supervision" library commands

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame.astype(np.dtype('uint8')), detections=detections)
    #   - set output directory and file names for object detection
    output_dir_sam2 = inference_models_parameters['output_dir_sam2']
    if subsampled_masks_flag:
        output_jpg_sam2 = f"{image_path.stem}_sam2_RANDOM_SUBSAMPLE_16.jpg"
    else:
        output_jpg_sam2 = f"{image_path.stem}_sam2.jpg"
    cv2.imwrite(os.path.join(output_dir_sam2, output_jpg_sam2), annotated_frame.astype(np.dtype('uint8')))

    # 2. Create JSON file
    # -------------------

    if inference_models_parameters['dump_json_results']:
        # convert mask into rle format
        mask_rles = [mask_to_rle(mask) for mask in masks]
        # convert bounding boxes and scores (confidences) to lists
        input_boxes = input_boxes.tolist()
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()

        # save the results in standard format
        results = {
            "image_path": image_path.as_posix(),
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": image_data.shape[1],
            "img_height": image_data.shape[0],
        }

        output_dir_masks_json = inference_models_parameters['output_dir_masks_json']
        output_json = f"{image_path.stem}_gsam2_results.json"

        with open(os.path.join(output_dir_masks_json, output_json), "w") as f:
            json.dump(results, f, indent=4)

    return None
