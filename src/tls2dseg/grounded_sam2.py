# Imports
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import Tuple
from PIL import Image
import os
import cv2
import json
import supervision as sv
import pycocotools.mask as mask_util
from supervision.draw.color import ColorPalette
from src.tls2dseg.supervision_utils import CUSTOM_COLOR_MAP


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


def run_grounded_sam2(image: Path | np.ndarray, text_prompt: str,
                      gdino_model: AutoModelForZeroShotObjectDetection,
                      gdino_processor: AutoProcessor,
                      sam2_predictor: SAM2ImagePredictor,
                      inference_models_parameters: dict) -> dict:


    # Set inference hardware
    device = inference_models_parameters['device']

    # Set image as Pillow (PIL) library object
    if isinstance(image, np.ndarray):
        # Transform np.ndarray to RGB pillow image
        image = Image.fromarray(image)
    elif isinstance(image, Path):
        image = Image.open(image)  # Load image as PIL Image
    else:
        print("Image passed to run_grounded_sam2() has to be np.ndarray or Path object")

    # Run Grounded DINO
    #   - Preprocess data: normalize and rescale images, tokenize text, transform into tensor
    inputs = gdino_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
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
        target_sizes=[image.size[::-1]]
    )

    # Set input for SAM2
    sam2_predictor.set_image(np.array(image.convert("RGB")))  # Assure np.ndarray with RGB channels
    input_boxes = gdino_results[0]["boxes"].cpu().numpy()  # get the bounding box prompts for SAM2

    # Run SAM2
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Move results in readable formats (tensor -> numpy, lists) and adjust them
    if masks.ndim == 4:
        masks = masks.squeeze(1)  # convert the shape to (n, H, W)

    confidences = gdino_results[0]["scores"].cpu().numpy().tolist()  # tensor -> ndarray
    class_names = gdino_results[0]["labels"]  # get class names
    class_ids = np.array(list(range(len(class_names))))  # get class ids

    # Create mask labels (class name + confidence scores)
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    # Store results in a dictionary
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
    output_jpg_od = f"{image_path.stem}_od.jpg"
    cv2.imwrite(os.path.join(output_dir_od, output_jpg_od), annotated_frame)

    # Save .jpg images of SAM masks (mask, semantic label, score)
    #   - "supervision" library commands
    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    #   - set output directory and file names for object detection
    output_dir_sam2 = inference_models_parameters['output_dir_sam2']
    output_jpg_sam2 = f"{image_path.stem}_sam2.jpg"
    cv2.imwrite(os.path.join(output_dir_sam2, output_jpg_sam2), annotated_frame)

    # 2. Create JSON file
    # -------------------

    if inference_models_parameters['dump_json_results']:
        # convert mask into rle format
        mask_rles = [mask_to_rle(mask) for mask in masks]
        # convert bounding boxes and scores (confidences) to lists
        input_boxes = input_boxes.tolist()
        # scores = scores

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







