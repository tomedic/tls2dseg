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
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src.tls2dseg.pc2img_utils import pc2img_run
from PIL import Image

# Grounded-SAM2:
import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from supervision.draw.color import ColorPalette
from src.tls2dseg.supervision_utils import CUSTOM_COLOR_MAP
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection



# Input parameters:
pcd_path = "./data/test_small.e57"  # Set path to point cloud
bbox_model_id = "IDEA-Research/grounding-dino-base"  # Set object detection model
# TODO: Currently SAM2 fixed to large model (maybe free choice necessary at some point...)

# Image generation parameters:
# TODO: Important - a lot of hyperparameters for generating image hidden in pc2img_run() function
rotate_pcd = False  # Should I rotate point cloud before generating an image?
theta = 0  # Rotate point cloud around Z-axis for theta degrees
image_width = 2000  # Image width in pixels (height adjusted according to aspect ratio)
rasterization_method: str = 'nanconv'
features: list = ["intensity"]

# Prompt object detection / segmentation
text_prompt = "house.window.bicycle.wall.grass.leaf"



def main():

    # Auto input parameters:
    pcd_path_pathlib = Path(pcd_path)  # Create pathlibpath
    pcd: PointCloudData = load_e57(pcd_path_pathlib, stay_global=False)  # Load point cloud
    image_height = int(image_width // pcd.fov.ratio())

    device = "cuda" if torch.cuda.is_available() else "cpu"  # Set inference hardware

    # Generate image:
    images_i = pc2img_run(pcd, pcd_path_pathlib, rotate_pcd, theta, image_height, image_width,
                     rasterization_method, features)

    # Load image:
    image4seg = Image.fromarray(images_i[0][1])

    # Hyper-parameters:

    # TODO: SET TO ALLOW FOR CLI calls
    parser = argparse.ArgumentParser()
    parser.add_argument('--grounding-model', default=bbox_model_id)
    parser.add_argument("--text-prompt", default=text_prompt)
    parser.add_argument("--img-path", default=str(images_i[0][2]))
    parser.add_argument("--sam2-checkpoint", default="/scratch/projects/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="outputs/test_sam2.1")
    parser.add_argument("--no-dump-json", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    GROUNDING_MODEL = args.grounding_model
    TEXT_PROMPT = args.text_prompt
    IMG_PATH = args.img_path
    SAM2_CHECKPOINT = args.sam2_checkpoint
    SAM2_MODEL_CONFIG = args.sam2_model_config
    DEVICE = device
    OUTPUT_DIR = Path(args.output_dir)
    DUMP_JSON_RESULTS = not args.no_dump_json

    # create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # environment settings
    # use bfloat16
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino from huggingface
    model_id = GROUNDING_MODEL
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = TEXT_PROMPT
    img_path = IMG_PATH

    image = Image.open(img_path)

    sam2_predictor.set_image(np.array(image.convert("RGB")))

    # Run Grounded DINO:
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # Post-process Grounded DINO predictions
    # TODO: PLAY WITH HYPER-PARAMETERS COULD BE IMPORTANT FOR WHEAT HEADS
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # RUN SAM2 with Grounded DINO Box Prompts
    # get the box prompt for SAM 2
    input_boxes = results[0]["boxes"].cpu().numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]
    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    """
    Visualize image with supervision useful API
    """
    img = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    """
    Note that if you want to use default color map,
    you can set color=ColorPalette.DEFAULT
    """
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

    """
    Dump the results in standard format and save as json files
    """

    def single_mask_to_rle(mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    if DUMP_JSON_RESULTS:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "image_path": img_path,
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
            "img_width": image.width,
            "img_height": image.height,
        }

        with open(os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
            json.dump(results, f, indent=4)

    test = 1
    banana = 2







if __name__ == "__main__":
    main()

