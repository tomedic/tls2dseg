
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

def initialize_sam2_everyting(inference_models_parameters: dict) -> SAM2AutomaticMaskGenerator:
    # build SAM2 image predictor (Meta GitHub workflow)
    # https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb
    sam2_checkpoint = inference_models_parameters['sam2-checkpoint']
    model_cfg = inference_models_parameters['sam2-model-config']
    device = inference_models_parameters['device']
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    #sam2_everything = SAM2AutomaticMaskGenerator(sam2_model)

    # sam2_everything = SAM2AutomaticMaskGenerator(
    #     model=sam2_model,
    #     points_per_side=128,
    #     points_per_batch=128,
    #     pred_iou_thresh=0.6,
    #     stability_score_thresh=0.8,
    #     stability_score_offset=0.7,
    #     crop_n_layers=1,
    #     box_nms_thresh=0.8,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=250.0,
    #     use_m2m=True,
    # )

    sam2_everything = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.8,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.8,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=250.0,
        use_m2m=True,
    )


    return sam2_everything



def run_sam2_everything(image: Path | np.ndarray,
                      sam2_everything: SAM2AutomaticMaskGenerator,
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

    # Run SAM2 everything
    masks = sam2_everything.generate(np.array(image.convert("RGB")))  # Assure np.ndarray with RGB channels

    # Show visualization
    np.random.seed(3)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()


    return masks