"""Shared inference helpers — Grounded-DINO post-processing + mask utilities.

Phase 4 plan 04-04 Task 3 (ENG-07, D-C-01).

This module is the single home for all helpers shared between both SAM2
inference engines (GroundedSAM2Engine and GroundedSAM2HFEngine).  It
deliberately imports ONLY numpy and stdlib at module level so that it is
importable in the tier_a venv (no torch/sam2/transformers/supervision/
pycocotools required for import-time — D-A-05).

Functions moved from:
- tls2dseg.grounded_sam2_utils lines 8-205
  (resolve_class_names, return_empty_detections, check_if_img_slice_empty,
   check_if_img_slice_complete, parse_gdino_results,
   remove_too_large_detections, remove_detections_touching_image_edges,
   post_process_gdino_results, convert_masks_to_sparse_masks)
- tls2dseg.pc2img_utils lines 435-551 + 615-710
  (get_instance_and_semantic_mask, get_instance_and_semantic_mask_with_confidence,
   img_1to3_channels_encoding)
- tls2dseg.pc2img_utils lines 713-758
  (get_per_mask_depth, get_per_mask_depth_parallel)
- tls2dseg.detections_2d
  (merge_list_of_2d_detections, filter_out_samples_in_2d_detections,
   _compute_2d_mask_features, _compute_2d_mask_features_worker, d2d_outlier_removal)
- tls2dseg.grounded_sam2_utils lines 243-258
  (mask_to_rle, rle_to_mask)
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import compress
from typing import Literal

import numpy as np

from tls2dseg.config.text import split_class_keys

logger = logging.getLogger("tls2dseg.engines.inference.shared")


# ─────────────────────────────────────────────────────────────────────────────
# DINO post-processing helpers (moved verbatim from grounded_sam2_utils.py)
# ─────────────────────────────────────────────────────────────────────────────


def resolve_class_names(class_names: list[str], valid_keys: list[str]) -> list[str | None]:
    """
    Resolves a list of class names to valid keys.
    If class_name is directly in valid_keys, keep it, else, search for the first valid_key that is a substring of
     class_name.

    Args:
        class_names: List of predicted or raw class names.
        valid_keys: List of valid class labels.

    Returns:
        List of resolved class names. May contain None for names with no valid_keys match;
        callers (see parse_gdino_results) filter those out downstream.
    """
    resolved: list[str | None] = []
    for name in class_names:
        if name in valid_keys:
            resolved.append(name)
        else:
            match = next((key for key in valid_keys if key in name), None)
            resolved.append(match)
    return resolved


# Create empty detections object for early terminations:
def return_empty_detections():
    """Return an empty supervision.Detections object (heavy import confined here)."""
    import supervision as sv  # tier_b only — supervision not in tier_a venv

    empty_detections = sv.Detections(xyxy=np.empty((0, 4)), confidence=np.array([]), class_id=np.array([]))
    empty_detections.data["sparse_masks"] = []
    return empty_detections


def check_if_img_slice_empty(image_slice: np.ndarray, slice_inference_parameters: dict) -> bool:
    # Check if image slice is predominantly empty space:
    max_val = np.max(image_slice)
    max_count = np.count_nonzero(image_slice == max_val)
    all_count = image_slice.size
    #   if more pixels than a threshold have value = max_val -> slice predominantly empty and will not be processed
    empty_slice_removal_threshold = slice_inference_parameters["empty_slice_removal_threshold"]

    return (max_count / all_count) <= empty_slice_removal_threshold


def check_if_img_slice_complete(image_slice: np.ndarray, slice_inference_parameters: dict) -> bool:
    #   get real image slice height and width
    slice_height, slice_width = image_slice.shape[:2]
    #   get expected image slice height and width
    slice_height_expected, slice_width_expected = slice_inference_parameters["slice_width_height"]
    #   if not matching -> exit
    return slice_height == slice_height_expected and slice_width == slice_width_expected


def parse_gdino_results(gdino_results, text_prompt: str) -> tuple:
    """
    Info:
        Prepare results for supervision library Detections object. Unpack Grounded Dino results
        objects containing GPU-based tensors and return relevant data in np.ndarrays and lists.

        Removes any detections that are eventually not related to any valid semantic class defined by a text-prompt

    """
    input_boxes = gdino_results[0]["boxes"].cpu().numpy()  # get the bounding box prompts for SAM2
    confidences = gdino_results[0]["scores"].cpu().numpy()  # tensor -> ndarray
    class_names = gdino_results[0]["labels"]  # get class names

    # Get class_ids corresponding defined relative to the original text_prompt and corresponding to class_names
    keys = split_class_keys(text_prompt)
    id_map = {k: i + 1 for i, k in enumerate(keys)}
    class_names = resolve_class_names(class_names, keys)  # if class name corresponds to 2 valid classes - pick 1

    # Removing eventual detections that are not related to any of the valid class categories
    # and assigning class_ids per detection based on the ID map
    none_indices = [i for i, name in enumerate(class_names) if name is None]
    class_names = [elem for i, elem in enumerate(class_names) if i not in none_indices]
    class_ids = np.array([id_map[q] for q in class_names])  # a list of corresponding class IDs
    mask = np.ones(confidences.shape[0], dtype=bool)
    mask[none_indices] = False
    input_boxes = input_boxes[mask]
    confidences = confidences[mask]

    return input_boxes, class_names, class_ids, confidences


def remove_too_large_detections(
    input_boxes,
    class_names,
    class_ids,
    confidences,
    inference_models_parameters,
    slice_width,
    slice_height,
) -> tuple:

    # Remove too-large object detections (when approaching SAHI slice-size/area, likely to be erroneous)
    lor_threshold = inference_models_parameters["large_object_removal_threshold"]  # lor = large object removal

    # Early stopping
    if lor_threshold is None:
        return input_boxes, class_names, class_ids, confidences

    # Detecting too large instances
    lor_max_area = lor_threshold * slice_width * slice_height  # percentage of SAHI slice area
    input_boxes_areas = (input_boxes[:, 2] - input_boxes[:, 0]) * (input_boxes[:, 3] - input_boxes[:, 1])
    keep_mask = input_boxes_areas < lor_max_area

    # Update main output variables
    class_names = list(compress(class_names, keep_mask))
    confidences = confidences[keep_mask]
    class_ids = class_ids[keep_mask]
    input_boxes = input_boxes[keep_mask]

    return input_boxes, class_names, class_ids, confidences


def remove_detections_touching_image_edges(
    input_boxes,
    class_names,
    class_ids,
    confidences,
    inference_models_parameters,
    slice_width,
    slice_height,
) -> tuple:

    # How close (in pixels) can a bounding box edge be to an image (or image slice) edge, and still be accepted
    threshold = inference_models_parameters["partial_detection_edge_touching_threshold"]

    # Early stopping:
    if threshold is None:
        return input_boxes, class_names, class_ids, confidences

    # Detect boxes touching left & bottom edge
    criteria_1 = np.any(input_boxes[:, :2] < threshold, axis=1)
    # Detect boxes touching right edge
    criteria_2 = input_boxes[:, 2] > slice_width - threshold
    # Detect boxes touching top edge
    criteria_3 = input_boxes[:, 3] > slice_height - threshold
    # Combine into single keep mask
    keep_mask = ~(criteria_1 | criteria_2 | criteria_3)

    # Update main output variables
    class_names = list(compress(class_names, keep_mask))
    confidences = confidences[keep_mask]
    class_ids = class_ids[keep_mask]
    input_boxes = input_boxes[keep_mask]

    return input_boxes, class_names, class_ids, confidences


def post_process_gdino_results(
    gdino_processor,
    outputs,
    inputs,
    text_prompt,
    inference_models_parameters,
    slice_height,
    slice_width,
) -> tuple:
    # Postprocess Grounded DINO results

    # - filters out bounding boxes and text predictions with low confidence scores,
    #   resizes the predictions to original size
    # transformers >=4.51 renamed the GroundingDino post-process kwarg
    # box_threshold -> threshold (same meaning: box/query confidence cutoff);
    # text_threshold is unchanged. The tls2dseg config key stays "box_threshold".
    gdino_results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=inference_models_parameters["box_threshold"],
        text_threshold=inference_models_parameters["text_threshold"],
        target_sizes=[(slice_height, slice_width)],
    )

    #   - prepare results for supervision library Detections object:
    input_boxes, class_names, class_ids, confidences = parse_gdino_results(gdino_results, text_prompt)

    #   - remove too-large object detections (when approaching SAHI slice-size/area, likely to be erroneous)
    input_boxes, class_names, class_ids, confidences = remove_too_large_detections(
        input_boxes,
        class_names,
        class_ids,
        confidences,
        inference_models_parameters,
        slice_width,
        slice_height,
    )

    #   - remove detections touching edges
    input_boxes, class_names, class_ids, confidences = remove_detections_touching_image_edges(
        input_boxes,
        class_names,
        class_ids,
        confidences,
        inference_models_parameters,
        slice_width,
        slice_height,
    )

    #   - everything is empty flag
    empty_results_flag = not class_names

    return input_boxes, class_names, class_ids, confidences, empty_results_flag


def convert_masks_to_sparse_masks(masks: list) -> list:
    sparse_masks = []
    if masks:
        # Transform masks into a single numpy.ndarray from a list of batches
        masks = np.concatenate(masks, axis=0)

        # Store individual masks j of batch i as sparse booleans
        for mask_i in masks:
            row, column = np.nonzero(mask_i)
            sparse_masks.append(np.vstack((row, column), dtype=np.int32).T)

    return sparse_masks


# ─────────────────────────────────────────────────────────────────────────────
# RLE helpers (moved verbatim from grounded_sam2_utils.py)
# ─────────────────────────────────────────────────────────────────────────────


def mask_to_rle(mask):
    # Convert binary masks into RLE (Run-Length Encoding) - common for e.g. COCO-style datasets
    import pycocotools.mask as mask_util  # tier_b only

    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def rle_to_mask(rle):
    # Convert RLE (Run-Length Encoding) into a binary mask
    import pycocotools.mask as mask_util  # tier_b only

    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    mask = mask_util.decode(rle)
    # Remove channel dimension if needed (shape: H x W x 1 -> H x W)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Instance/semantic mask helpers (moved verbatim from pc2img_utils.py)
# ─────────────────────────────────────────────────────────────────────────────


def get_instance_and_semantic_mask(results: dict, text_prompt) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Creates 1 representative instance and 1 semantic segmentation mask from N individual object masks.

    Args:
        results: A dictionary with grounded_sam2 results containing all instance/semantics segmentation info
                'masks' with M x w x h (M = mask number, w = width, h = height),
                'input_boxes' with input bounding boxes (results of object detection),
                'confidences' with confidence scores,
                'class_names', 'class_ids', ...
        text_prompt: a string with text prompts used for object detection with GroundedDINO
                     (each "object" separated by a dot ".")
    Returns:
        instance_mask: A NumPy array of shape (H, W) with unique labels for each instance.
        semantic_mask: A NumPy array of shape (H, W) with labels for each semantic class.
        class_ids: A dictionary with str class_name int class_id value-pairs
    """

    H, W = results["masks"][0].shape  # Mask/image size
    N = len(results["masks"])  # Number of detections

    # Get dictionary mapping "semantic classes" to unique IDs
    keys = split_class_keys(text_prompt)
    id_map = {k: i + 1 for i, k in enumerate(keys)}
    class_ids = [id_map[q] for q in results["class_names"]]  # a list of corresponding class IDs
    results["class_ids"] = class_ids  # Store real class IDs corresponding to detected classes, not range(#C)

    # Sorting masks from biggest to smallest, so if overlapping, the big ones do not superimpose the small ones
    mask_sizes = np.zeros(N, dtype=int)
    for i in range(N):
        mask_sizes[i] = results["masks"][i].nnz
    # Get indices sorted from biggest to smallest
    sorted_indices = np.argsort(mask_sizes)[::-1]

    # Initialize masks with zeros (background)
    instance_mask = np.zeros((H, W), dtype=np.int32)
    semantic_mask = np.zeros((H, W), dtype=np.int32)

    for i in range(N):
        # Get the instance mask
        mask_i = results["masks"][sorted_indices[i]].toarray().astype(bool)  # Shape: (H, W), dtype: bool
        # Assign a unique label to each instance in the instance mask
        # Labels start from 1 (to have 0 for background)
        instance_label = i + 1
        instance_mask[mask_i] = instance_label

        # Assign the semantic label to the semantic mask
        # Labels are class_id + 1 to avoid using 0
        semantic_mask[mask_i] = class_ids[sorted_indices[i]]

    return instance_mask, semantic_mask, id_map


def get_instance_and_semantic_mask_with_confidence(
    results: dict, text_prompt, image_hw: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Creates 1 representative instance and 1 semantic segmentation mask from N individual object masks.

    Args:
        results: A dictionary with grounded_sam2 results containing all instance/semantics segmentation info
                'masks' with M x w x h (M = mask number, w = width, h = height),
                'input_boxes' with input bounding boxes (results of object detection),
                'confidences' with confidence scores,
                'class_names', 'class_ids', ...
        text_prompt: a string with text prompts used for object detection with GroundedDINO
                     (each "object" separated by a dot ".")
        image_hw: a tuple with width and height of the image
    Returns:
        instance_mask: A NumPy array of shape (H, W) with unique labels for each instance.
        semantic_mask: A NumPy array of shape (H, W) with labels for each semantic class.
        confidence_mask: A NumPy array of shape (H, W) with confidences for each instance (detection).
        class_ids: A dictionary with str class_name int class_id value-pairs
    """

    H, W = image_hw  # Mask/image size
    N = len(results["masks"])  # Number of detections

    # Get dictionary mapping "semantic classes" to unique IDs
    keys = split_class_keys(text_prompt)
    id_map = {k: i + 1 for i, k in enumerate(keys)}
    class_ids = [id_map[q] for q in results["class_names"]]  # a list of corresponding class IDs
    results["class_ids"] = class_ids  # Store real class IDs corresponding to detected classes, not range(#C)

    # Get confidences
    confidences = results["confidences"]

    # Sorting masks from biggest to smallest, so if overlapping, the big ones do not superimpose the small ones
    mask_sizes = np.zeros(N, dtype=int)
    for i in range(N):
        mask_sizes[i] = results["masks"][i].shape[0]
    # Get indices sorted from biggest to smallest
    sorted_indices = np.argsort(mask_sizes)[::-1]

    # Initialize masks with zeros (background)
    instance_mask = np.zeros((H, W), dtype=np.uint32)
    semantic_mask = np.zeros((H, W), dtype=np.uint8)
    confidence_mask = np.zeros((H, W), dtype=np.float16)

    for i in range(N):
        # Get the instance mask row and column indices
        mask_i = results["masks"][sorted_indices[i]]
        mask_i = np.unique(mask_i, axis=0)

        # Assign a unique label to each instance in the instance mask
        # Labels start from 1 (to have 0 for background)
        instance_label = i + 1
        instance_mask[mask_i[:, 0], mask_i[:, 1]] = instance_label

        # Assign the semantic label to the semantic mask
        # Labels are class_id + 1 to avoid using 0
        semantic_mask[mask_i[:, 0], mask_i[:, 1]] = class_ids[sorted_indices[i]]

        # Assign confidence score i to confidence mask
        confidence_mask[mask_i[:, 0], mask_i[:, 1]] = confidences[sorted_indices[i]]

    return instance_mask, semantic_mask, confidence_mask, id_map


# ─────────────────────────────────────────────────────────────────────────────
# Image encoding helper (moved from pc2img_utils.py — TEST-05 target)
# Final home is this module (pure numpy, no heavy deps).
# ─────────────────────────────────────────────────────────────────────────────


def img_1to3_channels_encoding(
    img: np.ndarray,
    output_dtype: str | np.dtype | None = "float32",
    replace_nan_with: Literal["max", "min", "random", "zero"] | float = "max",
    normalize: Literal["0-1", "0-255"] | None = "0-1",
    broadcast: bool = True,
) -> np.ndarray:
    """
    Convert a HxWx, channel grayscale array with arbitrary value range and dtype into a HxWx3 channel grayscale array
    of selected dtype with 0-255 value range. Goal: Preparing image data for deep learning frameworks
    (e.g. HuggingFace transformers library, SAM/SAM2 by Facebook/Meta).

    Parameters
    ----------
    img : np.ndarray
        Input image 2-D array of Shape (H, W, ), dtype any, may contain NaNs.
    output_dtype : {'uint8', 'float32', None}
        Desired dtype of the output 3 channel image.
    replace_nan_with : {'max', 'min', 'random', 'zero'} or float
        Strategy for filling NaNs *before* further processing.
    normalize: {'0-1', '0-255'} or None
        If '0-1' or '0-255', normalize data to 0-1 or 0-255 values
        If None - do nothing.
    broadcast : bool
        If True return an O(1) broadcast view
        instead of materialising three copies.

    Returns
    -------
    img : np.ndarray
        Shape (H, W, 3) array of the requested dtype.
    """

    # Early stopping: if image already a 3 channel image with 0-255 value range, do nothing
    if img.ndim == 3 and img.shape[2] == 3:
        img_min, img_max = float(img.min()), float(img.max())
        if img_min >= 0.0 and img_max <= 255.0:
            return img

    # Checks for other cases:
    if img.ndim != 2:
        raise ValueError(
            "Input must be a 2-D array, check if your image is not already 3 channel image with 0-255 value range!"
        )

    if output_dtype not in ("uint8", "float32", None):
        raise ValueError("dtype must be 'uint8', 'float32', or None")

    # Make a copy to de-attach the image from the original ndarray
    img = img.copy()

    # ---- 1) handle NaNs ----------------------------------------------------
    nan_mask = np.isnan(img)
    img_min, img_max = np.nanmin(img), np.nanmax(img)
    if nan_mask.any():
        # Get values to replace nan with
        if replace_nan_with == "max":
            fill_val = img_max
        elif replace_nan_with == "min":
            fill_val = img_min
        elif replace_nan_with == "random":
            rng = np.random.default_rng()
            fill_val = rng.uniform(img_min, img_max, size=img.shape[:2])
        elif replace_nan_with == "zero":
            fill_val = 0.0
        elif isinstance(replace_nan_with, (int, float)):
            fill_val = float(replace_nan_with)
            img_max = max(replace_nan_with, img_max)
        else:
            raise ValueError("Invalid replace_nan_with option")
        # Replace values
        if replace_nan_with == "random":
            img[nan_mask] = fill_val[nan_mask]
        else:
            img[nan_mask] = fill_val

    # ---- 2) normalise slice-wise ------------------------------------------
    if normalize is not None and normalize in ("0-1", "0-255"):
        # constant slice -> all zeros, otherwise rescale to [0, 1]
        img = np.zeros_like(img, dtype=np.float32) if img_max == img_min else (img - img_min) / (img_max - img_min)
        if normalize == "0-255":
            img = img * 255.0
    elif normalize is None:
        pass
    else:
        raise ValueError("normalize must have one of the 3 following values: '0-1', '0-255', None")

    # ---- 3) cast to requested dtype ---------------------------------------
    if output_dtype is not None:
        output_dtype_np = np.dtype(output_dtype)
        img = img.astype(output_dtype_np)

    # ---- 4) replicate channels ---------------------------------------
    img = np.broadcast_to(img[..., None], (*img.shape, 3)) if broadcast else np.repeat(img[..., None], 3, axis=2)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Per-mask depth helpers (moved from pc2img_utils.py)
# ─────────────────────────────────────────────────────────────────────────────


def get_per_mask_depth(detections_2d: dict, images_of_pcd_i: list) -> dict:
    # Add a new field to detections_2d "object_distance" for each mask
    # Get masks:
    masks = detections_2d["masks"]
    # Get range image:
    image_features = [i for i in images_of_pcd_i[0]]
    range_image_index = image_features.index("range")
    range_image = images_of_pcd_i[range_image_index][1]

    object_distances = np.empty(len(masks), dtype=np.float32)
    for i, mask_i in enumerate(masks):
        mask_ranges = range_image[mask_i[:, 0], mask_i[:, 1]]
        object_distances[i] = np.nanmedian(mask_ranges)

    detections_2d["object_distances"] = object_distances

    return detections_2d


def _compute_mask_median(args):
    mask, range_image = args
    # extract all range values under this mask, compute nan-median
    return np.nanmedian(range_image[mask[:, 0], mask[:, 1]])


def get_per_mask_depth_parallel(detections_2d: dict, images_of_pcd_i: list, n_jobs: int | None = None) -> dict:
    # Add a new field to detections_2d "object_distance" for each mask
    # Get masks:
    masks = detections_2d["masks"]
    # pull out the range image the same way you already do:
    feature_names = [name[0] for name in images_of_pcd_i]
    range_image_index = feature_names.index("range")
    range_image = images_of_pcd_i[range_image_index][1]

    # prep arguments so each worker gets (mask, range_image)
    work_items = [(mask, range_image) for mask in masks]

    object_distances = np.empty(len(masks), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        # map returns in order if you use executor.map
        for i, med in enumerate(exe.map(_compute_mask_median, work_items)):
            object_distances[i] = med

    detections_2d["object_distances"] = object_distances
    return detections_2d


# ─────────────────────────────────────────────────────────────────────────────
# 2D detection helpers (moved verbatim from detections_2d.py)
# ─────────────────────────────────────────────────────────────────────────────


def merge_list_of_2d_detections(dict_list: list[dict]) -> dict:
    dict_keys = dict_list[0].keys()
    merged_dict = {}

    for key_i in dict_keys:
        vals_i = [dict_i[key_i] for dict_i in dict_list]

        # concatenate numpy arrays along axis 0
        if all(isinstance(val, np.ndarray) for val in vals_i):
            merged_dict[key_i] = np.concatenate(vals_i, axis=0)

        # flatten lists of lists using a list comprehension
        elif all(isinstance(val, list) for val in vals_i):
            merged_dict[key_i] = [item for sublist in vals_i for item in sublist]
        # rais an error if dtype under same key not consistent or not supported
        else:
            raise TypeError("not all dictionaries within a list of dictionary are consistently lists or np.ndarrays")

    return merged_dict


def filter_out_samples_in_2d_detections(detections_2d: dict, remove_mask: np.ndarray) -> dict:
    keep_mask = ~remove_mask
    N = len(remove_mask)
    detections_2d_filtered = {}

    for key, vals in detections_2d.items():
        if len(vals) != N:
            raise ValueError(f"Key '{key}' has length {len(vals)}, but remove_mask has {N}")

        if isinstance(vals, np.ndarray):
            detections_2d_filtered[key] = vals[keep_mask]
        elif isinstance(vals, list):
            detections_2d_filtered[key] = [v for v, keep in zip(vals, keep_mask, strict=False) if keep]
        else:
            raise TypeError(f"Filtering not supported for type {type(vals)} in key: '{key}'")

    return detections_2d_filtered


def _compute_2d_mask_features(mask_xy: np.ndarray, rng: float) -> np.ndarray:
    """
    Compute 5-D feature vector for a single 2-D mask.
    Returns: [log(major*range), log(minor*range),
              sin(2θ), cos(2θ),
              log(pixel_count*range²)]
    """
    # ----- PCA on 2-D pixel coordinates -----
    if mask_xy.shape[0] > 256:
        mask_xy = mask_xy[np.random.choice(mask_xy.shape[0], 256, replace=False)]

    # centre
    centred = mask_xy.astype(np.float64) - mask_xy.mean(axis=0)
    # covariance and eigen-decomposition
    cov = np.cov(centred, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort by descending eigenvalue
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]  # defining PCA-frame (ordered PCA eigenvectors)

    # get pc1 and project to pc1 to obtain extents
    pc1 = axes[:, 0]
    proj1 = centred @ pc1
    major = proj1.max() - proj1.min()

    # get pc2
    pc2 = axes[:, 1]
    proj2 = centred @ pc2
    minor = proj2.max() - proj2.min()

    # orientation (radians) and 2θ encoding
    theta = np.arctan2(pc1[1], pc1[0])
    sin2, cos2 = np.sin(2 * theta), np.cos(2 * theta)

    # pixel (point) count
    pix_cnt = mask_xy.shape[0]

    # ----- range normalisation & log transforms -----
    # range normalisation to account for different object distances
    major_phys = major * rng  # ∝ true length
    minor_phys = minor * rng  # ∝ true width
    area_phys = pix_cnt * rng**2  # ∝ true area
    # log transform to get closer to normal distribution
    major_phys = np.log1p(major_phys)
    minor_phys = np.log1p(minor_phys)
    area_phys = np.log1p(area_phys)

    # Final feature set
    features = np.array([major_phys, minor_phys, sin2, cos2, area_phys], dtype=np.float32)

    return features


def _compute_2d_mask_features_worker(args):
    mask, rng = args
    return _compute_2d_mask_features(mask, rng)


def d2d_outlier_removal(
    d2d_collection: dict,
    task_parameters: dict,
    per_class_separation: bool = False,
    confidence_interval: float = 0.99,
) -> tuple[dict, np.ndarray]:

    # Extract relevant values:
    masks = d2d_collection["masks"]
    ranges = d2d_collection["object_distances"]
    class_ids = d2d_collection["class_ids"]
    n_jobs = task_parameters["n_workers"]

    # Compute features for outlier removal using parallel computing:
    #   prep arguments for each worker (mask, range_image)
    work_items = list(zip(masks, ranges, strict=False))
    #   initialize empty results
    features_or = np.empty((len(masks), 5), dtype=np.float32)
    #   run computing PCA-based features in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        for i, feat in enumerate(exe.map(_compute_2d_mask_features_worker, work_items)):
            features_or[i] = feat

    from tls2dseg.statistics_generalizable import multivariate_normal_outlier_removal

    # Collapse different classes (optional):
    if per_class_separation is False:
        class_ids = np.ones_like(class_ids)

    # Get unique classes:
    unique_cls = np.unique(class_ids)

    # Compute statistics and outliers per class:
    is_outlier = np.zeros_like(class_ids, dtype=np.bool_)
    nd = features_or.shape[1]  # get number of dimensions
    for cls in unique_cls:
        class_mask = class_ids == cls
        nr_samples = np.sum(class_mask)
        if nr_samples > nd * 10:
            features_i = features_or[class_mask]
            is_outlier_i = multivariate_normal_outlier_removal(features_i, confidence_interval)
            is_outlier[class_mask] = is_outlier_i
        else:
            logger.warning("2d statistical outlier removal skipped for class %s due to too few samples", cls)

    # Filter out 2d detections
    d2d_collection = filter_out_samples_in_2d_detections(d2d_collection, is_outlier)

    # Return all outliers:
    return d2d_collection, is_outlier
