from typing import List, Tuple
import supervision as sv
import numpy as np
from itertools import compress
import pycocotools.mask as mask_util

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


# Create empty detections object for early terminations:
def return_empty_detections() -> sv.Detections:
    empty_detections = sv.Detections(xyxy=np.empty((0, 4)), confidence=np.array([]), class_id=np.array([]))
    empty_detections.data["sparse_masks"] = []
    return empty_detections


def check_if_img_slice_empty(image_slice: np.ndarray, slice_inference_parameters: dict) -> bool:
    # Check if image slice is predominantly empty space:
    max_val = np.max(image_slice)
    max_count = np.count_nonzero(image_slice == max_val)
    all_count = image_slice.size
    #   if more pixels than a threshold have value = max_val -> slice predominantly empty and will not be processed
    empty_slice_removal_threshold = slice_inference_parameters['empty_slice_removal_threshold']

    if (max_count/all_count) > empty_slice_removal_threshold:
        return False
    else:
        return True


def check_if_img_slice_complete(image_slice: np.ndarray, slice_inference_parameters: dict) -> bool:
    #   get real image slice height and width
    slice_height, slice_width = image_slice.shape[:2]
    #   get expected image slice height and width
    slice_height_expected, slice_width_expected = slice_inference_parameters["slice_width_height"]
    #   if not matching -> exit
    if slice_height != slice_height_expected or slice_width != slice_width_expected:
        return False
    else:
        return True


def parse_gdino_results(gdino_results, text_prompt: str) -> Tuple:
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
    keys = text_prompt.split('.')  # → ['house','window','bicycle','door','grass','leaf']
    id_map = {k: i + 1 for i, k in enumerate(keys)}  # → {'house':1, 'window':2, ..., 'leaf':6}
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


def remove_too_large_detections(input_boxes, class_names, class_ids, confidences,
                                inference_models_parameters, slice_width, slice_height) -> Tuple:

    # Remove too-large object detections (when approaching SAHI slice-size/area, likely to be erroneous)
    lor_threshold = inference_models_parameters["large_object_removal_threshold"]  # lor = large object removal
    lor_max_area = lor_threshold * slice_width * slice_height  # percentage of SAHI slice area
    input_boxes_areas = (input_boxes[:, 2] - input_boxes[:, 0]) * (input_boxes[:, 3] - input_boxes[:, 1])
    keep_mask = input_boxes_areas < lor_max_area

    # Update main output variables
    class_names = list(compress(class_names, keep_mask))
    confidences = confidences[keep_mask]
    class_ids = class_ids[keep_mask]
    input_boxes = input_boxes[keep_mask]

    return input_boxes, class_names, class_ids, confidences


def post_process_gdino_results(gdino_processor, outputs, inputs, text_prompt, inference_models_parameters,
                               slice_height, slice_width) -> Tuple:
    # Postprocess Grounded DINO results

    # - filters out bounding boxes and text predictions with low confidence scores,
    #   resizes the predictions to original size
    gdino_results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=inference_models_parameters['box_threshold'],
        text_threshold=inference_models_parameters['text_threshold'],
        target_sizes=[(slice_height, slice_width)]
    )

    #   - prepare results for supervision library Detections object:
    input_boxes, class_names, class_ids, confidences = parse_gdino_results(gdino_results, text_prompt)

    #   - remove too-large object detections (when approaching SAHI slice-size/area, likely to be erroneous)
    input_boxes, class_names, class_ids, confidences = remove_too_large_detections(input_boxes, class_names, class_ids,
                                                                                   confidences,
                                                                                   inference_models_parameters,
                                                                                   slice_width, slice_height)
    #   - everything is empty flag
    if not class_names:
        empty_results_flag = True
    else:
        empty_results_flag = False

    return input_boxes, class_names, class_ids, confidences, empty_results_flag


def run_sam2_bbox_prompt_inference_in_batches(sam2_predictor, input_boxes, sam_box_prompt_batch_size, masks) -> List:

    for batch_i in range(0, len(input_boxes), sam_box_prompt_batch_size):
        # Get batch
        batch_boxes = input_boxes[batch_i:batch_i + sam_box_prompt_batch_size]
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
        # Append to list
        masks.append(masks_i)

    return masks


def convert_masks_to_sparse_masks(masks: list) -> List:
    sparse_masks = []
    if masks:
        # Transform masks into a single numpy.ndarray from a list of batches
        masks = np.concatenate(masks, axis=0)

        # Store individual masks j of batch i as sparse booleans
        for mask_i in masks:
            row, column = np.nonzero(mask_i)
            sparse_masks.append(np.vstack((row, column), dtype=np.int32).T)

    return sparse_masks


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



