from typing import Tuple, List, Union
import numpy as np
import supervision as sv
from supervision.detection.tools.inference_slicer import InferenceSlicer, crop_image, move_detections
from supervision import Detections


def move_sparse_masks(
    detections: Detections,
    offset: np.ndarray[np.int32],
    resolution_wh: Tuple[int, int],
) -> Detections:
    """
    Shift sparse masks of an image slice to correct positions within full high-resolution image.

    Parameters:
        detections (sv.Detections): Detections object with sparse_masks in detections.data["sparse_masks"] to be moved
            - sparse_masks: List[scipy.sparse.spmatrix] — list of sparse slice-local masks (one per detected object)
        offset: NDArray[np.int32] - An array of shape `(2,)` containing non-negative
            int values `[dx, dy]`, upper left corner coordinates of an image slice within high-res. image.
        resolution_wh: Tuple[int, int] - The width and height of the full high-resolution image.

    Returns:
        masks: List[csr_matrix[np.bool_]] — list of sparse global (high-res. image) masks.
    """
    # Get masks from detection object

    # Check if masks empty, if empty, no need to offset anything
    masks = detections.data.get("sparse_masks", [])
    if not masks:
        return detections

    # Make checks
    if offset[0] < 0 or offset[1] < 0:
        raise ValueError(f"Offset values must be non-negative integers. Got: {offset}")

    # Image slice offsets
    y_offset, x_offset = offset[1], offset[0]
    # Full high-resolution image width and height
    w_image, h_image = resolution_wh[0], resolution_wh[1]

    # Apply offsets:
    for i, mask_i in enumerate(masks):

        # Apply offsets
        new_row = mask_i[:, 0] + y_offset
        new_col = mask_i[:, 1] + x_offset

        # Filter out-of-bounds entries (just in case)
        valid = (
            (new_row >= 0) & (new_row < h_image) &
            (new_col >= 0) & (new_col < w_image)
        )

        # Create new shifted sparse matrix in global space
        shifted_mask = np.vstack((new_row, new_col), dtype=np.int32).T
        shifted_mask = shifted_mask[valid]

        # Replace old with new in a list
        masks[i] = shifted_mask

        # Update detections:
        detections.data["sparse_masks"] = masks

    return detections


class SparseMasksInferenceSlicer(InferenceSlicer):
    """
    Extension for supervision.InferenceSlicer (SAHI) that:
        1 - stores sparse masks instead of full np.ndarray-s
        2 - stores image slice offsets (x0, y0) per object detection (to allow for mask transformation)
       (3 - merges detection masks into unified semantic and instance masks - 2x uint16 numpy ndarray)
        Remark: "3" is currently implemented and applied separately (after running this class)

    How stored:
        offsets -
        sparse masks -
    """
    def _run_callback(self, image: np.ndarray, offset: np.ndarray) -> sv.Detections:
        image_slice = crop_image(image=image, xyxy=offset)
        detections = self.callback(image_slice)

        # shift detection bounding boxes & masks from slice pixel location to full-image location
        resolution_wh = (image.shape[1], image.shape[0])
        detections = move_detections(detections, offset[:2], resolution_wh=resolution_wh)

        # Move masks
        offset_int32 = offset[:2].astype(np.int32)
        detections = move_sparse_masks(detections, offset=offset_int32, resolution_wh=resolution_wh)

        return detections

