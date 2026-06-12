"""GroundedSAM2Engine — direct sam2 inference engine.

Phase 4 plan 04-04 Task 4 (ENG-03, ENG-05, D-C-01).

This engine implements InferenceEngine using the direct sam2 package
(Meta GitHub workflow). Heavy deps (torch, sam2, transformers) are confined
to __init__ method bodies (D-A-05) so that importing this module never
triggers model loading.

The DINO + SAHI + post-processing half is SHARED with GroundedSAM2HFEngine;
the SAM2 half (run_sam2_bbox_prompt_inference_in_batches via
SAM2ImagePredictor.predict) is the seam specific to this engine (D-C-01).
"""

from __future__ import annotations

import logging
from functools import partial
from threading import Lock

import numpy as np

from tls2dseg.engines.inference.shared import (
    check_if_img_slice_complete,
    check_if_img_slice_empty,
    convert_masks_to_sparse_masks,
    img_1to3_channels_encoding,
    post_process_gdino_results,
    return_empty_detections,
)

# sahi_slicer imports supervision at module level (tier_b dep) — imported
# lazily inside detect() where slicing is requested (D-A-05).
from tls2dseg.grounded_sam2_utils import run_sam2_bbox_prompt_inference_in_batches
from tls2dseg.types import Detections2D, InferenceRequest

logger = logging.getLogger("tls2dseg.engines.inference.grounded_sam2")


class GroundedSAM2Engine:
    """Direct sam2 inference engine (Meta GitHub workflow).

    Conforms to the ``InferenceEngine`` Protocol (engines/protocols.py).

    Parameters
    ----------
    object_detection_model_id : str
        HF model id for Grounded-DINO.
    sam2_checkpoint : str
        Path to the local SAM2 checkpoint file.
    sam2_model_config : str
        Path to the SAM2 model config YAML (relative to the sam2 package).
    sam_box_prompt_batch_size : int
        Number of bounding boxes per SAM2 forward pass.
    device : str
        Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        *,
        object_detection_model_id: str,
        sam2_checkpoint: str,
        sam2_model_config: str,
        sam_box_prompt_batch_size: int,
        device: str,
    ) -> None:
        # Heavy imports confined here (D-A-05)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._sam_box_prompt_batch_size = sam_box_prompt_batch_size
        self._device = device
        self._sam_lock = Lock()

        # Grounded-DINO
        logger.info("Loading Grounded-DINO model: %s", object_detection_model_id)
        self._gdino_processor: AutoProcessor = AutoProcessor.from_pretrained(object_detection_model_id)
        self._gdino_model: AutoModelForZeroShotObjectDetection = AutoModelForZeroShotObjectDetection.from_pretrained(
            object_detection_model_id
        ).to(device)

        # SAM2 (direct)
        logger.info("Loading SAM2 checkpoint: %s", sam2_checkpoint)
        sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=device)
        self._sam2_predictor: SAM2ImagePredictor = SAM2ImagePredictor(sam2_model)

    def _run_callback(
        self,
        image_slice: np.ndarray,
        *,
        text_prompt: str,
        inference_models_parameters: dict,
        slice_inference_parameters: dict,
    ):
        """Run DINO + SAM2 inference on a single image or image slice."""
        import gc

        import torch

        slice_height, slice_width = image_slice.shape[:2]

        # Early-exit checks
        if not check_if_img_slice_empty(image_slice, slice_inference_parameters):
            return return_empty_detections()
        if not check_if_img_slice_complete(image_slice, slice_inference_parameters):
            return return_empty_detections()

        # Encode image: NaN handling + 3-channel normalisation
        image_slice = img_1to3_channels_encoding(
            image_slice, normalize="0-1", output_dtype="float32", replace_nan_with="max", broadcast=True
        )

        # Grounded-DINO
        inputs = self._gdino_processor(
            images=image_slice,
            text=text_prompt,
            return_tensors="pt",
            do_rescale=False,
        ).to(self._device)
        with torch.no_grad():
            outputs = self._gdino_model(**inputs)

        input_boxes, _, class_ids, confidences, empty_flag = post_process_gdino_results(
            self._gdino_processor,
            outputs,
            inputs,
            text_prompt,
            inference_models_parameters,
            slice_height,
            slice_width,
        )

        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        if empty_flag:
            return return_empty_detections()

        # SAM2 (direct)
        masks: list = []
        with self._sam_lock:
            self._sam2_predictor.set_image(image_slice)
            masks = run_sam2_bbox_prompt_inference_in_batches(
                self._sam2_predictor, input_boxes, self._sam_box_prompt_batch_size, masks
            )

        masks = convert_masks_to_sparse_masks(masks)

        torch.cuda.empty_cache()
        gc.collect()

        # Build supervision Detections
        import supervision as sv

        if not (len(masks) == input_boxes.shape[0] == confidences.shape[0] == class_ids.shape[0]):
            raise ValueError(
                "Grounded SAM2 inference produced mismatched output lengths "
                "(input_boxes, confidences, class_ids, sparse_masks)."
            )
        detections = sv.Detections(xyxy=input_boxes, confidence=confidences, class_id=class_ids)
        detections.data["sparse_masks"] = masks
        return detections

    def detect(self, image: np.ndarray, *, request: InferenceRequest) -> Detections2D:
        """Run 2D detection + segmentation on a single image.

        Parameters
        ----------
        image :
            Input image (H, W) float32 (single channel from spherical projection).
        request :
            Frozen job spec carrying text prompt + tuning-tagged settings.

        Returns
        -------
        Detections2D
            Per-detection boxes, sparse masks, class names/ids, confidences.
        """
        import gc

        import torch

        image_height, image_width = image.shape[:2]

        # Build the inference_models_parameters dict the shared helpers expect
        inference_models_parameters = {
            "bbox_model_id": None,  # already loaded; not used in post-proc
            "box_threshold": request.box_threshold,
            "text_threshold": request.text_threshold,
            "large_object_removal_threshold": request.large_object_removal_threshold,
            "partial_detection_edge_touching_threshold": request.partial_detection_edge_touching_threshold,
            "sam_box_prompt_batch_size": self._sam_box_prompt_batch_size,
            "device": self._device,
        }
        slice_inference_parameters = {
            "slice_width_height": request.slice_width_height,
            "overlap_width_height": request.overlap_width_height,
            "iou_threshold": request.iou_threshold,
            "overlap_filter_strategy": request.overlap_filter_strategy,
            "empty_slice_removal_threshold": 0.95,  # default
            "thread_workers": 1,
        }

        text_prompt = request.text_prompt

        if request.slicing_enabled:
            import supervision as sv

            from tls2dseg.engines.inference.sahi_slicer import SparseMasksInferenceSlicer

            callback_fn = partial(
                self._run_callback,
                text_prompt=text_prompt,
                inference_models_parameters=inference_models_parameters,
                slice_inference_parameters=slice_inference_parameters,
            )

            filter_strategy = sv.OverlapFilter.NON_MAX_SUPPRESSION
            slicer = SparseMasksInferenceSlicer(
                callback=callback_fn,
                slice_wh=request.slice_width_height,
                overlap_wh=request.overlap_width_height,
                overlap_ratio_wh=None,
                iou_threshold=request.iou_threshold,
                overlap_filter=filter_strategy,
                thread_workers=1,
            )
            detections = slicer(image)

            keys = text_prompt.split(".")
            class_id_map = {k: i + 1 for i, k in enumerate(keys)}
            inverted_map = {v: k for k, v in class_id_map.items()}

            class_names = [inverted_map[cid] for cid in detections.class_id]
            confidences = detections.confidence.tolist()
            class_ids = detections.class_id
            input_boxes = detections.xyxy
            masks = detections.data.get("sparse_masks", [])
        else:
            # No-SAHI: full-image inference
            image_encoded = img_1to3_channels_encoding(
                image, normalize="0-1", output_dtype="float32", replace_nan_with="max", broadcast=True
            )
            inputs = self._gdino_processor(
                images=image_encoded,
                text=text_prompt,
                return_tensors="pt",
                do_rescale=False,
            ).to(self._device)
            with torch.no_grad():
                outputs = self._gdino_model(**inputs)

            input_boxes, class_names, class_ids, confidences, _ = post_process_gdino_results(
                self._gdino_processor,
                outputs,
                inputs,
                text_prompt,
                inference_models_parameters,
                image_height,
                image_width,
            )
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()

            masks_batched: list = []
            with self._sam_lock:
                self._sam2_predictor.set_image(image_encoded)
                masks_batched = run_sam2_bbox_prompt_inference_in_batches(
                    self._sam2_predictor, input_boxes, self._sam_box_prompt_batch_size, masks_batched
                )
            masks = convert_masks_to_sparse_masks(masks_batched)
            torch.cuda.empty_cache()
            gc.collect()

            confidences = confidences.astype(float).tolist()

        conf_list = confidences if isinstance(confidences, list) else confidences.tolist()
        labels = [f"{name} {conf:.2f}" for name, conf in zip(class_names, conf_list, strict=False)]

        return Detections2D(
            masks=masks,
            input_boxes=input_boxes if isinstance(input_boxes, np.ndarray) else np.array(input_boxes),
            confidences=np.array(confidences, dtype=np.float32),
            class_names=list(class_names),
            class_ids=class_ids if isinstance(class_ids, np.ndarray) else np.array(class_ids, dtype=np.int32),
            mask_labels=labels,
        )
