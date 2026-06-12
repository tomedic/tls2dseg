"""GroundedSAM2HFEngine — HF transformers SAM2 inference engine.

Phase 4 plan 04-04 Task 4 (ENG-03, ENG-05, ENG-06, D-C-01).

This engine implements InferenceEngine using the HuggingFace transformers
SAM2 API (Sam2Processor / Sam2Model). Heavy deps (torch, transformers) are
confined to __init__ method bodies (D-A-05) so that importing this module
never triggers model loading.

Task 1 probe result (transformers 5.11.0, proceed-as-assumed):
  - Class names: Sam2Processor, Sam2Model  (transformers >= 4.46)
  - Model id: "facebook/sam2.1-hiera-large"
  - API delta vs direct sam2: post_process_masks(outputs.pred_masks, inputs["original_sizes"])
    takes a SINGLE original_sizes argument (not two-arg form assumed in RESEARCH).

The DINO + SAHI + post-processing half is SHARED with GroundedSAM2Engine;
the SAM2 half (Sam2Processor -> Sam2Model -> post_process_masks) is the seam
specific to this engine (D-C-01).
"""

from __future__ import annotations

import logging
from functools import partial

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
from tls2dseg.types import Detections2D, InferenceRequest

logger = logging.getLogger("tls2dseg.engines.inference.grounded_sam2_hf")


class GroundedSAM2HFEngine:
    """HF transformers SAM2 inference engine.

    Conforms to the ``InferenceEngine`` Protocol (engines/protocols.py).

    Parameters
    ----------
    object_detection_model_id : str
        HF model id for Grounded-DINO.
    sam2_hf_model_id : str
        HF model id for SAM2 (default: ``"facebook/sam2.1-hiera-large"``).
    sam_box_prompt_batch_size : int
        Number of bounding boxes per SAM2 forward pass.
    device : str
        Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        *,
        object_detection_model_id: str,
        sam2_hf_model_id: str = "facebook/sam2.1-hiera-large",
        sam_box_prompt_batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        # Heavy imports confined here (D-A-05)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

        self._sam_box_prompt_batch_size = sam_box_prompt_batch_size
        self._device = device

        # Grounded-DINO
        logger.info("Loading Grounded-DINO model: %s", object_detection_model_id)
        self._gdino_processor: AutoProcessor = AutoProcessor.from_pretrained(object_detection_model_id)
        self._gdino_model: AutoModelForZeroShotObjectDetection = AutoModelForZeroShotObjectDetection.from_pretrained(
            object_detection_model_id
        ).to(device)

        # SAM2 (HF)
        logger.info("Loading SAM2 HF model: %s", sam2_hf_model_id)
        self._sam2_processor: Sam2Processor = Sam2Processor.from_pretrained(sam2_hf_model_id)
        self._sam2_model: Sam2Model = Sam2Model.from_pretrained(sam2_hf_model_id).to(device)

    def _run_sam2_hf_batched(
        self,
        image: np.ndarray,
        input_boxes: np.ndarray,
    ) -> list:
        """Run SAM2 HF box-prompt inference in batches.

        Parameters
        ----------
        image :
            Encoded 3-channel (H, W, 3) float32 image [0, 1].
        input_boxes :
            (N, 4) float32 bounding boxes in xyxy format.

        Returns
        -------
        list
            Sparse masks as list of (M, 2) int32 coordinate arrays.
        """
        import torch

        all_masks: list = []
        batch_size = self._sam_box_prompt_batch_size

        for batch_i in range(0, len(input_boxes), batch_size):
            batch_boxes = input_boxes[batch_i : batch_i + batch_size]

            # Prepare SAM2 HF inputs
            # Sam2Processor expects PIL or numpy image + bounding boxes
            # boxes must be a list of lists (unnormalised pixel coordinates)
            inputs = self._sam2_processor(
                images=image,
                input_boxes=[batch_boxes.tolist()],
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                # multimask_output=False makes SAM2 return its single best mask,
                # matching the direct engine's predict(multimask_output=False) seam.
                # Sam2Model.forward defaults multimask_output=True (3 ranked candidates);
                # taking candidate [0] is NOT equivalent to multimask_output=False (it is
                # the first, not the highest-IoU mask) and yields lower-quality masks.
                outputs = self._sam2_model(**inputs, multimask_output=False)

            # post_process_masks(masks, original_sizes) — SINGLE original_sizes arg
            # (Task 1 probe result: transformers 5.11.0 API)
            masks_batch = self._sam2_processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
            )

            # masks_batch: list of tensors, each (N_prompts, 1, H, W) — the
            # candidate axis is length 1 because multimask_output=False above makes
            # SAM2 emit exactly one (best) mask per box.
            for mask_tensor in masks_batch:
                # shape: (n_boxes, 1, H, W) — squeeze the single-candidate axis
                if mask_tensor.ndim == 4:
                    mask_tensor = mask_tensor[:, 0, :, :]  # (n_boxes, H, W)
                mask_np = mask_tensor.cpu().numpy().astype(bool)
                all_masks.append(mask_np)

        return convert_masks_to_sparse_masks(all_masks)

    def _run_callback(
        self,
        image_slice: np.ndarray,
        *,
        text_prompt: str,
        inference_models_parameters: dict,
        slice_inference_parameters: dict,
    ):
        """Run DINO + SAM2-HF inference on a single image or image slice."""
        import gc

        import torch

        slice_height, slice_width = image_slice.shape[:2]

        # Early-exit checks
        if not check_if_img_slice_empty(image_slice, slice_inference_parameters):
            return return_empty_detections()
        if not check_if_img_slice_complete(image_slice, slice_inference_parameters):
            return return_empty_detections()

        # Encode image
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

        # SAM2 (HF)
        masks = self._run_sam2_hf_batched(image_slice, input_boxes)

        torch.cuda.empty_cache()
        gc.collect()

        import supervision as sv

        if not (len(masks) == input_boxes.shape[0] == confidences.shape[0] == class_ids.shape[0]):
            raise ValueError(
                "Grounded SAM2 HF inference produced mismatched output lengths "
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

        inference_models_parameters = {
            "bbox_model_id": None,
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
            "empty_slice_removal_threshold": 0.95,
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
            # Full-image inference (no SAHI)
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

            masks = self._run_sam2_hf_batched(image_encoded, input_boxes)
            torch.cuda.empty_cache()
            gc.collect()

            confidences = confidences.astype(float).tolist()

        labels = [
            f"{name} {conf:.2f}"
            for name, conf in zip(
                class_names,
                confidences if isinstance(confidences, list) else confidences.tolist(),
                strict=False,
            )
        ]

        return Detections2D(
            masks=masks,
            input_boxes=input_boxes if isinstance(input_boxes, np.ndarray) else np.array(input_boxes),
            confidences=np.array(confidences, dtype=np.float32),
            class_names=list(class_names),
            class_ids=class_ids if isinstance(class_ids, np.ndarray) else np.array(class_ids, dtype=np.int32),
            mask_labels=labels,
        )
