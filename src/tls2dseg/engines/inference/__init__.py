"""engines.inference — shared helpers + concrete inference engine classes.

Phase 4 plan 04-04 Task 3/4 (ENG-03, ENG-07, D-C-01).

Sub-modules:
- shared.py     — DINO/SAHI/post-proc helpers shared by both engines
- sahi_slicer.py — SparseMasksInferenceSlicer (verbatim move from sparse_masks_inference_slicer.py)
- grounded_sam2.py    — GroundedSAM2Engine (direct sam2 package)
- grounded_sam2_hf.py — GroundedSAM2HFEngine (HF transformers)

Heavy deps (torch, sam2, transformers) are confined to method bodies;
importing this package never triggers model loading (D-A-05).
"""

from __future__ import annotations
