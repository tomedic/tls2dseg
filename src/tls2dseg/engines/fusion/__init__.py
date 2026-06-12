"""Fusion sub-package — graph-clustering helpers + GraphClusterFusionEngine.

Phase 4 plan 03 Task 1 (ENG-07). Provides the fusion engine and helpers
under ``tls2dseg.engines.fusion.*``.

Sub-modules:
- ``bboxes_iou``: AABB and OBB IoU helpers (moved verbatim per D-D-03)
- ``connectivity``: sparse graph connectivity (KD-tree KNN/radius)
- ``edge_weights``: edge weight computation (IoU + supporter counts)
- ``clustering``: graph clustering algorithms (PCC/HCS/Leiden + outlier detection)
- ``graph``: GraphClusterFusionEngine (owns full stage-2 pipeline per D-A-06)

NOTE: This ``__init__.py`` is intentionally EMPTY (no eager imports).
Importing ``tls2dseg.engines.fusion`` for the package namespace is
tier_a safe. Sub-module symbols must be imported from their specific
modules (e.g. ``from tls2dseg.engines.fusion.connectivity import ...``)
to avoid triggering scipy/networkx at package-load time (D-A-05).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("tls2dseg.engines.fusion")

__all__: list[str] = []
