"""Shared type aggregates for tls2dseg.

This module is a Phase 2 stub. Phase 4 (ENG-01..07) populates it with the
engine-Protocol-facing shared types — concretely planned content:

- ``Detections2D`` aggregate (per-image detections with class + bbox + mask + confidence)
- ``Detections3D`` aggregate (per-instance 3D point cluster + bbox + class + confidence)
- ``ProjectionResult`` (per-scan 2D-image + per-pixel-to-3D mapping)
- Optional: ``RunContext`` fragment if not owned by ``runtime/`` subpackage (Phase 3 CFG-04 decides)

Until Phase 4 lands, this module intentionally exports nothing — its presence anchors
the import surface so downstream modules can ``from tls2dseg.types import ...`` without
the import path changing later.
"""

from __future__ import annotations

__all__: list[str] = []
