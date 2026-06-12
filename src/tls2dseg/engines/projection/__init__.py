"""Public surface for the ``tls2dseg.engines.projection`` sub-package.

Phase 4 plan 02 Task 2 (ENG-03). Re-exports ``SphericalProjectionEngine``
as the one concrete ``ProjectionEngine`` implementation for Phase 4.

Closest analog: ``tls2dseg.engines.__init__`` (registry pattern).
"""

from __future__ import annotations

import logging

from tls2dseg.engines.projection.spherical import SphericalProjectionEngine

logger = logging.getLogger("tls2dseg.engines.projection")

__all__ = [
    "SphericalProjectionEngine",
]
