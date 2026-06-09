"""Runtime sub-package public surface.

Phase 3 plan 02 (CPU-05/06) exports ``Runtime`` and ``probe_all`` only.
``RunContext`` and ``build_context`` will be appended by Phase 3 plan 04
(D-A2-01) when ``runtime/context.py`` lands. Until then, importers should
use only the symbols listed in ``__all__`` below.
"""

from __future__ import annotations

# RunContext, build_context added in plan 04 (D-A2-01)
from tls2dseg.runtime.capability import Runtime, probe_all

__all__ = ["Runtime", "probe_all"]
