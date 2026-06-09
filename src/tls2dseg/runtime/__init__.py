"""Runtime sub-package public surface.

Phase 3 plan 02 (CPU-05/06) exported ``Runtime`` and ``probe_all``.
Phase 3 plan 04 (CFG-04 + CPU-04) appends ``RunContext``, ``build_context``,
and ``resolve_device`` per D-A2-01. The full surface is now stable for
Phase 4+ consumers.
"""

from __future__ import annotations

from tls2dseg.runtime.capability import Runtime, probe_all
from tls2dseg.runtime.context import RunContext, build_context, resolve_device

__all__ = [
    "RunContext",
    "Runtime",
    "build_context",
    "probe_all",
    "resolve_device",
]
