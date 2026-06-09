"""Runtime capability probes — frozen Runtime dataclass + probe_all() factory.

Phase 3 CPU-05: a one-shot, never-raising snapshot of runtime module availability
(cuml, torch+CUDA, sam2, libvips) and dep versions (numpy, torch, cuml). The
``Runtime`` instance is stamped at process start by the CLI (plan 06) and threaded
through ``RunContext`` (plan 04) to every engine (Phase 4 ENG-01..07) so per-call
re-probing (``import torch.cuda`` at every site) is unnecessary.

Probes never import the heavy modules at module-load time. Availability uses
``importlib.util.find_spec`` (does NOT execute target module code). The single
exception is the ``torch.cuda.is_available()`` check, which requires importing
torch — that call is wrapped in a broad ``except Exception`` per the
``diagnostics.py`` "best-effort per D-11" pattern; any failure resolves to
``torch_cuda_available=False``.

Version lookups use ``importlib.metadata.version`` and resolve to ``None`` on
``PackageNotFoundError`` so callers can write provenance ``env.txt`` without
guarding every read site.

Closest analog: ``tls2dseg.diagnostics`` (Phase 1 D-09) — identical
``find_spec`` + try/import/broad-catch shape, but here the return is a
``@dataclasses.dataclass(frozen=True)`` per RESEARCH.md Pitfall 4 (mutation
raises ``dataclasses.FrozenInstanceError``, NOT ``pydantic.ValidationError`` —
``Runtime`` is a dataclass, not a pydantic model).
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import logging

logger = logging.getLogger("tls2dseg.runtime.capability")


@dataclasses.dataclass(frozen=True)
class Runtime:
    """Frozen snapshot of runtime module availability and dep versions.

    Fields per CONTEXT.md D-A2-03 + REQUIREMENTS.md CPU-05. All availability
    fields are ``bool``; all version fields are ``str | None`` (None for
    missing packages).

    Mutation attempts raise ``dataclasses.FrozenInstanceError`` per
    RESEARCH.md Pitfall 4. Construct exclusively via :func:`probe_all`.
    """

    cuml_available: bool
    torch_cuda_available: bool
    sam2_available: bool
    libvips_available: bool
    numpy_version: str | None
    torch_version: str | None
    cuml_version: str | None


def _probe(module_name: str) -> bool:
    """Return True iff ``module_name`` is importable, WITHOUT executing it.

    Uses ``importlib.util.find_spec`` per RESEARCH.md §Pattern 3 (verified).
    Does NOT raise on missing modules — returns False instead.
    """
    return importlib.util.find_spec(module_name) is not None


def _safe_version(dist_name: str) -> str | None:
    """Return installed-package version string, or None if not installed.

    Wraps ``importlib.metadata.version`` per RESEARCH.md §Pattern 3.
    Never raises ``PackageNotFoundError`` — returns None instead.
    """
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_all() -> Runtime:
    """Probe runtime once and return a populated frozen :class:`Runtime`.

    Availability checks use ``find_spec`` (no module code executed). The
    sole exception is ``torch.cuda.is_available()`` which requires importing
    torch; failure (CUDA-less build, broken install) resolves to False per
    the ``diagnostics.py`` best-effort pattern.

    Cheap to call (one-shot at process start); cache the result on
    ``RunContext.capability`` rather than re-probing per engine.
    """
    torch_cuda = False
    if _probe("torch"):
        try:
            import torch

            torch_cuda = bool(torch.cuda.is_available())
        except Exception:  # best-effort per D-11; broad catch is intentional
            pass

    return Runtime(
        cuml_available=_probe("cuml"),
        torch_cuda_available=torch_cuda,
        sam2_available=_probe("sam2"),
        libvips_available=_probe("pyvips"),
        numpy_version=_safe_version("numpy"),
        torch_version=_safe_version("torch"),
        cuml_version=_safe_version("cuml"),
    )
