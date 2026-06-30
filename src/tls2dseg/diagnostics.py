"""Environment diagnostics. All probes are best-effort and never raise."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("tls2dseg.diagnostics")


def doctor() -> dict[str, Any]:
    """Probe runtime environment. Returns structured report; never raises.

    Returns
    -------
    dict[str, Any]
        Keys: python, torch_cuda, rapids_cuml, libvips, pchandler, pc2img, sam2_checkpoint.
        Each value is a dict with status + detail + (where applicable) hint.
    """
    return {
        "python": _probe_python(),
        "torch_cuda": _probe_torch_cuda(),
        "rapids_cuml": _probe_rapids(),
        "libvips": _probe_libvips(),
        "pchandler": _probe_pchandler(),
        "pc2img": _probe_pc2img(),
        "sam2_checkpoint": _probe_sam2_checkpoint(),
    }


def _probe_python() -> dict[str, Any]:
    return {
        "status": "ok",
        "detail": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def _probe_torch_cuda() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {
            "status": "missing",
            "detail": "torch not installed",
            "hint": "pip install torch — see README for install instructions",
        }
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "status": "ok",
                "detail": (
                    f"CUDA available; device count={torch.cuda.device_count()}, name={torch.cuda.get_device_name(0)}"
                ),
            }
        return {
            "status": "cpu-only",
            "detail": "torch installed; CUDA unavailable",
            "hint": "Pipeline runs on CPU but slower; check NVIDIA driver and CUDA install.",
        }
    except Exception as e:  # best-effort per D-11; broad catch is intentional
        return {"status": "error", "detail": f"torch import failed: {e}"}


def _probe_rapids() -> dict[str, Any]:
    if importlib.util.find_spec("cuml") is None:
        return {
            "status": "missing",
            "detail": "cuml not installed",
            "hint": "Optional. CPU fallback (sklearn) lands in Phase 3.",
        }
    try:
        import cuml

        return {"status": "ok", "detail": f"cuml {cuml.__version__}"}
    except Exception as e:  # best-effort per D-11; broad catch is intentional
        return {"status": "error", "detail": f"cuml import failed: {e}"}


def _probe_libvips() -> dict[str, Any]:
    if importlib.util.find_spec("pyvips") is None:
        return {
            "status": "missing",
            "detail": "pyvips not installed",
            "hint": "pip install pyvips AND apt install libvips",
        }
    try:
        import pyvips

        return {
            "status": "ok",
            "detail": (f"libvips {pyvips.base.version(0)}.{pyvips.base.version(1)}.{pyvips.base.version(2)}"),
        }
    except Exception as e:  # best-effort per D-11; broad catch is intentional
        return {"status": "error", "detail": f"pyvips import failed: {e}"}


def _probe_pchandler() -> dict[str, Any]:
    if importlib.util.find_spec("pchandler") is None:
        return {
            "status": "missing",
            "detail": "pchandler not installed",
            "hint": "pip install pchandler — see README for install instructions",
        }
    try:
        import pchandler

        return {"status": "ok", "detail": f"pchandler {getattr(pchandler, '__version__', 'unknown')}"}
    except Exception as e:  # best-effort per D-11; broad catch is intentional
        return {"status": "error", "detail": f"pchandler import failed: {e}"}


def _probe_pc2img() -> dict[str, Any]:
    if importlib.util.find_spec("pc2img") is None:
        return {
            "status": "missing",
            "detail": "pc2img not installed",
            "hint": "pip install pc2img — see README for install instructions",
        }
    try:
        import pc2img

        return {"status": "ok", "detail": f"pc2img {getattr(pc2img, '__version__', 'unknown')}"}
    except Exception as e:  # best-effort per D-11; broad catch is intentional
        return {"status": "error", "detail": f"pc2img import failed: {e}"}


# Search-path strategy for SAM2 checkpoint (D-09, D-11):
#   1. $SAM2_CHECKPOINT_PATH if set and exists  (explicit user opt-in)
#   2. ./checkpoints/sam2*.pt  (project-local convention)
#   3. ~/.cache/tls2dseg/checkpoints/sam2*.pt  (XDG-style cache; standard for ML cks)
#   4. /scratch/projects/sam2/checkpoints/sam2*.pt  (backward compat with current main.py:139)
def _probe_sam2_checkpoint() -> dict[str, Any]:
    search_paths: list[Path] = []
    env_path = os.environ.get("SAM2_CHECKPOINT_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return {"status": "ok", "detail": f"found via $SAM2_CHECKPOINT_PATH: {p}"}
        search_paths.append(p)

    candidate_dirs: list[Path] = [
        Path.cwd() / "checkpoints",
        Path.home() / ".cache" / "tls2dseg" / "checkpoints",
        Path("/scratch/projects/sam2/checkpoints"),
    ]
    for d in candidate_dirs:
        if d.is_dir():
            matches = sorted(d.glob("sam2*.pt"))
            if matches:
                return {"status": "ok", "detail": f"found: {matches[0]}"}
            search_paths.append(d)
        else:
            search_paths.append(d)

    return {
        "status": "missing",
        "detail": "no SAM2 *.pt checkpoint found",
        "hint": (
            "Set $SAM2_CHECKPOINT_PATH to your checkpoint file, OR place one of "
            "(sam2.1_hiera_large.pt, sam2.1_hiera_base.pt, ...) into one of: " + ", ".join(str(p) for p in search_paths)
        ),
    }


def format_report(report: dict[str, Any]) -> str:
    """Human-readable single-table report for the CLI."""
    lines = ["tls2dseg doctor", "=" * 40]
    for key, val in report.items():
        status = val["status"]
        marker = {
            "ok": "[OK]    ",
            "cpu-only": "[WARN]  ",
            "missing": "[MISS]  ",
            "error": "[ERR]   ",
        }.get(status, "[?]     ")
        lines.append(f"{marker} {key:20s} {val['detail']}")
        if "hint" in val:
            lines.append(f"          hint: {val['hint']}")
    return "\n".join(lines)
