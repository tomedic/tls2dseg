"""Tier-A CFG-03 grep gate — no hardcoded ``/scratch/...`` or wheat-heads-
specific paths at module scope in :mod:`tls2dseg.config` (or
:mod:`tls2dseg.runtime` once that sub-package lands in plan 03-02/04).

The pre-Phase-3 codebase had hardcoded paths embedded in
``pipeline/run.py`` module-level dicts (``./data/wheat_heads/``,
``./results/results_wheat_only_2``, ``/scratch/projects/sam2/checkpoints/...``).
CFG-03 moves them to YAML; this test enforces the negative ("none of those
paths leak back into the new config/runtime sub-packages").

The check excludes comment lines so design notes referencing the legacy
paths in docstrings or comments do not trip the gate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import tls2dseg

# Resolve the tls2dseg/src/tls2dseg root once at module load.
_PKG_ROOT = Path(tls2dseg.__file__).resolve().parent
_CONFIG_DIR = _PKG_ROOT / "config"
_RUNTIME_DIR = _PKG_ROOT / "runtime"


def _scan_dir_for_pattern(directory: Path, pattern: str) -> list[str]:
    """Run ``grep -rnE pattern`` over ``directory`` for ``*.py``, strip
    comment-only lines, return the remaining matches as a list of
    ``file:line:content`` strings.

    Returns an empty list when no matches remain after the comment strip.
    Skips silently if the directory does not exist (e.g. ``runtime/`` may
    not yet exist when this plan runs; plan 03-02/04 creates it).
    """
    if not directory.is_dir():
        return []

    # `grep` exit codes: 0=match, 1=no match, 2=error. Don't raise on 1.
    proc = subprocess.run(
        [
            "grep",
            "-rnE",
            "--include=*.py",
            pattern,
            str(directory),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 1:
        return []
    if proc.returncode != 0:
        pytest.fail(f"grep failed (rc={proc.returncode}): {proc.stderr!r}")

    # Strip lines whose content (after the file:line: prefix) is a pure
    # comment. Format is ``path:lineno:content`` so we drop two colons.
    matches: list[str] = []
    for line in proc.stdout.splitlines():
        # Split on the first two colons only; content may contain colons.
        parts = line.split(":", 2)
        if len(parts) < 3:
            matches.append(line)
            continue
        content = parts[2].lstrip()
        if content.startswith("#"):
            continue
        matches.append(line)
    return matches


@pytest.mark.tier_a
def test_no_scratch_paths_in_config_sub_package() -> None:
    """No literal ``/scratch/`` references survive in ``config/`` modules
    (CFG-03). Comments referencing the legacy path are allowed.
    """
    hits = _scan_dir_for_pattern(_CONFIG_DIR, r"/scratch/")
    assert not hits, "CFG-03 gate failure — hardcoded /scratch/ path(s) found in tls2dseg.config:\n" + "\n".join(hits)


@pytest.mark.tier_a
def test_no_wheat_heads_paths_in_config_sub_package() -> None:
    """No literal ``wheat_heads`` references survive in ``config/`` modules
    (CFG-03). Wheat-heads dataset-specific defaults belong in
    ``examples/configs/`` YAMLs, not the schema.
    """
    hits = _scan_dir_for_pattern(_CONFIG_DIR, r"wheat_heads")
    assert not hits, "CFG-03 gate failure — hardcoded wheat_heads path(s) found in tls2dseg.config:\n" + "\n".join(hits)


@pytest.mark.tier_a
def test_no_scratch_or_wheat_heads_in_runtime_sub_package() -> None:
    """Same gate applied to ``runtime/`` once it exists (plan 03-02/04
    creates it). Skips silently if the directory is not yet present.
    """
    hits = _scan_dir_for_pattern(_RUNTIME_DIR, r"/scratch/|wheat_heads")
    assert not hits, (
        "CFG-03 gate failure — hardcoded /scratch/ or wheat_heads path(s) found in tls2dseg.runtime:\n"
        + "\n".join(hits)
    )
