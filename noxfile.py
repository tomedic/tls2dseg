"""nox session definitions for tls2dseg local testing automation.

Test sessions:
  - tier_a: lightweight, no-GPU, no heavy dependencies (e.g. pchandler/pc2img).
             Locally mirrors Github Actions (.github/workflows/ci.yml).
  - tier_b_light: lightweight, but requires pchandler + pc2img, does not require GPU.
  - tier_b_heavy: moderate, heavy dependancies (PyTorch) + GPU use.

Local usage:
  - ``nox -s tier_a``
  - ``nox -s tier_b_light``
  - ``nox -s tier_b_heavy``

This file is also the binding point for the Claude Code PreToolUse hook
(``.claude/settings.json`` at the workspace root): the hook runs all three
test sessions before any ``git push`` to gate the push on a green local test suite.

Reuse-venv: ``nox.options.reuse_venv = "yes"``.
Equivalent to ``--reuse-existing-venvs`` on the CLI. Critical for
pre-push hook latency — without it, every push rebuilds the venv from scratch.
"""

import os
import tomllib
from pathlib import Path

import nox

nox.options.reuse_venv = "yes"


_EDITABLE_SIBLINGS = {"pchandler", "pc2img"}


def _requirement_dist_name(req: str) -> str:
    """Leading distribution name of a requirement string, lowercased.

    Robust to ``name @ url``, ``name[extra] ~= 1.0``, ``name>=1.0`` forms:
    takes the token up to the first whitespace / ``@`` / ``[`` / version
    operator character.
    """
    name = req.strip().lower()
    for i, ch in enumerate(name):
        if ch in " \t@[<>=!~":
            return name[:i]
    return name


def _runtime_deps_excluding_editable_siblings() -> list[str]:
    """[project.dependencies] minus pchandler/pc2img.

    tier_b_light installs pchandler/pc2img as editable siblings, so they are
    excluded from the pip-resolved runtime deps to avoid double-installing them
    from git. Real URL deps (sam2) and all PyPI deps are kept.
    """
    data = tomllib.loads((Path(__file__).parent / "pyproject.toml").read_text())
    return [d for d in data["project"]["dependencies"] if _requirement_dist_name(d) not in _EDITABLE_SIBLINGS]


# pytest flags MUST match the tier_a job in ci.yml exactly.
# Update both atomically if flags change.
@nox.session(python="3.11", name="tier_a")
def tier_a(session: nox.Session) -> None:
    """Run tier_a tests (no-GPU, no-pchandler/pc2img, < 5 min)."""
    session.install("-e", ".", "--no-deps")
    session.install(
        "pydantic-settings[yaml]>=2.14",
        "typer>=0.13",
        "pytest",
        "numpy",
        "scipy",
    )
    session.run("pytest", "-m", "tier_a", "-v", *session.posargs)


# pytest flags MUST match any future tier_b_light cloud-CI job in ci.yml
# exactly. Today tier_b_light is local-only
@nox.session(python="3.11", name="tier_b_light")
def tier_b_light(session: nox.Session) -> None:
    """Run tier_b_light tests — pchandler+pc2img with deps; torch allowed; no real model runs.

    pchandler/pc2img are installed as editable siblings (the local-dev workflow)
    rather than resolved from pyproject's git URLs. tls2dseg installs ``--no-deps``
    so pip never re-resolves the siblings from git; its remaining runtime deps
    (PyPI + real sam2 URL) are installed explicitly so ``imageio`` etc. resolve
    and test collection works.
    """
    session.install("-e", "../PCHandler")  # editable sibling, WITH deps
    session.install("-e", "../pc2img")  # editable sibling, WITH deps
    session.install("-e", ".", "--no-deps")  # tls2dseg only; siblings stay editable
    session.install(*_runtime_deps_excluding_editable_siblings())
    session.install("pytest")  # explicit in case not in transitive deps
    # manifold3d: CPU-only trimesh boolean backend so the OBB-boolean IoU
    # tests (TEST-06, test_bboxes_iou.py) execute instead of skip-guarding on
    # an absent mesh-boolean engine. Pure pip dep, no GPU.
    session.install("manifold3d")
    session.run("pytest", "-m", "tier_b_light", "-v", *session.posargs)


# tier_b_heavy: real models / GPU / cudf / large fixtures. This session uses NO isolated
# venv and delegates to a project conda env that already has torch+CUDA+cudf+sam2 plus the
# editable pchandler/pc2img installs (development env). `nox` need NOT be installed in
# that conda env — nox stays the orchestrator in the base env and shells into the conda env via
# `conda run`. Override the env name with TLS2DSEG_HEAVY_CONDA_ENV.
@nox.session(venv_backend="none", name="tier_b_heavy")
def tier_b_heavy(session: nox.Session) -> None:
    """Run tier_b_heavy tests inside the project conda env (default tls2dseg_2025)."""
    env_name = os.environ.get("TLS2DSEG_HEAVY_CONDA_ENV", "tls2dseg_2025")
    session.run(
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        "-m",
        "pytest",
        "-m",
        "tier_b_heavy",
        "-v",
        *session.posargs,
        external=True,
    )
