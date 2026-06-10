"""Two-phase logging setup + misc env-var bootstrap for the CLI entry point.

Phase 3 plan 06 (LOG-02 + LOG-03 + LOG-04). Three public surfaces:

* :func:`bootstrap_logger` — Phase 1 of two-phase logging. ``basicConfig`` so
  any DEBUG/INFO emitted by ``config.loader`` during YAML parse + env-var
  interpolation (BEFORE ``configure_logging`` runs) is captured. Called from
  ``cli.run_cmd`` FIRST.
* :func:`configure_logging` — Phase 2 of two-phase logging. ``dictConfig`` with
  formatters, console + optional file handlers, per-package overrides. Called
  AFTER ``load_config()`` succeeds with the resolved ``cfg.logging`` values.
* :func:`bootstrap` — Misc env-var bootstrap (``TOKENIZERS_PARALLELISM`` —
  moved here from pipeline/run.py:18 so it lives at the CLI entry, not buried
  inside the legacy pipeline module).

Why two phases?
---------------
``logging.basicConfig`` is convenient for early-bootstrap (no config object
needed) but limited (single handler, no per-package overrides). ``dictConfig``
gives full structured setup but requires knowing ``cfg.logging``, which means
running AFTER ``load_config``. The loader emits its own DEBUG/INFO records
during ``${ENV_VAR}`` expansion and YAML parsing — these would be silently
lost if we waited for ``dictConfig``.

dictConfig's handler-replacement semantics are intentional here: when
:func:`configure_logging` runs, it cleanly replaces basicConfig's handlers
with the full structured setup (RESEARCH §Pattern 8 — "dictConfig replaces
root logger handlers as a unit"). No records emitted in Phase 1 are dropped;
they're flushed to the basicConfig handler before being replaced.

This design is locked by
:func:`test_bootstrap_logger_captures_records_before_dictconfig` in
``test_logging_setup.py`` — a DEBUG record emitted between the two calls is
asserted present in caplog.
"""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from collections.abc import Mapping
from typing import Any


def bootstrap_logger(level: str = "INFO") -> None:
    """Phase 1 — pre-load ``basicConfig`` so config-loader output is captured.

    Fires BEFORE :func:`load_config`. Sets the root logger to ``level``, adds
    a single StreamHandler to stderr, uses ``force=True`` to clear any
    pre-existing root handlers (test isolation).

    The late :func:`configure_logging` (Phase 2) cleanly replaces these
    handlers via dictConfig's "root logger handlers as a unit" semantic
    (RESEARCH §Pattern 8). No records emitted between the two calls are lost.

    Parameters
    ----------
    level
        Log level string (case-insensitive). One of ``"DEBUG"``, ``"INFO"``,
        ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``. Unknown levels fall back
        to INFO.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        stream=sys.stderr,
        format="[%(levelname)s|%(name)s] %(message)s",
        force=True,
    )


def configure_logging(
    level: str = "INFO",
    per_package: Mapping[str, str] | None = None,
    log_file: str | None = None,
) -> None:
    """Phase 2 — post-load ``dictConfig`` with structured handlers + per-package overrides.

    Replaces :func:`bootstrap_logger`'s basicConfig handlers cleanly per
    RESEARCH §Pattern 8 (dictConfig replaces root logger handlers as a unit).
    Called AFTER :func:`load_config` succeeds with the resolved
    ``cfg.logging`` values.

    Parameters
    ----------
    level
        Default level for the ``tls2dseg.*`` logger hierarchy (LOG-04 wiring).
    per_package
        Optional dict ``{package_name: level}`` of per-package logger overrides
        (LOG-03). Empty dict / None → no overrides (conservative D-A1-10
        default; pchandler/pc2img inherit root). Replaces the hardcoded
        ``logging.getLogger('pchandler').setLevel(ERROR)`` from
        pipeline/run.py:21.
    log_file
        Optional path for a FileHandler attached to the ``tls2dseg`` logger.
        ``cli.run_cmd`` sets this from ``ctx.logs_dir / 'run.log'`` when
        ``cfg.logging.log_to_file`` is True. None → console-only.

    Notes
    -----
    ``disable_existing_loggers: False`` is critical — pchandler/pc2img have
    their own ``logging.getLogger(__name__)`` calls executed at import time.
    Disabling them here would silence those loggers entirely; we want to
    REASSIGN their levels via ``per_package``, not kill them.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    level_name = logging.getLevelName(numeric_level)

    handlers_cfg: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "detailed",
            "level": "DEBUG",
        }
    }
    handler_names: list[str] = ["console"]

    if log_file is not None:
        handlers_cfg["file"] = {
            "class": "logging.FileHandler",
            "filename": log_file,
            "formatter": "detailed",
            "level": "DEBUG",
            "mode": "a",
            "encoding": "utf-8",
        }
        handler_names.append("file")

    loggers_cfg: dict[str, Any] = {
        "tls2dseg": {
            "level": level_name,
            "handlers": handler_names,
            "propagate": False,
        }
    }

    if per_package:
        for pkg, pkg_level in per_package.items():
            loggers_cfg[pkg] = {
                "level": pkg_level.upper(),
                "handlers": handler_names,
                "propagate": False,
            }

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "[%(levelname)s|%(name)s|L%(lineno)d] %(asctime)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
        },
        "handlers": handlers_cfg,
        "loggers": loggers_cfg,
        "root": {
            "level": "WARNING",
            "handlers": handler_names,
        },
    }

    logging.config.dictConfig(config)


def bootstrap() -> None:
    """Misc env-var bootstrap. Called once at CLI entry, before any pipeline import.

    Currently sets ``TOKENIZERS_PARALLELISM=true`` (moved from
    pipeline/run.py:18 per CONTEXT.md §"Integration Points"). Uses
    ``setdefault`` so a user-set value is preserved.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
