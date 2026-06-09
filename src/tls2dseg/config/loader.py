"""YAML loader for :class:`tls2dseg.config.RunConfig`.

The single public entry point :func:`load_config` reads a YAML file, expands
``${ENV_VAR}`` references against the OS environment, then constructs a
:class:`RunConfig` with the locked source precedence::

    cli_overrides  >  TLS2DSEG_* env vars  >  YAML file  >  schema defaults

(D-A3-04). Internally this wires pydantic-settings'
``settings_customise_sources`` with a custom :class:`ExpandingYamlSource`
subclass of :class:`YamlConfigSettingsSource` that runs
``os.path.expandvars`` on the raw YAML text *before*
:func:`yaml.safe_load` parses it — so a YAML containing
``sam2_checkpoint: ${SAM2_CHECKPOINT_PATH}`` resolves to the env-var value,
and any residual ``${...}`` after expansion raises a clear ``ValueError``
naming the unresolved variable (Pitfall 6 in RESEARCH.md).

Path validators on the :class:`RunConfig` sub-models are LAZY (D-CD-05) —
``validate-config`` accepts a well-formed-but-nonexistent path; the file
existence check fires later, at runtime, when the file is actually opened.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import YamlConfigSettingsSource
from pydantic_settings.sources import PydanticBaseSettingsSource

from tls2dseg.config.models import RunConfig

logger = logging.getLogger("tls2dseg.config.loader")

# Matches any ``${...}`` reference in the YAML text. Used both to detect
# unresolved env-var references after :func:`os.path.expandvars` and to
# avoid false-positive matches on things like ``$$`` literals.
_ENV_VAR_PATTERN = re.compile(r"\$\{[^}]+\}")


class ExpandingYamlSource(YamlConfigSettingsSource):
    """YamlConfigSettingsSource that expands ``${ENV_VAR}`` before parsing.

    Overrides :meth:`YamlConfigSettingsSource._read_file` to:

    1. Read the YAML file as raw text.
    2. Run :func:`os.path.expandvars` to substitute ``${VAR}`` references
       against the current process environment.
    3. Detect any residual ``${...}`` after expansion (unset vars stay
       literal — :func:`os.path.expandvars` does NOT raise on missing
       vars, per Pitfall 6) and raise :class:`ValueError` listing every
       unresolved reference.
    4. Hand the expanded text to :func:`yaml.safe_load` (the parent class's
       :meth:`_read_file` does the same — we replicate without re-opening
       the file).

    The :func:`yaml.safe_load` call rejects YAML tags that would
    arbitrary-construct Python objects (T-03-yaml-load mitigation).
    """

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        raw_text = file_path.read_text(encoding=self.yaml_file_encoding or "utf-8")
        expanded = os.path.expandvars(raw_text)

        # Detect unresolved ${VAR} references (Pitfall 6). os.path.expandvars
        # leaves unset references untouched rather than raising; we want a
        # clear, user-facing error here, not a downstream ValidationError on
        # a Path field that received the literal ``${VAR}`` string.
        unresolved = sorted(set(_ENV_VAR_PATTERN.findall(expanded)))
        if unresolved:
            raise ValueError(
                f"Config file {file_path} contains unresolved environment "
                f"variables: {unresolved}. Set these env vars before running."
            )

        parsed = yaml.safe_load(expanded)
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Config file {file_path} must parse to a YAML mapping (dict) at top level. "
                f"Got: {type(parsed).__name__}"
            )
        return parsed


def load_config(
    yaml_path: Path | str,
    cli_overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Load a :class:`RunConfig` from YAML + env vars + CLI overrides.

    Source precedence (highest first; locked in D-A3-04):

        1. ``cli_overrides`` keyword args (CLI flags, programmatic overrides)
        2. ``TLS2DSEG_*`` environment variables (nested via ``__`` delimiter)
        3. Values from the YAML file at ``yaml_path``
        4. Schema defaults declared on :class:`RunConfig` sub-models

    Parameters
    ----------
    yaml_path:
        Path to the YAML config file. Always passed explicitly to the
        underlying YAML source (Pitfall 7 — the default of ``.`` causes
        silent read failures).
    cli_overrides:
        Optional dict of overrides to inject at the highest-precedence
        level. Nested keys map to nested sub-models, e.g.
        ``{"inference": {"box_threshold": 0.25}}``.

    Returns
    -------
    RunConfig
        Fully-validated, frozen RunConfig instance.

    Raises
    ------
    FileNotFoundError
        ``yaml_path`` does not exist.
    ValueError
        YAML contains unresolved ``${ENV_VAR}`` references, or top-level
        YAML is not a mapping.
    pydantic.ValidationError
        Any required field missing; any field type-invalid; any unknown
        key (``extra='forbid'``); any value-validator failure.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    overrides = dict(cli_overrides) if cli_overrides else {}

    # Build a per-call RunConfig subclass whose customise-sources wires in
    # the path-bound ExpandingYamlSource. Done as a closure so the yaml_path
    # never leaks into module-level state (thread-safety + test-isolation
    # property).
    class _BoundRunConfig(RunConfig):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type,
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            # Tuple order = source precedence (highest priority first).
            # init_settings (the kwargs passed to __init__, i.e. cli_overrides)
            # > env_settings (TLS2DSEG_*) > YAML > dotenv > secrets.
            yaml_source = ExpandingYamlSource(settings_cls, yaml_file=yaml_path)
            return (
                init_settings,
                env_settings,
                yaml_source,
                dotenv_settings,
                file_secret_settings,
            )

    logger.debug("Loading RunConfig from %s with cli_overrides=%r", yaml_path, overrides)
    cfg = _BoundRunConfig(**overrides)
    # Cast back to RunConfig for the public return type — the _BoundRunConfig
    # subclass is purely a wiring vehicle.
    return cfg
