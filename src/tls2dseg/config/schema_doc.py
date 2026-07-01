"""Schema-documentation generator (DOC-02).

Walks RunConfig + all sub-models listed in ``_SUB_MODELS`` via
``model_fields``, reads the ``json_schema_extra["tag"]`` annotation on every
field, groups by tag (primary → tuning → pipings → other), and renders
Markdown reference docs.

Engine config variants (GroundedSAM2Config, GroundedSAM2HFConfig) appear
because they are listed explicitly in ``_SUB_MODELS`` — the discriminated
InferenceConfig union type is not traversed at runtime.
"""

from __future__ import annotations

from pydantic import BaseModel

from tls2dseg.config.models import (
    D3DExtractionConfig,
    FusionConfig,
    GroundedSAM2Config,
    GroundedSAM2HFConfig,
    InferenceSharedConfig,
    IOConfig,
    LoggingConfig,
    MultiZoomConfig,
    PreprocessingConfig,
    ProjectionConfig,
    PromptConfig,
    RunConfig,
    RuntimeConfig,
    SlicingConfig,
)

__all__ = ["generate_schema_doc"]

TAG_ORDER = ["primary", "tuning", "pipings", "other"]

# Explicit traversal order for top-level RunConfig sub-blocks.
_SUB_MODELS = [
    RunConfig,
    IOConfig,
    RuntimeConfig,
    PromptConfig,
    PreprocessingConfig,
    ProjectionConfig,
    SlicingConfig,
    MultiZoomConfig,
    InferenceSharedConfig,
    GroundedSAM2Config,
    GroundedSAM2HFConfig,
    D3DExtractionConfig,
    FusionConfig,
    LoggingConfig,
]


def _is_model(tp: object) -> bool:
    """Return True if tp is a pydantic BaseModel subclass."""
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _type_name(annotation: object) -> str:
    """Return a compact human-readable name for a field annotation."""
    if annotation is None:
        return "Any"
    if hasattr(annotation, "__name__"):
        return str(annotation.__name__)
    return str(annotation).replace("typing.", "").replace("tls2dseg.config.models.", "")


def _default_str(finfo: object) -> str:
    """Return the field default as a string, or '' if required."""
    import pydantic.fields as _pf
    from pydantic_core import PydanticUndefined

    if not isinstance(finfo, _pf.FieldInfo):
        return ""
    if finfo.is_required():
        return "*(required)*"
    if finfo.default is PydanticUndefined:
        if finfo.default_factory is not None:
            return repr(finfo.default_factory())
        return ""
    default = finfo.default
    if default is None:
        return "null"
    if isinstance(default, bool):
        return str(default).lower()
    return repr(default)


def _collect_fields(
    model: type[BaseModel],
    source_label: str,
    seen: set[tuple[str, str]],
) -> list[tuple[str, str, str, str, str, str]]:
    """Return list of (tag, field_name, type_str, default_str, description, source).

    Emits only fields declared directly on *model* (not inherited); inherited
    fields are documented when _SUB_MODELS reaches the base class directly.
    Skips fields already seen (source_label, field_name).
    """
    rows: list[tuple[str, str, str, str, str, str]] = []
    own_annotations = vars(model).get("__annotations__", {})
    for fname, finfo in model.model_fields.items():
        if fname not in own_annotations:
            continue
        key = (source_label, fname)
        if key in seen:
            continue
        seen.add(key)

        extra = finfo.json_schema_extra
        tag = extra.get("tag", "other") if isinstance(extra, dict) else "other"

        annotation = finfo.annotation

        # Nested BaseModel — emit a row for this container field; the sub-model's
        # own fields are rendered when _SUB_MODELS reaches that model directly.
        if _is_model(annotation):
            rows.append(
                (
                    tag,
                    fname,
                    _type_name(annotation),
                    _default_str(finfo),
                    finfo.description or "",
                    source_label,
                )
            )
            continue

        rows.append(
            (
                tag,
                fname,
                _type_name(annotation),
                _default_str(finfo),
                finfo.description or "",
                source_label,
            )
        )

    return rows


def generate_schema_doc() -> str:
    """Return a Markdown config reference grouped by tag (DOC-02).

    Sections are ordered: primary → tuning → pipings → other.
    The discriminated InferenceConfig union is unpacked so both
    GroundedSAM2Config and GroundedSAM2HFConfig fields appear.
    Output begins with a title header and a dataset permanent-link
    placeholder per D-19.
    """
    all_rows: list[tuple[str, str, str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for model in _SUB_MODELS:
        label = model.__name__
        all_rows.extend(_collect_fields(model, label, seen))

    # Group by tag.
    by_tag: dict[str, list[tuple[str, str, str, str, str, str]]] = {t: [] for t in TAG_ORDER}
    for row in all_rows:
        tag = row[0]
        if tag not in by_tag:
            by_tag.setdefault(tag, [])
        by_tag[tag].append(row)

    lines: list[str] = [
        "# tls2dseg Config Schema Reference",
        "",
        "Auto-generated by `tls2dseg schema` from the pydantic models.",
        "Fields are grouped by relevance tag: **primary** (user-facing), **tuning**",
        "(result quality), **pipings** (compute/infra), **other** (rarely changed).",
        "",
        "Reference datasets: <PLACEHOLDER_DATASET_URL>",
        "",
    ]

    for tag in TAG_ORDER:
        rows = by_tag.get(tag, [])
        if not rows:
            continue
        lines.append(f"## {tag}")
        lines.append("")
        lines.append("| Field | Type | Default | Description | Model |")
        lines.append("|-------|------|---------|-------------|-------|")
        for _tag, fname, type_str, default, desc, source in rows:
            # Escape pipe characters in values to avoid breaking the table.
            desc_safe = desc.replace("|", "\\|")
            type_safe = type_str.replace("|", "\\|")
            lines.append(f"| `{fname}` | `{type_safe}` | {default} | {desc_safe} | {source} |")
        lines.append("")

    return "\n".join(lines)
