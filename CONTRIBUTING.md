# Contributing to tls2dseg

## Engine architecture overview

The pipeline is built around three `Protocol`s (defined in
`src/tls2dseg/engines/protocols.py`). Each Protocol defines a single-method
contract that the orchestrator (`pipeline/stage1.py`, `pipeline/stage2.py`)
calls without knowing which concrete engine is behind it.

| Protocol | Single method | Role |
|----------|---------------|------|
| `ProjectionEngine` | `project(pcd, *, features, resolution, ...)` | Rasterizes a point cloud to a list of per-feature panoramic images |
| `InferenceEngine` | `detect(image, *, request)` | Runs Grounded-DINO + SAM2 (or equivalent) on one image; returns `Detections2D` |
| `FusionEngine` | `fuse(fusion_input)` | Runs cross-scan graph clustering on a complete `FusionInput`; returns `FusionResult` |

Heavy imports (torch, sam2, pyvips, igraph) belong **inside the engine's
`__init__`**, not at module level. This keeps `import tls2dseg.engines` fast
and tier_a tests GPU-free (D-A-05).

## Dict registries

`src/tls2dseg/engines/__init__.py` declares three plain dicts:

```python
PROJECTION_ENGINES: dict[str, type] = {"spherical": SphericalProjectionEngine}
INFERENCE_ENGINES:  dict[str, type] = {
    "grounded_sam2":    GroundedSAM2Engine,
    "grounded_sam2_hf": GroundedSAM2HFEngine,
}
FUSION_ENGINES:     dict[str, type] = {"graph_cluster": GraphClusterFusionEngine}
```

The config field `type` (e.g. `inference.type: grounded_sam2`) is the registry
key the pipeline uses at runtime. Companion builder functions
(`build_projection_engine`, `build_inference_engine`, `build_fusion_engine`)
construct the engine from the registry and raise a clear `ValueError` if the
name is not found.

## Adding a new engine

1. **Implement the Protocol.** Create a module under the relevant sub-package
   (e.g. `src/tls2dseg/engines/inference/my_engine.py`). The class must
   satisfy the Protocol's method signature; use `@runtime_checkable` to verify
   with `isinstance(engine, InferenceEngine)` in tests.

2. **Register it.** In `src/tls2dseg/engines/__init__.py`, add an entry to the
   appropriate dict inside the populator function (lazy import pattern):

   ```python
   def _populate_inference_registry() -> None:
       from tls2dseg.engines.inference.my_engine import MyEngine
       INFERENCE_ENGINES["my_engine"] = MyEngine
   ```

3. **Expose its config.** If the engine needs config fields, add a new
   `BaseModel` subclass to `src/tls2dseg/config/models.py` and wire it into
   the discriminated union for `InferenceConfig` (or the equivalent block for
   projection/fusion engines). Tag every new field with
   `json_schema_extra={"tag": "..."}`.

4. **Select it at runtime.** Set `inference.type: my_engine` (or
   `projection.type` / `fusion.type`) in the YAML config.

## FakeEngine convention (tier_a tests)

`tier_a` tests must not import torch, sam2, pchandler, or pyvips. Fake engines
satisfy the Protocol while staying lightweight:

```python
class FakeInferenceEngine:
    """Tier-A stand-in for InferenceEngine — returns empty Detections2D."""

    def detect(self, image, *, request):
        from tls2dseg.types import Detections2D
        return Detections2D.empty()
```

Place `FakeEngine` classes in `tests/fakes/` (or inline in the test module for
simple cases). Use `monkeypatch` or dependency injection to substitute them for
the real engine in tests marked `@pytest.mark.tier_a`.

## SAM2 default-engine note

The default `inference.type: grounded_sam2` (direct `sam2` package + local
checkpoint) is documented in README §Architecture and Extending. If you are
adding an alternative inference engine, follow the same Protocol, register it
with a unique key, and add a FakeEngine in the test suite. See the
`GroundedSAM2Engine` and `GroundedSAM2HFEngine` implementations for reference.
