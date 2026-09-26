"""napari front-end for ``tls2dseg label``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from tls2dseg.labeling import gt_io
from tls2dseg.labeling.session import DEFAULT_FEATURES, LabelingSession

logger = logging.getLogger("tls2dseg.labeling.app")

# GL textures are limited to 16384 px per side; larger mask images are edited
# through a movable full-resolution window instead of a downsampled texture.
FULL_MASK_MAX_PX = 16384
MASK_WINDOW_PX = 8192

HELP = (
    "Boxes layer: draw rectangles (R), delete (Del)\n"
    "Alt-1..9   set class (new + selected boxes)\n"
    "Shift-F    toggle intensity / range\n"
    "Shift-R    SAM2 on boxes without mask\n"
    "Alt-R      SAM2 on selected boxes (redo)\n"
    "Shift-B    boxes layer, rectangle tool\n"
    "Shift-M    masks layer, paint (E erase)\n"
    "Shift-W    move mask window to view\n"
    "Shift-S    save 2D ground truth\n"
    "Select one box to paint with its label.\n"
    "Optional: draw 'regions' = fully labeled\n"
    "areas; points outside become ignore (-2)."
)


def _signature(rows: np.ndarray, cols: np.ndarray, h: int) -> tuple[int, int]:
    lin = cols.astype(np.int64) * h + rows.astype(np.int64)
    return int(lin.size), int(lin.sum())


def _hex_to_rgba(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)] + [1.0])


class LabelingApp:
    """Box → SAM2 mask → 2D/3D ground-truth labeling on a spherical scan projection."""

    def __init__(
        self,
        viewer: Any,
        *,
        class_names: list[str],
        out_dir: Path,
        projection_params: dict,
        sam2_checkpoint: str | None,
        sam2_model_config: str,
        device: str,
        scan_path: Path | None = None,
        scan_index: int = 0,
    ) -> None:
        self.viewer = viewer
        self.class_names = list(class_names)
        self.out_dir = Path(out_dir)
        self.projection_params = dict(projection_params)
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_config = sam2_model_config
        self.device = device

        self.session: LabelingSession | None = None
        self.inst = np.zeros((0, 0), dtype=np.uint16)  # full-resolution instance image
        self.win = (0, 0, 0, 0)  # (r0, r1, c0, c1) of the editable mask window
        self.windowed = False
        self.mask_state: dict[int, tuple[str, str | None, tuple[int, int]]] = {}  # id → (source, feature, signature)
        self.next_id = 1
        self.feature = DEFAULT_FEATURES[0]
        self.current_class = self.class_names[0]
        self.predictor: Any = None
        self.busy = False
        self.dirty = False
        self._known_ids: set[int] = set()
        self._updating = False
        self.boxes: Any = None
        self.masks: Any = None
        self.regions: Any = None
        self.window_outline: Any = None

        self._build_dock(scan_path, scan_index)
        self._bind_keys()
        self._install_close_guard()
        if scan_path is not None:
            self.load()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_dock(self, scan_path: Path | None, scan_index: int) -> None:
        from magicgui import widgets as mw

        self.w_scan = mw.FileEdit(label="scan", mode="r", filter="*.e57")
        if scan_path is not None:
            self.w_scan.value = scan_path
        self.w_index = mw.SpinBox(label="scan index", min=0, max=9999, value=scan_index)
        w_load = mw.PushButton(text="Load scan")
        self.w_feature = mw.ComboBox(label="feature", choices=list(DEFAULT_FEATURES), value=self.feature)
        self.w_class = mw.ComboBox(label="class", choices=self.class_names, value=self.current_class)
        w_refine_new = mw.PushButton(text="SAM2: boxes without mask")
        w_refine_sel = mw.PushButton(text="SAM2: selected boxes")
        w_window = mw.PushButton(text="Move mask window to view")
        w_save = mw.PushButton(text="Save 2D")
        w_export = mw.PushButton(text="Save 2D + export 3D")
        self.w_status = mw.Label(value="")
        w_help = mw.Label(value=HELP)

        w_load.changed.connect(lambda: self.load())
        self.w_feature.changed.connect(lambda v: self.set_feature(v))
        self.w_class.changed.connect(lambda v: self.set_class(v))
        w_refine_new.changed.connect(lambda: self.refine(selected_only=False))
        w_refine_sel.changed.connect(lambda: self.refine(selected_only=True))
        w_window.changed.connect(lambda: self.move_window_to_view())
        w_save.changed.connect(lambda: self.save())
        w_export.changed.connect(lambda: self.export())

        container = mw.Container(
            widgets=[
                self.w_scan,
                self.w_index,
                w_load,
                self.w_feature,
                self.w_class,
                w_refine_new,
                w_refine_sel,
                w_window,
                w_save,
                w_export,
                self.w_status,
                w_help,
            ]
        )
        self.viewer.window.add_dock_widget(container, name="tls2dseg label", area="right")

    def _bind_keys(self) -> None:
        v = self.viewer

        def bind(key: str, fn: Any) -> None:
            v.bind_key(key, lambda _viewer: fn(), overwrite=True)

        for i, name in enumerate(self.class_names[:9], start=1):
            bind(f"Alt-{i}", lambda name=name: self.choose_class(name))
        bind("Shift-F", self.toggle_feature)
        bind("Shift-R", lambda: self.refine(selected_only=False))
        bind("Alt-R", lambda: self.refine(selected_only=True))
        bind("Shift-B", self.activate_boxes)
        bind("Shift-M", self.activate_masks)
        bind("Shift-W", self.move_window_to_view)
        bind("Shift-S", self.save)

    def _install_close_guard(self) -> None:
        from qtpy.QtCore import QEvent, QObject

        app = self

        class _CloseGuard(QObject):
            def eventFilter(self, obj: Any, event: Any) -> bool:
                if event.type() == QEvent.Close and app.dirty and app.session is not None:
                    app.save()
                return False

        try:
            self._close_guard = _CloseGuard()
            self.viewer.window._qt_window.installEventFilter(self._close_guard)
        except AttributeError:
            logger.warning("Could not install save-on-close hook; save with Shift-S before closing")

    @property
    def scan(self) -> LabelingSession:
        assert self.session is not None
        return self.session

    def status(self, msg: str) -> None:
        self.w_status.value = msg
        logger.info(msg)

    # ── Loading ──────────────────────────────────────────────────────────────

    def gt_dir(self) -> Path:
        assert self.session is not None
        return self.out_dir / self.scan.gt_name

    def load(self) -> None:
        from napari.qt.threading import create_worker

        from tls2dseg.labeling.session import e57_scan_count, load_scan

        if self.busy:
            return
        path = Path(self.w_scan.value)
        if not path.is_file():
            self.status(f"Not a file: {path}")
            return
        if self.session is not None and self.dirty:
            self.save()

        index = int(self.w_index.value)
        params = dict(self.projection_params)
        name = path.stem if e57_scan_count(path) == 1 else f"{path.stem}_s{index}"
        meta_path = self.out_dir / name / gt_io.META_JSON
        if meta_path.is_file():
            params = json.loads(meta_path.read_text(encoding="utf-8"))["projection_params"]
            logger.info("Existing ground truth found; re-projecting with its pinned parameters")

        self.busy = True
        self.status(f"Loading + projecting {path.name} …")
        worker = create_worker(load_scan, path, index, params, _start_thread=False)
        worker.returned.connect(self._on_loaded)
        worker.errored.connect(self._on_error)
        worker.start()

    def _on_error(self, exc: Exception) -> None:
        self.busy = False
        self.status(f"Error: {exc}")
        logger.error("Labeling tool error", exc_info=exc)

    def _on_loaded(self, session: LabelingSession) -> None:
        import pandas as pd

        from tls2dseg.labeling.session import build_multiscale, to_uint8

        self.busy = False
        self.session = session
        self.viewer.layers.clear()
        h, w = session.image_hw

        for feat, img in session.images.items():
            self.viewer.add_image(
                build_multiscale(to_uint8(img)),
                multiscale=True,
                name=feat,
                colormap="gray",
                contrast_limits=(0, 255),
                visible=feat == self.feature,
            )

        boxes: list[gt_io.BoxAnnotation] = []
        regions: list[tuple[float, float, float, float]] = []
        self.inst = np.zeros((h, w), dtype=np.uint16)
        self.mask_state = {}
        if (self.gt_dir() / gt_io.META_JSON).is_file():
            _, boxes, inst, regions = gt_io.load_2d(self.gt_dir())
            if inst.shape != (h, w):
                self.status(f"Saved labels are {inst.shape}, image is {(h, w)}: starting empty")
                boxes, regions = [], []
            else:
                self.inst = inst.astype(np.uint16)
                pixels = gt_io.instance_pixels(self.inst)
                for b in boxes:
                    if b.instance_id in pixels:
                        sig = _signature(*pixels[b.instance_id], h)
                        self.mask_state[b.instance_id] = (b.mask_source, b.mask_feature, sig)
        for b in boxes:
            if b.class_name not in self.class_names:
                self.class_names.append(b.class_name)
                self.w_class.choices = self.class_names

        self.windowed = max(h, w) > FULL_MASK_MAX_PX
        if self.windowed:
            self.win = (0, min(h, MASK_WINDOW_PX), 0, min(w, MASK_WINDOW_PX))
            self.masks = self.viewer.add_labels(self._window_view().copy(), name="masks", opacity=0.5)
            self.masks.translate = (self.win[0], self.win[2])
        else:
            self.win = (0, h, 0, w)
            self.masks = self.viewer.add_labels(self.inst, name="masks", opacity=0.5)
        self.masks.events.paint.connect(lambda _e: self._mark_dirty())

        self.regions = self.viewer.add_shapes(
            [[[y0, x0], [y1, x1]] for x0, y0, x1, y1 in regions] or None,
            shape_type="rectangle",
            name="regions",
            edge_color="yellow",
            face_color="transparent",
            edge_width=3,
        )
        self.regions.events.data.connect(lambda _e: self._mark_dirty())

        if self.windowed:
            self.window_outline = self.viewer.add_shapes(
                None, name="mask window", edge_color="cyan", face_color="transparent", edge_width=2
            )
            self._draw_window_outline()

        features = pd.DataFrame(
            {
                "class_name": pd.Series([b.class_name for b in boxes], dtype=object),
                "instance_id": pd.Series([b.instance_id for b in boxes], dtype=np.int64),
            }
        )
        self.boxes = self.viewer.add_shapes(
            [[[b.box_xyxy[1], b.box_xyxy[0]], [b.box_xyxy[3], b.box_xyxy[2]]] for b in boxes] or None,
            shape_type="rectangle",
            features=features,
            name="boxes",
            edge_width=2,
            face_color="transparent",
            text={"string": "{class_name}", "size": 9, "color": "white", "anchor": "upper_left"},
        )
        self.boxes.feature_defaults = {"class_name": self.current_class, "instance_id": 0}
        self.boxes.events.data.connect(self._on_boxes_changed)
        self.boxes.events.highlight.connect(self._on_box_selection)
        self._known_ids = {b.instance_id for b in boxes}
        self.next_id = max(self._known_ids, default=0) + 1
        self._recolor_boxes()
        self.dirty = False
        self.activate_boxes()
        self.status(
            f"{session.gt_name}: {w} x {h} px, {len(boxes)} boxes loaded"
            + (" (mask window: Shift-W to move)" if self.windowed else "")
        )

    # ── Layers / modes ───────────────────────────────────────────────────────

    def activate_boxes(self) -> None:
        if self.boxes is not None:
            self.viewer.layers.selection.active = self.boxes
            self.boxes.mode = "add_rectangle"

    def activate_masks(self) -> None:
        if self.masks is not None:
            self.viewer.layers.selection.active = self.masks
            self.masks.mode = "paint"

    def toggle_feature(self) -> None:
        feats = list(DEFAULT_FEATURES)
        self.w_feature.value = feats[(feats.index(self.feature) + 1) % len(feats)]

    def choose_class(self, name: str) -> None:
        self.w_class.value = name

    def set_feature(self, feature: str) -> None:
        self.feature = feature
        for feat in DEFAULT_FEATURES:
            if feat in self.viewer.layers:
                self.viewer.layers[feat].visible = feat == feature

    def set_class(self, name: str) -> None:
        self.current_class = name
        if self.boxes is None:
            return
        self.boxes.feature_defaults = {"class_name": name, "instance_id": 0}
        selected = sorted(self.boxes.selected_data)
        if selected:
            features = self.boxes.features.copy()
            features.loc[selected, "class_name"] = name
            self.boxes.features = features
            self._mark_dirty()
        self._recolor_boxes()

    def _class_color(self, name: str) -> np.ndarray:
        from tls2dseg.supervision_utils import CUSTOM_COLOR_MAP

        idx = self.class_names.index(name) if name in self.class_names else len(self.class_names)
        return _hex_to_rgba(CUSTOM_COLOR_MAP[idx % len(CUSTOM_COLOR_MAP)])

    def _recolor_boxes(self) -> None:
        self.boxes.current_edge_color = self._class_color(self.current_class)
        if len(self.boxes.data):
            self.boxes.edge_color = np.array([self._class_color(c) for c in self.boxes.features["class_name"]])
        self.boxes.refresh_text()

    # ── Box bookkeeping ──────────────────────────────────────────────────────

    def _mark_dirty(self) -> None:
        self.dirty = True

    def _on_boxes_changed(self, event: Any) -> None:
        if self._updating or str(getattr(event, "action", "")) not in ("added", "removed", "changed"):
            return
        self._updating = True
        try:
            features = self.boxes.features.copy()
            ids = features["instance_id"].to_numpy(dtype=np.int64, copy=True)
            seen: set[int] = set()
            for row, iid in enumerate(ids):
                if iid <= 0 or iid in seen:  # new or copy-pasted box
                    ids[row] = self.next_id
                    self.next_id += 1
                seen.add(int(ids[row]))
            if not np.array_equal(ids, features["instance_id"].to_numpy()):
                features["instance_id"] = ids
                self.boxes.features = features
            removed = self._known_ids - seen
            if removed:
                self._clear_instances(removed)
                for iid in removed:
                    self.mask_state.pop(iid, None)
            self._known_ids = seen
            self._recolor_boxes()
            self._mark_dirty()
        finally:
            self._updating = False

    def _on_box_selection(self, _event: Any) -> None:
        selected = list(self.boxes.selected_data)
        if len(selected) == 1:
            self.masks.selected_label = int(self.boxes.features["instance_id"].iloc[selected[0]])

    def _box_xyxy(self, row: int) -> np.ndarray:
        corners = np.asarray(self.boxes.data[row])[:, -2:]
        (y0, x0), (y1, x1) = corners.min(axis=0), corners.max(axis=0)
        return np.array([x0, y0, x1, y1], dtype=np.float64)

    # ── Full-resolution mask window ──────────────────────────────────────────

    def _window_view(self) -> np.ndarray:
        r0, r1, c0, c1 = self.win
        return self.inst[r0:r1, c0:c1]

    def _pull(self) -> None:
        """Copy edits made in the mask layer back into the full-resolution image."""
        if self.windowed and self.masks is not None:
            self._window_view()[...] = self.masks.data

    def _push(self) -> None:
        """Refresh the mask layer from the full-resolution image."""
        if self.windowed:
            self.masks.data = self._window_view().copy()
            self.masks.translate = (self.win[0], self.win[2])
        else:
            self.masks.refresh()

    def move_window_to_view(self) -> None:
        if not self.windowed or self.session is None:
            return
        self._pull()
        h, w = self.scan.image_hw
        cy, cx = self.viewer.camera.center[-2:]
        r0 = int(np.clip(cy - MASK_WINDOW_PX / 2, 0, max(h - MASK_WINDOW_PX, 0)))
        c0 = int(np.clip(cx - MASK_WINDOW_PX / 2, 0, max(w - MASK_WINDOW_PX, 0)))
        self.win = (r0, min(r0 + MASK_WINDOW_PX, h), c0, min(c0 + MASK_WINDOW_PX, w))
        self._push()
        self._draw_window_outline()

    def _draw_window_outline(self) -> None:
        r0, r1, c0, c1 = self.win
        layer = self.window_outline
        layer.selected_data = set(range(layer.nshapes))
        layer.remove_selected()
        layer.add_rectangles(
            [np.array([[r0, c0], [r1, c1]])], edge_color="cyan", edge_width=2, face_color="transparent"
        )

    def _clear_instances(self, ids: set[int]) -> None:
        self._pull()
        self.inst[np.isin(self.inst, np.fromiter(ids, dtype=np.int64))] = 0
        self._push()

    # ── SAM2 refinement ──────────────────────────────────────────────────────

    def refine(self, selected_only: bool) -> None:
        from napari.qt.threading import create_worker

        if self.busy or self.session is None:
            return
        if not self.sam2_checkpoint:
            self.status("No SAM2 checkpoint: pass --sam2-checkpoint or a --config with inference.sam2_checkpoint")
            return
        ids = self.boxes.features["instance_id"].to_numpy(dtype=np.int64)
        if selected_only:
            rows = sorted(self.boxes.selected_data)
        else:
            rows = [r for r, iid in enumerate(ids) if int(iid) not in self.mask_state]
        if not rows:
            self.status("No boxes to refine")
            return
        # Large boxes first so smaller (nested / foreground) masks end on top.
        jobs = sorted(((int(ids[r]), self._box_xyxy(r)) for r in rows), key=lambda j: -np.prod(j[1][2:] - j[1][:2]))
        image = self.scan.images[self.feature]

        self.busy = True
        worker = create_worker(self._refine_job, image, jobs, _start_thread=False)
        worker.yielded.connect(self.status)
        worker.returned.connect(lambda res: self._on_refined(res, jobs))
        worker.errored.connect(self._on_error)
        worker.start()

    def _refine_job(self, image: np.ndarray, jobs: list[tuple[int, np.ndarray]]) -> Any:
        from tls2dseg.labeling.sam2_refine import load_sam2_predictor, refine_boxes

        if self.predictor is None:
            yield "Loading SAM2 …"
            self.predictor = load_sam2_predictor(str(self.sam2_checkpoint), self.sam2_model_config, self.device)
        masks = []
        for i, (_, box) in enumerate(jobs, start=1):
            masks.extend(refine_boxes(self.predictor, image, box[None, :], self.device))
            yield f"SAM2 {i}/{len(jobs)}"
        return masks

    def _on_refined(self, masks: list[np.ndarray], jobs: list[tuple[int, np.ndarray]]) -> None:
        self.busy = False
        h = self.scan.image_hw[0]
        live = set(self.boxes.features["instance_id"].astype(int))
        self._pull()
        redo = {iid for iid, _ in jobs if iid in self.mask_state}
        if redo:
            self.inst[np.isin(self.inst, np.fromiter(redo, dtype=np.int64))] = 0
        empty = 0
        for (iid, _), sparse in zip(jobs, masks, strict=True):
            if iid not in live:
                continue
            if len(sparse) == 0:
                empty += 1
                continue
            self.inst[sparse[:, 0], sparse[:, 1]] = iid
        # Signatures after painting, since later (smaller) masks may cover earlier ones.
        for iid, box in jobs:
            if iid in live:
                self.mask_state[iid] = ("sam2", self.feature, _signature(*self._pixels_in_crop(iid, box), h))
        self._push()
        self._mark_dirty()
        self.status(f"SAM2 done: {len(jobs)} boxes" + (f", {empty} empty masks" if empty else ""))

    def _pixels_in_crop(self, iid: int, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixels of ``iid`` within the SAM2 crop of ``box`` (a fresh SAM2 mask never extends beyond it)."""
        from tls2dseg.labeling.sam2_refine import crop_window

        r0, r1, c0, c1 = crop_window(box, self.scan.image_hw)
        rows, cols = np.nonzero(self.inst[r0:r1, c0:c1] == iid)
        return rows + r0, cols + c0

    # ── Save / export ────────────────────────────────────────────────────────

    def _collect(self) -> tuple[list[gt_io.BoxAnnotation], list[tuple[float, float, float, float]]]:
        self._pull()
        h = self.scan.image_hw[0]
        pixels = gt_io.instance_pixels(self.inst)
        boxes = []
        for row, (cls, iid) in enumerate(
            zip(self.boxes.features["class_name"], self.boxes.features["instance_id"].astype(int), strict=True)
        ):
            source, feature = "none", None
            if iid in pixels:
                sig = _signature(*pixels[iid], h)
                prev = self.mask_state.get(iid)
                if prev is None:
                    source = "manual"
                else:
                    source, feature = prev[0], prev[1]
                    if prev[2] != sig:
                        source = "sam2_edited" if source in ("sam2", "sam2_edited") else "manual"
                self.mask_state[iid] = (source, feature, sig)
            x0, y0, x1, y1 = (float(v) for v in self._box_xyxy(row))
            boxes.append(gt_io.BoxAnnotation(iid, str(cls), (x0, y0, x1, y1), source, feature))

        regions = []
        for corners in self.regions.data:
            (y0, x0), (y1, x1) = np.asarray(corners)[:, -2:].min(axis=0), np.asarray(corners)[:, -2:].max(axis=0)
            regions.append((float(x0), float(y0), float(x1), float(y1)))
        return boxes, regions

    def _meta(self) -> dict:
        from tls2dseg._version import __version__

        s = self.scan
        stat = s.scan_path.stat()
        return {
            "tool": "tls2dseg label",
            "tls2dseg_version": __version__,
            "scan_path": str(s.scan_path.resolve()),
            "scan_index": s.scan_index,
            "scan_count": s.scan_count,
            "scan_file_size": stat.st_size,
            "scan_file_mtime": stat.st_mtime,
            "projection_params": s.projection_params,
            "d_azim_rad": s.d_azim_rad,
            "sam2_checkpoint": str(self.sam2_checkpoint) if self.sam2_checkpoint else None,
            "sam2_model_config": self.sam2_model_config,
        }

    def save(self) -> None:
        if self.session is None:
            return
        boxes, regions = self._collect()
        gt_io.save_2d(self.gt_dir(), self._meta(), self.class_names, boxes, self.inst, regions)
        self.dirty = False
        self.status(f"Saved {len(boxes)} boxes to {self.gt_dir()}")

    def export(self) -> None:
        from napari.qt.threading import create_worker

        if self.busy or self.session is None:
            return
        self.save()
        boxes, regions = self._collect()
        self.busy = True
        self.status("Exporting 3D ground truth …")
        worker = create_worker(
            gt_io.export_3d,
            self.gt_dir(),
            self.session,
            boxes,
            self.inst.copy(),
            list(self.class_names),
            regions,
            _start_thread=False,
        )
        worker.returned.connect(lambda path: self._on_exported(path))
        worker.errored.connect(self._on_error)
        worker.start()

    def _on_exported(self, path: Path) -> None:
        self.busy = False
        self.status(f"Exported {path}")


def run_app(
    *,
    class_names: list[str],
    out_dir: Path,
    projection_params: dict,
    sam2_checkpoint: str | None,
    sam2_model_config: str,
    device: str,
    scan_path: Path | None = None,
    scan_index: int = 0,
) -> None:
    """Open the napari labeling window and block until it is closed."""
    import napari

    viewer = napari.Viewer(title="tls2dseg label")
    LabelingApp(
        viewer,
        class_names=class_names,
        out_dir=out_dir,
        projection_params=projection_params,
        sam2_checkpoint=sam2_checkpoint,
        sam2_model_config=sam2_model_config,
        device=device,
        scan_path=scan_path,
        scan_index=scan_index,
    )
    napari.run()
