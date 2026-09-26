# Ground-truth labeling (`tls2dseg label`)

An interactive [napari](https://napari.org) tool for creating ground-truth labels for a few scans. It
projects a scan the same way the pipeline does, lets you draw class-labelled boxes, refines each box
into a pixel mask with SAM2, and lets you correct masks with a brush. It saves the result as 2D
(COCO) and 3D (per-point PLY) ground truth.

## Install and start

```bash
pip install "tls2dseg[label]"      # adds napari + PyQt5

tls2dseg label --scan scans/Scan_1.e57 --classes "plant,cabinet" --out gt/ \
               --sam2-checkpoint /path/to/sam2.1_hiera_large.pt
# or take projection settings, SAM2 checkpoint and classes from a pipeline config:
tls2dseg label --config examples/configs/office_small.yaml --scan examples/data/office_small/Scan_1.e57 --out gt/
```

You can also pick the scan (and the scan index, for multi-scan `.e57` files) in the tool's side
panel. Loading and projecting a large scan can take a few minutes.

## Workflow

1. **Load.** The scan is projected at full resolution (no subsampling or ROI filter). Both the
   `intensity` and `range` images are loaded, and `Shift-F` switches between them. Contrast and gamma
   are in napari's layer controls.
2. **Boxes.** Draw rectangles on the `boxes` layer (`Shift-B`, then drag). `Alt-1..9` or the
   *class* dropdown sets the class for new boxes and for the selected boxes. Box colour follows class.
3. **SAM2.** `Shift-R` runs SAM2 on every box that has no mask yet, and `Alt-R` re-runs it on the
   selected boxes. SAM2 runs on a crop around each box, so small objects are segmented at high
   detail. The mask is computed from the currently shown feature image.
4. **Fix masks.** Select one box, which makes its instance the active paint label. Then press
   `Shift-M` to switch to the `masks` layer and use the napari paint (`P`), erase (`E`) and fill
   (`F`) tools. Deleting a box (`Del`) also deletes its mask.
5. **Regions (optional).** Rectangles on the `regions` layer mark the areas you labelled
   *exhaustively*. Points outside every region are exported as *ignore* (`-2`). Without regions, the
   whole scan counts as exhaustively labelled.
6. **Save.** `Shift-S` saves the 2D ground truth. It is also saved automatically when you close the
   window or load another scan. *Save 2D + export 3D* additionally writes the labelled point cloud.

Re-opening the same scan with the same `--out` resumes the saved labels. The saved projection
parameters (image width, rotation and angular resolution) are reused, so the image is pixel-identical
even if the original settings were randomised `auto` estimates.

**Large panoramas.** Above 16384 px (the GPU texture limit), the `masks` layer is an 8192 px
full-resolution *mask window*, drawn with a cyan outline. `Shift-W` moves it to the current view.
SAM2 refinement and saving always use the full image.

## Output (`<out>/<scan name>/`)

| File | Content |
|---|---|
| `gt_instances.png` | 16-bit instance-id image (0 = none), the pixel source of truth |
| `gt_coco.json` | COCO: `bbox` = the drawn box, `segmentation` = compressed RLE of the final mask, `attributes` = `instance_id`, `mask_source` (`none`/`sam2`/`sam2_edited`/`manual`), `mask_feature` |
| `gt_meta.json` | scan path/index/size/mtime, pinned projection parameters, classes and ids, boxes, regions, SAM2 checkpoint |
| `<scan name>_gt.ply` | full-resolution cloud in the project frame (PRCS), fields `scalar_classes`, `scalar_instances`, `scalar_intensity` |
| `class_names_id_map.txt` | class id → name, same format as the pipeline output |

3D label values: `>= 1` class / instance id, `0` background, `-1` outside the projection's field of
view, `-2` ignore (outside the labelled regions). Class ids are 1-based in the order of `--classes`.

## Tips and limits

- Points on depth edges (mixed pixels) can inherit the label of the object in front. For precise 3D
  ground truth, clean these up in [CloudCompare](https://cloudcompare.org) (free), e.g. by segmenting
  and editing `scalar_classes` / `scalar_instances`.
- Ground truth is per scan. In multi-view data, instance ids are not linked across scans.
- The 2D ground truth is tied to the saved projection. Evaluate 2D detections only against images
  projected with the parameters in `gt_meta.json`; the 3D PLY does not depend on them.
