#!/usr/bin/env python3
"""
Option A) Direct NN-within-radius transfer (multi-class semantics + per-point instances)
             + optional smoothing + instance-purity cleanup

- Semantics and instances may come from DIFFERENT source PLYs.
- Label '0' means background for both classes and instances.
- Target gets scalar_fields["classes"] and ["instances"] filled.

DROP-IN TO DOs:
  - Replace the pchandler loaders/savers in `load_clouds()` and `save_cloud()`.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from pchandler.data_io import load_ply, save_ply

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG — set your paths + parameters here
# ────────────────────────────────────────────────────────────────────────────────

SRC_SEM_PLY_PATH = Path("data/label_transfer/source_semantic_instance.ply")   # classes live here
SRC_INS_PLY_PATH = Path("data/label_transfer/source_semantic_instance.ply")   # instances live here
TGT_PLY_PATH = Path("data/label_transfer/target.ply")

OUT_PLY_PATH = Path("../data/label_transfer/target_labels_transferred.ply")

# Geometry / radius
RESOLUTION = 0.003  # meters (point cloud resolution)
SCALE_TOLERANCE = 1.2  # scale neighborhood size r = SCALE_TOLERANCE * RESOLUTION * sqrt(3)
# TODO: figure out what is this for
EPS = 1e-6

# Optional post-processing of transferred labels - smoothing on the target (weighted majority in an r_smooth ball)
SMOOTHING_ENABLED = True
WEIGHT_EXP_P = 2.0     # weights = 1/(d+eps)^p
SMOOTH_RADIUS_MULT = 1.0     # r_smooth = r * SMOOTH_RADIUS_MULT
SMOOTH_CONF_TAU = 0.60    # ambiguity threshold; if max/total < tau → flip to 0
SMOOTH_MIN_NEIGH = 3       # require at least this many neighbors to attempt smoothing

# Final per-instance purity cleanup
PURITY_THRESHOLD = 0.80    # if instance majority semantic < this → zero-out instance & class

# ────────────────────────────────────────────────────────────────────────────────
# Core routines
# ────────────────────────────────────────────────────────────────────────────────


def weighted_majority(values: np.ndarray, weights: np.ndarray) -> tuple[int, float, float]:
    """
    Returns (winner_label, winner_weight, total_weight).
    values: (M,) int labels; weights: (M,) float >= 0
    """
    if values.size == 0:
        return 0, 0.0, 0.0
    uniq, inv = np.unique(values, return_inverse=True)
    wsum = np.bincount(inv, weights=weights.astype(np.float64), minlength=uniq.size)
    j = int(np.argmax(wsum))
    return int(uniq[j]), float(wsum[j]), float(wsum.sum())


def transfer_nn_within_radius_single(
    src_xyz: np.ndarray,
    src_lab: np.ndarray,
    tgt_xyz: np.ndarray,
    r: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generic 1-NN within radius. Returns (tgt_labels, nn_dist).
    If nearest distance > r, label = 0.
    """
    tree = cKDTree(src_xyz)
    dist, idx = tree.query(tgt_xyz, k=1, workers=-1)
    ok = dist <= r
    tgt_lab = np.zeros(tgt_xyz.shape[0], dtype=src_lab.dtype)
    tgt_lab[ok] = src_lab[idx[ok]]
    return tgt_lab, dist


def smoothing_weighted_majority(
    xyz: np.ndarray,
    sem: np.ndarray,
    ins: np.ndarray,
    r_smooth: float,
    p: float = 2.0,
    t_purity: float = 0.6,
    min_neigh: int = 3,
    gate_mode: str = "none",      # "none" | "soft" | "hard"
    w_scaledown: float = 0.2,       # for "soft": down-scale factor on mismatched semantics
    allow_instance_if_sem0: bool = True,  # allow instance smoothing even if semantics = 0 (background)
    eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-point smoothing on TARGET cloud.

    Semantics:
      - distance-weighted majority within r_smooth; if winner share < tau_sem -> set to 0.

    Instances:
      - If sem_s[i] > 0:
          * gate_mode="hard": vote only among neighbors whose ORIGINAL semantic == sem_s[i]
          * gate_mode="soft": down-weight neighbors with semantic != sem_s[i] by (1 - gate_beta)
          * gate_mode="none": ignore semantics
        If sem_s[i] == 0:
          * allow_instance_if_sem0=False -> set instance to 0 (hard dependency)
          * allow_instance_if_sem0=True  -> proceed with instance vote with **no semantic gating**,
                                           regardless of gate_mode.
      - Ambiguity: if winner share < tau_inst or margin < margin_m -> set instance to 0.
    """

    tree = cKDTree(xyz)
    N = xyz.shape[0]
    sem_s = sem.copy()
    ins_s = ins.copy()

    neighbors = tree.query_ball_point(xyz, r_smooth, workers=-1)

    for i in range(N):
        ids = neighbors[i]
        if len(ids) < min_neigh:
            continue

        d = np.linalg.norm(xyz[ids] - xyz[i], axis=1)
        w = 1.0 / (d + eps) ** p

        # --- Semantic smoothing
        sem_win, w_win, w_tot = weighted_majority(sem[ids], w)
        if w_tot > 0 and (w_win / w_tot) >= t_purity:
            sem_s[i] = sem_win
        else:
            sem_s[i] = 0

        # --- Instance smoothing
        if sem_s[i] == 0 and not allow_instance_if_sem0:
            ins_s[i] = 0
            continue

        # Decide gating behavior
        ids_inst = np.array(ids)
        w_inst = w.copy()
        if sem_s[i] > 0:
            if gate_mode == "hard":
                mask = (sem[ids_inst] == sem_s[i])
                ids_inst = ids_inst[mask]
                w_inst = w_inst[mask]
            elif gate_mode == "soft":
                mism = (sem[ids_inst] != sem_s[i])
                w_inst[mism] *= w_scaledown
            elif gate_mode == "none":
                pass
            else:
                raise ValueError(f"Unknown gate_mode: {gate_mode}")
        else:
            # sem_s[i] == 0 and allow_instance_if_sem0=True → no gating
            pass

        if ids_inst.size < min_neigh:
            ins_s[i] = 0
            continue

        inst_vals = ins[ids_inst]
        inst_win, wi_win, wi_tot = weighted_majority(inst_vals, w_inst)
        if wi_tot > 0 and (wi_win / wi_tot) >= t_purity:
            ins_s[i] = inst_win
        else:
            ins_s[i] = 0

    return sem_s, ins_s


def cleanup_instance_purity(
    sem: np.ndarray,
    ins: np.ndarray,
    purity: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enforce one semantic per instance:
      - For each instance id k>0: compute semantic histogram; if majority fraction >= purity,
        set all points in instance to that semantic; else flip both sem and ins to 0.
    """
    sem_c = sem.copy()
    ins_c = ins.copy()

    inst_ids = np.unique(ins_c)
    inst_ids = inst_ids[inst_ids > 0]
    for k in inst_ids:
        idx = np.where(ins_c == k)[0]
        if idx.size == 0:
            continue
        vals = sem_c[idx]
        uniq, counts = np.unique(vals, return_counts=True)

        # Exclude 0 when deciding majority; if all zero, maj=0
        pos_mask = (uniq > 0)
        if pos_mask.any():
            uniq_pos = uniq[pos_mask]
            counts_pos = counts[pos_mask]
            j = int(np.argmax(counts_pos))
            maj = int(uniq_pos[j])
            frac = counts_pos[j] / idx.size
        else:
            maj = 0
            frac = 1.0

        if frac >= purity and maj > 0:
            sem_c[idx] = maj
        else:
            sem_c[idx] = 0
            ins_c[idx] = 0

    return sem_c, ins_c

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # Radius from resolution
    r = float(SCALE_TOLERANCE) * float(RESOLUTION) * np.sqrt(3.0)
    r_smooth = r * float(SMOOTH_RADIUS_MULT)

    # project root = parent of the scripts folder
    global SRC_SEM_PLY_PATH, SRC_INS_PLY_PATH, TGT_PLY_PATH, OUT_PLY_PATH
    root = Path(__file__).resolve().parents[1]
    SRC_SEM_PLY_PATH = root / SRC_SEM_PLY_PATH
    SRC_INS_PLY_PATH = root / SRC_INS_PLY_PATH
    TGT_PLY_PATH = root / TGT_PLY_PATH
    OUT_PLY_PATH = root / OUT_PLY_PATH
    # Load three clouds (source for semantics, source for instances, target)

    sem_src = load_ply(SRC_SEM_PLY_PATH)
    if SRC_SEM_PLY_PATH.samefile(SRC_INS_PLY_PATH):
        ins_src = sem_src
    else:
        ins_src = load_ply(SRC_INS_PLY_PATH)
    tgt = load_ply(TGT_PLY_PATH)

    # Sanity checks for required fields on sources
    assert "classes" in sem_src.scalar_fields, "Semantic source must have scalar_field['classes']"
    assert "instances" in ins_src.scalar_fields, "Instance source must have scalar_field['instances']"

    src_sem = np.asarray(sem_src.scalar_fields["classes"]).astype(np.int32, copy=False)
    src_ins = np.asarray(ins_src.scalar_fields["instances"]).astype(np.int32, copy=False)

    # Ensure target label fields exist (& zero-init)
    N_t = tgt.xyz.shape[0]

    tgt.scalar_fields.__setitem__('instances', np.zeros(N_t, dtype=src_ins.dtype))
    tgt.scalar_fields.__setitem__('classes', np.zeros(N_t, dtype=src_sem.dtype))

    # Transfer semantics (from semantic source → target)
    tgt_sem, _dist_sem = transfer_nn_within_radius_single(
        src_xyz=sem_src.xyz,
        src_lab=src_sem,
        tgt_xyz=tgt.xyz,
        r=r
    )

    # Transfer instances (from instance source → target)
    tgt_ins, _dist_ins = transfer_nn_within_radius_single(
        src_xyz=ins_src.xyz,
        src_lab=src_ins,
        tgt_xyz=tgt.xyz,
        r=r
    )

    # Optional smoothing on target
    if SMOOTHING_ENABLED:
        tgt_sem, tgt_ins = smoothing_weighted_majority(
            xyz=tgt.xyz,
            sem=tgt_sem,
            ins=tgt_ins,
            r_smooth=r_smooth,
            p=WEIGHT_EXP_P,
            t_purity=SMOOTH_CONF_TAU,
            min_neigh=SMOOTH_MIN_NEIGH,
            gate_mode="none",
            w_scaledown=0.2,
            allow_instance_if_sem0=True,
            eps=EPS,
        )


    # Final per-instance purity cleanup on target
    tgt_sem, tgt_ins = cleanup_instance_purity(
        sem=tgt_sem,
        ins=tgt_ins,
        purity=PURITY_THRESHOLD,
    )

    # Write back and save
    tgt.scalar_fields["classes"] = tgt_sem
    tgt.scalar_fields["instances"] = tgt_ins
    save_ply(OUT_PLY_PATH, tgt)


if __name__ == "__main__":
    main()
