from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class SegPCDCollection:
    # Mandatory input: list of point clouds (paths) and image features
    raw_pcd_paths: list[Path]  # n_pcds
    features: list
    class_id_map: dict
    # Optional override: when set, n_seg_pcds = n_raw_pcds * slots_per_scan
    slots_per_scan: int | None = None
    # Self-populated at initialization:
    n_raw_pcds: int = field(init=False)
    n_seg_pcds: int = field(init=False)
    n_features: int = field(init=False)
    raw_pcd_id: NDArray[np.uint16] = field(default_factory=lambda: np.array([], dtype=np.uint16))
    feature_id: NDArray[np.uint16] = field(default_factory=lambda: np.array([], dtype=np.uint16))
    seg_pcd_id: NDArray[np.uint16] = field(default_factory=lambda: np.array([], dtype=np.uint16))
    class_ids: NDArray[np.uint16] = field(default_factory=lambda: np.array([], dtype=np.uint16))
    class_names: list[str] = field(default_factory=list)
    global_shift: NDArray[np.float64] = field(default_factory=lambda: np.zeros([3], dtype=np.float64))
    # Self-populated later:
    pcd_n_instances: NDArray[np.uint64] = field(default_factory=lambda: np.array([], dtype=np.uint64))
    seg_pcds: list = field(default_factory=list)
    # TODO: Eventually add other point-cloud processing related parameters (e.g. theta, alpha)

    def __post_init__(self):
        # Full initialization with correct/final values:
        self.n_raw_pcds = len(self.raw_pcd_paths)
        self.n_features = len(self.features)
        effective_slots = self.slots_per_scan if self.slots_per_scan is not None else self.n_features
        self.n_seg_pcds = self.n_raw_pcds * effective_slots

        if self.raw_pcd_id.size == 0:
            self.raw_pcd_id = np.repeat(np.arange(self.n_raw_pcds, dtype=np.uint16), self.n_features)
        if self.feature_id.size == 0:
            self.feature_id = np.repeat(np.arange(self.n_features, dtype=np.uint16), self.n_raw_pcds)
        if self.seg_pcd_id.size == 0:
            self.seg_pcd_id = np.arange(self.n_seg_pcds, dtype=np.uint16)

        if self.class_ids.size == 0:
            ids_list = list(self.class_id_map.values())
            self.class_ids = np.array(ids_list)
        if not self.class_names:
            self.class_names = list(self.class_id_map.keys())

        # Incomplete initialization (only correct sizes)
        if self.pcd_n_instances.size == 0:
            self.pcd_n_instances = np.zeros(self.n_seg_pcds, dtype=np.uint64)
        if not self.seg_pcds:
            self.seg_pcds = [None] * self.n_seg_pcds

    def filter_out_instances(self, pcd_or: NDArray, inst_or: NDArray) -> None:

        # Find all point clouds with outliers
        unique_seg_pcds_with_or = np.unique(pcd_or)

        for seg_pcd_id in unique_seg_pcds_with_or:
            # Get segmented point cloud to filter:
            seg_pcd_idx = int(np.where(self.seg_pcd_id == seg_pcd_id)[0][0])
            seg_pcd_i = self.seg_pcds[seg_pcd_idx]
            # Get instances to remove:
            inst_to_remove = inst_or[pcd_or == seg_pcd_id]
            # Get points in point cloud to keep:
            points_to_keep = ~np.isin(seg_pcd_i.scalar_fields["instances"].data, inst_to_remove)
            # Reduce point cloud and update pcd_collection:
            seg_pcd_i.reduce(points_to_keep)
            self.seg_pcds[seg_pcd_idx] = seg_pcd_i
            self.pcd_n_instances[seg_pcd_idx] -= inst_to_remove.size

        return None
