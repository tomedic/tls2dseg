"""Visualization helpers — matplotlib 3D scatter and histogram.

Phase 4 plan 05 Task 1 (ENG-07). Verbatim move from
``tls2dseg.visualization`` (all functions unchanged; only the module
docstring and logger are new — PATTERNS.md: verbatim move).

Heavy imports (matplotlib, scipy.spatial.transform) are at module level
because every function here requires them. The module does NOT force a
matplotlib backend (``matplotlib.use("TkAgg")`` was removed in Phase 2;
users configure their backend via matplotlibrc).

Closest analog: ``tls2dseg.visualization`` (exact source).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger("tls2dseg.viz.render")


def set_axes_equal(ax: object) -> None:
    """Set equal scaling for a 3D plot."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def plot_3d_scatter(points: np.ndarray) -> None:
    """
    Visualizes a 3D scatter plot of 3D points.
    Parameters:
    points (numpy.ndarray): Nx3 numpy array where each row represents a 3D point (x, y, z).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ax.scatter(x, y, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)
    plt.show()


def plot_2_point_clouds(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_color: str = "r",
    target_color: str = "b",
    title: str = "Point Cloud Comparison",
) -> None:
    """
    Plots two point clouds using Matplotlib.

    Parameters:
        source_points (numpy.ndarray): The source point cloud as a NumPy array of shape (N, 3).
        target_points (numpy.ndarray): The target point cloud as a NumPy array of shape (M, 3).
        source_color (str): Color for the source point cloud (default is 'r' for red).
        target_color (str): Color for the target point cloud (default is 'b' for blue).
        title (str): Title of the plot (default is 'Point Cloud Comparison').
    """
    assert source_points.shape[1] == 3, "Source point cloud must have shape (N, 3)"
    assert target_points.shape[1] == 3, "Target point cloud must have shape (M, 3)"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        source_points[:, 0],
        source_points[:, 1],
        source_points[:, 2],
        c=source_color,
        label="Source",
        alpha=0.5,
    )
    ax.scatter(
        target_points[:, 0],
        target_points[:, 1],
        target_points[:, 2],
        c=target_color,
        label="Target",
        alpha=0.5,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    set_axes_equal(ax)
    plt.show()


def display_histogram(
    d: object,
    bins: int = 10,
    title: str = "Histogram",
    xlabel: str = "Values",
    ylabel: str = "Frequency",
) -> None:
    """
    Displays a histogram of the values stored in vector d.

    Parameters:
    - d: Input data (array-like)
    - bins: Number of bins for the histogram (default is 10)
    - title: Title of the histogram plot (default is "Histogram")
    - xlabel: Label for the x-axis (default is "Values")
    - ylabel: Label for the y-axis (default is "Frequency")
    """
    plt.figure(figsize=(8, 6))
    plt.hist(d, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def compute_bbox_corners(bbox_data: np.ndarray) -> np.ndarray:
    """
    Computes the eight corners of a bounding box in a consistent order.
    For an oriented box, the local corners are computed using the half extents
    and then transformed to global coordinates.
    """
    center = bbox_data[:3]
    extent = bbox_data[3:6]
    quaternions = bbox_data[6:]
    R3x3 = R.from_quat(np.squeeze(quaternions))
    R3x3 = R3x3.as_matrix()
    half = extent / 2.0

    v0_local = np.array([-half[0], -half[1], -half[2]])
    v1_local = np.array([half[0], -half[1], -half[2]])
    v2_local = np.array([half[0], half[1], -half[2]])
    v3_local = np.array([-half[0], half[1], -half[2]])
    v4_local = np.array([-half[0], -half[1], half[2]])
    v5_local = np.array([half[0], -half[1], half[2]])
    v6_local = np.array([half[0], half[1], half[2]])
    v7_local = np.array([-half[0], half[1], half[2]])
    local_corners = np.array([v0_local, v1_local, v2_local, v3_local, v4_local, v5_local, v6_local, v7_local])
    corners = (R3x3 @ local_corners.T).T + center.T
    return corners


def visualize_bbox_3d(point_cloud: np.ndarray, bbox_data: dict) -> None:
    """
    Visualizes a 3D point cloud and its bounding box using Matplotlib.
        bbox_data details in bbox_functions -> extract_bounding_box
    """
    point_size = 1
    bbox_opacity = 0.3
    bbox_edge_color = "k"
    bbox_face_color = "red"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        s=point_size,
        color="blue",
        alpha=0.6,
    )

    corners = compute_bbox_corners(bbox_data)
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[3], corners[2], corners[6], corners[7]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[0], corners[3], corners[7], corners[4]],
    ]

    bbox_mesh = Poly3DCollection(faces, alpha=bbox_opacity, facecolor=bbox_face_color, edgecolor=bbox_edge_color)
    ax.add_collection3d(bbox_mesh)
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
