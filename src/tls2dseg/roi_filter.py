from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def _is_axis_aligned_rect(poly: NDArray[np.floating], tol: float) -> bool:
    # Axis-aligned iff each vertex x is ~minx or ~maxx and y is ~miny or ~maxy
    x, y = poly[:, 0], poly[:, 1]
    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()
    return (np.all(np.isclose(x, minx, atol=tol) | np.isclose(x, maxx, atol=tol)) and
            np.all(np.isclose(y, miny, atol=tol) | np.isclose(y, maxy, atol=tol)))


def _is_convex_ordered(poly: NDArray[np.floating]) -> bool:
    # Assumes vertices are given in order around the boundary (CW or CCW)
    # Convex iff cross products of successive edges have the same sign
    v = poly
    e1 = np.roll(v, -1, axis=0) - v
    e2 = np.roll(v, -2, axis=0) - np.roll(v, -1, axis=0)
    cross_z = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    # Allow tiny numerical wiggle
    pos = (cross_z > 1e-15).sum()
    neg = (cross_z < -1e-15).sum()
    return not (pos and neg)  # either all >=0 or all <=0


def _mask_halfspace_convex_quadrilateral(
        xy: NDArray[np.floating],
        quad: NDArray[np.floating],
        include_boundary: bool,
        tol: float
) -> NDArray[np.bool_]:
    """
    Inside test for convex 4-gon using half-spaces:
    For each edge (vi->vj) with outward normal n, require n·(p - vi) <= 0.
    We orient n so that the polygon centroid is inside (<= 0).
    """
    v = quad
    e = np.roll(v, -1, axis=0) - v  # edges
    n = np.stack([e[:, 1], -e[:, 0]], axis=1)  # rotate +90° to get a normal
    c = v.mean(axis=0)  # centroid
    # Orient normals outward so centroid satisfies n·(c - vi) <= 0
    orient = (np.einsum('ij,ij->i', n, (c - v)) > 0.0)
    n[orient] *= -1.0

    # Evaluate all four half-spaces: scores = n·(p - vi)
    # Memory-friendly: do it edge-by-edge and AND the result
    inside = np.ones(xy.shape[0], dtype=bool)
    for i in range(4):
        scores = (xy[:, 0] - v[i, 0]) * n[i, 0] + (xy[:, 1] - v[i, 1]) * n[i, 1]
        if include_boundary:
            inside &= (scores <= tol)
        else:
            inside &= (scores < 0.0)
        if not inside.any():
            break
    return inside


def roi_mask_xy_rectaware(
        xy: NDArray[np.floating],
        roi_xy: NDArray[np.floating],
        include_boundary: bool = False,
        tol: float = 1e-12,
        axis_aligned_tol: float = 1e-9) -> NDArray[np.bool_]:
    """
    Fast mask for rectangular/quad ROIs:
    - If axis-aligned rectangle: AABB test (fastest).
    - Else if convex quadrilateral: 4 half-space tests (fast).
    - Else: falls back to `fallback_general(xy, roi_xy, include_boundary, tol)` if provided.
    """
    xy = np.asarray(xy, dtype=np.float64)
    poly = np.asarray(roi_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (N,2).")
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("roi_xy must be (M,2) with M>=3.")

    # AABB prefilter (also serves as final result for axis-aligned rectangles)
    minx, maxx = poly[:, 0].min(), poly[:, 0].max()
    miny, maxy = poly[:, 1].min(), poly[:, 1].max()

    if poly.shape[0] == 4 and _is_axis_aligned_rect(poly, axis_aligned_tol):
        if include_boundary:
            return ((xy[:, 0] >= minx - tol) & (xy[:, 0] <= maxx + tol) &
                    (xy[:, 1] >= miny - tol) & (xy[:, 1] <= maxy + tol))
        else:
            return ((xy[:, 0] > minx) & (xy[:, 0] < maxx) &
                    (xy[:, 1] > miny) & (xy[:, 1] < maxy))

    elif poly.shape[0] == 4 and _is_convex_ordered(poly):
        return _mask_halfspace_convex_quadrilateral(xy, poly, include_boundary, tol)

    # Fallback to your general polygon routine (ray crossing) if provided
    else:
        return roi_mask_xy(xy, poly, include_boundary=include_boundary, tol=tol)


def roi_mask_xy(
        xy: NDArray[np.floating],
        roi_xy: NDArray[np.floating],
        include_boundary: bool = True,
        tol: float = 1e-12,
        max_points_per_chunk: int = 2_000_000,
) -> NDArray[np.bool_]:
    """
    Boolean mask for points whose (x,y) lies inside a 2D polygon (even-odd rule).
    - Works for non-convex polygons.
    - Includes points on edges/vertices if include_boundary=True.
    - Ignores z (all z are acceptable).

    Parameters
    ----------
    xy : (N,2) array
        Point coordinates (x,y) of your point cloud, e.g. pcd.xyz[:, :2].
    roi_xy : (M,2) array-like
        Polygon vertices in XY (need not be closed; last vertex will be connected to the first).
        Example: np.array([[-11.83, 7.53],[0.81, 0.29],[-0.09, -0.87],[-12.7, 6.37]], float)
    include_boundary : bool
        If True, points exactly on polygon edges/vertices are kept.
    tol : float
        Tolerance for boundary detection (in XY units). Uses a scale-relative check per edge.
    max_points_per_chunk : int
        To limit peak memory, points are processed in chunks of this size.

    Returns
    -------
    mask : (N,) bool array
        True for points inside the polygon (and on boundary if requested).
    """
    xy = np.asarray(xy, dtype=np.float64)
    poly = np.asarray(roi_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be an (N,2) array of [x,y] points.")
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("roi_xy must be an (M,2) array with M>=3 polygon vertices.")

    N = xy.shape[0]
    mask = np.zeros(N, dtype=bool)
    if N == 0:
        return mask

    # --- Broad-phase: quick AABB reject to avoid work on far points
    minx = np.min(poly[:, 0])
    maxx = np.max(poly[:, 0])
    miny = np.min(poly[:, 1])
    maxy = np.max(poly[:, 1])

    in_box = (
            (xy[:, 0] >= minx) & (xy[:, 0] <= maxx) &
            (xy[:, 1] >= miny) & (xy[:, 1] <= maxy)
    )
    idx = np.nonzero(in_box)[0]
    if idx.size == 0:
        return mask  # nothing to do

    # --- Prepare polygon edges (i -> j = next vertex, wraps around)
    xi = poly[:, 0]
    yi = poly[:, 1]
    xj = np.roll(xi, -1)
    yj = np.roll(yi, -1)
    ex = xj - xi
    ey = yj - yi
    elen = np.hypot(ex, ey)
    # Handle possible duplicate consecutive vertices (zero-length edges)
    valid_edge = elen > 0

    # --- Helper: process one chunk of points
    def _mask_chunk(xc: NDArray[np.floating], yc: NDArray[np.floating]) -> NDArray[np.bool_]:
        # Ray-crossing (even-odd) for "strictly inside"
        # Only consider edges that straddle the scanline y=yc
        # cond_y: shape (n_points, n_edges)
        yi_b = yi[None, :]
        yj_b = yj[None, :]
        xi_b = xi[None, :]
        xj_b = xj[None, :]
        ex_b = ex[None, :]
        ey_b = ey[None, :]
        elen_b = elen[None, :]
        valid_b = valid_edge[None, :]

        yc_b = yc[:, None]
        xc_b = xc[:, None]

        # Edge crosses the horizontal ray at yc? (strictly between yi and yj)
        cond_y = ((yi_b > yc_b) != (yj_b > yc_b)) & valid_b

        # Compute x-coordinate of intersection for those edges
        # t = (yc - yi) / (yj - yi); x_int = xi + t*(xj - xi)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (yc_b - yi_b) / (yj_b - yi_b)
            x_int = xi_b + t * ex_b

        crossings = np.count_nonzero(cond_y & (xc_b < x_int), axis=1)
        inside = (crossings % 2) == 1

        if not include_boundary:
            # Exclude boundary later
            return inside

        # Boundary test: point-on-segment within tolerance
        # Check colinearity via cross product and projection parameter tproj in [0,1]
        # cross = (p - a) x e ; |cross| <= tol * |e|
        # tproj = ((p - a)·e) / |e|^2  and  0<=tproj<=1
        # Vectorized over edges
        pa_x = xc_b - xi_b
        pa_y = yc_b - yi_b
        cross = pa_x * ey_b - pa_y * ex_b  # (n_pts, n_edges)
        # Scale-aware tolerance per edge:
        tol_scaled = tol * elen_b
        colinear = np.abs(cross) <= tol_scaled

        # Avoid division by 0 on zero-length edges (already masked by valid_b)
        with np.errstate(divide='ignore', invalid='ignore'):
            tproj = (pa_x * ex_b + pa_y * ey_b) / (elen_b * elen_b)

        on_seg = valid_b & colinear & (tproj >= -tol) & (tproj <= 1 + tol)
        on_boundary = np.any(on_seg, axis=1)

        return inside | on_boundary

    # --- Chunked processing of candidates
    for start in range(0, idx.size, max_points_per_chunk):
        sl = idx[start:start + max_points_per_chunk]
        m = _mask_chunk(xy[sl, 0], xy[sl, 1])
        mask[sl] = m

    return mask

# Alternative formulation by GPT:

# from __future__ import annotations
# import numpy as np
# from dataclasses import dataclass
# from numpy.typing import NDArray
#
# @dataclass(frozen=True)
# class PolyEdges:
#     xi: NDArray[np.float64]
#     yi: NDArray[np.float64]
#     xj: NDArray[np.float64]
#     yj: NDArray[np.float64]
#     ex: NDArray[np.float64]
#     ey: NDArray[np.float64]
#     elen: NDArray[np.float64]
#     valid_edge: NDArray[np.bool_]
#
# def build_poly_edges(roi_xy: NDArray[np.floating]) -> PolyEdges:
#     poly = np.asarray(roi_xy, dtype=np.float64)
#     if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
#         raise ValueError("roi_xy must be an (M,2) array with M>=3.")
#     xi, yi = poly[:, 0], poly[:, 1]
#     xj, yj = np.roll(xi, -1), np.roll(yi, -1)
#     ex, ey = xj - xi, yj - yi
#     elen = np.hypot(ex, ey)
#     valid_edge = elen > 0
#     return PolyEdges(xi, yi, xj, yj, ex, ey, elen, valid_edge)
#
# def mask_points_in_polygon_chunk(
#     xc: NDArray[np.floating],
#     yc: NDArray[np.floating],
#     edges: PolyEdges,
#     include_boundary: bool = True,
#     tol: float = 1e-12,
# ) -> NDArray[np.bool_]:
#     xi_b = edges.xi[None, :]; yi_b = edges.yi[None, :]
#     xj_b = edges.xj[None, :]; yj_b = edges.yj[None, :]
#     ex_b = edges.ex[None, :]; ey_b = edges.ey[None, :]
#     elen_b = edges.elen[None, :]
#     valid_b = edges.valid_edge[None, :]
#
#     xc_b = xc[:, None]; yc_b = yc[:, None]
#
#     cond_y = ((yi_b > yc_b) != (yj_b > yc_b)) & valid_b
#     with np.errstate(divide='ignore', invalid='ignore'):
#         t = (yc_b - yi_b) / (yj_b - yi_b)
#         x_int = xi_b + t * ex_b
#     crossings = np.count_nonzero(cond_y & (xc_b < x_int), axis=1)
#     inside = (crossings % 2) == 1
#
#     if not include_boundary:
#         return inside
#
#     # boundary test
#     pa_x = xc_b - xi_b
#     pa_y = yc_b - yi_b
#     cross = pa_x * ey_b - pa_y * ex_b
#     tol_scaled = tol * elen_b
#     colinear = np.abs(cross) <= tol_scaled
#     with np.errstate(divide='ignore', invalid='ignore'):
#         tproj = (pa_x * ex_b + pa_y * ey_b) / (elen_b * elen_b)
#     on_seg = valid_b & colinear & (tproj >= -tol) & (tproj <= 1 + tol)
#     return inside | np.any(on_seg, axis=1)
#
# def roi_mask_xy(
#     xy: NDArray[np.floating],
#     roi_xy: NDArray[np.floating],
#     include_boundary: bool = True,
#     tol: float = 1e-12,
#     max_points_per_chunk: int = 2_000_000,
# ) -> NDArray[np.bool_]:
#     xy = np.asarray(xy, dtype=np.float64)
#     if xy.ndim != 2 or xy.shape[1] != 2:
#         raise ValueError("xy must be (N,2).")
#
#     edges = build_poly_edges(roi_xy)
#
#     # AABB broad-phase
#     minx, maxx = edges.xi.min(), edges.xi.max()
#     miny, maxy = edges.yi.min(), edges.yi.max()
#     in_box = (
#         (xy[:, 0] >= minx) & (xy[:, 0] <= maxx) &
#         (xy[:, 1] >= miny) & (xy[:, 1] <= maxy)
#     )
#     idx = np.nonzero(in_box)[0]
#     mask = np.zeros(xy.shape[0], dtype=bool)
#     if idx.size == 0:
#         return mask
#
#     for s in range(0, idx.size, max_points_per_chunk):
#         sl = idx[s:s + max_points_per_chunk]
#         mask[sl] = mask_points_in_polygon_chunk(
#             xy[sl, 0], xy[sl, 1], edges, include_boundary, tol
#         )
#     return mask
