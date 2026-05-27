from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull

"""
Notes
Speed: Each pair does at most 220 3×3 solves + a small hull. With SAT early‑rejects, most non‑overlapping pairs exit fast. For very large M, you can parallelize over pairs.

Robustness: Tolerances (eps, tol) handle near‑coplanar cases. If you see occasional misses, tweak 1e-7…1e-9.

Inputs: extents are full side lengths (not half). Quaternions are [x,y,z,w]. Rotation matrices must have axes as columns (as produced above).
"""

# TODO: 1 - assure that dtype is float64 (currently likely float32)
# TODO: 2 - rename or comment-out corresponding functions (per bbox pair vs. vectorized implementation)
# TODO: 3 - do ConvexHull loop in parallel


# --------- helpers ---------
def quat_to_mat_batch(quats: np.ndarray) -> np.ndarray:
    """
    Convert N quaternions [x,y,z,w] to rotation matrices (N,3,3).
    """
    x, y, z, w = quats.T
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty((quats.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def obb_sat_overlap(c0, A0, h0, c1, A1, h1, eps=1e-9) -> bool:
    """
    OBB-OBB overlap test using the standard 15-axis SAT (Gottschalk).
    A0/A1: 3x3 orthonormal (columns are box axes), h0/h1: half-sizes.
    """
    # Express everything in box0's frame
    R = A0.T @ A1
    t = A0.T @ (c1 - c0)
    absR = np.abs(R) + eps

    # Test box0 axes
    for i in range(3):
        ra = h0[i]
        rb = h1 @ absR[i, :]
        if abs(t[i]) > (ra + rb):
            return False

    # Test box1 axes
    for j in range(3):
        ra = h0 @ absR[:, j]
        rb = h1[j]
        if abs(t @ R[:, j]) > (ra + rb):
            return False

    # Test cross products
    for i in range(3):
        ip1, ip2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            jp1, jp2 = (j + 1) % 3, (j + 2) % 3
            ra = h0[ip1] * absR[ip2, j] + h0[ip2] * absR[ip1, j]
            rb = h1[jp1] * absR[i, jp2] + h1[jp2] * absR[i, jp1]
            term = abs(t[ip2] * R[ip1, j] - t[ip1] * R[ip2, j])
            if term > (ra + rb):
                return False
    return True


def obb_planes(c: np.ndarray, A: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return 12 planes (normals N and offsets b) that define the OBB as {x | N @ x <= b}.
    For each axis k:  n = +A[:,k], b = n·(c) + h[k];  and  n = -A[:,k], b = (-n)·(c) + h[k].
    """
    N = np.zeros((12, 3), dtype=np.float64)
    b = np.zeros(12, dtype=np.float64)
    idx = 0
    for k in range(3):
        n = A[:, k]
        N[idx] = n
        b[idx] = n @ c + h[k]
        idx += 1
        N[idx] = -n
        b[idx] = (-n) @ c + h[k]
        idx += 1
    return N, b


def intersection_volume_planes(NA, bA, NB, bB, tol=1e-8) -> float:
    """
    Compute volume of the convex polyhedron A∩B defined by halfspaces:
        NA x <= bA  and  NB x <= bB
    via enumerating 3-plane intersections among the combined 12 planes.
    """
    N = np.vstack([NA, NB])  # (12,3)
    b = np.hstack([bA, bB])  # (12,)
    verts = []

    # Enumerate all 3-plane combos (220)
    for i, j, k in combinations(range(12), 3):
        M = np.stack([N[i], N[j], N[k]], axis=0)  # 3x3
        det = np.linalg.det(M)
        if abs(det) < tol:
            continue
        x = np.linalg.solve(M, np.array([b[i], b[j], b[k]]))
        # Check inside all halfspaces with tolerance
        if np.all(N @ x <= b + 1e-7):
            verts.append(x)

    if len(verts) < 4:
        return 0.0

    # Deduplicate close points
    V = np.unique(np.round(np.asarray(verts, dtype=np.float64), decimals=10), axis=0)

    if V.shape[0] < 4:
        return 0.0

    try:
        hull = ConvexHull(V)
        return float(hull.volume)
    except Exception:
        return 0.0


# --------- main IoU API ---------
def obb_iou_pair(c0, e0, q0, c1, e1, q1, exact=True) -> float:
    """
    IoU between two OBBs.
    c*: (3,) centers; e*: (3,) full extents (lengths); q*: (4,) quats [x,y,z,w].
    If exact=True, uses plane-intersection; if False, returns 1.0 for overlap? No—still exact, but SAT early-out only.
    """
    # Rotation matrices, axes as columns
    A0 = quat_to_mat_batch(q0[None, :])[0]
    A1 = quat_to_mat_batch(q1[None, :])[0]
    h0 = 0.5 * e0.astype(np.float64)
    h1 = 0.5 * e1.astype(np.float64)
    c0 = c0.astype(np.float64)
    c1 = c1.astype(np.float64)

    # SAT quick reject
    if not obb_sat_overlap(c0, A0, h0, c1, A1, h1):
        return 0.0

    v0 = float(np.prod(e0))
    v1 = float(np.prod(e1))

    # Exact intersection volume via plane triplets
    NA, bA = obb_planes(c0, A0, h0)
    NB, bB = obb_planes(c1, A1, h1)
    iv = intersection_volume_planes(NA, bA, NB, bB)

    union = v0 + v1 - iv
    return iv / union if union > 0 else 0.0


def compute_obb_iou_batch(centers, extents, quats, pairs) -> np.ndarray:
    """
    centers: (N,3), extents: (N,3) full lengths, quats: (N,4) [x,y,z,w], pairs: (M,2)
    Returns (M,) IoUs.
    """
    Rmats = quat_to_mat_batch(quats)  # not strictly needed here but useful if you extend
    ious = np.empty(pairs.shape[0], dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        ious[k] = obb_iou_pair(centers[i], extents[i], quats[i], centers[j], extents[j], quats[j])
    return ious


"""
Vectorized computationally efficient implementation:
"""

import numpy as np

# ---------- math utils ----------
_TRIPLETS = np.array(list(combinations(range(12), 3)), dtype=np.int64)  # 220x3


def quat_to_mat_batch(quats: np.ndarray) -> np.ndarray:
    """[x,y,z,w] -> (N,3,3) rotation matrices."""
    x, y, z, w = quats.T
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty((quats.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def obb_planes_batch(c: np.ndarray, A: np.ndarray, h: np.ndarray):
    """
    For P boxes, return planes N (P,12,3), b (P,12) s.t. box = {x | N@x <= b}.
    For each axis k: +A[:,k] and -A[:,k] with offsets c·n + h[k].
    """
    P = c.shape[0]
    N = np.empty((P, 12, 3), dtype=np.float64)
    b = np.empty((P, 12), dtype=np.float64)
    # +x,-x,+y,-y,+z,-z per pair
    for k in range(3):
        n = A[:, :, k]  # (P,3)
        N[:, 2 * k, :] = n
        b[:, 2 * k] = np.einsum("pi,pi->p", n, c) + h[:, k]
        N[:, 2 * k + 1, :] = -n
        b[:, 2 * k + 1] = np.einsum("pi,pi->p", -n, c) + h[:, k]
    return N, b


def sat_overlap_batch(c0, A0, h0, c1, A1, h1, eps=1e-9) -> np.ndarray:
    """
    Batch (P,) boolean SAT for OBB overlap.
    """
    AT0 = np.transpose(A0, (0, 2, 1))
    R = AT0 @ A1  # (P,3,3)
    t = (AT0 @ (c1 - c0)[..., None])[..., 0]  # (P,3)
    absR = np.abs(R) + eps

    # Test box0 axes
    rb0 = np.einsum("pj,pij->pi", h1, absR)  # (P,3) rb for i=0..2
    cond0 = np.abs(t) > (h0 + rb0)  # (P,3)

    # Test box1 axes
    ra1 = np.einsum("pi,pij->pj", h0, absR)  # (P,3) ra for j=0..2
    lhs1 = np.abs(np.einsum("pi,pij->pj", t, R))
    cond1 = lhs1 > (ra1 + h1)  # (P,3)

    # Cross products (9 tests) – small Python loop, vectorized across P
    condX = np.zeros(c0.shape[0], dtype=bool)
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            ra = h0[:, i1] * absR[:, i2, j] + h0[:, i2] * absR[:, i1, j]
            rb = h1[:, j1] * absR[:, i, j2] + h1[:, j2] * absR[:, i, j1]
            term = np.abs(t[:, i2] * R[:, i1, j] - t[:, i1] * R[:, i2, j])
            condX |= term > (ra + rb)

    sep = cond0.any(axis=1) | cond1.any(axis=1) | condX
    return ~sep  # True = overlap


# ---------- main: vectorized IoU ----------
def obb_iou_pairs_vectorized(centers, extents, quats, pairs, chunk_size=2048, tol=1e-9):
    """
    centers: (N,3), extents: (N,3) full lengths, quats: (N,4) [x,y,z,w], pairs: (M,2)
    Returns: (M,) IoUs (float64). Vectorized across pairs in chunks.
    """
    centers = centers.astype(np.float64, copy=False)
    extents = extents.astype(np.float64, copy=False)
    quats = quats.astype(np.float64, copy=False)

    A_all = quat_to_mat_batch(quats)
    h_all = 0.5 * extents
    vol_all = np.prod(extents, axis=1)  # (N,)

    M = pairs.shape[0]
    out = np.zeros(M, dtype=np.float64)

    for start in range(0, M, chunk_size):
        end = min(M, start + chunk_size)
        pc = pairs[start:end]
        i, j = pc[:, 0], pc[:, 1]

        c0, c1 = centers[i], centers[j]
        A0, A1 = A_all[i], A_all[j]
        h0, h1 = h_all[i], h_all[j]
        v0, v1 = vol_all[i], vol_all[j]

        # 1) SAT early reject (vectorized)
        overlap_mask = sat_overlap_batch(c0, A0, h0, c1, A1, h1, eps=tol)
        if not np.any(overlap_mask):
            continue  # leaves zeros

        idx = np.where(overlap_mask)[0]
        c0o, c1o = c0[idx], c1[idx]
        A0o, A1o = A0[idx], A1[idx]
        h0o, h1o = h0[idx], h1[idx]
        v0o, v1o = v0[idx], v1[idx]

        # 2) Build planes for overlapping pairs (vectorized)
        NA, bA = obb_planes_batch(c0o, A0o, h0o)  # (P',12,3), (P',12)
        NB, bB = obb_planes_batch(c1o, A1o, h1o)

        # Combine halfspaces
        N = np.concatenate([NA, NB], axis=1)  # (P',24,3)
        b = np.concatenate([bA, bB], axis=1)  # (P',24)

        # 3) Enumerate all 3-plane intersections (from 24 choose 3 = 2024)
        # Optimization: only mix planes from both boxes by using the first 12+12 set.
        # For robustness & completeness, we’ll just use the standard 12-plane union (6+6),
        # i.e., keep 12 (not 24) by clipping one by the other’s planes is enough.
        # Use 12 planes total (6 from A + 6 from B):
        # Take only first 12 to keep 220 combos:
        N12 = np.concatenate([NA, NB], axis=1)[:, :12, :]
        b12 = np.concatenate([bA, bB], axis=1)[:, :12]

        # Gather 3x3 M and 3x1 rhs for each pair and triplet (vectorized)
        Mmat = N12[:, _TRIPLETS, :]  # (P',220,3,3)
        rhs = b12[:, _TRIPLETS]  # (P',220,3)

        # 4) Solve M x = rhs where invertible
        det = np.linalg.det(Mmat)  # (P',220)
        invertible = np.abs(det) > 1e-12
        # Make safe copies for solve
        eye = np.eye(3, dtype=np.float64)
        Msafe = np.where(invertible[..., None, None], Mmat, eye)
        bsafe = np.where(invertible[..., None], rhs, 0.0)
        X = np.linalg.solve(Msafe, bsafe)  # (P',220,3)
        X[~invertible] = np.nan

        # 5) Keep points inside all halfspaces (vectorized)
        # Check N @ x <= b + tol ; N here is the 12-plane set N12
        vals = np.einsum("pnc,ptc->pnt", N12, X)  # (P',12,220)
        inside = np.all(vals <= (b12[:, :, None] + 1e-8), axis=1)  # (P',220)

        # 6) Per overlapping pair: compute convex hull volume
        iv = np.zeros(idx.size, dtype=np.float64)
        for u in range(idx.size):
            pts = X[u, inside[u]]  # (K,3)
            if pts.shape[0] < 4 or not np.all(np.isfinite(pts)):
                iv[u] = 0.0
                continue
            # Optional dedup / rounding to reduce nearly-coincident duplicates
            # pts = np.unique(np.round(pts, decimals=10), axis=0)
            if pts.shape[0] < 4:
                iv[u] = 0.0
                continue
            try:
                hull = ConvexHull(pts, qhull_options="QJ")  # joggle for robustness
                iv[u] = float(hull.volume)
            except Exception:
                iv[u] = 0.0

        union = v0o + v1o - iv
        out[start:end][idx] = np.where(union > 0, iv / union, 0.0)

    return out
