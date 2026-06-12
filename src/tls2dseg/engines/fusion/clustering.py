"""Graph clustering helpers — extracted verbatim from graph_clustering.py lines 237-595.

Phase 4 plan 03 Task 1 (ENG-07). Provides clustering algorithms and
outlier detection for the fusion pipeline:

* ``detect_upper_tail_outliers`` — upper-tail outlier detection (IQR/MAD/percentile/NB)
* ``UnionFind`` — union-find data structure for PCC
* ``pcc_strict_nondecreasing`` — PCC with multiplicity-preserving supporters
* ``hcs_labels`` — HCS via recursive global min-cut (networkx)
* ``graph_clustering`` — dispatcher for leiden/hcs/pcc methods

Import-cycle rule (RESEARCH Pitfall 2): this module imports ONLY numpy,
scipy.stats, collections, typing, and stdlib (networkx / igraph / leidenalg
are imported lazily inside method bodies). It MUST NOT import from
engines/__init__.py or preprocessing/.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Literal

import numpy as np
from scipy.stats import nbinom

logger = logging.getLogger("tls2dseg.engines.fusion.clustering")


def detect_upper_tail_outliers(
    data: np.ndarray,
    method: str = "negative_binomial",
    *,
    iqr_factor: float = 1.5,
    mad_factor: float = 3.0,
    percentile: float = 95.0,
    alpha: float = 0.05,
) -> tuple[np.ndarray, float]:
    """
    Detect upper-tail outliers in a 1D positive integer array.

    Parameters
    ----------
    data : (N,) array
        Any (developed for the per-detection counts)
    method : {"iqr","mad","percentile","negative_binomial"}
    iqr_factor : float
        multiplier for IQR fence: cutoff = Q3 + iqr_factor*(Q3-Q1)
    mad_factor : float
        multiplier for MAD fence: cutoff = median + mad_factor*MAD
    percentile : float
        for "percentile" method, cutoff = percentile-th quantile of data
    alpha : float
        for "negative_binomial" method, cutoff = nbinom.ppf(1 - alpha, r, p)

    Returns
    -------
    outlier_indices : 1D int array
        indices in `data` that exceed the cutoff.
    cutoff : float
        the numeric threshold used.
    """
    if method == "iqr":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        cutoff = q3 + iqr_factor * iqr
    elif method == "mad":
        med = np.median(data)
        mad = np.median(np.abs(data - med))
        cutoff = med + mad_factor * mad
    elif method == "percentile":
        cutoff = np.percentile(data, percentile)
    elif method == "negative_binomial":
        # Detect outliers based on negative binomial distribution
        mean, var = data.mean(), data.var(ddof=1)
        # Var = mean + mean^2 / r  ->  r = mean^2 / (var - mean)
        r = mean**2 / max(var - mean, 1e-6)
        p = r / (r + mean)
        cutoff = nbinom.ppf(1 - alpha, r, p)
    else:
        raise ValueError(f"Unknown method {method!r}")

    outliers = np.nonzero(data > cutoff)[0]
    outliers.astype(np.uint16)
    return outliers, cutoff


# ---------- utilities ----------
class UnionFind:
    def __init__(self, n):
        self.par = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x):
        while self.par[x] != x:
            self.par[x] = self.par[self.par[x]]
            x = self.par[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            self.par[ra] = rb
            return rb
        elif self.rank[rb] < self.rank[ra]:
            self.par[rb] = ra
            return ra
        else:
            self.par[rb] = ra
            self.rank[ra] += 1
            return ra


def pcc_strict_nondecreasing(
    num_nodes: int,
    pairs: np.ndarray,  # (M,2)
    supporters: np.ndarray,  # (M,)
    min_supporters: int = 2,
    quantiles: tuple[int, ...] = (99, 95, 90, 80, 70, 60, 50),
) -> tuple[np.ndarray, np.ndarray]:
    """
    PCC with multiplicity-preserving supporter counters.
    The supporter count of any surviving edge never goes down.
    """
    uf = UnionFind(num_nodes)

    # --- 1. Initial Counter for each node -------------------------------
    Support: list[Counter] = [Counter() for _ in range(num_nodes)]
    for (i, j), sup in zip(pairs, supporters, strict=False):
        Support[i][j] += sup
        Support[j][i] += sup

    # Sorted edge order
    order = np.argsort(-supporters)
    pairs_sorted, supp_sorted = pairs[order], supporters[order]

    thresholds = [max(int(np.percentile(supporters, q)), min_supporters) for q in quantiles]
    thresholds.append(min_supporters)

    current = 0  # pointer into sorted edge list
    active_edges = pairs_sorted  # edges still considered
    active_sup = supp_sorted.copy()  # their current weights

    for thr in thresholds:
        # ---- 2. merge all edges with supp >= thr ----------------------
        while current < len(active_sup) and active_sup[current] >= thr:
            u, v = active_edges[current]
            ru, rv = uf.find(u), uf.find(v)
            if ru != rv:
                # union; root_new is representative
                root_new = uf.union(ru, rv)
                root_old = rv if root_new == ru else ru
                # merge Counters: new counts = sum (keeps multiplicity)
                Support[root_new] += Support[root_old]
                Support[root_old].clear()
            current += 1

        # ---- 3. re-score surviving edges for NEXT round -------------
        if thr == thresholds[-1]:
            break  # last iteration

        root_of = np.fromiter((uf.find(i) for i in range(num_nodes)), dtype=np.int32)

        # Keep only inter-cluster edges
        keep = root_of[active_edges[:, 0]] != root_of[active_edges[:, 1]]
        active_edges = active_edges[keep]
        active_sup = active_sup[keep]

        # Recompute supporter counts (non-decreasing)
        new_sup = np.empty_like(active_sup)
        for idx, (u, v) in enumerate(active_edges):
            ru, rv = root_of[u], root_of[v]
            cu, cv = Support[ru], Support[rv]
            common = set(cu.keys()).intersection(cv.keys())
            s = sum(min(cu[k], cv[k]) for k in common)
            new_sup[idx] = s
        active_sup = new_sup

        # Sort edges for next threshold loop
        order2 = np.argsort(-active_sup)
        active_edges, active_sup = active_edges[order2], active_sup[order2]
        current = 0

    # ---- 4. final labels ---------------------------------------------
    roots = np.fromiter((uf.find(i) for i in range(num_nodes)), dtype=np.int32)
    _uniq, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32) + 1, active_sup


def hcs_labels(
    num_nodes: int,
    pairs: np.ndarray,  # shape (M, 2), int
    edge_weights: np.ndarray,  # shape (M,), float/int
    min_weight_for_connectivity: float = 1.0,  # treat w<=0 as "no edge"
) -> np.ndarray:
    """
    HCS via recursive global min-cut with robust connectivity handling.
    Returns 1-based cluster labels of shape (num_nodes,).
    """
    from collections.abc import Iterable

    import networkx as nx

    # Accumulate weights per undirected edge, filter <= 0 if desired --
    acc: dict[tuple[int, int], float] = defaultdict(float)
    for (i, j), w in zip(pairs, edge_weights, strict=False):
        if i == j:
            continue  # skip self-loops for min-cut
        w = float(w)
        if w < min_weight_for_connectivity:
            continue  # drop zero/negative supporter edges
        if j < i:
            i, j = j, i
        acc[(int(i), int(j))] += w

    # Build graph only from the edges we keep
    G = nx.Graph()
    if acc:
        G.add_weighted_edges_from([(u, v, w) for (u, v), w in acc.items()], weight="weight")

    # We still want every node to receive a label. Nodes not present in G become singletons.
    present = set(G.nodes())
    all_nodes = set(range(num_nodes))
    missing = sorted(all_nodes - present)

    labels = -np.ones(num_nodes, dtype=np.int32)
    current_label = 1

    def finalize(nodes: Iterable[int]):
        nonlocal current_label
        for n in nodes:
            labels[n] = current_label
        current_label += 1

    # Label isolated / missing nodes (no positive-weight edges)
    # You can choose: each as its own cluster, or group them together.
    for n in missing:
        finalize([n])

    # ---- 2) Recurse per connected component (guarantees connectivity) -------
    def recurse(H: nx.Graph):
        nonlocal current_label

        n = H.number_of_nodes()
        m = H.number_of_edges()

        # Base cases
        if n <= 1 or m == 0:
            finalize(H.nodes())
            return

        # Ensure connectivity (paranoia guard)
        comps = list(nx.connected_components(H))
        if len(comps) > 1:
            # Recurse per connected component; do NOT call stoer_wagner here
            for comp in comps:
                recurse(H.subgraph(comp).copy())
            return

        # Now safe: connected and has edges
        cut_value, (A, B) = nx.stoer_wagner(H, weight="weight")

        # HCS stopping rule — common heuristic uses min-cut > |V|/2
        # Adjust if your edge scale differs (e.g., normalize by average degree/weight).
        if cut_value > n / 2:
            finalize(H.nodes())
            return

        # Otherwise split and recurse
        recurse(H.subgraph(A).copy())
        recurse(H.subgraph(B).copy())

    # Kick off recursion for each component in G
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp).copy()
        recurse(sub)

    # Safety: any unlabeled node gets its own label (shouldn't happen now)
    unlab = np.where(labels < 0)[0]
    for u in unlab:
        finalize([int(u)])

    return labels


def graph_clustering(
    num_nodes: int,
    pairs: np.ndarray,  # (M,2) int32
    edge_weights: np.ndarray,  # (M,)  float  or int
    method: Literal["leiden", "hcs", "pcc"] = "leiden",
    *,
    # PCC-specific
    min_supporters: int = 1,
    quantiles: tuple[int, ...] = (90, 80, 70, 60, 50, 40, 30, 20, 10),
    # Leiden parameters
    leiden_resolution: float = 1.0,
) -> np.ndarray:
    """
    Parameters
    ----------
    num_nodes : total detections (rows in d3d_memory_bank)
    pairs      : edges (i,j)
    edge_weights : same length; for PCC must hold "supporter counts"
    method     : 'leiden' | 'hcs' | 'pcc'
    min_supporters: minimal number of supporting masks (detections 3d) needed for a mask merge
    quantiles: thresholds for "pcc"
    leiden_resolution: hyper-parameter steering resulting cluster sizes for Leiden
    Returns
    -------
    labels : (num_nodes,) int32 cluster id per detection ( -1 for isolated if HCS/PCC )
    """
    if method == "leiden":
        import igraph as ig
        import leidenalg as la

        # build igraph
        g = ig.Graph(n=num_nodes, edges=pairs.tolist(), edge_attrs={"weight": edge_weights})
        part = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=leiden_resolution,
        )
        labels = np.array(part.membership, dtype=np.int32) + 1

    elif method == "hcs":
        # Simple implementation of highly-connected-subgraphs ("quasi-clique") based on recursive min-cut with NetworkX
        labels = hcs_labels(num_nodes, pairs, edge_weights, min_weight_for_connectivity=1.0)
        labels = labels.astype(np.int32)

    elif method == "pcc":
        labels, _ = pcc_strict_nondecreasing(num_nodes, pairs, edge_weights, min_supporters, quantiles)
    else:
        raise ValueError(f"Unknown method {method}")

    return labels
