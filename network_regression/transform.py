from collections import deque
from dataclasses import dataclass

import mne
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


@dataclass
class TransformConfig:
    blossom_downsample: bool = True
    blossom_num_regions: int = 21   # ico1 = 84 roi -> 42 roi (21 per hemi)
    smoothmethod: str = 'bartlett'  # bartlett or gaussian or None
    smooth_kernel: float = 0.03     # bartlett radius or gaussian sigma
    smooth_across_hemis: bool = False   # prevent interhemispheric leakage
    num_native_roi: int = 84        # ico1 = 84 roi     


def create_balanced_transform(src, n_super_hemi):
    def build_graph(src_hemi):
        G = nx.Graph()
        tris = src_hemi['use_tris']
        for tri in tris:
            G.add_edge(tri[0], tri[1])
            G.add_edge(tri[1], tri[2])
            G.add_edge(tri[2], tri[0])
        return G

    def get_matching_priority(G):
        """
        Use Edmonds' blossom algorithm to define preferred neighbors.
        Returns dict: node -> set(preferred neighbors)
        """
        matching = nx.algorithms.matching.max_weight_matching(
            G, maxcardinality=True
        )

        priority = {v: set() for v in G.nodes}
        for u, v in matching:
            priority[u].add(v)
            priority[v].add(u)

        return priority

    def balanced_labels(G, n_super):
        n_vertices = G.number_of_nodes()
        target_size = n_vertices // n_super

        unassigned = set(G.nodes)
        labels = np.full(n_vertices, -1, dtype=int)

        # --- blossom-derived priority ---
        priority_neighbors = get_matching_priority(G)

        cluster_id = 0

        while unassigned and cluster_id < n_super:
            start = min(unassigned)  # preserves deterministic ordering
            unassigned.remove(start)

            queue = deque([start])
            cluster = [start]

            while queue and len(cluster) < target_size:
                v = queue.popleft()

                # --- PRIORITY: matched neighbors first ---
                neighbors = list(G.neighbors(v))

                matched = [n for n in neighbors 
                           if n in priority_neighbors[v]]
                unmatched = [n for n in neighbors 
                             if n not in priority_neighbors[v]]

                for group in (matched, unmatched):
                    for n in group:
                        if n in unassigned:
                            cluster.append(n)
                            queue.append(n)
                            unassigned.remove(n)

                        if len(cluster) >= target_size:
                            break
                    if len(cluster) >= target_size:
                        break

            for v in cluster:
                labels[v] = cluster_id

            cluster_id += 1

        # assign leftovers to last cluster
        for v in unassigned:
            labels[v] = cluster_id - 1

        return labels

    # --- process hemispheres ---
    G_lh = build_graph(src[0])
    labels_lh = balanced_labels(G_lh, n_super_hemi)

    G_rh = build_graph(src[1])
    labels_rh = balanced_labels(G_rh, n_super_hemi)

    # offset RH labels
    labels = np.concatenate([labels_lh, labels_rh + n_super_hemi])

    n_super_total = n_super_hemi * 2
    n_vertices_total = len(labels)

    # --- build transformation matrix ---
    T = np.zeros((n_super_total, n_vertices_total))
    for i in range(n_super_total):
        verts = np.where(labels == i)[0]
        if len(verts) > 0:
            T[i, verts] = 1.0 / len(verts)

    return T, labels


def make_gaussian_smoothing_matrix(src, sigma=0.015, hemispherewise=True):
    """
    Create a spatial smoothing matrix W for an MNE source space.

    Parameters
    ----------
    src : list of dict
        The MNE source space (src[0] = L hemisphere, src[1] = R hemisphere).
    sigma : float
        Gaussian kernel standard deviation in meters (default 0.015 = 15 mm).
    hemispherewise : bool
        If True, smoothing is restricted to each hemisphere separately.

    Returns
    -------
    W : ndarray, shape (n_vertices, n_vertices)
        The smoothing matrix. Apply to a connectivity matrix as:
            C_smooth = W @ C @ W.T
    """
    # get vertex coordinates for each hemisphere
    rr_lh = src[0]["rr"][src[0]["vertno"]]
    rr_rh = src[1]["rr"][src[1]["vertno"]]
    
    coords = np.vstack([rr_lh, rr_rh])  # shape (n_vertices, 3)

    # compute pairwise euclidean distances
    D = cdist(coords, coords)  # n_vertices x n_vertices

    # gaussian kernel
    W = np.exp(-(D ** 2) / (2 * sigma ** 2))

    if hemispherewise:
        n_lh = rr_lh.shape[0]
        W[:n_lh, n_lh:] = 0
        W[n_lh:, :n_lh] = 0

    # row-normalize
    W = W / W.sum(axis=1, keepdims=True)
    
    return W

def make_bartlett_smoothing_matrix(src, r_max=0.02, hemispherewise=True):
    """
    r_max : float
        The maximum distance for smoothing in meters (e.g., 0.02 = 20mm).
        Weights drop linearly to 0 at this distance.
    """
    rr_lh = src[0]["rr"][src[0]["vertno"]]
    rr_rh = src[1]["rr"][src[1]["vertno"]]
    coords = np.vstack([rr_lh, rr_rh])
    
    # Pairwise distances
    D = cdist(coords, coords)

    # Bartlett Kernel: 1 - (d/r_max) for d < r_max, else 0
    W = np.maximum(0, 1 - (D / r_max))
    
    # Apply hemisphere constraint
    if hemispherewise:
        n_lh = rr_lh.shape[0]
        W[:n_lh, n_lh:] = 0
        W[n_lh:, :n_lh] = 0

    # Row-normalize: ensures we don't scale the connectivity values up/down
    W = W / W.sum(axis=1, keepdims=True)
    
    return W


def downsample_smooth_normalize(
        df: pd.DataFrame, 
        src_target: mne.SourceSpaces,
        transformconfig: TransformConfig
    ) -> pd.DataFrame:
    ############################################################################
    # set up transforms
    ############################################################################
    if transformconfig.blossom_downsample:
        n_roi = transformconfig.blossom_num_regions
        anat, _ = create_balanced_transform(src_target, n_super_hemi=n_roi)
    else:
        assert transformconfig.blossom_num_regions == \
               transformconfig.num_native_roi, \
               "roi mismatch: no spatial reduction performed"
        
        anat = np.eye(transformconfig.num_native_roi)
    
    xhemismooth = transformconfig.smooth_across_hemis
    smoothkernel = transformconfig.smooth_kernel
    if transformconfig.smoothmethod == 'bartlett':
        W = make_bartlett_smoothing_matrix(src_target, 
                                           r_max=smoothkernel, 
                                           hemispherewise=xhemismooth)
    elif transformconfig.smoothmethod == 'gaussian':
        W = make_gaussian_smoothing_matrix(src_target, 
                                           sigma=smoothkernel, 
                                           hemispherewise=xhemismooth)
    else:
        W = np.eye(transformconfig.num_native_roi)
    
    morph = W @ anat.T  # smooth -> downsample

    ############################################################################
    # normalize -> smooth -> downsample
    ############################################################################
    
    df = df.copy()
    
    # invariant: for historical reasons, the 'B' statistic is what we call the
    # count-normalized J statistc
    df.insert(0, 'B', 
        df.apply(
            lambda x: morph.T @ x['J']/x['J'].sum() @ morph, 
            axis=1
        )
    )

    df.insert(0, 'total', df.apply(lambda x: x['J'].sum(), axis=1))

    return df