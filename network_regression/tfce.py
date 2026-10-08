from functools import partial

import numpy as np
from multiprocess import Pool
from tqdm.notebook import tqdm


def extract_strongly_connected_components(adj_matrix):
    n = len(adj_matrix)

    # Step 1: DFS to compute finishing times
    def dfs(v, visited, stack):
        # start at vertex v
        visited[v] = True
        # for every possible neighbor,
        for i in range(n):
            # if we haven't traversed yet, and there is a link,
            if adj_matrix[v][i] and not visited[i]:
                # go there and initiate a dfs from that vertex
                dfs(i, visited, stack)
        
        stack.append(v)

    # Step 3: DFS on transposed graph
    def dfs_transposed(v, visited, component, transposed):
        visited[v] = True
        component.append(v)
        for i in range(n):
            if transposed[v][i] and not visited[i]:
                dfs_transposed(i, visited, component, transposed)

    # Step 1
    visited = [False] * n
    stack = []
    for i in range(n):
        if not visited[i]:
            dfs(i, visited, stack)

    # Step 2
    transposed = adj_matrix.T

    # Step 3
    visited = [False] * n
    components = []
    while stack:
        v = stack.pop()
        if not visited[v]:
            component = []
            dfs_transposed(v, visited, component, transposed)
            components.append(component)

    return components


def nbs_tfce_scc_enhance(statmap, E=1, H=1, hmax=1, steps=100, 
                         intensity_extent=False):
    integrands = []
    for hidx, h_thresh in enumerate(np.linspace(1/1000, hmax, steps)):
        testmap = statmap * (statmap > h_thresh)
        components = extract_strongly_connected_components(testmap)

        integrand = np.zeros_like(testmap)
        for targ, src in zip(*np.where(testmap)):
            # singleton link is directed and only forms a strongly connected
            # component if reverse link exists
            extent = 0
            for comp in components:
                if src in comp and targ in comp:
                    if intensity_extent:
                        # sum of all statistic values in this level, 
                        # for this scc
                        extent = np.sum(np.nan_to_num(testmap[comp].T[comp].T))
                    else:
                        extent = len(comp) 
    
            integrand[targ, src] = extent**E * h_thresh**H

        integrands.append(integrand)

    return np.sum(integrands, axis=0)


def enhance_model_scc(model, E=1, H=2, hmin=0, hmax=1, hypothesis='two-sided', 
                      intensity_extent=False):
    if hypothesis=='two-sided':
        # hmin is the minimum non-negative cluster forming threshold
        neg_model = model * (model < -hmin) 
        neg_model_enhanced = nbs_tfce_scc_enhance(-neg_model, E=E, H=H, 
                                                  hmax=hmax, 
                                            intensity_extent=intensity_extent)
        pos_model = model * (model > hmin)
        pos_model_enhanced = nbs_tfce_scc_enhance(pos_model, E=E, H=H, 
                                                  hmax=hmax, 
                                            intensity_extent=intensity_extent)
        return pos_model_enhanced - neg_model_enhanced
    elif hypothesis == 'upper':
        pos_model = model * (model > hmin)
        pos_model_enhanced = nbs_tfce_scc_enhance(pos_model, E=E, H=H, 
                                                  hmax=hmax, 
                                            intensity_extent=intensity_extent)
        return pos_model_enhanced 
    elif hypothesis == 'lower':
        # hmin is the minimum non-negative cluster forming threshold
        neg_model = model * (model < -hmin) 
        neg_model_enhanced = nbs_tfce_scc_enhance(-neg_model, E=E, H=H, 
                                                  hmax=hmax, 
                                            intensity_extent=intensity_extent)
        return -neg_model_enhanced
    elif hypothesis == 'upperabs':
        pos_model = np.abs(model)
        pos_model_enhanced = nbs_tfce_scc_enhance(pos_model, E=E, H=H, 
                                                  hmax=hmax, 
                                            intensity_extent=intensity_extent)
        return pos_model_enhanced 
    else:
        raise ValueError(f"hypothesis type {hypothesis} must be one of "
                         "'two-sided', 'upper', or 'lower'")
    

def network_cluster_permutation_test(statistic, null_distribution, E=2, H=3, 
                                     hmin=0, 
                                     n_perm=10_000,
                                     verbose=False, 
                                     hypothesis='two-sided', 
                                     intensity_extent=False):
    
    hmax = np.max([np.max(np.abs(statistic)), 
                   np.max(np.abs(null_distribution))])

    if verbose:
        print("setting maximum absolute statistic value as upper "
              f"tfce height limit: {hmax=:.3f}")

    
    enhance_model_partial = partial(enhance_model_scc, 
                                    E=E,
                                    H=H, 
                                    hmin=hmin, 
                                    hmax=hmax,
                                    hypothesis=hypothesis, 
                                    intensity_extent=intensity_extent)

    # enhance the true t statistic map with network tfce. in normal tfce, the t
    # map only includes t values with significant p, but doing that here would
    # fragment the network (often by a lot) so instead we skip the p-value
    # thresholding and just enchance the raw t map and compare it to the
    # enhanced t maps from all permutations
    if verbose:
        print("processing true statistic map")
    enh = enhance_model_partial(statistic)

    # enhance all permuted model t statisic maps
    if verbose:
        print("processing null statistic maps")

    with Pool() as pool:
        null_distribution_enhanced = list(tqdm(pool.imap(enhance_model_partial, 
                                                         null_distribution),
                                            total=n_perm))

    null_distribution_enhanced = np.array(null_distribution_enhanced)

    # null distribution maximum tfce values 
    maxtfces = np.abs(null_distribution_enhanced).max(axis=1).max(axis=1)
    testval = np.abs(enh).max()
    monte_carlo_p = 1 - len(maxtfces[(maxtfces < testval)])/n_perm
    
    return monte_carlo_p, enh, null_distribution_enhanced


def nbstfce_test(statistic, null_distribution, E=1, H=1, n_perm=10_000, 
                intensity_extent=False, hypothesis='two-sided', verbose=False):

    # (n_draws, n_roi, n_roi)
    null_distribution = np.transpose(null_distribution, (2, 0, 1))
    
    monte_carlo_p, enh, null_distribution_enhanced = \
        network_cluster_permutation_test(statistic, null_distribution, 
                         E=E, 
                         H=H, 
                         hmin=0, 
                         hypothesis=hypothesis, 
                         intensity_extent=intensity_extent, 
                         n_perm=n_perm,
                         verbose=verbose)
    
    # single sided alpha
    alpha = 0.05

    nulltmax = null_distribution_enhanced.max(axis=1).max(axis=1)
    tmaxcutoff = np.sort(nulltmax)[::-1][int(n_perm*alpha)]
    
    nulltmin = null_distribution_enhanced.min(axis=1).min(axis=1)
    tmincutoff = -np.sort(-nulltmin)[::-1][int(n_perm*alpha)]
    
    return monte_carlo_p, \
           (enh, (enh * ((enh > tmaxcutoff) |(enh < tmincutoff)))), \
           null_distribution_enhanced