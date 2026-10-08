
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from network_regression.linkwise_regression import N_CHAINS, N_DRAWS

if TYPE_CHECKING:
    from network_regression.contrast import Contrast
    from network_regression.linkwise_regression import WorkerResult
    from network_regression.transform import TransformConfig


@dataclass
class StatmapResult:
    statmap: np.array
    statmap_null: np.array
    complete: bool = False


def marshal(
        n_files: int,
        resultsdir: str, 
        transformconfig: TransformConfig,
        contrast: Contrast,
    ) -> StatmapResult:

    n_nodes = transformconfig.blossom_num_regions * 2 # double for 2 hemis
    n_nulls = N_CHAINS * N_DRAWS

    statmap = np.zeros((n_nodes, n_nodes))
    statmap_null = np.zeros((n_nodes, n_nodes, n_nulls))

    completed = 0
    for machine in range(n_files):
        try:
            with open(f"{resultsdir}/machine_{machine}_chunkdata.pickle", 
                      "rb") as f:
                mdata_grouped: list[list[WorkerResult]] = pickle.load(f)

        # bad load
        except Exception as e:
            print(e)
            continue

        # flattened
        mdata: list[WorkerResult] = [j for i in mdata_grouped for j in i]
        
        # no data for this chunk
        for chunkidx in range(len(mdata)):
            if mdata[chunkidx] is None:
                continue

            truemodels = mdata[chunkidx].observed
            nullmodels = mdata[chunkidx].counterfactual

            src = truemodels.src
            targ = truemodels.targ

            if truemodels.posteriormode == 'applycontrastpositiveconditional':
                mu_a = truemodels.mu_a
                mu_b = truemodels.mu_b
                mu_a_cf = nullmodels.mu_a

            else:
                obsmu = truemodels.posterior_mu.values\
                            .reshape(N_CHAINS * N_DRAWS, -1)
                obsp = truemodels.posterior_p.values\
                            .reshape(N_CHAINS * N_DRAWS, -1)

                nullmu = nullmodels.posterior_mu.values\
                            .reshape(N_CHAINS * N_DRAWS, -1)
                nullp = nullmodels.posterior_p.values\
                            .reshape(N_CHAINS * N_DRAWS, -1)
                
                # point estimates
                mu_a = obsmu[:, ~contrast.group_A_indices].mean()
                mu_b = obsmu[:, contrast.group_A_indices].mean()

                # distribution
                mu_a_cf = nullmu[:, ~contrast.group_A_indices].mean(axis=1)
            
            statmap[targ, src] = mu_a - mu_b
            statmap_null[targ, src] = mu_a_cf - mu_b

        completed += 1

    return StatmapResult(
        statmap = statmap,
        statmap_null = statmap_null,
        complete = completed == n_files,
    )