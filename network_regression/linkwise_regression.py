from __future__ import annotations  # noqa: EXE002

import multiprocessing
import os
import pathlib
import subprocess
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import arviz as az
import bambi as bmb
import pandas as pd
import pymc as pm
import pytensor

if TYPE_CHECKING:
    from network_regression.farm import Chunk


N_CHAINS = 4
N_DRAWS = 4000


@dataclass
class InferenceResult:
    is_counterfactual: bool
    src: int
    targ: int
    rhat_nonzero: Any | None
    rhat_zero: Any | None
    posterior_mu: az.InferenceData
    posterior_p: az.InferenceData
    data_nonzero: pd.DataFrame | None
    data_zero: pd.DataFrame | None

@dataclass
class WorkerResult:
    observed: InferenceResult
    counterfactual: InferenceResult


def general_worker(
        chunk: Chunk, 
        compilebasedir: str
    ) -> list[WorkerResult]:
    print(f"running general model process {os.getpid()}")
    compiledir = f"{compilebasedir}/{os.getpid()}"
    pathlib.Path(compiledir).mkdir(parents=True, exist_ok=True)

    os.environ["PYTENSOR_FLAGS"] = f"compiledir={compiledir}"
    
    results = []

    for job in chunk.jobs:
        src = job.src 
        targ = job.targ 
        data_full = job.linkdata 
        formulae = job.contrast.formulae
        priors = job.contrast.priors
        interventions = job.contrast.interventions

        assert isinstance(data_full, pd.DataFrame), \
            "data object must be a pandas dataframe"
        assert "connectivity" in data_full.columns, \
            "data must have connectivity as a column name"
        assert 'zero' in formulae and 'nonzero' in formulae, \
            "zero and nonzero model formulas must be provided"
        assert 'zero' in priors and 'nonzero' in priors, \
            "zero and nonzero model priors must be provided (can be None)"
        
        data = data_full[data_full.connectivity > 0]
        data_binary = data_full.copy()
        data_binary['connectivity'] = data_binary.connectivity == 0
        
         # fit nonzero terms; hurdle doesn't work in bambi
        fullmodel_true = bmb.Model(formulae['nonzero'], 
                                   data, 
                                   priors=priors['nonzero'], 
                                   family='hurdle_lognormal')
        
        fullmodel_true.build()
        fullmodel_true = fullmodel_true.backend.model
        
        fullmodel_true_bool = bmb.Model(formulae['zero'], 
                                        data_binary, 
                                        priors=priors['zero'], 
                                        family='bernoulli')
        
        fullmodel_true_bool.build()
        fullmodel_true_bool = fullmodel_true_bool.backend.model
        
        try:
            with fullmodel_true:
                tracefulltrue = pm.sample(N_DRAWS, 
                                    chains=N_CHAINS, 
                                    return_inferencedata=True, 
                                    target_accept=0.97, 
                                    cores=1, 
                                    idata_kwargs={"log_likelihood": True}, 
                                    progressbar=False)

            with fullmodel_true_bool:
                tracefulltruebool = pm.sample(N_DRAWS, 
                                    chains=N_CHAINS, 
                                    return_inferencedata=True, 
                                    target_accept=0.97, 
                                    cores=1, 
                                    idata_kwargs={"log_likelihood": True}, 
                                    progressbar=False)
        except ValueError as ex:
            print("worker received valueerror for observed models"
                  f"({src}, {targ}): '{ex}' \n"
                  f"data: {data}")
            print("This is likely because one or more grouping variables "
                  "have no data")
            continue    
        
        try:
            # counterfactual for nonzero model
            with pm.do(fullmodel_true, interventions) as m_do:
                postpred_do = pm.sample_posterior_predictive(
                    tracefulltrue,
                    var_names=["mu"],  
                    random_seed=0
                )

            # counterfactual for bool model
            with pm.do(fullmodel_true_bool, interventions) as m_do:
                postpred_bool_do = pm.sample_posterior_predictive(
                    tracefulltruebool,
                    var_names=["p"],  
                    random_seed=0
                )
            
        except ValueError as ex:
            print("worker received valueerror for counterfactual models"
                  f"({src}, {targ}): '{ex}' \n"
                  f"data: {data}")
            print("This is likely because one or more grouping variables "
                  "have no data")
            continue    
            
        posterior = tracefulltrue.posterior
        posteriorbool = tracefulltruebool.posterior
        post_mu = posterior.mu
        post_cf_mu = postpred_do.posterior_predictive.mu
        
        
        rhatnz = az.rhat(tracefulltrue)

        if job.posteriormode is None:
            post_p = posteriorbool.p
            post_cf_p = postpred_bool_do.posterior_predictive.p
            rhatbool = az.rhat(tracefulltruebool)
        elif job.posteriormode == 'positiveconditional': # save mem
            post_p = None
            post_cf_p = None
            rhatbool = None
            data_binary = None
        else:
            raise ValueError(f"posteriormode {job.posteriormode} not one of "
                             "None or positiveconditional")
        
        obs_result = InferenceResult(
            is_counterfactual = False,
            src = src,
            targ = targ,
            rhat_nonzero = rhatnz,
            rhat_zero = rhatbool,
            posterior_mu = post_mu,
            posterior_p = post_p,
            data_nonzero = data, 
            data_zero = data_binary
        )

        cf_result = InferenceResult(
            is_counterfactual = True,
            src = src,
            targ = targ,
            rhat_nonzero = None,        # avoid duplicate in observed result 
            rhat_zero = None,           # avoid duplicate in observed result 
            posterior_mu = post_cf_mu,
            posterior_p = post_cf_p,
            data_nonzero = None,        # avoid duplicate in observed result 
            data_zero = None            # avoid duplicate in observed result 
        )

        results.append(WorkerResult(
            observed=obs_result,
            counterfactual=cf_result,
        ))

    return results


def counterfactual_run_general_worker(
        chunks: list[Chunk], 
        compiledir: str
    ) -> list[WorkerResult]:
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server
    pytensor.config.cxx = "/usr/bin/clang++"

    worker = partial(general_worker, compilebasedir=compiledir)

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(worker, chunks)) # each worker gets one chunk at a time

    print("transmit")
    subprocess.call(["rm", "-rf", compiledir])
    print("clear cache")

    return res
