import subprocess
import multiprocessing
from multiprocessing import Pool


import numpy as np
import pymc as pm
import arviz as az
import pytensor
import warnings
import pymc_extras as pmx 
import pytensor.tensor as pt
import os
import pandas as pd
import bambi as bmb
import copy
import pathlib


def general_worker(chunk):
    print(f"running general model process {os.getpid()}")
    compiledir = f"/export/vrishab/compile/{os.getpid()}"
    pathlib.Path(compiledir).mkdir(parents=True, exist_ok=True)

    os.environ["PYTENSOR_FLAGS"] = f"compiledir={compiledir}"
    
    trueresults = []
    nullresults = []
    for i, j, data_full, formula, priors, interventions, nullstartidx, nullblocksize, nullmethod in chunk:
        assert nullblocksize < 4*4000, "null blocksize must be less than chains (4) * draws (4000)"
        assert (4*4000)//nullblocksize == (4*4000)/nullblocksize, "null blocksize must evenly divide chains (4) * draws (4000)"
        assert isinstance(data_full, pd.DataFrame), "data object must be a pandas dataframe"
        assert "connectivity" in data_full.columns, "data must have connectivity as a column name"
        assert 'zero' in formula and 'nonzero' in formula, "zero and nonzero model formulas must be provided"
        assert 'zero' in priors and 'nonzero' in priors, "zero and nonzero model priors must be provided (can be None)"
        
        data = data_full[data_full.connectivity > 0]
        data_binary = data_full.copy()
        data_binary['connectivity'] = data_binary.connectivity == 0
        
         # fit nonzero terms; hurdle doesn't work in bambi
        fullmodel_true = bmb.Model(formula['nonzero'], data, priors=priors['nonzero'], family='hurdle_lognormal')
        fullmodel_true.build()
        fullmodel_true = fullmodel_true.backend.model
        
        fullmodel_true_bool = bmb.Model(formula['zero'], data_binary, priors=priors['zero'], family='bernoulli')
        fullmodel_true_bool.build()
        fullmodel_true_bool = fullmodel_true_bool.backend.model
        
        try:
            with fullmodel_true:
                tracefulltrue = pm.sample(4000, 
                                    chains=4, 
                                    return_inferencedata=True, 
                                    target_accept=0.97, 
                                    cores=1, 
                                    idata_kwargs={"log_likelihood": True}, 
                                    progressbar=False)

            with fullmodel_true_bool:
                tracefulltruebool = pm.sample(4000, 
                                    chains=4, 
                                    return_inferencedata=True, 
                                    target_accept=0.97, 
                                    cores=1, 
                                    idata_kwargs={"log_likelihood": True}, 
                                    progressbar=False)
            
        except ValueError as ex:
            print(f"worker received valueerror for cell ({i}, {j}): '{ex}'")
            print("This is likely because one or more grouping variables have no data")
            continue
            
        posterior = tracefulltrue.posterior
        posteriorbool = tracefulltruebool.posterior
        post_mu = posterior.mu
        post_p = posterior.p
        if nullmethod == 'onlymu':
            post_p = None
        
        rhatnz = az.rhat(tracefulltrue)
        rhatbool = az.rhat(tracefulltruebool)
        
        trueresults.append([i, j, 0, [rhatnz, rhatbool], post_mu, post_p, data, data_binary])

        # counterfactual for nonzero model
        with pm.do(fullmodel_true, interventions) as m_do:
            postpred_do = mu_cf = pm.sample_posterior_predictive(
                tracefulltrue,
                var_names=["mu"],  
                random_seed=0
            )

        # counterfactual for bool model
        with pm.do(fullmodel_true_bool, interventions) as m_do:
            postpred_bool_do = pm.sample_posterior_predictive(
                tracefulltrue,
                var_names=["p"],  
                random_seed=0
            )
            
        post_cf_mu = postpred_do.posterior_predictive.mu
        post_cf_p = postpred_bool_do.posterior_predictive.p
        
        if nullmethod == 'onlymu':
            post_cf_p = None
            
        nullresults.append([i, j, None, None, post_cf_mu, post_cf_p, None, None]) 

   
    return trueresults, nullresults


def counterfactual_run_general_worker(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(general_worker, chunks)) # each worker gets one chunk at a time

    print("transmit")
    subprocess.call(["rm", "-rf", "/export/vrishab/compile"])
    print("clear cache")
    return res
