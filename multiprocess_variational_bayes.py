import subprocess
import multiprocessing
from multiprocessing import Pool
# from workers import simple_worker_norandslopes_counterfactual, power_analysis

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


def fit_fullmodel(obs, obs_nonzero, cv, subs, exposure, samplemode='nuts'):
    with pm.Model() as fullmodel:
        cv_data = pm.Data("cv_data", cv)
        subs_data = pm.Data("sub_idx", subs)
        obs_data = pm.Data("obs_data", obs_nonzero)
        obs_full = pm.Data("obs_full", obs)
        
        # fixed effects parameters
        intercept = pm.Normal("intercept", mu=0, sigma=5)                 # population baseline log-mean
        beta = pm.Normal('beta', 0, 1, shape=(1))

        # random effects (1|subject)
        sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=2.5)
        subject_offset = pm.Normal("subject_offset", mu=0, sigma=0.5, shape=pm.math.max(subs_data) + 1)
        subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

        # linear predictor
        eta_fixed = (cv_data @ beta)   # fixed effects
        eta = intercept + eta_fixed + exposure + subject_effects[subs_data]

        mu  = pm.Deterministic('mu', pm.math.exp(eta))
        psi = pm.Beta("psi", 1, 1)

        y_obs = pm.Poisson("y_obs", mu=mu, observed=obs_data)
        pm.Bernoulli("y_bernoulli", p=1 - psi, observed=(obs_full == 0) * 1.0)

        # see if sampler converges with obtaining invalid values
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            try:
                if samplemode == 'advi':
                    approx = pm.fit(method='advi', progressbar=False, callbacks=[])
                    tracefulltrue = approx.sample(4000)
                else:
                    tracefulltrue = pm.sample(4000, chains=4, return_inferencedata=True, 
                            target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                            progressbar=False)
                badconvflag = False

            except RuntimeWarning as e:
                print(f"true model {i}->{j} caught a RuntimeWarning exception:", e)
                badconvflag = True

        # here the sampler will raise a runtime warning but will still proceed with sampling
        # this way we will still obtain estimates, just with the knowledge that something
        # didn't go right
        if badconvflag:
            print("badconv, retry with tighened target accept")
            if samplemode == 'advi':
                approx = pm.fit(method='advi', progressbar=False, callbacks=[])
                tracefulltrue = approx.sample(4000)
            else:
                tracefulltrue = pm.sample(4000, chains=4, return_inferencedata=True, 
                        target_accept=0.98, cores=1, idata_kwargs={"log_likelihood": True}, 
                        progressbar=False)

    post = tracefulltrue.posterior

    _intercept = post['intercept'].mean(dim=('chain', 'draw')).values
    _mu = post['mu'].mean(dim=('chain', 'draw')).values
    _psi = post['psi'].mean().item()
    posterior_subsamples = {
        "intercept": _intercept, 
        "beta": post['beta'].mean(dim=('chain', 'draw')).values,
        "RE": post["RE"].mean(dim=("chain", "draw")).values,
        "mu": _mu,
        "psi": _psi,
        "exposure": exposure,
        "obs": obs,
        "badconv": badconvflag,
    }
    rhat = None # az.rhat(tracefulltrue)
    return tracefulltrue, fullmodel, rhat, posterior_subsamples

def generate_data(idata, samplesize, N_trials, obs_nonzero, cv, subs, exposure):
    # subject RE offset is a zero-mean normal with std sigma the prior for
    # this is fairly wide, so feeding new unobserved (out of sample)
    # subjects through the likelihood with the default will overrepresent
    # the within-subject variability. instead, we take the average std for
    # subject offsets in our preliminary data and use that as the sigma for
    # new priors for the unobserved subjects.
    sigma_offset_hat = idata.posterior['subject_offset'].values\
                                    .reshape(-1, 5).std(axis=0).mean()

    # some proportion of the preliminary data for this cell was zero, and we
    # are fitting ztp models which exclude zeros, so our artificial data
    # should drop that proportion of counts
    p_hat = idata.posterior['psi'].mean().item()

    N_new_subs = samplesize
    # N_trials pre and post per subject
    new_cv = np.array(([0] * N_trials + [1] * N_trials) *\
                                    N_new_subs)[:, np.newaxis]
    new_subs = np.repeat(np.array(list(range(N_new_subs))) + max(subs), 
                                    N_trials * 2)
    np.random.seed(0)
    # drop p_hat proportion of data 
    new_usemask = np.random.binomial(n=1, 
                        p=idata.posterior['psi'].mean().item(), 
                        size=len(new_subs)).astype(bool)

    new_cv = np.concatenate([cv, new_cv[new_usemask]])
    new_subs = np.concatenate([subs, new_subs[new_usemask]])
    new_obs = np.zeros_like(new_subs)
    
    with pm.Model() as fullmodel_extended: 
        # dummy vars; names need to be in model graph for set_data
        cv_data = pm.Data("cv_data", cv)
        subs_data = pm.Data("sub_idx", subs)
        obs_data = pm.Data("obs_data", obs_nonzero)
        
        # extend data and overwrite dummy vars
        pm.set_data({"cv_data": new_cv,
                    "sub_idx": new_subs,
                    "obs_data": new_obs})
        
        # fixed effects parameters
        intercept = pm.Normal("intercept", mu=0, sigma=5)                 
        beta = pm.Normal('beta', 0, 1, shape=(1))

        # random effects (1|subject)
        sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=2.5)
        subject_offset = pm.Normal("subject_offset", mu=0, sigma=0.5, shape=pm.math.max(subs_data) + 1)
        subject_offset_extended = pm.Normal("subject_offset_extended", mu=0, sigma=sigma_offset_hat, shape=N_new_subs)
        subject_effects = pm.Deterministic("RE", pm.math.concatenate([subject_offset, subject_offset_extended]) * sigma_sub)

        # linear predictor
        eta_fixed = (cv_data @ beta)   # fixed effects
        eta = intercept + eta_fixed + exposure + subject_effects[subs_data]
                
        mu  = pm.Deterministic('mu', pm.math.exp(eta))
        psi = pm.Beta("psi", 1, 1)

        y_obs = pm.Poisson("y_obs", mu=mu, observed=obs_data)

        # meaningless here but left for consistency
        pm.Bernoulli("y_bernoulli", p=1 - psi, observed=(obs_data == 0) * 1.0) 
                                        
        postpred = pm.sample_posterior_predictive(idata)
    
    return postpred, fullmodel_extended, new_cv, new_subs


def power_analysis(chunk):
    print(f"running power analysis ztp models process {os.getpid()}")
    if len(chunk) == 0:
        return None
    
    S = chunk[0][-3]
    usemask = chunk[0][-2]
    mode = chunk[0][-1]
    samplesizes = mode['samplesizes']
    N_trials = mode['n_trials']
    n_subs = len(np.unique(S))

    trueresults = []
    simtrueresults = {i:[] for i in samplesizes}
    simnullresults = {i:[] for i in samplesizes}
    infos = {i:[] for i in samplesizes}

    for i, j, data, X, Z, nullstartidx, nullblocksize, S, usemask, mode in chunk:
        exposure = mode['exposure']
        samplemode = mode['samplemode']
        obs = data[usemask]
        cv = X[usemask]
        subs = S[usemask]
        design = Z[usemask]

        cv = cv[obs > 0, 1:]
        subs = subs[obs > 0]
        obs_nonzero = obs[obs > 0]

        # first fit on pilot data only
        # always use nuts sampler to generate first-pass at effects
        pilotmodeltrace, _, _, _ = fit_fullmodel(obs, obs_nonzero, cv, subs, 
                                                 exposure, 'nuts')

        for samplesize in samplesizes:
            postpred, generative_model, extended_cv, extended_subs = \
                generate_data(pilotmodeltrace, samplesize, 
                              N_trials, obs_nonzero, cv, subs, exposure)
            
            # artificial data containing weak beta fixed effect
            artificial_data = postpred.posterior_predictive['y_obs'].values\
                        .reshape(4 * 4000, -1) # (n_chain x n_samples, n_obs)

            # need to make sure we don't (likely) fit one model per chain within
            # each sample which would happen if we didn't shuffle the data (very
            # unlikely now)
            np.random.seed(0)
            np.random.shuffle(artificial_data) # (n_chain x 4000, n_obs)

            # clamp beta to 0 and generate another set of artificial data
            with pm.do(generative_model, {"beta": [0.0]}) as m_do:
                postpred_do = pm.sample_posterior_predictive(pilotmodeltrace, random_seed=0)

            null_data = postpred_do.posterior_predictive['y_obs'].values.reshape(4 * 4000, -1) # (n_chain x n_samples, n_obs * 3)

            # # need to make sure we don't (likely) fit one model per chain within each sample
            # # which would happen if we didn't shuffle the data (very unlikely now)
            np.random.seed(0)
            np.random.shuffle(null_data) # (n_chain x 4000, n_obs)

            # for this sample size we simulate N_sims fits on the aritificial
            # data set
            for simno in range(nullstartidx, nullstartidx+nullblocksize):
                print(f"chunk {i}->{j}: samplesize={samplesize}, simno={simno}, truemodel")
                sim_true_data = artificial_data[simno]
                sim_true_data_nonzero = sim_true_data[sim_true_data > 0]
                simtrace, simmodel, rhat, posterior_subsamples = \
                                    fit_fullmodel(sim_true_data, 
                                            sim_true_data_nonzero, 
                                            extended_cv[sim_true_data > 0], 
                                            extended_subs[sim_true_data > 0],
                                            exposure, samplemode)

                simtrueresults[samplesize].append([i, j, simno, rhat, posterior_subsamples])

                print(f"chunk {i}->{j}: samplesize={samplesize}, simno={simno}, nullmodel")
                sim_null_data = null_data[simno]
                sim_null_data_nonzero = sim_null_data[sim_null_data > 0]
                simtrace, simmodel, rhat, posterior_subsamples = \
                                    fit_fullmodel(sim_null_data, 
                                            sim_null_data_nonzero, 
                                            extended_cv[sim_null_data > 0], 
                                            extended_subs[sim_null_data > 0],
                                            exposure, samplemode)

                simnullresults[samplesize].append([i, j, simno, rhat, posterior_subsamples])

                infos[samplesize].append([i, j, simno, extended_cv, extended_subs, sim_true_data, sim_null_data])

    return simtrueresults, simnullresults, infos

# +
# def statvals(dfinput, interventions):
#     dfinput = dfinput.copy()
#     dfi_nonzero = dfinput[dfinput.connectivity > 0]
    
# #     inputmeans = {var:None for var in interventions}
# #     for var in interventions:
# #         inputmeans[var] = dfinput.groupby(var, observed=True)['connectivity'].mean()

# #     inputstds = {var:None for var in interventions}
# #     for var in interventions:
# #         inputstds[var] = dfinput.groupby(var, observed=True)['connectivity'].var(ddof=1)

#     inputsizes = {var:None for var in interventions}
#     for var in interventions:
#         inputsizes[var] = dfinput.groupby(var, observed=True).apply(lambda x: len(x), include_groups=False)
        
#     nzinputmeans = {var:None for var in interventions}
#     for var in interventions:
#         nzinputmeans[var] = dfi_nonzero.groupby(var, observed=True).apply(lambda x: np.log(x['connectivity']).mean(), 
#                                                                           include_groups=False)

#     nzinputstds = {var:None for var in interventions}
#     for var in interventions:
#         nzinputstds[var] = dfi_nonzero.groupby(var, observed=True).apply(lambda x: np.log(x['connectivity']).var(ddof=1), 
#                                                                          include_groups=False)

#     nzinputsizes = {var:None for var in interventions}
#     for var in interventions:
#         nzinputsizes[var] = dfi_nonzero.groupby(var, observed=True).apply(lambda x: len(x), include_groups=False)

#     return nzinputmeans, nzinputstds, nzinputsizes, inputsizes 

# def general_worker(chunk):
#     print(f"running general model process {os.getpid()}")
#     compiledir = f"/export/vrishab/compile/{os.getpid()}"
#     pathlib.Path(compiledir).mkdir(parents=True, exist_ok=True)

#     os.environ["PYTENSOR_FLAGS"] = f"compiledir={compiledir}"
    
#     trueresults = []
#     nullresults = []
#     for i, j, data_full, formula, priors, interventions, nullstartidx, nullblocksize, nullmethod in chunk:
#         assert isinstance(data_full, pd.DataFrame), "data object must be a pandas dataframe"
#         assert "connectivity" in data_full.columns, "data must have connectivity as a column name"
        
#         data = data_full[data_full.connectivity > 0]
#         data_zeros = data_full[data_full.connectivity == 0]
        
#         fullmodel_true = bmb.Model(formula, data, priors=priors, family='hurdle_lognormal')
#         fullmodel_true.build()
#         fullmodel_true = fullmodel_true.backend.model

#         with fullmodel_true:
#             tracefulltrue = pm.sample(4000, 
#                                 chains=4, 
#                                 return_inferencedata=True, 
#                                 target_accept=0.97, 
#                                 cores=1, 
#                                 idata_kwargs={"log_likelihood": True}, 
#                                 progressbar=False)

#         # posterior means for all model variables
#         posterior = tracefulltrue.posterior
#         nuts_means = {
#             var: posterior[var].mean(dim=("chain", "draw")).values
#             for var in posterior.data_vars
#         }
#         nuts_stds = {
#             var: posterior[var].std(dim=("chain", "draw")).values
#             for var in posterior.data_vars
#         }
        
#         rhat = az.rhat(tracefulltrue)
        
#         # we probably want to pass through the original data
#         # conditional on the intervention
#         trueresults.append([i, j, 0, rhat, nuts_means, nuts_stds, [statvals(data_full, interventions)]])

#         modelshape = list([f"{i}: {nuts_means[i].shape}" for i in nuts_means.keys()])
#         for iv in interventions:
#             assert iv in nuts_means, f"intervention must be on one of the model variables: {modelshape}"
#             assert np.array(interventions[iv]).shape == nuts_means[iv].shape, f"intervention '{iv}' of shape {np.array(interventions[iv]).shape} must match model internals {modelshape}"

#         # counterfactual model with intervention vars clamped to 0 
#         # but all other params as in full model
#         with pm.do(fullmodel_true, interventions) as m_do:
#             postpred_do = pm.sample_posterior_predictive(tracefulltrue, 
#                                                         random_seed=0)
            
#         artificial_data = postpred_do.posterior_predictive['connectivity']\
#                 .values.reshape(4 * 4000, -1) # (n_chain x n_samples, ...)
        
#         # sometimes posterior sampling creates spuriously-large values, which
#         # can throw off posterior model fits and particularly posterior means
#         # just clip these with a generous margin
#         ad_max = data.connectivity.max() * 2
#         artificial_data = np.clip(artificial_data, None, ad_max)

#         # need to make sure we don't (likely) fit one model per chain within each sample
#         # which would happen if we didn't shuffle the data (very unlikely now)
#         np.random.seed(0)
#         np.random.shuffle(artificial_data) # (n_chain x 4000, n_obs)        

#         initialization_means = copy.deepcopy(nuts_means)
#         # advi warm start initialization
#         for var in initialization_means:
#             if var in interventions:
#                 initialization_means[var] = interventions[var]

#         # second stage: fit null models to artificial data
#         for nmi in range(nullstartidx, nullstartidx+nullblocksize):
#             data_null = data.copy()
#             print(f"link {j}->{i} running artificial data null model {nmi}")
#             nullobs = artificial_data[nmi]
#             data_null['connectivity'] = nullobs
            
#             if nullmethod == 'posteriormeans':
#                 # since zeros aren't estimated by hurdle_lognormal (see pymc docs)
#                 # we manually re-add the zeros after sampling
#                 data_null = pd.concat([data_null, data_zeros], axis=0)
                
#                 nullresults.append([i, j, nmi, None, statvals(data_null, interventions)])         
#                 continue

#             nullmodel = bmb.Model(formula, data_null, priors=priors, family='hurdle_lognormal')
#             nullmodel.build()
#             nullmodel = nullmodel.backend.model

#             rhat = None # rhat not meaningful for advi
#             with nullmodel:
#                 if nullmethod == 'advi':
#                     approx = pm.fit(method='advi', progressbar=False, start=initialization_means, callbacks=[])
#                     tracefullnull = approx.sample(4000)
#                 elif nullmethod == 'nuts':
#                     tracefullnull = pm.sample(4000, 
#                                 chains=4, 
#                                 return_inferencedata=True, 
#                                 target_accept=0.97, 
#                                 cores=1, 
#                                 idata_kwargs={"log_likelihood": True}, 
#                                 progressbar=False)
                    
#                     rhat = az.rhat(tracefulltrue)
                    
#                 else:
#                     raise Exception(f"null model fit method {nullmethod} not supported")

#             posterior = tracefullnull.posterior
#             advi_means = {
#                 var: posterior[var].mean(dim=("chain", "draw")).values
#                 for var in posterior.data_vars
#             }
#             advi_stds = {
#                 var: posterior[var].std(dim=("chain", "draw")).values
#                 for var in posterior.data_vars
#             }
            
#             nullresults.append([i, j, nmi, rhat, advi_means, advi_stds])     

#     return trueresults, nullresults

# +
# def statvals(dfinput, interventions):
#     dfinput = dfinput.copy()
#     dfi_nonzero = dfinput[dfinput.connectivity > 0]

#     inputsizes = {var:None for var in interventions}
#     for var in interventions:
#         inputsizes[var] = dfinput.groupby(var, observed=True).apply(lambda x: len(x), include_groups=False)
        
#     nzinputmeans = {var:None for var in interventions}
#     for var in interventions:
#         nzinputmeans[var] = dfi_nonzero.groupby(var, observed=True).apply(lambda x: np.log(x['connectivity']).mean(), 
#                                                                           include_groups=False)

#     nzinputstds = {var:None for var in interventions}
#     for var in interventions:
#         nzinputstds[var] = dfi_nonzero.groupby(var, observed=True).apply(lambda x: np.log(x['connectivity']).var(ddof=1), 
#                                                                          include_groups=False)

#     nzinputsizes = {var:None for var in interventions}
#     for var in interventions:
#         nzinputsizes[var] = dfi_nonzero.groupby(var, observed=True).apply(lambda x: len(x), include_groups=False)

#     return nzinputmeans, nzinputstds, nzinputsizes, inputsizes 

# def general_worker(chunk):
#     print(f"running general model process {os.getpid()}")
#     compiledir = f"/export/vrishab/compile/{os.getpid()}"
#     pathlib.Path(compiledir).mkdir(parents=True, exist_ok=True)

#     os.environ["PYTENSOR_FLAGS"] = f"compiledir={compiledir}"
    
#     trueresults = []
#     nullresults = []
#     for i, j, data_full, formula, priors, interventions, nullstartidx, nullblocksize, nullmethod in chunk:
#         assert isinstance(data_full, pd.DataFrame), "data object must be a pandas dataframe"
#         assert "connectivity" in data_full.columns, "data must have connectivity as a column name"
#         assert 'zero' in formula and 'nonzero' in formula, "zero and nonzero model formulas must be provided"
#         assert 'zero' in priors and 'nonzero' in priors, "zero and nonzero model priors must be provided (can be None)"
        
#         data = data_full[data_full.connectivity > 0]
#         data_binary = data_full.copy()
#         data_binary['connectivity'] = data_binary.connectivity == 0
        
#          # fit nonzero terms; hurdle doesn't work in bambi
#         fullmodel_true = bmb.Model(formula['nonzero'], data, priors=priors['nonzero'], family='hurdle_lognormal')
#         fullmodel_true.build()
#         fullmodel_true = fullmodel_true.backend.model
        
#         fullmodel_true_bool = bmb.Model(formula['zero'], data_binary, priors=priors['zero'], family='bernoulli')
#         fullmodel_true_bool.build()
#         fullmodel_true_bool = fullmodel_true_bool.backend.model
        
#         try:
#             with fullmodel_true:
#                 tracefulltrue = pm.sample(4000, 
#                                     chains=4, 
#                                     return_inferencedata=True, 
#                                     target_accept=0.97, 
#                                     cores=1, 
#                                     idata_kwargs={"log_likelihood": True}, 
#                                     progressbar=False)

#             with fullmodel_true_bool:
#                 tracefulltruebool = pm.sample(4000, 
#                                     chains=4, 
#                                     return_inferencedata=True, 
#                                     target_accept=0.97, 
#                                     cores=1, 
#                                     idata_kwargs={"log_likelihood": True}, 
#                                     progressbar=False)
            
#         except ValueError as ex:
#             print(f"worker received valueerror for cell ({i}, {j}): '{ex}'")
#             print("This is likely because one or more grouping variables have no data")
#             continue
            
#         posterior = tracefulltrue.posterior
#         posteriorbool = tracefulltruebool.posterior
#         rhatnz = az.rhat(tracefulltrue)
#         rhatbool = az.rhat(tracefulltruebool)
        
#         # pass through the original data conditional on the intervention
#         nzinputmeans, nzinputstds, nzinputsizes, inputsizes = statvals(data_full, interventions)
#         trueresults.append([i, j, 0, [rhatnz, rhatbool], nzinputmeans, nzinputstds, nzinputsizes, inputsizes])

#         # counterfactual model with intervention vars clamped to 0 
#         # but all other params as in full model
#         with pm.do(fullmodel_true, interventions) as m_do:
#             postpred_do = pm.sample_posterior_predictive(tracefulltrue, 
#                                                         random_seed=0)
        
#         # counterfactual for bool model
#         with pm.do(fullmodel_true_bool, interventions) as m_do:
#             postpred_bool_do = pm.sample_posterior_predictive(tracefulltruebool, 
#                                                         random_seed=0)
            
#         artificial_data = postpred_do.posterior_predictive['connectivity']\
#                 .values.reshape(4 * 4000, -1) # (n_chain x n_samples, ...)
            
#         artificial_data_bool = postpred_bool_do.posterior_predictive['connectivity']\
#                 .values.reshape(4 * 4000, -1) # (n_chain x n_samples, ...)
        
#         # sometimes posterior sampling creates spuriously-large values, which
#         # can throw off posterior model fits and particularly posterior means
#         # just clip these with a generous margin
#         ad_max = data.connectivity.max() * 2
#         artificial_data = np.clip(artificial_data, None, ad_max)

#         # need to make sure we don't (likely) fit one model per chain within each sample
#         # which would happen if we didn't shuffle the data (very unlikely now)
#         np.random.seed(0)
#         np.random.shuffle(artificial_data) # (n_chain x 4000, n_obs)        
        
#         np.random.seed(0)
#         np.random.shuffle(artificial_data_bool) # (n_chain x 4000, n_obs)        

#         # second stage: null models from artificial data
#         for nmi in range(nullstartidx, nullstartidx+nullblocksize):
#             print(f"link {j}->{i} running artificial data null model {nmi}")
                
#             data_null = data.copy()
#             data_null_bool = data_binary.copy()
            
#             nullobs = artificial_data[nmi]
#             nullboolobs = artificial_data_bool[nmi]
            
#             data_null['connectivity'] = nullobs
#             data_null_bool['connectivity'] = nullboolobs
            
#             if nullmethod == 'posteriormeans':
#                 # estimate null model nonzero means and stds from nonzero data
#                 nzinputmeans, nzinputstds, _, _ = statvals(data_null, interventions)
#                 # estimate nonzero and zero null values from bernoulli model
#                 _, _, nzinputsizes, inputsizes = statvals(data_null_bool, interventions)
                
#                 nullresults.append([i, j, nmi, None, nzinputmeans, nzinputstds, nzinputsizes, inputsizes])         
#                 continue
   

#     return trueresults, nullresults
# -
def statvals(dfinput, interventions, posteriordata):        
    groups = dfinput[list(interventions.keys())[0]].unique()
    
    diff_means = {k:None for k in groups}
    for group in groups:
        idxs = (dfinput[list(interventions.keys())[0]] == group).values
        diff_means[group] = posteriordata[:, idxs].mean()
        
    diff_vars = {k:None for k in groups}
    for group in groups:
        idxs = (dfinput[list(interventions.keys())[0]] == group).values
        diff_vars[group] = posteriordata[:, idxs].var(ddof=1)

    
    return diff_means, diff_vars


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
        assert len(interventions) == 1, 'only single variable (length 1) interventions are supported'
        
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






def simple_worker_randint(chunk):   
    print(f"running integrated, random intercept ztp models process {os.getpid()}")
    if len(chunk) == 0:
        return None
    
    S = chunk[0][-3]
    usemask = chunk[0][-2]
    mode = chunk[0][-1]
    
    n_subs = len(np.unique(S))
    trueresults = []
    nullresults = []
    for i, j, data, X, Z, nullstartidx, nullblocksize, S, usemask, mode in chunk:
        nullmode = mode["nullmode"]
        nullppc = mode["nullppc"]
        exposure = mode["exposure"]
        modeltype = mode["modeltype"]
        clamp_idx = mode["clamp_indices"]

        obs = data[usemask]
        cv = X[usemask]
        subs = S[usemask]
        design = Z[usemask]
        exposure = exposure[usemask]

        with pm.Model() as fullmodel_true:
            # fixed effects parameters
            if modeltype != 'lognormal': # different parameterization
                intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean
                beta = pm.Normal('beta', 0, 5, shape=(cv.shape[1] - 1))

                # random effects (1|subject)
                sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
                subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
                subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

                # linear predictor
                # beta_intercept + ind(beta_0dB) + ind(beta_-6dB) + re_intercept + ind(re_0dB) + ind(re_-6dB)
                eta_fixed = (cv[obs > 0, 1:] @ beta)   # fixed effects
                eta = intercept + eta_fixed + exposure[obs > 0] + subject_effects[subs[obs > 0]]

                mu  = pm.Deterministic('mu', pm.math.exp(eta))
                psi = pm.Beta("psi", 1, 1)
            else:
                intercept = pm.Normal("intercept", mu=0, sigma=5)                 # population baseline log-mean
                beta = pm.Normal('beta', 0, 2, shape=(cv.shape[1] - 1))
                sigma = pm.Exponential("sigma", 1)

                # random effects (1|subject)
                sigma_sub = pm.Exponential("subject_re_sigma", 1)
                subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
                subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

                # linear predictor
                # beta_intercept + ind(beta_0dB) + ind(beta_-6dB) + re_intercept + ind(re_0dB) + ind(re_-6dB)
                eta_fixed = (cv[obs > 0, 1:] @ beta)   # fixed effects
                eta = intercept + eta_fixed + exposure[obs > 0] + subject_effects[subs[obs > 0]]

                mu  = pm.Deterministic('mu', eta)
                psi = pm.Beta("psi", 1, 1)

            if modeltype == 'gamma':
                sigma = pm.HalfNormal("sigma", sigma=1)
                y_obs = pm.Gamma("y_obs", mu=mu, sigma=sigma, observed=obs[obs > 0])
            elif modeltype == 'poisson':
                y_obs = pm.Poisson("y_obs", mu=mu, observed=obs[obs > 0])
            elif modeltype == 'genpois':
                lower = pm.math.maximum(-1.0, -mu / 4.0)
                # fraction between lower and 1
                lam_rel = pm.Beta("lam_rel", 2, 2)
                lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
                y_obs = pmx.distributions.GeneralizedPoisson(mu=mu, lam=lam, observed=obs[obs>0])
            elif modeltype == 'lognormal':
                y_obs = pm.LogNormal("y_obs", mu=mu, sigma=sigma, observed=obs[obs>0])

            pm.Bernoulli("y_bernoulli", p=1 - psi, observed=(obs == 0) * 1.0)

            # see if sampler converges with obtaining invalid values
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)

                try:
                    tracefulltrue = pm.sample(4000, chains=4, return_inferencedata=True, 
                            target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                            progressbar=False)
                    badconvflag = False

                except RuntimeWarning as e:
                    print(f"true model {i}->{j} caught a RuntimeWarning exception:", e)
                    badconvflag = True

            # here the sampler will raise a runtime warning but will still proceed with sampling
            # this way we will still obtain estimates, just with the knowledge that something
            # didn't go right
            if badconvflag:
                tracefulltrue = pm.sample(4000, chains=4, return_inferencedata=True, 
                            target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                            progressbar=False)

            post = tracefulltrue.posterior
            if nullstartidx == 0: # only do the first one to save mem
                postpred = pm.sample_posterior_predictive(tracefulltrue, random_seed=0)
            else:
                postpred = None
                
            _intercept = post['intercept'].mean(dim=('chain', 'draw')).values
            _mu = post['mu'].mean(dim=('chain', 'draw')).values
            _psi = post['psi'].mean().item()
            posterior_subsamples = {
                "intercept": _intercept, 
                "beta": post['beta'].mean(dim=('chain', 'draw')).values,
                "RE": post["RE"].mean(dim=("chain", "draw")).values,
                "mu": _mu,
                "psi": _psi,
                "exposure": exposure[obs > 0],
                "obs": obs,
                "predicted_samples": postpred,
                "badconv": badconvflag,
                "clamp": clamp_idx,
            }
            rhat = None # az.rhat(tracefulltrue)
            trueresults.append([i, j, 0, rhat, posterior_subsamples])
        
        # clamp relevant indices of symbolic beta vector to 0
        beta = fullmodel_true["beta"]
        mask_t = pt.constant(clamp_idx)
        beta_do = beta * mask_t

        # counterfactual model with beta forced to 0 but all other params as in full model
        with pm.do(fullmodel_true, {beta: beta_do}) as m_do:
            postpred_do = pm.sample_posterior_predictive(tracefulltrue, random_seed=0)
            
        artificial_data = postpred_do.posterior_predictive['y_obs'].values.reshape(4 * 4000, -1) # (n_chain x n_samples, n_obs * 3)
        
        # need to make sure we don't (likely) fit one model per chain within each sample
        # which would happen if we didn't shuffle the data (very unlikely now)
        np.random.seed(0)
        np.random.shuffle(artificial_data) # (n_chain x 4000, n_obs)

        mcmc_means = {var: tracefulltrue.posterior[var].mean(dim=("chain", "draw")).values 
                    for var in tracefulltrue.posterior.data_vars}
        mcmc_means['beta'] = mcmc_means['beta'] * mask_t # do operator set this to 0
        
        # second stage: fit null models to artificial data
        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            print(f"link {j}->{i} running artificial data null model {nmi}")
            
            nullobs = artificial_data[nmi]
            cvnull = cv[obs > 0] # the artificial data are derived from only nonzero observations
            subsnull = subs[obs > 0]
            exposurenull = exposure[obs > 0]

            print(f"link {j}->{i} running null model {nmi}")

            # everything else exactly the same as true full model
            with pm.Model() as nullmodel:
                # fixed effects parameters
                if modeltype != 'lognormal':
                    intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean
                    beta = pm.Normal('beta', 0, 5, shape=(cvnull.shape[1]-1))

                    # random effects (1|subject)
                    sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
                    subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
                    subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

                    # linear predictor
                    # beta_intercept + ind(beta_0dB) + ind(beta_-6dB) + re_intercept + ind(re_0dB) + ind(re_-6dB)
                    eta_fixed = (cvnull[nullobs > 0, 1:] @ beta)   # fixed effects
                    eta = intercept + eta_fixed + exposurenull[nullobs > 0] + subject_effects[subsnull[nullobs > 0]]

                    mu  = pm.Deterministic('mu', pm.math.exp(eta))
                    psi = pm.Beta("psi", 1, 1)

                else:
                    intercept = pm.Normal("intercept", mu=0, sigma=5)                 # population baseline log-mean
                    beta = pm.Normal('beta', 0, 2, shape=(cvnull.shape[1]-1))
                    sigma = pm.Exponential("sigma", 1)

                    # random effects (1|subject)
                    sigma_sub = pm.Exponential("subject_re_sigma", 1)
                    subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
                    subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

                    # linear predictor
                    # beta_intercept + ind(beta_0dB) + ind(beta_-6dB) + re_intercept + ind(re_0dB) + ind(re_-6dB)
                    eta_fixed = (cvnull[nullobs > 0, 1:] @ beta)   # fixed effects
                    eta = intercept + eta_fixed + exposurenull[nullobs > 0] + subject_effects[subsnull[nullobs > 0]]

                    mu  = pm.Deterministic('mu', eta)
                    psi = pm.Beta("psi", 1, 1)

                if modeltype == 'gamma':
                    sigma = pm.HalfNormal("sigma", sigma=1)
                    y_obs = pm.Gamma("y_obs", mu=mu, sigma=sigma, observed=nullobs[nullobs > 0])
                elif modeltype == 'poisson':
                    y_obs = pm.Poisson("y_obs", mu=mu, observed=nullobs[nullobs > 0])
                elif modeltype == 'genpois':
                    lower = pm.math.maximum(-1.0, -mu / 4.0)
                    # fraction between lower and 1
                    lam_rel = pm.Beta("lam_rel", 2, 2)
                    lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
                    y_obs = pmx.distributions.GeneralizedPoisson(mu=mu, lam=lam, observed=nullobs[nullobs > 0])
                elif modeltype == 'lognormal':
                    y_obs = pm.LogNormal("y_obs", mu=mu, sigma=sigma, observed=nullobs[nullobs>0])


                pm.Bernoulli("y_bernoulli", p=1 - psi, observed=(nullobs == 0) * 1.0)

                # see if sampler converges with obtaining invalid values
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)

                    try:
                        if nullmode == 'advi':
                            approx = pm.fit(method='advi', progressbar=False, start=mcmc_means, callbacks=[])
                            tracefullnull = approx.sample(4000)
                        else:
                            tracefullnull = pm.sample(4000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)
                        badconvflag = False

                    except RuntimeWarning as e:
                        print(f"true model {i}->{j} caught a RuntimeWarning exception:", e)
                        badconvflag = True

                # here the sampler will raise a runtime warning but will still proceed with sampling
                # this way we will still obtain estimates, just with the knowledge that something
                # didn't go right
                if badconvflag:
                    if nullmode == 'advi':
                            approx = pm.fit(method='advi', progressbar=False, start=mcmc_means, callbacks=[])
                            tracefullnull = approx.sample(4000)
                    else:
                        tracefullnull = pm.sample(4000, chains=4, return_inferencedata=True, 
                            target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                            progressbar=False)

                post = tracefullnull.posterior
                _intercept = post['intercept'].mean(dim=('chain', 'draw')).values
                _mu = post['mu'].mean(dim=('chain', 'draw')).values
                _psi = post['psi'].mean().item()
                posterior_subsamples = {
                    "intercept": _intercept, 
                    "beta": post['beta'].mean(dim=('chain', 'draw')).values,
                    "RE": post["RE"].mean(dim=("chain", "draw")).values,
                    "mu": _mu,
                    "psi": _psi,
                    "exposure": exposurenull[nullobs > 0],
                    "obs": nullobs,
                    "badconv": badconvflag,
                    "clamp": clamp_idx,
                }
                rhat = None # az.rhat(tracefulltrue)
                nullresults.append([i, j, nmi, rhat, posterior_subsamples])     

    return trueresults, nullresults


def simple_worker(chunk):   
    print(f"running integrated, covariate only ztp models process {os.getpid()}")
    if len(chunk) == 0:
        return None
    
    S = chunk[0][-3]
    usemask = chunk[0][-2]
    mode = chunk[0][-1]
    
    n_subs = len(np.unique(S))
    trueresults = []
    nullresults = []
    for i, j, data, X, Z, nullstartidx, nullblocksize, S, usemask, mode in chunk:
        nullmode = mode["nullmode"]
        nullppc = mode["nullppc"]
        exposure = mode["exposure"]
        modeltype = mode["modeltype"]

        obs = data[usemask]
        cv = X[usemask]
        subs = S[usemask]
        design = Z[usemask]
        exposure = exposure[usemask]

        with pm.Model() as fullmodel_true:
            # fixed effects parameters
            intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean
            beta = pm.Normal('beta', 0, 5, shape=(1))

            # linear predictor
            eta_fixed = (cv[obs > 0, 1:] @ beta)   # fixed effects
            eta = intercept + eta_fixed + exposure[obs > 0]

            mu  = pm.Deterministic('mu', pm.math.exp(eta))
            psi = pm.Beta("psi", 1, 1)

            if modeltype == 'gamma':
                sigma = pm.HalfNormal("sigma", sigma=1)
                y_obs = pm.Gamma("y_obs", mu=mu, sigma=sigma, observed=obs[obs > 0])
            elif modeltype == 'poisson':
                y_obs = pm.Poisson("y_obs", mu=mu, observed=obs[obs > 0])
            elif modeltype == 'genpois':
                lower = pm.math.maximum(-1.0, -mu / 4.0)
                # fraction between lower and 1
                lam_rel = pm.Beta("lam_rel", 2, 2)
                lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
                y_obs = pmx.distributions.GeneralizedPoisson(mu=mu, lam=lam, observed=obs[obs>0])

            pm.Bernoulli("y_bernoulli", p=1 - psi, observed=(obs == 0) * 1.0)

            # see if sampler converges with obtaining invalid values
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)

                try:
                    tracefulltrue = pm.sample(4000, chains=4, return_inferencedata=True, 
                            target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                            progressbar=False)
                    badconvflag = False

                except RuntimeWarning as e:
                    print(f"true model {i}->{j} caught a RuntimeWarning exception:", e)
                    badconvflag = True

            # here the sampler will raise a runtime warning but will still proceed with sampling
            # this way we will still obtain estimates, just with the knowledge that something
            # didn't go right
            if badconvflag:
                tracefulltrue = pm.sample(4000, chains=4, return_inferencedata=True, 
                            target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                            progressbar=False)

            post = tracefulltrue.posterior
            if nullstartidx == 0: # only do the first one to save mem
                postpred = pm.sample_posterior_predictive(tracefulltrue, random_seed=0)
            else:
                postpred = None
                
            _intercept = post['intercept'].mean(dim=('chain', 'draw')).values
            _mu = post['mu'].mean(dim=('chain', 'draw')).values
            _psi = post['psi'].mean().item()
            posterior_subsamples = {
                "intercept": _intercept, 
                "beta": post['beta'].mean(dim=('chain', 'draw')).values,
                "mu": _mu,
                "psi": _psi,
                "exposure": exposure,
                "obs": obs,
                "predicted_samples": postpred,
                "badconv": badconvflag,
            }
            rhat = None # az.rhat(tracefulltrue)
            trueresults.append([i, j, 0, rhat, posterior_subsamples])
        
        # counterfactual model with beta forced to 0 but all other params as in full model
        with pm.do(fullmodel_true, {beta: [0.0]}) as m_do:
            postpred_do = pm.sample_posterior_predictive(tracefulltrue, random_seed=0)
            
        artificial_data = postpred_do.posterior_predictive['y_obs'].values.reshape(4 * 4000, -1) # (n_chain x n_samples, n_obs * 3)
        
        # need to make sure we don't (likely) fit one model per chain within each sample
        # which would happen if we didn't shuffle the data (very unlikely now)
        np.random.seed(0)
        np.random.shuffle(artificial_data) # (n_chain x 4000, n_obs)
        
        # second stage: fit null models to artificial data
        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            print(f"link {j}->{i} running artificial data null model {nmi}")
            
            nullobs = artificial_data[nmi]
            cvnull = cv[obs > 0] # the artificial data are derived from only nonzero observations
            subsnull = subs[obs > 0]
            exposurenull = exposure[obs > 0]

            print(f"link {j}->{i} running null model {nmi}")

            # everything else exactly the same as true full model
            with pm.Model() as nullmodel:
                # fixed effects parameters
                intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean
                beta = pm.Normal('beta', 0, 5, shape=(1))

                # linear predictor
                eta_fixed = (cvnull[nullobs > 0, 1:] @ beta)   # fixed effects
                eta = intercept + eta_fixed + exposurenull[nullobs > 0] 

                mu  = pm.Deterministic('mu', pm.math.exp(eta))
                psi = pm.Beta("psi", 1, 1)

                if modeltype == 'gamma':
                    sigma = pm.HalfNormal("sigma", sigma=1)
                    y_obs = pm.Gamma("y_obs", mu=mu, sigma=sigma, observed=nullobs[nullobs > 0])
                elif modeltype == 'poisson':
                    y_obs = pm.Poisson("y_obs", mu=mu, observed=nullobs[nullobs > 0])
                elif modeltype == 'genpois':
                    lower = pm.math.maximum(-1.0, -mu / 4.0)
                    # fraction between lower and 1
                    lam_rel = pm.Beta("lam_rel", 2, 2)
                    lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
                    y_obs = pmx.distributions.GeneralizedPoisson(mu=mu, lam=lam, observed=nullobs[nullobs > 0])

                pm.Bernoulli("y_bernoulli", p=1 - psi, observed=(nullobs == 0) * 1.0)

                # see if sampler converges with obtaining invalid values
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)

                    try:
                        if nullmode == 'advi':
                            approx = pm.fit(method='advi', progressbar=False, callbacks=[])
                            tracefullnull = approx.sample(4000)
                        else:
                            tracefullnull = pm.sample(4000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)
                        badconvflag = False

                    except RuntimeWarning as e:
                        print(f"true model {i}->{j} caught a RuntimeWarning exception:", e)
                        badconvflag = True

                # here the sampler will raise a runtime warning but will still proceed with sampling
                # this way we will still obtain estimates, just with the knowledge that something
                # didn't go right
                if badconvflag:
                    if nullmode == 'advi':
                            approx = pm.fit(method='advi', progressbar=False, callbacks=[])
                            tracefullnull = approx.sample(4000)
                    else:
                        tracefullnull = pm.sample(4000, chains=4, return_inferencedata=True, 
                            target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                            progressbar=False)

                post = tracefullnull.posterior
                _intercept = post['intercept'].mean(dim=('chain', 'draw')).values
                _mu = post['mu'].mean(dim=('chain', 'draw')).values
                _psi = post['psi'].mean().item()
                posterior_subsamples = {
                    "intercept": _intercept, 
                    "beta": post['beta'].mean(dim=('chain', 'draw')).values,
                    "mu": _mu,
                    "psi": _psi,
                    "exposure": exposurenull[nullobs > 0],
                    "obs": obs,
                    "badconv": badconvflag,
                }
                rhat = None # az.rhat(tracefulltrue)
                nullresults.append([i, j, nmi, rhat, posterior_subsamples])     

    return trueresults, nullresults


def counterfactual_run_worker(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(simple_worker, chunks)) # each worker gets one chunk at a time

    print("transmit")
    subprocess.call(["rm", "-rf", "/export/vrishab/compile"])
    print("clear cache")
    return res

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


def counterfactual_run_worker_randint(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(simple_worker_randint, chunks)) # each worker gets one chunk at a time

    print("transmit")
    subprocess.call(["rm", "-rf", "/export/vrishab/compile"])
    print("clear cache")
    return res


def counterfactual_power_analysis(chunks):
    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(power_analysis, chunks)) # each worker gets one chunk at a time

    print("transmit")
    subprocess.call(["rm", "-rf", "/export/vrishab/compile"])
    print("clear cache")
    return res
