import numpy as np
import pymc as pm
import arviz as az
import multiprocessing
from multiprocessing import Pool
import pytensor
import os
import warnings
import pymc_extras as pmx 


def make_model(subs, usemask):
    # need to have a random intercept for all subjects, even if some have no 
    # usable data (filtered) out by usemask
    n_subs = len(np.unique(subs)) 
    subs = subs[usemask]
    datavec = np.zeros(usemask.sum())
    y_shared = pytensor.shared(datavec) # initialized value, not compiled
    covar_shared = pytensor.shared(datavec)
    covar_null_shared = pytensor.shared(datavec)

    with pm.Model(check_bounds=True) as fullmodel:
        # model random effects priors
        sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
        subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
        subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)

        # model fixed effects priors
        beta = pm.Normal('covar', mu=0, sigma=1)#, initval=0)
        intercept = pm.Normal('intercept', mu=0, sigma=10)#, initval=0)

        p = pm.Deterministic("p", pm.math.invlogit(intercept + pm.math.dot(beta, covar_shared) + subject_effects[subs]))

        # observations
        observed = pm.Bernoulli("bernoulli_obs", p, observed=y_shared)
        
    with pm.Model(check_bounds=True) as nullmodel:
        # model random effects priors
        sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
        subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
        subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)

        # model fixed effects priors
        beta = pm.Normal('covar', mu=0, sigma=1)#, initval=0)
        intercept = pm.Normal('intercept', mu=0, sigma=10)#, initval=0)

        p = pm.Deterministic("p", pm.math.invlogit(intercept + pm.math.dot(beta, covar_null_shared) + subject_effects[subs]))

        # observations
        observed = pm.Bernoulli("bernoulli_obs", p, observed=y_shared)
    return fullmodel, nullmodel, covar_shared, covar_null_shared, y_shared

def worker(chunk):
    print(f"running full model process {os.getpid()}")
    if len(chunk) == 0:
        return None
    
    subs = chunk[0][-3]
    usemask = chunk[0][-2]
    posteriormode = chunk[0][-1]
    if posteriormode not in ["fullposterior", 'posteriormeans']:
        raise Exception(f"posterior mode {posteriormode} not supported")
    
    n_subs = len(np.unique(subs))
    fullmodel, nullmodel, xs, xns, ys = make_model(subs, usemask) 
    trueresults = []
    nullresults = []
    for i, j, data, covar, n_nulls, subs, usemask, posteriormode in chunk:
        ys.set_value(data[usemask])
        xs.set_value(covar[usemask])

        # fit true model
        with fullmodel:
            approx = pm.fit(method='advi', progressbar=False, callbacks=[])
            trace_full = approx.sample(1000)
            trueresults.append([i, j, trace_full.posterior['covar'].values, 
                                trace_full.posterior['intercept'].values,
                                trace_full.posterior['subject_effects'].values])
            

        # fit null models
        for nmi in range(n_nulls):
            print(f"link {j}->{i} running null model {nmi}")
            covar_null = covar.copy().reshape(n_subs, -1)
            np.random.seed(nmi)
            np.random.shuffle(covar_null)
            covar_null = covar_null.flatten()
            covar_null = covar_null[usemask]
            xns.set_value(covar_null)
            with nullmodel:
                approx = pm.fit(method='advi', progressbar=False, callbacks=[])
                trace_null = approx.sample(1000)
                if posteriormode == 'fullposterior':
                    nullresults.append([i, j, trace_null.posterior['covar'].values])
                else:
                    nullresults.append([i, j, trace_null.posterior['covar'].values.mean()])
        
    return trueresults, nullresults


def simple_worker(chunk):
    print(f"running full model process {os.getpid()}")
    if len(chunk) == 0:
        return None
    
    subs = chunk[0][-3]
    usemask = chunk[0][-2]
    posteriormode = chunk[0][-1]
    if posteriormode not in ["fullposterior", 'posteriormeans', 'trace']:
        raise Exception(f"posterior mode {posteriormode} not supported")
    
    n_subs = len(np.unique(subs))
    trueresults = []
    nullresults = []
    for i, j, data, covar, n_nulls, subs, usemask, posteriormode in chunk:
        obs = data[usemask]
        cv = covar[usemask]

        # fit true model
        with pm.Model() as fullmodel:
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
            
            # model fixed effects priors
            beta = pm.Normal('covar', mu=0, sigma=1)
            intercept = pm.Normal('intercept', mu=0, sigma=10)

            p = pm.Deterministic("p", pm.math.invlogit(intercept + pm.math.dot(beta, cv) + subject_effects[subs[usemask]]))
            # observations
            observed = pm.Bernoulli("bernoulli_obs", p, observed=obs)
            approx = pm.fit(method='advi', progressbar=False, callbacks=[])
            trace_full = approx.sample(1000)
            trueresults.append([i, j, trace_full])
            

        # fit null models
        for nmi in range(n_nulls):
            print(f"link {j}->{i} running null model {nmi}")
            covar_null = covar.copy().reshape(n_subs, -1)
            np.random.seed(nmi)
            np.random.shuffle(covar_null)
            covar_null = covar_null.flatten()
            cvn = covar_null[usemask]
            
            with pm.Model() as nullmodel:
                # model fixed effects priors
                beta = pm.Normal('covar', mu=0, sigma=1)
                intercept = pm.Normal('intercept', mu=0, sigma=10)

                p = pm.Deterministic("p", pm.math.invlogit(intercept + pm.math.dot(beta, cvn) + subject_effects[subs[usemask]]))
                observed = pm.Bernoulli("bernoulli_obs", p, observed=obs)

                approx = pm.fit(method='advi', progressbar=False, callbacks=[])
                trace_null = approx.sample(1000)
                if posteriormode == 'fullposterior':
                    nullresults.append([i, j, trace_null.posterior['covar'].values])
                else:
                    # mean of every trace
                    nullresults.append([i, j, trace_null.posterior['covar'].values.mean()])
        
    return trueresults, nullresults


def simple_worker_negbin(chunk):
    print(f"running full negbin model process {os.getpid()}")
    if len(chunk) == 0:
        return None
    
    subs = chunk[0][-3]
    usemask = chunk[0][-2]
    posteriormode = chunk[0][-1]
    if posteriormode not in ["fullposterior", 'posteriormeans', 'trace']:
        raise Exception(f"posterior mode {posteriormode} not supported")
    
    n_subs = len(np.unique(subs))
    trueresults = []
    nullresults = []
    for i, j, data, covar, nullstartidx, nullblocksize, subs, usemask, posteriormode in chunk:
        obs = data[usemask]
        cv = covar[usemask]

        # fit true model
        with pm.Model() as truemodel:
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
            
            # Priors for intercept and slope
            alpha = pm.Normal("alpha", mu=0, sigma=5)
            beta = pm.Normal("covar", mu=0, sigma=5)
            
            # Linear predictor
            mu = pm.math.exp(alpha + beta * cv + subject_effects[subs[usemask]])
            
            # Likelihood
            alpha_disp = pm.HalfNormal("alpha_disp", sigma=5)
            y_obs = pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha_disp, observed=obs)

            # Sampling
            tracetrue = pm.sample(1000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)
            rhat = az.rhat(tracetrue)['covar'].values.flatten()[0]
            trueresults.append([i, j, 0, rhat, tracetrue])

        # fit null models
        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            print(f"link {j}->{i} running null model {nmi}")
            covar_null = covar.copy().reshape(n_subs, -1)
            np.random.seed(nmi)
            np.random.shuffle(covar_null)
            covar_null = covar_null.flatten()
            cvn = covar_null[usemask]
            
            with pm.Model() as nullmodel:
                sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
                subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
                subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
                
                # Priors for intercept and slope
                alpha = pm.Normal("alpha", mu=0, sigma=5)
                beta = pm.Normal("covar", mu=0, sigma=5)
                
                # Linear predictor
                mu = pm.math.exp(alpha + beta * cvn + subject_effects[subs[usemask]])

                # Likelihood
                alpha_disp = pm.HalfNormal("alpha_disp", sigma=5)
                y_obs = pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha_disp, observed=obs)

                # Sampling
                tracenull = pm.sample(1000, chains=4, return_inferencedata=True, 
                                    target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                    progressbar=False)
                rhat = az.rhat(tracenull)['covar'].values.flatten()[0]

                if posteriormode == 'fullposterior':
                    nullresults.append([i, j, nmi, rhat, tracenull.posterior['covar'].values])
                elif posteriormode == 'trace':
                    nullresults.append([i, j, nmi, rhat, tracenull])
                else:
                    # mean of every trace
                    nullresults.append([i, j, nmi, rhat, tracenull.posterior['covar'].values.mean(axis=1)])
        
    return trueresults, nullresults

def simple_worker_zip(chunk):
    print(f"running full zip model process {os.getpid()}")
    if len(chunk) == 0:
        return None
    
    subs = chunk[0][-3]
    usemask = chunk[0][-2]
    posteriormode = chunk[0][-1]
    if posteriormode not in ["fullposterior", 'posteriormeans', 'trace']:
        raise Exception(f"posterior mode {posteriormode} not supported")
    
    n_subs = len(np.unique(subs))
    trueresults = []
    nullresults = []
    for i, j, data, covar, nullstartidx, nullblocksize, subs, usemask, posteriormode in chunk:
        obs = data[usemask]
        cv = covar[usemask]

        # fit true model
        with pm.Model() as truemodel:
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
            
            # Priors for intercept and slope
            alpha = pm.Normal("alpha", mu=0, sigma=5)
            beta = pm.Normal("covar", mu=0, sigma=5)
            
            # Linear predictor
            mu = pm.math.exp(alpha + beta * cv + subject_effects[subs[usemask]])
            
            # Likelihood
            psi = pm.Beta("psi", 1, 1)
            y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=psi, observed=obs)

            # Sampling
            tracetrue = pm.sample(1000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)
            rhat = az.rhat(tracetrue)['covar'].values.flatten()[0]
            trueresults.append([i, j, 0, rhat, tracetrue])

        # fit null models
        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            print(f"link {j}->{i} running null model {nmi}")
            covar_null = covar.copy().reshape(n_subs, -1)
            np.random.seed(nmi)
            np.random.shuffle(covar_null)
            covar_null = covar_null.flatten()
            cvn = covar_null[usemask]
            
            with pm.Model() as nullmodel:
                sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
                subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
                subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
                
                # Priors for intercept and slope
                alpha = pm.Normal("alpha", mu=0, sigma=5)
                beta = pm.Normal("covar", mu=0, sigma=5)
                
                # Linear predictor
                mu = pm.math.exp(alpha + beta * cvn + subject_effects[subs[usemask]])

                # Likelihood
                psi = pm.Beta("psi", 1, 1)
                y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=psi, observed=obs)

                # Sampling
                tracenull = pm.sample(1000, chains=4, return_inferencedata=True, 
                                    target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                    progressbar=False)
                rhat = az.rhat(tracenull)['covar'].values.flatten()[0]
                if posteriormode == 'fullposterior':
                    nullresults.append([i, j, nmi, rhat, tracenull.posterior['covar'].values])
                elif posteriormode == 'trace':
                    nullresults.append([i, j, nmi, rhat, tracenull])
                else:
                    # mean of every trace
                    nullresults.append([i, j, nmi, rhat, tracenull.posterior['covar'].values.mean(axis=1)])
        
    return trueresults, nullresults


def simple_worker_zip_residualized(chunk):
    print(f"running full zip two-stage model process {os.getpid()}")
    if len(chunk) == 0:
        return None
    
    subs = chunk[0][-3]
    usemask = chunk[0][-2]
    posteriormode = chunk[0][-1]
    if posteriormode not in ["fullposterior", 'posteriormeans', 'trace']:
        raise Exception(f"posterior mode {posteriormode} not supported")
    
    n_subs = len(np.unique(subs))
    trueresults = []
    nullresults = []
    for i, j, data, covar, nullstartidx, nullblocksize, subs, usemask, posteriormode in chunk:
        obs = data[usemask]
        cv = covar[usemask]

        with pm.Model() as zmodel:
            sigma_sub = pm.HalfNormal("sigma_sub", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
            
            alpha = pm.Normal("alpha", mu=0, sigma=5)
            
            eta = alpha + subject_effects[subs[usemask]] # only random effects
            mu = pm.Deterministic("mu", pm.math.exp(eta))

            psi = pm.Beta("psi", alpha=1, beta=1)
            
            y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=psi, observed=obs)
            z_trace = pm.sample(1000, chains=4, return_inferencedata=True, 
                                    target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                    progressbar=False)

        mu_hat = z_trace.posterior["mu"].mean(dim=("chain", "draw")).values
        psi_hat = z_trace.posterior["psi"].mean().item()
        y_hat = (1 - psi_hat) * mu_hat
        residuals = obs - y_hat
        
        with pm.Model() as truemodel:
            beta = pm.Normal("covar", mu=0, sigma=5)
            mu = beta * cv
            sigma = pm.HalfNormal("sigma", sigma=1)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=residuals)
            tracetrue = pm.sample(1000, chains=4, return_inferencedata=True, 
                                    target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                    progressbar=False)
            rhat = az.rhat(tracetrue)['covar'].values.flatten()[0]
            trueresults.append([i, j, 0, rhat, tracetrue])

        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            print(f"link {j}->{i} running null model {nmi}")
            covar_null = covar.copy().reshape(n_subs, -1)
            np.random.seed(nmi)
            np.random.shuffle(covar_null)
            covar_null = covar_null.flatten()
            cvn = covar_null[usemask]

            with pm.Model() as nullmodel:
                beta = pm.Normal("covar", mu=0, sigma=5)
                mu = beta * cvn
                sigma = pm.HalfNormal("sigma", sigma=1)
                y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=residuals)
                # Sampling
                tracenull = pm.sample(1000, chains=4, return_inferencedata=True, 
                                    target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                    progressbar=False)
                rhat = az.rhat(tracenull)['covar'].values.flatten()[0]
                if posteriormode == 'fullposterior':
                    nullresults.append([i, j, nmi, rhat, tracenull.posterior['covar'].values])
                elif posteriormode == 'trace':
                    nullresults.append([i, j, nmi, rhat, tracenull])
                else:
                    # mean of every trace
                    nullresults.append([i, j, nmi, rhat, tracenull.posterior['covar'].values.mean(axis=1)])
        print("done")
        return trueresults, nullresults

def simple_worker_zip_twostage(chunk):
    print(f"running full zip model process {os.getpid()}")
    if len(chunk) == 0:
        return None
    
    subs = chunk[0][-3]
    usemask = chunk[0][-2]
    posteriormode = chunk[0][-1]
    if posteriormode not in ["fullposterior", 'posteriormeans', 'trace']:
        raise Exception(f"posterior mode {posteriormode} not supported")
    
    n_subs = len(np.unique(subs))
    trueresults = []
    nullresults = []
    for i, j, data, covar, nullstartidx, nullblocksize, subs, usemask, posteriormode in chunk:
        obs = data[usemask]
        cv = covar[usemask]

        # first stage: fit reduced model to derive random effects
        # with pm.Model() as reffmodel:
        #     sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
        #     subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
        #     subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
            
        #     # Priors for intercept and slope
        #     alpha = pm.Normal("alpha", mu=0, sigma=5)
            
        #     # Linear predictor
        #     mu = pm.math.exp(alpha + subject_effects[subs[usemask]])
            
        #     # Likelihood
        #     psi = pm.Beta("psi", 1, 1)
        #     y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=psi, observed=obs)

        #     # Sampling
        #     tracereff = pm.sample(1000, chains=4, return_inferencedata=True, 
        #                         target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
        #                         progressbar=False)
        
        # subject_effects_mean = tracereff.posterior["subject_effects"].mean(dim=("chain", "draw")).values
        # print(subject_effects_mean.shape)
        # alpha_mean = tracereff.posterior["alpha"].mean().item()
        # offset_vals = alpha_mean + subject_effects_mean[subs[usemask]]

        # second stage: fit true model using random effects posterior means
        with pm.Model() as truemodel:
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
            
            # Priors for intercept and slope
            alpha = pm.Normal("alpha", mu=0, sigma=5)
            beta = pm.Normal("covar", mu=0, sigma=5)
            
            # Linear predictor
            mu = pm.math.exp(alpha + beta * cv + subject_effects[subs[usemask]])
            
            # Likelihood
            psi = pm.Beta("psi", 1, 1)
            y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=psi, observed=obs)

            # Sampling
            tracetrue = pm.sample(1000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)
            rhat = az.rhat(tracetrue)['covar'].values.flatten()[0]
            trueresults.append([i, j, 0, rhat, tracetrue])

        subject_effects_mean = tracetrue.posterior["subject_effects"].mean(dim=("chain", "draw")).values
        alpha_mean = tracetrue.posterior["alpha"].mean().item()
        offset_vals = alpha_mean + subject_effects_mean[subs[usemask]]

        # second stage: fit null models using random effects posterior means
        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            print(f"link {j}->{i} running null model {nmi}")
            covar_null = covar.copy().reshape(n_subs, -1)
            np.random.seed(nmi)
            np.random.shuffle(covar_null)
            covar_null = covar_null.flatten()
            cvn = covar_null[usemask]
            
            with pm.Model() as nullmodel:
                beta = pm.Normal("covar", mu=0, sigma=5)
                
                # Linear predictor
                mu = pm.math.exp(beta * cvn + offset_vals)

                # Likelihood
                psi = pm.Beta("psi", 1, 1)
                y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=psi, observed=obs)

                # Sampling
                tracenull = pm.sample(1000, chains=4, return_inferencedata=True, 
                                    target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                    progressbar=False)
                rhat = az.rhat(tracenull)['covar'].values.flatten()[0]
                if posteriormode == 'fullposterior':
                    nullresults.append([i, j, nmi, rhat, tracenull.posterior['covar'].values])
                elif posteriormode == 'trace':
                    nullresults.append([i, j, nmi, rhat, tracenull])
                else:
                    # mean of every trace
                    nullresults.append([i, j, nmi, rhat, tracenull.posterior['covar'].values.mean(axis=1)])
        
    return trueresults, nullresults


def simple_worker_zip_reduced_full(chunk):
    print(f"running zip models process {os.getpid()}")
    if len(chunk) == 0:
        return None
    
    subs = chunk[0][-3]
    usemask = chunk[0][-2]
    posteriormode = chunk[0][-1]
    if posteriormode not in ["fullposterior", 'posteriormeans', 'trace']:
        raise Exception(f"posterior mode {posteriormode} not supported")
    
    n_subs = len(np.unique(subs))
    trueresults = []
    nullresults = []
    for i, j, data, covar, nullstartidx, nullblocksize, subs, usemask, posteriormode in chunk:
        obs = data[usemask]
        cv = covar[usemask]


        # first stage: fit reduced model using only random effects
        with pm.Model() as reducedmodel:
            print(f"link {j}->{i} running reduced model")
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
            
            # Priors for intercept and slope
            alpha = pm.Normal("alpha", mu=0, sigma=5)
            
            # Linear predictor
            mu = pm.math.exp(alpha + subject_effects[subs[usemask]])
            
            # Likelihood
            psi = pm.Beta("psi", 1, 1)
            y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=psi, observed=obs)

            # Sampling
            tracereduced = pm.sample(1000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)

            postpred = pm.sample_posterior_predictive(tracereduced, random_seed=0)

        artificial_data = postpred['posterior_predictive'].y_obs.values.reshape(-1, len(obs))
        
        # fit one full model to the true data. this is only done separately to conform to the return value api
        # ideally it should be included in the block below since the model is exactly the same and only the 
        # data are changed (from true observations to artificial data from posterior predictions)
        with pm.Model() as fullmodel_true:
            print(f"link {j}->{i} running full true model")
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
            
            # Priors for intercept and slope
            alpha = pm.Normal("alpha", mu=0, sigma=5)
            beta = pm.Normal("covar", mu=0, sigma=0.5)
            
            # Linear predictor
            mu = pm.math.exp(alpha + beta * cv + subject_effects[subs[usemask]])
            
            # Likelihood
            psi = pm.Beta("psi", 1, 1)
            y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=psi, observed=obs)

            # Sampling
            tracefulltrue = pm.sample(1000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)
            rhat = az.rhat(tracefulltrue)['covar'].values.flatten()[0]
            trueresults.append([i, j, 0, rhat, tracefulltrue])

        # second stage: fit null models using random effects posterior means
        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            assert(nmi < artificial_data.shape[0])
            print(f"link {j}->{i} running full null model {nmi}")
            artificial_obs = artificial_data[nmi]
            with pm.Model() as fullmodel_null:
                sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
                subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
                subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
                
                # Priors for intercept and slope
                alpha = pm.Normal("alpha", mu=0, sigma=5)
                beta = pm.Normal("covar", mu=0, sigma=0.5)
                
                # Linear predictor
                mu = pm.math.exp(alpha + beta * cv + subject_effects[subs[usemask]])

                # Likelihood
                psi = pm.Beta("psi", 1, 1)
                y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=psi, observed=artificial_obs)

                # Sampling
                tracefullppc = pm.sample(1000, chains=4, return_inferencedata=True, 
                                    target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                    progressbar=False)
                rhat = az.rhat(tracefullppc)['covar'].values.flatten()[0]
                if posteriormode == 'fullposterior':
                    nullresults.append([i, j, nmi, rhat, tracefullppc.posterior['covar'].values])
                elif posteriormode == 'trace':
                    nullresults.append([i, j, nmi, rhat, tracefullppc])
                else:
                    # mean of every trace
                    nullresults.append([i, j, nmi, rhat, tracefullppc.posterior['covar'].values.mean(axis=1)])
        
    return trueresults, nullresults

def integrated_worker_zi_reduced_full(chunk):
    import pymc_extras as pmx    

    print(f"running integrated zi models process {os.getpid()}")
    _supported_models = ["zinegbin", "zip", "zigenpois"]
    if len(chunk) == 0:
        return None
    
    S = chunk[0][-3]
    usemask = chunk[0][-2]
    mode = chunk[0][-1]
    
    n_subs = len(np.unique(S))
    trueresults = []
    nullresults = []
    for i, j, data, covar, design, nullstartidx, nullblocksize, S, usemask, mode in chunk:
        nullmode = mode["nullmode"]
        reducedmode = mode["reducedmode"]
        modeltype = mode['modeltype']
        if not modeltype in _supported_models:
            raise Exception(f"model type {modeltype} not one of {_supported_models}")
        obs = data[usemask]
        cv = covar[usemask]
        if cv.shape != covar.shape:
            raise Exception("usemask not supported. covariate must already be shaped to observations. do not impute values.")
        
        # first stage: fit reduced model using only random effects
        with pm.Model() as reducedmodel:
            # fixed effects: 6 cell means on log scale 
            # order: [Yq, Y0, Y6, Oq, O0, O6]

            # random effects (1 + condition | subject) 
#             sd = pm.HalfNormal.dist(1.0, shape=3)
#             chol, _, _ = pm.LKJCholeskyCov('chol', n=3, eta=2.0, sd_dist=sd)
#             re_raw = pm.Normal('re_raw', 0, 1, shape=(n_subs,3))
#             RE = pm.Deterministic('RE', re_raw @ chol.T)
#             b0  = RE[S, 0]            # random intercept
#             bM  = RE[S, 1]            # random slope for 0dB (vs quiet)
#             bH  = RE[S, 2]            # random slope for -6dB (vs quiet)

            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            RE = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
            b0 = RE[S]

            # linear predictor
            if reducedmode == 'omnibus':
                beta = pm.Normal('beta', 0, 1)
                eta_fixed = beta                  # population mean
            elif reducedmode == 'ageonly':
                beta = pm.Normal('beta', 0, 1, shape=2)
                X = cv.reshape(-1, 2, 3).max(axis=2) 
                eta_fixed = (X @ beta)      
            elif reducedmode == 'conditiononly':
                beta = pm.Normal('beta', 0, 1, shape=3)
                X = cv.reshape(-1, 2, 3).max(axis=1) 
                eta_fixed = (X @ beta)    
            else:
                raise Exception(f"reduced mode {nullmode} not supported")
                
            eta = eta_fixed + b0 # + bM*design[:,0] + bH*design[:,1]
            mu  = pm.Deterministic('mu', pm.math.exp(eta))

            # zero inflation (global psi) 
            alpha_psi = pm.Normal('alpha_psi', 0, 1)
            psi  = pm.Deterministic('psi', pm.math.sigmoid(alpha_psi))

            # likelihood 
            if modeltype == 'zip':
                pm.ZeroInflatedPoisson('y_obs', psi=psi, mu=mu, observed=obs)
            if modeltype == 'zinegbin':
                alpha_disp = pm.HalfNormal("alpha_disp", sigma=5)
                pm.ZeroInflatedNegativeBinomial('y_obs', psi=psi, mu=mu, alpha=alpha_disp, observed=obs)
            if modeltype == 'genpois':
                lower = pm.math.maximum(-1.0, -mu / 4.0)
                # fraction between lower and 1
                lam_rel = pm.Beta("lam_rel", 2, 2)
                lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
                pmx.distributions.GeneralizedPoisson('y_obs', mu=mu, lam=lam, observed=obs)
            if modeltype == 'zigenpois':
                lower = pm.math.maximum(-1.0, -mu / 4.0)
                # fraction between lower and 1
                lam_rel = pm.Beta("lam_rel", 2, 2)
                lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
                zero_component = pm.DiracDelta.dist(0)
                gp_component  = pmx.distributions.GeneralizedPoisson.dist(mu=mu, lam=lam)
                y_obs = pm.Mixture(
                    "y_obs",
                    w=[psi, 1 - psi],
                    comp_dists=[zero_component, gp_component],
                    observed=obs,  
                )


            # sampling
            tracereduced = pm.sample(4000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)

            postpred = pm.sample_posterior_predictive(tracereduced, random_seed=0)

        artificial_data = postpred['posterior_predictive'].y_obs.values.reshape(-1, len(obs))
        
        # fit one full model to the true data. this is only done separately to conform to the return value api
        # ideally it should be included in the block below since the model is exactly the same and only the 
        # data are changed (from true observations to artificial data from posterior predictions)
        with pm.Model() as fullmodel_true:
            # fixed effects: 6 cell means on log scale 
            # order: [Yq, Y0, Y6, Oq, O0, O6]
            beta = pm.Normal('beta', 0, 1, shape=6)

            # random effects (1 + condition | subject) 
            sd = pm.HalfNormal.dist(1.0, shape=3)
            chol, _, _ = pm.LKJCholeskyCov('chol', n=3, eta=2.0, sd_dist=sd)
            re_raw = pm.Normal('re_raw', 0, 1, shape=(n_subs,3))
            RE = pm.Deterministic('RE', re_raw @ chol.T)
            b0  = RE[S, 0]            # random intercept
            bM  = RE[S, 1]            # random slope for 0dB (vs quiet)
            bH  = RE[S, 2]            # random slope for -6dB (vs quiet)

            # linear predictor
            eta_fixed = (cv @ beta)                      # pick the appropriate cell mean
            eta = eta_fixed + b0 + bM*design[:,0] + bH*design[:,1]
            mu  = pm.Deterministic('mu', pm.math.exp(eta))

            # zero inflation (global psi) 
            alpha_psi = pm.Normal('alpha_psi', 0, 1)
            psi  = pm.Deterministic('psi', pm.math.sigmoid(alpha_psi))

            # likelihood 
            if modeltype == 'zip':
                pm.ZeroInflatedPoisson('y_obs', psi=psi, mu=mu, observed=obs)
            if modeltype == 'zinegbin':
                alpha_disp = pm.HalfNormal("alpha_disp", sigma=5)
                pm.ZeroInflatedNegativeBinomial('y_obs', psi=psi, mu=mu, alpha=alpha_disp, observed=obs)
            if modeltype == 'genpois':
                lower = pm.math.maximum(-1.0, -mu / 4.0)
                # fraction between lower and 1
                lam_rel = pm.Beta("lam_rel", 2, 2)
                lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
                pmx.distributions.GeneralizedPoisson('y_obs', mu=mu, lam=lam, observed=obs)
            if modeltype == 'zigenpois':
                lower = pm.math.maximum(-1.0, -mu / 4.0)
                # fraction between lower and 1
                lam_rel = pm.Beta("lam_rel", 2, 2)
                lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
                zero_component = pm.DiracDelta.dist(0)
                gp_component  = pmx.distributions.GeneralizedPoisson.dist(mu=mu, lam=lam)
                y_obs = pm.Mixture(
                    "y_obs",
                    w=[psi, 1 - psi],
                    comp_dists=[zero_component, gp_component],
                    observed=obs,  
                )

            # sampling
            tracefulltrue = pm.sample(4000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)
            rhat = None # unsupported
            trueresults.append([i, j, 0, rhat, tracefulltrue.posterior['beta'].values.mean(axis=1)])

        # second stage: fit null models using random effects posterior means
        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            assert(nmi < artificial_data.shape[0])
            print(f"link {j}->{i} running full null model {nmi}, {modeltype}")
            artificial_obs = artificial_data[nmi]
            with pm.Model() as fullmodel_null:
                # fixed effects: 6 cell means on log scale 
                # order: [Yq, Y0, Y6, Oq, O0, O6]
                beta = pm.Normal('beta', 0, 1, shape=6)

                # random effects (1 + condition | subject) 
                sd = pm.HalfNormal.dist(1.0, shape=3)
                chol, _, _ = pm.LKJCholeskyCov('chol', n=3, eta=2.0, sd_dist=sd)
                re_raw = pm.Normal('re_raw', 0, 1, shape=(n_subs,3))
                RE = pm.Deterministic('RE', re_raw @ chol.T)
                b0  = RE[S, 0]            # random intercept
                bM  = RE[S, 1]            # random slope for 0dB (vs quiet)
                bH  = RE[S, 2]            # random slope for -6dB (vs quiet)

                # linear predictor
                eta_fixed = (cv @ beta)                      # pick the appropriate cell mean
                eta = eta_fixed + b0 + bM*design[:,0] + bH*design[:,1]
                mu  = pm.Deterministic('mu', pm.math.exp(eta))

                # zero inflation (global psi) 
                alpha_psi = pm.Normal('alpha_psi', 0, 1)
                psi  = pm.Deterministic('psi', pm.math.sigmoid(alpha_psi))

                # likelihood 
                if modeltype == 'zip':
                    pm.ZeroInflatedPoisson('y_obs', psi=psi, mu=mu, observed=artificial_obs)
                if modeltype == 'zinegbin':
                    alpha_disp = pm.HalfNormal("alpha_disp", sigma=5)
                    pm.ZeroInflatedNegativeBinomial('y_obs', psi=psi, mu=mu, alpha=alpha_disp, observed=artificial_obs)
                if modeltype == 'genpois':
                    lower = pm.math.maximum(-1.0, -mu / 4.0)
                    # fraction between lower and 1
                    lam_rel = pm.Beta("lam_rel", 2, 2)
                    lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
                    pmx.distributions.GeneralizedPoisson('y_obs', mu=mu, lam=lam, observed=artificial_obs)
                if modeltype == 'zigenpois':
                    lower = pm.math.maximum(-1.0, -mu / 4.0)
                    # fraction between lower and 1
                    lam_rel = pm.Beta("lam_rel", 2, 2)
                    lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
                    zero_component = pm.DiracDelta.dist(0)
                    gp_component  = pmx.distributions.GeneralizedPoisson.dist(mu=mu, lam=lam)
                    y_obs = pm.Mixture(
                        "y_obs",
                        w=[psi, 1 - psi],
                        comp_dists=[zero_component, gp_component],
                        observed=artificial_obs,  
                    )

                # sampling
                if nullmode == 'advi':
                    try:
                        approx = pm.fit(method='advi', progressbar=False, callbacks=[])
                        tracefullnull = approx.sample(1000)
                    except FloatingPointError as e:
                        print(f"null model {nmi} for link {i}->{j} failed due to floating point error: {e}")
                        nullresults.append([i, j, nmi, None, np.nan*np.zeros((1, 6))])
                        continue
                elif nullmode == 'nuts':
                    tracefullnull = pm.sample(4000, chains=4, return_inferencedata=True, 
                                        target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                        progressbar=False)
                else:
                    raise Exception(f"null mode {nullmode} not supported")
                rhat = None # unsupported
                nullresults.append([i, j, nmi, rhat, tracefullnull.posterior['beta'].values.mean(axis=1)])
        
    return trueresults, nullresults


def simple_worker_zigenpois_interceptonly(chunk):
    import pymc_extras as pmx    
    print(f"running zip models process {os.getpid()}")
    if len(chunk) == 0:
        return None
    
    subs = chunk[0][-3]
    usemask = chunk[0][-2]
    posteriormode = chunk[0][-1]
    # if posteriormode not in ["fullposterior", 'posteriormeans', 'trace']:
    #     raise Exception(f"posterior mode {posteriormode} not supported")
    
    n_subs = len(np.unique(subs))
    trueresults = []
    # nullresults = []
    for i, j, data, covar, nullstartidx, nullblocksize, subs, usemask, posteriormode in chunk:
        obs = data[usemask]
        cv = covar[usemask]
        
        # fit one full model to the true data. this is only done separately to conform to the return value api
        # ideally it should be included in the block below since the model is exactly the same and only the 
        # data are changed (from true observations to artificial data from posterior predictions)
        with pm.Model() as fullmodel_true:
            print(f"link {j}->{i} running full true model")
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)
            
            # Priors for intercept and slope
            alpha = pm.Normal("alpha", mu=0, sigma=5)
            # beta = pm.Normal("covar", mu=0, sigma=0.5)
            
            # Linear predictor
            mu = pm.math.exp(alpha + subject_effects[subs[usemask]])
            
            # Likelihood
            psi = pm.Beta("psi", 1, 1)
            
            lower = pm.math.maximum(-1.0, -mu / 4.0)
            # fraction between lower and 1
            lam_rel = pm.Beta("lam_rel", 2, 2)
            lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
            zero_component = pm.DiracDelta.dist(0)
            gp_component  = pmx.distributions.GeneralizedPoisson.dist(mu=mu, lam=lam)
            y_obs = pm.Mixture(
                "y_obs",
                w=[psi, 1 - psi],
                comp_dists=[zero_component, gp_component],
                observed=obs,  
            )

            # Sampling
            tracefulltrue = pm.sample(5000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)
            rhat = az.rhat(tracefulltrue)['alpha'].values.flatten()[0]
            trueresults.append([i, j, 0, rhat, tracefulltrue])

    return trueresults, None

# +
def simple_worker_zip_conditiononly(chunk):
    import pymc_extras as pmx    

    print(f"running integrated zip models process {os.getpid()}")
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
        reducedmode = mode["reducedmode"]
        nullppc = mode["nullppc"]

        obs = data[usemask]
        cv = X[usemask]
        subs = S[usemask]
        design = Z[usemask]
        n_conditions = cv.shape[1]

        # first stage: fit true model
        with pm.Model() as fullmodel_true:
            # fixed effects parameters
            intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean
            beta = pm.Normal('beta', 0, 5, shape=(n_conditions-1))

            # random effects (1 + condition | subject) 
            sd = pm.HalfNormal.dist(1.0, shape=n_conditions)
            chol, _, _ = pm.LKJCholeskyCov('chol', n=n_conditions, eta=2.0, sd_dist=sd)
            re_raw = pm.Normal('re_raw', 0, 1, shape=(n_subs, n_conditions))
            RE = pm.Deterministic('RE', re_raw @ chol.T)
            bRE  = RE[subs, :]        # random effects (n_obs, n_conditions)

#             sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
#             subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
#             subject_effects = pm.Deterministic("subject_effects", subject_offset * sigma_sub)

            # linear predictor
            # beta_intercept + ind(beta_0dB) + ind(beta_-6dB) + re_intercept + ind(re_0dB) + ind(re_-6dB)
            eta_fixed = (cv[:, 1:] @ beta)   # fixed effects
#             eta = eta_fixed + subject_effects[subs]#+ pm.math.sum(bRE * design, axis=1)
            eta = intercept + eta_fixed + pm.math.sum(bRE * design, axis=1)

            mu  = pm.Deterministic('mu', pm.math.exp(eta))

            # likelihood zip
            conditions = cv
            conditions[:, 0] = (conditions[:, 1:].sum(axis=1) == 0).astype(int)
            
            # pooled per condition psis
#             alpha_psi_mu = pm.Normal('alpha_psi_mu', 0., 1.)
#             alpha_psi_sd = pm.HalfNormal('alpha_psi_sd', 1.)
#             alpha_psi_offset = pm.Normal('alpha_psi_offset', 0., 1., shape=n_conditions)
#             alpha_psi = pm.Deterministic('psi_latent', alpha_psi_mu + alpha_psi_offset * alpha_psi_sd)    # (n_conditions,)
#             psi_cond = pm.Deterministic('psi', pm.math.sigmoid(conditions @ alpha_psi))
            psi = pm.Beta("psi", 1, 1)
            
            y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=psi, observed=obs)

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
            postpred = pm.sample_posterior_predictive(tracefulltrue, random_seed=0)
            _intercept = post['intercept'].mean(dim=('chain', 'draw')).values
            _mu = post['mu'].mean(dim=('chain', 'draw')).values
            _psi = post['psi'].mean().item()
            posterior_subsamples = {
                "intercept": _intercept, 
                "beta": post['beta'].mean(dim=('chain', 'draw')).values,
                "RE": post["RE"].mean(dim=("chain", "draw")).values,
                "mu": _mu,
                "psi": _psi,
                "obs": obs,
                "predicted_samples": postpred,
                "badconv": badconvflag,
            }
            rhat = None # az.rhat(tracefulltrue)
            trueresults.append([i, j, 0, rhat, posterior_subsamples])
        
        subject_effects_mean = post["RE"].mean(dim=('chain', 'draw')).values
        offset_vals = subject_effects_mean[subs]
        fixed_psi = _psi
        fixed_intercept = post['intercept'].mean().item()

        # second stage: fit null models using random effects posterior means
        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            print(f"link {j}->{i} running permuted null model {nmi}")
            cvshape = X.shape
            usemaskshape = usemask.shape

            covar_null = X.copy().reshape(n_subs, -1)
#             usemask_null = usemask.copy().reshape(n_subs, -1)

            np.random.seed(nmi)
            idxs = list(range(n_subs))
            np.random.shuffle(idxs)

            covar_null = covar_null[idxs]
            covar_null = covar_null.reshape(*cvshape)
#             usemask_null = usemask_null[idxs]
#             usemask_null = usemask_null.reshape(*usemaskshape)

            cvn = covar_null[usemask]

            with pm.Model() as fullmodel_null:
                # fixed effects: 3 cell means 
                # order: [Yq, Y0, Y6]
                beta = pm.Normal('beta', 0, 5, shape=(n_conditions - 1)) # intercept is fixed

                # linear predictor
                eta_fixed = (cvn[:, 1:] @ beta)                  # pick the appropriate cell mean
                eta = fixed_intercept + eta_fixed + pm.math.sum(offset_vals * design, axis=1)

                mu = pm.Deterministic('mu', pm.math.exp(eta))

                # likelihood zip
                y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=fixed_psi, observed=obs)

                # see if sampler converges with obtaining invalid values
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)

                    try:
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
                    tracefullnull = pm.sample(4000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)

                post = tracefullnull.posterior
                if nullppc:
                    postpred = pm.sample_posterior_predictive(tracefullnull, random_seed=0)
                else:
                    postpred = None
                _mu = post['mu'].mean(dim=('chain', 'draw')).values
                posterior_subsamples = {
                    "beta": post['beta'].mean(dim=('chain', 'draw')).values,
                    "mu": _mu,
                    "psi": fixed_psi,
                    "intercept": fixed_intercept,
                    "predicted_samples": postpred,
                    "badconv": badconvflag,
                }

                rhat = None # az.rhat(tracefullnull)
                nullresults.append([i, j, nmi, rhat, posterior_subsamples])

    return trueresults, nullresults


# -
def simple_worker_zip_conditiononly_norandslopes(chunk):
    import pymc_extras as pmx    

    print(f"running integrated zip models process {os.getpid()}")
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
        reducedmode = mode["reducedmode"]
        nullppc = mode["nullppc"]

        obs = data[usemask]
        cv = X[usemask]
        subs = S[usemask]
        design = Z[usemask]
        n_conditions = cv.shape[1]

        # first stage: fit true model
        with pm.Model() as fullmodel_true:
            # fixed effects parameters
            intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean
            beta = pm.Normal('beta', 0, 5, shape=(n_conditions-1))

            # random effects (1 + condition | subject) 
#             sd = pm.HalfNormal.dist(1.0, shape=n_conditions)
#             chol, _, _ = pm.LKJCholeskyCov('chol', n=n_conditions, eta=2.0, sd_dist=sd)
#             re_raw = pm.Normal('re_raw', 0, 1, shape=(n_subs, n_conditions))
#             RE = pm.Deterministic('RE', re_raw @ chol.T)
#             bRE  = RE[subs, :]        # random effects (n_obs, n_conditions)
            
            # random effects (1|subject)
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

            # linear predictor
            # beta_intercept + ind(beta_0dB) + ind(beta_-6dB) + re_intercept + ind(re_0dB) + ind(re_-6dB)
            eta_fixed = (cv[:, 1:] @ beta)   # fixed effects
            eta = intercept + eta_fixed + subject_effects[subs] #+ pm.math.sum(bRE * design, axis=1)
#             eta = intercept + eta_fixed + pm.math.sum(bRE * design, axis=1)

            mu  = pm.Deterministic('mu', pm.math.exp(eta))

            # likelihood zip
            conditions = cv
            conditions[:, 0] = (conditions[:, 1:].sum(axis=1) == 0).astype(int)
            
            # pooled per condition psis
#             alpha_psi_mu = pm.Normal('alpha_psi_mu', 0., 1.)
#             alpha_psi_sd = pm.HalfNormal('alpha_psi_sd', 1.)
#             alpha_psi_offset = pm.Normal('alpha_psi_offset', 0., 1., shape=n_conditions)
#             alpha_psi = pm.Deterministic('psi_latent', alpha_psi_mu + alpha_psi_offset * alpha_psi_sd)    # (n_conditions,)
#             psi_cond = pm.Deterministic('psi', pm.math.sigmoid(conditions @ alpha_psi))
            psi = pm.Beta("psi", 1, 1)
            
            y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=psi, observed=obs)

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
            postpred = pm.sample_posterior_predictive(tracefulltrue, random_seed=0)
            _intercept = post['intercept'].mean(dim=('chain', 'draw')).values
            _mu = post['mu'].mean(dim=('chain', 'draw')).values
            _psi = post['psi'].mean().item()
            posterior_subsamples = {
                "intercept": _intercept, 
                "beta": post['beta'].mean(dim=('chain', 'draw')).values,
                "RE": post["RE"].mean(dim=("chain", "draw")).values,
                "mu": _mu,
                "psi": _psi,
                "obs": obs,
                "predicted_samples": postpred,
                "badconv": badconvflag,
            }
            rhat = None # az.rhat(tracefulltrue)
            trueresults.append([i, j, 0, rhat, posterior_subsamples])
        
        subject_effects_mean = post["RE"].mean(dim=('chain', 'draw')).values
        offset_vals = subject_effects_mean[subs]
        fixed_psi = _psi
        fixed_intercept = post['intercept'].mean().item()

        # second stage: fit null models using random effects posterior means
        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            print(f"link {j}->{i} running permuted null model {nmi}")
            cvshape = X.shape
            usemaskshape = usemask.shape

            covar_null = X.copy().reshape(n_subs, -1)
#             usemask_null = usemask.copy().reshape(n_subs, -1)

            np.random.seed(nmi)
            idxs = list(range(n_subs))
            np.random.shuffle(idxs)

            covar_null = covar_null[idxs]
            covar_null = covar_null.reshape(*cvshape)
#             usemask_null = usemask_null[idxs]
#             usemask_null = usemask_null.reshape(*usemaskshape)

            cvn = covar_null[usemask]

            with pm.Model() as fullmodel_null:
                # fixed effects: 3 cell means 
                # order: [Yq, Y0, Y6]
                beta = pm.Normal('beta', 0, 5, shape=(n_conditions - 1)) # intercept is fixed

                # linear predictor
                eta_fixed = (cvn[:, 1:] @ beta)                  # pick the appropriate cell mean
                eta = fixed_intercept + eta_fixed + offset_vals

                mu = pm.Deterministic('mu', pm.math.exp(eta))

                # likelihood zip
                y_obs = pm.ZeroInflatedPoisson("y_obs", mu=mu, psi=fixed_psi, observed=obs)

                # see if sampler converges with obtaining invalid values
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)

                    try:
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
                    tracefullnull = pm.sample(4000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)

                post = tracefullnull.posterior
                if nullppc:
                    postpred = pm.sample_posterior_predictive(tracefullnull, random_seed=0)
                else:
                    postpred = None
                _mu = post['mu'].mean(dim=('chain', 'draw')).values
                posterior_subsamples = {
                    "beta": post['beta'].mean(dim=('chain', 'draw')).values,
                    "mu": _mu,
                    "psi": fixed_psi,
                    "intercept": fixed_intercept,
                    "predicted_samples": postpred,
                    "badconv": badconvflag,
                }

                rhat = None # az.rhat(tracefullnull)
                nullresults.append([i, j, nmi, rhat, posterior_subsamples])

    return trueresults, nullresults


def simple_worker_conditiononly_norandslopes(chunk):
    import pymc_extras as pmx    

    print(f"running integrated zip models process {os.getpid()}")
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
        reducedmode = mode["reducedmode"]
        nullppc = mode["nullppc"]
        exposure = mode["exposure"]
        modeltype = mode["modeltype"]

        obs = data[usemask]
        cv = X[usemask]
        subs = S[usemask]
        design = Z[usemask]
        n_conditions = cv.shape[1]
        
        # hurdle models nonzero distinctly
        cv = cv[obs > 0]
        subs = subs[obs > 0]
        design = design[obs > 0]

        # first stage: fit true model
        with pm.Model() as fullmodel_true:
            # fixed effects parameters
            intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean
            beta = pm.Normal('beta', 0, 5, shape=(n_conditions-1))

            # random effects (1 + condition | subject) 
#             sd = pm.HalfNormal.dist(1.0, shape=n_conditions)
#             chol, _, _ = pm.LKJCholeskyCov('chol', n=n_conditions, eta=2.0, sd_dist=sd)
#             re_raw = pm.Normal('re_raw', 0, 1, shape=(n_subs, n_conditions))
#             RE = pm.Deterministic('RE', re_raw @ chol.T)
#             bRE  = RE[subs, :]        # random effects (n_obs, n_conditions)
            
            # random effects (1|subject)
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

            # linear predictor
            # beta_intercept + ind(beta_0dB) + ind(beta_-6dB) + re_intercept + ind(re_0dB) + ind(re_-6dB)
            eta_fixed = (cv[:, 1:] @ beta)   # fixed effects
            eta = intercept + eta_fixed + exposure + subject_effects[subs] #+ pm.math.sum(bRE * design, axis=1)
#             eta = intercept + eta_fixed + pm.math.sum(bRE * design, axis=1)

            mu  = pm.Deterministic('mu', pm.math.exp(eta))

            # likelihood zip
#             conditions = cv
#             conditions[:, 0] = (conditions[:, 1:].sum(axis=1) == 0).astype(int)
            
            # pooled per condition psis
#             alpha_psi_mu = pm.Normal('alpha_psi_mu', 0., 1.)
#             alpha_psi_sd = pm.HalfNormal('alpha_psi_sd', 1.)
#             alpha_psi_offset = pm.Normal('alpha_psi_offset', 0., 1., shape=n_conditions)
#             alpha_psi = pm.Deterministic('psi_latent', alpha_psi_mu + alpha_psi_offset * alpha_psi_sd)    # (n_conditions,)
#             psi_cond = pm.Deterministic('psi', pm.math.sigmoid(conditions @ alpha_psi))
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
            postpred = pm.sample_posterior_predictive(tracefulltrue, random_seed=0)
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
                "predicted_samples": postpred,
                "badconv": badconvflag,
            }
            rhat = None # az.rhat(tracefulltrue)
            trueresults.append([i, j, 0, rhat, posterior_subsamples])
        
        subject_effects_mean = post["RE"].mean(dim=('chain', 'draw')).values
        offset_vals = subject_effects_mean[subs]
        fixed_psi = _psi
        if modeltype == 'gamma':
            fixed_sigma = post['sigma'].mean().item()
        elif modeltype == 'genpois':
            fixed_lam_rel = post['lam_rel'].mean().item()
        fixed_intercept = post['intercept'].mean().item()

        # second stage: fit null models using random effects posterior means
        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            print(f"link {j}->{i} running permuted null model {nmi}")
            cvshape = X.shape
            usemaskshape = usemask.shape

            covar_null = X.copy().reshape(n_subs, -1)
#             usemask_null = usemask.copy().reshape(n_subs, -1)

            np.random.seed(nmi)
            idxs = list(range(n_subs))
            np.random.shuffle(idxs)

            covar_null = covar_null[idxs]
            covar_null = covar_null.reshape(*cvshape)
#             usemask_null = usemask_null[idxs]
#             usemask_null = usemask_null.reshape(*usemaskshape)
            
            

            cvn = covar_null[usemask] # usemask tells which obs not to use so should be unpermuted
            cvn = cvn[obs > 0]        

            with pm.Model() as fullmodel_null:
                # fixed effects: 3 cell means 
                # order: [Yq, Y0, Y6]
                beta = pm.Normal('beta', 0, 5, shape=(n_conditions - 1)) # intercept is fixed

                # linear predictor
                eta_fixed = (cvn[:, 1:] @ beta)                  # pick the appropriate cell mean
                eta = fixed_intercept + eta_fixed + exposure + offset_vals

                mu = pm.Deterministic('mu', pm.math.exp(eta))

                # likelihood zip
                if modeltype == 'gamma':
                    y_obs = pm.Gamma("y_obs", mu=mu, sigma=fixed_sigma, observed=obs[obs > 0])
                elif modeltype == 'poisson':
                    y_obs = pm.Poisson("y_obs", mu=mu, observed=obs[obs > 0])
                elif modeltype == 'genpois':
                    lower = pm.math.maximum(-1.0, -mu / 4.0)
                    # fraction between lower and 1
                    lam = pm.Deterministic("lam", lower + fixed_lam_rel * (1.0 - lower))
                    y_obs = pmx.distributions.GeneralizedPoisson(mu=mu, lam=lam, observed=obs[obs>0])
                
                pm.Bernoulli("y_bernoulli", p=1 - fixed_psi, observed=(obs == 0) * 1.0)

                # see if sampler converges with obtaining invalid values
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)

                    try:
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
                    tracefullnull = pm.sample(4000, chains=4, return_inferencedata=True, 
                                target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                                progressbar=False)

                post = tracefullnull.posterior
                if nullppc:
                    postpred = pm.sample_posterior_predictive(tracefullnull, random_seed=0)
                else:
                    postpred = None
                _mu = post['mu'].mean(dim=('chain', 'draw')).values
                posterior_subsamples = {
                    "beta": post['beta'].mean(dim=('chain', 'draw')).values,
                    "mu": _mu,
                    "psi": fixed_psi,
                    "intercept": fixed_intercept,
                    "predicted_samples": postpred,
                    "badconv": badconvflag,
                }

                rhat = None # az.rhat(tracefullnull)
                nullresults.append([i, j, nmi, rhat, posterior_subsamples])

    return trueresults, nullresults


def simple_worker_conditiononly_norandslopes_reduced_full(chunk):
    import pymc_extras as pmx    

    print(f"running integrated zip models process {os.getpid()}")
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
        reducedmode = mode["reducedmode"]
        nullppc = mode["nullppc"]
        exposure = np.log(mode["exposure"])
        modeltype = mode["modeltype"]

        obs = data[usemask]
        cv = X[usemask]
        subs = S[usemask]
        design = Z[usemask]
        n_conditions = cv.shape[1]
        
        # hurdle models nonzero distinctly
        intercept_mask = [bool(np.all(i == [1, 0, 0])) for i in cv] # only quiet condition
        intercept_subs = subs[(obs > 0) & (intercept_mask)]
        intercept_cv = cv[(obs > 0) & (intercept_mask)]

        # first stage: fit reduced model
        with pm.Model() as reducedmodel:
            # fixed effects parameters
            intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean

            # random effects (1|subject)
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

            # linear predictor
            eta = intercept + exposure + subject_effects[intercept_subs]

            mu  = pm.Deterministic('mu', pm.math.exp(eta))
            psi = pm.Beta("psi", 1, 1)

            if modeltype == 'gamma':
                sigma = pm.HalfNormal("sigma", sigma=1)
                y_obs = pm.Gamma("y_obs", mu=mu, sigma=sigma, observed=obs[(obs > 0) & (intercept_mask)])
            elif modeltype == 'poisson':
                y_obs = pm.Poisson("y_obs", mu=mu, observed=obs[(obs > 0) & (intercept_mask)])
            elif modeltype == 'genpois':
                lower = pm.math.maximum(-1.0, -mu / 4.0)
                # fraction between lower and 1
                lam_rel = pm.Beta("lam_rel", 2, 2)
                lam = pm.Deterministic("lam", lower + lam_rel * (1.0 - lower))
                y_obs = pmx.distributions.GeneralizedPoisson(mu=mu, lam=lam, observed=obs[(obs > 0) & (intercept_mask)])

            pm.Bernoulli("y_bernoulli", p=1 - psi, observed=(obs[intercept_mask] == 0) * 1.0)

            # see if sampler converges with obtaining invalid values
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)

                try:
                    tracereduced = pm.sample(4000, chains=4, return_inferencedata=True, 
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
                tracereduced = pm.sample(4000, chains=4, return_inferencedata=True, 
                            target_accept=0.95, cores=1, idata_kwargs={"log_likelihood": True}, 
                            progressbar=False)

            expanded_idata = tracereduced.copy()
            
            # to fit the intercept we are only using quiet sessions (~1/3 of data)
            # so multiply pp samples by 3 so we get 0dB and -6dB null sessions too. 
            # this does not repeat the quiet pp, but instead draws two additional 
            # samples from the same chains
            expanded_idata.posterior = tracereduced.posterior.expand_dims(y_obs_newdim=3)
            with reducedmodel:
                pm.sample_posterior_predictive(
                    expanded_idata,
                    sample_dims=["chain", "draw", "y_obs_newdim"],
                    extend_inferencedata=True,
                )

            postpred = expanded_idata.posterior_predictive
        
        
        artificial_data = postpred.y_obs.values.reshape(4 * 4000, -1) # (n_chain x n_samples, n_obs * 3)
        
        # need to make sure we don't (likely) fit one model per chain within each sample
        # which would happen if we didn't shuffle the data (very unlikely now)
        np.random.seed(0)
        np.random.shuffle(artificial_data) # (n_chain x 4000, n_obs)
        
        # second stage: fit null models to artificial data
        for nmi in range(nullstartidx, nullstartidx+nullblocksize):
            print(f"link {j}->{i} running artificial data null model {nmi}")
            
            # make 0dB and -6dB indicators
            cvzero = intercept_cv.copy()
            cvsix = intercept_cv.copy()
            cvzero[:, 1] = 1
            cvsix[:, 2] = 1

            cvnull = np.concatenate([intercept_cv, cvzero, cvsix], axis=0)
            subsnull = np.concatenate([intercept_subs, intercept_subs, intercept_subs], axis=0)
            
            nullobs = artificial_data[nmi]
            cvnull = cvnull[nullobs > 0]
            subsnull = subsnull[nullobs > 0]

            print(f"link {j}->{i} running null model {nmi}")

            # everything else exactly the same as true full model
            with pm.Model() as nullmodel:
                # fixed effects parameters
                intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean
                beta = pm.Normal('beta', 0, 5, shape=(n_conditions-1))

                # random effects (1|subject)
                sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
                subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
                subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

                # linear predictor
                # beta_intercept + ind(beta_0dB) + ind(beta_-6dB) + re_intercept + ind(re_0dB) + ind(re_-6dB)
                eta_fixed = (cvnull[:, 1:] @ beta)   # fixed effects
                eta = intercept + eta_fixed + exposure + subject_effects[subsnull]

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
                    "RE": post["RE"].mean(dim=("chain", "draw")).values,
                    "mu": _mu,
                    "psi": _psi,
                    "exposure": exposure,
                    "obs": obs,
                    "badconv": badconvflag,
                }
                rhat = None # az.rhat(tracefulltrue)
                nullresults.append([i, j, nmi, rhat, posterior_subsamples])
                
                
        # third stage: fit true model
        
        # fresh copy
        obs = data[usemask]
        cv = X[usemask]
        subs = S[usemask]
        design = Z[usemask]
        n_conditions = cv.shape[1]

        with pm.Model() as fullmodel_true:
            # fixed effects parameters
            intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean
            beta = pm.Normal('beta', 0, 5, shape=(n_conditions-1))

            # random effects (1|subject)
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

            # linear predictor
            # beta_intercept + ind(beta_0dB) + ind(beta_-6dB) + re_intercept + ind(re_0dB) + ind(re_-6dB)
            eta_fixed = (cv[obs > 0, 1:] @ beta)   # fixed effects
            eta = intercept + eta_fixed + exposure + subject_effects[subs[obs > 0]]

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
            postpred = pm.sample_posterior_predictive(tracefulltrue, random_seed=0)
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
                "predicted_samples": postpred,
                "artificial_data": artificial_data,
                "badconv": badconvflag,
            }
            rhat = None # az.rhat(tracefulltrue)
            trueresults.append([i, j, 0, rhat, posterior_subsamples])

    return trueresults, nullresults


def simple_worker_norandslopes_counterfactual(chunk):
    import pymc_extras as pmx    

    print(f"running integrated zip models process {os.getpid()}")
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
        reducedmode = mode["reducedmode"]
        nullppc = mode["nullppc"]
        exposure = np.log(mode["exposure"])
        modeltype = mode["modeltype"]

        obs = data[usemask]
        cv = X[usemask]
        subs = S[usemask]
        design = Z[usemask]
        n_conditions = cv.shape[1]

        with pm.Model() as fullmodel_true:
            # fixed effects parameters
            intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean
            beta = pm.Normal('beta', 0, 5, shape=(n_conditions-1))

            # random effects (1|subject)
            sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
            subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
            subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

            # linear predictor
            # beta_intercept + ind(beta_0dB) + ind(beta_-6dB) + re_intercept + ind(re_0dB) + ind(re_-6dB)
            eta_fixed = (cv[obs > 0, 1:] @ beta)   # fixed effects
            eta = intercept + eta_fixed + exposure + subject_effects[subs[obs > 0]]

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
                "RE": post["RE"].mean(dim=("chain", "draw")).values,
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
        with pm.do(fullmodel_true, {beta: [0.0] * (n_conditions-1)}) as m_do:
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

            print(f"link {j}->{i} running null model {nmi}")

            # everything else exactly the same as true full model
            with pm.Model() as nullmodel:
                # fixed effects parameters
                intercept = pm.Normal("intercept", mu=0, sigma=10)                 # population baseline log-mean
                beta = pm.Normal('beta', 0, 5, shape=(n_conditions-1))

                # random effects (1|subject)
                sigma_sub = pm.HalfNormal("subject_re_sigma", sigma=10)
                subject_offset = pm.Normal("subject_offset", mu=0, sigma=1, shape=n_subs)
                subject_effects = pm.Deterministic("RE", subject_offset * sigma_sub)

                # linear predictor
                # beta_intercept + ind(beta_0dB) + ind(beta_-6dB) + re_intercept + ind(re_0dB) + ind(re_-6dB)
                eta_fixed = (cvnull[nullobs > 0, 1:] @ beta)   # fixed effects
                eta = intercept + eta_fixed + exposure + subject_effects[subsnull[nullobs > 0]]

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
                    "RE": post["RE"].mean(dim=("chain", "draw")).values,
                    "mu": _mu,
                    "psi": _psi,
                    "exposure": exposure,
                    "obs": obs,
                    "badconv": badconvflag,
                }
                rhat = None # az.rhat(tracefulltrue)
                nullresults.append([i, j, nmi, rhat, posterior_subsamples])     

    return trueresults, nullresults


def run_variational_bayes(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with Pool() as pool:
        res = list(pool.imap(worker, chunks)) # each worker gets one chunk at a time

    return res

def run_variational_bayes_simple(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with Pool() as pool:
        res = list(pool.imap(simple_worker, chunks)) # each worker gets one chunk at a time

    return res

def run_variational_bayes_simple_negbin(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with Pool() as pool:
        res = list(pool.imap(simple_worker_negbin, chunks)) # each worker gets one chunk at a time

    return res

def run_variational_bayes_simple_zip(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with Pool() as pool:
        res = list(pool.imap(simple_worker_zip, chunks)) # each worker gets one chunk at a time

    return res


def run_variational_bayes_simple_zip_twostage(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(simple_worker_zip_twostage, chunks)) # each worker gets one chunk at a time

    print("transmit")
    return res

def run_variational_bayes_simple_zip_reduced_full(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(simple_worker_zip_reduced_full, chunks)) # each worker gets one chunk at a time

    print("transmit")
    return res

def run_variational_bayes_integrated_zi_reduced_full(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(integrated_worker_zi_reduced_full, chunks)) # each worker gets one chunk at a time

    print("transmit")
    return res

def run_variational_bayes_interceptonly_zigenpois(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(simple_worker_zigenpois_interceptonly, chunks)) # each worker gets one chunk at a time

    print("transmit")
    return res

def run_variational_bayes_conditiononly_zip(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(simple_worker_zip_conditiononly, chunks)) # each worker gets one chunk at a time

    print("transmit")
    return res

def run_variational_bayes_conditiononly_norandslopes_zip(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(simple_worker_zip_conditiononly_norandslopes, chunks)) # each worker gets one chunk at a time

    print("transmit")
    return res


def run_variational_bayes_conditiononly_norandslopes(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(simple_worker_conditiononly_norandslopes, chunks)) # each worker gets one chunk at a time

    print("transmit")
    return res


def run_variational_bayes_conditiononly_norandslopes_reduced_full(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(simple_worker_conditiononly_norandslopes_reduced_full, chunks)) # each worker gets one chunk at a time

    print("transmit")
    return res


def run_variational_bayes_norandslopes_counterfactual(chunks):
    import subprocess
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    with multiprocessing.get_context('spawn').Pool() as pool:
        res = list(pool.map(simple_worker_norandslopes_counterfactual, chunks)) # each worker gets one chunk at a time

    print("transmit")
    subprocess.call(["rm", "-rf", "/export/vrishab/compile"])
    print("clear cache")
    return res


def run_variational_bayes_clang(chunks):
    # receives a chunk of the connectivity data 
    # (link pairs and associated data) and splits the
    # task among the cores available on the machine.
    # aggregates the results and returns them to server

    pytensor.config.cxx = '/usr/bin/clang++'

    with Pool() as pool:
        res = list(pool.imap(worker, chunks)) # each worker gets one chunk at a time

    return res
