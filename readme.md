# Multiprocess Linkwise Regression and Post-Hoc Network Enhancement
Utilizes a mapreduce-type formulation to farm multiple single-link regressions
out to worker machines, and then collects the results into statistic maps.

Null (counterfactual) models are derived from the linkwise regressions using the
specified intervention terms. These also produce statistic maps.

The statistic maps are enhanced via nbs-TFCE and significance is assessed using
a max-statistic test.

General usage:
```
# df: pd.DataFrame
# src: mne.SourceSpaces

nt = nr.NetworkTest(src)

formulae = {
    'zero': "connectivity ~ session + (1|subject)",
    'nonzero': "connectivity ~ session + (1|subject)"
}

regressors = ['subject', 'session', 'J']

priors = {
    'zero': { # zero support
        "Intercept": bmb.Prior("Normal", mu=0, sigma=3),
        "session": bmb.Prior("Normal", mu=0, sigma=1),

        # distributional params
        "sigma": bmb.Prior("Exponential", lam=3),
        "subject|Intercept": bmb.Prior("Exponential", lam=4),
    },
    'nonzero': { # nonzero support
        "Intercept": bmb.Prior("Normal", mu=0, sigma=3),
        "session": bmb.Prior("Normal", mu=0, sigma=1),

        # distributional params
        "sigma": bmb.Prior("Exponential", lam=3),
        "subject|Intercept": bmb.Prior("Exponential", lam=4),
    },
}

interventions = {
    "session": [0],
}

nt.fit(df, 
       formulae=formulae,
       regressors=regressors, 
       priors=priors, 
       interventions=interventions)

server = nt.serve()

nt.submit() # spool workers in separate terminal via `eelfarm start 'localhost'`

# wait a long time...

nt.collect()

nt.infer() # returns enhanced statmaps, monte-carlo p values, etc.
```