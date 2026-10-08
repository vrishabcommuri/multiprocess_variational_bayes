from dataclasses import dataclass, field

import bambi as bmb
import numpy as np
import pandas as pd


@dataclass
class ContrastConfig:
    formulae: dict[str, str]    # user-provided formulae
    priors: dict[str, dict[str, bmb.Prior]]
    interventions: dict[str, list[int]]
    regressors: list[str] = field(default_factory=list)  # split regressor ids


@dataclass
class Contrast:
    group_A_regressor: str
    group_A_reference_level: int
    group_A_indices: np.array


def _triage_contrast(config: ContrastConfig) -> None:
    assert len(config.interventions.keys()) == 1, \
           "multivariate intervention not supported"
    
    ivar = next(iter(config.interventions.keys()))

    if isinstance(config.interventions[ivar], list):
        assert len(config.interventions[ivar]) > 0, \
           "intervention levels must be provided"

        assert np.all(np.equal(config.interventions[ivar], 
                           config.interventions[ivar][0])), \
           "all intervention levels must equal the reference level code"
    
        # if the regressor levels are binary, then intervention must be a
        # scalar, not list
        if len(config.interventions[ivar]) == 1:
            config.interventions[ivar] = config.interventions[ivar][0]
    

def build_contrast(df: pd.DataFrame, config: ContrastConfig) -> Contrast:
    _triage_contrast(config)
    
    group_A_regressor = next(iter(config.interventions.keys()))
    if isinstance(config.interventions[group_A_regressor], list):
        group_A_reference_level = config.interventions[group_A_regressor][0]
    else:
        group_A_reference_level = config.interventions[group_A_regressor]

    group_A_indices = df[group_A_regressor].values == group_A_reference_level

    return Contrast(
        group_A_regressor = group_A_regressor,
        group_A_reference_level = group_A_reference_level,
        group_A_indices = group_A_indices,
    )

