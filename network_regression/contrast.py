from dataclasses import dataclass
from typing import Any

import bambi as bmb
import numpy as np
import pandas as pd


@dataclass
class ContrastConfig:
    formulae: dict[str, str]    # user-provided formulae
    priors: dict[str, dict[str, bmb.Prior]]
    interventions: dict[str, list[int]] 
    contrast: dict[str, Any]


@dataclass
class Contrast:
    group_A_regressor: str
    group_A_reference_level: int
    group_A_indices: np.array
    nonzeroterm_indexed_regressors: list[str]
    zeroterm_indexed_regressors: list[str]


def _triage_formulae(form):
    assert len(list(form.keys())) == 2
    assert 'zero' in list(form.keys())
    assert 'nonzero' in list(form.keys())


def _triage_intervention(config: ContrastConfig) -> None:
    _triage_formulae(config.formulae)

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


def sanitize_formulae(form):
    zeroterm_indexed_regressors = [i.replace("[i->j]", "") \
                          for i in form['zero'].split(" ") if "[i->j]" in i]
    nonzeroterm_indexed_regressors = [i.replace("[i->j]", "") \
                          for i in form['nonzero'].split(" ") if "[i->j]" in i]
    
    form['zero'] = form['zero'].replace("[i->j]", "")
    form['nonzero'] = form['zero'].replace("[i->j]", "")
    return nonzeroterm_indexed_regressors, zeroterm_indexed_regressors


def build_contrast(df: pd.DataFrame, config: ContrastConfig) -> Contrast:
    _triage_intervention(config)

    nz_idxd_regressors, z_idxd_regressors = sanitize_formulae(config.formulae)

    return Contrast(
        group_A_regressor = None, #group_A_regressor,
        group_A_reference_level = None, #group_A_reference_level,
        group_A_indices = None, #group_A_indices,
        nonzeroterm_indexed_regressors = nz_idxd_regressors,
        zeroterm_indexed_regressors = z_idxd_regressors,
    )


def single_link_contrast_groups(
        df: pd.DataFrame,
        contrast: Contrast, 
        config: ContrastConfig
    ) -> Contrast:
    group_A_regressor = next(iter(config.contrast.keys()))
    group_A_reference_level = config.contrast[group_A_regressor]    
    assert isinstance(group_A_reference_level, str|int|float)
    
    if isinstance(df[group_A_regressor].values[0], str|int):
        assert group_A_reference_level in df[group_A_regressor].unique(), \
            f"reference level {group_A_reference_level} must be one of the " \
            f"levels of factor {group_A_regressor}"

    # cutpoint
    group_A_indices = df[group_A_regressor].values == group_A_reference_level

    contrast.group_A_indices = group_A_indices
    contrast.group_A_regressor = group_A_regressor
    contrast.group_A_reference_level = group_A_reference_level

    return contrast

