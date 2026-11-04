import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from network_regression.contrast import (
    Contrast,
    ContrastConfig,
    single_link_contrast_groups,
)


@dataclass
class DataOpsConfig:
    support_percentile: int = 50
    min_count: int = 25
    boost: int = 10_000           # offset applied to observation data
    observation_column: str = 'B' # default behavior derives 'B' from 'J' stats
    eps: float|None = 0.01

@dataclass 
class ScaledData:
    data_count: np.array
    valmask: np.array
    usemask: np.array


def data_rescale_threshold(
        df: pd.DataFrame, 
        config: DataOpsConfig
    ) -> ScaledData:
    
    data_raw = np.stack(df[config.observation_column].values)
    totals = np.stack(df['total'].values)

    # this is important for log scale models since small values become
    # numerically difficult to sample. the data rescaling does not affect the
    # posterior means or variances beyond a strictly arbitrary scaling factor
    boost = config.boost

    data_count = np.nan_to_num(data_raw)
    data_count = data_raw * boost

    # index nan data
    usemask = (~np.isnan(data_raw).sum(axis=1).sum(axis=1).astype(bool)) &\
              (totals > config.min_count)
    
    total_counts = data_count[usemask].sum(axis=0)

    # keep top 50%. this establishes a support that identifies link locations
    # that have enough data to fit and interpret the single-link regressions.
    # removing the support would ill-condition regression models with a paucity
    # of data and would substantially increase computational cost
    threshold = np.percentile(total_counts, config.support_percentile)  

    mask = total_counts > threshold
    valmask = mask & mask.T # zero both directions
    
    return ScaledData(
        data_count=data_count, 
        valmask=valmask, 
        usemask=usemask,
    )

def extract_single_link_data(
        data: ScaledData, 
        contrast: Contrast,
        df: pd.DataFrame, 
        src: int, 
        targ: int, 
        dataopsconfig: DataOpsConfig,
        contrastconfig: ContrastConfig,
    ) -> pd.DataFrame:
    
    data_count = data.data_count
    useidxs = data.usemask

    df = df.copy()
    subsetcontrast = copy.deepcopy(contrast)
    
    # replace connectivity data for this link with preprocessed values
    df[dataopsconfig.observation_column] = data_count[:, targ, src]

    if dataopsconfig.observation_column != 'B':
        df['B'] = np.stack(df['B'].values)[:, targ, src]
    if dataopsconfig.observation_column != 'J':
        df['J'] = np.stack(df['J'].values)[:, targ, src]

    idxd_regs = contrast.nonzeroterm_indexed_regressors + \
                contrast.zeroterm_indexed_regressors
    
    for indexed_regressor in np.unique(idxd_regs):
        # these were just sliced
        if indexed_regressor in ['J', 'B', dataopsconfig.observation_column]:
            continue
        
        df[indexed_regressor] = np.stack(df[indexed_regressor].values)\
                                         [:, targ, src]

    if dataopsconfig.eps:
        df.loc[df[dataopsconfig.observation_column] < dataopsconfig.eps, 
               dataopsconfig.observation_column] = 0 
        
    subsetcontrast = single_link_contrast_groups(df, 
                                                 subsetcontrast, 
                                                 contrastconfig)

    # drop trials with too few links
    df = df.iloc[useidxs]
    subsetcontrast.group_A_indices = subsetcontrast.group_A_indices[useidxs]

    return df, subsetcontrast