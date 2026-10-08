import bambi as bmb  # noqa: I001
import mne
import pandas as pd

from network_regression.contrast import ContrastConfig, build_contrast
from network_regression.dataops import DataOpsConfig, data_rescale_threshold
from network_regression.farm import chunk_data, start_server, send_chunks
from network_regression.transform import (TransformConfig, 
                                          downsample_smooth_normalize)
from network_regression.farm import FarmConfig
from network_regression.statmaps import marshal
from network_regression.tfce import nbstfce_test
from network_regression.linkwise_regression import N_CHAINS, N_DRAWS
import copy

class NetworkTest:
    def __init__(self, 
                 sourcespace: mne.SourceSpaces,
                 dataopsconfig: DataOpsConfig | None = None, 
                 transformconfig: TransformConfig | None = None,
                 farmconfig: FarmConfig | None = None,
        ) -> None:

        if dataopsconfig is None:
            dataopsconfig = DataOpsConfig()

        if transformconfig is None:
            transformconfig = TransformConfig()

        if farmconfig is None:
            farmconfig = FarmConfig()

        self.dataopsconfig = dataopsconfig
        self.transformconfig = transformconfig
        self.farmconfig = farmconfig
        self.sourcespace = sourcespace

    def fit(self, 
            df: pd.DataFrame, 
            formulae: dict[str, str],    
            regressors: list[str],
            priors: dict[str, dict[str, bmb.Prior]],
            interventions: dict[str, list[float]],
            posteriormode: str | None = 'applycontrastpositiveconditional',
        ):
        """
        initial fit.
        
        1. Set up configurations for various internal classes
        2. Marshal the data into dataframe objects, appropriately constructed to
           reflect the regressor variables and intervention arms chosen
        3. Evalute prior hysteresis and reconcile prior data scale with data
           scale
        4. Chunk data for farming.
        """
        contrastconfig = ContrastConfig(
            formulae=formulae,
            regressors=regressors,
            priors=priors,
            interventions=interventions
        )

        contrast = build_contrast(copy.deepcopy(df), 
                                  contrastconfig)
        self.contrast = contrast

        df_dsn = downsample_smooth_normalize(copy.deepcopy(df), 
                                             self.sourcespace, 
                                             self.transformconfig)

        scaleddata = data_rescale_threshold(df_dsn, self.dataopsconfig)

        chunks = chunk_data(scaleddata, df_dsn, self.farmconfig, 
                            contrastconfig, contrast, posteriormode)

        self.chunks = chunks

    def serve(self):
        server = start_server(self.farmconfig)
        self.server = server
        return server 

    def submit(self):
        """
        farm jobs to cluster
        """
        self.n_files = send_chunks(self.server, self.chunks, self.farmconfig)

    def collect(self):
        """
        collect farmed jobs and apply contrast to form statmaps
        """
        n_files = self.n_files
        resultsdir = self.farmconfig.results_dir
        statmaps = marshal(n_files, resultsdir, self.transformconfig, 
                           self.contrast)
        self.statmaps = statmaps
        return statmaps

    def infer(self):
        """
        TFCE and max-statistic testing
        """
        statistic = self.statmaps.statmap
        null_distribution = self.statmaps.statmap_null

        nbstfce_test(statistic, 
                     null_distribution, 
                     E = 0.75, 
                     H = 3.25, 
                     n_perm = N_CHAINS*N_DRAWS, 
                     intensity_extent = True, 
                     hypothesis = 'two-sided', 
                     verbose = False)