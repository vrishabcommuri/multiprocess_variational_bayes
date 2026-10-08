from dataclasses import dataclass, field
from typing import Any

import eelfarm
import numpy as np
import pandas as pd

from network_regression.contrast import ContrastConfig, Contrast
from network_regression.dataops import ScaledData, extract_single_link_data
from network_regression.linkwise_regression import counterfactual_run_general_worker


@dataclass
class Job:
    src: int
    targ: int
    contrastconfig: ContrastConfig
    contrast: Contrast
    linkdata: pd.DataFrame
    posteriormode: str | None = 'positiveconditional' # return mu only, save mem


@dataclass
class Chunk:
    jobs: list[Job] = field(default_factory=list)


@dataclass
class FarmConfig:
    n_compute_groups: int = 50   # each is a set of cores assigned to a task
    n_cores: int = 10            # cores per group
    server_ip: str = 'localhost'        
    results_dir: str = './'
    compile_dir: str = '~/compile'


def get_link_pairs(data: ScaledData) -> np.array:
    data_count = data.data_count
    usemask = data.usemask
    valmask = data.valmask

    pairs = list(zip(*np.where(data_count[usemask].sum(axis=0) * valmask)))
    return pairs

    
def chunk_data(
        data: ScaledData, 
        df: pd.DataFrame, 
        farmconfig: FarmConfig, 
        contrastconfig: ContrastConfig,
        contrast: Contrast,
        posteriormode: str | None = 'positiveconditional',
    ) -> list[Chunk]:
    """
    splits links into segments that define a chunk. chunks are appropriately
    sized to be farmed out to individual cores
    """
    pairs = get_link_pairs(data)

    splits = np.array_split(pairs, farmconfig.n_compute_groups * \
                                   farmconfig.n_cores) 

    chunks = []
    for splitpairs in splits:
        jobs = []
        for src, targ in splitpairs:
            linkdata, subsetcontrast = extract_single_link_data(data, contrast, 
                                                                df, src, targ, 
                                                                eps=0.01)

            job = Job(
                src = src,
                targ = targ,
                linkdata = linkdata,
                contrastconfig = contrastconfig,
                contrast = subsetcontrast,
                posteriormode = posteriormode,
            )
            jobs.append(job)

        chunk = Chunk(jobs)
        chunks.append(chunk)

    return chunks

def start_server(farmconfig: FarmConfig) -> Any:
    server_ip = farmconfig.server_ip
    server = eelfarm.start_server(server_ip)
    return server


def send_chunks(
        server: Any,
        chunks: list[Chunk], 
        farmconfig: FarmConfig,
    ) -> int:   

    n_cores = farmconfig.n_cores
    resultsdir = farmconfig.results_dir
    compiledir = farmconfig.compile_dir

    n_sent = (len(chunks)//n_cores) + 1

    for midx in range(n_sent):
        datastart = midx * n_cores 
        dataend = (midx + 1) * n_cores

        dst = f"{resultsdir}/machine_{midx}_chunkdata.pickle"
        print(f"starting job {dst}")

        try:
            server.put(
                dst, 
                counterfactual_run_general_worker, 
                chunks=chunks[datastart:dataend],
                compiledir=compiledir,
            )
        except Exception as exc:
            print(exc, "continuing")
            continue

    return n_sent



