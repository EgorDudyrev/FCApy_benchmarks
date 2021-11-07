from typing import Tuple, Dict
import os
import numpy as np
import pandas as pd
from itertools import product

from fcapy import LIB_INSTALLED
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from fcapy.visualizer import ConceptLatticeVisualizer


def load_classic_context(contexts_to_test: Tuple[str], tmp_dirname: str = 'tmp_contexts') -> Dict[str, pd.DataFrame]:
    try:
        os.system(f"rm -rf {tmp_dirname}")
    except FileNotFoundError as e:
        pass
    os.mkdir(tmp_dirname)

    frames = {}
    for K_name in contexts_to_test:
        fname = f"{tmp_dirname}/{K_name}.cxt"
        os.system(f"wget -O {fname} https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/{K_name}.cxt")
        df = FormalContext.read_cxt(fname).to_pandas()
        df.name = K_name
        frames[K_name] = df

    os.system(f"rm -rf {tmp_dirname}")
    return frames


def load_bob_ross_dataframe(K_name: str = 'bob_ross') -> pd.DataFrame:
    fname = f"{K_name}.csv"
    link = "https://raw.githubusercontent.com/fivethirtyeight/data/master/bob-ross/elements-by-episode.csv"
    os.system(f"wget -O {fname} -q {link}")
    df = pd.read_csv(fname)
    df['EPISODE_TITLE'] = df['EPISODE']+' '+df['TITLE']
    df = df.drop(['EPISODE','TITLE'],1).set_index('EPISODE_TITLE').astype(bool)
    df.name = K_name
    os.remove(fname)
    return df


def generate_random_contexts(
        n_objects_vars: Tuple[int], n_attributes_vars: Tuple[int], densities_vars: Tuple[float],
        random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    np.random.seed(random_state)

    frames = {}
    for comb in product(n_objects_vars, n_attributes_vars, densities_vars):
        n_objects, n_attributes, density = comb

        df = pd.DataFrame(np.random.binomial(1, density, size=(n_objects, n_attributes), ))
        df.columns = [f"m_{i}" for i in df.columns]
        df.index = [f"g_{i}" for i in df.index]
        df = df.astype(bool)

        df.name = f"random_{n_objects}_{n_attributes}_{density}"
        frames[df.name] = df
    return frames
