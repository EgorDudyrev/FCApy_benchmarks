from typing import Tuple, Dict, Union, Callable, Any
import os
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt
import multiprocessing, multiprocessing.sharedctypes
from tqdm.notebook import tqdm

from fcapy import LIB_INSTALLED
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from fcapy.visualizer import ConceptLatticeVisualizer

import concepts

SupportedContextType = Union[concepts.Context, FormalContext]
SupportedLatticeType = Union[concepts.lattices.Lattice, ConceptLattice]


#######################
#      Load Data      #
#######################
def load_classic_context(
        contexts_to_test: Tuple[str, ...], tmp_dirname: str = 'tmp_contexts'
) -> Dict[str, pd.DataFrame]:
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
        n_objects_vars: Tuple[int, ...], n_attributes_vars: Tuple[int, ...], densities_vars: Tuple[float, ...],
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


# Context statistics
def get_context_stat(frame: pd.DataFrame) -> Dict[str, Any]:
    K_stat = {
        'ctx_name': frame.name,
        'n_objects': frame.shape[0], 'n_attributes': frame.shape[1],
        'n_connections': frame.sum().sum(),
        'density': frame.sum().sum()/(frame.shape[0]*frame.shape[1]),
        'is_random': frame.name.startswith('random')
    }
    return K_stat


###############################
#      Visualize Lattice      #
###############################
def visualize_by_concepts(K_names: Tuple[str, ...], frames: Dict[str, pd.DataFrame], fname_out_prefix: str):
    for K_name in K_names:
        df = frames[K_name]
        print(K_name)
        t1 = datetime.now()
        K = concepts.Context(df.index, df.columns, df.values)
        L = K.lattice
        print(f'Lattice constructed in {(datetime.now() - t1).total_seconds()} seconds')
        L.graphviz(f'{fname_out_prefix}_{K_name}', render=True)
        t2 = datetime.now()
        dt = (t2 - t1).total_seconds()
        print(f"Executed in {dt} seconds")


def visualize_by_fcapy(K_names: Tuple[str, ...], frames: Dict[str, pd.DataFrame], fname_out_prefix: str):
    for K_name in K_names:
        df = frames[K_name]
        print(K_name)

        t1 = datetime.now()
        K = FormalContext.from_pandas(df)
        L = ConceptLattice.from_context(K)
        print(f'Lattice constructed in {(datetime.now() - t1).total_seconds()} seconds')
        vsl = ConceptLatticeVisualizer(L)
        print(f'Visualizer constructed in {(datetime.now() - t1).total_seconds()} seconds')

        plt.title('Networkx lattice')
        vsl.draw_networkx()
        plt.savefig(f'{fname_out_prefix}_{K_name}.png')
        plt.close()
        print(f'Png saved in {(datetime.now() - t1).total_seconds()} seconds')


##################################
#      Test basic functions      #
##################################

# Construct context from dataframe
def construct_context_by_lib(frame: pd.DataFrame, lib_name: str) -> SupportedContextType:
    if lib_name == 'concepts':
        K = concepts.Context(frame.index, frame.columns, frame.values)
    elif lib_name == 'fcapy':
        K = FormalContext.from_pandas(frame)
    # elif lib_name == 'fcapsy':
    #    K = fcapsy.Context.from_pandas(frame)
    else:
        raise ValueError(f'Given library "{lib_name}" is not supported')

    return K


# Test intext, extent time
def test_intent_extent_time_by_func(
        objects: Tuple, attributes: Tuple,
        extent_func: Callable[[Tuple], Tuple], intent_func: Callable[[Tuple], Tuple],
        samples_per_size: int = 100
) -> Tuple[float, float]:
    times = []
    for arr, fnc in [(objects, intent_func), (attributes, extent_func)]:
        subsample_sizes = np.logspace(0, np.log(len(arr)) / np.log(10), 10)
        subsample_sizes = subsample_sizes.round(0).astype(int)
        np.random.seed(42)
        samples = [
            sample
            for size in subsample_sizes
            for sample in np.random.choice(arr, size=(samples_per_size, size))
        ]

        t1 = datetime.now()
        intents = [fnc(sample) for sample in samples]
        t2 = datetime.now()
        dt = (t2 - t1).total_seconds() / len(samples)
        times.append(dt)
    intent_time, extent_time = times

    return intent_time, extent_time


def test_intent_extent_time_by_lib(
        frame: pd.DataFrame, K: SupportedContextType, lib_name: str,
        samples_per_size: int = 100
) -> Tuple[float, float]:
    if lib_name == 'concepts':
        intent_time, extent_time = test_intent_extent_time_by_func(
            frame.index, frame.columns, K.extension, K.intension, samples_per_size)
    elif lib_name == 'fcapy':
        intent_time, extent_time = test_intent_extent_time_by_func(
            frame.index, frame.columns, K.extension, K.intention, samples_per_size)
    # elif lib_name == 'fcapsy':
    #    intent_time, extent_time = test_intent_extent_time_by_func(
    #        frame.index, frame.columns,
    #        lambda ar: K.down(K.Attributes(ar)),
    #        lambda ar: K.up(K.Objects(ar)),
    #        samples_per_size
    #    )
    else:
        raise ValueError(f'Given library "{lib_name}" is not supported')

    return intent_time, extent_time


def test_intent_extent_time_by_lib_multiprocess(
        frame: pd.DataFrame, K: SupportedContextType, lib_name: str,
        intent_time: multiprocessing.sharedctypes.Synchronized, extent_time: multiprocessing.sharedctypes.Synchronized,
        samples_per_size: int = 100
):
    intent_time.value, extent_time.value = test_intent_extent_time_by_lib(
        frame, K, lib_name, samples_per_size
    )


# Test time to construct concept lattice
def test_lattice_time_by_func(
        K: SupportedContextType, L_func: Callable[[SupportedContextType], SupportedLatticeType]
) -> float:
    t1 = datetime.now()
    L_func(K)
    t2 = datetime.now()
    dt = (t2-t1).total_seconds()
    return dt


def test_lattice_time_by_lib(K: SupportedContextType, lib_name: str):
    if lib_name == 'concepts':
        L_time = test_lattice_time_by_func(K, lambda ctx: ctx.lattice)
    elif lib_name == 'fcapy':
        L_time = test_lattice_time_by_func(K, lambda ctx: ConceptLattice.from_context(ctx))
    #elif lib_name == 'fcapsy':
    #    L_time = test_lattice_time_by_func(K, lambda ctx: fcapsy.Lattice.from_context(ctx))
    else:
        raise ValueError(f'Given library "{lib_name}" is not supported')

    return L_time


def test_lattice_time_by_lib_multiprocess(
        K: SupportedContextType, lib_name: str, L_time: multiprocessing.sharedctypes.Synchronized
):
    L_time.value = test_lattice_time_by_lib(K, lib_name)


def run_func_multiprocess(
        frame: pd.DataFrame, lib_name: str, timeout_seconds: multiprocessing.sharedctypes.Synchronized
) -> Dict[str, float]:
    K = construct_context_by_lib(frame, lib_name)

    L_time, intent_time, extent_time = [multiprocessing.Value('f', -1, lock=False) for _ in range(3)]

    p = multiprocessing.Process(
        target=test_intent_extent_time_by_lib_multiprocess,
        name=f"test_intent_extent_{lib_name}",
        args=[frame, K, lib_name, intent_time, extent_time, 1000])
    p.start()
    p.join(timeout_seconds)
    if p.is_alive():
        p.terminate()

    p = multiprocessing.Process(
        target=test_lattice_time_by_lib_multiprocess,
        name=f"test_lattice_{lib_name}",
        args=[K, lib_name, L_time])
    p.start()
    p.join(timeout_seconds)
    if p.is_alive():
        p.terminate()

    def neg1_to_none(multiprocess_var):
        return multiprocess_var.value if multiprocess_var.value != -1 else None

    stat = {
        'lattice_construction_time (secs)': neg1_to_none(L_time),
        'intent_time (secs)': neg1_to_none(intent_time),
        'extent_time (secs)': neg1_to_none(extent_time),
        'timeout_seconds': timeout_seconds,
    }
    return stat


def compute_stats(
        ctx_names_vals: Tuple[str, ...], lib_names_vals: Tuple[str, ...],
        n_runs: int, timeout_secs: float,
        frames: Dict[str, pd.DataFrame],
        tmp_stats_fname: str = 'benchmark_stats_tmp.csv'
) -> pd.DataFrame:
    run_number_vals = list(range(n_runs))
    all_combs = list(product(run_number_vals, ctx_names_vals, lib_names_vals))

    stats_df = pd.DataFrame(all_combs, columns=['run_number', 'ctx_name', 'lib_name'])
    stats_df['is_computed'] = False
    stats_df.to_csv(tmp_stats_fname)

    df_to_compute = stats_df[~stats_df['is_computed']][['run_number', 'ctx_name', 'lib_name']]

    for comb in tqdm(df_to_compute.iterrows(), total=len(df_to_compute)):
        stats_df = pd.read_csv(tmp_stats_fname, index_col=0)
        row_idx, (run_number, ctx_name, lib_name) = comb

        frame = frames[ctx_name]
        frame_stat = get_context_stat(frame)

        stat = run_func_multiprocess(frame, lib_name, timeout_seconds=timeout_secs)
        stat = dict(stat, **frame_stat)

        for k, v in stat.items():
            stats_df.loc[row_idx, k] = v
        stats_df.loc[row_idx, 'is_computed'] = True

        stats_df.to_csv('benchmark_stats_tmp.csv')

    stats_df = pd.read_csv(tmp_stats_fname, index_col=0)
    os.system(f"rm {tmp_stats_fname}")
    return stats_df
