
import pandas as pd
import numpy as np
from src.models.nmf_runner_parallel import NMFDecomposer
import time
import matplotlib.pyplot as plt
import pickle
from src.models.clustering import consensus_signatures
import os
import multiprocessing

# set this to True to rerun the NMF
APPEND_RESULTS = True # set this to True if you want to append results to existing results

OUTPUT='data/nmf_runs/01_nmf_k_selection_K2-10_1E-8_KL.runs'

# max K
MAX_K = 10

def compare_dicts(d1, d2):
    mismatches = {}
    all_keys = set(d1.keys()).union(d2.keys())
    for key in all_keys:
        v1 = d1.get(key, '<missing>')
        v2 = d2.get(key, '<missing>')
        if v1 != v2:
            if key == 'n_components':
                continue
            mismatches[key] = (v1, v2)
    return mismatches


def main():
    df_sbs = pd.read_csv('data/processed/BRCA.SBS96.all', sep='\t', index_col=0)
    X = np.array(df_sbs)


    NMF_PARAMS = {
        'resample_method': 'poisson',
        'objective_function': 'kullback-leibler',
        'initialization_method': 'random',
        'normalization_method': 'GMM',
        'max_iter': 1000000,
        'num_factorizations': 100,
        'random_state': 42,
        'tolerance': 1e-8
    }

    if APPEND_RESULTS:
        try:
            with open(OUTPUT, 'rb') as f:
                results = pickle.load(f)
                print('Loaded NMF results from file to append NMF runs to', flush=True)
        except:
            print(f"No existing file, making new file {OUTPUT}")
            results = {}
            results['params'] = NMF_PARAMS
        mismatches = compare_dicts(results['params'], NMF_PARAMS)
        if mismatches:
            print("⚠️ NMF parameter mismatch:")
            for key, (v1, v2) in mismatches.items():
                print(f" - {key}: saved={v1} vs current={v2}")
            raise ValueError("Parameters in loaded file do not match with new parameters")

    else:
        print(f"Not appending data. Creating new file {OUTPUT}")
        results = {}
        results['params'] = NMF_PARAMS

    start_k = max(2,len(results) + 1)
    for k in range(start_k,MAX_K+1):
        print(f'Running NMF with k={k}', flush=True)
        NMF_PARAMS['n_components'] = k
        nmf_model = NMFDecomposer(**NMF_PARAMS, verbose=True, n_jobs=max(1, multiprocessing.cpu_count() - 1))
        #nmf_model = NMFDecomposer(**NMF_PARAMS, verbose=True)
        time_start = time.time()
        S_all, A_all, err_all, n_iter_all = nmf_model.run(X)
        time_end = time.time()
        print(f'Finished NMF with k={k}, runtime: {time_end - time_start}', flush=True)
        results[k] = {
            'S_all': S_all,
            'A_all': A_all,
            'err_all': err_all,
            'n_iter_all': n_iter_all,
            'time': time_end - time_start
        }
        with open(OUTPUT, 'wb') as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    main()