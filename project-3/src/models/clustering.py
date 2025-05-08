import numpy as np


import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def align_run_to_ref(S_ref, S_run):
    """
    Permute columns of S_run to best match S_ref.
    Both are (96 x k) arrays.
    """
    # compute cosine similarities between columns of S_ref and S_run
    sim = cosine_similarity(S_ref.T, S_run.T)
    # use Hungarian algorithm on cost = 1 - similarity
    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    # reorder columns of S_run to line up with S_ref
    return S_run[:, col_ind]
    
   

def consensus_signatures(S_runs, k, stability_threshold=0.8, run_threshold=0.8):
    """
    W_list: list of 100 aligned (96 x k) arrays
    returns: centroids of stable clusters (list of 96-vectors)
    """
    
    S_runs_aligned = []
    S_ref = S_runs[0]
    for S in S_runs:
        S_runs_aligned.append(align_run_to_ref(S_ref, S))
    
    
    # Stack all columns: shape (100*k, 96)
    all_sigs = np.hstack(S_runs_aligned).T  # now (100*k, 96)
    
    # Cluster into k clusters using k-means on cosine-distance features
    # We use normalized vectors so Euclidean ≈ cosine
    all_sigs_norm = all_sigs / (all_sigs.sum(axis=1, keepdims=True) + 1e-12)
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_sigs_norm)
    labels = kmeans.labels_

    centroids = []
    for cluster_id in range(k):
        members = all_sigs[labels == cluster_id]
        
        # Compute stability metrics
        # 1) fraction of runs present
        #    each member knows its run by index // k
        runs = {idx // k for idx in np.where(labels == cluster_id)[0]}
        frac_runs = len(runs) / len(S_runs_aligned)
        
        # 2) mean pairwise cosine
        sims = cosine_similarity(members)
        # exclude self-similarity by subtracting identity
        n = sims.shape[0]
        mean_sim = (sims.sum() - n) / (n*(n-1)) if n>1 else 1.0

        if frac_runs >= run_threshold and mean_sim >= stability_threshold:
            # compute centroid and renormalize
            centroid = members.mean(axis=0)
            centroid /= centroid.sum()
            centroids.append(centroid)
            
            
    return centroids

