import numpy as np
from scipy.optimize import linear_sum_assignment, nnls
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

def align_run_to_ref(S_ref, S_run):
    sim      = cosine_similarity(S_ref.T, S_run.T)
    cost     = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    return S_run[:, col_ind]

def consensus_signatures(X, S_runs, k,
                         average_threshold=0.8,
                         minimum_threshold=0.2):
    """
    X:       (n_samples x n_features) original data matrix
    S_runs:  list of (n_features x k) S-matrices from repeated NMF
    returns: centroids, silhouette_score, reconstruction_error
    """
    # 1) align all runs to the first
    S_ref = S_runs[0]
    # ensure X is shaped (n_samples, n_features) so features match S_ref rows
    if X.shape[1] != S_ref.shape[0]:
        X = X.T
    S_aligned = [ align_run_to_ref(S_ref, S) for S in S_runs ]

    # 2) stack all columns into shape (n_runs*k, n_features)
    all_sigs = np.hstack(S_aligned).T
    # normalize rows so Euclidean ~ cosine
    all_sigs_norm = all_sigs / (all_sigs.sum(axis=1, keepdims=True) + 1e-12)

    # 3) k-means clustering
    # 3) k-means with equal‐size clusters
    n_points = all_sigs_norm.shape[0]
    if n_points % k != 0:
        raise ValueError(f"Cannot partition {n_points} points into {k} equal clusters")
    size = n_points // k

    # initialize centroids with regular k‐means
    init_km = KMeans(n_clusters=k, random_state=0).fit(all_sigs_norm)
    centers = init_km.cluster_centers_

    # compute squared distances to each centroid
    dist2 = np.sum((all_sigs_norm[:, None, :] - centers[None, :, :])**2, axis=2)

    # greedy assignment: pick the smallest distance edges first
    labels = -np.ones(n_points, dtype=int)
    counts = np.zeros(k, dtype=int)
    rows, cols = np.unravel_index(np.argsort(dist2.ravel()), dist2.shape)
    for i, j in zip(rows, cols):
        if labels[i] < 0 and counts[j] < size:
            labels[i] = j
            counts[j] += 1
        if np.all(labels >= 0):
            break

    # wrap assigned labels in a dummy object to mimic sklearn API
    class DummyKMeans:
        pass
    kmeans = DummyKMeans()
    kmeans.labels_ = labels
    labels = kmeans.labels_

    # 4) compute stable centroids based on silhouette stability
    sil_samples = silhouette_samples(all_sigs_norm, labels, metric='cosine')

    centroids = []
    stable_centroids = []
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        sil_vals = sil_samples[idx]
        avg_sil = sil_vals.mean()
        min_sil = sil_vals.min()
        # require average stability ≥0.80 and no individual stability <0.20
        print(f"Cluster {cid}: avg_sil={avg_sil:.3f}, min_sil={min_sil:.3f}")
        members = all_sigs[idx]
        c = members.mean(axis=0)
        c = c / (c.sum() + 1e-12)
        if avg_sil >= average_threshold and min_sil >= minimum_threshold:
            stable_centroids.append(c)
        centroids.append(c)
            
    print(len(stable_centroids), "stable centroids found")

    # 5) global silhouette score
    sil_score = silhouette_score(all_sigs_norm, labels, metric='cosine')

   

    return stable_centroids, centroids,  sil_score
