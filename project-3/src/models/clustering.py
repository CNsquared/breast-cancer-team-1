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
                         stability_threshold=0.8,
                         min_sil=0.2):
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
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_sigs_norm)
    labels = kmeans.labels_

    # 4) compute stable centroids based on silhouette stability
    sil_samples = silhouette_samples(all_sigs_norm, labels, metric='cosine')

    centroids = []
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        sil_vals = sil_samples[idx]
        avg_sil = sil_vals.mean()
        min_sil = sil_vals.min()
        # require average stability ≥0.80 and no individual stability <0.20
        if avg_sil >= stability_threshold and min_sil >= min_sil:
            members = all_sigs[idx]
            c = members.mean(axis=0)
            c = c / (c.sum() + 1e-12)
            centroids.append(c)

    # 5) global silhouette score
    sil_score = silhouette_score(all_sigs_norm, labels, metric='cosine')

    # 6) reconstruction error: solve X ≈ W @ S_cons.T via NNLS
    #    centroids is list of (n_features,), stack into (n_features x c)
    if len(centroids) == 0:
        recon_err = np.nan
    else:
        S_cons = np.stack(centroids, axis=1)   # shape (n_features x c)
        n_samples = X.shape[0]
        W = np.zeros((n_samples, S_cons.shape[1]))
        for i in range(n_samples):
            # solve S_cons @ w_i = X[i,:]
            w_i, _ = nnls(S_cons, X[i, :])
            W[i, :] = w_i
        X_hat = W.dot(S_cons.T)
        numer   = np.linalg.norm(X - X_hat, 'fro')
        denom   = np.linalg.norm(X,       'fro') + 1e-12
        recon_err = numer / denom

    return centroids, sil_score, recon_err
