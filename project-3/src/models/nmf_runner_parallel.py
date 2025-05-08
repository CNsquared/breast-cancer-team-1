import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from sklearn.decomposition import NMF
from ..utils.mutation_matrix import normalize_matrix  # Adjust as needed


def _single_factorization_static(args):
    X, k, seed, resample_method, normalization_method, objective_function, initialization_method, max_iter, tolerance = args

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Resample
    if resample_method == 'poisson':
        X_resampled = np.random.poisson(X)
    elif resample_method == 'bootstrap':
        X_boot = np.zeros_like(X)
        for j in range(X.shape[1]):
            col = X[:, j]
            total = col.sum()
            if total == 0:
                continue
            probs = col / total
            X_boot[:, j] = rng.multinomial(total, probs)
        X_resampled = X_boot
    else:
        raise ValueError(f"Unknown resample method: {resample_method}")

    # Normalize
    X_normalized = normalize_matrix(X_resampled, method=normalization_method)

    # Fit NMF
    model = NMF(
        n_components=k,
        init=initialization_method,
        solver='mu',
        beta_loss=objective_function,
        random_state=seed,
        max_iter=max_iter,
        tol=tolerance
    )
    S = model.fit_transform(X_normalized)
    A = model.components_
    err = model.reconstruction_err_
    n_iter = model.n_iter_

    return S, A, err, n_iter


class NMFDecomposer:
    
    """Non-negative Matrix Factorization (NMF) Decomposer.
    
    Should be able to use different objectve functions (e.g., Frobenius, Kullback-Leibler).


    Input:
    M (original mutational matrix: n_channels × n_samples)
        ↓
    Repeat 100 times:
    ┌──────────────────────────────────────────────────────────────┐
    │ 1. (Poisson) resample M → M'                                   │
    │ 2. Normalize M' (e.g., log, 100X, GMM)                       │
    │ 3. Run NMF on M' with fixed k = 5                            │
    │    - Initialize W, H (random or NNDSVD)                     │
    │    - Minimize KL divergence (or Frobenius, Itakura-Saito)   │
    │    - Use multiplicative update rules                        │
    │    - Stop after convergence or max_iters                    │
    │                                                              │
    │ Output per replicate:                                       │
    │   W_i: n_channels × k (signatures)                          │
    │   H_i: k × n_samples (activities)                           │
    └──────────────────────────────────────────────────────────────┘
        ↓
    Store:
    - All W_i (shape: 100 × n_channels × k)
    - All H_i (shape: 100 × k × n_samples)

    Initialization methods: {‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’}

    Resampling methods: {'poisson', 'bootstrap'}

    Normalization methods: {'GMM', '100X', 'log2', None}

    Objective functions: ‘frobenius’, ‘kullback-leibler’, ‘itakura-saito’
        Beta divergence to be minimized, measuring the distance between X and the dot product WH. 
        Note that values different from ‘frobenius’ (or 2) and ‘kullback-leibler’ (or 1) lead to significantly slower fits. 
        Note that for beta_loss <= 0 (or ‘itakura-saito’), the input matrix X cannot contain zeros. Used only in ‘mu’ solver.
    """
    def __init__(self, n_components: int, resample_method: str = 'poisson', objective_function: str = 'frobenius', initialization_method: str = 'random', normalization_method: str = 'GMM', max_iter: int = 1000000, num_factorizations: int = 100, random_state: int = 42, tolerance: float = 1e-15):
        self.n_components = n_components
        self.resample_method = resample_method
        self.objective_function = objective_function
        self.initialization_method = initialization_method
        self.normalization_method = normalization_method
        self.max_iter = max_iter
        self.num_factorizations = num_factorizations
        self.random_state = random_state
        self.tolerance = tolerance

    def run(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        args_list = [
            (
                X,
                self.n_components,
                self.random_state + i,
                self.resample_method,
                self.normalization_method,
                self.objective_function,
                self.initialization_method,
                self.max_iter,
                self.tolerance
            )
            for i in range(self.num_factorizations)
        ]

        with ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), self.num_factorizations)) as executor:
            results = list(executor.map(_single_factorization_static, args_list))

        S_all, A_all, err_all, n_iter_all = zip(*results)
        return (
            np.array(S_all),
            np.array(A_all),
            np.array(err_all),
            np.array(n_iter_all)
        )