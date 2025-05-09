import numpy as np
from ..utils.mutation_matrix import normalize_matrix
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numpy.random import default_rng

from sklearn.decomposition import NMF

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
    def __init__(self, n_components: int, resample_method: str = 'poisson', objective_function: str = 'frobenius', initialization_method: str = 'random', normalization_method: str = 'GMM', max_iter: int = 1000000, num_factorizations: int = 100, random_state: int = 42, tolerance: float = 1e-15, verbose: bool = False):
        """Initialize NMF model parameters."""
        self.n_components = n_components
        self.random_state = random_state
        self.objective_function = objective_function
        self.num_factorizations = num_factorizations
        self.resample_method = resample_method
        self.normalization_method = normalization_method
        self.initialization_method = initialization_method
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.verbose = verbose

    def _random_initialization(self, X: np.ndarray, seed: int = None) -> tuple[np.ndarray, np.ndarray]:
        n_features, n_samples = X.shape
        rng = default_rng(seed)
        S = rng.uniform(low=1e-8, high=1.0 - 1e-8, size=(n_features, self.n_components))
        A = rng.uniform(low=1e-8, high=1.0 - 1e-8, size=(self.n_components, n_samples))
        return S, A

    def _fit(self, X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Run NMF and return S (samples x signatures) and A (signatures x features).
        
        Objective functions:
            - fro: Frobenius
            - kl: Kullback-Leibler
            - is: Itakura-Saito
        """
        # use custom random init
        if self.initialization_method == 'random':
            S, A = self._random_initialization(X, seed)
            model = NMF(
                n_components=self.n_components,
                init='custom',
                solver='mu',
                beta_loss=self.objective_function,
                random_state=seed,
                tol=self.tolerance,
                max_iter=self.max_iter
            )
            S = model.fit_transform(X, W=S, H=A)
        else:    
            model = NMF(
                n_components=self.n_components,
                init=self.initialization_method,
                solver='mu',
                beta_loss=self.objective_function,
                random_state=seed,
                tol=self.tolerance,
                max_iter=self.max_iter
            )
            S = model.fit_transform(X)
        A = model.components_
        err = model.reconstruction_err_
        n_iter = model.n_iter_

        # if err > self.tolerance:
        #     print(f"Warning: NMF did not converge. Error: {err}")

        return (S, A, err, n_iter)
        
    def _resample(self, X: np.ndarray, seed: int) -> np.ndarray:
        """Resample the data based on the specified method
        
        multinomial resample method: https://www.cell.com/cell-reports/fulltext/S2211-1247(12)00433-0#sec-4
        """
        np.random.seed(seed)
        if self.resample_method == 'poisson':
            # Poisson resampling, each entry in X is resampled from a Poisson distribution of the same mean
            return np.random.poisson(X)
        elif self.resample_method == 'bootstrap':
            return self._multinomial_bootstrap(X, seed)
        else:
            raise ValueError(f"Unknown resample method: {self.resample_method}, please use 'poisson' or 'bootstrap'")

    def _multinomial_bootstrap(self, X: np.ndarray, seed: int) -> np.ndarray:
        """
        Apply per-column multinomial bootstrap resampling to matrix M.

        Preserves the column sums of the original matrix M so each person will have the same number of mutations after resampling.
        
        Parameters:
        - X: np.ndarray of shape (n_types, n_samples), integer mutation counts
        
        Returns:
        - X_boot: np.ndarray of shape (n_types, n_samples), resampled counts
        """
        rng = np.random.default_rng(seed)
        n_types, n_samples = X.shape
        X_boot = np.zeros_like(X)

        for j in range(n_samples):
            col = X[:, j]
            total = col.sum()
            if total == 0:
                continue
            probs = col / total
            X_boot[:, j] = rng.multinomial(total, probs)
        
        return X_boot
    
    def _single_factorization(self, X: np.ndarray):
        """Wrapper to run one NMF replicate with a given seed."""
        X_resampled = self._resample(X)
        X_normalized = normalize_matrix(X_resampled, method=self.normalization_method)
        S, A, err, n_iter = self._fit(X_normalized)
        return S, A, err, n_iter


    def run(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run NMF decomposition on the input matrix X with defined NMF model parameters
        
        Parameters:
        - X: np.ndarray of shape (n_types, n_samples), original mutation matrix

        Returns:
        - Tuple of two numpy arrays: (S_all, A_all)
            - S_all: np.ndarray of shape (num_factorizations, n_types, n_components), all S matrices
            - A_all: np.ndarray of shape (num_factorizations, n_components, n_samples), all A matrices
        """
        S_all = []
        A_all = []
        err_all = []
        n_iter_all = []
        print(f"Running NMF with following parameters:", flush=True)
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
        print("", flush=True)
        
        for i in range(self.num_factorizations):
            if self.verbose:
                print(f"Running NMF factorization k={self.n_components}, iteration={i + 1}/{self.num_factorizations}", flush=True)
            seed = self.random_state + i
            X_resampled = self._resample(X, seed)
            X_normalized = normalize_matrix(X_resampled, method=self.normalization_method)
            S, A, err, n_iter = self._fit(X_normalized, seed)

            if np.isnan(err):
                if self.verbose:
                    print(f"⚠️  Skipping iteration {i + 1} due to NaN reconstruction error", flush=True)
                continue

            S_all.append(S)
            A_all.append(A)
            err_all.append(err)
            n_iter_all.append(n_iter)
        
        return np.array(S_all), np.array(A_all), np.array(err_all), np.array(n_iter_all)
    