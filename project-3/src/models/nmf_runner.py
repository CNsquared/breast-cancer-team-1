import numpy as np
from ..utils.mutation_matrix import normalize_matrix

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

    Initialization methods:
    - random (default): random initialization
    - nndsvd: Non-negative Double Singular Value Decomposition (NNDSVD)

    Resampling methods:
    - poisson: Poisson resampling (default)
    - bootstrap: multinomial Bootstrap resampling

    Normalization methods:
    - GMM: Gaussian mixture model (default)
    - 100X
    - log2
    - None: no normalization.

    Objective functions:
    - fro: Frobenius
    - kl: Kullback-Leibler
    - is: Itakura-Saito
    """
    def __init__(self, n_components: int, resample_method: str = 'poisson', objective_function: str = 'fro', initialization_method: str = 'random', normalization_method: str = 'GMM', num_factorizations = 100, random_state: int = 42):
        """Initialize NMF model parameters."""
        self.n_components = n_components
        self.random_state = random_state
        self.objective_function = objective_function
        self.num_factorizations = num_factorizations
        self.resample_method = resample_method
        self.normalization_method = normalization_method
        self.initialization_method = initialization_method

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run NMF and return S (samples x signatures) and A (signatures x features)."""
        # TODO: placeholder for actual NMF implementation
        W = np.random.rand(X.shape[0], self.n_components)
        H = np.random.rand(self.n_components, X.shape[1])
        return (W, H)
        
    def resample(self, X: np.ndarray) -> np.ndarray:
        """Resample the data based on the specified method
        
        multinomial resample method: https://www.cell.com/cell-reports/fulltext/S2211-1247(12)00433-0#sec-4
        """
        np.random.seed(self.random_state)
        if self.resample_method == 'poisson':
            # Poisson resampling, each entry in X is resampled from a Poisson distribution of the same mean
            return np.random.poisson(X)
        elif self.resample_method == 'bootstrap':
            return self.multinomial_bootstrap(X)
        else:
            raise ValueError(f"Unknown resample method: {self.resample_method}, please use 'poisson' or 'bootstrap'")

    def multinomial_bootstrap(self, X: np.ndarray) -> np.ndarray:
        """
        Apply per-column multinomial bootstrap resampling to matrix M.

        Preserves the column sums of the original matrix M so each person will have the same number of mutations after resampling.
        
        Parameters:
        - X: np.ndarray of shape (n_types, n_samples), integer mutation counts
        
        Returns:
        - X_boot: np.ndarray of shape (n_types, n_samples), resampled counts
        """
        rng = np.random.default_rng(self.random_state)
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
        
        for _ in range(self.num_factorizations):
            X_resampled = self.resample(X)
            X_normalized = normalize_matrix(X_resampled, method=self.normalization_method)
            S, A = self.fit(X_normalized)
            S_all.append(S)
            A_all.append(A)
        
        return np.array(S_all), np.array(A_all)
    