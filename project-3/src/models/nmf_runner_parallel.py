import numpy as np
from sklearn.decomposition import NMF
from joblib import Parallel, delayed
from ..utils.mutation_matrix import normalize_matrix  # Adjust as needed
import multiprocessing


def _single_factorization_static(
    X: np.ndarray,
    k: int,
    seed: int,
    resample_method: str,
    normalization_method: str,
    objective_function: str,
    initialization_method: str,
    max_iter: int,
    tolerance: float,
    i: int,
    verbose: bool
):
    if verbose:
        print(f"Running NMF factorization k={k}, iteration={i + 1}", flush=True)

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

    # if verbose:
    #     if n_iter >= max_iter:
    #         print(f"⚠️  Iteration {i + 1}: NMF hit max_iter ({max_iter}) without converging (reconstruction error: {err:.4f})", flush=True)
    #     else:
    #         print(f"✓  Iteration {i + 1}: NMF converged in {n_iter} iterations (reconstruction error: {err:.4f})", flush=True)

    return S, A, err, n_iter


class NMFDecomposer:
    def __init__(
        self,
        n_components: int,
        resample_method: str = 'poisson',
        objective_function: str = 'frobenius',
        initialization_method: str = 'random',
        normalization_method: str = 'GMM',
        max_iter: int = 1000000,
        num_factorizations: int = 100,
        random_state: int = 42,
        tolerance: float = 1e-15,
        verbose: bool = False,
        n_jobs: int = max(multiprocessing.cpu_count() - 2, 1)

    ):
        self.n_components = n_components
        self.resample_method = resample_method
        self.objective_function = objective_function
        self.initialization_method = initialization_method
        self.normalization_method = normalization_method
        self.max_iter = max_iter
        self.num_factorizations = num_factorizations
        self.random_state = random_state
        self.tolerance = tolerance
        self.verbose = verbose
        self.n_jobs = n_jobs

    def run(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.verbose:
            print(f"Running NMF with the following parameters:", flush=True)
            for attr, value in self.__dict__.items():
                print(f"{attr}: {value}", flush=True)
            print("", flush=True)

        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(_single_factorization_static)(
                X,
                self.n_components,
                self.random_state + i,
                self.resample_method,
                self.normalization_method,
                self.objective_function,
                self.initialization_method,
                self.max_iter,
                self.tolerance,
                i,
                self.verbose
            )
            for i in range(self.num_factorizations)
        )

        S_all, A_all, err_all, n_iter_all = zip(*results)
        return (
            np.array(S_all),
            np.array(A_all),
            np.array(err_all),
            np.array(n_iter_all)
        )
