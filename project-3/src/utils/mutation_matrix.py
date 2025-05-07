import pandas as pd
import numpy as np

def build_mutation_matrix(df_mut: pd.DataFrame) -> np.ndarray:
    """Builds a mutation matrix from mutation DataFrame."""
    # TODO: placehodler
    return np.zeros((len(df_mut), len(df_mut)))

def normalize_matrix(X: np.ndarray, method: str = 'GMM') -> np.ndarray:
    """normalize matrix by method
    
    should be able to use the following methods (from paper):
    - GMM: Gaussian mixture model (default)
    - 100X
    - log2
    - None: no normalization.
    """
    # TODO: placeholder
    return X