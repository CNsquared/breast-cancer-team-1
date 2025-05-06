import pandas as pd
import numpy as np

def build_mutation_matrix(df_mut: pd.DataFrame) -> np.ndarray:
    """Builds a mutation matrix from mutation DataFrame."""
    # TODO: placehodler
    return np.zeros((len(df_mut), len(df_mut)))

def normalize_matrix(X: np.ndarray, method: str = 'row-wise') -> np.ndarray:
    """normalize matrix by method
    
    should be able to return row-wise, column-wise, z-score, or whatever other methods we want to try
    """
    # TODO: placeholder
    return X