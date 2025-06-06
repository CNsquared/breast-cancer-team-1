import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataHandler:
    def __init__(self):
        pass

    @staticmethod
    def filter_by_mean(X: pd.DataFrame, threshold: float = 1, verbose=False) -> pd.DataFrame:
        """Filter based on mean value threshold"""
        above_threshold = X.mean(axis=0) > threshold
        X_filtered = X.loc[:, above_threshold]
        if verbose:
            print(f"Mean threshold: {threshold}")
            print(f"X shape before filtering: {X.shape}")
            print(f"X shape after filtering: {X_filtered.shape}")
        return X_filtered

    @staticmethod
    def top_n_variable(X: pd.DataFrame, k: int = 100, verbose=False) -> pd.DataFrame:
        """Select the top k most variable features"""
        # sort columns by variance
        X_var = X.var(axis=0).sort_values(ascending=False)
        # select top k most variable features
        top_k_genes = X_var.index[:k]
        if verbose:
            print(f"Selected top {k} most variable features")
        return X.loc[:, top_k_genes]
    
    @staticmethod
    def filter_by_variance(X: pd.DataFrame, threshold: float = 0.1, verbose=False) -> pd.DataFrame:
        """Filter features based on variance threshold"""
        above_threshold = X.var(axis=0) > threshold
        X_filtered = X.loc[:, above_threshold]
        if verbose:
            print(f"Variance threshold: {threshold}")
            print(f"X shape before filtering: {X.shape}")
            print(f"X shape after filtering: {X_filtered.shape}")
        return X_filtered