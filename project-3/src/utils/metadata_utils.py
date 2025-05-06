import pandas as pd
import numpy as np

def load_metadata(filepath) -> pd.DataFrame:
    """Load and clean patient metadata (age, sex, subtype, etc.)."""
    metadata = pd.read_csv(filepath, sep='\t', low_memory=False)
    # TODO: clean metadata
    return metadata

def merge_with_components(W: np.ndarray, sample_ids: list[str], metadata: pd.DataFrame) -> pd.DataFrame:
    """Join factor matrix W with metadata for statistical analysis."""
    df = pd.DataFrame()
    # TODO: implement this
    return df