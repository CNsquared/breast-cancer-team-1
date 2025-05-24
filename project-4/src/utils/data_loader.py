import pandas as pd

def load_expression_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=[0,1], low_memory=False)
    df = df.astype(float)
    return df.T  # transpose to shape (samples, genes)
