import pandas as pd

def test_association(factor_df: pd.DataFrame, test_cols: str) -> pd.DataFrame:
    """Test association between components and metadata"""
    # TODO: implement this
    results = {}
    for col in test_cols:
        results[col] = 0
    return pd.DataFrame(results)