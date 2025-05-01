from ..utils import dnds_crude_utils as dnds

import pandas as pd

def run_dnds_simple(mutations_processed: pd.DataFrame, reference_processed: pd.DataFrame) -> pd.DataFrame:
    counts_df = dnds.get_counts_df(mutations_processed)
    print("Counting observed synonymous and non-synonymous SNVs")
    counts_df = dnds.get_counts_df(mutations_processed)

    print("Calculating possible synonymous and non-synonymous SNVs")
    df_sizes = dnds.calculate_possible_mutations(reference_processed)

    print("Calculating dN/dS and generating p-values")
    results = dnds.evaluate(counts_df, df_sizes)

    return results

