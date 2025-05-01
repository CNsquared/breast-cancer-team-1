from ..utils import dnds_crude_utils as dnds
from ..utils import mutated_genes_normalize

import pandas as pd

def run_dnds_simple(mutations_processed: pd.DataFrame, df_sizes: pd.DataFrame) -> pd.DataFrame:
    counts_df = dnds.get_counts_df(mutations_processed)
    print("Counting observed synonymous and non-synonymous SNVs")
    counts_df = dnds.get_counts_df(mutations_processed)
    codons = dnds.precompute_codon_opportunities()
    opportunities = dnds.get_s_ns_opportunities_fast(df_sizes, codons)
    opportunities.to_csv("data/processed/dnds_opportunities.tsv", sep="\t", index=False)
    print("Calculating dN/dS")
    results = dnds.calculate_dnds(mutations_processed, opportunities)
    print("Calculate p-values")
    results = dnds.get_pval(results)
    return results


def run_CDS_length_normalized(mutations_processed: pd.DataFrame) -> pd.DataFrame:
    print("Normalizing mutation counts by CDS length")
    normalized_counts = mutated_genes_normalize.get_normalized_counts(mutations_processed)
    return normalized_counts