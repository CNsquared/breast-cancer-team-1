import pandas as pd

def count_mutations(mutations_processed):
    """
    Counts the number of mutations of various types for each gene in the processed mutation data

    Args: 
        mutations_processed (pd.DataFrame): Processed mutation data
    
    Returns:
        counts_df (pd.DataFrame): Dataframe where rows are genes, and columns are observed counts of SYN and non-SYN mutations
    """

    counts_df = None

    return counts_df

def calculate_possible_mutations(reference_processed):
    """
    Calculates the number of possible S and N mutations 

    Args:
        reference_processed (pd.DataFrame): Dataframe containing sequence of all CDSs in TCGA.BRCA

    Returns:
        reference_info (pd.DataFrame): Dataframe containing N sites, S sites, and length of all CDSs in TCGA.BRCA

    """

    reference_info = None

    return reference_info

def evaluate(counts_df, reference_info):
    """
    Calculated dN/dS and evaluates significance using normal approximation

    Args: 
        counts_df (pd.DataFrame): Dataframe where rows are genes, and columns are observed counts of SYN and non-SYN mutations
        reference_info (pd.DataFrame): Dataframe containing N sites, S sites, and length of all CDSs in TCGA.BRCA

    Returns:
        results_df (pd.DataFrame): Dataframe withr results
    """

    # calculate dN

    # approximate variance of dN using Binomial approximation

    # calculate dS

    # approximate variance of dS using Binomial approximation

    # calculate test statistic (Z-test) using approximated variances

    # apply bonferroni

    # apply Benjamini-Hochberg FDR

    # return results
    results_df = None
    return results_df

    

