import pandas as pd

def count_mutations(mutations_processed):
    """
    Counts the number of mutations of various types for each gene in the processed mutation data

    Args: 
        mutations_processed (pd.DataFrame): Processed mutation data
    
    Returns:
        counts_df (pd.DataFrame): Dataframe where rows are genes, and columns are observed counts of SYN and non-SYN mutations
    """

    
    mutations_processed['mutation_class'] = mutations_processed.apply(
        lambda x: x['Variant_Classification']
                if x['mutation_type'] == 'non-synonymous'
                else 'synonymous',
        axis=1
    )

    # 2. Define categories and order
    syn_col = 'synonymous'
    non_syn_classes = ["Missense_Mutation", "Nonsense_Mutation", "Translation_Start_Site", "Nonstop_Mutation"]
    all_classes = [syn_col] + non_syn_classes

    # 3. Group and pivot to get counts per gene per class
    mutation_counts = mutations_processed.groupby(['Hugo_Symbol', 'mutation_class']).size()
    counts_df = mutation_counts.unstack(fill_value=0)

    # 4. Ensure all classes exist
    for col in all_classes:
        if col not in counts_df.columns:
            counts_df[col] = 0

    # 5. Reorder columns
    counts_df = counts_df[all_classes]

    # 6. Filter genes with at least 5 total mutations
    counts_df = counts_df[counts_df.sum(axis=1) >= 5]

    return counts_df, mutations_processed

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

    

