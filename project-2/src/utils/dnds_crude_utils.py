def get_counts_df(df_mut):
    # 2. Define the desired categories and their plot order
    syn_col = 'synonymous'
    non_syn_classes = [
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Translation_Start_Site",
        "Nonstop_Mutation",
    ]
    all_classes = [syn_col] + non_syn_classes

    # 3. Group and pivot to get counts per gene per class
    mutation_counts = df_mut.groupby(['Hugo_Symbol', 'mutation_class']).size()
    counts_df = mutation_counts.unstack(fill_value=0)

    # 4. Ensure all columns exist
    for col in all_classes:
        if col not in counts_df.columns:
            counts_df[col] = 0

    # 5. Reorder columns
    counts_df = counts_df[all_classes]

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

    # calculate Fisher exact (null -> expected NS is SYN rate times NS sites, expected S is S)


    # return results
    results_df = None
    return results_df
