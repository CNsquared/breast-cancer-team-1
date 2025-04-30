
import math
import pandas as pd

def evalAccuracy(calculated_ranks, baseline_ranks):
    """
    Compares the ranking of genes to intogen rankings.
    Calculates:
      - DCG: discounted cumulative gain with binary relevance
      - Bpref: binary preference
    
    """
    # set of true (intogen) genes
    
    
    driver_genes = set(baseline_ranks.keys())
    sorted_genes = sorted(calculated_ranks, key=calculated_ranks.get, reverse=True)

    # --- DCG  --- WANT CLOSER TO 1
    # DCG is a measure of ranking quality that takes into account the position of relevant items.
    # It is calculated as the sum of the relevance of each relevant item, discounted by its position in the ranking.
    dcg = 0.0
    for i, g in enumerate(sorted_genes):
        if g in driver_genes:
            # (2^relevance - 1) / log2(position+1)
            dcg += ((2 ** baseline_ranks[g]) - 1)/ math.log2(i + 2)
    
    # --- Bpref --- WANT CLOSER TO 1
    # Bpref is a measure of ranking quality that takes into account the number of relevant items ranked above each relevant item.
    R = len(driver_genes)
    bp_sum = 0.0
    rel_seen = 0
    for i, g in enumerate(sorted_genes):
        if g in driver_genes:
            # number of non‑relevant ranked above this relevant
            irrel_before = i - rel_seen
            # cap at R for the standard Bpref formula
            bp_sum += 1 - min(irrel_before, R) / R
            rel_seen += 1
            
    bpref = bp_sum / R if R > 0 else 0.0
            
    # --- Accuracy --- WANT CLOSER TO 1
    # What percent of ground truth genes are missing from the top
    top_num = len(driver_genes)
    top_genes = set(sorted_genes[:top_num])
    missing = len(driver_genes - top_genes)
    accuracy = 1 - (missing / len(driver_genes)) if len(driver_genes) > 0 else 0.0
    
    
    return dcg, bpref, accuracy

def compareRankings(calculated_ranks, baseline_ranks):
    
    sorted_baseline = sorted(baseline_ranks.items(), key=lambda x: x[1], reverse=True)
    sorted_dn_ds = sorted(calculated_ranks.items(), key=lambda x: x[1], reverse=True)
    
    # create a lookup of gene → its rank in the dn/ds list
    dn_ds_rank_map = {gene: i+1 for i, (gene, _) in enumerate(sorted_dn_ds)}

    # build a mapping of baseline genes to (baseline_rank, dn_ds_rank)
    mapping_intogen_to_dn_ds = {
        gene: (i+1, dn_ds_rank_map.get(gene))
        for i, (gene, _) in enumerate(sorted_baseline)
    }

    # display as DataFrame for clarity
    df_gene_ranks = pd.DataFrame([
        {"Gene": gene, "IntOGen_Rank": ranks[0], "dN_dS_Rank": ranks[1]}
        for gene, ranks in mapping_intogen_to_dn_ds.items()
    ])
    
    return df_gene_ranks



if __name__ == "__main__":
    
    #COMPARING TO INTOGEN
    
    # load the IntOGen ranking TSV
    path_intogen = "IntOGen-DriverGenes_TCGA_WXS_BRCA.tsv"
    df_intogen = pd.read_csv(path_intogen, sep="\t")
    
    # build a dict mapping gene names to their IntOGen relevance.
    #RELEVANCE IS SAMPLES%
    baseline_ranks = dict(zip(df_intogen["Symbol"], (df_intogen["Samples (%)"] * 0.01)))
    
    path_dn_ds = "TCGA.BRCA.dN_dS.tsv"
    df_dn_ds = pd.read_csv(path_dn_ds, sep="\t")
    dn_ds_ranks = dict(zip(df_dn_ds["Hugo_Symbol"], df_dn_ds["dN/dS"]))
    
    # calculate accuracy metrics
    dcg, bpref, accuracy = evalAccuracy(dn_ds_ranks, baseline_ranks)
    
    df_gene_ranks = compareRankings(dn_ds_ranks, baseline_ranks)
    print(df_gene_ranks)import pandas as pd

def count_mutations(mutations_processed):
    """
    Counts the number of mutations of various types for each gene in the processed mutation data

    Args: 
        mutations_processed (pd.DataFrame): Processed mutation data
    
    Returns:
        counts_df (pd.DataFrame): Dataframe where rows are genes, and columns are observed counts of SYN and non-SYN mutations
    """

    # 2. Define categories and order
    syn_col = 'synonymous'
    non_syn_classes = ["Missense_Mutation", "Nonsense_Mutation", "Translation_Start_Site", "Nonstop_Mutation"]
  
    # 3. Group and pivot to get counts per gene per class
    mutation_counts = mutations_processed.groupby(['Hugo_Symbol', 'mutation_class']).size()
    counts_df = mutation_counts.unstack(fill_value=0)

    # 4. Filter genes with at least 5 total mutations
    counts_df = counts_df[counts_df.sum(axis=1) >= 5]

    # 5. Combine non-SYN columns
    counts_df['nonsynonymous'] = counts_df[non_syn_classes].sum(axis=1)

    # 6. add 0.5 pseudocounts
    # counts_df['nonsynonymous'] += 0.5
    # counts_df['synonymous'] += 0.5

    print(counts_df.head())

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

    

