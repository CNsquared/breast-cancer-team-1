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
    print(df_gene_ranks)
