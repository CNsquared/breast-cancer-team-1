from ..utils.eval import evalAccuracy, compareRankings

import pandas as pd

def get_intogen_ranks():
    #COMPARING TO INTOGEN
    
    # load the IntOGen ranking TSV
    path_intogen = "data/raw/IntOGen-DriverGenes_TCGA_WXS_BRCA.tsv"
    df_intogen = pd.read_csv(path_intogen, sep="\t")
    
    # build a dict mapping gene names to their IntOGen relevance.
    #RELEVANCE IS SAMPLES%
    baseline_ranks = dict(zip(df_intogen["Symbol"], (df_intogen["Samples (%)"] * 0.01)))
    
    path_dn_ds = "results/tables/dnds_simple_results.tsv"
    df_dn_ds = pd.read_csv(path_dn_ds, sep="\t")
    #dn_ds_ranks = dict(zip(df_dn_ds["Hugo_Symbol"], df_dn_ds["dN/dS"]))
    
    dn_ds_ranks = dict(zip(df_dn_ds["Hugo_Symbol"], -df_dn_ds["fisher_pval"]))
    
    # calculate accuracy metrics
    dcg, bpref, accuracy = evalAccuracy(dn_ds_ranks, baseline_ranks)
    
    df_gene_ranks = compareRankings(dn_ds_ranks, baseline_ranks)
    return df_gene_ranks, dcg, bpref, accuracy