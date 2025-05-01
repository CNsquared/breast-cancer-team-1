import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.stats import fisher_exact
from scipy.stats import chisquare

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
    print(counts_df.columns)
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

def translate(seq): 
       
    table = { 
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                  
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W', 
    } 
    protein ="" 
    if len(seq)%3 == 0: 
        for i in range(0, len(seq), 3): 
            codon = seq[i:i + 3] 
            protein+= table[codon] 
    return protein 

def precompute_codon_opportunities():
    bases = ['A', 'C', 'G', 'T']
    codon_opportunity = {}

    for b1 in bases:
        for b2 in bases:
            for b3 in bases:
                codon = b1 + b2 + b3
                ref_aa = translate(codon)
                syn_count = 0
                nonsyn_count = 0

                for pos in range(3):
                    orig_base = codon[pos]
                    for alt_base in bases:
                        if alt_base == orig_base:
                            continue
                        alt_codon = codon[:pos] + alt_base + codon[pos+1:]
                        alt_aa = translate(alt_codon)
                        
                        if alt_aa == ref_aa:
                            syn_count += 1
                        else:
                            nonsyn_count += 1
                codon_opportunity[codon] = (syn_count, nonsyn_count)

    return codon_opportunity

def get_s_ns_opportunities_fast(df_sizes: pd.DataFrame, codon_opportunity: dict) -> pd.DataFrame:
    syn_ops = []
    nonsyn_ops = []

    for gene, row in df_sizes.iterrows():
        seq = row["CDS sequence"]
        syn_count = 0
        nonsyn_count = 0

        if seq:
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i:i+3]
                if codon in codon_opportunity:
                    syn, nonsyn = codon_opportunity[codon]
                    syn_count += syn
                    nonsyn_count += nonsyn
                else:
                    pass

        syn_ops.append(syn_count)
        nonsyn_ops.append(nonsyn_count)

    df_sizes["synonymous_opportunity"] = syn_ops
    df_sizes["nonsynonymous_opportunity"] = nonsyn_ops
    return df_sizes

def get_s_ns_opportunities(df_sizes: pd.DataFrame) -> pd.DataFrame:
    bases = ['A', 'C', 'G', 'T']

    syn_ops = []
    nonsyn_ops = []

    for gene, row in df_sizes.iterrows():
        seq = row["CDS sequence"]
        syn_count = 0
        nonsyn_count = 0

        if seq:
            if gene == "IGHJ5":
                print(f"Processing gene: {gene}")
                print(f"CDS sequence: {seq}")
                print(f"CDS length: {len(seq)}")
                print(range(0, len(seq) - 2, 3))
            # walk codons
            for i in range(0, len(seq) - 3 + 1, 3):

                codon = seq[i:i+3]
                ref_aa = translate(codon)

                if gene == "IGHJ5":
                    print(codon, "->", ref_aa)
                # for each position in codon
                for pos in range(3):
                    orig_base = codon[pos]
                    for alt_base in bases:
                        if alt_base == orig_base:
                            continue
                        alt_codon = codon[:pos] + alt_base + codon[pos+1:]
                        aa = translate(alt_codon)
                        # Add at top of your script
                        bold_start = '\033[1m'
                        bold_end = '\033[0m'

                        # Then, inside your codon loop:
                        highlighted_codon = (
                            codon[:pos]
                            + bold_start + codon[pos] + bold_end
                            + codon[pos+1:]
                        )
                        highlighted_alt = (
                            alt_codon[:pos]
                            + bold_start + alt_codon[pos] + bold_end
                            + alt_codon[pos+1:]
                        )
                        if aa == ref_aa:
                            if gene == "IGHJ5":
                                print(f"{highlighted_codon}({ref_aa}) -> {highlighted_alt}({aa}) : S")
                            syn_count += 1
                        else:
                            if gene == "IGHJ5":
                                print(f"{highlighted_codon}({ref_aa}) -> {highlighted_alt}({aa}) : N")
                            nonsyn_count += 1

        syn_ops.append(syn_count)
        nonsyn_ops.append(nonsyn_count)

    df_sizes["synonymous_opportunity"] = syn_ops
    df_sizes["nonsynonymous_opportunity"] = nonsyn_ops

    return df_sizes

def calculate_dnds(df_mut, opportunities_df) -> pd.DataFrame:
    
    # 2. Define categories and order
    syn_col = 'synonymous'
    non_syn_classes = [
        'Frame_Shift_Del', 
        'Frame_Shift_Ins', 
        'In_Frame_Del', 
        'In_Frame_Ins',  
        'Missense_Mutation', 
        'Nonsense_Mutation', 
        'Nonstop_Mutation',
        'Translation_Start_Site'
    ]
    all_classes = [syn_col] + non_syn_classes

    # 3. Group and pivot to get counts per gene per class
    mutation_counts = df_mut.groupby(['Hugo_Symbol', 'mutation_class']).size()
    counts_df = mutation_counts.unstack(fill_value=0)
    # 4. Ensure all classes exist
    for col in all_classes:
        if col not in counts_df.columns:
            counts_df[col] = 0

    # 5. Reorder columns
    counts_df = counts_df[all_classes]

    # 6. Filter genes with at least 5 total mutations))
    counts_df = counts_df[counts_df.sum(axis=1) >= 5]

    # 7. Normalize by CDS length
    counts_df = counts_df.join(opportunities_df['CDS_length'], how='left')
    df = pd.merge(counts_df, opportunities_df[['Hugo_Symbol','synonymous_opportunity', 'nonsynonymous_opportunity']], how='inner', left_index=True, right_on='Hugo_Symbol')

    df['observed_nonsynonymous'] = df[non_syn_classes].sum(axis=1)

    # adjust nonsynonymous opportunities to include indels
    df['Indels'] = df['Frame_Shift_Del'] + df['Frame_Shift_Ins'] + df['In_Frame_Del'] + df['In_Frame_Ins']
    df['NS_SNV'] = df['Missense_Mutation'] + df['Nonsense_Mutation'] + df['Nonstop_Mutation']
    df['nonsynonymous_opportunity'] = df['nonsynonymous_opportunity'] * (1 + ((df['Indels'] + .5)/(df['NS_SNV'] + .5)).mean())

    df['dS'] = df.apply(
        lambda x: np.nan if x['synonymous_opportunity'] == 0 else x['synonymous'] / x['synonymous_opportunity'],
        axis=1
    )
    df['dN'] = df.apply(
        lambda x: np.nan if x['nonsynonymous_opportunity'] == 0 else x['observed_nonsynonymous'] / x['nonsynonymous_opportunity'],
        axis=1
    )

    df['dN/dS'] = df['dN'] / df['dS']
    # replace inf with NaN
    df['dN/dS'] = df['dN/dS'].replace([np.inf, -np.inf], np.nan)

    # # 3. Calculate dS = observed_synonymous / synonymous_opportunity
    # df['dS'] = df['synonymous'] / df['synonymous_opportunity']

    # # 4. Calculate dN = observed_nonsynonymous / nonsynonymous_opportunity
    # df['dN'] = df['observed_nonsynonymous'] / df['nonsynonymous_opportunity']

    # # 5. Calculate dN/dS ratio
    # df['dN/dS'] = df['dN'] / df['dS']
    # df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['dN/dS'])
    return df

def get_pval(df:pd.DataFrame) -> pd.DataFrame:
    """
    Calculate p-values for dN/dS ratios using two methods:
    1. Poisson
    2. Fisher exact

    We will assume under our null hypothesis that the dN/dS ratio is 1 and that
        the mutation rates are uniform for all sites and types of mutations across a gene.
        
    The p-values are added to the DataFrame as new columns.

    Parameters:
        df (pd.DataFrame): DataFrame containing dN/dS ratios
            Dataframe needs these columns:
            - 'synonymous_opportunity'
            - 'nonsynonymous_opportunity'
            - 'synonymous'
            - 'observed_nonsynonymous'
            
    Returns:
        df (pd.DataFrame): DataFrame with p-values added
    """
    # poisson exact test
    def poisson_pval(x):
        """P-value for observed nonsynonymous mutation count of the gene. Under null, expected should be equal to the observed synonymous mutations times the ratio of nonsynonymous to synonymous opportunities."""
        return 1 - poisson.cdf(x['observed_nonsynonymous']-1, x['synonymous']*x['nonsynonymous_opportunity']/x['synonymous_opportunity'])

    df['poisson_pval'] = df.apply(lambda x: poisson_pval(x), axis=1)
    # fisher exact test
    def fisher_pval(x):
        """P-value for the Fisher's exact test"""
        table = [[x['observed_nonsynonymous'], x['nonsynonymous_opportunity'] - x['observed_nonsynonymous']], 
                [x['synonymous'], x['synonymous_opportunity'] - x['synonymous']]
                ]
        odds, pval = fisher_exact(table, alternative='greater')
        return pd.Series({'fisher_odds': odds, 'fisher_pval': pval})
    
    df[['fisher_odds', 'fisher_pval']] = df.apply(fisher_pval, axis=1)

    def chi2_pval(x):
        """P-value for Chi2 test"""
        gene_mutation_rate = (x['observed_nonsynonymous'] + x['synonymous']) / (x['nonsynonymous_opportunity'] + x['synonymous_opportunity'])
        expected_synonymous = gene_mutation_rate * x['synonymous_opportunity']
        expected_nonsynonymous = gene_mutation_rate * x['nonsynonymous_opportunity']
        obs = [x['observed_nonsynonymous'], x['synonymous']]
        exp = [expected_nonsynonymous, expected_synonymous]

        statistic, pval = chisquare(f_obs=obs, f_exp=exp)
        return pd.Series({'chi2': statistic, 'chi2_pval': pval})

    df[['chi2', 'chi2_pval']] = df.apply(chi2_pval, axis=1)

    return df