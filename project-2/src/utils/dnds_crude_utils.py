import pandas as pd
import numpy as np

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
    non_syn_classes = ["Missense_Mutation", "Nonsense_Mutation", "Translation_Start_Site", "Nonstop_Mutation"]
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

    # 6. Filter genes with at least 5 total mutations
    counts_df = counts_df[counts_df.sum(axis=1) >= 5]

    # 7. Normalize by CDS length
    counts_df = counts_df.join(opportunities_df['CDS_length'], how='left')
    df = pd.merge(counts_df, opportunities_df[['Hugo_Symbol','synonymous_opportunity', 'nonsynonymous_opportunity']], how='inner', left_index=True, right_on='Hugo_Symbol')

    # 2. Compute observed nonsynonymous as sum of all non-synonymous classes
    nonsyn_cols = [
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Translation_Start_Site",
        "Nonstop_Mutation"
    ]
    df['observed_nonsynonymous'] = df[nonsyn_cols].sum(axis=1)

    # 3. Calculate dS = observed_synonymous / synonymous_opportunity
    df['dS'] = df['synonymous'] / df['synonymous_opportunity']

    # 4. Calculate dN = observed_nonsynonymous / nonsynonymous_opportunity
    df['dN'] = df['observed_nonsynonymous'] / df['nonsynonymous_opportunity']

    # 5. Calculate dN/dS ratio
    df['dN/dS'] = df['dN'] / df['dS']
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['dN/dS'])
    return df
    