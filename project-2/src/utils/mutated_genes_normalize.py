import pandas as pd

def get_normalized_counts(df_mut: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes mutation counts by CDS length and filters genes with at least 5 mutations.

    Args:
        df_mut (pd.DataFrame): DataFrame containing mutation data with columns 'Hugo_Symbol', 'mutation_type', and 'Variant_Classification'.

    Returns:
        pd.DataFrame: DataFrame with normalized counts by CDS length of mutations per gene.
    """
        
    # --- Ensure df_mut DataFrame is loaded and processed correctly before this point ---
    # Must contain 'Hugo_Symbol', 'mutation_type', and 'Variant_Classification'

    # 0. Load gene sizes
    sizes_path = "data/processed/gencode.v23lift37.pc_transcripts.transcripts_in_TCGA_MAF.cds_lengths.tsv"
    df_sizes = pd.read_csv(sizes_path, sep="\t", dtype={'Hugo_Symbol': str, 'CDS_length': int})
    # print(df_sizes.head())
    
    # Warn if any Hugo_Symbol appears more than once
    dup_counts = df_sizes['Hugo_Symbol'].value_counts()
    dups = dup_counts[dup_counts > 1].index.tolist()
    
    if dups:
        print(f"\tStill: {len(dups):,} duplicate Hugo_Symbols found in CDS lengths. Keeping the largest CDS length for each gene.")
        print(f"\tDuplicate genes: {', '.join(dups)}")


    # Keep largest CDS size per gene
    print(f"\tUsing largest CDS size per gene...")
    df_sizes = (
        df_sizes
        .sort_values('CDS_length', ascending=False)
        .drop_duplicates('Hugo_Symbol')
        .set_index('Hugo_Symbol') # Set Hugo_Symbol as index, no longer has it as a column
    )
    
    num_genes = df_sizes.index.nunique()
    print(f"\t{num_genes:,} genes with unique CDS_length entries from Gencode v23lift37")
    # 2. Define categories and order
    syn_col = 'synonymous'
    non_syn_classes = ["Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins", "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation", "Translation_Start_Site",]
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
    counts_df = counts_df.join(df_sizes['CDS_length'], how='left')
    norm_df = counts_df[all_classes].div(counts_df['CDS_length'], axis=0).fillna(0)
    norm_df.to_csv("data/processed/TCGA.BRCA.mutations.qc1.normalized_CDS_length.txt", sep="\t", index=True)
    return norm_df