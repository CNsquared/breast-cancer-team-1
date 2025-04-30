import pandas as pd


def preprocess_mutations():
    """
    Preprocesses mutation data

    Returns:
        df_mut (pd.DataFrame): Processed mutation data
    """
    # read in raw mutation data
    df_mut = pd.read_csv('data/raw/TCGA.BRCA.mutations.txt', sep='\t')

    # exclude samples the didn't pass QC
    df_mut = df_mut[df_mut['FILTER'] == 'PASS']

    # exclude indels
    df_mut = df_mut[df_mut['Variant_Type'] == 'SNP']

    # define mutation type
    df_mut.loc[:,'mutation_type'] = df_mut.apply(lambda row: 'synonymous' if row['Variant_Classification'] == 'Silent' else 'non-synonymous' if row['CDS_position'] != '.' else None, axis=1)

    # remove genes that don't have a symbol (they all have symbols FYI)
    df_mut = df_mut[df_mut['Hugo_Symbol'].notna()]

    # remove rows without a positiion
    df_mut = df_mut[df_mut['Start_Position'].notna()]

    # filter out hypermutators
    # count mutations per patient
    mutation_counts = df_mut['patient_id'].value_counts()

    # keep only patients with <= 500 mutations
    normal_mutators = mutation_counts[mutation_counts <= 500].index

    # filter out hypermutators
    df_mut = df_mut[df_mut['patient_id'].isin(normal_mutators)]

    # write output
    #df_mut.to_csv('../../data/processed/TCGA.BRCA.mutations.qc1.txt', sep='\t')

    # return output
    return df_mut


def preprocess_reference():
    """
    Preprocesses reference data

    Returns:
        reference_processed (pd.DataFrame): Processed mutation data
    """

    reference_processed = None
    return reference_processed