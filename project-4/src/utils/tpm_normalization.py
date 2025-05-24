import pandas as pd
import numpy as np

def parse_gencode_fasta(fasta_path):
    print(f"Parsing gencode fasta file: {fasta_path}")
    records = []
    with open(fasta_path, 'r') as f:
        current_record = {}
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_record:
                    current_record["sequence"] = ''.join(seq_lines)
                    records.append(current_record)
                    seq_lines = []

                parts = line[1:].split('|')
                cds_range = None
                cds_length = None
                for part in parts:
                    if part.startswith('CDS:'):
                        cds_range = part.replace('CDS:', '')
                        try:
                            start, end = map(int, cds_range.split('-'))
                            cds_length = end - start + 1
                        except:
                            cds_length = None

                current_record = {
                    'transcript_id': parts[0],
                    'gene_id': parts[1],
                    'transcript_symbol': parts[4],
                    'gene_symbol': parts[5],
                    'cds_range': cds_range,
                    'cds_length': cds_length
                }
            else:
                seq_lines.append(line)

        if current_record:
            current_record["sequence"] = ''.join(seq_lines)
            records.append(current_record)

    return pd.DataFrame(records)



def longest_CDS(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a DataFrame and returns a new DataFrame with the longest CDS for each gene_id.
    """
    print("Finding longest CDS length for each gene")
    # Convert cds_length to numeric, errors='coerce' will turn non-numeric values into NaN
    df.loc[:,'cds_length'] = pd.to_numeric(df.loc[:,'cds_length'], errors='coerce')
    
    # Group by gene_id and get the row with the maximum cds_length
    longest_cds_df = df.loc[df.groupby('gene_symbol')['cds_length'].idxmax()]
    
    return longest_cds_df

def log1p_TPM_normalize(df_exp: pd.DataFrame, df_lengths: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize gene expression data to TPM (Transcripts Per Million).
    
    Parameters:
    df_exp (pd.DataFrame): DataFrame with gene expression counts.
    df_lengths (pd.DataFrame): DataFrame with gene lengths.
    
    Returns:
    pd.DataFrame: DataFrame with TPM normalized values.
    """
    print("Normalizing expression data to log1p(TPM)")
    df_exp = df_exp.copy()

    df_lengths = df_lengths.copy()
    df_lengths = df_lengths.set_index('gene_symbol')

    # filter to genes with gene_symbol
    gene_symbols = df_exp.columns.str.split('|').str[0]
    df_exp.columns = gene_symbols
    df_exp = df_exp.loc[:, df_exp.columns != '?']

    # collapse isoforms
    df_exp = df_exp.T.groupby(df_exp.columns).sum().T

    # filter to common genes and reindex
    common_genes = df_exp.columns.intersection(df_lengths.index)
    df_exp = df_exp.reindex(columns=common_genes)
    df_lengths = df_lengths.reindex(index=common_genes)

    # calculate rpk
    rpk = df_exp.div(df_lengths['cds_length'], axis=1)
    rpk = rpk.div(1e3)

    # calculate scaling factor
    scaling_factor = rpk.sum(axis=1).div(1e6)

    # calculcate tpm
    tpm = rpk.div(scaling_factor, axis=0)
    log1p_tpm = np.log1p(tpm)

    return log1p_tpm