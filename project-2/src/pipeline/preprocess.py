import pandas as pd
import re

def preprocess_mutations():
    """
    Preprocesses mutation data

    Returns:
        df_mut (pd.DataFrame): Processed mutation data
    """
    HYPERMUTATOR_THRESHOLD = 500

    # read in raw mutation data
    df_mut = pd.read_csv('data/raw/TCGA.BRCA.mutations.txt', sep='\t')
    initial_mutation = len(df_mut)
    initial_patient = df_mut['patient_id'].nunique()
    print(f"\t{initial_mutation:,} mutations and {initial_patient:,} patients loaded")
    
    # exclude samples the didn't pass QC
    df_mut = df_mut[df_mut['FILTER'] == 'PASS']

    # exclude indels (no more, we're including indels now)
    #df_mut = df_mut[df_mut['Variant_Type'] == 'SNP']

    # define mutation type (Silent in a CDS region is synonymous and all other types in CDS region is nonsynonymous. Those not in a CDS region is None)
    df_mut.loc[:,'mutation_type'] = df_mut.apply(lambda row: 'synonymous' if row['Variant_Classification'] == 'Silent' else 'non-synonymous' if row['CDS_position'] != '.' else None, axis=1)

    # drop rows where mutation_type is None (non-coding mutations)
    df_mut = df_mut.dropna(subset=['mutation_type'])

    # remove genes that don't have a symbol (they all have symbols FYI)
    df_mut = df_mut[df_mut['Hugo_Symbol'].notna()]

    # remove rows without a position (they all have a start position FYI)
    df_mut = df_mut[df_mut['Start_Position'].notna()]

    # filter out hypermutators
    # count PASS coding synonymous and nonsynonymous SNPs and indels per patient
    mutation_counts = df_mut['patient_id'].value_counts()

    # a mask for only patients with <= 500 mutations
    normal_mutators = mutation_counts[mutation_counts <= HYPERMUTATOR_THRESHOLD].index

    # keep only the non-hypermutators
    df_mut = df_mut[df_mut['patient_id'].isin(normal_mutators)]

    # mutation_class will be used for stacked bar plot with detailed non-synonymous subtype breakdown (ex. Frame_Shift_Ins, Missense_Mutation, ...)
    df_mut['mutation_class'] = df_mut.apply(
        lambda x: x['Variant_Classification']
                if x['mutation_type'] == 'non-synonymous'
                else 'synonymous',
        axis=1
    )

    filter_mutation = len(df_mut)
    filter_patient = df_mut['patient_id'].nunique()
    print(f"\t{filter_mutation:,} ({(filter_mutation*100)/initial_mutation:.1f}%) mutations and {filter_patient:,} ({(filter_patient*100)/initial_patient:.1f}%)  patients used for analysis")

    # write output
    df_mut.to_csv('data/processed/TCGA.BRCA.mutations.qc1.txt', sep='\t', index=False)
    
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

def filter_fasta_and_get_cds_lengths() -> pd.DataFrame:
    """
    Filters a FASTA file to include only transcripts in the TSV,
    and computes CDS lengths from FASTA headers.

    Returns:
    - DataFrame with Hugo_Symbol, Transcript_ID, and CDS_length
    """
    # File paths
    tsv_path = "data/processed/TCGA.BRCA.mutations.qc1.txt"
    fasta_in = "data/raw/gencode.v23lift37.pc_transcripts.fa"
    fasta_out = "data/processed/gencode.v23lift37.pc_transcripts.transcripts_in_TCGA_MAF.fa"
    CDS_length_table = "data/processed/gencode.v23lift37.pc_transcripts.transcripts_in_TCGA_MAF.cds_lengths.tsv"

    # 1. Read TSV into DataFrame and build mapping Transcript_ID -> Hugo_Symbol
    tsv = pd.read_csv(tsv_path, sep="\t", dtype=str)
    if "Transcript_ID" not in tsv.columns or "Hugo_Symbol" not in tsv.columns:
        raise RuntimeError("TSV must contain 'Transcript_ID' and 'Hugo_Symbol' columns")
    mapping = dict(zip(tsv["Transcript_ID"], tsv["Hugo_Symbol"]))
    ids = set(mapping.keys())

    # 2. Iterate FASTA, filter entries, compute CDS length, and write filtered FASTA
    results = []
    with open(fasta_in, "r") as fin, open(fasta_out, "w") as fout:
        lines = iter(fin)
        for line in lines:
            if not line.startswith(">"):
                continue
            header = line.rstrip("\n")
            transcript_full = header[1:].split()[0]
            transcript = transcript_full.split(".")[0]
            if transcript in ids:
                # write header and sequence line
                fout.write(header + "\n")
                seq_line = next(lines, "")
                fout.write(seq_line)
                # extract CDS:start-end
                m = re.search(r"CDS:(\d+)-(\d+)", header)
                if m:
                    start, end = map(int, m.groups())
                    cds_length = end - start + 1
                else:
                    cds_length = None
                # collect result
                results.append({
                    "Hugo_Symbol": mapping[transcript],
                    "Transcript_ID": transcript,
                    "CDS_length": cds_length
                })

    # 3. Build DataFrame, sort by Hugo_Symbol, and save to TSV
    df = pd.DataFrame(results)
    df_sorted = df.sort_values("Hugo_Symbol").reset_index(drop=True)
    # df_sorted.to_csv(CDS_length_table, index=False, sep="\t")

    print(f"Filtered FASTA written to {fasta_out}")
    print(f"CDS lengths saved to {CDS_length_table}")

    transcripts = {}
    with open(fasta_out, "r") as f:
        header = None
        seq_chunks = []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    full_id = header[1:].split("|")[0]
                    base_id = full_id.split(".")[0]
                    transcripts[base_id] = (header, "".join(seq_chunks))
                header = line
                seq_chunks = []
            else:
                seq_chunks.append(line)
        # add last record
        if header is not None:
            full_id = header[1:].split("|")[0]
            base_id = full_id.split(".")[0]
            transcripts[base_id] = (header, "".join(seq_chunks))

    # 2. Extract CDS sequences based on header coordinates and verify frame
    cds_seqs = []
    for gene, row in df_sorted.iterrows():
        tid = row["Transcript_ID"]
        if tid not in transcripts:
            print(f"Warning: transcript {tid} for gene {gene} not found in FASTA.")
            cds_seqs.append(None)
            continue

        header, full_seq = transcripts[tid]
        m = re.search(r"CDS:(\d+)-(\d+)", header)
        if not m:
            print(f"Warning: CDS coordinates not found in header for {tid}.")
            cds_seqs.append(None)
            continue

        start, end = map(int, m.groups())
        cds_seq = full_seq[start-1:end]  # 1-based inclusive
        # if len(cds_seq) % 3 != 0:
        #     print(f"Warning: CDS sequence length for {tid} is {len(cds_seq)}, not a multiple of 3.")
        cds_seqs.append(cds_seq)

    # 3. Append to df_sizes
    df_sorted["CDS sequence"] = cds_seqs
    df_sorted.to_csv(CDS_length_table, index=False, sep="\t")
    return df_sorted
