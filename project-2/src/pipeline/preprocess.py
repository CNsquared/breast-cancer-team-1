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
    inital_genes = df_mut['Hugo_Symbol'].nunique()
    initial_patient = df_mut['patient_id'].nunique()
    print(f"\t{initial_mutation:,} mutations in {inital_genes:,} genes from {initial_patient:,} patients loaded")
    
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
    filter_genes = df_mut['Hugo_Symbol'].nunique()
    filter_patient = df_mut['patient_id'].nunique()
    print(f"\t{filter_mutation:,} ({(filter_mutation*100)/initial_mutation:.1f}%) mutations in {filter_genes:,} ({(filter_genes*100)/inital_genes:.1f}%) genes from {filter_patient:,} ({(filter_patient*100)/initial_patient:.1f}%) patients used for analysis")

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

    # 1. Load TSV and mapping
    tsv = pd.read_csv(tsv_path, sep="\t", dtype=str)
    mapping = dict(zip(tsv["Transcript_ID"], tsv["Hugo_Symbol"]))
    ids      = set(mapping)

    # 2. Parse FASTA into dict: transcript_id -> (header still containing separator "|", entire transcript sequence)
    transcripts = {}
    with open(fasta_in, "r") as f:
        header = None; seq_chunks = []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header:
                    full_id = header[1:].split("|")[0]
                    base_id = full_id.split(".")[0]
                    transcripts[base_id] = (header, "".join(seq_chunks))
                header = line; seq_chunks = []
            else:
                seq_chunks.append(line)
        if header:
            full_id = header[1:].split("|")[0]
            base_id = full_id.split(".")[0]
            transcripts[base_id] = (header, "".join(seq_chunks))
        
    results = []
    
    with open(fasta_out, "w") as fout:
        for tx_id in ids:
            gene = mapping[tx_id]
            entry = transcripts.get(tx_id)

            # if exact transcript not found, fallback to any transcripts matching gene
            if entry is None:
                # print(f"Warning: {tx_id} not found. Falling back on Hugo_Symbol {gene}.")
                candidates = []
                for bid, (hdr, seq) in transcripts.items():
                    # match gene field in header: |GENE|
                    if f"|{gene}|" in hdr:
                        m = re.search(r"CDS:(\d+)-(\d+)", hdr)
                        if m:
                            start, end = map(int, m.groups())
                            cds_len = end - start + 1
                            candidates.append((bid, hdr, seq, start, end, cds_len))
                if not candidates:
                    # print(f"Warning: No fallback transcripts for gene {gene}.")
                    continue
                # pick the one with longest CDS
                candidates.sort(key=lambda x: x[5], reverse=True)
                chosen = candidates[0]
                # print(f"  Fallback candidates:")
                # for bid, hdr, seq, st, en, clen in candidates:
                #     print(f"    {bid}: CDS:{st}-{en} (len={clen:,})")
                # print(f"  Chosen: {chosen[0]} with CDS:{chosen[3]}-{chosen[4]} (len={chosen[5]:,})")
                header, full_seq = chosen[1], chosen[2]
            else:
                header, full_seq = entry

            # write full FASTA entry
            fout.write(header + "\n")
            for i in range(0, len(full_seq), 60):
                fout.write(full_seq[i:i+60] + "\n")

            # extract CDS coords
            m = re.search(r"CDS:(\d+)-(\d+)", header)
            if not m:
                print(f"Warning: no CDS coords in header for {tx_id or chosen[0]}")
                continue
            start, end = map(int, m.groups())
            cds_seq    = full_seq[start-1:end]
            cds_len    = len(cds_seq)
            # if cds_len % 3 != 0:
            #     print(f"Warning: CDS length {cds_len:,} not multiple of 3 for {tx_id or chosen[0]}")

            results.append({
                "Hugo_Symbol"   : gene,
                "Transcript_ID" : tx_id,
                "CDS_start"     : start,
                "CDS_end"       : end,
                "CDS_length"    : cds_len,
                "CDS sequence"  : cds_seq
            })
            

    # 4. Save CSV
    df_sizes = pd.DataFrame(results)

    # Warn if any Hugo_Symbol appears more than once
    dup_counts = df_sizes['Hugo_Symbol'].value_counts()
    dups = dup_counts[dup_counts > 1].index.tolist()
    dup_genes = ", ".join(dups)
    if dups:
        print(f"\tWarning: {len(dups):,} transcripts found for genes: {dup_genes}")

    # Keep largest CDS size per gene
    print(f"\t—> Keeping largest CDS size per gene...")

    df_sorted = df_sizes.sort_values("CDS_length", ascending=False)
    df_sorted = df_sorted.drop_duplicates(subset="Hugo_Symbol", keep="first")
    df_sorted.to_csv(CDS_length_table, index=False, sep="\t")

    return df_sorted
