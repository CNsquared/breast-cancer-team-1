from src.pipeline import preprocess, run_analysis, postprocess

def main():

    print("Preprocesssing mutation data...")
    mutations_processed = preprocess.preprocess_mutations()

    print(f"Processing fasta and getting CDS lengths...")
    df_sizes = preprocess.filter_fasta_and_get_cds_lengths()

    print("Get normalized counts...")
    normalized_counts = run_analysis.run_CDS_length_normalized(mutations_processed)
    normalized_counts.to_csv('results/tables/normalized_counts.tsv', sep="\t", index=False)
    print("Normalized counts saved to results/tables/normalized_counts.tsv")

    print("Running simple dN/dS Analysis")
    dnds_simple_results = run_analysis.run_dnds_simple(mutations_processed, df_sizes)
    dnds_simple_results.to_csv('results/tables/dnds_simple_results.tsv', sep="\t", index=False)
    print("dN/dS simple results saved to results/tables/dnds_simple_results.tsv")
    #postprocess.something(dnds_simple_results)

    print("Job completed successfully!")

if __name__ == "__main__":
    main()