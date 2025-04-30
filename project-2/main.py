from src.pipeline import preprocess, run_analysis, postprocess

PROCESSED_OUTPUT= 'data/processed/TCGA.BRCA.mutations.qc1.txt'

def main():

    print("Preprocesssing mutation data...")
    mutations_processed = preprocess.preprocess_mutations()
    mutations_processed.to_csv(PROCESSED_OUTPUT)
    print(f"Preprocessed data saved to {PROCESSED_OUTPUT}")
    
    print("Preprocessing reference data...")
    reference_processed = preprocess.preprocess_reference()

    print("Counting observed synonymous and non-synonymous SNVs")
    counts_df = run_analysis.count_mutations(mutations_processed)

    print("Calculating possible synonymous and non-synonymous SNVs")
    df_sizes = run_analysis.calculate_possible_mutations(reference_processed)

    print("Calculating dN/dS and generating p-values")
    results = run_analysis.evaluate(counts_df, df_sizes)

    #postprocess.something(results)

if __name__ == "__main__":
    main()