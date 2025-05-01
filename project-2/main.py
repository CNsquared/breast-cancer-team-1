from src.pipeline import preprocess, run_analysis, postprocess

PROCESSED_OUTPUT= 'data/processed/TCGA.BRCA.mutations.qc1.txt'

def main():

    print("Preprocesssing mutation data...")
    mutations_processed = preprocess.preprocess_mutations()
    mutations_processed.to_csv(PROCESSED_OUTPUT, sep='\t', index=False)
    print(f"Preprocessed data saved to {PROCESSED_OUTPUT}")
    
    print("Preprocessing reference data...")
    reference_processed = preprocess.preprocess_reference()

    print("Running simple dN/dS Analysis")
    dnds_simple_results = run_analysis.run_dnds_simple(mutations_processed, reference_processed)

    #postprocess.something(dnds_simple_results)

    print("Job completed successfully!")

if __name__ == "__main__":
    main()