from src.pipeline import preprocess, run_analysis, postprocess

PROCESSED_OUTPUT= 'data/processed/TCGA.BRCA.mutations.qc1.txt'

def main():

    print("Preprocesssing mutation data...")
    data = preprocess.preprocess()
    data.to_csv(PROCESSED_OUTPUT)
    print(f"Preprocessed data saved to {PROCESSED_OUTPUT}")

    #results = run_analysis.something(data)

    #postprocess.something(results)
    print("Job completed successfully!")

if __name__ == "__main__":
    main()