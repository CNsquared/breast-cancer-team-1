from src.pipeline import preprocess, run_analysis, postprocess

def main():

    print("Preprocesssing mutation data...")
    data = preprocess.preprocess()
    data.to_csv('data/processed/TCGA.BRCA.mutations.qc1.txt')

    #results = run_analysis.something(data)

    #postprocess.something(results)

if __name__ == "__main__":
    main()