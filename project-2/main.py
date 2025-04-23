from src.pipeline import preprocess, run_analysis, postprocess

def main():

    data = preprocess.something()

    results = run_analysis.something(data)

    postprocess.something(results)

if __name__ == "__main__":
    main()