from src.pipeline import preprocess
from src.pipeline import run_analysis
from src.pipeline import postprocess

if __name__ == "__main__":

    print("Running pipeline")
    print("Preprocessing data")
    preprocess()
    print("Running analysis")
    run_analysis()
    print("Postprocessing data")
    postprocess()