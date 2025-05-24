from src.utils.preprocess import MutPreprocessor
from src.utils.preprocess import GeneExpPreprocessor
from src.models.autoencoder import Autoencoder
from src.utils.data_loader import load_expression_matrix

def main():
    # preprocess expression data
    df_exp = GeneExpPreprocessor(save=True).get_df()

if __name__ == "__main__":
    main()
