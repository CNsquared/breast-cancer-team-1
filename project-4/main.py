from src.utils.preprocess import MutPreprocessor
from src.utils.preprocess import GeneExpPreprocessor
from src.models.autoencoder_runner import GeneExpressionRunner
from src.models.autoencoder import GeneExpressionAutoencoder
from src.utils.data_loader import load_expression_matrix
import numpy as np
import pandas as pd

RUN_EACH_SUBTYPE = True

def main():
    SUBTYPE='BRCA_Basal' 
    LATENT_DIM=5
    # preprocess expression data
    df_exp = GeneExpPreprocessor(save=True, subtypes=[SUBTYPE]).get_df()
    # run cross validation
    runner = GeneExpressionRunner(df_exp, latent_dim=LATENT_DIM)
    cv_losses = runner.cross_validate()
    print(f'cv_losses: {cv_losses}')

    # train on all samples and get latent space
    latent = runner.train_all_and_encode()
    df_latent = pd.DataFrame(latent, index=df_exp.index, columns=[f"latent_{i}" for i in range(latent.shape[1])])
    df_latent.to_csv(f"results/tables/latent_space_{LATENT_DIM}dim_{SUBTYPE}.csv")
    cv_losses_df = pd.DataFrame(cv_losses, index=[f"fold_{i+1}" for i in range(len(cv_losses))])
    cv_losses_df.to_csv(f"results/tables/cv_losses_{LATENT_DIM}dim_{SUBTYPE}.csv", index=False, header=False)

def each_subtype():
    LATENT_DIM=5
    all_cv_losses = []
    subtypes = ['BRCA_LumA', 'BRCA_Her2', 'BRCA_LumB', 'BRCA_Normal', 'BRCA_Basal']
    for subtype in subtypes:
        print(f"Running for subtype: {subtype}")

        # preprocess expression data
        df_exp = GeneExpPreprocessor(subtypes=[subtype]).get_df()

        # run cross validation
        runner = GeneExpressionRunner(df_exp, latent_dim=LATENT_DIM)
        cv_losses = runner.cross_validate()
        all_cv_losses.append(cv_losses)
        print(f'cv_losses: {cv_losses}')

        # train on all samples and get latent space
        latent = runner.train_all_and_encode()
        df_latent = pd.DataFrame(latent, index=df_exp.index, columns=[f"latent_{i}" for i in range(latent.shape[1])])
        df_latent.to_csv(f"results/tables/latent_space_{LATENT_DIM}dim_{subtype}.csv")
    all_cv_losses_df = pd.DataFrame(all_cv_losses, index=subtypes, columns=[f"fold_{i+1}" for i in range(len(all_cv_losses[0]))])
    all_cv_losses_df.to_csv(f"results/tables/cv_losses_{LATENT_DIM}dim_all_subtypes.csv", index=True)

if __name__ == "__main__":
    main()
    if RUN_EACH_SUBTYPE:
        each_subtype()
