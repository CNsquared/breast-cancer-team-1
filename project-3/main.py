import joblib
import pandas as pd
from src.utils.preprocess import MutPreprocessor
from src.utils.metadata_utils import load_metadata, merge_with_components
from src.models.nmf_runner import NMFDecomposer
from src.utils.enrichment_tests import test_association
from src.models.signature_comparator import load_sigprofiler_results, cosine_similarity

# paths
MUTATIONS_PATH = "data/raw/TCGA.BRCA.mutations.txt"
METADATA_PATH = "data/raw/TCGA.BRCA.metadata.txt"
SIGPROFILER_PATH = "data/raw/sigprofiler_results.txt"
ASSOCIATION_COLUMNS=['sex','age','cancer_subtype']

# NMF parameters - Identify best parameters for our dataset
NMF_PARAMS = {
    'n_components': 5,
    'objective_function': 'fro',
    'num_factorizations': 100,
    'random_state': 42,
    'resample_method': 'poisson',
    'normalization_method': 'GMM',
    'initialization_method': 'nndsvd',
}

def main():

    #Preprocess MAF and generate mutation matrix
    print("Preprocessing mutation data...")
    pre = MutPreprocessor(MUTATIONS_PATH)
    matrix = pre.get_mutation_matrix()
    X, sample_ids, feature_names = matrix['X'], matrix['sample_ids'], matrix['feature_names']
    df_mut = pre.get_processed_df()

    # save processed data
    df_mut.to_csv("data/processed/TCGA.BRCA.mutations.qc1.csv", index=False)
    joblib.dump(matrix, "data/processed/mutation_matrix.joblib")    

    # -----------------------------------------------------------
    # run NMF for some value of k, num_factorizations times

    print("Running NMF decomposition...")
    nmf_model = NMFDecomposer(**NMF_PARAMS)
    S_all, A_all = nmf_model.run(X)

    # save NMF results
    joblib.dump({'S_all': S_all, 'A_all': A_all}, 'data/processed/nmf_replicates.joblib')

    # -----------------------------------------------------------
    # cluster NMF results to build consensus S and A

    print("Partition clustering NMF results...")
    # TODO: alex's function/class
    S, A = alex_cluster.consensus(S_all, A_all)

    # -----------------------------------------------------------
    # annotate metadata and see if we can find associations with signatures

    # load metadata
    print("Loading and merging metadata...")
    metadata = load_metadata(METADATA_PATH)
    S_annotated = merge_with_components(S, sample_ids, metadata)

    # save cleaned metadata and annotated W
    metadata.to_csv("data/processed/TCGA.BRCA.metadata.qc1.csv", index=False)
    joblib.dump(S_annotated, "data/processed/S_annotated.joblib")

    # statistical association tests
    print("Testing associations with metadata...")
    results = test_association(S_annotated, test_cols=ASSOCIATION_COLUMNS)

    # save results
    results.to_csv("reports/enrichment_results.csv", index=False)
    
    # -----------------------------------------------------------
    # compare with sigprofiler results

    print("Loading and merging sigprofiler results...")
    sigprofiler = load_sigprofiler_results(SIGPROFILER_PATH)
    cos_sim = cosine_similarity(W, sigprofiler)

    # save cosine similarity results
    cos_sim_df = pd.DataFrame(cos_sim, columns=["sigprofiler_signature_1", "sigprofiler_signature_2", "cosine_similarity"])
    cos_sim_df.to_csv("reports/cosine_similarity_results.csv", index=False)
    print("All steps completed successfully.")    

if __name__ == "__main__":
    main()
