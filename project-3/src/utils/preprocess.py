import pandas as pd
import numpy as np
from mutation_matrix import build_mutation_matrix, normalize_matrix

class MutPreprocessor:
    """This class preprocesses the MAF file and generates a mutation matrix."""
    def __init__(
            self, maf_path: str, 
            keep_variant_classes: list[str] = ['Missense_Mutation', 'Nonsense_Mutation', 'Silent']
        ):
        
        self.maf_path = maf_path
        self.keep_variant_classes = keep_variant_classes
        self.df_mut = None
        self.sample_ids = None
        self.feature_names = None
        self.X = None
        self._preprocess()

    def _load_maf(self):
        self.df_mut = pd.read_csv(self.maf_path, sep='\t', low_memory=False)
        print(f"Loaded {len(self.df_mut)} total mutations.")

    def _filter_mutations(self):
        before = len(self.df_mut)
        # TODO: do some filtering
        print(f"Filtered from {before} to {len(self.df_mut)} mutations.")

    def _generate_mutation_matrix(self):
        # this is all made up
        # TODO: implement generate mutation matrix either here or in mutation_matrix.py
        self.sample_ids = self.df_mut['patient_id'].unique()
        self.feature_names = self.df_mut['Hugo_Symbol'].unique()
        self.X = build_mutation_matrix(self.df_mut)

    def _preprocess(self):
        self._load_maf()
        self._filter_mutations()
        self._generate_mutation_matrix()

    def get_processed_df(self):
        return self.df_mut

    def get_mutation_matrix(self) -> tuple[np.ndarray, list[str], list[str]]:
        # TODO: double check if this makes sense
        return {
            'X': self.X,
            'sample_ids': self.sample_ids,
            'feature_names': self.feature_names
        }