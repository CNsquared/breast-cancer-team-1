import pandas as pd
import numpy as np

from src.utils.tpm_normalization import parse_gencode_fasta, longest_CDS, log1p_TPM_normalize
from sklearn.preprocessing import StandardScaler


class GeneExpPreprocessor:
    """This class preprocesses the gene expression file"""
    def __init__(
            self, expr_path: str = 'data/raw/TCGA.BRCA.expression.txt',
            top_N: int = 500,
            dndscv_path: str = 'data/raw/dndscv.csv',
            output_path: str = 'data/processed/processed_expression.tsv',
            subset_method: str = 'dndscv',
            gencode_fasta_path: str = 'data/raw/gencode.v23lift37.pc_transcripts.fa',
            save: bool = False,
            auto_preprocess: bool = True,
            filter_var_percentile_threshold: float = 0.5, # e.g. 0.1 means keep top 90% variable genes
            filter_exp_threshold: float = 1,
            metadata_path: str = 'data/raw/brca_tcga_pan_can_atlas_2018_clinical_data_PAM50_subype_and_progression_free_survival.tsv',
            subtypes: list[str] = ['BRCA_Basal'], # ['BRCA_LumA', 'BRCA_Her2', 'BRCA_LumB', 'BRCA_Normal', 'BRCA_Basal']
            filter_subtypes: bool = True
        ):
        self.expr_path = expr_path
        self.save = save
        self.dndscv_path = dndscv_path
        self.top_N = top_N
        self.output_path = output_path
        self.subset_method = subset_method
        self.gencode_fasta_path = gencode_fasta_path
        self.filter_var_percentile_threshold = filter_var_percentile_threshold
        self.filter_exp_threshold = filter_exp_threshold
        self.metadata_path = metadata_path
        self.subtypes = subtypes
        self.filter_subtypes = filter_subtypes
        if auto_preprocess:
            self._preprocess()

    def _load_expr(self) -> None:
        # Load expression data with first two columns as index
        self.df_exp = pd.read_csv(self.expr_path, index_col=[0, 1], sep='\t', low_memory=False)
        self.df_exp = self.df_exp.astype(float)
        print(f"Loaded matrix with {self.df_exp.shape[0]} samples x {self.df_exp.shape[1]} genes")

    def _filter_subtype(self) -> None:
        """Filter by subtype"""
        if self.filter_subtypes:
            print(f'Subset data to patients with subtype: {self.subtypes}')
            metadata = pd.read_csv(self.metadata_path, sep='\t', index_col=1)
            subset = metadata.loc[metadata.Subtype.isin(self.subtypes)]
            self.df_exp = self.df_exp.loc[self.df_exp.index.get_level_values('patient_id').intersection(subset.index)]
            print(f'Subset data to {len(self.df_exp.index.get_level_values("patient_id").unique())} patients with subtype: {self.subtypes}')

    def _filter_expr(self) -> None:
        """Filter out low variance genes and low expression genes
        This should be run after log1p(tpm)
        """
        print(f'Filtering out lowly expressed genes with mean expression of {self.filter_exp_threshold}')
        # filter out lowly expressed genes
        exp_mean = self.df_exp.mean()
        self.df_exp = self.df_exp.loc[:, exp_mean > self.filter_exp_threshold]

        print(f'Filtering out lowest {self.filter_var_percentile_threshold*100}% of genes')
        # remove genes whose variance < 1
        gene_variance = self.df_exp.var()

        # determine the variance cutoff 
        var_cutoff = gene_variance.quantile(self.filter_var_percentile_threshold)

        # filter expressio matrix
        self.df_exp = self.df_exp.loc[:, gene_variance > var_cutoff]
        
    def _subset(self) -> None:
        """Subset the expression data to only include top N genes"""
        print(f"Subsetting to top {self.top_N} genes using method: {self.subset_method}")
        if self.subset_method == 'dndscv':
            # Load dndscv data
            df_dndscv = pd.read_csv(self.dndscv_path, sep=',', low_memory=False)
            # filter to only include genes in the expression data
            df_dndscv = df_dndscv[df_dndscv['gene_name'].isin(self.df_exp.columns)]
            # Get the top N genes
            top_genes = df_dndscv.nlargest(self.top_N, 'wmis_cv')['gene_name']
            # Subset the expression data
            self.df_exp = self.df_exp.loc[:,top_genes]
        elif self.subset_method == 'variance':
            gene_variance = self.df_exp.var(axis=0)  # axis=0 for gene-wise variance
            # Get the top N genes
            top_genes = gene_variance.nlargest(self.top_N).index
            # Subset the expression data
            self.df_exp = self.df_exp.loc[:,top_genes]
        else:
            raise ValueError(f"Unknown subset method: {self.subset_method}. Use 'dndscv' or 'variance'.")
        
    def _log1p_tpm_normalization(self) -> None: 
        # this should be run before subsetting for better normalization
        df_lengths = parse_gencode_fasta(self.gencode_fasta_path)
        df_lengths = longest_CDS(df_lengths)
        self.df_exp = log1p_TPM_normalize(self.df_exp, df_lengths)

    def _preprocess(self) -> None:
        self._load_expr()
        self._log1p_tpm_normalization()
        self._filter_subtype()
        self._filter_expr()
        self._subset()
        
        if self.save:
            self.df_exp.to_csv(self.output_path, sep='\t', index=True)

    def get_df(self):
        return self.df_exp
    

def generateFeatureSpace():
    print('generating initial feature space....')
    metadata = pd.read_csv('data/raw/brca_tcga_pan_can_atlas_2018_clinical_data_PAM50_subype_and_progression_free_survival.tsv', sep='\t')

    # subset to female
    metadata = metadata[metadata['Sex'] == 'Female']

    # subset to stages
    metadata = metadata[metadata['Neoplasm Disease Stage American Joint Committee on Cancer Code'].notna()]
    metadata = metadata[metadata['Neoplasm Disease Stage American Joint Committee on Cancer Code'] != 'STAGE X']
    metadata['Stage'] = metadata['Neoplasm Disease Stage American Joint Committee on Cancer Code']

    # has a PAM50 subtype
    metadata = metadata[metadata['Subtype'].notna()]

    # adjust ethnicity
    metadata['Race Category'] = metadata['Race Category'].apply(lambda x: 'White' if x == 'White' else 'Non-white')

    # add outcome 
    metadata['PFI_over60mo'] = metadata.apply(lambda row: 'No' if (row['Progression Free Status'] == '1:PROGRESSION') and (row['Progress Free Survival (Months)'] < 60) else 'Yes', axis=1)

    # divide age by 10
    metadata['Diagnosis Age'] = metadata['Diagnosis Age']/10

    featureSpace_metadata = metadata[['Patient ID', 'Sample ID', 'Diagnosis Age', 'Race Category', 'Tumor Type', 'Subtype', 'PFI_over60mo']]


    mutations = pd.read_csv('data/raw/TCGA.BRCA.mutations.txt', sep="\t")

    # Filter for nonsynoymous pass SBS and indels that cause in top 10 IntOGen breast cancer drivers. They end up being Missense_Mutation, Frame_Shift_Del, Nonsense_Mutation, Frame_Shift_Ins, Splice_Site, In_Frame_Del
    pass_sbs_indels = mutations[mutations["FILTER"] == "PASS"]
    pass_sbs_indels_in_top10_intogen_breast_drivers = pass_sbs_indels[pass_sbs_indels["Hugo_Symbol"].isin(["TP53", "PIK3CA", "GATA3", "KMT2C", "CDH1", "MAP3K1", "ESR1", "PTEN", "AKT1", "ARID1A"])]
    pass_nonsyn_sbs_indels_in_top10_intogen_breast_drivers = pass_sbs_indels_in_top10_intogen_breast_drivers[~pass_sbs_indels_in_top10_intogen_breast_drivers["Variant_Classification"].isin(["Silent", "Intron", "3'Flank", "3'UTR", "5'UTR"])]
    pass_nonsyn_sbs_indels_in_top10_intogen_breast_drivers_feature_space = pd.crosstab(
        index=pass_nonsyn_sbs_indels_in_top10_intogen_breast_drivers['patient_id'],
        columns=pass_nonsyn_sbs_indels_in_top10_intogen_breast_drivers['Hugo_Symbol']
    )
        # The highest mutated primary samples have only 11 nonsyn sbs and indels in these top 10 genes
    # pass_nonsyn_sbs_indels_in_top10_intogen_breast_drivers_feature_space.sum(axis=1).nlargest(5)

    metadata_with_mutation_feature_space = pd.merge(
        featureSpace_metadata,
        pass_nonsyn_sbs_indels_in_top10_intogen_breast_drivers_feature_space,
        
        left_on="Patient ID",
        right_index=True,
        how="left"
    )
    
    # Fill NA values in any of these 10 mutation columns with 0, so patients with no nonsyn mutations in these genes are represented as 0
    gene_columns = ['AKT1', 'ARID1A', 'CDH1', 'ESR1', 'GATA3', 'KMT2C', 'MAP3K1', 'PIK3CA', 'PTEN', 'TP53']
    initial_na_rows = metadata_with_mutation_feature_space[gene_columns].isna().all(axis=1).sum()
    metadata_with_mutation_feature_space[gene_columns] = metadata_with_mutation_feature_space[gene_columns].fillna(0)
    final_na_rows = metadata_with_mutation_feature_space[gene_columns].isna().all(axis=1).sum()
    print(f"Number of rows with all NA in mutation columns (before): {initial_na_rows}")
    print(f"Number of rows with all NA in mutation columns (after): {final_na_rows}")
        

    print("Adding latent dims to feature space...")
    
    latent_dims = pd.DataFrame()
    filenames = ['data/raw/latent_space_5dim_BRCA_Normal.csv',
                 'data/raw/latent_space_5dim_BRCA_Her2.csv',
                 'data/raw/latent_space_5dim_BRCA_LumA.csv',
                 'data/raw/latent_space_5dim_BRCA_LumB.csv',
                 'data/raw/latent_space_5dim_BRCA_Basal.csv']

    for f in filenames:
        print(f)
        new_df = pd.read_csv(f)
        latent_dims = pd.concat([latent_dims, new_df])
    
    featureMatrix_mutations_latent_dims = pd.merge(
        metadata_with_mutation_feature_space, latent_dims,
        left_on=['Patient ID'],
        right_on=['patient_id'],
        how='left'
    )

    return featureMatrix_mutations_latent_dims
