import pandas as pd
import numpy as np

from src.utils.tpm_normalization import parse_gencode_fasta, longest_CDS, log1p_TPM_normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as TTS

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
            filter_exp_threshold: float = 1
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
        if auto_preprocess:
            self._preprocess()

    def _load_expr(self) -> None:
        # Load expression data with first two columns as index
        self.df_exp = pd.read_csv(self.expr_path, index_col=[0, 1], sep='\t', low_memory=False)
        self.df_exp = self.df_exp.astype(float)
        print(f"Loaded matrix with {self.df_exp.shape[0]} samples x {self.df_exp.shape[1]} genes")

    def _filter(self) -> None:
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
        self._filter()
        self._subset()
        self._normalize()
        if self.save:
            self.df_exp.to_csv(self.output_path, sep='\t', index=True)

    def _normalize(self) -> None:
        """Z-score genes"""
        print("Normalizing (Z-score) Gene Expression")
        self.raw_tpm = self.df_exp.copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df_exp)
        self.df_exp = pd.DataFrame(X_scaled, index=self.df_exp.index, columns=self.df_exp.columns)

    def get_df(self):
        return self.df_exp
    
class MutPreprocessor:
    """This class preprocesses the MAF file"""
    # TODO: this doesn't really do anything right now, other than put a copy of the MAF in processed folder
    def __init__(
            self, maf_path: str = 'data/raw/TCGA.BRCA.mutations.txt',
            output_path: str = 'data/processed/processed_mutations.tsv',
            save: bool = False
        ):
        self.save = save
        self.output_path = output_path
        self.maf_path = maf_path
        self._preprocess()

    def _load_maf(self):
        self.df_mut = pd.read_csv(self.maf_path, sep='\t', low_memory=False, index_col=0)
        print(f"Loaded {len(self.df_mut)} total mutations.")

    def _preprocess(self):
        self._load_maf()
        if self.save:
            self.df_mut.to_csv(self.output_path, sep='\t', index=True)

    def get_preprocess_df(self):
        return self.df_mut