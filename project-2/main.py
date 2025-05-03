from src.pipeline import preprocess, run_analysis, postprocess, plot

import pandas as pd

def main():

    print("Preprocesssing mutation data...")
    mutations_processed = preprocess.preprocess_mutations()

    print(f"Processing Gencode CDS FASTA sequence using the Transcript ID or Hugo Symbol in the mutation data and getting lengths...")
    df_sizes = preprocess.filter_fasta_and_get_cds_lengths()
    
    print("Get normalized counts...")
    normalized_counts = run_analysis.run_CDS_length_normalized(mutations_processed)
    normalized_counts.to_csv('results/tables/normalized_counts.tsv', sep="\t", index=True) # index=True to keep Hugo_Symbol
    print("Normalized counts saved to results/tables/normalized_counts.tsv")

    print("Running simple dN/dS Analysis")
    dnds_simple_results = run_analysis.run_dnds_simple(mutations_processed, df_sizes)
    dnds_simple_results.to_csv('results/tables/dnds_simple_results.tsv', sep="\t", index=False)
    print("dN/dS simple results saved to results/tables/dnds_simple_results.tsv")


    plot.plot_raw_stacked_bar(dnds_simple_results, top_n=46, save_path="results/figures/raw_stacked_bar_plot.png")
    print("Raw count stacked bar plots using results/tables/dnds_simple_results.tsv saved to results/figures/raw_stacked_bar_plots.png")
    
    plot.plot_normalized_stacked_bar(normalized_counts, top_n=46, save_path="results/figures/normalized_stacked_bar_plot.png")
    print("CDS length normalized stacked bar plots using results/tables/normalized_counts.tsv saved to results/figures/normalized_stacked_bar_plots.png")

    plot.plot_dNdS_stacked_bar(dnds_simple_results, columns_to_plot=["dN/dS"], top_n=46, pval_col='fisher_pval', pval_threshold=0.05, save_path="results/figures/dNdS_stacked_bar_plot.png")
    print("dN/dS stacked bar plots using results/tables/dnds_simple_results.tsv saved to results/figures")

    print("Running Evaluation Metrics on dN/dS results against IntOGen...")
    df_gene_ranks, dcg, bpref, accuracy = postprocess.get_intogen_ranks()
    df_gene_ranks.to_csv('results/tables/dnds_simple_results_intogen.tsv', sep="\t", index=False)
    print("dN/dS results compared to IntOGen saved to results/tables/dnds_simple_results_intogen.tsv")
    summary = pd.DataFrame({
        "DCG": [dcg],
        "Bpref": [bpref],
        "Accuracy": [accuracy]
    })
    summary.to_csv('results/tables/dnds_simple_results_summary.tsv', sep="\t", index=False)
    print("Evaluation metrics saved to results/tables/dnds_simple_results_summary.tsv")

    print("Job completed successfully!")

if __name__ == "__main__":
    main()
