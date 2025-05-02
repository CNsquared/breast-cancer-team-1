import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib_venn import venn2
from matplotlib_venn import venn3, venn3_circles
import seaborn as sns
import numpy as np

def plot_raw_mutation_counts(dnds_simple_results):
    """
    Preprocesses mutation data

    Returns:
        df_mut (pd.DataFrame): Processed mutation data
    """
    syn_col = 'synonymous'
    non_syn_classes = [
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Translation_Start_Site",
        "Nonstop_Mutation",
    ] #  and Nonstop_Mutation don't have recurrently mutated genes
    all_classes = [syn_col] + non_syn_classes

    # 3. Group and pivot to get counts per gene per class
    mutation_counts = dnds_simple_results.groupby(['Hugo_Symbol', 'mutation_class']).size()
    counts_df = mutation_counts.unstack(fill_value=0)
    
    # 4. Ensure all columns exist
    for col in all_classes:
        if col not in counts_df.columns:
            counts_df[col] = 0

    # 5. Reorder columns
    counts_df = counts_df[all_classes]

    # 6. Optional: select top N genes by total mutations
    counts_df['TotalMutations'] = counts_df.sum(axis=1)
    N = 46
    top_n = counts_df.sort_values('TotalMutations', ascending=False).head(N)
    plot_df = top_n.drop(columns='TotalMutations')
    top_46_genes_raw_count = list(plot_df.index)

    # 7. Define a color palette: one blue for synonymous, shades of red for non-synonymous
    palette = [
        'cadetblue',      # synonymous
        '#fcae91',        # Missense_Mutation (light red) 
        '#fb6a4a',        # Nonsense_Mutation
        "tab:orange",  # Translation_Start_Site
        "red",  # Nonstop_Mutation
    ]

    # 8. Plot
    if plot_df.empty:
        print("No data available to plot.")
    else:
        print(", ".join(top_46_genes_raw_count))
        print(f"Generating stacked bar plot for top {(len(set(top_46_genes_raw_count)))} genes...")
        fig, ax = plt.subplots(
            figsize=(max(10, len(plot_df) * 0.3), 8),
            dpi=500
        )

        # Plot on this axis
        plot_df.plot(
            kind='bar',
            stacked=True,
            width=0.8,
            color=palette,
            ax=ax
        )

        # Labels and styling
        ax.set_title(f'Top {N} Most Mutated Genes in BRCA (n=773)', fontsize=22)
        ax.set_xlabel('', fontsize=18)
        ax.set_ylabel('Number of SBS', fontsize=20)
        ax.set_xticklabels(plot_df.index, rotation=45, fontsize=12, ha='right')
        ax.tick_params(axis='x', which='major', pad=1)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(title='Mutation Class', fontsize=16, title_fontsize=18)
        ax.grid(False)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 400)

        plt.tight_layout()