import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib_venn import venn2
from matplotlib_venn import venn3, venn3_circles
import seaborn as sns
import numpy as np

IntOGen = '/Users/zichenjiang/Downloads/BENG 285 projects SP25/breast-cancer-team-1/project-2/data/processed/IntOGen-DriverGenes_TCGA_WXS_BRCA.csv'
df_intogen = pd.read_csv(IntOGen, dtype={'Symbol': str, 'Mutations': int})
intogen_genes = list(df_intogen['Symbol'].unique())

dnds_qallsubs_cv_significant = ["TP53",     "PIK3CA",   "HIST1H3B", "AKT1",     "PTEN",     "FOXA1",    "CBFB",     "MAP2K4",   "CASP8",    "RUNX1",    "ERBB2",    "GATA3",    "MAP3K1",   "CDH1",     "NCOR1",    "KMT2C",    "RB1",      "ARID1A",   "NF1"]

# Define the columns representing mutation types in the desired plotting order
mutation_columns = [
    'synonymous',
    'Frame_Shift_Del',
    'Frame_Shift_Ins',
    'In_Frame_Del',
    'In_Frame_Ins',
    'Missense_Mutation',
    'Nonsense_Mutation',
    'Nonstop_Mutation',
    'Translation_Start_Site'
]

def plot_raw_stacked_bar(df, columns_to_plot=mutation_columns, top_n=46, IntOGen_list=intogen_genes, dNdScv_list=dnds_qallsubs_cv_significant, save_path="results/figures/raw_stacked_bar_plot.png"):
    """
    Generates a stacked bar plot for raw mutation counts and saves it.

    Args:
        df (pd.DataFrame): DataFrame with mutation counts
                           Must contain the columns specified in columns_to_plot, such as Hugo Symbol in the column.
        columns_to_plot (list): List of column names (mutation types) to include
                                in the sum and the stacked plot, in the desired
                                stacking order (bottom first).
        top_n (int): Number of top rows (genes) to plot based on the sum.
        save_path (str, optional): The full path (including filename and extension, e.g., .png)
                                   to save the plot. If None, the plot is not saved.
                                   Defaults to None.
    """
    # Ensure Hugo_Symbol is the index for easier plotting
    if 'Hugo_Symbol' in df.columns:
        df = df.set_index('Hugo_Symbol')

    # 1. Make a new column that sums each row's specified columns
    # Make sure all columns to plot actually exist in the DataFrame
    valid_columns_to_plot = [col for col in columns_to_plot if col in df.columns]
    if not valid_columns_to_plot:
        print("Error: None of the specified columns_to_plot exist in the DataFrame.")
        return
    elif len(valid_columns_to_plot) < len(columns_to_plot):
        missing_cols = set(columns_to_plot) - set(valid_columns_to_plot)
        print(f"Warning: The following columns were not found and ignored: {missing_cols}")

    df['Total_Mutations'] = df[valid_columns_to_plot].sum(axis=1)

    # 2. Sort by that column from largest to smallest
    df_sorted = df.sort_values('Total_Mutations', ascending=False)

    # 3. Select the top N rows
    df_top = df_sorted.head(top_n)

    # Check if df_top is empty after filtering
    if df_top.empty:
        print(f"No data to plot for the top {top_n} genes.")
        return

    # 4. Plot a stacked bar plot
    plt.style.use('seaborn-v0_8-talk') # Use a style for better aesthetics
    fig, ax = plt.subplots(figsize=(20, 8)) # Adjust figure size as needed

    # Plot the specified valid columns in the desired order
    df_top[valid_columns_to_plot].plot(kind='bar', stacked=True, ax=ax,
                                       colormap='tab10') # Use a colormap

    # Customize the plot
    # Labels and styling
    ax.set_title(f'Top {top_n} Most Mutated Genes by Raw Count in BRCA (n=773)', fontsize=22)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('Number of PASS Coding Mutations', fontsize=20)
    
    # --- X-axis annotation and labeling ---
    # Get the gene symbols from the index of df_top
    gene_symbols = df_top.index.tolist()
    x_ticks = range(len(gene_symbols)) # Generate numerical positions for ticks

    # Build annotated x labels
    annot_labels = []
    for gene in gene_symbols:
        label = ""
        if gene in IntOGen_list:
            label += '★'       # mark IntOGen drivers
        if gene in dNdScv_list:
            label += '▲'       # mark significant by dNdScv tool
        label += gene
        annot_labels.append(label)

    # Set x-ticks and annotated labels
    ax.set_xticks(x_ticks) # Set tick positions
    ax.set_xticklabels(annot_labels, rotation=45, fontsize=12, ha='right') # Set annotated labels
    ax.tick_params(axis='x', which='major', pad=1) # Make x tick labels closer to the axis
    ax.set_xlim(-0.75, len(gene_symbols) - 0.25) # Adjust x-limits for bar plot centering


    # --- Legends ---
    # 6. Original Legend for mutation types
    handles1, labels1 = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles1, labels1, title='Mutation Type', loc='right', title_fontsize=16)
    ax.add_artist(legend1) # Add the first legend manually

    # 7. Legend for annotation symbols
    legend_elements = [
        Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=8,
               label='IntOGen Driver'),
        Line2D([0], [0], marker='^', color='black', linestyle='None', markersize=8,
               label='dNdScv Significant')
    ]
    # Create the second legend
    legend2 = ax.legend(handles=legend_elements, title='Annotations', fontsize=12,
                        title_fontsize=16, loc='upper right')
    ax.add_artist(legend2)

    # 5. Save the plot if save_path is provided
    plt.savefig(save_path, bbox_inches='tight', dpi=500)




def plot_normalized_stacked_bar(df, columns_to_plot=mutation_columns, top_n=46, IntOGen_list=intogen_genes, dNdScv_list=dnds_qallsubs_cv_significant, save_path="results/figures/normalized_stacked_bar_plot.png"):

    # Ensure Hugo_Symbol is the index for easier plotting
    if 'Hugo_Symbol' in df.columns:
        df = df.set_index('Hugo_Symbol')

    # 1. Make a new column that sums each row's specified columns
    # Make sure all columns to plot actually exist in the DataFrame
    valid_columns_to_plot = [col for col in columns_to_plot if col in df.columns]
    if not valid_columns_to_plot:
        print("Error: None of the specified columns_to_plot exist in the DataFrame.")
        return
    elif len(valid_columns_to_plot) < len(columns_to_plot):
        missing_cols = set(columns_to_plot) - set(valid_columns_to_plot)
        print(f"Warning: The following columns were not found and ignored: {missing_cols}")

    df['Total_Mutations'] = df[valid_columns_to_plot].sum(axis=1)

    # 2. Sort by that column from largest to smallest
    df_sorted = df.sort_values('Total_Mutations', ascending=False)

    # 3. Select the top N rows
    df_top = df_sorted.head(top_n)

    # Check if df_top is empty after filtering
    if df_top.empty:
        print(f"No data to plot for the top {top_n} genes.")
        return

    # 4. Plot a stacked bar plot
    plt.style.use('seaborn-v0_8-talk') # Use a style for better aesthetics
    fig, ax = plt.subplots(figsize=(20, 8)) # Adjust figure size as needed

    # Plot the specified valid columns in the desired order
    df_top[valid_columns_to_plot].plot(kind='bar', stacked=True, ax=ax,
                                       colormap='tab10') # Use a colormap

    # Customize the plot
    # Labels and styling
    ax.set_title(f'Top {top_n} Most Mutated Genes Normalized by CDS Length in BRCA (n=773)', fontsize=22)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('Number of Normalized PASS Coding Mutations', fontsize=20)
    
    # --- X-axis annotation and labeling ---
    # Get the gene symbols from the index of df_top
    gene_symbols = df_top.index.tolist()
    x_ticks = range(len(gene_symbols)) # Generate numerical positions for ticks

    # Build annotated x labels
    annot_labels = []
    for gene in gene_symbols:
        label = ""
        if gene in IntOGen_list:
            label += '★'       # mark IntOGen drivers
        if gene in dNdScv_list:
            label += '▲'       # mark significant by dNdScv tool
        label += gene
        annot_labels.append(label)

    # Set x-ticks and annotated labels
    ax.set_xticks(x_ticks) # Set tick positions
    ax.set_xticklabels(annot_labels, rotation=45, fontsize=12, ha='right') # Set annotated labels
    ax.tick_params(axis='x', which='major', pad=1) # Make x tick labels closer to the axis
    ax.set_xlim(-0.75, len(gene_symbols) - 0.25) # Adjust x-limits for bar plot centering


    # --- Legends ---
    # 6. Original Legend for mutation types
    handles1, labels1 = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles1, labels1, title='Mutation Type', loc='right', title_fontsize=16)
    ax.add_artist(legend1) # Add the first legend manually

    # 7. Legend for annotation symbols
    legend_elements = [
        Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=8,
               label='IntOGen Driver'),
        Line2D([0], [0], marker='^', color='black', linestyle='None', markersize=8,
               label='dNdScv Significant')
    ]
    # Create the second legend
    legend2 = ax.legend(handles=legend_elements, title='Annotations', fontsize=12,
                        title_fontsize=16, loc='upper right')
    ax.add_artist(legend2)

    # 5. Save the plot if save_path is provided
    plt.savefig(save_path, bbox_inches='tight', dpi=500)



def plot_dNdS_stacked_bar(df, columns_to_plot=["dN/dS"], top_n=46, IntOGen_list=intogen_genes, dNdScv_list=dnds_qallsubs_cv_significant, pval_col='fisher_pval', pval_threshold=0.05, save_path="results/figures/dNdS_stacked_bar_plot.png"):

    # Ensure Hugo_Symbol is the index for easier plotting
    if 'Hugo_Symbol' in df.columns:
        df = df.set_index('Hugo_Symbol')
    else:
        print("Error: 'Hugo_Symbol' column not found.")
        return
    
    # Check if p-value column exists
    if pval_col not in df.columns:
        print(f"Warning: P-value column '{pval_col}' not found. Skipping '+' annotations.")
        add_pval_markers = False
    else:
        add_pval_markers = True
         # Ensure p-value column is numeric, coerce errors to NaN
        df[pval_col] = pd.to_numeric(df[pval_col], errors='coerce')

    # 1. Make a new column that sums each row's specified columns
    # Make sure all columns to plot actually exist in the DataFrame
    valid_columns_to_plot = [col for col in columns_to_plot if col in df.columns]
    if not valid_columns_to_plot:
        print("Error: None of the specified columns_to_plot exist in the DataFrame.")
        return
    elif len(valid_columns_to_plot) < len(columns_to_plot):
        missing_cols = set(columns_to_plot) - set(valid_columns_to_plot)
        print(f"Warning: The following columns were not found and ignored: {missing_cols}")

    df['Total_Mutations'] = df[valid_columns_to_plot].sum(axis=1)

    # 2. Sort by that column from largest to smallest
    df_sorted = df.sort_values('Total_Mutations', ascending=False)

    # 3. Select the top N rows
    df_top = df_sorted.head(top_n)

    # Check if df_top is empty after filtering
    if df_top.empty:
        print(f"No data to plot for the top {top_n} genes.")
        return

    # 4. Plot a stacked bar plot
    plt.style.use('seaborn-v0_8-talk') # Use a style for better aesthetics
    fig, ax = plt.subplots(figsize=(20, 8)) # Adjust figure size as needed

    # Plot the specified valid columns in the desired order
    df_top[valid_columns_to_plot].plot(kind='bar', stacked=True, ax=ax,
                                       colormap='tab10') # Use a colormap

    # Customize the plot
    # Labels and styling
    ax.set_title(f'Top {top_n} Most Mutated Genes by Raw Count in BRCA (n=773)', fontsize=22)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('dN/dS', fontsize=20)
    
    # --- X-axis annotation and labeling ---
    # Get the gene symbols from the index of df_top
    gene_symbols = df_top.index.tolist()
    x_ticks = range(len(gene_symbols)) # Generate numerical positions for ticks

    # Build annotated x labels
    annot_labels = []
    for gene in gene_symbols:
        label = ""
        if gene in IntOGen_list:
            label += '★'       # mark IntOGen drivers
        if gene in dNdScv_list:
            label += '▲'       # mark significant by dNdScv tool
        label += gene
        annot_labels.append(label)

    # Set x-ticks and annotated labels
    ax.set_xticks(x_ticks) # Set tick positions
    ax.set_xticklabels(annot_labels, rotation=45, fontsize=12, ha='right') # Set annotated labels
    ax.tick_params(axis='x', which='major', pad=1) # Make x tick labels closer to the axis
    ax.set_xlim(-0.75, len(gene_symbols) - 0.25) # Adjust x-limits for bar plot centering


    # --- Legends ---
    # 6. Original Legend for mutation types
    handles1, labels1 = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles1, labels1, title='Mutation Type', loc='right', title_fontsize=16)
    ax.add_artist(legend1) # Add the first legend manually

    # 7. Legend for annotation symbols
    legend_elements = [
        Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=8,
               label='IntOGen Driver'),
        Line2D([0], [0], marker='^', color='black', linestyle='None', markersize=8,
               label='dNdScv Significant')
    ]
    # Create the second legend
    legend2 = ax.legend(handles=legend_elements, title='Annotations', fontsize=12,
                        title_fontsize=16, loc='upper right')
    ax.add_artist(legend2)


    # --- Add '+' Annotations for significant Fisher Q-value ---
    our_significant_dnds_genes_by_fischer_for_venn_diagram = []
    if add_pval_markers:
        # Calculate total height for each bar (sum of plotted columns for df_top)
        # Ensure we use the same fillna logic as in plotting
        # Disable the SettingWithCopyWarning
        pd.options.mode.chained_assignment = None
        df_top['Total_Height'] = df_top[valid_columns_to_plot].fillna(0).sum(axis=1)

        for i, gene in enumerate(gene_symbols):
            # Check p-value condition (handle potential NaNs)
            pval = df_top.loc[gene, pval_col]
            if pd.notna(pval) and pval < pval_threshold:
                our_significant_dnds_genes_by_fischer_for_venn_diagram.append(gene)
                bar_height = df_top.loc[gene, 'Total_Height']
                # Place '+' at the index of the gene name and slightly above the dNdS y value
                ax.text(i, bar_height, '+', ha='center', va='bottom', fontsize=14, color='red', fontweight='bold') # Make it stand out



    # 5. Save the plot if save_path is provided
    plt.savefig(save_path, bbox_inches='tight', dpi=500)



    dnds_col_hist = "dN/dS" # Assuming this is the standard column name
    if dnds_col_hist not in df.columns:
        print(f"Warning: Column '{dnds_col_hist}' not found. Cannot generate histogram.")
        return
    else:
        # Convert dN/dS column to numeric for histogram, coercing errors
        df[dnds_col_hist] = pd.to_numeric(df[dnds_col_hist], errors='coerce')
        # Drop rows where dN/dS could not be converted for histogram plotting
        df_hist_data = df.dropna(subset=[dnds_col_hist])
        
    fig2, ax2 = plt.subplots(figsize=(10, 6)) # New figure for histogram

    # Prepare data for histogram: separate significant and non-significant
    dnds_values = df_hist_data[dnds_col_hist]
    significant_dnds = pd.Series(dtype=float) # Initialize empty series
    non_significant_dnds = dnds_values # Start with all values

    if add_pval_markers: # Only separate if pval column is valid
        is_significant = (df_hist_data[pval_col] < pval_threshold) & pd.notna(df_hist_data[pval_col])
        significant_dnds = df_hist_data.loc[is_significant, dnds_col_hist]
        non_significant_dnds = df_hist_data.loc[~is_significant, dnds_col_hist]
    print(our_significant_dnds_genes_by_fischer_for_venn_diagram, " our dN/dS significant genes")

    # Define bins (calculate once for consistency)
    # Adjust range based on your data, maybe exclude extreme outliers if needed
    min_val = max(-1, dnds_values.min()) # Limit lower bound if desired
    max_val = min(15, dnds_values.max())+0.5 # Limit upper bound if desired
    bins = np.linspace(min_val, max_val, 101) # 100 bins between min_val and max_val

    # Plot histograms: non-significant first, then significant on top
    ax2.hist(non_significant_dnds.clip(min_val, max_val), bins=bins, color='gray', label='Non-significant', log=True)
    if not significant_dnds.empty:
        ax2.hist(significant_dnds.clip(min_val, max_val), bins=bins, color='red', label=f'Significant ({pval_col} < {pval_threshold})', log=True)

    # Histogram customization
    ax2.set_xlabel("dN/dS")
    ax2.set_ylabel("Frequency (log scale)")
    ax2.set_title("Distribution of dN/dS", fontsize=16)
    ax2.set_xlim(min_val, max_val) # Use calculated limits
    # ax2.set_yscale('log') # Already set in hist call
    ax2.grid(False)
    ax2.axvline(x=1, color='black', linestyle='--') # Vertical line at x=1
    if not significant_dnds.empty: # Only add legend if significant data exists
        ax2.legend()

    fig2.tight_layout()

    
    plt.savefig('results/figures/dNdS_histogram.png', bbox_inches='tight', dpi=500)



    # 1. Take all dN/dS significant genes
    # 3. Convert to sets
    set_top_n = set(our_significant_dnds_genes_by_fischer_for_venn_diagram)
    set_dNdScv = set(dNdScv_list)
    set_intogen = set(IntOGen_list)

    # 4. Plot Venn diagram
    plt.figure(figsize=(5, 5), dpi=500) # Adjusted figure size for 3 sets

    # Create the Venn diagram - store the output to modify fonts
    v = venn3([set_top_n, set_dNdScv, set_intogen],
            set_labels=('Top '+ str(top_n) + ' dN/dS', 'dNdScv traditional model', 'IntOGen Drivers'),
            set_colors=('skyblue', 'lightgreen', 'lightcoral'), # Optional: set colors
            alpha=0.7 # Optional: set transparency
            )
    plt.savefig('results/figures/dNdS_venn_digram.png', bbox_inches='tight', dpi=500)