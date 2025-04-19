def create_pam50_embeddings():
    import pandas as pd
    import numpy as np
    from openTSNE import TSNE
    
    # Load the CSV file
    df = pd.read_csv("expression_data_with_subtype.csv")
    expression_full = df.drop(columns=["patient_id", "sample_id", "Subtype"])
    y = df["Subtype"].values
    
    #normalize
    q_low_full = np.quantile(expression_full, 0.025, axis=0)
    q_high_full = np.quantile(expression_full, 0.975, axis=0)

    # Robust standardization: For each gene, subtract the 0.025 quantile and scale by the interquantile range
    X_robust_full = (expression_full - q_low_full) / (q_high_full - q_low_full)
    
    #select pam50
    pam50_genes = [
    "UBE2T", "BIRC5", "NUF2", "CDC6", "CCNB1", "TYMS", "MYBL2", "CEP55", "MELK", "NDC80", "RRM2", "UBE2C", "CENPF", "PTTG1", "EXO1", "ORC6L", "ANLN", "CCNE1", "CDC20", "MKI67", "KIF2C", "ACTR3B", "MYC", "EGFR", "KRT5", "PHGDH", "CDH3", "MIA", "KRT17", "FOXC1", "SFRP1", "KRT14", "ESR1", "SLC39A6", "BAG1", "MAPT", "PGR", "CXXC5", "MLPH", "BCL2", "MDM2", "NAT1", "FOXA1", "BLVRA", "MMP11", "GPR160", "FGFR4", "GRB7", "TMEM45B", "ERBB2"
    ]
    expr_pam50 = X_robust_full[pam50_genes]
    
    def tsne_reduce(data_no_label,n_comp = 5, r_seed = 42):
        embedding_tsne = TSNE(n_components=n_comp, perplexity=90, random_state= r_seed).fit(data_no_label)
        #print('t-SNE running time is: ' + str(t1-t0) + ' s' )

        return embedding_tsne
    
    
    #create embedding
    embedding = tsne_reduce(expr_pam50.values, n_comp=5, r_seed=42)

    return embedding, y


def evaluation(embedding, ground_truth=None):
    import hdbscan
    from itertools import permutations
    import numpy as np
    
    
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
        jaccard_score,
    )
    
    if ground_truth is None:
        ground_truth , y = create_pam50_embeddings()
    
    embeddings = [embedding, ground_truth]
    # Perform clustering
    # Use HDBSCAN for clustering

    # Dictionary to store scores for each embedding type
    scores = {}
    labels = ['ARI', 'NMI', 'Silhouette', 'Jaccard', 'Accuracy', 'Average']
    embedding_names = ["Test", "Truth"]

    for i, embedding in enumerate(embeddings):
        cluster = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(embedding)
        
        unique_pred = np.unique(cluster)
        unique_true = np.unique(y)
        
        best_ari = -1
        best_nmi = -1
        best_jaccard = -1
        best_acc = -1
        
        if len(unique_pred) <= len(unique_true):
            for perm in permutations(unique_true, len(unique_pred)):
                mapping = dict(zip(unique_pred, perm))
                mapped_cluster = np.array([mapping[c] for c in cluster])
                current_ari = adjusted_rand_score(y, mapped_cluster)
                current_nmi = normalized_mutual_info_score(y, mapped_cluster)
                current_jaccard = jaccard_score(y, mapped_cluster, average='weighted')
                current_acc = np.mean(y == mapped_cluster)
                if current_ari > best_ari:
                    best_ari = current_ari
                if current_nmi > best_nmi:
                    best_nmi = current_nmi
                if current_jaccard > best_jaccard:
                    best_jaccard = current_jaccard
                if current_acc > best_acc:
                    best_acc = current_acc
            ari, nmi, jaccard, accuracy = best_ari, best_nmi, best_jaccard, best_acc
        else:
            ari = adjusted_rand_score(y, cluster)
            nmi = normalized_mutual_info_score(y, cluster)
            jaccard = jaccard_score(y, cluster, average='weighted')
            accuracy = np.mean(y == cluster)
        
        silhouette = silhouette_score(embedding, cluster)
        average_score = np.mean([ari, nmi, silhouette, jaccard, accuracy])
        
        scores[embedding_names[i]] = [ari, nmi, silhouette, jaccard, accuracy, average_score]

    import matplotlib.pyplot as plt

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for name, metric_values in scores.items():
        values = metric_values + metric_values[:1]
        ax.plot(angles, values, label=name)
        ax.fill(angles, values, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Clustering Metrics Radar Chart")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Save the figure to a file
    plt.savefig("clustering_metrics_radar.png", bbox_inches='tight')
    plt.show()
    
    
    
    return ari, nmi, silhouette, jaccard, average_score


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("Evaluating PAM50 Embeddings...")
    print("Creating PAM50 Embeddings...")
    embedding, y = create_pam50_embeddings()
    print("Evaluating...")
    evaluation(embedding)
    t1 = time.time()
    print('Evaluation running time is: ' + str(t1-t0) + ' s' )