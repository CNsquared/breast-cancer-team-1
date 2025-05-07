import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

def build_mutation_matrix(df_mut: pd.DataFrame) -> np.ndarray:
    """Builds a mutation matrix from mutation DataFrame."""
    # TODO: placehodler
    return np.zeros((len(df_mut), len(df_mut)))

def normalize_matrix(X: np.ndarray, method: str = 'GMM', max_iter: int = 100, random_state: int = 42) -> np.ndarray:
    """normalize matrix by method
    
    should be able to use the following methods (from sigprofiler extractor):
    - GMM: Gaussian mixture model (default). GMM normalization encompasses a two-step process. 
        The first step derives the normalization cutoff value in a data-driven manner using a Gaussian mixture 
        model (GMM). The second step normalizes the appropriate columns using the derived cutoff value. The first
        step uses a GMM to separate the samples into two groups based on their total number of mutations; the 
        total number of mutations in a sample reflects the sum of a column in the matrix. The group with larger
        number of samples is subsequently selected, and the same process is applied iteratively until it converges.
        Convergence is achieved when the mean of the two groups is separated by no more than four standard 
        deviations of the larger group. A cutoff value is derived as the average value plus two standard deviations
        from the total number of somatic mutations in the last large group. If the derived cutoff value is below 100
        times the number of mutational channels, the cutoff value is adjusted to 100 times the number of mutational
        channels. For each column where the sum exceeds the derived cutoff value, each cell in the column is multiplied
        by the cutoff value and subsequently divided by the original column sum. Note that 100X normalization is 
        performed if the means of the first two groups are not separated by at least four standard deviations. 
    - 100X: the sum of each column in the matrix is derived. For each column where the sum exceeds 100 times
        the number of mutational channels (i.e., 100 times the number of rows in the matrix), each cell in 
        the column is multiplied by the 100 times the number of mutational channels and subsequently divided
        by the original column sum. This normalization ensures that no sample has a total number of mutations
        above 100 times the number of mutational channels
    - log2: In log2 normalization, the sum of each column in the matrix is derived and logarithm 
        with base 2 is calculated for each of these sums. Each cell in a column of the matrix is 
        multiplied by the log2 of the column-sum and subsequently divided by the original column sum
    - None: no normalization.

    TODO: In all cases, fractional values after normalization are used as input for the factorization, and columns
        with a sum of zero, reflecting genomes without any somatic mutations, are ignored to avoid division by zero
    """
    if method is None or method == 'None':
        return X
    elif method == 'GMM':
        # first step: fit GMM to the data to get normalization cutoff value
        X = X.astype(float)
        col_sums = np.sum(X, axis=0)
        n_channels = X.shape[0]

        first_pass = True
        current_group = col_sums.copy()
        for _ in range(max_iter):
            # fit GMM to current group
            gmm = GaussianMixture(n_components=2, random_state=random_state)
            gmm.fit(current_group.reshape(-1, 1))
            # get group labels
            labels = gmm.predict(current_group.reshape(-1, 1))
            group0 = current_group[labels == 0]
            group1 = current_group[labels == 1]

            # Identify larger group
            if len(group0) >= len(group1):
                large_group = group0
            else:
                large_group = group1
            
            # check that large group is not empty
            if len(large_group) == 0:
                break

            # check for convergence, where mean of two groups separated by at most 4 std dev
            std_large_group = np.std(large_group)
            mean_diff = abs(group0.mean() - group1.mean())
            if mean_diff <= 4 * std_large_group or std_large_group == 0:
                if first_pass:
                    # do 100X normalization if the means of the first two groups are not separated by at least 4 std dev
                    col_sums = np.sum(X, axis=0)
                    num_channels = X.shape[0]
                    threshold = 100 * num_channels
                    normalized_matrix = np.where(col_sums > threshold, (X * threshold) / col_sums, X)
                    return normalized_matrix
                else:
                    break
            # update current group to be the large group
            current_group = large_group
            first_pass = False

        if len(current_group) == 0:
            raise ValueError("GMM normalization failed to converge. No large group found.")
        
        # cutoff value is avg value plus two std devations from total number of somatics mutations in last large group
        cutoff_value = np.mean(large_group) + 2 * np.std(large_group)

        # if derived cutoff value is below 100 times number of mutation channels, cutoff value adjusted
        if cutoff_value < 100 * n_channels:
            cutoff_value = 100 * n_channels

        # adjust each column where sum exceeds cutoff value
        normalized_matrix = X.copy()
        for j in range(X.shape[1]):
            if col_sums[j] > cutoff_value:
                normalized_matrix[:, j] *= cutoff_value / col_sums[j]

        return normalized_matrix

    elif method == '100X':
        col_sums = np.sum(X, axis=0)
        num_channels = X.shape[0]
        threshold = 100 * num_channels
        normalized_matrix = X.copy()
        for j in range(X.shape[1]):
            if col_sums[j] > threshold:
                normalized_matrix[:, j] *= threshold / col_sums[j]
        return normalized_matrix
    elif method == 'log2':
        col_sums = np.sum(X, axis=0)
        log_col_sums = np.log2(col_sums)
        normalized_matrix = (X * log_col_sums) / col_sums
        return normalized_matrix
    else:
        raise ValueError(f"Unknown normalization method: {method}. Please use 'None', 'GMM', '100X', or 'log2'")