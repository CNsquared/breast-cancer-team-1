run './setup.sh' to get file directory set up

run 'conda env create -f project1_env.yml' to create conda environment project1_env

use 'project1_env' as python environment when running jupyter notebook

dim_reduce_cluster.ipynb:
This notebook describes how samples are filtered out, how 50 genes from PAM50 are selected, the "ground truth" PAM50 labels, grid search for best embedding and clustering, and the best embedding and clustering. As well as colored by metadata.
Figure 1, 2, 3 left side panels

ZAID50_grid_search_found_UMAP.ipynb:
Samples are filtered the same way, 50 genes from Zaid50 are selected, grid search performed the same way and the best embedding and clustering result is shown. As well as colored by metadata.
Figure 1, 2, 3 right panels

check_zaid50.ipynb:
This checks the Canonical Correlation Analysis and relationship between PAM50 and Zaid50. Related to "Discussion and lesson learned" section, top right of page 2.

If any line stops running, please refer to whether the path pointing to the files like expression text file is correct.
