#!/bin/bash

# This script sets up the environment for the project.
# It installs the required packages and sets up the virtual environment.
# Downloads any necessary data.

# Usage: ./setup.sh
# Make sure you are in the root of the project-5 directory before running this script.

YML_FILE="project5_env.yml"

# Extract environment name from yml file
ENV_NAME=$(grep '^name:' "$YML_FILE" | awk '{print $2}')

echo "Setting up conda environment '$ENV_NAME' from '$YML_FILE'..."

# Flag to optionally skip env update
SKIP_CONDA_UPDATE=false
for arg in "$@"; do
    if [[ "$arg" == "--skip-conda-update" ]]; then
        SKIP_CONDA_UPDATE=true
    fi
done

# Check if env exists
if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda env create -f "$YML_FILE"
fi

# Update if not skipped
if [[ "$SKIP_CONDA_UPDATE" == false ]]; then
    echo "Updating conda environment '$ENV_NAME'..."
    conda env update -f "$YML_FILE"
else
    echo "Skipping environment update (per --skip-conda-update)."
fi

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"



# Download TCGA BRCA expression data from Dropbox

mkdir -p data/raw
cd data/raw

TCGA_LINK='https://www.dropbox.com/scl/fo/7d37xqur5vlb8jni61b0t/AIULq2j8qwiKujKLUdwZ1fA/Team_1_BRCA?rlkey=pfw7xmb7slnz7d398gzfzpju7&subfolder_nav_tracking=1&st=4c0wzon9&dl=0'
if [ -f "TCGA.BRCA.expression.txt" ]; then
    echo "TCGA.BRCA.expression.txt already exists. Skipping download."
else
    echo "Downloading TCGA data from Dropbox"
    wget "$TCGA_LINK" -O "TCGA_BRCA.zip"
    unzip TCGA_BRCA.zip
    rm TCGA_BRCA.zip
fi

# Download GENCODE transcripts
TRANSCRIPTS_ID='1ioyMfGNvF7FC9734FX1pDq9CkUo9Ogva'
if [ -f "gencode.v23lift37.pc_transcripts.fa" ]; then
    echo "gencode.v23lift37.pc_transcripts.fa already exists. Skipping download."
else
    echo "Downloading gencode.v23lift37.pc_transcripts.fa.gz"
    gdown "https://drive.google.com/uc?id=${TRANSCRIPTS_ID}"
    gunzip gencode.v23lift37.pc_transcripts.fa.gz
fi

# download pam50 gene list
PAM50_GENESET_ID='1ggB-Ds39xU4POwOx020aPiLN3Y3nqQ-v'
if [ -f "pam50.tsv" ]; then
    echo "pam50.tsv already exists. Skipping download."
else
    echo "Downloading pam50 gene list"
    gdown "https://drive.google.com/uc?id=${PAM50_GENESET_ID}"
fi

# download survival data table
SURVIVAL_DATA_ID='1gdDy5GgMYg3Ir_iV-OEjFMzYy2usd9OJ'
if [ -f "brca_tcga_pan_can_atlas_2018_clinical_data_PAM50_subype_and_progression_free_survival.tsv" ]; then
    echo "brca_tcga_pan_can_atlas_2018_clinical_data_PAM50_subype_and_progression_free_survival.tsv already exists. Skipping download."
else
    echo "Downloading brca_tcga_pan_can_atlas_2018_clinical_data_PAM50_subype_and_progression_free_survival.tsv"
    gdown "https://drive.google.com/uc?id=${SURVIVAL_DATA_ID}"
fi

# download dndscv genes
DNDS_GENES_ID='1a_Zd6pZrCpGowaM9bYDmtPy6uSvPx8ik'
if [ -f "dndscv.csv" ]; then
    echo "dndscv.csv already exists. Skipping download."
else
    echo "Downloading dndscv.csv"
    gdown "https://drive.google.com/uc?id=${DNDS_GENES_ID}"
fi

echo "Setup complete."
