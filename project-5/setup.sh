#!/bin/bash

# This script sets up the environment for the project.
# It installs the required packages and sets up the virtual environment.
# Downloads any necessary data.

# Usage: ./setup.sh
# Make sure you are in the root of the project-3 directory before running this script.


# Download TCGA BRCA expression data from Dropbox

mkdir data
mkdir data/raw
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
