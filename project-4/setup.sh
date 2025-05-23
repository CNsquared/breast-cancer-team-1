#!/bin/bash

# This script sets up the environment for the project.
# It installs the required packages and sets up the virtual environment.
# Downloads any necessary data.

# Usage: ./setup.sh
# Make sure you are in the root of the project-3 directory before running this script.

conda env create -f project4_env.yml
conda env update -f project4_env.yml

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project4_env

cd data/raw

# Download TCGA BRCA expression data from Dropbox
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

# download supplementary tables
SUPL_TABLE_ID='1M7PHOeb4AxAr0qaV6MSOVs_rSn391Eg-'
if [ -f "Supplementary Tables 1-4.xls" ]; then
    echo "Supplementary Tables 1-4.xls already exists. Skipping download."
else
    echo "Downloading Supplementary Tables 1-4.xls"
    gdown "https://drive.google.com/uc?id=${SUPL_TABLE_ID}"
fi

echo "Setup complete."
