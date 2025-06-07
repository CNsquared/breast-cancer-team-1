#!/bin/bash

# This script sets up the environment for the project.
# It installs the required packages and sets up the virtual environment.
# Downloads any necessary data.

# Usage: ./setup.sh
# Make sure you are in the root of the project-5 directory before running this script.

YML_FILE="project5_env.yml"


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


LATENT_DIMS_BASAL_ID='1EhjERXK4LH3GWBWn3sVXD_u0Tz0Buwg4'
LATENT_DIMS_NORMAL_ID='1PD15G4p-4yzdcICmXrpUadgq1IYE5A2e'
LATENT_DIMS_LUMB_ID='1JfKM0I6AAchjT9tW9x-fCYn9o82UXeR7'
LATENT_DIMS_HER2_ID='1vV5v09Xe_fMMltzN2-YoElk5LO7y5BX1'
LATENT_DIMS_LUMA_ID='1mAqHTgcbtT5XbEsHoEHEfzktiInSI6aA'


if [ -f "latent_space_5dim_BRCA_Basal.csv" ]; then
    echo "latent_space_5dim_BRCA_Basal.csv already exists. Skipping download."
else
    echo "Downloading latent dims Basal"
    gdown "https://drive.google.com/uc?export=download&id=${LATENT_DIMS_BASAL_ID}"
fi
# download latent dims
if [ -f "latent_space_5dim_BRCA_Normal.csv" ]; then
    echo "latent_space_5dim_BRCA_Normal.csv already exists. Skipping download."
else
    echo "Downloading latent dims Normal"
    gdown "https://drive.google.com/uc?export=download&id=${LATENT_DIMS_NORMAL_ID}"
fi

if [ -f "latent_space_5dim_BRCA_LUMB.csv" ]; then
    echo "latent_space_5dim_BRCA_LUMB.csv already exists. Skipping download."
else
    echo "Downloading latent dims LumB"
    gdown "https://drive.google.com/uc?export=download&id=${LATENT_DIMS_LUMB_ID}"
fi

if [ -f "latent_space_5dim_BRCA_LUMA.csv" ]; then
    echo "latent_space_5dim_BRCA_LUMA.csv already exists. Skipping download."
else
    echo "Downloading latent dims LumA"
    gdown "https://drive.google.com/uc?export=download&id=${LATENT_DIMS_LUMA_ID}"
fi

if [ -f "latent_space_5dim_BRCA_HER2.csv" ]; then
    echo "latent_space_5dim_BRCA_HER2.csv already exists. Skipping download."
else
    echo "Downloading latent dims LumA"
    gdown "https://drive.google.com/uc?export=download&id=${LATENT_DIMS_HER2_ID}"
fi     
