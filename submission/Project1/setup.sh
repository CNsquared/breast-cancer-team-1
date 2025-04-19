#!/bin/bash

# This script sets up the environment for the project.
# It creates a virtual environment, installs dependencies, and sets up the database.
# Usage: ./setup.sh
# Make sure you are in the root of the project directory before running this script.

conda env create -f environment.yml

mkdir -p data
mkdir -p output
mkdir -p output/figures
mkdir -p output/tables

echo "get supplementary table"
SUPL_TABLE_LINK='https://docs.google.com/spreadsheets/d/1M7PHOeb4AxAr0qaV6MSOVs_rSn391Eg-/edit?usp=sharing&ouid=106435358685362048028&rtpof=true&sd=true'
wget $SUPL_TABLE_LINK -O "data/Supplementary Tables 1-4.xls"

echo "get pam50 gene list"
PAM50_GENESET_LINK='https://drive.google.com/file/d/1ggB-Ds39xU4POwOx020aPiLN3Y3nqQ-v/view?usp=sharing'
wget $PAM50_GENESET_LINK -O "data/pam50.tsv"

echo "get TCGA data"
TCGA_LINK='https://www.dropbox.com/scl/fo/7d37xqur5vlb8jni61b0t/AIULq2j8qwiKujKLUdwZ1fA/Team_1_BRCA?rlkey=pfw7xmb7slnz7d398gzfzpju7&subfolder_nav_tracking=1&st=4c0wzon9&dl=0'
wget $TCGA_LINK -O "data/TCGA_BRCA.zip"

echo "unzip TCGA data"
unzip data/TCGA_BRCA.zip
rm data/TCGA_BRCA.zip

echo "complete"