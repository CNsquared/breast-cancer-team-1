SBS-96 catalogue is created by SigProfilerMatrixGenerator (v1.2.30)
```
# Filter for PASS SNVs
head -n1 TCGA.BRCA.mutations.txt > TCGA.BRCA.mutations_PASS.txt
grep "PASS" TCGA.BRCA.mutations.txt >> TCGA.BRCA.mutations_PASS.txt # 789 unique samples


# Format into VCF
awk '
BEGIN {
  FS = "\t";  # Input field separator is tab
  OFS = "\t"; # Output field separator is tab
  print "#CHROM\tPOS\tID\tREF\tALT\tHugo_Symbol\tConsequence"; # Print the header line
}
NR > 1 { # Skip the header line of the input file
  print $6, $7, $1, $12, $14, $2, $30; # Print the specified columns in the desired order
}
' TCGA.BRCA.mutations_PASS.txt > TCGA.BRCA.mutations_PASS.vcf


# Split into individual sample VCFs
awk '
  BEGIN { FS = OFS = "\t" }
  /^#/ { header = header $0 "\n"; next }
  {
    out = $3 ".vcf"
    if (!(out in seen)) {
      printf "%s", header > out
      seen[out] = 1
    }
    print >> out
  }
' TCGA.BRCA.mutations_PASS.vcf # 789 unique samples

# Run SigProfilerMatrixGenerator with no downsampling
rm TCGA.BRCA.mutations_PASS.txt TCGA.BRCA.mutations_PASS.vcf

conda activate SigProfilerAssignment

SigProfilerMatrixGenerator matrix_generator "BRCA" "GRCh37" "/Users/zichenjiang/Downloads/BENG 285 projects SP25/breast-cancer-team-1/project-3/data/processed" --plot
```

# To run SigProfilerExtractor

1. **Create and activate a virtual environment** using the following commands:

    ```bash
    python -m venv sigprof
    source sigprof/bin/activate
    ```

2. **Install SigProfilerExtractor and its dependencies** inside the virtual environment:

    ```bash
    pip install -r requirements.txt
    ```


