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
# Visualization of how samples and mutation counts are affected by our filtering criteria
project-3/notebooks/personal/cardiff/preprocess.ipynb

# To run our NMF
` python main.py ` \
By changing the parameter sets in the beginning of main.py, we repeated this many times \
This saves each NMF run to project-3/data/processed/*joblib and S centroid matrix or signature matrix (and A matrices if you use clustering) as .txt to project-3/data/nmf_runs \ 
Stability from K-means clustering of the S matrices and the A matrices can be seen in the std out from running the main.py

# To visualize our parameter set sweep
project-3/notebooks/makefigs/02_nmf_parameter_sweep.ipynb

# To visualize why k=4 is chosen
project-3/notebooks/personal/cardiff/visualize_signatures_and_choose_k.ipynb

# To visualize our 4 de novo signatures and their cosine similarities with the SigProfiler result
project-3/notebooks/personal/cardiff/visualize_signatures_and_choose_k.ipynb\

# Decomposition of our 4 de novo signatures to COSMIC reference
project-3/notebooks/personal/ari/Project3_gpu_extraction_and_decomposition.ipynb

# Comparison with SigProfiler
## To run SigProfilerExtractor

1. **Create and activate a virtual environment** using the following commands:

    ```bash
    python -m venv sigprof
    source sigprof/bin/activate
    ```

2. **Install SigProfilerExtractor and its dependencies** inside the virtual environment:

    ```bash
    pip install -r requirements.txt
    ```
    
For more information about SigProfilerExtractor : https://github.com/AlexandrovLab/SigProfilerExtractor

## To estimate exome signatures
```
$ python

>> from SigProfilerExtractor import estimate_best_solution as ebs
>> ebs.estimate_solution(base_csvfile="All_solutions_stat.csv", 
          All_solution="All_Solutions", 
          genomes="Samples.txt", 
          output="results", 
          title="Selection_Plot",
          stability=0.8, 
          min_stability=0.2, 
          combined_stability=1.0,
          allow_stability_drop=False,
          exome=True) 
```

## To run SigProfilerAssignment
```
$ python
>> from SigProfilerAssignment import Analyzer as Analyze
>> parent="extraction_SBS96"
>> samples=parent+"/SBS96/Samples.txt"
>> output=parent+"/SBS96/Suggested_Solution/"
>> denovo=parent+"/SBS96/Suggested_Solution/SBS96_De-Novo_Solution/Signatures/SBS96_De-Novo_Signatures.txt"
Analyze.decompose_fit(samples, output, signatures=denovo, genome_build="GRCh37", exome=True, cosmic_version="3.4")
```
