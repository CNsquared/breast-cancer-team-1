#Install libraries

install.packages('devtools')
install.packages('tidyverse')

library(devtools)
install_github("im3sanger/dndscv")
library(tidyverse)
library(dndscv)

#Load mutations 
Mutations <- read.table('./project-2/data/processed/TCGA.BRCA.mutations.qc1.txt',header = T)

#Select and rename them for dNdScv
Mutations_sub <- Mutations %>% select(patient_id,Chromosome,Start_Position,Reference_Allele,Allele)
colnames(Mutations_sub) <- c('sampleID','chr','pos','ref','mut')

#Run dNdScv
dndsout = dndscv(Mutations_sub)

sel_cv = dndsout$sel_cv

#Print top 10 genes in statistical significance
print('Top 10 genes in statistical significance')
print(head(sel_cv,10), digits = 3)

#Export dNdScv results as csv file 
print('Export dNdScv results as csv file')
write.csv(sel_cv,'./project-2/results/tables/dNdScv_output.csv')