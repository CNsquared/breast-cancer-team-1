install.packages('tidyverse')
install.packages('car')
install.packages('patchwork')
install.packages('ggpubr')
install.packages('gtsummary')
install.packages('survival')
install.packages('survminer')
install.packages('umap')

library(tidyverse)
library(car)
library(patchwork)
library(ggpubr)
library(gtsummary)
library(survival)
library(survminer)
library(umap)

setwd('./project-4')

#Load Metadata
Patient_metadata <- read_tsv('./data/raw/brca_tcga_pan_can_atlas_2018_clinical_data_PAM50_subype_and_progression_free_survival.tsv')
#Load latent space
Latent_Space <- read_csv('./results/tables/latent_space_5dim_BRCA_Basal.csv')
#Load mutation data
Mutations <- read.table('./data/processed/TCGA.BRCA.mutations.qc1.txt',header = T)

#Keep patients in latent space only
Mutations <- Mutations %>% semi_join(Latent_Space)
#Keep patients with BRCA1 and BRCA2 only
Mutations_BRCA <- Mutations %>% filter(str_detect(Hugo_Symbol,'BRCA1|BRCA2')) %>% filter(mutation_type=='non-synonymous')
Mutations_BRCA <- Mutations_BRCA %>% select(patient_id) %>% mutate(BRCA=TRUE)

#Data Prep
#Rename variables
Patient_metadata <- Patient_metadata %>% rename(Age=`Diagnosis Age`)
Patient_metadata <- Patient_metadata %>% rename(patient_id=`Patient ID`,sample_id=`Sample ID`)
Patient_metadata <- Patient_metadata %>% rename(`histological_type` = `Tumor Type`)
Patient_metadata <- Patient_metadata %>% rename(ajcc_pathologic_tumor_stage=`Neoplasm Disease Stage American Joint Committee on Cancer Code`)

#Scale age to 10 years
Patient_metadata <- Patient_metadata %>% mutate(Age10=Age/10)

#Group cancer stages in I, II,III, and IV
Patient_metadata <- Patient_metadata %>% filter(!ajcc_pathologic_tumor_stage  %in% c('[Discrepancy]', '[Not Available]','Stage X','NA'))
Patient_metadata <- Patient_metadata %>% mutate(Stage = case_when(
  ajcc_pathologic_tumor_stage %in% c('STAGE I','STAGE IA', 'STAGE IB') ~ 'I',
  ajcc_pathologic_tumor_stage %in% c('STAGE II',  'STAGE IIA',  'STAGE IIB' ) ~ 'II',
  ajcc_pathologic_tumor_stage %in% c('STAGE III', 'STAGE IIIA', 'STAGE IIIB', 'STAGE IIIC') ~ 'III',
  ajcc_pathologic_tumor_stage %in% c('STAGE IV') ~ 'STAGE IV',
))

#Modify latent space sample ids
Latent_Space <- Latent_Space %>% mutate(sample_id=str_sub(sample_id,1,15))

#Joint clinical data and latent space data
Joint_data <- Patient_metadata %>% inner_join(Latent_Space) 

#Convert ethnicity into White and Non-White 
Joint_data <- Joint_data %>% mutate(Ethnicity_white=ifelse(`Race Category`=='White','White','Non-White'))
Joint_data <- Joint_data %>% mutate(Ethnicity_white=ifelse(is.na(Ethnicity_white),'Non-White',Ethnicity_white))
Joint_data <- Joint_data %>% mutate(Ethnicity_white=as.factor(Ethnicity_white))
Joint_data$Ethnicity_white = relevel(Joint_data$Ethnicity_white,ref='White')

#Histological type
Joint_data <- Joint_data %>% filter(histological_type %in% c("Infiltrating Lobular Carcinoma","Infiltrating Ductal Carcinoma"))
Joint_data$histological_type <- as.factor(Joint_data$histological_type)

#Join BRCA status
Joint_data <- Joint_data %>% left_join(Mutations_BRCA)
Joint_data <- Joint_data %>% mutate(BRCA=ifelse(is.na(BRCA),F,T))

#Rename survival variables
Joint_data_Survival <- Joint_data %>% mutate(DSS.time=`Disease Free (Months)`)
#Censor patients after 100 months of follow up
Joint_data_Survival <- Joint_data_Survival %>% mutate(DSS.time=ifelse(DSS.time>100,100,DSS.time))
Joint_data_Survival <- Joint_data_Survival %>% mutate(DSS=`Disease Free Status`)
Joint_data_Survival <- Joint_data_Survival %>% mutate(DSS=ifelse(str_starts(DSS,'0'),0,1))


#Remove patients  with missing follow up data
Joint_data_Survival <- Joint_data_Survival %>% filter(!is.na(DSS) & !is.na(DSS.time))

#Generate Kaplan-Meier curve
km.plot <- survfit(Surv(DSS.time, DSS) ~ histological_type, data = Joint_data_Survival)
ggsurvplot(km.plot,
           risk.table = TRUE,
           pval = T,
           conf.int = T,
           title = "Kaplan-Meier curves")

#Scale latent space data to per 10 points
Joint_data_Survival <- Joint_data_Survival %>% mutate(across(starts_with('latent'), ~ .x/10, .names = "{.col}_10"))

#Fit full cox model
res.cox <- coxph(Surv(DSS.time, DSS) ~ Age + BRCA + Ethnicity_white + Stage   + latent_0_10 + latent_1_10  + latent_2_10 + latent_3_10 +latent_4_10  , data = Joint_data_Survival %>% mutate(latent_4_10=latent_4/10))
tbl_regression(res.cox,exponentiate = TRUE, label = list(Ethnicity_white = "Ethnicity") )

#Perform backward selection
res.cox.backward <- step(res.cox, direction = "backward")
tbl_regression(res.cox.backward, exponentiate = TRUE, label = list(Stage = 'Cancer stage',Ethnicity_white = "Ethnicity", latent_2_10= 'Latent space 2, per 10 units', latent_4_10= 'Latent space 4, per 10 units'))

#Perform LRT tests
anova(res.cox.backward, res.cox, test = "LRT")
