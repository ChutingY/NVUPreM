# --- 0. Initial Setup and Library Loading ---
library(lattice)   # For stripplot visualization from mice package
library(MASS)      # Dependency for nnet (used by mice)
library(nnet)      # Dependency for mice
library(mice)      # Multiple Imputation by Chained Equations
library(foreign)   # To read data from other statistical systems (optional, if not needed remove)
library(ranger)    # Fast random forests (used by mlr3verse or directly for imputation)
library(mlr3verse) # Machine learning framework (if specific mlr3 functionalities are used later)
library(dplyr)     # Data manipulation (e.g., mutate, %>% )
library(ggplot2)   # Data visualization
library(forcats)   # For working with factors (part of tidyverse, if needed)
library(VIM)       # Visualization and Imputation of Missing Values

# --- 1. Load Data ---
data <- read.csv()

# --- 2. Clinical Score Calculation ---
# 2.1 Rockall Score
data <- data %>%
  mutate(
    AGE_SCORE = case_when(age < 60 ~ 0,age >= 60 & age <= 79 ~ 1,age >= 80 ~ 2,TRUE ~ 0 ),
    SHOCK_SCORE = case_when(sbp_min < 100 ~ 2,sbp_min >= 100 & heart_rate_max > 100 ~ 1,TRUE ~ 0),
    CO_MORBIDITY_SCORE = case_when(
      rowSums(across(c(renal_disease, severe_liver_disease, metastatic_solid_tumor)), na.rm = TRUE) > 0 ~ 2,
      rowSums(across(c(myocardial_infarct, congestive_heart_failure, peripheral_vascular_disease,
                       cerebrovascular_disease, dementia, chronic_pulmonary_disease, rheumatic_disease,
                       paraplegia, malignant_cancer, aids)), na.rm = TRUE) > 0 ~ 1,
      TRUE ~ 0
    )
  ) %>%
  mutate(
    across(c(AGE_SCORE, SHOCK_SCORE, CO_MORBIDITY_SCORE), ~ replace_na(., 0)), # Treat NA component scores as 0
    Rockall = AGE_SCORE + SHOCK_SCORE + CO_MORBIDITY_SCORE # Sum components for total Rockall score
  )

# 2.2 Glasgow Blatchford Score (GBS)
data <- data %>%
  mutate(
    BUN_SCORE = case_when(
      bun_max >= 70 ~ 6,
      bun_max >= 28 & bun_max < 70 ~ 4,
      bun_max >= 22.4 & bun_max < 28 ~ 3,
      bun_max >= 18.2 & bun_max < 22.4 ~ 2,
      TRUE ~ 0
    ),
    Hemoglobin_SCORE = case_when(
      hemoglobin_min < 10.0 ~ 6,
      gender == "M" & hemoglobin_min >= 10 & hemoglobin_min < 12 ~ 3,
      gender == "F" & hemoglobin_min >= 10 & hemoglobin_min < 12 ~ 1,
      gender == "M" & hemoglobin_min >= 12 & hemoglobin_min < 13 ~ 1,
      TRUE ~ 0
    ),
    SBP_SCORE = case_when(
      sbp_min < 90 ~ 3,
      sbp_min >= 90 & sbp_min <= 99 ~ 2,
      sbp_min >= 100 & sbp_min <= 109 ~ 1,
      TRUE ~ 0
    ),
    HR_SCORE = ifelse(heart_rate_max >= 100, 1, 0),
    Syncope_SCORE = ifelse(gcs_min < 15, 2, 0),
    Liver_SCORE = ifelse(severe_liver_disease >= 1, 2, 0),
    Cardiac_SCORE = ifelse(congestive_heart_failure >= 1, 2, 0)
  ) %>%
  mutate(
    across(c(BUN_SCORE, Hemoglobin_SCORE, SBP_SCORE, HR_SCORE, Syncope_SCORE, Liver_SCORE, Cardiac_SCORE), ~ replace_na(., 0)),
    GBS = BUN_SCORE + Hemoglobin_SCORE + SBP_SCORE + HR_SCORE + Syncope_SCORE + Liver_SCORE + Cardiac_SCORE
  )

# 2.3 AIMS65 Score
data <- data %>%
  mutate(
    AGE_SCORE_2 = ifelse(age > 65, 1, 0), # AIMS65 score for age >= 65
    Albumin_score = ifelse(albumin_min < 3, 1, 0),
    INR_score = ifelse(inr_max > 1.5, 1, 0),
    mental_score = ifelse(gcs_min < 15, 1, 0),
    # !!! IMPORTANT CORRECTION !!!
    # Original code incorrectly used gcs_min for SBP_score in AIMS65.
    # AIMS65 SBP component is typically SBP < 90 mmHg.
    # Correcting to use sbp_min for SBP_score.
    SBP_score_AIMS65 = ifelse(sbp_min < 90, 1, 0) 
  ) %>%
  mutate(
    across(c(AGE_SCORE_2, Albumin_score, INR_score, mental_score, SBP_score_AIMS65), ~ replace_na(., 0)),
    AIMS65 = AGE_SCORE_2 + Albumin_score + INR_score + mental_score + SBP_score_AIMS65
  )

# --- 3. Data Filtering based on Missingness ---
# 3.1 Identify and select columns with <= 40% missing values
na_ratios <- colMeans(is.na(data))
selected_columns <- names(na_ratios[na_ratios <= 0.4])
FIND <- data[, selected_columns]

# 3.2 Filter rows where at least 80% of selected columns are not empty
FIND2 <- FIND[apply(!is.na(FIND), 1, function(x) sum(x) >= ncol(FIND) * 0.8), ]

# 3.3 Explore mortality statistics for filtered data
# Calculate median time to event for deceased patients.
dead <- FIND2[FIND2$OS == 1, ]
median_value <- median(dead$Time, na.rm = TRUE)
print(paste("Median time to death (days):", median_value))

# Plot distribution of time to death
ggplot(dead, aes(Time)) +
  geom_bar(fill = "#4d97cd") +
  theme_bw() +
  xlim(-1, 200) + # Adjust x-axis limit as per data distribution
  labs(title = "Distribution of Time to Death", x = "Time (days)", y = "Count") 
ggsave(here("R", "1.1", "0. Distribution of dead.pdf"), width = 10, height = 6) # Save plot

# --- 4. Prepare Data for Imputation ---
# 4.1 Recode categorical variables to numerical ranks
FIND2 <- FIND2 %>%
  mutate(
    admission_type = case_when(admission_type == "EMER" ~ 1, "SURGERY" ~ 2, "URGENT" ~ 3, 
      admission_type == "ELECTIVE" ~ 4, "OBSERVATION" ~ 5,TRUE ~ NA_real_ ),
    insurance = case_when(insurance == "Medicare" ~ 1, "Private" ~ 2, "Medicaid" ~ 3, insurance == "Other" ~ 4,TRUE ~ NA_real_),
    marital_status = case_when(
      marital_status == "MARRIED" ~ 1, "SINGLE" ~ 2, "DIVORCED" ~ 3, 
      marital_status == "WIDOWED" ~ 4,
      TRUE ~ NA_real_
    ),
    race = case_when(
      race == "WHITE" ~ 1, "BLACK" ~ 2, "ASIAN" ~ 3, "OTHER" ~ 4,
      TRUE ~ NA_real_
    ),
    gender = case_when(
      gender == "M" ~ 1, "F" ~ 2,
      TRUE ~ NA_real_
    )
  )
# head(FIND2) # Uncomment to view converted data

# --- 5. Impute Missing Values using MICE ---
# 5.1 Perform multiple imputation using Random Forest (rf) method
mice_imputations <- mice(FIND2, 
                         method = "rf", # Random Forest imputation method
                         seed = 123, 
                         print = FALSE, # Suppress detailed output during imputation
                         m = 10,        # Number of imputed datasets
                         maxit = 5)     # Number of iterations per imputation
# 5.2 Visualize imputation quality
imputation_figure <- stripplot(mice_imputations, col = c("grey", mdc(2)), pch = c(1, 20))
# 5.3 Save imputation quality plot to PDF
pdf(here("R", "1.1", "Imputation_Quality_rf_50times.pdf"), height = 10, width = 10)
print(imputation_figure)
dev.off()
# MICE imputation methods available:
# "pmm": Predictive Mean Matching (default for numerical)
# "norm": Bayesian linear regression
# "norm.boot": Bootstrap-based linear regression
# "norm.predict": Linear regression predictive mean
# "cart": Classification And Regression Trees
# "rf": Random Forest

# 5.4 Extract one complete dataset (e.g., the first one)
MICE_PREPARE <- complete(mice_imputations, 1) # Extracts the 1st imputed dataset
rownames(MICE_PREPARE) <- rownames(FIND2) # Restore original row names

# --- 6. Save Imputed Data ---
write.csv(MICE_PREPARE)

