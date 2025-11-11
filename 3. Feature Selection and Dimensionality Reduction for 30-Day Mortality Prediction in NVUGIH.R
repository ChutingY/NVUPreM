# --- 1. Environment Setup and Library Loading ---
library(here) 
library(DataExplorer)   # For data profiling reports and visualizations
library(dplyr)          # Data manipulation
library(ggplot2)        # Data visualization
library(tableone)       # For Table 1 generation and group comparisons
library(autoReg)        # Automated logistic regression table generation
library(rrtable)        # Table export to Word/PowerPoint
library(glmnet)         # Lasso regression
library(car)            # For VIF calculation (multicollinearity)
library(FactoMineR)     # Principal Component Analysis (PCA)
library(factoextra)     # Visualization for PCA
library(lattice)        # Dependency for mice (retained for consistency, though mice not primary here)
library(MASS)           # Dependency for nnet
library(nnet)           # Dependency for mice
library(mice)           # Multiple Imputation (not used in this specific snippet but in project)
library(foreign)        # Optional: For reading foreign data formats
library(ranger)         # Optional: Fast random forests
library(mlr3verse)      # Optional: Machine learning framework
library(forcats)        # Optional: For factor manipulation
library(VIM)            # Optional: Visualization of Missing Values


# --- 2. Data Loading and Initial Processing ---
data <- read.csv(here("R", "1.1", "2.0 model_datafram.csv"), header = TRUE, row.names = 1)


# --- 3. Data Transformation and Exploration ---
# Extract target variable and select independent variables
OS_30day <- data$OS_30day
normalized_columns <- data[, c(7:89)] # Assuming these are the features
data_scale <- as.data.frame(cbind(OS_30day, normalized_columns))

# Generate data exploration report
create_report(data_scale, output_file = here("R", "1.1", "1.2 Data_Frequency_Distribution.html"),y = "OS_30day")

DataExplorer::plot_bar(data_scale, ggtheme = theme_bw())
DataExplorer::plot_histogram(data_scale, ggtheme = theme_bw())

# Log transformation for skewed continuous variables
data_log <- data_scale %>%
  mutate(across(c(abs_basophils_max, abs_basophils_min, abs_eosinophils_max, abs_eosinophils_min,
                  abs_lymphocytes_max, abs_lymphocytes_min, abs_monocytes_max, abs_monocytes_min,
                  abs_neutrophils_max, abs_neutrophils_min, alp_max, alp_min, alt_max, alt_min,
                  aniongap_max, aniongap_min, ast_max, ast_min, bilirubin_total_max, bilirubin_total_min,
                  bun_max, bun_min, creatinine_max, creatinine_min, dbp_max, dbp_min, glucose_max,
                  glucose_min, heart_rate_max, heart_rate_min, hematocrit_max, hematocrit_min,
                  hemoglobin_max, hemoglobin_min, inr_max, inr_min, mbp_max, mbp_min, platelets_max,
                  platelets_min, pt_max, pt_min, ptt_max, ptt_min, resp_rate_max, resp_rate_min,
                  sbp_max, sbp_min, spo2_max, spo2_min, temperature_max, temperature_min, wbc_max,
                  wbc_min), 
                ~ log(. + 1))) # Add 1 to handle zeros during log transformation

DataExplorer::plot_histogram(data_log, ncol = 3)
DataExplorer::plot_boxplot(data_log, by = 'OS_30day')

# Generate updated data exploration report after log transformation
create_report(data_log, 
              output_file = here("R", "1.1", "1.2 Data_Frequency_Distribution_Log_Transformed.html"),
              y = "OS_30day")


# --- 4. Feature Selection ---
# 4.1 Univariate Analysis: Wilcoxon Rank-Sum Test (or t-test/chi-square as appropriate)
# Define categorical variables
var_class <- c(
  "gender", "admission_type", "insurance", "marital_status", "race",
  "gcs_motor", "gcs_verbal", "gcs_eyes", "gcs_unable", "myocardial_infarct",
  "congestive_heart_failure", "peripheral_vascular_disease", "cerebrovascular_disease",
  "dementia", "chronic_pulmonary_disease", "rheumatic_disease", "peptic_ulcer_disease",
  "mild_liver_disease", "diabetes_without_cc", "diabetes_with_cc", "paraplegia",
  "renal_disease", "malignant_cancer", "severe_liver_disease", "metastatic_solid_tumor",
  "aids"
)
# Convert specified columns to factors in the original 'data' object for CreateTableOne
data[var_class] <- lapply(data[var_class], function(x) factor(x, ordered = FALSE)) # ordered=FALSE for typical chi-square/Fisher
# Create Table 1 for group comparisons
table_one_result <- CreateTableOne(vars = colnames(data)[c(7:89)], data = data,factorVars = var_class, strata = 'OS_30day', addOverall = FALSE)
printed_table <- print(table_one_result, showAllLevels = TRUE, test = TRUE) # test=TRUE to include p-values
write.csv(printed_table, here("R", "1.1", "3.1 OS_30day", "1.1 Univariate_RankSum_Test.csv"))  


# 4.2 Univariate Logistic Regression
# Fit logistic regression model with all predictors
full_logistic_model <- glm(OS_30day ~ gender + age + admission_type + insurance + marital_status + race + weight + heart_rate_min + heart_rate_max + sbp_min + sbp_max + dbp_min + dbp_max + mbp_min + mbp_max + resp_rate_min + resp_rate_max + temperature_min + temperature_max + spo2_min + spo2_max + glucose_min + glucose_max + urineoutput + hemoglobin_min + hemoglobin_max + hematocrit_min + hematocrit_max + platelets_min + platelets_max + wbc_min + wbc_max + aniongap_min + aniongap_max + bun_min + bun_max + creatinine_min + creatinine_max + abs_basophils_min + abs_basophils_max + abs_eosinophils_min + abs_eosinophils_max + abs_lymphocytes_min + abs_lymphocytes_max + abs_monocytes_min + abs_monocytes_max + abs_neutrophils_min + abs_neutrophils_max + inr_min + inr_max + pt_min + pt_max + ptt_min + ptt_max + alt_min + alt_max + alp_min + alp_max + ast_min + ast_max + bilirubin_total_min + bilirubin_total_max + gcs_motor + gcs_verbal + gcs_eyes + gcs_unable + myocardial_infarct + congestive_heart_failure + peripheral_vascular_disease + cerebrovascular_disease + dementia + chronic_pulmonary_disease + rheumatic_disease + peptic_ulcer_disease + mild_liver_disease + diabetes_without_cc + diabetes_with_cc + paraplegia + renal_disease + malignant_cancer + severe_liver_disease + metastatic_solid_tumor + aids,
                           family = binomial(link = "logit"),data = data)

# Generate univariate logistic regression results table
logistic_results_uni <- autoReg(full_logistic_model, uni = TRUE, threshold = 0.05) %>% myft()
table2pptx(logistic_results_uni, file = here("R", "1.1", "3.1 OS_30day", "1.2 Univariate_LogisticRegression.pptx"), title = "Univariate Logistic Regression Analysis")
table2docx(logistic_results_uni, file = here("R", "1.1", "3.1 OS_30day", "1.2 Univariate_LogisticRegression.docx"), title = "Univariate Logistic Regression Analysis")


# 4.3 Multicollinearity Check (VIF and Condition Number)
data_log_numeric_for_vif <- data_log %>%  mutate(across(var_class, as.numeric)) 

# Fit linear model for VIF calculation. OS_30day should be numeric here for lm.
lm_model_for_vif <- lm(as.numeric(OS_30day) ~ ., data = data_log_numeric_for_vif[,-1]) # Exclude OS_30day from predictors
vif_values <- as.data.frame(vif(lm_model_for_vif))
write.csv(vif_values, here("R", "1.1", "3.1 OS_30day", "VIF_Scores_PreLasso.csv"))

# Select predictor variables for correlation matrix and kappa.
predictors_for_kappa <- data_log_numeric_for_vif[, -1] # Exclude OS_30day
correlation_matrix <- cor(predictors_for_kappa)
condition_number <- kappa(correlation_matrix, exact = TRUE)$kappa # Extract kappa value

# 4.4 Lasso Regression for Feature Selection

# Prepare data for glmnet. Factors need to be converted to dummy variables.
x_lasso <- model.matrix(OS_30day ~ ., data = data_log)[,-1] # Create design matrix, remove intercept
y_lasso <- as.factor(ifelse(data_log$OS_30day == "0", 0, 1)) 

# Fit Lasso model
set.seed(123) # For reproducibility
fit_lasso <- glmnet(x_lasso, y_lasso, family = "binomial", alpha = 1) 

pdf(here("R", "1.1", "3.1 OS_30day", "1.3 Lasso_Regression_Paths.pdf"), width = 8, height = 8)
plot(fit_lasso, xvar = "dev", label = TRUE)
dev.off()

# Perform 10-fold cross-validation for Lasso
set.seed(123) # For reproducibility
cvfit_lasso <- cv.glmnet(x_lasso, y_lasso, nfolds = 10, family = "binomial", type.measure = "class")

pdf(here("R", "1.1", "3.1 OS_30day", "1.3 CV_Lasso_Regression_Plot.pdf"))
plot(cvfit_lasso)
dev.off()

# Extract coefficients at lambda.min
lasso_coefs <- coef(cvfit_lasso, s = "lambda.min")
lasso_features <- rownames(lasso_coefs)[which(lasso_coefs != 0)]
lasso_features <- lasso_features[-1] # Remove intercept
write.csv(data.frame(Feature = lasso_features), here("R", "1.1", "3.1 OS_30day", "1.3 Selected_Features_Lasso.csv"), row.names = FALSE)

# Re-check multicollinearity for Lasso-selected features
# Re-construct data with only Lasso features for numeric conversion
data_lasso_selected <- data_log[, c("OS_30day", lasso_features)]
data_lasso_numeric <- data_lasso_selected %>%
  mutate(across(intersect(lasso_features, var_class), as.numeric)) # Only convert selected categorical features

predictors_lasso_numeric <- data_lasso_numeric[, -1] 
correlation_matrix_lasso <- cor(predictors_lasso_numeric)
condition_number_lasso <- kappa(correlation_matrix_lasso, exact = TRUE)$kappa 

# Prepare data for Machine Learning models (Lasso-selected features)
data_ready_for_ml <- data_lasso_selected
write.csv(data_ready_for_ml, here("R", "1.1", "3.1 OS_30day", "1.3 Lasso_Ready_for_ML.csv"), row.names = FALSE)
