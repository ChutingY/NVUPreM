# --- 1. Initial Setup and Library Loading ---
library(PerformanceAnalytics) # For correlation matrix visualization
library(scitb)              # For quick descriptive statistics and group comparisons
library(autoReg)            # For automated logistic regression table generation
library(rrtable)            # For exporting tables to Word/PowerPoint
library(dplyr)              # Data manipulation (e.g., mutate, %>% )
library(ggplot2)            # Data visualization (though not explicitly used in this snippet)
library(lattice)            # For potential visualization (dependency for mice, though mice isn't main focus here)
library(MASS)               # Dependency for nnet
library(nnet)               # Dependency for mice
library(mice)               # Multiple Imputation (though na.omit is used here)
library(foreign)            # To read data from other statistical systems (optional)
library(ranger)             # Fast random forests (optional, if mlr3verse or imputation uses it)
library(mlr3verse)          # Machine learning framework (optional, if advanced ML tasks follow)
library(forcats)            # For working with factors (optional)
library(VIM)

# --- 2. Data Loading and Initial Preprocessing ---
data <- read.csv(here("R", "1.1", "1.1 signature.csv"), header = TRUE, row.names = 1)
mydata <- na.omit(data)
mydata$status <- ifelse(mydata$status == 2, 1, 0)
str(mydata)

# --- 3. Exploratory Data Analysis ---
# 3.1 Quick Statistical Description and Correlation Visualization
chart.Correlation(mydata[, c("Rockall", "GBS", "AIMS65", "sapsii", "sofa", "charlson", "gcs", "apsiii")], 
                  histogram = TRUE, pch = 19, method = 'spearman')
# 3.2 Group Comparisons for Clinical Scores (Table 1 Generation)
selected_vars <- c("Rockall", "GBS", "AIMS65", "sapsii", "sofa", "charlson", "gcs", "apsiii")
factor_vars <- c("GBS")
strata_var <- "Composite_Endpoint"
table1_output <- scitb1(vars = selected_vars, fvars = factor_vars, strata = strata_var, data = mydata) 
print(table1_output)
write.csv(table1_output, file = here("R", "1.1", "Table1_GroupComparison.csv"), row.names = FALSE) 

# --- 4. Univariate Logistic Regression Analysis ---
logistic_model <- glm(status ~ Rockall + GBS + AIMS65 + sapsii + sofa + charlson + gcs + apsiii,
                      family = binomial(link = "logit"),data = mydata)
univariate_results <- autoReg(logistic_model, uni = TRUE, threshold = 0.05)
print(univariate_results)
result_df <- univariate_results %>% myft()
print(result_df)
table2pptx(result_df, file = here("R", "1.1", "Univariate_LogisticRegression_Results.pptx")) 
table2docx(result_df, file = here("R", "1.1", "Univariate_LogisticRegression_Results.docx")) 

