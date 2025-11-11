# --- 1. Environment Setup and Library Loading ---
library(here) 
library(dplyr)
library(ggplot2)
library(tidyr)        # For data reshaping (gather)
library(caret)        # For ML model training, evaluation, and data splitting
library(caretEnsemble) # For combining caret models (stacking)
library(pROC)         # For ROC curve and AUC calculation with confidence intervals
library(ROCR)         # For general performance curves (used in plotting ROC)
library(data.table)   # For efficient data frame operations (rbindlist)
library(cutoff)       # For optimal cutoff point determination
library(DALEX)        # For model interpretability (SHAP values)
library(forcats)      # For factor manipulation (used in plotting)
library(lattice)      # Dependency for mice (retained for consistency)
library(MASS)         # Dependency for nnet (retained for consistency)
library(nnet)         # Dependency for mice (retained for consistency)
library(mice)         # Multiple Imputation (not used in this snippet but in project)
library(foreign)      # Optional: For reading foreign data formats
library(ranger)       # Optional: Fast random forests
library(mlr3verse)    # Optional: Machine learning framework
library(VIM)          # Optional: Visualization of Missing Values
library(rms)          # For nomogram, calibration (ddist, lrm)
library(regplot)      # For nomogram visualization
library(rmda)         # For Decision Curve Analysis (DCA)
library(survival)     # Dependency for rms

# --- 2. Data Loading and Preparation ---
data <- read.csv(here("R", "1.1", "3.1 OS_30day", "1.3 Lasso后ready for ML.csv"), header = TRUE, row.names = 1)

# --- 3. Data Splitting ---
data$OS_30day <- as.factor(data$OS_30day)
levels(data$OS_30day) <- c("X0", "X1")
index <- createDataPartition(data$OS_30day, p = 0.8, list = FALSE) 
train_data <- data[index, ]
test_data <- data[-index, ]

# Display class balance
table(train_data$OS_30day)
table(test_data$OS_30day)


# --- 4. Model Training with caretList ---
set.seed(456) # For reproducible cross-validation folds
control <- trainControl(method = "cv",
                        number = 10, 
                        savePredictions = "final",
                        classProbs = TRUE,
                        index = createResample(train_data$OS_30day, 10), # Use createResample for stratified resampling indices
                        sampling = "up", # Handles imbalanced classes by up-sampling the minority class
                        summaryFunction = twoClassSummary)

# List of machine learning methods to train
model_methods <- c(
  "vglmContRatio", "vglmCumulative", "bayesglm", "glmboost", "glm", "glmStepAIC", # GLM variants
  "xgbLinear", "xgbTree", # XGBoost
  "glmnet", # Regularized regression
  "pda", "pda2", # Probabilistic Discriminant Analysis
  "rf", "RRF", # Random Forest variants
  "C5.0", "C5.0Tree", # Decision Trees (C5.0)
  "kernelpls", "pls", "simpls", "widekernelpls", "spls", # Partial Least Squares
  "nnet", "pcaNNet", # Neural Networks
  "rpart", "rpart1SE", "rpart2", # Recursive Partitioning
  "ctree", "ctree2", # Conditional Inference Trees
  "svmLinear", "svmRadial", "svmRadialCost", "svmRadialSigma", # Support Vector Machines
  "LogitBoost", "blackboost", "cforest", "fda", "kknn", "knn", "lda", "lda2", "earth", "gcvEarth", # Other ML algorithms
  "multinom", "gbm", "ada" # Additional common methods
)
# Filter for methods supported by caretList
supported_methods <- intersect(model_methods, names(modelLookup()))

set.seed(789) # For reproducible model training
models <- caretList(OS_30day ~ ., 
                    data = train_data,  
                    trControl = control,  
                    metric = "ROC",
                    methodList = supported_methods)

# Compare models using resampling results
resamps <- resamples(models)
summary(resamps)
bwplot(resamps)

# Plot average ROC values for all models
roc_columns <- colnames(resamps$values)[grep("ROC", colnames(resamps$values))]
average_roc_values <- resamps$values[, roc_columns]

long_format_roc <- gather(average_roc_values, key = "model_name", value = "roc_value")
long_format_roc$model_name <- gsub("~ROC", "", long_format_roc$model_name)
long_format_roc$model_name <- as.factor(long_format_roc$model_name)

average_values_roc <- aggregate(roc_value ~ model_name, data = long_format_roc, FUN = mean)
sorted_model_names <- average_values_roc$model_name[order(average_values_roc$roc_value)]
long_format_roc$model_name <- factor(long_format_roc$model_name, levels = sorted_model_names)

ggplot(long_format_roc, aes(x = model_name, y = roc_value)) +
  geom_boxplot(aes(fill = model_name, outlier.color = NA)) +
  labs(title = "Boxplot of Average ROC Across Models (10-Fold CV)",
       y = "Receiver Operating Characteristic", x = "") +
  coord_flip() + theme_bw() +
  scale_fill_manual(values = colorRampPalette(c("#2A77AC", "#0071C2", "#78AB31", "#EDB11A", "#D75615", "#D55535", "#7E318A"))(length(unique(long_format_roc$model_name)))) +
  theme(plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), units = "cm"),
        axis.line = element_line(color = "black", size = 0.2),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(size = 0.2, color = "#e5e5e5"),
        panel.background = element_blank(),
        axis.text.y = element_text(color = "black", size = 11, face = "bold"),
        axis.text.x = element_text(color = "black", size = 10, vjust = 0.5, hjust = 1, angle = 90),
        axis.line.x.top = element_line(color = "black"), 
        axis.text.x.top = element_blank(),
        axis.ticks.y.right = element_blank(),
        axis.text.y.right = element_blank(),
        axis.ticks.x.top = element_blank(),
        panel.spacing.x = unit(0, "cm"),
        legend.position = "none",
        panel.spacing = unit(0, "lines")) +
  guides(x.sec = "axis", y.sec = "axis")
ggsave(filename = here("R", "1.1", "3.1 OS_30day", "3.1 Average_AUC_10Fold_CV.pdf"), width = 5, height = 7, device = "pdf", family = "Times")


# --- 5. Best Model Analysis (Variable Importance & SHAP) ---

# Identify the best performing model (e.g., based on ROC)
best_model_name <- names(which.max(colMeans(average_roc_values)))
best_model_caret <- models[[best_model_name]]

# Retrain the best model explicitly for variable importance if needed
Fit_best <- train(OS_30day ~ ., data = train_data, method = "spls", 
                  tuneGrid = expand.grid(K = best_model_caret$bestTune$K, 
                                         eta = best_model_caret$bestTune$eta, 
                                         kappa = best_model_caret$bestTune$kappa),
                  trControl = control)

# Variable Importance Plot
feature_importance <- as.data.frame(caret::varImp(Fit_best, scale = FALSE)[["importance"]])
feature_importance$feature <- rownames(feature_importance)
feature_importance <- feature_importance[, c("Overall", "feature")]
colnames(feature_importance) <- c("score", "feature")

feature_importance <- feature_importance %>%
  mutate(feature = fct_reorder(feature, score))

ggplot(data = feature_importance, aes(x = feature, y = score)) +
  geom_bar(stat = "identity", aes(fill = feature)) +
  scale_fill_manual(values = colorRampPalette(c("#2A77AC", "#0071C2", "#78AB31", "#EDB11A", "#D75615", "#D55535", "#7E318A"))(nrow(feature_importance))) +
  geom_text(aes(label = round(score, 3)), vjust = 0.5, hjust = -0.05, size = 3, color = "black") +
  coord_flip() +
  labs(x = NULL, y = "Importance Score", title = paste("Variable Importance for", best_model_name)) +
  theme_bw() +
  ylim(0, max(feature_importance$score) * 1.1) +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        legend.position = "none",
        axis.text.x = element_text(size = 8, face = "bold", color = "black"),
        axis.text.y = element_text(size = 8, face = "bold", color = "black"),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
ggsave(here("R", "1.1", "3.1 OS_30day", "3.2 Model_Variable_Importance.pdf"), width = 6, height = 8)

# SHAPley Values for Model Interpretability
explainer <- explain(
  model = best_model_caret,
  data = train_data[, -which(names(train_data) == "OS_30day")],
  y = as.numeric(train_data$OS_30day == "X1"), # DALEX expects numeric target for SHAP
  label = best_model_name
)

# Individual SHAP plot for a single observation
shap_values_individual <- predict_parts(
  explainer = explainer,
  new_observation = test_data[1, -which(names(test_data) == "OS_30day")],
  type = "shap"
)
plot(shap_values_individual)

# Global SHAP for external validation cohort (assuming test_data is the external cohort here)
test_data_features <- test_data[, -which(names(test_data) == "OS_30day")]
shap_values_global_list <- lapply(1:nrow(test_data_features), function(i) {
  predict_parts(
    explainer = explainer,
    new_observation = test_data_features[i, ],
    type = "shap"
  )
})
shap_values_global <- do.call(rbind, shap_values_global_list)
data_shap_plot <- as.data.frame(shap_values_global)

# Reorder variables by median contribution for plotting
data_shap_plot <- data_shap_plot %>%
  group_by(variable_name) %>%
  mutate(median_contribution = median(contribution)) %>%
  ungroup() %>%
  arrange(median_contribution) %>%
  mutate(variable_name = factor(variable_name, levels = unique(variable_name)))

my_colors_shap <- colorRampPalette(c("#7E318A", "#0071C2", "#2A77AC", "#78AB31", "#D55535", "#D75615", "#EDB11A"))(length(unique(data_shap_plot$variable_name)))

ggplot(data_shap_plot, aes(x = variable_name, y = contribution, fill = variable_name)) +
  geom_violin(alpha = 0.8, scale = "width", color = "black") +
  scale_fill_manual(values = my_colors_shap) +
  labs(title = "SHAP Values by Variable (Test Cohort)",
       x = "Variable",
       y = "SHAP Value (Contribution)") +
  theme_bw() +
  coord_flip() + # Flip coordinates for better readability
  theme(axis.text.y = element_text(size = 10, face = "bold"),
        axis.text.x = element_text(size = 10),
        plot.title = element_text(size = 14, face = "bold"),
        legend.position = "none")
ggsave(here("R", "1.1", "3.1 OS_30day", "3.3 SHAP_Test_Validation.pdf"), width = 10, height = 6)


# --- 6. Ensemble Learning (Stacking) ---
# Check model correlations before stacking
modelCor(resamps)
splom(resamps)

# Stack models using Random Forest as the meta-learner
set.seed(101) # For reproducible stacking
stack <- caretStack(models, method = "rf",
                    metric = "ROC",
                    trControl = trainControl(method = "cv", number = 10, savePredictions = "final", classProbs = TRUE,
                                             index = createResample(train_data$OS_30day, 10),
                                             summaryFunction = twoClassSummary))
summary(stack)
print(stack)


# --- 7. Model Performance Metrics (AUC, Brier Score, Calibration) ---
# Save all trained models and data partitions for future use
save(models, stack, index, Fit_best, file = here("R", "1.1", "3.1 OS_30day", "ML结果_OS_30day.Rdata"))

# 7.1 Internal Validation (Training Data)

# Get predictions for all models on the training data
model_preds_train <- lapply(models, predict, newdata = train_data, type = "prob")
model_preds_train <- lapply(model_preds_train, function(x) x[, "X1"])
model_preds_train <- data.frame(model_preds_train)
rownames(model_preds_train) <- rownames(train_data)
true_labels_train_factor <- train_data$OS_30day
true_labels_train_numeric <- as.numeric(true_labels_train_factor == "X1")

# Calculate AUC with DeLong's CI
internal_auc_results <- list()
for (i in 1:ncol(model_preds_train)) {
  model_name <- colnames(model_preds_train)[i]
  predictions <- model_preds_train[, i]
  roc_obj_train <- pROC::roc(response = true_labels_train_factor, predictor = predictions, levels = c('X0', 'X1'), direction = "<")
  ci_auc_delong_train <- ci.auc(roc_obj_train, method = "delong")
  internal_auc_results[[model_name]] <- data.frame(
    Validation_Type = "Train", Model = model_name,
    AUC = as.numeric(ci_auc_delong_train[2]), CI_Lower = as.numeric(ci_auc_delong_train[1]), CI_Upper = as.numeric(ci_auc_delong_train[3]))
}
internal_auc_table <- do.call(rbind, internal_auc_results)
write.csv(internal_auc_table, here("R", "1.1", "3.1 OS_30day", "3.3 AUC_DeLong_CI_train.csv"), row.names = FALSE)

# Calculate Brier Score, Calibration Intercept and Slope with CI
internal_calibration_results <- list()
for (i in 1:ncol(model_preds_train)) {
  model_name <- colnames(model_preds_train)[i]
  predictions <- model_preds_train[, i]
  brier_score_train <- mean((predictions - true_labels_train_numeric)^2)
  cal_data_train <- data.frame(prob = predictions, outcome = true_labels_train_numeric)
  cal_model_train <- glm(outcome ~ prob, data = cal_data_train, family = binomial())
  cal_intercept_train <- coef(cal_model_train)[1]
  cal_slope_train <- coef(cal_model_train)[2]
  conf_int_train <- confint(cal_model_train, level = 0.95)
  internal_calibration_results[[model_name]] <- data.frame(
    Validation_Type = "Train", Model = model_name,
    Brier_Score = brier_score_train, Calibration_Intercept = cal_intercept_train, 
    CI_Intercept_Lower = conf_int_train[1, 1], CI_Intercept_Upper = conf_int_train[1, 2],
    Calibration_Slope = cal_slope_train, CI_Slope_Lower = conf_int_train[2, 1], 
    CI_Slope_Upper = conf_int_train[2, 2])
}
internal_calibration_table <- do.call(rbind, internal_calibration_results)
write.csv(internal_calibration_table, here("R", "1.1", "3.1 OS_30day", "3.4 Calibration_Brier_train.csv"), row.names = FALSE)


# 7.2 External Validation (Test Data)
# Get predictions for all models on the test data
model_preds_test <- lapply(models, predict, newdata = test_data, type = "prob")
model_preds_test <- lapply(model_preds_test, function(x) x[, "X1"])
model_preds_test <- data.frame(model_preds_test)
rownames(model_preds_test) <- rownames(test_data)
true_labels_test_factor <- test_data$OS_30day
true_labels_test_numeric <- as.numeric(true_labels_test_factor == "X1")

# Calculate AUC with DeLong's CI
external_auc_results <- list()
for (i in 1:ncol(model_preds_test)) {
  model_name <- colnames(model_preds_test)[i]
  predictions <- model_preds_test[, i]
  roc_obj_test <- pROC::roc(response = true_labels_test_factor, predictor = predictions, levels = c('X0', 'X1'), direction = "<")
  ci_auc_delong_test <- ci.auc(roc_obj_test, method = "delong")
  external_auc_results[[model_name]] <- data.frame(
    Validation_Type = "Test", Model = model_name,
    AUC = as.numeric(ci_auc_delong_test[2]), CI_Lower = as.numeric(ci_auc_delong_test[1]), CI_Upper = as.numeric(ci_auc_delong_test[3]))
}
external_auc_table <- do.call(rbind, external_auc_results)
write.csv(external_auc_table, here("R", "1.1", "3.1 OS_30day", "3.3 AUC_DeLong_CI_test.csv"), row.names = FALSE)

# Calculate Brier Score, Calibration Intercept and Slope with CI
external_calibration_results <- list()
for (i in 1:ncol(model_preds_test)) {
  model_name <- colnames(model_preds_test)[i]
  predictions <- model_preds_test[, i]
  brier_score_test <- mean((predictions - true_labels_test_numeric)^2)
  cal_data_test <- data.frame(prob = predictions, outcome = true_labels_test_numeric)
  cal_model_test <- glm(outcome ~ prob, data = cal_data_test, family = binomial())
  cal_intercept_test <- coef(cal_model_test)[1]
  cal_slope_test <- coef(cal_model_test)[2]
  conf_int_test <- confint(cal_model_test, level = 0.95)
  external_calibration_results[[model_name]] <- data.frame(
    Validation_Type = "Test", Model = model_name,
    Brier_Score = brier_score_test, Calibration_Intercept = cal_intercept_test, 
    CI_Intercept_Lower = conf_int_test[1, 1], CI_Intercept_Upper = conf_int_test[1, 2],
    Calibration_Slope = cal_slope_test, CI_Slope_Lower = conf_int_test[2, 1], 
    CI_Slope_Upper = conf_int_test[2, 2])
}
external_calibration_table <- do.call(rbind, external_calibration_results)
write.csv(external_calibration_table, here("R", "1.1", "3.1 OS_30day", "3.4 Calibration_Brier_test.csv"), row.names = FALSE)


# --- 8. Confusion Matrix and Performance Metrics ---
# Prepare test data with model predictions for confusion matrix calculation
cal_test_data <- test_data # Rename for clarity matching original script
cal_test_data <- cbind(cal_test_data, model_preds_test)

# Loop through models and calculate confusion matrix metrics
confusion_metrics_results <- data.table()
for (model_name in colnames(model_preds_test)) {
  # Find optimal cutoff point using Youden's J statistic
  roc_obj_cutoff <- pROC::roc(response = cal_test_data$OS_30day, predictor = cal_test_data[[model_name]], levels = c('X0', 'X1'), direction = "<")
  # cutoff::roc returns a list where the third element is the optimal cutoff
  optimal_cutoff <- as.numeric(cutoff::roc(cal_test_data[[model_name]], as.numeric(cal_test_data$OS_30day == "X1"))[3]) 
  
  # Classify predictions based on optimal cutoff
  predicted_group <- ifelse(cal_test_data[[model_name]] > optimal_cutoff, "X1", "X0")
  
  # Ensure factor levels are consistent for confusionMatrix
  predicted_group <- factor(predicted_group, levels = c("X0", "X1"))
  actual_group <- factor(cal_test_data$OS_30day, levels = c("X0", "X1"))
  
  confusion_mn <- confusionMatrix(predicted_group, actual_group)
  
  metric_df <- data.frame(
    Method = model_name,
    Accuracy = confusion_mn$overall["Accuracy"],
    Balanced_Accuracy = confusion_mn$byClass["Balanced Accuracy"],
    Precision = confusion_mn$byClass["Precision"],
    Recall = confusion_mn$byClass["Recall"],
    F1_score = confusion_mn$byClass["F1"],
    Specificity = confusion_mn$byClass["Specificity"],
    Sensitivity = confusion_mn$byClass["Sensitivity"],
    NPV = confusion_mn$byClass["Neg Pred Value"],
    PPV = confusion_mn$byClass["Pos Pred Value"]
  )
  confusion_metrics_results <- rbindlist(list(confusion_metrics_results, metric_df), use.names = TRUE, fill = TRUE)
}
write.csv(confusion_metrics_results, file = here("R", "1.1", "3.1 OS_30day", "4. Confusion_Matrix_Metrics_OS_30day.csv"), row.names = FALSE)

# 8.1 Best Model Confusion Matrix Visualization (e.g., spls)
# Assuming 'spls' was chosen as the best_model_name, adjust if different
best_model_test_preds <- cal_test_data[[best_model_name]]
roc_obj_best <- pROC::roc(response = cal_test_data$OS_30day, predictor = best_model_test_preds, levels = c('X0', 'X1'), direction = "<")
best_optimal_cutoff <- as.numeric(cutoff::roc(best_model_test_preds, as.numeric(cal_test_data$OS_30day == "X1"))[3]) 

best_model_predicted_group <- ifelse(best_model_test_preds > best_optimal_cutoff, "X1", "X0")
best_model_predicted_group <- factor(best_model_predicted_group, levels = c("X0", "X1"))
actual_group_factor <- factor(cal_test_data$OS_30day, levels = c("X0", "X1"))

confusion_mn_best <- confusionMatrix(best_model_predicted_group, actual_group_factor)
print(confusion_mn_best)

confusion_table_df <- as.data.frame.matrix(confusion_mn_best$table)
confusion_table_df$actual <- rownames(confusion_table_df)
confusion_matrix_long <- gather(confusion_table_df, key = "predicted", value = "count", -actual)
confusion_matrix_long$count_label <- paste("n =", confusion_matrix_long$count)

ggplot(data = confusion_matrix_long, aes(x = predicted, y = actual, fill = count)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "#D55535") +
  geom_text(aes(label = count_label), size = 4, color = "black") +
  labs(title = paste("Confusion Matrix for", best_model_name), x = "Predicted", y = "Actual") +
  theme_minimal()
ggsave(here("R", "1.1", "3.1 OS_30day", "5.1 Confusion_Matrix_Best_Model_OS_30day.pdf"), width = 5, height = 4.5)

# 8.2 Best Model Performance Bar Chart
metric_df_best <- data.frame(
  Metric = c("Accuracy", "Specificity", "Sensitivity"),
  Value = c(confusion_mn_best$overall["Accuracy"], confusion_mn_best$byClass["Specificity"], confusion_mn_best$byClass["Sensitivity"])
)

ggplot(data = metric_df_best, aes(x = Metric, y = Value)) +
  geom_bar(stat = "identity", aes(fill = Metric)) +
  scale_fill_manual(values = c("#2A77AC", "#0071C2", "#78AB31")) +
  geom_text(aes(label = round(Value, 2)), vjust = 0.5, hjust = -0.05, size = 5, color = "black") +
  coord_flip() +
  labs(x = NULL, y = "Value", title = "Best Model Performance Evaluation") +
  theme_classic() +
  ylim(0, 1.1) +
  theme(plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
        legend.position = "none",
        axis.text.x = element_text(size = 14, face = "bold", color = "black"),
        axis.text.y = element_text(size = 14, face = "bold", color = "black"),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
ggsave(here("R", "1.1", "3.1 OS_30day", "5. Model_Performance_Best_Model_OS_30day.pdf"), width = 6, height = 3)


# --- 9. ROC Curve Plot for All Models ---
roc_plot_data <- data.frame()
for (model_name in colnames(model_preds_test)) {
  roc_obj_plot <- pROC::roc(response = cal_test_data$OS_30day, predictor = cal_test_data[[model_name]], levels = c('X0', 'X1'), direction = "<")
  auc_val <- as.numeric(pROC::auc(roc_obj_plot))
  ci_auc_obj <- pROC::ci.auc(roc_obj_plot, method = "delong")
  auc_label <- paste0("AUC=", round(auc_val, 3), ", 95%CI (", round(ci_auc_obj[1], 3), " - ", round(ci_auc_obj[3], 3), ")")
  
  if (auc_val < 0.6) {next} # Skip models with AUC less than 0.6
  
  pred_obj <- prediction(cal_test_data[[model_name]], cal_test_data$OS_30day)  
  perf_obj <- performance(pred_obj, "tpr", "fpr") 
  
  current_model_roc <- data.frame(
    name = model_name,
    TPR = unlist(perf_obj@y.values),
    FPR = unlist(perf_obj@x.values),
    AUC_Label = paste0(model_name, ": ", auc_label)
  )
  roc_plot_data <- rbind(roc_plot_data, current_model_roc)
}

ggplot(roc_plot_data, aes(x = FPR, y = TPR, color = AUC_Label)) +
  geom_line(size = 0.8) +
  labs(title = "ROC Curve", x = "False Positive Rate (1 - Specificity)", y = "True Positive Rate (Sensitivity)") +
  scale_color_manual(values = colorRampPalette(c("#2A77AC", "#0071C2", "#78AB31", "#EDB11A", "#D75615", "#D55535", "#7E318A"))(length(unique(roc_plot_data$AUC_Label)))) +                 
  geom_abline(lty = 2) +
  theme_minimal() + theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), 
        legend.title = element_blank(), 
        legend.background = element_rect(fill = NULL, size = 0.5, linetype = "solid", colour = "black"))
ggsave(here("R", "1.1", "3.1 OS_30day", "6. ROC_Test_OS_30day.pdf"), width = 12, height = 8)


# --- 10. Nomogram and Decision Curve Analysis (DCA) ---
# Create 'test_for_dca' by adding predictions to 'test_data'
test_for_dca <- test_data
for (model_name in colnames(model_preds_test)) {
  test_for_dca[[model_name]] <- model_preds_test[[model_name]]
}

# 10.1 Nomogram Construction
# Define factor variables for nomogram (adjust column names based on your Lasso selection)
# The 'Degree' variable is not previously defined; removing or requiring explicit definition.
test_for_dca$gender <- factor(test_for_dca$gender, levels = c(0, 1), labels = c('Male', 'Female'))
test_for_dca$admission_type <- factor(test_for_dca$admission_type, 
                                      levels = c(0, 1, 2, 3), # Adjust levels based on your data encoding
                                      labels = c('EMER', 'SURGERY', 'URGENT', 'ELECTIVE'))
test_for_dca$insurance <- factor(test_for_dca$insurance, 
                                 levels = c(0, 1, 2, 3), # Adjust levels
                                 labels = c('Medicare', 'Private', 'Medicaid', 'Other'))
test_for_dca$marital_status <- factor(test_for_dca$marital_status, 
                                      levels = c(0, 1, 2, 3), # Adjust levels
                                      labels = c('MARRIED', 'SINGLE', 'DIVORCED', 'WIDOWED'))
test_for_dca$race <- factor(test_for_dca$race, 
                            levels = c(0, 1, 2), # Adjust levels
                            labels = c('WHITE', 'BLACK', 'ASIAN/OTHER')) # Combine small groups if needed

ddist <- datadist(test_for_dca)
options(datadist = 'ddist')

# Build Logistic model for nomogram (using selected features and best model's predictions)
# Replace 'best_model_name' with an actual prediction column if preferred (e.g., 'spls')
# And ensure all listed features (Age, Gender etc.) are in test_for_dca
# This model uses the *prediction* from a previous ML model as a single predictor, plus demographics.
nomogram_model_fit <- lrm(OS_30day ~ Age + gender + admission_type + insurance + marital_status + race + .data[[best_model_name]], 
                          data = test_for_dca, x = TRUE, y = TRUE)

pdf(here("R", "1.1", "3.1 OS_30day", "Nomogram_OS_30day.pdf"), width = 8, height = 5)
regplot(nomogram_model_fit,
        plots = c("violin", "boxes"), 
        observation = TRUE, 
        center = TRUE, 
        subticks = TRUE,
        droplines = TRUE,
        title = "Prediction Nomogram",
        dencol = "#fdc58f", boxcol = "#99cbeb",
        points = FALSE, 
        odds = TRUE, 
        showP = TRUE, 
        rank = "sd", 
        interval = "confidence", 
        clickable = FALSE)
dev.off()

# 10.2 Decision Curve Analysis (DCA)
# Ensure OS_30day is 0/1 numeric for rmda functions
test_for_dca$OS_30day_numeric <- as.numeric(test_for_dca$OS_30day == "X1")

# The list of models for DCA should be actual prediction columns in test_for_dca
dca_model_list <- list()
for (m_name in colnames(model_preds_test)) {
  # Add demographic features to the model for DCA, if desired.
  # This implies a full model for each prediction.
  # Simplified for just the prediction: decision_curve(OS_30day_numeric ~ .data[[m_name]], ...)
  # If you want to include demographic variables again, ensure they are in test_for_dca
  if (m_name %in% supported_methods) { # Ensure method was actually trained
    dca_model_list[[m_name]] <- decision_curve(OS_30day_numeric ~ .data[[m_name]] + age + gender + admission_type + insurance + marital_status + race,
                                               data = test_for_dca,
                                               family = binomial(link = 'logit'),
                                               thresholds = seq(0, 1, by = 0.01),
                                               confidence.intervals = 0.95,
                                               study.design = "cohort")
  }
}

pdf(here("R", "1.1", "3.1 OS_30day", "9. Plot_Decision_Curve_OS_30day.pdf"), height = 10, width = 20)
plot_decision_curve(dca_model_list,
                    curve.names = names(dca_model_list),
                    cost.benefit.axis = TRUE,
                    col = colorRampPalette(c("#2A77AC", "#0071C2", "#78AB31", "#EDB11A", "#D75615", "#D55535", "#7E318A"))(length(dca_model_list)),
                    confidence.intervals = FALSE,
                    standardize = TRUE)
dev.off()

# Plot Clinical Impact Curve (CIC) for the best model
pdf(here("R", "1.1", "3.1 OS_30day", "10. Clinical_Impact_Curve_CIC_OS_30day.pdf"), height = 6, width = 10)
plot_clinical_impact(dca_model_list[[best_model_name]], 
                     population.size = 2000,
                     cost.benefit.axis = TRUE,
                     n.cost.benefits = 10,
                     col = c('#D75615', '#0071C2'),
                     confidence.intervals = TRUE,
                     ylim = c(0, 2000),
                     legend.position = "bottomleft")
dev.off()
