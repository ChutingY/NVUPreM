# --- 1. Environment Setup and Library Loading ---
library(here) 
library(dplyr)
library(ggplot2)
library(PerformanceAnalytics) # For correlation matrix visualization
library(scitb)              # For quick descriptive statistics and group comparisons
library(ggpubr)             # For enhanced ggplot2 visualizations and statistical tests
library(autoReg)            # For automated logistic regression table generation
library(rrtable)            # For exporting tables to Word/PowerPoint
library(pROC)               # For ROC curve and AUC calculation with confidence intervals
library(ROCR)               # For general performance curves
library(cutoff)             # For optimal cutoff point determination
library(rmda)               # For Decision Curve Analysis (DCA)
library(rms)                # For calibration curves (lrm, calibrate)
library(PredictABEL)        # Potentially for additional prediction functionalities
library(forcats)            # For factor manipulation in plotting

# --- 2. Data Loading and Initial Processing ---
data <- read.csv(here("R", "1.1", "3.0 compare_with_sig.csv"), header = TRUE, row.names = 1)
# Determine optimal cutoff for NVUPreM and create a binary variable
nvuprem_roc_obj <- cutoff::roc(data$NVUPreM, data$OS_30day)
optimal_nvuprem_cutoff <- as.numeric(nvuprem_roc_obj[3])
data$NVUPreM_2 <- ifelse(data$NVUPreM <= optimal_nvuprem_cutoff, 1, 2) # 1=Low risk, 2=High risk

# Handle missing values by omission (consider imputation for robust analysis)
mydata <- na.omit(data)

# Ensure target variable is a factor with specified levels for consistency across analyses
mydata$OS_30day <- factor(mydata$OS_30day, levels = c(0, 1), labels = c("X0", "X1"))
mydata$NVUPreM_2 <- factor(mydata$NVUPreM_2, levels = c(1, 2), labels = c("Low", "High"))

# --- 3. Exploratory Data Analysis ---
# 3.1 Quick Statistical Description and Correlation Visualization
chart.Correlation(mydata[, c("Rockall", "GBS", "AIMS65", "sapsii", "sofa", "charlson", "gcs", "NVUPreM")], 
                  histogram = TRUE, pch = 19, method = 'spearman')

# 3.2 Group Comparisons for Clinical Scores (Table 1 Generation)
allVars <- c("Rockall", "GBS", "AIMS65", "sapsii", "sofa", "charlson", "gcs", "NVUPreM", "NVUPreM_2")
strata_var <- "OS_30day"

table1_output <- scitb1(vars = allVars, 
                        strata = strata_var, data = mydata, 
                        atotest = TRUE, statistic = TRUE, Overall = TRUE)
write.csv(table1_output, file = here("R", "1.1", "4.1 OS_30day compare", "1_Group_Comparison_Table.csv"), row.names = FALSE)

# 3.3 Violin Plots for Group Comparisons
my_comparisons <- list(c("X0", "X1"))
plot_theme <- theme_test() + 
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.text.x = element_text(hjust = 0.5, size = 16), 
        axis.text.y = element_text(hjust = 0.5, size = 16), 
        axis.title.x = element_blank(), 
        axis.title.y = element_text(size = 16), 
        axis.line = element_line(size = 1), 
        legend.position = "none")

# Exclude NVUPreM_2 (already a factor for table1 but continuous for violin plot)
variables_to_plot_violin <- c("Rockall", "GBS", "AIMS65", "sapsii", "sofa", "charlson", "gcs", "NVUPreM")

for (variable in variables_to_plot_violin) {
  p <- ggviolin(mydata, x = 'OS_30day', y = variable, fill = 'OS_30day',
                palette = c("#2A77AC", "#D75615"), add = 'boxplot', 
                add.params = list(fill = "white", color = "black")) + 
    plot_theme +
    stat_compare_means(comparisons = my_comparisons, label = "p.signif", size = 8,
                       bracket.size = 0.5, tip.length = 0.02, method = 'wilcox.test')
  
  pdf(here("R", "1.1", "4.1 OS_30day compare", "COMPARE_FIGURE", paste0(variable, "_violin.pdf")), width = 5, height = 7)
  print(p)
  dev.off()
}

# --- 4. Univariate Logistic Regression Analysis ---
logistic_model_fit <- glm(OS_30day ~ Rockall + GBS + AIMS65 + sapsii + sofa + charlson + gcs + NVUPreM,
                          family = binomial(link = "logit"),
                          data = mydata)

univariate_logistic_results <- autoReg(logistic_model_fit, uni = TRUE, threshold = 0.05) %>% myft()
table2pptx(univariate_logistic_results, file = here("R", "1.1", "4.1 OS_30day compare", "2_Univariate_Logistic_Regression.pptx"))
table2docx(univariate_logistic_results, file = here("R", "1.1", "4.1 OS_30day compare", "2_Univariate_Logistic_Regression.docx"))


# --- 5. Receiver Operating Characteristic (ROC) Curve Analysis ---
models_for_roc <- c("Rockall", "GBS", "AIMS65", "sapsii", "sofa", "charlson", "gcs", "NVUPreM")
roc_plot_data <- data.frame()
score_interpretations <- c(Rockall = "<", GBS = "<", AIMS65 = "<", sapsii = "<", sofa = "<", charlson = "<",
                           gcs = ">", # For GCS, lower scores mean worse outcome/higher risk
                           NVUPreM = "<") 

for (model_name in models_for_roc) {
  roc_obj <- pROC::roc(response = mydata$OS_30day, 
                       predictor = mydata[[model_name]], 
                       levels = c('X0', 'X1'), 
                       direction = score_interpretations[[model_name]])
  
  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci_auc_obj <- pROC::ci.auc(roc_obj, method = "delong")
  auc_label <- paste0("AUC=", round(auc_val, 3), ", 95%CI (", round(ci_auc_obj[1], 3), " - ", round(ci_auc_obj[3], 3), ")")
  
  # Skip models with AUC below a threshold (e.g., 0.5 for no predictive power)
  if (auc_val < 0.5) {next} 
  
  pred_obj <- ROCR::prediction(mydata[[model_name]], mydata$OS_30day, 
                               label.ordering = c('X0', 'X1')) # ROCR needs label ordering
  perf_obj <- ROCR::performance(pred_obj, "tpr", "fpr") 
  
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
  labs(title = "ROC Curve Comparison", x = "False Positive Rate (1 - Specificity)", y = "True Positive Rate (Sensitivity)") +
  scale_color_manual(values = colorRampPalette(c("#0071C2", "#D75615", "#EDB11A", "#7E318A", "#78AB31", "#2A77AC", "#D55535"))(length(unique(roc_plot_data$AUC_Label)))) +                 
  geom_abline(lty = 2) +
  theme_minimal() + theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), 
        legend.title = element_blank(), 
        legend.background = element_rect(fill = NULL, size = 0.5, linetype = "solid", colour = "black"))
ggsave(here("R", "1.1", "4.1 OS_30day compare", "3_ROC_Comparison_OS_30day.pdf"), width = 10, height = 7)


# --- 6. Decision Curve Analysis (DCA) ---

# Ensure OS_30day is numeric 0/1 for rmda functions
mydata_for_dca <- mydata %>%
  mutate(OS_30day_numeric = as.numeric(OS_30day == "X1"))

dca_results <- list()
variables_for_dca <- c("Rockall", "GBS", "AIMS65", "sapsii", "sofa", "charlson", "gcs", "NVUPreM")

for (var in variables_for_dca) {
  if (var == "gcs") {
    temp_data <- mydata_for_dca %>%
      mutate(transformed_var = max(gcs, na.rm = TRUE) - gcs) # Invert GCS for "higher is worse"
    formula <- as.formula(paste("OS_30day_numeric ~ transformed_var"))
  } else {
    temp_data <- mydata_for_dca
    formula <- as.formula(paste("OS_30day_numeric ~", var))
  }
  
  result <- decision_curve(formula, data = temp_data, 
                           thresholds = seq(0, 1, by = 0.005), 
                           bootstraps = 10,
                           study.design = "cohort") # Assuming cohort study design
  dca_results[[paste0(var, " curve")]] <- result
}

pdf(here("R", "1.1", "4.1 OS_30day compare", "4_DCA_Plot_OS_30day.pdf"), width = 6, height = 5)
plot_decision_curve(dca_results, 
                    curve.names = names(dca_results), # Use automatically generated names
                    col = colorRampPalette(c("#0071C2", "#D75615", "#EDB11A", "#7E318A", "#78AB31", "#2A77AC", "#D55535"))(length(dca_results)), 
                    confidence.intervals = FALSE,
                    cost.benefit.axis = FALSE,
                    legend.position = "topright",
                    main = "Decision Curve Analysis")
dev.off()


# --- 7. Calibration Curve Analysis ---
mydata_for_cal <- mydata %>%
  mutate(OS_30day_numeric = as.numeric(OS_30day == "X1"))

calibration_results <- list()
variables_for_calibration <- c("NVUPreM", "Rockall", "GBS", "AIMS65", "gcs", "charlson", "sofa", "sapsii") # Use continuous NVUPreM

# Set up datadist for rms package
ddist <- datadist(mydata_for_cal)
options(datadist = 'ddist')

for (var in variables_for_calibration) {
  temp_data_for_lrm <- mydata_for_cal
  if (var == "gcs") {
    temp_data_for_lrm <- temp_data_for_lrm %>%
      mutate(gcs_inverted = max(gcs, na.rm = TRUE) - gcs)
    formula <- as.formula("OS_30day_numeric ~ gcs_inverted")
  } else {
    formula <- as.formula(paste("OS_30day_numeric ~", var))
  }
  
  lrm_fit <- lrm(formula = formula, data = temp_data_for_lrm, x = TRUE, y = TRUE)
  cal_obj <- calibrate(lrm_fit, method = "boot", B = 1000)
  calibration_results[[var]] <- cal_obj
}

pdf(here("R", "1.1", "4.1 OS_30day compare", "5_Calibration_Curve_OS_30day.pdf"), width = 6, height = 6)
plot(1, type = "n",
     xlim = c(0, 1), ylim = c(0, 1),
     xaxs = "i", yaxs = "i",
     xlab = "Predicted Probability", ylab = "Observed Probability",
     legend = FALSE, subtitles = FALSE,
     cex = 1.5, cex.axis = 1.5, cex.lab = 1.5)
abline(0, 1, col = "black", lty = 2, lwd = 2)

# Plot calibration curves for each model
cal_colors <- colorRampPalette(c("#7E318A", "#0071C2", "#2A77AC", "#78AB31", "#EDB11A", "#D75615", "#D55535"))(length(variables_for_calibration))
cal_legend_labels <- c("NVUPreM", "Rockall", "GBS", "AIMS65", "GCS", "Charlson", "Sofa", "SAPSII") # Match order of variables_for_calibration

for (i in seq_along(variables_for_calibration)) {
  var <- variables_for_calibration[i]
  plot_data <- calibration_results[[var]]
  lines(plot_data[, c("predy", "calibrated.orig")], lty = 1, lwd = 2, col = cal_colors[i])
}

legend(0.01, 0.98,
       paste0(cal_legend_labels, "'s Calibration curve"),
       col = cal_colors,
       lty = rep(1, length(cal_colors)),
       lwd = rep(2, length(cal_colors)),
       bty = "n", cex = 1)
dev.off()

options(datadist = NULL) # Clear datadist
