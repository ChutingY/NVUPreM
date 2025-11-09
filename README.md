# NVUPreM: A Machine Learning Model for 30-Day Mortality Prediction in Non-Variceal Upper Gastrointestinal Bleeding (NVUGIB)

## Abstract

**Background:** Accurate early prognostication is critical for managing non-variceal upper gastrointestinal bleeding (NVUGIB) mortality in critically ill patients. Machine learning (ML) offers potential advantages over traditional clinical scores. This study aims to develop and validate a ML model named NVUPreM for superior prediction of 30-day NVUGIB mortality.

**Methods:** Using data from the Medical Information Mart for Intensive Care IV (n=11,237) and the eICU Collaborative Research Database (n=7,742), we retrospectively developed and externally validated the NVUPreM model. Predictors were selected via least absolute shrinkage and selection operator regression. Performance of 36 algorithms was evaluated using tenfold cross-validation. The optimal NVUPreM model was compared against eight clinical scoring systems (AIMS65, Charlson, GBS, GCS, Admission-Rockall, SAPSII, SOFA) using the area under the receiver operating characteristic curve (AUC), calibration, decision curve analysis (DCA), and SHapley Additive exPlanations for interpretability.

**Results:** The NVUPreM model demonstrated superior discrimination (AUC=0.876, [95% CI 0.846-0.907]) and sensitivity (0.86) and outperformed all clinical scores internally via AUCs (AIMS65: AUC=0.693; Charlson: AUC=0.636; GBS: AUC=0.575; GCS: AUC=0.707; NVUPreM: AUC=0.876; Admission-Rockall: AUC=0.633; SAPSII: AUC=0.777; SOFA: AUC=0.665), DCA and calibration curve. External validation in eICU confirmed robustness in terms of discrimination (AUC=0.82, [95% CI 0.803-0.837]), calibration, and clinical application. The interpretability analysis revealed directional feature contributions, identifying predictors with significantly positive and negative impacts on the model output.

**Conclusion:** The NVUPreM model significantly outperforms existing clinical scores in predicting 30-day NVUGIB mortality, offering improved accuracy and interpretability for early high-risk patient identification and personalized intervention.

**Keywords:** Non-variceal upper gastrointestinal bleeding; Machine learning; Mortality prediction; Intensive care unit; Explainable artificial intelligence

---

## 1. Project Overview

This repository hosts the R code and associated files for the development and validation of **NVUPreM**, a novel machine learning model designed to predict 30-day mortality in critically ill patients with non-variceal upper gastrointestinal bleeding (NVUGIB). The project employs a rigorous methodology, including comprehensive feature selection, evaluation of numerous machine learning algorithms, and robust external validation using large-scale intensive care unit (ICU) datasets.

Our findings demonstrate that NVUPreM significantly surpasses the predictive capabilities of conventional clinical scoring systems, offering a more accurate and interpretable tool for early identification of high-risk patients, thereby facilitating personalized clinical interventions.

## 2. Key Features

*   **Advanced Machine Learning:** Utilizes a diverse set of 36 ML algorithms to identify the optimal predictive model.
*   **Comprehensive Feature Selection:** Employs LASSO regression for robust and parsimonious feature selection.
*   **Extensive Validation:** Developed on MIMIC-IV and externally validated on the eICU Collaborative Research Database, ensuring generalizability.
*   **Comparative Analysis:** Directly compares NVUPreM's performance against eight widely-used clinical scoring systems using multiple robust metrics (AUC, calibration, DCA).
*   **Explainable AI (XAI):** Incorporates SHapley Additive exPlanations (SHAP) for transparent and interpretable model insights, revealing individual feature contributions.
*   **Reproducible Research:** All analysis scripts are provided to ensure full transparency and reproducibility of the study's findings.

## 3. Data Sources

The study utilizes two large, publicly available, de-identified critical care datasets:

*   **Medical Information Mart for Intensive Care IV (MIMIC-IV):** Used for model development. Access requires completion of CITI training and application through [PhysioNet](https://physionet.org/content/mimiciv/2.2/).
*   **eICU Collaborative Research Database:** Used for external validation. Access requires completion of CITI training and application through [PhysioNet](https://physionet.org/content/eicu-crd/2.0/).

**Note:** Due to data privacy regulations, the raw datasets are not included in this repository. Users intending to reproduce the analysis must obtain access to these databases independently and follow their respective data use agreements.

## 4. R Scripts Description:

*   `1_Clinical_Score_Calculation_and_Imputation_Pipeline.R`: Contains code for calculating various clinical scores (e.g., AIMS65, Charlson, GBS) and implementing multiple imputation techniques for handling missing data, preparing the dataset for subsequent modeling.
*   `2_Exploratory_Analysis_of_Clinical_Scores.R`: Performs initial descriptive statistics and exploratory data analysis focusing on existing clinical scores and their distribution concerning the composite endpoint.
*   `3_Feature_Selection_and_Dimensionality_Reduction.R`: Implements feature selection using LASSO regression and explores dimensionality reduction techniques like PCA to refine the set of predictors for the machine learning models.
*   `4_Comparative_Performance_Evaluation_ML_Models.R`: The core script for training and evaluating 36 different machine learning algorithms using 10-fold cross-validation. It also includes the generation of SHAP values for interpretability of the best model.
*   `5_Comparative_Evaluation_Clinical_Risk_Scores_and_Novel_Prediction_Model.R`: Conducts a head-to-head comparison of the developed NVUPreM model against traditional clinical scores using AUC, calibration curves, and Decision Curve Analysis (DCA).

