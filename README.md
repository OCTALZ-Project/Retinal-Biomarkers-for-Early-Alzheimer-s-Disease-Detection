# Retinal Biomarkers for Early Alzheimer's Disease Detection

This repository contains the code and analysis pipeline for investigating retinal biomarkers as potential indicators for early detection of Alzheimer's Disease (AD) and dementia, using data from the UK Biobank.

## Project Overview

The study leverages retinal Optical Coherence Tomography (OCT) measurements, along with demographic and clinical features, to build predictive models for Alzheimer's Disease and dementia. The pipeline includes data preprocessing, XGBoost-based classification, SHAP-based model interpretability, and survival analysis using Cox proportional hazards models and Nelson-Aalen estimators.

## Repository Structure

```
.
├── Data Cleaning/
│   ├── create_data.ipynb                          # UK Biobank field code processing and dataset creation
│   ├── data_preprocessing.ipynb                   # Data preprocessing (downsampling, age-matching, imputation)
│   └── data_preprocessing_with_deprecated.ipynb   # Legacy preprocessing script
├── XGBoost/
│   ├── main.ipynb                # Main training script (XGBoost across 5 datasets x 5 splits)
│   ├── functions.py              # Core helper functions for classification and evaluation
│   ├── functions_cox.py          # Survival analysis helper functions (Cox PH, Nelson-Aalen)
│   ├── ad_cox.ipynb              # Survival analysis script (Cox & Nelson-Aalen)
│   ├── create_graphs.ipynb       # Graph/figure generation
│   ├── get_roc_curves.ipynb      # ROC curve generation
│   └── main deprecated.ipynb    # Legacy training script
├── requirements.txt              # Conda environment specification
└── README.md
```

## Pipeline

### 1. Data Creation (`Data Cleaning/create_data.ipynb`)
Processes raw UK Biobank data by mapping demographic and numerical field codes to human-readable labels. Handles OCT measurements from multiple instances and eyes (left/right, instance 0/1).

### 2. Data Preprocessing (`Data Cleaning/data_preprocessing.ipynb`)
Designed to be executed **5 times** with consecutive seed values to produce 5 distinct dataset variants for robustness analysis. Includes:
- Random downsampling of healthy records (~90%) to reduce class imbalance
- Age-matched cohort construction
- KNN imputation for missing values
- Random Cohort Selection (RCS) imputation as an alternative strategy

### 3. XGBoost Classification (`XGBoost/main.ipynb`)
Main execution script that automates model training and evaluation across 5 datasets and 5 train-test splits (25 total runs). For each configuration, it trains:
- **Weighted XGBoost** (with balanced class weights)
- **Unweighted XGBoost** (baseline)

Supports both binary (AD vs. Healthy, Dementia vs. Healthy) and multiclass (Healthy, Dementia, AD) classification tasks.

**Evaluation metrics:** ROC-AUC, AUPRC, MCC, F1 Score, Accuracy, Sensitivity, Specificity

**Sampling strategies:**
- Custom cluster-based undersampling (KMeans with majority voting for categorical features)
- SMOTE-NC upsampling with TomekLinks

### 4. Survival Analysis (`XGBoost/ad_cox.ipynb`)
Performs survival analysis using:
- **Cox Proportional Hazards models** to evaluate feature effects on survival time with hazard ratios and statistical significance
- **Nelson-Aalen cumulative hazard estimation** with risk tables and stratified plots

## Key Features Used

- **Retinal OCT:** mRNFL thickness, mGCIPL thickness
- **Demographics:** Age, Sex, Educational status
- **Clinical:** BMI, Systolic/Diastolic blood pressure, Spherical equivalent
- **Lifestyle:** Diabetes, Alcohol consumption, Smoking status, Antihypertensive usage

## Requirements

- Python 3.10+
- Conda (for environment setup)

### Setup

```bash
conda create --name retinal-ad --file requirements.txt
conda activate retinal-ad
```

### Key Dependencies

- xgboost
- scikit-learn
- imbalanced-learn (SMOTE-NC, TomekLinks)
- lifelines (Cox PH, Nelson-Aalen, Kaplan-Meier)
- shap
- pandas, numpy, scipy
- matplotlib, seaborn, plotly

## Usage

1. **Prepare the data:**
   ```
   Run Data Cleaning/create_data.ipynb
   ```

2. **Preprocess (run 5 times with SEED = 1..5):**
   ```
   Run Data Cleaning/data_preprocessing.ipynb
   ```

3. **Train and evaluate models:**
   ```
   Run XGBoost/main.ipynb
   ```

4. **Run survival analysis:**
   ```
   Run XGBoost/ad_cox.ipynb
   ```

## Data Availability

The dataset used in this study is subject to UK Biobank access restrictions and can only be obtained through the official application process via the [UK Biobank website](https://www.ukbiobank.ac.uk/).
