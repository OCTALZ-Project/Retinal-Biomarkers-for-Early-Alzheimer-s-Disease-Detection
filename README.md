# Retinal Biomarkers for Early Alzheimer's Disease Detection

This repository contains the code and analysis pipeline for investigating retinal biomarkers as potential indicators for early detection of Alzheimer's Disease (AD) and dementia, using data from the UK Biobank.

## Project Overview

The study leverages retinal Optical Coherence Tomography (OCT) measurements, along with demographic and clinical features, to build predictive models for Alzheimer's Disease and dementia. The pipeline includes data preprocessing, XGBoost-based classification, SHAP-based model interpretability, and survival analysis using Cox proportional hazards models and Nelson-Aalen estimators.

## Repository Structure

```
.
├── Data Cleaning/
│   ├── create_data.ipynb          # UK Biobank field-code processing and dataset creation
│   └── data_preprocessing.ipynb   # Preprocessing (missingness matching, age matching, imputation)
├── Fix Imbalance/
│   ├── main.ipynb                 # Main training script (XGBoost across 10 datasets x 5 folds)
│   ├── functions.py               # Core helper functions for classification and evaluation
│   ├── functions_cox.py           # Survival analysis helper functions (Cox PH, Nelson-Aalen)
│   ├── ad_cox.ipynb               # Survival analysis notebook (Cox & Nelson-Aalen)
│   ├── create_graphs.ipynb        # Graph/figure generation
│   ├── get_roc_curves.ipynb       # ROC curve generation
│   └── make_fig2_histogram.py     # Cohort histogram figure
├── requirements.txt               # Python dependencies (pip)
└── README.md
```

## Pipeline

### 1. Data Creation (`Data Cleaning/create_data.ipynb`)
Processes raw UK Biobank data by mapping demographic and numerical field codes to human-readable labels. Handles OCT measurements from multiple instances and eyes (left/right, instance 0/1), and writes the assembled table to `./results/df.csv`.

### 2. Data Preprocessing (`Data Cleaning/data_preprocessing.ipynb`)
Designed to be executed **10 times** with consecutive seed values (`SEED = 1..10`) to produce 10 distinct dataset variants for robustness analysis. It builds:
- **Missingness-aware matching (missing-matched):** healthy controls are matched to each AD/dementia case by their pattern of missing features. This is the primary cohort used in the paper, produced alongside an **age-matched** version.
- Each cohort is written in three versions:
  - **unimputed** — missing values kept (XGBoost handles them natively);
  - **KNN-imputed** — custom cosine-distance nearest-neighbour imputation;
  - **RCS-imputed** — Random Case Sampling: each missing value is filled with a randomly drawn observed value from a balanced reference pool.

### 3. XGBoost Classification (`Fix Imbalance/main.ipynb`)
Main execution notebook that automates model training and evaluation across 10 datasets with 5-fold cross-validation (50 total runs). For each configuration, it trains:
- **Weighted XGBoost** (with balanced class weights)
- **Unweighted XGBoost** (baseline)

Supports both binary (AD vs. Healthy, Dementia vs. Healthy) and multiclass (Healthy, Dementia, AD) classification tasks.

**Evaluation metrics:** ROC-AUC, AUPRC, MCC, F1 Score, Accuracy, Sensitivity, Specificity

**Sampling strategies:**
- Custom cluster-based undersampling (KMeans with majority voting for categorical features)
- SMOTE-NC upsampling with TomekLinks

### 4. Survival Analysis (`Fix Imbalance/ad_cox.ipynb`)
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

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Figures embed **Arial** as a Type-42 (TrueType) font. If Arial is not installed, matplotlib falls back to another sans-serif face (Liberation Sans / Nimbus Sans / DejaVu Sans). See the header of `requirements.txt` for how to install the Microsoft core fonts.

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

2. **Preprocess (run 10 times with SEED = 1..10):**
   ```
   Run Data Cleaning/data_preprocessing.ipynb
   ```

3. **Train and evaluate models:**
   ```
   Run Fix Imbalance/main.ipynb
   ```

4. **Run survival analysis:**
   ```
   Run Fix Imbalance/ad_cox.ipynb
   ```

## Data Availability

The dataset used in this study is subject to UK Biobank access restrictions and can only be obtained through the official application process via the [UK Biobank website](https://www.ukbiobank.ac.uk/).
