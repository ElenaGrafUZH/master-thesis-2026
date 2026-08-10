# Prediction of Insulin Sensitivity using Smartwatch-derived Lifestyle Features in Early Postmenopausal Women with Diabetes
## Master Thesis, MSc Computational Biology and Bioinformatics, University of Bern

This repository contains the analysis code developed for my Master's thesis. 
The thesis investigates smartwatch-derived lifestyle features in early postmenopausal and obese women with diabetes. 
The analyses explore relationships among the feature domains sleep, activity, and HR & HRV, investigate underlying lifestyle dimensions
using principal component analysis (PCA), and assess whether smartwatch-derived features can improve the classification of insulin sensitivity status beyond anthropometric predictors. 

## Repository Structure

```text
MasterThesisFinal/
├── scripts/
│   ├── 0_preprocessing/
│   ├── 1_descriptive_analysis/
│   ├── 2_logistic_regression/
│   ├── data_helper/
│   └── plot_helper/
│
├── requirements.txt
├── r_packages.txt
├── settings.ini
└── README.md
```

### `scripts/`

The analysis code is organized according to the main stages of the thesis workflow.

- **`0_preprocessing/`** – preprocessing of the original study data and generation of data files required for subsequent analyses.
- **`1_descriptive_analysis/`** – descriptive statistics, feature distributions, correlation analyses, and exploratory principal component analysis.
- **`2_logistic_regression/`** – prediction analyses using regularized logistic regression, including PCA-based predictor construction, primary model evaluation, stability analyses, and cutoff sensitivity analyses.
- **`data_helper/`** – shared functions for data loading, cleaning, transformation, and preparation.
- **`plot_helper/`** – shared plotting utilities and definitions used throughout the analyses.

## Data Availability

The data used for this thesis originate from the DECLARED study and are subject to data protection and study-specific access restrictions.

**No participant-level, raw, processed, or intermediate study data are included in this repository.**

Consequently, the analyses cannot be reproduced from this repository alone without authorized access to the underlying study data. 
The scripts are provided to document the preprocessing, feature engineering, statistical analysis, and prediction modeling workflow used for the thesis. Paths to data files as well as output folders must be adjusted accordingly in each file. 


## Software Requirements

The analyses were performed using Python (v3.14.3) and R (v4.5.3).

### Python

Python dependencies are listed in:

```text
requirements.txt
```

They can be installed using:

```bash
pip install -r requirements.txt
```

The main Python packages used in the project include:

- NumPy
- pandas
- SciPy
- scikit-learn
- statsmodels
- matplotlib
- seaborn
- Plotly
- joblib

### R

R dependencies are listed in:

```text
R-packages.txt
```

The R code is used for the calculation of the heart rate and heart rate variability features.

## Configuration

Configuration parameters for the access to the RedCAP API are defined in:

```text
settings.ini
```

The file needs to be updated with valid credentials before running the scripts.

## Analysis Workflow

The main analysis workflow consists of:

1. preprocessing smartwatch and clinical data
2. generating participant-level smartwatch-derived lifestyle features
3. describing feature distributions and summary statistics
4. examining within and across domain correlations
5. performing exploratory PCA across smartwatch-derived features
6. constructing domain-specific PCA predictors
7. fitting L2 regularized logistic regression models for insulin sensitivity classification
8. evaluating predictive performance using nested repeated cross validation
9. assessing the stability of PCA-derived components and prediction results
10. evaluating the robustness of the prediction results across alternative Matsuda index cutoffs

## Prediction Models

The prediction analysis compares several predictor sets based on anthropometric variables and smartwatch-derived feature domains.

Smartwatch-derived features are reduced using principal component analysis within the relevant training data before inclusion in the prediction models. 
L2-regularized logistic regression is used for classification, and model performance is evaluated using repeated nested cross-validation.

The Matsuda index is used as the measure of insulin sensitivity.

**Author:** Elena Graf  
**Year:** 2026
