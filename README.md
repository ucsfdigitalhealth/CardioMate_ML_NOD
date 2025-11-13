# CardioMate BP Spike Prediction

*(Public Results Repository)*

This repository provides **publicly shareable results** from the
CardioMate machine-learning pipeline for predicting short-horizon
blood-pressure (BP) spikes using continuous wearable biosignals and
lightweight self-reports. It is intended to support **transparency**,
**reproducibility review**, and **journal submissions**.

> **Important:**\
> This *public* repository contains **only the processed outputs,
> figures, and summaries** necessary for readers and reviewers to
> understand the model behavior.\
> To fully run the end-to-end preprocessing and model-training pipeline,
> please follow the guidelines in the section below.

------------------------------------------------------------------------

## 🚀 Full Pipeline Access & Dataset Availability

A **small, de-identified subset** of the data (Fitbit HR/steps, EMA
stress, BP readings) is publicly available at:

**https://github.com/ucsfdigitalhealth/CardioMate_ML_NOD**

This subset is sufficient to:

-   execute the entire preprocessing pipeline\
-   perform feature engineering\
-   train and evaluate all models described in the manuscript\
-   reproduce SHAP analyses and threshold-based performance curves

The **full dataset** (\~40 participants across two studies, \~28--30
days each) will be released in a companion dataset paper with:

-   persistent identifiers\
-   comprehensive metadata and documentation\
-   formal access via IRB and Data Use Agreement (DUA)

Requests for early research access (e.g., for reproducibility
verification) may be submitted to the **corresponding author** and will
be reviewed under IRB/DUA constraints.

------------------------------------------------------------------------

## 📦 Repository Structure (Public Results Only)

This public repository includes **processed outputs and model artifacts
only**, not raw data:

    processed/
    └── hp{ID}/
        ├── processed_bp_prediction_data.csv     De-identified feature matrix
        ├── sens_spec_plot.png                   Sensitivity–specificity curve
        ├── shap_summary.png                     SHAP feature importance summary
        └── results.txt                          Model performance summary

    preprocess.py                                (reference stub – full version in NOD repo)
    train.py                                     (reference stub – full version in NOD repo)
    README.md

These outputs allow researchers and reviewers to inspect model behavior
without exposing sensitive data.

------------------------------------------------------------------------

## 🧠 Running the Full Pipeline (via the NOD Repository)

To run the full CardioMate pipeline end-to-end---including
preprocessing, feature engineering, modeling, and evaluation---please
use the scripts in:

📂 **Full pipeline:**\
https://github.com/ucsfdigitalhealth/CardioMate_ML_NOD

### **1. Install dependencies**

``` bash
pip install -r requirements.txt
```

### **2. Preprocess data**

``` bash
python preprocess.py --participant_id 31
```

### **3. Train models**

``` bash
python train.py --participant_id 31 --models xgb,attn --verbose
```

Pipeline outputs will be stored under:

    processed/hp{ID}/

------------------------------------------------------------------------

## ⚙️ Prerequisites

-   Python 3.8+
-   Required packages:\
    `pandas` `numpy` `scikit-learn` `xgboost` `imbalanced-learn`\
    `tensorflow` `keras-tuner` `shap` `matplotlib`

Complete dependency list is available in the NOD repository.

------------------------------------------------------------------------

## 📄 Notes

-   No raw participant data appear in this repository.\
-   All entries are de-identified and suitable for public sharing.\
-   This repository is safe to cite in manuscripts or supplementary
    materials.

------------------------------------------------------------------------

## 📬 Contact

For dataset access, reproducibility requests, or pipeline inquiries,
contact the **corresponding author** listed in the manuscript.
