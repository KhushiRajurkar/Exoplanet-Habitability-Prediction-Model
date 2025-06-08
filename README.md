# Exoplanet-Habitability-Prediction-Model

A machine learning pipeline to predict whether an exoplanet is habitable using pre-processed astronomical features. This project explores multiple classification models, handles class imbalance with resampling techniques, and evaluates performance using robust metrics and visualization.

---

## Overview

This project applies advanced ML techniques to classify exoplanets into:
- Not Habitable (0)
- Potentially Habitable (1)
- Habitable (2)

It includes:
- GridSearchCV for hyperparameter tuning  
- Logistic Regression, SVM, MLP, KNN models  
- ADASYN, SMOTE, SMOTE-Tomek, ClusterCentroids samplers  
- ROC curves, confusion matrices, and F1/Recall comparison

---

## Dataset

- The dataset is derived from exoplanet observational features and cleaned for modeling.
- Final file: [`hwc.xlsx`](Dataset/hwc.xlsx)
- Contains ~99 features with target label: `P_HABITABLE`

To convert to `.csv`:
```python
import pandas as pd
pd.read_excel('hwc.xlsx').to_csv('hwc.csv', index=False)
```

## Modeling Approach

The modeling pipeline follows these key stages:

1. **Preprocessing**  
   - Missing values handled
   - Feature scaling with `StandardScaler`

2. **Resampling**  
   - Addressed class imbalance using `ADASYN`
   - Oversampled rare classes to match dominant ones

3. **Model Training**  
   - Best results achieved using **Logistic Regression**
   - Hyperparameters tuned via `GridSearchCV`

4. **Evaluation Metrics**  
   - F1 Score (Macro & Weighted)
   - Recall
   - ROC AUC (for each class and overall)

5. **Output Artifacts**  
   - Saved model
   - Confusion matrices
   - Metrics export

## 📈 Performance Summary

The best-performing combination was **Logistic Regression + ADASYN**, achieving strong metrics across the board despite the imbalanced dataset.

### 🔍 Key Results:

| Metric            | Score     |
|-------------------|-----------|
| **Accuracy**      | 99%       |
| **Macro F1 Score**| 72%       |
| **Macro Recall**  | 72%       |
| **Weighted F1**   | 99%       |
| **Overall AUC**   | 0.993     |

- **Class 0 (Not Habitable):** Precision = 1.00, Recall = 0.99  
- **Class 1 (Habitable Class 1):** Precision = 0.20, Recall = 0.33  
- **Class 2 (Habitable Class 2):** Precision = 0.62, Recall = 0.83

## Confusion Matrices

Visual evaluations for all model + sampler combinations are stored in the folder: [`Confusion Matrices`)(Confusion Matrices)

Each image follows the naming format:  
`<ModelName>_<SamplerName>_ConfusionMatrix.png`
