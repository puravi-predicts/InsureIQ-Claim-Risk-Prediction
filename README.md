# InsureIQ — Predictive Risk Scoring for Insurance Claims

> An end-to-end machine learning pipeline for rare-event insurance claim prediction — benchmarking 9 algorithms, stacking the best of them, calibrating probabilities, explaining predictions with SHAP, and optimizing decisions against a real business cost matrix.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-success)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-Deployment-teal)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

Insurance companies need to predict which customers are likely to file a claim — but the event is rare (**~3.6% positive class**), the raw features are anonymized for privacy (`ps_ind_*`, `ps_car_*`, `ps_reg_*`), and a naive model can hit 96% "accuracy" by always predicting "no claim." InsureIQ tackles this as a serious, production-minded ML problem rather than a toy classification exercise:

- **9 benchmarked models** — Logistic Regression, SVM, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, and a Keras MLP/ANN.
- **Leakage-safe pipeline** — all imputation, scaling, and resampling statistics are fit on the training split only.
- **Rare-event-aware evaluation** — F1, PR-AUC, and per-model threshold calibration instead of raw accuracy.
- **Advanced ensembling** — a stacked meta-learner and soft-voting ensemble built on top of the strongest base models.
- **Calibrated probabilities** — isotonic regression so risk scores are usable for actual pricing decisions, not just ranking.
- **Explainability** — SHAP global feature impact + per-customer local explanations.
- **Business-cost-aware decisioning** — the deployment threshold is chosen to minimize ₹-denominated expected cost, not abstract F1.
- **Statistical rigor** — bootstrap confidence intervals and McNemar's test to confirm model differences are real, not noise.
- **Production readiness** — a serialized inference pipeline (`predict_risk()`) and a FastAPI deployment sketch.

---

## Project Structure

```
InsureIQ-Claim-Risk-Prediction/
│
├── notebooks/
│   └── InsClaimPred-Pipeline.ipynb                # Core pipeline: EDA → preprocessing → 9 models → tuning → comparison
│                                                  # Stacking, calibration, SHAP, cost optimization, deployment
│
├── model_artifacts/                               # Generated after running the notebooks
│   ├── best_model_calibrated.joblib
│   ├── scaler.joblib
│   ├── impute_values.joblib
│   ├── feature_columns.joblib
│   ├── inference_config.json
│   ├── experiment_log.csv
│   └── experiment_log.json
│
├── serve.py                                       # FastAPI inference service (reference deployment)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dataset

The dataset follows the **Porto Seguro Safe Driver Prediction** convention:
- ~595K customer records with anonymized `ps_ind_*`, `ps_car_*`, `ps_reg_*`, `ps_calc_*` feature families.
- Binary target: `1` = claim filed, `0` = no claim.
- Missing values encoded as the sentinel `-1` rather than `NaN`.
- Severe class imbalance: positives are ~3.6% of all records.

> Place `train.csv` in the expected data path before running the notebooks (see the **Data Loading** section in `InsClaimPred-Pipeline.ipynb`).

---

## Pipeline Architecture

```
Raw Data
   │
   ▼
┌─────────────────────────┐
│ Phase 0–1: EDA           │  Distributions, missingness, correlation, outlier profiling
└─────────────────────────┘
   │
   ▼
┌─────────────────────────┐
│ Phase 2: Preprocessing   │  -1 → NaN, Yeo-Johnson transform, IQR capping, encoding
└─────────────────────────┘
   │
   ▼
┌─────────────────────────┐
│ Phase 3: Leak-Safe Split │  Train/test split → fit imputers & scalers on train only → SMOTE on train only
└─────────────────────────┘
   │
   ▼
┌─────────────────────────┐
│ Phase 4: Model Bench     │  9 models trained & evaluated (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
└─────────────────────────┘
   │
   ▼
┌─────────────────────────┐
│ Phase 5: Tuning & Thresh.│  RandomizedSearchCV + per-model F1-optimal threshold calibration
└─────────────────────────┘
   │
   ▼
┌─────────────────────────┐
│ Phase 6–10: Advanced ML  │  Stacking, isotonic calibration, SHAP, cost-optimal thresholding,
│                          │  bootstrap CIs & McNemar significance testing
└─────────────────────────┘
   │
   ▼
┌─────────────────────────┐
│ Phase 11–13: Production  │  Model persistence, predict_risk() API, FastAPI service, experiment log
└─────────────────────────┘
```

---

## Results

By default, standard classifiers struggle with the ~3.6% severe class imbalance, often predicting the majority class exclusively at a standard 0.5 decision boundary. To extract real business value, prediction thresholds were optimized individually per model to maximize the F1-Score.

### Threshold-Optimized Performance Matrix

| Model | Optimized Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Winner)** | **0.0719** | **0.8970** | **0.0821** | **0.1793** | **0.1126** | **0.6229** | **0.0634** |
| LightGBM | 0.0642 | 0.8772 | 0.0751 | 0.2095 | 0.1106 | 0.6243 | 0.0617 |
| Gradient Boosting (Tuned) | 0.0926 | 0.8557 | 0.0613 | 0.2067 | 0.0946 | 0.5915 | 0.0513 |
| Gradient Boosting | 0.0847 | 0.8638 | 0.0618 | 0.1931 | 0.0937 | 0.5908 | 0.0515 |
| XGBoost (Tuned) | 0.1738 | 0.7921 | 0.0511 | 0.2680 | 0.0859 | 0.5683 | 0.0466 |
| Random Forest | 0.3311 | 0.6444 | 0.0459 | 0.4427 | 0.0832 | 0.5675 | 0.0450 |
| Random Forest (Tuned) | 0.3410 | 0.6632 | 0.0458 | 0.4155 | 0.0825 | 0.5656 | 0.0448 |

### Key Takeaways
* **The Baseline Illusion:** Under default settings (0.5 threshold), powerful boosting ensembles outputted an F1-score of `0.0000` because they completely ignored the minority class. Moving the decision boundary to reflect the rare-event prior distribution (~0.07) successfully activated the models.
* **The Champion:** **XGBoost** achieved the highest overall trade-off balance, leading the pack with an optimized **F1-Score of 0.1126** and a solid **PR-AUC of 0.0634**.

**Business Impact:** By shifting from a default boundary to a cost-optimal threshold of **~0.07**, the pipeline dynamically balances false positives and false negatives — minimizing net expected enterprise risk costs on held-out samples.

---

## Key Engineering Decisions

| Challenge | Mitigation |
|---|---|
| **Severe class imbalance (~3.6%)** | SMOTE on the training set only + F1/PR-AUC-driven model selection instead of accuracy |
| **`-1` missing-value sentinel** | Explicitly mapped to `NaN` before any statistical operation |
| **Skewed, outlier-heavy continuous features** | Yeo-Johnson power transform + IQR-based Winsorization (no row deletion) |
| **Untrustworthy raw probabilities** | Isotonic regression calibration, validated via reliability diagrams + Brier score |
| **F1-optimal ≠ business-optimal threshold** | Explicit cost matrix (FN/FP/TP cost-benefit) drives the deployed decision threshold |
| **"Is the best model actually better?"** | Bootstrap 95% confidence intervals + McNemar's paired significance test |
| **Black-box predictions** | SHAP global summary plots + per-customer local force plots |

---

## How to Run

### 1. Clone and set up the environment
```bash
git clone https://github.com/puravi-predicts/InsureIQ-Claim-Risk-Prediction.git
cd InsureIQ-Claim-Risk-Prediction
pip install -r requirements.txt
```

### 2. Add the dataset
Place `train.csv` in the path expected by `notebooks/InsClaimPred-Pipeline.ipynb`.

### 3. Run the end-to-end pipeline
Open and run **`InsClaimPred-Pipeline.ipynb`** top to bottom. This single notebook executes all sequential phases of the project: complete EDA, preprocessing, training the 9 benchmarked models, hyperparameter tuning, stacked ensembling, SHAP explainability calculations, and the serialization of production artifacts into `model_artifacts/`.

### 4. Serve the model (optional)
```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"ps_ind_01": 2, "ps_car_13": 0.64, "ps_reg_03": 0.71}'
```

---

## Tech Stack

**Core:** Python, pandas, NumPy, scikit-learn
**Boosting:** XGBoost, LightGBM
**Deep Learning:** TensorFlow / Keras
**Imbalance Handling:** imbalanced-learn (SMOTE)
**Explainability:** SHAP
**Deployment:** FastAPI, joblib
**Visualization:** Matplotlib, Seaborn

---

## Future Work

- Engineered interaction features between top SHAP-ranked predictors.
- Full grid search (vs. randomized) on the winning model family at full dataset scale with distributed compute.
- Integration with an MLflow tracking server for richer experiment versioning beyond the current CSV/JSON log.
- A monitoring layer for prediction drift once deployed (e.g., population stability index on incoming feature distributions).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Puravi Pradhan**

<p align="left">
  <a href="https://www.linkedin.com/in/purvi-pradhan-593465383"><strong>LinkedIn</strong></a> · 
  <a href="https://github.com/puravi-predicts"><strong>GitHub Portfolio</strong></a> · 
  <a href="mailto:puravipradhan15@gmail.com"><strong>Email</strong></a>
</p>

If you found this project useful or interesting, consider starring the repo ⭐
