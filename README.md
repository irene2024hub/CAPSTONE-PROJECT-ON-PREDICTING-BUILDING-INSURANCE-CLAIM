<div align="center">

# 🏢 PREDICTING BUILDING INSURANCE CLAIMS
### for Olusola Insurance

**Flagging high-risk buildings before a claim is ever filed — turning years of policy history into a proactive underwriting weapon.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-2E8B57?style=for-the-badge)
![SMOTE](https://img.shields.io/badge/Imbalance-SMOTE-FF6B35?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-00C853?style=for-the-badge)
![Recall](https://img.shields.io/badge/Recall-43%25-critical?style=for-the-badge)
![AUC](https://img.shields.io/badge/AUC--ROC-0.71-blueviolet?style=for-the-badge)

</div>

<br>

> ### ⚡ The Bottom Line
> A tuned **XGBoost + SMOTE** pipeline more than **doubled claim-detection recall** versus the baseline model — from 11% to 43% — giving Olusola Insurance an early-warning signal on risk that didn't exist before.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Project Objectives](#-project-objectives)
4. [Business Questions](#-business-questions)
5. [Dataset Description](#-dataset-description)
6. [Technology Stack](#-technology-stack)
7. [Project Workflow](#-project-workflow)
8. [Exploratory Data Analysis](#-exploratory-data-analysis)
9. [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
10. [Handling Class Imbalance](#-handling-class-imbalance)
11. [Machine Learning Methodology](#-machine-learning-methodology)
12. [Model Comparison](#-model-comparison)
13. [Final Model Evaluation](#-final-model-evaluation)
14. [Feature Importance](#-feature-importance)
15. [Threshold Tuning](#-threshold-tuning)
16. [Key Insights](#-key-insights)
17. [Business Impact & Recommendations](#-business-impact--recommendations)
18. [Limitations](#-limitations)
19. [Future Improvements](#-future-improvements)
20. [Model Deployment](#-model-deployment)
21. [Key Skills Demonstrated](#-key-skills-demonstrated)
22. [Author](#-author)
23. [Acknowledgements](#-acknowledgements)

---

## 🎯 Project Overview

Insurance claims are unpredictable by nature — but the buildings behind them are not. Roof condition, occupancy status, location, and building type all leave a data trail that, if read correctly, can tell an insurer where risk is concentrated long before a claim is filed.

This project builds an end-to-end machine learning pipeline that predicts the probability of a building filing **at least one insurance claim** during its coverage period, using historical policy and building-characteristic data. The final model is packaged as a deployable pipeline ready for integration into an underwriting workflow.

## 💼 Business Problem

Olusola Insurance underwrites building policies without a systematic, data-driven way to estimate a building's claim risk at the point of sale. This creates two costly problems:

- **Under-pricing high-risk buildings**, which erodes profitability when claims materialize.
- **Over-pricing low-risk buildings**, which drives away good customers to competitors.

Without a predictive risk signal, underwriting decisions rely on manual judgment and static rules rather than patterns learned from thousands of historical policies.

## 🎯 Project Objectives

- Build a classification model that predicts claim likelihood from building and policy attributes.
- Address the natural class imbalance in claims data (most buildings never claim) without sacrificing the ability to catch true risk.
- Compare multiple modeling strategies and select the one that best balances precision and recall for the minority (claim) class.
- Package the winning model into a reusable pipeline for production deployment.

## ❓ Business Questions

- Which building characteristics are most predictive of a claim?
- How well can we distinguish claim-prone buildings from safe ones (AUC-ROC)?
- What trade-off between false alarms and missed claims best serves an early-warning underwriting process?
- Can model output support a tiered, risk-based pricing or inspection strategy?

## 📊 Dataset Description

The dataset contains **7,160 building records** with the following fields:

| Feature | Description |
|---|---|
| `YearOfObservation` | Year the building was observed (2012–2016) |
| `Insured_Period` | Proportion of the year the building was insured |
| `Residential` | Whether the building is residential (1) or not (0) |
| `Building_Dimension` | Building size |
| `Building_Type` | Categorical building type (1–4) |
| `Date_of_Occupancy` | Year the building was first occupied |
| `Building_Painted` | Whether the building is painted |
| `Building_Fenced` | Whether the building is fenced |
| `Garden` | Whether the building has a garden |
| `Settlement` | Urban or Rural |
| `NumberOfWindows` | Number of windows |
| `Claim` | Target variable — 1 if a claim was filed, 0 otherwise |

**Class balance:** 77.2% no-claim vs. 22.8% claim — a moderate imbalance that shapes the entire modeling strategy.

**Settlement split:** buildings are almost evenly split between Urban (50.4%) and Rural (49.6%) locations.

## 🛠 Technology Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | Scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Hyperparameter Tuning | GridSearchCV, RandomizedSearchCV |
| Deployment | Pickle (`.pkl` pipeline) |

## 🔄 Project Workflow

```
Data Collection → Cleaning & Preprocessing → EDA → Feature Engineering
      → Class Imbalance Handling (SMOTE) → Model Training & Tuning
      → Evaluation → Threshold Optimization → Deployment Packaging
```

## 🔍 Exploratory Data Analysis

EDA focused on understanding both the shape of individual features and how they relate to claim outcomes.

**Feature distributions** across the dataset showed that most buildings are small-to-mid-sized, insured for the full year, and concentrated in occupancy years from the mid-1900s onward, with claims making up a clear minority of records.

**Claim vs. no-claim comparisons** (via box plots) showed that buildings which went on to file a claim tended to have **larger building dimensions** and a higher median building type, suggesting size and structure type carry real signal for risk.

*(Suggested image placement: `assets/feature_distributions.png`, `assets/boxplots_by_claim.png`, `assets/claim_distribution.png`, `assets/settlement_distribution.png`)*

## 🧹 Data Cleaning & Preprocessing

- Converted boolean columns to integer type for model compatibility.
- Dropped raw datetime columns (e.g., `Date_of_Occupancy`) after deriving usable numeric features.
- One-hot encoded categorical variables (e.g., `Settlement`).
- Verified and handled missing values prior to modeling.

## ⚖️ Handling Class Imbalance

With claims representing less than a quarter of records, a model trained naively would default to predicting "no claim" almost every time — technically accurate, but useless for risk detection. **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data to balance the classes, giving the model enough signal from the minority class to learn meaningful claim patterns.

## 🤖 Machine Learning Methodology

Six modeling strategies were tested to find the best balance between overall accuracy and — critically — the ability to catch actual claims (recall on Class 1):

1. Random Forest (baseline)
2. Random Forest + SMOTE
3. XGBoost + SMOTE
4. XGBoost + SMOTE + GridSearchCV
5. Random Forest with Class Weighting
6. **XGBoost + SMOTE + RandomizedSearchCV** *(final selected model)*

Each model was evaluated with a full classification report, confusion matrix, and AUC-ROC curve, with particular attention to **recall and F1-score for Class 1 (claims)** — since missing a true claim carries far more business cost than a false alarm.

## 📈 Model Comparison

| Model | Recall (Class 1) | F1-Score (Class 1) | Accuracy |
|---|:---:|:---:|:---:|
| Random Forest (baseline) | 11% | 0.19 | 78.4% |
| Random Forest + SMOTE | 20% | 0.30 | 78.1% |
| XGBoost + SMOTE | 24% | 0.32 | 75.8% |
| XGBoost + SMOTE + GridSearchCV | 29% | 0.37 | 76.3% |
| Random Forest with Class Weighting | 30% | 0.34 | 72.3% |
| 🏆 **XGBoost + SMOTE + RandomizedSearchCV** | **43%** | **0.44** | 74.3% |

> ⚠️ **Why not the highest-accuracy model?** Accuracy is misleading on imbalanced data — a model that never predicts a claim can still score ~77% accuracy while catching zero real risk. The selected model trades a modest amount of overall accuracy for **more than double the recall** of the baseline, which is what actually matters for catching risk.

## ✅ Final Model Evaluation

<div align="center">

### 🏆 XGBoost + SMOTE + RandomizedSearchCV
**The model that ships.**

</div>

Best hyperparameters found:
```
subsample: 1.0
n_estimators: 100
max_depth: 9
learning_rate: 0.01
colsample_bytree: 0.8
```

**Confusion Matrix**

| | Predicted: No Claim | Predicted: Claim |
|---|---|---|
| **Actual: No Claim** | 905 (TN) | 168 (FP) |
| **Actual: Claim** | 194 (FN) | 144 (TP) |

**Classification Report**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 (No Claim) | 0.82 | 0.84 | 0.83 |
| 1 (Claim) | 0.46 | 0.43 | 0.44 |

- **AUC-ROC ≈ 0.71** — the model shows solid separation between claim and non-claim buildings, well above the random-guess baseline.
- The model correctly rules out the majority of safe buildings while catching nearly half of actual claims — a strong improvement over every baseline tested.

*(Suggested image: `assets/confusion_matrix_roc.png`)*

## 🌟 Feature Importance

Two views were generated — one from the Random Forest baseline, one from the final XGBoost model — and they highlight different but complementary risk signals:

**Random Forest** ranked **Building Dimension**, **Insured Period**, and **Number of Windows** as the top predictors — larger, longer-insured buildings carry more physical exposure.

**XGBoost (final model)** ranked **Settlement**, **Garden**, and **Number of Windows** highest — suggesting location type and property amenities are strong differentiators once the model has been tuned and balanced.

*(Suggested images: `assets/rf_feature_importance.png`, `assets/xgb_feature_importance.png`)*

## 🎚 Threshold Tuning

By default, classifiers use a 0.5 probability cutoff to decide "claim" vs. "no claim." Lowering that threshold to **0.3** dramatically shifts the model's behavior:

| Metric | Value |
|---|---|
| Recall (Class 1) | 96% |
| Precision (Class 1) | 26% |
| Accuracy | 33% |

This configuration is not meant for standalone decision-making — it's designed for an **early-warning / triage use case**, where the cost of missing a real claim outweighs the cost of flagging extra buildings for manual review.

## 💡 Key Insights

- Claims are a minority event (22.8%), and class imbalance handling was essential to building a usable model.
- Building size and structural type are consistently associated with claim likelihood.
- Settlement type (urban vs. rural) emerged as the single strongest predictor in the tuned XGBoost model.
- There is a clear precision–recall trade-off: the standard-threshold model is balanced for general use, while the lowered-threshold version is built for maximum sensitivity in triage scenarios.

## 📌 Business Impact & Recommendations

| 💰 Lever | What It Does |
|---|---|
| **Risk-based underwriting** | Use claim probability scores to inform premium pricing at the point of sale, instead of static rules. |
| **Proactive inspection targeting** | Route high-probability buildings flagged at the 0.3 threshold to inspection or risk-mitigation teams *before* issues arise. |
| **Portfolio monitoring** | Periodically re-score the existing book of business to catch policies whose risk profile has shifted. |
| **Tiered decisioning** | Pair the balanced model with the high-sensitivity "early warning" version depending on whether the use case favors caution or coverage. |

## ⚠️ Limitations

- Precision on the claim class remains moderate (46%), meaning a portion of flagged buildings will not actually file a claim.
- The dataset spans a limited historical window (2012–2016), which may not capture longer-term risk trends.
- Some fields (e.g., historical occupancy dates) required simplification during preprocessing, which may have removed nuance.

## 🚀 Future Improvements

- Incorporate external data sources (e.g., regional weather/flood risk, crime statistics) to enrich building risk signals.
- Experiment with alternative imbalance techniques (ADASYN, cost-sensitive learning) alongside SMOTE.
- Build a monitoring pipeline to track model drift as new policy data arrives.
- Deploy the model behind an API for real-time scoring at the point of underwriting.

## 📦 Model Deployment

The final tuned pipeline is saved as **`xgb_smote_pipeline.pkl`**, bundling:

- SMOTE resampling logic
- The tuned XGBoost classifier
- Best parameters identified via RandomizedSearchCV

This pipeline is ready for integration into a production scoring service or API, taking raw building/policy features as input and returning a claim-probability score.

## 🧠 Key Skills Demonstrated

- End-to-end ML pipeline design (preprocessing → modeling → evaluation → deployment)
- Handling class imbalance with SMOTE
- Model comparison and hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Threshold optimization for business-specific risk tolerance
- Translating model output into underwriting-relevant business recommendations

## 👤 Author

*(Add your full name and links — e.g., LinkedIn, GitHub, portfolio site)*

## 🙏 Acknowledgements

Sincere gratitude to **AINOW** and **The Incubator Hub** for the scholarship and platform that made this capstone possible, and to course instructors **Isaac Odeajo** and **Ezekiel Adebayo Ogundepo** for their mentorship and guidance throughout the project.

---

*(Insert dashboard/demo screenshot here, if available)*
