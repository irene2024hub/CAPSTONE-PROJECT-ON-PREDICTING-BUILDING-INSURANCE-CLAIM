# CAPSTONE-PROJECT-ON-PREDICTING-BUILDING-INSURANCE-CLAIM
## PROJECT OVERVIEW: Predictive model for identifying buildings likely to file insurance claims during coverage. Uses historical data, 
SMOTE for class imbalance, and XGBoost with hyperparameter tuning. Helps Olusola Insurance improve underwriting, assess risk, and reduce 
losses through data-driven decisions

## Insurance Building Claim Prediction
This capstone project focuses on building a predictive model to estimate the likelihood of a building filing at least one insurance 
claim during its insured period. Accurate risk assessment is critical in the insurance industry to ensure profitability, reduce potential 
losses, and enhance customer satisfaction.

##  Objective
To support Olusola Insurance in making data-driven underwriting decisions by developing a machine learning pipeline that predicts claim probability 
based on building characteristics and historical claim data.

##  Role
- Lead Data Analyst
- Responsible for end-to-end pipeline development including data preprocessing, EDA, feature engineering, model training, evaluation,
  and deployment preparation.

 ## Methodology
 
- Data Preprocessing: Handling missing values, encoding categorical variables, and scaling.
- Exploratory Data Analysis (EDA): Visualizing distributions, correlations, and class imbalance.
- Feature Engineering: Creating new features to enhance model performance.
- Class Imbalance Handling: Applied SMOTE to address ~3.5:1 imbalance (non-claims to claims).
- Modeling: Trained and compared multiple models including Random Forest and XGBoost.
- Hyperparameter Tuning: Used RandomizedSearchCV for optimal performance.
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC.
  
##  Results

- The final model — an **XGBoost classifier enhanced with SMOTE** and optimized through hyperparameter tuning — demonstrated strong predictive performance. 
Notably, it significantly improved recall and F1-score for the minority class (claims), making it a valuable tool for proactive risk assessment and 
more informed underwriting decisions at Olusola Insurance.

 Project Structure
IRENEDSMLCAPTONEPROJECT
├── Train_data.csv
├── Variable Description.csv
├── xgb_smote_pipeline.ipynb
├── irenecapstoneprojectdsml.ipynb
├── irenecapstoneprojectdsml.html
└── README.md


# Acknowledgment
This project was completed as part of the Data Science & Machine Learning Capstone under a scholarship awarded by AINOW, empowered by The Incubator Hub.
Special thanks to instructors Mr. Isaac Odeajo and Mr. Ezekiel Adebayo Ogundepo for their invaluable guidance and support.
📅 Date
January 15, 2026
