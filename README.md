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

### **METHODOLOGY**

1. Data Preprocessing
   
- Converted boolean columns to integers
- Dropped datetime columns (e.g., Date_of_Occupancy)
- One-hot encoded categorical variables
- Handled class imbalance using SMOTE
  
2. **Model Experiments**
   
- Random Forest (baseline)
- Random Forest + SMOTE
- XGBoost + SMOTE
- XGBoost + SMOTE + GridSearchCV
- Random Forest with Class Weighting
- **XGBoost + SMOTE + RandomizedSearchCV (best-performing model)**
  
3. **Evaluation Metrics**
4. - Accuracy
- Precision, Recall, F1-score (especially for Class 1)
- Confusion Matrix
- AUC-ROC Curve

 ## MODEL COMPARISION SUMMARY
 
| Model | Recall (Class 1) | F1 Score | 
| Random Forest (baseline) | 11% | 0.19 | 
| RF + SMOTE | 20% | 0.30 | 
| XGBoost + SMOTE | 24% | 0.32 | 
| XGBoost + SMOTE + GridSearchCV | 29% | 0.37 | 
| RF with Class Weighting | 30% | 0.34 | 
| **XGBoost + SMOTE + RandomizedSearchCV | 43% | 0.44 |** 
8### CONCLUSION

**The XGBoost + SMOTE + RandomizedSearchCV pipeline significantly improved claim detection performance,
especially in recall and F1-score for the minority class. This model is well-suited for identifying 
high-risk policies and can serve as a foundation for proactive risk mitigation strategies at Olusola Insurance**





## CONFUSION MATRIX INSIGHT 
- True Negatives (968): Strong performance in identifying non-claims
- True Positives (122): Moderate success in detecting actual claims
- False Negatives (216): Many claims missed — recall still a challenge
- False Positives (105): Some non-claims flagged incorrectly
 The model is reliable at ruling out non-claims but needs improvement in catching all true claims.


 **AUC-ROC Curve Insight**
 
- AUC Score: 0.7354
- The ROC curve shows strong separation from the random baseline
- Indicates good overall discrimination between claim and non-claim classes

 ## THRESHOLD TUNING (Threshold = 0.3)
 
- Purpose: Increase sensitivity to high-risk policies
- Results:
- Recall (Class 1): 96%
- Precision (Class 1): 26%
- Accuracy: 33%
- Insight: Excellent for early warning systems or triage, though with high false positives


 ## Model Deployment
- Final model saved as: xgb_smote_pipeline.pkl
- Includes:
- SMOTE resampling logic
- Tuned XGBoost classifier
- RandomizedSearchCV best parameters
- Ready for integration into production systems or APIs
  
 # CONCLUSION
- The **XGBoost + SMOTE + RandomizedSearchCV pipeline** significantly improved claim detection performance, especially in recall and 
F1-score for the minority class. This model is well-suited for identifying high-risk policies and can serve as a foundation for

## ACKNOWLEDGEMENT 

I would like to express my sincere gratitude to AINOW and The Incubator Hub for awarding me this scholarship and providing the platform to 
grow my skills in data science and machine learning. Special thanks to my course instructors, **. Isaac Odeajo and Mr. Ezekiel Adebayo Ogundepo**,for their expert guidance, mentorship, and unwavering support throughout this capstone journey. Their insights and encouragement were instrumental in the successful completion of this project.


proactive risk mitigation strategies at Olusola Insurance
