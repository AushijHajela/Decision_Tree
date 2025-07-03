***Car Evaluation using Decision Tree***
This project focuses on predicting the quality of car evaluations using a Decision Tree Classifier, based on categorical input features like buying price,
maintenance cost, safety, etc. The project applies data preprocessing, handling class imbalance (SMOTE), hyperparameter tuning (GridSearchCV),
and model evaluation techniques.

***Dataset***
Source: UCI Machine Learning Repository - Car Evaluation

***Attributes:***
buying, maint, doors, persons, lug_boot, safety
Target: class — (unacc, acc, good, vgood)

***Project Workflow***
Data Preprocessing
Loaded CSV without headers
Renamed columns
Label encoded all categorical columns
Train-Test Split
Used train_test_split with stratify to preserve class distribution
Class Imbalance Handling
Applied SMOTE to oversample minority classes
Model Training
Used DecisionTreeClassifier with class_weight='balanced'
Tuned hyperparameters (max_depth, min_samples_split, etc.) using GridSearchCV

Model Evaluation
Accuracy: 0.97
ROC AUC: 0.89

Plotted:
Confusion Matrix
ROC Curve
Decision Tree Visualization
Feature Importance

***Results***
Metric	Value
Accuracy	0.97
AUC Score	0.89
Best Parameters	Found via GridSearchCV
Classes	unacc, acc, good, vgood

***Classification Report:***
              precision    recall  f1-score   support

           0       0.89      0.96      0.93        77
           1       1.00      0.79      0.88        14
           2       0.99      0.98      0.99       242
           3       1.00      0.85      0.92        13

    accuracy                           0.97       346
   macro avg       0.97      0.89      0.93       346
weighted avg       0.97      0.97      0.97       346

***Most important feature: safety***

***unacc was the most predictable class***

***Minimal overfitting observed***

***Tech Stack***
Python
scikit-learn
imbalanced-learn
pandas, matplotlib, seaborn

