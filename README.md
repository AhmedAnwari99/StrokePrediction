**Heart Disease Prediction Using Random Forest Models**

This project applies Random Forest Regressor and Random Forest Classifier models to predict heart disease using the Cleveland Heart Disease Dataset. The primary goal was to assess the performance of these models under various parameter settings and compare their effectiveness using different evaluation metrics.

**Dataset**

The Cleveland Heart Disease Dataset contains 303 patient records with 14 attributes. This dataset is widely used in cardiovascular disease prediction research.

Target variable:

0: No heart disease
1-4: Different severity levels of heart disease (simplified to binary classification for the classifier).

**Objectives**
Use Random Forest Regressor to predict the target values (severity levels of heart disease).
Use Random Forest Classifier to classify whether heart disease is present.
Compare models based on metrics such as:
For Regression: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
For Classification: Accuracy, Precision, Recall, F1 Score, and ROC-AUC Score.
Identify optimal hyperparameters for both models using grid search.

**Dependencies**

Python 3.8+
scikit-learn
pandas
numpy
matplotlib
seaborn

**Methodology**

Data Preprocessing:

Handled missing values.
Encoded categorical variables.
Normalized numerical features.
Split data into training and testing sets.
Modeling:

Used Random Forest Regressor to predict heart disease severity scores.
Used Random Forest Classifier for binary classification of heart disease presence.
Experimented with hyperparameter tuning (e.g., number of estimators, max depth, and min samples split).
Evaluation:

Evaluated model performance using appropriate metrics.
Visualized feature importance to understand which features contribute most to predictions.



**Key Insights**
Random Forest Classifier achieved better results in binary classification tasks, indicating its suitability for predicting the presence of heart disease.
Feature importance analysis revealed that attributes such as thalach (maximum heart rate) and cp (chest pain type) were among the most predictive.

**Conclusion**
This project highlights the effectiveness of Random Forest models in predicting and classifying heart disease. Future work may involve:
Testing other advanced algorithms like Gradient Boosting or XGBoost.

Applying feature engineering techniques for improved accuracy.

Evaluating models on larger datasets.


