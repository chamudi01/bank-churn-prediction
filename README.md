#📊 Bank Customer Churn Prediction

📌 Project Overview
This project focuses on predicting customer churn in a banking system using machine learning techniques. Customer churn refers to customers who stop using the bank’s services. Identifying such customers early helps banks take preventive actions and improve customer retention.

The project follows a complete data science pipeline, including data cleaning, exploratory data analysis, feature engineering, model development, evaluation, and interpretability.

🎯 Objectives
Analyze customer demographic and behavioral data
Identify key factors contributing to customer churn
Build and compare multiple machine learning models
Select the best-performing model using appropriate evaluation metrics
Provide business insights and recommendations based on model results

🗂 Dataset
Records: 9,985
Target Variable: Exited (0 = Retained, 1 = Churned)

🧹 Data Preparation
Removed irrelevant columns (RowNumber, CustomerId, Surname)
Outlier handling applied only to CreditScore using the IQR method
Encoded categorical features (Geography, Gender)
Standardized numerical features
Used an 80/20 stratified train-test split

🤖 Models Used
Logistic Regression
Ridge Regression
Lasso Regression
Decision Tree Classifier

📈 Evaluation Metrics
MSE and R² (for comparison)
Accuracy, Precision, Recall, F1-Score, ROC-AUC
Classification metrics were prioritized due to class imbalance and the binary nature of churn prediction.

🏆 Best Model
Decision Tree Classifier
Accuracy: 85.38%
F1-Score: 0.57
ROC-AUC: 0.83

🔍 Key Insights
Age is the strongest predictor of churn
Inactive customers are more likely to leave
Customers from Germany show higher churn rates

🛠 Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn



