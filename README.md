# 📊 Bank Customer Churn Prediction

## 📌 Project Overview
This project focuses on predicting customer churn in a banking system using machine learning techniques.  
Customer churn refers to customers who stop using the bank’s services. Identifying such customers early helps banks take preventive actions and improve customer retention.

The project follows a complete data science pipeline, including data cleaning, exploratory data analysis, feature engineering, model development, evaluation, and interpretability.

## 🎯 Objectives
- Analyze customer demographic and behavioral data  
- Identify key factors contributing to customer churn  
- Build and compare multiple machine learning models  
- Select the best-performing model using appropriate evaluation metrics  
- Provide business insights and recommendations based on model results  # Customer Churn Analytics

**A machine learning application for understanding customer churn, evaluating prediction models, and identifying customers who may be at risk of leaving a bank.**

Customer churn is an important challenge in retail banking. Understanding customer behaviour and identifying potential churners can help banks make better-informed decisions about customer retention.

This project explores that problem using a structured machine learning pipeline and an interactive Streamlit dashboard. It brings together data exploration, feature engineering, model evaluation, churn risk prediction, and customer-level analysis in one application.

The project began as an exploratory Jupyter notebook and was later reorganised into a modular codebase to make model training, testing, and prediction easier to maintain and reproduce.

## Dashboard

After running the application, the dashboard provides several views of the customer portfolio and the trained models.

<!-- Add screenshots after running the application.
![Portfolio Overview](docs/screenshots/overview.png)
![Model Performance](docs/screenshots/model-performance.png)
![Customer Risk Analysis](docs/screenshots/single-customer.png)
-->

## What the project offers

| Section | Description |
|---|---|
| **Portfolio Overview** | Explore customer segments and compare churn rates across geography, age groups, product counts, account balances, and membership status. |
| **Model Performance** | Compare classification models, inspect ROC curves, and adjust the decision threshold to examine changes in precision, recall, and the confusion matrix. |
| **Churn Drivers** | Examine permutation feature importance and customer-level response patterns to understand which attributes are associated with churn predictions. |
| **Single Customer Analysis** | Enter an individual customer profile and obtain a predicted churn risk, together with an explanation of how individual attributes influence the prediction. |
| **Batch Scoring** | Upload a customer CSV file, validate its contents, generate predictions for multiple customers, and download the scored results. |
| **Methodology** | Review the data preparation process, feature engineering, model selection approach, risk categories, and known limitations. |

## Machine learning approach

The project uses a reproducible training pipeline built with scikit-learn. Data preparation, feature engineering, model training, and prediction are organised into separate modules rather than being handled entirely inside a notebook.

### Dataset and preparation

The project uses the public Bank Customer Churn dataset, which contains 10,000 customer records and a churn rate of approximately 20%.

The preparation process includes:

- Removing identifier columns that are not intended to contribute to predictions.
- Handling credit-score outliers using the 1.5 × IQR rule.
- Preparing customer attributes for use in the modelling pipeline.
- Separating training and test data before fitting data-dependent transformations.

### Feature engineering

The pipeline transforms 10 raw customer attributes into 20 model inputs. Engineered features include:

- Balance-to-salary ratio.
- Age and tenure bands.
- A customer engagement score.
- A high-value customer indicator.

Feature engineering is implemented as a scikit-learn transformer. This allows the same transformations to be applied during model training and when new customers are scored.

### Models

Three classification approaches are evaluated:

- Logistic Regression with L2 regularisation.
- Logistic Regression with L1 regularisation.
- A decision tree with a limited depth.

The default model is selected using five-fold cross-validation on the training data, with ROC-AUC as the selection metric. The held-out test set is reserved for final evaluation.

### Model interpretation

The dashboard includes permutation feature importance and one-attribute-at-a-time response analysis for individual customer profiles.

These methods help users explore the relationship between customer characteristics and model predictions. They provide insight into model behaviour, but they do not establish that a particular characteristic causes churn.

### Churn risk and decision thresholds

The dashboard allows users to adjust the classification threshold and examine the resulting precision, recall, and confusion matrix.

This is important because a model that achieves reasonable overall accuracy can still miss a substantial proportion of customers who actually churn.

The application also groups predicted probabilities into configurable risk categories. These categories support portfolio exploration and should not be treated as definitive assessments of an individual customer.

## Getting started

### Requirements

- Python 3.10 or newer
- pip
- Git
- The project dataset: `Churn_Modelling.csv`

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd churn-dashboard
```

Replace `<your-repository-url>` with the URL of your GitHub repository.

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place `Churn_Modelling.csv` in the following directory:

```text
data/raw/Churn_Modelling.csv
```

Ensure the file follows the column structure expected by the project.

### 5. Train the models

Run the training pipeline from the project root:

```bash
python -m churn.train
```

The pipeline trains and evaluates the candidate models, selects the default model according to the configured selection rule, and generates the artifacts required by the dashboard.

To specify a dataset at another location:

```bash
python -m churn.train --data path/to/Churn_Modelling.csv
```

To select a particular model as the default:

```bash
python -m churn.train --model "Decision Tree"
```

### 6. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

### 7. Run the tests

```bash
pytest
```

The automated tests use synthetic data to test the pipeline without requiring the original customer dataset.

## Project structure

```text
churn-dashboard/
├── app.py
├── churn/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── modeling.py
│   ├── evaluate.py
│   ├── train.py
│   └── predict.py
├── dashboard/
│   ├── theme.py
│   ├── components.py
│   ├── state.py
│   └── views/
├── data/
│   └── raw/
├── notebooks/
│   └── ChurnModelling.ipynb
├── tests/
├── artifacts/
├── requirements.txt
└── README.md
```

**Module overview**

- `app.py` — Application entry point and dashboard navigation.
- `churn/config.py` — Configuration, data schema, paths, and risk thresholds.
- `churn/data.py` — Data loading, cleaning, and input validation.
- `churn/features.py` — Reusable feature engineering transformations.
- `churn/modeling.py` — Model definitions and pipeline construction.
- `churn/evaluate.py` — Evaluation metrics and decision-threshold calculations.
- `churn/train.py` — Training workflow and artifact generation.
- `churn/predict.py` — Model loading, customer scoring, and individual prediction analysis.
- `dashboard/` — Streamlit presentation, shared components, and dashboard pages.
- `notebooks/` — Original exploratory analysis.
- `tests/` — Automated tests for the modelling pipeline.
- `artifacts/` — Generated model files and training metadata.

Generated model artifacts are excluded from version control. Run the training pipeline to create the artifacts required by your local application.

## Initial model results

The original exploratory notebook produced the following baseline results at a classification threshold of 0.5.

| Model | ROC-AUC | Recall |
|---|---:|---:|
| Logistic Regression (L2) | 0.788 | 0.31 |
| Logistic Regression (L1) | 0.788 | 0.31 |
| Decision Tree (depth 5) | 0.832 | 0.49 |

These figures are included as historical baseline results from the original notebook, not as the final performance of the refactored pipeline.

The original notebook fitted scaling parameters using the full dataset before splitting and used test-set results during model selection. The refactored pipeline addresses these issues by fitting data-dependent transformations on the training split and selecting the default model through cross-validation.

Consequently, the results from the refactored pipeline may differ from the baseline figures above.

To generate the current results, run:

```bash
python -m churn.train
```

The latest evaluation results are displayed in the Model Performance dashboard and recorded in `artifacts/metadata.json`.

## Design considerations and limitations

This project is intended for learning, experimentation, and demonstrating a complete machine learning workflow. It is not a production-ready banking decision system.

- **Historical data:** The dataset represents a single snapshot, so the project does not evaluate how performance changes over time or under changing customer behaviour.
- **Prediction versus causation:** Feature importance and individual response analysis describe model behaviour; they do not establish causal relationships.
- **Probability calibration:** Decision tree probabilities can be coarse and may not correspond closely to actual churn frequencies. Calibration should be evaluated before probabilities are interpreted as reliable risk estimates.
- **Responsible use:** Gender and geography are included as model inputs. Their use should be reviewed for fairness, legal compliance, and suitability for the intended application.
- **Decision thresholds:** Risk categories and classification thresholds are configurable analytical choices, not universally valid banking standards.
- **Model security:** Generated artifacts use Python serialization. Only load trusted artifacts, and retrain when incompatible library versions or pipeline changes require it.
- **Operational readiness:** Real-world deployment would require additional validation, monitoring, access controls, data protection, and governance.

## Future improvements

Possible extensions include:

- Evaluating additional classification algorithms.
- Adding probability calibration and more detailed model diagnostics.
- Introducing time-based validation when longitudinal customer data becomes available.
- Monitoring prediction quality and changes in customer behaviour.
- Expanding fairness and subgroup performance analysis.
- Integrating the dashboard with an authorised, regularly updated customer data source.

## Technology stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Dashboard:** Streamlit
- **Visualisation:** Plotly
- **Testing:** pytest
- **Exploratory analysis:** Jupyter Notebook

## License

No license has been specified yet. If you plan to make this repository public and allow others to reuse the code, consider adding an appropriate open-source license, such as the MIT License, after confirming that the dataset and other included materials permit your intended use.

---

*Developed as a data science project exploring customer churn prediction, model evaluation, and interactive analytics in retail banking.*

## 🗂 Dataset
- **Records**: 9,985  
- **Target Variable**: `Exited`  
  - 0 = Retained  
  - 1 = Churned  

## 🧹 Data Preparation
- Removed irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
- Outlier handling applied **only to CreditScore** using the IQR method
- Encoded categorical features (`Geography`, `Gender`)
- Standardized numerical features
- Used an **80/20 stratified train-test split**

## 🤖 Models Used
- Logistic Regression  
- Ridge Regression  
- Lasso Regression  
- Decision Tree Classifier  

## 📈 Evaluation Metrics
- Mean Squared Error (MSE) and R² (for comparison)
- Accuracy, Precision, Recall, F1-Score, ROC-AUC  

> Classification metrics were prioritized due to class imbalance and the binary nature of churn prediction.

## 🏆 Best Model
**Decision Tree Classifier**
- Accuracy: **85.38%**
- F1-Score: **0.57**
- ROC-AUC: **0.83**

## 🔍 Key Insights
- Age is the strongest predictor of churn
- Inactive customers are more likely to leave
- Customers from Germany show higher churn rates

## 🛠 Technologies
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
