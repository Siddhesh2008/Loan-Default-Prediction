# Loan Default Prediction

## Project Overview

This project uses machine learning to predict whether a loan applicant is likely to default based on financial, employment, and loan-related information.

The project follows an end-to-end machine learning workflow, including data cleaning, exploratory data analysis, preprocessing, model training, evaluation, and feature importance analysis.

The main goal was to understand how different applicant and loan characteristics can be used to predict loan default risk.

---

## Dataset

The dataset contains **32,581 loan application records** with information about applicants and their loans.

### Features

* `person_age` — Applicant's age
* `person_income` — Applicant's income
* `person_home_ownership` — Home ownership status
* `person_emp_length` — Employment length
* `loan_intent` — Purpose of the loan
* `loan_grade` — Loan grade
* `loan_amnt` — Loan amount
* `loan_int_rate` — Loan interest rate
* `loan_percent_income` — Loan amount relative to income
* `cb_person_default_on_file` — Previous default record
* `cb_person_cred_hist_length` — Length of credit history

### Target

* `loan_status`

  * `0` → No default
  * `1` → Default

---

## Data Cleaning

The dataset was inspected for missing values, duplicate records, and invalid values.

The cleaning process included:

* Handling missing values
* Removing duplicate records
* Identifying unrealistic age values
* Identifying invalid employment-length values
* Handling extreme income values
* Checking data types and feature distributions

After cleaning and preprocessing, the dataset was prepared for machine learning.

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand patterns and relationships within the dataset.

Some of the analysis included:

* Distribution of applicant age
* Income distribution
* Loan amount distribution
* Interest-rate distribution
* Default distribution
* Default rate by home ownership
* Default rate by loan intent
* Default rate by loan grade
* Default rate based on previous default history
* Correlation analysis

Some notable patterns observed during EDA included higher default rates among certain loan grades, higher-risk loan intents, and applicants with higher loan-to-income burdens.

---

## Preprocessing

The dataset was divided into training and testing sets using an **80/20 split**.

* Training samples: **25,932**
* Testing samples: **6,484**

Numerical features were standardized using `StandardScaler`.

Categorical features were converted into numerical representations using `OneHotEncoder`.

A `ColumnTransformer` was used to apply the appropriate preprocessing to each feature type.

After preprocessing, the 11 original input features were transformed into **26 model-ready features**.

---

## Models

### 1. Logistic Regression

Logistic Regression was used as the baseline classification model.

**Accuracy:** 86.94%

### 2. Random Forest

A Random Forest classifier was then trained to capture more complex relationships within the data.

The model was configured with:

* 200 decision trees
* `random_state = 42`
* All available CPU cores used during training

Random Forest significantly outperformed the Logistic Regression baseline.

---

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Precision-Recall AUC (PR-AUC)

### Logistic Regression

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          | 86.94% |
| Default Precision |    77% |
| Default Recall    |    57% |
| Default F1-score  |    66% |

### Random Forest

| Metric            |     Result |
| ----------------- | ---------: |
| Accuracy          | **93.46%** |
| Default Precision |    **97%** |
| Default Recall    |    **72%** |
| Default F1-score  |    **83%** |
| PR-AUC            |   **0.88** |

### Random Forest Confusion Matrix

|          | Predicted 0 | Predicted 1 |
| -------- | ----------: | ----------: |
| Actual 0 |        5036 |          30 |
| Actual 1 |         394 |        1024 |

The Random Forest correctly identified **1,024 out of 1,418 actual default cases**, giving a default recall of approximately **72%**.

---

## Feature Importance

Random Forest feature importance was used to identify the most influential individual processed features.

### Top Features

| Rank | Feature               | Importance |
| ---: | --------------------- | ---------: |
|    1 | Loan % of Income      |     22.25% |
|    2 | Person Income         |     13.97% |
|    3 | Loan Interest Rate    |     10.81% |
|    4 | Loan Amount           |      7.41% |
|    5 | Employment Length     |      6.04% |
|    6 | Home Ownership — Rent |      6.00% |
|    7 | Loan Grade — D        |      5.40% |
|    8 | Person Age            |      4.50% |

These values represent the features the Random Forest relied on most when making predictions. Feature importance indicates predictive contribution within the model and should not be interpreted as proof of causation.

---

## Results

Random Forest performed substantially better than the Logistic Regression baseline.

| Model               |   Accuracy | Default F1 |   PR-AUC |
| ------------------- | ---------: | ---------: | -------: |
| Logistic Regression |     86.94% |       0.66 |        — |
| Random Forest       | **93.46%** |   **0.83** | **0.88** |

The results demonstrate that Random Forest was better able to capture the patterns associated with loan defaults in this dataset.

The project also showed that **loan-to-income ratio, income, interest rate, loan amount, and employment length** were among the most influential individual predictive features.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

### Machine Learning Techniques

* Data preprocessing
* Standardization
* One-hot encoding
* Logistic Regression
* Random Forest Classification
* Classification metrics
* Precision-Recall AUC
* Feature importance analysis

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Siddhesh2008/Loan-Default-Prediction
cd Loan-Default-Prediction
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the project notebook

Open the `.ipynb` file and run the cells **from top to bottom**.

Make sure the dataset file is placed in the expected project directory before running the notebook.

---

## Project Outcome

This project provided hands-on experience with the complete machine learning workflow, from raw data cleaning and exploratory analysis to preprocessing, model training, evaluation, and model interpretation.
