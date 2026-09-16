# E-Commerce Conversion Dynamics 🛒

A comprehensive **Machine Learning project for e-commerce conversion analysis** using Python and Scikit-learn to analyze online shopping behavior, engineer session-level features, develop multiple classification models, and build an ensemble-based conversion prediction system.

The project focuses on understanding the factors associated with purchase behavior, predicting whether an online shopping session will result in a purchase, evaluating model performance under class imbalance, and explaining model predictions through feature importance and error analysis.

---

## 🌟 Features

### Core Features

* **Data Understanding & Cleaning:** Inspection of dataset structure, data types, missing values, duplicates, and data quality.
* **Exploratory Data Analysis:** Analysis of browsing behavior, visitor types, traffic sources, page interactions, session duration, and conversion patterns.
* **Feature Engineering:** Creation of session-level browsing, product engagement, duration, and page-share features.
* **Data Preprocessing:** Preparation of numerical and categorical features using Scikit-learn preprocessing pipelines.
* **Class Imbalance Handling:** Application of SMOTE to address the imbalance between converting and non-converting sessions.
* **Multiple Machine Learning Models:** Development and comparison of multiple classification models.
* **Ensemble Learning:** Combination of multiple classifiers using ensemble techniques.
* **Model Evaluation:** Evaluation using classification metrics, confusion matrices, and prediction analysis.
* **Error Analysis:** Investigation of incorrect predictions and model behavior across different customer-session characteristics.
* **Model Explainability:** Analysis of feature importance and investigation of important factors influencing conversion predictions.
* **Business Insights:** Translation of model findings into practical e-commerce conversion insights.

---

## 🗂 Project Structure

The project is organized as follows:

```text
E-Commerce_Conversion_Dynamics/
│
├── data/
│   ├── raw/
│   │   └── online_shoppers_intention.csv   # Original dataset
│   │
│   └── processed/                          # Processed and feature-engineered data
│
├── notebooks/
│   ├── 01_data_understanding.ipynb         # Dataset inspection and understanding
│   ├── 02_eda.ipynb                        # Exploratory Data Analysis
│   ├── 03_data_preparation.ipynb           # Data cleaning and preprocessing
│   ├── 04_feature_engineering.ipynb        # Feature creation
│   ├── 05_model_development.ipynb          # Model development and tuning
│   ├── 06_model_evaluation.ipynb           # Model evaluation and error analysis
│   └── 07_model_explainability.ipynb       # Model explainability
│
├── models/                                 # Saved trained models
│
├── reports/
│   └── figures/                            # Project visualizations
│
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
└── .gitignore                              # Git ignored files
```

---

## 🏗️ Project Architecture

```text
+-----------------------------+
|         DATA LAYER          |
|                             |
|  Online Shopper Sessions    |
|  online_shoppers_intention  |
|  .csv                       |
+--------------+--------------+
               ↓
+-----------------------------+
|    DATA UNDERSTANDING       |
|                             |
|  - Dataset inspection       |
|  - Data types               |
|  - Missing values           |
|  - Duplicate analysis       |
|  - Data quality checks      |
+--------------+--------------+
               ↓
+-----------------------------+
|          EDA LAYER          |
|                             |
|  - Session behavior         |
|  - Page interactions        |
|  - Visitor analysis         |
|  - Traffic analysis         |
|  - Conversion patterns      |
|  - Numerical relationships  |
+--------------+--------------+
               ↓
+-----------------------------+
|     PREPROCESSING LAYER     |
|                             |
|  - Duplicate removal        |
|  - Feature preparation      |
|  - Categorical encoding     |
|  - Numerical preparation    |
|  - Train-test preparation   |
+--------------+--------------+
               ↓
+-----------------------------+
|    FEATURE ENGINEERING      |
|                             |
|  - Total pages viewed       |
|  - Browsing duration        |
|  - Product page share       |
|  - Average page duration    |
|  - Product duration share   |
+--------------+--------------+
               ↓
+-----------------------------+
|       CLASSIFICATION        |
|                             |
|  - Multiple classifiers     |
|  - Hyperparameter tuning    |
|  - SMOTE                    |
|  - Ensemble learning       |
+--------------+--------------+
               ↓
+-----------------------------+
|      EVALUATION LAYER       |
|                             |
|  - Classification metrics  |
|  - Confusion matrix         |
|  - Prediction analysis      |
|  - Error analysis           |
+--------------+--------------+
               ↓
+-----------------------------+
|    EXPLAINABILITY LAYER     |
|                             |
|  - Feature importance       |
|  - Feature analysis         |
|  - Prediction investigation |
+--------------+--------------+
               ↓
+-----------------------------+
|      BUSINESS INSIGHTS      |
|                             |
|  Conversion drivers         |
|  Customer behavior          |
|  Session-level insights     |
|  Model limitations          |
+-----------------------------+
```

---

## 🛠 Tech Stack

* **Language:** Python 3.8+
* **Environment:** Jupyter Notebook
* **Libraries:**
  * **Data Manipulation:** Pandas, NumPy
  * **Visualization:** Matplotlib, Seaborn, Plotly
  * **Machine Learning:** Scikit-learn
  * **Imbalanced Learning:** imbalanced-learn
  * **Model Persistence:** Joblib
  * **Experiment Tracking:** MLflow
* **Version Control:** Git & GitHub

---

## 📊 Dataset Overview

The project uses the **Online Shoppers Purchasing Intention Dataset**, which contains session-level information about visitors browsing an e-commerce website.

The original dataset contains **12,330 observations and 18 features**. During data preparation, **125 duplicate observations** are identified and removed, resulting in **12,205 unique sessions** used for modeling.

### Dataset Features

```text
Administrative
Administrative_Duration
Informational
Informational_Duration
ProductRelated
ProductRelated_Duration
BounceRates
ExitRates
PageValues
SpecialDay
Month
OperatingSystems
Browser
Region
TrafficType
VisitorType
Weekend
Revenue
```

### Feature Categories

#### Page Interaction Features

* `Administrative`
* `Administrative_Duration`
* `Informational`
* `Informational_Duration`
* `ProductRelated`
* `ProductRelated_Duration`

These variables describe the number of pages visited and the time spent across different types of pages during a shopping session.

#### Session Behavior Features

* `BounceRates`
* `ExitRates`
* `PageValues`
* `SpecialDay`

These variables provide information about session behavior, page exits, and the relationship of the session to special shopping periods.

#### Visitor & Traffic Features

* `VisitorType`
* `TrafficType`
* `Region`
* `OperatingSystems`
* `Browser`

These variables describe visitor characteristics, traffic sources, geographic region, and technical browsing information.

#### Temporal Features

* `Month`
* `Weekend`

These variables capture the timing and calendar context of each shopping session.

---

## 🎯 Target Variable

**Revenue**

The target variable indicates whether an online shopping session resulted in a purchase.

```text
False → No purchase
True  → Purchase
```

The target is therefore treated as a **binary classification problem**.

---

## 🎯 Problem Statement

E-commerce businesses generate large amounts of behavioral data during customer browsing sessions. Understanding which sessions are more likely to result in a purchase can help businesses identify conversion patterns and improve the online shopping experience.

The objective of this project is to develop a machine learning classification system that predicts whether an online shopping session will result in a purchase using session behavior, page interactions, visitor characteristics, traffic information, and other available variables.

---

## 🚀 Project Objectives

* Understand the behavioral characteristics of online shopping sessions.
* Analyze differences between purchasing and non-purchasing sessions.
* Identify relationships between browsing activity and conversion.
* Engineer meaningful session-level behavioral features.
* Prepare numerical and categorical variables for machine learning.
* Address class imbalance using SMOTE.
* Develop and compare multiple classification models.
* Apply ensemble learning to combine model predictions.
* Evaluate model performance using appropriate classification metrics.
* Investigate incorrect predictions and model errors.
* Identify the features that contribute most to conversion predictions.
* Translate model findings into practical e-commerce insights.

---

## 🔍 Exploratory Data Analysis

The EDA stage investigates the underlying behavior of online shopping sessions and examines how session characteristics differ between converting and non-converting visitors.

### Analysis Areas

* Target distribution
* Page interaction patterns
* Session duration
* Product-related browsing behavior
* Administrative and informational activity
* Bounce and exit rates
* Page value
* Visitor type
* Traffic type
* Monthly conversion patterns
* Weekend vs weekday behavior
* Special-day activity
* Relationships between numerical variables
* Conversion patterns across categorical variables

The EDA stage is used to identify behavioral patterns and relationships that guide subsequent preprocessing and feature engineering decisions.

---

## ⚙️ Feature Engineering

Additional session-level features are created to capture broader measures of browsing activity and product engagement.

### Engineered Features

* `Total_Pages_Viewed`
* `Total_Browsing_Duration`
* `Product_Page_Share`
* `Avg_Duration_Per_Page`
* `Product_Duration_Share`

### Feature Descriptions

**Total_Pages_Viewed**

Combines administrative, informational, and product-related page counts to represent overall browsing activity.

**Total_Browsing_Duration**

Combines the duration spent across administrative, informational, and product-related pages.

**Product_Page_Share**

Measures the proportion of total pages viewed that are product-related.

**Avg_Duration_Per_Page**

Represents the average browsing duration relative to the total number of pages viewed.

**Product_Duration_Share**

Measures the proportion of total browsing duration spent on product-related pages.

These features are designed to provide a broader representation of session engagement than the original variables individually.

---

## ⚖️ Class Imbalance

The target variable is imbalanced, with non-purchasing sessions representing the majority of observations.

The prepared training data contains approximately:

```text
False → 84.37%
True  → 15.63%
```

A similar distribution is maintained in the test set:

```text
False → 84.35%
True  → 15.65%
```

Because the minority class represents purchasing sessions, simply predicting the majority class can produce misleadingly high accuracy.

To address this imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training data.

The test data remains untouched so that final model performance is evaluated on the original class distribution.

---

## 🤖 Machine Learning Approach

The project treats conversion prediction as a **supervised binary classification problem**.

The modeling workflow includes:

```text
Data Preparation
       ↓
Feature Engineering
       ↓
Train-Test Split
       ↓
Preprocessing Pipeline
       ↓
SMOTE
       ↓
Multiple Classification Models
       ↓
Hyperparameter Tuning
       ↓
Model Comparison
       ↓
Ensemble Learning
       ↓
Final Evaluation
```

Numerical and categorical features are handled through the preprocessing pipeline.

Categorical variables are encoded before being provided to the machine learning models, while numerical variables are prepared according to the requirements of the selected models.

SMOTE is applied to the training data to improve the representation of purchasing sessions during model training.

---

## 📈 Model Evaluation

Model performance is evaluated using multiple classification metrics rather than relying only on accuracy.

### Evaluation Metrics

**Accuracy**

Measures the proportion of correctly classified sessions.

**Precision**

Measures how many sessions predicted as purchases actually result in purchases.

**Recall**

Measures how many actual purchasing sessions are correctly identified.

**F1-Score**

Provides a combined measure of precision and recall.

**ROC-AUC**

Measures the model's ability to distinguish between purchasing and non-purchasing sessions across classification thresholds.

### Evaluation Analysis

In addition to numerical metrics, the project includes:

* Confusion matrix analysis
* Actual vs predicted class analysis
* False-positive investigation
* False-negative investigation
* Class-level performance analysis
* Prediction error analysis

> **Final model metrics will be reported here after the final evaluation results are verified against the completed evaluation pipeline.**

---

## 🔎 Model Explainability

Model explainability is used to understand which session characteristics contribute most to conversion predictions.

The explainability stage focuses on:

* Feature importance analysis
* Comparison of important behavioral features
* Analysis of model-driving variables
* Investigation of individual prediction errors
* Examination of incorrectly classified purchasing sessions

The analysis helps connect model predictions with observable customer-session behavior rather than treating the classifier as a black box.

---

## 🧪 Baseline Model

A simple baseline is established before evaluating machine learning models.

The baseline predicts the **training-set majority class** for every test observation.

This provides a reference point for determining whether the trained classification models provide meaningful predictive improvement beyond a naive majority-class strategy.

---

## 📊 Data Preparation Summary

The data preparation stage includes the following steps:

```text
Original Dataset
12,330 rows × 18 columns
          ↓
Duplicate Detection
125 duplicate rows
          ↓
Duplicate Removal
12,205 rows × 18 columns
          ↓
Data Type Preparation
Categorical / Boolean / Numerical
          ↓
Feature Engineering
Additional session-level features
          ↓
Missing Value Handling
Final modeling dataset
          ↓
Train-Test Split
9,764 training observations
2,441 test observations
          ↓
SMOTE on Training Data
Balanced training representation
```

After preparation, the final dataset contains no missing values and is ready for model development.

---

## 🧪 Experiment Tracking

**MLflow** is used to track machine learning experiments and maintain a record of model development.

Tracked information includes:

* Model name
* Model parameters
* SMOTE configuration
* Random state
* Model performance
* Notebook source

The project uses an MLflow database backend for experiment tracking.

The MLflow experiment is named:

```text
E-commerce Conversion Dynamics
```

This allows different classification models and configurations to be compared systematically during model development.

---

## 📓 Notebook Workflow

The project is divided into seven notebooks to maintain a structured and reproducible workflow.

| Notebook | Purpose |
|---|---|
| `01_data_understanding.ipynb` | Dataset inspection and initial understanding |
| `02_eda.ipynb` | Exploratory Data Analysis and visualization |
| `03_data_preparation.ipynb` | Data cleaning, validation, and preprocessing |
| `04_feature_engineering.ipynb` | Creation of session-level behavioral features |
| `05_model_development.ipynb` | Model development, comparison, tuning, and MLflow tracking |
| `06_model_evaluation.ipynb` | Model performance and prediction error analysis |
| `07_model_explainability.ipynb` | Feature importance and prediction explainability |

The notebooks are intended to be executed in numerical order.

---

## 📁 Data Organization

The project maintains a separation between raw and processed data.

### Raw Data

```text
data/raw/online_shoppers_intention.csv
```

The raw dataset remains unchanged and represents the original source data used by the project.

### Processed Data

Processed datasets are stored separately under:

```text
data/processed/
```

These files contain the cleaned and transformed data required by subsequent modeling and evaluation stages.

Keeping raw and processed data separate makes the workflow more reproducible and prevents accidental modification of the original dataset.

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd E-Commerce_Conversion_Dynamics
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

#### Windows

```bash
.\.venv\Scripts\activate
```

#### Mac/Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the `notebooks/` directory and execute the notebooks in numerical order.

---

## 💼 Business Applications

The developed conversion prediction system can support several e-commerce use cases, including:

* Identifying sessions with stronger purchase intent.
* Understanding browsing behaviors associated with conversion.
* Supporting targeted engagement strategies.
* Identifying potential friction in the shopping journey.
* Prioritizing high-intent sessions for appropriate interventions.
* Supporting data-driven optimization of the online shopping experience.

The model should be used as a decision-support system rather than as a standalone replacement for business rules or experimentation.

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

* The dataset represents individual shopping sessions rather than complete customer journeys across multiple sessions.
* The target variable indicates whether a purchase occurs but does not represent purchase value or customer lifetime value.
* The dataset contains historical session information and may not represent current e-commerce browsing behavior.
* SMOTE is applied to address class imbalance during model training, while the original test distribution is retained for evaluation.
* Model performance depends on the behavioral and technical variables available in the dataset.

### Future Improvements

* Introduce customer-level historical behavior where available.
* Incorporate product-level information and product availability.
* Include real-time behavioral signals.
* Explore probability calibration and threshold optimization.
* Evaluate cost-sensitive classification strategies.
* Investigate more advanced ensemble methods.
* Introduce explainability methods for individual session predictions.
* Evaluate model performance using time-based validation.
* Develop a deployment-ready prediction pipeline.
* Monitor model performance and conversion patterns after deployment.

---

## 📌 Project Status

**Status:** Completed ✅

The project includes:

* Data understanding
* Exploratory data analysis
* Data preparation
* Feature engineering
* Class imbalance analysis
* SMOTE-based training preparation
* Multiple classification models
* Model comparison
* Ensemble learning
* MLflow experiment tracking
* Model evaluation
* Error analysis
* Model explainability
* Business insights

---

## 👨‍💻 Author

**Tejas Jadhav**

* GitHub: [@tejas-jadhav](https://github.com/TejasMJ)
* LinkedIn: [Tejas Jadhav](https://www.linkedin.com/in/tejas-m-jadhav/)
