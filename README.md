# ML Lab 08: Ensemble Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)

---

## Academic Details
- **Student Name:** Shrri Dharshan D R
- **Register Number:** 23BPS1090
- **Course Code:** BCSE209P
- **Course Title:** Machine Learning Laboratory
- **Faculty:** Dr. S. Shridevi
- **Institution:** School of Computer Science and Engineering (SCOPE), VIT Chennai

---

## Overview
This repository contains laboratory implementations and empirical comparative evaluations of **Ensemble Learning Techniques** across three benchmark datasets from the UCI Machine Learning Repository.

The laboratory investigates bias-variance decomposition and compares major ensemble paradigms:
1. **Bagging (Bootstrap Aggregation):** Variance reduction through parallel independent bootstrap sampling and aggregation (Random Forest, BaggingClassifier with Out-Of-Bag evaluation).
2. **Boosting:** Bias reduction through sequential adaptive learning on residual errors (AdaBoost with adaptive sample weights, Gradient Tree Boosting, and XGBoost).
3. **Voting / Stacking:** Combining diverse heterogeneous base estimators (Soft and Hard Voting Classifiers).

---

## Experiments & Datasets

### 1. Bank Marketing Term Deposit Subscription (lab/)
- **Dataset:** Bank Marketing Dataset ([UCI ID: 222](https://archive.ics.uci.edu/dataset/222))
- **Objective:** Predict client term deposit subscription (`yes` / `no`) using ensemble models on demographic, financial, and campaign contact features.
- **Notebook:** [lab/23BPS1090_ShrriDharshan_ML_Lab8_Ensemble_BankMarketing.ipynb](./lab/23BPS1090_ShrriDharshan_ML_Lab8_Ensemble_BankMarketing.ipynb)
- **Data Directory:** `lab/bank_marketing_data/`
- **Techniques:** Random Forest, AdaBoost, Gradient Boosting, Soft Voting Classifier, feature importance ranking, and threshold calibration.

### 2. Adult Census Income Prediction (lab/)
- **Dataset:** Adult Census Income Dataset ([UCI ID: 2](https://archive.ics.uci.edu/dataset/2))
- **Objective:** Predict whether an individual's annual income exceeds $50K/year based on census demographic features (age, education, occupation, marital status, hours per week).
- **Notebook:** [lab/23BPS1090_ShrriDharshan_ML_Lab_Ensemble_Adult.ipynb](./lab/23BPS1090_ShrriDharshan_ML_Lab_Ensemble_Adult.ipynb)
- **Techniques:** One-hot encoding of categorical features, class imbalance handling, Random Forest Out-of-Bag (OOB) scoring, AdaBoost, Gradient Boosting, and Soft Voting Ensemble.

### 3. Student Academic Performance Prediction (lab/)
- **Dataset:** Student Performance Dataset ([UCI ID: 320](https://archive.ics.uci.edu/dataset/320))
- **Objective:** Predict secondary student academic outcome / final exam pass performance using socio-economic, familial, and behavioral school attributes.
- **Notebook:** [lab/23BPS1090_ShrriDharshan_ML_Lab_Ensemble_StudentPerf.ipynb](./lab/23BPS1090_ShrriDharshan_ML_Lab_Ensemble_StudentPerf.ipynb)
- **Techniques:** Bagging vs. Boosting comparisons, Gini impurity feature importances, and multi-model voting.

### 4. Reference Implementations (Reference materials/)
- **Notebook:** [Reference materials/Ensemble.ipynb](./Reference%20materials/Ensemble.ipynb)
- **Objective:** Baseline demonstrations and exploratory reference code for ensemble implementations.

---

## Evaluation Metrics
- **Classification Accuracy & Balanced Accuracy**
- **Precision, Recall, and F1-Score** (Macro and Weighted)
- **ROC-AUC & Precision-Recall AUC curves**
- **Confusion Matrices**
- **Out-of-Bag (OOB) Error Score** for Bagging / Random Forest
- **Feature Importance Distributions:** Comparison of Gini importance across ensemble architectures

---

## Repository Structure
```text
ML-Lab-08-Ensemble-Learning/
├── lab/
│   ├── 23BPS1090_ShrriDharshan_ML_Lab8_Ensemble_BankMarketing.ipynb
│   ├── 23BPS1090_ShrriDharshan_ML_Lab_Ensemble_Adult.ipynb
│   ├── 23BPS1090_ShrriDharshan_ML_Lab_Ensemble_StudentPerf.ipynb
│   └── bank_marketing_data/
├── Reference materials/
│   └── Ensemble.ipynb
├── .gitignore
└── README.md
```

---

## How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/shrridharshan27/ML-Lab-08-Ensemble-Learning.git
   cd ML-Lab-08-Ensemble-Learning
   ```
2. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn ucimlrepo xgboost jupyter
   ```
3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
