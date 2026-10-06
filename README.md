# Ensemble Learning: Bagging, Boosting & Voting Classifiers

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Author
- **Shrri Dharshan D R** — [@shrridharshan27](https://github.com/shrridharshan27)

---

## Overview
This repository contains implementations and empirical comparisons of **Ensemble Learning Paradigms** across three benchmark datasets from the UCI Machine Learning Repository.

The project investigates bias-variance decomposition and compares major ensemble architectures:
1. **Bagging (Bootstrap Aggregation):** Variance reduction through parallel bootstrap sampling and aggregation (Random Forest, BaggingClassifier with Out-Of-Bag evaluation).
2. **Boosting:** Bias reduction via sequential adaptive learning on residual errors (AdaBoost with adaptive sample weights, Gradient Tree Boosting, and XGBoost).
3. **Voting / Stacking:** Combining diverse heterogeneous base estimators (Soft and Hard Voting Classifiers).

---

## Project Modules & Datasets

### 1. Bank Marketing Term Deposit Subscription (`notebooks/`)
- **Dataset:** Bank Marketing Dataset ([UCI ID: 222](https://archive.ics.uci.edu/dataset/222))
- **Objective:** Predict client term deposit subscription (`yes` / `no`) using ensemble models on demographic, financial, and campaign contact features.
- **Notebook:** [`notebooks/23BPS1090_ShrriDharshan_ML_Lab8_Ensemble_BankMarketing.ipynb`](./notebooks/23BPS1090_ShrriDharshan_ML_Lab8_Ensemble_BankMarketing.ipynb)
- **Data Directory:** `notebooks/bank_marketing_data/`
- **Methodology:** Random Forest, AdaBoost, Gradient Boosting, Soft Voting Classifier, feature importance ranking, and threshold calibration.

### 2. Adult Census Income Prediction (`notebooks/`)
- **Dataset:** Adult Census Income Dataset ([UCI ID: 2](https://archive.ics.uci.edu/dataset/2))
- **Objective:** Predict whether an individual's annual income exceeds $50K/year based on census demographic features (age, education, occupation, marital status, hours per week).
- **Notebook:** [`notebooks/23BPS1090_ShrriDharshan_ML_Lab_Ensemble_Adult.ipynb`](./notebooks/23BPS1090_ShrriDharshan_ML_Lab_Ensemble_Adult.ipynb)
- **Methodology:** One-hot encoding of categorical features, class imbalance handling, Random Forest Out-of-Bag (OOB) scoring, AdaBoost, Gradient Boosting, and Soft Voting Ensemble.

### 3. Student Academic Performance Prediction (`notebooks/`)
- **Dataset:** Student Performance Dataset ([UCI ID: 320](https://archive.ics.uci.edu/dataset/320))
- **Objective:** Predict secondary student academic outcome / final exam pass performance using socio-economic, familial, and behavioral school attributes.
- **Notebook:** [`notebooks/23BPS1090_ShrriDharshan_ML_Lab_Ensemble_StudentPerf.ipynb`](./notebooks/23BPS1090_ShrriDharshan_ML_Lab_Ensemble_StudentPerf.ipynb)
- **Methodology:** Bagging vs. Boosting comparisons, Gini impurity feature importances, and multi-model voting.

### 4. Reference Benchmarks (`reference_benchmarks/`)
- **Notebook:** [`reference_benchmarks/Ensemble.ipynb`](./reference_benchmarks/Ensemble.ipynb)
- **Objective:** Baseline implementations and exploratory reference code for ensemble implementations.

---

## Evaluation Metrics
- **Classification Accuracy & Balanced Accuracy**
- **Precision, Recall, and F1-Score** (Macro and Weighted)
- **ROC-AUC & Precision-Recall AUC curves**
- **Confusion Matrices**
- **Out-of-Bag (OOB) Error Score** for Bagging / Random Forest
- **Feature Importance Distributions:** Comparison of Gini importance across ensemble architectures
- **In-Depth Comparative Analysis:** Benchmarking runtime efficiency against predictive accuracy

---

## Project Structure
```text
Ensemble-Learning-ML/
├── notebooks/
│   ├── 23BPS1090_ShrriDharshan_ML_Lab8_Ensemble_BankMarketing.ipynb
│   ├── 23BPS1090_ShrriDharshan_ML_Lab_Ensemble_Adult.ipynb
│   ├── 23BPS1090_ShrriDharshan_ML_Lab_Ensemble_StudentPerf.ipynb
│   └── bank_marketing_data/
├── reference_benchmarks/
│   └── Ensemble.ipynb
├── .gitignore
└── README.md
```

---

## Quickstart & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/shrridharshan27/Ensemble-Learning-ML.git
   cd Ensemble-Learning-ML
   ```
2. **Install requirements:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn ucimlrepo xgboost jupyter
   ```
3. **Run the notebooks:**
   ```bash
   jupyter notebook
   ```

---

## License
Distributed under the [MIT License](https://opensource.org/licenses/MIT).
