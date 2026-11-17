# 🎓 TP 2 — Supervised Learning: Linear & Regularized Regression
**Master 1 Artificial Intelligence — Machine Learning Course**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## 📋 Table of Contents
- [1. Practical Work Overview](#1-practical-work-overview)
- [2. Learning Objectives](#2-learning-objectives)
- [3. Dataset Description](#3-dataset-description)
- [4. Project Structure](#4-project-structure)
- [5. Implemented Regression Models](#5-implemented-regression-models)
- [6. Evaluation Metrics & Benchmark Results](#6-evaluation-metrics--benchmark-results)
- [7. Multicollinearity & Regularization Insights](#7-multicollinearity--regularization-insights)
- [8. Installation & Execution Guide](#8-installation--execution-guide)
- [9. Key Academic Takeaways](#9-key-academic-takeaways)

---

## 1. Practical Work Overview

This practical work (**TP 2**) covers the end-to-end Machine Learning lifecycle for **continuous numerical prediction (Supervised Regression)**. Building upon the exploratory data analysis and data preprocessing principles developed in **TP 1**, this assignment investigates both ordinary and regularized linear models using Scikit-Learn's clinical **Diabetes dataset** (Efron et al., 2004).

---

## 2. Learning Objectives

By completing this practical assignment, students will master:
1. **Univariate Linear Regression:** Formulating OLS normal equations, estimating slope ($\beta_1$) and intercept ($\beta_0$), and evaluating residuals.
2. **Multivariate Linear Regression:** Expanding to multiple simultaneous features, interpreting partial regression coefficients (*ceteris paribus*), and identifying parameter instability.
3. **Multicollinearity Diagnostics:** Calculating Variance Inflation Factors (VIF) to detect ill-conditioned predictors.
4. **$L_2$ Regularization (Ridge):** Mitigating multicollinearity and estimation variance via parameter shrinkage and 5-fold cross-validation.
5. **$L_1$ Regularization (Lasso):** Inducing model sparsity, analyzing geometric corner solutions, and performing automatic feature selection.
6. **Leakage-Free Cross-Validation:** Utilizing Scikit-Learn `Pipeline` architectures to ensure preprocessing parameters ($\mu, \sigma$) are fitted strictly on training folds.
7. **Comprehensive Model Evaluation:** Comparing models across MAE, MSE, RMSE, $R^2$, and Adjusted $R^2$.

---

## 3. Dataset Description

The dataset used is Scikit-Learn's built-in **Diabetes dataset** (`load_diabetes(as_frame=True)`):
- **Sample Size:** 442 patients ($n = 442$).
- **Features:** 10 physiological and clinical predictor variables ($p = 10$).
- **Target Variable:** Quantitative measure of diabetes disease progression recorded one year after baseline (continuous score ranging from 25 to 346).

### Feature Dictionary:

| Feature | Medical Measurement | Description & Significance |
| :--- | :--- | :--- |
| `age` | Patient Age | Age in years |
| `sex` | Biological Sex | Categorical indicator numerically encoded (-0.0446 vs +0.0507) |
| `bmi` | Body Mass Index | Ratio of weight to height squared ($kg/m^2$); primary obesity index |
| `bp` | Mean Blood Pressure | Average blood pressure ($mmHg$); vascular risk indicator |
| `s1` (`tc`) | Total Serum Cholesterol | Complete blood lipid measurement |
| `s2` (`ldl`) | Low-Density Lipoproteins | "Bad" cholesterol carrying lipids into arterial walls |
| `s3` (`hdl`) | High-Density Lipoproteins | "Good" cholesterol removing excess cholesterol |
| `s4` (`tch`) | Total Cholesterol / HDL | Atherogenic ratio assessing cardiac risk |
| `s5` (`ltg`) | Serum Triglycerides | Blood lipid concentration linked to insulin sensitivity |
| `s6` (`glu`) | Fasting Blood Sugar | Direct biomarker of glycemic control |

> **Crucial Preprocessing Note:** Each of the 10 feature variables in Scikit-Learn has already been mean-centered to 0 and scaled such that $\sum_{i=1}^n x_{ij}^2 = 1$ ($\text{std} = 1/\sqrt{n} \approx 0.0476$). The target values are preserved in their original clinical units.

A local CSV copy is saved in [`data/diabetes.csv`](./data/diabetes.csv).

---

## 4. Project Structure

The `tp2` directory follows a clean, modular academic structure:

```text
tp2/
│
├── data/
│   └── diabetes.csv                                  # Full 442-patient dataset
│
├── notebooks/
│   └── TP2_Supervised_Regression.ipynb               # Primary executed Jupyter Notebook (51 cells)
│
├── plots/
│   ├── distributions/
│   │   ├── features_boxplots.png                     # Boxplots for outlier screening
│   │   ├── features_distribution.png                 # Histograms & KDEs of all 10 features
│   │   └── target_distribution.png                   # Histogram, KDE, and boxplot of target
│   │
│   ├── correlations/
│   │   ├── correlation_heatmap.png                   # Annotated Pearson correlation matrix
│   │   └── feature_vs_target_scatter.png             # 10 bivariate scatter plots with OLS trendlines
│   │
│   ├── simple_regression/
│   │   ├── bmi_regression_line.png                   # Univariate OLS regression line on BMI
│   │   ├── actual_vs_predicted_simple.png            # Actual vs Predicted scatter (SLR)
│   │   └── residuals_simple.png                      # Residual scatter & distribution diagnostics
│   │
│   ├── multiple_regression/
│   │   ├── vif_multicollinearity.png                 # VIF diagnostic bar chart with threshold lines
│   │   ├── coefficients_multiple.png                 # Horizontal bar chart of OLS weights
│   │   ├── actual_vs_predicted_multiple.png          # Actual vs Predicted scatter (MLR)
│   │   └── residuals_multiple.png                    # Multivariate residual diagnostics
│   │
│   ├── regularization/
│   │   ├── ridge_alpha_vs_cv_error.png               # Ridge 5-fold CV error vs alpha
│   │   ├── ridge_coefficient_shrinkage.png           # OLS vs Ridge coefficient comparison
│   │   ├── ridge_actual_vs_predicted.png             # Actual vs Predicted scatter (Ridge)
│   │   ├── lasso_alpha_vs_cv_error.png               # Lasso 5-fold CV error vs alpha
│   │   ├── lasso_coefficients_sparsity.png           # Lasso feature selection & zeroed weights
│   │   └── lasso_actual_vs_predicted.png             # Actual vs Predicted scatter (Lasso)
│   │
│   └── evaluation/
│       ├── comparison_actual_vs_predicted.png        # 4-panel Actual vs Predicted grid
│       ├── comparison_residuals.png                  # Overlaid residual density curves (KDE)
│       ├── comparison_metrics_bars.png               # Grouped bar charts (MAE, RMSE, R2, Adj R2)
│       └── comparison_coefficients_grouped.png       # Grouped bar chart comparing all model weights
│
├── results/
│   ├── model_comparison.csv                          # Comprehensive train & test evaluation metrics
│   └── model_coefficients.csv                        # Comparative weights across OLS, Ridge, and Lasso
│
├── README.md                                         # Practical work documentation
└── requirements.txt                                  # Python dependencies
```

---

## 5. Implemented Regression Models

### 1. Simple Linear Regression (SLR — Baseline)
- **Predictor:** Body Mass Index (`bmi`) chosen due to strongest univariate Pearson correlation ($r = +0.59$).
- **Equation:** $\hat{y} = 152.00 + 998.58 \cdot \text{bmi}$

### 2. Multiple Linear Regression (MLR — OLS)
- **Predictors:** All 10 baseline physiological measurements.
- **Equation:** $\hat{y} = \beta_0 + \sum_{j=1}^{10} \beta_j x_j = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
- **Finding:** Near-doubles explanatory variance ($R^2: 23.3\% \to 45.3\%$), but exhibits high parameter variance due to collinear blood serum markers.

### 3. Ridge Regression ($L_2$ Regularization)
- **Objective:** $\min_{\boldsymbol{\beta}} \left\{ \frac{1}{m}\sum (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p \beta_j^2 \right\}$
- **Architecture:** `StandardScaler` + `Ridge` within a Scikit-Learn `Pipeline` tuned via 5-fold `GridSearchCV`.
- **Optimal Hyperparameter:** $\alpha = 20.0$.
- **Finding:** Stabilizes erratic opposing coefficients (e.g., `s1` and `s2`) and improves test generalization.

### 4. Lasso Regression ($L_1$ Regularization)
- **Objective:** $\min_{\boldsymbol{\beta}} \left\{ \frac{1}{m}\sum (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p |\beta_j| \right\}$
- **Architecture:** `StandardScaler` + `Lasso` within a Scikit-Learn `Pipeline` tuned via 5-fold `GridSearchCV`.
- **Optimal Hyperparameter:** $\alpha = 2.0$.
- **Finding:** Achieves the best test performance while inducing sparsity: **eliminates 3 features** (`age`, `s2`, `s4`) by driving their coefficients to exactly $0.00$.

---

## 6. Evaluation Metrics & Benchmark Results

All metrics were computed on the untouched 20% test partition ($n = 89$, seed 42):

| Model | Parameters ($p$) | Test MAE | Test MSE | Test RMSE | Test $R^2$ | Test Adjusted $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple Linear Regression (`bmi`)** | 1 | 52.26 | 4061.83 | 63.73 | 0.2334 | 0.2245 |
| **Multiple Linear Regression (All 10)** | 10 | 42.79 | 2900.19 | 53.85 | 0.4526 | 0.3824 |
| **Ridge Regression ($\alpha=20.0$)** | 10 | 42.89 | 2867.95 | 53.55 | 0.4587 | 0.3893 |
| **Lasso Regression ($\alpha=2.0$)** | **7** | **42.84** | **2798.60** | **52.90** | **0.4718** | **0.4261** |

> *Full table including training metrics is available at [`results/model_comparison.csv`](./results/model_comparison.csv).*

### Estimated Model Coefficients:

| Feature | OLS Multiple Linear | Ridge ($\alpha=20.0$) | Lasso ($\alpha=2.0$) | Clinical Role |
| :--- | :---: | :---: | :---: | :--- |
| `age` | +37.90 | +1.97 | **0.00** *(Eliminated)* | Demographic baseline |
| `sex` | -241.96 | -10.63 | -7.44 | Biological gender |
| `bmi` | **+542.43** | **+25.12** | **+26.13** | Strongest positive driver |
| `bp` | +347.70 | +15.97 | +14.74 | Vascular risk factor |
| `s1` (`tc`) | -931.49 | -8.39 | -4.05 | Total serum lipids |
| `s2` (`ldl`) | +518.06 | -2.54 | **0.00** *(Eliminated)* | Low-density lipids |
| `s3` (`hdl`) | +163.42 | -7.98 | -10.88 | Protective high-density lipids |
| `s4` (`tch`) | +275.32 | +7.81 | **0.00** *(Eliminated)* | Cholesterol / HDL ratio |
| `s5` (`ltg`) | **+736.20** | **+20.69** | **+21.50** | Triglycerides concentration |
| `s6` (`glu`) | +48.67 | +3.25 | +1.47 | Glycemic blood sugar |

> *Full table is saved in [`results/model_coefficients.csv`](./results/model_coefficients.csv).*

---

## 7. Multicollinearity & Regularization Insights

### Variance Inflation Factor (VIF) Analysis
Calculating VIF on the training partition revealed severe linear dependence among serum lipids:
- `s1` (Total Cholesterol): **$\text{VIF} = 55.26$** (Critical collinearity $> 10$)
- `s2` (LDL Cholesterol): **$\text{VIF} = 35.77$** (Critical collinearity $> 10$)
- `s3` (HDL Cholesterol): **$\text{VIF} = 14.29$**
- `s5` (Triglycerides): **$\text{VIF} = 10.08$**

Because LDL is a biological component of total cholesterol, Pearson correlation $r(s1, s2) = +0.897$. In unconstrained OLS, $(\mathbf{X}^T\mathbf{X})$ is near-singular, causing `s1` and `s2` to take on massive opposing coefficients (-931.49 vs +518.06).

### Regularization Solutions:
1. **Ridge ($L_2$):** Reconditions the matrix by adding $\alpha \mathbf{I}$, compressing coefficients toward zero and reducing test RMSE from 53.85 to 53.55.
2. **Lasso ($L_1$):** Explains redundant variance by eliminating `s2` and `s4` entirely, producing the lowest test RMSE (52.90) and highest $R^2$ (0.4718).

---

## 8. Installation & Execution Guide

### Prerequisites
Python 3.10+ and a virtual environment.

### 1. Set Up Environment
```bash
# From repository root:
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r tp2/requirements.txt
```

### 2. Launch Jupyter Notebook
```bash
jupyter notebook tp2/notebooks/TP2_Supervised_Regression.ipynb
```
Select the Python 3 kernel and run all cells sequentially (`Cell > Run All`). All outputs, Markdown analyses, and figures will render interactively.

---

## 9. Key Academic Takeaways

1. **Multivariate Power:** Incorporating clinical multidimensionality increased explained variance from $23.3\%$ (BMI alone) to $45.3\%$ (All features).
2. **Collinearity Hazard:** High correlations between features cause OLS coefficients to become physically uninterpretable without affecting training predictions.
3. **Regularization Utility:** Ridge stabilizes collinear weights; Lasso produces a sparser, more interpretable model while outperforming OLS on test data ($R^2 = 0.4718$).
4. **Data Leakage Hygiene:** Standardizing within cross-validation folds using Scikit-Learn `Pipeline` is mandatory to avoid optimistic performance inflation.
5. **Linear Bounds:** The $\approx 47\%$ $R^2$ ceiling highlights that physiological progression contains inherent non-linear interactions requiring non-linear modeling in future practical works.
