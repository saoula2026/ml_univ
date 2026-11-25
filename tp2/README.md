# 🎓 TP 2 — Supervised Learning: Linear, Polynomial & Regularized Regression
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
- [5. Implemented Regression Models (7 Models)](#5-implemented-regression-models-7-models)
- [6. Evaluation Metrics & Benchmark Results](#6-evaluation-metrics--benchmark-results)
- [7. Key Insights & Overfitting Analysis](#7-key-insights--overfitting-analysis)
- [8. Installation & Execution Guide](#8-installation--execution-guide)
- [9. Key Academic Takeaways](#9-key-academic-takeaways)

---

## 1. Practical Work Overview

This practical work (**TP 2**) covers the end-to-end Machine Learning lifecycle for **continuous numerical prediction (Supervised Regression)**. Using Scikit-Learn's clinical **Diabetes dataset** (Efron et al., 2004), we evaluate a clean, organized progression of 7 regression models across an **80% training ($n=353$)** and **20% testing ($n=89$)** partition without generating any artificial data.

---

## 2. Learning Objectives

By completing this practical assignment, students will master:
1. **Univariate Linear Regression:** Formulating OLS normal equations, estimating slope ($\beta_1$) and intercept ($\beta_0$).
2. **Multivariate Linear Regression:** Expanding to multiple simultaneous features and identifying multicollinearity.
3. **Polynomial Regression & Overfitting:** Observing high variance when expanding to 65 non-linear terms with unconstrained OLS.
4. **$L_2$ Regularization (Ridge):** Mitigating collinearity and parameter variance via weight shrinkage and 5-fold cross-validation.
5. **$L_1$ Regularization (Lasso):** Inducing model sparsity and automatic feature selection.
6. **Regularized Polynomial Models:** Combining `PolynomialFeatures` with `StandardScaler` and Ridge/Lasso to eliminate polynomial overfitting.
7. **Leakage-Free Cross-Validation:** Utilizing Scikit-Learn `Pipeline` architectures to ensure preprocessing parameters ($\mu, \sigma$) are fitted strictly on training folds.
8. **Comprehensive Model Evaluation:** Comparing models across MAE, RMSE, and $R^2$ on the unseen test set.

---

## 3. Dataset Description

The dataset used is Scikit-Learn's built-in **Diabetes dataset** (`load_diabetes(as_frame=True)`):
- **Sample Size:** 442 patients ($n = 442$).
- **Partition:** 80% Train ($n = 353$), 20% Test ($n = 89$), fixed seed `random_state=42`.
- **Features:** 10 physiological predictor variables ($p = 10$).
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
│   └── TP2_Supervised_Regression.ipynb               # Clean, executed Jupyter Notebook (50 cells)
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
│   │   ├── ridge_alpha_vs_cv_error.png               # Ridge 5-fold CV error & coefficient shrinkage
│   │   └── lasso_alpha_vs_cv_error.png               # Lasso 5-fold CV error & feature sparsity
│   │
│   ├── polynomial_regression/
│   │   ├── poly_ols_diagnostics.png                  # Polynomial OLS Actual vs Predicted & Overfitting Residuals
│   │   ├── poly_regularization_diagnostics.png       # Poly Ridge/Lasso alpha curves & 14 active biomarkers
│   │   └── poly_actual_vs_predicted.png              # Side-by-side Actual vs Predicted for Poly Ridge & Lasso
│   │
│   └── evaluation/
│       ├── comparison_actual_vs_predicted.png        # 7-panel Actual vs Predicted grid across all models
│       ├── comparison_residuals.png                  # Overlaid residual density curves (KDE)
│       └── comparison_metrics_bars.png               # Grouped bar charts (Test RMSE & R2) across all 7 models
│
├── results/
│   ├── model_comparison.csv                          # Comprehensive train & test evaluation metrics (all 7 models)
│   └── model_coefficients.csv                        # Comparative weights across OLS, Ridge, and Lasso
│
├── README.md                                         # Practical work documentation
└── requirements.txt                                  # Python dependencies
```

---

## 5. Implemented Regression Models (7 Models)

### Model 1: Simple Linear Regression (SLR — Baseline)
- **Predictor:** Body Mass Index (`bmi`).
- **Equation:** $\hat{y} = 152.00 + 998.58 \cdot \text{bmi}$
- **Performance:** Test $R^2 = 0.2334$, Test RMSE = $63.73$.

### Model 2: Multiple Linear Regression (MLR — OLS)
- **Predictors:** All 10 baseline physiological measurements.
- **Finding:** Doubles explanatory power ($R^2: 0.2334 \to 0.4526$, RMSE: $63.73 \to 53.85$), but exhibits parameter instability from collinear blood serum features (`s1`, `s2`).

### Model 3: Polynomial Regression (Degree 2, OLS)
- **Predictors:** 65 polynomial features (10 linear + 10 quadratic $x_j^2$ + 45 interactions $x_j x_k$).
- **Architecture:** `Pipeline([('poly', PolynomialFeatures(degree=2)), ('scaler', StandardScaler()), ('ols', LinearRegression())])`.
- **Finding:** **Demonstrates Overfitting!** Training $R^2$ jumps to $0.6062$ (Train RMSE drops to $48.92$), but Test RMSE worsens to **$55.64$** (Test $R^2$ drops to $0.4156$). High model flexibility leads to high variance.

### Model 4: Ridge Regression ($L_2$ Regularization)
- **Objective:** $\min_{\boldsymbol{\beta}} \{ \text{MSE} + \alpha \sum_{j=1}^p \beta_j^2 \}$
- **Architecture:** `Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])` tuned via 5-fold CV.
- **Optimal Hyperparameter:** $\alpha = 20.0$.
- **Finding:** Stabilizes collinear weights and improves test generalization (Test RMSE = $53.55$, Test $R^2 = 0.4587$).

### Model 5: Lasso Regression ($L_1$ Regularization)
- **Objective:** $\min_{\boldsymbol{\beta}} \{ \text{MSE} + \alpha \sum_{j=1}^p |\beta_j| \}$
- **Architecture:** `Pipeline([('scaler', StandardScaler()), ('lasso', Lasso())])` tuned via 5-fold CV.
- **Optimal Hyperparameter:** $\alpha = 2.0$.
- **Finding:** Achieves sparsity by eliminating 3 features (`age`, `s2`, `s4`) with $\beta = 0.00$, outperforming linear OLS (Test RMSE = $52.90$, Test $R^2 = 0.4718$).

### Model 6: Polynomial Ridge Regression (Degree 2 + $L_2$)
- **Architecture:** `Pipeline([('poly', PolynomialFeatures(2)), ('scaler', StandardScaler()), ('ridge', Ridge())])`.
- **Optimal Hyperparameter:** $\alpha = 100.0$ (via 5-fold CV).
- **Finding:** **Cures polynomial overfitting!** Shrinks all 65 polynomial weights, reducing Test RMSE from $55.64$ down to **$51.57$** (Test MAE = **$41.57$**, Test $R^2 = 0.4981$).

### Model 7: Polynomial Lasso Regression (Degree 2 + $L_1$)
- **Architecture:** `Pipeline([('poly', PolynomialFeatures(2)), ('scaler', StandardScaler()), ('lasso', Lasso())])`.
- **Optimal Hyperparameter:** $\alpha = 4.0$ (via 5-fold CV).
- **Finding:** **Overall Benchmark Winner.** Eliminates 51 redundant polynomial terms (78.5% sparsity), keeping **14 active non-linear biomarkers**. Achieves the lowest Test RMSE (**$51.23$**) and highest explained variance (**Test $R^2 = 0.5047$**).

---

## 6. Evaluation Metrics & Benchmark Results

All metrics were evaluated on the untouched 20% test partition ($n = 89$, seed 42) and recorded in [`results/model_comparison.csv`](./results/model_comparison.csv):

| Model | Parameters | Train MAE | Train RMSE | Train $R^2$ | Test MAE | Test RMSE | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Simple Linear Regression (BMI)** | 1 | 51.38 | 62.08 | 0.3657 | 52.26 | 63.73 | 0.2334 |
| **2. Multiple Linear Regression (OLS)** | 10 | 43.48 | 53.56 | 0.5279 | 42.79 | 53.85 | 0.4526 |
| **3. Polynomial Regression (Degree 2, OLS)** | 65 | 39.14 | 48.92 | 0.6062 | 43.58 | 55.64 | 0.4156 |
| **4. Ridge Regression ($L_2$, $\alpha=20.0$)** | 10 | 43.71 | 53.81 | 0.5234 | 42.89 | 53.55 | 0.4587 |
| **5. Lasso Regression ($L_1$, $\alpha=2.0$)** | 7 | 44.23 | 54.15 | 0.5174 | 42.84 | 52.90 | 0.4718 |
| **6. Polynomial Ridge (Deg 2 + $L_2$, $\alpha=100.0$)** | 65 | **41.95** | 51.66 | 0.5608 | **41.57** | 51.57 | 0.4981 |
| **7. Polynomial Lasso (Deg 2 + $L_1$, $\alpha=4.0$)** | **14** | 44.28 | 53.95 | 0.5209 | 41.90 | **51.23** | **0.5047** |

---

## 7. Key Insights & Overfitting Analysis

### 1. The Overfitting Phenomenon in Polynomial Regression
- Expanding to Degree 2 generates 65 features.
- Without regularization, OLS achieves a deceptively low training RMSE ($48.92$), but test RMSE degrades to **$55.64$** (worse than linear regression at $53.85$).
- **Pedagogical Takeaway:** More features $\neq$ better test performance. When $p$ increases relative to $n$, unconstrained models fit random noise in the training set.

### 2. How Regularization Solves the Problem
- **$L_2$ (Ridge):** Smoothly contracts all 65 weights, dropping test RMSE to **$51.57$**.
- **$L_1$ (Lasso):** Prunes 51 of 65 candidate terms (78.5% sparsity), keeping 14 true non-linear biomarkers and achieving the global minimum test RMSE of **$51.23$**.

### 3. Key Non-Linear Biomarkers Retained by Polynomial Lasso:
- `bmi^2` ($+2.90$): Accelerating risk with higher obesity.
- `bmi * bp` ($+2.27$): **Metabolic Syndrome synergy** (adiposity combined with hypertension).
- `s6^2` ($+1.32$): Non-linear metabolic deterioration from elevated blood sugar.
- `age * sex` ($+3.09$) and `age * bp` ($+0.71$): Compounding demographic and vascular risk factors.

### 4. Mandatory Pipeline Standardization
Squaring and multiplying variables alters scales drastically ($0.0476^2 \approx 0.0022$). Placing `StandardScaler` inside the `Pipeline` ensures all 65 dimensions have zero mean and unit variance for fair regularization penalties, with zero data leakage during cross-validation.

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
Select the Python 3 kernel and run all cells sequentially (`Cell > Run All`). All 50 cells execute cleanly and display interactive figures and evaluation tables.

---

## 9. Key Academic Takeaways

1. **Multivariate Progression:** Adding physiological predictors improved $R^2$ from $23.3\%$ (BMI alone) to $45.3\%$ (All features).
2. **Overfitting Illustrated:** Unregularized polynomial regression overfits (Train $R^2 = 0.6062 \to$ Test $R^2 = 0.4156$).
3. **Regularization Essential:** Ridge and Lasso control polynomial complexity, with Polynomial Lasso achieving the best test performance (**$R^2 = 0.5047$, $\text{RMSE} = 51.23$**).
4. **Data Leakage Immunity:** Using Scikit-Learn `Pipeline` ensures preprocessing parameters are learned strictly from the 80% training set.
