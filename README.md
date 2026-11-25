# 🎓 University Machine Learning Practical Works (`ml_univ`)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Git Flow](https://img.shields.io/badge/Git-Flow-brightgreen.svg?logo=git&logoColor=white)](https://github.com/saoula2026/ml_univ)

This repository hosts all university laboratory practical works (**Travaux Pratiques — TPs**), assignments, empirical benchmarks, and datasets for the **Machine Learning & Statistical Pattern Recognition** academic course.

---

## 📂 Repository Structure

The repository is modularly organized by practical work (TP) units. Each module is self-contained with executed Jupyter Notebooks, datasets, and structured documentation:

```text
ml_univ/
├── README.md                 # Main repository documentation & curriculum guide
├── .gitignore                # Git ignore rules for Python, Jupyter & checkpoints
├── requirements.txt          # Python dependencies required to run all notebooks
│
├── tp01/                     # TP 1: Data Exploration & Preprocessing
│   ├── TP1.ipynb             # Jupyter Notebook with exercises and solutions
│   └── train.csv             # Titanic survival benchmark dataset
│
├── tp2/                      # TP 2: Supervised Learning: Regression
│   ├── notebooks/
│   │   └── TP2_Supervised_Regression.ipynb   # Linear, Polynomial, Ridge & Lasso models
│   ├── data/                 # Clinical Diabetes dataset
│   ├── plots/                # Evaluation plots, residual diagnostics & regularization curves
│   ├── results/              # Model comparison metrics (CSV)
│   └── README.md             # In-depth technical report for TP2
│
├── TP3/                      # TP 3: Supervised Learning: Classification (Chapter IV)
│   ├── breast_cancer_dataset/
│   │   ├── TP3_Breast_Cancer.ipynb          # Binary classification benchmark (English)
│   │   └── TP3_Breast_Cancer_Arabic.ipynb   # النسخة الكاملة باللغة العربية الأكاديمية
│   └── iris_dataset/
│       ├── TP3_Iris.ipynb                   # Multi-class classification benchmark (English)
│       └── TP3_Iris_Arabic.ipynb            # النسخة الكاملة باللغة العربية الأكاديمية
│
└── tpXX/                     # Future modules (Clustering, PCA, Deep Learning)
```

---

## 📋 Curriculum & List of Practical Works (TPs)

| TP | Topic / Subject | Dataset(s) | Key Models & Methods | Status |
| :--- | :--- | :--- | :--- | :---: |
| **[TP 01](./tp01/)** | **Data Exploration, Cleaning & Preprocessing** | Titanic Dataset (`train.csv`) | • Missing value imputation<br>• Outlier detection & IQR bounds<br>• One-Hot & Ordinal Encoding<br>• Descriptive statistical analysis | ✅ Completed |
| **[TP 02](./tp2/)** | **Supervised Learning: Regression** | Diabetes Clinical Dataset (`diabetes.csv`) | • Simple & Multivariate Linear Regression<br>• Polynomial Regression (Deg 2)<br>• Multicollinearity & VIF Diagnostics<br>• Ridge ($L_2$) & Lasso ($L_1$) Regularization<br>• MAE, RMSE, $R^2$, and Adjusted $R^2$ | ✅ Completed |
| **[TP 03](./TP3/)** | **Supervised Learning: Classification**<br>*(Chapter IV: Classification)* | • Breast Cancer Wisconsin Diagnostic ($N=569, d=30$)<br>• Fisher's Iris Dataset ($N=150, d=4, C=3$) | • 11 Classifiers (Linear & Non-linear)<br>• Stratified Hold-out (80/20) & 5-Fold CV<br>• Confusion Matrices & ROC-AUC curves<br>• Decision Tree Structure plots (`plot_tree`)<br>• Hyperparameter Tuning (`GridSearchCV`)<br>• **Bilingual:** English & Arabic Editions | ✅ Completed |
| **TP 04** | **Unsupervised Learning & Clustering** | *TBD* | • $K$-Means & Hierarchical Clustering<br>• PCA & Dimensionality Reduction | ⏳ Upcoming |

---

## 🔬 TP 03 Deep Dive: Classification Algorithms (Chapter IV)

**TP 03** provides an exhaustive, mathematically rigorous empirical study of **11 core classification algorithms** covered in **Chapter IV: Classification**, benchmarked across both binary and multi-class tasks:

### 1. Evaluated Models & Theoretical Formulations
1. **Logistic Regression:** Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$, log-odds logit link, and binary/multinomial cross-entropy with $L_2$ regularization.
2. **Polynomial Logistic Regression:** Quadratic basis expansion $\phi(x)$ via `PolynomialFeatures(degree=2)` generating non-linear decision hypersurfaces.
3. **Simple Perceptron:** Rosenblatt's linear threshold unit with Heaviside step activation $f(z) = \mathbb{I}(z \ge 0)$, weight update rule $w \leftarrow w + \eta (y - \hat{y}) x$, and One-vs-Rest for multi-class.
4. **Multi-Layer Perceptron (MLP):** Feedforward artificial neural network with ReLU hidden activations ($g(z) = \max(0, z)$), softmax output, and Adam backpropagation via the chain rule.
5. **Decision Tree (Entropy):** Shannon Entropy minimization $H(S) = -\sum p_i \log_2(p_i)$ and Information Gain ($IG$) splitting with `max_depth=3`.
6. **Decision Tree (Gini Impurity):** CART impurity criterion $I_G(S) = 1 - \sum p_i^2$ with `max_depth=3`, comparing computational speed and split choices vs. Entropy.
7. **Random Forest Classifier:** Bootstrap Aggregation (Bagging) with random feature subspaces ($m = \lfloor\sqrt{d}\rfloor$), variance reduction theorem, and majority voting.
8. **K-Nearest Neighbors ($k$-NN):** Non-parametric instance-based metric learning with Euclidean distance $d(x, x') = \|x - x'\|_2$, demonstrating the necessity of feature scaling.
9. **Gaussian Naive Bayes:** Bayes' Theorem $P(Y|X) \propto P(Y) \prod P(X_j|Y)$ with continuous Gaussian likelihoods $\mathcal{N}(\mu, \sigma^2)$ and class-conditional independence.
10. **Linear Support Vector Machine (Linear SVM):** Soft-margin optimization $\min \frac{1}{2}\|w\|^2 + C \sum \xi_i$, maximizing geometric margin $\gamma = \frac{2}{\|w\|}$.
11. **Non-Linear Support Vector Machine (RBF Kernel SVM):** The Kernel Trick via Gaussian Radial Basis Function $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ mapping into infinite-dimensional Hilbert space.

### 2. Full Multi-Metric Evaluation Matrix
Every classifier is evaluated on unseen test data across:
* **Accuracy:** $\frac{TP + TN}{TP + FP + TN + FN}$
* **Precision:** $\frac{TP}{TP + FP}$
* **Recall (Sensitivity / TPR):** $\frac{TP}{TP + FN}$
* **Specificity (TNR):** $\frac{TN}{TN + FP}$ (One-vs-Rest weighted average for multi-class)
* **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
* **ROC-AUC Score:** Area under the True Positive Rate vs. False Positive Rate curve (binary & OvR weighted)
* **5-Fold Stratified Cross-Validation:** Training fold stability estimation ($\mu \pm \sigma$).

### 3. Notebook Editions & Direct Links

| Dataset | Language | Direct Link | Colab Badge |
| :--- | :---: | :--- | :---: |
| **Breast Cancer Wisconsin** | English | [TP3_Breast_Cancer.ipynb](./TP3/breast_cancer_dataset/TP3_Breast_Cancer.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saoula2026/ml_univ/blob/main/TP3/breast_cancer_dataset/TP3_Breast_Cancer.ipynb) |
| **Breast Cancer Wisconsin** | العربية | [TP3_Breast_Cancer_Arabic.ipynb](./TP3/breast_cancer_dataset/TP3_Breast_Cancer_Arabic.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saoula2026/ml_univ/blob/main/TP3/breast_cancer_dataset/TP3_Breast_Cancer_Arabic.ipynb) |
| **Fisher's Iris** | English | [TP3_Iris.ipynb](./TP3/iris_dataset/TP3_Iris.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saoula2026/ml_univ/blob/main/TP3/iris_dataset/TP3_Iris.ipynb) |
| **Fisher's Iris** | العربية | [TP3_Iris_Arabic.ipynb](./TP3/iris_dataset/TP3_Iris_Arabic.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saoula2026/ml_univ/blob/main/TP3/iris_dataset/TP3_Iris_Arabic.ipynb) |

---

## 📊 Dataset Guidelines

> **Repository Dataset Policy:**
> - **Small to Medium Datasets (< 50 MB)**: Kept inside each respective TP folder (e.g., `tp01/train.csv`). This ensures notebooks can be executed immediately after cloning without broken paths.
> - **Built-in Benchmark Datasets**: Loaded directly via `sklearn.datasets` (`load_diabetes()`, `load_breast_cancer()`, `load_iris()`).

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone git@github.com:saoula2026/ml_univ.git
cd ml_univ
```

### 2. Set up a Python Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook or VS Code
```bash
jupyter notebook
# Or open directly in VS Code:
code .
```

---

## 🌿 Git Flow for University TPs

To keep the repository clean and structured throughout the academic semester, follow this simple **Git Flow**:

### Step 1: Create a feature branch for each new TP
```bash
# Ensure main is up to date
git checkout main
git pull origin main

# Create and switch to a branch for the new TP
git checkout -b feature/tp03-classification
```

### Step 2: Work on the assignment
```bash
# Add notebooks and resources inside TP3/
```

### Step 3: Commit changes with Conventional Commit messages
```bash
git add TP3/
git commit -m "feat(tp03): implement 11 classification models, evaluations and bilingual notebooks"
```

### Step 4: Merge to `main` and Push
```bash
git checkout main
git merge feature/tp03-classification
git push origin main
```

---

## 👤 Author
- **GitHub**: [@saoula2026](https://github.com/saoula2026)
- **Email**: a.ysaoula@gmail.com
- **Institution**: University Machine Learning & Artificial Intelligence Program
