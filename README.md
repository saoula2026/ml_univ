# 🎓 University Machine Learning Practical Works (`ml_univ`)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Git Flow](https://img.shields.io/badge/Git-Flow-brightgreen.svg?logo=git&logoColor=white)](https://github.com/saoula2026/ml_univ)

This repository contains all university practical works (**Travaux Pratiques - TPs**), laboratory exercises, assignments, and datasets for the **Machine Learning** course.

---

## 📂 Repository Structure

The repository is organized by practical work (TP) modules. Each TP has its dedicated directory containing the notebook, source code, and associated dataset(s):

```text
ml_univ/
├── README.md                 # Main repository documentation & guide
├── .gitignore                # Git ignore rules for ML & Jupyter
├── requirements.txt          # Python dependencies required to run the notebooks
│
├── tp01/                     # TP 1: Data Exploration & Preprocessing
│   ├── TP1.ipynb             # Jupyter Notebook with exercises and solutions
│   └── train.csv             # Dataset used (Titanic survival dataset)
│
├── tp02/                     # TP 2: (Upcoming / Next module)
│   ├── TP2.ipynb
│   └── dataset.csv
│
└── tpXX/                     # Future TPs (Classification, Regression, Clustering, etc.)
```

---

## 📋 List of Practical Works (TPs)

| TP | Topic / Subject | Dataset | Status |
| :--- | :--- | :--- | :---: |
| **[TP 01](./tp01/)** | **Data Exploration, Cleaning & Preprocessing**<br>• Missing values handling<br>• Outliers & duplicates detection<br>• Feature transformation & encoding<br>• Descriptive statistics | Titanic Dataset (`train.csv`) | ✅ Completed |
| **[TP 02](./tp2/)** | **Supervised Learning: Regression**<br>• Simple & Multiple Linear Regression<br>• Multicollinearity & VIF diagnostics<br>• Ridge (L2) & Lasso (L1) Regularization<br>• Model evaluation & metrics (MAE, RMSE, R²) | Diabetes Dataset (`diabetes.csv`) | ✅ Completed |
| **[TP 03](./tp03/)** | *To be added* | *TBD* | ⏳ Pending |

---

## 📊 Dataset Guidelines

> **Should you push datasets to this repository?**
>
> - **Small to Medium Datasets (< 50 MB)**: **YES!** Keep them inside each respective TP folder (e.g., `tp01/train.csv`). This ensures notebooks can be run immediately after cloning without broken paths.
> - **Large Datasets (> 50 MB / > 100 MB)**: **NO!** GitHub has a strict file size limit (100 MB max per file). For large datasets:
>   1. Add the large files to `.gitignore`.
>   2. Provide a download link (Google Drive, Kaggle, Hugging Face) in the TP's notes, or include a python download script.

---

## 🚀 Getting Started

### 1. Clone the repository using SSH
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
# Or launch VS Code directly in the repository:
code .
```

---

## 🌿 Git Flow for University TPs

To keep your repository clean and structured throughout the semester, follow this simple **Git Flow**:

### Step 1: Create a feature branch for each new TP
Never work directly on dirty uncommitted code on `main`. Start each TP on a dedicated branch:
```bash
# Ensure main is up to date
git checkout main
git pull origin main

# Create and switch to a branch for the new TP
git checkout -b tp02-classification
```

### Step 2: Work on your assignment
Add your notebook and dataset to the corresponding folder:
```bash
mkdir -p tp02
# Place your TP2.ipynb and dataset inside tp02/
```

### Step 3: Commit your work with clear messages
```bash
git add tp02/
git commit -m "feat(tp02): implement classification models and evaluation metrics"
```

### Step 4: Push the branch and merge to `main`
```bash
# Push branch to GitHub
git push -u origin tp02-classification

# Merge into main when completed:
git checkout main
git merge tp02-classification
git push origin main
```

*(Optional)* Tag your submission:
```bash
git tag -a v1.0-tp01 -m "Submission for TP01"
git push origin --tags
```

---

## 👤 Author
- **GitHub**: [@saoula2026](https://github.com/saoula2026)
- **Email**: a.ysaoula@gmail.com
