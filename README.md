# Peakflo Data Analyst Take-Home Task (2026)
## Automated Invoice Categorization using Machine Learning

> **Submitted by:** Samarth Saxena  
> **Date:** April 3, 2026  
> **Position Applied For:** Machine Learning Engineer Intern (Paid – India/Remote)

---

## 📋 Task Overview

This project builds a **multi-class classification model** that automatically predicts the correct **account category** (`accountName`) for each financial bill/expense record, based on the item name, description, vendor identity, and total amount. The dataset `accounts-bills.json` contains **4,894 expense records** across **103 distinct categories**.

**Key Achievement:** The final LinearSVC model achieves **92.32% accuracy**, surpassing both the ≥85% required threshold and the ≥92% bonus threshold.

---

## 📁 Repository Structure

```
peakflo-data-analyst-task/
│
├── accounts-bills.json           # Original dataset (4,894 records, 7 fields)
├── assignment.ipynb              # ✅ Primary Jupyter Notebook (end-to-end pipeline)
├── eda_and_model.py              # Equivalent standalone Python script
│
├── report.md                     # Written analysis report (6 pages)
├── README.md                     # This file
├── requirements.txt              # Python dependencies
│
├── account_distribution.png      # EDA: Top 30 categories + class frequency histogram
├── model_comparison.png          # Results: Accuracy bar chart across all 3 models
├── confidence_distribution.png   # Deployment: Decision score distribution
│
├── lr_classification_report.txt  # Detailed metrics: Logistic Regression
├── rf_classification_report.txt  # Detailed metrics: Random Forest
└── svc_classification_report.txt # Detailed metrics: LinearSVC (final model)
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### 1. Clone or Download the Repository

```bash
git clone https://github.com/samarthsaxena2004/peakflo-data-analyst-task.git
cd peakflo-data-analyst-task
```

Or download and extract the ZIP archive.

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate (macOS/Linux):
source venv/bin/activate

# Activate (Windows):
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Jupyter Notebook ✅ (Preferred)

The Jupyter Notebook (`assignment.ipynb`) is the primary deliverable. It includes:
- Inline visualizations
- Step-by-step narrative explanation
- All outputs pre-executed for review

```bash
jupyter notebook
```

Open `assignment.ipynb` in your browser and run all cells sequentially (Cell → Run All).

### Option 2: Standalone Python Script

For a quick terminal-based run without a notebook:

```bash
python eda_and_model.py
```

**Script outputs:**
- Console logs: dataset shape, class distribution, model accuracy for all 3 models
- `account_distribution.png` — Top 30 categories + class frequency histogram
- `model_comparison.png` — Accuracy bar chart comparing all models against thresholds
- `confidence_distribution.png` — Decision score distribution (correct vs incorrect)
- `lr_classification_report.txt` — per-class precision/recall/F1 for Logistic Regression
- `rf_classification_report.txt` — per-class precision/recall/F1 for Random Forest
- `svc_classification_report.txt` — per-class precision/recall/F1 for LinearSVC (final)

> **Note:** All random operations use `random_state=42` for full reproducibility.

---

## 🧠 Approach Summary

| Stage | Details |
|---|---|
| **Data** | 4,878 bills (post-filtering), 87 classes, severe imbalance (top class = 24%) |
| **Features** | TF-IDF (15k tokens, bigrams, sublinear_tf), OneHot for `vendorId`, **vendor-category prior**, StandardScaler for `itemTotalAmount` |
| **Imbalance Handling** | Pruned 16 single-occurrence samples; `class_weight='balanced'` in all models |
| **Models** | LR (baseline: 80.94%) → RF (88.52%) → **LinearSVC C=8 (final: 92.32%)** |
| **Validation** | 80/20 stratified train/test split with `random_state=42` |

---

## 📊 Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression | 80.94% | 0.789 | 0.811 |
| Random Forest (200 trees) | 88.52% | 0.798 | 0.878 |
| **LinearSVC C=8 (final)** | **92.32%** ✅ | **0.837** | **0.923** |

The LinearSVC model **exceeds both the 85% required threshold and the 92% bonus threshold**. The key drivers of this improvement were the **vendor-category prior feature** and switching to LinearSVC, which is the gold-standard classifier for high-dimensional sparse text problems.

---

## 📝 Full Report

For the complete methodology, EDA insights, result discussion, and business recommendations, please read **[`report.md`](./report.md)**.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization |
| `seaborn` | Enhanced visualization |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `jupyter` | Notebook environment |

Install all via: `pip install -r requirements.txt`
