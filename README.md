# Peakflo Data Analyst Take-Home Task (2026)
## Automated Invoice Categorization using Machine Learning

> **Submitted by:** Samarth Saxena  
> **Date:** April 3, 2026  
> **Position Applied For:** Machine Learning Engineer Intern (Paid – India/Remote)

---

## 📋 Task Overview

This project builds a **multi-class classification model** that automatically predicts the correct **account category** (`accountName`) for each financial bill/expense record, based on the item name, description, vendor identity, and total amount. The dataset `accounts-bills.json` contains **4,894 expense records** across **103 distinct categories**.

**Key Achievement:** The final Random Forest model achieves **86.37% accuracy**, surpassing the required ≥85% threshold.

---

## 📁 Repository Structure

```
peakflo-data-analyst-task/
│
├── accounts-bills.json          # Original dataset (4,894 records, 7 fields)
├── assignment.ipynb             # ✅ Primary Jupyter Notebook (end-to-end pipeline)
├── eda_and_model.py             # Equivalent standalone Python script
│
├── report.md                    # Written analysis report (4–6 pages)
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── account_distribution.png    # EDA visualization: Top 30 account categories
├── lr_classification_report.txt # Detailed metrics: Logistic Regression
└── rf_classification_report.txt # Detailed metrics: Random Forest (final model)
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
- Console logs: dataset shape, class distribution, model accuracy
- `account_distribution.png` — bar chart of top 30 expense categories
- `lr_classification_report.txt` — per-class precision/recall/F1 for Logistic Regression
- `rf_classification_report.txt` — per-class precision/recall/F1 for Random Forest

> **Note:** All random operations use `random_state=42` for full reproducibility.

---

## 🧠 Approach Summary

| Stage | Details |
|---|---|
| **Data** | 4,894 bills, 103 categories, severe class imbalance (top class = 24% of data) |
| **Features** | TF-IDF on `itemName + itemDescription` (5,000 tokens, unigrams + bigrams), OneHotEncoding for `vendorId`, StandardScaler for `itemTotalAmount` |
| **Imbalance Handling** | Pruned 16 single-occurrence class samples; used `class_weight='balanced'` in all models |
| **Models** | Logistic Regression (baseline: 77.36%) → Random Forest (final: **86.37%**) |
| **Validation** | 80/20 stratified train/test split with `random_state=42` |

---

## 📊 Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression | 77.36% | 0.75 | 0.79 |
| **Random Forest** | **86.37%** | **0.82** | **0.86** |

The Random Forest model **exceeds the 85% accuracy requirement** and shows strong performance across high-frequency categories. Rare classes (≤ 5 instances) remain challenging due to data scarcity.

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
