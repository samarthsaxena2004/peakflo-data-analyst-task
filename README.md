# Invoice Categorization using Machine Learning

This repository contains the solution for the Peakflo Data Analyst Take Home Task (2026), aiming to build a classification algorithm to categorize financial bills automatically.

## Project Structure
- `accounts-bills.json`: The dataset containing 4,894 expense records.
- `assignment.ipynb`: Jupyter Notebook containing the end-to-end data analysis, preprocessing, and model training.
- `eda_and_model.py`: A Python script containing the exact same pipeline if executed via terminal.
- `report.md`: Detailed written description of the methodology, results, and discussion.
- `requirements.txt`: List of dependencies necessary to reproduce the work.

## Setup Instructions

1. **Clone or Download the Repository**
2. **Set up a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**
   Install the required Python packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Instructions to Run

### Option 1: Using Jupyter Notebook (Preferred)
1. Start the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
2. Open `assignment.ipynb` in your browser.
3. Run all cells sequentially to reproduce the exploratory data analysis, data preprocessing, and model evaluation pipeline.

### Option 2: Using the Python Script
If you prefer to run the analysis via terminal without a notebook, you can execute the standalone Python script:
```bash
python eda_and_model.py
```
This script will:
- Read `accounts-bills.json`.
- Output log messages showing dataset shape, data distributions, and modeling results.
- Save a visualization in `account_distribution.png`.
- Output two text files (`lr_classification_report.txt` and `rf_classification_report.txt`) encompassing detailed classification metrics.

## Approach Summary
Our methodology leverages TF-IDF combined with `StandardScaler` and `OneHotEncoder` mapped through a `ColumnTransformer` to deal with mixed unstructured fields (Item Names/Descriptions, Total Amounts, and Vendor IDs). Due to massive class imbalances (103 categories), we incorporate automated algorithm-level reweighting (`class_weight='balanced'`). The final `RandomForestClassifier` achieves an operational accuracy of over **86%**, successfully meeting the requirements threshold. Please read `report.md` for full implementation logic.
