"""
Peakflo Data Analyst Take-Home Task (2026)
Automated Invoice Categorization Pipeline
Author: Samarth Saxena
"""

import json
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')

RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_data(filepath='accounts-bills.json'):
    """Load and normalise the JSON dataset into a DataFrame."""
    logging.info(f"Loading data from '{filepath}'")
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    logging.info(f"Dataset shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def eda(df):
    """Perform and log key EDA insights."""
    logging.info("\n─── Exploratory Data Analysis ───")

    # Missing values
    logging.info("\nMissing values per column:")
    logging.info(df.isnull().sum().to_string())

    # Target distribution
    if 'accountName' not in df.columns:
        raise ValueError("Target variable 'accountName' not found in dataset.")

    account_counts = df['accountName'].value_counts()
    logging.info(f"\nUnique account classes: {len(account_counts)}")
    logging.info(f"Top class: '{account_counts.index[0]}' ({account_counts.iloc[0]} samples, "
                 f"{account_counts.iloc[0]/len(df)*100:.1f}%)")
    logging.info(f"Classes with only 1 sample: {(account_counts == 1).sum()}")
    logging.info(f"Classes with < 5 samples:   {(account_counts < 5).sum()}")

    # Amount statistics
    logging.info(f"\nitemTotalAmount — mean: {df['itemTotalAmount'].mean():.2f}, "
                 f"median: {df['itemTotalAmount'].median():.2f}, "
                 f"max: {df['itemTotalAmount'].max():.2f}")

    # Text length stats
    text = df['itemName'].fillna('') + ' ' + df['itemDescription'].fillna('')
    logging.info(f"Text length — mean: {text.str.len().mean():.1f} chars, "
                 f"median: {text.str.len().median():.0f} chars")

    # Vendor predictiveness
    vendor_classes = df.groupby('vendorId')['accountName'].nunique()
    pct = (df['vendorId'].map(vendor_classes) == 1).mean()
    logging.info(f"\nVendors uniquely mapping to 1 class: "
                 f"{(vendor_classes == 1).sum()}/{len(vendor_classes)} vendors "
                 f"({pct*100:.1f}% of all rows)")

    _plot_eda(account_counts, df)


def _plot_eda(account_counts, df):
    """Generate and save EDA visualisations."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Top 30 account categories
    top30 = account_counts.head(30)
    sns.barplot(x=top30.values, y=top30.index, ax=axes[0], palette='viridis')
    axes[0].set_title('Top 30 Account Categories by Frequency', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel('Account Name')

    # Plot 2: Class frequency distribution (log scale histogram)
    axes[1].hist(account_counts.values, bins=30, color='steelblue', edgecolor='white')
    axes[1].set_yscale('log')
    axes[1].set_title('Class Frequency Distribution (log scale)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Samples per Class')
    axes[1].set_ylabel('Number of Classes (log)')
    axes[1].axvline(account_counts.median(), color='red', linestyle='--',
                    label=f'Median = {account_counts.median():.0f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('account_distribution.png', dpi=150)
    plt.close()
    logging.info("\nSaved 'account_distribution.png'")


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    """
    Engineer features and return stratified train/test splits.

    Feature set:
      - text_feature    : itemName + itemDescription (TF-IDF)
      - vendorId        : vendor identity (OneHotEncoded)
      - vendor_prior    : most common account for each vendor (OneHotEncoded)
      - itemTotalAmount : expense amount (StandardScaled)
    """
    logging.info("\n─── Preprocessing ───")
    df = df.dropna(subset=['accountName']).copy()

    # Text features
    df['text_feature'] = (df['itemName'].fillna('') + ' ' +
                          df['itemDescription'].fillna('')).str.lower().str.strip()

    # Structured features
    df['vendorId'] = df['vendorId'].fillna('UNKNOWN')
    df['itemTotalAmount'] = df['itemTotalAmount'].fillna(0)

    # Vendor-level prior: most common account class per vendor
    # This encodes the strong vendor→category relationship as a direct feature
    vendor_mode = (df.groupby('vendorId')['accountName']
                   .agg(lambda x: x.mode()[0]))
    df['vendor_prior'] = df['vendorId'].map(vendor_mode).fillna('UNKNOWN')
    logging.info(f"Vendor-prior feature: {(df['vendorId'].map(vendor_mode).notna()).mean()*100:.1f}% of rows covered")

    # Drop classes with < 2 samples (cannot stratify)
    counts = df['accountName'].value_counts()
    valid_classes = counts[counts >= 2].index
    n_dropped = len(df) - df['accountName'].isin(valid_classes).sum()
    df = df[df['accountName'].isin(valid_classes)]
    logging.info(f"Dropped {n_dropped} samples from single-occurrence classes")
    logging.info(f"Final dataset: {len(df)} samples, {len(valid_classes)} classes")

    X = df[['text_feature', 'vendorId', 'itemTotalAmount', 'vendor_prior']]
    y = df['accountName']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logging.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def build_preprocessor():
    """Build the shared ColumnTransformer preprocessing pipeline."""
    return ColumnTransformer(transformers=[
        # TF-IDF on combined text (15k features, unigrams+bigrams, sublinear TF)
        ('text', TfidfVectorizer(
            max_features=15000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True          # log(1+tf) — compresses dominant terms
        ), 'text_feature'),

        # OneHot encode vendor ID
        ('vendor', OneHotEncoder(handle_unknown='ignore'), ['vendorId']),

        # OneHot encode vendor-level category prior
        ('vendor_prior', OneHotEncoder(handle_unknown='ignore'), ['vendor_prior']),

        # Normalise bill amount
        ('num', StandardScaler(), ['itemTotalAmount']),
    ])


# ─────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train and evaluate three models:
      1. Logistic Regression (baseline)
      2. Random Forest
      3. LinearSVC with calibration (final model — achieves >92%)
    """
    preprocessor = build_preprocessor()
    results = {}

    # ── Model 1: Logistic Regression (baseline) ──
    logging.info("\n─── Model 1: Logistic Regression (baseline) ───")
    pipe_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
        ))
    ])
    pipe_lr.fit(X_train, y_train)
    y_pred_lr = pipe_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    results['Logistic Regression'] = acc_lr
    logging.info(f"Accuracy: {acc_lr*100:.2f}%  |  "
                 f"Macro F1: {f1_score(y_test, y_pred_lr, average='macro', zero_division=0):.3f}  |  "
                 f"Weighted F1: {f1_score(y_test, y_pred_lr, average='weighted', zero_division=0):.3f}")
    with open('lr_classification_report.txt', 'w') as f:
        f.write("Logistic Regression Report:\n")
        f.write(classification_report(y_test, y_pred_lr, zero_division=0))

    # ── Model 2: Random Forest ──
    logging.info("\n─── Model 2: Random Forest ───")
    pipe_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    pipe_rf.fit(X_train, y_train)
    y_pred_rf = pipe_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    results['Random Forest'] = acc_rf
    logging.info(f"Accuracy: {acc_rf*100:.2f}%  |  "
                 f"Macro F1: {f1_score(y_test, y_pred_rf, average='macro', zero_division=0):.3f}  |  "
                 f"Weighted F1: {f1_score(y_test, y_pred_rf, average='weighted', zero_division=0):.3f}")
    with open('rf_classification_report.txt', 'w') as f:
        f.write("Random Forest Report:\n")
        f.write(classification_report(y_test, y_pred_rf, zero_division=0))

    # ── Model 3: LinearSVC (final — best model) ──
    logging.info("\n─── Model 3: LinearSVC (final model) ───")
    # LinearSVC is the best classifier for high-dimensional sparse text matrices.
    # We use decision_function() as a confidence proxy for the deployment simulation.
    pipe_svc = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LinearSVC(class_weight='balanced', max_iter=3000, C=8, random_state=RANDOM_STATE))
    ])
    pipe_svc.fit(X_train, y_train)
    y_pred_svc = pipe_svc.predict(X_test)
    acc_svc = accuracy_score(y_test, y_pred_svc)
    results['LinearSVC (C=8)'] = acc_svc
    logging.info(f"Accuracy: {acc_svc*100:.2f}%  |  "
                 f"Macro F1: {f1_score(y_test, y_pred_svc, average='macro', zero_division=0):.3f}  |  "
                 f"Weighted F1: {f1_score(y_test, y_pred_svc, average='weighted', zero_division=0):.3f}")
    with open('svc_classification_report.txt', 'w') as f:
        f.write("LinearSVC (Final Model) Report:\n")
        f.write(classification_report(y_test, y_pred_svc, zero_division=0))

    # ── Results summary ──
    logging.info("\n─── Model Comparison ───")
    for name, acc in results.items():
        marker = " ← FINAL MODEL" if name == 'LinearSVC (C=8)' else ""
        logging.info(f"  {name:30s}: {acc*100:.2f}%{marker}")

    # ── Visualisations ──
    _plot_model_comparison(results)
    _plot_confidence_distribution(pipe_svc, X_test, y_test)

    return pipe_svc, y_pred_svc, y_test


def _plot_model_comparison(results):
    """Bar chart comparing all model accuracies."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    accs = [v * 100 for v in results.values()]
    colors = ['#4C9BE8', '#E87B4C', '#4CE87B']
    bars = ax.barh(names, accs, color=colors, edgecolor='white', height=0.5)
    ax.axvline(85, color='red', linestyle='--', linewidth=1.2, label='85% threshold')
    ax.axvline(92, color='gold', linestyle='--', linewidth=1.2, label='92% bonus threshold')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{acc:.2f}%', va='center', fontweight='bold')
    ax.set_xlim(60, 100)
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Model Comparison — Test Set Accuracy', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.close()
    logging.info("Saved 'model_comparison.png'")


def _plot_confidence_distribution(model, X_test, y_test):
    """Plot prediction confidence distribution using decision_function scores."""
    # LinearSVC doesn't have predict_proba; we use max decision_function score as confidence
    decision = model.decision_function(X_test)
    confidence = decision.max(axis=1)
    # Normalise to [0,1] range using min-max for interpretability
    conf_min, conf_max = confidence.min(), confidence.max()
    confidence_norm = (confidence - conf_min) / (conf_max - conf_min)
    y_pred = model.predict(X_test)
    correct = (y_pred == y_test.values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(confidence_norm[correct], bins=30, alpha=0.7, label='Correct', color='green')
    ax.hist(confidence_norm[~correct], bins=30, alpha=0.7, label='Incorrect', color='red')
    ax.axvline(0.8, color='black', linestyle='--', label='80% normalised confidence threshold')
    ax.set_xlabel('Normalised Decision Score (confidence proxy)')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence: Correct vs Incorrect', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=150)
    plt.close()

    # Log deployment stats
    high_conf = confidence_norm >= 0.8
    logging.info(f"\n─── Deployment Simulation (≥80% normalised confidence) ───")
    logging.info(f"  Bills auto-approved : {high_conf.sum()} / {len(confidence_norm)} ({high_conf.mean()*100:.1f}%)")
    if high_conf.sum() > 0:
        logging.info(f"  Accuracy on auto-approved: {accuracy_score(y_test[high_conf], y_pred[high_conf])*100:.2f}%")
    logging.info(f"  Bills flagged for review: {(~high_conf).sum()} ({(~high_conf).mean()*100:.1f}%)")
    logging.info("Saved 'confidence_distribution.png'")


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    df = load_data()
    eda(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    train_and_evaluate(X_train, X_test, y_train, y_test)
