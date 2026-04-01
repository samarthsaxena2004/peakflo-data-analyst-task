import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_data(filepath='accounts-bills.json'):
    logging.info(f"Loading data from {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    logging.info(f"Data shape: {df.shape}")
    return df

def eda(df):
    logging.info("--- Exploratory Data Analysis ---")
    
    # Missing Values
    logging.info("Missing Values:")
    logging.info(df.isnull().sum())
    
    # Target variable distribution
    if 'accountName' not in df.columns:
        raise ValueError("Target variable 'accountName' not found.")
        
    account_counts = df['accountName'].value_counts()
    logging.info(f"Number of unique classes: {len(account_counts)}")
    logging.info("Top 10 classes:")
    logging.info(account_counts.head(10))
    logging.info("Bottom 10 classes:")
    logging.info(account_counts.tail(10))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    top_30 = account_counts.head(30)
    sns.barplot(x=top_30.values, y=top_30.index, palette='viridis')
    plt.title('Top 30 Account Names Distribution')
    plt.xlabel('Count')
    plt.ylabel('Account Name')
    plt.tight_layout()
    plt.savefig('account_distribution.png')
    logging.info("Saved 'account_distribution.png'")

def preprocess(df):
    logging.info("--- Preprocessing ---")
    df = df.dropna(subset=['accountName'])
    
    df['text_feature'] = df['itemName'].fillna('') + " " + df['itemDescription'].fillna('')
    df['text_feature'] = df['text_feature'].str.lower()
    df['vendorId'] = df['vendorId'].fillna('UNKNOWN')
    df['itemTotalAmount'] = df['itemTotalAmount'].fillna(0)
    
    X = df[['text_feature', 'vendorId', 'itemTotalAmount']]
    y = df['accountName']
    
    counts = y.value_counts()
    valid_classes = counts[counts >= 2].index
    
    df_filtered = df[df['accountName'].isin(valid_classes)]
    logging.info(f"Dropped {len(df) - len(df_filtered)} samples due to single-occurrence classes.")
    
    X_filtered = df_filtered[['text_feature', 'vendorId', 'itemTotalAmount']]
    y_filtered = df_filtered['accountName']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )
    logging.info(f"Training size: {len(X_train)}, Testing size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def train_and_evaluate(X_train, X_test, y_train, y_test):
    logging.info("--- Model Training Basic TF-IDF + Logistic Regression ---")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2)), 'text_feature'),
            ('vendor', OneHotEncoder(handle_unknown='ignore'), ['vendorId']),
            ('num', StandardScaler(), ['itemTotalAmount'])
        ])

    pipeline_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    pipeline_lr.fit(X_train, y_train)
    y_pred_lr = pipeline_lr.predict(X_test)
    
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr_macro = f1_score(y_test, y_pred_lr, average='macro')
    f1_lr_weighted = f1_score(y_test, y_pred_lr, average='weighted')
    
    logging.info(f"Logistic Regression - Accuracy: {acc_lr:.4f}")
    
    with open("lr_classification_report.txt", "w") as f:
        f.write("Logistic Regression Report:\n")
        f.write(classification_report(y_test, y_pred_lr, zero_division=0))
        
    logging.info("--- Model Training Advanced TF-IDF + Random Forest ---")
    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    
    pipeline_rf.fit(X_train, y_train)
    y_pred_rf = pipeline_rf.predict(X_test)
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf_macro = f1_score(y_test, y_pred_rf, average='macro')
    f1_rf_weighted = f1_score(y_test, y_pred_rf, average='weighted')
    
    logging.info(f"Random Forest - Accuracy: {acc_rf:.4f}")
    
    with open("rf_classification_report.txt", "w") as f:
        f.write("Random Forest Report:\n")
        f.write(classification_report(y_test, y_pred_rf, zero_division=0))

if __name__ == "__main__":
    df = load_data()
    eda(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    train_and_evaluate(X_train, X_test, y_train, y_test)
