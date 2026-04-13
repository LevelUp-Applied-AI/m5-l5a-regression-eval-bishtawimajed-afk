import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             classification_report, mean_absolute_error, r2_score)

# --- Task 1: Data Loading ---
def load_data(filepath="data/telecom_churn.csv"):
    possible_paths = [
        filepath,
        os.path.join("starter", filepath),
        "../data/telecom_churn.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"Successfully loaded data from: {path}")
                return df
            except Exception as e:
                print(f"Error reading file at {path}: {e}")
    return None

# --- Task 2: Data Splitting ---
def split_data(df, target_col, test_size=0.2, random_state=42, is_regression=False):
    X = df.drop(columns=[target_col, 'customer_id'], errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    y = df[target_col]
    
    if not is_regression and (y.dtype == "object" or y.nunique() <= 10):
        stratify = y
    else:
        stratify = None
        
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

# --- Task 3: Classification ---
def build_logistic_pipeline(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
    ])
    pipeline.fit(X_train, y_train)
    metrics = evaluate_classifier(pipeline, X_test, y_test)
    return pipeline, metrics

def evaluate_classifier(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

# --- Task 4 & 5: Regression ---
def build_ridge_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(random_state=42))
    ])

def build_lasso_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(random_state=42))
    ])

def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

# --- Task 6: Cross-Validation (التغيير المهم هنا) ---
def run_cross_validation(pipeline, X_train, y_train, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')
    return scores

# --- Main Execution ---
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        # Classification
        X_train_c, X_test_c, y_train_c, y_test_c = split_data(df, 'churned')
        log_pipe, log_metrics = build_logistic_pipeline(X_train_c, X_test_c, y_train_c, y_test_c)
        
        # Regression
        X_train_r, X_test_r, y_train_r, y_test_r = split_data(df, 'monthly_charges', is_regression=True)
        ridge_pipe = build_ridge_pipeline()
        reg_metrics = evaluate_regressor(ridge_pipe, X_train_r, X_test_r, y_train_r, y_test_r)
        
        # Cross Validation
        cv_scores = run_cross_validation(log_pipe, X_train_c, y_train_c)
        print(f"CV Accuracy: {cv_scores.mean():.4f}")

        # --- Tier 1: Threshold Tuning ---
def run_threshold_tuning(pipeline, X_test, y_test):
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\n--- Tier 1: Threshold Tuning Analysis ---")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)
    for thresh in thresholds:
        y_pred_thresh = (y_probs >= thresh).astype(int)
        p = precision_score(y_test, y_pred_thresh, zero_division=0)
        r = recall_score(y_test, y_pred_thresh, zero_division=0)
        f = f1_score(y_test, y_pred_thresh, zero_division=0)
        print(f"{thresh:<10.1f} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f}")

# --- Tier 2: Model Sweep ---
def run_model_sweep(X, y):
    config = [
        {"name": "LR_C0.1", "model": LogisticRegression(C=0.1, max_iter=1000)},
        {"name": "Ridge_A10", "model": Ridge(alpha=10.0)},
        {"name": "Lasso_A1", "model": Lasso(alpha=1.0)}
    ]
    print("\n--- Tier 2: Model Sweep Results ---")
    for item in config:
        pipe = Pipeline([('scaler', StandardScaler()), ('model', item['model'])])
        metric = 'accuracy' if isinstance(item['model'], LogisticRegression) else 'r2'
        try:
            score = cross_val_score(pipe, X, y, cv=3, scoring=metric).mean()
            print(f"Model: {item['name']:<10} | Mean {metric.upper()}: {score:.4f}")
        except:
            continue

# --- Tier 3: Logistic Regression from Scratch ---
class MyLogisticRegression:
    def __init__(self, lr=0.01, iters=1000):
        self.lr, self.iters = lr, iters
        self.w, self.b = None, None
    def _sigmoid(self, z): return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w, self.b = np.zeros(n_features), 0
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        for _ in range(self.iters):
            pred = self._sigmoid(np.dot(X_array, self.w) + self.b)
            self.w -= self.lr * (1/n_samples) * np.dot(X_array.T, (pred - y_array))
            self.b -= self.lr * (1/n_samples) * np.sum(pred - y_array)
    def predict(self, X):
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return [1 if i > 0.5 else 0 for i in self._sigmoid(np.dot(X_array, self.w) + self.b)]

def run_scratch_comparison(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = MyLogisticRegression(lr=0.1, iters=1500)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    print(f"\n--- Tier 3: Scratch Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    """
SUMMARY OF THE ISSUE AND FIX:
I encountered errors during the autograder tests mainly due to path mismatches. 
The tests expected the dataset and module to exist under the starter/ directory, 
while my project structure had them in different locations.

To fix this, I updated the load_data() function to handle multiple possible paths, 
including both data/telecom_churn.csv and starter/data/telecom_churn.csv. 
I also made the code more flexible to ensure compatibility with the autograder 
environment. After these changes, all local tests passed successfully.
"""