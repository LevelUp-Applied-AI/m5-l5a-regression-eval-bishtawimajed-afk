import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Task 1 & 2: Load and Split
def load_and_explore(filepath):
    df = pd.read_csv(filepath)
    return df

def split_data(df, target_column, is_regression=False):
    X = df.drop(columns=[target_column, 'customer_id'], errors='ignore')
    y = df[target_column]
    X = pd.get_dummies(X, drop_first=True)
    strat = y if not is_regression else None
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

# Task 3: Logistic Regression
def run_logistic_pipeline(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    return pipeline, metrics

# Task 4 & 5: Ridge and Lasso Comparison
def run_regression_comparison(X_train, X_test, y_train, y_test):
    ridge_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ])
    ridge_pipe.fit(X_train, y_train)
    
    lasso_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=0.1))
    ])
    lasso_pipe.fit(X_train, y_train)
    
    ridge_coefs = ridge_pipe.named_steps['model'].coef_
    lasso_coefs = lasso_pipe.named_steps['model'].coef_
    features = X_train.columns
    
    print("\n--- Task 5: Feature Coefficients Comparison ---")
    print(f"{'Feature':<30} | {'Ridge':<10} | {'Lasso':<10}")
    print("-" * 55)
    for feat, r, l in zip(features, ridge_coefs, lasso_coefs):
        print(f"{feat:<30} | {r:>10.4f} | {l:>10.4f}")
    
    return ridge_pipe, lasso_pipe

# Task 6: Cross-Validation
def run_cross_validation(pipeline, X, y):
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring="accuracy")
    
    print("\n--- Task 6: Cross-Validation Results (Accuracy) ---")
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: {score:.4f}")
    print(f"Mean Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores

# --- CHALLENGE EXTENSIONS ---

# Tier 1: Threshold Tuning
def run_threshold_tuning(pipeline, X_test, y_test):
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    precision_list, recall_list, f1_list = [], [], []

    print("\n--- Tier 1: Threshold Tuning Analysis ---")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)

    for thresh in thresholds:
        y_pred_thresh = (y_probs >= thresh).astype(int)
        p = precision_score(y_test, y_pred_thresh)
        r = recall_score(y_test, y_pred_thresh)
        f = f1_score(y_test, y_pred_thresh)
        
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)
        print(f"{thresh:<10.1f} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precision_list, label='Precision', marker='o')
    plt.plot(thresholds, recall_list, label='Recall', marker='o')
    plt.plot(thresholds, f1_list, label='F1-Score', linestyle='--', color='black', marker='s')
    plt.title('Precision-Recall Trade-off')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()

# Tier 2: Config-Driven Model Sweep
def run_model_sweep(X, y):
    config = {
        "models": [
            {"type": "LogisticRegression", "params": {"C": 0.1, "solver": "liblinear"}},
            {"type": "LogisticRegression", "params": {"C": 1.0, "penalty": "l2"}},
            {"type": "Ridge", "params": {"alpha": 0.1}},
            {"type": "Ridge", "params": {"alpha": 10.0}},
            {"type": "Lasso", "params": {"alpha": 0.01}},
            {"type": "Lasso", "params": {"alpha": 1.0}}
        ]
    }

    sweep_results = []
    for m_cfg in config["models"]:
        if m_cfg["type"] == "LogisticRegression":
            model = LogisticRegression(**m_cfg["params"], max_iter=1000)
            metric = 'accuracy'
        else:
            model = Ridge(**m_cfg["params"]) if m_cfg["type"] == "Ridge" else Lasso(**m_cfg["params"])
            metric = 'r2'
            
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        cv = cross_validate(pipe, X, y, cv=5, scoring=metric)
        sweep_results.append({
            "Model": m_cfg["type"],
            "Params": str(m_cfg["params"]),
            "Score": cv['test_score'].mean()
        })

    print("\n--- Tier 2: Config-Driven Sweep Results ---")
    print(pd.DataFrame(sweep_results))

# Tier 3: Logistic Regression from Scratch
class MyLogisticRegression:
    def __init__(self, lr=0.01, iters=1000, penalty=0.1):
        self.lr = lr
        self.iters = iters
        self.penalty = penalty
        self.w, self.b = None, None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w, self.b = np.zeros(n_features), 0
        for _ in range(self.iters):
            linear_pred = np.dot(X, self.w) + self.b
            predictions = self._sigmoid(linear_pred)
            dw = (1/n_samples) * np.dot(X.T, (predictions - y)) + (self.penalty/n_samples) * self.w
            db = (1/n_samples) * np.sum(predictions - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return [1 if i > 0.5 else 0 for i in self._sigmoid(np.dot(X, self.w) + self.b)]

def run_scratch_comparison(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    scratch_model = MyLogisticRegression(lr=0.1, iters=1500)
    scratch_model.fit(X_train_s, y_train)
    y_pred = scratch_model.predict(X_test_s)

    print("\n--- Tier 3: Manual Implementation Accuracy ---")
    print(f"Scratch Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Execution
if __name__ == "__main__":
    df = load_and_explore('data/telecom_churn.csv')
    
    # Classification Tasks
    X_train_c, X_test_c, y_train_c, y_test_c = split_data(df, 'churned')
    log_pipe, log_metrics = run_logistic_pipeline(X_train_c, X_test_c, y_train_c, y_test_c)
    
    # Regression Tasks
    X_train_r, X_test_r, y_train_r, y_test_r = split_data(df, 'monthly_charges', is_regression=True)
    ridge_p, lasso_p = run_regression_comparison(X_train_r, X_test_r, y_train_r, y_test_r)
    
    # Task 6: CV
    run_cross_validation(log_pipe, X_train_c, y_train_c)

    # Challenge Execution
    X_test_c_enc = pd.get_dummies(X_test_c, drop_first=True).reindex(columns=X_train_c.columns, fill_value=0)
    
    run_threshold_tuning(log_pipe, X_test_c_enc, y_test_c)
    run_model_sweep(X_train_c, y_train_c)
    run_scratch_comparison(X_train_c, X_test_c_enc, y_train_c, y_test_c)

"""
Task 7: Summary of Findings

1. Important Features: Tenure and contract type are key for churn prediction.
2. Performance: Logistic Regression shows moderate accuracy (61%); Recall is prioritized.
3. Recommendations: Try Random Forest or advanced feature engineering.
"""