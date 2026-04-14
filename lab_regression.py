"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, r2_score, recall_score)



def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
    # TODO: Load the CSV and return the DataFrame
    df = pd.read_csv(filepath)
    return df


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # TODO: Separate features and target, then split with stratification
    X=df.drop(columns=[target_col])
    y=df[target_col]
    # إذا classification (binary)
    if y.nunique() <= 2:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    # إذا regression
    else:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    # TODO: Create and return a Pipeline with two steps
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    return pipe


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    # TODO: Create and return a Pipeline for Ridge regression

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    return pipe



def build_lasso_pipeline():
    """Build a Pipeline with StandardScaler and Lasso regression."""
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))
    ])
    
    return pipe    

def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    # TODO: Fit the pipeline on training data, predict on test, compute metrics
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }
  

def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    # TODO: Fit the pipeline, predict, and compute MAE and R²
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
   


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    # TODO: Run cross_val_score with StratifiedKFold
    cv_splitter = StratifiedKFold(
    n_splits=cv,
    shuffle=True,
    random_state=42
    )
#Task 6    
#CV: 0.607 +/- 0.019
#0.607 → متوسط الدقة (accuracy) عبر 5 folds
#± 0.019 → التذبذب (variance) بين الفولدز

# Cross-validation results show that the model achieves an average accuracy of ~0.61
# with low variance (+/- 0.019), indicating consistent performance across folds.
# However, the accuracy is relatively moderate, which may be due to class imbalance
# in the churn dataset (~16% churn rate), making accuracy less reliable.

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="accuracy"
    )

    return scores
# ==========================================
# --- CHALLENGE EXTENSIONS: Tiers 1, 2, 3 ---
# ==========================================

def run_threshold_tuning(pipeline, X_test, y_test):
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    precision_list, recall_list, f1_list = [], [], []

    print("\n--- Tier 1: Threshold Tuning Analysis ---")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)

    for thresh in thresholds:
        y_pred_thresh = (y_probs >= thresh).astype(int)
        p = precision_score(y_test, y_pred_thresh, zero_division=0)
        r = recall_score(y_test, y_pred_thresh, zero_division=0)
        f = f1_score(y_test, y_pred_thresh, zero_division=0)
        
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

def run_model_sweep(X, y):
    config = {
        "models": [
            {"type": "LogisticRegression", "params": {"C": 0.1, "solver": "liblinear"}},
            {"type": "Ridge", "params": {"alpha": 10.0}},
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
        cv = cross_val_score(pipe, X, y, cv=5, scoring=metric)
        sweep_results.append({"Model": m_cfg["type"], "Params": str(m_cfg["params"]), "Mean Score": cv.mean()})
    
    print("\n--- Tier 2: Config-Driven Sweep Results ---")
    print(pd.DataFrame(sweep_results))

class MyLogisticRegression:
    def __init__(self, lr=0.01, iters=1000):
        self.lr, self.iters = lr, iters
        self.w, self.b = None, None
    def _sigmoid(self, z): return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w, self.b = np.zeros(n_features), 0
        X_arr, y_arr = X.values, y.values
        for _ in range(self.iters):
            model_out = np.dot(X_arr, self.w) + self.b
            preds = self._sigmoid(model_out)
            self.w -= self.lr * (1/n_samples) * np.dot(X_arr.T, (preds - y_arr))
            self.b -= self.lr * (1/n_samples) * np.sum(preds - y_arr)
    def predict(self, X):
        return [1 if i > 0.5 else 0 for i in self._sigmoid(np.dot(X.values, self.w) + self.b)]

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Select numeric features for classification
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                           "num_support_calls", "senior_citizen",
                           "has_partner", "has_dependents"]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            if pipe:
                pipe.fit(X_train, y_train)#تدريب الموديل
                y_pred = pipe.predict(X_test)#التنبؤ
                #احسب الميتريكس باستخدام الفنكشن evaluate_classifier
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            if ridge_pipe:
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                print(f"Ridge Regression: {reg_metrics}")
            lasso_pipe = build_lasso_pipeline()

            if lasso_pipe:
                lasso_pipe.fit(X_tr, y_tr)
                ridge_pipe.fit(X_tr, y_tr)

                ridge_coef = ridge_pipe.named_steps["model"].coef_
                lasso_coef = lasso_pipe.named_steps["model"].coef_

                feature_names = X_tr.columns

                print("\nFeature Coefficients Comparison:")
                for name, r_coef, l_coef in zip(feature_names, ridge_coef, lasso_coef):
                    print(f"{name:20} | Ridge: {r_coef:.4f} | Lasso: {l_coef:.4f}")    
# In this case, Lasso did not drive any feature coefficients to zero.
# This suggests that all features contribute to predicting monthly charges.
# It may also indicate that the regularization strength (alpha=0.1) is not strong enough
# to eliminate less important features.
# To see more sparsity, we could try increasing alpha or using a different dataset with more irrelevant features. 
    ##print(df.head())            
    # --- تشغيل الشالينج ---
        print("\n" + "="*30 + "\nRUNNING CHALLENGE TIERS\n" + "="*30)
        
        # Tier 1: Tuning
        run_threshold_tuning(pipe, X_test, y_test)
        
        # Tier 2: Sweep
        run_model_sweep(X_train, y_train)
        
        # Tier 3: Scratch Comparison
        # بنستخدم StandardScaler عشان السكراتش موديل يشتغل صح
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        scratch = MyLogisticRegression(lr=0.1, iters=1000)
        scratch.fit(X_train_scaled, y_train)
        s_preds = scratch.predict(X_test_scaled)
        print(f"\nTier 3: Scratch Accuracy: {accuracy_score(y_test, s_preds):.4f}")



"""
Summary of Findings

1. Which features appear most important for predicting churn?
Based on the model coefficients, features such as total_charges and tenure appear to have the strongest influence on predicting churn. Other features like number of support calls and customer demographics also contribute, but to a lesser extent.

2. Model Performance:
The logistic regression model achieved moderate performance with an accuracy of around 0.61. However, due to class imbalance in the dataset, accuracy is not the most reliable metric. The recall score is particularly important, as it reflects the model's ability to correctly identify customers who are likely to churn.

3. Key Concern:
Recall is more critical than precision in this problem because failing to identify a customer who will churn (false negative) can result in lost revenue. Therefore, improving recall should be prioritized.

4. Recommendations for Improvement:
- Try different models such as Random Forest or Gradient Boosting
- Tune hyperparameters (e.g., regularization strength)
- Apply feature engineering to create more informative variables
- Use resampling techniques such as SMOTE to address class imbalance
- Evaluate using additional metrics such as ROC-AUC

Overall, the model provides a solid baseline but can be improved with more advanced techniques and better handling of imbalanced data.
"""