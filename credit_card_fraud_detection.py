
"""
Credit Card Fraud Detection
- Binary classification (fraud=1, legit=0). Default target column guesses: 'Class' or 'fraud'.
- Handles class imbalance via class_weight='balanced'.
- Tries Logistic Regression and Random Forest, selects best by PR-AUC on validation.

"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def infer_columns(df, target_col):
    if target_col is None:
        for guess in ["Class", "class", "Fraud", "fraud", "is_fraud"]:
            if guess in df.columns:
                target_col = guess
                break
    if target_col is None or target_col not in df.columns:
        raise ValueError(f"Target column not found. Available: {list(df.columns)}")
    X_cols = [c for c in df.columns if c != target_col]
    cat_cols = [c for c in X_cols if df[c].dtype == "object"]
    num_cols = [c for c in X_cols if c not in cat_cols]
    return target_col, num_cols, cat_cols

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def train_and_eval(X_train, y_train, X_test, y_test, preprocessor):
    models = {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=42
        )
    }
    results = {}
    best_model = None
    best_name = None
    best_score = -1.0

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        probas = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else None
        preds = pipe.predict(X_test)

        roc = roc_auc_score(y_test, probas) if probas is not None else np.nan
        pr = average_precision_score(y_test, probas) if probas is not None else np.nan
        results[name] = {"roc_auc": roc, "pr_auc": pr, "pipe": pipe}

        # Select by PR-AUC primarily (better for imbalance); fallback to ROC-AUC
        score_for_sel = (pr if not np.isnan(pr) else roc)
        if score_for_sel > best_score:
            best_score = score_for_sel
            best_name = name
            best_model = pipe

    return best_name, best_model, results

def main(args):
    df = pd.read_csv(args.csv)
    target_col, num_cols, cat_cols = infer_columns(df, args.target_col)

    X = df[num_cols + cat_cols]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)
    best_name, best_model, results = train_and_eval(X_train, y_train, X_test, y_test, preprocessor)

    print("=== Model Comparison (on test set) ===")
    for name, r in results.items():
        print(f"{name}: PR-AUC={r['pr_auc']:.4f}, ROC-AUC={r['roc_auc']:.4f}")

    preds = best_model.predict(X_test)
    probas = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model.named_steps["clf"], "predict_proba") else None

    print("\n=== Best Model:", best_name, "===")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    if probas is not None:
        roc = roc_auc_score(y_test, probas)
        pr = average_precision_score(y_test, probas)
        print(f"ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}")

    # Save model
    out_path = args.model_out
    meta = {
        "best_model": best_name,
        "target_col": target_col,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "test_size": args.test_size
    }
    joblib.dump({"pipeline": best_model, "meta": meta}, out_path)
    print(f"\nSaved model to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV with features and target.")
    parser.add_argument("--target_col", default=None, help="Target column name. Defaults to a smart guess (Class/Fraud/etc.).")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--model_out", default="fraud_model.joblib")
    args = parser.parse_args()
    main(args)
