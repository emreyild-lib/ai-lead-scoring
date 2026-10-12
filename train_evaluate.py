# -*- coding: utf-8 -*-
"""
Train + evaluate an AI lead-scoring model, then save a production pipeline.

INPUT: leads_history.csv  (one row per lead/opportunity)
Required columns:
- lead_id : unique id
- won     : 1 if won, 0 if lost
Other columns become features. Numeric and categorical are auto-detected.

USAGE:
python train_evaluate.py --in leads_history.csv --out lead_scoring_model.pkl
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt


def infer_feature_types(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    # drop obvious IDs from features if present
    for drop in ["lead_id", "account_id", "opportunity_id"]:
        if drop in cat_cols:
            cat_cols.remove(drop)
        if drop in num_cols:
            num_cols.remove(drop)
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return pre


def build_models():
    logreg = Pipeline(steps=[
        ("clf", LogisticRegression(max_iter=500, n_jobs=None, solver="lbfgs"))
    ])
    rf = Pipeline(steps=[
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
        ))
    ])
    return {"logreg": logreg, "rf": rf}


def evaluate(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, proba)
    ap = average_precision_score(y_val, proba)  # PR-AUC
    return {"roc_auc": float(roc), "pr_auc": float(ap)}, proba


def plot_curves(y_true, proba, out_prefix="eval"):
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)

    plt.figure(figsize=(4,3))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_roc.png", dpi=140); plt.close()

    plt.figure(figsize=(4,3))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_pr.png", dpi=140); plt.close()


def main(args):
    df = pd.read_csv(args.infile)
    if "won" not in df.columns:
        raise ValueError("Missing required column 'won' (0/1).")

    num_cols, cat_cols = infer_feature_types(df, target_col="won")
    pre = build_preprocessor(num_cols, cat_cols)

    X = df.drop(columns=["won"])
    y = df["won"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = build_models()
    results = {}
    best_name, best_score, best_pipeline = None, -1.0, None

    for name, base in models.items():
        pipe = Pipeline(steps=[("prep", pre), (name, base["clf"] if isinstance(base, dict) else base)])
        # handle the dict vs pipeline from build_models()
        if isinstance(base, dict):
            clf = base["clf"]
            pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])
        else:
            pipe = Pipeline(steps=[("prep", pre), (name, base)])

        pipe.fit(X_train, y_train)
        metrics, proba = evaluate(pipe, X_val, y_val)
        results[name] = metrics
        if metrics["roc_auc"] > best_score:
            best_name, best_score, best_pipeline = name, metrics["roc_auc"], pipe
            best_proba = proba

    # curves
    plot_curves(y_val, best_proba, out_prefix="eval")

    # permutation importance on validation set (slow but model-agnostic)
    prepped = best_pipeline.named_steps["prep"].transform(X_val)
    clf_step_name = [k for k in best_pipeline.named_steps.keys() if k != "prep"][0]
    clf = best_pipeline.named_steps[clf_step_name]

    # build feature names after OHE
    ohe = best_pipeline.named_steps["prep"].named_transformers_["cat"].named_steps["oh"]
    cat_feature_names = list(ohe.get_feature_names_out())
    feature_names = list(num_cols) + cat_feature_names

    try:
        imp = permutation_importance(clf, prepped, y_val, n_repeats=5, random_state=42, n_jobs=-1)
        fi = pd.DataFrame({"feature": feature_names, "importance": imp.importances_mean}).sort_values(
            "importance", ascending=False)
        fi.to_csv("feature_importance.csv", index=False)
    except Exception:
        pass  # some classifiers may not work cleanly with permutation_importance on sparse inputs

    # save final pipeline retrained on all data
    best_pipeline.fit(X, y)
    joblib.dump({
        "pipeline": best_pipeline,
        "features_num": num_cols,
        "features_cat": cat_cols,
        "results": results,
        "best_model": best_name
    }, args.outfile)

    Path("training_summary.json").write_text(json.dumps({
        "results": results, "best_model": best_name, "val_roc_auc": best_score
    }, indent=2))

    print(f"Saved model → {args.outfile}")
    print("Validation curves: eval_roc.png, eval_pr.png")
    print("Feature importance: feature_importance.csv (if computed).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True, help="leads_history.csv")
    p.add_argument("--out", dest="outfile", default="lead_scoring_model.pkl")
    args = p.parse_args()
    main(args)
