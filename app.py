# -*- coding: utf-8 -*-
"""
Streamlit app: train and/or score leads.

Run:
streamlit run app.py

Tabs:
- Train: upload historical CSV (with 'won'), train, see metrics, download model.
- Score: upload model + new leads CSV (without 'won'), get scores and download.
"""

import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="AI Lead Scoring", layout="wide")

st.title("AI-Powered Lead Scoring")

def infer_feature_types(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    for drop in ["lead_id", "account_id", "opportunity_id"]:
        if drop in cat_cols: cat_cols.remove(drop)
        if drop in num_cols: num_cols.remove(drop)
    return num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=500, solver="lbfgs"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    }

def eval_curves(y, proba):
    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)
    fig1, ax1 = plt.subplots(figsize=(4,3))
    ax1.plot(fpr, tpr); ax1.plot([0,1],[0,1],"--")
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC")
    fig2, ax2 = plt.subplots(figsize=(4,3))
    ax2.plot(rec, prec)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision-Recall")
    return fig1, fig2

tab_train, tab_score = st.tabs(["Train model", "Score leads"])

with tab_train:
    st.subheader("Upload historical leads CSV (must include 'won' 0/1)")
    hist_file = st.file_uploader("leads_history.csv", type=["csv"], key="hist")
    if hist_file is not None:
        df = pd.read_csv(hist_file)
        if "won" not in df.columns:
            st.error("Missing required column 'won'.")
        else:
            st.write("Preview", df.head())

            num_cols, cat_cols = infer_feature_types(df, "won")
            pre = build_preprocessor(num_cols, cat_cols)
            X, y = df.drop(columns=["won"]), df["won"].astype(int)
            Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            model_choice = st.selectbox("Model", list(build_models().keys()))
            mdl = build_models()[model_choice]
            pipe = Pipeline([("prep", pre), ("clf", mdl)])
            train_btn = st.button("Train")
            if train_btn:
                pipe.fit(Xtr, ytr)
                proba = pipe.predict_proba(Xva)[:,1]
                roc = roc_auc_score(yva, proba)
                prauc = average_precision_score(yva, proba)

                c1, c2, c3 = st.columns(3)
                c1.metric("ROC AUC", f"{roc:.3f}")
                c2.metric("PR AUC", f"{prauc:.3f}")
                c3.metric("Val size", f"{len(yva)}")

                fig_roc, fig_pr = eval_curves(yva, proba)
                st.pyplot(fig_roc); st.pyplot(fig_pr)

                # Save model in memory for download
                payload = {
                    "pipeline": pipe,
                    "features_num": num_cols,
                    "features_cat": cat_cols,
                    "model": model_choice
                }
                buf = io.BytesIO()
                joblib.dump(payload, buf)
                st.download_button("Download trained model (.pkl)", buf.getvalue(), file_name="lead_scoring_model.pkl")

with tab_score:
    st.subheader("Upload model (.pkl) and new leads CSV (no 'won')")
    mdl_file = st.file_uploader("lead_scoring_model.pkl", type=["pkl"], key="mdl")
    new_file = st.file_uploader("new_leads.csv", type=["csv"], key="new")
    top_n = st.number_input("Show top-N leads", min_value=5, max_value=500, value=50, step=5)

    if mdl_file and new_file:
        model_obj = joblib.load(mdl_file)
        pipe = model_obj["pipeline"]
        df_new = pd.read_csv(new_file)
        if "won" in df_new.columns:
            st.warning("Column 'won' found. It will be ignored for scoring.")
            df_new = df_new.drop(columns=["won"])
        proba = pipe.predict_proba(df_new)[:,1]
        out = df_new.copy()
        out["lead_score"] = proba
        out = out.sort_values("lead_score", ascending=False)

        st.write("Top leads", out.head(top_n))

        fig, ax = plt.subplots(figsize=(5,3))
        ax.hist(proba, bins=20)
        ax.set_title("Score distribution")
        st.pyplot(fig)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored leads CSV", csv_bytes, file_name="scored_leads.csv")
